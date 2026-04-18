use crate::core::BitCollection;
use crate::enums::{BitIndexing, Codec, Endianness};
use crate::helpers;
use crate::helpers::{
    BS, BV, bv_from_bin, bv_from_bools, bv_from_bytes_slice, bv_from_f64, bv_from_hex,
    bv_from_i128, bv_from_oct, bv_from_ones, bv_from_random, bv_from_u128, bv_from_zeros,
    compute_lps, find_bitvec, logical_range_to_physical, physical_match_to_logical_start,
    promote_to_bv, rfind_bitvec, str_to_bv, validate_logical_op_lengths, validate_shift,
    validate_slice,
};
use crate::iterator::{BoolIterator, ChunksIterator, FindAllIterator};
use crate::mutibs::Mutibs;
use bitvec::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PySlice, PyType};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::Not;
use std::sync::Arc;

impl Hash for Tibs {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);

        let bits = self.to_bitslice();

        let mut words = bits.chunks_exact(64);
        for chunk in words.by_ref() {
            state.write_u64(chunk.load_be::<u64>());
        }

        let mut bytes = words.remainder().chunks_exact(8);
        for chunk in bytes.by_ref() {
            state.write_u8(chunk.load_be::<u8>());
        }

        let tail = bytes.remainder();
        if !tail.is_empty() {
            let mut last = 0u8;
            for bit in tail {
                last = (last << 1) | (*bit as u8);
            }
            last <<= 8 - tail.len();
            state.write_u8(last);
        }
    }
}

// ---- Tibs private helper methods. Not part of the Python interface. ----

impl Tibs {
    fn raw_encoded_bit_length(bit_length: usize) -> usize {
        let data_byte_length = bit_length.div_ceil(8);
        5 + Self::encode_varint(data_byte_length as u64).len() + data_byte_length * 8
    }

    fn rice_encode_int(value: usize, k: u8) -> BV {
        let mut out = BV::new();
        let quotient = value >> k;
        for _ in 0..quotient {
            out.push(true);
        }
        out.push(false);
        if k > 0 {
            let remainder_mask = (1usize << k) - 1;
            let remainder = value & remainder_mask;
            for shift in (0..k).rev() {
                out.push(((remainder >> shift) & 1) == 1);
            }
        }
        out
    }

    fn rice_decode_int(bits: &BS, start: usize, k: u8) -> PyResult<(usize, usize)> {
        let mut pos = start;
        while pos < bits.len() && bits[pos] {
            pos += 1;
        }
        if pos >= bits.len() {
            return Err(PyValueError::new_err("The encoded sequence ended unexpectedly."));
        }
        let quotient = pos - start;
        pos += 1; // separator bit

        let k_usize = k as usize;
        if bits.len() - pos < k_usize {
            return Err(PyValueError::new_err("The encoded sequence ended unexpectedly."));
        }
        let remainder = if k == 0 {
            0
        } else {
            bits[pos..pos + k_usize].load_be::<usize>()
        };
        pos += k_usize;

        let base = quotient
            .checked_shl(k as u32)
            .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
        let value = base
            .checked_add(remainder)
            .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
        Ok((value, pos))
    }

    fn zstd_compress_bytes(&self) -> Vec<u8> {
        let bit_length = self.len();
        let data_byte_length = bit_length.div_ceil(8);
        let raw_bit_padding = data_byte_length * 8 - bit_length;

        let mut raw = self.to_bitvec();
        for _ in 0..raw_bit_padding {
            raw.push(false);
        }

        zstd::bulk::compress(&raw.into_vec(), 0)
            .expect("zstd compression failed") // TODO
    }

    fn encode_as_zstd_from_compressed(&self, compressed: Vec<u8>) -> BV {
        let mut bv = BV::new();
        bv.push(true); // codec bit 0
        bv.push(false); // codec bit 1
        bv.extend_from_bitslice(bits![0, 0, 0]);
        bv.extend(Self::encode_varint(compressed.len() as u64));
        bv.extend(BV::from_vec(compressed));
        bv
    }

    fn encode_as_zstd(&self) -> BV {
        self.encode_as_zstd_from_compressed(Self::zstd_compress_bytes(self))
    }

    fn encode_as_raw(&self) -> BV {
        let bit_length = self.len();
        let data_byte_length = bit_length.div_ceil(8);
        let bit_padding = data_byte_length * 8 - bit_length;

        let mut bv = BV::new();
        bv.push(false); // codec bit 0
        bv.push(false); // codec bit 1
        for shift in (0..3).rev() {
            bv.push((bit_padding >> shift) & 1 == 1);
        }
        bv.extend(Self::encode_varint(data_byte_length as u64));
        bv.extend(self.to_bitvec());
        for _ in 0..bit_padding {
            bv.push(false);
        }
        bv
    }

    fn rice_encoded_gaps(bits: &BS, sparse_bit: bool) -> Vec<usize> {
        let mut gaps = Vec::new();
        let mut gap = 0usize;
        let opposite_bit = !sparse_bit;

        for bit in bits {
            if *bit == sparse_bit {
                gaps.push(gap);
                gap = 0;
            } else {
                debug_assert_eq!(*bit, opposite_bit);
                gap += 1;
            }
        }

        if let Some(last) = bits.last() {
            if *last != sparse_bit {
                debug_assert!(gap > 0);
                gaps.push(gap - 1);
            }
        }

        gaps
    }

    fn estimated_rice_k(gaps: &[usize]) -> u8 {
        if gaps.is_empty() {
            return 0;
        }

        let total_gap: usize = gaps.iter().sum();
        if total_gap == 0 {
            return 0;
        }

        let mean_gap = total_gap as f64 / gaps.len() as f64;
        let estimate = (mean_gap * std::f64::consts::LN_2).log2().round();
        estimate.clamp(0.0, 31.0) as u8
    }

    fn rice_payload_bit_length(gaps: &[usize], k: u8) -> usize {
        gaps.iter().map(|gap| (gap >> k) + 1 + k as usize).sum()
    }

    fn rice_encoded_bit_length(&self, sparse_bit: bool) -> usize {
        let gaps = Self::rice_encoded_gaps(self.to_bitslice(), sparse_bit);
        let estimated_k = Self::estimated_rice_k(&gaps);
        let payload_bit_length = Self::rice_payload_bit_length(&gaps, estimated_k);
        let payload_byte_length = payload_bit_length.div_ceil(8);
        5 + Self::encode_varint(payload_byte_length as u64).len() + 8 + payload_byte_length * 8
    }

    fn encode_as_rice(&self, sparse_bit: bool) -> BV {
        let bits = self.to_bitslice();

        let gaps = Self::rice_encoded_gaps(bits, sparse_bit);
        let final_bit = *bits.last().unwrap();
        let estimated_k = Self::estimated_rice_k(&gaps);

        let payload_bit_length = Self::rice_payload_bit_length(&gaps, estimated_k);
        let mut payload = BV::new();
        for gap in &gaps {
            payload.extend(Self::rice_encode_int(*gap, estimated_k));
        }
        debug_assert_eq!(payload.len(), payload_bit_length);
        let payload_byte_length = payload_bit_length.div_ceil(8);
        let bit_padding = payload_byte_length * 8 - payload_bit_length;
        for _ in 0..bit_padding {
            payload.push(false);
        }

        let mut encoded = BV::new();
        encoded.push(false); // codec bit 0
        encoded.push(true); // codec bit 1 => 01
        for shift in (0..3).rev() {
            encoded.push((bit_padding >> shift) & 1 == 1);
        }
        encoded.extend(Self::encode_varint(payload_byte_length as u64));
        for shift in (0..5).rev() {
            encoded.push((estimated_k >> shift) & 1 == 1);
        }
        encoded.push(sparse_bit);
        encoded.push(final_bit);
        encoded.push(false); // reserved
        encoded.extend(payload);

        encoded
    }

    fn decode_raw_long(
        bv: &BS,
        msb0_flag: bool,
        bit_padding: usize,
        data_start: usize,
        data_bits: usize,
    ) -> PyResult<Tibs> {
        let data_end = data_start
            .checked_add(data_bits)
            .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
        if bv.len() < data_end {
            return Err(PyValueError::new_err("The encoded sequence ended unexpectedly."));
        }
        if bv.len() != data_end {
            return Err(PyValueError::new_err("The encoded sequence has unexpected trailing bytes."));
        }
        if bit_padding > data_bits {
            return Err(PyValueError::new_err("The encoded sequence is reserved."));
        }

        let out_end = data_end - bit_padding;
        Ok(Tibs::from_bv(bv[data_start..out_end].to_bitvec(), msb0_flag))
    }

    fn decode_rice_long(
        bv: &BS,
        msb0_flag: bool,
        bit_padding: usize,
        data_start: usize,
        payload_bits: usize,
    ) -> PyResult<Tibs> {
        let config_end = data_start
            .checked_add(8)
            .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
        if bv.len() < config_end {
            return Err(PyValueError::new_err("The encoded sequence ended unexpectedly."));
        }

        let payload_start = config_end;
        let payload_end = payload_start
            .checked_add(payload_bits)
            .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
        if bv.len() < payload_end {
            return Err(PyValueError::new_err("The encoded sequence ended unexpectedly."));
        }
        if bv.len() != payload_end {
            return Err(PyValueError::new_err("The encoded sequence has unexpected trailing bytes."));
        }
        if bit_padding > payload_bits {
            return Err(PyValueError::new_err("The encoded sequence is reserved."));
        }

        let config = &bv[data_start..config_end];
        if config[7] {
            return Err(PyValueError::new_err("The encoded sequence is reserved."));
        }
        let k = config[0..5].load_be::<u8>();
        let sparse_bit = config[5];
        let final_bit = config[6];

        let encoded_gaps_end = payload_end - bit_padding;
        let encoded_gaps = &bv[payload_start..encoded_gaps_end];

        let mut decoded = BV::new();
        let mut pos = 0usize;
        while pos < encoded_gaps.len() {
            let (gap, next_pos) = Self::rice_decode_int(encoded_gaps, pos, k)?;
            pos = next_pos;

            for _ in 0..gap {
                decoded.push(!sparse_bit);
            }
            decoded.push(sparse_bit);
        }

        if decoded.is_empty() {
            return Err(PyValueError::new_err("The encoded sequence is reserved."));
        }
        let final_pos = decoded.len() - 1;
        decoded.set(final_pos, final_bit);
        Ok(Tibs::from_bv(decoded, msb0_flag))
    }

    fn encode_varint(mut u: u64) -> BV {
        let mut chunks: Vec<u8> = Vec::new();
        loop {
            chunks.push((u & 0x7f) as u8);
            u >>= 7;
            if u == 0 {
                break;
            }
        }
        chunks.reverse();

        let mut out: BV = BV::with_capacity(chunks.len() * 8);
        for (i, chunk) in chunks.iter().enumerate() {
            let continuation = i + 1 < chunks.len(); // 1 if another varint byte follows
            out.push(continuation);
            for shift in (0..7).rev() {
                out.push(((chunk >> shift) & 1) == 1);
            }
        }
        out
    }

    fn decode_varint(bits: &BS) -> PyResult<(usize, usize)> {
        let mut value: usize = 0;
        let mut bits_consumed: usize = 0;
        let mut saw_final = false;

        for byte in bits.chunks(8) {
            if byte.len() < 8 {
                break;
            }
            let continuation = byte[0];
            let payload = byte[1..8].load_be::<u8>() as usize;

            // Per spec, a first varint byte of 10000000 is reserved.
            if bits_consumed == 0 && continuation && payload == 0 {
                return Err(PyValueError::new_err("The encoded sequence is reserved."));
            }
            if value > (usize::MAX >> 7) {
                return Err(PyValueError::new_err("The encoded sequence is too large to decode."));
            }
            value = (value << 7) | payload;
            bits_consumed += 8;

            if !continuation {
                saw_final = true;
                break;
            }
        }

        if !saw_final {
            return Err(PyValueError::new_err("The encoded sequence ended unexpectedly."));
        }
        Ok((value, bits_consumed))
    }

    pub(crate) fn from_bv(bv: BV, msb0: bool) -> Self {
        let length = bv.len();
        Tibs {
            data: Arc::new(bv),
            offset: 0,
            length,
            msb0,
        }
    }

    pub(crate) fn get_slice_unchecked(&self, offset: usize, length: usize) -> Self {
        Tibs {
            data: self.data.clone(),
            offset: self.offset + offset,
            length,
            msb0: self.msb0,
        }
    }

    #[inline]
    pub(crate) fn to_bitslice(&self) -> &BS {
        if self.msb0 {
            &self.data[self.offset..self.offset + self.length]
        } else {
            let start_bit = self.data.len() - self.length - self.offset;
            &self.data[start_bit..start_bit + self.length]
        }
    }

    #[inline]
    pub(crate) fn to_bitvec(&self) -> BV {
        self.to_bitslice().to_bitvec()
    }

    #[inline]
    pub(crate) fn raw_bytes(&self) -> Vec<u8> {
        // Given the bit offset self._offset and the bit length self._length
        // return the byte data from the bitvec self._data. The data should cover just
        // enough bytes and should not realign them in any way.
        let byte_offset = self.offset / 8;
        let final_byte = (self.offset + self.length).div_ceil(8);
        let full_bytes = self.data.as_raw_slice();
        full_bytes[byte_offset..final_byte].to_vec()
    }

    pub(crate) fn find_impl(
        &self,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        reverse: bool,
    ) -> PyResult<Option<usize>> {
        if needle.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to find."));
        }
        let len = self.len();
        let (start, end) = validate_slice(len, start, end)?;
        let needle = if self.msb0 { needle } else { needle.reversed() };
        let (start, end) = logical_range_to_physical(len, start, end, self.msb0);

        let use_find = self.msb0 ^ reverse;
        let found = if !self.msb0 && byte_aligned {
            let mut search_start = start;
            let mut search_end = end;
            let needle_len = needle.len();
            loop {
                let candidate = if use_find {
                    find_bitvec(
                        self.to_bitslice(),
                        needle.as_bitslice(),
                        search_start,
                        search_end,
                        false,
                    )
                } else {
                    rfind_bitvec(
                        self.to_bitslice(),
                        needle.as_bitslice(),
                        search_start,
                        search_end,
                        false,
                    )
                };
                let Some(pos) = candidate else {
                    break None;
                };
                let logical = physical_match_to_logical_start(len, needle_len, pos, self.msb0);
                if logical % 8 == 0 {
                    break Some(pos);
                }
                if use_find {
                    search_start = pos.saturating_add(1);
                    if search_start >= search_end {
                        break None;
                    }
                } else {
                    search_end = pos.saturating_add(needle_len.saturating_sub(1));
                    if search_end <= search_start {
                        break None;
                    }
                }
            }
        } else if use_find {
            find_bitvec(
                self.to_bitslice(),
                needle.as_bitslice(),
                start,
                end,
                byte_aligned,
            )
        } else {
            rfind_bitvec(
                self.to_bitslice(),
                needle.as_bitslice(),
                start,
                end,
                byte_aligned,
            )
        };
        Ok(found.map(|pos| physical_match_to_logical_start(len, needle.len(), pos, self.msb0)))
    }
}

///     An immutable container of binary data.
///
///     The constructor is a convenient way to delegate to the ``from_string``,
///     ``from_bytes`` or ``from_bools`` builder methods, depending on the type of ``auto``.
///
///     * ``Tibs('0x13')`` - Equivalent to ``Tibs.from_string('0x13')``.
///     * ``Tibs([1, 0])`` - Equivalent to ``Tibs.from_bools([1, 0])``.
///     * ``Tibs(b'hello')`` - Equivalent to ``Tibs.from_bytes(b'hello')``.
///
///     Otherwise, to construct use a builder 'from' method:
///
///     * ``Tibs.from_bin(s)`` - Create from a binary string, optionally starting with '0b'.
///     * ``Tibs.from_oct(s)`` - Create from an octal string, optionally starting with '0o'.
///     * ``Tibs.from_hex(s)`` - Create from a hex string, optionally starting with '0x'.
///     * ``Tibs.from_u(u, length, [endianness])`` - Create from an unsigned int to a given length.
///     * ``Tibs.from_i(i, length, [endianness])`` - Create from a signed int to a given length.
///     * ``Tibs.from_f(f, length, [endianness])`` - Create from an IEEE float to a 16, 32 or 64 bit length.
///     * ``Tibs.from_bytes(b)`` - Create directly from a ``bytes`` or ``bytearray`` object.
///     * ``Tibs.from_string(s)`` - Use a formatted string.
///     * ``Tibs.from_bools(iterable)`` - Convert each element in ``iterable`` to a bool.
///     * ``Tibs.from_zeros(length)`` - Initialise with ``length`` ``0`` bits.
///     * ``Tibs.from_ones(length)`` - Initialise with ``length`` ``1`` bits.
///     * ``Tibs.from_random(length, [secure, seed])`` - Initialise with ``length`` randomly set bits.
///     * ``Tibs.from_joined(iterable)`` - Concatenate an iterable of objects.
///
#[derive(Clone)]
#[pyclass(frozen, sequence, skip_from_py_object, module = "tibs")]
pub struct Tibs {
    data: Arc<BV>,
    offset: usize,
    length: usize,
    pub msb0: bool,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Tibs {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(tibs_ref) = obj.extract::<PyRef<Tibs>>() {
            return Ok(tibs_ref.clone());
        }
        if let Ok(mutibs_ref) = obj.extract::<PyRef<Mutibs>>() {
            return Ok(mutibs_ref.to_tibs());
        }
        // Default to msb0 when creating from other types.
        let bv = promote_to_bv(&obj)?;
        Ok(Tibs::from_bv(bv, true))
    }
}

/// Public Python-facing methods.
#[pymethods]
impl Tibs {
    #[new]
    #[pyo3(signature = (auto = None, bit_indexing = BitIndexing::Msb0), text_signature = "(auto=None, bit_indexing)")]
    pub fn py_new(
        auto: Option<&Bound<'_, PyAny>>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let Some(auto) = auto else {
            return Ok(BitCollection::empty(msb0));
        };
        let mut tibs = Tibs::extract(auto.as_borrowed())?;
        tibs.msb0 = msb0;
        Ok(tibs)
    }

    /// Whether the bits are indexed from the most significant bit (BitIndexing.Msb0, the default) or from the
    /// least significant bit (BitIndexing.Lsb0). This doesn't affect the actual data stored, just how it's
    /// accessed.
    #[getter]
    pub fn bit_indexing(&self) -> BitIndexing {
        if self.msb0 {
            BitIndexing::Msb0
        } else {
            BitIndexing::Lsb0
        }
    }

    /// Return a new instance with the bits reversed.
    ///
    /// :return: Tibs
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Tibs('0b00011')
    ///     >>> a.reversed()
    ///     >>> Tibs('0b11000')
    ///
    fn reversed(&self) -> Self {
        BitCollection::reverse_copy(self)
    }

    /// Return a new instance with the byte endianness swapped.
    ///
    /// The whole of the data will be byte-swapped. It must be a multiple
    /// of byte_length long.
    ///
    /// :param int | None byte_length: An int giving the number of bytes in each swap, or None (the default)
    ///   to do a single reverse over the whole data.
    /// :return: Tibs
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Tibs('0x12345678')
    ///     >>> b = a.byte_swapped(2)
    ///     >>> b
    ///     Tibs('0x34127856')
    ///
    #[pyo3(signature = (byte_length = None), text_signature = "($self, byte_length=None)")]
    pub fn byte_swapped(&self, byte_length: Option<i64>) -> PyResult<Tibs> {
        Ok(BitCollection::byte_swap_copy(self, byte_length)?)
    }

    /// Return a copy of the raw byte information.
    ///
    /// This returns the underlying byte data and can contain leading and trailing
    /// bits that are not considered part of the object's data. Usually using
    /// :meth:`~to_bytes` is what you really need.
    ///
    /// The way that the data is stored is not considered part of the public interface
    /// and so the output of this method may change between point releases, and even
    /// during the running of a program.
    ///
    /// :return: A tuple of the raw bytes, the bit offset and the bit length.
    ///
    /// .. code-block:: python
    ///
    ///     raw_bytes, offset, length = t.to_raw_data()
    ///     assert t == Tibs.from_bytes(raw_bytes)[offset:offset + length]
    ///
    pub fn to_raw_data(&self) -> (Vec<u8>, usize, usize) {
        self.raw_data()
    }

    /// Return string representations for printing.
    pub fn __str__(&self) -> String {
        self.to_string()
    }

    /// Return representation that could be used to recreate the instance.
    pub fn __repr__(&self) -> String {
        if self.is_empty() {
            let bit_indexing = if self.msb0 {
                "".to_string()
            } else {
                "bit_indexing=BitIndexing.Lsb0".to_string()
            };
            format!("Tibs({})", bit_indexing)
        } else {
            let bit_indexing = if self.msb0 {
                "".to_string()
            } else {
                ", BitIndexing.Lsb0".to_string()
            };
            format!("Tibs('{}'{})", self.__str__(), bit_indexing)
        }
    }

    /// Iterate over the bits of the Tibs, yielding each bit as a boolean.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<BoolIterator>> {
        let py = slf.py();
        let length = slf.len() as isize;
        Py::new(
            py,
            BoolIterator {
                bits: slf.into(),
                index: 0,
                length,
            },
        )
    }

    /// Return Tibs generator by cutting into chunks.
    ///
    /// :param int chunk_size: The size in bits of the chunks to generate.
    /// :param int | None count: If specified, at most count items are generated. Default is to cut as many times as possible.
    /// :return: A generator yielding Tibs chunks.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b110011').chunks(2))
    ///     [Tibs('0b11'), Tibs('0b00'), Tibs('0b11')]
    ///
    #[pyo3(signature = (chunk_size, count = None), text_signature = "($self, chunk_size, count=None)")]
    pub fn chunks(
        slf: PyRef<'_, Self>,
        chunk_size: i64,
        count: Option<i64>,
    ) -> PyResult<Py<ChunksIterator>> {
        if chunk_size <= 0 {
            return Err(PyValueError::new_err(format!(
                "Cannot create chunk generator - chunk_size of {chunk_size} given, but it must be > 0."
            )));
        }
        let max_chunks = match count {
            Some(c) => {
                if c < 0 {
                    return Err(PyValueError::new_err(format!(
                        "Cannot create chunk generator - count of {c} given, but it must be > 0 if present."
                    )));
                }
                c as usize
            }
            None => usize::MAX,
        };

        let py = slf.py();
        let bits_len = slf.len();
        let iter = ChunksIterator {
            bits_object: slf.into(),
            chunk_size: chunk_size as usize,
            max_chunks,
            current_pos: 0,
            chunks_generated: 0,
            bits_len,
            is_reverse: false,
        };
        Py::new(py, iter)
    }

    /// Return reverse Tibs generator by cutting into chunks, starting from the end.
    ///
    /// :param int chunk_size: The size in bits of the chunks to generate.
    /// :param int | None count: If specified, at most count items are generated. Default is to cut as many times as possible.
    /// :return: A generator yielding Tibs chunks.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b1100111').rchunks(3))
    ///     [Tibs('0b111'), Tibs('0b100'), Tibs('0b11')]
    ///
    #[pyo3(signature = (chunk_size, count = None), text_signature = "($self, chunk_size, count=None)")]
    pub fn rchunks(
        slf: PyRef<'_, Self>,
        chunk_size: i64,
        count: Option<i64>,
    ) -> PyResult<Py<ChunksIterator>> {
        if chunk_size <= 0 {
            return Err(PyValueError::new_err(format!(
                "Cannot create chunk generator - chunk_size of {chunk_size} given, but it must be > 0."
            )));
        }
        let max_chunks = match count {
            Some(c) => {
                if c < 0 {
                    return Err(PyValueError::new_err(format!(
                        "Cannot create chunk generator - count of {c} given, but it must be > 0 if present."
                    )));
                }
                c as usize
            }
            None => usize::MAX,
        };

        let py = slf.py();
        let bits_len = slf.len();
        let iter = ChunksIterator {
            bits_object: slf.into(),
            chunk_size: chunk_size as usize,
            max_chunks,
            current_pos: bits_len,
            chunks_generated: 0,
            bits_len,
            is_reverse: true,
        };
        Py::new(py, iter)
    }

    /// Return True if two Tibs have the same binary representation.
    ///
    /// The right hand side will be promoted to a Tibs if needed and possible.
    ///
    /// >>> Tibs('0b1110') == '0xe'
    /// True
    ///
    pub fn __eq__(&self, other: Tibs) -> bool {
        *self.to_bitslice() == *other.as_bitslice()
    }

    #[pyo3(name = "__hash__")]
    /// Return a hash of the Tibs.
    pub fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish() as isize
    }

    /// Find all occurrences of a bit sequence. Return generator of bit positions.
    ///
    /// :param Tibs needle: The bit sequence to find.
    /// :param int | None start: The starting bit position of the slice to search. Defaults to 0.
    /// :param int | None end: The end bit position of the slice to search. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries. Defaults to ``False``.
    /// :return: A generator yielding bit positions.
    ///
    /// :raises ValueError: if needle is empty, if start or end are out of range or if end is before start.
    ///
    /// All occurrences of needle are found, even if they overlap.
    ///
    /// Note that this method is not available for :class:`Mutibs` as its value could change while the
    /// generator is still active. For that case you should convert to a :class:`Tibs` first with :meth:`Mutibs.to_tibs`.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b10111011').find_all('0b11'))
    ///     [2, 3, 6]
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn find_all(
        slf: PyRef<'_, Self>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
    ) -> PyResult<Py<FindAllIterator>> {
        if needle.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to find."));
        }
        // TODO: For single bits we could use more specialised methods
        // See https://docs.rs/bitvec/1.0.1/bitvec/slice/struct.BitSlice.html#method.iter_ones
        let (start, end) = validate_slice(slf.len(), start, end)?;
        let needle = if slf.msb0 { needle } else { needle.reversed() };
        let (start, end) = logical_range_to_physical(slf.len(), start, end, slf.msb0);
        let is_reverse = !slf.msb0;
        let step = if byte_aligned { 8 } else { 1 };
        let py = slf.py();
        let lps = { compute_lps(needle.to_bitslice()) };
        let iter_obj = FindAllIterator {
            haystack: slf.into(),
            needle,
            lps,
            start,
            end,
            byte_aligned,
            step,
            current_pos: if is_reverse { end } else { start },
            is_reverse,
        };
        Py::new(py, iter_obj)
    }

    /// Find all occurrences of a bit sequence, searching in reverse. Return generator of bit positions.
    ///
    /// :param Tibs needle: The bit sequence to find.
    /// :param int | None start: The starting bit position of the slice to search. Defaults to 0.
    /// :param int | None end: The end bit position of the slice to search. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries. Defaults to ``False``.
    /// :return: A generator yielding bit positions.
    ///
    /// :raises ValueError: if needle is empty, if start or end are out of range or end is before start.
    ///
    /// All occurrences of needle are found, even if they overlap.
    ///
    /// Note that this method is not available for :class:`Mutibs` as its value could change while the
    /// generator is still active. For that case you should convert to a :class:`Tibs` first with :meth:`Mutibs.to_tibs`.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b10111011').rfind_all('0b11'))
    ///     [6, 3, 2]
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn rfind_all(
        slf: PyRef<'_, Self>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
    ) -> PyResult<Py<FindAllIterator>> {
        if needle.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to find."));
        }
        let (start, end) = validate_slice(slf.len(), start, end)?;
        let needle = if slf.msb0 { needle } else { needle.reversed() };
        let (start, end) = logical_range_to_physical(slf.len(), start, end, slf.msb0);
        let is_reverse = slf.msb0;
        let step = if byte_aligned { 8 } else { 1 };
        let py = slf.py();
        let lps = { compute_lps(needle.to_bitslice()) };
        let iter_obj = FindAllIterator {
            haystack: slf.into(),
            needle,
            lps,
            start,
            end,
            byte_aligned,
            step,
            current_pos: if is_reverse { end } else { start },
            is_reverse,
        };
        Py::new(py, iter_obj)
    }

    /// The bit length of the Tibs.
    #[inline]
    pub fn __len__(&self) -> usize {
        self.len()
    }

    /// Create a new instance with all bits set to '0'.
    ///
    /// :param int length: The number of bits to set.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    /// :return: A Tibs object with all bits set to zero.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_zeros(500)  # 500 zero bits
    ///
    #[classmethod]
    #[pyo3(signature = (length, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, length, bit_indexing)")]
    pub fn from_zeros(
        _cls: &Bound<'_, PyType>,
        length: i64,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        if length < 0 {
            return Err(PyValueError::new_err(format!(
                "Negative bit length given: {}.",
                length
            )));
        }
        Ok(Self::from_bv(bv_from_zeros(length as usize), msb0))
    }

    /// Create a new instance with all bits set to '1'.
    ///
    /// :param int length: The number of bits to set.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_ones(5)
    ///     Tibs('0b11111')
    ///
    #[classmethod]
    #[pyo3(signature = (length, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, length, bit_indexing)")]
    pub fn from_ones(
        _cls: &Bound<'_, PyType>,
        length: i64,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        if length < 0 {
            return Err(PyValueError::new_err(format!(
                "Negative bit length given: {}.",
                length
            )));
        }
        Ok(Tibs::from_bv(bv_from_ones(length as usize), msb0))
    }

    /// Create a new instance from a formatted string.
    ///
    /// :param str s: The formatted string to convert. This can begin with '0b', '0o' or '0x' to indicate binary, octal or hexadecimal, and commas can be used to separate items.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_string("0xff01")
    ///     b = Tibs.from_string("0o775, 0b1")
    ///
    /// The ``__init__`` method can also redirect to ``from_string``:
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs("0xff01")
    ///
    #[classmethod]
    #[pyo3(signature = (s, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, s, bit_indexing)")]
    pub fn from_string(
        _cls: &Bound<'_, PyType>,
        s: String,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = str_to_bv(s)?;
        Ok(Tibs::from_bv(bv, msb0))
    }

    /// Create a new instance from an unsigned integer.
    ///
    /// :param int u: An unsigned integer.
    /// :param int length: The bit length to create. Can be up to 128.
    /// :param Endianness endianness: The byte endianness used to store the integer. Defaults to Endianness.Unspecified.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// :raises ValueError: if the integer doesn't fit in the length given.
    ///
    #[classmethod]
    #[pyo3(signature = (u, /, length, endianness = Endianness::Unspecified, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, u, length, endianness, bit_indexing)")]
    pub fn from_u(
        _cls: &Bound<'_, PyType>,
        u: u128,
        length: i64,
        endianness: Option<Endianness>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let is_msb0 = BitIndexing::is_msb0(bit_indexing);
        let is_little_endian = Endianness::is_little_endian(endianness, length as usize)?;
        Ok(Tibs::from_bv(bv_from_u128(u, length, is_little_endian)?, is_msb0))
    }

    /// Return the unsigned integer representation of the Tibs.
    ///
    /// :param Endianness endianness: The byte endianness used to interpret the integer. Defaults to Endianness.Unspecified.
    #[pyo3(signature = (endianness = Endianness::Unspecified), text_signature = "($self, endianness)")]
    pub fn to_u(&self, endianness: Option<Endianness>) -> PyResult<u128> {
        let is_little_endian = Endianness::is_little_endian(endianness, self.len())?;
        BitCollection::to_u128(self, is_little_endian)
    }

    /// Create a new instance from a signed integer.
    ///
    /// :param int i: A signed integer.
    /// :param int length: The bit length to create. Can be up to 128.
    /// :param Endianness endianness: The byte endianness used to store the integer. Defaults to Endianness.Unspecified.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// :raises ValueError: if the integer doesn't fit in the length given.
    ///
    #[classmethod]
    #[pyo3(signature = (i, /, length, endianness = Endianness::Unspecified, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, i, length, endianness, bit_indexing)")]
    pub fn from_i(
        _cls: &Bound<'_, PyType>,
        i: i128,
        length: i64,
        endianness: Option<Endianness>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let is_msb0 = BitIndexing::is_msb0(bit_indexing);
        let is_little_endian = Endianness::is_little_endian(endianness, length as usize)?;
        Ok(Tibs::from_bv(bv_from_i128(i, length, is_little_endian)?, is_msb0))
    }

    /// Return the signed integer representation of the Tibs.
    ///
    /// :param Endianness endianness: The byte endianness used to interpret the integer. Defaults to Endianness.Unspecified.
    #[pyo3(signature = (endianness = Endianness::Unspecified), text_signature = "($self, endianness)")]
    pub fn to_i(&self, endianness: Option<Endianness>) -> PyResult<i128> {
        let is_little_endian = Endianness::is_little_endian(endianness, self.len())?;
        BitCollection::to_i128(self, is_little_endian)
    }

    /// Create a new instance from a floating point number.
    ///
    /// :param float f: A floating point value.
    /// :param int length: The bit length to create. Must be 16, 32 or 64.
    /// :param Endianness endianness: The byte endianness used to store the float. Defaults to Endianness.Unspecified.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    #[classmethod]
    #[pyo3(signature = (f, /, length, endianness = Endianness::Unspecified, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, f, length, endianness, bit_indexing)")]
    pub fn from_f(
        _cls: &Bound<'_, PyType>,
        f: f64,
        length: i64,
        endianness: Option<Endianness>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let is_msb0 = BitIndexing::is_msb0(bit_indexing);
        let is_little_endian = Endianness::is_little_endian(endianness, length as usize)?;
        let bv = bv_from_f64(f, length, is_little_endian)?;
        Ok(Tibs::from_bv(bv, is_msb0))
    }

    /// Return the floating point representation of the Tibs.
    ///
    /// The length must be 16, 32 or 64.
    ///
    /// :param Endianness endianness: The byte endianness used to interpret the float. Defaults to Endianness.Unspecified.
    #[pyo3(signature = (endianness = Endianness::Unspecified), text_signature = "($self, endianness)")]
    pub fn to_f(&self, endianness: Option<Endianness>) -> PyResult<f64> {
        let is_little_endian = Endianness::is_little_endian(endianness, self.len())?;
        BitCollection::to_f64(self, is_little_endian)
    }

    /// Create a new instance from a binary string.
    ///
    /// :param str s: A string of ``0`` and ``1`` s, optionally preceded with ``0b`` and optionally containing underscores.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_bin("0000_1111_0101")
    ///
    #[classmethod]
    #[pyo3(signature = (s, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, s, bit_indexing)")]
    pub fn from_bin(
        _cls: &Bound<'_, PyType>,
        s: &str,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = bv_from_bin(s)?;
        Ok(Tibs::from_bv(bv, msb0))
    }

    /// Return the binary representation of the Tibs as a string.
    ///
    /// Equivalent to using the ``bin`` property.
    ///
    /// :return: The binary representation.
    pub fn to_bin(&self) -> String {
        BitCollection::to_binary(self)
    }

    /// Read-only property of the binary representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_bin`.
    ///
    /// :return: The binary representation.
    #[getter]
    fn bin(&self) -> String {
        BitCollection::to_binary(self)
    }

    /// Create a new instance from an octal string.
    ///
    /// :param str s: A string of octal digits, optionally preceded with ``0o`` and optionally containing underscores.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    #[classmethod]
    #[pyo3(signature = (s, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, s, bit_indexing)")]
    pub fn from_oct(
        _cls: &Bound<'_, PyType>,
        s: &str,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = bv_from_oct(s)?;
        Ok(Tibs::from_bv(bv, msb0))
    }

    /// Return the octal representation of the Tibs as a string.
    ///
    /// Equivalent to using the ``oct`` property.
    ///
    /// :return: The octal representation.
    /// :raises ValueError: if the length is not a multiple of 3.
    pub fn to_oct(&self) -> PyResult<String> {
        BitCollection::to_octal(self)
    }

    /// Read-only property of the octal representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_oct`.
    ///
    /// :return: The octal representation.
    /// :raises ValueError: if the length is not a multiple of 3.
    #[getter]
    fn oct(&self) -> PyResult<String> {
        BitCollection::to_octal(self)
    }

    /// Create a new instance from a hexadecimal string.
    ///
    /// :param str s: A string of hexadecimal digits, optionally preceded with ``0x`` and optionally containing underscores.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    #[classmethod]
    #[pyo3(signature = (s, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, s, bit_indexing)")]
    pub fn from_hex(
        _cls: &Bound<'_, PyType>,
        s: &str,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = bv_from_hex(s)?;
        Ok(Tibs::from_bv(bv, msb0))
    }

    /// Return the hexadecimal representation of the Tibs as a string.
    ///
    /// Equivalent to using the ``hex`` property.
    ///
    /// :return: The hexadecimal representation.
    /// :raises ValueError: if the length is not a multiple of 4.
    pub fn to_hex(&self) -> PyResult<String> {
        BitCollection::to_hexadecimal(self)
    }

    /// Read-only property of the hexadecimal representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_hex`.
    ///
    /// :return: The hexadecimal representation.
    /// :raises ValueError: if the length is not a multiple of 4.
    #[getter]
    fn hex(&self) -> PyResult<String> {
        BitCollection::to_hexadecimal(self)
    }

    /// Create a new instance from a bytes object.
    ///
    /// :param bytes | bytearray | memoryview data: The bytes, bytearray or memoryview object to convert to a :class:`Tibs`.
    /// :param int | None offset: The bit offset from the start. Defaults to zero.
    /// :param int | None length: The bit length to use. Defaults to the whole of the data.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_bytes(b"some_bytes_maybe_from_a_file")
    ///
    #[classmethod]
    #[inline]
    #[pyo3(signature = (data, /, offset=None, length=None, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, data, offset=None, length=None, bit_indexing=None)")]
    pub fn from_bytes(
        _cls: &Bound<'_, PyType>,
        data: Vec<u8>,
        offset: Option<i64>,
        length: Option<i64>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = bv_from_bytes_slice(data, offset, length)?;
        Ok(Self::from_bv(bv, msb0))
    }

    /// Create a new instance from an iterable by converting each element to a bool.
    ///
    /// :param Iterable iterable: The iterable to convert to a :class:`Tibs`.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_bools([False, 0, 1, "Steven"])  # binary 0011
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, iterable, bit_indexing)")]
    pub fn from_bools(
        _cls: &Bound<'_, PyType>,
        iterable: &Bound<'_, PyAny>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = bv_from_bools(iterable)?;
        Ok(Tibs::from_bv(bv, msb0))
    }

    /// Create a new instance with all bits randomly set.
    ///
    /// :param int length: The number of bits to set. Must be positive.
    /// :param bool secure: If ``True``, use the OS's cryptographically secure generator. Default is ``False``.
    /// :param bytes | bytearray | None seed: A bytes or bytearray to use as an optional seed, only if ``secure`` is ``False``.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    /// :return: A newly constructed ``Tibs`` with random data.
    ///
    /// The 'secure' option uses the OS's random data source, so will be slower and could potentially
    /// fail.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_random(1000000)  # A million random bits
    ///     b = Tibs.from_random(100, seed=b'a_seed')
    ///
    #[classmethod]
    #[pyo3(signature = (length, /, secure=false, seed=None, bit_indexing = BitIndexing::Msb0), text_signature="(cls, length, secure=False, seed=None, bit_indexing=None)")]
    pub fn from_random(
        _cls: &Bound<'_, PyType>,
        length: i64,
        secure: bool,
        seed: Option<Vec<u8>>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = bv_from_random(length, secure, &seed)?;
        Ok(Tibs::from_bv(bv, msb0))
    }

    /// Create a new instance by concatenating a sequence of Tibs objects.
    ///
    /// This method concatenates a sequence of Tibs objects into a single Tibs object.
    ///
    /// :param Iterable iterable: An iterable to concatenate. Items can be anything that can be promoted to a Tibs.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_joined(['0x01', [1, 0], b'some_bytes'])
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, iterable, bit_indexing)")]
    pub fn from_joined(
        _cls: &Bound<'_, PyType>,
        iterable: &Bound<'_, PyAny>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        Ok(Mutibs::from_joined(_cls, iterable, bit_indexing)?.as_tibs())
    }

    /// Return the Tibs as a bytes object.
    ///
    /// :return: The bytes representation.
    /// :raises ValueError: if the length is not a multiple of 8.
    pub fn to_bytes(&self) -> PyResult<Vec<u8>> {
        BitCollection::to_byte_data(self)
    }

    /// Read-only property of the ``bytes`` representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_bytes`.
    ///
    /// :return: The bytes representation.
    /// :raises ValueError: if the length is not a multiple of 8.
    #[getter]
    fn bytes(&self) -> PyResult<Vec<u8>> {
        BitCollection::to_byte_data(self)
    }

    /// Find first occurrence of a bit sequence.
    ///
    /// Returns the bit position if found, or None if not found.
    ///
    /// :param Tibs needle: The bit sequence to find.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries.
    /// :return: The bit position if found, or None if not found.
    ///
    /// :raises ValueError: if ``needle`` is empty, or if the slice parameters are invalid.
    ///
    /// .. code-block:: pycon
    ///
    ///      >>> Tibs('0xc3e').find('0b1111')
    ///      6
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn find(
        &self,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
    ) -> PyResult<Option<usize>> {
        self.find_impl(needle, start, end, byte_aligned, false)
    }

    /// Return True if b is a sub-sequence of self.
    pub fn __contains__(&self, b: Tibs) -> bool {
        match self.find(b, None, None, false) {
            Ok(Some(_)) => true,
            _ => false,
        }
    }

    /// As Tibs is immutable, this returns the same instance.
    pub fn __copy__(slf: PyRef<'_, Self>) -> Py<Self> {
        slf.into()
    }

    /// Find last occurrence of a bit sequence.
    ///
    /// Returns the bit position if found, or None if not found.
    ///
    /// :param Tibs needle: The bit sequence to find.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries.
    /// :return: The bit position if found, or None if not found.
    ///
    /// :raises ValueError: if ``needle`` is empty, or if the slice parameters are invalid.
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn rfind(
        &self,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
    ) -> PyResult<Option<usize>> {
        self.find_impl(needle, start, end, byte_aligned, true)
    }

    /// Return whether the current Tibs starts with prefix.
    ///
    /// :param Tibs prefix: The bits to search for.
    /// :return: True if the Tibs starts with the prefix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b101100').starts_with('0b101')
    ///     True
    ///     >>> Tibs('0b101100').starts_with('0b100')
    ///     False
    ///
    pub fn starts_with(&self, prefix: Tibs) -> PyResult<bool> {
        Ok(<Tibs as BitCollection>::starts_with(self, prefix))
    }

    /// Return whether the current Tibs ends with suffix.
    ///
    /// :param Tibs suffix: The bits to search for.
    /// :return: True if the Tibs ends with the suffix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b101100').ends_with('0b100')
    ///     True
    ///     >>> Tibs('0b101100').ends_with('0b101')
    ///     False
    ///
    pub fn ends_with(&self, suffix: Tibs) -> PyResult<bool> {
        Ok(<Tibs as BitCollection>::ends_with(self, suffix))
    }

    /// Counts the total number of occurrences of a bit pattern.
    ///
    /// :param object value: Either something that can be converted to a ``Tibs``, or a single bit (one of ``0``, ``1``, ``False`` or ``True``).
    ///
    /// :return: The number of times the bit pattern is found.
    ///
    ///     .. code-block:: pycon
    ///
    ///         >>> Tibs('0xef').count(1)
    ///         7
    ///         >>> Tibs.from_bin('0011010101100').count('0b01')
    ///         4
    ///
    pub fn count(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        match Tibs::extract(value.as_borrowed()) {
            Ok(v) => {
                if v.len() == 1 {
                    Ok(<Tibs as BitCollection>::count(self, v.get_index(0)?))
                } else {
                    Ok(helpers::count_bitvec(self.to_bitslice(), v.as_bitslice()))
                }
            }
            Err(_) => {
                let count_ones = helpers::convert_to_bool(value);
                match count_ones {
                    Some(b) => Ok(<Tibs as BitCollection>::count(self, b)),
                    None => Err(PyValueError::new_err(
                        "Cannot convert value to 0, 1 or a Tibs",
                    )),
                }
            }
        }
    }

    /// Return True if all bits are equal to 1, otherwise return False.
    ///
    /// :return: ``True`` if all bits are 1, otherwise ``False``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b1111').all()
    ///     True
    ///     >>> Tibs('0b1011').all()
    ///     False
    ///
    #[inline]
    pub fn all(&self) -> bool {
        self.to_bitslice().all()
    }

    /// Return True if any bits are equal to 1, otherwise return False.
    ///
    /// :return: ``True`` if any bits are 1, otherwise ``False``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b0000').any()
    ///     False
    ///     >>> Tibs('0b1000').any()
    ///     True
    ///
    #[inline]
    pub fn any(&self) -> bool {
        self.to_bitslice().any()
    }

    /// Return a new Tibs with one or many bits set to 1.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.set`.
    ///
    /// :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
    /// :return: A new Tibs.
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    pub fn set_at(&self, pos: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut out = self.to_mutibs();
        out.apply_set_positions(true, pos)?;
        Ok(out.to_tibs())
    }

    /// Return a new Tibs with one or many bits set to 0.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.unset`.
    ///
    /// :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
    /// :return: A new Tibs.
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    pub fn unset_at(&self, pos: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut out = self.to_mutibs();
        out.apply_set_positions(false, pos)?;
        Ok(out.to_tibs())
    }

    /// Return a new Tibs with selected bits inverted.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.invert`.
    ///
    #[pyo3(signature = (pos = None), text_signature = "($self, pos=None)")]
    pub fn inverted(&self, pos: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut out = self.to_mutibs();
        out.apply_invert_positions(pos)?;
        Ok(out.to_tibs())
    }

    /// Insert bits at position pos and return a new Tibs.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.insert`.
    ///
    #[pyo3(signature = (pos, bs, /), text_signature = "($self, pos, bs, /)")]
    pub fn inserted(&self, pos: isize, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bs = Tibs::extract(bs.as_borrowed())?;
        let mut out = self.to_mutibs();
        out.apply_insert_bits(pos, &bs)?;
        Ok(out.to_tibs())
    }

    /// Search and replace and return a new Tibs.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.replace`.
    #[pyo3(signature = (old, new, start=None, end=None, count=None, byte_aligned=false), text_signature = "($self, old, new, start=None, end=None, count=None, byte_aligned=False)")]
    pub fn replaced(
        &self,
        old: &Bound<'_, PyAny>,
        new: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
        count: Option<i64>,
        byte_aligned: bool,
    ) -> PyResult<Self> {
        let old = Tibs::extract(old.as_borrowed())?;
        let new = Tibs::extract(new.as_borrowed())?;
        let mut out = self.to_mutibs();
        out.apply_replace_bits(old, new, start, end, count, byte_aligned)?;
        Ok(out.to_tibs())
    }

    /// Create and return a mutable copy of the Tibs as a Mutibs instance.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> t = Tibs.from_hex('abc')
    ///     >>> m = t.to_mutibs()
    ///     >>> m *= 4
    ///     >>> t.hex
    ///     abc
    ///     >>> m.hex
    ///     abcabcabcabc
    ///
    pub fn to_mutibs(&self) -> Mutibs {
        Mutibs::from_bv(self.to_bitvec(), self.msb0)
    }

    #[inline]
    /// Get a bit or a slice of bits.
    ///
    /// :param int | slice key: The index or slice to get.
    /// :return: A bool for a single index, or a new Tibs for a slice.
    /// :raises IndexError: If the index is out of range.
    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = key.py();
        // Handle integer indexing
        if let Ok(index) = key.extract::<isize>() {
            let value: bool = self.get_index(index)?;
            let py_value = PyBool::new(py, value);
            return Ok(py_value.to_owned().into());
        }

        // Handle slice indexing
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(self.len() as isize)?;
            let (start, stop, step) = (
                isize::try_from(indices.start)?,
                isize::try_from(indices.stop)?,
                isize::try_from(indices.step)?,
                );

            let result = if step == 1 {
                if start < stop {
                    self.get_slice_unchecked(start as usize, (stop - start) as usize)
                } else {
                    Tibs::empty(self.msb0)
                }
            } else {
                self.get_slice_with_step(start, stop, step)?
            };
            let py_obj = Py::new(py, result)?.into_pyobject(py)?;
            return Ok(py_obj.into());
        }

        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    /// Return new Tibs shifted by n to the left.
    ///
    /// n -- the number of bits to shift. Must be >= 0.
    ///
    pub fn __lshift__(&self, n: i64) -> PyResult<Self> {
        let shift = validate_shift(self, n)?;
        Ok(self.lshift(shift))
    }

    /// Return new Tibs shifted by n to the right.
    ///
    /// n -- the number of bits to shift. Must be >= 0.
    ///
    pub fn __rshift__(&self, n: i64) -> PyResult<Self> {
        let shift = validate_shift(self, n)?;
        Ok(self.rshift(shift))
    }

    /// Concatenates two Tibs and return a newly constructed Tibs.
    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        let mut data = BV::with_capacity(self.len() + other.len());
        data.extend_from_bitslice(self.to_bitslice());
        data.extend_from_bitslice(other.as_bitslice());
        Ok(Tibs::from_bv(data, self.msb0))
    }

    /// Concatenates two Tibs and return a newly constructed Tibs.
    pub fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        let mut data = BV::with_capacity(other.len() + self.len());
        data.extend_from_bitslice(other.as_bitslice());
        data.extend_from_bitslice(self.to_bitslice());
        Ok(Tibs::from_bv(data, self.msb0))
    }

    /// Bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        // TODO: Return early `if other is self`.
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_and(self, &other))
    }

    /// Bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        // TODO: Return early `if other is self`.
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_or(self, &other))
    }

    /// Bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;

        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_xor(self, &other))
    }

    /// Reverse bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__and__(other)
    }

    /// Reverse bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__or__(other)
    }

    /// Reverse bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__xor__(other)
    }

    /// Return a new Tibs with the bits rotated to the left.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.rotate_left`.
    ///
    #[pyo3(signature = (n, start=None, end=None), text_signature = "($self, n, start=None, end=None)")]
    pub fn rotated_left(&self, n: i64, start: Option<isize>, end: Option<isize>) -> PyResult<Self> {
        let mut out = self.to_mutibs();
        out.apply_rotation(n, start, end, true)?;
        Ok(out.to_tibs())
    }

    /// Return a new Tibs with the bits rotated to the right.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.rotate_right`.
    ///
    #[pyo3(signature = (n, start=None, end=None), text_signature = "($self, n, start=None, end=None)")]
    pub fn rotated_right(&self, n: i64, start: Option<isize>, end: Option<isize>) -> PyResult<Self> {
        let mut out = self.to_mutibs();
        out.apply_rotation(n, start, end, false)?;
        Ok(out.to_tibs())
    }

    /// Create a Tibs by decoding bytes created via Tibs.encode()
    ///
    /// :return: A new Tibs.
    /// :raises ValueError: for badly formed, truncated or extended input bytes.
    ///
    #[classmethod]
    #[pyo3(signature = (b, /), text_signature = "(cls, b)")]
    pub fn decode(_cls: &Bound<'_, PyType>, b: Vec<u8>) -> PyResult<Tibs> {
        if b.len() == 0 {
            return Err(PyValueError::new_err("Cannot decode an empty bytes."));
        }
        let bv = BV::from_vec(b);
        let single_byte_flag = bv[0];
        let msb0_flag = bv[1];
        if single_byte_flag {
            if bv.len() != 8 {
                return Err(PyValueError::new_err("The encoded sequence has unexpected trailing bytes."));
            }
            for bit_pos in 2..7 {
                if bv[bit_pos] == true {
                    return Ok(Tibs::from_bv(bv[bit_pos + 1..].to_bitvec(), msb0_flag));
                }
            }
            if bv[7] == false {
                return Err(PyValueError::new_err("The encoded sequence is reserved."));
            }
            return Ok(Tibs::empty(msb0_flag));
        }
        let short_form_flag = bv[2];
        if short_form_flag {
            let length_minus_6 = bv[3..8].load_be::<u8>() as usize;
            let bit_length = length_minus_6 + 6;
            if bv.len() < bit_length + 8 {
                return Err(PyValueError::new_err("The encoded sequence ended unexpectedly."));
            }
            if bv.len() != (1 + (bit_length + 7) / 8) * 8 {
                return Err(PyValueError::new_err("The encoded sequence has unexpected trailing bytes."));
            }
            // TODO: Should we check that padding bits are all zeros?
            return Ok(Tibs::from_bv(bv[8..8 + bit_length].to_bitvec(), msb0_flag));

        }

        let codec = bv[3..5].load_be::<u8>();
        let bit_padding = bv[5..8].load_be::<u8>() as usize;

        let (byte_length, varint_bits) = Self::decode_varint(&bv[8..])?;
        let data_start = 8 + varint_bits;
        let data_bits = byte_length
            .checked_mul(8)
            .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
        match codec {
            0 => Self::decode_raw_long(&bv, msb0_flag, bit_padding, data_start, data_bits),
            1 => Self::decode_rice_long(&bv, msb0_flag, bit_padding, data_start, data_bits),
            _ => Err(PyValueError::new_err("The codec value is reserved.")),
        }
    }


    /// Encode the tibs as a bytes instance.
    ///
    /// The bit length and the bit indexing are stored in the encoded bytes.
    ///
    /// The bytes instance can be used to recreate the Tibs exactly -
    /// see :meth:`Tibs.decode`.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> t = Tibs([1, 0, 1, 1, 1], bit_indexing=BitIndexing.Lsb0)
    ///     >>> b = t.encode()
    ///     >>> b
    ///     b'\xb7'
    ///     >>> Tibs.decode(b)
    ///     Tibs('0b10111', BitIndexing.Lsb0)
    ///
    #[pyo3(signature = (codec=Codec::Auto), text_signature = "($self, codec=Codec.Auto)")]
    pub fn encode(&self, codec: Option<Codec>) -> Vec<u8> {
        let bit_length = self.len();
        let mut bv: BV = BV::new();
        match bit_length {
            0..=5 => {
                bv.push(true);  // single_byte_flag
                bv.push(self.msb0);
                let leading_zeros = 5 - bit_length;
                for _ in 0..leading_zeros {
                    bv.push(false);
                }
                bv.push(true);
                bv.extend_from_bitslice(self.to_bitslice());
                bv.into_vec()
            },
            6..=37 => {
                bv.push(false);  // single_byte_flag
                bv.push(self.msb0);
                bv.push(true);  // short_form_flag
                let length_minus_6 = (bit_length - 6) as u8;
                for shift in (0..5).rev() {
                    bv.push((length_minus_6 >> shift) & 1 == 1);
                }
                bv.extend(self.to_bitvec());
                let padding_bits = 8 - bv.len() % 8;
                if padding_bits != 8 {
                    for _ in 0..padding_bits {
                        bv.push(false);
                    }
                }
                bv.into_vec()
            },
            38.. => {
                bv.push(false);  // single_byte_flag
                bv.push(self.msb0);
                bv.push(false);  // short_form_flag
                match codec.unwrap_or(Codec::Auto) {
                    Codec::Auto => {
                        let raw_bit_length = Self::raw_encoded_bit_length(bit_length);
                        let mut best_codec = Codec::Raw;
                        let mut best_bit_length = raw_bit_length;
                        let mut sparse_bit = false;

                        if bit_length <= 128 {
                            let ones_count = <Tibs as BitCollection>::count(self, true);
                            sparse_bit = ones_count < bit_length / 2;
                            let rice_bit_length = self.rice_encoded_bit_length(sparse_bit);
                            if rice_bit_length < best_bit_length {
                                best_codec = Codec::Rice;
                                best_bit_length = rice_bit_length;
                            }
                        }

                        let zstd_compressed = Self::zstd_compress_bytes(self);
                        let zstd_bit_length =
                            5 + Self::encode_varint(zstd_compressed.len() as u64).len() + zstd_compressed.len() * 8;

                        if zstd_bit_length < best_bit_length {
                            bv.extend(self.encode_as_zstd_from_compressed(zstd_compressed));
                        } else {
                            match best_codec {
                                Codec::Raw => bv.extend(self.encode_as_raw()),
                                Codec::Rice => bv.extend(self.encode_as_rice(sparse_bit)),
                                Codec::Auto | Codec::Zstd => unreachable!(),
                            }
                        }
                    },
                    Codec::Raw => {
                        bv.extend(self.encode_as_raw());
                    },
                    Codec::Rice => {
                        let sparse_bit = <Tibs as BitCollection>::count(self, true) < self.len() / 2;
                        bv.extend(self.encode_as_rice(sparse_bit));
                    },
                    Codec::Zstd => {
                        bv.extend(self.encode_as_zstd());
                    },
                }
                bv.into_vec()
            }
        }
    }


    /// Return the instance with every bit inverted.
    ///
    /// :raises ValueError: if the Tibs is empty.
    ///
    pub fn __invert__(&self) -> PyResult<Self> {
        if self.to_bitslice().is_empty() {
            return Err(PyValueError::new_err("Cannot invert empty Tibs."));
        }
        Ok(Tibs::from_bv(self.to_bitvec().not(), self.msb0))
    }

    /// Return the Tibs as a bytes object.
    ///
    /// :raises ValueError: if the length is not a multiple of 8.
    pub fn __bytes__(&self) -> PyResult<Vec<u8>> {
        self.to_bytes()
    }

    /// Return new Tibs consisting of n concatenations of self.
    ///
    /// Called for expression of the form 'a = b*3'.
    ///
    /// n -- The number of concatenations. Must be >= 0.
    ///
    pub fn __mul__(&self, n: i64) -> PyResult<Self> {
        if n < 0 {
            return Err(PyValueError::new_err(
                "Cannot multiply by a negative integer.",
            ));
        }
        Ok(self.multiply(n as usize))
    }

    /// Return Tibs consisting of n concatenations of self.
    ///
    /// Called for expressions of the form 'a = 3*b'.
    ///
    /// n -- The number of concatenations. Must be >= 0.
    ///
    pub fn __rmul__(&self, n: i64) -> PyResult<Self> {
        self.__mul__(n)
    }

    /// Item assignment is not supported for immutable Tibs objects.
    pub fn __setitem__(&self, _key: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "Tibs objects do not support item assignment. Did you mean to use the Mutibs class? Call to_mutibs() to convert to a Mutibs.",
        ))
    }

    /// Item deletion is not supported for immutable Tibs objects.
    pub fn __delitem__(&self, _key: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "Tibs objects do not support item deletion. Did you mean to use the Mutibs class? Call to_mutibs() to convert to a Mutibs.",
        ))
    }
}
