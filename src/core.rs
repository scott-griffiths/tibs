use crate::helpers::{
    BS, BV, FAST_INT_BITS, LogicalOp, bin_from_padded_bytes, bv_from_zeros, byte_order_name,
    copy_unaligned_padded_bytes, count_bitslice, count_pair_bits, for_each_pair_word,
    for_each_pair_word_bitslice, head_bit_offset, hex_from_padded_bytes,
    logical_op_with_aligned_bytes, logical_op_with_matching_bytes, mask_padding_bits,
    normalize_split_position, oct_from_padded_bytes, reverse_padded_bits, validate_index,
    validate_slice,
};
use crate::mutibs::Mutibs;
use crate::tibs_::Tibs;
use bitvec::prelude::*;
use half::f16;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyInt};
use std::borrow::Cow;
use std::fmt;

// Trait used for commonality between the Tibs and Mutibs structs.
pub(crate) trait BitCollection: Sized + Clone {
    fn from_bv(bv: BV) -> Self;
    fn to_bitvec(&self) -> BV;
    fn as_bitslice(&self) -> &BS;
    fn get_slice_unchecked(&self, start_bit: usize, length: usize) -> Self;

    fn get_raw_bytes(&self) -> Vec<u8>;

    fn raw_data_ref(&self) -> Option<(&[u8], usize, usize)> {
        None
    }

    fn raw_data(&self) -> (Vec<u8>, usize, usize) {
        let raw_bytes = self.get_raw_bytes();
        let offset = head_bit_offset(self.as_bitslice());
        (raw_bytes, offset, self.len())
    }

    #[inline]
    fn logical_op(&self, other: &impl BitCollection, op: LogicalOp) -> Self {
        debug_assert!(self.len() == other.len());

        let (Some((lhs, lhs_offset, _)), Some((rhs, rhs_offset, _))) =
            (self.raw_data_ref(), other.raw_data_ref())
        else {
            let mut result = self.to_bitvec();
            op.bitslice(&mut result, other.as_bitslice());
            return Self::from_bv(result);
        };

        let data = if lhs_offset == rhs_offset {
            logical_op_with_matching_bytes(lhs, rhs, op)
        } else {
            logical_op_with_aligned_bytes(lhs, lhs_offset, rhs, rhs_offset, op)
        };
        Self::from_bv(BV::from_vec(data)).get_slice_unchecked(lhs_offset, self.len())
    }

    /// The number of set bits in `op(self, other)`, without building it.
    fn pairwise_count(&self, other: &impl BitCollection, op: LogicalOp) -> usize {
        debug_assert!(self.len() == other.len());
        match (self.raw_data_ref(), other.raw_data_ref()) {
            (Some((lhs, lhs_offset, _)), Some((rhs, rhs_offset, _))) => {
                count_pair_bits(lhs, lhs_offset, rhs, rhs_offset, self.len(), op)
            }
            _ => {
                let mut count = 0;
                for_each_pair_word_bitslice(self.as_bitslice(), other.as_bitslice(), op, |word| {
                    count += word.count_ones() as usize;
                    true
                });
                count
            }
        }
    }

    /// Whether `op(self, other)` has any set bit, stopping at the first one.
    fn pairwise_any(&self, other: &impl BitCollection, op: LogicalOp) -> bool {
        debug_assert!(self.len() == other.len());
        let mut found = false;
        let test = |word: u64| {
            found = word != 0;
            !found
        };
        match (self.raw_data_ref(), other.raw_data_ref()) {
            (Some((lhs, lhs_offset, _)), Some((rhs, rhs_offset, _))) => {
                for_each_pair_word(lhs, lhs_offset, rhs, rhs_offset, self.len(), op, test)
            }
            _ => for_each_pair_word_bitslice(self.as_bitslice(), other.as_bitslice(), op, test),
        }
        found
    }

    /// Read the bits of `self` where `mask` is set, compacted into a new
    /// bit vector of length `mask.count_ones()` (the PEXT operation). `self`
    /// and `mask` must be the same length.
    fn extract_masked(&self, mask: &impl BitCollection) -> BV {
        debug_assert!(self.len() == mask.len());
        let bits = self.as_bitslice();
        let mask_bits = mask.as_bitslice();
        let mut out = BV::with_capacity(mask_bits.count_ones());
        // Walking the mask's set positions skips its zero runs.
        for pos in mask_bits.iter_ones() {
            out.push(bits[pos]);
        }
        out
    }

    #[inline]
    fn logical_or(&self, other: &impl BitCollection) -> Self {
        self.logical_op(other, LogicalOp::Or)
    }

    #[inline]
    fn logical_and(&self, other: &impl BitCollection) -> Self {
        self.logical_op(other, LogicalOp::And)
    }

    #[inline]
    fn logical_xor(&self, other: &impl BitCollection) -> Self {
        self.logical_op(other, LogicalOp::Xor)
    }

    #[inline]
    fn map_slice<R>(
        &self,
        start: Option<isize>,
        end: Option<isize>,
        f: impl FnOnce(&Self) -> PyResult<R>,
    ) -> PyResult<R> {
        if start.is_none() && end.is_none() {
            return f(self);
        }
        let (start, end) = validate_slice(self.len(), start, end)?;
        f(&self.get_slice_unchecked(start, end - start))
    }

    fn to_string(&self) -> String {
        if self.is_empty() {
            return "".to_string();
        }
        const MAX_BITS_TO_PRINT: usize = 10000;
        const {
            assert!(MAX_BITS_TO_PRINT.is_multiple_of(4));
        }
        if self.len() <= MAX_BITS_TO_PRINT {
            match self.to_hexadecimal() {
                Ok(hex) => format!("0x{}", hex),
                Err(_) => format!("0b{}", self.to_binary()),
            }
        } else {
            format!(
                "0x{}... # length={}",
                self.get_slice_unchecked(0, MAX_BITS_TO_PRINT)
                    .to_hexadecimal()
                    .unwrap(),
                self.len()
            )
        }
    }

    fn starts_with(&self, prefix: impl BitCollection) -> bool {
        let n = prefix.len();
        if n <= self.len() {
            *prefix.as_bitslice() == self.as_bitslice()[..n]
        } else {
            false
        }
    }

    #[inline]
    fn empty() -> Self {
        Self::from_bv(BV::new())
    }

    fn ends_with(&self, suffix: impl BitCollection) -> bool {
        let n = suffix.len();
        if n <= self.len() {
            *suffix.as_bitslice() == self.as_bitslice()[self.len() - n..]
        } else {
            false
        }
    }

    /// Returns the bool value at a given bit index.
    #[inline]
    fn get_index(&self, bit_index: isize) -> PyResult<bool> {
        let index = validate_index(bit_index, self.len())?;
        Ok(self.as_bitslice()[index])
    }

    fn get_slice_with_step(&self, start_bit: isize, end_bit: isize, step: isize) -> PyResult<Self> {
        if step == 0 {
            return Err(PyValueError::new_err(
                "Slice step cannot be zero.".to_string(),
            ));
        }
        // Note that a start_bit or end_bit of -1 means to stop at the beginning when using a negative step.
        // Otherwise they should both be positive indices.
        debug_assert!(start_bit >= -1);
        debug_assert!(end_bit >= -1);
        debug_assert!(step != 0);
        if start_bit < -1 || end_bit < -1 {
            return Err(PyValueError::new_err(
                "Indices less than -1 are not valid values.".to_string(),
            ));
        }
        if step > 0 {
            if start_bit >= end_bit {
                return Ok(BitCollection::empty());
            }
            if end_bit as usize > self.len() {
                return Err(PyValueError::new_err(
                    "Slice end goes past the end of the container.".to_string(),
                ));
            }
            Ok(Self::from_bv(
                self.as_bitslice()[start_bit as usize..end_bit as usize]
                    .iter()
                    .step_by(step as usize)
                    .collect(),
            ))
        } else {
            if start_bit <= end_bit || start_bit == -1 {
                return Ok(BitCollection::empty());
            }
            if start_bit as usize > self.len() {
                return Err(PyValueError::new_err(
                    "Slice start bit is past the end of the container.".to_string(),
                ));
            }
            // For negative step, the end_bit is inclusive, but the start_bit is exclusive.
            debug_assert!(step < 0);
            let adjusted_end_bit = (end_bit + 1) as usize;
            Ok(Self::from_bv(
                self.as_bitslice()[adjusted_end_bit..=start_bit as usize]
                    .iter()
                    .rev()
                    .step_by(-step as usize)
                    .collect(),
            ))
        }
    }

    fn count(&self, count_ones: bool) -> usize {
        count_bitslice(self.as_bitslice(), count_ones)
    }

    fn multiply(&self, n: usize) -> Self {
        let len = self.len();
        if n == 0 || len == 0 {
            return BitCollection::empty();
        }
        let mut bv = BV::with_capacity(len * n);
        bv.extend_from_bitslice(self.as_bitslice());

        let mut copies = 1;
        while copies <= n / 2 {
            let current = bv.clone();
            bv.extend_from_bitslice(&current);
            copies *= 2;
        }
        while copies < n {
            bv.extend_from_bitslice(self.as_bitslice());
            copies += 1;
        }
        Self::from_bv(bv)
    }

    fn collect_chunks(&self, chunk_size: i64, count: Option<i64>) -> PyResult<Vec<Self>> {
        if chunk_size <= 0 {
            return Err(PyValueError::new_err(format!(
                "Cannot create chunk list - chunk_size of {chunk_size} given, but it must be > 0."
            )));
        }
        let max_chunks = match count {
            Some(c) => {
                if c < 0 {
                    return Err(PyValueError::new_err(format!(
                        "Cannot create chunk list - count of {c} given, but it must be > 0."
                    )));
                }
                c as usize
            }
            None => usize::MAX,
        };

        let bits_len = self.len();
        let chunk_size = chunk_size as usize;
        let mut current_pos = 0;
        let mut chunks_generated = 0;
        let mut chunks = Vec::new();

        while chunks_generated < max_chunks {
            if current_pos >= bits_len {
                break;
            }
            let take = std::cmp::min(chunk_size, bits_len - current_pos);
            let start = current_pos;
            chunks.push(self.get_slice_unchecked(start, take));
            current_pos += take;
            chunks_generated += 1;
        }

        Ok(chunks)
    }

    fn collect_split_at(&self, pos: &Bound<'_, PyAny>) -> PyResult<Vec<Self>> {
        let len = self.len();
        let positions = if let Ok(position) = pos.extract::<isize>() {
            vec![normalize_split_position(position, len)?]
        } else {
            let capacity = pos.len().ok().unwrap_or(1);
            let mut positions = Vec::with_capacity(capacity);
            for item in pos.try_iter()? {
                let position = item?.extract::<isize>()?;
                positions.push(normalize_split_position(position, len)?);
            }
            positions
        };

        let mut pieces = Vec::with_capacity(positions.len() + 1);
        let mut start = 0;
        for position in positions {
            if position < start {
                return Err(PyValueError::new_err(
                    "Split positions must be in nondecreasing order.",
                ));
            }
            pieces.push(self.get_slice_unchecked(start, position - start));
            start = position;
        }
        pieces.push(self.get_slice_unchecked(start, len - start));
        Ok(pieces)
    }

    fn lshift(&self, n: usize) -> Self {
        if n == 0 {
            return self.clone();
        }
        let len = self.len();
        if n >= len {
            return Self::from_bv(bv_from_zeros(len));
        }

        let data = self.padded_byte_data_cow();
        let mut result = vec![0u8; data.len()];
        let byte_shift = n / 8;
        let bit_shift = n & 7;
        if bit_shift == 0 {
            result[..data.len() - byte_shift].copy_from_slice(&data[byte_shift..]);
        } else {
            let right_shift = 8 - bit_shift;
            let source = &data[byte_shift..];
            for (byte, pair) in result.iter_mut().zip(source.windows(2)) {
                *byte = pair[0] << bit_shift | pair[1] >> right_shift;
            }
            if let Some((&last, byte)) = source.last().zip(result.get_mut(source.len() - 1)) {
                *byte = last << bit_shift;
            }
        }
        mask_padding_bits(&mut result, len);
        let mut result = BV::from_vec(result);
        result.truncate(len);
        Self::from_bv(result)
    }

    fn rshift(&self, n: usize) -> Self {
        if n == 0 {
            return self.clone();
        }
        let len = self.len();
        if n >= len {
            return Self::from_bv(bv_from_zeros(len));
        }

        let data = self.padded_byte_data_cow();
        let mut result = vec![0u8; data.len()];
        let byte_shift = n / 8;
        let bit_shift = n & 7;
        if bit_shift == 0 {
            result[byte_shift..].copy_from_slice(&data[..data.len() - byte_shift]);
        } else {
            let left_shift = 8 - bit_shift;
            let source = &data[..data.len() - byte_shift];
            let output = &mut result[byte_shift..];
            output[0] = source[0] >> bit_shift;
            for (byte, pair) in output[1..].iter_mut().zip(source.windows(2)) {
                *byte = pair[0] << left_shift | pair[1] >> bit_shift;
            }
        }
        mask_padding_bits(&mut result, len);
        let mut result = BV::from_vec(result);
        result.truncate(len);
        Self::from_bv(result)
    }

    /// Return a bit reversed copy
    fn reverse_copy(&self) -> Self {
        let len = self.len();
        if len < 2 {
            return self.clone();
        }
        let mut bytes = self.to_padded_byte_data();
        reverse_padded_bits(&mut bytes, len);
        let mut bv = BV::from_vec(bytes);
        bv.truncate(len);
        Self::from_bv(bv)
    }

    fn invert_copy(&self) -> Self {
        let len = self.len();
        let mut bytes = self.to_padded_byte_data();
        bytes.iter_mut().for_each(|byte| *byte = !*byte);
        mask_padding_bits(&mut bytes, len);
        let mut result = BV::from_vec(bytes);
        result.truncate(len);
        Self::from_bv(result)
    }

    /// Return a byte swapped copy
    fn byte_swap_copy(&self, byte_length: Option<i64>) -> PyResult<Self> {
        let len = self.len();
        if !len.is_multiple_of(8) {
            return Err(PyValueError::new_err(format!(
                "Bit length must be a multiple of 8 to use byte_swap (got length of {len} bits). This error can also be caused by using a byte-order modifier on non-whole byte data."
            )));
        }
        let byte_length = byte_length.unwrap_or((len as i64) / 8);
        if byte_length == 0 && len == 0 {
            return Ok(BitCollection::empty());
        }
        if byte_length <= 0 {
            return Err(PyValueError::new_err(format!(
                "Need a positive byte length for byte_swap. Received '{byte_length}'."
            )));
        }
        let byte_length = byte_length as usize;
        let self_byte_length = len / 8;
        if !self_byte_length.is_multiple_of(byte_length) {
            return Err(PyValueError::new_err(format!(
                "The data to byte_swap is {self_byte_length} bytes long, but it needs to be a multiple of {byte_length} bytes."
            )));
        }

        let mut bytes = self.to_byte_data()?;
        for chunk in bytes.chunks_mut(byte_length) {
            chunk.reverse();
        }
        Ok(BitCollection::from_bv(BV::from_vec(bytes)))
    }

    #[inline]
    fn to_binary(&self) -> String {
        bin_from_padded_bytes(&self.left_aligned_byte_data(), self.len())
    }

    #[inline]
    fn to_octal(&self) -> PyResult<String> {
        let len = self.len();
        if !len.is_multiple_of(3) {
            return Err(PyValueError::new_err(format!(
                "Cannot interpret as octal - length of {} is not a multiple of 3 bits.",
                len
            )));
        }
        Ok(self.build_oct_string())
    }

    #[inline]
    fn to_hexadecimal(&self) -> PyResult<String> {
        let len = self.len();
        if !len.is_multiple_of(4) {
            return Err(PyValueError::new_err(format!(
                "Cannot interpret as hex - length of {} is not a multiple of 4 bits.",
                len
            )));
        }
        Ok(self.build_hex_string())
    }

    #[inline]
    fn build_oct_string(&self) -> String {
        debug_assert!(self.len().is_multiple_of(3));
        oct_from_padded_bytes(&self.left_aligned_byte_data(), self.len())
    }

    #[inline]
    fn build_hex_string(&self) -> String {
        debug_assert!(self.len().is_multiple_of(4));
        hex_from_padded_bytes(&self.left_aligned_byte_data(), self.len())
    }

    /// The bits left aligned into whole bytes, borrowed when the storage
    /// already starts on a byte boundary and copied into place when it does
    /// not. The padding bits of the final byte are left as they are, so
    /// callers must only read the first `self.len()` bits.
    #[inline]
    fn left_aligned_byte_data(&self) -> Cow<'_, [u8]> {
        match self.byte_aligned_raw_data() {
            Some(bytes) => Cow::Borrowed(bytes),
            None => Cow::Owned(self.to_padded_byte_data()),
        }
    }

    #[inline]
    fn to_byte_data(&self) -> PyResult<Vec<u8>> {
        let len_bits = self.len();
        if !len_bits.is_multiple_of(8) {
            return Err(PyValueError::new_err(format!(
                "Cannot interpret as bytes - length of {len_bits} is not a multiple of 8 bits."
            )));
        }
        Ok(self.to_padded_byte_data())
    }

    #[inline]
    fn to_padded_byte_data(&self) -> Vec<u8> {
        let len_bits = self.len();
        if len_bits == 0 {
            return Vec::new();
        }

        if let Some(bytes) = self.byte_aligned_raw_data() {
            let mut out = bytes.to_vec();
            mask_padding_bits(&mut out, len_bits);
            return out;
        }
        if let Some((bytes, bit_offset, _)) = self.raw_data_ref()
            && bit_offset != 0
        {
            let mut out = vec![0u8; len_bits.div_ceil(8)];
            copy_unaligned_padded_bytes(bytes, bit_offset, len_bits, &mut out);
            return out;
        }

        let new_len = (len_bits + 7) & !7;
        let mut bv = BV::with_capacity(new_len);
        bv.extend_from_bitslice(self.as_bitslice());
        bv.resize(new_len, false);
        bv.into_vec()
    }

    fn padded_byte_data_cow(&self) -> Cow<'_, [u8]> {
        if self.len().is_multiple_of(8)
            && let Some(bytes) = self.byte_aligned_raw_data()
        {
            Cow::Borrowed(bytes)
        } else {
            Cow::Owned(self.to_padded_byte_data())
        }
    }

    #[inline]
    fn to_py_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let len_bits = self.len();
        if !len_bits.is_multiple_of(8) {
            return Err(PyValueError::new_err(format!(
                "Cannot interpret as bytes - length of {len_bits} is not a multiple of 8 bits."
            )));
        }
        self.to_padded_py_bytes(py)
    }

    #[inline]
    fn to_padded_py_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        if self.is_empty() {
            return Ok(PyBytes::new(py, &[]).unbind());
        }
        let len_bits = self.len();
        if let Some(bytes) = self.byte_aligned_raw_data() {
            if len_bits.is_multiple_of(8) {
                return Ok(PyBytes::new(py, bytes).unbind());
            }
            return PyBytes::new_with(py, bytes.len(), |out| {
                out.copy_from_slice(bytes);
                mask_padding_bits(out, len_bits);
                Ok(())
            })
            .map(|bytes| bytes.unbind());
        }
        if let Some((bytes, bit_offset, _)) = self.raw_data_ref()
            && bit_offset != 0
        {
            return PyBytes::new_with(py, len_bits.div_ceil(8), |out| {
                copy_unaligned_padded_bytes(bytes, bit_offset, len_bits, out);
                Ok(())
            })
            .map(|bytes| bytes.unbind());
        }

        let bytes = self.to_padded_byte_data();
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    #[inline]
    fn byte_aligned_raw_data(&self) -> Option<&[u8]> {
        let (bytes, bit_offset, len_bits) = self.raw_data_ref()?;
        if bit_offset == 0 {
            Some(&bytes[..len_bits.div_ceil(8)])
        } else {
            None
        }
    }

    /// Build a Python int from more bits than fit in a machine word.
    ///
    /// The bits are left padded to a whole number of bytes so that
    /// `int.from_bytes` can do the arithmetic. For a signed reading the pad
    /// repeats the sign bit, which is exactly the sign extension that
    /// `from_bytes(..., signed=True)` then expects.
    fn to_big_int<'py>(
        &self,
        py: Python<'py>,
        is_little_endian: bool,
        signed: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let length = self.len();
        let pad = (8 - length % 8) % 8;
        let bytes = if pad == 0 {
            self.to_padded_byte_data()
        } else {
            // A byte order is only allowed for whole-byte lengths, so padding
            // and little-endian never have to be reconciled.
            debug_assert!(!is_little_endian);
            let mut bv = BV::repeat(signed && self.as_bitslice()[0], pad);
            bv.extend_from_bitslice(self.as_bitslice());
            bv.into_vec()
        };
        let args = (PyBytes::new(py, &bytes), byte_order_name(is_little_endian));
        let int_type = py.get_type::<PyInt>();
        // `signed` is keyword-only and defaults to False, so the unsigned case
        // can skip building a kwargs dict.
        if signed {
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "signed"), true)?;
            int_type.call_method(intern!(py, "from_bytes"), args, Some(&kwargs))
        } else {
            int_type.call_method1(intern!(py, "from_bytes"), args)
        }
    }

    #[inline]
    fn to_uint<'py>(&self, py: Python<'py>, is_little_endian: bool) -> PyResult<Bound<'py, PyAny>> {
        let length = self.len();
        if length == 0 {
            return Err(PyValueError::new_err(
                "Cannot convert to unsigned int when bit length is zero.",
            ));
        }
        if length > FAST_INT_BITS {
            return self.to_big_int(py, is_little_endian, false);
        }
        let raw = if is_little_endian {
            self.as_bitslice().load_le::<u64>()
        } else {
            self.as_bitslice().load_be::<u64>()
        };
        raw.into_bound_py_any(py)
    }

    #[inline]
    fn to_int<'py>(&self, py: Python<'py>, is_little_endian: bool) -> PyResult<Bound<'py, PyAny>> {
        let length = self.len();
        if length == 0 {
            return Err(PyValueError::new_err(
                "Cannot convert to signed int when bit length is zero.",
            ));
        }
        if length > FAST_INT_BITS {
            return self.to_big_int(py, is_little_endian, true);
        }
        let raw = if is_little_endian {
            self.as_bitslice().load_le::<u64>()
        } else {
            self.as_bitslice().load_be::<u64>()
        };

        let shift = FAST_INT_BITS - length;
        (((raw << shift) as i64) >> shift).into_bound_py_any(py)
    }

    fn to_f64(&self, is_little_endian: bool) -> PyResult<f64> {
        let length = self.len();
        match length {
            64 => {
                let bits = if is_little_endian {
                    self.as_bitslice().load_le::<u64>()
                } else {
                    self.as_bitslice().load_be::<u64>()
                };
                Ok(f64::from_bits(bits))
            }
            32 => {
                let bits = if is_little_endian {
                    self.as_bitslice().load_le::<u32>()
                } else {
                    self.as_bitslice().load_be::<u32>()
                };
                Ok(f32::from_bits(bits) as f64)
            }
            16 => {
                let bits = if is_little_endian {
                    self.as_bitslice().load_le::<u16>()
                } else {
                    self.as_bitslice().load_be::<u16>()
                };
                Ok(f16::from_bits(bits).to_f64())
            }
            _ => Err(PyValueError::new_err(format!(
                "Unsupported float bit length '{length}'. Only 16, 32 and 64 are supported."
            ))),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.as_bitslice().is_empty()
    }

    #[inline]
    fn len(&self) -> usize {
        self.as_bitslice().len()
    }

    #[inline]
    fn get_slice(&self, start_bit: usize, length: usize) -> PyResult<Self> {
        if length == 0 {
            return Ok(BitCollection::empty());
        }
        if start_bit + length > self.len() {
            return Err(PyIndexError::new_err(
                "End bit of the slice goes past the end of the container.".to_string(),
            ));
        }
        Ok(self.get_slice_unchecked(start_bit, length))
    }
}

pub(crate) fn concatenate_bitcollections(
    left: &impl BitCollection,
    right: &impl BitCollection,
) -> BV {
    let len = left.len() + right.len();
    if left.len().is_multiple_of(8) {
        let left = left.padded_byte_data_cow();
        let right = right.padded_byte_data_cow();
        let mut bytes = Vec::with_capacity(len.div_ceil(8));
        bytes.extend_from_slice(&left);
        bytes.extend_from_slice(&right);
        let mut result = BV::from_vec(bytes);
        result.truncate(len);
        return result;
    }

    let mut result = BV::with_capacity(len);
    result.extend_from_bitslice(left.as_bitslice());
    result.extend_from_bitslice(right.as_bitslice());
    result
}

impl BitCollection for Tibs {
    fn from_bv(bv: BV) -> Self {
        Tibs::from_bv(bv)
    }

    #[inline]
    fn to_bitvec(&self) -> BV {
        Tibs::to_bitvec(self)
    }

    #[inline]
    fn as_bitslice(&self) -> &BS {
        Tibs::as_bitslice(self)
    }

    #[inline]
    fn get_slice_unchecked(&self, start_bit: usize, length: usize) -> Self {
        Tibs::get_slice_unchecked(self, start_bit, length)
    }

    #[inline]
    fn get_raw_bytes(&self) -> Vec<u8> {
        Tibs::raw_bytes(self)
    }

    #[inline]
    fn raw_data_ref(&self) -> Option<(&[u8], usize, usize)> {
        Tibs::raw_data_ref(self)
    }
}

impl BitCollection for Mutibs {
    fn from_bv(bv: BV) -> Self {
        Mutibs::from_bv(bv)
    }

    #[inline]
    fn to_bitvec(&self) -> BV {
        Mutibs::to_bitvec(self)
    }

    #[inline]
    fn as_bitslice(&self) -> &BS {
        Mutibs::as_bitslice(self)
    }

    #[inline]
    fn get_slice_unchecked(&self, start_bit: usize, length: usize) -> Self {
        Self::from_bv(self.as_bitslice()[start_bit..start_bit + length].to_bitvec())
    }

    #[inline]
    fn get_raw_bytes(&self) -> Vec<u8> {
        Mutibs::raw_bytes(self)
    }

    #[inline]
    fn raw_data_ref(&self) -> Option<(&[u8], usize, usize)> {
        let slice = self.as_bitslice();
        let offset = match slice.domain() {
            bitvec::domain::Domain::Enclave(elem) => elem.head().into_inner() as usize,
            bitvec::domain::Domain::Region {
                head: Some(elem), ..
            } => elem.head().into_inner() as usize,
            _ => 0,
        };
        if offset == 0 {
            Some((self.data.as_raw_slice(), offset, slice.len()))
        } else {
            None
        }
    }
}

impl fmt::Debug for Tibs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len() > 100 {
            return f
                .debug_struct("Tibs")
                .field(
                    "hex",
                    &self.get_slice_unchecked(0, 100).to_hex(None, None).unwrap(),
                )
                .field("length", &self.len())
                .finish();
        }
        if self.len().is_multiple_of(4) {
            return f
                .debug_struct("Tibs")
                .field("hex", &self.to_hex(None, None).unwrap())
                .field("length", &self.len())
                .finish();
        }
        f.debug_struct("Tibs")
            .field("bin", &BitCollection::to_binary(self))
            .field("length", &self.len())
            .finish()
    }
}

impl PartialEq for Tibs {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_bitslice() == other.as_bitslice()
    }
}

impl PartialEq<Mutibs> for Tibs {
    #[inline]
    fn eq(&self, other: &Mutibs) -> bool {
        self.as_bitslice() == other.as_bitvec_ref()
    }
}

impl PartialEq for Mutibs {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_bitvec_ref() == other.as_bitvec_ref()
    }
}

impl PartialEq<Tibs> for Mutibs {
    #[inline]
    fn eq(&self, other: &Tibs) -> bool {
        self.as_bitvec_ref() == other.as_bitslice()
    }
}
