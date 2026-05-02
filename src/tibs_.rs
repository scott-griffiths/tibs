use crate::core::BitCollection;
use crate::enums::{BitOrder, Codec, Endianness};
use crate::helpers;
use crate::helpers::{
    BS, BV, bv_from_bin, bv_from_bools, bv_from_bytes_slice, bv_from_f64, bv_from_hex,
    bv_from_i128, bv_from_oct, bv_from_ones, bv_from_random, bv_from_u128, bv_from_zeros,
    compute_lps, find_bitvec_aligned, promote_to_bv, rfind_bitvec_aligned, str_to_bv,
    validate_logical_op_lengths, validate_shift, validate_slice,
};
use crate::iterator::{BoolIterator, ChunksIterator, FindAllIterator};
use crate::mutibs::Mutibs;
use crate::view::View;
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
    pub(crate) fn from_bv(bv: BV) -> Self {
        let length = bv.len();
        Tibs {
            data: Arc::new(bv),
            offset: 0,
            length,
        }
    }

    pub(crate) fn get_slice_unchecked(&self, offset: usize, length: usize) -> Self {
        Tibs {
            data: self.data.clone(),
            offset: self.offset + offset,
            length,
        }
    }

    #[inline]
    fn shares_view_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
            && self.offset == other.offset
            && self.length == other.length
    }

    #[inline]
    pub(crate) fn as_bitslice(&self) -> &BS {
        &self.data[self.offset..self.offset + self.length]
    }

    #[inline]
    pub(crate) fn to_bitvec(&self) -> BV {
        // Materialize a single owned copy of the current logical view.
        self.as_bitslice().to_bitvec()
    }

    #[inline]
    pub(crate) fn to_bitslice(&self) -> &BS {
        self.as_bitslice()
    }

    #[inline]
    pub(crate) fn raw_bytes(&self) -> Vec<u8> {
        let bit_offset = match self.as_bitslice().domain() {
            bitvec::domain::Domain::Enclave(elem) => elem.head().into_inner() as usize,
            bitvec::domain::Domain::Region {
                head: Some(elem), ..
            } => elem.head().into_inner() as usize,
            _ => 0,
        };
        let physical_start = self.offset;
        let byte_start = physical_start / 8;
        let byte_len = (bit_offset + self.length).div_ceil(8);
        self.data.as_raw_slice()[byte_start..byte_start + byte_len].to_vec()
    }

    #[inline]
    pub(crate) fn raw_data_ref(&self) -> Option<(&[u8], usize, usize)> {
        let data_head_offset = match self.data.as_bitslice().domain() {
            bitvec::domain::Domain::Enclave(elem) => elem.head().into_inner() as usize,
            bitvec::domain::Domain::Region {
                head: Some(elem), ..
            } => elem.head().into_inner() as usize,
            _ => 0,
        };
        if data_head_offset != 0 {
            return None;
        }

        let physical_start = self.offset;
        let byte_start = physical_start / 8;
        let bit_offset = physical_start % 8;
        let byte_len = (bit_offset + self.length).div_ceil(8);
        Some((
            &self.data.as_raw_slice()[byte_start..byte_start + byte_len],
            bit_offset,
            self.length,
        ))
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
        let (start, end) = validate_slice(self.len(), start, end)?;
        let alignment_mod8 = if byte_aligned { Some(0) } else { None };

        let found = if !reverse {
            find_bitvec_aligned(
                self.to_bitslice(),
                needle.as_bitslice(),
                start,
                end,
                alignment_mod8,
            )
        } else {
            rfind_bitvec_aligned(
                self.to_bitslice(),
                needle.as_bitslice(),
                start,
                end,
                alignment_mod8,
            )
        };
        Ok(found)
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
        let bv = promote_to_bv(&obj)?;
        Ok(Tibs::from_bv(bv))
    }
}

/// Public Python-facing methods.
#[pymethods]
impl Tibs {
    #[new]
    #[pyo3(signature = (auto = None), text_signature = "(auto=None)")]
    pub fn py_new(auto: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(auto) = auto else {
            return Ok(BitCollection::empty());
        };
        Tibs::extract(auto.as_borrowed())
    }

    /// Return a new instance with the bits reversed.
    ///
    /// :return: Tibs
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Tibs('0b00011')
    ///     >>> a.reversed()
    ///     Tibs('0b11000')
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
            "Tibs()".to_string()
        } else {
            format!("Tibs('{}')", self.__str__())
        }
    }

    #[pyo3(signature = (byte_order = Endianness::Unspecified, byte_bit_order = BitOrder::Msb0), text_signature = "($self, byte_order=Endianness.Unspecified, byte_bit_order=BitOrder.Msb0)")]
    pub fn view(
        slf: PyRef<'_, Self>,
        byte_order: Option<Endianness>,
        byte_bit_order: Option<BitOrder>,
    ) -> View {
        View::from_tibs(
            slf.into(),
            byte_order.unwrap_or(Endianness::Unspecified),
            byte_bit_order.unwrap_or(BitOrder::Msb0),
        )
    }

    #[getter]
    pub fn le(slf: PyRef<'_, Self>) -> View {
        View::from_tibs(slf.into(), Endianness::Little, BitOrder::Msb0)
    }

    #[getter]
    pub fn be(slf: PyRef<'_, Self>) -> View {
        View::from_tibs(slf.into(), Endianness::Big, BitOrder::Msb0)
    }

    #[getter]
    pub fn lsb0(slf: PyRef<'_, Self>) -> View {
        View::from_tibs(slf.into(), Endianness::Unspecified, BitOrder::Lsb0)
    }

    #[getter]
    pub fn msb0(slf: PyRef<'_, Self>) -> View {
        View::from_tibs(slf.into(), Endianness::Unspecified, BitOrder::Msb0)
    }

    /// Iterate over the bits of the Tibs, yielding each bit as a boolean.
    ///
    /// :return: An iterator yielding bool values.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b101'))
    ///     [True, False, True]
    ///
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

    /// Return a list of Tibs by cutting into chunks.
    ///
    /// :param int chunk_size: The size in bits of the chunks to create.
    /// :param int | None count: If specified, at most count items are created. Default is to cut as many times as possible.
    /// :return: A list of Tibs chunks.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b110011').chunks(2)
    ///     [Tibs('0b11'), Tibs('0b00'), Tibs('0b11')]
    ///
    #[pyo3(signature = (chunk_size, count = None), text_signature = "($self, chunk_size, count=None)")]
    pub fn chunks(&self, chunk_size: i64, count: Option<i64>) -> PyResult<Vec<Self>> {
        BitCollection::collect_chunks(self, chunk_size, count)
    }

    /// Return Tibs generator by cutting into chunks.
    ///
    /// :param int chunk_size: The size in bits of the chunks to generate.
    /// :param int | None count: If specified, at most count items are generated. Default is to cut as many times as possible.
    /// :return: A generator yielding Tibs chunks.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b110011').chunks_iter(2))
    ///     [Tibs('0b11'), Tibs('0b00'), Tibs('0b11')]
    ///
    #[pyo3(signature = (chunk_size, count = None), text_signature = "($self, chunk_size, count=None)")]
    pub fn chunks_iter(
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
    ///     >>> list(Tibs('0b1100111').rchunks_iter(3))
    ///     [Tibs('0b111'), Tibs('0b100'), Tibs('0b11')]
    ///
    #[pyo3(signature = (chunk_size, count = None), text_signature = "($self, chunk_size, count=None)")]
    pub fn rchunks_iter(
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

    /// Find all occurrences of a bit sequence.
    ///
    /// :param Tibs needle: The bit sequence to find.
    /// :param int | None start: The starting bit position of the slice to search. Defaults to 0.
    /// :param int | None end: The end bit position of the slice to search. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries. Defaults to ``False``.
    /// :return: A list of bit positions.
    ///
    /// :raises ValueError: if needle is empty, if start or end are out of range or if end is before start.
    ///
    /// All occurrences of needle are found, even if they overlap.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b10111011').find_all('0b11')
    ///     [2, 3, 6]
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn find_all(
        slf: PyRef<'_, Self>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
    ) -> PyResult<Vec<u64>> {
        if needle.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to find."));
        }

        let haystack_len = slf.len();
        let (start, end) = validate_slice(haystack_len, start, end)?;

        Ok(helpers::collect_find_all_positions(
            slf.as_bitslice(),
            needle.as_bitslice(),
            haystack_len,
            start,
            end,
            byte_aligned,
        ))
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
    ///     >>> list(Tibs('0b10111011').find_all_iter('0b11'))
    ///     [2, 3, 6]
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn find_all_iter(
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
        let haystack_len = slf.len();
        let is_reverse = false;
        let step = if byte_aligned { 8 } else { 1 };
        let alignment_mod8 = if byte_aligned { Some(0) } else { None };
        let (byte_haystack, byte_needle, byte_base) = helpers::byte_search_prep(
            slf.as_bitslice(),
            needle.as_bitslice(),
            start,
            end,
            alignment_mod8,
        )
        .map_or((None, None, 0), |(haystack, needle, base)| {
            (Some(haystack), Some(needle), base)
        });
        let py = slf.py();
        let lps = { compute_lps(needle.to_bitslice()) };
        let iter_obj = FindAllIterator {
            haystack: slf.into(),
            haystack_len,
            needle,
            lps,
            start,
            end,
            byte_aligned,
            step,
            current_pos: if is_reverse { end } else { start },
            is_reverse,
            byte_haystack,
            byte_needle,
            byte_base,
            byte_current: if is_reverse { end / 8 - byte_base } else { 0 },
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
    ///     >>> list(Tibs('0b10111011').rfind_all_iter('0b11'))
    ///     [6, 3, 2]
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn rfind_all_iter(
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
        let haystack_len = slf.len();
        let is_reverse = true;
        let step = if byte_aligned { 8 } else { 1 };
        let alignment_mod8 = if byte_aligned { Some(0) } else { None };
        let (byte_haystack, byte_needle, byte_base) = helpers::byte_search_prep(
            slf.as_bitslice(),
            needle.as_bitslice(),
            start,
            end,
            alignment_mod8,
        )
        .map_or((None, None, 0), |(haystack, needle, base)| {
            (Some(haystack), Some(needle), base)
        });
        let py = slf.py();
        let lps = { compute_lps(needle.to_bitslice()) };
        let iter_obj = FindAllIterator {
            haystack: slf.into(),
            haystack_len,
            needle,
            lps,
            start,
            end,
            byte_aligned,
            step,
            current_pos: if is_reverse { end } else { start },
            is_reverse,
            byte_haystack,
            byte_needle,
            byte_base,
            byte_current: if is_reverse { end / 8 - byte_base } else { 0 },
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
    /// :return: A Tibs object with all bits set to zero.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_zeros(500)  # 500 zero bits
    ///
    #[classmethod]
    #[pyo3(signature = (length, /), text_signature = "(cls, length, /)")]
    pub fn from_zeros(_cls: &Bound<'_, PyType>, length: i64) -> PyResult<Self> {
        if length < 0 {
            return Err(PyValueError::new_err(format!(
                "Negative bit length given: {}.",
                length
            )));
        }
        Ok(Self::from_bv(bv_from_zeros(length as usize)))
    }

    /// Create a new instance with all bits set to '1'.
    ///
    /// :param int length: The number of bits to set.
    /// :return: A Tibs object with all bits set to one.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_ones(5)
    ///     Tibs('0b11111')
    ///
    #[classmethod]
    #[pyo3(signature = (length, /), text_signature = "(cls, length, /)")]
    pub fn from_ones(_cls: &Bound<'_, PyType>, length: i64) -> PyResult<Self> {
        if length < 0 {
            return Err(PyValueError::new_err(format!(
                "Negative bit length given: {}.",
                length
            )));
        }
        Ok(Tibs::from_bv(bv_from_ones(length as usize)))
    }

    /// Create a new instance from a formatted string.
    ///
    /// :param str s: The formatted string to convert. This can begin with '0b', '0o' or '0x' to indicate binary, octal or hexadecimal, and commas can be used to separate items.
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
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_string(_cls: &Bound<'_, PyType>, s: String) -> PyResult<Self> {
        let bv = str_to_bv(s)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Create a new instance from an unsigned integer.
    ///
    /// :param int u: An unsigned integer.
    /// :param int length: The bit length to create. Can be up to 128.
    /// :param Endianness endianness: The byte endianness used to store the integer. Defaults to Endianness.Unspecified.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// :raises ValueError: if the integer doesn't fit in the length given.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_u(15, length=8)
    ///     Tibs('0x0f')
    ///
    #[classmethod]
    #[pyo3(signature = (u, /, length, endianness = Endianness::Unspecified), text_signature = "(cls, u, /, length, endianness=Endianness.Unspecified)")]
    pub fn from_u(
        _cls: &Bound<'_, PyType>,
        u: u128,
        length: i64,
        endianness: Option<Endianness>,
    ) -> PyResult<Self> {
        let is_little_endian = Endianness::is_little_endian(endianness, length as usize)?;
        Ok(Tibs::from_bv(bv_from_u128(u, length, is_little_endian)?))
    }

    /// Return the unsigned integer representation of the Tibs.
    ///
    /// :return: The value as an unsigned integer.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0x0f').to_u()
    ///     15
    ///
    pub fn to_u(&self) -> PyResult<u128> {
        BitCollection::to_u128(self, false)
    }

    /// Read-only property of the unsigned integer representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_u`.
    ///
    /// :return: The value as an unsigned integer.
    #[getter]
    fn u(&self) -> PyResult<u128> {
        self.to_u()
    }

    /// Create a new instance from a signed integer.
    ///
    /// :param int i: A signed integer.
    /// :param int length: The bit length to create. Can be up to 128.
    /// :param Endianness endianness: The byte endianness used to store the integer. Defaults to Endianness.Unspecified.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// :raises ValueError: if the integer doesn't fit in the length given.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_i(-2, length=4)
    ///     Tibs('0xe')
    ///
    #[classmethod]
    #[pyo3(signature = (i, /, length, endianness = Endianness::Unspecified), text_signature = "(cls, i, /, length, endianness=Endianness.Unspecified)")]
    pub fn from_i(
        _cls: &Bound<'_, PyType>,
        i: i128,
        length: i64,
        endianness: Option<Endianness>,
    ) -> PyResult<Self> {
        let is_little_endian = Endianness::is_little_endian(endianness, length as usize)?;
        Ok(Tibs::from_bv(bv_from_i128(i, length, is_little_endian)?))
    }

    /// Return the signed integer representation of the Tibs.
    ///
    /// :return: The value as a signed integer.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0xe').to_i()
    ///     -2
    ///
    pub fn to_i(&self) -> PyResult<i128> {
        BitCollection::to_i128(self, false)
    }

    /// Read-only property of the signed integer representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_i`.
    ///
    /// :return: The value as a signed integer.
    #[getter]
    fn i(&self) -> PyResult<i128> {
        self.to_i()
    }

    /// Create a new instance from a floating point number.
    ///
    /// :param float f: A floating point value.
    /// :param int length: The bit length to create. Must be 16, 32 or 64.
    /// :param Endianness endianness: The byte endianness used to store the float. Defaults to Endianness.Unspecified.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_f(1.5, length=32)
    ///     Tibs('0x3fc00000')
    ///
    #[classmethod]
    #[pyo3(signature = (f, /, length, endianness = Endianness::Unspecified), text_signature = "(cls, f, /, length, endianness=Endianness.Unspecified)")]
    pub fn from_f(
        _cls: &Bound<'_, PyType>,
        f: f64,
        length: i64,
        endianness: Option<Endianness>,
    ) -> PyResult<Self> {
        let is_little_endian = Endianness::is_little_endian(endianness, length as usize)?;
        let bv = bv_from_f64(f, length, is_little_endian)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Return the floating point representation of the Tibs.
    ///
    /// The length must be 16, 32 or 64.
    ///
    /// :return: The value as a Python float.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0x3fc00000').to_f()
    ///     1.5
    ///
    pub fn to_f(&self) -> PyResult<f64> {
        BitCollection::to_f64(self, false)
    }

    /// Read-only property of the floating point representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_f`.
    ///
    /// :return: The value as a Python float.
    #[getter]
    fn f(&self) -> PyResult<f64> {
        self.to_f()
    }

    /// Create a new instance from a binary string.
    ///
    /// :param str s: A string of ``0`` and ``1`` s, optionally preceded with ``0b`` and optionally containing underscores.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_bin("0000_1111_0101")
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_bin(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        let bv = bv_from_bin(s)?;
        Ok(Tibs::from_bv(bv))
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
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_oct("17")
    ///     Tibs('0b001111')
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_oct(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        let bv = bv_from_oct(s)?;
        Ok(Tibs::from_bv(bv))
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
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_hex("0f")
    ///     Tibs('0x0f')
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_hex(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        let bv = bv_from_hex(s)?;
        Ok(Tibs::from_bv(bv))
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
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_bytes(b"some_bytes_maybe_from_a_file")
    ///
    #[classmethod]
    #[inline]
    #[pyo3(signature = (data, /, offset=None, length=None), text_signature = "(cls, data, /, offset=None, length=None)")]
    pub fn from_bytes(
        _cls: &Bound<'_, PyType>,
        data: Vec<u8>,
        offset: Option<i64>,
        length: Option<i64>,
    ) -> PyResult<Self> {
        let bv = bv_from_bytes_slice(data, offset, length)?;
        Ok(Self::from_bv(bv))
    }

    /// Create a new instance from an iterable by converting each element to a bool.
    ///
    /// :param Iterable iterable: The iterable to convert to a :class:`Tibs`.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_bools([False, 0, 1, "Steven"])  # binary 0011
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /), text_signature = "(cls, iterable, /)")]
    pub fn from_bools(_cls: &Bound<'_, PyType>, iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bv = bv_from_bools(iterable)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Create a new instance with all bits randomly set.
    ///
    /// :param int length: The number of bits to set. Must be positive.
    /// :param bool secure: If ``True``, use the OS's cryptographically secure generator. Default is ``False``.
    /// :param bytes | bytearray | None seed: A bytes or bytearray to use as an optional seed, only if ``secure`` is ``False``.
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
    #[pyo3(signature = (length, /, secure=false, seed=None), text_signature="(cls, length, /, secure=False, seed=None)")]
    pub fn from_random(
        _cls: &Bound<'_, PyType>,
        length: i64,
        secure: bool,
        seed: Option<Vec<u8>>,
    ) -> PyResult<Self> {
        let bv = bv_from_random(length, secure, &seed)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Create a new instance by concatenating a sequence of Tibs objects.
    ///
    /// This method concatenates a sequence of Tibs objects into a single Tibs object.
    ///
    /// :param Iterable iterable: An iterable to concatenate. Items can be anything that can be promoted to a Tibs.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_joined(['0x01', [1, 0], b'some_bytes'])
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /), text_signature = "(cls, iterable, /)")]
    pub fn from_joined(_cls: &Bound<'_, PyType>, iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Build the immutable result directly; going through Mutibs::as_tibs
        // would move through an unnecessary mutable wrapper.
        Ok(Tibs::from_bv(Mutibs::joined_bv_from_iterable(iterable)?))
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
    ///
    /// .. code-block:: pycon
    ///
    ///      >>> Tibs('0b10111011').rfind('0b11')
    ///      6
    ///
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
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0xef').count(1)
    ///     7
    ///     >>> Tibs.from_bin('0011010101100').count('0b01')
    ///     4
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
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_zeros(5).set_at([1, 3])
    ///     Tibs('0b01010')
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
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_ones(5).unset_at([1, 3])
    ///     Tibs('0b10101')
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
    /// :param int | Iterable[int] | None pos: Either a single bit position, an iterable of bit positions,
    ///   or None to invert every bit. Defaults to None.
    /// :return: A new Tibs.
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b10110').inverted([0, 2])
    ///     Tibs('0b00010')
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
    /// :param int pos: The bit position to insert at. Clips to the start or end if out of range.
    /// :param Tibs bs: The bits to insert.
    /// :return: A new Tibs.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b1011').inserted(2, '0b00')
    ///     Tibs('0b100011')
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
    ///
    /// :param Tibs old: The bits to search for.
    /// :param Tibs new: The bits to replace with.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param int | None count: If present, the maximum number of replacements to make.
    /// :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
    /// :return: A new Tibs.
    /// :raises ValueError: if old is empty, count is negative or the slice parameters are invalid.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b00010010').replaced([0, 1], [1, 1, 1])
    ///     Tibs('0b0011101110')
    ///
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
    /// :return: A new Mutibs with the same bit data.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> t = Tibs.from_hex('abc')
    ///     >>> m = t.to_mutibs()
    ///     >>> m *= 4
    ///     >>> print(t.hex)
    ///     abc
    ///     >>> print(m.hex)
    ///     abcabcabcabc
    ///
    pub fn to_mutibs(&self) -> Mutibs {
        Mutibs::from_bv(self.to_bitvec())
    }

    #[inline]
    /// Get a bit or a slice of bits.
    ///
    /// :param int | slice key: The index or slice to get.
    /// :return: A bool for a single index, or a new Tibs for a slice.
    /// :raises IndexError: If the index is out of range.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> t = Tibs('0b101100')
    ///     >>> t[0]
    ///     True
    ///     >>> t[1:4]
    ///     Tibs('0b011')
    ///
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
                    Tibs::empty()
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
    /// :param int n: The number of bits to shift. Must be >= 0.
    /// :return: A new Tibs.
    /// :raises ValueError: if n < 0.
    ///
    pub fn __lshift__(&self, n: i64) -> PyResult<Self> {
        let shift = validate_shift(self, n)?;
        Ok(self.lshift(shift))
    }

    /// Return new Tibs shifted by n to the right.
    ///
    /// :param int n: The number of bits to shift. Must be >= 0.
    /// :return: A new Tibs.
    /// :raises ValueError: if n < 0.
    ///
    pub fn __rshift__(&self, n: i64) -> PyResult<Self> {
        let shift = validate_shift(self, n)?;
        Ok(self.rshift(shift))
    }

    /// Concatenates two Tibs and return a newly constructed Tibs.
    ///
    /// :param Tibs other: The bits to append.
    /// :return: A new Tibs.
    ///
    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        let mut data = BV::with_capacity(self.len() + other.len());
        data.extend_from_bitslice(self.to_bitslice());
        data.extend_from_bitslice(other.as_bitslice());
        Ok(Tibs::from_bv(data))
    }

    /// Concatenates two Tibs and return a newly constructed Tibs.
    ///
    /// :param Tibs other: The bits to prepend.
    /// :return: A new Tibs.
    ///
    pub fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        let mut data = BV::with_capacity(other.len() + self.len());
        data.extend_from_bitslice(other.as_bitslice());
        data.extend_from_bitslice(self.to_bitslice());
        Ok(Tibs::from_bv(data))
    }

    /// Bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// :param Tibs other: The other bits.
    /// :return: A new Tibs.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        if self.shares_view_with(&other) {
            return Ok(self.clone());
        }
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_and(self, &other))
    }

    /// Bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// :param Tibs other: The other bits.
    /// :return: A new Tibs.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        if self.shares_view_with(&other) {
            return Ok(self.clone());
        }
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_or(self, &other))
    }

    /// Bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// :param Tibs other: The other bits.
    /// :return: A new Tibs.
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
    /// :param int n: The number of bits to rotate by.
    /// :param int | None start: Start of slice to rotate. Defaults to 0.
    /// :param int | None end: End of slice to rotate. Defaults to len(self).
    /// :return: A new Tibs.
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b10110').rotated_left(2)
    ///     Tibs('0b11010')
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
    /// :param int n: The number of bits to rotate by.
    /// :param int | None start: Start of slice to rotate. Defaults to 0.
    /// :param int | None end: End of slice to rotate. Defaults to len(self).
    /// :return: A new Tibs.
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b10110').rotated_right(1)
    ///     Tibs('0b01011')
    ///
    #[pyo3(signature = (n, start=None, end=None), text_signature = "($self, n, start=None, end=None)")]
    pub fn rotated_right(
        &self,
        n: i64,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Self> {
        let mut out = self.to_mutibs();
        out.apply_rotation(n, start, end, false)?;
        Ok(out.to_tibs())
    }

    /// Create a Tibs by decoding bytes created via Tibs.encode()
    ///
    /// :param bytes | bytearray b: The encoded bytes to decode.
    /// :return: A new Tibs.
    /// :raises ValueError: for badly formed, truncated or extended input bytes.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.decode(Tibs('0b101').encode())
    ///     Tibs('0b101')
    ///
    #[classmethod]
    #[pyo3(signature = (b, /), text_signature = "(cls, b, /)")]
    pub fn decode(_cls: &Bound<'_, PyType>, b: Vec<u8>) -> PyResult<Tibs> {
        <Tibs as BitCollection>::decode_bytes(b)
    }

    /// Encode the tibs as a bytes instance.
    ///
    /// The bit length and the bit indexing are stored in the encoded bytes.
    ///
    /// The bytes instance can be used to recreate the Tibs exactly -
    /// see :meth:`Tibs.decode`.
    ///
    /// :param Codec codec: The codec to use. Defaults to Codec.Auto.
    /// :return: The encoded bytes.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> b = t.encode()
    ///     >>> b
    ///     b'\xb7'
    ///     >>> Tibs.decode(b)
    ///
    #[pyo3(signature = (codec=Codec::Auto), text_signature = "($self, codec=Codec.Auto)")]
    pub fn encode(&self, codec: Option<Codec>) -> PyResult<Vec<u8>> {
        <Tibs as BitCollection>::encode(self, codec)
    }

    /// Return the instance with every bit inverted.
    ///
    /// :return: A new Tibs.
    /// :raises ValueError: if the Tibs is empty.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> ~Tibs('0b10110')
    ///     Tibs('0b01001')
    ///
    pub fn __invert__(&self) -> PyResult<Self> {
        if self.to_bitslice().is_empty() {
            return Err(PyValueError::new_err("Cannot invert empty Tibs."));
        }
        Ok(Tibs::from_bv(self.to_bitvec().not()))
    }

    /// Return the Tibs as a bytes object.
    ///
    /// :return: The bytes representation.
    /// :raises ValueError: if the length is not a multiple of 8.
    pub fn __bytes__(&self) -> PyResult<Vec<u8>> {
        self.to_bytes()
    }

    /// Return new Tibs consisting of n concatenations of self.
    ///
    /// Called for expression of the form 'a = b*3'.
    ///
    /// :param int n: The number of concatenations. Must be >= 0.
    /// :return: A new Tibs.
    /// :raises ValueError: if n < 0.
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
    /// :param int n: The number of concatenations. Must be >= 0.
    /// :return: A new Tibs.
    /// :raises ValueError: if n < 0.
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
