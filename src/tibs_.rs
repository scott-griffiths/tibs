use crate::core::BitCollection;
use crate::helpers::{
    BS, BV, bv_from_bin, bv_from_bools, bv_from_bytes_slice, bv_from_f64, bv_from_hex,
    bv_from_i128, bv_from_oct, bv_from_ones, bv_from_random, bv_from_u128, bv_from_zeros,
    find_bitvec, promote_to_bv, str_to_bv, validate_logical_op_lengths, validate_shift,
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

// Enum allows us to wrap and use a Tibs without copying it or owning it.
pub(crate) enum BorrowedOrOwnedTibs<'a> {
    Borrowed(PyRef<'a, Tibs>),
    Owned(Tibs),
}

impl BitCollection for BorrowedOrOwnedTibs<'_> {
    fn from_bv(bv: BV) -> Self {
        BorrowedOrOwnedTibs::Owned(Tibs::from_bv(bv))
    }

    fn to_bitvec(&self) -> BV {
        match self {
            BorrowedOrOwnedTibs::Borrowed(t) => t.to_bitvec(),
            BorrowedOrOwnedTibs::Owned(t) => t.to_bitvec(),
        }
    }

    fn as_bitvec_ref(&self) -> &BV {
        match self {
            BorrowedOrOwnedTibs::Borrowed(t) => t.as_bitvec_ref(),
            BorrowedOrOwnedTibs::Owned(t) => t.as_bitvec_ref(),
        }
    }

    fn as_bitslice(&self) -> &BS {
        match self {
            BorrowedOrOwnedTibs::Borrowed(t) => t.as_bitslice(),
            BorrowedOrOwnedTibs::Owned(t) => t.as_bitslice(),
        }
    }

    #[inline]
    fn get_slice_unchecked(&self, start_bit: usize, length: usize) -> Self {
        Self::from_bv(self.as_bitslice()[start_bit..start_bit + length].to_bitvec())
    }

    #[inline]
    fn get_raw_bytes(&self) -> Vec<u8> {
        match self {
            BorrowedOrOwnedTibs::Borrowed(t) => t.raw_bytes(),
            BorrowedOrOwnedTibs::Owned(t) => t.raw_bytes(),
        }
    }
}

impl<'a> Clone for BorrowedOrOwnedTibs<'a> {
    fn clone(&self) -> Self {
        match self {
            BorrowedOrOwnedTibs::Borrowed(t) => BorrowedOrOwnedTibs::Owned((**t).clone()),
            BorrowedOrOwnedTibs::Owned(t) => BorrowedOrOwnedTibs::Owned(t.clone()),
        }
    }
}

pub(crate) fn tibs_from_any<'a>(any: &'a Bound<'a, PyAny>) -> PyResult<BorrowedOrOwnedTibs<'a>> {
    // Is it of type Tibs?
    if let Ok(tibs_ref) = any.extract::<PyRef<Tibs>>() {
        return Ok(BorrowedOrOwnedTibs::Borrowed(tibs_ref));
    }

    // Is it of type Mutibs?
    if let Ok(mutibs_ref) = any.extract::<PyRef<Mutibs>>() {
        return Ok(BorrowedOrOwnedTibs::Owned(mutibs_ref.to_tibs()));
    }
    let bv = promote_to_bv(any)?;

    let tibs = Tibs::from_bv(bv);
    Ok(BorrowedOrOwnedTibs::Owned(tibs))
}

impl Hash for Tibs {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);

        let bits = self.as_bitslice();

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
            _data: Arc::new(bv),
            _offset: 0,
            _length: length,
        }
    }

    pub(crate) fn from_slice_unchecked(&self, offset: usize, length: usize) -> Self {
        Tibs {
            _data: self._data.clone(),
            _offset: self._offset + offset,
            _length: length,
        }
    }

    #[inline]
    pub(crate) fn as_bitslice(&self) -> &BS {
        &self._data[self._offset..self._offset + self._length]
    }

    #[inline]
    pub(crate) fn to_bitvec(&self) -> BV {
        self.as_bitslice().to_bitvec()
    }

    #[inline]
    pub(crate) fn as_bitvec_ref(&self) -> &BV {
        &self._data
    }

    #[inline]
    pub(crate) fn raw_bytes(&self) -> Vec<u8> {
        // Given the bit offset self._offset and the bit length self._length
        // return the byte data from the bitvec self._data. The data should cover just
        // enough bytes and should not realign them in any way.
        let byte_offset = self._offset / 8;
        let final_byte = (self._offset + self._length).div_ceil(8);
        let full_bytes = self._data.as_raw_slice();
        full_bytes[byte_offset..final_byte].to_vec()
    }
}

///     An immutable container of binary data.
///
///     To construct, use a builder 'from' method:
///
///     * ``Tibs.from_bin(s)`` - Create from a binary string, optionally starting with '0b'.
///     * ``Tibs.from_oct(s)`` - Create from an octal string, optionally starting with '0o'.
///     * ``Tibs.from_hex(s)`` - Create from a hex string, optionally starting with '0x'.
///     * ``Tibs.from_u(u, length)`` - Create from an unsigned int to a given length.
///     * ``Tibs.from_i(i, length)`` - Create from a signed int to a given length.
///     * ``Tibs.from_f(f, length)`` - Create from an IEEE float to a 16, 32 or 64 bit length.
///     * ``Tibs.from_bytes(b)`` - Create directly from a ``bytes`` or ``bytearray`` object.
///     * ``Tibs.from_string(s)`` - Use a formatted string.
///     * ``Tibs.from_bools(iterable)`` - Convert each element in ``iterable`` to a bool.
///     * ``Tibs.from_zeros(length)`` - Initialise with ``length`` '0' bits.
///     * ``Tibs.from_ones(length)`` - Initialise with ``length`` '1' bits.
///     * ``Tibs.from_random(length, [secure, seed])`` - Initialise with ``length`` randomly set bits.
///     * ``Tibs.from_joined(iterable)`` - Concatenate an iterable of objects.
///
///     Using ``Tibs(auto)`` will try to delegate to ``from_string``, ``from_bytes`` or ``from_bools``.
///
#[derive(Clone)]
#[pyclass(frozen, sequence, module = "tibs")]
pub struct Tibs {
    _data: Arc<BV>,
    _offset: usize,
    _length: usize,
}

/// Public Python-facing methods.
#[pymethods]
impl Tibs {
    #[new]
    #[pyo3(signature = (auto = None))]
    pub fn py_new(auto: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(auto) = auto else {
            return Ok(BitCollection::empty());
        };
        let bv = promote_to_bv(auto)?;
        Ok(Tibs::from_bv(bv))
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
    pub fn __repr__(&self, py: Python) -> String {
        let class_name = py.get_type::<Self>().name().unwrap();
        if self.is_empty() {
            format!("{}()", class_name)
        } else {
            format!("{}('{}')", class_name, self.__str__())
        }
    }

    /// Iterate over the bits of the Tibs, yielding each bit as a boolean.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<BoolIterator>> {
        let py = slf.py();
        let length = slf.len();
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
    /// :param chunk_size: The size in bits of the chunks to generate.
    /// :param count: If specified, at most count items are generated. Default is to cut as many times as possible.
    /// :return: A generator yielding Tibs chunks.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b110011').chunks(2))
    ///     [Tibs('0b11'), Tibs('0b00'), Tibs('0b11')]
    ///
    #[pyo3(signature = (chunk_size, count = None))]
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
    pub fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match tibs_from_any(other) {
            Ok(b) => *self.as_bitslice() == *b.as_bitslice(),
            Err(_) => false,
        }
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
    /// :param b: The Tibs to find.
    /// :param start: The starting bit position of the slice to search. Defaults to 0.
    /// :param end: The end bit position of the slice to search. Defaults to len(self).
    /// :param count: The maximum number of occurrences to find.
    /// :param byte_aligned: If True, the Tibs will only be found on byte boundaries.
    /// :return: A generator yielding bit positions.
    ///
    /// Raises ValueError if b is empty, if start < 0, if end > len(self) or
    /// if end < start.
    ///
    /// All occurrences of b are found, even if they overlap.
    ///
    /// Note that this method is not available for :class:`Mutibs` as its value could change while the
    /// generator is still active. For that case you should convert to a :class:`Tibs` first with :meth:`Mutibs.to_tibs`.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b10111011').find_all('0b11'))
    ///     [2, 3, 6]
    ///
    #[pyo3(signature = (b, start=None, end=None, byte_aligned=false))]
    pub fn find_all(
        slf: PyRef<'_, Self>,
        b: &Bound<'_, PyAny>,
        start: Option<i64>,
        end: Option<i64>,
        byte_aligned: bool,
    ) -> PyResult<Py<FindAllIterator>> {
        let b = tibs_from_any(b)?;
        let (start, end) = validate_slice(slf.len(), start, end)?;
        let step = if byte_aligned { 8 } else { 1 };
        let py = slf.py();
        let needle: Py<Tibs> = match b {
            BorrowedOrOwnedTibs::Borrowed(t) => t.into(),
            BorrowedOrOwnedTibs::Owned(t) => Py::new(py, t)?,
        };
        let iter_obj = FindAllIterator {
            haystack: slf.into(),
            needle,
            start,
            end,
            byte_aligned,
            step,
            current_pos: start,
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
    /// :param length: The number of bits to set.
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
    /// :param length: The number of bits to set.
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
    /// :param s: The formatted string to convert.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_string("0xff01")
    ///     b = Tibs.from_string("0b1")
    ///
    /// The ``__init__`` method can also redirect to ``from_string`` method:
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
    /// :param u: An unsigned integer.
    /// :param length: The bit length to create. Can be up to 128.
    ///
    /// Raises ValueError if the integer doesn't fit in the length given.
    ///
    #[classmethod]
    #[pyo3(signature = (u, /, length), text_signature = "(cls, u, /, length)")]
    pub fn from_u(_cls: &Bound<'_, PyType>, u: u128, length: i64) -> PyResult<Self> {
        let bv = bv_from_u128(u, length)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Return the unsigned integer representation of the Tibs.
    pub fn to_u(&self) -> PyResult<u128> {
        BitCollection::to_u128(self)
    }

    /// Create a new instance from a signed integer.
    ///
    /// :param i: A signed integer.
    /// :param length: The bit length to create. Can be up to 128.
    ///
    /// Raises ValueError if the integer doesn't fit in the length given.
    ///
    #[classmethod]
    #[pyo3(signature = (i, /, length), text_signature = "(cls, i, /, length)")]
    pub fn from_i(_cls: &Bound<'_, PyType>, i: i128, length: i64) -> PyResult<Self> {
        let bv = bv_from_i128(i, length)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Return the signed integer representation of the Tibs.
    pub fn to_i(&self) -> PyResult<i128> {
        BitCollection::to_i128(self)
    }

    #[classmethod]
    #[pyo3(signature = (f, /, length), text_signature = "(cls, f, /, length)")]
    pub fn from_f(_cls: &Bound<'_, PyType>, f: f64, length: i64) -> PyResult<Self> {
        let bv = bv_from_f64(f, length)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Return the floating point representation of the Tibs.
    ///
    /// The length must be 16, 32 or 64.
    pub fn to_f(&self) -> PyResult<f64> {
        BitCollection::to_f64(self)
    }

    /// Create a new instance from a binary string.
    ///
    /// :param s: A string of '0' and '1's, optionally preceded with '0b'.
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
    pub fn to_bin(&self) -> String {
        BitCollection::to_binary(self)
    }

    /// Create a new instance from an octal string.
    ///
    /// :param s: A string of octal digits, optionally preceded with '0o'.
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_oct(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        let bv = bv_from_oct(s)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Return the octal representation of the Tibs as a string.
    ///
    /// Raises ValueError if the length is not a multiple of 3.
    pub fn to_oct(&self) -> PyResult<String> {
        BitCollection::to_octal(self)
    }

    /// Create a new instance from a hexadecimal string.
    ///
    /// :param s: A string of hexadecimal digits, optionally preceded with '0x'.
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_hex(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        let bv = bv_from_hex(s)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Return the hexadecimal representation of the Tibs as a string.
    ///
    /// Raises ValueError if the length is not a multiple of 4.
    pub fn to_hex(&self) -> PyResult<String> {
        BitCollection::to_hexadecimal(self)
    }

    /// Create a new instance from a bytes object.
    ///
    /// :param data: The bytes, bytearray or memoryview object to convert to a :class:`Tibs`.
    /// :param length: The bit length to use. Defaults to the whole of the data.
    /// :param offset: The bit offset from the start. Defaults to zero.
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
    /// :param iterable: The iterable to convert to a :class:`Tibs`.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_bools([False, 0, 1, "Steven"])  # binary 0011
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /), text_signature = "(cls, values, /)")]
    pub fn from_bools(_cls: &Bound<'_, PyType>, iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bv = bv_from_bools(iterable)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Create a new instance with all bits randomly set.
    ///
    /// :param length: The number of bits to set. Must be positive.
    /// :param secure: If ``True``, use the OS's cryptographically secure generator. Default is ``False``.
    /// :param seed: A bytes or bytearray to use as an optional seed, only if ``secure`` is ``False``.
    /// :return: A newly constructed ``Tibs`` with random data.
    ///
    /// The 'secure' option uses the OS's random data source, so will be slower and could potentially
    /// fail.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_random(1000000)  # A million random bits
    ///     b = Tibs.from_random(100, b'a_seed')
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
    /// :param iterable: An iterable to concatenate. Items can either be a Tibs object, or a string or bytes-like object that could create one via the :meth:`from_string` or :meth:`from_bytes` methods.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_joined(['0x01', [1, 0], b'some_bytes'])
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /), text_signature = "(cls, iterable, /)")]
    pub fn from_joined(_cls: &Bound<'_, PyType>, iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Convert each item to BV, store, and sum total length for a single allocation.
        let iter = iterable.try_iter()?;
        let mut bv_parts: Vec<BV> = Vec::new();
        let mut total_len: usize = 0;
        for item in iter {
            let obj = item?;
            let bits = tibs_from_any(&obj)?;
            total_len += bits.len();
            let owned_tibs = match bits {
                BorrowedOrOwnedTibs::Borrowed(t) => t.clone(),
                BorrowedOrOwnedTibs::Owned(t) => t,
            };
            bv_parts.push(owned_tibs.to_bitvec());
        }

        // Concatenate.
        let mut bv = BV::with_capacity(total_len);
        for b in bv_parts {
            bv.extend_from_bitslice(&b);
        }
        Ok(Tibs::from_bv(bv))
    }

    /// Return the Tibs as a bytes object.
    ///
    /// Raises ValueError if the length is not a multiple of 8.
    pub fn to_bytes(&self) -> PyResult<Vec<u8>> {
        BitCollection::to_byte_data(self)
    }

    /// Find first occurrence of a bit sequence.
    ///
    /// Returns the bit position if found, or None if not found.
    ///
    /// :param b: The Tibs to find.
    /// :param start: The starting bit position. Defaults to 0.
    /// :param end: The end position. Defaults to len(self).
    /// :param byte_aligned: If ``True``, the Tibs will only be found on byte boundaries.
    /// :return: The bit position if found, or None if not found.
    ///
    /// .. code-block:: pycon
    ///
    ///      >>> Tibs('0xc3e').find('0b1111')
    ///      6
    ///
    #[pyo3(signature = (b, start=None, end=None, byte_aligned=false))]
    pub fn find(
        &self,
        b: &Bound<'_, PyAny>,
        start: Option<i64>,
        end: Option<i64>,
        byte_aligned: bool,
    ) -> PyResult<Option<usize>> {
        let b = tibs_from_any(b)?;
        if b.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to find."));
        }
        let (start, end) = validate_slice(self.len(), start, end)?;

        Ok(find_bitvec(
            self.as_bitslice(),
            b.as_bitslice(),
            start,
            end,
            byte_aligned,
        ))
    }

    /// Return True if b is a sub-sequence of self.
    pub fn __contains__(&self, b: &Bound<'_, PyAny>) -> bool {
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
    /// :param b: The Tibs to find.
    /// :param start: The starting bit position. Defaults to 0.
    /// :param end: The end position. Defaults to len(self).
    /// :param byte_aligned: If ``True``, the Tibs will only be found on byte boundaries.
    /// :return: The bit position if found, or None if not found.
    #[pyo3(signature = (b, start=None, end=None, byte_aligned=false))]
    pub fn rfind(
        &self,
        b: &Bound<'_, PyAny>,
        start: Option<i64>,
        end: Option<i64>,
        byte_aligned: bool,
    ) -> PyResult<Option<usize>> {
        let b = tibs_from_any(b)?;
        if b.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to rfind."));
        }

        let (start, end) = validate_slice(self.len(), start, end)?;
        if b.len() + start > end {
            return Ok(None);
        }
        let step = if byte_aligned { 8 } else { 1 };
        let mut pos = end - b.len();
        if byte_aligned {
            pos = pos / 8 * 8;
        }
        while pos >= start {
            if self.as_bitslice()[pos..pos + b.len()] == *b.as_bitslice() {
                return Ok(Some(pos));
            }
            if pos < step {
                break;
            }
            pos -= step;
        }
        Ok(None)
    }

    /// Return whether the current Tibs starts with prefix.
    ///
    /// :param prefix: The bits to search for.
    /// :return: True if the Tibs starts with the prefix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b101100').starts_with('0b101')
    ///     True
    ///     >>> Tibs('0b101100').starts_with('0b100')
    ///     False
    ///
    pub fn starts_with(&self, prefix: &Bound<'_, PyAny>) -> PyResult<bool> {
        let prefix = tibs_from_any(prefix)?;
        Ok(<Tibs as BitCollection>::starts_with(self, prefix))
    }

    /// Return whether the current Tibs ends with suffix.
    ///
    /// :param suffix: The bits to search for.
    /// :return: True if the Tibs ends with the suffix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b101100').ends_with('0b100')
    ///     True
    ///     >>> Tibs('0b101100').ends_with('0b101')
    ///     False
    ///
    pub fn ends_with(&self, suffix: &Bound<'_, PyAny>) -> PyResult<bool> {
        let suffix = tibs_from_any(suffix)?;
        Ok(<Tibs as BitCollection>::ends_with(self, suffix))
    }

    /// Return count of total number of either zero or one bits.
    ///
    ///     :param value: If `bool(value)` is True, bits set to 1 are counted; otherwise, bits set to 0 are counted.
    ///     :return: The count of bits set to 1 or 0.
    ///
    ///     .. code-block:: pycon
    ///
    ///         >>> Tibs('0xef').count(1)
    ///         7
    ///
    pub fn count(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let count_ones = value.is_truthy()?;
        Ok(<Tibs as BitCollection>::count(self, count_ones))
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
        self.as_bitslice().all()
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
        self.as_bitslice().any()
    }

    /// Create and return a mutable copy of the Tibs as a Mutibs instance.
    pub fn to_mutibs(&self) -> Mutibs {
        Mutibs::from_bv(self.to_bitvec())
    }

    #[inline]
    /// Get a bit or a slice of bits.
    ///
    /// :param key: The index or slice to get.
    /// :return: A bool for a single index, or a new Tibs for a slice.
    /// :raises IndexError: If the index is out of range.
    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = key.py();
        // Handle integer indexing
        if let Ok(index) = key.extract::<i64>() {
            let value: bool = self.get_index(index)?;
            let py_value = PyBool::new(py, value);
            return Ok(py_value.to_owned().into());
        }

        // Handle slice indexing
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(self.len() as isize)?;
            let start: i64 = indices.start.try_into()?;
            let stop: i64 = indices.stop.try_into()?;
            let step: i64 = indices.step.try_into()?;

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
    pub fn __add__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bs = tibs_from_any(bs)?;
        let mut data = BV::with_capacity(self.len() + bs.len());
        data.extend_from_bitslice(self.as_bitslice());
        data.extend_from_bitslice(bs.as_bitslice());
        Ok(Tibs::from_bv(data))
    }

    /// Concatenates two Tibs and return a newly constructed Tibs.
    pub fn __radd__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bs = tibs_from_any(bs)?;
        let mut data = BV::with_capacity(bs.len() + self.len());
        data.extend_from_bitslice(bs.as_bitslice());
        data.extend_from_bitslice(self.as_bitslice());
        Ok(Tibs::from_bv(data))
    }

    /// Bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __and__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        // TODO: Return early `if bs is self`.
        let other = tibs_from_any(bs)?;
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_and(self, &other))
    }

    /// Bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __or__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        // TODO: Return early `if bs is self`.
        let other = tibs_from_any(bs)?;
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_or(self, &other))
    }

    /// Bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __xor__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = tibs_from_any(bs)?;
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_xor(self, &other))
    }

    /// Reverse bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __rand__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__and__(bs)
    }

    /// Reverse bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __ror__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__or__(bs)
    }

    /// Reverse bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __rxor__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__xor__(bs)
    }

    /// Return the instance with every bit inverted.
    ///
    /// Raises ValueError if the Tibs is empty.
    ///
    pub fn __invert__(&self) -> PyResult<Self> {
        if self.as_bitslice().is_empty() {
            return Err(PyValueError::new_err("Cannot invert empty Tibs."));
        }
        Ok(Tibs::from_bv(self.to_bitvec().not()))
    }

    /// Return the Tibs as a bytes object.
    ///
    /// Raises ValueError if the length is not a multiple of 8.
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
