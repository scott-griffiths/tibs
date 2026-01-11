use crate::core::BitCollection;
use crate::helpers::{
    BS, BV, bv_from_bin, bv_from_bools, bv_from_bytes_slice, bv_from_f64, bv_from_hex,
    bv_from_i128, bv_from_oct, bv_from_ones, bv_from_random, bv_from_u128, bv_from_zeros,
    find_bitvec, promote_to_bv, str_to_bv, validate_index, validate_logical_op_lengths,
    validate_shift, validate_slice,
};
use crate::tibs_::{BorrowedOrOwnedTibs, Tibs, tibs_from_any};

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PySlice, PyType};
use std::ops::Not;

///     A mutable container of binary data.
///
///     To construct, use a builder 'from' method:
///
///     * ``Mutibs.from_bin(s)`` - Create from a binary string, optionally starting with '0b'.
///     * ``Mutibs.from_oct(s)`` - Create from an octal string, optionally starting with '0o'.
///     * ``Mutibs.from_hex(s)`` - Create from a hex string, optionally starting with '0x'.
///     * ``Mutibs.from_u(u, length)`` - Create from an unsigned int to a given length.
///     * ``Mutibs.from_i(i, length)`` - Create from a signed int to a given length.
///     * ``Mutibs.from_f(f, length)`` - Create from an IEEE float to a 16, 32 or 64 bit length.
///     * ``Mutibs.from_bytes(b)`` - Create directly from a ``bytes`` or ``bytearray`` object.
///     * ``Mutibs.from_string(s)`` - Use a formatted string.
///     * ``Mutibs.from_bools(iterable)`` - Convert each element in ``iterable`` to a bool.
///     * ``Mutibs.from_zeros(length)`` - Initialise with ``length`` '0' bits.
///     * ``Mutibs.from_ones(length)`` - Initialise with ``length`` '1' bits.
///     * ``Mutibs.from_random(length, [secure, seed])`` - Initialise with ``length`` randomly set bits.
///     * ``Mutibs.from_joined(iterable)`` - Concatenate an iterable of objects.
///
///     Using ``Mutibs(auto)`` will try to delegate to ``from_string``, ``from_bytes`` or ``from_bools``.
///
#[pyclass(freelist = 8, module = "tibs")]
#[derive(Clone)]
pub struct Mutibs {
    _data: BV,
}

// Internal methods, not exported to Python
impl Mutibs {
    pub(crate) fn from_bv(bv: BV) -> Self {
        Mutibs { _data: bv }
    }

    #[inline]
    pub(crate) fn as_bitvec_ref(&self) -> &BV {
        &self._data
    }

    #[inline]
    pub(crate) fn as_mut_bitvec_ref(&mut self) -> &mut BV {
        &mut self._data
    }

    #[inline]
    pub(crate) fn raw_bytes(&self) -> Vec<u8> {
        self._data.as_raw_slice().to_vec()
    }

    pub fn set_index(&mut self, value: bool, index: i64) -> PyResult<()> {
        self.set_from_sequence(value, vec![index])
    }

    pub(crate) fn set_slice(&mut self, start: usize, end: usize, value: &BS) {
        if end - start == value.len() {
            // This is an overwrite, so no need to move data around.
            self.as_mut_bitvec_ref()[start..start + value.len()].copy_from_bitslice(value);
        } else if start == end {
            // Not sure why but splice doesn't work for this case, so we do it explicitly
            let tail = self.as_mut_bitvec_ref().split_off(start);
            self.as_mut_bitvec_ref().extend_from_bitslice(value);
            self.as_mut_bitvec_ref().extend_from_bitslice(&tail);
        } else {
            let tail = self.as_mut_bitvec_ref().split_off(end);
            self.as_mut_bitvec_ref().truncate(start);
            self.as_mut_bitvec_ref().extend_from_bitslice(value);
            self.as_mut_bitvec_ref().extend_from_bitslice(&tail);
        }
    }

    pub(crate) fn ixor(&mut self, other: &BS) -> PyResult<()> {
        validate_logical_op_lengths(self.len(), other.len())?;
        *self.as_mut_bitvec_ref() ^= other;
        Ok(())
    }

    pub(crate) fn ior(&mut self, other: &BS) -> PyResult<()> {
        validate_logical_op_lengths(self.len(), other.len())?;
        *self.as_mut_bitvec_ref() |= other;
        Ok(())
    }

    pub(crate) fn iand(&mut self, other: &BS) -> PyResult<()> {
        validate_logical_op_lengths(self.len(), other.len())?;
        *self.as_mut_bitvec_ref() &= other;
        Ok(())
    }

    pub(crate) fn set_from_sequence(&mut self, value: bool, indices: Vec<i64>) -> PyResult<()> {
        let mut validated = Vec::with_capacity(indices.len());
        for idx in indices {
            validated.push(validate_index(idx, self.len())?);
        }
        for idx in validated {
            self.as_mut_bitvec_ref().set(idx, value);
        }
        Ok(())
    }

    pub(crate) fn set_from_slice(
        &mut self,
        value: bool,
        start: i64,
        stop: i64,
        step: i64,
    ) -> PyResult<()> {
        let len = self.len() as i64;
        if len == 0 {
            return Ok(());
        }
        let mut positive_start = if start < 0 { start + len } else { start };
        let mut positive_stop = if stop < 0 { stop + len } else { stop };
        if positive_start < 0 || positive_start >= len {
            return Err(PyIndexError::new_err("Start of slice out of bounds."));
        }
        if positive_stop < 0 || positive_stop > len {
            return Err(PyIndexError::new_err("End of slice out of bounds."));
        }
        if step == 0 {
            return Err(PyValueError::new_err("Step cannot be zero."));
        }
        if step < 0 {
            positive_stop = positive_start - 1;
            positive_start = positive_stop - (positive_stop - positive_start) / step;
        }
        let positive_step = if step > 0 {
            step as usize
        } else {
            -step as usize
        };

        let mut index = positive_start as usize;
        let stop = positive_stop as usize;

        while index < stop {
            unsafe {
                self.as_mut_bitvec_ref().set_unchecked(index, value);
            }
            index += positive_step;
        }

        Ok(())
    }
}

#[pymethods]
impl Mutibs {
    #[new]
    #[pyo3(signature = (auto = None))]
    pub fn py_new(auto: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(auto) = auto else {
            return Ok(BitCollection::empty());
        };
        let bv = promote_to_bv(auto)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Return True if two Mutibs have the same binary representation.
    ///
    /// The right hand side will be promoted to a Mutibs if needed and possible.
    ///
    /// >>> Mutibs('0xf2') == '0b11110010'
    /// True
    ///
    pub fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        if let Ok(b) = other.extract::<PyRef<Tibs>>() {
            return *self.as_bitvec_ref() == *b.as_bitslice();
        }
        if let Ok(b) = other.extract::<PyRef<Mutibs>>() {
            return *self.as_bitvec_ref() == *b.as_bitvec_ref();
        }
        match tibs_from_any(other) {
            Ok(b) => *self.as_bitvec_ref() == *b.as_bitslice(),
            Err(_) => false,
        }
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

    /// Create a new instance from a formatted string.
    ///
    /// This method initializes a new instance of :class:`Mutibs` using a formatted string.
    ///
    /// :param s: The formatted string to convert.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_string("0xff01")
    ///     b = Mutibs.from_string("0b1")
    ///
    /// The `__init__` method for `Mutibs` can also redirect to `from_string` method:
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs("0xff01")
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_string(_cls: &Bound<'_, PyType>, s: String) -> PyResult<Self> {
        let bv = str_to_bv(s)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Create a new instance from a binary string.
    ///
    /// :param s: A string of '0' and '1's, optionally preceded with '0b'.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_bin("0000_1111_0101")
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_bin(_cls: &Bound<'_, PyType>, s: String) -> PyResult<Self> {
        let bv = bv_from_bin(&s)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the binary representation of the Mutibs as a string.
    pub fn to_bin(&self) -> String {
        BitCollection::to_binary(self)
    }

    /// Create a new instance from an octal string.
    ///
    /// :param s: A string of octal digits, optionally preceded with '0o'.
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_oct(_cls: &Bound<'_, PyType>, s: String) -> PyResult<Self> {
        let bv = bv_from_oct(&s)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the octal representation of the Mutibs as a string.
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
    pub fn from_hex(_cls: &Bound<'_, PyType>, s: String) -> PyResult<Self> {
        let bv = bv_from_hex(&s)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the hexadecimal representation of the Mutibs as a string.
    ///
    /// Raises ValueError if the length is not a multiple of 4.
    pub fn to_hex(&self) -> PyResult<String> {
        BitCollection::to_hexadecimal(self)
    }

    /// Return the Mutibs as a bytes object.
    ///
    /// Raises ValueError if the length is not a multiple of 8.
    pub fn to_bytes(&self) -> PyResult<Vec<u8>> {
        BitCollection::to_byte_data(self)
    }

    /// Return a copy of the raw byte information.
    ///
    /// This returns the underlying byte data and can contain leading and trailing
    /// bits that are not considered part of the object's value. Usually using
    /// :meth:`~to_bytes` is what you really need.
    ///
    /// The way that the data is stored is not considered part of the public interface
    /// and so the output of this method may change between point releases, and even
    /// during the running of a program.
    ///
    /// See also :meth:`~as_raw_data` which moves the byte data instead of copying it.
    ///
    /// :return: A tuple of the raw bytes, the bit offset and the bit length.
    ///
    /// .. code-block:: python
    ///
    ///     raw_bytes, offset, length = t.to_raw_data()
    ///     assert t == Mutibs.from_bytes(raw_bytes)[offset:offset + length]
    ///
    pub fn to_raw_data(&self) -> (Vec<u8>, usize, usize) {
        self.raw_data()
    }

    /// Return the raw bytes and offset information, leaving the Mutibs empty.
    ///
    /// This returns the underlying byte data using a move rather than a copy, and can contain
    /// leading and trailing bits that are not considered part of the object's value. Usually using
    /// :meth:`~to_bytes` is what you really need.
    ///
    /// The way that the data is stored is not considered part of the public interface
    /// and so the output of this method may change between point releases, and even
    /// during the running of a program.
    ///
    /// See also :meth:`~to_raw_data` which copies the byte data instead of moving it.
    ///
    /// :return: A tuple of the raw bytes, the bit offset and the bit length.
    ///
    /// .. code-block:: python
    ///
    ///     raw_bytes, offset, length = t.as_raw_data()
    ///     assert t == []
    ///
    pub fn as_raw_data(&mut self) -> (Vec<u8>, usize, usize) {
        let slice = self.as_bitvec_ref().as_bitslice();
        let offset = match slice.domain() {
            bitvec::domain::Domain::Enclave(elem) => elem.head().into_inner() as usize,
            bitvec::domain::Domain::Region {
                head: Some(elem), ..
            } => elem.head().into_inner() as usize,
            _ => 0,
        };
        let len = self.len();
        let bv = std::mem::take(&mut *self.as_mut_bitvec_ref());
        let raw_bytes = bv.into_vec();
        (raw_bytes, offset, len)
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
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the unsigned integer representation of the Mutibs.
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
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the signed integer representation of the Mutibs.
    pub fn to_i(&self) -> PyResult<i128> {
        BitCollection::to_i128(self)
    }

    /// Create a new instance from a floating point number.
    ///
    /// :param f: A float.
    /// :param length: The bit length to create. Must be 16, 32 or 64.
    #[classmethod]
    #[pyo3(signature = (f, /, length), text_signature = "(cls, f, /, length)")]
    pub fn from_f(_cls: &Bound<'_, PyType>, f: f64, length: i64) -> PyResult<Self> {
        let bv = bv_from_f64(f, length)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the floating point representation of the Mutibs.
    ///
    /// The length must be 16, 32 or 64.
    pub fn to_f(&self) -> PyResult<f64> {
        BitCollection::to_f64(self)
    }

    /// Create a new instance with all bits set to zero.
    ///
    /// :param length: The number of bits to set.
    /// :return: A Mutibs object with all bits set to zero.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_zeros(500)  # 500 zero bits
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

    /// Create a new instance with all bits set to one.
    ///
    /// :param length: The number of bits to set.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.from_ones(5)
    ///     Mutibs('0b11111')
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
        Ok(Mutibs::from_bv(bv_from_ones(length as usize)))
    }

    /// Create a new instance from an iterable by converting each element to a bool.
    ///
    /// :param iterable: The iterable to convert to a :class:`Mutibs`.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_bools([False, 0, 1, "Steven"])  # binary 0011
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /), text_signature = "(cls, iterable, /)")]
    pub fn from_bools(_cls: &Bound<'_, PyType>, iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bv = bv_from_bools(iterable)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Create a new instance with all bits randomly set.
    ///
    /// :param length: The number of bits to set. Must be positive.
    /// :param secure: If ``True``, use the OS's cryptographically secure generator. Default is ``False``.
    /// :param seed: A bytes or bytearray to use as an optional seed, only if ``secure`` is ``False``.
    /// :return: A newly constructed ``Mutibs`` with random data.
    ///
    /// The 'secure' option uses the OS's random data source, so will be slower and could potentially
    /// fail.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_random(1000000)  # A million random bits
    ///     b = Mutibs.from_random(100, b'a_seed')
    ///
    #[classmethod]
    #[pyo3(signature = (length, /, secure=false, seed=None), text_signature="(cls, length, /, secure=False, seed=None)"
    )]
    pub fn from_random(
        _cls: &Bound<'_, PyType>,
        length: i64,
        secure: bool,
        seed: Option<Vec<u8>>,
    ) -> PyResult<Self> {
        let bv = bv_from_random(length, secure, &seed)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Create a new instance from a bytes object.
    ///
    /// :param data: The bytes, bytearray or memoryview object to convert to a :class:`Mutibs`.
    /// :param length: The bit length to use. Defaults to the whole of the data.
    /// :param offset: The bit offset from the start. Defaults to zero.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_bytes(b"some_bytes_maybe_from_a_file")
    ///
    #[classmethod]
    #[inline]
    #[pyo3(signature = (data, /, offset=None, length=None), text_signature = "(cls, data, /, offset=None, length=None)"
    )]
    pub fn from_bytes(
        _cls: &Bound<'_, PyType>,
        data: Vec<u8>,
        offset: Option<i64>,
        length: Option<i64>,
    ) -> PyResult<Self> {
        let bv = bv_from_bytes_slice(data, offset, length)?;
        Ok(Self::from_bv(bv))
    }

    /// Create a new instance by concatenating a sequence of Tibs objects.
    ///
    /// This method concatenates a sequence of Tibs objects into a single Mutibs object.
    ///
    /// :param iterable: An iterable to concatenate. Items can either be a Tibs object, or a string or bytes-like object that could create one via the :meth:`from_string` or :meth:`from_bytes` methods.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_joined([f'u6={x}' for x in range(64)])
    ///     b = Mutibs.from_joined(['0x01', [1, 0], b'some_bytes'])
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /), text_signature = "(cls, iterable, /)")]
    pub fn from_joined(_cls: &Bound<'_, PyType>, iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Tibs::from_joined(_cls, iterable)?.to_mutibs())
    }

    /// The bit length of the Mutibs.
    pub fn __len__(&self) -> usize {
        self.len()
    }

    /// Get a bit or a slice of bits.
    ///
    /// :param key: The index or slice to get.
    /// :return: A bool for a single index, or a new Mutibs for a slice.
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
                    self.get_slice(start as usize, (stop - start) as usize)?
                } else {
                    Mutibs::empty()
                }
            } else {
                self.get_slice_with_step(start, stop, step)?
            };
            let py_obj = Py::new(py, result)?.into_pyobject(py)?;
            return Ok(py_obj.into());
        }

        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    /// Set a bit or a slice of bits.
    ///
    /// :param key: The index or slice to set.
    /// :param value: For a single index, a boolean value. For a slice, anything that can be converted to Tibs.
    /// :raises ValueError: If the slice has a step other than 1, or if the length of the value doesn't match the slice.
    /// :raises IndexError: If the index is out of range.
    ///
    /// Examples:
    ///     >>> b = Mutibs('0b0000')
    ///     >>> b[1] = True
    ///     >>> b.bin
    ///     '0100'
    ///     >>> b[1:3] = '0b11111'
    ///     >>> b.bin
    ///     '0111110'
    ///
    pub fn __setitem__(
        mut slf: PyRefMut<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let length = slf.len();
        if let Ok(index) = key.extract::<i64>() {
            slf.set_index(value.is_truthy()?, index)?;
            return Ok(());
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            // Need to guard against value being self
            let bs = if value.as_ptr() == slf.as_ptr() {
                BorrowedOrOwnedTibs::Owned(Tibs::from_bv(slf.to_bitvec()))
            } else {
                tibs_from_any(value)?
            };

            let indices = slice.indices(length as isize)?;
            let start: i64 = indices.start.try_into()?;
            let stop: i64 = indices.stop.try_into()?;
            let step: i64 = indices.step.try_into()?;

            if step == 1 {
                debug_assert!(start >= 0);
                debug_assert!(stop >= 0);
                slf.set_slice(start as usize, stop as usize, bs.as_bitslice());
                return Ok(());
            }
            if step == 0 {
                return Err(PyValueError::new_err(
                    "The step in __setitem__ must not be zero.",
                ));
            }
            // Compute target indices in the natural slice order (respecting step sign).
            let mut positions: Vec<usize> = Vec::new();
            if step > 0 {
                debug_assert!(start >= 0);
                debug_assert!(stop >= 0);
                let mut i = start;
                while i < stop {
                    positions.push(i as usize);
                    i += step;
                }
            } else {
                // TODO: with a negative step I think start or stop could be -1.
                let mut i = start;
                while i > stop {
                    positions.push(i as usize);
                    i += step; // step < 0
                }
            }

            // Enforce equal sizes.
            if bs.len() != positions.len() {
                return Err(PyValueError::new_err(format!(
                    "Attempt to assign sequence of size {} to extended slice of size {}",
                    bs.len(),
                    positions.len()
                )));
            }

            // Assign element-wise.
            for (k, &pos) in positions.iter().enumerate() {
                let v = bs.as_bitslice()[k];
                slf.as_mut_bitvec_ref().set(pos, v);
            }

            return Ok(());
        }
        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    pub fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let length = self.len();
        if let Ok(mut index) = key.extract::<i64>() {
            if index < 0 {
                index += length as i64;
            }
            if index < 0 || index >= length as i64 {
                return Err(PyIndexError::new_err(format!(
                    "Bit index {index} out of range for length {length}"
                )));
            }
            self.as_mut_bitvec_ref().remove(index as usize);
            return Ok(());
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(length as isize)?;
            let start: i64 = indices.start.try_into()?;
            let stop: i64 = indices.stop.try_into()?;
            let step: i64 = indices.step.try_into()?;
            if step == 1 {
                if stop > start {
                    self.as_mut_bitvec_ref()
                        .drain(start as usize..stop as usize);
                }
            } else {
                // Collect indices to remove, then remove from highest to lowest.
                let mut to_remove: Vec<usize> = if step > 0 {
                    let mut v = Vec::new();
                    let mut i = start;
                    while i < stop {
                        v.push(i as usize);
                        i += step;
                    }
                    v
                } else {
                    let mut v = Vec::new();
                    let mut i = start;
                    while i > stop {
                        v.push(i as usize);
                        i += step; // step < 0
                    }
                    v
                };

                to_remove.sort();
                for i in to_remove.into_iter().rev() {
                    self.as_mut_bitvec_ref().remove(i);
                }
            }
            return Ok(());
        }
        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    /// Return whether the current Mutibs starts with prefix.
    ///
    /// :param prefix: The bits to search for.
    /// :return: True if the Mutibs starts with the prefix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b101100').starts_with('0b101')
    ///     True
    ///     >>> Mutibs('0b101100').starts_with('0b100')
    ///     False
    ///
    pub fn starts_with(&self, prefix: &Bound<'_, PyAny>) -> PyResult<bool> {
        let prefix = tibs_from_any(prefix)?;
        Ok(<Mutibs as BitCollection>::starts_with(self, prefix))
    }

    /// Return whether the current Mutibs ends with suffix.
    ///
    /// :param suffix: The bits to search for.
    /// :return: True if the Mutibs ends with the suffix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b101100').ends_with('0b100')
    ///     True
    ///     >>> Mutibs('0b101100').ends_with('0b101')
    ///     False
    ///
    pub fn ends_with(&self, suffix: &Bound<'_, PyAny>) -> PyResult<bool> {
        let suffix = tibs_from_any(suffix)?;
        Ok(<Mutibs as BitCollection>::ends_with(self, suffix))
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
    ///      >>> Mutibs('0xc3e').find('0b1111')
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
            self.as_bitvec_ref(),
            b.as_bitslice(),
            start,
            end,
            byte_aligned,
        ))
    }

    /// Bit-wise 'and' between two Mutibs. Returns new Mutibs.
    ///
    /// Raises ValueError if the two Mutibs have differing lengths.
    ///
    pub fn __and__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = tibs_from_any(bs)?;
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_and(self, &other))
    }

    /// Bit-wise 'or' between two Mutibs. Returns new Mutibs.
    ///
    /// Raises ValueError if the two Mutibs have differing lengths.
    ///
    pub fn __or__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = tibs_from_any(bs)?;
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_or(self, &other))
    }

    /// Bit-wise 'xor' between two Mutibs. Returns new Mutibs.
    ///
    /// Raises ValueError if the two Mutibs have differing lengths.
    ///
    pub fn __xor__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = tibs_from_any(bs)?;
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_xor(self, &other))
    }

    /// Reverse bit-wise 'and' between two Mutibs. Returns new Mutibs.
    ///
    /// This method is used when the RHS is a Mutibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Mutibs have differing lengths.
    ///
    pub fn __rand__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__and__(bs)
    }

    /// Reverse bit-wise 'or' between two Mutibs. Returns new Mutibs.
    ///
    /// This method is used when the RHS is a Mutibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Mutibs have differing lengths.
    ///
    pub fn __ror__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__or__(bs)
    }

    /// Reverse bit-wise 'xor' between two Mutibs. Returns new Mutibs.
    ///
    /// This method is used when the RHS is a Mutibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Mutibs have differing lengths.
    ///
    pub fn __rxor__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__xor__(bs)
    }

    /// Rotates bit pattern to the left. Returns self.
    ///
    /// :param n: The number of bits to rotate by.
    /// :param start: Start of slice to rotate. Defaults to 0.
    /// :param end: End of slice to rotate. Defaults to len(self).
    /// :return: self
    ///
    /// Raises ValueError if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b1011')
    ///     >>> a.rol(2)
    ///     Mutibs('0b1110')
    ///
    #[pyo3(signature = (n, start=None, end=None))]
    pub fn rol(
        mut slf: PyRefMut<'_, Self>,
        n: i64,
        start: Option<i64>,
        end: Option<i64>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        if slf.is_empty() {
            return Err(PyValueError::new_err("Cannot rotate an empty Mutibs."));
        }
        if n < 0 {
            return Err(PyValueError::new_err("Cannot rotate by a negative amount."));
        }

        let (start, end) = validate_slice(slf.len(), start, end)?;
        let n = (n % (end as i64 - start as i64)) as usize;
        slf.as_mut_bitvec_ref()[start..end].rotate_left(n);
        Ok(slf)
    }

    /// Rotates bit pattern to the right. Returns self.
    ///
    /// :param n: The number of bits to rotate by.
    /// :param start: Start of slice to rotate. Defaults to 0.
    /// :param end: End of slice to rotate. Defaults to len(self).
    /// :return: self
    ///
    /// Raises ValueError if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b1011')
    ///     >>> a.ror(1)
    ///     Mutibs('0b1101')
    ///
    #[pyo3(signature = (n, start=None, end=None))]
    pub fn ror(
        mut slf: PyRefMut<'_, Self>,
        n: i64,
        start: Option<i64>,
        end: Option<i64>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        if slf.is_empty() {
            return Err(PyValueError::new_err("Cannot rotate an empty Mutibs."));
        }
        if n < 0 {
            return Err(PyValueError::new_err("Cannot rotate by a negative amount."));
        }

        let (start, end) = validate_slice(slf.len(), start, end)?;
        let n = (n % (end as i64 - start as i64)) as usize;
        slf.as_mut_bitvec_ref()[start..end].rotate_right(n);
        Ok(slf)
    }

    /// Set one or many bits set to 1 or 0. Returns self.
    ///
    /// :param value: If bool(value) is True, bits are set to 1, otherwise they are set to 0.
    /// :param pos: Either a single bit position or an iterable of bit positions.
    /// :return: self
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs.from_zeros(10)
    ///     >>> a.set(1, 5)
    ///     Mutibs('0b0000010000')
    ///     >>> a.set(1, [-1, -2])
    ///     Mutibs('0b0000010011')
    ///     >>> a.set(0, range(8, 10))
    ///     Mutibs('0b0000010000')
    ///
    pub fn set<'a>(
        mut slf: PyRefMut<'a, Self>,
        value: &Bound<'_, PyAny>,
        pos: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let v = value.is_truthy()?;

        if let Ok(index) = pos.extract::<i64>() {
            slf.set_index(v, index)?;
        } else if pos.is_instance_of::<pyo3::types::PyRange>() {
            let start = pos.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
            let stop = pos.getattr("stop")?.extract::<i64>()?;
            let step = pos.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
            slf.set_from_slice(v, start, stop, step)?;
        }
        // Otherwise treat as a sequence
        else {
            // Convert to Vec<i64> if possible
            let indices = pos.extract::<Vec<i64>>()?;
            slf.set_from_sequence(v, indices)?;
        }

        Ok(slf)
    }

    /// Count of total number of either zero or one bits.
    ///
    /// :param value: If `bool(value)` is True, bits set to 1 are counted; otherwise, bits set to 0 are counted.
    /// :return: The count of bits set to 1 or 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0xef').count(1)
    ///     7
    ///
    pub fn count(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let count_ones = value.is_truthy()?;
        Ok(<Mutibs as BitCollection>::count(self, count_ones))
    }

    /// Return True if all bits are equal to 1, otherwise return False.
    ///
    /// :return: ``True`` if all bits are 1, otherwise ``False``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b1111').all()
    ///     True
    ///     >>> Mutibs('0b1011').all()
    ///     False
    ///
    pub fn all(&self) -> bool {
        self.as_bitvec_ref().all()
    }

    /// Return True if any bits are equal to 1, otherwise return False.
    ///
    /// :return: ``True`` if any bits are 1, otherwise ``False``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b0000').any()
    ///     False
    ///     >>> Mutibs('0b1000').any()
    ///     True
    ///
    pub fn any(&self) -> bool {
        self.as_bitvec_ref().any()
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
        // TODO: Completely redo how rfind works!
        let t = Tibs::from_bv(self.to_bitvec());
        t.rfind(b, start, end, byte_aligned)
    }

    /// Return the Mutibs with one or many bits inverted between 0 and 1.
    ///
    /// :param pos: Either a single bit position or an iterable of bit positions.
    /// :return: self
    ///
    /// Raises IndexError if pos < -len(self) or pos >= len(self).
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b10111')
    ///     >>> a.invert(1)
    ///     Mutibs('0b11111')
    ///     >>> a.invert([0, 2])
    ///     Mutibs('0b01011')
    ///     >>> a.invert()
    ///     Mutibs('0b10100')
    ///
    #[pyo3(signature = (pos = None))]
    pub fn invert<'a>(
        mut slf: PyRefMut<'a, Self>,
        pos: Option<&Bound<'a, PyAny>>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        match pos {
            None => {
                *slf.as_mut_bitvec_ref() = std::mem::take(&mut *slf.as_mut_bitvec_ref()).not();
            }
            Some(p) => {
                if let Ok(pos) = p.extract::<i64>() {
                    let pos: usize = validate_index(pos, slf.len())?;
                    let value = slf.as_bitvec_ref()[pos];
                    slf.as_mut_bitvec_ref().set(pos, !value);
                } else if let Ok(pos_list) = p.extract::<Vec<i64>>() {
                    for pos in pos_list {
                        let pos: usize = validate_index(pos, slf.len())?;
                        let value = slf.as_bitvec_ref()[pos];
                        slf.as_mut_bitvec_ref().set(pos, !value);
                    }
                } else {
                    return Err(PyTypeError::new_err(
                        "invert() argument must be an integer, an iterable of ints, or None",
                    ));
                }
            }
        }
        Ok(slf)
    }

    /// Reverse bits in-place.
    ///
    /// :return: self
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b1011')
    ///     >>> a.reverse()
    ///     Mutibs('0b1101')
    ///
    pub fn reverse(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.as_mut_bitvec_ref().reverse();
        slf
    }

    /// Change the byte endianness in-place. Returns self.
    ///
    /// The whole of the Mutibs will be byte-swapped. It must be a multiple
    /// of byte_length long.
    ///
    /// :param byte_length: An int giving the number of bytes in each swap.
    /// :return: self
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0x12345678')
    ///     >>> a.byte_swap(2)
    ///     Mutibs('0x34127856')
    ///
    #[pyo3(signature = (byte_length = None))]
    pub fn byte_swap(
        mut slf: PyRefMut<'_, Self>,
        byte_length: Option<i64>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let len = slf.len();
        if !len.is_multiple_of(8) {
            return Err(PyValueError::new_err(format!(
                "Bit length must be an multiple of 8 to use byte_swap (got length of {len} bits). This error can also be caused by using an endianness modifier on non-whole byte data."
            )));
        }
        let byte_length = byte_length.unwrap_or((len as i64) / 8);
        if byte_length == 0 && len == 0 {
            return Ok(slf);
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
                "The Mutibs to byte_swap is {self_byte_length} bytes long, but it needs to be a multiple of {byte_length} bytes."
            )));
        }

        let mut bytes = slf.to_bytes()?;
        for chunk in bytes.chunks_mut(byte_length) {
            chunk.reverse();
        }
        *slf.as_mut_bitvec_ref() = BV::from_vec(bytes);
        Ok(slf)
    }

    /// Return the instance with every bit inverted.
    ///
    /// Raises ValueError if the Mutibs is empty.
    ///
    pub fn __invert__(&self) -> PyResult<Self> {
        if self.as_bitvec_ref().is_empty() {
            return Err(PyValueError::new_err("Cannot invert empty Mutibs."));
        }
        Ok(Mutibs::from_bv(self.to_bitvec().not()))
    }

    /// Return new Mutibs shifted by n to the left.
    ///
    /// n -- the number of bits to shift. Must be >= 0.
    ///
    pub fn __lshift__(&self, n: i64) -> PyResult<Self> {
        let shift = validate_shift(self, n)?;
        Ok(self.lshift(shift))
    }

    /// Return new Mutibs shifted by n to the right.
    ///
    /// n -- the number of bits to shift. Must be >= 0.
    ///
    pub fn __rshift__(&self, n: i64) -> PyResult<Self> {
        let shift = validate_shift(self, n)?;
        Ok(self.rshift(shift))
    }

    /// Return a new copy of the Mutibs for the copy module.
    pub fn __copy__(&self) -> Self {
        Mutibs::from_bv(self.to_bitvec())
    }

    /// Create and return a Tibs instance from a copy of the Mutibs data.
    ///
    /// This copies the underlying binary data, giving a new independent Tibs object.
    /// If you no longer need the Mutibs, consider using :meth:`as_tibs` instead to avoid the copy.
    ///
    /// :return: A new Tibs instance with the same bit data.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b1011')
    ///     >>> b = a.to_tibs()
    ///     >>> a
    ///     Mutibs('0b1011')
    ///     >>> b
    ///     Tibs('0b1101')
    ///
    pub fn to_tibs(&self) -> Tibs {
        Tibs::from_bv(self.to_bitvec())
    }

    /// Create and return a Tibs instance by moving the Mutibs data.
    ///
    /// The data is moved to the new Tibs, so the Mutibs will be empty after the operation.
    /// This is more efficient than :meth:`to_tibs` if you no longer need the Mutibs.
    ///
    /// It will try to reclaim any excess memory capacity that the Mutibs may have had.
    ///
    /// :return: A Tibs instance with the same bit data.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b1011')
    ///     >>> b = a.as_tibs()
    ///     >>> a
    ///     Mutibs()
    ///     >>> b
    ///     Tibs('0b1101')
    ///
    pub fn as_tibs(&mut self) -> Tibs {
        let mut data = std::mem::take(&mut *self.as_mut_bitvec_ref());
        data.shrink_to_fit();
        Tibs::from_bv(data)
    }

    /// Clear all bits, making the Mutibs empty.
    ///
    /// This doesn't change the allocated capacity, so won't free up any memory.
    ///
    pub fn clear(&mut self) {
        self.as_mut_bitvec_ref().clear();
    }

    /// Return the number of bits the Mutibs can hold without reallocating memory.
    ///
    /// The capacity is always equal to or greater than the current length of the Mutibs.
    /// If the length ever exceeds the capacity then memory will have to be reallocated, and the
    /// capacity will increase.
    ///
    /// It can be helpful as a performance optimization to reserve enough capacity before
    /// constructing a large Mutibs incrementally. See also :meth:`reserve`.
    ///
    pub fn capacity(&self) -> usize {
        self.as_bitvec_ref().capacity()
    }

    /// Reserve memory for at least `additional` more bits to be appended to the Mutibs.
    ///
    /// This can be helpful as a performance optimization to avoid multiple memory reallocations when
    /// constructing a large Mutibs incrementally. If enough memory is already reserved then
    /// this method will have no effect. See also :meth:`capacity`.
    ///
    /// :param additional: The number of bits that can be appended without any further memory reallocations.
    ///
    pub fn reserve(&mut self, additional: usize) {
        self.as_mut_bitvec_ref().reserve(additional);
    }

    /// Concatenate Mutibs and return a new Mutibs.
    pub fn __add__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bs = tibs_from_any(bs)?;
        let mut data = BV::with_capacity(self.len() + bs.len());
        data.extend_from_bitslice(self.as_bitvec_ref());
        data.extend_from_bitslice(bs.as_bitslice());
        Ok(Mutibs::from_bv(data))
    }

    /// Concatenate Mutibs and return a new Mutibs.
    pub fn __radd__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bs = tibs_from_any(bs)?;
        let mut data = BV::with_capacity(self.len() + bs.len());
        data.extend_from_bitslice(bs.as_bitslice());
        data.extend_from_bitslice(self.as_bitvec_ref());
        Ok(Mutibs::from_bv(data))
    }

    /// Concatenate in-place.
    pub fn __iadd__(slf: PyRefMut<'_, Self>, bs: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::extend(slf, bs)?;
        Ok(())
    }

    /// Append a single bit to the current Mutibs in-place.
    ///
    /// :param bit: Either `0`, `1`, `True` or `False` to append.
    /// :return: self
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs()
    ///     >>> a.append(True)
    ///     Mutibs('0b1')
    ///
    pub fn append<'a>(
        mut slf: PyRefMut<'a, Self>,
        bit: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let bit = if let Ok(b) = bit.cast::<PyBool>() {
            b.is_true()
        } else if let Ok(val) = bit.extract::<i64>() {
            match val {
                0 => false,
                1 => true,
                _ => {
                    return Err(PyValueError::new_err(
                        "Only True, False, 0 or 1 can be appended.",
                    ));
                }
            }
        } else {
            return Err(PyTypeError::new_err(
                "Can only append a bool or an integer (0 or 1).",
            ));
        };
        slf.as_mut_bitvec_ref().push(bit);
        Ok(slf)
    }

    /// Extend the current Mutibs in-place.
    ///
    /// :param bs: The bits to extend with.
    /// :return: self
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0x0f')
    ///     >>> a.extend('0x0a')
    ///     Mutibs('0x0f0a')
    ///
    pub fn extend<'a>(
        mut slf: PyRefMut<'a, Self>,
        bs: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        // Check if bs is the same object as slf
        if bs.as_ptr() == slf.as_ptr() {
            // If bs is slf, clone inner bits first then extend
            let bits_clone = slf.to_bitvec();
            slf.as_mut_bitvec_ref().extend_from_bitslice(&bits_clone);
        } else {
            let bs = tibs_from_any(bs)?;
            slf.as_mut_bitvec_ref()
                .extend_from_bitslice(bs.as_bitslice());
        }
        Ok(slf)
    }

    /// Extend the current Mutibs in-place from the start.
    ///
    /// This is broadly equivalent to `current = new + current`.
    /// Note that this method is inherently slower than :meth:`extend` and
    /// should be avoided in performance critical code. See also :meth:`from_joined`.
    ///
    /// :param bs: The bits to prepend to the current Mutibs.
    /// :return: self
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0x0f')
    ///     >>> a.extend_left('0x0a')
    ///     Mutibs('0x0a0f')
    ///
    pub fn extend_left<'a>(
        mut slf: PyRefMut<'a, Self>,
        bs: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        // Check for self-prepending
        if bs.as_ptr() == slf.as_ptr() {
            let mut new_data = slf.to_bitvec();
            new_data.extend_from_bitslice(slf.as_bitvec_ref());
            *slf.as_mut_bitvec_ref() = new_data;
        } else {
            let to_prepend = tibs_from_any(bs)?;
            if to_prepend.is_empty() {
                return Ok(slf);
            }
            let mut new_data = BV::with_capacity(to_prepend.len() + slf.len());
            new_data.extend_from_bitslice(to_prepend.as_bitslice());
            new_data.extend_from_bitslice(slf.as_bitvec_ref());
            *slf.as_mut_bitvec_ref() = new_data;
        }
        Ok(slf)
    }

    #[pyo3(signature = (old, new, start=None, end=None, count=None, byte_aligned=false))]
    pub fn replace<'a>(
        mut slf: PyRefMut<'a, Self>,
        old: &Bound<'_, PyAny>,
        new: &Bound<'_, PyAny>,
        start: Option<i64>,
        end: Option<i64>,
        count: Option<i64>,
        byte_aligned: bool,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let old = if old.as_ptr() == slf.as_ptr() {
            BorrowedOrOwnedTibs::Owned(slf.to_tibs())
        } else {
            tibs_from_any(old)?
        };

        if old.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to replace."));
        }
        let new = if new.as_ptr() == slf.as_ptr() {
            BorrowedOrOwnedTibs::Owned(slf.to_tibs())
        } else {
            tibs_from_any(new)?
        };

        let (start, end) = validate_slice(slf.len(), start, end)?;

        // Find all non-overlapping occurrences
        let mut starting_points: Vec<usize> = Vec::new();
        let mut current_pos = start;
        while current_pos < end {
            if let Some(count) = count
                && starting_points.len() >= count as usize
            {
                break;
            }
            if let Some(found_pos) = find_bitvec(
                slf.as_bitvec_ref(),
                old.as_bitslice(),
                current_pos,
                end,
                byte_aligned,
            ) {
                starting_points.push(found_pos);
                current_pos = found_pos + old.len();
            } else {
                break;
            }
        }

        if starting_points.is_empty() {
            return Ok(slf);
        }

        // Rebuild the bitstring with replacements
        let mut result = BV::new();
        let mut last_pos = 0;
        for &pos in &starting_points {
            result.extend_from_bitslice(&slf.as_bitvec_ref()[last_pos..pos]);
            result.extend_from_bitslice(new.as_bitslice());
            last_pos = pos + old.len();
        }
        result.extend_from_bitslice(&slf.as_bitvec_ref()[last_pos..]);

        *slf.as_mut_bitvec_ref() = result;
        Ok(slf)
    }

    /// Insert bits at position pos. Returns self.
    ///
    /// :param pos: The bit position to insert at.
    /// :param bs: The bits to insert.
    /// :return: self
    ///
    /// Raises ValueError if pos < 0 or pos > len(self).
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b1011')
    ///     >>> a.insert(2, '0b00')
    ///     Mutibs('0b100011')
    ///
    pub fn insert<'a>(
        mut slf: PyRefMut<'a, Self>,
        mut pos: i64,
        bs: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        // Check for self assignment
        let bs = if bs.as_ptr() == slf.as_ptr() {
            BorrowedOrOwnedTibs::Owned(slf.__copy__().as_tibs())
        } else {
            tibs_from_any(bs)?
        };
        if bs.len() == 0 {
            return Ok(slf);
        }
        if pos < 0 {
            pos += slf.len() as i64;
        }
        // Keep Python insert behaviour. Clips to start and end.
        if pos < 0 {
            pos = 0;
        } else if pos > slf.len() as i64 {
            pos = slf.len() as i64;
        }
        if bs.len() == 1 {
            slf.as_mut_bitvec_ref()
                .insert(pos as usize, bs.as_bitslice()[0]);
            return Ok(slf);
        }
        let tail = slf.as_mut_bitvec_ref().split_off(pos as usize);
        slf.as_mut_bitvec_ref()
            .extend_from_bitslice(bs.as_bitslice());
        slf.as_mut_bitvec_ref().extend_from_bitslice(&tail);
        Ok(slf)
    }

    /// Shift bits to the left in-place.
    ///
    /// :param n: The number of bits to shift. Must be >= 0.
    /// :return: self
    ///
    /// Raises ValueError if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> b = Mutibs('0b001100')
    ///     >>> b <<= 2
    ///     >>> b.bin
    ///     '110000'
    ///
    pub fn __ilshift__(mut slf: PyRefMut<'_, Self>, n: i64) -> PyResult<()> {
        let shift = validate_shift(&*slf, n)?;
        slf.as_mut_bitvec_ref().shift_left(shift);
        Ok(())
    }

    /// Shift bits to the right in-place.
    ///
    /// :param n: The number of bits to shift. Must be >= 0.
    /// :return: self
    ///
    /// Raises ValueError if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> b = Mutibs('0b001100')
    ///     >>> b >>= 2
    ///     >>> b.bin
    ///     '000011'
    ///
    pub fn __irshift__(mut slf: PyRefMut<'_, Self>, n: i64) -> PyResult<()> {
        let shift = validate_shift(&*slf, n)?;
        slf.as_mut_bitvec_ref().shift_right(shift);
        Ok(())
    }

    /// Return the Mutibs as a bytes object.
    ///
    /// Raises ValueError if the length is not a multiple of 8.
    pub fn __bytes__(&self) -> PyResult<Vec<u8>> {
        self.to_bytes()
    }

    /// Return new Mutibs consisting of n concatenations of self.
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

    /// Return Mutibs consisting of n concatenations of self.
    ///
    /// Called for expressions of the form 'a = 3*b'.
    ///
    /// n -- The number of concatenations. Must be >= 0.
    ///
    pub fn __rmul__(&self, n: i64) -> PyResult<Self> {
        self.__mul__(n)
    }

    /// Iteration is not supported for mutable objects.
    pub fn __iter__(&self) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "Mutibs objects are not iterable. You can use .to_tibs() or .as_tibs() to convert to a Tibs object that does support iteration.",
        ))
    }

    /// In-place bit-wise 'and'.
    pub fn __iand__(mut slf: PyRefMut<'_, Self>, bs: &Bound<'_, PyAny>) -> PyResult<()> {
        let other = tibs_from_any(bs)?;
        slf.iand(other.as_bitslice())
    }

    /// In-place bit-wise 'or'.
    pub fn __ior__(mut slf: PyRefMut<'_, Self>, bs: &Bound<'_, PyAny>) -> PyResult<()> {
        let other = tibs_from_any(bs)?;
        slf.ior(other.as_bitslice())
    }

    /// In-place bit-wise 'xor'.
    pub fn __ixor__(mut slf: PyRefMut<'_, Self>, bs: &Bound<'_, PyAny>) -> PyResult<()> {
        let other = tibs_from_any(bs)?;
        slf.ixor(other.as_bitslice())
    }

    /// In-place multiplication by a non-negative integer.
    pub fn __imul__(mut slf: PyRefMut<'_, Self>, n: i64) -> PyResult<()> {
        match n {
            i if i < 0 => Err(PyValueError::new_err(
                "Cannot multiply by a negative integer.",
            )),
            0 => {
                slf.clear();
                Ok(())
            }
            1 => Ok(()),
            i => {
                let n = i as usize;
                let orig_data = slf.to_bitvec();
                let len = slf.len();
                slf.reserve(len * (n - 1));
                let mut mul = 1;
                while mul * 2 <= n {
                    // Double the length
                    let current = slf.to_bitvec();
                    slf.as_mut_bitvec_ref().extend_from_bitslice(&current);
                    mul *= 2;
                }
                while mul < n {
                    slf.as_mut_bitvec_ref().extend_from_bitslice(&orig_data);
                    mul += 1;
                }
                Ok(())
            }
        }
    }
}
