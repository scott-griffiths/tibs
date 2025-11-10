use crate::bits::{bits_from_any, Tibs};
use crate::core::str_to_bits;
use crate::core::validate_logical_op_lengths;
use crate::core::BitCollection;
use crate::helpers::{validate_index, validate_slice, BV};
use crate::iterator::ChunksIterator;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::{PyAnyMethods, PyTypeMethods};
use pyo3::types::{PyBool, PySlice};
use pyo3::types::{PySliceMethods, PyType};
use pyo3::{pyclass, pymethods, PyRef, PyResult, Python};
use pyo3::{pyfunction, PyRefMut};
use pyo3::{Bound, IntoPyObject, Py, PyAny};
use std::ops::Not;

#[pyfunction]
pub fn mutable_bits_from_any(any: Py<PyAny>, py: Python) -> PyResult<Mutibs> {
    let any_bound = any.bind(py);

    if let Ok(any_bits) = any_bound.extract::<PyRef<Tibs>>() {
        return Ok(any_bits.to_mutable_bits());
    }

    if let Ok(any_mutable_bits) = any_bound.extract::<PyRef<Mutibs>>() {
        return Ok(any_mutable_bits.__copy__());
    }

    if let Ok(any_string) = any_bound.extract::<String>() {
        let bits = str_to_bits(any_string)?;
        return Ok(bits.to_mutable_bits());
    }
    if let Ok(any_bytes) = any_bound.extract::<Vec<u8>>() {
        let bits = <Tibs as BitCollection>::from_bytes(any_bytes);
        return Ok(bits.to_mutable_bits());
    }
    let type_name = match any_bound.get_type().name() {
        Ok(name) => name.to_string(),
        Err(_) => "<unknown>".to_string(),
    };
    Err(PyTypeError::new_err(format!(
        "Cannot convert object of type {} to a Mutibs object.",
        type_name
    )))
}

///     A mutable container of binary data.
///
///     To construct, use a builder 'from' method:
///
///     * ``Mutibs.from_bytes(b)`` - Create directly from a ``bytes`` object.
///     * ``Mutibs.from_string(s)`` - Use a formatted string.
///     * ``Mutibs.from_bools(i)`` - Convert each element in ``i`` to a bool.
///     * ``Mutibs.from_zeros(length)`` - Initialise with ``length`` '0' bits.
///     * ``Mutibs.from_ones(length)`` - Initialise with ``length`` '1' bits.
///     * ``Mutibs.from_random(length, [seed])`` - Initialise with ``length`` pseudo-randomly set bits.
///     * ``Mutibs.from_dtype(dtype, value)`` - Combine a data type with a value.
///     * ``Mutibs.from_joined(iterable)`` - Concatenate an iterable of objects.
///
///     Using the constructor ``Mutibs(s)`` is an alias for ``Mutibs.from_string(s)``.
///
#[pyclass(freelist = 8, module = "tibs")]
pub struct Mutibs {
    pub(crate) inner: Tibs,
}

impl Mutibs {
    fn _getslice_with_step(&self, start_bit: i64, end_bit: i64, step: i64) -> PyResult<Self> {
        self.inner
            ._getslice_with_step(start_bit, end_bit, step)
            .map(|bits| Mutibs { inner: bits })
    }
}

#[pymethods]
impl Mutibs {
    #[new]
    #[pyo3(signature = (s = None))]
    pub fn py_new(s: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(s) = s else {
            return Ok(BitCollection::empty());
        };
        if let Ok(string_s) = s.extract::<String>() {
            return str_to_bits(string_s).map(|bits| bits.to_mutable_bits());
        }

        // If it's not a string, build a more helpful error message.
        let type_name = s.get_type().name()?;
        let mut err = format!(
            "Expected a str for Mutibs constructor, but received a {}. ",
            type_name
        );

        if s.is_instance_of::<Tibs>() {
            err.push_str(
                "You can use the 'to_mutable_bits()' method on the `Bits` instance instead.",
            );
        } else if s.is_instance_of::<pyo3::types::PyBytes>()
            || s.is_instance_of::<pyo3::types::PyByteArray>()
            || s.is_instance_of::<pyo3::types::PyMemoryView>()
        {
            err.push_str("You can use 'Mutibs.from_bytes()' instead.");
        } else if s.is_instance_of::<pyo3::types::PyInt>() {
            err.push_str("Perhaps you want to use 'Mutibs.from_zeros()', 'Mutibs.from_ones()' or 'Mutibs.from_random()'?");
        } else if s.is_instance_of::<pyo3::types::PyTuple>()
            || s.is_instance_of::<pyo3::types::PyList>()
        {
            err.push_str(
                "Perhaps you want to use 'Mutibs.from_joined()' or 'Mutibs.from_bools()' instead?",
            );
        } else {
            err.push_str(
                "To create from other types use from_bytes(), from_bools(), from_joined(), \
                 from_ones(), from_zeros(), from_dtype() or from_random().",
            );
        }
        Err(PyTypeError::new_err(err))
    }

    /// Return True if two Mutibs have the same binary representation.
    ///
    /// The right hand side will be promoted to a Mutibs if needed and possible.
    ///
    /// >>> Mutibs('0xf2') == '0b11110010'
    /// True
    ///
    pub fn __eq__(&self, other: Py<PyAny>, py: Python) -> bool {
        let obj = other.bind(py);
        if let Ok(b) = obj.extract::<PyRef<Tibs>>() {
            return self.inner.data == b.data;
        }
        if let Ok(b) = obj.extract::<PyRef<Mutibs>>() {
            return self.inner.data == b.inner.data;
        }
        match bits_from_any(other, py) {
            Ok(b) => self.inner.data == b.data,
            Err(_) => false,
        }
    }

    /// Return string representations for printing.
    pub fn __str__(&self) -> String {
        self.inner.__str__()
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



    pub fn _overwrite(&mut self, start: usize, value: &Tibs) {
        self.inner.data[start..start + value.len()].copy_from_bitslice(&value.data);
    }

    pub fn _set_slice(&mut self, start: usize, end: usize, value: &Tibs) {
        if end - start == value.len() {
            // This is an overwrite, so no need to move data around.
            self._overwrite(start, value);
        } else {
            if start == end {
                // Not sure why but splice doesn't work for this case, so we do it explicitly
                let tail = self.inner.data.split_off(start);
                self.inner.data.extend_from_bitslice(&value.data);
                self.inner.data.extend_from_bitslice(&tail);
            } else {
                let tail = self.inner.data.split_off(end);
                self.inner.data.truncate(start);
                self.inner.data.extend_from_bitslice(&value.data);
                self.inner.data.extend_from_bitslice(&tail);
            }
        }
    }

    pub fn _ixor(&mut self, other: &Mutibs) -> PyResult<()> {
        validate_logical_op_lengths(self.len(), other.len())?;
        self.inner.data ^= &other.inner.data;
        Ok(())
    }

    pub fn _ior(&mut self, other: &Mutibs) -> PyResult<()> {
        validate_logical_op_lengths(self.len(), other.len())?;
        self.inner.data |= &other.inner.data;
        Ok(())
    }

    pub fn _iand(&mut self, other: &Mutibs) -> PyResult<()> {
        validate_logical_op_lengths(self.len(), other.len())?;
        self.inner.data &= &other.inner.data;
        Ok(())
    }

    pub fn _or(&self, other: &Tibs) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(Mutibs::logical_or(self, other))
    }

    pub fn _and(&self, other: &Tibs) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(Mutibs::logical_and(self, other))
    }

    pub fn _xor(&self, other: &Tibs) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(Mutibs::logical_xor(self, other))
    }

    #[staticmethod]
    pub fn _from_u64(value: u64, length: usize) -> Self {
        BitCollection::from_u64(value, length)
    }

    #[staticmethod]
    pub fn _from_i64(value: i64, length: usize) -> Self {
        BitCollection::from_i64(value, length)
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
    ///     c = Mutibs.from_string("u12 = 31, f16=-0.25")
    ///
    /// The `__init__` method for `Mutibs` redirects to the `from_string` method and is sometimes more convenient:
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs("0xff01")  # Mutibs(s) is equivalent to Mutibs.from_string(s)
    #[classmethod]
    pub fn from_string(_cls: &Bound<'_, PyType>, s: String) -> PyResult<Self> {
        str_to_bits(s).map(|bits| bits.to_mutable_bits())
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
    pub fn from_zeros(_cls: &Bound<'_, PyType>, length: i64) -> PyResult<Self> {
        if length < 0 {
            return Err(PyValueError::new_err(format!(
                "Negative bit length given: {}.",
                length
            )));
        }
        Ok(BitCollection::from_zeros(length as usize))
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
    pub fn from_ones(_cls: &Bound<'_, PyType>, length: i64) -> PyResult<Self> {
        if length < 0 {
            return Err(PyValueError::new_err(format!(
                "Negative bit length given: {}.",
                length
            )));
        }
        Ok(BitCollection::from_ones(length as usize))
    }

    /// Create a new instance from an iterable by converting each element to a bool.
    ///
    /// :param i: The iterable to convert to a :class:`Mutibs`.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_bools([False, 0, 1, "Steven"])  # binary 0011
    ///
    #[classmethod]
    pub fn from_bools(
        _cls: &Bound<'_, PyType>,
        values: Vec<Py<PyAny>>,
        py: Python,
    ) -> PyResult<Self> {
        Ok(Tibs::from_bools(_cls, values, py)?.to_mutable_bits())
    }

    /// Create a new instance with all bits pseudo-randomly set.
    ///
    /// :param length: The number of bits to set. Must be positive.
    /// :param seed: An optional seed as a bytes or bytearray.
    /// :return: A newly constructed ``Mutibs`` with random data.
    ///
    /// Note that this uses a pseudo-random number generator and so
    /// might not suitable for cryptographic or other more serious purposes.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_random(1000000)  # A million random bits
    ///     b = Mutibs.from_random(100, b'a_seed')
    ///
    #[classmethod]
    #[pyo3(signature = (length, seed=None))]
    pub fn from_random(_cls: &Bound<'_, PyType>, length: i64, seed: Option<Vec<u8>>) -> PyResult<Self> {
        Ok(Tibs::from_random(_cls, length, seed)?.to_mutable_bits())
    }


    /// Create a new instance from a bytes object.
    ///
    /// :param b: The bytes object to convert to a :class:`Mutibs`.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_bytes(b"some_bytes_maybe_from_a_file")
    ///
    #[classmethod]
    pub fn from_bytes(_cls: &Bound<'_, PyType>, data: Vec<u8>) -> Self {
        BitCollection::from_bytes(data)
    }

    #[staticmethod]
    pub fn _from_bytes_with_offset(data: Vec<u8>, offset: usize) -> Self {
        Self {
            inner: Tibs::_from_bytes_with_offset(data, offset),
        }
    }

    /// Create a new instance by concatenating a sequence of Bits objects.
    ///
    /// This method concatenates a sequence of Bits objects into a single Mutibs object.
    ///
    /// :param sequence: A sequence to concatenate. Items can either be a Bits object, or a string or bytes-like object that could create one via the :meth:`from_string` or :meth:`from_bytes` methods.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_joined([f'u6={x}' for x in range(64)])
    ///     b = Mutibs.from_joined(['0x01', 'i4 = -1', b'some_bytes'])
    ///
    #[classmethod]
    pub fn from_joined(
        _cls: &Bound<'_, PyType>,
        sequence: &Bound<'_, PyAny>,
        py: Python,
    ) -> PyResult<Self> {
        Ok(Tibs::from_joined(_cls, sequence, py)?.to_mutable_bits())
    }

    pub fn _to_u64(&self, start: usize, length: usize) -> u64 {
        self.inner._to_u64(start, length)
    }

    pub fn _to_i64(&self, start: usize, length: usize) -> i64 {
        self.inner._to_i64(start, length)
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn _getindex(&self, bit_index: i64) -> PyResult<bool> {
        self.inner._getindex(bit_index)
    }

    pub fn _getslice(&self, start_bit: usize, length: usize) -> PyResult<Self> {
        self.inner
            ._getslice(start_bit, length)
            .map(|bits| Mutibs { inner: bits })
    }

    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = key.py();
        // Handle integer indexing
        if let Ok(index) = key.extract::<i64>() {
            let value: bool = self._getindex(index)?;
            let py_value = PyBool::new(py, value);
            return Ok(py_value.to_owned().into());
        }

        // Handle slice indexing
        if let Ok(slice) = key.downcast::<PySlice>() {
            let indices = slice.indices(self.len() as isize)?;
            let start: i64 = indices.start.try_into()?;
            let stop: i64 = indices.stop.try_into()?;
            let step: i64 = indices.step.try_into()?;

            let result = if step == 1 {
                if start < stop {
                    self._getslice(start as usize, (stop - start) as usize)?
                } else {
                    Mutibs::empty()
                }

            } else {
                self._getslice_with_step(start, stop, step)?
            };
            let py_obj = Py::new(py, result)?.into_pyobject(py)?;
            return Ok(py_obj.into());
        }

        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    /// Set a bit or a slice of bits.
    ///
    /// :param key: The index or slice to set.
    /// :param value: For a single index, a boolean value. For a slice, anything that can be converted to Bits.
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
        value: Py<PyAny>,
        py: Python,
    ) -> PyResult<()> {
        let length = slf.len();
        if let Ok(mut index) = key.extract::<i64>() {
            if index < 0 {
                index += length as i64;
            }
            if index < 0 || index >= length as i64 {
                return Err(PyIndexError::new_err(format!(
                    "Bit index {index} out of range for length {length}"
                )));
            }
            slf._set_index(value.is_truthy(py)?, index)?;
            return Ok(());
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            // Need to guard against value being self
            let bs = if value.as_ptr() == slf.as_ptr() {
                Tibs::new(slf.inner.data.clone())
            } else {
                bits_from_any(value, py)?
            };

            let indices = slice.indices(length as isize)?;
            let start: i64 = indices.start.try_into()?;
            let stop: i64 = indices.stop.try_into()?;
            let step: i64 = indices.step.try_into()?;

            if step == 1 {
                debug_assert!(start >= 0);
                debug_assert!(stop >= 0);
                slf._set_slice(start as usize, stop as usize, &bs);
                return Ok(());
            }
            if step == 0 {
                return Err(PyValueError::new_err("The step in __setitem__ must not be zero."));
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
                let v = bs.data[k];
                slf.inner.data.set(pos, v);
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
            self.inner.data.remove(index as usize);
            return Ok(());
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let indices = slice.indices(length as isize)?;
            let start: i64 = indices.start.try_into()?;
            let stop: i64 = indices.stop.try_into()?;
            let step: i64 = indices.step.try_into()?;
            if step == 1 {
                if stop > start {
                    self.inner.data.drain(start as usize..stop as usize);
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
                    self.inner.data.remove(i);
                }
            }
            return Ok(());
        }
        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    /// Return the Mutibs as bytes, padding with zero bits if needed.
    ///
    /// Up to seven zero bits will be added at the end to byte align.
    ///
    /// :return: The Mutibs as bytes.
    ///
    pub fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    pub fn _slice_to_bin(&self, start: usize, end: usize) -> String {
        self.inner._slice_to_bin(start, end)
    }

    pub fn _slice_to_oct(&self, start: usize, end: usize) -> PyResult<String> {
        self.inner._slice_to_oct(start, end)
    }

    pub fn _slice_to_hex(&self, start: usize, end: usize) -> PyResult<String> {
        self.inner._slice_to_hex(start, end)
    }

    pub fn _slice_to_bytes(&self, start: usize, end: usize) -> PyResult<Vec<u8>> {
        self.inner._slice_to_bytes(start, end)
    }

    pub fn _to_int_byte_data(&self, signed: bool) -> Vec<u8> {
        self.inner._to_int_byte_data(signed)
    }

    /// Return whether the current Mutibs starts with prefix.
    ///
    /// :param prefix: The Bits to search for.
    /// :return: True if the Bits starts with the prefix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b101100').starts_with('0b101')
    ///     True
    ///     >>> Mutibs('0b101100').starts_with('0b100')
    ///     False
    ///
    pub fn starts_with(&self, prefix: Py<PyAny>, py: Python) -> PyResult<bool> {
        self.inner.starts_with(prefix, py)
    }

    /// Return whether the current Mutibs ends with suffix.
    ///
    /// :param suffix: The Bits to search for.
    /// :return: True if the Bits ends with the suffix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b101100').ends_with('0b10-')
    ///     True
    ///     >>> Mutibs('0b101100').ends_with('0b101')
    ///     False
    ///
    pub fn ends_with(&self, suffix: Py<PyAny>, py: Python) -> PyResult<bool> {
        self.inner.ends_with(suffix, py)
    }

    /// Bit-wise 'and' between two Mutibs. Returns new Mutibs.
    ///
    /// Raises ValueError if the two Mutibs have differing lengths.
    ///
    pub fn __and__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let other = bits_from_any(bs, py)?;
        self._and(&other)
    }

    /// Bit-wise 'or' between two Mutibs. Returns new Mutibs.
    ///
    /// Raises ValueError if the two Mutibs have differing lengths.
    ///
    pub fn __or__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let other = bits_from_any(bs, py)?;
        self._or(&other)
    }

    /// Bit-wise 'xor' between two Mutibs. Returns new Mutibs.
    ///
    /// Raises ValueError if the two Mutibs have differing lengths.
    ///
    pub fn __xor__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let other = bits_from_any(bs, py)?;
        self._xor(&other)
    }

    /// Reverse bit-wise 'and' between two Mutibs. Returns new Mutibs.
    ///
    /// This method is used when the RHS is a Mutibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Mutibs have differing lengths.
    ///
    pub fn __rand__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let other = mutable_bits_from_any(bs, py)?;
        other._and(&self.inner)
    }

    /// Reverse bit-wise 'or' between two Mutibs. Returns new Mutibs.
    ///
    /// This method is used when the RHS is a Mutibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Mutibs have differing lengths.
    ///
    pub fn __ror__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let other = mutable_bits_from_any(bs, py)?;
        other._or(&self.inner)
    }

    /// Reverse bit-wise 'xor' between two Mutibs. Returns new Mutibs.
    ///
    /// This method is used when the RHS is a Mutibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Mutibs have differing lengths.
    ///
    pub fn __rxor__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let other = mutable_bits_from_any(bs, py)?;
        other._xor(&self.inner)
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
    pub fn rol<'a>(
        mut slf: PyRefMut<'a, Self>,
        n: i64,
        start: Option<i64>,
        end: Option<i64>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if slf.is_empty() {
            return Err(PyValueError::new_err("Cannot rotate an empty Mutibs."));
        }
        if n < 0 {
            return Err(PyValueError::new_err("Cannot rotate by a negative amount."));
        }

        let (start, end) = validate_slice(slf.len(), start, end)?;
        let n = (n % (end as i64 - start as i64)) as usize;
        slf.inner.data[start..end].rotate_left(n);
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
    pub fn ror<'a>(
        mut slf: PyRefMut<'a, Self>,
        n: i64,
        start: Option<i64>,
        end: Option<i64>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if slf.is_empty() {
            return Err(PyValueError::new_err("Cannot rotate an empty Mutibs."));
        }
        if n < 0 {
            return Err(PyValueError::new_err("Cannot rotate by a negative amount."));
        }

        let (start, end) = validate_slice(slf.len(), start, end)?;
        let n = (n % (end as i64 - start as i64)) as usize;
        slf.inner.data[start..end].rotate_right(n);
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
            slf._set_index(v, index)?;
        } else if pos.is_instance_of::<pyo3::types::PyRange>() {
            let start = pos.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
            let stop = pos.getattr("stop")?.extract::<i64>()?;
            let step = pos.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
            slf._set_from_slice(v, start, stop, step)?;
        }
        // Otherwise treat as a sequence
        else {
            // Convert to Vec<i64> if possible
            let indices = pos.extract::<Vec<i64>>()?;
            slf._set_from_sequence(v, indices)?;
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
    pub fn count(&self, value: Py<PyAny>, py: Python) -> PyResult<usize> {
        self.inner.count(value, py)
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
        self.inner.all()
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
        self.inner.any()
    }

    pub fn _find(&self, b: &Tibs, start: usize, end: usize, bytealigned: bool) -> Option<usize> {
        self.inner._find(b, start, end, bytealigned)
    }

    pub fn _rfind(&self, b: &Tibs, start: usize, end: usize, bytealigned: bool) -> Option<usize> {
        self.inner._rfind(b, start, end, bytealigned)
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
                slf.inner.data = std::mem::take(&mut slf.inner.data).not();
            }
            Some(p) => {
                if let Ok(pos) = p.extract::<i64>() {
                    let pos: usize = validate_index(pos, slf.len())?;
                    let value = slf.inner.data[pos];
                    slf.inner.data.set(pos, !value);
                } else if let Ok(pos_list) = p.extract::<Vec<i64>>() {
                    for pos in pos_list {
                        let pos: usize = validate_index(pos, slf.len())?;
                        let value = slf.inner.data[pos];
                        slf.inner.data.set(pos, !value);
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
        slf.inner.data.reverse();
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
    pub fn byte_swap(mut slf: PyRefMut<'_, Self>, byte_length: Option<i64>) -> PyResult<PyRefMut<'_, Self>> {
        let len = slf.len();
        if len % 8 != 0 {
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
        if self_byte_length % byte_length != 0 {
            return Err(PyValueError::new_err(format!(
                "The Mutibs to byte_swap is {self_byte_length} bytes long, but it needs to be a multiple of {byte_length} bytes."
            )));
        }

        let mut bytes = slf.inner._slice_to_bytes(0, len)?;
        for chunk in bytes.chunks_mut(byte_length) {
            chunk.reverse();
        }
        slf.inner.data = BV::from_vec(bytes);
        Ok(slf)
    }

    /// Return the instance with every bit inverted.
    ///
    /// Raises ValueError if the Mutibs is empty.
    ///
    pub fn __invert__(&self) -> PyResult<Self> {
        if self.inner.data.is_empty() {
            return Err(PyValueError::new_err("Cannot invert empty Mutibs."));
        }
        Ok(Mutibs::new(self.inner.data.clone().not()))
    }

    /// Return new Mutibs shifted by n to the left.
    ///
    /// n -- the number of bits to shift. Must be >= 0.
    ///
    pub fn __lshift__(&self, n: i64) -> PyResult<Self> {
        Ok(Mutibs::new(self.inner.__lshift__(n)?.data))
    }

    /// Return new Mutibs shifted by n to the right.
    ///
    /// n -- the number of bits to shift. Must be >= 0.
    ///
    pub fn __rshift__(&self, n: i64) -> PyResult<Self> {
        Ok(Mutibs::new(self.inner.__rshift__(n)?.data))
    }

    pub fn _set_index(&mut self, value: bool, index: i64) -> PyResult<()> {
        self._set_from_sequence(value, vec![index])
    }

    // Just redirects to the Bits._chunks method. Not public part of Python interface
    // as it's only used internally in things like pp().
    #[pyo3(signature = (chunk_size, count = None))]
    pub fn _chunks(
        slf: PyRef<'_, Self>,
        chunk_size: usize,
        count: Option<usize>,
    ) -> PyResult<Py<ChunksIterator>> {
        let py = slf.py();
        let bits_instance = slf.to_bits();
        let bits_py_obj = Py::new(py, bits_instance)?;
        let bits_py_ref = bits_py_obj.bind(py);
        bits_py_ref
            .call_method1("_chunks", (chunk_size, count))?
            .extract()
    }

    pub fn _set_from_slice(
        &mut self,
        value: bool,
        start: i64,
        stop: i64,
        step: i64,
    ) -> PyResult<()> {
        let len = self.inner.len() as i64;
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
                self.inner.data.set_unchecked(index, value);
            }
            index += positive_step;
        }

        Ok(())
    }

    /// Return a new copy of the Mutibs for the copy module.
    pub fn __copy__(&self) -> Self {
        Mutibs::new(self.inner.data.clone())
    }

    /// Create and return a Bits instance from a copy of the Mutibs data.
    ///
    /// This copies the underlying binary data, giving a new independent Bits object.
    /// If you no longer need the Mutibs, consider using :meth:`as_bits` instead to avoid the copy.
    ///
    /// :return: A new Bits instance with the same bit data.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b1011')
    ///     >>> b = a.to_bits()
    ///     >>> a
    ///     Mutibs('0b1011')
    ///     >>> b
    ///     Bits('0b1101')
    ///
    pub fn to_bits(&self) -> Tibs {
        Tibs::new(self.inner.data.clone())
    }

    /// Create and return a Bits instance by moving the Mutibs data.
    ///
    /// The data is moved to the new Bits, so this Mutibs will be empty after the operation.
    /// This is more efficient than :meth:`to_bits` if you no longer need the Mutibs.
    ///
    /// It will try to reclaim any excess memory capacity that the Mutibs may have had.
    ///
    /// :return: A Bits instance with the same bit data.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b1011')
    ///     >>> b = a.as_bits()
    ///     >>> a
    ///     Mutibs()
    ///     >>> b
    ///     Bits('0b1101')
    ///
    pub fn as_bits(&mut self) -> Tibs {
        let mut data = std::mem::take(&mut self.inner.data);
        data.shrink_to_fit();
        Tibs::new(data)
    }

    /// Clear all bits, making the Mutibs empty.
    ///
    /// This doesn't change the allocated capacity, so won't free up any memory.
    ///
    pub fn clear(&mut self) {
        self.inner.data.clear();
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
        self.inner.data.capacity()
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
        self.inner.data.reserve(additional);
    }

    /// Concatenate Mutibs and return a new Mutibs.
    pub fn __add__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let bs = bits_from_any(bs, py)?;
        let mut data = BV::with_capacity(self.len() + bs.len());
        data.extend_from_bitslice(&self.inner.data);
        data.extend_from_bitslice(&bs.data);
        Ok(Mutibs::new(data))
    }

    /// Concatenate Mutibs and return a new Mutibs.
    pub fn __radd__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let mut bs = mutable_bits_from_any(bs, py)?;
        bs.inner.data.extend_from_bitslice(&self.inner.data);
        Ok(bs)
    }

    /// Concatenate Bits in-place.
    pub fn __iadd__<'a>(
        mut slf: PyRefMut<'a, Self>,
        bs: Py<PyAny>,
        py: Python<'_>,
    ) -> PyResult<()> {
        // Check if bs is the same object as slf
        if bs.as_ptr() == slf.as_ptr() {
            // If bs is slf, clone inner bits first then append
            let bits_clone = slf.inner.data.clone();
            slf.inner.data.extend_from_bitslice(&bits_clone);
        } else {
            // Normal case - convert bs to Bits and append
            let bs = bits_from_any(bs, py)?;
            slf.inner.data.extend_from_bitslice(&bs.data);
        }
        Ok(())
    }

    /// Append bits to the end of the current Mutibs in-place.
    ///
    /// :param bs: The bits to append.
    /// :return: self
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0x0f')
    ///     >>> a.append('0x0a')
    ///     Mutibs('0x0f0a')
    ///
    pub fn append<'a>(
        mut slf: PyRefMut<'a, Self>,
        bs: Py<PyAny>,
        py: Python<'_>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        // Check if bs is the same object as slf
        if bs.as_ptr() == slf.as_ptr() {
            // If bs is slf, clone inner bits first then append
            let bits_clone = slf.inner.data.clone();
            slf.inner.data.extend_from_bitslice(&bits_clone);
        } else {
            // Normal case - convert bs to Bits and append
            let bs = bits_from_any(bs, py)?;
            slf.inner.data.extend_from_bitslice(&bs.data);
        }
        Ok(slf)
    }

    ///Prepend bits to the beginning of the current Mutibs in-place.
    ///
    /// :param bs: The bits to prepend.
    /// :return: self
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0x0f')
    ///     >>> a.prepend('0x0a')
    ///     Mutibs('0x0a0f')
    ///
    pub fn prepend<'a>(
        mut slf: PyRefMut<'a, Self>,
        bs: Py<PyAny>,
        py: Python<'_>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        // Check for self-prepending
        if bs.as_ptr() == slf.as_ptr() {
            let mut new_data = slf.inner.data.clone();
            new_data.extend_from_bitslice(&slf.inner.data);
            slf.inner.data = new_data;
        } else {
            let to_prepend = bits_from_any(bs, py)?;
            if to_prepend.is_empty() {
                return Ok(slf);
            }
            let mut new_data = to_prepend.data;
            new_data.extend_from_bitslice(&slf.inner.data);
            slf.inner.data = new_data;
        }
        Ok(slf)
    }

    /// Inserts another Bits or Mutibs at bit position pos. Returns self.
    ///
    /// :param pos: The bit position to insert at.
    /// :param bs: The Bits to insert.
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
        bs: Py<PyAny>,
        py: Python<'_>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        // Check for self assignment
        let bs = if bs.as_ptr() == slf.as_ptr() {
            Mutibs::new(slf.inner.data.clone())
        } else {
            mutable_bits_from_any(bs, py)?
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
            slf.inner.data.insert(pos as usize, bs.inner.data[0]);
            return Ok(slf);
        }
        let tail = slf.inner.data.split_off(pos as usize);
        slf.inner.data.extend_from_bitslice(&bs.inner.data);
        slf.inner.data.extend_from_bitslice(&tail);
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
    pub fn __ilshift__<'a>(mut slf: PyRefMut<'a, Self>, n: i64) -> PyResult<()> {
        let shift = slf.inner._validate_shift(n)?;
        slf.inner.data.shift_left(shift);
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
    pub fn __irshift__<'a>(mut slf: PyRefMut<'a, Self>, n: i64) -> PyResult<()> {
        let shift = slf.inner._validate_shift(n)?;
        slf.inner.data.shift_right(shift);
        Ok(())
    }

    pub fn __bytes__(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    /// Return new Mutibs consisting of n concatenations of self.
    ///
    /// Called for expression of the form 'a = b*3'.
    ///
    /// n -- The number of concatenations. Must be >= 0.
    ///
    pub fn __mul__(&self, n: i64) -> PyResult<Self> {
        let x = self.inner.__mul__(n)?;
        Ok(Mutibs::new(x.data))
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

    pub fn __iter__(&self) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "Mutibs objects are not iterable. You can use .to_bits() or .as_bits() to convert to a Bits object that does support iteration."
        ))
    }

}