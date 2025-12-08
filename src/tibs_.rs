use crate::core::validate_logical_op_lengths;
use crate::core::{str_to_mutibs, BitCollection};
use crate::helpers::{find_bitvec, validate_index, validate_slice, BV};
use crate::iterator::{BoolIterator, ChunksIterator, FindAllIterator};
use crate::mutibs::mutibs_from_any;
use crate::mutibs::Mutibs;
use bitvec::prelude::*;
use bytemuck;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyFloat, PyInt, PyMemoryView, PySlice, PyType};
use pyo3::{pyclass, pymethods, IntoPyObject, PyRef, PyResult};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::Not;

fn promote_to_tibs(any: &Bound<'_, PyAny>) -> PyResult<Tibs> {
    // Is it a string?
    if let Ok(any_string) = any.extract::<String>() {
        return Ok(str_to_mutibs(any_string)?.as_tibs());
    }

    // Is it a bytes, bytearray or memoryview?
    if any.is_instance_of::<PyBytes>()
        || any.is_instance_of::<PyByteArray>()
        || any.is_instance_of::<PyMemoryView>()
    {
        if let Ok(any_bytes) = any.extract::<Vec<u8>>() {
            return Ok(<Tibs as BitCollection>::from_bytes(any_bytes));
        }
    }

    // Is it an iterable that we can convert each element to a bool?
    if let Ok(iter) = any.try_iter() {
        let mut bv = BV::new();
        for item in iter {
            bv.push(item?.is_truthy()?);
        }
        return Ok(Tibs::new(bv));
    }
    let type_name = match any.get_type().name() {
        Ok(name) => name.to_string(),
        Err(_) => "<unknown>".to_string(),
    };
    let mut err = format!("Cannot promote object of type {type_name} to a Tibs object. ");
    if any.is_instance_of::<PyInt>() {
        err.push_str("Perhaps you want to use 'Tibs.from_zeros()', 'Tibs.from_ones()' or 'Tibs.from_random()'?");
    };
    Err(PyTypeError::new_err(err))
}

pub fn tibs_from_any(any: &Bound<'_, PyAny>) -> PyResult<Tibs> {
    // Is it of type Tibs?
    if let Ok(tibs_ref) = any.extract::<PyRef<Tibs>>() {
        return Ok(tibs_ref.clone()); // TODO: Expensive clone
    }

    // Is it of type Mutibs?
    if let Ok(mutibs_ref) = any.extract::<PyRef<Mutibs>>() {
        return Ok(mutibs_ref.to_tibs()); // TODO: Expensive clone
    }

    promote_to_tibs(any)
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
///     * ``Tibs.from_bools(i)`` - Convert each element in ``i`` to a bool.
///     * ``Tibs.from_zeros(length)`` - Initialise with ``length`` '0' bits.
///     * ``Tibs.from_ones(length)`` - Initialise with ``length`` '1' bits.
///     * ``Tibs.from_random(length, [seed])`` - Initialise with ``length`` pseudo-randomly set bits.
///     * ``Tibs.from_joined(iterable)`` - Concatenate an iterable of objects.
///
///     Using ``Tibs(auto)`` will try to delegate to ``from_string``, ``from_bytes`` or ``from_bools``.
///
#[derive(Clone)]
#[pyclass(frozen, module = "tibs")]
pub struct Tibs {
    pub(crate) data: BV,
}

impl Hash for Tibs {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);

        let bits = self.data.as_bitslice();

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

impl Tibs {
    pub(crate) fn _getslice_with_step(
        &self,
        start_bit: i64,
        end_bit: i64,
        step: i64,
    ) -> PyResult<Self> {
        if step == 0 {
            return Err(PyValueError::new_err("Slice step cannot be zero."));
        }
        // Note that a start_bit or end_bit of -1 means to stop at the beginning when using a negative step.
        // Otherwise they should both be positive indices.
        debug_assert!(start_bit >= -1);
        debug_assert!(end_bit >= -1);
        debug_assert!(step != 0);
        if start_bit < -1 || end_bit < -1 {
            return Err(PyValueError::new_err(
                "Indices less than -1 are not valid values.",
            ));
        }
        if step > 0 {
            if start_bit >= end_bit {
                return Ok(BitCollection::empty());
            }
            if end_bit as usize > self.len() {
                return Err(PyValueError::new_err(
                    "Slice end goes past the end of the Tibs.",
                ));
            }
            Ok(Tibs::new(
                self.data[start_bit as usize..end_bit as usize]
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
                    "Slice start bit is past the end of the Tibs.",
                ));
            }
            // For negative step, the end_bit is inclusive, but the start_bit is exclusive.
            debug_assert!(step < 0);
            let adjusted_end_bit = (end_bit + 1) as usize;
            Ok(Tibs::new(
                self.data[adjusted_end_bit..=start_bit as usize]
                    .iter()
                    .rev()
                    .step_by(-step as usize)
                    .collect(),
            ))
        }
    }
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
        promote_to_tibs(auto)
    }

    /// Return string representations for printing.
    pub fn __str__(&self) -> String {
        if self.is_empty() {
            return "".to_string();
        }
        const MAX_BITS_TO_PRINT: usize = 10000;
        debug_assert!(MAX_BITS_TO_PRINT % 4 == 0);
        if self.len() <= MAX_BITS_TO_PRINT {
            match self.to_hexadecimal() {
                Ok(hex) => format!("0x{}", hex),
                Err(_) => format!("0b{}", self.to_bin()),
            }
        } else {
            format!(
                "0x{}... # length={}",
                self.slice(0, MAX_BITS_TO_PRINT).to_hexadecimal().unwrap(),
                self.len()
            )
        }
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
            return Err(PyValueError::new_err(
                format!("Cannot create chunk generator - chunk_size of {chunk_size} given, but it must be > 0."),
            ));
        }
        let max_chunks = match count {
            Some(c) => {
                if c < 0 {
                    return Err(PyValueError::new_err(
                        format!("Cannot create chunk generator - count of {c} given, but it must be > 0 if present.")
                    ));
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
        if let Ok(b) = other.extract::<PyRef<Tibs>>() {
            return self.data == b.data;
        }
        if let Ok(b) = other.extract::<PyRef<Mutibs>>() {
            return self.data == b.inner.data;
        }
        let maybe = tibs_from_any(other);
        match maybe {
            Ok(b) => self.data == b.data,
            Err(_) => false,
        }
    }

    #[pyo3(name = "__hash__")]
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
        let iter_obj = FindAllIterator {
            haystack: slf.into(),
            needle: Py::new(py, b)?,
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
        Ok(BitCollection::from_zeros(length as usize))
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
        Ok(BitCollection::from_ones(length as usize))
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
        Ok(str_to_mutibs(s)?.as_tibs())
    }

    /// Create a new instance from an unsigned integer.
    ///
    /// :param u: An unsigned integer.
    /// :param length: The bit length to create.
    ///
    /// Raises ValueError if the integer doesn't fit in the length given.
    ///
    #[classmethod]
    #[pyo3(signature = (u, /, length), text_signature = "(cls, u, /, length)")]
    pub fn from_u(
        _cls: &Bound<'_, PyType>,
        u: &Bound<'_, PyInt>,
        length: i64,
        py: Python,
    ) -> PyResult<Self> {
        if length <= 0 {
            return Err(PyValueError::new_err(
                "The length for an unsigned integer must be positive.",
            ));
        }
        let length = length as usize;
        if length <= 128 {
            let value = u.extract::<u128>().map_err(PyValueError::new_err)?;
            Ok(BitCollection::from_u128(value, length).map_err(PyValueError::new_err)?)
        } else {
            // For integers longer than 128 bits, use Python's to_bytes method.
            let num_bytes = (length + 7) / 8;
            let kwargs = [("signed", false)].into_py_dict(py)?;
            let bytes_obj = u
                .call_method("to_bytes", (num_bytes, "big"), Some(&kwargs))?
                .extract::<Vec<u8>>()?;

            let mut bv = <Tibs as BitCollection>::from_bytes(bytes_obj).data;
            if bv.len() > length {
                // Trim excess bits from the most significant end.
                bv.drain(..(bv.len() - length));
            }
            Ok(Tibs::new(bv))
        }
    }

    pub fn to_u(&self, py: Python) -> PyResult<Py<PyInt>> {
        if self.len() <= 128 {
            Ok(BitCollection::to_u128(self)
                .map_err(PyValueError::new_err)?
                .into_pyobject(py)?
                .unbind())
        } else {
            let bytes = self.to_int_byte_data(false);
            let kwargs = [("signed", false)].into_py_dict(py)?;
            let int_type = py.get_type::<PyInt>();
            let result = int_type.call_method("from_bytes", (&bytes, "big"), Some(&kwargs))?;
            Ok(result.cast::<PyInt>()?.clone().unbind())
        }
    }

    #[classmethod]
    #[pyo3(signature = (i, /, length), text_signature = "(cls, i, /, length)")]
    pub fn from_i(
        _cls: &Bound<'_, PyType>,
        i: &Bound<'_, PyInt>,
        length: i64,
        py: Python,
    ) -> PyResult<Self> {
        if length <= 0 {
            return Err(PyValueError::new_err(
                "The length for a signed integer must be positive.",
            ));
        }
        let length = length as usize;
        if length <= 128 {
            let value = i.extract::<i128>().map_err(PyValueError::new_err)?;
            Ok(BitCollection::from_i128(value, length).map_err(PyValueError::new_err)?)
        } else {
            // For integers longer than 128 bits, use Python's to_bytes method.
            let num_bytes = (length + 7) / 8;
            let kwargs = [("signed", true)].into_py_dict(py)?;
            let bytes_obj = i
                .call_method("to_bytes", (num_bytes, "big"), Some(&kwargs))?
                .extract::<Vec<u8>>()?;

            let mut bv = <Tibs as BitCollection>::from_bytes(bytes_obj).data;
            if bv.len() > length {
                // Trim excess bits from the most significant end.
                bv.drain(..(bv.len() - length));
            }
            Ok(Tibs::new(bv))
        }
    }

    pub fn to_i(&self, py: Python) -> PyResult<Py<PyInt>> {
        if self.len() <= 128 {
            Ok(BitCollection::to_i128(self)
                .map_err(PyValueError::new_err)?
                .into_pyobject(py)?
                .unbind())
        } else {
            let bytes = self.to_int_byte_data(false);
            let kwargs = [("signed", true)].into_py_dict(py)?;
            let int_type = py.get_type::<PyInt>();
            let result = int_type.call_method("from_bytes", (&bytes, "big"), Some(&kwargs))?;
            Ok(result.cast::<PyInt>()?.clone().unbind())
        }
    }

    #[classmethod]
    #[pyo3(signature = (f, /, length), text_signature = "(cls, f, /, length)")]
    pub fn from_f(_cls: &Bound<'_, PyType>, f: &Bound<'_, PyFloat>, length: i64) -> PyResult<Self> {
        let value = f.extract::<f64>().map_err(PyValueError::new_err)?;
        Ok(BitCollection::from_f64(value, length).map_err(PyValueError::new_err)?)
    }

    pub fn to_f(&self) -> PyResult<f64> {
        Ok(BitCollection::to_f64(self).map_err(PyValueError::new_err)?)
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
        BitCollection::from_binary(s).map_err(PyValueError::new_err)
    }

    pub fn to_bin(&self) -> String {
        BitCollection::to_binary(self)
    }

    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_oct(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        BitCollection::from_octal(s).map_err(PyValueError::new_err)
    }

    pub fn to_oct(&self) -> PyResult<String> {
        BitCollection::to_octal(self).map_err(PyValueError::new_err)
    }

    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_hex(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        BitCollection::from_hexadecimal(s).map_err(PyValueError::new_err)
    }

    pub fn to_hex(&self) -> PyResult<String> {
        BitCollection::to_hexadecimal(self).map_err(PyValueError::new_err)
    }

    /// Create a new instance from a bytes object.
    ///
    /// :param data: The bytes, bytearray or memoryview object to convert to a :class:`Tibs`.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_bytes(b"some_bytes_maybe_from_a_file")
    ///
    #[classmethod]
    #[inline]
    #[pyo3(signature = (data, /), text_signature = "(cls, data, /)")]
    pub fn from_bytes(_cls: &Bound<'_, PyType>, data: Vec<u8>) -> Self {
        BitCollection::from_bytes(data)
    }

    #[staticmethod]
    pub fn _from_bytes_with_offset(data: Vec<u8>, offset: usize) -> Self {
        debug_assert!(offset < 8);
        let mut bv: BV = <Tibs as BitCollection>::from_bytes(data).data;
        bv.drain(..offset);
        Tibs::new(bv)
    }

    /// Create a new instance from an iterable by converting each element to a bool.
    ///
    /// :param i: The iterable to convert to a :class:`Tibs`.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_bools([False, 0, 1, "Steven"])  # binary 0011
    ///
    #[classmethod]
    #[pyo3(signature = (values, /), text_signature = "(cls, values, /)")]
    pub fn from_bools(
        _cls: &Bound<'_, PyType>,
        values: Vec<Py<PyAny>>,
        py: Python,
    ) -> PyResult<Self> {
        let mut bv = BV::with_capacity(values.len());

        for value in values {
            let b = value.is_truthy(py)?;
            bv.push(b);
        }
        Ok(Tibs::new(bv))
    }

    /// Create a new instance with all bits pseudo-randomly set.
    ///
    /// :param length: The number of bits to set. Must be positive.
    /// :param seed: A bytes or bytearray to use as an optional seed.
    /// :return: A newly constructed ``Tibs`` with random data.
    ///
    /// Note that this uses a pseudo-random number generator and so
    /// might not suitable for cryptographic or other more serious purposes.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_random(1000000)  # A million random bits
    ///     b = Tibs.from_random(100, b'a_seed')
    ///
    #[classmethod]
    #[pyo3(signature = (length, /, seed=None), text_signature="(cls, length, /, seed=None)")]
    pub fn from_random(
        _cls: &Bound<'_, PyType>,
        length: i64,
        seed: Option<Vec<u8>>,
    ) -> PyResult<Self> {
        let bv = crate::helpers::bv_from_random(length, &seed)?;
        Ok(Tibs::new(bv))
    }

    /// Create a new instance by concatenating a sequence of Tibs objects.
    ///
    /// This method concatenates a sequence of Tibs objects into a single Tibs object.
    ///
    /// :param sequence: A sequence to concatenate. Items can either be a Tibs object, or a string or bytes-like object that could create one via the :meth:`from_string` or :meth:`from_bytes` methods.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_joined([f'u6={x}' for x in range(64)])
    ///     b = Tibs.from_joined(['0x01', 'i4 = -1', b'some_bytes'])
    ///
    #[classmethod]
    #[pyo3(signature = (sequence, /), text_signature = "(cls, sequence, /)")]
    pub fn from_joined(_cls: &Bound<'_, PyType>, sequence: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Convert each item to Tibs, store, and sum total length for a single allocation.
        let iter = sequence.try_iter()?;
        let mut parts: Vec<Tibs> = Vec::new();
        let mut total_len: usize = 0;
        for item in iter {
            let obj = item?;
            let bits = tibs_from_any(&obj)?;
            total_len += bits.len();
            parts.push(bits);
        }

        // Concatenate.
        let mut bv = BV::with_capacity(total_len);
        for bits in &parts {
            bv.extend_from_bitslice(&bits.data);
        }
        Ok(Tibs::new(bv))
    }

    pub fn to_bytes(&self) -> PyResult<Vec<u8>> {
        BitCollection::to_byte_data(self).map_err(|e| PyValueError::new_err(e))
    }

    pub fn _and(&self, other: &Tibs) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_and(self, other))
    }

    pub fn _or(&self, other: &Tibs) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_or(self, other))
    }

    pub fn _xor(&self, other: &Tibs) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_xor(self, other))
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

        Ok(find_bitvec(self, &b, start, end, byte_aligned))
    }

    pub fn __contains__(&self, b: &Bound<'_, PyAny>) -> bool {
        match self.find(b, None, None, false) {
            Ok(Some(_)) => true,
            _ => false,
        }
    }

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
            if &self.data[pos..pos + b.len()] == &b.data {
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
        let n = prefix.len();
        if n <= self.len() {
            Ok(&prefix.data == &self.data[..n])
        } else {
            Ok(false)
        }
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
        let n = suffix.len();
        if n <= self.len() {
            Ok(&suffix.data == &self.data[self.len() - n..])
        } else {
            Ok(false)
        }
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
        let len = self.len();

        let (mut ones, raw) = (0usize, self.data.as_raw_slice());
        if let Ok(words) = bytemuck::try_cast_slice::<u8, usize>(raw) {
            // Considerable speed increase by casting data to usize if possible.
            for word in words {
                ones += word.count_ones() as usize;
            }
            let used_bits = words.len() * usize::BITS as usize;
            if used_bits > len {
                let extra = used_bits - len;
                if let Some(last) = words.last() {
                    ones -= (last & (!0usize >> extra)).count_ones() as usize;
                }
            }
        } else {
            // Fallback to library method
            ones = self.data.count_ones();
        }

        Ok(if count_ones { ones } else { len - ones })
    }

    /// Return a slice of the current Tibs.
    pub fn _getslice(&self, start_bit: usize, length: usize) -> PyResult<Self> {
        if length == 0 {
            return Ok(BitCollection::empty());
        }
        if start_bit + length > self.len() {
            return Err(PyValueError::new_err(
                "End bit of the slice goes past the end of the Tibs.",
            ));
        }
        Ok(self.slice(start_bit, length))
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
        self.data.all()
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
        self.data.any()
    }

    /// Create and return a mutable copy of the Tibs as a Mutibs instance.
    pub fn to_mutibs(&self) -> Mutibs {
        Mutibs {
            inner: Tibs::new(self.data.clone()),
        }
    }

    /// Returns the bool value at a given bit index.
    #[inline]
    pub fn _getindex(&self, bit_index: i64) -> PyResult<bool> {
        let index = validate_index(bit_index, self.len())?;
        Ok(self.data[index])
    }

    #[inline]
    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = key.py();
        // Handle integer indexing
        if let Ok(index) = key.extract::<i64>() {
            let value: bool = self._getindex(index)?;
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
                self._getslice(
                    start as usize,
                    if stop > start {
                        (stop - start) as usize
                    } else {
                        0
                    },
                )?
            } else {
                self._getslice_with_step(start, stop, step)?
            };
            let py_obj = Py::new(py, result)?.into_pyobject(py)?;
            return Ok(py_obj.into());
        }

        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    #[inline]
    pub(crate) fn _validate_shift(&self, n: i64) -> PyResult<usize> {
        if self.is_empty() {
            return Err(PyValueError::new_err("Cannot shift an empty Tibs."));
        }
        if n < 0 {
            return Err(PyValueError::new_err("Cannot shift by a negative amount."));
        }
        Ok(n as usize)
    }

    /// Return new Tibs shifted by n to the left.
    ///
    /// n -- the number of bits to shift. Must be >= 0.
    ///
    pub fn __lshift__(&self, n: i64) -> PyResult<Self> {
        let shift = self._validate_shift(n)?;
        if shift == 0 {
            return Ok(self.clone());
        }
        let len = self.len();
        if shift >= len {
            return Ok(BitCollection::from_zeros(len));
        }
        let mut result_data = BV::with_capacity(len);
        result_data.extend_from_bitslice(&self.data[shift..]);
        result_data.resize(len, false);
        Ok(Self::new(result_data))
    }

    /// Return new Tibs shifted by n to the right.
    ///
    /// n -- the number of bits to shift. Must be >= 0.
    ///
    pub fn __rshift__(&self, n: i64) -> PyResult<Self> {
        let shift = self._validate_shift(n)?;
        if shift == 0 {
            return Ok(self.clone());
        }
        let len = self.len();
        if shift >= len {
            return Ok(BitCollection::from_zeros(len));
        }
        let mut result_data = BV::repeat(false, shift);
        result_data.extend_from_bitslice(&self.data[..len - shift]);
        Ok(Self::new(result_data))
    }

    /// Concatenates two Tibs and return a newly constructed Tibs.
    pub fn __add__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bs = tibs_from_any(bs)?;
        let mut data = BV::with_capacity(self.len() + bs.len());
        data.extend_from_bitslice(&self.data);
        data.extend_from_bitslice(&bs.data);
        Ok(Tibs::new(data))
    }

    /// Concatenates two Tibs and return a newly constructed Tibs.
    pub fn __radd__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut bs = mutibs_from_any(bs)?;
        bs.inner.data.extend_from_bitslice(&self.data);
        Ok(Tibs::new(bs.inner.data))
    }

    /// Bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __and__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        // TODO: Return early `if bs is self`.
        let other = tibs_from_any(bs)?;
        self._and(&other)
    }

    /// Bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __or__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        // TODO: Return early `if bs is self`.
        let other = tibs_from_any(bs)?;
        self._or(&other)
    }

    /// Bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __xor__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = tibs_from_any(bs)?;
        self._xor(&other)
    }

    /// Reverse bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __rand__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = tibs_from_any(bs)?;
        other._and(&self)
    }

    /// Reverse bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __ror__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = tibs_from_any(bs)?;
        other._or(&self)
    }

    /// Reverse bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __rxor__(&self, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = tibs_from_any(bs)?;
        other._xor(&self)
    }

    /// Return the instance with every bit inverted.
    ///
    /// Raises ValueError if the Tibs is empty.
    ///
    pub fn __invert__(&self) -> PyResult<Self> {
        if self.data.is_empty() {
            return Err(PyValueError::new_err("Cannot invert empty Tibs."));
        }
        Ok(Tibs::new(self.data.clone().not()))
    }

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
        let n = n as usize;
        let len = self.len();
        if n == 0 || len == 0 {
            return Ok(BitCollection::empty());
        }
        let mut bv = BV::with_capacity(len * n);
        bv.extend_from_bitslice(&self.data);
        // TODO: This could be done more efficiently with doubling.
        for _ in 1..n {
            bv.extend_from_bitslice(&self.data);
        }
        Ok(Tibs::new(bv))
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

    pub fn __setitem__(&self, _key: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "Tibs objects do not support item assignment. Did you mean to use the Mutibs class? Call to_mutibs() to convert to a Mutibs."
        ))
    }

    pub fn __delitem__(&self, _key: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "Tibs objects do not support item deletion. Did you mean to use the Mutibs class? Call to_mutibs() to convert to a Mutibs."
        ))
    }
}
