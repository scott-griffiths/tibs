use crate::mutable::mutable_bits_from_any;
use crate::core::validate_logical_op_lengths;
use crate::core::{str_to_bits, BitCollection, DTYPE_PARSER};
use crate::helpers::{find_bitvec, validate_index, BV};
use crate::iterator::{BoolIterator, ChunksIterator, FindAllIterator};
use crate::mutable::Mutibs;
use bitvec::prelude::*;
use bytemuck;
use pyo3::conversion::IntoPyObject;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyMemoryView, PySlice, PyType, PyInt, PyTuple, PyList};
use pyo3::{pyclass, pymethods, PyRef, PyResult};
use std::ops::Not;
use rand::{RngCore, SeedableRng};
use rand::rngs::StdRng;
use once_cell::sync::OnceCell;

// ---- Exported Python helper methods ----

#[pyfunction]
pub fn set_dtype_parser(dtype_parser: Py<PyAny>) -> PyResult<()> {
    // Store the Python object directly - no conversion needed
    let mut parser_guard = DTYPE_PARSER.lock().unwrap();
    *parser_guard = Some(dtype_parser);
    Ok(())
}

#[pyfunction]
pub fn bits_from_any(any: Py<PyAny>, py: Python) -> PyResult<Tibs> {
    let any_bound = any.bind(py);

    // Is it of type Tibs?
    if let Ok(any_bits) = any_bound.extract::<PyRef<Tibs>>() {
        return Ok(any_bits.clone());
    }

    // Is it of type MutableBits?
    if let Ok(any_mutable_bits) = any_bound.extract::<PyRef<Mutibs>>() {
        return Ok(any_mutable_bits.to_bits());
    }

    // Is it a string?
    if let Ok(any_string) = any_bound.extract::<String>() {
        return str_to_bits(any_string);
    }

    // Is it a bytes, bytearray or memoryview?
    if any_bound.is_instance_of::<PyBytes>()
        || any_bound.is_instance_of::<PyByteArray>()
        || any_bound.is_instance_of::<PyMemoryView>()
    {
        if let Ok(any_bytes) = any_bound.extract::<Vec<u8>>() {
            return Ok(<Tibs as BitCollection>::from_bytes(any_bytes));
        }
    }

    // Is it an iterable that we can convert each element to a bool?
    if let Ok(iter) = any_bound.try_iter() {
        let mut bv = BV::new();
        for item in iter {
            bv.push(item?.is_truthy()?);
        }
        return Ok(Tibs::new(bv));
    }

    let type_name = match any_bound.get_type().name() {
        Ok(name) => name.to_string(),
        Err(_) => "<unknown>".to_string(),
    };
    Err(PyTypeError::new_err(format!(
        "Cannot convert object of type {type_name} to a Tibs object."
    )))
}

///     An immutable container of binary data.
///
///     To construct, use a builder 'from' method:
///
///     * ``Tibs.from_bytes(b)`` - Create directly from a ``bytes`` object.
///     * ``Tibs.from_string(s)`` - Use a formatted string.
///     * ``Tibs.from_bools(i)`` - Convert each element in ``i`` to a bool.
///     * ``Tibs.from_zeros(length)`` - Initialise with ``length`` '0' bits.
///     * ``Tibs.from_ones(length)`` - Initialise with ``length`` '1' bits.
///     * ``Tibs.from_random(length, [seed])`` - Initialise with ``length`` pseudo-randomly set bits.
///     * ``Tibs.from_dtype(dtype, value)`` - Combine a data type with a value.
///     * ``Tibs.from_joined(iterable)`` - Concatenate an iterable of objects.
///
///     Using the constructor ``Tibs(s)`` is an alias for ``Tibs.from_string(s)``.
///
#[derive(Clone)]
#[pyclass(module = "tibs")]
pub struct Tibs {
    pub(crate) data: BV,
    pub(crate) bin_cache: OnceCell<String>,
    pub(crate) oct_cache: OnceCell<String>,
    pub(crate) hex_cache: OnceCell<String>,
}

impl Tibs {
    pub(crate) fn _getslice_with_step(&self, start_bit: i64, end_bit: i64, step: i64) -> PyResult<Self> {
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
    #[pyo3(signature = (s = None))]
    pub fn py_new(s: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(s) = s else {
            return Ok(BitCollection::empty());
        };
        if let Ok(string_s) = s.extract::<String>() {
            return str_to_bits(string_s);
        }

        // If it's not a string, build a more helpful error message.
        let type_name = s.get_type().name()?;
        let mut err = format!(
            "Expected a str for Tibs constructor, but received a {}. ",
            type_name
        );

        if s.is_instance_of::<Mutibs>() {
            err.push_str(
                "You can use the 'to_bits()' method on the `MutableBits` instance instead.",
            );
        } else if s.is_instance_of::<PyBytes>()
            || s.is_instance_of::<PyByteArray>()
            || s.is_instance_of::<PyMemoryView>()
        {
            err.push_str("You can use 'Tibs.from_bytes()' instead.");
        } else if s.is_instance_of::<PyInt>() {
            err.push_str("Perhaps you want to use 'Tibs.from_zeros()', 'Tibs.from_ones()' or 'Tibs.from_random()'?");
        } else if s.is_instance_of::<PyTuple>()
            || s.is_instance_of::<PyList>()
        {
            err.push_str(
                "Perhaps you want to use 'Tibs.from_joined()' or 'Tibs.from_bools()' instead?",
            );
        } else {
            err.push_str(
                "To create from other types use from_bytes(), from_bools(), from_joined(), \
                 from_ones(), from_zeros(), from_dtype() or from_random().",
            );
        }
        Err(PyTypeError::new_err(err))
    }

    /// Return string representations for printing.
    pub fn __str__(&self) -> String {
        if self.is_empty() {
            return "".to_string();
        }
        match self.to_hex() {
            Ok(hex) => format!("0x{}", hex),
            Err(_) => format!("0b{}", self.to_bin()),
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

    // A bit of a hack so that the Python can use _chunks on Tibs and MutableBits. Can remove later.
    #[pyo3(signature = (chunk_size, count = None))]
    pub fn _chunks(
        slf: PyRef<'_, Self>,
        chunk_size: i64,
        count: Option<i64>,
    ) -> PyResult<Py<ChunksIterator>> {
        Tibs::chunks(slf, chunk_size, count)
    }

    /// Return True if two Tibs have the same binary representation.
    ///
    /// The right hand side will be promoted to a Tibs if needed and possible.
    ///
    /// >>> Tibs('0b1110') == '0xe'
    /// True
    ///
    pub fn __eq__(&self, other: Py<PyAny>, py: Python) -> bool {
        let obj = other.bind(py);
        if let Ok(b) = obj.extract::<PyRef<Tibs>>() {
            return self.data == b.data;
        }
        if let Ok(b) = obj.extract::<PyRef<Mutibs>>() {
            return self.data == b.inner.data;
        }
        match bits_from_any(other, py) {
            Ok(b) => self.data == b.data,
            Err(_) => false,
        }
    }

    #[staticmethod]
    pub fn _from_u64(value: u64, length: usize) -> Self {
        BitCollection::from_u64(value, length)
    }

    #[staticmethod]
    pub fn _from_i64(value: i64, length: usize) -> Self {
        BitCollection::from_i64(value, length)
    }

    pub fn _to_u64(&self, start: usize, length: usize) -> u64 {
        self.data[start..start + length].load_be::<u64>()
    }

    pub fn _to_i64(&self, start: usize, length: usize) -> i64 {
        self.data[start..start + length].load_be::<i64>()
    }

    #[pyo3(signature = (needle_obj, start=None, end=None, byte_aligned=false))]
    pub fn _findall(
        slf: PyRef<'_, Self>,
        needle_obj: Py<Tibs>,
        start: Option<usize>,
        end: Option<usize>,
        byte_aligned: bool,
    ) -> PyResult<Py<FindAllIterator>> {
        let py = slf.py();
        let haystack_obj: Py<Tibs> = slf.into(); // Get a Py<Tibs> for the haystack (self)

        let step = if byte_aligned { 8 } else { 1 };
        let start = start.unwrap_or(0);
        let iter_obj = FindAllIterator {
            haystack: haystack_obj,
            needle: needle_obj,
            start,
            end,
            byte_aligned,
            step,
            current_pos: start,
        };
        Py::new(py, iter_obj)
    }

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
    /// This method initializes a new instance of :class:`Tibs` using a formatted string.
    ///
    /// :param s: The formatted string to convert.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_string("0xff01")
    ///     b = Tibs.from_string("0b1")
    ///     c = Tibs.from_string("u12 = 31, f16=-0.25")
    ///
    /// The `__init__` method for `Tibs` redirects to the `from_string` method and is sometimes more convenient:
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs("0xff01")  # Tibs(s) is equivalent to Tibs.from_string(s)
    ///
    #[classmethod]
    pub fn from_string(_cls: &Bound<'_, PyType>, s: String) -> PyResult<Self> {
        str_to_bits(s)
    }

    /// Create a new instance from a bytes object.
    ///
    /// :param b: The bytes object to convert to a :class:`Tibs`.
    ///
    /// .. code-block:: python
    ///
    /// a = Tibs.from_bytes(b"some_bytes_maybe_from_a_file")
    ///
    #[classmethod]
    #[inline]
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
    /// :param seed: An optional seed as a bytes or bytearray.
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
    #[pyo3(signature = (length, seed=None))]
    pub fn from_random(_cls: &Bound<'_, PyType>, length: i64, seed: Option<Vec<u8>>) -> PyResult<Self> {
        if length < 0 {
            return Err(PyValueError::new_err(format!(
                "Negative bit length given: {}.",
                length
            )));
        }
        let length = length as usize;
        if length == 0 {
            return Ok(BitCollection::empty());
        }
        let seed_arr = crate::helpers::process_seed(seed);
        let mut rng = StdRng::from_seed(seed_arr);

        let num_bytes = (length + 7) / 8;
        let mut data = vec![0u8; num_bytes];
        rng.fill_bytes(&mut data);
        let mut bv = BV::from_vec(data);
        bv.truncate(length);
        Ok(Tibs::new(bv))
    }

    #[staticmethod]
    pub fn _from_bin(binary_string: &str) -> PyResult<Self> {
        BitCollection::from_bin(binary_string).map_err(PyValueError::new_err)
    }

    #[staticmethod]
    pub fn _from_hex(hex: &str) -> PyResult<Self> {
        BitCollection::from_hex(hex).map_err(PyValueError::new_err)
    }

    #[staticmethod]
    pub fn _from_oct(oct: &str) -> PyResult<Self> {
        BitCollection::from_oct(oct).map_err(PyValueError::new_err)
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
    pub fn from_joined(
        _cls: &Bound<'_, PyType>,
        sequence: &Bound<'_, PyAny>,
        py: Python,
    ) -> PyResult<Self> {
        // Convert each item to Tibs, store, and sum total length for a single allocation.
        let iter = sequence.try_iter()?;
        let mut parts: Vec<Tibs> = Vec::new();
        let mut total_len: usize = 0;
        for item in iter {
            let obj = item?;
            let bits = bits_from_any(obj.into(), py)?;
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

    /// Return bytes that can easily be converted to an int in Python
    pub fn _to_int_byte_data(&self, signed: bool) -> Vec<u8> {
        if self.is_empty() {
            return Vec::new();
        }

        // TODO: Is this next line right?
        let needed_bits = (self.len() + 7) & !7;
        let mut bv = BV::with_capacity(needed_bits);

        let sign_bit = signed && self.data[0];
        let padding = needed_bits - self.len();

        for _ in 0..padding {
            bv.push(sign_bit);
        }
        bv.extend_from_bitslice(&self.data);

        bv.into_vec()
    }

    /// Return the Tibs as bytes, padding with zero bits if needed.
    ///
    /// Up to seven zero bits will be added at the end to byte align.
    ///
    /// :return: The Tibs as bytes.
    ///
    pub fn to_bytes(&self) -> Vec<u8> {
        if self.data.is_empty() {
            return Vec::new();
        }
        let len_bits = self.len();
        match self.data.as_bitslice().domain() {
            // Fast path: element-aligned and length is a multiple of 8
            bitvec::domain::Domain::Region { head: None, body, tail: None } => {
                // Already byte-aligned; copy the bytes directly.
                body.to_vec()
            }

            // Aligned but not a multiple of 8: clone and pad to the next byte
            bitvec::domain::Domain::Region { head: None, .. } => {
                let mut bv = self.data.clone();
                let new_len = (len_bits + 7) & !7;
                bv.resize(new_len, false);
                bv.into_vec()
            }

            // Misaligned: repack by extending from the bitslice, then pad
            _ => {
                let mut bv = BV::with_capacity(len_bits);
                bv.extend_from_bitslice(&self.data);
                let new_len = (len_bits + 7) & !7;
                bv.resize(new_len, false);
                bv.into_vec()
            }
        }
    }

    pub fn _slice_to_bytes(&self, start: usize, length: usize) -> PyResult<Vec<u8>> {
        if length % 8 != 0 {
            return Err(PyValueError::new_err(format!(
                "Cannot interpret as bytes - length of {} is not a multiple of 8 bits. Use the to_bytes() method if you want to add zero padding bits.",
                length
            )));
        }
        if length == 0 {
            return Ok(Vec::new());
        }
        let mut bv = BV::with_capacity(length);
        bv.extend_from_bitslice(&self.data[start..start + length]);
        Ok(bv.into_vec())
    }

    pub fn _slice_to_bin(&self, start: usize, length: usize) -> String {
        self.slice(start, length).to_bin()
    }

    pub fn _slice_to_oct(&self, start: usize, length: usize) -> PyResult<String> {
        self.slice(start, length)
            .to_oct()
            .map_err(PyValueError::new_err)
    }

    pub fn _slice_to_hex(&self, start: usize, length: usize) -> PyResult<String> {
        self.slice(start, length)
            .to_hex()
            .map_err(PyValueError::new_err)
    }

    pub fn _and(&self, other: &Tibs) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), other.len())?;
        let result = self.data.clone() & &other.data;
        Ok(Tibs::new(result))
    }

    pub fn _or(&self, other: &Tibs) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), other.len())?;
        let result = self.data.clone() | &other.data;
        Ok(Tibs::new(result))
    }

    pub fn _xor(&self, other: &Tibs) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), other.len())?;
        let result = self.data.clone() ^ &other.data;
        Ok(Tibs::new(result))
    }

    pub fn _find(&self, b: &Tibs, start: usize, end: usize, bytealigned: bool) -> Option<usize> {
        debug_assert!(end >= start);
        debug_assert!(end <= self.len());
        find_bitvec(self, b, start, end, bytealigned)
    }

    pub fn _rfind(&self, b: &Tibs, start: usize, end: usize, bytealigned: bool) -> Option<usize> {
        debug_assert!(end >= start);
        debug_assert!(end <= self.len());
        if b.len() + start > end {
            return None;
        }
        let step = if bytealigned { 8 } else { 1 };
        let mut pos = end - b.len();
        if bytealigned {
            pos = pos / 8 * 8;
        }
        while pos >= start {
            if &self.data[pos..pos + b.len()] == &b.data {
                return Some(pos);
            }
            if pos < step {
                break;
            }
            pos -= step;
        }
        None
    }

    /// Return whether the current Tibs starts with prefix.
    ///
    /// :param prefix: The Tibs to search for.
    /// :return: True if the Tibs starts with the prefix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b101100').starts_with('0b101')
    ///     True
    ///     >>> Tibs('0b101100').starts_with('0b100')
    ///     False
    ///
    pub fn starts_with(&self, prefix: Py<PyAny>, py: Python) -> PyResult<bool> {
        let prefix = bits_from_any(prefix, py)?;
        let n = prefix.len();
        if n <= self.len() {
            Ok(&prefix.data == &self.data[..n])
        } else {
            Ok(false)
        }
    }

    /// Return whether the current Tibs ends with suffix.
    ///
    /// :param suffix: The Tibs to search for.
    /// :return: True if the Tibs ends with the suffix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b101100').ends_with('0b10-')
    ///     True
    ///     >>> Tibs('0b101100').ends_with('0b101')
    ///     False
    ///
    pub fn ends_with(&self, suffix: Py<PyAny>, py: Python) -> PyResult<bool> {
        let suffix = bits_from_any(suffix, py)?;
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
    pub fn count(&self, value: Py<PyAny>, py: Python) -> PyResult<usize> {
        let count_ones = value.is_truthy(py)?;
        let len = self.len();
        let ones = py.detach(|| {
            // Note that using hamming::weight is about twice as fast as:
            // self.data.count_ones()
            // which is the way that bitvec suggests.
            let bytes: &[u8] = bytemuck::cast_slice(self.data.as_raw_slice());
            hamming::weight(bytes) as usize
        });
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

    /// Create and return a mutable copy of the Tibs as a MutableBits instance.
    pub fn to_mutable_bits(&self) -> Mutibs {
        Mutibs {
            inner: Tibs::new(self.data.clone()),
        }
    }

    /// Move the bitvec out, leaving this Tibs empty.
    /// Only to be done as part of MutableBits construction
    /// when the transient Tibs isn't visible externally.
    /// Definitely not part of public interface!
    pub fn _as_mutable_bits(mut slf: PyRefMut<Self>) -> Mutibs {
        let data = std::mem::take(&mut slf.data);
        Mutibs {
            inner: Tibs::new(data),
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
        if let Ok(slice) = key.downcast::<PySlice>() {
            let indices = slice.indices(self.len() as isize)?;
            let start: i64 = indices.start.try_into()?;
            let stop: i64 = indices.stop.try_into()?;
            let step: i64 = indices.step.try_into()?;

            let result = if step == 1 {
                self._getslice(start as usize, if stop > start { (stop - start) as usize } else { 0 })?
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
    pub fn __add__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let bs = bits_from_any(bs, py)?;
        let mut data = BV::with_capacity(self.len() + bs.len());
        data.extend_from_bitslice(&self.data);
        data.extend_from_bitslice(&bs.data);
        Ok(Tibs::new(data))
    }

    /// Concatenates two Tibs and return a newly constructed Tibs.
    pub fn __radd__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let mut bs = mutable_bits_from_any(bs, py)?;
        bs.inner.data.extend_from_bitslice(&self.data);
        Ok(Tibs::new(bs.inner.data))
    }

    /// Bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __and__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        // TODO: Return early `if bs is self`.
        let other = bits_from_any(bs, py)?;
        self._and(&other)
    }

    /// Bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __or__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        // TODO: Return early `if bs is self`.
        let other = bits_from_any(bs, py)?;
        self._or(&other)
    }

    /// Bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __xor__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let other = bits_from_any(bs, py)?;
        self._xor(&other)
    }

    /// Reverse bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __rand__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let other = bits_from_any(bs, py)?;
        other._and(&self)
    }

    /// Reverse bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __ror__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let other = bits_from_any(bs, py)?;
        other._or(&self)
    }

    /// Reverse bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    ///
    pub fn __rxor__(&self, bs: Py<PyAny>, py: Python) -> PyResult<Self> {
        let other = bits_from_any(bs, py)?;
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

    pub fn __bytes__(&self) -> Vec<u8> {
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
            return Err(PyValueError::new_err("Cannot multiply by a negative integer."))
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

    pub fn __setitem__(&self, _key: Py<PyAny>, _value: Py<PyAny>) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "Tibs objects do not support item assignment. Did you mean to use the MutableBits class? Call to_mutable_bits() to convert to a MutableBits."
        ))
    }

    pub fn __delitem__(&self, _key: Py<PyAny>) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "Tibs objects do not support item deletion. Did you mean to use the MutableBits class? Call to_mutable_bits() to convert to a MutableBits."
        ))
    }
}
