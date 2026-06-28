use crate::core::BitCollection;
use crate::enums::{DtypeKind, Endianness};
use crate::helpers::validate_slice;
use crate::iterator::ValuesIterator;
use crate::tibs_::{Tibs, bv_from_value, bv_from_values_iter, py_from_value, py_values_from_range};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, Py, PyAny, PyRef, PyResult, Python, pyclass, pymethods};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

///     A data type which determines how a value is encoded as a fixed-width bit sequence.
///
///     ``Dtype`` is used by :meth:`Tibs.from_value`, :meth:`Tibs.to_value`,
///     :meth:`Tibs.from_values` and related methods to describe the kind, length and
///     optional byte order for encoded values.
///
///     .. code-block:: pycon
///
///         >>> Dtype("u16_le")
///         Dtype('u16_le')
///         >>> Tibs.from_value("u8", 15)
///         Tibs('0x0f')
///
#[pyclass(module = "tibs", frozen, skip_from_py_object)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Dtype {
    pub(crate) kind: DtypeKind,
    pub(crate) length: usize,
    pub(crate) byte_order: Endianness,
}

impl Dtype {
    fn from_parts(kind: DtypeKind, length: i64, byte_order: Endianness) -> PyResult<Self> {
        if length <= 0 {
            return Err(PyValueError::new_err(format!(
                "Dtype length must be greater than zero, but received {}.",
                length
            )));
        }
        let length = length as usize;
        if kind == DtypeKind::Bool && length != 1 {
            return Err(PyValueError::new_err(format!(
                "A Dtype of type {} must have length 1.",
                kind.repr_name()
            )));
        }
        if byte_order != Endianness::Unspecified {
            match kind {
                DtypeKind::Uint | DtypeKind::Int | DtypeKind::Float => {
                    if !length.is_multiple_of(8) {
                        return Err(PyValueError::new_err(format!(
                            "If a Dtype byte_order is given, the length must be a multiple of 8 (length = {}).",
                            length
                        )));
                    }
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "A byte order cannot be specified for a Dtype of type {}.",
                        kind.repr_name()
                    )));
                }
            }
        }
        Ok(Dtype {
            kind,
            length,
            byte_order,
        })
    }

    fn parse_spec(spec: &str) -> PyResult<Self> {
        let spec = spec.trim().to_ascii_lowercase();
        let (base, byte_order) = if let Some(base) = spec.strip_suffix("_le") {
            (base, Endianness::Little)
        } else if let Some(base) = spec.strip_suffix("_be") {
            (base, Endianness::Big)
        } else {
            (spec.as_str(), Endianness::Unspecified)
        };

        if base == "bool" {
            return Self::from_parts(DtypeKind::Bool, 1, byte_order);
        }

        let (kind, length_text) = if let Some(length) = base.strip_prefix("bytes") {
            (DtypeKind::Bytes, length)
        } else if let Some(length) = base.strip_prefix("bits") {
            (DtypeKind::Bits, length)
        } else if let Some(length) = base.strip_prefix("bin") {
            (DtypeKind::Bin, length)
        } else if let Some(length) = base.strip_prefix("oct") {
            (DtypeKind::Oct, length)
        } else if let Some(length) = base.strip_prefix("hex") {
            (DtypeKind::Hex, length)
        } else if let Some(length) = base.strip_prefix('u') {
            (DtypeKind::Uint, length)
        } else if let Some(length) = base.strip_prefix('i') {
            (DtypeKind::Int, length)
        } else if let Some(length) = base.strip_prefix('f') {
            (DtypeKind::Float, length)
        } else {
            return Err(PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}'."
            )));
        };

        if length_text.is_empty() || !length_text.chars().all(|c| c.is_ascii_digit()) {
            return Err(PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}': missing or invalid bit length."
            )));
        }
        let length = length_text.parse::<i64>().map_err(|_| {
            PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}': bit length is too large."
            ))
        })?;
        Self::from_parts(kind, length, byte_order)
    }
}

pub(crate) fn extract_dtype(obj: &Bound<'_, PyAny>) -> PyResult<Dtype> {
    if let Ok(dtype) = obj.extract::<PyRef<'_, Dtype>>() {
        return Ok(dtype.clone());
    }
    if let Ok(spec) = obj.extract::<String>() {
        return Dtype::parse_spec(&spec);
    }
    Err(PyTypeError::new_err(
        "dtype must be a Dtype instance or dtype string.",
    ))
}

#[pymethods]
impl Dtype {
    /// Create a dtype from a compact string specification.
    ///
    /// :param str spec: A dtype string such as ``"u8"``, ``"i16"``, ``"f32_le"``, ``"bool"``, ``"bits32"``, ``"hex32"`` or ``"bytes64"``.
    /// :return: A new ``Dtype``.
    ///
    /// :raises ValueError: if the string cannot be parsed, if ``length`` is not greater than zero, if ``bool`` is given a length other than 1, if byte order is used with a non-numeric kind, or if byte order is used with a non-byte length.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("u8")
    ///     Dtype('u8')
    ///     >>> Dtype("f32_le")
    ///     Dtype('f32_le')
    ///
    #[new]
    #[pyo3(signature = (spec, /), text_signature = "(spec, /)")]
    pub fn py_new(spec: &str) -> PyResult<Self> {
        Self::parse_spec(spec)
    }

    /// Create a dtype from explicit parameters.
    ///
    /// :param DtypeKind kind: The kind of value to encode or decode.
    /// :param int length: The number of bits used by one value.
    /// :param Endianness byte_order: The byte order for integer and floating-point values. Defaults to ``Endianness.Unspecified``.
    /// :return: A new ``Dtype``.
    ///
    /// :raises ValueError: if ``length`` is not greater than zero, if ``DtypeKind.Bool`` is given a length other than 1, if byte order is used with a non-numeric kind, or if byte order is used with a non-byte length.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype.from_params(DtypeKind.Uint, 16, Endianness.Little)
    ///     Dtype('u16_le')
    ///
    #[classmethod]
    #[pyo3(signature = (kind, length, byte_order = Endianness::Unspecified), text_signature = "(cls, kind, length, byte_order=None)")]
    pub fn from_params(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        kind: DtypeKind,
        length: i64,
        byte_order: Option<Endianness>,
    ) -> PyResult<Self> {
        let byte_order = byte_order.unwrap_or(Endianness::Unspecified);
        Self::from_parts(kind, length, byte_order)
    }

    /// The value kind described by this dtype.
    #[getter]
    fn kind(&self) -> DtypeKind {
        self.kind
    }

    /// The number of bits used by one value.
    #[getter]
    fn length(&self) -> usize {
        self.length
    }

    /// The byte order used by integer and floating-point values.
    #[getter]
    fn byte_order(&self) -> Endianness {
        self.byte_order
    }

    /// Encode one Python value as a :class:`Tibs`.
    ///
    /// :param object value: The value to encode.
    /// :return: A new :class:`Tibs`.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("u8").pack(15)
    ///     Tibs('0x0f')
    ///
    #[pyo3(signature = (value, /), text_signature = "($self, value, /)")]
    pub fn pack(&self, value: &Bound<'_, PyAny>) -> PyResult<Tibs> {
        Ok(Tibs::from_bv(bv_from_value(self, value)?))
    }

    /// Encode and concatenate Python values as a :class:`Tibs`.
    ///
    /// :param Iterable iterable: The values to encode.
    /// :return: A new :class:`Tibs`.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("u8").pack_values([1, 2, 3])
    ///     Tibs('0x010203')
    ///
    #[pyo3(signature = (iterable, /), text_signature = "($self, iterable, /)")]
    pub fn pack_values(&self, py: Python<'_>, iterable: &Bound<'_, PyAny>) -> PyResult<Tibs> {
        Ok(Tibs::from_bv(bv_from_values_iter(py, self, iterable)?))
    }

    /// Decode one value from a bit sequence.
    ///
    /// :param Tibs bits: The bit sequence to decode.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(bits).
    /// :return: The decoded Python value.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("u8").unpack("0x0f")
    ///     15
    ///
    #[pyo3(signature = (bits, /, start = None, end = None), text_signature = "($self, bits, /, start=None, end=None)")]
    pub fn unpack(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<PyAny>> {
        let bits = bits.extract::<Tibs>()?;
        let (start, end) = validate_slice(bits.len(), start, end)?;
        let value = bits.get_slice_unchecked(start, end - start);
        py_from_value(py, self, &value)
    }

    /// Decode a list of values from a bit sequence.
    ///
    /// The selected range must be a whole number of dtype values.
    ///
    /// :param Tibs bits: The bit sequence to decode.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(bits).
    /// :return: A list of decoded Python values.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("u8").unpack_values("0x010203")
    ///     [1, 2, 3]
    ///
    #[pyo3(signature = (bits, /, start = None, end = None), text_signature = "($self, bits, /, start=None, end=None)")]
    pub fn unpack_values(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let bits = bits.extract::<Tibs>()?;
        py_values_from_range(py, &bits, self, start, end)
    }

    /// Return an iterator over values decoded from a bit sequence.
    ///
    /// The selected range must be a whole number of dtype values.
    ///
    /// :param Tibs bits: The bit sequence to decode.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(bits).
    /// :return: An iterator yielding decoded Python values.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Dtype("u8").unpack_values_iter("0x010203"))
    ///     [1, 2, 3]
    ///
    #[pyo3(signature = (bits, /, start = None, end = None), text_signature = "($self, bits, /, start=None, end=None)")]
    pub fn unpack_values_iter(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<ValuesIterator>> {
        let bits = bits.extract::<Tibs>()?;
        let (start, end) = validate_slice(bits.len(), start, end)?;
        let selected_len = end - start;
        let chunk_size = self.length;
        if !selected_len.is_multiple_of(chunk_size) {
            return Err(PyValueError::new_err(format!(
                "Cannot create values iterator - selected length of {selected_len} bits is not a multiple of dtype length {} bits.",
                self.length
            )));
        }

        Py::new(
            py,
            ValuesIterator {
                bits_object: Py::new(py, bits)?,
                dtype_kind: self.kind,
                dtype_length: self.length,
                byte_order: self.byte_order,
                chunk_size,
                current_pos: start,
                end_pos: end,
            },
        )
    }

    pub fn __repr__(&self) -> String {
        let byte_order_str = match self.byte_order {
            Endianness::Unspecified => "",
            Endianness::Little => "_le",
            Endianness::Big => "_be",
        };
        let spec = match self.kind {
            DtypeKind::Uint => format!("u{}{}", self.length, byte_order_str),
            DtypeKind::Int => format!("i{}{}", self.length, byte_order_str),
            DtypeKind::Float => format!("f{}{}", self.length, byte_order_str),
            DtypeKind::Bool => "bool".to_string(),
            DtypeKind::Bits => format!("bits{}", self.length),
            DtypeKind::Bin => format!("bin{}", self.length),
            DtypeKind::Oct => format!("oct{}", self.length),
            DtypeKind::Hex => format!("hex{}", self.length),
            DtypeKind::Bytes => format!("bytes{}", self.length),
        };
        format!("Dtype('{spec}')")
    }

    pub fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(other) = other.extract::<PyRef<'_, Dtype>>() else {
            return Ok(false);
        };
        Ok(self == &*other)
    }

    pub fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let hash = hasher.finish() as isize;
        // Python reserves -1 as the error return value from tp_hash.
        if hash == -1 { -2 } else { hash }
    }
}
