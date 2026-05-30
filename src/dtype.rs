use crate::enums::{DtypeKind, Endianness};
use pyo3::exceptions::PyValueError;
use pyo3::{PyResult, pyclass, pymethods};

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
///         >>> Tibs.from_value(Dtype("u8"), 15)
///         Tibs('0x0f')
///
#[pyclass(module = "tibs", frozen)]
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
        match byte_order {
            Endianness::Unspecified => (),
            _ => match kind {
                DtypeKind::Bin | DtypeKind::Hex | DtypeKind::Oct | DtypeKind::Bytes => {
                    return Err(PyValueError::new_err(format!(
                        "A byte order cannot be specified for a Dtype of type {}.",
                        kind.repr_name()
                    )));
                }
                _ => {
                    if !length.is_multiple_of(8) {
                        return Err(PyValueError::new_err(format!(
                            "If a Dtype byte_order is given, the length must be a multiple of 8 (length = {}).",
                            length
                        )));
                    }
                }
            },
        }
        if byte_order != Endianness::Unspecified {
            match kind {
                DtypeKind::Bin | DtypeKind::Hex | DtypeKind::Oct | DtypeKind::Bytes => {
                    return Err(PyValueError::new_err(format!(
                        "A byte order cannot be specified for a Dtype of type {}.",
                        kind.repr_name()
                    )));
                }
                _ => (),
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

        let (kind, length_text) = if let Some(length) = base.strip_prefix("bytes") {
            (DtypeKind::Bytes, length)
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

#[pymethods]
impl Dtype {
    /// Create a dtype from a compact string specification.
    ///
    /// :param str spec: A dtype string such as ``"u8"``, ``"i16"``, ``"f32_le"``, ``"hex32"`` or ``"bytes64"``.
    /// :return: A new ``Dtype``.
    ///
    /// :raises ValueError: if the string cannot be parsed, if ``length`` is not greater than zero, if byte order is used with a non-numeric kind, or if byte order is used with a non-byte length.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("u8")
    ///     Dtype('u8')
    ///     >>> Dtype("f32_le")
    ///     Dtype('f32_le')
    ///
    #[new]
    #[pyo3(signature = (spec, /), text_signature = "($self, spec, /)")]
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
    /// :raises ValueError: if ``length`` is not greater than zero, if byte order is used with a non-numeric kind, or if byte order is used with a non-byte length.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype.from_params(DtypeKind.Uint, 16, Endianness.Little)
    ///     Dtype('u16_le')
    ///
    #[classmethod]
    #[pyo3(signature = (kind, length, byte_order = Endianness::Unspecified), text_signature = "(cls, kind, length, byte_order)")]
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
            DtypeKind::Bin => format!("bin{}", self.length),
            DtypeKind::Oct => format!("oct{}", self.length),
            DtypeKind::Hex => format!("hex{}", self.length),
            DtypeKind::Bytes => format!("bytes{}", self.length),
        };
        format!("Dtype('{spec}')")
    }
}
