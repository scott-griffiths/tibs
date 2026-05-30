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
///         >>> Dtype.u(16, Endianness.Little)
///         Dtype.u(16, Endianness.Little)
///         >>> Tibs.from_value(Dtype.u(8), 15)
///         Tibs('0x0f')
///
#[pyclass(module = "tibs", frozen)]
pub struct Dtype {
    pub(crate) kind: DtypeKind,
    pub(crate) length: usize,
    pub(crate) byte_order: Endianness,
}

#[pymethods]
impl Dtype {
    /// Create a value encoding description.
    ///
    /// :param DtypeKind kind: The kind of value to encode or decode.
    /// :param int length: The number of bits used by one value.
    /// :param Endianness byte_order: The byte order for integer and floating-point values. Defaults to ``Endianness.Unspecified``.
    /// :return: A new ``Dtype``.
    ///
    /// :raises ValueError: if ``length`` is not greater than zero, if byte order is used with a non-numeric kind, or if byte order is used with a non-byte length.
    ///
    #[new]
    #[pyo3(signature = (kind, length, byte_order = Endianness::Unspecified), text_signature = "($self, kind, length, byte_order)")]
    pub fn py_new(kind: DtypeKind, length: i64, byte_order: Option<Endianness>) -> PyResult<Self> {
        if length <= 0 {
            return Err(PyValueError::new_err(format!(
                "Dtype length must be greater than zero, but received {}.",
                length
            )));
        }
        let length = length as usize;
        let byte_order = byte_order.unwrap_or(Endianness::Unspecified);
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

    /// Create an unsigned integer dtype.
    ///
    /// :param int length: The number of bits used by one unsigned integer value.
    /// :param Endianness byte_order: The byte order for byte-wide values. Defaults to ``Endianness.Unspecified``.
    /// :return: A new unsigned integer ``Dtype``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype.u(8)
    ///     Dtype.u(8)
    ///
    #[classmethod]
    #[pyo3(signature = (length, byte_order = Endianness::Unspecified), text_signature = "(cls, length, byte_order)")]
    pub fn u(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        length: i64,
        byte_order: Option<Endianness>,
    ) -> PyResult<Self> {
        Self::py_new(DtypeKind::Uint, length, byte_order)
    }

    /// Create a signed integer dtype.
    ///
    /// :param int length: The number of bits used by one signed integer value.
    /// :param Endianness byte_order: The byte order for byte-wide values. Defaults to ``Endianness.Unspecified``.
    /// :return: A new signed integer ``Dtype``.
    ///
    #[classmethod]
    #[pyo3(signature = (length, byte_order = Endianness::Unspecified), text_signature = "(cls, length, byte_order)")]
    pub fn i(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        length: i64,
        byte_order: Option<Endianness>,
    ) -> PyResult<Self> {
        Self::py_new(DtypeKind::Int, length, byte_order)
    }

    /// Create a floating-point dtype.
    ///
    /// :param int length: The number of bits used by one floating-point value. Supported value conversion lengths are 16, 32 and 64.
    /// :param Endianness byte_order: The byte order for byte-wide values. Defaults to ``Endianness.Unspecified``.
    /// :return: A new floating-point ``Dtype``.
    ///
    #[classmethod]
    #[pyo3(signature = (length, byte_order = Endianness::Unspecified), text_signature = "(cls, length, byte_order)")]
    pub fn f(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        length: i64,
        byte_order: Option<Endianness>,
    ) -> PyResult<Self> {
        Self::py_new(DtypeKind::Float, length, byte_order)
    }

    /// Create a bytes dtype.
    ///
    /// :param int length: The number of bits used by one bytes value.
    /// :return: A new bytes ``Dtype``.
    ///
    #[classmethod]
    #[pyo3(signature = (length), text_signature = "(cls, length)")]
    pub fn bytes(_cls: &pyo3::Bound<'_, pyo3::types::PyType>, length: i64) -> PyResult<Self> {
        Self::py_new(DtypeKind::Bytes, length, None)
    }

    /// Create a binary string dtype.
    ///
    /// :param int length: The number of bits used by one binary string value.
    /// :return: A new binary string ``Dtype``.
    ///
    #[classmethod]
    #[pyo3(signature = (length), text_signature = "(cls, length)")]
    pub fn bin(_cls: &pyo3::Bound<'_, pyo3::types::PyType>, length: i64) -> PyResult<Self> {
        Self::py_new(DtypeKind::Bin, length, None)
    }

    /// Create an octal string dtype.
    ///
    /// :param int length: The number of bits used by one octal string value.
    /// :return: A new octal string ``Dtype``.
    ///
    #[classmethod]
    #[pyo3(signature = (length), text_signature = "(cls, length)")]
    pub fn oct(_cls: &pyo3::Bound<'_, pyo3::types::PyType>, length: i64) -> PyResult<Self> {
        Self::py_new(DtypeKind::Oct, length, None)
    }

    /// Create a hexadecimal string dtype.
    ///
    /// :param int length: The number of bits used by one hexadecimal string value.
    /// :return: A new hexadecimal string ``Dtype``.
    ///
    #[classmethod]
    #[pyo3(signature = (length), text_signature = "(cls, length)")]
    pub fn hex(_cls: &pyo3::Bound<'_, pyo3::types::PyType>, length: i64) -> PyResult<Self> {
        Self::py_new(DtypeKind::Hex, length, None)
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
            Endianness::Unspecified => "".to_string(),
            _ => format!(", {}", self.byte_order.repr_name()),
        };
        match self.kind {
            DtypeKind::Uint => {
                format!("Dtype.u({}{})", self.length, byte_order_str)
            }
            DtypeKind::Int => {
                format!("Dtype.i({}{})", self.length, byte_order_str)
            }
            DtypeKind::Float => {
                format!("Dtype.f({}{})", self.length, byte_order_str)
            }
            DtypeKind::Bin => {
                format!("Dtype.bin({})", self.length)
            }
            DtypeKind::Oct => {
                format!("Dtype.oct({})", self.length)
            }
            DtypeKind::Hex => {
                format!("Dtype.hex({})", self.length)
            }
            DtypeKind::Bytes => {
                format!("Dtype.bytes({})", self.length)
            }
        }
    }
}
