use crate::enums::{DtypeKind, Endianness};
use pyo3::exceptions::PyValueError;
use pyo3::{PyResult, pyclass, pymethods};

#[pyclass(module = "tibs", frozen)]
pub struct Dtype {
    pub(crate) kind: DtypeKind,
    pub(crate) length: usize,
    pub(crate) byte_order: Endianness,
}

#[pymethods]
impl Dtype {
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

    #[classmethod]
    #[pyo3(signature = (length, byte_order = Endianness::Unspecified), text_signature = "(cls, length, byte_order)")]
    pub fn u(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        length: i64,
        byte_order: Option<Endianness>,
    ) -> PyResult<Self> {
        Self::py_new(DtypeKind::Uint, length, byte_order)
    }

    #[classmethod]
    #[pyo3(signature = (length, byte_order = Endianness::Unspecified), text_signature = "(cls, length, byte_order)")]
    pub fn i(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        length: i64,
        byte_order: Option<Endianness>,
    ) -> PyResult<Self> {
        Self::py_new(DtypeKind::Int, length, byte_order)
    }

    #[classmethod]
    #[pyo3(signature = (length, byte_order = Endianness::Unspecified), text_signature = "(cls, length, byte_order)")]
    pub fn f(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        length: i64,
        byte_order: Option<Endianness>,
    ) -> PyResult<Self> {
        Self::py_new(DtypeKind::Float, length, byte_order)
    }

    #[classmethod]
    #[pyo3(signature = (length), text_signature = "(cls, length)")]
    pub fn bytes(_cls: &pyo3::Bound<'_, pyo3::types::PyType>, length: i64) -> PyResult<Self> {
        Self::py_new(DtypeKind::Bytes, length, None)
    }

    #[classmethod]
    #[pyo3(signature = (length), text_signature = "(cls, length)")]
    pub fn bin(_cls: &pyo3::Bound<'_, pyo3::types::PyType>, length: i64) -> PyResult<Self> {
        Self::py_new(DtypeKind::Bin, length, None)
    }

    #[classmethod]
    #[pyo3(signature = (length), text_signature = "(cls, length)")]
    pub fn oct(_cls: &pyo3::Bound<'_, pyo3::types::PyType>, length: i64) -> PyResult<Self> {
        Self::py_new(DtypeKind::Oct, length, None)
    }

    #[classmethod]
    #[pyo3(signature = (length), text_signature = "(cls, length)")]
    pub fn hex(_cls: &pyo3::Bound<'_, pyo3::types::PyType>, length: i64) -> PyResult<Self> {
        Self::py_new(DtypeKind::Hex, length, None)
    }

    #[getter]
    fn kind(&self) -> DtypeKind {
        self.kind
    }

    #[getter]
    fn length(&self) -> usize {
        self.length
    }

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
