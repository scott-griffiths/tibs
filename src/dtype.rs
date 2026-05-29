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
        Ok(Dtype {
            kind,
            length: length as usize,
            byte_order: byte_order.unwrap_or(Endianness::Unspecified),
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
    #[pyo3(signature = (length, byte_order = Endianness::Unspecified), text_signature = "(cls, length, byte_order)")]
    pub fn bytes(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        length: i64,
        byte_order: Option<Endianness>,
    ) -> PyResult<Self> {
        Self::py_new(DtypeKind::Bytes, length, byte_order)
    }

    #[classmethod]
    #[pyo3(signature = (length, byte_order = Endianness::Unspecified), text_signature = "(cls, length, byte_order)")]
    pub fn bin(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        length: i64,
        byte_order: Option<Endianness>,
    ) -> PyResult<Self> {
        Self::py_new(DtypeKind::Bin, length, byte_order)
    }

    #[classmethod]
    #[pyo3(signature = (length, byte_order = Endianness::Unspecified), text_signature = "(cls, length, byte_order)")]
    pub fn oct(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        length: i64,
        byte_order: Option<Endianness>,
    ) -> PyResult<Self> {
        Self::py_new(DtypeKind::Oct, length, byte_order)
    }

    #[classmethod]
    #[pyo3(signature = (length, byte_order = Endianness::Unspecified), text_signature = "(cls, length, byte_order)")]
    pub fn hex(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        length: i64,
        byte_order: Option<Endianness>,
    ) -> PyResult<Self> {
        Self::py_new(DtypeKind::Hex, length, byte_order)
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
        format!(
            "Dtype({}, {}, {})",
            self.kind.repr_name(),
            self.length,
            self.byte_order.repr_name(),
        )
    }
}
