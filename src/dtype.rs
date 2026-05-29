use crate::enums::{DtypeKind, Endianness};
use pyo3::{PyResult, pyclass, pymethods};
use pyo3::exceptions::PyValueError;

#[pyclass(module = "tibs", frozen)]
pub struct Dtype {
    pub(crate) kind: DtypeKind,
    pub(crate) length: i64,
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
            length,
            byte_order: byte_order.unwrap_or(Endianness::Unspecified),
        })
    }

    #[getter]
    fn kind(&self) -> DtypeKind {
        self.kind
    }

    #[getter]
    fn length(&self) -> i64 {
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
