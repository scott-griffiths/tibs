use pyo3::{pyclass, pymethods, PyResult};
use crate::enums::{DtypeKind, Endianness, BitOrder};

#[pyclass(module = "tibs", frozen)]
pub struct Dtype {
    pub(crate) kind: DtypeKind,
    pub(crate) length: i64,
    pub(crate) byte_order: Endianness,
    pub(crate) bit_order: BitOrder,
}

#[pymethods]
impl Dtype {
    #[new]
    pub fn py_new(kind: DtypeKind, length: i64, byte_order: Endianness, bit_order: BitOrder) -> PyResult<Self> {
        Ok(Dtype {
            kind,
            length,
            byte_order,
            bit_order,
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

    #[getter]
    fn bit_order(&self) -> BitOrder {
        self.bit_order
    }

    pub fn __repr__(&self) -> String {
        format!("Dtype({}, {}, {}, {})", self.kind.repr_name(), self.length, self.byte_order.repr_name(), self.bit_order.repr_name())

    }

}