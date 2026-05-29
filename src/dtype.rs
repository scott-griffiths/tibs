use crate::enums::{BitOrder, DtypeKind, Endianness};
use pyo3::{PyResult, pyclass, pymethods};

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
    #[pyo3(signature = (kind, length, byte_order = Endianness::Unspecified, bit_order = BitOrder::Msb0), text_signature = "($self, kind, length, byte_order, bit_order)")]
    pub fn py_new(
        kind: DtypeKind,
        length: i64,
        byte_order: Option<Endianness>,
        bit_order: Option<BitOrder>,
    ) -> PyResult<Self> {
        Ok(Dtype {
            kind,
            length,
            byte_order: byte_order.unwrap_or(Endianness::Unspecified),
            bit_order: bit_order.unwrap_or(BitOrder::Msb0),
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
        format!(
            "Dtype({}, {}, {}, {})",
            self.kind.repr_name(),
            self.length,
            self.byte_order.repr_name(),
            self.bit_order.repr_name()
        )
    }
}
