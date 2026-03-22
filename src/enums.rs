use pyo3::prelude::*;

#[pyclass(from_py_object, module = "tibs")]
#[derive(Clone, Copy)]
pub enum BitIndexing {
    Msb0,
    Lsb0,
}

impl BitIndexing {
    pub fn is_msb0(optional_bit_indexing: Option<Self>) -> bool {
        match optional_bit_indexing {
            None => true, // Default to Msb0
            Some(BitIndexing::Msb0) => true,
            Some(BitIndexing::Lsb0) => false,
        }
    }
}
