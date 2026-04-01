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

#[pyclass(from_py_object, module = "tibs")]
#[derive(Clone, Copy)]
pub enum Endianness {
    Big,
    Little,
}

impl Endianness {
    pub fn is_big_endian(optional_endianness: Option<Self>) -> bool {
        match optional_endianness {
            None => true, // Default to big endianness
            Some(Endianness::Big) => true,
            Some(Endianness::Little) => false,
        }
    }
}
