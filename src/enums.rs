use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(from_py_object, module = "tibs")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteOrder {
    Unspecified,
    Big,
    Little,
}

impl ByteOrder {
    pub(crate) fn repr_name(self) -> &'static str {
        match self {
            ByteOrder::Unspecified => "ByteOrder.Unspecified",
            ByteOrder::Big => "ByteOrder.Big",
            ByteOrder::Little => "ByteOrder.Little",
        }
    }

    pub fn is_little_endian(optional_byte_order: Option<Self>, length: usize) -> PyResult<bool> {
        match optional_byte_order {
            Some(ByteOrder::Big) => {
                if !length.is_multiple_of(8) {
                    return Err(PyValueError::new_err(format!(
                        "Cannot create a big-endian byte-order value with a length of {length} bits. It must be a whole number of bytes long."
                    )));
                }
                Ok(false)
            }
            Some(ByteOrder::Little) => {
                if !length.is_multiple_of(8) {
                    return Err(PyValueError::new_err(format!(
                        "Cannot create a little-endian byte-order value with a length of {length} bits. It must be a whole number of bytes long."
                    )));
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

#[pyclass(from_py_object, module = "tibs")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BitOrder {
    Msb0,
    Lsb0,
}

impl BitOrder {
    pub(crate) fn repr_name(self) -> &'static str {
        match self {
            BitOrder::Msb0 => "BitOrder.Msb0",
            BitOrder::Lsb0 => "BitOrder.Lsb0",
        }
    }
}

#[pyclass(from_py_object, module = "tibs")]
#[derive(Clone, Copy)]
pub enum Codec {
    Auto,
    Raw,
    Rice,
    Zstd,
}

#[pyclass(from_py_object, module = "tibs")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtypeKind {
    Uint,
    Int,
    Float,
    BFloat,
    Bool,
    Bits,
    Bytes,
    Bin,
    Oct,
    Hex,
}

impl DtypeKind {
    pub(crate) fn repr_name(self) -> &'static str {
        match self {
            DtypeKind::Uint => "DtypeKind.Uint",
            DtypeKind::Int => "DtypeKind.Int",
            DtypeKind::Float => "DtypeKind.Float",
            DtypeKind::BFloat => "DtypeKind.BFloat",
            DtypeKind::Bool => "DtypeKind.Bool",
            DtypeKind::Bits => "DtypeKind.Bits",
            DtypeKind::Bytes => "DtypeKind.Bytes",
            DtypeKind::Bin => "DtypeKind.Bin",
            DtypeKind::Oct => "DtypeKind.Oct",
            DtypeKind::Hex => "DtypeKind.Hex",
        }
    }
}
