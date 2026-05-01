use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(from_py_object, module = "tibs")]
#[derive(Clone, Copy)]
pub enum Endianness {
    Unspecified,
    Big,
    Little,
}

impl Endianness {
    pub fn is_little_endian(optional_endianness: Option<Self>, length: usize) -> PyResult<bool> {
        match optional_endianness {
            Some(Endianness::Big) => {
                if length % 8 != 0 {
                    return Err(PyValueError::new_err(format!(
                        "Cannot create a big byte-endian value with a length of {length} bits. It must be a whole number of bytes long."
                    )));
                }
                Ok(false)
            }
            Some(Endianness::Little) => {
                if length % 8 != 0 {
                    return Err(PyValueError::new_err(format!(
                        "Cannot create a little byte-endian value with a length of {length} bits. It must be a whole number of bytes long."
                    )));
                }
                Ok(true)
            }
            _ => Ok(false),
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
