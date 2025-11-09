pub mod bits;
pub mod core;
pub mod helpers;
pub mod iterator;
pub mod mutable;

use pyo3::prelude::*;

#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bits::Bits>()?;
    m.add_class::<mutable::MutableBits>()?;

    m.add_function(wrap_pyfunction!(bits::set_dtype_parser, m)?)?;
    m.add_function(wrap_pyfunction!(bits::bits_from_any, m)?)?;
    m.add_function(wrap_pyfunction!(mutable::mutable_bits_from_any, m)?)?;
    Ok(())
}

#[cfg(test)]
mod mutable_test;
mod bits_tests;