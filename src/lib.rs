pub mod tibs_;
pub mod core;
pub mod helpers;
pub mod iterator;
pub mod mutibs;

use pyo3::prelude::*;

#[pymodule]
fn tibs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tibs_::Tibs>()?;
    m.add_class::<mutibs::Mutibs>()?;

    m.add_function(wrap_pyfunction!(tibs_::set_dtype_parser, m)?)?;
    m.add_function(wrap_pyfunction!(tibs_::bits_from_any, m)?)?;
    m.add_function(wrap_pyfunction!(mutibs::mutable_bits_from_any, m)?)?;
    Ok(())
}

#[cfg(test)]
mod mutable_test;
mod bits_tests;