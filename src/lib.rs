pub mod core;
pub mod helpers;
pub mod iterator;
pub mod mutibs;
pub mod tibs_;

use pyo3::prelude::*;

#[pymodule]
fn tibs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tibs_::Tibs>()?;
    m.add_class::<mutibs::Mutibs>()?;
    Ok(())
}

mod bits_tests;
#[cfg(test)]
mod mutable_test;
