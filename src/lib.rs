pub mod core;
pub mod enums;
pub mod helpers;
pub mod iterator;
pub mod mutibs;
pub mod tibs_;

use pyo3::prelude::*;

#[pymodule]
fn tibs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tibs_::Tibs>()?;
    m.add_class::<mutibs::Mutibs>()?;
    m.add_class::<enums::BitIndexing>()?;
    m.add_class::<enums::Endianness>()?;
    Ok(())
}
