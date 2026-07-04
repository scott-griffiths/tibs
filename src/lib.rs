#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::len_zero)]
#![allow(clippy::collapsible_if)]

pub mod core;
pub mod dtype;
pub mod enums;
pub mod helpers;
pub mod iterator;
pub mod mutibs;
pub mod tibs_;
pub mod view;

use pyo3::prelude::*;

#[pymodule]
fn tibs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tibs_::Tibs>()?;
    m.add_class::<mutibs::Mutibs>()?;
    m.add_class::<enums::ByteOrder>()?;
    m.add_class::<enums::BitOrder>()?;
    m.add_class::<enums::Codec>()?;
    m.add_class::<enums::DtypeKind>()?;
    m.add_class::<view::View>()?;
    m.add_class::<view::MutableView>()?;
    m.add_class::<dtype::Dtype>()?;
    Ok(())
}
