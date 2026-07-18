#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::len_zero)]
#![allow(clippy::collapsible_if)]

mod codec;
mod core;
mod dtype;
mod enums;
mod helpers;
mod iterator;
mod mutibs;
mod tibs_;
mod view;

use pyo3::prelude::*;

#[pymodule]
fn tibs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__author__", "Scott Griffiths")?;
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
