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
mod reader;
mod tibs_;
mod view;

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

create_exception!(
    tibs,
    ReadError,
    PyValueError,
    "A requested read could not be completed."
);
create_exception!(
    tibs,
    DecodeError,
    PyValueError,
    "Encoded Tibs data could not be decoded."
);

#[pymodule]
fn tibs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Scott Griffiths")?;
    m.add("ReadError", m.py().get_type::<ReadError>())?;
    m.add("DecodeError", m.py().get_type::<DecodeError>())?;
    m.add_class::<tibs_::Tibs>()?;
    m.add_class::<mutibs::Mutibs>()?;
    m.add_class::<enums::ByteOrder>()?;
    m.add_class::<enums::BitOrder>()?;
    m.add_class::<enums::Codec>()?;
    m.add_class::<enums::DtypeKind>()?;
    m.add_class::<view::View>()?;
    m.add_class::<view::MutableView>()?;
    m.add_class::<reader::Reader>()?;
    m.add_class::<reader::Bookmark>()?;
    m.add_class::<dtype::Dtype>()?;
    m.add_class::<dtype::DtypeSingle>()?;
    m.add_class::<dtype::DtypeArray>()?;
    m.add_class::<dtype::DtypeTuple>()?;
    Ok(())
}
