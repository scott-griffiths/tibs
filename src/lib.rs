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

/// Fast, bit-level binary data.
///
/// ``Tibs`` is an immutable sequence of bits, as ``bytes`` is of bytes, and
/// ``Mutibs`` is its mutable counterpart, as ``bytearray`` is. Lengths and
/// positions are in bits, so nothing needs to be byte aligned. Use them for
/// packets, registers, instruction formats, file headers, bitsets and streams
/// whose fields can be any number of bits long.
///
/// There are three ways to use them:
///
/// * As a container of bits: slice, concatenate, compare, ``find``,
///   ``find_all``, ``replace``, ``count``, ``starts_with``, ``split_at`` and
///   ``chunks``, all with bit positions.
/// * As typed fields: build with ``from_u``, ``from_i``, ``from_f``,
///   ``from_bytes`` or ``from_value``, and read back with ``.u``, ``.i``,
///   ``.f``, ``.hex``, ``.bin`` and ``.bytes``. ``.le`` and ``.be`` give
///   byte-order views, ``.lsb0`` numbers bits from the least significant end,
///   ``field`` takes inclusive datasheet-style labels, ``Dtype`` describes
///   compound records, and ``Reader`` walks through a stream.
/// * As a set of bits: ``&``, ``|``, ``^`` and ``~``, plus ``count_and``,
///   ``count_xor``, ``intersects`` and ``is_subset_of``, which count or test
///   without building an intermediate result.
///
/// Anywhere a ``Tibs`` is expected, a string such as ``'0xff'``, ``'0b101'``
/// or ``'0o7, 0b1'``, a ``bytes``-like object or a list of 0s and 1s is
/// accepted too.
///
/// .. code-block:: pycon
///
///     >>> from tibs import Tibs, Mutibs, Reader
///     >>> frame = Tibs('0b101') + Tibs.from_u(1000, 12) + b'ok'
///     >>> len(frame), frame[3:15].u, frame.find(b'ok')
///     (31, 1000, 15)
///     >>> Tibs(b'\x34\x12').le.u
///     4660
///     >>> r = Reader(frame)
///     >>> r.read_value('u3'), r.read_value('u12'), r.read_bits(16).bytes
///     (5, 1000, b'ok')
///     >>> m = Mutibs.from_zeros(8)
///     >>> m.set([0, 7])
///     >>> m.bin
///     '10000001'
///
/// The full documentation is at https://tibs.readthedocs.io/.
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
