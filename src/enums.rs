use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

///     The order of the bytes of a whole-byte value.
///
///     ``Unspecified`` is the default and is interpreted bitwise big-endian, which
///     for whole-byte data is the same as ``Big``, but can be used at any length.
///     ``Big`` and ``Little`` require a whole number of bytes.
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

///     How bit labels are mapped within each byte by a :class:`View`.
///
///     ``Msb0`` is the default, and matches ordinary indexing: label 0 is the most
///     significant bit of the byte. ``Lsb0`` numbers from the least significant bit
///     instead, as many hardware manuals do. Only the labels differ, never the
///     stored data.
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

///     The storage strategy used by :meth:`Tibs.encode`.
///
///     ``Auto`` is the default and picks whatever is most compact. ``Raw`` stores the
///     bits directly and is the canonical form to use when the exact bytes matter.
///     ``Rice`` suits sparse data, and ``Zstd`` larger byte-like data.
#[pyclass(from_py_object, module = "tibs")]
#[derive(Clone, Copy)]
pub enum Codec {
    Auto,
    Raw,
    Rice,
    Zstd,
}

///     The kind of value a :class:`DtypeSingle` encodes.
///
///     The kind says how bits become a Python value, and the dtype's length says how
///     many bits one value takes. ``Uint``, ``Int``, ``Float``, ``Bits``, ``Bin``,
///     ``Oct``, ``Hex`` and ``Bytes`` are families of widths, so a dtype using one
///     always needs a length. The remaining thirteen - ``Bool``, ``BFloat`` and the
///     narrow float formats - have an intrinsic width, so the kind on its own can be
///     used wherever a dtype is accepted.
///
///     Every kind, and the dtype string it corresponds to, is listed in the
///     documentation for this enum.
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
    Binary8P3,
    Binary8P4,
    OcpE4M3Saturate,
    OcpE4M3Overflow,
    OcpE5M2Saturate,
    OcpE5M2Overflow,
    OcpE3M2,
    OcpE2M3,
    OcpE2M1,
    OcpE8M0,
    OcpInt8,
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
            DtypeKind::Binary8P3 => "DtypeKind.Binary8P3",
            DtypeKind::Binary8P4 => "DtypeKind.Binary8P4",
            DtypeKind::OcpE4M3Saturate => "DtypeKind.OcpE4M3Saturate",
            DtypeKind::OcpE4M3Overflow => "DtypeKind.OcpE4M3Overflow",
            DtypeKind::OcpE5M2Saturate => "DtypeKind.OcpE5M2Saturate",
            DtypeKind::OcpE5M2Overflow => "DtypeKind.OcpE5M2Overflow",
            DtypeKind::OcpE3M2 => "DtypeKind.OcpE3M2",
            DtypeKind::OcpE2M3 => "DtypeKind.OcpE2M3",
            DtypeKind::OcpE2M1 => "DtypeKind.OcpE2M1",
            DtypeKind::OcpE8M0 => "DtypeKind.OcpE8M0",
            DtypeKind::OcpInt8 => "DtypeKind.OcpInt8",
        }
    }
}
