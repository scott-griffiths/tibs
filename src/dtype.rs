use crate::core::BitCollection;
use crate::enums::{BitOrder, ByteOrder, DtypeKind};
use crate::helpers::{
    BS, BV, BitAccumulator, bv_from_bytes_slice, head_bit_offset, push_lsb0_field,
    store_msb0_field_as_lsb0, try_bv_from_bytes, try_byte_buffer, validate_slice, whole_bytes,
};
use crate::iterator::ValuesIterator;
use crate::tibs_::{
    BytewisePacker, Tibs, WordField, bv_from_value, bv_from_values_iter, py_from_raw_range,
    py_from_value, py_value_from_lsb0, py_values_from_lsb0, py_values_from_range,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyBytes, PyString, PyTuple, PyType};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock, RwLock};

/// The tail of the message given when a spec doesn't start with any known kind.
const KIND_HINT: &str = "expected a kind: either a fixed format such as 'binary8p3' or \
     'ocp_e4m3_saturate', or 'u', 'i', 'f', 'bf', 'bool', 'bits', 'bin', 'oct', 'hex' or \
     'bytes' followed by a length in bits, for example 'u12'";

/// Scalar encodings whose bit length is intrinsic to the named format rather
/// than written as a suffix in the dtype spec.
const FIXED_FORMATS: [(&str, DtypeKind, usize); 11] = [
    ("binary8p3", DtypeKind::Binary8P3, 8),
    ("binary8p4", DtypeKind::Binary8P4, 8),
    ("ocp_e4m3_saturate", DtypeKind::OcpE4M3Saturate, 8),
    ("ocp_e4m3_overflow", DtypeKind::OcpE4M3Overflow, 8),
    ("ocp_e5m2_saturate", DtypeKind::OcpE5M2Saturate, 8),
    ("ocp_e5m2_overflow", DtypeKind::OcpE5M2Overflow, 8),
    ("ocp_e3m2", DtypeKind::OcpE3M2, 6),
    ("ocp_e2m3", DtypeKind::OcpE2M3, 6),
    ("ocp_e2m1", DtypeKind::OcpE2M1, 4),
    ("ocp_e8m0", DtypeKind::OcpE8M0, 8),
    ("ocp_int8", DtypeKind::OcpInt8, 8),
];

/// Scalar kind prefixes whose bit length is written as a suffix.
const LENGTH_SUFFIX_FORMATS: [(&str, DtypeKind, &str); 9] = [
    ("bytes", DtypeKind::Bytes, "bytes16"),
    ("bits", DtypeKind::Bits, "bits12"),
    ("bf", DtypeKind::BFloat, "bf16"),
    ("bin", DtypeKind::Bin, "bin12"),
    ("oct", DtypeKind::Oct, "oct12"),
    ("hex", DtypeKind::Hex, "hex12"),
    ("u", DtypeKind::Uint, "u12"),
    ("i", DtypeKind::Int, "i12"),
    ("f", DtypeKind::Float, "f32"),
];

fn fixed_format_for_kind(kind: DtypeKind) -> Option<(&'static str, usize)> {
    FIXED_FORMATS
        .iter()
        .find_map(|&(spec, candidate, length)| (candidate == kind).then_some((spec, length)))
}

/// The bit length a kind fixes on its own, if it fixes one.
///
/// These are exactly the kinds for which a bare [`DtypeKind`] already describes
/// a complete dtype — the fixed formats, plus `Bool` (always 1 bit) and
/// `BFloat` (always 16). Every other kind is a family of widths, so a length
/// has to be written alongside it.
pub(crate) fn intrinsic_length(kind: DtypeKind) -> Option<usize> {
    match kind {
        DtypeKind::Bool => Some(1),
        DtypeKind::BFloat => Some(16),
        _ => fixed_format_for_kind(kind).map(|(_, length)| length),
    }
}

/// Resolve the length to build a scalar dtype with, filling in the intrinsic
/// one when the caller omitted it.
fn resolve_length(kind: DtypeKind, length: Option<i64>) -> PyResult<i64> {
    match (length, intrinsic_length(kind)) {
        (Some(length), _) => Ok(length),
        (None, Some(intrinsic)) => Ok(intrinsic as i64),
        (None, None) => {
            let example = match example_spec(kind) {
                Some(spec) => format!(" For example, '{spec}'."),
                None => String::new(),
            };
            Err(PyValueError::new_err(format!(
                "{} does not determine a length on its own, so one must be given.{example}",
                kind.repr_name(),
            )))
        }
    }
}

/// A valid dtype of a kind that takes a length, for use in the message shown
/// when the length is missing.
///
/// Each one has to satisfy that kind's own length rule — `Float` admits only
/// 16, 32 and 64, and `Bytes` only multiples of 8 — so these are not a single
/// width with different prefixes. `None` for the kinds that fix their own
/// length, which never reach here from [`resolve_length`].
fn example_spec(kind: DtypeKind) -> Option<&'static str> {
    LENGTH_SUFFIX_FORMATS
        .iter()
        .find_map(|&(_, candidate, example)| (candidate == kind).then_some(example))
}

/// Translate a numpy/struct style spec such as `"u4"` (four *bytes*) into the
/// equivalent tibs spec, e.g. `"u32_le"`. `None` if the translation isn't a
/// valid dtype, so that the caller falls back to a hint without a suggestion.
fn numpy_style_suggestion(rest: &str, order: &str) -> Option<String> {
    let (kind, digits) = rest.split_at_checked(1)?;
    if !matches!(kind, "u" | "i" | "f") || digits.is_empty() {
        return None;
    }
    let bits = digits.parse::<usize>().ok()?.checked_mul(8)?;
    let candidate = format!("{kind}{bits}{order}");
    SingleDtype::parse(&candidate).is_ok().then_some(candidate)
}

/// Translate a long-form kind name borrowed from numpy, C or Python — `"uint12"`,
/// `"float32"`, `"double"` — into the short tibs spelling. `None` if `base` isn't
/// one of those or the translation isn't a valid dtype.
///
/// Every suggestion produced here starts with a bare `u`, `i` or `f`, so
/// validating one can never re-enter this function.
fn long_kind_name_suggestion(base: &str, byte_order: ByteOrder) -> Option<String> {
    let order = match byte_order {
        ByteOrder::Unspecified => "",
        ByteOrder::Little => "_le",
        ByteOrder::Big => "_be",
    };
    let candidate = match base {
        "double" => format!("f64{order}"),
        "single" => format!("f32{order}"),
        "half" => format!("f16{order}"),
        _ => {
            let (short, digits) = [("uint", "u"), ("int", "i"), ("float", "f"), ("sint", "i")]
                .into_iter()
                .find_map(|(word, short)| Some((short, base.strip_prefix(word)?)))?;
            if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            format!("{short}{digits}{order}")
        }
    };
    SingleDtype::parse(&candidate).is_ok().then_some(candidate)
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SingleDtype {
    pub(crate) kind: DtypeKind,
    pub(crate) length: usize,
    pub(crate) byte_order: ByteOrder,
}

impl SingleDtype {
    fn from_parts(kind: DtypeKind, length: i64, byte_order: ByteOrder) -> PyResult<Self> {
        if length < 0 {
            return Err(PyValueError::new_err(format!(
                "Dtype length can't be negative, but received {length}."
            )));
        }
        let length = length as usize;
        if let Some(required_length) = intrinsic_length(kind)
            && length != required_length
        {
            return Err(PyValueError::new_err(format!(
                "A Dtype of kind {} must have length {required_length} bits. Received {length}.",
                kind.repr_name()
            )));
        }
        // The literal kinds have an obvious empty value (`b''`, `''`, an empty
        // `Tibs`), so they may be zero bits long. An integer has no zero-bit
        // value, and every other kind is caught by its own length rule.
        match kind {
            DtypeKind::Uint | DtypeKind::Int if length == 0 => {
                let prefix = if kind == DtypeKind::Uint { "u" } else { "i" };
                return Err(PyValueError::new_err(format!(
                    "A Dtype of kind {} ('{prefix}') needs a length of at least 1 bit. Received 0.",
                    kind.repr_name()
                )));
            }
            DtypeKind::Float if !matches!(length, 16 | 32 | 64) => {
                return Err(PyValueError::new_err(format!(
                    "A Dtype of kind {} must have length 16, 32 or 64 bits. Received {length}.",
                    kind.repr_name()
                )));
            }
            DtypeKind::Bytes if !length.is_multiple_of(8) => {
                return Err(PyValueError::new_err(format!(
                    "A Dtype of kind {} must have a length that is a multiple of 8 bits. Received {length}.",
                    kind.repr_name()
                )));
            }
            DtypeKind::Hex if !length.is_multiple_of(4) => {
                return Err(PyValueError::new_err(format!(
                    "A Dtype of kind {} must have a length that is a multiple of 4 bits. Received {length}.",
                    kind.repr_name()
                )));
            }
            DtypeKind::Oct if !length.is_multiple_of(3) => {
                return Err(PyValueError::new_err(format!(
                    "A Dtype of kind {} must have a length that is a multiple of 3 bits. Received {length}.",
                    kind.repr_name()
                )));
            }
            _ => {}
        }
        if byte_order != ByteOrder::Unspecified {
            match kind {
                DtypeKind::Uint | DtypeKind::Int | DtypeKind::Float | DtypeKind::BFloat => {
                    if !length.is_multiple_of(8) {
                        return Err(PyValueError::new_err(format!(
                            "If a Dtype byte_order is given, the length must be a multiple of 8 (length = {length})."
                        )));
                    }
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "A byte order cannot be specified for a Dtype of kind {}.",
                        kind.repr_name()
                    )));
                }
            }
        }
        Ok(Self {
            kind,
            length,
            byte_order,
        })
    }

    fn parse(spec: &str) -> PyResult<Self> {
        let spec = spec.trim().to_ascii_lowercase();

        // numpy and struct write the byte order as a leading character and
        // count bytes rather than bits, so both differences need pointing out
        // before anything else rejects the spec as an unknown kind.
        if let Some(first) = spec.chars().next()
            && matches!(first, '<' | '>' | '=' | '|')
        {
            let order = match first {
                '<' => "_le",
                '>' => "_be",
                _ => "",
            };
            let hint = match numpy_style_suggestion(&spec[1..], order) {
                Some(suggestion) => format!(" Did you mean '{suggestion}'?"),
                None => String::new(),
            };
            return Err(PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}': a byte order is written as a '_le' or '_be' \
                 suffix, not a leading character, and lengths are in bits, not bytes.{hint}"
            )));
        }

        let (base, byte_order) = if let Some(base) = spec.strip_suffix("_le") {
            (base, ByteOrder::Little)
        } else if let Some(base) = spec.strip_suffix("_be") {
            (base, ByteOrder::Big)
        } else {
            (spec.as_str(), ByteOrder::Unspecified)
        };

        if let Some(&(_, kind, length)) = FIXED_FORMATS
            .iter()
            .find(|(candidate, _, _)| *candidate == base)
        {
            return Self::from_parts(kind, length as i64, byte_order);
        }

        if base == "bool" {
            return Self::from_parts(DtypeKind::Bool, 1, byte_order);
        }

        // A bool is one bit by definition, so 'bool8' is nearly always someone
        // reaching for eight of them.
        if let Some(count) = base.strip_prefix("bool")
            && count.chars().all(|c| c.is_ascii_digit())
            && !count.is_empty()
        {
            return Err(PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}': a 'bool' is always 1 bit. \
                 Did you mean '[bool; {count}]'?"
            )));
        }

        // 'bf16' is the only spelling accepted, but the format is usually
        // written out in full elsewhere, so send that spelling somewhere
        // rather than letting it fail as an unparseable length.
        if base == "bfloat" || base == "bfloat16" {
            return Err(PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}': did you mean 'bf16'?"
            )));
        }

        // Long-form kind names taken from numpy, C or Python. These have to be
        // caught before the prefix matching below, which would otherwise read
        // 'int8' as kind 'i' with a bit length of 'nt8'.
        if let Some(suggestion) = long_kind_name_suggestion(base, byte_order) {
            return Err(PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}': did you mean '{suggestion}'?"
            )));
        }

        let Some((kind, kind_name, length_text)) =
            LENGTH_SUFFIX_FORMATS
                .iter()
                .find_map(|&(kind_name, kind, _)| {
                    base.strip_prefix(kind_name)
                        .map(|length| (kind, kind_name, length))
                })
        else {
            return Err(PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}': {KIND_HINT}."
            )));
        };

        if length_text.is_empty() {
            return Err(PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}': expected a bit length after '{kind_name}', \
                 for example '{kind_name}8'."
            )));
        }
        if !length_text.chars().all(|c| c.is_ascii_digit()) {
            // A byte order written without its underscore, or a length written
            // with Python's numeric underscores, both land here.
            // Both corrections below are only offered once they are known to
            // parse: 'u12_le' is not a valid dtype even though 'u12le' looks
            // like it is missing nothing but an underscore.
            let valid = |candidate: String| Self::parse(&candidate).is_ok().then_some(candidate);
            let hint = if let Some(candidate) = length_text
                .strip_suffix("le")
                .or_else(|| length_text.strip_suffix("be"))
                .filter(|digits| !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()))
                .and_then(|digits| {
                    valid(format!(
                        "{kind_name}{digits}_{}",
                        &length_text[length_text.len() - 2..]
                    ))
                }) {
                format!(" Did you mean '{candidate}'?")
            } else if let Some(candidate) = Some(length_text)
                .filter(|text| {
                    text.contains('_') && text.chars().all(|c| c.is_ascii_digit() || c == '_')
                })
                .and_then(|text| valid(format!("{kind_name}{}", text.replace('_', ""))))
            {
                format!(" Underscores are not allowed in a bit length: did you mean '{candidate}'?")
            } else if let Some(suffix) = length_text.rsplit_once('_') {
                format!(
                    " '_{}' is not a valid byte order: only '_le' and '_be' are supported.",
                    suffix.1
                )
            } else {
                String::new()
            };
            return Err(PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}': expected a whole number of bits after \
                 '{kind_name}', but found '{length_text}'.{hint}"
            )));
        }
        let length = length_text.parse::<i64>().map_err(|_| {
            PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}': bit length is too large."
            ))
        })?;
        Self::from_parts(kind, length, byte_order)
    }

    fn spec(&self) -> String {
        let byte_order = match self.byte_order {
            ByteOrder::Unspecified => "",
            ByteOrder::Little => "_le",
            ByteOrder::Big => "_be",
        };
        if let Some((spec, _)) = fixed_format_for_kind(self.kind) {
            return spec.to_string();
        }
        match self.kind {
            DtypeKind::Uint => format!("u{}{byte_order}", self.length),
            DtypeKind::Int => format!("i{}{byte_order}", self.length),
            DtypeKind::Float => format!("f{}{byte_order}", self.length),
            DtypeKind::BFloat => format!("bf{}{byte_order}", self.length),
            DtypeKind::Bool => "bool".to_string(),
            DtypeKind::Bits => format!("bits{}", self.length),
            DtypeKind::Bin => format!("bin{}", self.length),
            DtypeKind::Oct => format!("oct{}", self.length),
            DtypeKind::Hex => format!("hex{}", self.length),
            DtypeKind::Bytes => format!("bytes{}", self.length),
            _ => unreachable!("fixed formats returned above"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum DtypeRepr {
    Single(SingleDtype),
    Array { dtype: Box<DtypeRepr>, count: usize },
    Tuple(Vec<DtypeRepr>),
}

impl DtypeRepr {
    pub(crate) fn length(&self) -> PyResult<usize> {
        match self {
            Self::Single(dtype) => Ok(dtype.length),
            Self::Array { dtype, count } => dtype
                .length()?
                .checked_mul(*count)
                .ok_or_else(|| PyValueError::new_err("Dtype length is too large to represent.")),
            Self::Tuple(dtypes) => dtypes.iter().try_fold(0usize, |total, dtype| {
                total
                    .checked_add(dtype.length()?)
                    .ok_or_else(|| PyValueError::new_err("Dtype length is too large to represent."))
            }),
        }
    }

    /// Whether any part of this dtype names a byte order, at any nesting depth.
    ///
    /// A byte order is a property of each scalar field rather than of the whole
    /// value, so `"(u8, u16_le)"` counts even though only one field carries it.
    fn has_explicit_byte_order(&self) -> bool {
        match self {
            Self::Single(dtype) => dtype.byte_order != ByteOrder::Unspecified,
            Self::Array { dtype, .. } => dtype.has_explicit_byte_order(),
            Self::Tuple(dtypes) => dtypes.iter().any(Self::has_explicit_byte_order),
        }
    }

    /// Call `f` with the bit offset and length of each scalar field in turn,
    /// the first starting at `offset`, returning the offset just past the last.
    ///
    /// An array of zero-length items is skipped whole, so that a huge count of
    /// them costs nothing.
    fn for_each_field<F: FnMut(usize, usize)>(&self, offset: usize, f: &mut F) -> usize {
        match self {
            Self::Single(dtype) => {
                f(offset, dtype.length);
                offset + dtype.length
            }
            Self::Array { dtype, count } => {
                if dtype.length().unwrap_or(0) == 0 {
                    return offset;
                }
                (0..*count).fold(offset, |offset, _| dtype.for_each_field(offset, f))
            }
            Self::Tuple(dtypes) => dtypes
                .iter()
                .fold(offset, |offset, dtype| dtype.for_each_field(offset, f)),
        }
    }

    fn spec(&self) -> String {
        match self {
            Self::Single(dtype) => dtype.spec(),
            Self::Array { dtype, count } => format!("[{}; {count}]", dtype.spec()),
            Self::Tuple(dtypes) if dtypes.len() == 1 => {
                format!("({},)", dtypes[0].spec())
            }
            Self::Tuple(dtypes) => {
                let fields = dtypes.iter().map(Self::spec).collect::<Vec<_>>().join(", ");
                format!("({fields})")
            }
        }
    }
}

/// A precomputed layout for a record of scalar fields: a [`DtypeRepr::Tuple`]
/// of [`DtypeRepr::Single`]s (`struct`'s `">hhl"`, or an MPEG-header-style
/// fixed table), a [`DtypeRepr::Array`] of one, or a tuple whose items include
/// arrays of a scalar or tuples of scalars, such as `(u8, [u8; 4], (u4, u4))`.
/// It lets pack and unpack address each field directly instead of re-walking
/// `DtypeRepr` and recomputing offsets on every record. Deeper nesting
/// (array-of-tuple, a tuple inside a tuple inside a tuple, ...) has no layout,
/// and pack/unpack walk `DtypeRepr` recursively for it.
///
/// An array stores one field however long it is, so `Dtype("[u8; 1_000_000]")`
/// is as cheap to build as `Dtype("[u8; 4]")`, and a tuple stores one per
/// scalar or array in its spec, a count bounded by the spec's own arity.
pub(crate) struct RecordLayout {
    pub(crate) shape: RecordShape,
    /// The number of bits in one record.
    pub(crate) length: usize,
    /// The fields as [`BytewisePacker`]s, when every one is a whole-byte field
    /// one can write.
    pub(crate) packers: Option<Box<[BytewisePacker]>>,
    /// The fields as [`WordField`]s, when every one is a field one can read and
    /// write in place, which lets a record whose fields aren't all whole bytes
    /// skip the per-field `Tibs` and `BV` the recursive walk builds.
    pub(crate) words: Option<Box<[WordField]>>,
}

/// How the fields of a [`RecordLayout`] make up a record's value.
pub(crate) enum RecordShape {
    /// A tuple of scalars, one field per item.
    Flat,
    /// An array of a scalar: the one field, repeated this many times.
    Array(usize),
    /// A tuple whose items include arrays or tuples.
    Members(Box<[RecordMember]>),
}

/// One item of a [`RecordShape::Members`] tuple, addressing the layout's fields.
#[derive(Clone, Copy)]
pub(crate) enum RecordMember {
    /// A scalar, the field at this index.
    Field(usize),
    /// A tuple of scalars, the `len` fields from `start`.
    Tuple { start: usize, len: usize },
    /// An array of a scalar: the field at `field`, which is its first element,
    /// repeated `count` times end to end.
    Array { field: usize, count: usize },
}

fn single(repr: &DtypeRepr) -> Option<&SingleDtype> {
    match repr {
        DtypeRepr::Single(single) => Some(single),
        _ => None,
    }
}

fn build_record_layout(repr: &DtypeRepr) -> Option<RecordLayout> {
    // Each field with the bit offset of its start within the record.
    let mut fields: Vec<(SingleDtype, usize)> = Vec::new();
    let mut length = 0;
    let mut add = |single: &SingleDtype, count: usize, fields: &mut Vec<_>| {
        fields.push((single.clone(), length));
        length += single.length * count;
        fields.len() - 1
    };
    let shape = match repr {
        DtypeRepr::Single(_) => return None,
        DtypeRepr::Array { dtype, count } => {
            add(single(dtype)?, *count, &mut fields);
            RecordShape::Array(*count)
        }
        DtypeRepr::Tuple(dtypes) => {
            let mut members = Vec::with_capacity(dtypes.len());
            for dtype in dtypes {
                members.push(match dtype {
                    DtypeRepr::Single(single) => RecordMember::Field(add(single, 1, &mut fields)),
                    DtypeRepr::Array { dtype, count } => RecordMember::Array {
                        field: add(single(dtype)?, *count, &mut fields),
                        count: *count,
                    },
                    DtypeRepr::Tuple(inner) => {
                        let start = fields.len();
                        for dtype in inner {
                            add(single(dtype)?, 1, &mut fields);
                        }
                        RecordMember::Tuple {
                            start,
                            len: inner.len(),
                        }
                    }
                });
            }
            if members
                .iter()
                .all(|member| matches!(member, RecordMember::Field(_)))
            {
                RecordShape::Flat
            } else {
                RecordShape::Members(members.into_boxed_slice())
            }
        }
    };
    Some(RecordLayout {
        shape,
        length,
        packers: fields
            .iter()
            .map(|(single, _)| BytewisePacker::for_dtype(single))
            .collect(),
        words: fields
            .iter()
            .map(|(single, offset)| WordField::new(single, *offset))
            .collect(),
    })
}

struct DtypeParser<'a> {
    spec: &'a str,
    bytes: &'a [u8],
    pos: usize,
    /// Whether this parser may propose corrected specs in its error messages.
    /// Cleared for the throwaway parsers that check a proposal, so that
    /// checking a suggestion can't itself go looking for suggestions.
    suggest: bool,
}

impl<'a> DtypeParser<'a> {
    fn new(spec: &'a str) -> Self {
        Self {
            spec,
            bytes: spec.as_bytes(),
            pos: 0,
            suggest: true,
        }
    }

    fn parse(mut self) -> PyResult<DtypeRepr> {
        let dtype = self.parse_dtype()?;
        self.skip_whitespace();
        if self.pos != self.bytes.len() {
            // A separator left at the top level is a missing bracket: the
            // fields of a tuple need '(...)' around them, and an array needs
            // '[...]'. Only suggest the bracketed form if it actually parses.
            let trimmed = self.spec.trim();
            return match self.peek() {
                Some(b',') => self.error_hinted(
                    "a tuple dtype needs its fields inside parentheses",
                    self.checked_suggestion(format!("({trimmed})")),
                ),
                Some(b';') => self.error_hinted(
                    "an array dtype needs its dtype and count inside brackets",
                    self.checked_suggestion(format!("[{trimmed}]")),
                ),
                // Two dtypes separated by nothing but space: most likely a
                // tuple missing both its comma and its parentheses.
                _ => self.error_hinted(
                    "unexpected trailing text",
                    self.checked_suggestion(format!(
                        "({}, {})",
                        self.spec[..self.pos].trim(),
                        self.spec[self.pos..].trim()
                    )),
                ),
            };
        }
        Ok(dtype)
    }

    /// The canonical spelling of `candidate` if it parses as a dtype spec,
    /// otherwise `None`. Suggesting the canonical form rather than the
    /// candidate keeps whatever odd spacing was in the original out of the
    /// suggestion: `'[u8 4]'` is answered with `'[u8; 4]'`, not `'[u8 ; 4]'`.
    fn checked_suggestion(&self, candidate: String) -> Option<String> {
        if !self.suggest {
            return None;
        }
        let mut parser = DtypeParser::new(&candidate);
        parser.suggest = false;
        parser.parse().ok().map(|repr| repr.spec())
    }

    /// The spec with the byte at `self.pos` replaced by `replacement`.
    fn spec_with_replacement(&self, replacement: char) -> String {
        let mut candidate = String::with_capacity(self.spec.len());
        candidate.push_str(&self.spec[..self.pos]);
        candidate.push(replacement);
        candidate.push_str(&self.spec[self.pos + 1..]);
        candidate
    }

    fn parse_dtype(&mut self) -> PyResult<DtypeRepr> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'[') => self.parse_array(),
            Some(b'(') => self.parse_tuple(),
            Some(_) => self.parse_single(),
            None if self.spec.trim().is_empty() => Err(PyValueError::new_err(format!(
                "Cannot parse a Dtype from an empty spec: {KIND_HINT}."
            ))),
            None => self.error("expected a dtype, but reached the end of the spec"),
        }
    }

    fn parse_single(&mut self) -> PyResult<DtypeRepr> {
        let start = self.pos;
        while let Some(byte) = self.peek() {
            if byte.is_ascii_whitespace() || matches!(byte, b'[' | b']' | b'(' | b')' | b',' | b';')
            {
                break;
            }
            self.pos += 1;
        }
        if self.pos == start {
            return self.error("expected a scalar dtype");
        }
        Ok(DtypeRepr::Single(SingleDtype::parse(
            &self.spec[start..self.pos],
        )?))
    }

    fn parse_array(&mut self) -> PyResult<DtypeRepr> {
        self.pos += 1;
        let open = self.pos - 1;
        let dtype = self.parse_dtype()?;
        self.skip_whitespace();
        if self.peek() != Some(b';') {
            // '[u8, 4]' means '[u8; 4]' and '[u8 4]' means the same, but
            // '[u12, u12]' means '(u12, u12)'. Try the array readings first,
            // since only a spec whose second half is a count can be an array.
            let as_array = if self.peek() == Some(b',') {
                self.spec_with_replacement(';')
            } else {
                format!("{}; {}", &self.spec[..self.pos], &self.spec[self.pos..])
            };
            let mut as_tuple = self.spec.to_string();
            as_tuple.replace_range(open..open + 1, "(");
            if let Some(close) = as_tuple.rfind(']') {
                as_tuple.replace_range(close..close + 1, ")");
            }
            let suggestion = self
                .checked_suggestion(as_array)
                .or_else(|| self.checked_suggestion(as_tuple));
            return self.error_hinted(
                "expected ';' between the array dtype and count, \
                 as an array dtype is written '[dtype; count]'",
                suggestion,
            );
        }
        self.pos += 1;
        self.skip_whitespace();
        let start = self.pos;
        while matches!(self.peek(), Some(byte) if byte.is_ascii_digit()) {
            self.pos += 1;
        }
        if start == self.pos {
            return self.error("expected an array count");
        }
        let count = self.spec[start..self.pos].parse::<usize>().map_err(|_| {
            PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{}': array count is too large.",
                self.spec
            ))
        })?;
        self.skip_whitespace();
        if self.peek().is_none() {
            return self.error("unterminated array dtype: expected ']'");
        }
        self.expect(b']', "expected ']' after the array count")?;
        let repr = DtypeRepr::Array {
            dtype: Box::new(dtype),
            count,
        };
        repr.length()?;
        Ok(repr)
    }

    fn parse_tuple(&mut self) -> PyResult<DtypeRepr> {
        self.pos += 1;
        self.skip_whitespace();
        if self.peek() == Some(b')') {
            return self.error("tuple dtypes must contain at least one dtype");
        }

        let mut dtypes = vec![self.parse_dtype()?];
        self.skip_whitespace();
        match self.peek() {
            Some(b',') => self.pos += 1,
            Some(b')') => {
                return self.error(
                    "a one-element tuple dtype needs a trailing comma, for example '(u8,)'",
                );
            }
            _ => return self.error("expected ',' between the fields of a tuple dtype"),
        }

        loop {
            self.skip_whitespace();
            if self.peek() == Some(b')') {
                self.pos += 1;
                break;
            }
            dtypes.push(self.parse_dtype()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b')') if dtypes.len() > 1 => {
                    self.pos += 1;
                    break;
                }
                None => return self.error("unterminated tuple dtype: expected ')'"),
                _ => return self.error("expected ',' or ')' after a tuple field"),
            }
        }

        let repr = DtypeRepr::Tuple(dtypes);
        repr.length()?;
        Ok(repr)
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(byte) if byte.is_ascii_whitespace()) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, expected: u8, message: &str) -> PyResult<()> {
        if self.peek() != Some(expected) {
            return self.error(message);
        }
        self.pos += 1;
        Ok(())
    }

    fn error<T>(&self, message: &str) -> PyResult<T> {
        self.error_hinted(message, None)
    }

    /// As [`Self::error`], but with a corrected spec suggested after the
    /// position, so that the message reads as one sentence then the fix.
    fn error_hinted<T>(&self, message: &str, suggestion: Option<String>) -> PyResult<T> {
        let hint = match suggestion {
            Some(suggestion) => format!(" Did you mean '{suggestion}'?"),
            None => String::new(),
        };
        Err(PyValueError::new_err(format!(
            "Cannot parse Dtype spec '{}': {message} at position {}.{hint}",
            self.spec, self.pos
        )))
    }
}

///     The base class for fixed-width value descriptors.
///
///     Constructing ``Dtype`` parses ``spec`` and returns a
///     :class:`DtypeSingle`, :class:`DtypeArray` or :class:`DtypeTuple`.
///
///     :param str spec: A scalar, array or tuple dtype specification.
///     :return: The corresponding concrete dtype.
///
///     .. code-block:: pycon
///
///         >>> Dtype("u16_le")
///         DtypeSingle('u16_le')
///         >>> Dtype("[(u8, bool); 2]")
///         DtypeArray('[(u8, bool); 2]')
///
#[pyclass(module = "tibs", frozen, subclass, skip_from_py_object)]
#[derive(Clone)]
pub struct Dtype {
    pub(crate) repr: DtypeRepr,
    pub(crate) length: usize,
    pub(crate) record_layout: Option<Arc<RecordLayout>>,
}

// `length` and `record_layout` are both pure functions of `repr`, so equality
// and hashing are defined over `repr` alone rather than derived over every
// field — deriving would need `RecordLayout` to carry its own `PartialEq`/
// `Hash` for no semantic benefit, since two dtypes with equal `repr` always
// have equal `length`/`record_layout` already.
impl PartialEq for Dtype {
    fn eq(&self, other: &Self) -> bool {
        self.repr == other.repr
    }
}

impl Eq for Dtype {}

impl Hash for Dtype {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.repr.hash(state);
    }
}

impl Dtype {
    fn from_repr(repr: DtypeRepr) -> PyResult<Self> {
        let length = repr.length()?;
        // Nothing this long could be packed or unpacked, since no container can
        // hold it, so it is refused here rather than at its first use.
        if length > BS::MAX_BITS {
            return Err(PyValueError::new_err(format!(
                "Dtype length of {length} bits is more than the {} bits this build of tibs \
                 supports in one container.",
                BS::MAX_BITS
            )));
        }
        let record_layout = build_record_layout(&repr).map(Arc::new);
        Ok(Self {
            repr,
            length,
            record_layout,
        })
    }

    fn parse_spec(spec: &str) -> PyResult<Self> {
        Self::from_repr(DtypeParser::new(spec).parse()?)
    }

    fn into_python(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.repr {
            DtypeRepr::Single(_) => Ok(Py::new(
                py,
                PyClassInitializer::from(self).add_subclass(DtypeSingle),
            )?
            .into_any()),
            DtypeRepr::Array { .. } => Ok(Py::new(
                py,
                PyClassInitializer::from(self).add_subclass(DtypeArray),
            )?
            .into_any()),
            DtypeRepr::Tuple(_) => Ok(Py::new(
                py,
                PyClassInitializer::from(self).add_subclass(DtypeTuple),
            )?
            .into_any()),
        }
    }

    pub(crate) fn single(&self) -> Option<&SingleDtype> {
        match &self.repr {
            DtypeRepr::Single(dtype) => Some(dtype),
            _ => None,
        }
    }

    pub(crate) fn spec(&self) -> String {
        self.repr.spec()
    }

    pub(crate) fn has_explicit_byte_order(&self) -> bool {
        self.repr.has_explicit_byte_order()
    }

    /// Refuse a dtype that names its own byte order when the values are in
    /// LSB-first order, which already decides it.
    ///
    /// Under [`BitOrder::Lsb0`] every value is read least significant bit
    /// first, so a whole-byte value is little-endian by construction. A `_le`
    /// or `_be` dtype would state the byte order a second time, as it would
    /// through an `lsb0` view, and is refused for the same reason.
    pub(crate) fn check_bit_order(&self, bit_order: BitOrder) -> PyResult<()> {
        if bit_order == BitOrder::Lsb0 && self.has_explicit_byte_order() {
            return Err(PyValueError::new_err(format!(
                "Cannot use the dtype '{}' with BitOrder.Lsb0. LSB0 order reads each value \
                 least significant bit first, which already decides the byte order, so a '_le' \
                 or '_be' dtype would state it a second time. Remove the byte order from the dtype.",
                self.spec()
            )));
        }
        Ok(())
    }

    /// Bits `start..end` of the LSB-first stream `bits`, rearranged into the
    /// order the ordinary decoders read: each field of each value in turn,
    /// most significant bit first.
    ///
    /// Value `i` of an `n`-bit scalar dtype is labels `start + i*n` (its least
    /// significant bit) to `start + i*n + n - 1`, which is what
    /// `bits.lsb0.field(start + i*n + n - 1, start + i*n)` selects. An array or
    /// tuple applies the same rule to each of its scalar fields.
    ///
    /// `bits` must be a whole number of bytes long, which
    /// [`validate_lsb0_length`] checks, and `start..end` must lie within it.
    /// Any bits after the last whole value are passed on as one more field, so
    /// that a range that isn't a whole number of values reaches the decoder at
    /// its true length and is refused with the usual message.
    pub(crate) fn lsb0_to_msb0(&self, bits: &BS, start: usize, end: usize) -> Tibs {
        debug_assert!(bits.len().is_multiple_of(8) && start <= end && end <= bits.len());
        let first = start / 8;
        let bytes = whole_bytes(&bits[first * 8..end.div_ceil(8) * 8]);
        let base = start - first * 8;
        let total = end - start;
        let records = total.checked_div(self.length).unwrap_or(0);
        let mut out = BitAccumulator::with_bit_capacity(Some(total));
        match self.single() {
            Some(single) => {
                let length = single.length;
                for index in 0..records {
                    push_lsb0_field(&mut out, &bytes, base + index * length, length);
                }
            }
            None => {
                let mut push = |label, length| push_lsb0_field(&mut out, &bytes, label, length);
                for index in 0..records {
                    self.repr
                        .for_each_field(base + index * self.length, &mut push);
                }
            }
        }
        let done = records * self.length;
        push_lsb0_field(&mut out, &bytes, base + done, total - done);
        Tibs::from_bv(out.into_bitvec())
    }

    /// Decode one value from labels `start..end` of the LSB-first stream
    /// `bits`, as [`Self::lsb0_to_msb0`] lays it out.
    ///
    /// A scalar whose value is a word is read straight out of the stream;
    /// anything else is rearranged for the ordinary decoder first.
    pub(crate) fn py_value_lsb0(
        &self,
        py: Python<'_>,
        bits: &BS,
        start: usize,
        end: usize,
    ) -> PyResult<Py<PyAny>> {
        if let Some(value) = py_value_from_lsb0(py, self, bits, start, end)? {
            return Ok(value);
        }
        py_from_value(py, self, &self.lsb0_to_msb0(bits, start, end))
    }

    /// [`Self::py_value_lsb0`] for every value in `start..end`.
    pub(crate) fn py_values_lsb0(
        &self,
        py: Python<'_>,
        bits: &BS,
        start: usize,
        end: usize,
    ) -> PyResult<Vec<Py<PyAny>>> {
        if let Some(values) = py_values_from_lsb0(py, self, bits, start, end)? {
            return Ok(values);
        }
        py_values_from_range(py, &self.lsb0_to_msb0(bits, start, end), self, None, None)
    }

    /// Whole values of this dtype packed MSB-first, as the encoders write
    /// them, rearranged into an LSB-first stream: the inverse of
    /// [`Self::lsb0_to_msb0`].
    ///
    /// The stream is padded with zeros to a whole number of bytes, the padding
    /// taking the highest labels, as LSB0 labels only exist in whole bytes.
    fn msb0_to_lsb0(&self, packed: &BV) -> PyResult<BV> {
        let total = packed.len();
        let mut out = try_byte_buffer(total)?;
        out.resize(total.div_ceil(8), 0);
        let raw = packed.as_raw_slice();
        let offset = head_bit_offset(packed.as_bitslice());
        let records = total.checked_div(self.length).unwrap_or(0);
        debug_assert_eq!(records * self.length, total);
        match self.single() {
            Some(single) => {
                let length = single.length;
                for index in 0..records {
                    store_msb0_field_as_lsb0(&mut out, raw, offset, index * length, length);
                }
            }
            None => {
                let mut store =
                    |bit, length| store_msb0_field_as_lsb0(&mut out, raw, offset, bit, length);
                for index in 0..records {
                    self.repr.for_each_field(index * self.length, &mut store);
                }
            }
        }
        try_bv_from_bytes(out)
    }

    /// Encode one value in `bit_order`, for `Dtype.pack` and the `from_value`
    /// constructors.
    pub(crate) fn pack_bv(&self, value: &Bound<'_, PyAny>, bit_order: BitOrder) -> PyResult<BV> {
        self.check_bit_order(bit_order)?;
        let packed = bv_from_value(self, value)?;
        match bit_order {
            BitOrder::Msb0 => Ok(packed),
            BitOrder::Lsb0 => self.msb0_to_lsb0(&packed),
        }
    }

    /// Encode and concatenate values in `bit_order`, for `Dtype.pack_values`
    /// and the `from_values` constructors.
    pub(crate) fn pack_values_bv(
        &self,
        py: Python<'_>,
        iterable: &Bound<'_, PyAny>,
        bit_order: BitOrder,
    ) -> PyResult<BV> {
        self.check_bit_order(bit_order)?;
        let packed = bv_from_values_iter(py, self, iterable)?;
        match bit_order {
            BitOrder::Msb0 => Ok(packed),
            BitOrder::Lsb0 => self.msb0_to_lsb0(&packed),
        }
    }

    /// The range `start..end` of an LSB-first stream `len` bits long, after
    /// every check that reading it with [`BitOrder::Lsb0`] needs: this dtype
    /// names no byte order, the stream is whole bytes, and the range is in it.
    pub(crate) fn lsb0_slice(
        &self,
        len: usize,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<(usize, usize)> {
        self.check_bit_order(BitOrder::Lsb0)?;
        validate_lsb0_length(len)?;
        validate_slice(len, start, end)
    }

    /// Run `read` over `bits`, an LSB-first stream, with the range
    /// `start..end` resolved, after the checks that reading with
    /// [`BitOrder::Lsb0`] needs.
    ///
    /// A `Tibs` or a `bytes` is read where it is rather than copied first.
    fn with_lsb0_range<R>(
        &self,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
        read: impl FnOnce(&BS, usize, usize) -> PyResult<R>,
    ) -> PyResult<R> {
        let checked = |bits: &BS| {
            let (start, end) = self.lsb0_slice(bits.len(), start, end)?;
            read(bits, start, end)
        };
        if let Ok(tibs) = bits.cast::<Tibs>() {
            return checked(tibs.get().as_bitslice());
        }
        if let Ok(data) = bits.cast_exact::<PyBytes>() {
            return checked(BS::from_slice(data.as_bytes()));
        }
        checked(bits.extract::<Tibs>()?.as_bitslice())
    }

    /// The error for reading values of this dtype until the bits run out when
    /// it is zero bits long, as any number of them fit in any length.
    #[cold]
    pub(crate) fn zero_length_values_error(&self) -> PyErr {
        PyValueError::new_err(format!(
            "Can't tell how many zero-length values to read: the dtype '{}' has length 0, \
             so any number of its values fit in any number of bits.",
            self.spec()
        ))
    }

    fn class_name(&self) -> &'static str {
        match self.repr {
            DtypeRepr::Single(_) => "DtypeSingle",
            DtypeRepr::Array { .. } => "DtypeArray",
            DtypeRepr::Tuple(_) => "DtypeTuple",
        }
    }
}

/// Refuse a run that isn't a whole number of bytes as an LSB-first stream.
///
/// LSB0 labels count from the least significant bit of each byte, so a final
/// part-byte has no labels for its bits, just as an `lsb0` view needs whole
/// bytes.
pub(crate) fn validate_lsb0_length(len: usize) -> PyResult<()> {
    if !len.is_multiple_of(8) {
        return Err(PyValueError::new_err(format!(
            "Cannot read {len} bits with BitOrder.Lsb0. LSB0 labels count from the least \
             significant bit of each byte, so the bits must be a whole number of bytes long."
        )));
    }
    Ok(())
}

/// The bits of the whole bytes that cover `start..end`, as a start and a
/// length, for a caller that copies them out of a source it has to lock.
pub(crate) fn lsb0_covering_bytes(start: usize, end: usize) -> (usize, usize) {
    let first = start / 8 * 8;
    (first, end.div_ceil(8) * 8 - first)
}

/// Upper bound on the number of parsed specs kept in [`SPEC_CACHE`].
///
/// Specs come from source code in the overwhelming majority of programs, so a
/// handful of entries serve every call site. The cap exists only so that a
/// program building specs dynamically — `f"u{n}"` over a wide range of `n` —
/// grows the cache to a bounded size rather than leaking. Past the cap, lookups
/// still hit for everything already cached and misses simply parse as before.
const SPEC_CACHE_LIMIT: usize = 256;

/// Parsed dtype specs, keyed by the exact string passed in.
///
/// Parsing a spec costs appreciably more than the bulk operation it precedes
/// for short reads — a scalar `to_value("u8")` spent roughly three quarters of
/// its time in the parser — and the parse result is a pure function of the
/// string, so caching it is transparent. Two spellings of the same dtype (say
/// `"u8"` and `"U8 "`) simply occupy separate entries.
static SPEC_CACHE: LazyLock<RwLock<HashMap<String, Dtype>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Parse a dtype spec, going through [`SPEC_CACHE`].
pub(crate) fn parse_spec_cached(spec: &str) -> PyResult<Dtype> {
    if let Ok(cache) = SPEC_CACHE.read()
        && let Some(dtype) = cache.get(spec)
    {
        return Ok(dtype.clone());
    }
    // Invalid specs are never cached: they are an error path, not a hot one,
    // and caching them would let a typo loop fill the cache with entries that
    // only ever produce errors.
    let dtype = Dtype::parse_spec(spec)?;
    if let Ok(mut cache) = SPEC_CACHE.write()
        && cache.len() < SPEC_CACHE_LIMIT
    {
        cache.insert(spec.to_string(), dtype.clone());
    }
    Ok(dtype)
}

/// Build the dtype described by a bare kind, which is possible exactly when the
/// kind fixes its own length.
fn dtype_from_kind(kind: DtypeKind) -> PyResult<Dtype> {
    let length = resolve_length(kind, None)?;
    Dtype::from_repr(DtypeRepr::Single(SingleDtype::from_parts(
        kind,
        length,
        ByteOrder::Unspecified,
    )?))
}

/// Build a scalar dtype of `kind` and `length`, with the same checks as
/// parsing its spec.
pub(crate) fn dtype_from_kind_and_length(kind: DtypeKind, length: usize) -> PyResult<Dtype> {
    let length = i64::try_from(length)
        .map_err(|_| PyValueError::new_err("Dtype length is too large to represent."))?;
    Dtype::from_repr(DtypeRepr::Single(SingleDtype::from_parts(
        kind,
        length,
        ByteOrder::Unspecified,
    )?))
}

/// The kind named by a length-suffix prefix written without its length, such
/// as `'hex'` or `'u'`, for callers that supply the length themselves.
pub(crate) fn kind_from_prefix(prefix: &str) -> Option<DtypeKind> {
    LENGTH_SUFFIX_FORMATS.iter().find_map(|&(name, kind, _)| {
        (name == prefix && intrinsic_length(kind).is_none()).then_some(kind)
    })
}

/// The prefix that names `kind` in a spec, for a kind whose length has to be
/// written after it.
pub(crate) fn prefix_for_kind(kind: DtypeKind) -> Option<&'static str> {
    LENGTH_SUFFIX_FORMATS
        .iter()
        .find_map(|&(name, candidate, _)| (candidate == kind).then_some(name))
        .filter(|_| intrinsic_length(kind).is_none())
}

pub(crate) fn extract_dtype(obj: &Bound<'_, PyAny>) -> PyResult<Dtype> {
    if let Ok(dtype) = obj.extract::<PyRef<'_, Dtype>>() {
        return Ok(dtype.clone());
    }
    // A kind that fixes its own length already carries everything a dtype does,
    // so accepting one here saves wrapping `DtypeKind.OcpE2M1` in a
    // `DtypeSingle` that adds no information.
    if let Ok(kind) = obj.extract::<DtypeKind>() {
        return dtype_from_kind(kind);
    }
    if let Ok(spec) = obj.cast::<PyString>() {
        return parse_spec_cached(spec.to_str()?);
    }
    Err(PyTypeError::new_err(
        "dtype must be a Dtype instance, a DtypeKind with a fixed length, or a dtype string.",
    ))
}

#[pymethods]
impl Dtype {
    /// Parse a scalar, array or tuple dtype specification.
    ///
    /// :param str spec: The dtype specification.
    /// :return: A concrete dtype instance.
    /// :raises ValueError: if the specification is invalid.
    #[new]
    #[pyo3(signature = (spec, /), text_signature = "(spec, /)")]
    fn py_new(py: Python<'_>, spec: &str) -> PyResult<Py<PyAny>> {
        parse_spec_cached(spec)?.into_python(py)
    }

    /// The number of bits used by one complete value.
    #[getter]
    fn length(&self) -> usize {
        self.length
    }

    /// Encode one Python value as a :class:`Tibs`.
    ///
    /// With ``bit_order=BitOrder.Lsb0`` the value is written least significant
    /// bit first, from the least significant bit of the first byte, as
    /// :meth:`unpack` reads it with the same option. The result is then padded
    /// with zero bits to a whole number of bytes, the padding taking the
    /// highest labels.
    ///
    /// :param object value: A scalar or structured value matching this dtype.
    /// :param BitOrder bit_order: The order the bits are written in. Defaults to ``BitOrder.Msb0``.
    /// :return: The encoded bits.
    /// :raises ValueError: if a structured value has the wrong number of items, or if
    ///     ``bit_order`` is ``BitOrder.Lsb0`` and the dtype names a byte order.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("(u8, bool)").pack((15, True))
    ///     Tibs('0b000011111')
    ///     >>> Dtype("u12").pack(0x321, bit_order=BitOrder.Lsb0)
    ///     Tibs('0x2103')
    #[pyo3(signature = (value, /, bit_order = BitOrder::Msb0), text_signature = "($self, value, /, bit_order=None)")]
    fn pack(&self, value: &Bound<'_, PyAny>, bit_order: Option<BitOrder>) -> PyResult<Tibs> {
        let bit_order = bit_order.unwrap_or(BitOrder::Msb0);
        Ok(Tibs::from_bv(self.pack_bv(value, bit_order)?))
    }

    /// Encode and concatenate Python values as a :class:`Tibs`.
    ///
    /// With ``bit_order=BitOrder.Lsb0`` the values are packed least significant
    /// bit first, filling each byte from its least significant bit, as
    /// :meth:`unpack_values` reads them with the same option. The result is
    /// then padded with zero bits to a whole number of bytes, the padding
    /// taking the highest labels.
    ///
    /// :param Iterable iterable: Values matching this dtype.
    /// :param BitOrder bit_order: The order the bits are written in. Defaults to ``BitOrder.Msb0``.
    /// :return: The concatenated encoded bits.
    /// :raises ValueError: if ``bit_order`` is ``BitOrder.Lsb0`` and the dtype names a byte order.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("u8").pack_values([1, 2, 3])
    ///     Tibs('0x010203')
    ///     >>> Dtype("u12").pack_values([0x321, 0x654], bit_order=BitOrder.Lsb0)
    ///     Tibs('0x214365')
    #[pyo3(signature = (iterable, /, bit_order = BitOrder::Msb0), text_signature = "($self, iterable, /, bit_order=None)")]
    fn pack_values(
        &self,
        py: Python<'_>,
        iterable: &Bound<'_, PyAny>,
        bit_order: Option<BitOrder>,
    ) -> PyResult<Tibs> {
        let bit_order = bit_order.unwrap_or(BitOrder::Msb0);
        Ok(Tibs::from_bv(self.pack_values_bv(py, iterable, bit_order)?))
    }

    /// Decode one complete value from a bit sequence.
    ///
    /// With ``bit_order=BitOrder.Lsb0``, ``start`` and ``end`` are LSB0 bit
    /// labels, counted from the least significant bit of each byte, and the
    /// value is read with ``start`` as its least significant bit. That is the
    /// value ``bits.lsb0.field(end - 1, start)`` gives, and ``bits`` must be a
    /// whole number of bytes long.
    ///
    /// :param object bits: Anything promotable to :class:`Tibs`.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to ``len(bits)``.
    /// :param BitOrder bit_order: The order the bits are read in. Defaults to ``BitOrder.Msb0``.
    /// :return: One scalar value, or a tuple for an array or tuple dtype.
    /// :raises ValueError: if the selected range is not exactly :attr:`length` bits, or if
    ///     ``bit_order`` is ``BitOrder.Lsb0`` and either the dtype names a byte order or
    ///     ``bits`` is not a whole number of bytes long.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("u12").unpack(b'\x21\x43', end=12, bit_order=BitOrder.Lsb0) == 0x321
    ///     True
    #[pyo3(signature = (bits, /, start = None, end = None, bit_order = BitOrder::Msb0), text_signature = "($self, bits, /, start=None, end=None, bit_order=None)")]
    fn unpack(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
        bit_order: Option<BitOrder>,
    ) -> PyResult<Py<PyAny>> {
        if bit_order == Some(BitOrder::Lsb0) {
            return self.with_lsb0_range(bits, start, end, |bits, start, end| {
                self.py_value_lsb0(py, bits, start, end)
            });
        }
        // A `Tibs` or a `bytes` is read where it already is, rather than copied
        // into a new `Tibs` first, which for a short record costs about as much
        // as decoding it.
        if let Ok(tibs) = bits.cast::<Tibs>() {
            let tibs = tibs.get();
            let (start, end) = validate_slice(tibs.len(), start, end)?;
            let (bytes, shift, _) = tibs.raw_data_ref();
            return py_from_raw_range(py, self, bytes, shift, start, end, || {
                Ok(tibs.get_slice_unchecked(start, end - start))
            });
        }
        if let Ok(data) = bits.cast_exact::<PyBytes>() {
            let data = data.as_bytes();
            let (start, end) = validate_slice(data.len() * 8, start, end)?;
            // Only the bytes the range covers are copied, so reading a short
            // field from a long buffer doesn't copy the whole buffer.
            return py_from_raw_range(py, self, data, 0, start, end, || {
                let covered = data[start / 8..end.div_ceil(8)].to_vec();
                let bits = bv_from_bytes_slice(covered, Some(start % 8), Some(end - start))?;
                Ok(Tibs::from_bv(bits))
            });
        }
        let bits = bits.extract::<Tibs>()?;
        let (start, end) = validate_slice(bits.len(), start, end)?;
        py_from_value(py, self, &bits.get_slice_unchecked(start, end - start))
    }

    /// Decode a list of complete values from a bit sequence.
    ///
    /// With ``bit_order=BitOrder.Lsb0`` the values are packed least significant
    /// bit first, filling each byte from its least significant bit, as in
    /// FAT12 tables or 1-bit pixel formats such as Qt's ``MonoLSB``. Value
    /// ``i`` of an ``n``-bit dtype is then ``bits.lsb0.field(start + i*n + n-1,
    /// start + i*n)``, ``start`` and ``end`` are LSB0 bit labels, and ``bits``
    /// must be a whole number of bytes long. An array or tuple dtype applies
    /// the same rule to each of its fields in turn.
    ///
    /// :param object bits: Anything promotable to :class:`Tibs`.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to ``len(bits)``.
    /// :param BitOrder bit_order: The order the bits are read in. Defaults to ``BitOrder.Msb0``.
    /// :return: A list of decoded values.
    /// :raises ValueError: if the selected range is not a multiple of :attr:`length`, if
    ///     :attr:`length` is zero, as the number of values could then be anything, or if
    ///     ``bit_order`` is ``BitOrder.Lsb0`` and either the dtype names a byte order or
    ///     ``bits`` is not a whole number of bytes long.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("u2").unpack_values(b'\xe4', bit_order=BitOrder.Lsb0)
    ///     [0, 1, 2, 3]
    ///     >>> [hex(x) for x in Dtype("u12").unpack_values(b'\x21\x43\x65', bit_order=BitOrder.Lsb0)]
    ///     ['0x321', '0x654']
    #[pyo3(signature = (bits, /, start = None, end = None, bit_order = BitOrder::Msb0), text_signature = "($self, bits, /, start=None, end=None, bit_order=None)")]
    fn unpack_values(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
        bit_order: Option<BitOrder>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        if bit_order == Some(BitOrder::Lsb0) {
            return self.with_lsb0_range(bits, start, end, |bits, start, end| {
                self.py_values_lsb0(py, bits, start, end)
            });
        }
        let bits = bits.extract::<Tibs>()?;
        py_values_from_range(py, &bits, self, start, end)
    }

    /// Lazily decode complete values from a bit sequence.
    ///
    /// ``bit_order`` has the same meaning as for :meth:`unpack_values`. With
    /// ``BitOrder.Lsb0`` the selected bits are rearranged once, up front, and
    /// the values are then decoded lazily from that.
    ///
    /// :param object bits: Anything promotable to :class:`Tibs`.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to ``len(bits)``.
    /// :param BitOrder bit_order: The order the bits are read in. Defaults to ``BitOrder.Msb0``.
    /// :return: An iterator of decoded values.
    /// :raises ValueError: if the selected range is not a multiple of :attr:`length`, if
    ///     :attr:`length` is zero, as the number of values could then be anything, or if
    ///     ``bit_order`` is ``BitOrder.Lsb0`` and either the dtype names a byte order or
    ///     ``bits`` is not a whole number of bytes long.
    #[pyo3(signature = (bits, /, start = None, end = None, bit_order = BitOrder::Msb0), text_signature = "($self, bits, /, start=None, end=None, bit_order=None)")]
    fn unpack_values_iter(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
        bit_order: Option<BitOrder>,
    ) -> PyResult<Py<ValuesIterator>> {
        let (bits, start, end) = if bit_order == Some(BitOrder::Lsb0) {
            let bits = self.with_lsb0_range(bits, start, end, |bits, start, end| {
                Ok(self.lsb0_to_msb0(bits, start, end))
            })?;
            let len = bits.len();
            (bits, 0, len)
        } else {
            let bits = bits.extract::<Tibs>()?;
            let (start, end) = validate_slice(bits.len(), start, end)?;
            (bits, start, end)
        };
        ValuesIterator::new(py, Py::new(py, bits)?, self.clone(), start, end)
    }

    fn __str__(&self) -> String {
        self.spec()
    }

    fn __repr__(&self) -> String {
        format!("{}('{}')", self.class_name(), self.spec())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        let Ok(other) = other.extract::<PyRef<'_, Dtype>>() else {
            return false;
        };
        self == &*other
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let hash = hasher.finish() as isize;
        if hash == -1 { -2 } else { hash }
    }
}

///     A scalar dtype with a kind, bit length and optional byte order.
///
///     Construct directly from a scalar specification, use
///     :meth:`from_params`, or obtain one from the :class:`Dtype` factory.
///
#[pyclass(module = "tibs", frozen, extends = Dtype, skip_from_py_object)]
pub struct DtypeSingle;

#[pymethods]
impl DtypeSingle {
    /// Parse a scalar dtype specification.
    ///
    /// :param str spec: A specification such as ``"u8"`` or ``"f32_le"``.
    /// :raises ValueError: if ``spec`` describes an array or tuple.
    #[new]
    #[pyo3(signature = (spec, /), text_signature = "(spec, /)")]
    fn py_new(spec: &str) -> PyResult<PyClassInitializer<Self>> {
        let dtype = parse_spec_cached(spec)?;
        if !matches!(dtype.repr, DtypeRepr::Single(_)) {
            return Err(PyValueError::new_err(
                "DtypeSingle requires a scalar dtype specification.",
            ));
        }
        Ok(PyClassInitializer::from(dtype).add_subclass(Self))
    }

    /// Construct a scalar dtype from explicit parameters.
    ///
    /// :param DtypeKind kind: The scalar value kind.
    /// :param int | None length: The bit length. May be omitted for a kind that
    ///     fixes its own length, such as ``DtypeKind.OcpE2M1`` or
    ///     ``DtypeKind.Bool``. Only the ``Bits``, ``Bytes``, ``Bin``, ``Oct``
    ///     and ``Hex`` kinds can have a length of zero.
    /// :param ByteOrder | None byte_order: The byte order. Defaults to unspecified.
    /// :return: A scalar dtype.
    ///
    ///     .. code-block:: pycon
    ///
    ///         >>> DtypeSingle.from_params(DtypeKind.Uint, 12)
    ///         DtypeSingle('u12')
    ///         >>> DtypeSingle.from_params(DtypeKind.OcpE2M1)
    ///         DtypeSingle('ocp_e2m1')
    ///
    #[classmethod]
    #[pyo3(signature = (kind, length = None, /, byte_order = ByteOrder::Unspecified), text_signature = "(cls, kind, length=None, /, byte_order=None)")]
    fn from_params(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        kind: DtypeKind,
        length: Option<i64>,
        byte_order: Option<ByteOrder>,
    ) -> PyResult<Py<Self>> {
        let byte_order = byte_order.unwrap_or(ByteOrder::Unspecified);
        let length = resolve_length(kind, length)?;
        let repr = DtypeRepr::Single(SingleDtype::from_parts(kind, length, byte_order)?);
        Py::new(
            py,
            PyClassInitializer::from(Dtype::from_repr(repr)?).add_subclass(Self),
        )
    }

    /// The scalar value kind.
    #[getter]
    fn kind(slf: PyRef<'_, Self>) -> DtypeKind {
        slf.as_super().single().unwrap().kind
    }

    /// The scalar byte order.
    #[getter]
    fn byte_order(slf: PyRef<'_, Self>) -> ByteOrder {
        slf.as_super().single().unwrap().byte_order
    }
}

///     A fixed number of repetitions of another dtype.
///
///     Array values pack from any iterable with exactly :attr:`count` items
///     and unpack to Python tuples. The count may be zero, giving a dtype of
///     length zero whose value is ``()``.
///
#[pyclass(module = "tibs", frozen, extends = Dtype, skip_from_py_object)]
pub struct DtypeArray;

#[pymethods]
impl DtypeArray {
    /// Parse an array dtype specification.
    ///
    /// :param str spec: A specification such as ``"[u8; 4]"``.
    /// :raises ValueError: if ``spec`` does not describe an array.
    #[new]
    #[pyo3(signature = (spec, /), text_signature = "(spec, /)")]
    fn py_new(spec: &str) -> PyResult<PyClassInitializer<Self>> {
        let dtype = parse_spec_cached(spec)?;
        if !matches!(dtype.repr, DtypeRepr::Array { .. }) {
            return Err(PyValueError::new_err(
                "DtypeArray requires an array dtype specification.",
            ));
        }
        Ok(PyClassInitializer::from(dtype).add_subclass(Self))
    }

    /// Construct an array dtype from its element dtype and count.
    ///
    /// :param Dtype | str dtype: The element dtype.
    /// :param int count: The number of elements, which may be zero.
    /// :return: An array dtype.
    #[classmethod]
    #[pyo3(signature = (dtype, count, /), text_signature = "(cls, dtype, count, /)")]
    fn from_params(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        count: i64,
    ) -> PyResult<Py<Self>> {
        if count < 0 {
            return Err(PyValueError::new_err(format!(
                "DtypeArray count can't be negative, but received {count}."
            )));
        }
        let dtype = extract_dtype(dtype)?;
        let repr = DtypeRepr::Array {
            dtype: Box::new(dtype.repr),
            count: count as usize,
        };
        Py::new(
            py,
            PyClassInitializer::from(Dtype::from_repr(repr)?).add_subclass(Self),
        )
    }

    /// The dtype repeated by this array.
    #[getter]
    fn dtype(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let DtypeRepr::Array { dtype, .. } = &slf.as_super().repr else {
            unreachable!()
        };
        Dtype::from_repr((**dtype).clone())?.into_python(py)
    }

    /// The number of values in this array.
    #[getter]
    fn count(slf: PyRef<'_, Self>) -> usize {
        let DtypeRepr::Array { count, .. } = &slf.as_super().repr else {
            unreachable!()
        };
        *count
    }
}

///     An ordered, non-empty tuple of dtypes.
///
///     Tuple values pack from any iterable with exactly one item per field and
///     unpack to Python tuples.
///
#[pyclass(module = "tibs", frozen, extends = Dtype, skip_from_py_object)]
pub struct DtypeTuple;

#[pymethods]
impl DtypeTuple {
    /// Parse a tuple dtype specification.
    ///
    /// :param str spec: A specification such as ``"(u8, u16_le)"``.
    /// :raises ValueError: if ``spec`` does not describe a tuple.
    #[new]
    #[pyo3(signature = (spec, /), text_signature = "(spec, /)")]
    fn py_new(spec: &str) -> PyResult<PyClassInitializer<Self>> {
        let dtype = parse_spec_cached(spec)?;
        if !matches!(dtype.repr, DtypeRepr::Tuple(_)) {
            return Err(PyValueError::new_err(
                "DtypeTuple requires a tuple dtype specification.",
            ));
        }
        Ok(PyClassInitializer::from(dtype).add_subclass(Self))
    }

    /// Construct a tuple dtype from its field dtypes.
    ///
    /// :param Iterable dtypes: A non-empty iterable of dtypes or dtype strings.
    /// :return: A tuple dtype.
    #[classmethod]
    #[pyo3(signature = (dtypes, /), text_signature = "(cls, dtypes, /)")]
    fn from_params(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        dtypes: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let reprs = dtypes
            .try_iter()?
            .map(|item| extract_dtype(&item?).map(|dtype| dtype.repr))
            .collect::<PyResult<Vec<_>>>()?;
        if reprs.is_empty() {
            return Err(PyValueError::new_err(
                "DtypeTuple must contain at least one dtype.",
            ));
        }
        Py::new(
            py,
            PyClassInitializer::from(Dtype::from_repr(DtypeRepr::Tuple(reprs))?).add_subclass(Self),
        )
    }

    /// The field dtypes as an immutable Python tuple.
    #[getter]
    fn dtypes(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let DtypeRepr::Tuple(dtypes) = &slf.as_super().repr else {
            unreachable!()
        };
        let objects = dtypes
            .iter()
            .cloned()
            .map(|repr| Dtype::from_repr(repr)?.into_python(py))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, objects)?.unbind())
    }
}
