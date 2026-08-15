use crate::core::BitCollection;
use crate::enums::{ByteOrder, DtypeKind};
use crate::helpers::validate_slice;
use crate::iterator::ValuesIterator;
use crate::tibs_::{Tibs, bv_from_value, bv_from_values_iter, py_from_value, py_values_from_range};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyString, PyTuple, PyType};
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
        if length <= 0 {
            return Err(PyValueError::new_err(format!(
                "Dtype length must be greater than zero, but received {length}."
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
        match kind {
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

/// One field of a precomputed flat record layout: its kind/length/byte_order
/// plus its bit offset within one record. See [`RecordLayout`].
#[derive(Clone, Copy)]
pub(crate) struct RecordField {
    pub(crate) kind: DtypeKind,
    pub(crate) length: usize,
    pub(crate) byte_order: ByteOrder,
    pub(crate) bit_offset: usize,
}

/// A precomputed flat layout for a [`DtypeRepr::Tuple`] whose fields are all
/// [`DtypeRepr::Single`], or a [`DtypeRepr::Array`] whose element is
/// [`DtypeRepr::Single`] — the "record of scalar fields" shape (e.g.
/// `struct`'s `">hhl"`, or an MPEG-header-style fixed table). It lets pack and
/// unpack address each field directly instead of re-walking `DtypeRepr` and
/// recomputing offsets via `DtypeRepr::length()` on every single record.
///
/// Deeper nesting (tuple-of-tuple, array-of-tuple, ...) is never represented
/// here: `Dtype::record_layout` is `None` for those, and pack/unpack keep
/// walking `DtypeRepr` recursively exactly as before.
///
/// `Array` stores one element descriptor plus `count`, not `count` cloned
/// entries, so a large array stays cheap to represent: `Dtype("[u8; 1_000_000]")`
/// is as cheap to build as `Dtype("[u8; 4]")`. `Tuple` does flatten its fields
/// literally, which is safe because that count is bounded by the dtype spec's
/// own field arity, not by data volume.
pub(crate) enum RecordLayout {
    Tuple(Vec<RecordField>),
    Array { element: RecordField, count: usize },
}

fn build_record_layout(repr: &DtypeRepr) -> Option<RecordLayout> {
    match repr {
        DtypeRepr::Tuple(dtypes) => {
            let mut fields = Vec::with_capacity(dtypes.len());
            let mut bit_offset = 0;
            for dtype in dtypes {
                let DtypeRepr::Single(single) = dtype else {
                    return None;
                };
                fields.push(RecordField {
                    kind: single.kind,
                    length: single.length,
                    byte_order: single.byte_order,
                    bit_offset,
                });
                bit_offset += single.length;
            }
            Some(RecordLayout::Tuple(fields))
        }
        DtypeRepr::Array { dtype, count } => {
            let DtypeRepr::Single(single) = dtype.as_ref() else {
                return None;
            };
            Some(RecordLayout::Array {
                element: RecordField {
                    kind: single.kind,
                    length: single.length,
                    byte_order: single.byte_order,
                    bit_offset: 0,
                },
                count: *count,
            })
        }
        DtypeRepr::Single(_) => None,
    }
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
            return self.error("expected a positive array count");
        }
        let count = self.spec[start..self.pos].parse::<usize>().map_err(|_| {
            PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{}': array count is too large.",
                self.spec
            ))
        })?;
        if count == 0 {
            return self.error("array count must be greater than zero");
        }
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

    fn class_name(&self) -> &'static str {
        match self.repr {
            DtypeRepr::Single(_) => "DtypeSingle",
            DtypeRepr::Array { .. } => "DtypeArray",
            DtypeRepr::Tuple(_) => "DtypeTuple",
        }
    }
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
fn parse_spec_cached(spec: &str) -> PyResult<Dtype> {
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
    /// :raises ValueError: if the specification is invalid or does not have a positive fixed length.
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
    /// :param object value: A scalar or structured value matching this dtype.
    /// :return: The encoded bits.
    /// :raises ValueError: if a structured value has the wrong number of items.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("(u8, bool)").pack((15, True))
    ///     Tibs('0b000011111')
    #[pyo3(signature = (value, /), text_signature = "($self, value, /)")]
    fn pack(&self, value: &Bound<'_, PyAny>) -> PyResult<Tibs> {
        Ok(Tibs::from_bv(bv_from_value(self, value)?))
    }

    /// Encode and concatenate Python values as a :class:`Tibs`.
    ///
    /// :param Iterable iterable: Values matching this dtype.
    /// :return: The concatenated encoded bits.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("u8").pack_values([1, 2, 3])
    ///     Tibs('0x010203')
    #[pyo3(signature = (iterable, /), text_signature = "($self, iterable, /)")]
    fn pack_values(&self, py: Python<'_>, iterable: &Bound<'_, PyAny>) -> PyResult<Tibs> {
        Ok(Tibs::from_bv(bv_from_values_iter(py, self, iterable)?))
    }

    /// Decode one complete value from a bit sequence.
    ///
    /// :param object bits: Anything promotable to :class:`Tibs`.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to ``len(bits)``.
    /// :return: One scalar value, or a tuple for an array or tuple dtype.
    /// :raises ValueError: if the selected range is not exactly :attr:`length` bits.
    #[pyo3(signature = (bits, /, start = None, end = None), text_signature = "($self, bits, /, start=None, end=None)")]
    fn unpack(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<PyAny>> {
        let bits = bits.extract::<Tibs>()?;
        let (start, end) = validate_slice(bits.len(), start, end)?;
        py_from_value(py, self, &bits.get_slice_unchecked(start, end - start))
    }

    /// Decode a list of complete values from a bit sequence.
    ///
    /// :param object bits: Anything promotable to :class:`Tibs`.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to ``len(bits)``.
    /// :return: A list of decoded values.
    /// :raises ValueError: if the selected range is not a multiple of :attr:`length`.
    #[pyo3(signature = (bits, /, start = None, end = None), text_signature = "($self, bits, /, start=None, end=None)")]
    fn unpack_values(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let bits = bits.extract::<Tibs>()?;
        py_values_from_range(py, &bits, self, start, end)
    }

    /// Lazily decode complete values from a bit sequence.
    ///
    /// :param object bits: Anything promotable to :class:`Tibs`.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to ``len(bits)``.
    /// :return: An iterator of decoded values.
    /// :raises ValueError: if the selected range is not a multiple of :attr:`length`.
    #[pyo3(signature = (bits, /, start = None, end = None), text_signature = "($self, bits, /, start=None, end=None)")]
    fn unpack_values_iter(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<ValuesIterator>> {
        let bits = bits.extract::<Tibs>()?;
        let (start, end) = validate_slice(bits.len(), start, end)?;
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
    /// :param int | None length: The positive bit length. May be omitted for a
    ///     kind that fixes its own length, such as ``DtypeKind.OcpE2M1`` or
    ///     ``DtypeKind.Bool``.
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

///     A fixed positive number of repetitions of another dtype.
///
///     Array values pack from any iterable with exactly :attr:`count` items
///     and unpack to Python tuples.
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
    /// :param int count: The positive number of elements.
    /// :return: An array dtype.
    #[classmethod]
    #[pyo3(signature = (dtype, count, /), text_signature = "(cls, dtype, count, /)")]
    fn from_params(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        count: i64,
    ) -> PyResult<Py<Self>> {
        if count <= 0 {
            return Err(PyValueError::new_err(
                "DtypeArray count must be greater than zero.",
            ));
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
