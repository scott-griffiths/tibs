//! The pretty-printer behind `pp` on `Tibs` and `Mutibs`.
//!
//! The bits are split into groups whose size is the length of the formats'
//! dtypes, and each line shows the same groups in every format side by side,
//! with the offset of the line in the first column. A [`Printer`] makes the
//! output one line at a time, so `pp` can write a value of any size without
//! building all of it first, and can check for Ctrl-C between lines.

use crate::core::BitCollection;
use crate::dtype::{
    Dtype, DtypeRepr, SingleDtype, dtype_from_kind_and_length, extract_dtype, kind_from_prefix,
    parse_spec_cached, prefix_for_kind, validate_lsb0_length,
};
use crate::enums::{BitOrder, DtypeKind};
use crate::tibs_::{Tibs, py_from_value_parts};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyString, PyTuple};

// Cyan and magenta are readable on both dark and light backgrounds in the
// common terminal palettes, and distinct from each other in all of them. Blue
// is avoided as it's the darkest colour in most palettes, and green because
// several make it barely different from cyan. With more than two formats the
// two colours alternate. Offsets are faint rather than coloured, so that they
// recede and the data stands out.
const FORMAT_COLORS: [&str; 2] = ["\x1b[36m", "\x1b[35m"];
const FAINT: &str = "\x1b[2m";
const OFF: &str = "\x1b[0m";

/// Put between the formats on each line, and between their names in the header.
const FORMAT_SEP: &str = " : ";
/// Starts each line that describes output rather than showing it.
const NOTE_MARKER: &str = "...";
/// Output is passed to the stream in pieces of about this many bytes, so a
/// large `Tibs` isn't rendered into one enormous string first.
const FLUSH_BYTES: usize = 1 << 16;
/// The line width when none is given and the output isn't a terminal.
const DEFAULT_WIDTH: usize = 120;
/// Ctrl-C is checked for every this many lines. The stream can't be relied on
/// to do it: a file or terminal checks when written to, but `io.StringIO`
/// doesn't, and squeezed lines write nothing at all.
const SIGNAL_CHECK_LINES: usize = 256;

/// The `show_offset` argument.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Offsets {
    /// Bytes for a whole number of bytes, otherwise bits.
    Auto,
    Hidden,
    Bits,
    Bytes,
}

impl Offsets {
    pub(crate) fn extract(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(obj) = obj else {
            return Ok(Offsets::Auto);
        };
        if let Ok(spec) = obj.cast::<PyString>() {
            return match spec.to_str()? {
                "bits" => Ok(Offsets::Bits),
                "bytes" => Ok(Offsets::Bytes),
                other => Err(PyValueError::new_err(format!(
                    "show_offset must be True, False, 'bits' or 'bytes', but received '{other}'."
                ))),
            };
        }
        if obj.is_instance_of::<PyBool>() {
            return Ok(if obj.is_truthy()? {
                Offsets::Auto
            } else {
                Offsets::Hidden
            });
        }
        Err(PyTypeError::new_err(format!(
            "show_offset must be True, False, 'bits' or 'bytes', but received an object of type '{}'.",
            obj.get_type().name()?
        )))
    }
}

/// Everything `pp` takes apart from the bits and where the output goes.
pub(crate) struct PpOptions<'py> {
    pub(crate) dtypes: Bound<'py, PyTuple>,
    pub(crate) sep: String,
    pub(crate) offsets: Offsets,
    pub(crate) groups: Option<isize>,
    pub(crate) squeeze: bool,
    pub(crate) max_lines: Option<isize>,
    pub(crate) bit_order: BitOrder,
}

/// One format as given, before the group length is known.
enum Token {
    /// A complete dtype, such as `'u8'` or `'(u4, u4)'`.
    Dtype(Dtype),
    /// A kind given without its length, such as `'hex'`, `'u'` or
    /// `DtypeKind.Hex`.
    Bare(DtypeKind, String),
}

impl Token {
    fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(spec) = obj.cast::<PyString>() {
            return Self::parse(spec.to_str()?);
        }
        if let Ok(kind) = obj.extract::<DtypeKind>()
            && let Some(prefix) = prefix_for_kind(kind)
        {
            return Ok(Token::Bare(kind, prefix.to_string()));
        }
        extract_dtype(obj)
            .map(Token::Dtype)
            .map_err(|_| match obj.get_type().name() {
                Ok(type_name) => PyTypeError::new_err(format!(
                    "A format must be a Dtype, a DtypeKind or a dtype string, but received an \
                     object of type '{type_name}'."
                )),
                Err(e) => e,
            })
    }

    fn parse(spec: &str) -> PyResult<Self> {
        let spec = spec.trim();
        let error = match parse_spec_cached(spec) {
            Ok(dtype) => return Ok(Token::Dtype(dtype)),
            Err(error) => error,
        };
        if let Some(kind) = kind_from_prefix(spec) {
            return Ok(Token::Bare(kind, spec.to_string()));
        }
        // bitstring's pp takes all its formats in one string, as in 'u8, hex'.
        let parts = split_top_level(spec);
        if parts.len() > 1 {
            let quoted = parts
                .iter()
                .map(|part| format!("'{}'", part.trim()))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(PyValueError::new_err(format!(
                "Cannot use '{spec}' as a format. To show several formats, pass them as \
                 separate arguments: pp({quoted})."
            )));
        }
        Err(error)
    }

    fn label(&self) -> String {
        match self {
            Token::Dtype(dtype) => dtype.spec(),
            Token::Bare(_, name) => name.clone(),
        }
    }

    fn length(&self) -> Option<usize> {
        match self {
            Token::Dtype(dtype) => Some(dtype.length),
            Token::Bare(..) => None,
        }
    }
}

/// Split a string at its top-level commas, leaving the commas inside a tuple
/// or array dtype alone.
fn split_top_level(spec: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in spec.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&spec[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&spec[start..]);
    parts
}

fn tokens_from_args(dtypes: &Bound<'_, PyTuple>, length: usize) -> PyResult<Vec<Token>> {
    if dtypes.is_empty() {
        let mut tokens = vec![Token::Bare(DtypeKind::Bin, "bin".to_string())];
        if length >= 8 && length.is_multiple_of(8) {
            tokens.push(Token::Bare(DtypeKind::Hex, "hex".to_string()));
        }
        return Ok(tokens);
    }
    dtypes.iter().map(|dtype| Token::from_py(&dtype)).collect()
}

/// Bits per character for the kinds that print as text, which are the only
/// ones with a default group length.
fn bits_per_char(kind: DtypeKind) -> Option<usize> {
    match kind {
        DtypeKind::Bin => Some(1),
        DtypeKind::Oct => Some(3),
        DtypeKind::Hex => Some(4),
        DtypeKind::Bytes => Some(8),
        _ => None,
    }
}

fn text_kind(dtype: &Dtype) -> Option<DtypeKind> {
    dtype
        .single()
        .map(|single| single.kind)
        .filter(|&kind| bits_per_char(kind).is_some())
}

/// The group length for formats that were all given without one: the
/// shortest that is whole characters in every format and at least as long as
/// each format's own default.
fn default_group_length(tokens: &[Token]) -> PyResult<usize> {
    let mut step = 1;
    let mut minimum: usize = 0;
    for token in tokens {
        let default = match token {
            Token::Bare(kind, _) => bits_per_char(*kind).map(|per_char| (per_char, *kind)),
            Token::Dtype(_) => None,
        };
        let Some((per_char, kind)) = default else {
            let labels = tokens
                .iter()
                .map(|t| format!("'{}'", t.label()))
                .collect::<Vec<_>>();
            return Err(PyValueError::new_err(if tokens.len() == 1 {
                format!(
                    "pp() has no default group length for {}, so a length has to be given, \
                     for example '{}32'.",
                    labels[0],
                    token.label()
                )
            } else {
                let (last, rest) = labels.split_last().expect("more than one format");
                format!(
                    "pp() has no default group length for {} and {last}, so a length has to be \
                     given to at least one of them.",
                    rest.join(", ")
                )
            }));
        };
        step = lcm(step, per_char);
        minimum = minimum.max(match kind {
            DtypeKind::Oct => 12,
            DtypeKind::Bytes => 32,
            _ => 8,
        });
    }
    Ok(minimum.div_ceil(step) * step)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

fn decimal_digits(n: usize) -> usize {
    n.checked_ilog10().unwrap_or(0) as usize + 1
}

fn hex_digits(n: usize) -> usize {
    (n.checked_ilog2().unwrap_or(0) / 4) as usize + 1
}

/// The largest power of two that is no more than `n`, which must be positive.
fn power_of_two_at_most(n: usize) -> usize {
    1 << n.ilog2()
}

/// The number of decimal digits in `2**n`, which for `n > 0` is also the
/// number in `2**n - 1`, as no power of two is a power of ten.
fn digits_in_power_of_two(n: usize) -> usize {
    (n as f64 * std::f64::consts::LOG10_2).floor() as usize + 1
}

/// The most characters one value of `single` can print as.
fn single_width(single: &SingleDtype) -> usize {
    let n = single.length;
    match single.kind {
        DtypeKind::Uint => digits_in_power_of_two(n),
        DtypeKind::Int => digits_in_power_of_two(n - 1) + 1,
        DtypeKind::Bin => n,
        DtypeKind::Oct => n.div_ceil(3),
        DtypeKind::Hex => n.div_ceil(4),
        DtypeKind::Bytes => n.div_ceil(8),
        DtypeKind::Bits => match n {
            0 => 0,
            _ if n.is_multiple_of(4) => 2 + n / 4,
            _ => 2 + n,
        },
        DtypeKind::Bool => 1,
        // The widest values Python prints for each float format. The narrow
        // formats were found by printing every value they have.
        DtypeKind::Float if n == 64 => 24,
        DtypeKind::Float | DtypeKind::BFloat => 23,
        DtypeKind::OcpE8M0 => 22,
        DtypeKind::Binary8P3 | DtypeKind::OcpE5M2Saturate | DtypeKind::OcpE5M2Overflow => 19,
        DtypeKind::Binary8P4 => 13,
        DtypeKind::OcpE4M3Saturate | DtypeKind::OcpE4M3Overflow => 12,
        DtypeKind::OcpInt8 => 9,
        DtypeKind::OcpE3M2 => 7,
        DtypeKind::OcpE2M3 => 6,
        DtypeKind::OcpE2M1 => 4,
    }
}

/// Characters taken by `items` values in a tuple, written as Python would.
fn tuple_width(item_widths: impl Iterator<Item = usize>) -> usize {
    let (count, total) = item_widths.fold((0, 0), |(count, total), w| (count + 1, total + w));
    match count {
        0 => 2,
        1 => total + 3,
        _ => total + 2 + 2 * (count - 1),
    }
}

fn repr_width(repr: &DtypeRepr) -> usize {
    match repr {
        DtypeRepr::Single(single) => single_width(single),
        DtypeRepr::Array { dtype, count } => {
            let w = repr_width(dtype);
            tuple_width(std::iter::repeat_n(w, *count))
        }
        DtypeRepr::Tuple(dtypes) => tuple_width(dtypes.iter().map(repr_width)),
    }
}

fn is_left_aligned(repr: &DtypeRepr) -> bool {
    matches!(
        repr,
        DtypeRepr::Single(SingleDtype {
            kind: DtypeKind::Bin
                | DtypeKind::Oct
                | DtypeKind::Hex
                | DtypeKind::Bytes
                | DtypeKind::Bits,
            ..
        })
    )
}

/// Append `bits` as bytes, one character per byte, with a `.` for each byte
/// that isn't printable ASCII. The hex alongside gives the exact values.
fn push_printable_bytes(out: &mut String, bits: &Tibs) {
    for byte in bits.to_padded_byte_data() {
        out.push(if (0x20..0x7f).contains(&byte) {
            char::from(byte)
        } else {
            '.'
        });
    }
}

/// Append the text for one scalar, without padding. `bits` can be shorter
/// than `single.length` for the text kinds, in the final group.
fn push_single(
    py: Python<'_>,
    out: &mut String,
    single: &SingleDtype,
    bits: &Tibs,
) -> PyResult<()> {
    match single.kind {
        DtypeKind::Bin => out.push_str(&bits.to_binary()),
        DtypeKind::Oct => out.push_str(&bits.to_octal()?),
        DtypeKind::Hex => out.push_str(&bits.to_hexadecimal()?),
        DtypeKind::Bytes => push_printable_bytes(out, bits),
        DtypeKind::Bits => out.push_str(&bits.to_string()),
        DtypeKind::Bool => out.push(if bits.as_bitslice()[0] { '1' } else { '0' }),
        kind => {
            let value = py_from_value_parts(py, kind, single.length, single.byte_order, bits)?;
            out.push_str(value.bind(py).str()?.to_str()?);
        }
    }
    Ok(())
}

fn push_padded(out: &mut String, text: &str, width: usize, left: bool) {
    let padding = width.saturating_sub(text.chars().count());
    if left {
        out.push_str(text);
        out.extend(std::iter::repeat_n(' ', padding));
    } else {
        out.extend(std::iter::repeat_n(' ', padding));
        out.push_str(text);
    }
}

/// Append one value of `repr`, padded to its full width.
fn push_value(py: Python<'_>, out: &mut String, repr: &DtypeRepr, bits: &Tibs) -> PyResult<()> {
    let items: Vec<&DtypeRepr> = match repr {
        DtypeRepr::Single(single) => {
            let mut text = String::new();
            push_single(py, &mut text, single, bits)?;
            push_padded(out, &text, single_width(single), is_left_aligned(repr));
            return Ok(());
        }
        DtypeRepr::Array { dtype, count } => std::iter::repeat_n(dtype.as_ref(), *count).collect(),
        DtypeRepr::Tuple(dtypes) => dtypes.iter().collect(),
    };
    out.push('(');
    let mut pos = 0;
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let length = item.length()?;
        push_value(py, out, item, &bits.get_slice_unchecked(pos, length))?;
        pos += length;
    }
    if items.len() == 1 {
        out.push(',');
    }
    out.push(')');
    Ok(())
}

fn is_terminal(stream: &Bound<'_, PyAny>) -> bool {
    stream
        .call_method0("isatty")
        .and_then(|tty| tty.is_truthy())
        .unwrap_or(false)
}

/// Whether `pp` should colour its output.
fn use_color(color: Option<bool>, terminal: bool) -> bool {
    if let Some(color) = color {
        return color;
    }
    let no_color = std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty());
    terminal && !no_color
}

/// The line width when none is given: the terminal's, when writing to one.
/// As for `shutil.get_terminal_size`, the `COLUMNS` environment variable
/// comes first. The last column is left free, as some terminals start a new
/// line as soon as it's written to.
fn default_width(py: Python<'_>, stream: &Bound<'_, PyAny>, terminal: bool) -> usize {
    if !terminal {
        return DEFAULT_WIDTH;
    }
    let from_environment = std::env::var("COLUMNS")
        .ok()
        .and_then(|columns| columns.trim().parse::<usize>().ok());
    let columns = from_environment.filter(|&c| c > 0).or_else(|| {
        let fd = stream.call_method0("fileno").ok()?.extract::<i32>().ok()?;
        let size = py
            .import("os")
            .ok()?
            .call_method1("get_terminal_size", (fd,))
            .ok()?;
        size.getattr("columns").ok()?.extract::<usize>().ok()
    });
    columns.filter(|&c| c > 1).map_or(DEFAULT_WIDTH, |c| c - 1)
}

/// `n` with commas between each group of three digits.
fn with_commas(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `n` followed by `unit`, made plural unless `n` is 1.
fn plural(n: usize, unit: &str) -> String {
    let s = if n == 1 { "" } else { "s" };
    format!("{} {unit}{s}", with_commas(n))
}

/// Check that an optional count is at least 1.
fn positive(value: Option<isize>, name: &str) -> PyResult<Option<usize>> {
    match value {
        Some(n) if n < 1 => Err(PyValueError::new_err(format!(
            "{name} must be at least 1, but received {n}."
        ))),
        n => Ok(n.map(|n| n as usize)),
    }
}

struct Format {
    dtype: Dtype,
    color: &'static str,
    /// Characters in one full group.
    group_width: usize,
    /// The width of this format on the first line, which later lines are
    /// padded to so that the formats after it stay in their columns.
    column_width: Option<usize>,
}

/// How the offset column is shown, once `Offsets::Auto` has been decided.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OffsetColumn {
    Hidden,
    /// Decimal bit offsets, right-aligned in this many digits.
    Bits(usize),
    /// Hex byte offsets, zero-padded to this many digits.
    Bytes(usize),
}

impl OffsetColumn {
    fn width(self) -> usize {
        match self {
            OffsetColumn::Hidden => 0,
            OffsetColumn::Bits(digits) | OffsetColumn::Bytes(digits) => digits + 2,
        }
    }

    /// The text for a line starting at bit `position`, which is blank if the
    /// position isn't a whole number of bytes.
    fn text(self, position: usize) -> String {
        match self {
            OffsetColumn::Hidden => String::new(),
            OffsetColumn::Bits(digits) => format!("{position:>digits$}: "),
            OffsetColumn::Bytes(digits) if position.is_multiple_of(8) => {
                format!("{:0digits$x}: ", position / 8)
            }
            OffsetColumn::Bytes(digits) => " ".repeat(digits + 2),
        }
    }
}

enum Stage {
    Header,
    Body,
    Leftover,
    Done,
}

/// Makes the lines of `pp` output, one at a time.
struct Printer {
    bits: Tibs,
    header: String,
    formats: Vec<Format>,
    group_length: usize,
    /// The bits shown in the formats. Any after these are shown on their own
    /// line at the end.
    data_length: usize,
    line_bits: usize,
    offsets: OffsetColumn,
    sep: String,
    squeeze: bool,
    max_lines: Option<usize>,
    /// With `Lsb0`, each line is rearranged for each format by
    /// [`Dtype::lsb0_to_msb0`] before it is split into groups, so every group
    /// shows what `to_values` with the same `bit_order` would give.
    bit_order: BitOrder,
    faint: &'static str,
    off: &'static str,
    stage: Stage,
    line_start: usize,
    lines_written: usize,
    lines_visited: usize,
}

impl Printer {
    fn new(
        py: Python<'_>,
        bits: Tibs,
        class_name: &str,
        options: PpOptions<'_>,
        width: usize,
        color: bool,
    ) -> PyResult<Self> {
        let length = bits.len();
        let groups = positive(options.groups, "groups")?;
        let max_lines = positive(options.max_lines, "max_lines")?;
        let tokens = tokens_from_args(&options.dtypes, length)?;
        let lsb0 = options.bit_order == BitOrder::Lsb0;
        if lsb0 {
            validate_lsb0_length(length)?;
        }

        let mut explicit = tokens.iter().filter_map(|t| t.length().map(|n| (t, n)));
        let first_explicit = explicit.next();
        if let Some((first, a)) = first_explicit
            && let Some((other, b)) = explicit.find(|&(_, b)| b != a)
        {
            return Err(PyValueError::new_err(format!(
                "The formats '{}' and '{}' can't be shown together, as their lengths of {a} and \
                 {b} bits differ.",
                first.label(),
                other.label()
            )));
        }
        let has_explicit_length = first_explicit.is_some();
        let group_length = match first_explicit {
            Some((token, 0)) => {
                let hint = match token {
                    Token::Dtype(dtype) if text_kind(dtype).is_some() => {
                        " To show the bits without gaps between the groups, use sep=''."
                    }
                    _ => "",
                };
                return Err(PyValueError::new_err(format!(
                    "The format '{}' has a length of 0, so there's nothing to show.{hint}",
                    token.label()
                )));
            }
            Some((_, length)) => length,
            None => default_group_length(&tokens)?,
        };

        let paint = |code: &'static str| if color { code } else { "" };
        let mut formats = Vec::with_capacity(tokens.len());
        for (index, token) in tokens.iter().enumerate() {
            let dtype = match token {
                Token::Dtype(dtype) => dtype.clone(),
                Token::Bare(kind, _) => {
                    dtype_from_kind_and_length(*kind, group_length).map_err(|e| {
                        PyValueError::new_err(format!(
                            "The format '{}' can't be used for groups of {group_length} bits. {}",
                            token.label(),
                            e.value(py)
                        ))
                    })?
                }
            };
            formats.push(Format {
                group_width: repr_width(&dtype.repr),
                dtype,
                color: paint(FORMAT_COLORS[index % FORMAT_COLORS.len()]),
                column_width: None,
            });
        }

        // A final group that is short is still shown if it's whole characters
        // in every format, unless a length was asked for. Bits after that are
        // left over.
        let text_step = formats
            .iter()
            .map(|f| text_kind(&f.dtype).and_then(bits_per_char))
            .try_fold(1, |step, per_char| per_char.map(|p| lcm(step, p)));
        let leftover = match text_step {
            Some(step) if !has_explicit_length => length % group_length % step,
            _ => length % group_length,
        };

        // Bytes need each line to start on a byte boundary, which a number of
        // groups that is a multiple of this always does.
        let byte_step = 8 / gcd(group_length, 8);
        // LSB0 labels map onto the stored bits a byte at a time, so a line that
        // started part way into a byte would need bits from outside its own
        // span, and two lines with the same stored bits could show different
        // values, which would make squeezing wrong.
        if let Some(groups) = groups
            && !groups.is_multiple_of(byte_step)
            && (lsb0 || options.offsets == Offsets::Bytes)
        {
            let reason = if lsb0 {
                "BitOrder.Lsb0 needs"
            } else {
                "Offsets in bytes need"
            };
            return Err(PyValueError::new_err(format!(
                "{reason} a whole number of bytes on each line, but {groups} groups of \
                 {group_length} bits is {} bits.",
                groups.saturating_mul(group_length)
            )));
        }
        let use_bytes = match options.offsets {
            Offsets::Hidden | Offsets::Bits => false,
            Offsets::Bytes => true,
            Offsets::Auto => {
                length > 0
                    && length.is_multiple_of(8)
                    && groups.is_none_or(|groups| groups.is_multiple_of(byte_step))
            }
        };
        let last_offset = length.saturating_sub(1);
        let offsets = match options.offsets {
            Offsets::Hidden => OffsetColumn::Hidden,
            _ if use_bytes => OffsetColumn::Bytes(hex_digits(last_offset / 8).next_multiple_of(2)),
            _ => OffsetColumn::Bits(decimal_digits(last_offset)),
        };

        let groups_per_line = match groups {
            Some(groups) => groups,
            None => {
                let widths: usize = formats.iter().map(|f| f.group_width).sum();
                let fixed = offsets.width() + widths + FORMAT_SEP.len() * (formats.len() - 1);
                // Every group after the first adds its characters plus a
                // separator in each format.
                let per_group = widths + options.sep.chars().count() * formats.len();
                let fit = 1 + width.saturating_sub(fixed) / per_group.max(1);
                // A power of two keeps the offsets round numbers.
                let groups = power_of_two_at_most(fit);
                if use_bytes || lsb0 {
                    groups.max(byte_step)
                } else {
                    groups
                }
            }
        };

        let labels = formats
            .iter()
            .zip(&tokens)
            .map(|(f, token)| format!("{}{}{}", f.color, token.label(), paint(OFF)))
            .collect::<Vec<_>>()
            .join(FORMAT_SEP);
        let order = if lsb0 { "BitOrder.Lsb0, " } else { "" };
        Ok(Self {
            header: format!("{class_name}, {}, {order}{labels}", plural(length, "bit")),
            bits,
            formats,
            group_length,
            data_length: length - leftover,
            line_bits: groups_per_line.saturating_mul(group_length),
            offsets,
            sep: options.sep,
            squeeze: options.squeeze,
            max_lines,
            bit_order: options.bit_order,
            faint: paint(FAINT),
            off: paint(OFF),
            stage: Stage::Header,
            line_start: 0,
            lines_written: 0,
            lines_visited: 0,
        })
    }

    /// The next line, without its newline, or `None` when there are no more.
    fn next_line(&mut self, py: Python<'_>) -> PyResult<Option<String>> {
        loop {
            match self.stage {
                Stage::Header => {
                    self.stage = Stage::Body;
                    return Ok(Some(std::mem::take(&mut self.header)));
                }
                Stage::Body if self.line_start >= self.data_length => {
                    self.stage = Stage::Leftover;
                }
                Stage::Body => {
                    self.lines_visited += 1;
                    if self.lines_visited.is_multiple_of(SIGNAL_CHECK_LINES) {
                        py.check_signals()?;
                    }
                    if self.max_lines == Some(self.lines_written) {
                        return Ok(Some(self.truncate()));
                    }
                    // A run of one is shown as itself, as describing it would
                    // take a line anyway.
                    let run = self.repeat_run(py)?;
                    if run > 1 {
                        let line = self.squeezed_line(run);
                        self.line_start += run * self.line_bits;
                        return Ok(Some(self.counted(line)));
                    }
                    let line_length = self.line_bits.min(self.data_length - self.line_start);
                    let line = self.render_line(py, line_length)?;
                    self.line_start += line_length;
                    return Ok(Some(self.counted(line)));
                }
                Stage::Leftover => {
                    self.stage = Stage::Done;
                    let leftover = self.bits.len() - self.data_length;
                    if leftover == 0 {
                        continue;
                    }
                    if self.max_lines == Some(self.lines_written) {
                        return Ok(Some(self.truncate()));
                    }
                    let bits = match self.bit_order {
                        BitOrder::Msb0 => self.bits.get_slice_unchecked(self.data_length, leftover),
                        // Shorter than any value, so the dtype only decides
                        // that the leftover is one field, in significance order.
                        BitOrder::Lsb0 => self.formats[0].dtype.lsb0_to_msb0(
                            self.bits.as_bitslice(),
                            self.data_length,
                            self.bits.len(),
                        ),
                    };
                    return Ok(Some(format!(
                        "{}{}{}{}",
                        self.faint,
                        self.offsets.text(self.data_length),
                        bits.to_string(),
                        self.off
                    )));
                }
                Stage::Done => return Ok(None),
            }
        }
    }

    /// Whether the line at `start` repeats the line before it. The last line
    /// never counts, so it's always shown and it's clear where the data ends.
    fn is_repeat(&self, start: usize) -> bool {
        let n = self.line_bits;
        start > 0
            && start + n < self.data_length
            && self
                .bits
                .get_slice_unchecked(start, n)
                .bits_equal(&self.bits.get_slice_unchecked(start - n, n))
    }

    /// How many lines from `line_start` on repeat the line before them.
    fn repeat_run(&mut self, py: Python<'_>) -> PyResult<usize> {
        if !self.squeeze {
            return Ok(0);
        }
        let mut run = 0;
        while self.is_repeat(self.line_start + run * self.line_bits) {
            run += 1;
            self.lines_visited += 1;
            if self.lines_visited.is_multiple_of(SIGNAL_CHECK_LINES) {
                py.check_signals()?;
            }
        }
        Ok(run)
    }

    /// The line that stands in for `run` hidden lines, indented to where the
    /// data starts.
    fn squeezed_line(&self, run: usize) -> String {
        let (n, unit) = self.size(run * self.line_bits);
        self.note(&format!(
            "x{} identical lines, {}",
            with_commas(run),
            plural(n, unit)
        ))
    }

    /// A line that describes output rather than showing it. It starts with
    /// `...` in the offset column, so that the text lines up with the data
    /// where there is room.
    fn note(&self, text: &str) -> String {
        let column = self.offsets.width().max(NOTE_MARKER.len() + 1);
        format!("{}{NOTE_MARKER:<column$}{text}{}", self.faint, self.off)
    }

    /// `bits` in the unit the offsets use, or in bytes when there are no
    /// offsets and it's a whole number of them.
    fn size(&self, bits: usize) -> (usize, &'static str) {
        match self.offsets {
            OffsetColumn::Bits(_) => (bits, "bit"),
            _ if bits.is_multiple_of(8) => (bits / 8, "byte"),
            _ => (bits, "bit"),
        }
    }

    /// Count `line` against `max_lines`, which the caller has checked.
    fn counted(&mut self, line: String) -> String {
        self.lines_written += 1;
        line
    }

    /// End the output at `line_start`, with a line saying why and how much
    /// isn't shown.
    fn truncate(&mut self) -> String {
        self.stage = Stage::Done;
        let max_lines = self.max_lines.expect("only truncated when there's a limit");
        let (n, unit) = self.size(self.bits.len() - self.line_start);
        let s = if n == 1 { "" } else { "s" };
        self.note(&format!(
            "max_lines={max_lines}, {} {unit}{s} not shown",
            with_commas(n)
        ))
    }

    fn render_line(&mut self, py: Python<'_>, line_length: usize) -> PyResult<String> {
        let stored = self.bits.get_slice_unchecked(self.line_start, line_length);
        let mut out = String::new();
        if self.offsets != OffsetColumn::Hidden {
            out.push_str(self.faint);
            out.push_str(&self.offsets.text(self.line_start));
            out.push_str(self.off);
        }
        let count = self.formats.len();
        for (index, format) in self.formats.iter_mut().enumerate() {
            // A short final group is passed on by `lsb0_to_msb0` as one field,
            // which is how `to_value` would read it.
            let rearranged;
            let line = match self.bit_order {
                BitOrder::Msb0 => &stored,
                BitOrder::Lsb0 => {
                    rearranged = format.dtype.lsb0_to_msb0(
                        self.bits.as_bitslice(),
                        self.line_start,
                        self.line_start + line_length,
                    );
                    &rearranged
                }
            };
            let mut text = String::new();
            let mut group_start = 0;
            while group_start < line_length {
                if group_start > 0 {
                    text.push_str(&self.sep);
                }
                let n = self.group_length.min(line_length - group_start);
                push_value(
                    py,
                    &mut text,
                    &format.dtype.repr,
                    &line.get_slice_unchecked(group_start, n),
                )?;
                group_start += n;
            }
            if index > 0 {
                out.push_str(FORMAT_SEP);
            }
            let is_last_format = index + 1 == count;
            // A short final group is padded to a full one by push_value,
            // which would otherwise leave spaces at the end of the line.
            let shown = if is_last_format {
                text.trim_end_matches(' ')
            } else {
                &text
            };
            out.push_str(format.color);
            out.push_str(shown);
            out.push_str(self.off);
            if !is_last_format {
                let used = text.chars().count();
                let column = *format.column_width.get_or_insert(used);
                out.extend(std::iter::repeat_n(' ', column.saturating_sub(used)));
            }
        }
        Ok(out)
    }
}

/// Write every line of `printer` to `stream`, or to `sys.stdout`.
pub(crate) fn pp(
    py: Python<'_>,
    bits: Tibs,
    class_name: &str,
    options: PpOptions<'_>,
    width: Option<isize>,
    stream: Option<Bound<'_, PyAny>>,
    color: Option<bool>,
) -> PyResult<()> {
    let stream = match stream {
        Some(stream) => stream,
        None => py.import("sys")?.getattr("stdout")?,
    };
    let terminal = is_terminal(&stream);
    let width = match width {
        Some(width) => usize::try_from(width).unwrap_or(0),
        None => default_width(py, &stream, terminal),
    };
    let color = use_color(color, terminal);
    let mut printer = Printer::new(py, bits, class_name, options, width, color)?;
    let mut buffer = String::new();
    while let Some(line) = printer.next_line(py)? {
        buffer.push_str(&line);
        buffer.push('\n');
        if buffer.len() >= FLUSH_BYTES {
            stream.call_method1("write", (buffer.as_str(),))?;
            buffer.clear();
        }
    }
    if !buffer.is_empty() {
        stream.call_method1("write", (buffer.as_str(),))?;
    }
    Ok(())
}
