use crate::core::BitCollection;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The default number of digits between separators when grouping is requested.
const DEFAULT_GROUP_SIZE: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Align {
    Left,
    Right,
    AfterPrefix,
    Center,
}

impl Align {
    fn from_char(c: char) -> Option<Self> {
        match c {
            '<' => Some(Align::Left),
            '>' => Some(Align::Right),
            '=' => Some(Align::AfterPrefix),
            '^' => Some(Align::Center),
            _ => None,
        }
    }
}

/// A parsed Python format mini-language specifier.
///
/// The grammar accepted is a subset of Python's, with the precision field reused
/// to set the digit group size:
///
/// `[[fill]align][sign]["#"]["0"][width][grouping]["." group_size][type]`
struct FormatSpec {
    fill: Option<char>,
    align: Option<Align>,
    sign: Option<char>,
    alternate: bool,
    zero_pad: bool,
    width: usize,
    grouping: Option<char>,
    group_size: Option<usize>,
    ty: Option<char>,
}

fn invalid_spec(spec: &str, type_name: &str) -> PyErr {
    PyValueError::new_err(format!(
        "Invalid format specifier '{spec}' for object of type '{type_name}'."
    ))
}

impl FormatSpec {
    fn parse(spec: &str, type_name: &str) -> PyResult<Self> {
        let chars: Vec<char> = spec.chars().collect();
        let mut i = 0;
        let mut parsed = FormatSpec {
            fill: None,
            align: None,
            sign: None,
            alternate: false,
            zero_pad: false,
            width: 0,
            grouping: None,
            group_size: None,
            ty: None,
        };

        // A fill character is only a fill if an alignment character follows it.
        if chars.len() >= 2
            && let Some(align) = Align::from_char(chars[1])
        {
            parsed.fill = Some(chars[0]);
            parsed.align = Some(align);
            i = 2;
        } else if !chars.is_empty()
            && let Some(align) = Align::from_char(chars[0])
        {
            parsed.align = Some(align);
            i = 1;
        }

        if let Some(&c) = chars.get(i)
            && matches!(c, '+' | '-' | ' ')
        {
            parsed.sign = Some(c);
            i += 1;
        }

        if chars.get(i) == Some(&'#') {
            parsed.alternate = true;
            i += 1;
        }

        if chars.get(i) == Some(&'0') {
            parsed.zero_pad = true;
            i += 1;
        }

        while let Some(&c) = chars.get(i)
            && c.is_ascii_digit()
        {
            parsed.width = parsed
                .width
                .checked_mul(10)
                .and_then(|w| w.checked_add(c.to_digit(10).unwrap() as usize))
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Width is too large in format specifier '{spec}'."
                    ))
                })?;
            i += 1;
        }

        if let Some(&c) = chars.get(i)
            && matches!(c, '_' | ',')
        {
            parsed.grouping = Some(c);
            i += 1;
        }

        if chars.get(i) == Some(&'.') {
            i += 1;
            let start = i;
            let mut size = 0usize;
            while let Some(&c) = chars.get(i)
                && c.is_ascii_digit()
            {
                size = size
                    .checked_mul(10)
                    .and_then(|s| s.checked_add(c.to_digit(10).unwrap() as usize))
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "Group size is too large in format specifier '{spec}'."
                        ))
                    })?;
                i += 1;
            }
            if i == start {
                return Err(invalid_spec(spec, type_name));
            }
            parsed.group_size = Some(size);
        }

        match chars.len() - i {
            0 => {}
            1 => {
                let c = chars[i];
                if !matches!(c, 'b' | 'o' | 'x' | 'X' | 'u' | 'i') {
                    return Err(PyValueError::new_err(format!(
                        "Unknown format code '{c}' for object of type '{type_name}'. \
                         Valid codes are 'b', 'o', 'x', 'X', 'u' and 'i'."
                    )));
                }
                parsed.ty = Some(c);
            }
            _ => return Err(invalid_spec(spec, type_name)),
        }

        Ok(parsed)
    }

    /// Apply fill, alignment and width to an already-built body.
    ///
    /// `prefix` stays attached to the front of the body except under `=` alignment,
    /// where the padding is inserted between the two. `default_align` is used when no
    /// alignment was given, and `zero_align` replaces it when the ``0`` option was
    /// used. The two differ because a bit representation pads like a number, keeping
    /// the padding after the prefix, whereas the plain string form pads like a string.
    fn pad(&self, body: &str, prefix: &str, default_align: Align, zero_align: Align) -> String {
        let total_len = prefix.chars().count() + body.chars().count();
        if total_len >= self.width {
            return format!("{prefix}{body}");
        }
        let padding_len = self.width - total_len;
        let fill = self.fill.unwrap_or(if self.zero_pad { '0' } else { ' ' });
        let align = self.align.unwrap_or(if self.zero_pad {
            zero_align
        } else {
            default_align
        });
        let (before, between, after) = match align {
            Align::Left => (0, 0, padding_len),
            Align::Right => (padding_len, 0, 0),
            Align::AfterPrefix => (0, padding_len, 0),
            Align::Center => (padding_len / 2, 0, padding_len - padding_len / 2),
        };
        let mut padded =
            String::with_capacity(prefix.len() + body.len() + padding_len * fill.len_utf8());
        padded.extend(std::iter::repeat_n(fill, before));
        padded.push_str(prefix);
        padded.extend(std::iter::repeat_n(fill, between));
        padded.push_str(body);
        padded.extend(std::iter::repeat_n(fill, after));
        padded
    }
}

/// Insert `sep` after every `size` digits, counting from the start of the sequence.
///
/// Unlike integer formatting this runs left to right, because a bit sequence starts
/// at bit zero rather than at a least significant digit. Any short group is therefore
/// the last one rather than the first.
fn group_from_left(digits: &str, sep: char, size: usize) -> String {
    debug_assert!(size > 0);
    if digits.len() <= size {
        return digits.to_string();
    }
    let separator_count = (digits.len() - 1) / size;
    let mut grouped = String::with_capacity(digits.len() + separator_count);
    for (index, c) in digits.chars().enumerate() {
        if index > 0 && index.is_multiple_of(size) {
            grouped.push(sep);
        }
        grouped.push(c);
    }
    grouped
}

/// Format a bit sequence using the Python format mini-language.
///
/// See the `Formatting` section of the user manual for the accepted grammar.
/// `type_name` only appears in error messages.
pub(crate) fn format_bit_collection(
    py: Python<'_>,
    bits: &impl BitCollection,
    spec: &str,
    type_name: &str,
) -> PyResult<String> {
    if spec.is_empty() {
        return Ok(bits.to_string());
    }
    let parsed = FormatSpec::parse(spec, type_name)?;

    let Some(ty) = parsed.ty else {
        if parsed.sign.is_some()
            || parsed.alternate
            || parsed.grouping.is_some()
            || parsed.group_size.is_some()
        {
            return Err(PyValueError::new_err(format!(
                "Format specifier '{spec}' needs a type code to go with it. \
                 Use 'b', 'o', 'x' or 'X' for the bit representation, or 'u' or 'i' \
                 for a numeric interpretation."
            )));
        }
        return Ok(parsed.pad(&bits.to_string(), "", Align::Left, Align::Left));
    };

    // The numeric interpretations really are numbers, so hand the rest of the spec
    // to the Python int. That keeps '+', ',', '_', 'n' and zero padding behaving
    // exactly as they do everywhere else, including their error messages.
    if matches!(ty, 'u' | 'i') {
        let value = if ty == 'u' {
            bits.to_u128(false)?.into_bound_py_any(py)?
        } else {
            bits.to_i128(false)?.into_bound_py_any(py)?
        };
        let mut int_spec = String::with_capacity(spec.len());
        int_spec.push_str(&spec[..spec.len() - ty.len_utf8()]);
        int_spec.push('d');
        return value.call_method1("__format__", (int_spec,))?.extract();
    }

    if let Some(sign) = parsed.sign {
        return Err(PyValueError::new_err(format!(
            "Sign '{sign}' is not allowed with the '{ty}' format type as a bit sequence \
             has no sign. Use the 'u' or 'i' type code for a numeric interpretation."
        )));
    }
    if parsed.grouping == Some(',') {
        return Err(PyValueError::new_err(format!(
            "Comma grouping is not allowed with the '{ty}' format type. Use '_' to group \
             digits, or the 'u' or 'i' type code for a numeric interpretation."
        )));
    }
    if parsed.group_size.is_some() && parsed.grouping.is_none() {
        return Err(PyValueError::new_err(format!(
            "A group size was given without a grouping character in format specifier \
             '{spec}'. Add '_' before the '.', for example '_.8{ty}'."
        )));
    }

    let mut digits = match ty {
        'b' => bits.to_binary(),
        'o' => bits.to_octal()?,
        'x' => bits.to_hexadecimal()?,
        'X' => {
            let mut hex = bits.to_hexadecimal()?;
            hex.make_ascii_uppercase();
            hex
        }
        _ => unreachable!("type code already validated"),
    };

    if let Some(sep) = parsed.grouping {
        let size = parsed.group_size.unwrap_or(DEFAULT_GROUP_SIZE);
        if size == 0 {
            return Err(PyValueError::new_err(format!(
                "The group size must be greater than zero in format specifier '{spec}'."
            )));
        }
        digits = group_from_left(&digits, sep, size);
    }

    let prefix = if parsed.alternate {
        match ty {
            'b' => "0b",
            'o' => "0o",
            'x' => "0x",
            'X' => "0X",
            _ => unreachable!("type code already validated"),
        }
    } else {
        ""
    };

    Ok(parsed.pad(&digits, prefix, Align::Right, Align::AfterPrefix))
}
