use super::bits::BV;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Each byte of eight binary digits is `0x30` or `0x31`, so masking off the
/// low bit of every byte leaves the same value for any valid run.
const BIN_DIGIT_MASK: u64 = 0xfefe_fefe_fefe_fefe;
/// Each byte of eight octal digits is `0x30` to `0x37`, so the same check
/// masks off three bits per byte instead of one.
const OCT_DIGIT_MASK: u64 = 0xf8f8_f8f8_f8f8_f8f8;
const ASCII_DIGITS: u64 = 0x3030_3030_3030_3030;

const INVALID_HEX: u8 = 0xff;

/// The value of each hex digit, with `INVALID_HEX` for every other byte.
const HEX_VALUES: [u8; 256] = {
    let mut table = [INVALID_HEX; 256];
    let mut digit = 0u8;
    while digit < 10 {
        table[(b'0' + digit) as usize] = digit;
        digit += 1;
    }
    let mut letter = 0u8;
    while letter < 6 {
        table[(b'a' + letter) as usize] = 10 + letter;
        table[(b'A' + letter) as usize] = 10 + letter;
        letter += 1;
    }
    table
};

/// A partly filled byte being built up from digits.
///
/// Digits arrive with fewer bits than a byte and do not have to divide into
/// one, so bits are accumulated at the bottom of a word and pushed out a
/// whole byte at a time.
struct BitAccumulator {
    bytes: Vec<u8>,
    bits: u32,
    held: u32,
    len_bits: usize,
}

impl BitAccumulator {
    fn with_capacity(bytes: usize) -> Self {
        BitAccumulator {
            bytes: Vec::with_capacity(bytes),
            bits: 0,
            held: 0,
            len_bits: 0,
        }
    }

    /// Whether the next digit would start on a byte boundary, which is what
    /// the whole word fast paths need.
    #[inline]
    fn is_aligned(&self) -> bool {
        self.held == 0
    }

    /// Add `count` bits, given as the low bits of `digit`.
    #[inline]
    fn push_digit(&mut self, digit: u8, count: u32) {
        self.bits = (self.bits << count) | u32::from(digit);
        self.held += count;
        self.len_bits += count as usize;
        while self.held >= 8 {
            self.held -= 8;
            self.bytes.push((self.bits >> self.held) as u8);
        }
    }

    /// Add the whole bytes spelled out by each leading group of `DIGITS`
    /// digits that `decode` accepts, stopping at the first group it rejects.
    /// Returns how many digits were consumed.
    ///
    /// This is where the work is done for any input that is all digits, so
    /// the loop is kept to the decode and the store. It is only valid while
    /// aligned, which is what makes whole bytes come out of it.
    fn extend_groups<const DIGITS: usize, const BYTES: usize>(
        &mut self,
        source: &[u8],
        decode: impl Fn(&[u8; DIGITS]) -> Option<[u8; BYTES]>,
    ) -> usize {
        debug_assert!(self.is_aligned());
        let mut groups = 0;
        for group in source.chunks_exact(DIGITS) {
            let Some(bytes) = decode(group.try_into().unwrap()) else {
                break;
            };
            self.bytes.extend_from_slice(&bytes);
            groups += 1;
        }
        self.len_bits += groups * BYTES * 8;
        groups * DIGITS
    }

    fn into_bv(mut self) -> BV {
        if self.held != 0 {
            // The final digits are left aligned in a last, part used byte.
            self.bytes.push((self.bits << (8 - self.held)) as u8);
        }
        let mut bv = BV::from_vec(self.bytes);
        bv.truncate(self.len_bits);
        bv
    }
}

/// How many bytes to skip over a character that is not a digit, or the
/// character itself when it is not one that may be ignored.
///
/// `index` must be on a character boundary, which it always is because the
/// callers only ever step over whole characters.
fn skip_non_digit(s: &str, index: usize) -> Result<usize, char> {
    let byte = s.as_bytes()[index];
    if byte == b'_' || byte.is_ascii_whitespace() {
        return Ok(1);
    }
    let c = s[index..].chars().next().expect("index is a char boundary");
    if c.is_whitespace() {
        Ok(c.len_utf8())
    } else {
        Err(c)
    }
}

fn invalid_character(base: &str, source: &str, c: char) -> PyErr {
    PyValueError::new_err(format!(
        "Cannot convert from {base} '{source}': Invalid character '{c}'."
    ))
}

/// Pack eight binary digits into the byte they spell out.
#[inline]
fn decode_bin_group(group: &[u8; 8]) -> Option<[u8; 1]> {
    let word = u64::from_be_bytes(*group);
    if word & BIN_DIGIT_MASK != ASCII_DIGITS {
        return None;
    }
    // Every digit is one bit at the bottom of its byte, and the multiply
    // gathers all eight into the top byte of the product. No two of the
    // partial products land on the same bit, so nothing can carry.
    let packed = (word & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080);
    Some([(packed >> 56) as u8])
}

/// Pack eight octal digits into the three bytes they spell out.
#[inline]
fn decode_oct_group(group: &[u8; 8]) -> Option<[u8; 3]> {
    let word = u64::from_be_bytes(*group);
    if word & OCT_DIGIT_MASK != ASCII_DIGITS {
        return None;
    }
    // Fold the eight three bit digits together, halving the gap between them
    // each time, until all twenty four bits sit at the bottom of the word.
    let digits = word & 0x0707_0707_0707_0707;
    let pairs = (digits | (digits >> 5)) & 0x003f_003f_003f_003f;
    let quads = (pairs | (pairs >> 10)) & 0x0000_0fff_0000_0fff;
    let packed = ((quads | (quads >> 20)) & 0xff_ffff) as u32;
    let [_, high, middle, low] = packed.to_be_bytes();
    Some([high, middle, low])
}

/// Pack two hex digits into the byte they spell out.
#[inline]
fn decode_hex_group(group: &[u8; 2]) -> Option<[u8; 1]> {
    let high = HEX_VALUES[group[0] as usize];
    let low = HEX_VALUES[group[1] as usize];
    if (high | low) < 0x10 {
        Some([(high << 4) | low])
    } else {
        None
    }
}

pub(crate) fn bv_from_bin(binary_string: &str) -> PyResult<BV> {
    // Ignore any leading '0b' or '0B'
    let s = binary_string
        .strip_prefix("0b")
        .or_else(|| binary_string.strip_prefix("0B"))
        .unwrap_or(binary_string);
    let source = s.as_bytes();
    let mut out = BitAccumulator::with_capacity(source.len().div_ceil(8));
    let mut index = 0;
    while index < source.len() {
        // Eight digits are a whole byte, so runs of them go straight out.
        // Barring separators, that is the whole string in one pass.
        if out.is_aligned() {
            index += out.extend_groups(&source[index..], decode_bin_group);
            if index == source.len() {
                break;
            }
        }
        let digit = source[index] ^ b'0';
        if digit > 1 {
            index +=
                skip_non_digit(s, index).map_err(|c| invalid_character("bin", binary_string, c))?;
            continue;
        }
        out.push_digit(digit, 1);
        index += 1;
    }
    Ok(out.into_bv())
}

pub(crate) fn bv_from_oct(octal_string: &str) -> PyResult<BV> {
    // Ignore any leading '0o' or '0O'
    let s = octal_string
        .strip_prefix("0o")
        .or_else(|| octal_string.strip_prefix("0O"))
        .unwrap_or(octal_string);
    let source = s.as_bytes();
    let mut out = BitAccumulator::with_capacity((source.len() * 3).div_ceil(8));
    let mut index = 0;
    while index < source.len() {
        // Eight digits are three whole bytes, so runs of them go straight
        // out. Barring separators, that is the whole string in one pass.
        if out.is_aligned() {
            index += out.extend_groups(&source[index..], decode_oct_group);
            if index == source.len() {
                break;
            }
        }
        let digit = source[index] ^ b'0';
        if digit > 7 {
            index +=
                skip_non_digit(s, index).map_err(|c| invalid_character("oct", octal_string, c))?;
            continue;
        }
        out.push_digit(digit, 3);
        index += 1;
    }
    Ok(out.into_bv())
}

pub(crate) fn bv_from_hex(hex: &str) -> PyResult<BV> {
    // Ignore any leading '0x' or '0X'
    let s = hex
        .strip_prefix("0x")
        .or_else(|| hex.strip_prefix("0X"))
        .unwrap_or(hex);
    let source = s.as_bytes();
    let mut out = BitAccumulator::with_capacity(source.len().div_ceil(2));
    let mut index = 0;
    while index < source.len() {
        // A pair of digits is a whole byte, so runs of them go straight out.
        // Barring separators, that is the whole string in one pass.
        if out.is_aligned() {
            index += out.extend_groups(&source[index..], decode_hex_group);
            if index == source.len() {
                break;
            }
        }
        let digit = HEX_VALUES[source[index] as usize];
        if digit == INVALID_HEX {
            index += skip_non_digit(s, index).map_err(|c| invalid_character("hex", hex, c))?;
            continue;
        }
        out.push_digit(digit, 4);
        index += 1;
    }
    Ok(out.into_bv())
}

/// Whether every byte is a printable non-space ASCII character.
///
/// Folded rather than short circuited so that it vectorizes: the answer is
/// almost always yes, which means reading the whole string either way, and a
/// bailing out loop is several times slower over the length of a long one.
fn is_all_graphic(bytes: &[u8]) -> bool {
    bytes.iter().fold(true, |ok, b| ok & b.is_ascii_graphic())
}

fn string_literal_to_bv(s: &str) -> PyResult<BV> {
    match s.as_bytes() {
        [b'0', b'b' | b'B', ..] => bv_from_bin(s),
        [b'0', b'x' | b'X', ..] => bv_from_hex(s),
        [b'0', b'o' | b'O', ..] => bv_from_oct(s),
        _ => Err(PyValueError::new_err(format!(
            "Can't parse token '{s}'. Did you mean to prefix with '0x', '0b' or '0o'?"
        ))),
    }
}

pub(crate) fn str_to_bv(s: &str) -> PyResult<BV> {
    // Whitespace has to come out before the string is split into tokens, but
    // removing it means building a new string. Nearly every input is already
    // free of it, and a string of printable non-space characters can only be
    // ASCII, so one scan avoids the copy in the usual case.
    let stripped;
    let s = if is_all_graphic(s.as_bytes()) {
        s
    } else {
        stripped = s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        &stripped
    };
    // Nearly every string is a single token, so the first one is parsed into
    // the result and any others are appended to it.
    let mut tokens = s.split(',').filter(|token| !token.is_empty());
    let Some(first) = tokens.next() else {
        return Ok(BV::new());
    };
    let mut result = string_literal_to_bv(first)?;
    for token in tokens {
        result.extend_from_bitslice(&string_literal_to_bv(token)?);
    }
    Ok(result)
}
