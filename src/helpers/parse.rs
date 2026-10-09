use super::bits::{BV, BitAccumulator};
use super::bitwise::BitConcat;
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

/// Add the whole bytes spelled out by each leading group of `DIGITS` digits
/// that `decode` accepts, stopping at the first group it rejects. Returns how
/// many digits were consumed.
fn extend_groups<const DIGITS: usize, const BYTES: usize>(
    out: &mut BitAccumulator,
    source: &[u8],
    decode: impl Fn(&[u8; DIGITS]) -> Option<[u8; BYTES]>,
) -> usize {
    debug_assert!(out.is_byte_aligned());
    let mut groups = 0;
    for group in source.chunks_exact(DIGITS) {
        let Some(bytes) = decode(group.try_into().unwrap()) else {
            break;
        };
        out.push_aligned_bytes(&bytes);
        groups += 1;
    }
    groups * DIGITS
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

/// `byte` in every byte of a word.
const fn repeated_byte(byte: u8) -> u64 {
    byte as u64 * 0x0101_0101_0101_0101
}

const HIGH_BITS: u64 = repeated_byte(0x80);

/// The top bit of each byte of `word` that is at least `floor`.
///
/// Every byte of `word` must be below `0x80`. Adding `0x80 - floor` then sets
/// a byte's top bit exactly when it reaches `floor`, and no byte can carry
/// into the next.
#[inline(always)]
fn bytes_at_least(word: u64, floor: u8) -> u64 {
    word.wrapping_add(repeated_byte(0x80 - floor)) & HIGH_BITS
}

/// Pack eight hex digits into the four bytes they spell out.
///
/// The digits are read as a little-endian word, so the first is the low byte,
/// and every byte is checked and converted at once rather than through the
/// lookup table a pair at a time.
#[inline]
fn decode_hex_word(digits: &[u8; 8]) -> Option<[u8; 4]> {
    let word = u64::from_le_bytes(*digits);
    if word & HIGH_BITS != 0 {
        return None;
    }
    let is_digit = bytes_at_least(word, b'0') & !bytes_at_least(word, b'9' + 1);
    // Setting 0x20 takes 'A' to 'F' onto 'a' to 'f', and only they land there.
    let folded = word | repeated_byte(0x20);
    let is_letter = bytes_at_least(folded, b'a') & !bytes_at_least(folded, b'f' + 1);
    if is_digit | is_letter != HIGH_BITS {
        return None;
    }
    // A digit's value is its low four bits, and a letter's is nine more.
    let nibbles = (word & repeated_byte(0x0f)) + (is_letter >> 7) * 9;
    // Each even byte takes its own nibble on top and the next byte's below,
    // and then the even bytes are gathered together at the bottom.
    let pairs = ((nibbles << 4) | (nibbles >> 8)) & 0x00ff_00ff_00ff_00ff;
    let quads = (pairs | (pairs >> 8)) & 0x0000_ffff_0000_ffff;
    Some((((quads | (quads >> 16)) & 0xffff_ffff) as u32).to_le_bytes())
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
    let mut out = BitAccumulator::with_bit_capacity(Some(source.len()));
    let mut index = 0;
    while index < source.len() {
        // Eight digits are a whole byte, so runs of them go straight out.
        // Barring separators, that is the whole string in one pass.
        if out.is_byte_aligned() {
            index += extend_groups(&mut out, &source[index..], decode_bin_group);
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
        out.push(u64::from(digit), 1);
        index += 1;
    }
    Ok(out.into_bitvec())
}

pub(crate) fn bv_from_oct(octal_string: &str) -> PyResult<BV> {
    // Ignore any leading '0o' or '0O'
    let s = octal_string
        .strip_prefix("0o")
        .or_else(|| octal_string.strip_prefix("0O"))
        .unwrap_or(octal_string);
    let source = s.as_bytes();
    let mut out = BitAccumulator::with_bit_capacity(Some(source.len() * 3));
    let mut index = 0;
    while index < source.len() {
        // Eight digits are three whole bytes, so runs of them go straight
        // out. Barring separators, that is the whole string in one pass.
        if out.is_byte_aligned() {
            index += extend_groups(&mut out, &source[index..], decode_oct_group);
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
        out.push(u64::from(digit), 3);
        index += 1;
    }
    Ok(out.into_bitvec())
}

/// Append the hex digits from `source[index..]` to `out`, up to the first byte
/// that is not one, and return where that is.
///
/// Inlined into both callers so that a caller handing over short runs one
/// after another, as the comma-separated path does, sets up the word decode's
/// constants once rather than once per run.
#[inline(always)]
fn push_hex_run(out: &mut BitAccumulator, source: &[u8], mut index: usize) -> usize {
    while index < source.len() {
        // A pair of digits is a whole byte, so runs of them go straight out,
        // eight digits at a time and then in pairs for the rest. Barring
        // separators, that is the whole string in one pass. Eight rather than
        // more because a group that runs into a separator is thrown away and
        // redone in pairs, which for short comma-separated tokens is most of
        // them.
        if out.is_byte_aligned() {
            index += extend_groups(out, &source[index..], decode_hex_word);
            index += extend_groups(out, &source[index..], decode_hex_group);
            if index == source.len() {
                break;
            }
        }
        let digit = HEX_VALUES[source[index] as usize];
        if digit == INVALID_HEX {
            break;
        }
        out.push(u64::from(digit), 4);
        index += 1;
    }
    index
}

pub(crate) fn bv_from_hex(hex: &str) -> PyResult<BV> {
    // Ignore any leading '0x' or '0X'
    let s = hex
        .strip_prefix("0x")
        .or_else(|| hex.strip_prefix("0X"))
        .unwrap_or(hex);
    let source = s.as_bytes();
    let mut out = BitAccumulator::with_bit_capacity(Some(source.len() * 4));
    let mut index = push_hex_run(&mut out, source, 0);
    while index < source.len() {
        index += skip_non_digit(s, index).map_err(|c| invalid_character("hex", hex, c))?;
        index = push_hex_run(&mut out, source, index);
    }
    Ok(out.into_bitvec())
}

/// Whether every byte is a printable non-space ASCII character, and whether
/// any of them is a comma.
///
/// Folded rather than short circuited so that it vectorizes: the answer is
/// almost always yes, which means reading the whole string either way, and a
/// bailing out loop is several times slower over the length of a long one.
/// The comma is looked for in the same pass because a string without one is a
/// single token, and finding that out separately meant two more passes over it.
fn scan_literal(bytes: &[u8]) -> (bool, bool) {
    bytes.iter().fold((true, false), |(graphic, comma), &b| {
        (graphic & b.is_ascii_graphic(), comma | (b == b','))
    })
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

/// Decode comma-separated hex literals without allocating per token.
///
/// Removing the prefixes and commas leaves exactly the same hexadecimal bit
/// stream, including when a token contains an odd number of digits, so each
/// token's digits go straight into one accumulator. Anything else - another
/// base, an underscore, an invalid digit - returns `None`, so the general path
/// can parse it or reproduce its usual error.
///
/// One pass over the bytes rather than a split into tokens: the tokens are
/// typically a few digits long, which left splitting them out costing as much
/// as decoding them.
fn try_bv_from_hex_tokens(s: &str) -> Option<BV> {
    let source = s.as_bytes();
    let mut out = BitAccumulator::with_bit_capacity(Some(source.len() * 4));
    let mut index = 0;
    while index < source.len() {
        // An empty token, as between two adjacent commas, adds nothing.
        if source[index] == b',' {
            index += 1;
            continue;
        }
        // Setting 0x20 takes 'X' to 'x', and nothing else there.
        if source[index] != b'0' || source.get(index + 1).map(|c| c | 0x20) != Some(b'x') {
            return None;
        }
        index = push_hex_run(&mut out, source, index + 2);
        if index < source.len() && source[index] != b',' {
            return None;
        }
    }
    Some(out.into_bitvec())
}

pub(crate) fn str_to_bv(s: &str) -> PyResult<BV> {
    // Whitespace has to come out before the string is split into tokens, but
    // removing it means building a new string. Nearly every input is already
    // free of it, and a string of printable non-space characters can only be
    // ASCII, so one scan avoids the copy in the usual case.
    let (graphic, has_comma) = scan_literal(s.as_bytes());
    let stripped;
    let s = if graphic {
        s
    } else {
        stripped = s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        &stripped
    };
    // Taking out whitespace leaves the commas alone, so the scan still holds.
    if !has_comma {
        return if s.is_empty() {
            Ok(BV::new())
        } else {
            string_literal_to_bv(s)
        };
    }
    if let Some(result) = try_bv_from_hex_tokens(s) {
        return Ok(result);
    }
    // The first token is parsed into the result and any others are appended
    // to it.
    let mut tokens = s.split(',').filter(|token| !token.is_empty());
    let Some(first) = tokens.next() else {
        return Ok(BV::new());
    };
    let first = string_literal_to_bv(first)?;
    let Some(second) = tokens.next() else {
        return Ok(first);
    };
    let mut result = BitConcat::with_bit_capacity(first.len());
    result.push_run(first.as_raw_slice(), 0, first.len());
    let second = string_literal_to_bv(second)?;
    result.push_run(second.as_raw_slice(), 0, second.len());
    for token in tokens {
        let bits = string_literal_to_bv(token)?;
        result.push_run(bits.as_raw_slice(), 0, bits.len());
    }
    Ok(result.into_bitvec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pairwise table decode over the same eight digits, as the reference.
    fn decode_by_pairs(digits: [u8; 8]) -> Option<[u8; 4]> {
        let mut bytes = [0u8; 4];
        for (byte, pair) in bytes.iter_mut().zip(digits.chunks_exact(2)) {
            *byte = decode_hex_group(pair.try_into().unwrap())?[0];
        }
        Some(bytes)
    }

    #[test]
    fn word_decode_agrees_with_the_table_for_every_byte_in_every_position() {
        let valid = *b"0aF9b8C7";
        for position in 0..8 {
            for byte in 0..=255u8 {
                let mut digits = valid;
                digits[position] = byte;
                assert_eq!(
                    decode_hex_word(&digits),
                    decode_by_pairs(digits),
                    "{digits:?}"
                );
            }
        }
    }

    #[test]
    fn word_decode_agrees_with_the_table_for_every_pair_of_digits() {
        let digits = b"0123456789abcdefABCDEF";
        for &high in digits {
            for &low in digits {
                for position in (0..8).step_by(2) {
                    let mut word = *b"00000000";
                    word[position] = high;
                    word[position + 1] = low;
                    assert_eq!(decode_hex_word(&word), decode_by_pairs(word), "{word:?}");
                }
            }
        }
    }

    /// Hex digits with no prefix or separators, decoded without going through
    /// anything that can build a Python error, which a test binary cannot link.
    fn bv_from_digits(digits: &str) -> BV {
        let mut out = BitAccumulator::with_bit_capacity(None);
        assert_eq!(push_hex_run(&mut out, digits.as_bytes(), 0), digits.len());
        out.into_bitvec()
    }

    #[test]
    fn hex_tokens_match_the_single_literal() {
        for (tokens, single) in [
            ("0x1,0x23,0x456", "123456"),
            ("0xabc,0X0,0x", "abc0"),
            ("0x0123456789abcdef0,0xF", "0123456789abcdef0F"),
        ] {
            assert_eq!(try_bv_from_hex_tokens(tokens), Some(bv_from_digits(single)));
        }
        assert_eq!(try_bv_from_hex_tokens("0x12,0b1"), None);
        assert_eq!(try_bv_from_hex_tokens("0x1_2,0x3"), None);
        assert_eq!(try_bv_from_hex_tokens("0x12,0xg"), None);
    }
}
