//! Rendering left aligned byte data as binary, octal and hexadecimal text.
//!
//! Every function here takes the same shape of input: a byte buffer whose first
//! bit is the first bit of the data, holding at least `len_bits` bits. Only
//! those bits are read, so the padding in the final byte does not have to be
//! masked. The output is built once into a buffer of the exact final size and
//! filled from a lookup table, rather than a digit at a time.

const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

/// The two hex digits of each byte value.
const HEX_PAIRS: [[u8; 2]; 256] = {
    let mut table = [[0u8; 2]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        table[byte] = [HEX_DIGITS[byte >> 4], HEX_DIGITS[byte & 0xf]];
        byte += 1;
    }
    table
};

/// The eight binary digits of each byte value.
const BIN_OCTETS: [[u8; 8]; 256] = {
    let mut table = [[b'0'; 8]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut bit = 0usize;
        while bit < 8 {
            table[byte][bit] = b'0' + ((byte >> (7 - bit)) & 1) as u8;
            bit += 1;
        }
        byte += 1;
    }
    table
};

/// Render the first `len_bits` bits of `bytes` as hex digits.
pub(crate) fn hex_from_padded_bytes(bytes: &[u8], len_bits: usize) -> String {
    debug_assert!(len_bits.is_multiple_of(4));
    debug_assert!(bytes.len() >= len_bits.div_ceil(8));
    let mut out = vec![0u8; len_bits / 4];
    let whole_bytes = len_bits / 8;
    for (pair, &byte) in out.chunks_exact_mut(2).zip(&bytes[..whole_bytes]) {
        pair.copy_from_slice(&HEX_PAIRS[byte as usize]);
    }
    // An odd digit count leaves a nibble that `chunks_exact_mut` skipped.
    if !len_bits.is_multiple_of(8)
        && let Some(last) = out.last_mut()
    {
        *last = HEX_PAIRS[bytes[whole_bytes] as usize][0];
    }
    into_ascii_string(out)
}

/// Render the first `len_bits` bits of `bytes` as binary digits.
pub(crate) fn bin_from_padded_bytes(bytes: &[u8], len_bits: usize) -> String {
    debug_assert!(bytes.len() >= len_bits.div_ceil(8));
    let mut out = vec![0u8; len_bits];
    let whole_bytes = len_bits / 8;
    for (octet, &byte) in out.chunks_exact_mut(8).zip(&bytes[..whole_bytes]) {
        octet.copy_from_slice(&BIN_OCTETS[byte as usize]);
    }
    let remainder = len_bits & 7;
    if remainder != 0 {
        out[whole_bytes * 8..]
            .copy_from_slice(&BIN_OCTETS[bytes[whole_bytes] as usize][..remainder]);
    }
    into_ascii_string(out)
}

/// Render the first `len_bits` bits of `bytes` as octal digits.
pub(crate) fn oct_from_padded_bytes(bytes: &[u8], len_bits: usize) -> String {
    debug_assert!(len_bits.is_multiple_of(3));
    debug_assert!(bytes.len() >= len_bits.div_ceil(8));
    let digits = len_bits / 3;
    let mut out = vec![0u8; digits];
    // Three bytes hold exactly eight octal digits, so a digit never straddles a
    // group boundary and each group can be read as one big endian word.
    let full_groups = digits / 8;
    let (grouped, tail) = bytes.split_at(full_groups * 3);
    for (group, group_out) in grouped.chunks_exact(3).zip(out.chunks_exact_mut(8)) {
        write_oct_digits(
            u32::from_be_bytes([0, group[0], group[1], group[2]]),
            group_out,
        );
    }
    let remaining = digits - full_groups * 8;
    if remaining != 0 {
        // The tail is under a full group, so the bytes it needs may run out.
        let mut word = 0u32;
        for index in 0..3 {
            word = (word << 8) | u32::from(tail.get(index).copied().unwrap_or(0));
        }
        write_oct_digits(word, &mut out[full_groups * 8..]);
    }
    into_ascii_string(out)
}

/// Write the leading octal digits of a 24 bit big endian `word`, one per
/// element of `out`, which is never longer than the eight digits of a word.
#[inline]
fn write_oct_digits(word: u32, out: &mut [u8]) {
    debug_assert!(out.len() <= 8);
    for (index, digit) in out.iter_mut().enumerate() {
        *digit = b'0' + ((word >> (21 - 3 * index)) & 7) as u8;
    }
}

#[inline]
fn into_ascii_string(bytes: Vec<u8>) -> String {
    debug_assert!(bytes.is_ascii());
    // SAFETY: every byte written by this module is an ASCII digit, so the
    // buffer is valid UTF-8 and does not need checking again.
    unsafe { String::from_utf8_unchecked(bytes) }
}
