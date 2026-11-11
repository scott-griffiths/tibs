//! Moving fields between an LSB-first stream and the MSB-first order the
//! ordinary decoders and encoders use.
//!
//! In an LSB-first stream bit label `L` is bit `L % 8` of byte `L / 8`,
//! counting from the least significant bit, and a field of `n` bits at label
//! `L` has label `L` as its least significant bit. Read as a little-endian
//! integer, the whole stream holds each field as `(stream >> L) & mask`. These
//! functions read and write fields under that rule, a word at a time.

use super::bits::{BS, BitAccumulator, load_be_bits};
use super::raw_bytes::{bitslice_storage, copy_shifted_bytes};
use std::borrow::Cow;

/// The bytes of a run that is a whole number of bytes long, starting on a byte
/// boundary.
///
/// Borrowed when the run's storage already starts on a byte boundary, which is
/// the usual case; otherwise the bytes are shifted into a copy.
pub(crate) fn whole_bytes(bits: &BS) -> Cow<'_, [u8]> {
    debug_assert!(bits.len().is_multiple_of(8));
    let (storage, offset) = bitslice_storage(bits);
    if offset == 0 {
        return storage;
    }
    let mut out = vec![0u8; bits.len() / 8];
    copy_shifted_bytes(&storage, offset, &mut out);
    Cow::Owned(out)
}

/// The `length` bits at `label` of the LSB-first stream in `bytes`, with
/// `label` as the least significant bit, right-aligned in a `u64`.
///
/// `length` is 1 to 64 and the bits must lie within `bytes`.
#[inline]
pub(crate) fn load_lsb0_bits(bytes: &[u8], label: usize, length: usize) -> u64 {
    debug_assert!((1..=64).contains(&length));
    debug_assert!(label + length <= bytes.len() * 8);
    let byte = label / 8;
    let offset = label % 8;
    let word = match bytes.get(byte..byte + 8) {
        Some(word) => u64::from_le_bytes(word.try_into().unwrap()),
        None => {
            let mut buf = [0u8; 8];
            let tail = &bytes[byte..];
            buf[..tail.len()].copy_from_slice(tail);
            u64::from_le_bytes(buf)
        }
    };
    let mut value = word >> offset;
    // Only a field that starts part way into a byte and runs past the eighth
    // byte needs a ninth.
    if offset + length > 64 {
        value |= u64::from(bytes[byte + 8]) << (64 - offset);
    }
    if length == 64 {
        value
    } else {
        value & ((1u64 << length) - 1)
    }
}

/// Write the low `length` bits of `value` at `label` of the LSB-first stream in
/// `bytes`, `label` taking the least significant bit.
///
/// The bits are ORed in, so the target bits must already be clear. `length` is
/// 1 to 64, the bits of `value` above it must be zero, and the bits must lie
/// within `bytes`.
#[inline]
pub(crate) fn store_lsb0_bits(bytes: &mut [u8], label: usize, length: usize, value: u64) {
    debug_assert!((1..=64).contains(&length));
    debug_assert!(length == 64 || value >> length == 0);
    debug_assert!(label + length <= bytes.len() * 8);
    let byte = label / 8;
    let offset = label % 8;
    if offset + length <= 64
        && let Some(window) = bytes.get_mut(byte..byte + 8)
    {
        let word = u64::from_le_bytes((&*window).try_into().unwrap()) | (value << offset);
        window.copy_from_slice(&word.to_le_bytes());
        return;
    }
    let shifted = u128::from(value) << offset;
    for (index, target) in bytes[byte..(label + length).div_ceil(8)]
        .iter_mut()
        .enumerate()
    {
        *target |= (shifted >> (8 * index)) as u8;
    }
}

/// Append the `length`-bit field at `label` of the LSB-first stream in `bytes`
/// to `out`, most significant bit first, which is the order every decoder
/// reads.
///
/// A field wider than 64 bits goes out a word at a time from its top.
#[inline]
pub(crate) fn push_lsb0_field(out: &mut BitAccumulator, bytes: &[u8], label: usize, length: usize) {
    let mut remaining = length;
    while remaining > 0 {
        let chunk = remaining.min(64);
        remaining -= chunk;
        out.push_wide(load_lsb0_bits(bytes, label + remaining, chunk), chunk);
    }
}

/// [`push_lsb0_field`] for a field whose dtype names a byte order, which is a
/// whole number of bytes long: the bytes go out lowest labels first.
///
/// That is the natural value with its bytes reversed. A `_le` decoder reverses
/// them back, so the dtype reads the natural, little-endian value; a `_be`
/// decoder takes them as they are, so the first byte in the stream is the
/// most significant.
#[inline]
pub(crate) fn push_lsb0_field_bytes(
    out: &mut BitAccumulator,
    bytes: &[u8],
    label: usize,
    length: usize,
) {
    debug_assert!(length.is_multiple_of(8));
    let mut done = 0;
    while done < length {
        let chunk = (length - done).min(64);
        let value = load_lsb0_bits(bytes, label + done, chunk);
        out.push_wide(value.swap_bytes() >> (64 - chunk), chunk);
        done += chunk;
    }
}

/// [`store_msb0_field_as_lsb0`] for a field whose dtype names a byte order:
/// the inverse of [`push_lsb0_field_bytes`], the first byte of the field in
/// `source` taking the lowest labels.
#[inline]
pub(crate) fn store_msb0_field_bytes_as_lsb0(
    target: &mut [u8],
    source: &[u8],
    source_offset: usize,
    bit: usize,
    length: usize,
) {
    debug_assert!(length.is_multiple_of(8));
    let mut done = 0;
    while done < length {
        let chunk = (length - done).min(64);
        let value = load_be_bits(source, source_offset + bit + done, chunk);
        store_lsb0_bits(
            target,
            bit + done,
            chunk,
            value.swap_bytes() >> (64 - chunk),
        );
        done += chunk;
    }
}

/// Copy the `length`-bit field that starts `bit` bits into the MSB-first
/// `source` to the same position, as a label, of the LSB-first stream in
/// `target`.
///
/// This is [`push_lsb0_field`] run backwards: the field's most significant bit
/// is the first of `source` and the highest label of `target`.
#[inline]
pub(crate) fn store_msb0_field_as_lsb0(
    target: &mut [u8],
    source: &[u8],
    source_offset: usize,
    bit: usize,
    length: usize,
) {
    let mut done = 0;
    while done < length {
        let chunk = (length - done).min(64);
        let value = load_be_bits(source, source_offset + bit + length - done - chunk, chunk);
        store_lsb0_bits(target, bit + done, chunk, value);
        done += chunk;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference rule: the field at `label` is the stream read as one
    /// little-endian integer, shifted down and masked.
    fn reference(bytes: &[u8], label: usize, length: usize) -> u128 {
        let mut whole = 0u128;
        for (index, &byte) in bytes.iter().enumerate().take(16) {
            whole |= u128::from(byte) << (8 * index);
        }
        (whole >> label) & ((1u128 << length) - 1)
    }

    #[test]
    fn load_agrees_with_the_little_endian_integer() {
        let bytes: Vec<u8> = (0u8..15).map(|i| i.wrapping_mul(0x9d) ^ 0x5a).collect();
        for length in 1..=64 {
            for label in 0..=bytes.len() * 8 - length {
                assert_eq!(
                    u128::from(load_lsb0_bits(&bytes, label, length)),
                    reference(&bytes, label, length),
                    "{label} {length}"
                );
            }
        }
    }

    #[test]
    fn store_is_the_inverse_of_load() {
        let bytes: Vec<u8> = (0u8..15).map(|i| i.wrapping_mul(0x9d) ^ 0x5a).collect();
        for length in 1..=64 {
            for label in 0..=bytes.len() * 8 - length {
                let value = load_lsb0_bits(&bytes, label, length);
                let mut target = vec![0u8; bytes.len()];
                store_lsb0_bits(&mut target, label, length, value);
                assert_eq!(load_lsb0_bits(&target, label, length), value);
                let mut cleared = bytes.clone();
                for index in label..label + length {
                    cleared[index / 8] &= !(1 << (index % 8));
                }
                for (written, (original, cleared)) in target.iter().zip(bytes.iter().zip(&cleared))
                {
                    assert_eq!(*written, original ^ cleared, "{label} {length}");
                }
            }
        }
    }
}
