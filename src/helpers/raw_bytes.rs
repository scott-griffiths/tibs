use super::bits::{BS, BV};
use bitvec::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::borrow::Cow;

pub(crate) type ByteSearchPrep<'h, 'n> = (Cow<'h, [u8]>, Cow<'n, [u8]>, usize);

#[inline]
pub(crate) fn bits_to_bytes(bits: &BS) -> Vec<u8> {
    debug_assert!(bits.len().is_multiple_of(8));
    bits.chunks_exact(8)
        .map(|chunk| chunk.load_be::<u8>())
        .collect()
}

pub(crate) fn byte_search_prep<'h, 'n>(
    haystack: &'h BS,
    needle: &'n BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> Option<ByteSearchPrep<'h, 'n>> {
    if alignment_mod8 != Some(0) || !needle.len().is_multiple_of(8) {
        return None;
    }

    let start_byte = start.div_ceil(8);
    let end_byte = end / 8;
    if start_byte > end_byte {
        return None;
    }
    let haystack_bytes = bits_to_byte_cow(&haystack[start_byte * 8..end_byte * 8]);
    let needle_bytes = bits_to_byte_cow(needle);
    Some((haystack_bytes, needle_bytes, start_byte))
}

fn bits_to_byte_cow(bits: &BS) -> Cow<'_, [u8]> {
    debug_assert!(bits.len().is_multiple_of(8));
    match bits.domain() {
        bitvec::domain::Domain::Region {
            head: None,
            body,
            tail: None,
        } => Cow::Borrowed(body),
        _ => Cow::Owned(bits_to_bytes(bits)),
    }
}

pub(crate) fn bv_from_bytes_slice(
    data: Vec<u8>,
    offset: Option<usize>,
    length: Option<usize>,
) -> PyResult<BV> {
    let offset = offset.unwrap_or(0);
    let data_length = data.len() * 8;
    if offset > data_length {
        return Err(PyValueError::new_err(format!(
            "Offset of {offset} is greater than the data length ({data_length} bits)."
        )));
    }
    let length = length.unwrap_or(data_length - offset);
    let Some(_end) = offset.checked_add(length).filter(|&end| end <= data_length) else {
        return Err(PyValueError::new_err(format!(
            "Length of {length} with offset of {offset} is greater than the data length ({data_length} bits)."
        )));
    };
    if offset == 0 && length == data_length {
        return Ok(BV::from_vec(data));
    }
    if length == 0 {
        return Ok(BV::new());
    }

    let byte_offset = offset / 8;
    let bit_offset = offset & 7;
    let byte_len = length.div_ceil(8);
    let mut bytes = if bit_offset == 0 {
        data[byte_offset..byte_offset + byte_len].to_vec()
    } else {
        let input_len = (bit_offset + length).div_ceil(8);
        shifted_padded_bytes(
            &data[byte_offset..byte_offset + input_len],
            bit_offset,
            length,
        )
    };

    mask_padding_bits(&mut bytes, length);
    let mut bv = BV::from_vec(bytes);
    bv.truncate(length);
    Ok(bv)
}

#[inline]
fn shifted_padded_bytes(data: &[u8], bit_offset: usize, len_bits: usize) -> Vec<u8> {
    let mut out = vec![0u8; len_bits.div_ceil(8)];
    copy_shifted_bytes(data, bit_offset, &mut out);
    out
}

#[inline]
pub(crate) fn copy_shifted_bytes(data: &[u8], bit_offset: usize, out: &mut [u8]) {
    debug_assert!((1..8).contains(&bit_offset));
    debug_assert!(data.len() >= out.len());
    let Some((last, prefix)) = out.split_last_mut() else {
        return;
    };

    let right_shift = 8 - bit_offset;
    for (index, byte) in prefix.iter_mut().enumerate() {
        *byte = (data[index] << bit_offset) | (data[index + 1] >> right_shift);
    }

    let last_index = prefix.len();
    let next = if data.len() > last_index + 1 {
        data[last_index + 1]
    } else {
        0
    };
    *last = (data[last_index] << bit_offset) | (next >> right_shift);
}

/// Reverse the first `len_bits` bits of `bytes` in place.
///
/// The bits must be left aligned (bit zero is the most significant bit of
/// `bytes[0]`) with the trailing `bytes.len() * 8 - len_bits` padding bits
/// zeroed. The reversed bits are left aligned again, and the padding is
/// still zero afterwards.
/// Reverse each `width`-byte group of `bytes` in place, which is the byte
/// order swap.
///
/// The widths that matter are the ones integers come in, and for those the
/// reversal is one `bswap` instruction rather than a loop over the group, so
/// they get their own arms and the run vectorises.
pub(crate) fn reverse_byte_groups(bytes: &mut [u8], width: usize) {
    macro_rules! swap_as {
        ($ty:ty, $n:expr) => {
            for group in bytes.chunks_exact_mut($n) {
                let value = <$ty>::from_ne_bytes(group.try_into().unwrap());
                group.copy_from_slice(&value.swap_bytes().to_ne_bytes());
            }
        };
    }
    match width {
        1 => {}
        2 => swap_as!(u16, 2),
        4 => swap_as!(u32, 4),
        8 => swap_as!(u64, 8),
        16 => swap_as!(u128, 16),
        _ => {
            for group in bytes.chunks_exact_mut(width) {
                group.reverse();
            }
        }
    }
}

pub(crate) fn reverse_padded_bits(bytes: &mut [u8], len_bits: usize) {
    debug_assert_eq!(bytes.len(), len_bits.div_ceil(8));
    reverse_all_bits(bytes);
    // Reversing the whole buffer pushes the padding to the front, so the
    // data now sits `pad` bits in and has to be pulled back to the start.
    let pad = bytes.len() * 8 - len_bits;
    if pad != 0 {
        shift_bytes_left(bytes, pad);
    }
}

/// Reverse every bit of `bytes`, so that bit `i` ends up at `8 * len - 1 - i`.
///
/// Kept as two separate sweeps rather than one combined pass because both of
/// them auto-vectorize, which beats a scalar loop working in from both ends.
#[inline]
fn reverse_all_bits(bytes: &mut [u8]) {
    bytes.reverse();
    bytes
        .iter_mut()
        .for_each(|byte| *byte = byte.reverse_bits());
}

/// Shift the bits of `bytes` towards the front by `shift` (1..8) bits,
/// zero filling at the end.
fn shift_bytes_left(bytes: &mut [u8], shift: usize) {
    debug_assert!((1..8).contains(&shift));
    const WORD: usize = size_of::<u64>();
    let len = bytes.len();
    let Some(last) = len.checked_sub(1) else {
        return;
    };
    let right = 8 - shift;
    let mut index = 0usize;
    // Each word needs the byte after it to supply the bits shifted in.
    while index + WORD < len {
        let word = u64::from_be_bytes(bytes[index..index + WORD].try_into().unwrap());
        let carry = u64::from(bytes[index + WORD] >> right);
        bytes[index..index + WORD].copy_from_slice(&((word << shift) | carry).to_be_bytes());
        index += WORD;
    }
    while index < last {
        bytes[index] = (bytes[index] << shift) | (bytes[index + 1] >> right);
        index += 1;
    }
    bytes[last] <<= shift;
}

#[inline]
pub(crate) fn mask_padding_bits(bytes: &mut [u8], len_bits: usize) {
    let remainder = len_bits & 7;
    if remainder != 0
        && let Some(last) = bytes.last_mut()
    {
        *last &= 0xffu8 << (8 - remainder);
    }
}
