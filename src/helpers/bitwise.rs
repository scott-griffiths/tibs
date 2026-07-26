//! Low-level bit and byte machinery shared by the `BitCollection` operations.
//!
//! Everything here works on raw `&[u8]` storage plus a bit offset, or on a
//! `BitSlice` when the storage does not start on a byte boundary. Nothing in
//! this module knows about `Tibs`, `Mutibs` or Python; the collection-level
//! logic that drives it lives in `core.rs`.

use super::bits::{BS, BV, head_bit_offset};
use super::raw_bytes::{copy_shifted_bytes, mask_padding_bits, reverse_padded_bits};
use bitvec::prelude::*;

#[inline]
fn align_byte(bytes: &[u8], byte_index: usize, bit_shift: isize) -> u8 {
    debug_assert!((-7..=7).contains(&bit_shift));
    match bit_shift.cmp(&0) {
        std::cmp::Ordering::Equal => bytes.get(byte_index).copied().unwrap_or(0),
        std::cmp::Ordering::Greater => {
            let shift = bit_shift as u32;
            let current = bytes.get(byte_index).copied().unwrap_or(0);
            let next = bytes.get(byte_index + 1).copied().unwrap_or(0);
            (current << shift) | (next >> (8 - shift))
        }
        std::cmp::Ordering::Less => {
            let shift = (-bit_shift) as u32;
            let previous = byte_index
                .checked_sub(1)
                .and_then(|index| bytes.get(index))
                .copied()
                .unwrap_or(0);
            let current = bytes.get(byte_index).copied().unwrap_or(0);
            (current >> shift) | (previous << (8 - shift))
        }
    }
}

#[inline]
pub(crate) fn copy_unaligned_padded_bytes(
    bytes: &[u8],
    bit_offset: usize,
    len_bits: usize,
    out: &mut [u8],
) {
    debug_assert!((1..8).contains(&bit_offset));
    debug_assert_eq!(out.len(), len_bits.div_ceil(8));

    copy_shifted_bytes(bytes, bit_offset, out);
    mask_padding_bits(out, len_bits);
}

/// Reverse the bits of `bits` in place, working over the raw storage.
///
/// `BitSlice::reverse` swaps one bit at a time through bit pointers, which
/// is hundreds of times slower than sweeping the bytes.
pub(crate) fn reverse_bitvec_in_place(bits: &mut BV) {
    let len = bits.len();
    if len < 2 {
        return;
    }
    let offset = head_bit_offset(bits.as_bitslice());
    if offset == 0 {
        let bytes = &mut bits.as_raw_mut_slice()[..len.div_ceil(8)];
        // The bits past the end are dead storage and may hold anything, but
        // the reversal moves them to the front, so clear them first.
        mask_padding_bits(bytes, len);
        reverse_padded_bits(bytes, len);
        return;
    }
    // Storage starting mid byte cannot be reversed within itself, so realign
    // it on the way through. The result then starts at bit zero.
    let mut bytes = vec![0u8; len.div_ceil(8)];
    copy_unaligned_padded_bytes(bits.as_raw_slice(), offset, len, &mut bytes);
    reverse_padded_bits(&mut bytes, len);
    let mut reversed = BV::from_vec(bytes);
    reversed.truncate(len);
    *bits = reversed;
}

#[derive(Clone, Copy)]
pub(crate) enum LogicalOp {
    Or,
    And,
    Xor,
    /// Only used by the fused counting and predicate paths; there is no
    /// public `and not` operator.
    AndNot,
}

impl LogicalOp {
    #[inline]
    fn byte(self, lhs: u8, rhs: u8) -> u8 {
        match self {
            LogicalOp::Or => lhs | rhs,
            LogicalOp::And => lhs & rhs,
            LogicalOp::Xor => lhs ^ rhs,
            LogicalOp::AndNot => lhs & !rhs,
        }
    }

    #[inline]
    fn word(self, lhs: u64, rhs: u64) -> u64 {
        match self {
            LogicalOp::Or => lhs | rhs,
            LogicalOp::And => lhs & rhs,
            LogicalOp::Xor => lhs ^ rhs,
            LogicalOp::AndNot => lhs & !rhs,
        }
    }

    #[inline]
    pub(crate) fn bitslice(self, result: &mut BV, rhs: &BS) {
        match self {
            LogicalOp::Or => *result |= rhs,
            LogicalOp::And => *result &= rhs,
            LogicalOp::Xor => *result ^= rhs,
            // Never reached from `logical_op`, which is only called with the
            // three operators that have a Python spelling.
            LogicalOp::AndNot => *result &= (!rhs.to_bitvec()).as_bitslice(),
        }
    }
}

#[inline]
fn read_be_u64(bytes: &[u8], index: usize) -> u64 {
    u64::from_be_bytes(bytes[index..index + 8].try_into().unwrap())
}

#[inline]
pub(crate) fn logical_op_with_matching_bytes(lhs: &[u8], rhs: &[u8], op: LogicalOp) -> Vec<u8> {
    debug_assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&left, &right)| op.byte(left, right))
        .collect()
}

#[inline]
pub(crate) fn logical_op_with_aligned_bytes(
    lhs: &[u8],
    lhs_offset: usize,
    rhs: &[u8],
    rhs_offset: usize,
    op: LogicalOp,
) -> Vec<u8> {
    debug_assert!(lhs_offset < 8);
    debug_assert!(rhs_offset < 8);

    let rhs_shift = rhs_offset as isize - lhs_offset as isize;
    let mut out = Vec::with_capacity(lhs.len());
    let mut index = 0;
    match rhs_shift.cmp(&0) {
        std::cmp::Ordering::Equal => {
            return logical_op_with_matching_bytes(lhs, rhs, op);
        }
        std::cmp::Ordering::Greater => {
            let left_shift = rhs_shift as u32;
            let right_shift = 8 - left_shift;
            while index + 8 <= lhs.len() && index + 8 <= rhs.len() {
                let next = rhs.get(index + 8).copied().unwrap_or(0) as u64;
                let aligned = (read_be_u64(rhs, index) << left_shift) | (next >> right_shift);
                let word = op.word(read_be_u64(lhs, index), aligned);
                out.extend_from_slice(&word.to_be_bytes());
                index += 8;
            }
        }
        std::cmp::Ordering::Less => {
            let right_shift = (-rhs_shift) as u32;
            while index + 8 <= lhs.len() && index + 8 <= rhs.len() {
                let previous = if index == 0 { 0 } else { rhs[index - 1] as u64 };
                let aligned =
                    (read_be_u64(rhs, index) >> right_shift) | (previous << (64 - right_shift));
                let word = op.word(read_be_u64(lhs, index), aligned);
                out.extend_from_slice(&word.to_be_bytes());
                index += 8;
            }
        }
    }
    out.extend(
        (index..lhs.len())
            .map(|byte_index| op.byte(lhs[byte_index], align_byte(rhs, byte_index, rhs_shift))),
    );
    out
}

/// The 64 bits of `rhs` starting at byte `index`, shifted so they line up with
/// `lhs`'s bit offset. Requires `index + 8 <= rhs.len()`.
#[inline]
fn aligned_rhs_word(rhs: &[u8], index: usize, rhs_shift: isize) -> u64 {
    debug_assert!((-7..=7).contains(&rhs_shift));
    match rhs_shift.cmp(&0) {
        std::cmp::Ordering::Equal => read_be_u64(rhs, index),
        std::cmp::Ordering::Greater => {
            let left_shift = rhs_shift as u32;
            let next = rhs.get(index + 8).copied().unwrap_or(0) as u64;
            (read_be_u64(rhs, index) << left_shift) | (next >> (8 - left_shift))
        }
        std::cmp::Ordering::Less => {
            let right_shift = (-rhs_shift) as u32;
            let previous = if index == 0 { 0 } else { rhs[index - 1] as u64 };
            (read_be_u64(rhs, index) >> right_shift) | (previous << (64 - right_shift))
        }
    }
}

/// Masks for the partial bits at each end of the live range, which the raw byte
/// slices carry either side of it. `logical_op` can ignore them because it
/// slices them off the result afterwards; anything that *reduces* over the bits
/// has to mask them out instead, or they are folded into the answer.
#[inline]
fn edge_masks(offset: usize, len: usize) -> (u8, u8) {
    let tail = match (offset + len) & 7 {
        0 => 0xffu8,
        bits => !(0xffu8 >> bits),
    };
    (0xffu8 >> offset, tail)
}

/// The number of set bits in `op(lhs, rhs)` over the live range only.
///
/// Kept separate from `for_each_pair_word` because it must not carry a
/// per-word branch: counting is pure throughput, and a callback that can stop
/// the loop early stops the compiler vectorising it. For the same reason the
/// operation is dispatched here, once, so that each inner loop is compiled with
/// it inlined rather than re-matching an enum on every word.
#[inline]
pub(crate) fn count_pair_bits(
    lhs: &[u8],
    lhs_offset: usize,
    rhs: &[u8],
    rhs_offset: usize,
    len: usize,
    op: LogicalOp,
) -> usize {
    macro_rules! counted {
        ($word:expr, $byte:expr) => {
            count_pair_bits_with(lhs, lhs_offset, rhs, rhs_offset, len, $word, $byte)
        };
    }
    match op {
        LogicalOp::Or => counted!(|a, b| a | b, |a, b| a | b),
        LogicalOp::And => counted!(|a, b| a & b, |a, b| a & b),
        LogicalOp::Xor => counted!(|a, b| a ^ b, |a, b| a ^ b),
        LogicalOp::AndNot => counted!(|a, b| a & !b, |a, b| a & !b),
    }
}

fn count_pair_bits_with<W, B>(
    lhs: &[u8],
    lhs_offset: usize,
    rhs: &[u8],
    rhs_offset: usize,
    len: usize,
    word_op: W,
    byte_op: B,
) -> usize
where
    W: Fn(u64, u64) -> u64,
    B: Fn(u8, u8) -> u8,
{
    debug_assert!(lhs_offset < 8);
    debug_assert!(rhs_offset < 8);
    if len == 0 {
        return 0;
    }
    debug_assert_eq!(lhs.len(), (lhs_offset + len).div_ceil(8));
    let last = lhs.len() - 1;
    let rhs_shift = rhs_offset as isize - lhs_offset as isize;
    let (head_mask, tail_mask) = edge_masks(lhs_offset, len);
    let pair_byte = |index: usize| byte_op(lhs[index], align_byte(rhs, index, rhs_shift));

    if last == 0 {
        return (pair_byte(0) & head_mask & tail_mask).count_ones() as usize;
    }
    let mut count = (pair_byte(0) & head_mask).count_ones() as usize
        + (pair_byte(last) & tail_mask).count_ones() as usize;

    // Bytes strictly between the two partial ends are wholly live, so they need
    // no masking and can go 64 bits at a time.
    if rhs_shift == 0 {
        // Byte order is irrelevant to a popcount, so read native-endian and
        // skip the byte swap that the shifted path needs.
        debug_assert_eq!(lhs.len(), rhs.len());
        let (left, right) = (&lhs[1..last], &rhs[1..last]);
        let mut left_chunks = left.chunks_exact(8);
        let mut right_chunks = right.chunks_exact(8);
        for (left_chunk, right_chunk) in left_chunks.by_ref().zip(right_chunks.by_ref()) {
            let word = word_op(
                u64::from_ne_bytes(left_chunk.try_into().unwrap()),
                u64::from_ne_bytes(right_chunk.try_into().unwrap()),
            );
            count += word.count_ones() as usize;
        }
        for (&left_byte, &right_byte) in
            left_chunks.remainder().iter().zip(right_chunks.remainder())
        {
            count += byte_op(left_byte, right_byte).count_ones() as usize;
        }
        return count;
    }
    let mut index = 1;
    while index + 8 <= last && index + 8 <= rhs.len() {
        let word = word_op(
            read_be_u64(lhs, index),
            aligned_rhs_word(rhs, index, rhs_shift),
        );
        count += word.count_ones() as usize;
        index += 8;
    }
    while index < last {
        count += pair_byte(index).count_ones() as usize;
        index += 1;
    }
    count
}

/// Feed `op(lhs, rhs)` to `on_word` in chunks, covering the live bit range and
/// nothing else, and stopping early if `on_word` returns false. Used by the
/// predicates, where stopping early is the whole point.
pub(crate) fn for_each_pair_word<F>(
    lhs: &[u8],
    lhs_offset: usize,
    rhs: &[u8],
    rhs_offset: usize,
    len: usize,
    op: LogicalOp,
    mut on_word: F,
) where
    F: FnMut(u64) -> bool,
{
    debug_assert!(lhs_offset < 8);
    debug_assert!(rhs_offset < 8);
    if len == 0 {
        return;
    }
    let last = lhs.len() - 1;
    debug_assert_eq!(lhs.len(), (lhs_offset + len).div_ceil(8));

    let rhs_shift = rhs_offset as isize - lhs_offset as isize;
    let (head_mask, tail_mask) = edge_masks(lhs_offset, len);
    let pair_byte = |index: usize| op.byte(lhs[index], align_byte(rhs, index, rhs_shift));

    if last == 0 {
        on_word((pair_byte(0) & head_mask & tail_mask) as u64);
        return;
    }
    if !on_word((pair_byte(0) & head_mask) as u64) {
        return;
    }
    // Bytes strictly between the two partial ends are wholly live, so they need
    // no masking and can go 64 bits at a time.
    let mut index = 1;
    while index + 8 <= last && index + 8 <= rhs.len() {
        let word = op.word(
            read_be_u64(lhs, index),
            aligned_rhs_word(rhs, index, rhs_shift),
        );
        if !on_word(word) {
            return;
        }
        index += 8;
    }
    while index < last {
        if !on_word(pair_byte(index) as u64) {
            return;
        }
        index += 1;
    }
    on_word((pair_byte(last) & tail_mask) as u64);
}

/// Fallback for operands whose underlying storage does not start on a byte
/// boundary, so `raw_data_ref` gives nothing to work with. `load_be` zero-fills
/// a partial chunk, and both operands are the same length, so the padding
/// contributes nothing to any of the four operations and needs no masking.
pub(crate) fn for_each_pair_word_bitslice<F>(lhs: &BS, rhs: &BS, op: LogicalOp, mut on_word: F)
where
    F: FnMut(u64) -> bool,
{
    debug_assert_eq!(lhs.len(), rhs.len());
    for (left, right) in lhs.chunks(64).zip(rhs.chunks(64)) {
        if !on_word(op.word(left.load_be::<u64>(), right.load_be::<u64>())) {
            return;
        }
    }
}

/// Scatter `value`'s bits into the positions of `bits` where `mask` is set,
/// leaving the other bits untouched (the PDEP operation). `bits` and `mask` must
/// be the same length and `value` must be `mask.count_ones()` bits long; both
/// are the caller's responsibility to check.
pub(crate) fn deposit_masked(bits: &mut BS, value: &BS, mask: &BS) {
    debug_assert_eq!(bits.len(), mask.len());
    debug_assert_eq!(value.len(), mask.count_ones());
    for (value_index, pos) in mask.iter_ones().enumerate() {
        bits.set(pos, value[value_index]);
    }
}

/// The number of set bits in a run of whole bytes.
///
/// Sixty-four bits at a time, with the up-to-seven bytes left over counted
/// singly. The word loop reads through `chunks_exact` rather than casting the
/// run to `&[u64]`: `body` is a slice into the middle of someone's storage, so
/// it need not start on a word boundary and need not be a whole number of
/// words, and a cast that has to reject either case would send the entire scan
/// - not just the odd bytes at the end - down the byte-at-a-time path.
#[inline]
fn count_ones_in_bytes(bytes: &[u8]) -> usize {
    let mut chunks = bytes.chunks_exact(8);
    let mut ones = 0;
    for chunk in chunks.by_ref() {
        // Byte order is irrelevant to a popcount, so read native-endian.
        ones += u64::from_ne_bytes(chunk.try_into().unwrap()).count_ones() as usize;
    }
    for &byte in chunks.remainder() {
        ones += byte.count_ones() as usize;
    }
    ones
}

pub(crate) fn count_bitslice(slice: &BS, count_ones: bool) -> usize {
    let mut ones = 0;

    match slice.domain() {
        bitvec::domain::Domain::Region { head, body, tail } => {
            if let Some(h) = head {
                ones += h.into_bitslice().count_ones();
            }
            ones += count_ones_in_bytes(body);
            if let Some(t) = tail {
                ones += t.into_bitslice().count_ones();
            }
        }
        _ => {
            ones = slice.count_ones();
        }
    }

    if count_ones { ones } else { slice.len() - ones }
}
