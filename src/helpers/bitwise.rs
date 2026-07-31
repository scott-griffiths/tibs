//! Low-level bit and byte machinery shared by the `BitCollection` operations.
//!
//! Bulk operations work on raw `&[u8]` storage plus a bit offset. Nothing in
//! this module knows about `Tibs`, `Mutibs` or Python; the collection-level
//! logic that drives it lives in `core.rs`.

use super::bits::{BS, BV, BitAccumulator, head_bit_offset};
use super::raw_bytes::{copy_shifted_bytes, mask_padding_bits, reverse_padded_bits};
use super::splice::{copy_bits, move_bits};

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

/// The `len_bits` bits starting `bit_offset` bits into `bytes`, returned as
/// left aligned bytes with any trailing padding cleared.
///
/// `bytes` must hold at least `(bit_offset + len_bits)` bits.
pub(crate) fn padded_bytes_from_offset(
    bytes: &[u8],
    bit_offset: usize,
    len_bits: usize,
) -> Vec<u8> {
    debug_assert!(bit_offset < 8);
    if len_bits == 0 {
        return Vec::new();
    }
    let byte_length = len_bits.div_ceil(8);
    debug_assert!(bytes.len() >= byte_length);
    if bit_offset == 0 {
        let mut out = bytes[..byte_length].to_vec();
        mask_padding_bits(&mut out, len_bits);
        return out;
    }
    let mut out = vec![0u8; byte_length];
    copy_unaligned_padded_bytes(bytes, bit_offset, len_bits, &mut out);
    out
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

/// Rotate the `len` bits starting `offset` bits into `bytes` left by `by`.
///
/// `BitSlice::rotate_left` carries at most one word per pass and does a
/// full-width `copy_within` on every one of them, so it costs `O(len * by)`:
/// rotating a megabit by half its length takes seconds. Rotating is really
/// just a three-way move, so this costs `O(len)` however far it rotates.
///
/// The bits outside the rotated span are left alone, so this is safe to point
/// at a range inside a larger buffer.
pub(crate) fn rotate_bits_left(bytes: &mut [u8], offset: usize, len: usize, by: usize) {
    debug_assert!(by <= len);
    debug_assert!(offset + len <= bytes.len() * 8);
    if by == 0 || by == len {
        return;
    }
    // Whichever of the two pieces is smaller becomes the temporary, so the
    // extra allocation is at most half the span however lopsided the rotation.
    let wrapping = by.min(len - by);
    let mut saved = vec![0u8; wrapping.div_ceil(8)];
    if by <= len - by {
        // The leading `by` bits wrap round to the end: stash them, slide the
        // rest down over where they were, then write them back at the end.
        copy_bits(&mut saved, 0, bytes, offset, by);
        move_bits(bytes, offset + by, offset, len - by);
        copy_bits(bytes, offset + len - by, &saved, 0, by);
    } else {
        // The same rotation read from the other end: the trailing `len - by`
        // bits move to the front, so stash those instead.
        let tail = len - by;
        copy_bits(&mut saved, 0, bytes, offset + by, tail);
        move_bits(bytes, offset, offset + tail, by);
        copy_bits(bytes, offset, &saved, 0, tail);
    }
}

/// Lays bit runs end to end into one buffer.
///
/// Growing a `BitVec` with `extend_from_bitslice` per piece copies a bit at a
/// time. This moves whole bytes instead, which is the difference between
/// memcpy speed and roughly fifty times slower.
///
/// A run that starts on a byte boundary and comes from byte-aligned storage -
/// the common case, and the whole of an aligned concatenation - is appended
/// without the buffer ever being zeroed. Only a run landing part way through a
/// byte pays for that, and only over its own bytes.
pub(crate) struct BitConcat {
    bytes: Vec<u8>,
    length: usize,
}

impl BitConcat {
    pub(crate) fn with_bit_capacity(total_bits: usize) -> Self {
        BitConcat {
            bytes: Vec::with_capacity(total_bits.div_ceil(8)),
            length: 0,
        }
    }

    /// Append the `len` bits starting `offset` bits into `src`.
    pub(crate) fn push_run(&mut self, src: &[u8], offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        debug_assert!(offset < 8);
        debug_assert!(src.len() >= (offset + len).div_ceil(8));
        if offset == 0 && self.length.is_multiple_of(8) {
            debug_assert_eq!(self.bytes.len(), self.length / 8);
            // Both ends are on byte boundaries, so this is a straight copy and
            // nothing has to be cleared first.
            let whole = len / 8;
            self.bytes.extend_from_slice(&src[..whole]);
            let tail = len - whole * 8;
            if tail > 0 {
                // Drop the source's padding; a later run overwrites those bits.
                self.bytes.push(src[whole] & (!0u8 << (8 - tail)));
            }
        } else {
            let end = self.length + len;
            // Only the bytes this run reaches into need to exist yet.
            self.bytes.resize(end.div_ceil(8), 0);
            copy_bits(&mut self.bytes, self.length, src, offset, len);
        }
        self.length += len;
    }

    pub(crate) fn into_bitvec(self) -> BV {
        let mut bv = BV::from_vec(self.bytes);
        bv.truncate(self.length);
        bv
    }
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
    pub(crate) fn word(self, lhs: u64, rhs: u64) -> u64 {
        match self {
            LogicalOp::Or => lhs | rhs,
            LogicalOp::And => lhs & rhs,
            LogicalOp::Xor => lhs ^ rhs,
            LogicalOp::AndNot => lhs & !rhs,
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

/// Apply a logical operation to the `len` live bits of `lhs` in place.
///
/// Each slice begins with the byte containing its first live bit. The offsets
/// say where that bit sits in the byte, and may differ. Bits outside the live
/// range in the first and last `lhs` bytes are preserved.
#[inline]
pub(crate) fn logical_op_assign_bytes(
    lhs: &mut [u8],
    lhs_offset: usize,
    rhs: &[u8],
    rhs_offset: usize,
    len: usize,
    op: LogicalOp,
) {
    macro_rules! applied {
        ($byte_op:expr) => {
            logical_op_assign_bytes_with(lhs, lhs_offset, rhs, rhs_offset, len, $byte_op)
        };
    }
    match op {
        LogicalOp::Or => applied!(|a, b| a | b),
        LogicalOp::And => applied!(|a, b| a & b),
        LogicalOp::Xor => applied!(|a, b| a ^ b),
        LogicalOp::AndNot => applied!(|a, b| a & !b),
    }
}

fn logical_op_assign_bytes_with<F>(
    lhs: &mut [u8],
    lhs_offset: usize,
    rhs: &[u8],
    rhs_offset: usize,
    len: usize,
    byte_op: F,
) where
    F: Fn(u8, u8) -> u8,
{
    debug_assert!(lhs_offset < 8);
    debug_assert!(rhs_offset < 8);
    if len == 0 {
        return;
    }
    debug_assert_eq!(lhs.len(), (lhs_offset + len).div_ceil(8));
    debug_assert_eq!(rhs.len(), (rhs_offset + len).div_ceil(8));

    let last = lhs.len() - 1;
    let rhs_shift = rhs_offset as isize - lhs_offset as isize;
    let (head_mask, tail_mask) = edge_masks(lhs_offset, len);

    let old = lhs[0];
    let mask = if last == 0 {
        head_mask & tail_mask
    } else {
        head_mask
    };
    let updated = byte_op(old, align_byte(rhs, 0, rhs_shift));
    lhs[0] = (old & !mask) | (updated & mask);
    if last == 0 {
        return;
    }

    if rhs_shift == 0 {
        debug_assert_eq!(lhs.len(), rhs.len());
        for (left, &right) in lhs[1..last].iter_mut().zip(&rhs[1..last]) {
            *left = byte_op(*left, right);
        }
    } else {
        for (index, left) in lhs[1..last].iter_mut().enumerate() {
            let index = index + 1;
            *left = byte_op(*left, align_byte(rhs, index, rhs_shift));
        }
    }

    let old = lhs[last];
    let updated = byte_op(old, align_byte(rhs, last, rhs_shift));
    lhs[last] = (old & !tail_mask) | (updated & tail_mask);
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
/// The operation is dispatched here, once, so that each inner loop is compiled
/// with it inlined rather than re-matching an enum on every word.
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

/// How many words the predicate scan folds together before testing for a hit.
///
/// Testing each word on its own puts a branch in the inner loop, which stops
/// the compiler vectorising it and costs the predicates several times the
/// throughput of the counting path over the same data. Folding a block of words
/// together with `or` and testing once keeps the loop straight, and the early
/// exit only loses whatever is left of the block it is in: at most 512 bits of
/// extra scanning, against a full pass over the operands saved.
const ANY_BLOCK_WORDS: usize = 8;
const ANY_BLOCK_BYTES: usize = ANY_BLOCK_WORDS * 8;

/// Whether `op(lhs, rhs)` has any set bit in the live range, stopping early.
///
/// The counterpart to [`count_pair_bits`], and dispatched the same way: the
/// operation is matched here, once, so that each inner loop is compiled with it
/// inlined instead of re-matching an enum on every word.
#[inline]
pub(crate) fn any_pair_bits(
    lhs: &[u8],
    lhs_offset: usize,
    rhs: &[u8],
    rhs_offset: usize,
    len: usize,
    op: LogicalOp,
) -> bool {
    macro_rules! tested {
        ($word:expr, $byte:expr) => {
            any_pair_bits_with(lhs, lhs_offset, rhs, rhs_offset, len, $word, $byte)
        };
    }
    match op {
        LogicalOp::Or => tested!(|a, b| a | b, |a, b| a | b),
        LogicalOp::And => tested!(|a, b| a & b, |a, b| a & b),
        LogicalOp::Xor => tested!(|a, b| a ^ b, |a, b| a ^ b),
        LogicalOp::AndNot => tested!(|a, b| a & !b, |a, b| a & !b),
    }
}

/// Whether the live range contains `value`, stopping after the block that
/// contains the first match.
///
/// This is the unary form of [`any_pair_bits`]. Reusing its blocked word scan
/// keeps the full-scan case vectorisable while preserving early exit to within
/// 512 bits.
#[inline]
pub(crate) fn contains_bit(bytes: &[u8], bit_offset: usize, len: usize, value: bool) -> bool {
    any_pair_bits_with(
        bytes,
        bit_offset,
        bytes,
        bit_offset,
        len,
        |word, _| if value { word } else { !word },
        |byte, _| if value { byte } else { !byte },
    )
}

fn any_pair_bits_with<W, B>(
    lhs: &[u8],
    lhs_offset: usize,
    rhs: &[u8],
    rhs_offset: usize,
    len: usize,
    word_op: W,
    byte_op: B,
) -> bool
where
    W: Fn(u64, u64) -> u64,
    B: Fn(u8, u8) -> u8,
{
    debug_assert!(lhs_offset < 8);
    debug_assert!(rhs_offset < 8);
    if len == 0 {
        return false;
    }
    debug_assert_eq!(lhs.len(), (lhs_offset + len).div_ceil(8));
    let last = lhs.len() - 1;
    let rhs_shift = rhs_offset as isize - lhs_offset as isize;
    let (head_mask, tail_mask) = edge_masks(lhs_offset, len);
    let pair_byte = |index: usize| byte_op(lhs[index], align_byte(rhs, index, rhs_shift));

    if last == 0 {
        return pair_byte(0) & head_mask & tail_mask != 0;
    }
    // The two partial ends are the only bytes needing masking, so take them
    // first and leave the middle as a clean run that no branch has to skirt.
    if pair_byte(0) & head_mask != 0 || pair_byte(last) & tail_mask != 0 {
        return true;
    }

    if rhs_shift == 0 {
        // Byte order is irrelevant to a test against zero, so read
        // native-endian and skip the byte swap that the shifted path needs.
        debug_assert_eq!(lhs.len(), rhs.len());
        let (left, right) = (&lhs[1..last], &rhs[1..last]);
        let mut left_blocks = left.chunks_exact(ANY_BLOCK_BYTES);
        let mut right_blocks = right.chunks_exact(ANY_BLOCK_BYTES);
        for (left_block, right_block) in left_blocks.by_ref().zip(right_blocks.by_ref()) {
            let mut hits = 0u64;
            for (left_chunk, right_chunk) in
                left_block.chunks_exact(8).zip(right_block.chunks_exact(8))
            {
                hits |= word_op(
                    u64::from_ne_bytes(left_chunk.try_into().unwrap()),
                    u64::from_ne_bytes(right_chunk.try_into().unwrap()),
                );
            }
            if hits != 0 {
                return true;
            }
        }
        return any_pair_bytes(
            left_blocks.remainder(),
            right_blocks.remainder(),
            word_op,
            byte_op,
        );
    }

    // Storage at differing bit offsets: the same blocking, over words that have
    // to be shifted into line one at a time.
    let mut index = 1;
    while index + ANY_BLOCK_BYTES <= last && index + ANY_BLOCK_BYTES <= rhs.len() {
        let mut hits = 0u64;
        for word_index in (index..index + ANY_BLOCK_BYTES).step_by(8) {
            hits |= word_op(
                read_be_u64(lhs, word_index),
                aligned_rhs_word(rhs, word_index, rhs_shift),
            );
        }
        if hits != 0 {
            return true;
        }
        index += ANY_BLOCK_BYTES;
    }
    while index + 8 <= last && index + 8 <= rhs.len() {
        if word_op(
            read_be_u64(lhs, index),
            aligned_rhs_word(rhs, index, rhs_shift),
        ) != 0
        {
            return true;
        }
        index += 8;
    }
    while index < last {
        if pair_byte(index) != 0 {
            return true;
        }
        index += 1;
    }
    false
}

/// The under-a-block tail of a pair of byte runs held at the same bit offset,
/// tested a word at a time and then a byte at a time.
fn any_pair_bytes<W, B>(left: &[u8], right: &[u8], word_op: W, byte_op: B) -> bool
where
    W: Fn(u64, u64) -> u64,
    B: Fn(u8, u8) -> u8,
{
    debug_assert_eq!(left.len(), right.len());
    let mut left_chunks = left.chunks_exact(8);
    let mut right_chunks = right.chunks_exact(8);
    for (left_chunk, right_chunk) in left_chunks.by_ref().zip(right_chunks.by_ref()) {
        if word_op(
            u64::from_ne_bytes(left_chunk.try_into().unwrap()),
            u64::from_ne_bytes(right_chunk.try_into().unwrap()),
        ) != 0
        {
            return true;
        }
    }
    left_chunks
        .remainder()
        .iter()
        .zip(right_chunks.remainder())
        .any(|(&left_byte, &right_byte)| byte_op(left_byte, right_byte) != 0)
}

/// Move the bits of `x` picked out by `m` down to the low end, keeping their
/// order and clearing everything above them.
///
/// Hacker's Delight figure 7-9, "compress right". Six parallel-prefix rounds
/// move each selected bit past the cleared bits below it, so the whole word is
/// done branchlessly in a fixed number of operations however the mask falls.
/// x86 has this as a single `PEXT` instruction, but it needs runtime feature
/// detection, is famously slow on pre-Zen 3 AMD, and has no aarch64 equivalent,
/// so the portable version earns its place.
#[inline]
fn compress_bits(mut x: u64, mut m: u64) -> u64 {
    x &= m;
    let mut mk = !m << 1;
    for i in 0..6 {
        let mut mp = mk ^ (mk << 1);
        mp ^= mp << 2;
        mp ^= mp << 4;
        mp ^= mp << 8;
        mp ^= mp << 16;
        mp ^= mp << 32;
        let mv = mp & m;
        m = (m ^ mv) | (mv >> (1 << i));
        let t = x & mv;
        x = (x ^ t) | (t >> (1 << i));
        mk &= !mp;
    }
    x
}

/// Move the low bits of `x` up into the positions picked out by `m`, keeping
/// their order and clearing every position outside the mask.
///
/// This is Hacker's Delight's "expand right", the inverse of
/// [`compress_bits`]. The first pass records how compression would move each
/// group of mask bits down; the second applies those moves in reverse.
#[inline]
fn expand_bits(mut x: u64, mut m: u64) -> u64 {
    let original_mask = m;
    let mut moves = [0u64; 6];
    let mut mk = !m << 1;
    for (i, movement) in moves.iter_mut().enumerate() {
        let mut mp = mk ^ (mk << 1);
        mp ^= mp << 2;
        mp ^= mp << 4;
        mp ^= mp << 8;
        mp ^= mp << 16;
        mp ^= mp << 32;
        let mv = mp & m;
        *movement = mv;
        m = (m ^ mv) | (mv >> (1 << i));
        mk &= !mp;
    }
    for (i, &mv) in moves.iter().enumerate().rev() {
        let moved = x << (1 << i);
        x = (x & !mv) | (moved & mv);
    }
    x & original_mask
}

/// Above this many set bits in a word, [`compress_bits`] costs less than
/// picking the bits out one at a time.
///
/// Compress is a fixed six rounds however few bits it moves, while picking
/// costs a handful of instructions per bit, so the two cross over at a low
/// count. A sparse mask - one set bit every word or two, which is what a mask
/// built from scattered positions looks like - would otherwise pay the full
/// fixed cost per word to collect a single bit.
const SPARSE_WORD_BITS: usize = 8;

/// The bits of `s` picked out by `m`, in order, right-aligned in the low
/// `selected` bits. Walks the set positions, so it costs per set bit.
#[inline]
fn pick_bits(s: u64, m: u64, selected: usize) -> u64 {
    debug_assert_eq!(m.count_ones() as usize, selected);
    let mut rest = m;
    let mut value = 0u64;
    while rest != 0 {
        // The highest remaining set bit is the earliest in the run, so taking
        // them from the top builds the value in order.
        let from_top = rest.leading_zeros();
        value = (value << 1) | ((s >> (63 - from_top)) & 1);
        rest &= !(1u64 << (63 - from_top));
    }
    value
}

/// The inverse of [`pick_bits`]: scatter `selected` low bits of `value` into
/// the set positions of `mask`. Walking the positions is cheaper than
/// [`expand_bits`] for a sparse mask.
#[inline]
fn place_bits(value: u64, mask: u64, selected: usize) -> u64 {
    debug_assert_eq!(mask.count_ones() as usize, selected);
    let mut rest = mask;
    let mut source_bit = selected;
    let mut placed = 0u64;
    while rest != 0 {
        source_bit -= 1;
        let from_top = rest.leading_zeros();
        placed |= ((value >> source_bit) & 1) << (63 - from_top);
        rest &= !(1u64 << (63 - from_top));
    }
    placed
}

#[inline]
fn low_mask(bits: usize) -> u64 {
    debug_assert!(bits <= 64);
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Read at most one word from an arbitrary bit offset, with the first bit at
/// the high end of the returned run and the whole run right-aligned.
#[inline]
fn read_bit_run(bytes: &[u8], bit_offset: usize, len_bits: usize) -> u64 {
    debug_assert!((1..=64).contains(&len_bits));
    debug_assert!(bytes.len() * 8 >= bit_offset + len_bits);
    let byte_start = bit_offset / 8;
    let head = bit_offset % 8;
    if head == 0 && len_bits == 64 {
        return read_be_u64(bytes, byte_start);
    }

    let byte_count = (head + len_bits).div_ceil(8);
    let mut value = 0u128;
    for &byte in &bytes[byte_start..byte_start + byte_count] {
        value = (value << 8) | u128::from(byte);
    }
    let trailing = byte_count * 8 - head - len_bits;
    ((value >> trailing) as u64) & low_mask(len_bits)
}

/// Replace at most one word at an arbitrary bit offset, preserving the bits
/// before and after the run in its edge bytes.
#[inline]
fn write_bit_run(bytes: &mut [u8], bit_offset: usize, len_bits: usize, value: u64) {
    debug_assert!((1..=64).contains(&len_bits));
    debug_assert!(bytes.len() * 8 >= bit_offset + len_bits);
    let byte_start = bit_offset / 8;
    let head = bit_offset % 8;
    if head == 0 && len_bits == 64 {
        bytes[byte_start..byte_start + 8].copy_from_slice(&value.to_be_bytes());
        return;
    }

    let byte_count = (head + len_bits).div_ceil(8);
    let run = &mut bytes[byte_start..byte_start + byte_count];
    let mut stored = 0u128;
    for &byte in run.iter() {
        stored = (stored << 8) | u128::from(byte);
    }
    let trailing = byte_count * 8 - head - len_bits;
    let field_mask = ((1u128 << len_bits) - 1) << trailing;
    stored = (stored & !field_mask) | ((u128::from(value) << trailing) & field_mask);
    for byte in run.iter_mut().rev() {
        *byte = stored as u8;
        stored >>= 8;
    }
}

/// Gather the bits of `src` at the positions set in `mask`, packed together
/// (the PEXT operation).
///
/// `src` and `mask` are left-aligned runs holding `len_bits` bits each, and
/// `ones` is `mask`'s population count. Walking the mask's set positions and
/// pushing one bit at a time costs a bit-pointer round trip per set bit; this
/// takes a word at a time, and skips or copies whole words outright where the
/// mask is uniform, which is the shape most masks actually have.
pub(crate) fn extract_masked_bytes(src: &[u8], mask: &[u8], len_bits: usize, ones: usize) -> BV {
    debug_assert!(src.len() >= len_bits.div_ceil(8));
    debug_assert!(mask.len() >= len_bits.div_ceil(8));
    let mut out = BitAccumulator::with_bit_capacity(Some(ones));

    let whole_words = len_bits / 64;
    for index in 0..whole_words {
        let at = index * 8;
        // Big-endian reads because `Msb0` makes the first bit of the run the
        // most significant bit of the word, which is where compress wants it.
        let m = read_be_u64(mask, at);
        if m == 0 {
            continue;
        }
        if m == u64::MAX && out.is_byte_aligned() {
            // The whole word is selected and the output happens to sit on a
            // byte boundary, so these bytes go straight across untouched.
            out.push_aligned_bytes(&src[at..at + 8]);
            continue;
        }
        let s = read_be_u64(src, at);
        let selected = m.count_ones() as usize;
        if m == u64::MAX {
            out.push_wide(s, 64);
        } else if selected <= SPARSE_WORD_BITS {
            out.push(pick_bits(s, m, selected), selected);
        } else {
            out.push_wide(compress_bits(s, m), selected);
        }
    }

    // Fewer than 64 bits left over, so gather them into one final word.
    let done = whole_words * 64;
    if done < len_bits {
        let rest = len_bits - done;
        let byte_start = done / 8;
        let byte_count = rest.div_ceil(8);
        let mut source = [0u8; 8];
        let mut selector = [0u8; 8];
        source[..byte_count].copy_from_slice(&src[byte_start..byte_start + byte_count]);
        selector[..byte_count].copy_from_slice(&mask[byte_start..byte_start + byte_count]);
        // Drop anything past the end of the value, which the padding may hold.
        let m = u64::from_be_bytes(selector) & (!0u64 << (64 - rest));
        if m != 0 {
            let s = u64::from_be_bytes(source);
            out.push_wide(compress_bits(s, m), m.count_ones() as usize);
        }
    }

    out.into_bitvec()
}

/// Scatter `value`'s bits into the positions of `bits` where `mask` is set,
/// leaving the other bits untouched (the PDEP operation).
///
/// Each input is a bit run at an arbitrary offset in its byte storage. The
/// caller must ensure that the destination and mask are `len_bits` long and
/// that `value_len` equals the mask's population count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deposit_masked_bytes(
    bits: &mut [u8],
    bits_offset: usize,
    value: &[u8],
    value_offset: usize,
    value_len: usize,
    mask: &[u8],
    mask_offset: usize,
    len_bits: usize,
) {
    debug_assert!(bits.len() * 8 >= bits_offset + len_bits);
    debug_assert!(value.len() * 8 >= value_offset + value_len);
    debug_assert!(mask.len() * 8 >= mask_offset + len_bits);

    if value_len == 0 {
        return;
    }
    if value_len == len_bits {
        copy_bits(bits, bits_offset, value, value_offset, len_bits);
        return;
    }

    let mut done = 0;
    let mut consumed = 0;
    while done < len_bits {
        let run_len = (len_bits - done).min(64);
        let selector = read_bit_run(mask, mask_offset + done, run_len);
        let selected = selector.count_ones() as usize;
        if selected != 0 {
            let packed = read_bit_run(value, value_offset + consumed, selected);
            let placed = if selected == run_len {
                packed
            } else if selected <= SPARSE_WORD_BITS {
                // `read_bit_run` right-aligns a short run, while `place_bits`
                // expects word positions. Shift both into the same frame.
                place_bits(packed, selector << (64 - run_len), selected) >> (64 - run_len)
            } else {
                expand_bits(packed, selector)
            };
            let updated = if selected == run_len {
                placed
            } else {
                (read_bit_run(bits, bits_offset + done, run_len) & !selector) | placed
            };
            write_bit_run(bits, bits_offset + done, run_len, updated);
            consumed += selected;
        }
        done += run_len;
    }
    debug_assert_eq!(consumed, value_len);
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
