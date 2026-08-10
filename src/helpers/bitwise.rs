//! Low-level bit and byte machinery shared by the `BitCollection` operations.
//!
//! Bulk operations work on raw `&[u8]` storage plus a bit offset. Nothing in
//! this module knows about `Tibs`, `Mutibs` or Python; the collection-level
//! logic that drives it lives in `core.rs`.

use super::bits::{BS, BV, BitAccumulator, head_bit_offset};
use super::raw_bytes::{copy_shifted_bytes, mask_padding_bits, reverse_padded_bits};
use super::splice::{copy_bits, move_bits};

// ---- Selecting a wider instruction set at run time ----------------------
//
// The x86 wheels are built for the base architecture, which has neither
// `popcnt` nor AVX. `count_ones` therefore compiles to a software population
// count and the byte loops vectorise no wider than 128 bits. Building the same
// source at `x86-64-v3` and timing 8 Mbit operands showed what that costs:
//
//     count()      4.1x     count_and()  3.7x     count_xor()  3.8x
//     &, |, ^      1.8x     ~            1.8x     ==           1.3x
//
// A wheel cannot simply be built that way - it would fault on any CPU older
// than 2013 - so each kernel that carries the difference is compiled twice and
// the pair selected between here, once, on what the CPU actually has.
//
// Only x86 is gated. On aarch64 NEON is in the baseline the wheels already
// target, population count included, so the single compilation is already the
// wide one and none of this is built at all.

/// Whether this CPU has the instruction set the `_wide` kernels are compiled
/// for.
///
/// `is_x86_feature_detected!` caches its own answer, but behind a call that
/// does not inline. These kernels are entered on operands short enough for
/// that to show, so the answer is cached again here where the branch can be
/// predicted and the load folded into the surrounding code.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn wide_simd() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    /// Holds `2` until the first probe, then `0` or `1`. Relaxed throughout:
    /// racing threads compute the same answer from the same CPU.
    static AVAILABLE: AtomicU8 = AtomicU8::new(2);
    match AVAILABLE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let found = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("popcnt");
            AVAILABLE.store(found as u8, Ordering::Relaxed);
            found
        }
    }
}

/// Below this many live bits, a kernel is left to its baseline compilation
/// whatever the CPU can do.
///
/// A `#[target_feature]` function cannot be inlined into a caller that is not
/// one, so reaching a widened kernel always costs a real call where the
/// baseline can be folded into its caller outright. Over a few bytes there are
/// too few words for the wider registers to win that back: measured on
/// operands of eight to sixty-four bits, dispatching cost five to ten
/// nanoseconds against calls of about a hundred, and only past a few hundred
/// bits did the wide kernels pull ahead.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const WIDE_MIN_BITS: usize = 512;

/// Whether a run of `len_bits` should go to the widened kernels.
///
/// The length is tested first and deliberately: it is already in a register,
/// where the feature flag is a load, so a short operand never touches it.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn use_wide(len_bits: usize) -> bool {
    len_bits >= WIDE_MIN_BITS && wide_simd()
}

/// Whether this CPU has `pext`/`pdep`, the gather and scatter instructions that
/// [`compress_bits`] and [`expand_bits`] emulate.
///
/// Cached the same way, and for the same reason, as [`wide_simd`]. Separate
/// from it because BMI2 and AVX2 are separate features: a CPU can have either
/// without the other.
#[cfg(target_arch = "x86_64")]
#[inline]
fn bmi2_available() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static AVAILABLE: AtomicU8 = AtomicU8::new(2);
    match AVAILABLE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let found = is_x86_feature_detected!("bmi2");
            AVAILABLE.store(found as u8, Ordering::Relaxed);
            found
        }
    }
}

/// Whether a masked gather or scatter over `len_bits` should use `pext`/`pdep`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn use_bmi2(len_bits: usize) -> bool {
    len_bits >= WIDE_MIN_BITS && bmi2_available()
}

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

    /// Append `count` copies of the same bit run.
    ///
    /// After writing the first copy, repeatedly duplicate the completed
    /// prefix. This keeps the number of moves logarithmic in `count` while
    /// moving each result byte only once overall.
    pub(crate) fn push_repeated_run(
        &mut self,
        src: &[u8],
        offset: usize,
        len: usize,
        count: usize,
    ) {
        if count == 0 || len == 0 {
            return;
        }

        let start = self.length;
        let total = len * count;
        self.push_run(src, offset, len);
        self.bytes.resize((start + total).div_ceil(8), 0);

        let mut completed = len;
        while completed <= total / 2 {
            move_bits(&mut self.bytes, start, start + completed, completed);
            completed *= 2;
        }
        if completed < total {
            move_bits(&mut self.bytes, start, start + completed, total - completed);
        }
        self.length = start + total;
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

/// Below this many live bits, a mismatched operand is shifted word by word
/// inside the loop rather than realigned up front.
///
/// Realigning buys a vectorisable loop at the cost of an allocation and one
/// extra sequential pass, so it only pays once the loop is long enough to
/// absorb them. Measured either way over a size sweep of `count_and`, in
/// nanoseconds per call:
///
/// ```text
///  bits    512   1024   2048   4096   8192  16384
///  shift   213    271    338    494    849   1339
///  realign 275    302    403    398    409    562
/// ```
///
/// The two cross between two and four kilobits, so the threshold sits at the
/// first size where realigning is clearly ahead rather than at the crossing
/// itself. Below it the fixed cost - an allocation, and a pass that the short
/// loop would not otherwise make - is most of the call.
const REALIGN_MIN_BITS: usize = 4096;

/// `rhs`'s live bits copied into a fresh buffer shaped like `lhs`'s, sitting at
/// `lhs_offset` so that the two can be read as a matching pair.
///
/// Every kernel below splits on whether its operands hold their bits at the
/// same offset in their storage. The matching case reads native-endian words
/// and vectorises; the mismatched case shifts each word into place inside the
/// loop, behind a byte swap and a three-way branch, and neither of those
/// vectorises - which measured six to eight times dearer over a megabit.
/// Shifting once up front and then taking the matching path costs one extra
/// sequential pass and wins most of that back.
fn realigned_to(
    lhs_len: usize,
    lhs_offset: usize,
    rhs: &[u8],
    rhs_offset: usize,
    len: usize,
) -> Vec<u8> {
    debug_assert_eq!(lhs_len, (lhs_offset + len).div_ceil(8));
    let mut out = vec![0u8; lhs_len];
    copy_bits(&mut out, lhs_offset, rhs, rhs_offset, len);
    out
}

/// Whether realigning `rhs` onto `lhs`'s offset is worth it here.
#[inline]
fn worth_realigning(lhs_offset: usize, rhs_offset: usize, len: usize) -> bool {
    lhs_offset != rhs_offset && len >= REALIGN_MIN_BITS
}

#[inline]
pub(crate) fn logical_op_with_matching_bytes(lhs: &[u8], rhs: &[u8], op: LogicalOp) -> Vec<u8> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_wide(lhs.len() * 8) {
        // SAFETY: `use_wide` reported the features this is compiled for.
        return unsafe { logical_op_with_matching_bytes_wide(lhs, rhs, op) };
    }
    logical_op_with_matching_bytes_impl(lhs, rhs, op)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "popcnt,avx2")]
unsafe fn logical_op_with_matching_bytes_wide(lhs: &[u8], rhs: &[u8], op: LogicalOp) -> Vec<u8> {
    logical_op_with_matching_bytes_impl(lhs, rhs, op)
}

/// `#[inline(always)]`, here and on every other `_impl` below, is what makes
/// the second compilation worth anything: `#[target_feature]` widens only what
/// is inlined into it, so a body left as an out-of-line call would be compiled
/// once, at baseline, and both wrappers would reach the same narrow code.
#[inline(always)]
fn logical_op_with_matching_bytes_impl(lhs: &[u8], rhs: &[u8], op: LogicalOp) -> Vec<u8> {
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
    len: usize,
    op: LogicalOp,
) -> Vec<u8> {
    debug_assert!(lhs_offset < 8);
    debug_assert!(rhs_offset < 8);

    // The result outside the live range is discarded by the caller, so the
    // zeros the realigned copy carries there are as good as the shifted bits
    // this would otherwise compute.
    if worth_realigning(lhs_offset, rhs_offset, len) {
        let rhs = realigned_to(lhs.len(), lhs_offset, rhs, rhs_offset, len);
        return logical_op_with_matching_bytes(lhs, &rhs, op);
    }

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
    // An in-place op writes every live bit, so realigning is never wasted.
    // See `realigned_to`.
    if worth_realigning(lhs_offset, rhs_offset, len) {
        let rhs = realigned_to(lhs.len(), lhs_offset, rhs, rhs_offset, len);
        return logical_op_assign_bytes(lhs, lhs_offset, &rhs, lhs_offset, len, op);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let wide = use_wide(len);
    macro_rules! applied {
        ($byte_op:expr) => {{
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if wide {
                // SAFETY: `use_wide` reported the features this is compiled for.
                return unsafe {
                    logical_op_assign_bytes_with_wide(
                        lhs, lhs_offset, rhs, rhs_offset, len, $byte_op,
                    )
                };
            }
            logical_op_assign_bytes_with(lhs, lhs_offset, rhs, rhs_offset, len, $byte_op)
        }};
    }
    match op {
        LogicalOp::Or => applied!(|a, b| a | b),
        LogicalOp::And => applied!(|a, b| a & b),
        LogicalOp::Xor => applied!(|a, b| a ^ b),
        LogicalOp::AndNot => applied!(|a, b| a & !b),
    }
}

/// See [`count_pair_bits_with_wide`] for why the choice is made inside each arm
/// of the match rather than around it.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "popcnt,avx2")]
unsafe fn logical_op_assign_bytes_with_wide<F>(
    lhs: &mut [u8],
    lhs_offset: usize,
    rhs: &[u8],
    rhs_offset: usize,
    len: usize,
    byte_op: F,
) where
    F: Fn(u8, u8) -> u8,
{
    logical_op_assign_bytes_with(lhs, lhs_offset, rhs, rhs_offset, len, byte_op)
}

#[inline(always)]
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
    // A count always reads every bit, so realigning can never be wasted work.
    // See `realigned_to`.
    if worth_realigning(lhs_offset, rhs_offset, len) {
        let rhs = realigned_to(lhs.len(), lhs_offset, rhs, rhs_offset, len);
        return count_pair_bits(lhs, lhs_offset, &rhs, lhs_offset, len, op);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let wide = use_wide(len);
    macro_rules! counted {
        ($word:expr, $byte:expr) => {{
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if wide {
                // SAFETY: `use_wide` reported the features this is compiled for.
                return unsafe {
                    count_pair_bits_with_wide(lhs, lhs_offset, rhs, rhs_offset, len, $word, $byte)
                };
            }
            count_pair_bits_with(lhs, lhs_offset, rhs, rhs_offset, len, $word, $byte)
        }};
    }
    match op {
        LogicalOp::Or => counted!(|a, b| a | b, |a, b| a | b),
        LogicalOp::And => counted!(|a, b| a & b, |a, b| a & b),
        LogicalOp::Xor => counted!(|a, b| a ^ b, |a, b| a ^ b),
        LogicalOp::AndNot => counted!(|a, b| a & !b, |a, b| a & !b),
    }
}

/// The widened copy is selected inside each arm of the match rather than around
/// it, so that the two compilations hold one monomorphisation of the loop per
/// operation between them.
///
/// Hoisting the choice above the match reads better and cost `count_or` and
/// `count_xor` a factor of three: it puts all four operations in both
/// compilations, eight bodies of an `#[inline(always)]` loop in one function,
/// and past that much inlining the optimiser stopped vectorising some of them.
/// `count_and` stayed fast throughout, which is what makes this worth a comment
/// - the shape looks harmless and only two of the four arms show the damage.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "popcnt,avx2")]
unsafe fn count_pair_bits_with_wide<W, B>(
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
    count_pair_bits_with(lhs, lhs_offset, rhs, rhs_offset, len, word_op, byte_op)
}

#[inline(always)]
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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let wide = use_wide(len);
    macro_rules! tested {
        ($word:expr, $byte:expr) => {{
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if wide {
                // SAFETY: `use_wide` reported the features this is compiled for.
                return unsafe {
                    any_pair_bits_with_wide(lhs, lhs_offset, rhs, rhs_offset, len, $word, $byte)
                };
            }
            any_pair_bits_with(lhs, lhs_offset, rhs, rhs_offset, len, $word, $byte)
        }};
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
    let word_op = |word: u64, _| if value { word } else { !word };
    let byte_op = |byte: u8, _| if value { byte } else { !byte };
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_wide(len) {
        // SAFETY: `use_wide` reported the features this is compiled for.
        return unsafe {
            any_pair_bits_with_wide(bytes, bit_offset, bytes, bit_offset, len, word_op, byte_op)
        };
    }
    any_pair_bits_with(bytes, bit_offset, bytes, bit_offset, len, word_op, byte_op)
}

/// See [`count_pair_bits_with_wide`] for why the choice is made inside each arm
/// of the match rather than around it.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "popcnt,avx2")]
unsafe fn any_pair_bits_with_wide<W, B>(
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
    any_pair_bits_with(lhs, lhs_offset, rhs, rhs_offset, len, word_op, byte_op)
}

#[inline(always)]
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

/// [`compress_bits`], or the single instruction that does the same thing.
///
/// `PEXT` is a constant so the branch folds away and each loop below is
/// compiled once per answer rather than testing per word.
///
/// SAFETY of the intrinsic: `PEXT` is only ever instantiated as `true` behind a
/// [`bmi2_available`] check - by [`extract_masked_bytes_pext`] in this module,
/// and by the agreement test below - so the instruction exists on every CPU
/// that reaches it. That holds through the call graph and does not depend on
/// this being inlined.
#[inline(always)]
fn compress_with<const PEXT: bool>(x: u64, m: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if PEXT {
        return unsafe { std::arch::x86_64::_pext_u64(x, m) };
    }
    compress_bits(x, m)
}

/// [`expand_bits`], or the single instruction that does the same thing. The
/// scattering counterpart to [`compress_with`]; see it for the safety argument.
#[inline(always)]
fn expand_with<const PDEP: bool>(x: u64, m: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if PDEP {
        return unsafe { std::arch::x86_64::_pdep_u64(x, m) };
    }
    expand_bits(x, m)
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
    #[cfg(target_arch = "x86_64")]
    if use_bmi2(len_bits) {
        // SAFETY: `use_bmi2` reported the feature this is compiled for.
        return unsafe { extract_masked_bytes_pext(src, mask, len_bits, ones) };
    }
    extract_masked_bytes_impl::<false>(src, mask, len_bits, ones)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn extract_masked_bytes_pext(src: &[u8], mask: &[u8], len_bits: usize, ones: usize) -> BV {
    extract_masked_bytes_impl::<true>(src, mask, len_bits, ones)
}

#[inline(always)]
fn extract_masked_bytes_impl<const PEXT: bool>(
    src: &[u8],
    mask: &[u8],
    len_bits: usize,
    ones: usize,
) -> BV {
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
            // Kept even when `pext` is available. Picking one or two bits out
            // singly is a handful of operations either way, and measured a few
            // percent ahead of the instruction on a mask that sparse - unlike
            // the scattering side, where `place_bits` loses to `pdep` outright.
            out.push(pick_bits(s, m, selected), selected);
        } else {
            out.push_wide(compress_with::<PEXT>(s, m), selected);
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
            out.push_wide(compress_with::<PEXT>(s, m), m.count_ones() as usize);
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
    #[cfg(target_arch = "x86_64")]
    if use_bmi2(len_bits) {
        // SAFETY: `use_bmi2` reported the feature this is compiled for.
        return unsafe {
            deposit_masked_bytes_pdep(
                bits,
                bits_offset,
                value,
                value_offset,
                value_len,
                mask,
                mask_offset,
                len_bits,
            )
        };
    }
    deposit_masked_bytes_impl::<false>(
        bits,
        bits_offset,
        value,
        value_offset,
        value_len,
        mask,
        mask_offset,
        len_bits,
    )
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "bmi2")]
unsafe fn deposit_masked_bytes_pdep(
    bits: &mut [u8],
    bits_offset: usize,
    value: &[u8],
    value_offset: usize,
    value_len: usize,
    mask: &[u8],
    mask_offset: usize,
    len_bits: usize,
) {
    deposit_masked_bytes_impl::<true>(
        bits,
        bits_offset,
        value,
        value_offset,
        value_len,
        mask,
        mask_offset,
        len_bits,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn deposit_masked_bytes_impl<const PDEP: bool>(
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
            } else if !PDEP && selected <= SPARSE_WORD_BITS {
                // `read_bit_run` right-aligns a short run, while `place_bits`
                // expects word positions. Shift both into the same frame. Only
                // worth it against the software expand; see `extract`'s twin.
                place_bits(packed, selector << (64 - run_len), selected) >> (64 - run_len)
            } else {
                expand_with::<PDEP>(packed, selector)
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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_wide(bytes.len() * 8) {
        // SAFETY: `use_wide` reported the features this is compiled for.
        return unsafe { count_ones_in_bytes_wide(bytes) };
    }
    count_ones_in_bytes_impl(bytes)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "popcnt,avx2")]
unsafe fn count_ones_in_bytes_wide(bytes: &[u8]) -> usize {
    count_ones_in_bytes_impl(bytes)
}

#[inline(always)]
fn count_ones_in_bytes_impl(bytes: &[u8]) -> usize {
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

/// The widened kernels are a second compilation of the same source, so they
/// cannot disagree with the baseline over what the source says. What they can
/// disagree over is the dispatch around them: a wrapper handed its arguments in
/// the wrong order, or one left behind when the function it shadows is edited.
///
/// Nothing else catches that. Every CI runner and every developer machine of
/// the last decade takes the wide path, so the baseline compilation is the one
/// that never runs - right up until it runs on a user's older CPU, which is the
/// only reason it exists.
#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod tests {
    use super::*;

    /// A deterministic spread of bytes. The kernels only care that the bits
    /// vary, not how, so a small congruential generator saves pulling `rand`
    /// into a test that would gain nothing from it.
    fn pattern(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect()
    }

    #[test]
    fn wide_kernels_agree_with_their_baseline() {
        if !wide_simd() {
            // This CPU runs the baseline everywhere, so there is no second
            // compilation to compare it against.
            return;
        }
        // Lengths either side of a byte, of a word, of the eight-word block the
        // predicate scan folds over, and of the point where a mismatched
        // operand starts being realigned instead of shifted.
        for len in [
            1usize, 7, 8, 9, 15, 63, 64, 65, 127, 511, 512, 513, 1023, 1024, 4095, 4096, 4103,
        ] {
            for lhs_offset in 0usize..8 {
                for rhs_offset in 0usize..8 {
                    let (lo, ro) = (lhs_offset, rhs_offset);
                    let lhs = pattern(len as u64 * 31 + lo as u64, (lo + len).div_ceil(8));
                    let rhs = pattern(len as u64 * 17 + ro as u64 + 9, (ro + len).div_ceil(8));

                    macro_rules! assert_agrees {
                        ($name:expr, $word:expr, $byte:expr, $assign:expr) => {{
                            let base = count_pair_bits_with(&lhs, lo, &rhs, ro, len, $word, $byte);
                            // SAFETY: guarded by `wide_simd` above.
                            let wide = unsafe {
                                count_pair_bits_with_wide(&lhs, lo, &rhs, ro, len, $word, $byte)
                            };
                            assert_eq!(base, wide, "count {} len={len} {lo} {ro}", $name);

                            let base = any_pair_bits_with(&lhs, lo, &rhs, ro, len, $word, $byte);
                            // SAFETY: guarded by `wide_simd` above.
                            let wide = unsafe {
                                any_pair_bits_with_wide(&lhs, lo, &rhs, ro, len, $word, $byte)
                            };
                            assert_eq!(base, wide, "any {} len={len} {lo} {ro}", $name);

                            let mut base = lhs.clone();
                            let mut wide = lhs.clone();
                            logical_op_assign_bytes_with(&mut base, lo, &rhs, ro, len, $assign);
                            // SAFETY: guarded by `wide_simd` above.
                            unsafe {
                                logical_op_assign_bytes_with_wide(
                                    &mut wide, lo, &rhs, ro, len, $assign,
                                )
                            };
                            assert_eq!(base, wide, "assign {} len={len} {lo} {ro}", $name);
                        }};
                    }
                    assert_agrees!(
                        "or",
                        |a: u64, b: u64| a | b,
                        |a: u8, b: u8| a | b,
                        |a: u8, b: u8| a | b
                    );
                    assert_agrees!(
                        "and",
                        |a: u64, b: u64| a & b,
                        |a: u8, b: u8| a & b,
                        |a: u8, b: u8| a & b
                    );
                    assert_agrees!(
                        "xor",
                        |a: u64, b: u64| a ^ b,
                        |a: u8, b: u8| a ^ b,
                        |a: u8, b: u8| a ^ b
                    );
                    assert_agrees!(
                        "andnot",
                        |a: u64, b: u64| a & !b,
                        |a: u8, b: u8| a & !b,
                        |a: u8, b: u8| a & !b
                    );

                    if lhs.len() == rhs.len() {
                        for op in [
                            LogicalOp::Or,
                            LogicalOp::And,
                            LogicalOp::Xor,
                            LogicalOp::AndNot,
                        ] {
                            let base = logical_op_with_matching_bytes_impl(&lhs, &rhs, op);
                            // SAFETY: guarded by `wide_simd` above.
                            let wide =
                                unsafe { logical_op_with_matching_bytes_wide(&lhs, &rhs, op) };
                            assert_eq!(base, wide, "logical_op_with_matching_bytes len={len}");
                        }
                    }
                }
            }
            let bytes = pattern(len as u64, len.div_ceil(8));
            // SAFETY: guarded by `wide_simd` above.
            assert_eq!(
                count_ones_in_bytes_impl(&bytes),
                unsafe { count_ones_in_bytes_wide(&bytes) },
                "count_ones_in_bytes len={len}"
            );
        }
    }

    /// The number of set bits in the `len` bits starting `offset` bits in.
    #[cfg(target_arch = "x86_64")]
    fn count_run(bytes: &[u8], offset: usize, len: usize) -> usize {
        (0..len)
            .filter(|index| {
                let at = offset + index;
                bytes[at / 8] & (0x80u8 >> (at % 8)) != 0
            })
            .count()
    }

    /// `pext`/`pdep` replace a parallel-prefix emulation of themselves, so
    /// unlike the widened kernels these really are two different algorithms and
    /// could disagree on their own terms - not only through the dispatch. The
    /// selecting branch differs too: with the instruction available the sparse
    /// path is skipped, so the two instantiations do not even take the same
    /// route through the loop.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn pext_and_pdep_agree_with_the_software_forms() {
        if !bmi2_available() {
            return;
        }
        // The primitives first, over masks that are empty, full, sparse, dense
        // and arbitrary, since those are what select between the loop's arms.
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..4096 {
            let x = next();
            let m = match round % 8 {
                0 => 0,
                1 => u64::MAX,
                2 => 1 << (round % 64),
                3 => next() & next() & next(), // sparse
                4 => next() | next() | next(), // dense
                _ => next(),
            };
            // Instantiating the intrinsic arm is only sound behind the
            // `bmi2_available` check above.
            assert_eq!(compress_bits(x, m), compress_with::<true>(x, m));
            assert_eq!(expand_bits(x, m), expand_with::<true>(x, m));
        }

        // Then the whole loops, which choose different arms per instantiation.
        for len in [1usize, 63, 64, 65, 127, 512, 513, 1000, 4096, 4103] {
            let bytes = len.div_ceil(8);
            let src = pattern(len as u64 * 7 + 1, bytes);
            for mask in [
                pattern(len as u64 * 11 + 2, bytes),
                vec![0u8; bytes],
                vec![0xffu8; bytes],
                vec![0x80u8; bytes], // one bit per word group, the sparse arm
            ] {
                let ones = count_run(&mask, 0, len);
                let base = extract_masked_bytes_impl::<false>(&src, &mask, len, ones);
                // SAFETY: guarded by `bmi2_available` above.
                let wide = unsafe { extract_masked_bytes_pext(&src, &mask, len, ones) };
                assert_eq!(base, wide, "extract len={len} ones={ones}");

                for offset in 0usize..8 {
                    let store = len.div_ceil(8) + 1;
                    let selected = count_run(&mask, 0, len);
                    let value = pattern(len as u64 * 13 + offset as u64, store);
                    let mut base = pattern(len as u64 * 17, store);
                    let mut wide = base.clone();
                    deposit_masked_bytes_impl::<false>(
                        &mut base, offset, &value, 0, selected, &mask, 0, len,
                    );
                    // SAFETY: guarded by `bmi2_available` above.
                    unsafe {
                        deposit_masked_bytes_pdep(
                            &mut wide, offset, &value, 0, selected, &mask, 0, len,
                        )
                    };
                    assert_eq!(base, wide, "deposit len={len} offset={offset}");
                }
            }
        }
    }
}
