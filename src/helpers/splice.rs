//! Byte level bit moves for the operations that add or remove bits.
//!
//! `BitVec` inserts and removes one bit at a time, so editing a large buffer
//! costs microseconds per kilobit. Working over the raw storage instead makes
//! an edit cost a slide of the bits after it and nothing more: a `memmove`
//! when the replacement is a whole number of bytes, and otherwise a sweep
//! that carries each byte across the boundary by up to seven bits.
//!
//! Bit `n` of a buffer is bit `7 - (n % 8)` of byte `n / 8`, matching the
//! `Msb0` ordering used throughout, so a buffer reads as one big endian bit
//! stream.

/// Bytes staged per iteration of the sweeping loops. Copying a run into a
/// buffer of its own and back is what lets the shift vectorise, and is also
/// what makes a slide that overlaps its own source safe. This size is big
/// enough for the copy in and out to disappear next to the shift, and small
/// enough not to cost a short edit anything.
const CHUNK: usize = 256;

#[inline]
fn get_bit(bytes: &[u8], bit: usize) -> bool {
    (bytes[bit >> 3] >> (7 - (bit & 7))) & 1 == 1
}

#[inline]
fn set_bit(bytes: &mut [u8], bit: usize, value: bool) {
    let mask = 0x80u8 >> (bit & 7);
    if value {
        bytes[bit >> 3] |= mask;
    } else {
        bytes[bit >> 3] &= !mask;
    }
}

/// The eight bits of `bytes` starting at bit `offset`.
///
/// Bits past the end read as zero, so only the first bit has to be in bounds.
#[inline]
fn read_u8(bytes: &[u8], offset: usize) -> u8 {
    let index = offset >> 3;
    let shift = offset & 7;
    if shift == 0 {
        return bytes[index];
    }
    let next = bytes.get(index + 1).copied().unwrap_or(0);
    (bytes[index] << shift) | (next >> (8 - shift))
}

/// Write the top `count` bits of `value` over the start of `bytes[index]`,
/// leaving the rest of the byte alone. `count` must be in `1..8`.
#[inline]
fn write_partial_byte(bytes: &mut [u8], index: usize, value: u8, count: usize) {
    debug_assert!((1..8).contains(&count));
    let mask = !(0xffu8 >> count);
    bytes[index] = (bytes[index] & !mask) | (value & mask);
}

/// The number of bits from `offset` to the next byte boundary, at most `len`.
#[inline]
fn bits_to_boundary(offset: usize, len: usize) -> usize {
    ((8 - (offset & 7)) & 7).min(len)
}

/// Fill `out` with `window` shifted up by `shift` bits, so that `out[i]` holds
/// the eight bits starting `shift` bits into `window[i]`.
///
/// `window` has to be one byte longer than `out`, since every output byte
/// straddles two input bytes. Walking the two input runs as separate slices of
/// a known equal length is what gets this compiled to vector shifts rather
/// than a byte at a time loop.
#[inline]
fn shift_window(out: &mut [u8], window: &[u8], shift: usize) {
    debug_assert_eq!(window.len(), out.len() + 1);
    debug_assert!((1..8).contains(&shift));
    let back = 8 - shift;
    let (low, high) = (&window[..out.len()], &window[1..]);
    for ((byte, &first), &second) in out.iter_mut().zip(low).zip(high) {
        *byte = (first << shift) | (second >> back);
    }
}

/// Copy `len` bits from `src` at bit `src_offset` to `dst` at bit
/// `dst_offset`.
///
/// The two buffers must not overlap; use [`move_bits`] to slide bits around
/// within a single buffer.
pub(crate) fn copy_bits(
    dst: &mut [u8],
    dst_offset: usize,
    src: &[u8],
    src_offset: usize,
    len: usize,
) {
    if len == 0 {
        return;
    }
    debug_assert!(src_offset + len <= src.len() * 8);
    debug_assert!(dst_offset + len <= dst.len() * 8);

    // Bring the destination up to a byte boundary so the bulk of the copy
    // writes whole bytes.
    let lead = bits_to_boundary(dst_offset, len);
    for i in 0..lead {
        set_bit(dst, dst_offset + i, get_bit(src, src_offset + i));
    }

    let whole = (len - lead) >> 3;
    let target = (dst_offset + lead) >> 3;
    let source = (src_offset + lead) >> 3;
    let shift = (src_offset + lead) & 7;
    if shift == 0 {
        dst[target..target + whole].copy_from_slice(&src[source..source + whole]);
    } else {
        shift_window(
            &mut dst[target..target + whole],
            &src[source..source + whole + 1],
            shift,
        );
    }

    let done = lead + whole * 8;
    if done < len {
        let byte = read_u8(src, src_offset + done);
        write_partial_byte(dst, target + whole, byte, len - done);
    }
}

/// Move `len` bits within `bytes` from bit `src_offset` to bit `dst_offset`,
/// leaving every bit outside the destination range untouched.
/// Set the `len` bits starting at `offset` to `value`.
///
/// Whole bytes go out with a `memset`; only the up-to-seven bits at each end
/// are written singly. `BitSlice::fill` walks every bit through a bit pointer.
pub(crate) fn fill_bits(bytes: &mut [u8], offset: usize, len: usize, value: bool) {
    if len == 0 {
        return;
    }
    debug_assert!(offset + len <= bytes.len() * 8);

    let lead = bits_to_boundary(offset, len);
    for i in 0..lead {
        set_bit(bytes, offset + i, value);
    }

    let whole = (len - lead) >> 3;
    let first = (offset + lead) >> 3;
    bytes[first..first + whole].fill(if value { !0u8 } else { 0u8 });

    for i in lead + whole * 8..len {
        set_bit(bytes, offset + i, value);
    }
}

pub(crate) fn move_bits(bytes: &mut [u8], src_offset: usize, dst_offset: usize, len: usize) {
    if len == 0 || src_offset == dst_offset {
        return;
    }
    if src_offset & 7 == dst_offset & 7 {
        move_bits_whole_bytes(bytes, src_offset, dst_offset, len);
    } else if dst_offset < src_offset {
        move_bits_down(bytes, src_offset, dst_offset, len);
    } else {
        move_bits_up(bytes, src_offset, dst_offset, len);
    }
}

/// Copy the bits of `range`, counted from `src_offset`, to the same positions
/// counted from `dst_offset`, sweeping away from the destination so that no
/// bit is read after it has been overwritten.
#[inline]
fn move_boundary_bits(
    bytes: &mut [u8],
    src_offset: usize,
    dst_offset: usize,
    range: std::ops::Range<usize>,
) {
    if dst_offset < src_offset {
        for i in range {
            let bit = get_bit(bytes, src_offset + i);
            set_bit(bytes, dst_offset + i, bit);
        }
    } else {
        for i in range.rev() {
            let bit = get_bit(bytes, src_offset + i);
            set_bit(bytes, dst_offset + i, bit);
        }
    }
}

/// Move bits whose source and destination share a bit position within their
/// byte, which keeps every byte of the run intact and reduces the move to a
/// `memmove` plus the partial byte at each end.
fn move_bits_whole_bytes(bytes: &mut [u8], src_offset: usize, dst_offset: usize, len: usize) {
    let lead = bits_to_boundary(dst_offset, len);
    let whole = (len - lead) >> 3;
    let trail = len - lead - whole * 8;

    // Each of the three steps has to read bits the other two have not written
    // yet, which holds as long as the end nearest the destination goes first.
    let (first, last) = if dst_offset < src_offset {
        (0..lead, len - trail..len)
    } else {
        (len - trail..len, 0..lead)
    };
    move_boundary_bits(bytes, src_offset, dst_offset, first);
    let source = (src_offset + lead) >> 3;
    let target = (dst_offset + lead) >> 3;
    if target < source && source - target < CHUNK {
        // `memmove` drops to a narrow loop when the two ranges overlap by all
        // but a few bytes - the shape of every small deletion - so a short
        // slide is staged through a buffer instead.
        let mut buffer = [0u8; CHUNK];
        let mut done = 0;
        while done < whole {
            let count = CHUNK.min(whole - done);
            buffer[..count].copy_from_slice(&bytes[source + done..source + done + count]);
            bytes[target + done..target + done + count].copy_from_slice(&buffer[..count]);
            done += count;
        }
    } else {
        bytes.copy_within(source..source + whole, target);
    }
    move_boundary_bits(bytes, src_offset, dst_offset, last);
}

/// Move bits towards the front of the buffer, across a byte boundary.
///
/// The sweep runs forwards: each chunk is read before anything below it is
/// written, so the run can overlap its own source.
fn move_bits_down(bytes: &mut [u8], src_offset: usize, dst_offset: usize, len: usize) {
    let lead = bits_to_boundary(dst_offset, len);
    move_boundary_bits(bytes, src_offset, dst_offset, 0..lead);

    let whole = (len - lead) >> 3;
    let target = (dst_offset + lead) >> 3;
    let source = (src_offset + lead) >> 3;
    let shift = (src_offset + lead) & 7;
    let mut window = [0u8; CHUNK + 1];
    let mut done = 0;
    while done < whole {
        let count = CHUNK.min(whole - done);
        window[..count + 1].copy_from_slice(&bytes[source + done..source + done + count + 1]);
        shift_window(
            &mut bytes[target + done..target + done + count],
            &window[..count + 1],
            shift,
        );
        done += count;
    }

    let moved = lead + whole * 8;
    if moved < len {
        let byte = read_u8(bytes, src_offset + moved);
        write_partial_byte(bytes, target + whole, byte, len - moved);
    }
}

/// Move bits towards the back of the buffer, across a byte boundary.
///
/// The sweep runs backwards, so each chunk is read before anything above it is
/// written. The one byte that does not fit that pattern is the one straddling
/// the top of a chunk, which the chunk below still needs after it has been
/// overwritten, so it is carried down in a register.
fn move_bits_up(bytes: &mut [u8], src_offset: usize, dst_offset: usize, len: usize) {
    // Bring the end of the destination down to a byte boundary.
    let trail = ((dst_offset + len) & 7).min(len);
    move_boundary_bits(bytes, src_offset, dst_offset, len - trail..len);

    let left = len - trail;
    let whole = left >> 3;
    let lead = left - whole * 8;
    // The destination byte range sits strictly above the source one, so the
    // bits below it are still intact once the sweep is done.
    let target = ((dst_offset + left) >> 3) - whole;
    let source = (src_offset + lead) >> 3;
    let shift = (src_offset + lead) & 7;

    let mut window = [0u8; CHUNK + 1];
    let mut done = whole;
    let mut carry = bytes[source + whole];
    while done > 0 {
        let count = CHUNK.min(done);
        done -= count;
        window[..count].copy_from_slice(&bytes[source + done..source + done + count]);
        window[count] = carry;
        carry = window[0];
        shift_window(
            &mut bytes[target + done..target + done + count],
            &window[..count + 1],
            shift,
        );
    }

    move_boundary_bits(bytes, src_offset, dst_offset, 0..lead);
}
