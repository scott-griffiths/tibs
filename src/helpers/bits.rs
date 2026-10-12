use bitvec::prelude::*;
use pyo3::exceptions::PyMemoryError;
use pyo3::prelude::*;
use std::collections::TryReserveError;

pub(crate) type BV = BitVec<u8, Msb0>;
pub(crate) type BS = BitSlice<u8, Msb0>;

pub(crate) fn bv_from_zeros(length: usize) -> BV {
    BV::repeat(false, length)
}

/// An empty byte buffer with room for `bit_length` bits.
///
/// For a length that a caller has asked for rather than one that already
/// exists in memory. An infallible allocation aborts the process when it
/// fails, and a `MemoryError` is what Python raises for the same request.
pub(crate) fn try_byte_buffer(bit_length: usize) -> PyResult<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(bit_length.div_ceil(8))
        .map_err(|_| memory_error(bit_length))?;
    Ok(bytes)
}

/// `length` copies of `bit`, as [`bv_from_zeros`] gives for zeros, but
/// raising a `MemoryError` rather than aborting when the space cannot be had.
pub(crate) fn try_bv_filled(bit: bool, length: usize) -> PyResult<BV> {
    let mut bytes = try_byte_buffer(length)?;
    bytes.resize(length.div_ceil(8), if bit { 0xff } else { 0 });
    Ok(bv_from_padded_bytes(bytes, length))
}

/// The first `length` bits of `bytes`, which may run on past them.
///
/// `BV::from_vec` followed by a truncate is not enough on its own: it spans
/// every bit of every byte first, and rounding a length up to whole bytes can
/// carry it past `BS::MAX_BITS` when the length itself is within it. On a
/// 32-bit build anything from 2^29 - 7 to 2^29 - 1 bits does this, and
/// `from_vec` panics rather than fail.
///
/// `length` must be at most `bytes.len() * 8` and at most `BS::MAX_BITS`.
#[inline]
pub(crate) fn bv_from_padded_bytes(bytes: Vec<u8>, length: usize) -> BV {
    debug_assert!(length <= BS::MAX_BITS);
    debug_assert!(length.div_ceil(8) <= bytes.len());
    // On a 64-bit build the buffer would need to be over 2^58 bytes, which no
    // machine can allocate, so the check folds away there and costs nothing.
    if usize::BITS < 64 && bytes.len() > BS::MAX_BITS / 8 {
        return bv_from_overlong_bytes(bytes, length);
    }
    let mut bv = BV::from_vec(bytes);
    bv.truncate(length);
    bv
}

/// [`bv_from_padded_bytes`] for a buffer too long to view as bits in one go:
/// take the whole bytes, then append the bits of the part byte. Growing back
/// into the buffer's own capacity does not reallocate.
#[cold]
#[inline(never)]
fn bv_from_overlong_bytes(mut bytes: Vec<u8>, length: usize) -> BV {
    let whole = length / 8;
    let tail = length % 8;
    let last = bytes.get(whole).copied().unwrap_or(0);
    bytes.truncate(whole);
    let mut bv = BV::from_vec(bytes);
    if tail > 0 {
        bv.extend_from_bitslice(&last.view_bits::<Msb0>()[..tail]);
    }
    bv
}

/// All the bits of `bytes`, whose length came from outside rather than from
/// a container that already exists, so may be more than one can hold.
pub(crate) fn try_bv_from_bytes(bytes: Vec<u8>) -> PyResult<BV> {
    BV::try_from_vec(bytes).map_err(|bytes| {
        PyMemoryError::new_err(format!(
            "Cannot create {} bytes of bits: this build of tibs supports at most {} bits \
             ({} bytes) in one container.",
            bytes.len(),
            BS::MAX_BITS,
            BS::MAX_BITS / 8
        ))
    })
}

pub(crate) fn memory_error(bit_length: usize) -> PyErr {
    PyMemoryError::new_err(format!("Not enough memory for {bit_length} bits."))
}

/// The most bits worth reserving up front on the strength of a claim about
/// values still to come: a dtype's length, or a sequence's `len()`.
///
/// A value of the wrong shape is only found out while packing, so reserving
/// the claimed size on trust let `Tibs.from_value('[u1; 2**60]', [])` abort
/// the process allocating for a value it was about to reject. Past this a
/// buffer grows as it fills, which costs little at that size. It is also below
/// `BS::MAX_BITS` on a 32-bit build, where a larger `BV::with_capacity` panics.
const MAX_RESERVE_BITS: usize = 1 << 27;

/// Bits to reserve for a claimed `bits`, capped at [`MAX_RESERVE_BITS`].
pub(crate) fn reserve_bits(bits: Option<usize>) -> usize {
    bits.unwrap_or(0).min(MAX_RESERVE_BITS)
}

/// Bytes to reserve for a claimed `bytes`, capped at [`MAX_RESERVE_BITS`].
pub(crate) fn reserve_bytes(bytes: Option<usize>) -> usize {
    bytes.unwrap_or(0).min(MAX_RESERVE_BITS / 8)
}

/// The largest run of bits that [`BitAccumulator::push`] takes at once.
///
/// The accumulator carries up to seven bits over from the previous value, so a
/// push of this size still leaves the whole run inside the 64-bit register.
const MAX_PUSH_BITS: usize = 57;

/// Packs runs of bits end to end into a byte buffer, most significant bit
/// first.
///
/// A `BitVec` grown a value at a time costs a heap allocation for the value
/// and then a bit-at-a-time append; this holds the straddling bits in a
/// register instead and writes out whole bytes, which is what makes packing a
/// sequence of same-width values cheap. `Msb0` ordering means the bytes it
/// produces are exactly a `BV`'s backing store, so [`Self::into_bitvec`] is
/// free beyond the truncation.
pub(crate) struct BitAccumulator {
    bytes: Vec<u8>,
    /// The bits not yet written out, right-aligned in the low `pending` bits.
    /// Anything above them is stale and is masked off by the cast to `u8`.
    ///
    /// Every bit pushed is either in `bytes` or pending, so the length is
    /// worked out from the two rather than kept as well. Keeping it was a
    /// load, add and store on every push, which a loop pushing whole words
    /// had to wait on from one push to the next.
    carry: u64,
    pending: usize,
}

impl BitAccumulator {
    /// Start an accumulator sized for `bit_capacity` bits, rounded up to a
    /// whole byte.
    pub(crate) fn with_bit_capacity(bit_capacity: Option<usize>) -> Self {
        let bytes = match bit_capacity {
            Some(bits) => Vec::with_capacity(bits.div_ceil(8)),
            None => Vec::new(),
        };
        BitAccumulator {
            bytes,
            carry: 0,
            pending: 0,
        }
    }

    /// Append the low `count` bits of `value`, most significant first.
    ///
    /// `count` must be at most [`MAX_PUSH_BITS`], and the bits of `value`
    /// above `count` must be zero.
    #[inline]
    pub(crate) fn push(&mut self, value: u64, count: usize) {
        debug_assert!(count <= MAX_PUSH_BITS);
        debug_assert!(value >> count == 0, "value has bits above the field");
        self.carry = (self.carry << count) | value;
        self.pending += count;
        while self.pending >= 8 {
            self.pending -= 8;
            self.bytes.push((self.carry >> self.pending) as u8);
        }
    }

    /// Append the low `count` bits of `value`, most significant first, for a
    /// `count` of up to 64.
    ///
    /// [`Self::push`] takes at most [`MAX_PUSH_BITS`] at a time because of the
    /// bits carried over from the previous value, so anything wider than that
    /// goes out as two runs.
    #[inline]
    pub(crate) fn push_wide(&mut self, value: u64, count: usize) {
        debug_assert!(count <= 64);
        debug_assert!(count == 64 || value >> count == 0);
        if count <= MAX_PUSH_BITS {
            self.push(value, count);
            return;
        }
        let low = count / 2;
        self.push(value >> low, count - low);
        self.push(value & ((1u64 << low) - 1), low);
    }

    /// Append `count` copies of `bit`.
    ///
    /// The whole bytes of a run are written as a fill rather than through the
    /// carry, and their space is reserved first, so a run too long for memory
    /// is an error here rather than an abort.
    pub(crate) fn try_push_repeated(
        &mut self,
        bit: bool,
        count: usize,
    ) -> Result<(), TryReserveError> {
        let fill = if bit { u64::MAX } else { 0 };
        let run = |n: usize| if n == 0 { 0 } else { fill >> (64 - n) };
        // Up to the next byte boundary through the carry, after which nothing
        // is pending if any of the run is left.
        let head = ((8 - self.pending) % 8).min(count);
        self.push(run(head), head);
        let rest = count - head;
        let whole = rest / 8;
        if whole > 0 {
            debug_assert!(self.is_byte_aligned());
            self.bytes.try_reserve(whole)?;
            self.bytes.resize(self.bytes.len() + whole, fill as u8);
        }
        let tail = rest % 8;
        self.push(run(tail), tail);
        Ok(())
    }

    /// The number of bits pushed so far.
    pub(crate) fn len(&self) -> usize {
        self.bytes.len() * 8 + self.pending
    }

    /// Whether the next push would start on a byte boundary, and so whether
    /// [`Self::push_aligned_bytes`] may be used.
    #[inline]
    pub(crate) fn is_byte_aligned(&self) -> bool {
        self.pending == 0
    }

    /// Append whole bytes directly, skipping the carry register.
    ///
    /// Only valid while the accumulator sits on a byte boundary, which
    /// [`Self::is_byte_aligned`] reports. Runs that are already whole bytes
    /// are common enough - any stretch of a mask that selects everything, or
    /// a field copied wholesale - to be worth not funnelling a byte at a time
    /// through the carry.
    #[inline]
    pub(crate) fn push_aligned_bytes(&mut self, bytes: &[u8]) {
        debug_assert!(self.is_byte_aligned());
        self.bytes.extend_from_slice(bytes);
    }

    /// Append every bit of `bits`, most significant first.
    ///
    /// The general-purpose entry point, for values that the caller could not
    /// reduce to a single `u64` run.
    pub(crate) fn push_bits(&mut self, bits: &BS) {
        for chunk in bits.chunks(MAX_PUSH_BITS) {
            self.push(chunk.load_be::<u64>(), chunk.len());
        }
    }

    /// Finish, returning the packed bits. Any final part-byte is padded with
    /// zeros, then trimmed off by the truncation.
    pub(crate) fn into_bitvec(mut self) -> BV {
        let length = self.len();
        if self.pending > 0 {
            self.bytes.push((self.carry << (8 - self.pending)) as u8);
        }
        bv_from_padded_bytes(self.bytes, length)
    }
}

/// The `length` bits starting `bit` bits into `bytes`, most significant first,
/// right-aligned in a `u64`.
///
/// `length` is 1 to 64 and the bits must lie within `bytes`. A field that
/// starts part way into a byte and runs to its full width spans nine bytes,
/// which is the one case where a ninth byte is read after the word.
#[inline]
pub(crate) fn load_be_bits(bytes: &[u8], bit: usize, length: usize) -> u64 {
    debug_assert!((1..=64).contains(&length));
    debug_assert!(bit + length <= bytes.len() * 8);
    let byte = bit / 8;
    let offset = bit % 8;
    let word = match bytes.get(byte..byte + 8) {
        Some(word) => u64::from_be_bytes(word.try_into().unwrap()),
        None => {
            let mut buf = [0u8; 8];
            let tail = &bytes[byte..];
            buf[..tail.len()].copy_from_slice(tail);
            u64::from_be_bytes(buf)
        }
    };
    let mut aligned = word << offset;
    if offset + length > 64 {
        aligned |= u64::from(bytes[byte + 8]) >> (8 - offset);
    }
    aligned >> (64 - length)
}

/// The bit position within the first raw byte at which `bits` starts.
///
/// Storage does not have to begin on a byte boundary: slicing a `BitSlice`
/// and calling `to_bitvec` keeps the original head index, so even an owned
/// `BitVec` can start part way into its first byte.
///
/// Read straight off the bit pointer, where it is a mask and a shift. Going
/// through `domain` gives the same answer but splits the whole slice into its
/// partial and whole elements to do it, and this is called on every access to
/// a `Mutibs`'s storage. An empty slice has no first live bit and reports zero,
/// as `domain` does, whatever its pointer holds.
#[inline]
pub(crate) fn head_bit_offset(bits: &BS) -> usize {
    let head = if bits.is_empty() {
        0
    } else {
        bits.as_bitptr().bit().into_inner() as usize
    };
    debug_assert_eq!(
        head,
        match bits.domain() {
            bitvec::domain::Domain::Enclave(elem)
            | bitvec::domain::Domain::Region {
                head: Some(elem), ..
            } => elem.head().into_inner() as usize,
            bitvec::domain::Domain::Region { .. } => 0,
        }
    );
    head
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_be_bits_agrees_with_bitvec() {
        let bytes: Vec<u8> = (0u8..11).map(|i| i.wrapping_mul(0x9d) ^ 0x5a).collect();
        let bits = BS::from_slice(&bytes);
        for length in 1..=64 {
            for start in 0..=bits.len() - length {
                let expected = bits[start..start + length].load_be::<u64>();
                assert_eq!(
                    load_be_bits(&bytes, start, length),
                    expected,
                    "{start} {length}"
                );
            }
        }
    }

    /// The overlong path is only reached for real at 2^29 - 7 bits on a 32-bit
    /// build, so check it against the plain one on lengths it can be given here.
    #[test]
    fn overlong_bytes_agree_with_from_vec_and_truncate() {
        let bytes: Vec<u8> = (0u8..9).map(|i| i.wrapping_mul(0x9d) ^ 0x5a).collect();
        for length in 0..=bytes.len() * 8 {
            let mut expected = BV::from_vec(bytes.clone());
            expected.truncate(length);
            let mut padded = bytes.clone();
            padded.truncate(length.div_ceil(8));
            assert_eq!(bv_from_overlong_bytes(bytes.clone(), length), expected);
            assert_eq!(bv_from_overlong_bytes(padded, length), expected);
        }
    }
}
