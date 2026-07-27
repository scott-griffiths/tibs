use bitvec::prelude::*;

pub(crate) type BV = BitVec<u8, Msb0>;
pub(crate) type BS = BitSlice<u8, Msb0>;

pub(crate) fn bv_from_zeros(length: usize) -> BV {
    BV::repeat(false, length)
}

pub(crate) fn bv_from_ones(length: usize) -> BV {
    BV::repeat(true, length)
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
    carry: u64,
    pending: usize,
    length: usize,
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
            length: 0,
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
        self.length += count;
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
        self.length += bytes.len() * 8;
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
        if self.pending > 0 {
            self.bytes.push((self.carry << (8 - self.pending)) as u8);
        }
        let mut bv = BV::from_vec(self.bytes);
        bv.truncate(self.length);
        bv
    }
}

/// The bit position within the first raw byte at which `bits` starts.
///
/// Storage does not have to begin on a byte boundary: slicing a `BitSlice`
/// and calling `to_bitvec` keeps the original head index, so even an owned
/// `BitVec` can start part way into its first byte.
#[inline]
pub(crate) fn head_bit_offset(bits: &BS) -> usize {
    match bits.domain() {
        bitvec::domain::Domain::Enclave(elem) => elem.head().into_inner() as usize,
        bitvec::domain::Domain::Region {
            head: Some(elem), ..
        } => elem.head().into_inner() as usize,
        _ => 0,
    }
}
