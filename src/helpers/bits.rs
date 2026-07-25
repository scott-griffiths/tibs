use bitvec::prelude::*;

pub(crate) type BV = BitVec<u8, Msb0>;
pub(crate) type BS = BitSlice<u8, Msb0>;

pub(crate) fn bv_from_zeros(length: usize) -> BV {
    BV::repeat(false, length)
}

pub(crate) fn bv_from_ones(length: usize) -> BV {
    BV::repeat(true, length)
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
