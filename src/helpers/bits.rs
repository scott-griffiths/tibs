use bitvec::prelude::*;

pub(crate) type BV = BitVec<u8, Msb0>;
pub(crate) type BS = BitSlice<u8, Msb0>;

pub(crate) fn bv_from_zeros(length: usize) -> BV {
    BV::repeat(false, length)
}

pub(crate) fn bv_from_ones(length: usize) -> BV {
    BV::repeat(true, length)
}
