mod bits;
mod bitwise;
mod digits;
mod format;
mod numeric;
mod parse;
mod python;
mod random;
mod raw_bytes;
mod search;
mod splice;
mod validation;

pub(crate) use bits::{BS, BV, BitAccumulator, bv_from_ones, bv_from_zeros, head_bit_offset};
pub(crate) use bitwise::{
    BitConcat, LogicalOp, any_pair_bits, copy_unaligned_padded_bytes, count_bitslice,
    count_pair_bits, deposit_masked, extract_masked_bytes, logical_op_assign_bytes,
    logical_op_with_aligned_bytes, logical_op_with_matching_bytes, padded_bytes_from_offset,
    reverse_bitvec_in_place, rotate_bits_left,
};
pub(crate) use digits::{bin_from_padded_bytes, hex_from_padded_bytes, oct_from_padded_bytes};
pub(crate) use format::format_bit_collection;
pub(crate) use numeric::{
    FAST_INT_BITS, bv_from_f64, bv_from_int, bv_from_uint, byte_order_name, push_f64_bytes,
    push_int_bits, push_int_bytes,
};
pub(crate) use parse::{bv_from_bin, bv_from_hex, bv_from_oct, str_to_bv};
pub(crate) use python::{
    bitslice_to_bool_list, bv_from_bools, bytes_like_to_vec, convert_to_bool, promote_to_bv,
};
pub(crate) use random::bv_from_random;
pub(crate) use raw_bytes::{
    bv_from_bytes_slice, byte_search_prep, mask_padding_bits, reverse_byte_groups,
    reverse_padded_bits,
};
pub(crate) use search::{
    MaskedMatcher, SIGNAL_CHECK_INTERVAL, collect_find_all_positions,
    collect_find_all_positions_masked, compute_lps, count_bitvec, count_bitvec_masked,
    count_candidate_positions, count_single_bit, find_bitvec, find_bitvec_aligned,
    find_bitvec_masked_aligned, rfind_bitvec_aligned, rfind_bitvec_with_reversed_lps_aligned,
};
pub(crate) use splice::{copy_bits, fill_bits, move_bits};
pub(crate) use validation::{
    normalize_split_position, validate_index, validate_length, validate_logical_op_lengths,
    validate_shift, validate_slice,
};
