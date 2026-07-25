mod bits;
mod bitwise;
mod format;
mod numeric;
mod parse;
mod python;
mod random;
mod raw_bytes;
mod search;
mod validation;

pub(crate) use bits::{BS, BV, bv_from_ones, bv_from_zeros};
pub(crate) use bitwise::{
    LogicalOp, copy_unaligned_padded_bytes, count_bitslice, count_pair_bits, deposit_masked,
    for_each_pair_word, for_each_pair_word_bitslice, logical_op_with_aligned_bytes,
    logical_op_with_matching_bytes,
};
pub(crate) use format::format_bit_collection;
pub(crate) use numeric::{FAST_INT_BITS, bv_from_f64, bv_from_int, bv_from_uint, byte_order_name};
pub(crate) use parse::{bv_from_bin, bv_from_hex, bv_from_oct, str_to_bv};
pub(crate) use python::{
    bitslice_to_bool_list, bv_from_bools, bytes_like_to_vec, convert_to_bool, promote_to_bv,
};
pub(crate) use random::bv_from_random;
pub(crate) use raw_bytes::{bv_from_bytes_slice, byte_search_prep, mask_padding_bits};
pub(crate) use search::{
    MaskedMatcher, SIGNAL_CHECK_INTERVAL, collect_find_all_positions,
    collect_find_all_positions_masked, compute_lps, count_bitvec, count_bitvec_masked,
    count_candidate_positions, count_single_bit, find_bitvec, find_bitvec_aligned,
    find_bitvec_masked_aligned, find_bitvec_with_lps_aligned, rfind_bitvec_aligned,
    rfind_bitvec_with_reversed_lps_aligned,
};
pub(crate) use validation::{
    normalize_split_position, validate_index, validate_length, validate_logical_op_lengths,
    validate_shift, validate_slice,
};
