use crate::DecodeError;
use crate::core::BitCollection;
use crate::enums::Codec;
use crate::helpers::{BS, BV};
use bitvec::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn raw_encoded_bit_length(bit_length: usize) -> usize {
    let data_byte_length = bit_length.div_ceil(8);
    8 + encode_varint(data_byte_length as u64).len() + data_byte_length * 8
}

fn short_raw_encoded_bit_length(bit_length: usize) -> usize {
    8 + bit_length.div_ceil(8) * 8
}

fn rice_encode_int(value: usize, k: u8) -> BV {
    let mut out = BV::new();
    let quotient = value >> k;
    for _ in 0..quotient {
        out.push(true);
    }
    out.push(false);
    if k > 0 {
        let remainder_mask = (1usize << k) - 1;
        let remainder = value & remainder_mask;
        for shift in (0..k).rev() {
            out.push(((remainder >> shift) & 1) == 1);
        }
    }
    out
}

fn rice_decode_int(bits: &BS, start: usize, k: u8) -> PyResult<(usize, usize)> {
    let mut pos = start;
    while pos < bits.len() && bits[pos] {
        pos += 1;
    }
    if pos >= bits.len() {
        return Err(PyValueError::new_err(
            "The encoded sequence ended unexpectedly.",
        ));
    }
    let quotient = pos - start;
    pos += 1;

    let k_usize = k as usize;
    if bits.len() - pos < k_usize {
        return Err(PyValueError::new_err(
            "The encoded sequence ended unexpectedly.",
        ));
    }
    let remainder = if k == 0 {
        0
    } else {
        bits[pos..pos + k_usize].load_be::<usize>()
    };
    pos += k_usize;

    let base = quotient
        .checked_shl(k as u32)
        .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
    let value = base
        .checked_add(remainder)
        .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
    Ok((value, pos))
}

fn zstd_compress_bytes<C: BitCollection>(bits: &C) -> PyResult<Vec<u8>> {
    zstd::bulk::compress(&bits.padded_byte_data_cow(), 0)
        .map_err(|e| PyValueError::new_err(format!("The zstd payload could not be encoded: {e}")))
}

/// Three-bit markers that follow the two-bit codec tag.
const RAW_MARKER: u8 = 0b000;
const ZSTD_MARKER: u8 = 0b010;

/// The byte a Raw or Zstd encoding opens with.
///
/// Two zero bits of codec tag, a three-bit marker, then the count of padding
/// bits: eight bits exactly, so the varint that follows and the payload after
/// it both land on byte boundaries. That is what lets those two encodings be
/// assembled as bytes rather than pushed a bit at a time.
#[inline]
fn body_header_byte(marker: u8, bit_padding: usize) -> u8 {
    debug_assert!(marker < 8);
    debug_assert!(bit_padding < 8);
    (marker << 3) | bit_padding as u8
}

fn encode_as_zstd_bytes<C: BitCollection>(bits: &C, compressed: Vec<u8>) -> Vec<u8> {
    let bit_padding = if bits.len().is_multiple_of(8) {
        0
    } else {
        8 - bits.len() % 8
    };
    let varint = encode_varint_bytes(compressed.len() as u64);
    let mut out = Vec::with_capacity(1 + varint.len() + compressed.len());
    out.push(body_header_byte(ZSTD_MARKER, bit_padding));
    out.extend_from_slice(&varint);
    out.extend_from_slice(&compressed);
    out
}

fn encode_as_raw_bytes<C: BitCollection>(bits: &C) -> Vec<u8> {
    let bit_length = bits.len();
    let data_byte_length = bit_length.div_ceil(8);
    let bit_padding = data_byte_length * 8 - bit_length;

    let varint = encode_varint_bytes(data_byte_length as u64);
    let mut out = Vec::with_capacity(1 + varint.len() + data_byte_length);
    out.push(body_header_byte(RAW_MARKER, bit_padding));
    out.extend_from_slice(&varint);
    // The payload is already the padded byte data, so it copies straight in.
    out.extend_from_slice(&bits.padded_byte_data_cow());
    out
}

fn rice_encoded_gaps(bits: &BS, sparse_bit: bool) -> Vec<usize> {
    let mut gaps = Vec::new();

    let mut previous = 0;
    if sparse_bit {
        for p in bits.iter_ones() {
            gaps.push(p - previous);
            previous = p + 1;
        }
    } else {
        for p in bits.iter_zeros() {
            gaps.push(p - previous);
            previous = p + 1;
        }
    }

    if let Some(last) = bits.last()
        && *last != sparse_bit
    {
        gaps.push(bits.len() - previous - 1);
    }

    gaps
}

fn estimated_rice_k(gaps: &[usize]) -> u8 {
    if gaps.is_empty() {
        return 0;
    }

    let total_gap: usize = gaps.iter().sum();
    if total_gap == 0 {
        return 0;
    }

    let mean_gap = total_gap as f64 / gaps.len() as f64;
    let estimate = (mean_gap * std::f64::consts::LN_2).log2().round();
    estimate.clamp(0.0, 31.0) as u8
}

fn rice_payload_bit_length(gaps: &[usize], k: u8) -> usize {
    gaps.iter().map(|gap| (gap >> k) + 1 + k as usize).sum()
}

fn rice_encoded_bit_length<C: BitCollection>(bits: &C, sparse_bit: bool) -> usize {
    let gaps = rice_encoded_gaps(bits.as_bitslice(), sparse_bit);
    let estimated_k = estimated_rice_k(&gaps);
    let payload_bit_length = rice_payload_bit_length(&gaps, estimated_k);
    let payload_byte_length = payload_bit_length.div_ceil(8);
    8 + encode_varint(payload_byte_length as u64).len() + 8 + payload_byte_length * 8
}

fn encode_as_rice<C: BitCollection>(bits: &C, sparse_bit: bool) -> BV {
    let bitslice = bits.as_bitslice();

    let gaps = rice_encoded_gaps(bitslice, sparse_bit);
    debug_assert!(bitslice.len() > 0);
    let final_bit = *bitslice
        .last()
        .expect("Rice encoding not supported for empty Tibs.");
    let estimated_k = estimated_rice_k(&gaps);

    let payload_bit_length = rice_payload_bit_length(&gaps, estimated_k);
    let mut payload = BV::new();
    for gap in &gaps {
        payload.extend(rice_encode_int(*gap, estimated_k));
    }
    debug_assert_eq!(payload.len(), payload_bit_length);
    let payload_byte_length = payload_bit_length.div_ceil(8);
    let bit_padding = payload_byte_length * 8 - payload_bit_length;
    for _ in 0..bit_padding {
        payload.push(false);
    }

    let mut encoded = BV::new();
    encoded.push(false);
    encoded.push(false);
    encoded.push(true);
    for shift in (0..3).rev() {
        encoded.push((bit_padding >> shift) & 1 == 1);
    }
    encoded.extend(encode_varint(payload_byte_length as u64));
    for shift in (0..5).rev() {
        encoded.push((estimated_k >> shift) & 1 == 1);
    }
    encoded.push(sparse_bit);
    encoded.push(final_bit);
    encoded.push(false);
    encoded.extend(payload);

    encoded
}

fn exact_payload_end(total: usize, start: usize, len: usize) -> PyResult<usize> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
    if total < end {
        return Err(PyValueError::new_err(
            "The encoded sequence ended unexpectedly.",
        ));
    }
    if total > end {
        return Err(PyValueError::new_err(
            "The encoded sequence has unexpected trailing bytes.",
        ));
    }
    Ok(end)
}

fn decode_raw_payload<C: BitCollection>(
    bv: BV,
    bit_padding: usize,
    data_start: usize,
    data_bits: usize,
) -> PyResult<C> {
    exact_payload_end(bv.len(), data_start, data_bits)?;
    if bit_padding > data_bits {
        return Err(PyValueError::new_err("The encoded sequence is reserved."));
    }

    Ok(C::from_bv(bv).get_slice_unchecked(data_start, data_bits - bit_padding))
}

fn decode_rice_payload<C: BitCollection>(
    bv: &BS,
    bit_padding: usize,
    data_start: usize,
    payload_bits: usize,
) -> PyResult<C> {
    let config_end = data_start
        .checked_add(8)
        .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
    let payload_start = config_end;
    let payload_end = exact_payload_end(bv.len(), payload_start, payload_bits)?;
    if bit_padding > payload_bits {
        return Err(PyValueError::new_err("The encoded sequence is reserved."));
    }

    let config = &bv[data_start..config_end];
    if config[7] {
        return Err(PyValueError::new_err("The encoded sequence is reserved."));
    }
    let k = config[0..5].load_be::<u8>();
    let sparse_bit = config[5];
    let final_bit = config[6];

    let encoded_gaps_end = payload_end - bit_padding;
    let encoded_gaps = &bv[payload_start..encoded_gaps_end];

    let mut decoded = BV::new();
    let mut pos = 0usize;
    while pos < encoded_gaps.len() {
        let (gap, next_pos) = rice_decode_int(encoded_gaps, pos, k)?;
        pos = next_pos;

        for _ in 0..gap {
            decoded.push(!sparse_bit);
        }
        decoded.push(sparse_bit);
    }

    if decoded.is_empty() {
        return Err(PyValueError::new_err("The encoded sequence is reserved."));
    }
    let final_pos = decoded.len() - 1;
    decoded.set(final_pos, final_bit);
    Ok(C::from_bv(decoded))
}

fn decode_zstd_payload<C: BitCollection>(
    bv: &BV,
    bit_padding: usize,
    data_start: usize,
    payload_bits: usize,
) -> PyResult<C> {
    let payload_end = exact_payload_end(bv.len(), data_start, payload_bits)?;
    debug_assert!(data_start.is_multiple_of(8));
    debug_assert!(payload_end.is_multiple_of(8));
    let compressed = &bv.as_raw_slice()[data_start / 8..payload_end / 8];
    let decompressed_size = zstd::zstd_safe::get_frame_content_size(compressed)
        .map_err(|e| PyValueError::new_err(format!("The zstd payload could not be decoded: {e}")))?
        .ok_or_else(|| {
            PyValueError::new_err("The zstd payload did not include its decompressed size.")
        })?;

    let decompressed =
        zstd::bulk::decompress(compressed, decompressed_size as usize).map_err(|e| {
            PyValueError::new_err(format!("The zstd payload could not be decoded: {e}"))
        })?;

    let data_bits = decompressed.len() * 8;
    if bit_padding > data_bits {
        return Err(PyValueError::new_err("The encoded sequence is reserved."));
    }
    let out_end = data_bits - bit_padding;
    let mut decompressed = BV::from_vec(decompressed);
    decompressed.truncate(out_end);
    Ok(C::from_bv(decompressed))
}

/// A varint as whole bytes: seven bits of payload each, most significant group
/// first, with the top bit set on every byte but the last.
fn encode_varint_bytes(mut u: u64) -> Vec<u8> {
    let mut chunks: Vec<u8> = Vec::new();
    loop {
        chunks.push((u & 0x7f) as u8);
        u >>= 7;
        if u == 0 {
            break;
        }
    }
    chunks.reverse();

    let last = chunks.len() - 1;
    for chunk in &mut chunks[..last] {
        *chunk |= 0x80;
    }
    chunks
}

fn encode_varint(u: u64) -> BV {
    BV::from_vec(encode_varint_bytes(u))
}

fn decode_varint(bits: &BS) -> PyResult<(usize, usize)> {
    let mut value: usize = 0;
    let mut bits_consumed: usize = 0;
    let mut saw_final = false;

    for byte in bits.chunks(8) {
        if byte.len() < 8 {
            break;
        }
        let continuation = byte[0];
        let payload = byte[1..8].load_be::<u8>() as usize;

        if bits_consumed == 0 && continuation && payload == 0 {
            return Err(PyValueError::new_err("The encoded sequence is reserved."));
        }
        if value > (usize::MAX >> 7) {
            return Err(PyValueError::new_err(
                "The encoded sequence is too large to decode.",
            ));
        }
        value = (value << 7) | payload;
        bits_consumed += 8;

        if !continuation {
            saw_final = true;
            break;
        }
    }

    if !saw_final {
        return Err(PyValueError::new_err(
            "The encoded sequence ended unexpectedly.",
        ));
    }
    Ok((value, bits_consumed))
}

fn decode_bytes_inner<C: BitCollection>(b: Vec<u8>) -> PyResult<C> {
    if b.is_empty() {
        return Err(PyValueError::new_err(
            "Cannot decode an empty byte sequence.",
        ));
    }
    let bv = BV::from_vec(b);
    let single_byte_flag = bv[0];
    if single_byte_flag {
        exact_payload_end(bv.len(), 0, 8)?;
        for bit_pos in 1..8 {
            if bv[bit_pos] {
                return Ok(C::from_bv(bv).get_slice_unchecked(bit_pos + 1, 7 - bit_pos));
            }
        }
        return Err(PyValueError::new_err("The encoded sequence is reserved."));
    }
    let short_form_flag = bv[1];
    if short_form_flag {
        let byte_length = bv[2..5].load_be::<u8>() as usize + 1;
        let bit_padding = bv[5..8].load_be::<u8>() as usize;
        let data_bits = byte_length * 8;
        let bit_length = data_bits - bit_padding;
        if bit_length <= 6 {
            return Err(PyValueError::new_err("The encoded sequence is reserved."));
        }
        exact_payload_end(bv.len(), 8, data_bits)?;
        return Ok(C::from_bv(bv).get_slice_unchecked(8, bit_length));
    }

    let codec = bv[2..5].load_be::<u8>();
    let bit_padding = bv[5..8].load_be::<u8>() as usize;

    let (byte_length, varint_bits) = decode_varint(&bv[8..])?;
    let data_start = 8 + varint_bits;
    let data_bits = byte_length
        .checked_mul(8)
        .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
    match codec {
        0b000 => decode_raw_payload(bv, bit_padding, data_start, data_bits),
        0b001 => decode_rice_payload(&bv, bit_padding, data_start, data_bits),
        0b010 => decode_zstd_payload(&bv, bit_padding, data_start, data_bits),
        _ => Err(PyValueError::new_err("The codec value is reserved.")),
    }
}

pub(crate) fn decode_bytes<C: BitCollection>(py: Python<'_>, b: Vec<u8>) -> PyResult<C> {
    decode_bytes_inner(b).map_err(|error| {
        if error.is_instance_of::<PyValueError>(py) {
            DecodeError::new_err(error.value(py).to_string())
        } else {
            error
        }
    })
}

pub(crate) fn encode<C: BitCollection>(bits: &C, codec: Option<Codec>) -> PyResult<Vec<u8>> {
    let bit_length = bits.len();
    let mut bv: BV = BV::new();

    // Length of zero treated as a special case and ignores the codec.
    // Uses the Auto codec, and encodes as a single byte.
    if bit_length == 0 {
        bv.push(true);
        for _ in 0..6 {
            bv.push(false);
        }
        bv.push(true);
        return Ok(bv.into_vec());
    }

    match codec.unwrap_or(Codec::Auto) {
        Codec::Auto => match bit_length {
            0..=6 => {
                bv.push(true);
                let leading_zeros = 6 - bit_length;
                for _ in 0..leading_zeros {
                    bv.push(false);
                }
                bv.push(true);
                bv.extend_from_bitslice(bits.as_bitslice());
            }
            7..=64 => {
                bv.push(false);
                bv.push(true);
                let byte_length = bit_length.div_ceil(8);
                let bit_padding = byte_length * 8 - bit_length;
                let byte_length_minus_1 = (byte_length - 1) as u8;
                for shift in (0..3).rev() {
                    bv.push((byte_length_minus_1 >> shift) & 1 == 1);
                }
                for shift in (0..3).rev() {
                    bv.push((bit_padding >> shift) & 1 == 1);
                }
                let mut short_encoded = bv.clone();
                short_encoded.extend(bits.to_bitvec());
                for _ in 0..bit_padding {
                    short_encoded.push(false);
                }

                if bit_length > 24 {
                    let ones_count = bits.count(true);
                    let sparse_bit = ones_count < bit_length / 2;
                    let rice_bit_length = rice_encoded_bit_length(bits, sparse_bit);
                    if rice_bit_length < short_raw_encoded_bit_length(bit_length) {
                        bv.clear();
                        bv.push(false);
                        bv.push(false);
                        bv.extend(encode_as_rice(bits, sparse_bit));
                    } else {
                        bv = short_encoded;
                    }
                } else {
                    bv = short_encoded;
                }
            }
            65.. => {
                bv.push(false);
                bv.push(false);

                let raw_bit_length = raw_encoded_bit_length(bit_length);
                let mut best_codec = Codec::Raw;
                let mut best_bit_length = raw_bit_length;
                let mut best_compressed: Option<Vec<u8>> = None;

                let ones_count = bits.count(true);
                let sparse_bit = ones_count < bit_length / 2;
                let sparseness = if sparse_bit {
                    ones_count as f64 / bits.len() as f64
                } else {
                    (bits.len() - ones_count) as f64 / bits.len() as f64
                };
                if bit_length <= 128 || sparseness < 0.25 {
                    let rice_bit_length = rice_encoded_bit_length(bits, sparse_bit);
                    if rice_bit_length < best_bit_length {
                        best_codec = Codec::Rice;
                        best_bit_length = rice_bit_length;
                    }
                }

                if let Ok(zstd_compressed) = zstd_compress_bytes(bits) {
                    let zstd_bit_length = 8
                        + encode_varint_bytes(zstd_compressed.len() as u64).len() * 8
                        + zstd_compressed.len() * 8;

                    if zstd_bit_length < best_bit_length {
                        best_codec = Codec::Zstd;
                        best_compressed = Some(zstd_compressed);
                    }
                }
                match best_codec {
                    // Raw and Zstd are whole bytes from their first bit, so
                    // they are returned as bytes rather than rebuilt through
                    // the bit vector.
                    Codec::Raw => return Ok(encode_as_raw_bytes(bits)),
                    Codec::Zstd => {
                        let compressed = best_compressed
                            .expect("zstd encoding should be available when selected");
                        return Ok(encode_as_zstd_bytes(bits, compressed));
                    }
                    Codec::Rice => bv.extend(encode_as_rice(bits, sparse_bit)),
                    Codec::Auto => unreachable!(),
                }
            }
        },
        Codec::Raw => return Ok(encode_as_raw_bytes(bits)),
        Codec::Rice => {
            bv.push(false);
            bv.push(false);
            let sparse_bit = bits.count(true) < bits.len() / 2;
            bv.extend(encode_as_rice(bits, sparse_bit));
        }
        Codec::Zstd => return Ok(encode_as_zstd_bytes(bits, zstd_compress_bytes(bits)?)),
    }

    Ok(bv.into_vec())
}
