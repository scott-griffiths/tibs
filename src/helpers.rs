use crate::core::BitCollection;
use bitvec::prelude::*;
use half::f16;
use lru::LruCache;
use once_cell::sync::Lazy;
use pyo3::exceptions::{PyIndexError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyInt, PyMemoryView};
use rand::rngs::{OsRng, StdRng};
use rand::{RngCore, SeedableRng, TryRngCore};
use sha2::{Digest, Sha256};
use std::num::NonZeroUsize;
use std::sync::Mutex;

pub type BV = BitVec<u8, Msb0>;
pub type BS = BitSlice<u8, Msb0>;

// Define a static LRU cache.
const BITS_CACHE_SIZE: usize = 1024;
static BITS_CACHE: Lazy<Mutex<LruCache<String, BV>>> =
    Lazy::new(|| Mutex::new(LruCache::new(NonZeroUsize::new(BITS_CACHE_SIZE).unwrap())));

pub(crate) fn validate_logical_op_lengths(a: usize, b: usize) -> PyResult<()> {
    if a != b {
        Err(PyValueError::new_err(format!(
            "For logical operations the lengths of both objects must match. Received lengths of {a} and {b} bits."
        )))
    } else {
        Ok(())
    }
}

// An implementation of the KMP algorithm for bit slices.
fn compute_lps(pattern: &BS) -> Vec<usize> {
    let len = pattern.len();
    let mut lps = vec![0; len];
    let mut i = 1;
    let mut len_prev = 0;

    while i < len {
        match pattern[i] == pattern[len_prev] {
            true => {
                len_prev += 1;
                lps[i] = len_prev;
                i += 1;
            }
            false if len_prev != 0 => len_prev = lps[len_prev - 1],
            false => {
                lps[i] = 0;
                i += 1;
            }
        }
    }
    lps
}

pub(crate) fn find_bitvec(
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    byte_aligned: bool,
) -> Option<usize> {
    debug_assert!(end >= start);
    debug_assert!(end <= haystack.len());
    if byte_aligned {
        find_bitvec_impl::<true>(haystack, needle, start, end)
    } else {
        find_bitvec_impl::<false>(haystack, needle, start, end)
    }
}

#[inline]
fn find_bitvec_impl<const BYTE_ALIGNED: bool>(
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() - start {
        return None;
    }

    let lps = compute_lps(needle);
    let needle_len = needle.len();
    let mut i = start;
    let mut j = 0;

    while i < end {
        if needle[j] == haystack[i] {
            i += 1;
            j += 1;

            if j == needle_len {
                let match_pos = i - j;
                if !BYTE_ALIGNED || (match_pos & 7) == 0 {
                    return Some(match_pos);
                }
                // Continue searching for a byte-aligned match
                j = lps[j - 1];
            }
        } else if j != 0 {
            j = lps[j - 1];
        } else {
            i += 1;
        }
    }
    None
}

pub(crate) fn validate_index(index: i64, length: usize) -> PyResult<usize> {
    let index_p = if index < 0 {
        length as i64 + index
    } else {
        index
    };
    if index_p >= length as i64 || index_p < 0 {
        return Err(PyIndexError::new_err(format!(
            "Index of {index} is out of range for length of {length}"
        )));
    }
    Ok(index_p as usize)
}

pub(crate) fn validate_shift(s: &impl BitCollection, n: i64) -> PyResult<usize> {
    if s.is_empty() {
        return Err(PyValueError::new_err(
            "Cannot use a bit shift on an empty container.",
        ));
    }
    if n < 0 {
        return Err(PyValueError::new_err(
            "Cannot bit shift by a negative amount.",
        ));
    }
    Ok(n as usize)
}

#[inline]
pub(crate) fn validate_slice(
    length: usize,
    start: Option<i64>,
    end: Option<i64>,
) -> PyResult<(usize, usize)> {
    let mut start = start.unwrap_or(0);
    let mut end = end.unwrap_or(length as i64);
    if start < 0 {
        start += length as i64;
    }
    if end < 0 {
        end += length as i64;
    }

    if !(0 <= start && start <= end && end <= length as i64) {
        return Err(PyValueError::new_err(format!(
            "Invalid slice positions for length of {length}: start={start}, end={end}."
        )));
    }
    Ok((start as usize, end as usize))
}

pub(crate) fn process_seed(seed: &Option<Vec<u8>>) -> [u8; 32] {
    match seed {
        None => {
            let mut seed_arr = [0u8; 32];
            rand::rng().fill_bytes(&mut seed_arr);
            seed_arr
        }
        Some(seed_bytes) => {
            let mut hasher = Sha256::new();
            hasher.update(seed_bytes);
            let digest = hasher.finalize();
            let mut seed_arr = [0u8; 32];
            seed_arr.copy_from_slice(&digest);
            seed_arr
        }
    }
}

pub(crate) fn bv_from_random(length: i64, secure: bool, seed: &Option<Vec<u8>>) -> PyResult<BV> {
    if length < 0 {
        return Err(PyValueError::new_err(format!(
            "Negative bit length given: {}.",
            length
        )));
    }
    if secure && seed.is_some() {
        return Err(PyValueError::new_err(
            "A seed cannot be used when generating secure random data.",
        ));
    }
    let length = length as usize;
    if length == 0 {
        return Ok(BV::new());
    }
    let seed_arr = process_seed(seed);
    let num_bytes = length.div_ceil(8);
    let mut data = vec![0u8; num_bytes];
    if secure {
        OsRng
            .try_fill_bytes(&mut data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    } else {
        let mut rng = StdRng::from_seed(seed_arr);
        rng.fill_bytes(&mut data);
    }
    let mut bv = BV::from_vec(data);
    if bv.len() > length {
        bv.truncate(length);
    }
    Ok(bv)
}

pub(crate) fn bv_from_zeros(length: usize) -> BV {
    BV::repeat(false, length)
}

pub(crate) fn bv_from_ones(length: usize) -> BV {
    BV::repeat(true, length)
}

#[inline]
pub(crate) fn bv_from_bin(binary_string: &str) -> PyResult<BV> {
    // Ignore any leading '0b' or '0B'
    let s = binary_string
        .strip_prefix("0b")
        .or_else(|| binary_string.strip_prefix("0B"))
        .unwrap_or(binary_string);
    let mut bv: BV = BV::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '0' => bv.push(false),
            '1' => bv.push(true),
            '_' => continue,
            c if c.is_whitespace() => continue,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Cannot convert from bin '{binary_string}: Invalid character '{c}'."
                )));
            }
        }
    }
    bv.set_uninitialized(false);
    Ok(bv)
}

#[inline]
pub(crate) fn bv_from_oct(octal_string: &str) -> PyResult<BV> {
    // Ignore any leading '0o' or '0O'
    let s = octal_string
        .strip_prefix("0o")
        .or_else(|| octal_string.strip_prefix("0O"))
        .unwrap_or(octal_string);
    let mut bv: BV = BV::with_capacity(s.len() * 3);
    for c in s.chars() {
        match c {
            '0' => bv.extend_from_bitslice(bits![0, 0, 0]),
            '1' => bv.extend_from_bitslice(bits![0, 0, 1]),
            '2' => bv.extend_from_bitslice(bits![0, 1, 0]),
            '3' => bv.extend_from_bitslice(bits![0, 1, 1]),
            '4' => bv.extend_from_bitslice(bits![1, 0, 0]),
            '5' => bv.extend_from_bitslice(bits![1, 0, 1]),
            '6' => bv.extend_from_bitslice(bits![1, 1, 0]),
            '7' => bv.extend_from_bitslice(bits![1, 1, 1]),
            '_' => continue,
            c if c.is_whitespace() => continue,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Cannot convert from oct '{octal_string}': Invalid character '{c}'."
                )));
            }
        }
    }
    bv.set_uninitialized(false);
    Ok(bv)
}

#[inline]
pub(crate) fn bv_from_hex(hex: &str) -> PyResult<BV> {
    // Ignore any leading '0x' or '0X'
    let mut new_hex = hex
        .strip_prefix("0x")
        .or_else(|| hex.strip_prefix("0X"))
        .unwrap_or(hex)
        .to_string();
    // Remove any underscores or whitespace characters
    new_hex.retain(|c| c != '_' && !c.is_whitespace());
    let new_hex_length = new_hex.len() as i64;
    if new_hex_length % 2 != 0 {
        new_hex.push('0');
    }
    let data = match hex::decode(&new_hex) {
        Ok(d) => d,
        Err(e) => {
            return Err(PyValueError::new_err(format!(
                "Cannot convert from hex '{hex}': {}",
                e
            )))
        }
    };
    let bv = bv_from_bytes_slice(data, None, Some(new_hex_length * 4))?;
    Ok(bv)
}

pub(crate) fn bv_from_bytes_slice(
    data: Vec<u8>,
    offset: Option<i64>,
    length: Option<i64>,
) -> PyResult<BV> {
    if length.is_none() && offset.is_none() {
        return Ok(BV::from_vec(data));
    }
    let start_bit = offset.unwrap_or(0);
    if start_bit < 0 {
        return Err(PyValueError::new_err(format!(
            "Cannot create using a negative offset of {start_bit}."
        )));
    }
    let start_bit = start_bit as usize;
    let data_length = data.len() * 8;
    if start_bit > data_length {
        return Err(PyValueError::new_err(format!(
            "Offset of {start_bit} is greater than the data length ({data_length} bits)."
        )));
    }
    let length = length.unwrap_or(data_length as i64 - start_bit as i64);
    if length < 0 {
        return Err(PyValueError::new_err(format!(
            "Negative length of {length} bits provided."
        )));
    }
    let length = length as usize;
    if start_bit + length > data_length {
        return Err(PyValueError::new_err(format!(
            "Length of {length} with offset of {start_bit} is greater than the data length ({data_length} bits)."
        )));
    }
    let bs = BS::from_slice(&data);
    Ok(bs[start_bit..start_bit + length].to_bitvec())
}

#[inline]
pub(crate) fn bv_from_u128(value: u128, length: i64) -> PyResult<BV> {
    if length <= 0 || length > 128 {
        return Err(PyValueError::new_err(format!(
            "Bit length for unsigned int must be between 1 and 128. Received {length}."
        )));
    }
    if value >= (1u128 << length) {
        return Err(PyOverflowError::new_err(format!(
            "Value {value} does not fit in {length} bits."
        )));
    }
    let mut bv = BV::repeat(false, length as usize);
    bv.store_be(value);
    Ok(bv)
}

#[inline]
pub(crate) fn bv_from_i128(value: i128, length: i64) -> PyResult<BV> {
    if length <= 0 || length > 128 {
        return Err(PyValueError::new_err(format!(
            "Bit length for signed int must be between 1 and 128. Received {length}."
        )));
    }
    let min_val = -(1i128 << (length - 1));
    let max_val = (1i128 << (length - 1)) - 1;
    if value < min_val || value > max_val {
        return Err(PyOverflowError::new_err(format!(
            "Value {value} does not fit in {length} signed bits."
        )));
    }
    let repeat_bit = value < 0;
    let mut bv = BV::repeat(repeat_bit, length as usize);
    bv.store_be(value);
    Ok(bv)
}

pub(crate) fn bv_from_f64(value: f64, length: i64) -> PyResult<BV> {
    let bv = match length {
        64 => {
            let mut bv = BV::repeat(false, 64);
            bv.store_be(value.to_bits());
            bv
        }
        32 => {
            let value_f32 = value as f32;
            let mut bv = BV::repeat(false, 32);
            bv.store_be(value_f32.to_bits());
            bv
        }
        16 => {
            let value_f16 = f16::from_f64(value);
            let mut bv = BV::repeat(false, 16);
            bv.store_be(value_f16.to_bits());
            bv
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported float bit length '{length}'. Only 16, 32 and 64 are supported."
            )));
        }
    };
    Ok(bv)
}

pub(crate) fn bv_from_bools(iterable: &Bound<'_, PyAny>) -> PyResult<BV> {
    // For sequences, we can pre-allocate the capacity.
    let capacity = iterable.len().ok().unwrap_or(64);
    let mut bv = BV::with_capacity(capacity);

    for value in iterable.try_iter()? {
        bv.push(value?.is_truthy()?);
    }
    Ok(bv)
}

fn string_literal_to_bv(s: &str) -> PyResult<BV> {
    match s.get(0..2).map(|p| p.to_ascii_lowercase()).as_deref() {
        Some("0b") => {
            let bv = bv_from_bin(s)?;
            Ok(bv)
        }
        Some("0x") => {
            let bv = bv_from_hex(s)?;
            Ok(bv)
        }
        Some("0o") => {
            let bv = bv_from_oct(s)?;
            Ok(bv)
        }
        _ => Err(PyValueError::new_err(format!(
            "Can't parse token '{s}'. Did you mean to prefix with '0x', '0b' or '0o'?"
        ))),
    }
}

pub(crate) fn str_to_bv(s: String) -> PyResult<BV> {
    // First remove whitespace
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    // Check if it's already in the cache
    {
        let mut cache = BITS_CACHE.lock().unwrap();
        if let Some(cached_data) = cache.get(&s) {
            return Ok(cached_data.clone());
        }
    }
    let tokens = s.split(',');
    let mut bv_array = Vec::<BV>::new();
    let mut total_bit_length = 0;
    for token in tokens {
        if token.is_empty() {
            continue;
        }
        let x = string_literal_to_bv(token)?;
        total_bit_length += x.len();
        bv_array.push(x);
    }
    if bv_array.is_empty() {
        return Ok(BV::new());
    }
    // Combine all bits
    let result = if bv_array.len() == 1 {
        bv_array.pop().unwrap()
    } else {
        let mut result = BV::with_capacity(total_bit_length);
        for bv in bv_array {
            result.extend_from_bitslice(&bv);
        }
        result
    };
    // Update cache with new result
    {
        let mut cache = BITS_CACHE.lock().unwrap();
        cache.put(s, result.clone());
    }
    Ok(result)
}

pub(crate) fn promote_to_bv(any: &Bound<'_, PyAny>) -> PyResult<BV> {
    // Is it a string?
    if let Ok(any_string) = any.extract::<String>() {
        let bv = str_to_bv(any_string)?;
        return Ok(bv);
    }

    // Is it a bytes, bytearray or memoryview?
    if (any.is_instance_of::<PyBytes>()
        || any.is_instance_of::<PyByteArray>()
        || any.is_instance_of::<PyMemoryView>())
        && let Ok(any_bytes) = any.extract::<Vec<u8>>() {
            return Ok(BV::from_vec(any_bytes));
        }

    // Is it an iterable that we can convert each element to a bool?
    if let Ok(iter) = any.try_iter() {
        let mut bv = BV::new();
        for item in iter {
            bv.push(item?.is_truthy()?);
        }
        return Ok(bv);
    }
    let type_name = match any.get_type().name() {
        Ok(name) => name.to_string(),
        Err(_) => "<unknown>".to_string(),
    };
    let mut err = format!("Cannot promote object of type {type_name} to a Mutibs object. ");
    if any.is_instance_of::<PyInt>() {
        err.push_str("Perhaps you want to use 'Mutibs.from_zeros()', 'Mutibs.from_ones()' or 'Mutibs.from_random()'?");
    };
    Err(PyTypeError::new_err(err))
}
