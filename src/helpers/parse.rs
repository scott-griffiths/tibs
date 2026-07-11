use super::bits::BV;
use super::raw_bytes::bv_from_bytes_slice;
use bitvec::prelude::*;
use lru::LruCache;
use once_cell::sync::Lazy;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::num::NonZeroUsize;
use std::sync::Mutex;

const BITS_CACHE_SIZE: usize = 1024;
static BITS_CACHE: Lazy<Mutex<LruCache<String, BV>>> =
    Lazy::new(|| Mutex::new(LruCache::new(NonZeroUsize::new(BITS_CACHE_SIZE).unwrap())));

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
    let new_hex_length = new_hex.len();
    if !new_hex_length.is_multiple_of(2) {
        new_hex.push('0');
    }
    let data = match hex::decode(&new_hex) {
        Ok(d) => d,
        Err(e) => {
            return Err(PyValueError::new_err(format!(
                "Cannot convert from hex '{hex}': {}",
                e
            )));
        }
    };
    let bv = bv_from_bytes_slice(data, None, Some(new_hex_length * 4))?;
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
        let mut cache = BITS_CACHE
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Internal bits cache mutex poisoned?"))?;
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
        let mut cache = BITS_CACHE
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Internal bits cache mutex poisoned?"))?;
        cache.put(s, result.clone());
    }
    Ok(result)
}
