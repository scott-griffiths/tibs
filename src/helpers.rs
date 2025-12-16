use crate::core::BitCollection;
use bitvec::prelude::*;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::rngs::{OsRng, StdRng};
use rand::{RngCore, SeedableRng, TryRngCore};
use sha2::{Digest, Sha256};

pub type BV = BitVec<u8, Msb0>;
pub type BS = BitSlice<u8, Msb0>;

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
    haystack: &BV,
    needle: &BV,
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
    haystack: &BV,
    needle: &BV,
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
            let mut seed_arr = [0u8; 32]; //  TODO: from entropy?
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

// TODO: Similar helper methods for from_joined, from_bools etc.
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
    let num_bytes = (length + 7) / 8;
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
