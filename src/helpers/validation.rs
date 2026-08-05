use crate::core::BitCollection;
use crate::helpers::BS;
use pyo3::exceptions::{PyIndexError, PyMemoryError, PyValueError};
use pyo3::prelude::*;

/// Check a caller-supplied bit length and narrow it to a `usize`.
///
/// The upper bound is what a container can actually hold: bitvec addresses a
/// bit with a `usize` and spends three of those bits on the position within an
/// element, so `BS::MAX_BITS` is 2^61 - 1 on a 64-bit build but only
/// 2^29 - 1 (about 64 MB) on a 32-bit one, which the x86 wheels are.
///
/// Both halves of this matter, and neither is theoretical:
///
/// * Without the check bitvec panics, and a panic crossing the FFI boundary
///   becomes pyo3's `PanicException`, which derives from `BaseException`
///   specifically so that it tears down the interpreter. An ordinary
///   `except Exception` does not catch it.
/// * The comparison has to happen before the cast. On a 32-bit build `usize`
///   is narrower than the `i64` that comes in from Python, so casting first
///   would silently truncate: a length of 2^32 + 100 would become 100 and
///   quietly succeed with the wrong size.
pub(crate) fn validate_length(length: i64) -> PyResult<usize> {
    if length < 0 {
        return Err(PyValueError::new_err(format!(
            "Negative bit length given: {length}."
        )));
    }
    if length as u64 > BS::MAX_BITS as u64 {
        return Err(PyMemoryError::new_err(format!(
            "Cannot create {length} bits: this build of tibs supports at most {} bits \
             ({} bytes) in one container.",
            BS::MAX_BITS,
            BS::MAX_BITS / 8
        )));
    }
    Ok(length as usize)
}

pub(crate) fn validate_offset(offset: i64) -> PyResult<usize> {
    if offset < 0 {
        return Err(PyValueError::new_err(format!(
            "Negative bit offset given: {offset}."
        )));
    }
    validate_length(offset)
}

pub(crate) fn validate_logical_op_lengths(a: usize, b: usize) -> PyResult<()> {
    if a != b {
        Err(PyValueError::new_err(format!(
            "For logical operations the lengths of both objects must match. Received lengths of {a} and {b} bits."
        )))
    } else {
        Ok(())
    }
}

/// Validates the index is in range and returns an absolute bit index.
#[inline(always)]
pub(crate) fn validate_index(index: isize, length: usize) -> PyResult<usize> {
    let index_p = if index < 0 {
        length as isize + index
    } else {
        index
    };
    if index_p >= length as isize || index_p < 0 {
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
    start: Option<isize>,
    end: Option<isize>,
) -> PyResult<(usize, usize)> {
    let mut start = start.unwrap_or(0);
    let mut end = end.unwrap_or(length as isize);
    if start < 0 {
        start += length as isize;
    }
    if end < 0 {
        end += length as isize;
    }

    if !(0 <= start && start <= end && end <= length as isize) {
        return Err(PyValueError::new_err(format!(
            "Invalid slice positions for length of {length}: start={start}, end={end}."
        )));
    }
    Ok((start as usize, end as usize))
}

pub(crate) fn normalize_split_position(position: isize, length: usize) -> PyResult<usize> {
    let mut normalized = position;
    if normalized < 0 {
        normalized += length as isize;
    }
    if normalized < 0 || normalized > length as isize {
        return Err(PyValueError::new_err(format!(
            "Split position {position} is out of range for length of {length}."
        )));
    }
    Ok(normalized as usize)
}
