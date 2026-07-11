use crate::core::BitCollection;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

pub(crate) fn validate_length(length: i64) -> PyResult<usize> {
    if length < 0 {
        Err(PyValueError::new_err(format!(
            "Negative bit length given: {length}."
        )))
    } else {
        Ok(length as usize)
    }
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
