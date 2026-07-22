use super::bits::{BS, BV};
use bitvec::prelude::*;
use half::f16;
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyInt};
use pyo3::{PyErr, ffi, intern};

/// Integers up to this many bits are converted through `u64` / `i64`, which is
/// a single C API call in each direction. Longer ones go through
/// `int.to_bytes` and `int.from_bytes`, which has no length limit but costs a
/// Python method call. There's no upper bound on the length either way.
pub(crate) const FAST_INT_BITS: usize = 64;

/// The `byteorder` argument shared by `int.to_bytes` and `int.from_bytes`.
pub(crate) fn byte_order_name(is_little_endian: bool) -> &'static str {
    if is_little_endian { "little" } else { "big" }
}

fn overflow_error(value: &Bound<'_, PyAny>, length: usize, signed: bool) -> PyErr {
    let signed_text = if signed { " signed" } else { "" };
    // A value with more than a few thousand digits can't be rendered at all,
    // because of CPython's integer to string conversion limit. Such a value is
    // certainly too big for the field, so report the field on its own.
    match value.str() {
        Ok(shown) => PyOverflowError::new_err(format!(
            "Value {shown} does not fit in {length}{signed_text} bits."
        )),
        Err(_) => {
            PyOverflowError::new_err(format!("Value does not fit in {length}{signed_text} bits."))
        }
    }
}

fn zero_length_error(signed: bool) -> PyErr {
    let kind = if signed { "signed" } else { "unsigned" };
    PyValueError::new_err(format!(
        "Bit length for {kind} int must be at least 1. Received 0."
    ))
}

/// Apply `__index__`, so that anything Python accepts as an integer reaches
/// the `int.to_bytes` call below with the right type and the right error.
fn to_index<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if value.is_instance_of::<PyInt>() {
        return Ok(value.clone());
    }
    let py = value.py();
    // PyNumber_Index returns a new reference, or null with the error set.
    let indexed = unsafe { ffi::PyNumber_Index(value.as_ptr()) };
    if indexed.is_null() {
        return Err(PyErr::fetch(py));
    }
    unsafe { Bound::from_owned_ptr_or_err(py, indexed) }
}

/// Convert an integer of any size through `int.to_bytes`.
///
/// The bytes come back with the value right-aligned in `length.div_ceil(8)`
/// bytes, so for a length that isn't a whole number of bytes the leading pad
/// bits have to be checked before they're dropped: `to_bytes` only knows
/// whether the value fitted in the bytes, not whether it fitted in the field.
fn bv_from_big_int(
    value: &Bound<'_, PyAny>,
    length: usize,
    is_little_endian: bool,
    signed: bool,
) -> PyResult<BV> {
    let value = to_index(value)?;
    let py = value.py();
    let byte_length = length.div_ceil(8);
    let byte_order = byte_order_name(is_little_endian);
    let args = (byte_length, byte_order);

    // `signed` is keyword-only and defaults to False, so the unsigned case can
    // skip building a kwargs dict.
    let call = if signed {
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "signed"), true)?;
        value.call_method(intern!(py, "to_bytes"), args, Some(&kwargs))
    } else {
        value.call_method1(intern!(py, "to_bytes"), args)
    };
    let bytes = match call {
        Ok(bytes) => bytes,
        Err(err) if err.is_instance_of::<PyOverflowError>(py) => {
            return Err(overflow_error(&value, length, signed));
        }
        Err(err) => return Err(err),
    };
    let bytes = bytes.cast_into::<PyBytes>()?;
    let bits = BS::from_slice(bytes.as_bytes());

    // A little-endian byte order is only allowed for whole-byte lengths, so
    // there are never any pad bits to trim in that case.
    let pad = byte_length * 8 - length;
    if pad == 0 {
        return Ok(bits.to_bitvec());
    }
    debug_assert!(!is_little_endian);
    let fits = if signed {
        // to_bytes sign extended into the pad, so the value only fits if the
        // field's own sign bit continues that run.
        let sign = bits[pad];
        bits[..pad].iter().all(|bit| *bit == sign)
    } else {
        bits[..pad].not_any()
    };
    if !fits {
        return Err(overflow_error(&value, length, signed));
    }
    Ok(bits[pad..].to_bitvec())
}

#[inline]
pub(crate) fn bv_from_uint(
    value: &Bound<'_, PyAny>,
    length: usize,
    is_little_endian: bool,
) -> PyResult<BV> {
    if length == 0 {
        return Err(zero_length_error(false));
    }
    if length <= FAST_INT_BITS {
        // A failed extraction means the value is negative, too big for a u64,
        // or not an integer at all. All three are handled properly by the
        // general path below, which reports them against the field length.
        if let Ok(value) = value.extract::<u64>() {
            if length < FAST_INT_BITS && value >= (1u64 << length) {
                return Err(PyOverflowError::new_err(format!(
                    "Value {value} does not fit in {length} bits."
                )));
            }
            let mut bv = BV::repeat(false, length);
            if is_little_endian {
                bv.store_le(value);
            } else {
                bv.store_be(value);
            }
            return Ok(bv);
        }
    }
    bv_from_big_int(value, length, is_little_endian, false)
}

#[inline]
pub(crate) fn bv_from_int(
    value: &Bound<'_, PyAny>,
    length: usize,
    is_little_endian: bool,
) -> PyResult<BV> {
    if length == 0 {
        return Err(zero_length_error(true));
    }
    if length <= FAST_INT_BITS {
        if let Ok(value) = value.extract::<i64>() {
            if length < FAST_INT_BITS {
                let min_val = -(1i64 << (length - 1));
                let max_val = (1i64 << (length - 1)) - 1;
                if value < min_val || value > max_val {
                    return Err(PyOverflowError::new_err(format!(
                        "Value {value} does not fit in {length} signed bits."
                    )));
                }
            }
            let mut bv = BV::repeat(value < 0, length);
            if is_little_endian {
                bv.store_le(value);
            } else {
                bv.store_be(value);
            }
            return Ok(bv);
        }
    }
    bv_from_big_int(value, length, is_little_endian, true)
}

pub(crate) fn bv_from_f64(value: f64, length: usize, is_little_endian: bool) -> PyResult<BV> {
    let bv = match length {
        64 => {
            let mut bv = BV::repeat(false, 64);
            if is_little_endian {
                bv.store_le(value.to_bits());
            } else {
                bv.store_be(value.to_bits());
            }
            bv
        }
        32 => {
            let value_f32 = value as f32;
            let mut bv = BV::repeat(false, 32);
            if is_little_endian {
                bv.store_le(value_f32.to_bits());
            } else {
                bv.store_be(value_f32.to_bits());
            }
            bv
        }
        16 => {
            let value_f16 = f16::from_f64(value);
            let mut bv = BV::repeat(false, 16);
            if is_little_endian {
                bv.store_le(value_f16.to_bits());
            } else {
                bv.store_be(value_f16.to_bits());
            }
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
