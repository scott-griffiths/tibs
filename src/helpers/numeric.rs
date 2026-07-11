use super::bits::BV;
use bitvec::prelude::*;
use half::f16;
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

#[inline]
pub(crate) fn bv_from_u128(value: u128, length: usize, is_little_endian: bool) -> PyResult<BV> {
    if length == 0 || length > 128 {
        return Err(PyValueError::new_err(format!(
            "Bit length for unsigned int must be between 1 and 128. Received {length}."
        )));
    }
    // Special case for 128 to avoid overflow in more general case
    if length == 128 {
        let mut bv = BV::repeat(false, 128);
        if is_little_endian {
            bv.store_le(value);
        } else {
            bv.store_be(value);
        }
        return Ok(bv);
    }
    if value >= (1u128 << length) {
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
    Ok(bv)
}

#[inline]
pub(crate) fn bv_from_i128(value: i128, length: usize, is_little_endian: bool) -> PyResult<BV> {
    if length == 0 || length > 128 {
        return Err(PyValueError::new_err(format!(
            "Bit length for signed int must be between 1 and 128. Received {length}."
        )));
    }
    // Special case for 128 to avoid overflow in more general case
    if length == 128 {
        let mut bv = BV::repeat(value < 0, 128);
        if is_little_endian {
            bv.store_le(value);
        } else {
            bv.store_be(value);
        }
        return Ok(bv);
    }
    let min_val = -(1i128 << (length - 1));
    let max_val = (1i128 << (length - 1)) - 1;
    if value < min_val || value > max_val {
        return Err(PyOverflowError::new_err(format!(
            "Value {value} does not fit in {length} signed bits."
        )));
    }
    let repeat_bit = value < 0;
    let mut bv = BV::repeat(repeat_bit, length);
    if is_little_endian {
        bv.store_le(value);
    } else {
        bv.store_be(value);
    }
    Ok(bv)
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
