use super::bits::BV;
use super::parse::str_to_bv;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyInt, PyList, PyMemoryView, PyTuple};
use pyo3::{PyErr, ffi};

pub(crate) fn convert_to_bool(bit: &Bound<'_, PyAny>) -> Option<bool> {
    if let Ok(b) = bit.cast::<PyBool>() {
        Some(b.is_true())
    } else if let Ok(val) = bit.extract::<i64>() {
        match val {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    } else {
        None
    }
}

pub(crate) fn bytes_like_to_vec(data: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if data.is_instance_of::<PyBytes>()
        || data.is_instance_of::<PyByteArray>()
        || data.is_instance_of::<PyMemoryView>()
    {
        data.extract::<Vec<u8>>()
    } else {
        Err(PyTypeError::new_err(
            "Expected a bytes-like object: bytes, bytearray or memoryview.",
        ))
    }
}

pub(crate) fn bv_from_bools(iterable: &Bound<'_, PyAny>) -> PyResult<BV> {
    // Lists and tuples are the common bulk path. Reading their items through
    // the C API avoids creating a Bound<PyAny> wrapper for every bit.
    if let Ok(list) = iterable.cast::<PyList>() {
        return unsafe {
            bv_from_py_sequence_items(
                iterable.py(),
                list.len(),
                |index| ffi::PyList_GetItem(list.as_ptr(), index as ffi::Py_ssize_t),
                py_truthy,
            )
        };
    }
    if let Ok(tuple) = iterable.cast::<PyTuple>() {
        return unsafe {
            bv_from_py_sequence_items(
                iterable.py(),
                tuple.len(),
                |index| ffi::PyTuple_GetItem(tuple.as_ptr(), index as ffi::Py_ssize_t),
                py_truthy,
            )
        };
    }

    // For sequences, we can pre-allocate the capacity.
    let capacity = iterable.len().ok().unwrap_or(64);
    let mut bv = BV::with_capacity(capacity);

    for value in iterable.try_iter()? {
        bv.push(value?.is_truthy()?);
    }
    Ok(bv)
}

fn bv_from_strict_bit_pattern(any: &Bound<'_, PyAny>) -> PyResult<Option<BV>> {
    if let Ok(list) = any.cast::<PyList>() {
        return unsafe {
            bv_from_py_sequence_items(
                any.py(),
                list.len(),
                |index| ffi::PyList_GetItem(list.as_ptr(), index as ffi::Py_ssize_t),
                |_py, index, item| strict_bit(index, &Bound::from_borrowed_ptr(any.py(), item)),
            )
            .map(Some)
        };
    }
    if let Ok(tuple) = any.cast::<PyTuple>() {
        return unsafe {
            bv_from_py_sequence_items(
                any.py(),
                tuple.len(),
                |index| ffi::PyTuple_GetItem(tuple.as_ptr(), index as ffi::Py_ssize_t),
                |_py, index, item| strict_bit(index, &Bound::from_borrowed_ptr(any.py(), item)),
            )
            .map(Some)
        };
    }
    Ok(None)
}

unsafe fn bv_from_py_sequence_items(
    py: Python<'_>,
    len: usize,
    mut get_item: impl FnMut(usize) -> *mut ffi::PyObject,
    mut convert_other: impl FnMut(Python<'_>, usize, *mut ffi::PyObject) -> PyResult<bool>,
) -> PyResult<BV> {
    // The callers pass list/tuple indices in bounds, so borrowed item pointers
    // are valid while the GIL is held. Pack directly into Msb0 bytes to avoid
    // BitVec::push overhead for large Python bool sequences.
    let mut bytes = vec![0u8; len.div_ceil(8)];
    let py_true = unsafe { ffi::Py_True() };
    let py_false = unsafe { ffi::Py_False() };

    for index in 0..len {
        let item = get_item(index);
        // Python bools are singletons, so pointer comparison handles the
        // benchmark path cheaply. Other objects keep normal truthiness.
        let bit = if item == py_true {
            true
        } else if item == py_false {
            false
        } else {
            convert_other(py, index, item)?
        };
        if bit {
            bytes[index / 8] |= 0x80 >> (index & 7);
        }
    }
    let mut bv = BV::from_vec(bytes);
    bv.truncate(len);
    Ok(bv)
}

fn py_truthy(py: Python<'_>, _index: usize, item: *mut ffi::PyObject) -> PyResult<bool> {
    match unsafe { ffi::PyObject_IsTrue(item) } {
        0 => Ok(false),
        1 => Ok(true),
        -1 => Err(PyErr::fetch(py)),
        _ => unreachable!("PyObject_IsTrue only returns -1, 0, or 1"),
    }
}

fn strict_bit(index: usize, item: &Bound<'_, PyAny>) -> PyResult<bool> {
    let Some(bit) = convert_to_bool(item) else {
        let type_name = py_type_name(item);
        let repr = match item.repr() {
            Ok(repr) => repr.to_string(),
            Err(_) => "<unrepresentable>".to_string(),
        };
        return Err(PyTypeError::new_err(format!(
            "Implicit bit patterns only accept True, False, 0 or 1; item at index {index} is {repr} of type <{type_name}>. Use from_bools(...) to convert truthy values, from_values(...) to pack numeric values, or from_bytes(...) for bytes."
        )));
    };
    Ok(bit)
}

fn py_type_name(any: &Bound<'_, PyAny>) -> String {
    match any.get_type().name() {
        Ok(name) => name.to_string(),
        Err(_) => "<unknown>".to_string(),
    }
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
        && let Ok(any_bytes) = any.extract::<Vec<u8>>()
    {
        return Ok(BV::from_vec(any_bytes));
    }

    // Is it an explicit bit pattern shorthand?
    if let Some(bv) = bv_from_strict_bit_pattern(any)? {
        return Ok(bv);
    }
    let type_name = py_type_name(any);
    let mut err = format!("Cannot promote object of type <{type_name}> to a Tibs/Mutibs object. ");
    if any.is_instance_of::<PyInt>() {
        err.push_str("Perhaps you want to use the class methods 'from_zeros()', 'from_ones()' or 'from_random()'?");
    } else {
        err.push_str("Use from_bytes(...) for bytes-like data, from_bools(...) for truthy iterables, or from_values(...) for typed numeric values.");
    };
    Err(PyTypeError::new_err(err))
}
