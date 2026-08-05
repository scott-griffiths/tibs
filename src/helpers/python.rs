use super::bits::BV;
use super::parse::str_to_bv;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyInt, PyList, PyMemoryView, PyTuple};
use pyo3::{PyErr, ffi};

/// One pre-built list of 8 bools for every byte value, so whole bytes can be
/// converted to list items with a single C API call. The cached lists never
/// escape: `PyList_SetSlice` copies their items out.
static BOOL_CHUNKS: pyo3::sync::PyOnceLock<Vec<Py<PyList>>> = pyo3::sync::PyOnceLock::new();

fn build_bool_chunks(py: Python<'_>) -> PyResult<Vec<Py<PyList>>> {
    (0u16..256)
        .map(|value| {
            let bools: [bool; 8] = std::array::from_fn(|i| value & (0x80 >> i) != 0);
            Ok(PyList::new(py, bools)?.unbind())
        })
        .collect()
}

/// Build a Python list of bools from a bit slice.
///
/// This is called for potentially millions of bits, where per-item C API
/// calls dominate, so whole storage bytes are appended via the byte-value
/// lookup table and only the partial edge bits are appended individually.
pub(crate) fn bitslice_to_bool_list(
    py: Python<'_>,
    slice: &super::bits::BS,
) -> PyResult<Py<PyList>> {
    let chunks = BOOL_CHUNKS.get_or_try_init(py, || build_bool_chunks(py))?;

    unsafe fn append_bits(
        py: Python<'_>,
        list: *mut ffi::PyObject,
        bits: u8,
        n: usize,
    ) -> PyResult<()> {
        // The n bits are right-aligned in `bits`, most significant first.
        for i in 0..n {
            let obj = if bits & (1 << (n - 1 - i)) != 0 {
                unsafe { ffi::Py_True() }
            } else {
                unsafe { ffi::Py_False() }
            };
            if unsafe { ffi::PyList_Append(list, obj) } != 0 {
                return Err(PyErr::fetch(py));
            }
        }
        Ok(())
    }

    unsafe {
        match slice.domain() {
            bitvec::domain::Domain::Enclave(elem) => {
                let list = ffi::PyList_New(0);
                if list.is_null() {
                    return Err(PyErr::fetch(py));
                }
                let list_guard = Bound::from_owned_ptr(py, list).cast_into::<PyList>()?;
                let head = elem.head().into_inner() as usize;
                let bits = elem.load_value() >> (8 - head - slice.len());
                append_bits(py, list, bits, slice.len())?;
                Ok(list_guard.unbind())
            }
            bitvec::domain::Domain::Region { head, body, tail } => {
                let live_head = match &head {
                    Some(elem) => 8 - elem.head().into_inner() as usize,
                    None => 0,
                };
                // Allocate the body at full size with one call, then only the
                // bytes that differ from the all-zeros seed need a slice
                // replacement. This keeps the C API call count per byte, not
                // per bit, and avoids incremental list growth.
                let list =
                    ffi::PySequence_Repeat(chunks[0].as_ptr(), body.len() as ffi::Py_ssize_t);
                if list.is_null() {
                    return Err(PyErr::fetch(py));
                }
                let list_guard = Bound::from_owned_ptr(py, list).cast_into::<PyList>()?;
                if let Some(elem) = head {
                    let head_list = ffi::PyList_New(0);
                    if head_list.is_null() {
                        return Err(PyErr::fetch(py));
                    }
                    let head_guard = Bound::from_owned_ptr(py, head_list);
                    append_bits(py, head_list, elem.load_value(), live_head)?;
                    if ffi::PyList_SetSlice(list, 0, 0, head_list) != 0 {
                        return Err(PyErr::fetch(py));
                    }
                    drop(head_guard);
                }
                for (index, &byte) in body.iter().enumerate() {
                    if byte == 0 {
                        continue;
                    }
                    let at = (live_head + index * 8) as ffi::Py_ssize_t;
                    if ffi::PyList_SetSlice(list, at, at + 8, chunks[byte as usize].as_ptr()) != 0 {
                        return Err(PyErr::fetch(py));
                    }
                }
                let live_tail = slice.len() - live_head - body.len() * 8;
                if live_tail > 0
                    && let Some(elem) = tail
                {
                    append_bits(py, list, elem.load_value() >> (8 - live_tail), live_tail)?;
                }
                Ok(list_guard.unbind())
            }
        }
    }
}

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

pub(crate) fn try_extract_index(index: &Bound<'_, PyAny>) -> PyResult<Option<isize>> {
    let py = index.py();
    let indexed = unsafe { ffi::PyNumber_Index(index.as_ptr()) };
    if indexed.is_null() {
        let error = PyErr::fetch(py);
        return if error.is_instance_of::<PyTypeError>(py) {
            Ok(None)
        } else {
            Err(error)
        };
    }

    let value = unsafe { ffi::PyLong_AsSsize_t(indexed) };
    unsafe { ffi::Py_DECREF(indexed) };
    if value == -1 && unsafe { !ffi::PyErr_Occurred().is_null() } {
        Err(PyErr::fetch(py))
    } else {
        Ok(Some(value))
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
    if let Ok(any_string) = any.extract::<&str>() {
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
