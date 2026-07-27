use crate::codec as tibs_codec;
use crate::core::{
    BitCollection, concatenate_bitcollections, push_collection_run, repeat_bitcollection,
};
use crate::dtype::extract_dtype;
use crate::enums::{BitOrder, ByteOrder, Codec};
use crate::helpers::{
    BS, BV, BitConcat, LogicalOp, MaskedMatcher, bv_from_bin, bv_from_bools, bv_from_bytes_slice,
    bv_from_f64, bv_from_hex, bv_from_int, bv_from_oct, bv_from_ones, bv_from_random, bv_from_uint,
    bv_from_zeros, bytes_like_to_vec, copy_bits, deposit_masked, find_bitvec, find_bitvec_aligned,
    head_bit_offset, move_bits, padded_bytes_from_offset, promote_to_bv, rotate_bits_left,
    str_to_bv, validate_index, validate_length, validate_logical_op_lengths, validate_shift,
    validate_slice,
};
use crate::tibs_::{
    Tibs, bv_from_value, bv_from_values_iter, prepare_mask, py_from_value, py_values_from_range,
};
use crate::view::{MutableView, View};

use crate::helpers;
use pyo3::exceptions::{
    PyAttributeError, PyIndexError, PyOverflowError, PyTypeError, PyValueError,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyList, PySlice, PyTuple, PyType};
use std::ops::Not;

///     A mutable container of binary data.
///
///     To construct, use a builder 'from' method:
///
///     * ``Mutibs.from_bin(s)`` - Create from a binary string, optionally starting with '0b'.
///     * ``Mutibs.from_oct(s)`` - Create from an octal string, optionally starting with '0o'.
///     * ``Mutibs.from_hex(s)`` - Create from a hex string, optionally starting with '0x'.
///     * ``Mutibs.from_u(u, length, [byte_order])`` - Create from an unsigned int to a given length.
///     * ``Mutibs.from_i(i, length, [byte_order])`` - Create from a signed int to a given length.
///     * ``Mutibs.from_f(f, length, [byte_order])`` - Create from an IEEE float to a 16, 32 or 64 bit length.
///     * ``Mutibs.from_bytes(b)`` - Create directly from a ``bytes``, ``bytearray`` or ``memoryview`` object.
///     * ``Mutibs.from_string(s)`` - Use a formatted string.
///     * ``Mutibs.from_bools(iterable)`` - Convert each element in ``iterable`` to a bool.
///     * ``Mutibs.from_zeros(length)`` - Initialise with ``length`` ``0`` bits.
///     * ``Mutibs.from_ones(length)`` - Initialise with ``length`` ``1`` bits.
///     * ``Mutibs.from_random(length, [secure, seed])`` - Initialise with ``length`` randomly set bits.
///     * ``Mutibs.from_joined(iterable)`` - Concatenate an iterable of objects.
///
///     Using ``Mutibs(auto)`` will try to delegate to ``from_string``, ``from_bytes`` or ``from_bools``.
///
#[pyclass(freelist = 8, sequence, skip_from_py_object, module = "tibs")]
pub struct Mutibs {
    pub data: BV,
}

// Not derived: a derived Clone would call BitVec's own Clone, which rebuilds the
// storage one element at a time. Going through to_bitvec copies bytes instead.
// Every copying method here (inverted, reversed, rotated_*, inserted, ...) is
// written as `self.clone()` followed by a mutation, so this one impl carries the
// whole family.
impl Clone for Mutibs {
    #[inline]
    fn clone(&self) -> Self {
        Mutibs::from_bv(self.to_bitvec())
    }
}

enum JoinedPart<'py> {
    // Keep existing bit containers borrowed during a join. Promoted Python
    // values still need owned storage because their BitVec is created here.
    Tibs(PyRef<'py, Tibs>),
    Mutibs(PyRef<'py, Mutibs>),
    Owned(BV),
}

// Internal methods, not exported to Python
impl Mutibs {
    pub(crate) fn from_bv(bv: BV) -> Self {
        Mutibs { data: bv }
    }

    #[inline]
    pub(crate) fn as_bitvec_ref(&self) -> &BV {
        &self.data
    }

    #[inline]
    pub(crate) fn as_bitslice(&self) -> &BS {
        self.as_bitvec_ref().as_bitslice()
    }

    /// A copy of `length` bits starting `start_bit` bits in, as an owned
    /// BitVec whose own storage starts on a byte boundary.
    ///
    /// Copying through the raw bytes is a shift-and-copy sweep. The obvious
    /// spelling, `self.as_bitslice()[range].to_bitvec()`, instead rebuilds the
    /// storage one element at a time *and* keeps whatever head offset the
    /// source had, which then denies the byte-wide paths to everything done to
    /// the result afterwards. Landing on bit zero also makes this agree with
    /// `Tibs::to_bitvec`, so both promotion routes now produce the same shape.
    ///
    /// `start_bit + length` must be within the bit length.
    pub(crate) fn copied_range(&self, start_bit: usize, length: usize) -> BV {
        debug_assert!(start_bit + length <= self.len());
        if length == 0 {
            return BV::new();
        }
        // Where the wanted bits begin within the backing storage, which itself
        // need not start on a byte boundary.
        let absolute = self.storage_head_offset() + start_bit;
        let bytes = &self.data.as_raw_slice()[absolute / 8..];
        let mut result = BV::from_vec(padded_bytes_from_offset(bytes, absolute % 8, length));
        result.truncate(length);
        result
    }

    #[inline]
    pub(crate) fn to_bitvec(&self) -> BV {
        self.copied_range(0, self.len())
    }

    /// Append the `len` bits starting `offset` bits into `src`, in place.
    ///
    /// Grows the byte storage and copies over bytes. Rebuilding into a fresh
    /// buffer would make a loop of appends quadratic, so this keeps the
    /// `Vec`'s amortised growth, and `extend_from_bitslice` - which does grow
    /// amortised - copies a bit at a time.
    pub(crate) fn append_run(&mut self, src: &[u8], offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        if self.storage_head_offset() != 0 {
            // Storage starting mid byte cannot be appended to over bytes, so
            // realign it once; the result starts at bit zero.
            self.data = self.to_bitvec();
        }
        let old_len = self.len();
        let new_len = old_len + len;
        let mut bytes = std::mem::take(&mut self.data).into_vec();
        bytes.resize(new_len.div_ceil(8), 0);
        copy_bits(&mut bytes, old_len, src, offset, len);
        let mut grown = BV::from_vec(bytes);
        grown.truncate(new_len);
        self.data = grown;
    }

    /// Append every bit of `bits` in place.
    pub(crate) fn append_collection(&mut self, bits: &impl BitCollection) {
        match bits.raw_data_ref() {
            Some((bytes, offset, _)) => self.append_run(bytes, offset, bits.len()),
            None => {
                let bytes = bits.padded_byte_data_cow().into_owned();
                self.append_run(&bytes, 0, bits.len())
            }
        }
    }

    #[inline]
    pub(crate) fn as_mut_bitvec_ref(&mut self) -> &mut BV {
        &mut self.data
    }

    #[inline]
    pub(crate) fn raw_bytes(&self) -> Vec<u8> {
        self.data.as_raw_slice().to_vec()
    }

    pub(crate) fn joined_bv_from_iterable(iterable: &Bound<'_, PyAny>) -> PyResult<BV> {
        if let Ok(list) = iterable.cast::<PyList>()
            && let Some(bv) = Self::joined_bv_from_repeated_list(list)
        {
            return Ok(bv);
        }

        // Walk the iterable once to collect bit views and compute the final
        // length, so the destination BitVec can be allocated exactly once.
        let iter = iterable.try_iter()?;
        let mut parts = Vec::new();
        let mut total_len: usize = 0;
        for item in iter {
            Self::push_joined_part(&mut parts, &mut total_len, item?)?;
        }
        Self::join_parts(parts, total_len)
    }

    fn joined_bv_from_repeated_list(list: &Bound<'_, PyList>) -> Option<BV> {
        let count = list.len();
        if count == 0 {
            return Some(BV::new());
        }

        // All indices are in bounds, and list items remain valid while the GIL is held.
        let first = unsafe { ffi::PyList_GetItem(list.as_ptr(), 0) };
        for index in 1..count {
            if unsafe { ffi::PyList_GetItem(list.as_ptr(), index as ffi::Py_ssize_t) } != first {
                return None;
            }
        }

        let item = unsafe { Bound::from_borrowed_ptr(list.py(), first) };
        if let Ok(tibs) = item.extract::<PyRef<Tibs>>() {
            Some(repeat_bitcollection(&*tibs, count))
        } else if let Ok(mutibs) = item.extract::<PyRef<Mutibs>>() {
            Some(repeat_bitcollection(&*mutibs, count))
        } else {
            None
        }
    }

    fn push_joined_part<'py>(
        parts: &mut Vec<JoinedPart<'py>>,
        total_len: &mut usize,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<()> {
        if let Ok(tibs) = obj.extract::<PyRef<Tibs>>() {
            *total_len += tibs.len();
            parts.push(JoinedPart::Tibs(tibs));
        } else if let Ok(mutibs) = obj.extract::<PyRef<Mutibs>>() {
            *total_len += mutibs.len();
            parts.push(JoinedPart::Mutibs(mutibs));
        } else {
            let bv = promote_to_bv(&obj)?;
            *total_len += bv.len();
            parts.push(JoinedPart::Owned(bv));
        }
        Ok(())
    }

    fn join_parts(parts: Vec<JoinedPart<'_>>, total_len: usize) -> PyResult<BV> {
        // Copy into storage sized once, over bytes. `copy_from_bitslice` moves
        // a bit at a time even when both sides are byte aligned.
        let mut out = BitConcat::with_bit_capacity(total_len);
        for part in parts {
            match &part {
                JoinedPart::Tibs(tibs) => push_collection_run(&mut out, &**tibs),
                JoinedPart::Mutibs(mutibs) => push_collection_run(&mut out, &**mutibs),
                JoinedPart::Owned(bv) => out.push_run(
                    bv.as_raw_slice(),
                    head_bit_offset(bv.as_bitslice()),
                    bv.len(),
                ),
            }
        }
        Ok(out.into_bitvec())
    }

    #[inline]
    pub fn set_index(&mut self, index: isize) -> PyResult<()> {
        let index = validate_index(index, self.len())?;
        self.write_bits_raw(&[index], true);
        Ok(())
    }

    #[inline]
    pub fn unset_index(&mut self, index: isize) -> PyResult<()> {
        let index = validate_index(index, self.len())?;
        self.write_bits_raw(&[index], false);
        Ok(())
    }

    /// The bit position within the underlying storage at which this Mutibs
    /// starts. Usually zero, but slicing can produce a bit vector whose
    /// storage begins mid-byte.
    #[inline]
    pub(crate) fn storage_head_offset(&self) -> usize {
        match self.data.as_bitslice().domain() {
            bitvec::domain::Domain::Enclave(elem) => elem.head().into_inner() as usize,
            bitvec::domain::Domain::Region {
                head: Some(elem), ..
            } => elem.head().into_inner() as usize,
            _ => 0,
        }
    }

    /// Write already-validated bit indices directly into the underlying bytes.
    #[inline]
    fn write_bits_raw(&mut self, indices: &[usize], value: bool) {
        let head = self.storage_head_offset();
        let raw = self.data.as_raw_mut_slice();
        if value {
            for &index in indices {
                let index = head + index;
                raw[index >> 3] |= 0x80u8 >> (index & 7);
            }
        } else {
            for &index in indices {
                let index = head + index;
                raw[index >> 3] &= !(0x80u8 >> (index & 7));
            }
        }
    }

    /// Install `bytes` as the storage, keeping only the first `length` bits.
    #[inline]
    fn set_data_from_bytes(&mut self, mut bytes: Vec<u8>, length: usize) {
        bytes.truncate(length.div_ceil(8));
        let mut data = BV::from_vec(bytes);
        data.truncate(length);
        self.data = data;
    }

    /// Replace the bits in `[start, end)` with `value`.
    ///
    /// The bits before `start` never move, so the edit is a slide of the bits
    /// after `end` and nothing more. Storage that starts mid-byte cannot be
    /// grown in place, so that case is realigned into a new buffer instead.
    fn splice_raw(
        &mut self,
        start: usize,
        end: usize,
        value: &[u8],
        value_offset: usize,
        value_len: usize,
    ) {
        let length = self.len();
        debug_assert!(start <= end && end <= length);
        let new_length = length - (end - start) + value_len;
        let target = start + value_len;
        let head = self.storage_head_offset();

        if head != 0 {
            let mut bytes = vec![0u8; new_length.div_ceil(8)];
            let source = self.data.as_raw_slice();
            copy_bits(&mut bytes, 0, source, head, start);
            copy_bits(&mut bytes, start, value, value_offset, value_len);
            copy_bits(&mut bytes, target, source, head + end, length - end);
            self.set_data_from_bytes(bytes, new_length);
            return;
        }

        let mut bytes = std::mem::take(&mut self.data).into_vec();
        let byte_length = new_length.div_ceil(8);
        if bytes.len() < byte_length {
            bytes.resize(byte_length, 0);
        }
        move_bits(&mut bytes, end, target, length - end);
        copy_bits(&mut bytes, start, value, value_offset, value_len);
        self.set_data_from_bytes(bytes, new_length);
    }

    pub(crate) fn set_slice(&mut self, start: usize, end: usize, value: &Tibs) {
        // A slice that runs backwards is an insertion at `start` in Python.
        let end = end.max(start);
        let (value_bytes, value_offset) = value.raw_data_with_offset();
        if end - start == value.len() {
            // An overwrite moves no data, so write straight over the bytes.
            let storage_start = self.storage_head_offset() + start;
            copy_bits(
                self.data.as_raw_mut_slice(),
                storage_start,
                value_bytes,
                value_offset,
                value.len(),
            );
            return;
        }
        self.splice_raw(start, end, value_bytes, value_offset, value.len());
    }

    fn delete_slice(&mut self, start: usize, end: usize) {
        self.splice_raw(start, end, &[], 0, 0);
    }

    /// Remove the bits at `positions`, which must be sorted and distinct.
    ///
    /// The surviving runs are packed into a new buffer in one pass, rather
    /// than closing up the gap once per deleted bit.
    fn delete_positions(&mut self, positions: &[usize]) {
        if positions.is_empty() {
            return;
        }
        let length = self.len();
        debug_assert!(positions.iter().all(|&pos| pos < length));
        let new_length = length - positions.len();
        let head = self.storage_head_offset();
        let source = self.data.as_raw_slice();

        let mut bytes = vec![0u8; new_length.div_ceil(8)];
        let mut written = 0;
        let mut read = 0;
        for &pos in positions {
            copy_bits(&mut bytes, written, source, head + read, pos - read);
            written += pos - read;
            read = pos + 1;
        }
        copy_bits(&mut bytes, written, source, head + read, length - read);

        self.set_data_from_bytes(bytes, new_length);
    }

    #[inline]
    fn assign_from_bv(&mut self, value: BV) {
        debug_assert_eq!(self.len(), value.len());
        self.as_mut_bitvec_ref()
            .copy_from_bitslice(value.as_bitslice());
    }

    #[inline]
    fn assign_u(&mut self, u: &Bound<'_, PyAny>) -> PyResult<()> {
        let length = self.len();
        let value = bv_from_uint(u, length, false)?;
        self.assign_from_bv(value);
        Ok(())
    }

    #[inline]
    fn assign_i(&mut self, i: &Bound<'_, PyAny>) -> PyResult<()> {
        let length = self.len();
        let value = bv_from_int(i, length, false)?;
        self.assign_from_bv(value);
        Ok(())
    }

    #[inline]
    fn assign_f(&mut self, f: f64) -> PyResult<()> {
        let length = self.len();
        let value = bv_from_f64(f, length, false)?;
        self.assign_from_bv(value);
        Ok(())
    }

    #[inline]
    fn replace_with_bv(&mut self, value: BV) {
        self.data = value;
    }

    pub(crate) fn ixor(&mut self, other: &BS) -> PyResult<()> {
        validate_logical_op_lengths(self.len(), other.len())?;
        *self.as_mut_bitvec_ref() ^= other;
        Ok(())
    }

    pub(crate) fn ior(&mut self, other: &BS) -> PyResult<()> {
        validate_logical_op_lengths(self.len(), other.len())?;
        *self.as_mut_bitvec_ref() |= other;
        Ok(())
    }

    pub(crate) fn iand(&mut self, other: &BS) -> PyResult<()> {
        validate_logical_op_lengths(self.len(), other.len())?;
        *self.as_mut_bitvec_ref() &= other;
        Ok(())
    }

    pub(crate) fn set_from_sequence(
        &mut self,
        value: bool,
        mut indices: Vec<isize>,
    ) -> PyResult<()> {
        let len = self.len();
        for index in &mut indices {
            *index = validate_index(*index, len)? as isize;
        }
        let head = self.storage_head_offset();
        let raw = self.data.as_raw_mut_slice();
        for index in indices {
            let index = head + index as usize;
            let mask = 0x80u8 >> (index & 7);
            if value {
                raw[index >> 3] |= mask;
            } else {
                raw[index >> 3] &= !mask;
            }
        }
        Ok(())
    }

    fn set_from_list(&mut self, value: bool, list: &Bound<'_, PyList>) -> PyResult<()> {
        const STACK_CAPACITY: usize = 16;

        let count = list.len();
        if count <= STACK_CAPACITY {
            let mut indices = [0usize; STACK_CAPACITY];
            self.validate_list_indices(list, &mut indices[..count])?;
            self.write_bits_raw(&indices[..count], value);
        } else {
            let mut indices = vec![0usize; count];
            self.validate_list_indices(list, &mut indices)?;
            self.write_bits_raw(&indices, value);
        }
        Ok(())
    }

    fn validate_list_indices(
        &self,
        list: &Bound<'_, PyList>,
        indices: &mut [usize],
    ) -> PyResult<()> {
        debug_assert_eq!(list.len(), indices.len());
        let py = list.py();
        let len = self.len();
        for (position, index) in indices.iter_mut().enumerate() {
            // The index is in bounds and the borrowed item remains valid while
            // the list is held under the GIL.
            let item = unsafe { ffi::PyList_GetItem(list.as_ptr(), position as ffi::Py_ssize_t) };
            let value = if unsafe { ffi::PyLong_Check(item) } != 0 {
                unsafe { ffi::PyLong_AsSsize_t(item) }
            } else {
                let indexed = unsafe { ffi::PyNumber_Index(item) };
                if indexed.is_null() {
                    return Err(PyErr::fetch(py));
                }
                let value = unsafe { ffi::PyLong_AsSsize_t(indexed) };
                unsafe { ffi::Py_DECREF(indexed) };
                value
            };
            if value == -1 && unsafe { !ffi::PyErr_Occurred().is_null() } {
                return Err(PyErr::fetch(py));
            }
            *index = validate_index(value, len)?;
        }
        Ok(())
    }

    /// Collect bit positions from any iterable of ints, with a fast path for
    /// sequences. Errors raised while iterating or converting items propagate.
    fn collect_position_indices(pos: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
        if let Ok(indices) = pos.extract::<Vec<isize>>() {
            return Ok(indices);
        }
        let capacity = pos.len().ok().unwrap_or(8);
        let mut indices = Vec::with_capacity(capacity);
        for item in pos.try_iter()? {
            indices.push(item?.extract::<isize>()?);
        }
        Ok(indices)
    }

    pub(crate) fn apply_set_positions(
        &mut self,
        value: bool,
        pos: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if let Ok(list) = pos.cast::<PyList>() {
            self.set_from_list(value, list)?;
        } else if let Ok(index) = pos.extract::<isize>() {
            if value {
                self.set_index(index)?;
            } else {
                self.unset_index(index)?;
            }
        } else if pos.is_instance_of::<pyo3::types::PyRange>() {
            let start = pos
                .getattr("start")?
                .extract::<Option<isize>>()?
                .unwrap_or(0);
            let stop = pos.getattr("stop")?.extract::<isize>()?;
            let step = pos
                .getattr("step")?
                .extract::<Option<isize>>()?
                .unwrap_or(1);
            self.set_from_slice(value, start, stop, step)?;
        } else {
            let indices = Self::collect_position_indices(pos)?;
            self.set_from_sequence(value, indices)?;
        }

        Ok(())
    }

    pub(crate) fn apply_rotation(
        &mut self,
        n: i64,
        start: Option<isize>,
        end: Option<isize>,
        rotate_left: bool,
    ) -> PyResult<()> {
        if self.is_empty() {
            return Err(PyValueError::new_err("Cannot rotate an empty Mutibs."));
        }
        let n = match n {
            ..0 => {
                return Err(PyValueError::new_err("Cannot rotate by a negative amount."));
            }
            _ => n as usize,
        };

        let (start, end) = validate_slice(self.len(), start, end)?;
        if start != end {
            let span = end - start;
            let n = n % span;
            // Rotating right by n is rotating left by the rest of the span.
            let by = if rotate_left { n } else { (span - n) % span };
            // The span is measured from the first bit of the value, which need
            // not be the first bit of the storage.
            let head = self.storage_head_offset();
            let bytes = self.as_mut_bitvec_ref().as_raw_mut_slice();
            rotate_bits_left(bytes, head + start, span, by);
        }
        Ok(())
    }

    pub(crate) fn apply_byte_swap(
        &mut self,
        byte_length: Option<i64>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<()> {
        let (start, end) = validate_slice(self.len(), start, end)?;
        let swapped = BitCollection::byte_swap_copy(
            &self.get_slice_unchecked(start, end - start),
            byte_length,
        )?;
        self.as_mut_bitvec_ref()[start..end].copy_from_bitslice(swapped.as_bitslice());
        Ok(())
    }

    pub(crate) fn apply_invert_positions(
        &mut self,
        pos: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        match pos {
            None => {
                *self.as_mut_bitvec_ref() = std::mem::take(&mut *self.as_mut_bitvec_ref()).not();
            }
            Some(p) => {
                if let Ok(pos) = p.extract::<isize>() {
                    let pos: usize = validate_index(pos, self.len())?;
                    let value = self.as_bitvec_ref()[pos];
                    self.as_mut_bitvec_ref().set(pos, !value);
                } else {
                    if p.try_iter().is_err() {
                        return Err(PyTypeError::new_err(
                            "invert() argument must be an integer, an iterable of ints, or None",
                        ));
                    }
                    let pos_list = Self::collect_position_indices(p)?;
                    for pos in pos_list {
                        let pos: usize = validate_index(pos, self.len())?;
                        let value = self.as_bitvec_ref()[pos];
                        self.as_mut_bitvec_ref().set(pos, !value);
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn apply_replace_bits(
        &mut self,
        py: Python<'_>,
        old: Tibs,
        new: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        count: Option<i64>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<usize> {
        if old.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to replace."));
        }
        let mask = prepare_mask(mask, old.len())?;

        let (search_start, search_end) = validate_slice(self.len(), start, end)?;
        let search_old = old;
        let replace_new = new;
        let mut countdown = count.unwrap_or(i64::MAX);
        if countdown < 0 {
            return Err(PyValueError::new_err(format!(
                "The count in replace() should not be negative. Received {}.",
                countdown
            )));
        }

        let alignment_mod8 = if byte_aligned { Some(0) } else { None };
        let matcher = mask
            .as_ref()
            .map(|mask| MaskedMatcher::new(search_old.as_bitslice(), mask.as_bitslice(), false));

        let mut starting_points: Vec<usize> = Vec::new();
        let mut current_pos = search_start;
        while current_pos < search_end && countdown > 0 {
            let found = match &matcher {
                Some(matcher) => matcher.find(
                    py,
                    self.as_bitvec_ref(),
                    current_pos,
                    search_end,
                    alignment_mod8,
                )?,
                None => find_bitvec(
                    py,
                    self.as_bitvec_ref(),
                    search_old.as_bitslice(),
                    current_pos,
                    search_end,
                    byte_aligned,
                )?,
            };
            if let Some(found_pos) = found {
                starting_points.push(found_pos);
                current_pos = found_pos + search_old.len();
                countdown -= 1;
            } else {
                break;
            }
        }

        if starting_points.is_empty() {
            return Ok(0);
        }

        let replacements = starting_points.len();
        starting_points.sort_unstable();
        let mut result = BV::new();
        let mut last_pos = 0;
        for &pos in &starting_points {
            result.extend_from_bitslice(&self.as_bitvec_ref()[last_pos..pos]);
            result.extend_from_bitslice(replace_new.as_bitslice());
            last_pos = pos + search_old.len();
        }
        result.extend_from_bitslice(&self.as_bitvec_ref()[last_pos..]);

        *self.as_mut_bitvec_ref() = result;
        Ok(replacements)
    }

    pub(crate) fn apply_deposit(&mut self, value: &Tibs, mask: &Tibs) -> PyResult<()> {
        validate_logical_op_lengths(self.len(), mask.len())?;
        let set_bits = mask.as_bitslice().count_ones();
        if value.len() != set_bits {
            return Err(PyValueError::new_err(format!(
                "The value to deposit is {} bits long, but the mask selects {set_bits} bits.",
                value.len()
            )));
        }
        deposit_masked(
            self.as_mut_bitvec_ref(),
            value.as_bitslice(),
            mask.as_bitslice(),
        );
        Ok(())
    }

    pub(crate) fn apply_insert_bits(&mut self, mut pos: isize, bs: &Tibs) -> PyResult<()> {
        if bs.is_empty() {
            return Ok(());
        }
        if pos < 0 {
            pos += self.len() as isize;
        }
        if pos < 0 {
            pos = 0;
        } else if pos > self.len() as isize {
            pos = self.len() as isize;
        }
        let insert_pos = pos as usize;
        let (bytes, offset) = bs.raw_data_with_offset();
        self.splice_raw(insert_pos, insert_pos, bytes, offset, bs.len());
        Ok(())
    }

    pub(crate) fn find_impl(
        &self,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
        reverse: bool,
    ) -> PyResult<Option<usize>> {
        if needle.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to find."));
        }
        let mask = prepare_mask(mask, needle.len())?;
        let (start, end) = validate_slice(self.len(), start, end)?;
        let alignment_mod8 = if byte_aligned { Some(0) } else { None };

        let found = match (&mask, reverse) {
            (Some(mask), _) => helpers::find_bitvec_masked_aligned(
                py,
                self.as_bitvec_ref(),
                needle.as_bitslice(),
                mask.as_bitslice(),
                start,
                end,
                alignment_mod8,
                reverse,
            )?,
            (None, false) => find_bitvec_aligned(
                py,
                self.as_bitvec_ref(),
                needle.as_bitslice(),
                start,
                end,
                alignment_mod8,
            )?,
            (None, true) => helpers::rfind_bitvec_aligned(
                py,
                self.as_bitvec_ref(),
                needle.as_bitslice(),
                start,
                end,
                alignment_mod8,
            )?,
        };
        Ok(found)
    }

    pub(crate) fn set_from_slice(
        &mut self,
        value: bool,
        start: isize,
        stop: isize,
        step: isize,
    ) -> PyResult<()> {
        if step == 0 {
            return Err(PyValueError::new_err("Step cannot be zero."));
        }
        // The arguments describe a Python range whose elements are the bit
        // positions to change, exactly as if the range had been passed as a
        // list: negative values index from the end, and empty ranges are
        // no-ops.
        if (step > 0 && start >= stop) || (step < 0 && start <= stop) {
            return Ok(());
        }
        let len = self.len();
        validate_index(start, len)?;
        // Every element lies between the first and the last, so validating
        // those two covers the whole range. Use i128 so extreme range
        // endpoints cannot overflow before they are rejected.
        let count = if step > 0 {
            (stop as i128 - start as i128 - 1) / step as i128 + 1
        } else {
            (start as i128 - stop as i128 - 1) / (-step) as i128 + 1
        };
        let last = start as i128 + step as i128 * (count - 1);
        if last < -(len as i128) || last >= len as i128 {
            return Err(PyIndexError::new_err(format!(
                "Index of {last} is out of range for length of {len}"
            )));
        }
        let last = last as isize;
        let count = count as usize;
        let len_isize = len as isize;
        let bv = self.as_mut_bitvec_ref();

        // Contiguous fast paths: the values form one interval, which wraps
        // to at most two index regions.
        if step == 1 || step == -1 {
            let (lo, hi) = if step == 1 {
                (start, last)
            } else {
                (last, start)
            };
            if lo >= 0 {
                bv[lo as usize..(hi + 1) as usize].fill(value);
            } else if hi < 0 {
                bv[(lo + len_isize) as usize..(hi + len_isize + 1) as usize].fill(value);
            } else {
                bv[(lo + len_isize) as usize..len].fill(value);
                bv[0..(hi + 1) as usize].fill(value);
            }
            return Ok(());
        }

        // General strided path
        let mut element = start;
        for _ in 0..count {
            let index = if element < 0 {
                element + len_isize
            } else {
                element
            } as usize;
            debug_assert!(index < len);
            unsafe { bv.set_unchecked(index, value) };
            element += step;
        }
        Ok(())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Mutibs {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(tibs_ref) = obj.extract::<PyRef<Tibs>>() {
            return Ok(tibs_ref.to_mutibs());
        }
        if let Ok(mutibs_ref) = obj.extract::<PyRef<Mutibs>>() {
            return Ok(mutibs_ref.clone());
        }
        let bv = promote_to_bv(&obj)?;
        Ok(Mutibs::from_bv(bv))
    }
}

#[pymethods]
impl Mutibs {
    #[new]
    #[pyo3(signature = (auto = None), text_signature = "(auto=None)")]
    pub fn py_new(auto: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(auto) = auto else {
            return Ok(BitCollection::empty());
        };
        Mutibs::extract(auto.as_borrowed())
    }

    /// Return True if two Mutibs have the same binary representation.
    ///
    /// Equality is only defined against :class:`Tibs` and :class:`Mutibs`.
    ///
    /// >>> Mutibs('0xf2') == Tibs('0b11110010')
    /// True
    ///
    pub fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other) = other.extract::<PyRef<'_, Tibs>>() {
            return Ok(self.as_bitslice() == other.as_bitslice());
        }
        if let Ok(other) = other.extract::<PyRef<'_, Mutibs>>() {
            return Ok(self.as_bitslice() == other.as_bitslice());
        }
        Ok(false)
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    /// Return string representations for printing.
    pub fn __str__(&self) -> String {
        self.to_string()
    }

    /// Return representation that could be used to recreate the instance.
    pub fn __repr__(&self) -> String {
        if self.is_empty() {
            "Mutibs()".to_string()
        } else {
            format!("Mutibs('{}')", self.__str__())
        }
    }

    /// Return a string formatted according to the Python format mini-language.
    ///
    /// The type codes ``b``, ``o``, ``x`` and ``X`` give the bit representation, and so
    /// keep any leading zeros. They are equivalent to the :attr:`~Mutibs.bin`,
    /// :attr:`~Mutibs.oct` and :attr:`~Mutibs.hex` properties. The type codes ``u`` and
    /// ``i`` give the unsigned and signed integer interpretations instead.
    ///
    /// The ``#`` flag adds a ``0b``, ``0o``, ``0x`` or ``0X`` prefix. The ``_`` option
    /// groups the digits, with the group size taken from the otherwise unused precision
    /// field and defaulting to 4. Groups are counted from bit zero, so a short group
    /// comes last. Fill, alignment and width work as they do elsewhere in Python.
    ///
    /// An empty format spec gives the same string as :func:`str`.
    ///
    /// :param str format_spec: The format specification.
    /// :return: The formatted string.
    ///
    /// :raises ValueError: if the spec cannot be parsed, if a type code is used that needs a length that is a different multiple, or if a sign or comma grouping is used with a bit representation type code.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> f"{Mutibs('0xac804f4b'):#_.2x}"
    ///     '0xac_80_4f_4b'
    ///     >>> f"{Mutibs('0x0f'):b}"
    ///     '00001111'
    ///
    #[pyo3(signature = (format_spec, /), text_signature = "($self, format_spec, /)")]
    pub fn __format__(&self, py: Python<'_>, format_spec: &str) -> PyResult<String> {
        helpers::format_bit_collection(py, self, format_spec, "Mutibs")
    }

    /// Return a mutable view with interpretation settings.
    ///
    /// A mutable view keeps a live reference to the source ``Mutibs``. Later
    /// changes to the ``Mutibs`` are reflected in the view, and assignment through
    /// the view mutates the source.
    ///
    /// Byte-oriented views must have a whole-byte length. This applies when using
    /// little-endian or big-endian byte order, or when using ``BitOrder.Lsb0``.
    ///
    /// :param ByteOrder byte_order: The byte order used when interpreting whole-byte values. Defaults to ``ByteOrder.Unspecified``.
    /// :param BitOrder bit_order: The bit numbering order used for field labels. Defaults to ``BitOrder.Msb0``.
    /// :return: A new :class:`MutableView`.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs('0x0100')
    ///     >>> v = m.le
    ///     >>> v.write_u(2)
    ///     >>> m
    ///     Mutibs('0x0002')
    ///
    #[pyo3(signature = (byte_order = ByteOrder::Unspecified, bit_order = BitOrder::Msb0), text_signature = "($self, byte_order=None, bit_order=None)")]
    pub fn view(
        slf: PyRef<'_, Self>,
        byte_order: Option<ByteOrder>,
        bit_order: Option<BitOrder>,
    ) -> PyResult<MutableView> {
        let byte_order = byte_order.unwrap_or(ByteOrder::Unspecified);
        let bit_order = bit_order.unwrap_or(BitOrder::Msb0);
        View::validate_layout(slf.len(), byte_order, bit_order)?;
        Ok(MutableView::from_mutibs(slf.into(), byte_order, bit_order))
    }

    /// Return a little-endian byte-order view.
    ///
    /// Equivalent to ``view(byte_order=ByteOrder.Little)``.
    ///
    /// The ``Mutibs`` length must be a whole number of bytes.
    ///
    #[getter]
    pub fn le(slf: PyRef<'_, Self>) -> PyResult<MutableView> {
        View::validate_layout(slf.len(), ByteOrder::Little, BitOrder::Msb0)?;
        Ok(MutableView::from_mutibs(
            slf.into(),
            ByteOrder::Little,
            BitOrder::Msb0,
        ))
    }

    /// Return a big-endian byte-order view.
    ///
    /// Equivalent to ``view(byte_order=ByteOrder.Big)``.
    ///
    /// The ``Mutibs`` length must be a whole number of bytes.
    ///
    #[getter]
    pub fn be(slf: PyRef<'_, Self>) -> PyResult<MutableView> {
        View::validate_layout(slf.len(), ByteOrder::Big, BitOrder::Msb0)?;
        Ok(MutableView::from_mutibs(
            slf.into(),
            ByteOrder::Big,
            BitOrder::Msb0,
        ))
    }

    /// Return an LSB0 bit-order view.
    ///
    /// ``BitOrder.Lsb0`` means that field labels are counted from the least
    /// significant bit of each byte. The ``Mutibs`` length must be a whole number
    /// of bytes.
    ///
    /// Equivalent to ``view(bit_order=BitOrder.Lsb0)``.
    ///
    #[getter]
    pub fn lsb0(slf: PyRef<'_, Self>) -> PyResult<MutableView> {
        View::validate_layout(slf.len(), ByteOrder::Unspecified, BitOrder::Lsb0)?;
        Ok(MutableView::from_mutibs(
            slf.into(),
            ByteOrder::Unspecified,
            BitOrder::Lsb0,
        ))
    }

    /// Return an MSB0 bit-order view.
    ///
    /// ``BitOrder.Msb0`` means that field labels are counted from the most
    /// significant bit of each byte. This is the default bit order.
    ///
    /// Equivalent to ``view(bit_order=BitOrder.Msb0)``.
    ///
    #[getter]
    pub fn msb0(slf: PyRef<'_, Self>) -> PyResult<MutableView> {
        Ok(MutableView::from_mutibs(
            slf.into(),
            ByteOrder::Unspecified,
            BitOrder::Msb0,
        ))
    }

    /// Extract a mutable field using inclusive MSB0 bit labels.
    ///
    /// ``a`` and ``b`` must be zero or positive bit labels. The two endpoints
    /// are inclusive and may be provided in either order. This is equivalent to
    /// ``self.msb0.field(a, b)``.
    ///
    /// :param int a: One non-negative inclusive field endpoint.
    /// :param int b: The other non-negative inclusive field endpoint.
    /// :return: A new :class:`MutableView`.
    ///
    #[pyo3(signature = (a, b), text_signature = "($self, a, b)")]
    pub fn field(slf: PyRef<'_, Self>, a: i64, b: i64) -> PyResult<MutableView> {
        let py = slf.py();
        MutableView::from_mutibs(slf.into(), ByteOrder::Unspecified, BitOrder::Msb0).field(py, a, b)
    }

    /// Create a new instance from a formatted string.
    ///
    /// This method initializes a new instance of :class:`Mutibs` using a formatted string.
    ///
    /// :param str s: The formatted string to convert. This can begin with '0b', '0o' or '0x' to indicate binary, octal or hexadecimal, and commas can be used to separate items.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_string("0xff01")
    ///     b = Mutibs.from_string("0b1")
    ///
    /// The ``__init__`` method for ``Mutibs`` can also redirect to ``from_string``:
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs("0xff01")
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)"
    )]
    pub fn from_string(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        let bv = str_to_bv(s)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Create a new instance from a binary string.
    ///
    /// :param str s: A string of ``0`` and ``1`` s, optionally preceded with ``0b`` and optionally containing underscores.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_bin("0000_1111_0101")
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)"
    )]
    pub fn from_bin(_cls: &Bound<'_, PyType>, s: String) -> PyResult<Self> {
        let bv = bv_from_bin(&s)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the binary representation of the Mutibs as a string.
    ///
    /// Equivalent to using the ``bin`` property when called with no parameters.
    ///
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    ///
    /// :return: The binary representation.
    #[pyo3(signature = (start = None, end = None), text_signature = "($self, start=None, end=None)")]
    pub fn to_bin(&self, start: Option<isize>, end: Option<isize>) -> PyResult<String> {
        self.map_slice(start, end, |bits| Ok(BitCollection::to_binary(bits)))
    }

    /// Replace the current bits from a binary string.
    ///
    /// This can change the length of the ``Mutibs``.
    ///
    /// :param str s: A string of ``0`` and ``1`` s, optionally preceded with ``0b`` and optionally containing underscores.
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs()
    ///     >>> m.write_bin('101')
    ///     >>> m
    ///     Mutibs('0b101')
    ///
    #[pyo3(signature = (s, /), text_signature = "($self, s, /)")]
    pub fn write_bin(&mut self, s: &str) -> PyResult<()> {
        let bv = bv_from_bin(s)?;
        self.replace_with_bv(bv);
        Ok(())
    }

    /// Property of the binary representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_bin` with no parameters.
    /// Assigning is equivalent to using :meth:`~write_bin` and can change the length.
    ///
    /// :return: The binary representation.
    #[getter]
    fn bin(&self) -> String {
        BitCollection::to_binary(self)
    }

    #[setter(bin)]
    fn write_bin_property(&mut self, s: &str) -> PyResult<()> {
        self.write_bin(s)
    }

    /// Create a new instance from an octal string.
    ///
    /// :param str s: A string of octal digits, optionally preceded with ``0o`` and optionally containing underscores.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.from_oct("17")
    ///     Mutibs('0b001111')
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)"
    )]
    pub fn from_oct(_cls: &Bound<'_, PyType>, s: String) -> PyResult<Self> {
        let bv = bv_from_oct(&s)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the octal representation of the Mutibs as a string.
    ///
    /// Equivalent to using the ``oct`` property when called with no parameters.
    ///
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    ///
    /// :return: The octal representation.
    /// :raises ValueError: if the length is not a multiple of 3.
    #[pyo3(signature = (start = None, end = None), text_signature = "($self, start=None, end=None)")]
    pub fn to_oct(&self, start: Option<isize>, end: Option<isize>) -> PyResult<String> {
        self.map_slice(start, end, BitCollection::to_octal)
    }

    /// Replace the current bits from an octal string.
    ///
    /// This can change the length of the ``Mutibs``.
    ///
    /// :param str s: A string of octal digits, optionally preceded with ``0o`` and optionally containing underscores.
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs()
    ///     >>> m.write_oct('17')
    ///     >>> m
    ///     Mutibs('0b001111')
    ///
    #[pyo3(signature = (s, /), text_signature = "($self, s, /)")]
    pub fn write_oct(&mut self, s: &str) -> PyResult<()> {
        let bv = bv_from_oct(s)?;
        self.replace_with_bv(bv);
        Ok(())
    }

    /// Property of the octal representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_oct` with no parameters.
    /// Assigning is equivalent to using :meth:`~write_oct` and can change the length.
    ///
    /// :return: The octal representation.
    /// :raises ValueError: if the length is not a multiple of 3.
    #[getter]
    fn oct(&self) -> PyResult<String> {
        BitCollection::to_octal(self)
    }

    #[setter(oct)]
    fn write_oct_property(&mut self, s: &str) -> PyResult<()> {
        self.write_oct(s)
    }

    /// Create a new instance from a hexadecimal string.
    ///
    /// Equivalent to using the ``hex`` property.
    ///
    /// :param str s: A string of hexadecimal digits, optionally preceded with ``0x`` and optionally containing underscores.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.from_hex("0f")
    ///     Mutibs('0x0f')
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)"
    )]
    pub fn from_hex(_cls: &Bound<'_, PyType>, s: String) -> PyResult<Self> {
        let bv = bv_from_hex(&s)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the hexadecimal representation of the Mutibs as a string.
    ///
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    ///
    /// :return: The hexadecimal representation.
    /// :raises ValueError: if the length is not a multiple of 4.
    #[pyo3(signature = (start = None, end = None), text_signature = "($self, start=None, end=None)")]
    pub fn to_hex(&self, start: Option<isize>, end: Option<isize>) -> PyResult<String> {
        self.map_slice(start, end, BitCollection::to_hexadecimal)
    }

    /// Replace the current bits from a hexadecimal string.
    ///
    /// This can change the length of the ``Mutibs``.
    ///
    /// :param str s: A string of hexadecimal digits, optionally preceded with ``0x`` and optionally containing underscores.
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs()
    ///     >>> m.write_hex('0f')
    ///     >>> m
    ///     Mutibs('0x0f')
    ///
    #[pyo3(signature = (s, /), text_signature = "($self, s, /)")]
    pub fn write_hex(&mut self, s: &str) -> PyResult<()> {
        let bv = bv_from_hex(s)?;
        self.replace_with_bv(bv);
        Ok(())
    }

    /// Property of the hexadecimal representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_hex` with no parameters.
    /// Assigning is equivalent to using :meth:`~write_hex` and can change the length.
    ///
    /// :return: The hexadecimal representation.
    /// :raises ValueError: if the length is not a multiple of 4.
    #[getter]
    fn hex(&self) -> PyResult<String> {
        BitCollection::to_hexadecimal(self)
    }

    #[setter(hex)]
    fn write_hex_property(&mut self, s: &str) -> PyResult<()> {
        self.write_hex(s)
    }

    /// Return the Mutibs as a bytes object.
    ///
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    ///
    /// :return: The bytes representation.
    /// :raises ValueError: if the length is not a multiple of 8.
    #[pyo3(signature = (start = None, end = None), text_signature = "($self, start=None, end=None)")]
    pub fn to_bytes(
        &self,
        py: Python<'_>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<PyBytes>> {
        self.map_slice(start, end, |bits| BitCollection::to_py_bytes(bits, py))
    }

    /// Return the Mutibs as a bytes object, padding the right-hand side with zero bits.
    ///
    /// This appends 0 to 7 zero bits to the end of the selected bit sequence so
    /// the returned value has a whole number of bytes. If the selected length is
    /// already a multiple of 8, this is equivalent to :meth:`~to_bytes`.
    ///
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    ///
    /// :return: The padded bytes representation.
    #[pyo3(signature = (start = None, end = None), text_signature = "($self, start=None, end=None)")]
    pub fn to_padded_bytes(
        &self,
        py: Python<'_>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<PyBytes>> {
        self.map_slice(start, end, |bits| {
            BitCollection::to_padded_py_bytes(bits, py)
        })
    }

    /// Replace the current bits from a bytes-like object.
    ///
    /// This can change the length of the ``Mutibs``.
    ///
    /// :param bytes data: A bytes-like object.
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs()
    ///     >>> m.write_bytes(b'A')
    ///     >>> m
    ///     Mutibs('0x41')
    ///
    #[pyo3(signature = (data, /), text_signature = "($self, data, /)")]
    pub fn write_bytes(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let bv = bv_from_bytes_slice(bytes_like_to_vec(data)?, None, None)?;
        self.replace_with_bv(bv);
        Ok(())
    }

    /// Property of the ``bytes`` representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_bytes` with no parameters.
    /// Assigning is equivalent to using :meth:`~write_bytes` and can change the length.
    ///
    /// :return: The bytes representation.
    /// :raises ValueError: if the length is not a multiple of 8.
    #[getter]
    fn bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        BitCollection::to_py_bytes(self, py)
    }

    #[setter(bytes)]
    fn write_bytes_property(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        self.write_bytes(data)
    }

    /// Return a copy of the raw byte information.
    ///
    /// This returns the underlying byte data and can contain leading and trailing
    /// bits that are not considered part of the object's value. Usually using
    /// :meth:`~to_bytes` is what you really need.
    ///
    /// See also :meth:`~as_raw_data` which moves the byte data instead of copying it.
    ///
    /// :return: A tuple of the raw bytes, the bit offset and the bit length.
    ///
    /// .. code-block:: python
    ///
    ///     raw_bytes, offset, length = t.to_raw_data()
    ///     assert t == Mutibs.from_bytes(raw_bytes, offset=offset, length=length)
    ///
    pub fn to_raw_data(&self) -> (Vec<u8>, usize, usize) {
        self.raw_data()
    }

    /// Return the raw bytes and offset information, leaving the Mutibs empty.
    ///
    /// This returns the underlying byte data using a move rather than a copy, and can contain
    /// leading and trailing bits that are not considered part of the object's value. Usually using
    /// :meth:`~to_bytes` is what you really need.
    ///
    /// See also :meth:`~to_raw_data` which copies the byte data instead of moving it.
    ///
    /// :return: A tuple of the raw bytes, the bit offset and the bit length.
    ///
    /// .. code-block:: python
    ///
    ///     raw_bytes, offset, length = t.as_raw_data()
    ///     assert t == []
    ///
    pub fn as_raw_data(&mut self) -> (Vec<u8>, usize, usize) {
        let slice = self.as_bitvec_ref().as_bitslice();
        let offset = match slice.domain() {
            bitvec::domain::Domain::Enclave(elem) => elem.head().into_inner() as usize,
            bitvec::domain::Domain::Region {
                head: Some(elem), ..
            } => elem.head().into_inner() as usize,
            _ => 0,
        };
        let len = self.len();
        let bv = std::mem::take(&mut *self.as_mut_bitvec_ref());
        let raw_bytes = bv.into_vec();
        (raw_bytes, offset, len)
    }

    /// Create a new instance from an unsigned integer.
    ///
    /// :param int u: An unsigned integer.
    /// :param int length: The bit length to create. Can be any positive number of bits.
    /// :param ByteOrder byte_order: The byte order used to store the integer. Defaults to ByteOrder.Unspecified.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// :raises ValueError: if the integer doesn't fit in the length given.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.from_u(15, length=8)
    ///     Mutibs('0x0f')
    ///
    #[classmethod]
    #[pyo3(signature = (u, /, length, byte_order = ByteOrder::Unspecified), text_signature = "(cls, u, /, length, byte_order=None)")]
    pub fn from_u(
        _cls: &Bound<'_, PyType>,
        u: &Bound<'_, PyAny>,
        length: i64,
        byte_order: Option<ByteOrder>,
    ) -> PyResult<Self> {
        let length = validate_length(length)?;
        let is_little_endian = ByteOrder::is_little_endian(byte_order, length)?;
        let bv = bv_from_uint(u, length, is_little_endian)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the unsigned integer representation of the Mutibs.
    ///
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    ///
    /// :return: The value as an unsigned integer.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0x0f').to_u()
    ///     15
    ///
    #[pyo3(signature = (start = None, end = None), text_signature = "($self, start=None, end=None)")]
    pub fn to_u<'py>(
        &self,
        py: Python<'py>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.map_slice(start, end, |bits| BitCollection::to_uint(bits, py, false))
    }

    /// Write the current bits from an unsigned integer without changing the length.
    ///
    /// :param int u: An unsigned integer.
    /// :return: None
    ///
    /// :raises ValueError: if the current length is zero.
    /// :raises OverflowError: if the integer doesn't fit in the current length.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs.from_zeros(8)
    ///     >>> m.write_u(15)
    ///     >>> m
    ///     Mutibs('0x0f')
    ///
    #[pyo3(signature = (u, /), text_signature = "($self, u, /)")]
    pub fn write_u(&mut self, u: &Bound<'_, PyAny>) -> PyResult<()> {
        self.assign_u(u)
    }

    /// Property of the unsigned integer representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_u` with no parameters. Assigning
    /// is equivalent to using :meth:`~write_u`.
    ///
    /// :return: The value as an unsigned integer.
    #[getter]
    fn u<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.to_u(py, None, None)
    }

    #[setter(u)]
    fn write_u_property(&mut self, u: &Bound<'_, PyAny>) -> PyResult<()> {
        self.assign_u(u)
    }

    /// Create a new instance from a signed integer.
    ///
    /// :param int i: A signed integer.
    /// :param int length: The bit length to create. Can be any positive number of bits.
    /// :param ByteOrder byte_order: The byte order used to store the integer. Defaults to ByteOrder.Unspecified.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// :raises ValueError: if the integer doesn't fit in the length given.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.from_i(-2, length=4)
    ///     Mutibs('0xe')
    ///
    #[classmethod]
    #[pyo3(signature = (i, /, length, byte_order = ByteOrder::Unspecified), text_signature = "(cls, i, /, length, byte_order=None)")]
    pub fn from_i(
        _cls: &Bound<'_, PyType>,
        i: &Bound<'_, PyAny>,
        length: i64,
        byte_order: Option<ByteOrder>,
    ) -> PyResult<Self> {
        let length = validate_length(length)?;
        let is_little_endian = ByteOrder::is_little_endian(byte_order, length)?;
        let bv = bv_from_int(i, length, is_little_endian)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the signed integer representation of the Mutibs.
    ///
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    ///
    /// :return: The value as a signed integer.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0xe').to_i()
    ///     -2
    ///
    #[pyo3(signature = (start = None, end = None), text_signature = "($self, start=None, end=None)")]
    pub fn to_i<'py>(
        &self,
        py: Python<'py>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.map_slice(start, end, |bits| BitCollection::to_int(bits, py, false))
    }

    /// Write the current bits from a signed integer without changing the length.
    ///
    /// :param int i: A signed integer.
    /// :return: None
    ///
    /// :raises ValueError: if the current length is zero.
    /// :raises OverflowError: if the integer doesn't fit in the current length.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs.from_zeros(4)
    ///     >>> m.write_i(-2)
    ///     >>> m
    ///     Mutibs('0xe')
    ///
    #[pyo3(signature = (i, /), text_signature = "($self, i, /)")]
    pub fn write_i(&mut self, i: &Bound<'_, PyAny>) -> PyResult<()> {
        self.assign_i(i)
    }

    /// Property of the signed integer representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_i` with no parameters. Assigning
    /// is equivalent to using :meth:`~write_i`.
    ///
    /// :return: The value as a signed integer.
    #[getter]
    fn i<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.to_i(py, None, None)
    }

    #[setter(i)]
    fn write_i_property(&mut self, i: &Bound<'_, PyAny>) -> PyResult<()> {
        self.assign_i(i)
    }

    /// Create a new instance from a floating point number.
    ///
    /// :param float f: A floating point value.
    /// :param int length: The bit length to create. Must be 16, 32 or 64.
    /// :param ByteOrder byte_order: The byte order used to store the float. Defaults to ByteOrder.Unspecified.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.from_f(1.5, length=32)
    ///     Mutibs('0x3fc00000')
    ///
    #[classmethod]
    #[pyo3(signature = (f, /, length, byte_order = ByteOrder::Unspecified), text_signature = "(cls, f, /, length, byte_order=None)")]
    pub fn from_f(
        _cls: &Bound<'_, PyType>,
        f: f64,
        length: i64,
        byte_order: Option<ByteOrder>,
    ) -> PyResult<Self> {
        let length = validate_length(length)?;
        let is_little_endian = ByteOrder::is_little_endian(byte_order, length)?;
        let bv = bv_from_f64(f, length, is_little_endian)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the floating point representation of the Mutibs.
    ///
    /// The length must be 16, 32 or 64.
    ///
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    ///
    /// :return: The value as a Python float.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0x3fc00000').to_f()
    ///     1.5
    ///
    #[pyo3(signature = (start = None, end = None), text_signature = "($self, start=None, end=None)")]
    pub fn to_f(&self, start: Option<isize>, end: Option<isize>) -> PyResult<f64> {
        self.map_slice(start, end, |bits| BitCollection::to_f64(bits, false))
    }

    /// Write the current bits from a floating point number without changing the length.
    ///
    /// The current length must be 16, 32 or 64 bits.
    ///
    /// :param float f: A floating point value.
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs.from_zeros(32)
    ///     >>> m.write_f(1.5)
    ///     >>> m
    ///     Mutibs('0x3fc00000')
    ///
    #[pyo3(signature = (f, /), text_signature = "($self, f, /)")]
    pub fn write_f(&mut self, f: f64) -> PyResult<()> {
        self.assign_f(f)
    }

    /// Property of the floating point representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_f` with no parameters. Assigning
    /// is equivalent to using :meth:`~write_f`.
    ///
    /// :return: The value as a Python float.
    #[getter]
    fn f(&self) -> PyResult<f64> {
        self.to_f(None, None)
    }

    #[setter(f)]
    fn write_f_property(&mut self, f: f64) -> PyResult<()> {
        self.assign_f(f)
    }

    /// Create a new instance with all bits set to zero.
    ///
    /// :param int length: The number of bits to set.
    /// :return: A Mutibs object with all bits set to zero.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_zeros(500)  # 500 zero bits
    ///
    #[classmethod]
    #[pyo3(signature = (length, /), text_signature = "(cls, length, /)")]
    pub fn from_zeros(_cls: &Bound<'_, PyType>, length: i64) -> PyResult<Self> {
        let length = validate_length(length)?;
        Ok(Self::from_bv(bv_from_zeros(length)))
    }

    /// Create a new instance with all bits set to one.
    ///
    /// :param int length: The number of bits to set.
    /// :return: A Mutibs object with all bits set to one.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.from_ones(5)
    ///     Mutibs('0b11111')
    ///
    #[classmethod]
    #[pyo3(signature = (length, /), text_signature = "(cls, length, /)")]
    pub fn from_ones(_cls: &Bound<'_, PyType>, length: i64) -> PyResult<Self> {
        let length = validate_length(length)?;
        Ok(Mutibs::from_bv(bv_from_ones(length)))
    }

    /// Create a new instance from an iterable by converting each element to a bool.
    ///
    /// :param Iterable iterable: The iterable to convert to a :class:`Mutibs`.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_bools([False, 0, 1, "Steven"])  # binary 0011
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /), text_signature = "(cls, iterable, /)")]
    pub fn from_bools(_cls: &Bound<'_, PyType>, iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bv = bv_from_bools(iterable)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Return the bits as a list of bools.
    ///
    /// This is much faster than using ``list()`` on the Mutibs, which iterates bit by bit.
    ///
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    /// :return: A list of bools.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b101').to_bools()
    ///     [True, False, True]
    ///
    #[pyo3(signature = (start = None, end = None), text_signature = "($self, start=None, end=None)")]
    pub fn to_bools(
        &self,
        py: Python<'_>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<PyList>> {
        let (start, end) = validate_slice(self.len(), start, end)?;
        helpers::bitslice_to_bool_list(py, &self.as_bitslice()[start..end])
    }

    /// Create a new instance with all bits randomly set.
    ///
    /// :param int length: The number of bits to set. Must be non-negative.
    /// :param bool secure: If ``True``, use the OS's cryptographically secure generator. Default is ``False``.
    /// :param bytes | bytearray | None seed: A bytes or bytearray to use as an optional seed, only if ``secure`` is ``False``.
    /// :return: A newly constructed ``Mutibs`` with random data.
    ///
    /// The 'secure' option uses the OS's random data source, so will be slower and could potentially
    /// fail.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_random(1000000)  # A million random bits
    ///     b = Mutibs.from_random(100, seed=b'a_seed')
    ///
    #[classmethod]
    #[pyo3(signature = (length, /, secure=false, seed=None), text_signature="(cls, length, /, secure=False, seed=None)")]
    pub fn from_random(
        _cls: &Bound<'_, PyType>,
        length: i64,
        secure: bool,
        seed: Option<Vec<u8>>,
    ) -> PyResult<Self> {
        let bv = bv_from_random(length, secure, &seed)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Create a new instance from a bytes object.
    ///
    /// :param bytes | bytearray | memoryview data: The bytes, bytearray or memoryview object to convert to a :class:`Mutibs`.
    /// :param int | None offset: The bit offset from the start. Defaults to zero.
    /// :param int | None length: The bit length to use. Defaults to the whole of the data.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_bytes(b"some_bytes_maybe_from_a_file")
    ///
    #[classmethod]
    #[inline]
    #[pyo3(signature = (data, /, offset=None, length=None), text_signature = "(cls, data, /, offset=None, length=None)")]
    pub fn from_bytes(
        _cls: &Bound<'_, PyType>,
        data: &Bound<'_, PyAny>,
        offset: Option<i64>,
        length: Option<i64>,
    ) -> PyResult<Self> {
        let length = match length {
            Some(length) => Some(validate_length(length)?),
            None => None,
        };
        let offset = match offset {
            Some(offset) => Some(validate_length(offset)?),
            None => None,
        };
        let bv = bv_from_bytes_slice(bytes_like_to_vec(data)?, offset, length)?;
        Ok(Self::from_bv(bv))
    }

    /// Create a new instance by concatenating a sequence of Mutibs objects.
    ///
    /// This method concatenates a sequence of Mutibs objects into a single Mutibs object.
    ///
    /// :param Iterable iterable: An iterable to concatenate. Items can be anything that can be promoted to a :class:`Mutibs`.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_joined(['0x01', [1, 0], b'some_bytes'])
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /), text_signature = "(cls, iterable, /)")]
    pub fn from_joined(_cls: &Bound<'_, PyType>, iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bv = Self::joined_bv_from_iterable(iterable)?;
        Ok(Mutibs::from_bv(bv))
    }

    /// Create a new instance by encoding one value with a dtype.
    ///
    /// :param Dtype | str dtype: The value encoding to use.
    /// :param object value: The value to encode.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.from_value("u8", 15)
    ///     Mutibs('0x0f')
    ///
    #[classmethod]
    #[pyo3(signature = (dtype, value, /), text_signature = "(cls, dtype, value, /)")]
    pub fn from_value(
        _cls: &Bound<'_, PyType>,
        dtype: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let dtype = extract_dtype(dtype)?;
        Ok(Mutibs::from_bv(bv_from_value(&dtype, value)?))
    }

    /// Create a new instance by encoding and concatenating values with a dtype.
    ///
    /// :param Dtype | str dtype: The value encoding to use for each item.
    /// :param Iterable iterable: The values to encode.
    /// :return: A newly constructed ``Mutibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.from_values("u8", [1, 2, 3])
    ///     Mutibs('0x010203')
    ///
    #[classmethod]
    #[pyo3(signature = (dtype, iterable, /), text_signature = "(cls, dtype, iterable, /)")]
    pub fn from_values(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        iterable: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let dtype = extract_dtype(dtype)?;
        Ok(Mutibs::from_bv(bv_from_values_iter(py, &dtype, iterable)?))
    }

    /// The bit length of the Mutibs.
    pub fn __len__(&self) -> usize {
        self.len()
    }

    /// Whether the Mutibs has any bits.
    pub fn __bool__(&self) -> bool {
        !self.as_bitvec_ref().is_empty()
    }

    /// Return a list of values decoded with a dtype.
    ///
    /// The selected range must be a whole number of dtype values. The values are
    /// decoded from the current contents when the method is called.
    ///
    /// :param Dtype | str dtype: The value encoding to use for each item.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    /// :return: A list of decoded Python values.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0x010203').to_values("u8")
    ///     [1, 2, 3]
    ///
    #[pyo3(signature = (dtype, start = None, end = None), text_signature = "($self, dtype, start=None, end=None)")]
    pub fn to_values(
        &self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let dtype = extract_dtype(dtype)?;
        let snapshot = self.to_tibs();
        py_values_from_range(py, &snapshot, &dtype, start, end)
    }

    /// Return one value decoded with a dtype.
    ///
    /// The selected range must have exactly the dtype length.
    ///
    /// :param Dtype | str dtype: The value encoding to use.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    /// :return: The decoded Python value.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0x0f').to_value("u8")
    ///     15
    ///
    #[pyo3(signature = (dtype, start = None, end = None), text_signature = "($self, dtype, start=None, end=None)")]
    pub fn to_value(
        &self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<PyAny>> {
        let dtype = extract_dtype(dtype)?;
        let snapshot = self.to_tibs();
        let (start, end) = validate_slice(snapshot.len(), start, end)?;
        let value = snapshot.get_slice_unchecked(start, end - start);
        py_from_value(py, &dtype, &value)
    }

    /// Get a bit or a slice of bits.
    ///
    /// :param int | slice key: The index or slice to get.
    /// :return: A bool for a single index, or a new Mutibs for a slice.
    /// :raises IndexError: If the index is out of range.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs('0b101100')
    ///     >>> m[0]
    ///     True
    ///     >>> m[1:4]
    ///     Mutibs('0b011')
    ///
    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = key.py();
        // Fast path for exact int keys via direct ffi, mirroring
        // Tibs::__getitem__ (see there for why the bool singleton is chosen
        // with a conditional move rather than a branch).
        unsafe {
            if ffi::PyLong_Check(key.as_ptr()) != 0 {
                let index = ffi::PyLong_AsSsize_t(key.as_ptr());
                if index == -1 && !ffi::PyErr_Occurred().is_null() {
                    let err = PyErr::fetch(py);
                    return Err(if err.is_instance_of::<PyOverflowError>(py) {
                        PyIndexError::new_err(format!(
                            "Index is out of range for length of {}",
                            self.data.len()
                        ))
                    } else {
                        err
                    });
                }
                let index = validate_index(index, self.data.len())?;
                // SAFETY: validate_index guarantees index < self.data.len().
                let value = *self.data.as_bitslice().get_unchecked(index);
                let obj = std::hint::select_unpredictable(value, ffi::Py_True(), ffi::Py_False());
                ffi::Py_INCREF(obj);
                return Ok(Bound::from_owned_ptr(py, obj).unbind());
            }
        }
        // Handle integer indexing
        if let Ok(index) = key.extract::<isize>() {
            let value: bool = self.get_index(index)?;
            let py_value = PyBool::new(py, value);
            return Ok(py_value.to_owned().into());
        }

        // Handle slice indexing
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(self.len() as isize)?;
            let (start, stop, step) = (indices.start, indices.stop, indices.step);

            let result = if step == 1 {
                if start < stop {
                    self.get_slice(start as usize, (stop - start) as usize)?
                } else {
                    Mutibs::empty()
                }
            } else {
                self.get_slice_with_step(start, stop, step)?
            };
            let py_obj = Py::new(py, result)?.into_pyobject(py)?;
            return Ok(py_obj.into());
        }

        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    /// Set a bit or a slice of bits.
    ///
    /// :param int | slice key: The index or slice to set.
    /// :param object value: For a single index, a boolean value. For a slice, anything that can be converted to Tibs.
    ///
    /// Slice assignment follows standard Python semantics:
    ///
    /// - For a simple slice with a step of 1 (e.g. ``m[2:5]``), the slice is replaced and the length of the Mutibs
    ///   may change.
    /// - For an extended slice (step != 1), the slice length is fixed and the assigned value must have exactly the
    ///   same number of bits as the number of targeted indices.
    ///
    /// :raises ValueError: If the slice step is 0, or if the length of the value doesn't match an extended slice.
    /// :raises IndexError: If the index is out of range.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> b = Mutibs('0b0000')
    ///     >>> b[1] = True
    ///     >>> b.to_bin()
    ///     '0100'
    ///     >>> b[1:3] = '0b11111'
    ///     >>> b.to_bin()
    ///     '0111110'
    ///
    pub fn __setitem__(
        mut slf: PyRefMut<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let length = slf.len();
        if let Ok(index) = key.extract::<isize>() {
            if value.is_truthy()? {
                slf.set_index(index)?;
            } else {
                slf.unset_index(index)?;
            }
            return Ok(());
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            // Need to guard against value being self
            let tibs = if value.as_ptr() == slf.as_ptr() {
                Tibs::from_bv(slf.to_bitvec())
            } else {
                Tibs::extract(value.as_borrowed())?
            };

            let indices = slice.indices(length as isize)?;
            let start = indices.start;
            let stop = indices.stop;
            let step = indices.step;

            if step == 1 {
                debug_assert!(start >= 0);
                debug_assert!(stop >= 0);
                slf.set_slice(start as usize, stop as usize, &tibs);
                return Ok(());
            }
            if step == 0 {
                return Err(PyValueError::new_err(
                    "The step in __setitem__ must not be zero.",
                ));
            }
            // Compute target indices in the natural slice order (respecting step sign).
            let mut positions: Vec<usize> = Vec::new();
            if step > 0 {
                debug_assert!(start >= 0);
                debug_assert!(stop >= 0);
                let mut i = start;
                while i < stop {
                    positions.push(validate_index(i, length)?);
                    i += step;
                }
            } else {
                // TODO: with a negative step I think start or stop could be -1.
                let mut i = start;
                while i > stop {
                    positions.push(validate_index(i, length)?);
                    i += step; // step < 0
                }
            }

            // Enforce equal sizes.
            if tibs.len() != positions.len() {
                return Err(PyValueError::new_err(format!(
                    "Attempt to assign sequence of size {} to extended slice of size {}",
                    tibs.len(),
                    positions.len()
                )));
            }

            // Assign element-wise in logical order.
            for (k, &pos) in positions.iter().enumerate() {
                let v = tibs.get_index(k as isize)?;
                slf.as_mut_bitvec_ref().set(pos, v);
            }

            return Ok(());
        }
        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    /// Delete a bit or a slice of bits.
    ///
    /// :param int | slice key: The index or slice to delete.
    ///
    /// For a single index, one bit is removed.
    /// For a slice, all targeted indices are removed (for an extended slice, indices are removed in a way that
    /// matches Python's behavior).
    ///
    /// :raises IndexError: If the index is out of range.
    /// :raises TypeError: If the key is not an int or slice.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> b = Mutibs('0b1011001')
    ///     >>> del b[1:3]
    ///     >>> b
    ///     Mutibs('0b11001')
    ///
    pub fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let length = self.len();
        if let Ok(mut index) = key.extract::<i64>() {
            if index < 0 {
                index += length as i64;
            }
            if index < 0 || index >= length as i64 {
                return Err(PyIndexError::new_err(format!(
                    "Bit index {index} out of range for length {length}"
                )));
            }
            self.delete_slice(index as usize, index as usize + 1);
            return Ok(());
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(length as isize)?;
            let start: i64 = indices.start.try_into()?;
            let stop: i64 = indices.stop.try_into()?;
            let step: i64 = indices.step.try_into()?;

            if step == 1 {
                if stop > start {
                    self.delete_slice(start as usize, stop as usize);
                }
            } else {
                // Collect indices to remove, then remove from highest to lowest.
                let mut to_remove: Vec<usize> = if step > 0 {
                    let mut v = Vec::new();
                    let mut i = start;
                    while i < stop {
                        v.push(i as usize);
                        i += step;
                    }
                    v
                } else {
                    let mut v = Vec::new();
                    let mut i = start;
                    while i > stop {
                        v.push(i as usize);
                        i += step; // step < 0
                    }
                    v
                };

                to_remove.sort();
                self.delete_positions(&to_remove);
            }
            return Ok(());
        }
        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    /// Return whether the current Mutibs starts with prefix.
    ///
    /// :param object prefix: The bits to search for. This can be anything promotable to ``Tibs``.
    /// :return: True if the Mutibs starts with the prefix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b101100').starts_with('0b101')
    ///     True
    ///     >>> Mutibs('0b101100').starts_with('0b100')
    ///     False
    ///
    pub fn starts_with(&self, prefix: Tibs) -> PyResult<bool> {
        Ok(<Mutibs as BitCollection>::starts_with(self, prefix))
    }

    /// Return True if b is a sub-sequence of self.
    pub fn __contains__(&self, py: Python<'_>, b: Tibs) -> PyResult<bool> {
        self.find(py, b, None, None, false, None)
            .map(|found| found.is_some())
    }

    /// Return whether the current Mutibs ends with suffix.
    ///
    /// :param object suffix: The bits to search for. This can be anything promotable to ``Tibs``.
    /// :return: True if the Mutibs ends with the suffix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b101100').ends_with('0b100')
    ///     True
    ///     >>> Mutibs('0b101100').ends_with('0b101')
    ///     False
    ///
    pub fn ends_with(&self, suffix: Tibs) -> PyResult<bool> {
        Ok(<Mutibs as BitCollection>::ends_with(self, suffix))
    }

    /// Find first occurrence of a bit sequence.
    ///
    /// Returns the bit position if found, or None if not found.
    ///
    /// :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: The bit position if found, or None if not found.
    /// :raises ValueError: if ``needle`` is empty, if the slice parameters are invalid, or if the
    ///     mask length doesn't match the needle length.
    ///
    /// .. code-block:: pycon
    ///
    ///      >>> Mutibs('0xc3e').find('0b1111')
    ///      6
    ///      >>> Mutibs('0x3a5f').find('0x0f', mask='0x0f', byte_aligned=True)
    ///      8
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, needle, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn find(
        &self,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Option<usize>> {
        self.find_impl(py, needle, start, end, byte_aligned, mask, false)
    }

    /// Find all occurrences of a bit sequence.
    ///
    /// :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: A list of bit positions.
    /// :raises ValueError: if ``needle`` is empty, if the slice parameters are invalid, or if the
    ///     mask length doesn't match the needle length.
    ///
    /// All occurrences of needle are found, even if they overlap.
    ///
    /// .. code-block:: pycon
    ///
    ///      >>> Mutibs('0xc3e').find_all('0b1111')
    ///      [6]
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, needle, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn find_all(
        &self,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Vec<u64>> {
        if needle.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to find."));
        }
        let mask = prepare_mask(mask, needle.len())?;

        let haystack_len = self.len();
        let (start, end) = validate_slice(haystack_len, start, end)?;

        match mask {
            Some(mask) => helpers::collect_find_all_positions_masked(
                py,
                self.as_bitslice(),
                needle.as_bitslice(),
                mask.as_bitslice(),
                start,
                end,
                byte_aligned,
            ),
            None => helpers::collect_find_all_positions(
                py,
                self.as_bitslice(),
                needle.as_bitslice(),
                start,
                end,
                byte_aligned,
            ),
        }
    }

    /// Return a list of Mutibs by cutting into chunks.
    ///
    /// :param int chunk_size: The size in bits of the chunks to create.
    /// :param int | None count: If specified, at most count items are created. Default is to cut as many times as possible.
    /// :return: A list of Mutibs chunks.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b110011').chunks(2)
    ///     [Mutibs('0b11'), Mutibs('0b00'), Mutibs('0b11')]
    ///
    #[pyo3(signature = (chunk_size, count = None), text_signature = "($self, chunk_size, count=None)")]
    pub fn chunks(&self, chunk_size: i64, count: Option<i64>) -> PyResult<Vec<Self>> {
        BitCollection::collect_chunks(self, chunk_size, count)
    }

    /// Split at one or more bit positions.
    ///
    /// ``pos`` may be a single integer or an iterable of integers. Negative
    /// positions count from the end. Positions must be in nondecreasing order
    /// after normalization, and each position must be in the range
    /// ``0`` through ``len(self)``, inclusive.
    ///
    /// The returned pieces are new ``Mutibs`` objects, matching normal
    /// ``Mutibs`` slice behavior.
    ///
    /// :param int | Iterable[int] pos: The bit position or positions where the split should occur.
    /// :return: A tuple of ``Mutibs`` pieces.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b101100').split_at(3)
    ///     (Mutibs('0b101'), Mutibs('0b100'))
    ///     >>> Mutibs('0b101100').split_at([2, 5])
    ///     (Mutibs('0b10'), Mutibs('0b110'), Mutibs('0b0'))
    ///
    #[pyo3(signature = (pos, /), text_signature = "($self, pos, /)")]
    pub fn split_at(&self, py: Python<'_>, pos: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
        let pieces = BitCollection::collect_split_at(self, pos)?;
        Ok(PyTuple::new(py, pieces)?.unbind())
    }

    /// Count the bits set in both this Mutibs and another.
    ///
    /// Equivalent to ``(self & other).count(1)``, but without building the
    /// intermediate object.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: The number of positions set in both.
    /// :raises ValueError: if the two lengths differ.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b1100').count_and('0b1010')
    ///     1
    ///
    pub fn count_and(&self, other: Tibs) -> PyResult<usize> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(self.pairwise_count(&other, LogicalOp::And))
    }

    /// Count the bits set in either this Mutibs or another.
    ///
    /// Equivalent to ``(self | other).count(1)``, but without building the
    /// intermediate object.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: The number of positions set in either.
    /// :raises ValueError: if the two lengths differ.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b1100').count_or('0b1010')
    ///     3
    ///
    pub fn count_or(&self, other: Tibs) -> PyResult<usize> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(self.pairwise_count(&other, LogicalOp::Or))
    }

    /// Count the bits that differ between this Mutibs and another.
    ///
    /// This is the Hamming distance. Equivalent to ``(self ^ other).count(1)``,
    /// but without building the intermediate object.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: The number of positions where the two differ.
    /// :raises ValueError: if the two lengths differ.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b1100').count_xor('0b1010')
    ///     2
    ///
    pub fn count_xor(&self, other: Tibs) -> PyResult<usize> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(self.pairwise_count(&other, LogicalOp::Xor))
    }

    /// Count the bits set in this Mutibs but not in another.
    ///
    /// Equivalent to ``self.count(1) - self.count_and(other)``, but in a single pass.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: The number of positions set here but not in the other.
    /// :raises ValueError: if the two lengths differ.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b1100').count_andnot('0b1010')
    ///     1
    ///
    pub fn count_andnot(&self, other: Tibs) -> PyResult<usize> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(self.pairwise_count(&other, LogicalOp::AndNot))
    }

    /// Return whether any bit is set in both this Mutibs and another.
    ///
    /// Equivalent to ``(self & other).any()``, but stops at the first bit set in
    /// both instead of building the intermediate object.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: ``True`` if some position is set in both, otherwise ``False``.
    /// :raises ValueError: if the two lengths differ.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b1100').intersects('0b1010')
    ///     True
    ///     >>> Mutibs('0b1100').intersects('0b0011')
    ///     False
    ///
    pub fn intersects(&self, other: Tibs) -> PyResult<bool> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(self.pairwise_any(&other, LogicalOp::And))
    }

    /// Return whether every bit set in this Mutibs is also set in another.
    ///
    /// Equivalent to ``(self & other) == self``, but stops at the first bit set
    /// here and not there.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: ``True`` if every position set here is set in the other, otherwise ``False``.
    /// :raises ValueError: if the two lengths differ.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b1000').is_subset_of('0b1010')
    ///     True
    ///     >>> Mutibs('0b1100').is_subset_of('0b1010')
    ///     False
    ///
    pub fn is_subset_of(&self, other: Tibs) -> PyResult<bool> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(!self.pairwise_any(&other, LogicalOp::AndNot))
    }

    /// Read the bits at the positions set in a mask, packed together.
    ///
    /// This reads a bit field whose bits are scattered through the Mutibs by the
    /// mask, the way :meth:`field` reads a contiguous one. The result has one
    /// bit for each set bit of the mask, in order.
    ///
    /// :param object mask: The mask selecting which bits to read. This can be anything promotable to ``Tibs``, and must be the same length as ``self``.
    /// :return: A new Mutibs of length ``mask.count()``.
    /// :raises ValueError: if the mask length doesn't match the length of ``self``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b11010110').extract('0b10110000')
    ///     Mutibs('0b101')
    ///
    // Named `extract_field` because `Mutibs::extract` is the FromPyObject
    // promotion method used throughout the crate.
    #[pyo3(name = "extract")]
    pub fn extract_field(&self, mask: Tibs) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), mask.len())?;
        Ok(Self::from_bv(self.extract_masked(&mask)))
    }

    /// Write a scattered bit field into the Mutibs in place.
    ///
    /// This is the inverse of :meth:`extract`, and writes a scattered field the
    /// way slice assignment writes a contiguous one: the bits of ``value`` are
    /// written into the positions set in ``mask``, and the other bits are left
    /// unchanged.
    ///
    /// :param object value: The bits to deposit. This can be anything promotable to ``Tibs``, and must be ``mask.count()`` bits long.
    /// :param object mask: The mask selecting which positions to write. This can be anything promotable to ``Tibs``, and must be the same length as ``self``.
    /// :return: None
    /// :raises ValueError: if the mask length doesn't match the length of ``self``, or ``value`` is not ``mask.count()`` bits long.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs('0b11010110')
    ///     >>> m.deposit('0b111', '0b10110000')
    ///     >>> m.bin
    ///     '11110110'
    ///
    pub fn deposit(
        mut slf: PyRefMut<'_, Self>,
        value: &Bound<'_, PyAny>,
        mask: Tibs,
    ) -> PyResult<()> {
        // Snapshot value first, both to avoid re-borrowing self when value is
        // self, and so a self-deposit reads the pre-write bits.
        let value = if value.as_ptr() == slf.as_ptr() {
            slf.to_tibs()
        } else {
            Tibs::extract(value.as_borrowed())?
        };
        slf.apply_deposit(&value, &mask)
    }

    /// Return a new Mutibs with a scattered bit field written into it.
    ///
    /// This is the non-inplace version of :meth:`deposit`.
    ///
    /// :param object value: The bits to deposit. This can be anything promotable to ``Tibs``, and must be ``mask.count()`` bits long.
    /// :param object mask: The mask selecting which positions to write. This can be anything promotable to ``Tibs``, and must be the same length as ``self``.
    /// :return: A new Mutibs.
    /// :raises ValueError: if the mask length doesn't match the length of ``self``, or ``value`` is not ``mask.count()`` bits long.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b11010110').deposited('0b111', '0b10110000').bin
    ///     '11110110'
    ///
    pub fn deposited(&self, value: &Bound<'_, PyAny>, mask: Tibs) -> PyResult<Self> {
        let value = Tibs::extract(value.as_borrowed())?;
        let mut out = self.clone();
        out.apply_deposit(&value, &mask)?;
        Ok(out)
    }

    /// Bit-wise 'and' between two Mutibs. Returns new Mutibs.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: A new Mutibs.
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_and(self, &other))
    }

    /// Bit-wise 'or' between two Mutibs. Returns new Mutibs.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: A new Mutibs.
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_or(self, &other))
    }

    /// Bit-wise 'xor' between two Mutibs. Returns new Mutibs.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: A new Mutibs.
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_xor(self, &other))
    }

    /// Reverse bit-wise 'and' between two Mutibs. Returns new Mutibs.
    ///
    /// This method is used when the RHS is a Mutibs and the LHS is not, but can be converted to one.
    ///
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__and__(other)
    }

    /// Reverse bit-wise 'or' between two Mutibs. Returns new Mutibs.
    ///
    /// This method is used when the RHS is a Mutibs and the LHS is not, but can be converted to one.
    ///
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__or__(other)
    }

    /// Reverse bit-wise 'xor' between two Mutibs. Returns new Mutibs.
    ///
    /// This method is used when the RHS is a Mutibs and the LHS is not, but can be converted to one.
    ///
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__xor__(other)
    }

    /// Rotates bit pattern to the left in-place.
    ///
    /// :param int n: The number of bits to rotate by.
    /// :param int | None start: Start of slice to rotate. Defaults to 0.
    /// :param int | None end: End of slice to rotate. Defaults to len(self).
    /// :return: None
    ///
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b10110')
    ///     >>> a.rotate_left(2)
    ///     >>> a
    ///     Mutibs('0b11010')
    ///
    #[pyo3(signature = (n, start=None, end=None), text_signature = "($self, n, start=None, end=None)")]
    pub fn rotate_left(
        mut slf: PyRefMut<'_, Self>,
        n: i64,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<()> {
        slf.apply_rotation(n, start, end, true)
    }

    /// Rotates bit pattern to the right in-place.
    ///
    /// :param int n: The number of bits to rotate by.
    /// :param int | None start: Start of slice to rotate. Defaults to 0.
    /// :param int | None end: End of slice to rotate. Defaults to len(self).
    /// :return: None
    ///
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b10110')
    ///     >>> a.rotate_right(1)
    ///     >>> a
    ///     Mutibs('0b01011')
    ///
    #[pyo3(signature = (n, start=None, end=None), text_signature = "($self, n, start=None, end=None)")]
    pub fn rotate_right(
        mut slf: PyRefMut<'_, Self>,
        n: i64,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<()> {
        slf.apply_rotation(n, start, end, false)
    }

    /// Return a new Mutibs with the bits rotated to the left.
    ///
    /// This is the non-inplace version of :meth:`rotate_left`.
    ///
    /// :param int n: The number of bits to rotate by.
    /// :param int | None start: Start of slice to rotate. Defaults to 0.
    /// :param int | None end: End of slice to rotate. Defaults to len(self).
    /// :return: A new Mutibs.
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b10110').rotated_left(2)
    ///     Mutibs('0b11010')
    ///
    #[pyo3(signature = (n, start=None, end=None), text_signature = "($self, n, start=None, end=None)")]
    pub fn rotated_left(&self, n: i64, start: Option<isize>, end: Option<isize>) -> PyResult<Self> {
        let mut out = self.clone();
        out.apply_rotation(n, start, end, true)?;
        Ok(out)
    }

    /// Return a new Mutibs with the bits rotated to the right.
    ///
    /// This is the non-inplace version of :meth:`rotate_right`.
    ///
    /// :param int n: The number of bits to rotate by.
    /// :param int | None start: Start of slice to rotate. Defaults to 0.
    /// :param int | None end: End of slice to rotate. Defaults to len(self).
    /// :return: A new Mutibs.
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b10110').rotated_right(1)
    ///     Mutibs('0b01011')
    ///
    #[pyo3(signature = (n, start=None, end=None), text_signature = "($self, n, start=None, end=None)")]
    pub fn rotated_right(
        &self,
        n: i64,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Self> {
        let mut out = self.clone();
        out.apply_rotation(n, start, end, false)?;
        Ok(out)
    }

    /// Create a Mutibs by decoding bytes created via `encode()`.
    ///
    /// :param bytes | bytearray b: The encoded bytes to decode.
    /// :return: A new Mutibs.
    /// :raises ValueError: for badly formed, truncated or extended input bytes.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.decode(Mutibs('0b101').encode())
    ///     Mutibs('0b101')
    ///
    #[classmethod]
    #[pyo3(signature = (b, /), text_signature = "(cls, b, /)")]
    pub fn decode(_cls: &Bound<'_, PyType>, b: Vec<u8>) -> PyResult<Self> {
        tibs_codec::decode_bytes::<Mutibs>(b)
    }

    /// Encode the Mutibs as a bytes instance.
    ///
    /// The bytes instance can be used to recreate the Mutibs exactly with :meth:`decode`.
    ///
    /// Use ``Codec.Raw`` when the encoded bytes themselves need to be a stable,
    /// canonical representation. The default ``Codec.Auto`` chooses a valid
    /// encoding for compactness and may produce different bytes for the same
    /// value in a future release.
    ///
    /// :param Codec codec: The codec to use. Defaults to Codec.Auto.
    /// :return: The encoded bytes.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.decode(Mutibs('0b101').encode())
    ///     Mutibs('0b101')
    ///
    #[pyo3(signature = (codec=Codec::Auto), text_signature = "($self, codec=None)")]
    pub fn encode(&self, codec: Option<Codec>) -> PyResult<Vec<u8>> {
        tibs_codec::encode(self, codec)
    }

    /// Set one or many bits set to 1.
    ///
    /// :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
    /// :return: None
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    /// See also :meth:`unset`.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs.from_zeros(10)
    ///     >>> a.set(5)
    ///     >>> a
    ///     Mutibs('0b0000010000')
    ///     >>> a.set([-1, -2])
    ///     >>> a
    ///     Mutibs('0b0000010011')
    ///
    pub fn set<'a>(mut slf: PyRefMut<'a, Self>, pos: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.apply_set_positions(true, pos)
    }

    /// Set one or many bits set to 0.
    ///
    /// :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
    /// :return: None
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    /// See also :meth:`set`.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs.from_ones(10)
    ///     >>> a.unset(5)
    ///     >>> a
    ///     Mutibs('0b1111101111')
    ///     >>> a.unset([-1, -2])
    ///     >>> a
    ///     Mutibs('0b1111101100')
    ///
    pub fn unset<'a>(mut slf: PyRefMut<'a, Self>, pos: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.apply_set_positions(false, pos)
    }

    /// Return a new Mutibs with one or many bits set to 1.
    ///
    /// This is the non-inplace version of :meth:`set`.
    ///
    /// :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
    /// :return: A new Mutibs.
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.from_zeros(5).set_at([1, 3])
    ///     Mutibs('0b01010')
    ///
    pub fn set_at(&self, pos: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut out = self.clone();
        out.apply_set_positions(true, pos)?;
        Ok(out)
    }

    /// Return a new Mutibs with one or many bits set to 0.
    ///
    /// This is the non-inplace version of :meth:`unset`.
    ///
    /// :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
    /// :return: A new Mutibs.
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.from_ones(5).unset_at([1, 3])
    ///     Mutibs('0b10101')
    ///
    pub fn unset_at(&self, pos: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut out = self.clone();
        out.apply_set_positions(false, pos)?;
        Ok(out)
    }

    /// Counts the total number of occurrences of a bit pattern.
    ///
    /// :param object | None value: Either something that can be converted to a ``Tibs``, or a single bit (one of ``0``, ``1``, ``False`` or ``True``). Defaults to counting the set bits.
    /// :param int | None start: The start of the slice to count within. Defaults to 0.
    /// :param int | None end: The end of the slice to count within. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, only occurrences on byte boundaries are counted. Defaults to ``False``.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    ///
    /// :return: The number of times the bit pattern is found.
    /// :raises ValueError: if the slice parameters are invalid, or if the mask length doesn't match
    ///     the length of the value.
    ///
    /// With no ``value`` this counts the set bits, so ``count()`` is the same as ``count(1)``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0xef').count()
    ///     7
    ///     >>> Mutibs('0xef').count(1, 0, 4)
    ///     3
    ///     >>> Mutibs('0xff00ff').count([1, 1, 1])
    ///     12
    ///
    #[pyo3(signature = (value=None, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, value=None, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn count(
        &self,
        py: Python<'_>,
        value: Option<&Bound<'_, PyAny>>,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<usize> {
        let (start, end) = validate_slice(self.len(), start, end)?;
        let haystack = self.as_bitslice();

        let Some(value) = value else {
            // No value given, so count the set bits.
            return match prepare_mask(mask, 1)? {
                Some(_) => Ok(helpers::count_candidate_positions(start, end, byte_aligned)),
                None => Ok(helpers::count_single_bit(
                    haystack,
                    true,
                    start,
                    end,
                    byte_aligned,
                )),
            };
        };

        if let Some(b) = helpers::convert_to_bool(value) {
            return match prepare_mask(mask, 1)? {
                // The only unset single-bit mask matches every bit.
                Some(_) => Ok(helpers::count_candidate_positions(start, end, byte_aligned)),
                None => Ok(helpers::count_single_bit(
                    haystack,
                    b,
                    start,
                    end,
                    byte_aligned,
                )),
            };
        }

        match Tibs::extract(value.as_borrowed()) {
            Ok(v) => {
                let mask = prepare_mask(mask, v.len())?;
                match mask {
                    Some(mask) => helpers::count_bitvec_masked(
                        py,
                        haystack,
                        v.as_bitslice(),
                        mask.as_bitslice(),
                        start,
                        end,
                        byte_aligned,
                    ),
                    None => helpers::count_bitvec(
                        py,
                        haystack,
                        v.as_bitslice(),
                        start,
                        end,
                        byte_aligned,
                    ),
                }
            }
            Err(err) => {
                if err.is_instance_of::<PyTypeError>(py)
                    && (value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>())
                {
                    Err(err)
                } else {
                    Err(PyValueError::new_err(
                        "Cannot convert value to 0, 1 or a Tibs",
                    ))
                }
            }
        }
    }

    /// Return True if all bits are equal to 1, otherwise return False.
    ///
    /// :return: ``True`` if all bits are 1, otherwise ``False``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b1111').all()
    ///     True
    ///     >>> Mutibs('0b1011').all()
    ///     False
    ///
    pub fn all(&self) -> bool {
        self.as_bitvec_ref().all()
    }

    /// Return True if any bits are equal to 1, otherwise return False.
    ///
    /// :return: ``True`` if any bits are 1, otherwise ``False``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b0000').any()
    ///     False
    ///     >>> Mutibs('0b1000').any()
    ///     True
    ///
    pub fn any(&self) -> bool {
        self.as_bitvec_ref().any()
    }

    /// Find last occurrence of a bit sequence.
    ///
    /// Returns the bit position if found, or None if not found.
    ///
    /// :param object needle: The bits to find. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: The bit position if found, or None if not found.
    /// :raises ValueError: if ``needle`` is empty, if the slice parameters are invalid, or if the
    ///     mask length doesn't match the needle length.
    ///
    /// .. code-block:: pycon
    ///
    ///      >>> Mutibs('0b10111011').rfind('0b11')
    ///      6
    ///      >>> Mutibs('0b10111011').rfind('0b00', mask='0b10')
    ///      5
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, needle, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn rfind(
        &self,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Option<usize>> {
        self.find_impl(py, needle, start, end, byte_aligned, mask, true)
    }

    /// Invert one or many bits in place.
    ///
    /// :param int | Iterable[int] | None pos: Either a single bit position or an iterable of bit positions.
    /// :return: None
    ///
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b10111')
    ///     >>> a.invert(1)
    ///     >>> a
    ///     Mutibs('0b11111')
    ///     >>> a.invert([0, 2])
    ///     >>> a
    ///     Mutibs('0b01011')
    ///     >>> a.invert()
    ///     >>> a
    ///     Mutibs('0b10100')
    ///
    #[pyo3(signature = (pos = None), text_signature = "($self, pos=None)")]
    pub fn invert<'a>(mut slf: PyRefMut<'a, Self>, pos: Option<&Bound<'a, PyAny>>) -> PyResult<()> {
        slf.apply_invert_positions(pos)
    }

    /// Return a new Mutibs with selected bits inverted.
    ///
    /// This is the non-inplace version of :meth:`invert`.
    ///
    /// :param int | Iterable[int] | None pos: Either a single bit position, an iterable of bit positions,
    ///   or None to invert every bit. Defaults to None.
    /// :return: A new Mutibs.
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b10110').inverted([0, 2])
    ///     Mutibs('0b00010')
    ///
    #[pyo3(signature = (pos = None), text_signature = "($self, pos=None)")]
    pub fn inverted(&self, pos: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut out = self.clone();
        out.apply_invert_positions(pos)?;
        Ok(out)
    }

    /// Reverse bits in-place.
    ///
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b10110')
    ///     >>> a.reverse()
    ///     >>> a
    ///     Mutibs('0b01101')
    ///
    pub fn reverse(mut slf: PyRefMut<'_, Self>) {
        helpers::reverse_bitvec_in_place(slf.as_mut_bitvec_ref());
    }

    /// Return a new instance with the bits reversed.
    ///
    /// :return: Mutibs
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b00011')
    ///     >>> a.reversed()
    ///     Mutibs('0b11000')
    ///
    pub fn reversed(&self) -> Self {
        BitCollection::reverse_copy(self)
    }

    /// Swap byte order in-place.
    ///
    /// The selected slice will be byte-swapped. It must be a multiple of
    /// byte_length long.
    ///
    /// :param int | None byte_length: An int giving the number of bytes in each swap, or None (the default)
    ///   to do a single reverse over the selected slice.
    /// :param int | None start: Start of slice to byte-swap. Defaults to 0.
    /// :param int | None end: End of slice to byte-swap. Defaults to len(self).
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0x12345678')
    ///     >>> a.byte_swap(2)
    ///     >>> a
    ///     Mutibs('0x34127856')
    ///
    #[pyo3(signature = (byte_length = None, start=None, end=None), text_signature = "($self, byte_length=None, start=None, end=None)")]
    pub fn byte_swap(
        mut slf: PyRefMut<'_, Self>,
        byte_length: Option<i64>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<()> {
        slf.apply_byte_swap(byte_length, start, end)
    }

    /// Return a new instance with the byte order swapped.
    ///
    /// The selected slice will be byte-swapped. It must be a multiple of
    /// byte_length long.
    ///
    /// :param int | None byte_length: An int giving the number of bytes in each swap, or None (the default)
    ///   to do a single reverse over the selected slice.
    /// :param int | None start: Start of slice to byte-swap. Defaults to 0.
    /// :param int | None end: End of slice to byte-swap. Defaults to len(self).
    /// :return: Mutibs
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0x12345678')
    ///     >>> b = a.byte_swapped(2)
    ///     >>> b
    ///     Mutibs('0x34127856')
    ///
    #[pyo3(signature = (byte_length = None, start=None, end=None), text_signature = "($self, byte_length=None, start=None, end=None)")]
    pub fn byte_swapped(
        &self,
        byte_length: Option<i64>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Mutibs> {
        let mut out = self.clone();
        out.apply_byte_swap(byte_length, start, end)?;
        Ok(out)
    }

    /// Return the instance with every bit inverted.
    ///
    /// :return: A new Mutibs.
    ///
    /// Inverting an empty Mutibs gives an empty Mutibs, as :meth:`invert` does.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> ~Mutibs('0b10110')
    ///     Mutibs('0b01001')
    ///
    pub fn __invert__(&self) -> PyResult<Self> {
        Ok(BitCollection::invert_copy(self))
    }

    /// Return new Mutibs shifted by n to the left.
    ///
    /// :param int n: The number of bits to shift. Must be >= 0.
    /// :return: A new Mutibs.
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b001100') << 2
    ///     Mutibs('0b110000')
    ///
    pub fn __lshift__(&self, n: i64) -> PyResult<Self> {
        let shift = validate_shift(self, n)?;
        Ok(self.lshift(shift))
    }

    /// Return new Mutibs shifted by n to the right.
    ///
    /// :param int n: The number of bits to shift. Must be >= 0.
    /// :return: A new Mutibs.
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b001100') >> 2
    ///     Mutibs('0b000011')
    ///
    pub fn __rshift__(&self, n: i64) -> PyResult<Self> {
        let shift = validate_shift(self, n)?;
        Ok(self.rshift(shift))
    }

    /// Return a new copy of the Mutibs for the copy module.
    pub fn __copy__(&self) -> Self {
        Mutibs::from_bv(self.to_bitvec())
    }

    /// Create and return a Tibs instance from a copy of the Mutibs data.
    ///
    /// This copies the underlying binary data, giving a new independent Tibs object.
    /// If you no longer need the Mutibs, consider using :meth:`as_tibs` instead to avoid the copy.
    ///
    /// :return: A new Tibs instance with the same bit data.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b10110')
    ///     >>> b = a.to_tibs()
    ///     >>> a
    ///     Mutibs('0b10110')
    ///     >>> b
    ///     Tibs('0b10110')
    ///
    pub fn to_tibs(&self) -> Tibs {
        Tibs::from_bv(self.to_bitvec())
    }

    /// Create and return a Tibs instance by moving the Mutibs data.
    ///
    /// The data is moved to the new Tibs, so the Mutibs will be empty after the operation.
    /// This is more efficient than :meth:`to_tibs` if you no longer need the Mutibs.
    ///
    /// It will try to reclaim any excess memory capacity that the Mutibs may have had.
    ///
    /// :return: A Tibs instance with the same bit data.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b10110')
    ///     >>> b = a.as_tibs()
    ///     >>> a
    ///     Mutibs()
    ///     >>> b
    ///     Tibs('0b10110')
    ///
    pub fn as_tibs(&mut self) -> Tibs {
        let mut data = std::mem::take(&mut *self.as_mut_bitvec_ref());
        data.shrink_to_fit();
        Tibs::from_bv(data)
    }

    /// Clear all bits, making the Mutibs empty.
    ///
    /// This doesn't change the allocated capacity, so won't free up any memory.
    ///
    /// :return: None.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs('0b1011')
    ///     >>> m.clear()
    ///     >>> m
    ///     Mutibs()
    ///
    pub fn clear(&mut self) {
        self.as_mut_bitvec_ref().clear();
    }

    /// Return the number of bits the Mutibs can hold without reallocating memory.
    ///
    /// The capacity is always equal to or greater than the current length of the Mutibs.
    /// If the length ever exceeds the capacity then memory will have to be reallocated, and the
    /// capacity will increase.
    ///
    /// It can be helpful as a performance optimization to reserve enough capacity before
    /// constructing a large Mutibs incrementally. See also :meth:`reserve`.
    ///
    /// :return: The current capacity in bits.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs()
    ///     >>> m.capacity() >= len(m)
    ///     True
    ///
    pub fn capacity(&self) -> usize {
        self.as_bitvec_ref().capacity()
    }

    /// Reserve memory for at least `additional` more bits to be appended to the Mutibs.
    ///
    /// This can be helpful as a performance optimization to avoid multiple memory reallocations when
    /// constructing a large Mutibs incrementally. If enough memory is already reserved then
    /// this method will have no effect. See also :meth:`capacity`.
    ///
    /// :param int additional: The number of bits that can be appended without any further memory reallocations.
    /// :return: None.
    ///
    /// .. code-block:: python
    ///
    ///     m = Mutibs()
    ///     m.reserve(1000)
    ///
    pub fn reserve(&mut self, additional: usize) {
        self.as_mut_bitvec_ref().reserve(additional);
    }

    /// Concatenate Mutibs and return a new Mutibs.
    ///
    /// :param object other: The bits to append. This can be anything promotable to ``Tibs``.
    /// :return: A new Mutibs.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b10') + '0b1'
    ///     Mutibs('0b101')
    ///
    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // We accept the PyAny and convert manually here because if we instead
        // accept a Tibs, then correct types with wrong values (e.g. a malformed string)
        // will fail and return a TypeError instead of ValueError which we can't control.
        let other = Tibs::extract(other.as_borrowed())?;
        Ok(Mutibs::from_bv(concatenate_bitcollections(self, &other)))
    }

    /// Concatenate Mutibs and return a new Mutibs.
    ///
    /// :param object other: The bits to prepend. This can be anything promotable to ``Tibs``.
    /// :return: A new Mutibs.
    ///
    pub fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        Ok(Mutibs::from_bv(concatenate_bitcollections(&other, self)))
    }

    /// Concatenate in-place.
    ///
    /// :param object other: The bits to append. This can be anything promotable to ``Tibs``.
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs('0b10')
    ///     >>> m += '0b1'
    ///     >>> m
    ///     Mutibs('0b101')
    ///
    pub fn __iadd__(slf: PyRefMut<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::extend(slf, other)?;
        Ok(())
    }

    /// Append a single bit to the current Mutibs in-place.
    ///
    /// :param bool | int bit: Either ``0``, ``1``, ``True`` or ``False`` to append.
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs()
    ///     >>> a.append(True)
    ///     >>> a
    ///     Mutibs('0b1')
    ///
    pub fn append<'a>(mut slf: PyRefMut<'a, Self>, bit: &Bound<'_, PyAny>) -> PyResult<()> {
        match helpers::convert_to_bool(bit) {
            Some(b) => {
                slf.as_mut_bitvec_ref().push(b);
                Ok(())
            }
            None => Err(PyTypeError::new_err(
                "Only True, False, 0 or 1 can be appended.",
            )),
        }
    }

    /// Remove and return the final bit.
    ///
    /// :return: bool
    /// :raises IndexError: if the Mutibs is empty.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b101')
    ///     >>> a.pop()
    ///     True
    ///     >>> a
    ///     Mutibs('0b10')
    ///
    pub fn pop<'py>(&mut self, py: Python<'py>) -> PyResult<pyo3::Borrowed<'py, 'py, PyBool>> {
        match self.as_mut_bitvec_ref().pop() {
            Some(bit) => Ok(PyBool::new(py, bit)),
            None => Err(PyIndexError::new_err("pop from empty Mutibs.")),
        }
    }

    /// Extend the current Mutibs in-place.
    ///
    /// :param object bs: The bits to extend with. This can be anything promotable to ``Tibs``.
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0x0f')
    ///     >>> a.extend('0x0a')
    ///     >>> a
    ///     Mutibs('0x0f0a')
    ///
    #[pyo3(signature = (bs, /), text_signature = "($self, bs, /)")]
    pub fn extend<'a>(mut slf: PyRefMut<'a, Self>, bs: &Bound<'_, PyAny>) -> PyResult<()> {
        // Check if bs is the same object as slf
        if bs.as_ptr() == slf.as_ptr() {
            // If bs is slf, clone inner bits first then extend
            let bits_clone = slf.to_bitvec();
            slf.append_run(bits_clone.as_raw_slice(), 0, bits_clone.len());
        } else if let Ok(tibs) = bs.extract::<PyRef<Tibs>>() {
            // Existing tibs containers can be borrowed directly; avoid
            // materializing a temporary Tibs through the generic converter.
            slf.append_collection(&*tibs);
        } else if let Ok(mutibs) = bs.extract::<PyRef<Mutibs>>() {
            // Mutibs inputs also expose a stable bit slice while we hold the
            // Python reference.
            slf.append_collection(&*mutibs);
        } else {
            let bits = promote_to_bv(bs)?;
            if slf.is_empty() {
                // For an empty receiver, move the promoted BitVec into place
                // rather than copying it into another allocation.
                *slf.as_mut_bitvec_ref() = bits;
            } else {
                slf.append_run(
                    bits.as_raw_slice(),
                    head_bit_offset(bits.as_bitslice()),
                    bits.len(),
                );
            }
        }
        Ok(())
    }

    /// Extend the current Mutibs in-place from the start.
    ///
    /// This is broadly equivalent to ``self = bs + self``.
    /// Note that this method is inherently slower than :meth:`extend` and
    /// should be avoided in performance critical code. See also :meth:`from_joined`.
    ///
    /// :param object bs: The bits to prepend to the current Mutibs. This can be anything promotable to ``Tibs``.
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0x0f')
    ///     >>> a.extend_left('0x0a')
    ///     >>> a
    ///     Mutibs('0x0a0f')
    ///
    #[pyo3(signature = (bs, /), text_signature = "($self, bs, /)")]
    pub fn extend_left<'a>(mut slf: PyRefMut<'a, Self>, bs: &Bound<'_, PyAny>) -> PyResult<()> {
        // Check for self-prepending
        // Prepending has to build a new buffer either way, so both arms lay
        // the two runs down over bytes rather than growing a BitVec by bits.
        if bs.as_ptr() == slf.as_ptr() {
            let doubled = concatenate_bitcollections(&*slf, &*slf);
            *slf.as_mut_bitvec_ref() = doubled;
        } else {
            let to_prepend = Tibs::extract(bs.as_borrowed())?;
            if to_prepend.is_empty() {
                return Ok(());
            }
            let joined = concatenate_bitcollections(&to_prepend, &*slf);
            *slf.as_mut_bitvec_ref() = joined;
        }
        Ok(())
    }

    /// Search and replace in-place.
    ///
    /// :param object old: The bits to search for. This can be anything promotable to ``Tibs``.
    /// :param object new: The bits to replace with. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param int | None count: If present, the maximum number of replacements to make.
    /// :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: The number of replacements made.
    /// :raises ValueError: if old is empty, count is negative, the slice parameters are invalid or
    ///     the mask length doesn't match the length of old.
    ///
    /// The ``mask`` affects only which bits have to match; the whole of each match is still
    /// replaced by ``new``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs('0b00010010')
    ///     >>> m.replace([0, 1], [1, 1, 1])
    ///     2
    ///     >>> m
    ///     Mutibs('0b0011101110')
    ///
    #[pyo3(signature = (old, new, start=None, end=None, count=None, byte_aligned=false, mask=None), text_signature = "($self, old, new, start=None, end=None, count=None, byte_aligned=False, mask=None)")]
    pub fn replace<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'_>,
        old: &Bound<'_, PyAny>,
        new: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
        count: Option<i64>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<usize> {
        let old = if old.as_ptr() == slf.as_ptr() {
            slf.to_tibs()
        } else {
            Tibs::extract(old.as_borrowed())?
        };

        if old.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to replace."));
        }
        let new = if new.as_ptr() == slf.as_ptr() {
            slf.to_tibs()
        } else {
            Tibs::extract(new.as_borrowed())?
        };
        slf.apply_replace_bits(py, old, new, start, end, count, byte_aligned, mask)
    }

    /// Search and replace and return a new Mutibs.
    ///
    /// This is the non-inplace version of :meth:`replace`.
    ///
    /// :param object old: The bits to search for. This can be anything promotable to ``Tibs``.
    /// :param object new: The bits to replace with. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param int | None count: If present, the maximum number of replacements to make.
    /// :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: A new Mutibs.
    /// :raises ValueError: if old is empty, count is negative, the slice parameters are invalid or
    ///     the mask length doesn't match the length of old.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b00010010').replaced([0, 1], [1, 1, 1])
    ///     Mutibs('0b0011101110')
    ///     >>> Mutibs('0x1f2e3f').replaced('0x0f', '0x00', mask='0x0f', byte_aligned=True)
    ///     Mutibs('0x002e00')
    ///
    #[pyo3(signature = (old, new, start=None, end=None, count=None, byte_aligned=false, mask=None), text_signature = "($self, old, new, start=None, end=None, count=None, byte_aligned=False, mask=None)")]
    pub fn replaced(
        &self,
        py: Python<'_>,
        old: &Bound<'_, PyAny>,
        new: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
        count: Option<i64>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Self> {
        let old = Tibs::extract(old.as_borrowed())?;
        let new = Tibs::extract(new.as_borrowed())?;
        let mut out = self.clone();
        let _ = out.apply_replace_bits(py, old, new, start, end, count, byte_aligned, mask)?;
        Ok(out)
    }

    /// Insert bits at position pos.
    ///
    /// Clips to start or end if insert position is out of range.
    ///
    /// :param int pos: The bit position to insert at.
    /// :param object bs: The bits to insert. This can be anything promotable to ``Tibs``.
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b1011')
    ///     >>> a.insert(2, '0b00')
    ///     >>> a
    ///     Mutibs('0b100011')
    ///
    #[pyo3(signature = (pos, bs, /), text_signature = "($self, pos, bs, /)")]
    pub fn insert<'a>(
        mut slf: PyRefMut<'a, Self>,
        pos: isize,
        bs: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // Check for self assignment
        let bs = if bs.as_ptr() == slf.as_ptr() {
            slf.to_tibs()
        } else {
            Tibs::extract(bs.as_borrowed())?
        };
        slf.apply_insert_bits(pos, &bs)
    }

    /// Insert bits at position pos and return a new Mutibs.
    ///
    /// This is the non-inplace version of :meth:`insert`.
    ///
    /// :param int pos: The bit position to insert at. Clips to the start or end if out of range.
    /// :param object bs: The bits to insert. This can be anything promotable to ``Tibs``.
    /// :return: A new Mutibs.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b1011').inserted(2, '0b00')
    ///     Mutibs('0b100011')
    ///
    #[pyo3(signature = (pos, bs, /), text_signature = "($self, pos, bs, /)")]
    pub fn inserted(&self, pos: isize, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bs = Tibs::extract(bs.as_borrowed())?;
        let mut out = self.clone();
        out.apply_insert_bits(pos, &bs)?;
        Ok(out)
    }

    /// Shift bits to the left in-place.
    ///
    /// :param int n: The number of bits to shift. Must be >= 0.
    /// :return: self
    ///
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> b = Mutibs('0b001100')
    ///     >>> b <<= 2
    ///     >>> b.bin
    ///     '110000'
    ///
    pub fn __ilshift__(mut slf: PyRefMut<'_, Self>, n: i64) -> PyResult<()> {
        // Shifting by the whole length or more zero-fills, matching __lshift__.
        let shift = validate_shift(&*slf, n)?.min(slf.len());
        slf.as_mut_bitvec_ref().shift_start(shift);
        Ok(())
    }

    /// Shift bits to the right in-place.
    ///
    /// :param int n: The number of bits to shift. Must be >= 0.
    /// :return: self
    ///
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> b = Mutibs('0b001100')
    ///     >>> b >>= 2
    ///     >>> b.bin
    ///     '000011'
    ///
    pub fn __irshift__(mut slf: PyRefMut<'_, Self>, n: i64) -> PyResult<()> {
        // Shifting by the whole length or more zero-fills, matching __rshift__.
        let shift = validate_shift(&*slf, n)?.min(slf.len());
        slf.as_mut_bitvec_ref().shift_end(shift);
        Ok(())
    }

    /// Return the Mutibs as a bytes object.
    ///
    /// :return: The bytes representation.
    /// :raises ValueError: if the length is not a multiple of 8.
    pub fn __bytes__(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        BitCollection::to_py_bytes(self, py)
    }

    /// Return new Mutibs consisting of n concatenations of self.
    ///
    /// Called for expression of the form 'a = b*3'.
    ///
    /// :param int n: The number of concatenations. Must be >= 0.
    /// :return: A new Mutibs.
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b10') * 3
    ///     Mutibs('0b101010')
    ///
    pub fn __mul__(&self, n: i64) -> PyResult<Self> {
        if n < 0 {
            return Err(PyValueError::new_err(
                "Cannot multiply by a negative integer.",
            ));
        }
        Ok(self.multiply(n as usize))
    }

    /// Return Mutibs consisting of n concatenations of self.
    ///
    /// Called for expressions of the form 'a = 3*b'.
    ///
    /// :param int n: The number of concatenations. Must be >= 0.
    /// :return: A new Mutibs.
    /// :raises ValueError: if n < 0.
    ///
    pub fn __rmul__(&self, n: i64) -> PyResult<Self> {
        self.__mul__(n)
    }

    /// In-place bit-wise 'and'.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: None
    /// :raises ValueError: if the two bit sequences have differing lengths.
    ///
    pub fn __iand__(mut slf: PyRefMut<'_, Self>, other: Tibs) -> PyResult<()> {
        slf.iand(other.as_bitslice())
    }

    /// In-place bit-wise 'or'.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: None
    /// :raises ValueError: if the two bit sequences have differing lengths.
    ///
    pub fn __ior__(mut slf: PyRefMut<'_, Self>, other: Tibs) -> PyResult<()> {
        slf.ior(other.as_bitslice())
    }

    /// In-place bit-wise 'xor'.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: None
    /// :raises ValueError: if the two bit sequences have differing lengths.
    ///
    pub fn __ixor__(mut slf: PyRefMut<'_, Self>, other: Tibs) -> PyResult<()> {
        slf.ixor(other.as_bitslice())
    }

    /// In-place multiplication by a non-negative integer.
    ///
    /// :param int n: The number of concatenations. Must be >= 0.
    /// :return: None
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs('0b10')
    ///     >>> m *= 3
    ///     >>> m
    ///     Mutibs('0b101010')
    ///
    pub fn __imul__(mut slf: PyRefMut<'_, Self>, n: i64) -> PyResult<()> {
        match n {
            i if i < 0 => Err(PyValueError::new_err(
                "Cannot multiply by a negative integer.",
            )),
            0 => {
                slf.clear();
                Ok(())
            }
            1 => Ok(()),
            i => {
                let n = i as usize;
                // One byte-wide pass per copy, so the total is the size of the
                // result. Doubling moved the same bits but a bit at a time.
                let original = slf.to_bitvec();
                let len = original.len();
                for _ in 1..n {
                    slf.append_run(original.as_raw_slice(), 0, len);
                }
                Ok(())
            }
        }
    }

    // Supply some more helpful errors for things which aren't supported for Mutibs, but are for Tibs.
    pub fn __iter__(&self) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "'Mutibs' objects are not iterable. You can use '.to_tibs()' or '.as_tibs()' to convert to a 'Tibs' object that does support iteration.",
        ))
    }

    pub fn __getattr__(&self, name: String) -> PyResult<()> {
        if name == "find_all_iter"
            || name == "rfind_all_iter"
            || name == "chunks_iter"
            || name == "rchunks_iter"
            || name == "to_values_iter"
        {
            Err(PyAttributeError::new_err(format!(
                "'Mutibs' object has no attribute '{name}', but `Tibs` does. Perhaps try '.to_tibs().{name}()' instead."
            )))
        } else {
            Err(PyAttributeError::new_err(format!(
                "'Mutibs' object has no attribute '{name}'"
            )))
        }
    }
}
