use crate::codec as tibs_codec;
use crate::core::{
    BitCollection, concatenate_bitcollections, push_collection_run, read_split_positions,
    repeat_bitcollection,
};
use crate::dtype::extract_dtype;
use crate::enums::{BitOrder, ByteOrder, Codec};
use crate::helpers::{
    BS, BV, BitConcat, LogicalOp, MaskedMatcher, bv_from_bin, bv_from_bools, bv_from_bytes_slice,
    bv_from_f64, bv_from_hex, bv_from_int, bv_from_oct, bv_from_ones, bv_from_random, bv_from_uint,
    bv_from_zeros, bytes_like_to_vec, copy_bits, deposit_masked_bytes, fill_bits, find_bitvec,
    head_bit_offset, logical_op_assign_bytes, move_bits, padded_bytes_from_offset, promote_to_bv,
    rotate_bits_left, str_to_bv, try_extract_index, validate_index, validate_length,
    validate_logical_op_lengths, validate_offset, validate_shift, validate_slice, with_locked,
    with_locked_mut, with_locked_mut2, with_locked2,
};
use crate::tibs_::{
    SearchParams, Tibs, bv_from_value, bv_from_values_iter, count_in_bits, find_all_in_bits,
    find_in_bits, prepare_mask, py_from_value, py_values_from_range, resolve_count_target,
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

/// Bit positions for `set`, `unset` and `invert` once read out of Python.
///
/// The Python side of these methods - indexing a list, walking an iterable,
/// reading a `range`'s attributes, calling `__index__` - has to finish before
/// the critical section is entered, so it produces one of these and the write
/// works from it. See [`crate::helpers::locking`].
pub(crate) enum Positions {
    /// Every bit: `invert()` with no argument.
    All,
    /// One index, not yet range-checked.
    One(isize),
    /// Up to `INLINE_POSITIONS` indices, held inline so that the common short
    /// `set([1, 3, 5])` does not allocate on its way out of Python.
    Few {
        buf: [isize; INLINE_POSITIONS],
        count: usize,
    },
    /// More indices than fit inline, none yet range-checked.
    Many(Vec<isize>),
    /// A `range`, kept as its parts so the stride is walked at write time.
    Range {
        start: isize,
        stop: isize,
        step: isize,
    },
    /// The argument was neither an index nor iterable. The error is raised at
    /// write time so that the message can name the calling method.
    NotIndexable,
    /// A single index too large to be one. Raised at write time so the message
    /// can quote the length actually in force.
    IndexOverflow,
}

/// How many positions `Positions::Few` holds without allocating.
///
/// Deliberately small. Every `Positions` value is as large as this array, and
/// it is moved out of the reader on every call, so a generous inline buffer
/// makes the single-index `set(7)` pay for a capacity it never uses: at 16 the
/// enum reached 136 bytes and `set(7)` measured 24% slower than not splitting
/// at all. Four covers the short literal lists and tuples without that.
const INLINE_POSITIONS: usize = 4;

/// An indexing key for `__setitem__` and `__delitem__`, read out of Python.
///
/// Converting a key calls `__index__`, so it happens before the critical
/// section and the outcome is carried in here. A slice is kept as the object:
/// resolving it needs the length, and `PySlice::indices` is a C call that
/// reaches Python only for a slice built from exotic components.
enum ItemKey<'py> {
    /// An integer index, not yet range-checked.
    Index(isize),
    /// A slice, to be resolved against the length in force at write time.
    Slice(Bound<'py, PySlice>),
    /// Neither an index nor a slice.
    Unusable,
    /// The conversion raised. Carried rather than returned so that the message
    /// can quote the length actually in force.
    Failed(PyErr),
}

fn read_item_key<'py>(key: &Bound<'py, PyAny>) -> ItemKey<'py> {
    // Exact ints first, then slices, and only then the general integer
    // extraction, which raises and discards a Python exception for a key it
    // cannot convert.
    if unsafe { ffi::PyLong_Check(key.as_ptr()) } != 0 {
        let index = unsafe { ffi::PyLong_AsSsize_t(key.as_ptr()) };
        if index == -1 && unsafe { !ffi::PyErr_Occurred().is_null() } {
            return ItemKey::Failed(PyErr::fetch(key.py()));
        }
        return ItemKey::Index(index);
    }
    if let Ok(slice) = key.cast::<PySlice>() {
        return ItemKey::Slice(slice.clone());
    }
    match try_extract_index(key) {
        Ok(Some(index)) => ItemKey::Index(index),
        Ok(None) => ItemKey::Unusable,
        Err(error) => ItemKey::Failed(error),
    }
}

fn index_overflow_error(length: usize) -> PyErr {
    PyIndexError::new_err(format!("Index is out of range for length of {length}"))
}

fn index_conversion_error(py: Python<'_>, error: PyErr, length: usize) -> PyErr {
    if error.is_instance_of::<PyOverflowError>(py) {
        PyIndexError::new_err(format!("Index is out of range for length of {length}"))
    } else {
        error
    }
}

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
#[pyclass(sequence, skip_from_py_object, module = "tibs")]
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

    /// A copy of `length` bits starting `start_bit` bits in, as a `Tibs`.
    ///
    /// This is what a windowed read of a `Mutibs` should go through. Copying
    /// only the window is the point: reaching for `to_tibs` and slicing the
    /// result would copy the whole container on every read, which turns a loop
    /// of reads over a large `Mutibs` quadratic.
    ///
    /// A window short enough to live in a `Tibs`'s inline storage is assembled
    /// straight into it. Going through `copied_range` would heap-allocate one
    /// or two bytes for `Tibs::from_bv` to copy inline and drop again, and a
    /// loop of small reads is exactly where that allocation is the cost.
    ///
    /// `start_bit + length` must be within the bit length.
    pub(crate) fn window(&self, start_bit: usize, length: usize) -> Tibs {
        debug_assert!(start_bit + length <= self.len());
        if length == 0 || length > helpers::FAST_INT_BITS {
            return Tibs::from_bv(self.copied_range(start_bit, length));
        }
        let absolute = self.storage_head_offset() + start_bit;
        let bytes = &self.data.as_raw_slice()[absolute / 8..];
        let mut inline = [0u8; helpers::FAST_INT_BITS / 8];
        let out = &mut inline[..length.div_ceil(8)];
        match absolute % 8 {
            0 => {
                out.copy_from_slice(&bytes[..out.len()]);
                helpers::mask_padding_bits(out, length);
            }
            offset => helpers::copy_unaligned_padded_bytes(bytes, offset, length, out),
        }
        Tibs::from_inline_bytes(inline, length)
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

    /// Shift the value in place by `by` bits, zero filling the vacated end.
    ///
    /// `by` must not exceed the bit length. bitvec's `shift_start`/`shift_end`
    /// carry one bit at a time; this is a byte-wide slide and a `memset`.
    pub(crate) fn shift_in_place(&mut self, by: usize, towards_start: bool) {
        let len = self.len();
        debug_assert!(by <= len);
        if by == 0 || len == 0 {
            return;
        }
        let head = self.storage_head_offset();
        let bytes = self.as_mut_bitvec_ref().as_raw_mut_slice();
        let keep = len - by;
        if towards_start {
            move_bits(bytes, head + by, head, keep);
            fill_bits(bytes, head + keep, by, false);
        } else {
            move_bits(bytes, head, head + by, keep);
            fill_bits(bytes, head, by, false);
        }
    }

    /// Append every bit of `bits` in place.
    pub(crate) fn append_collection(&mut self, bits: &impl BitCollection) {
        let (bytes, offset, _) = bits.raw_data_ref();
        self.append_run(bytes, offset, bits.len());
    }

    #[inline]
    pub(crate) fn as_mut_bitvec_ref(&mut self) -> &mut BV {
        &mut self.data
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
        Ok(Self::join_parts(parts, total_len))
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

    fn join_parts(parts: Vec<JoinedPart<'_>>, total_len: usize) -> BV {
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
        out.into_bitvec()
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
        head_bit_offset(self.data.as_bitslice())
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
        let (value_bytes, value_offset, _) = value.raw_data_ref();
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

    /// Install a replacement value that has to be built for the current length.
    ///
    /// `bv_from_uint` and `bv_from_int` call `__index__` on their argument, and
    /// `int.to_bytes` for anything past a `u64`, so the value cannot be built
    /// under the lock. That leaves a window in which another thread - or a
    /// re-entrant `__index__` - can resize the object, so the length is
    /// confirmed before the value is installed. These writes promise not to
    /// change the length, so installing a value built for a stale one is not an
    /// option, and there is no correct length to fall back on.
    fn assign_sized(
        slf: &Bound<'_, Self>,
        build: impl FnOnce(usize) -> PyResult<BV>,
    ) -> PyResult<()> {
        let length = with_locked(slf, |m| Ok(m.len()))?;
        let value = build(length)?;
        with_locked_mut(slf, |m| {
            if m.len() != length {
                return Err(PyValueError::new_err(format!(
                    "The Mutibs changed length from {length} to {} while the value was being converted, so a write that must preserve the length cannot be applied.",
                    m.len()
                )));
            }
            m.replace_with_bv(value);
            Ok(())
        })
    }

    #[inline]
    fn assign_u(slf: &Bound<'_, Self>, u: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::assign_sized(slf, |length| bv_from_uint(u, length, false))
    }

    #[inline]
    fn assign_i(slf: &Bound<'_, Self>, i: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::assign_sized(slf, |length| bv_from_int(i, length, false))
    }

    /// Unlike `assign_u` and `assign_i`, `f` arrives already converted and
    /// `bv_from_f64` runs no Python, so this needs only the one lock.
    #[inline]
    fn assign_f(slf: &Bound<'_, Self>, f: f64) -> PyResult<()> {
        with_locked_mut(slf, |m| {
            let value = bv_from_f64(f, m.len(), false)?;
            m.replace_with_bv(value);
            Ok(())
        })
    }

    #[inline]
    fn replace_with_bv(&mut self, value: BV) {
        self.data = value;
    }

    fn apply_logical_op(&mut self, other: &Tibs, op: LogicalOp) -> PyResult<()> {
        let len = self.len();
        validate_logical_op_lengths(len, other.len())?;
        if len == 0 {
            return Ok(());
        }
        let (rhs, rhs_offset, _) = other.raw_data_ref();
        let lhs_offset = self.storage_head_offset();
        logical_op_assign_bytes(
            self.as_mut_bitvec_ref().as_raw_mut_slice(),
            lhs_offset,
            rhs,
            rhs_offset,
            len,
            op,
        );
        Ok(())
    }

    pub(crate) fn ixor(&mut self, other: &Tibs) -> PyResult<()> {
        self.apply_logical_op(other, LogicalOp::Xor)
    }

    pub(crate) fn ior(&mut self, other: &Tibs) -> PyResult<()> {
        self.apply_logical_op(other, LogicalOp::Or)
    }

    pub(crate) fn iand(&mut self, other: &Tibs) -> PyResult<()> {
        self.apply_logical_op(other, LogicalOp::And)
    }

    pub(crate) fn set_from_sequence(&mut self, value: bool, indices: &[isize]) -> PyResult<()> {
        let len = self.len();
        // Every index is checked before any bit is written, so a bad one leaves
        // the Mutibs untouched.
        for index in indices {
            validate_index(*index, len)?;
        }
        let head = self.storage_head_offset();
        let raw = self.data.as_raw_mut_slice();
        for index in indices {
            // Checked above, so this normalisation cannot leave the range.
            let index = head
                + if *index < 0 {
                    (len as isize + *index) as usize
                } else {
                    *index as usize
                };
            let mask = 0x80u8 >> (index & 7);
            if value {
                raw[index >> 3] |= mask;
            } else {
                raw[index >> 3] &= !mask;
            }
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

    /// Bit positions for `set`, `unset` and `invert`, read out of Python.
    ///
    /// Indices are kept unvalidated: they are range-checked against the length
    /// in force when the write happens, not the one seen while reading.
    pub(crate) fn read_positions(pos: Option<&Bound<'_, PyAny>>) -> PyResult<Positions> {
        let Some(pos) = pos else {
            return Ok(Positions::All);
        };
        if let Ok(list) = pos.cast::<PyList>() {
            let items = list.as_ptr();
            return Self::read_sequence_items(list.py(), list.len(), |index| unsafe {
                ffi::PyList_GetItem(items, index)
            });
        }
        if let Ok(tuple) = pos.cast::<PyTuple>() {
            let items = tuple.as_ptr();
            return Self::read_sequence_items(tuple.py(), tuple.len(), |index| unsafe {
                ffi::PyTuple_GetItem(items, index)
            });
        }
        let index = match try_extract_index(pos) {
            Ok(index) => index,
            Err(error) if error.is_instance_of::<PyOverflowError>(pos.py()) => {
                return Ok(Positions::IndexOverflow);
            }
            Err(error) => return Err(error),
        };
        match index {
            Some(index) => Ok(Positions::One(index)),
            None if pos.is_instance_of::<pyo3::types::PyRange>() => Ok(Positions::Range {
                start: pos
                    .getattr("start")?
                    .extract::<Option<isize>>()?
                    .unwrap_or(0),
                stop: pos.getattr("stop")?.extract::<isize>()?,
                step: pos
                    .getattr("step")?
                    .extract::<Option<isize>>()?
                    .unwrap_or(1),
            }),
            None => {
                if pos.try_iter().is_err() {
                    return Ok(Positions::NotIndexable);
                }
                Ok(Positions::Many(Self::collect_position_indices(pos)?))
            }
        }
    }

    /// Read one bit position per sequence entry, without range-checking them.
    ///
    /// `get` returns a borrowed reference to the item at an index, valid while
    /// the sequence is held. Reading the items directly avoids pyo3's generic
    /// extraction building its own `Vec`, which costs about a third more.
    fn read_sequence_items(
        py: Python<'_>,
        count: usize,
        get: impl Fn(ffi::Py_ssize_t) -> *mut ffi::PyObject,
    ) -> PyResult<Positions> {
        let read = |position: usize| -> PyResult<isize> {
            let item = get(position as ffi::Py_ssize_t);
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
            Ok(value)
        };
        if count <= INLINE_POSITIONS {
            let mut buf = [0isize; INLINE_POSITIONS];
            for (position, slot) in buf[..count].iter_mut().enumerate() {
                *slot = read(position)?;
            }
            return Ok(Positions::Few { buf, count });
        }
        let mut indices = Vec::with_capacity(count);
        for position in 0..count {
            indices.push(read(position)?);
        }
        Ok(Positions::Many(indices))
    }

    /// Apply already-read positions. Runs no Python.
    pub(crate) fn apply_set_positions(
        &mut self,
        value: bool,
        positions: &Positions,
    ) -> PyResult<()> {
        match positions {
            Positions::All => {
                let len = self.len();
                self.set_from_slice(value, 0, len as isize, 1)
            }
            Positions::One(index) => {
                if value {
                    self.set_index(*index)
                } else {
                    self.unset_index(*index)
                }
            }
            Positions::Few { buf, count } => self.set_from_sequence(value, &buf[..*count]),
            Positions::Many(indices) => self.set_from_sequence(value, indices),
            Positions::Range { start, stop, step } => {
                self.set_from_slice(value, *start, *stop, *step)
            }
            // Neutral wording: this arm is shared by set, unset, set_at and
            // unset_at, so it cannot name one of them.
            Positions::NotIndexable => Err(PyTypeError::new_err(
                "The positions argument must be an integer, an iterable of ints, or a range.",
            )),
            Positions::IndexOverflow => Err(index_overflow_error(self.len())),
        }
    }

    /// `__getitem__` over an already-read key.
    ///
    /// The result objects are built in here rather than outside: creating a
    /// bool or a `Mutibs` touches the C API but never user Python, which is the
    /// line drawn in [`crate::helpers::locking`].
    fn get_key(&self, py: Python<'_>, key: ItemKey<'_>) -> PyResult<Py<PyAny>> {
        let index = match key {
            ItemKey::Failed(error) => {
                return Err(index_conversion_error(py, error, self.data.len()));
            }
            ItemKey::Unusable => {
                return Err(PyTypeError::new_err("Index must be an integer or a slice."));
            }
            ItemKey::Slice(slice) => {
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
                return Ok(Py::new(py, result)?.into_pyobject(py)?.into());
            }
            ItemKey::Index(index) => index,
        };
        let index = validate_index(index, self.data.len())?;
        // SAFETY: validate_index guarantees index < self.data.len(). The bool
        // singleton is chosen with a conditional move rather than a branch; see
        // `Tibs::__getitem__`.
        let value = unsafe { *self.data.as_bitslice().get_unchecked(index) };
        unsafe {
            let obj = std::hint::select_unpredictable(value, ffi::Py_True(), ffi::Py_False());
            ffi::Py_INCREF(obj);
            Ok(Bound::from_owned_ptr(py, obj).unbind())
        }
    }

    /// `__setitem__` over an already-read key and value.
    fn set_key(
        &mut self,
        py: Python<'_>,
        key: ItemKey<'_>,
        bit: Option<bool>,
        bits: Option<Tibs>,
    ) -> PyResult<()> {
        let length = self.len();
        let index = match key {
            ItemKey::Failed(error) => {
                return Err(index_conversion_error(py, error, length));
            }
            ItemKey::Unusable => {
                return Err(PyTypeError::new_err("Index must be an integer or a slice."));
            }
            ItemKey::Slice(slice) => {
                // `bits` is None only when the value is the receiver, in which
                // case the pre-write bits are what should be assigned.
                let tibs = match bits {
                    Some(tibs) => tibs,
                    None => Tibs::from_bv(self.to_bitvec()),
                };
                let indices = slice.indices(length as isize)?;
                let (start, stop, step) = (indices.start, indices.stop, indices.step);

                if step == 1 {
                    debug_assert!(start >= 0);
                    debug_assert!(stop >= 0);
                    self.set_slice(start as usize, stop as usize, &tibs);
                    return Ok(());
                }
                if step == 0 {
                    return Err(PyValueError::new_err(
                        "The step in __setitem__ must not be zero.",
                    ));
                }
                // Target indices in the natural slice order, respecting the sign.
                let mut positions: Vec<usize> = Vec::new();
                let mut i = start;
                if step > 0 {
                    debug_assert!(start >= 0);
                    debug_assert!(stop >= 0);
                    while i < stop {
                        positions.push(validate_index(i, length)?);
                        i += step;
                    }
                } else {
                    // TODO: with a negative step I think start or stop could be -1.
                    while i > stop {
                        positions.push(validate_index(i, length)?);
                        i += step; // step < 0
                    }
                }

                if tibs.len() != positions.len() {
                    return Err(PyValueError::new_err(format!(
                        "Attempt to assign sequence of size {} to extended slice of size {}",
                        tibs.len(),
                        positions.len()
                    )));
                }

                for (k, &pos) in positions.iter().enumerate() {
                    let v = tibs.get_index(k as isize)?;
                    self.as_mut_bitvec_ref().set(pos, v);
                }
                return Ok(());
            }
            ItemKey::Index(index) => index,
        };
        // `bit` is Some for every index key; see `__setitem__`.
        if bit.unwrap_or(false) {
            self.set_index(index)
        } else {
            self.unset_index(index)
        }
    }

    /// `__delitem__` over an already-read key.
    fn delete_key(&mut self, py: Python<'_>, key: ItemKey<'_>) -> PyResult<()> {
        let length = self.len();
        let index = match key {
            ItemKey::Failed(error) => {
                return Err(index_conversion_error(py, error, length));
            }
            ItemKey::Unusable => {
                return Err(PyTypeError::new_err("Index must be an integer or a slice."));
            }
            ItemKey::Slice(slice) => {
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
                    let mut to_remove: Vec<usize> = Vec::new();
                    let mut i = start;
                    if step > 0 {
                        while i < stop {
                            to_remove.push(i as usize);
                            i += step;
                        }
                    } else {
                        while i > stop {
                            to_remove.push(i as usize);
                            i += step; // step < 0
                        }
                    }
                    to_remove.sort();
                    self.delete_positions(&to_remove);
                }
                return Ok(());
            }
            ItemKey::Index(index) => index as i64,
        };
        let index = if index < 0 {
            index + length as i64
        } else {
            index
        };
        if index < 0 || index >= length as i64 {
            return Err(PyIndexError::new_err(format!(
                "Bit index {index} out of range for length {length}"
            )));
        }
        self.delete_slice(index as usize, index as usize + 1);
        Ok(())
    }

    /// A `Tibs` copy as plain Rust, for callers that already hold the object.
    pub(crate) fn tibs_copy(&self) -> Tibs {
        Tibs::from_bv(self.to_bitvec())
    }

    /// The `repr` as plain Rust, for callers that already hold the object.
    pub(crate) fn repr_string(&self) -> String {
        if self.is_empty() {
            "Mutibs()".to_string()
        } else {
            format!("Mutibs('{}')", self.to_string())
        }
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
        // Write back over bytes; `copy_from_bitslice` moves a bit at a time
        // even when, as here, both sides are byte aligned.
        let span = end - start;
        let head = self.storage_head_offset();
        let source_head = swapped.storage_head_offset();
        let source = swapped.as_bitvec_ref().as_raw_slice();
        copy_bits(
            self.data.as_raw_mut_slice(),
            head + start,
            source,
            source_head,
            span,
        );
        Ok(())
    }

    /// Apply already-read positions. Runs no Python.
    pub(crate) fn apply_invert_positions(&mut self, positions: &Positions) -> PyResult<()> {
        let mut flip = |index: isize| -> PyResult<()> {
            let index: usize = validate_index(index, self.len())?;
            let value = self.as_bitvec_ref()[index];
            self.as_mut_bitvec_ref().set(index, !value);
            Ok(())
        };
        match positions {
            Positions::All => {
                *self.as_mut_bitvec_ref() = std::mem::take(&mut *self.as_mut_bitvec_ref()).not();
                Ok(())
            }
            Positions::One(index) => flip(*index),
            Positions::Few { buf, count } => {
                for index in &buf[..*count] {
                    flip(*index)?;
                }
                Ok(())
            }
            Positions::Many(indices) => {
                for index in indices {
                    flip(*index)?;
                }
                Ok(())
            }
            Positions::Range { start, stop, step } => {
                if *step == 0 {
                    return Err(PyValueError::new_err("Step cannot be zero."));
                }
                let mut index = *start;
                while (*step > 0 && index < *stop) || (*step < 0 && index > *stop) {
                    flip(index)?;
                    index += *step;
                }
                Ok(())
            }
            Positions::NotIndexable => Err(PyTypeError::new_err(
                "invert() argument must be an integer, an iterable of ints, or None",
            )),
            Positions::IndexOverflow => Err(index_overflow_error(self.len())),
        }
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
        let mut countdown = count.unwrap_or(i64::MAX);
        if countdown < 0 {
            return Err(PyValueError::new_err(format!(
                "The count in replace() should not be negative. Received {}.",
                countdown
            )));
        }

        if byte_aligned
            && mask.is_none()
            && old.len() == 8
            && new.len() == 8
            && self.storage_head_offset() == 0
        {
            let byte_value = |bits: &Tibs| {
                let (bytes, offset, _) = bits.raw_data_ref();
                if offset == 0 {
                    bytes[0]
                } else {
                    (bytes[0] << offset) | (bytes[1] >> (8 - offset))
                }
            };
            let old_byte = byte_value(&old);
            let new_byte = byte_value(&new);
            let start_byte = search_start.div_ceil(8);
            let end_byte = search_end / 8;
            if start_byte >= end_byte {
                return Ok(0);
            }
            let limit = usize::try_from(countdown).unwrap_or(usize::MAX);
            let target = &mut self.data.as_raw_mut_slice()[start_byte..end_byte];
            let mut replacements = 0;
            let mut current = 0;
            while replacements < limit {
                let Some(found) = memchr::memchr(old_byte, &target[current..]) else {
                    break;
                };
                let position = current + found;
                target[position] = new_byte;
                replacements += 1;
                current = position + 1;
            }
            return Ok(replacements);
        }

        let alignment_mod8 = if byte_aligned { Some(0) } else { None };
        let matcher = mask
            .as_ref()
            .map(|mask| MaskedMatcher::new(old.as_bitslice(), mask.as_bitslice(), false));

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
                    old.as_bitslice(),
                    current_pos,
                    search_end,
                    byte_aligned,
                )?,
            };
            if let Some(found_pos) = found {
                starting_points.push(found_pos);
                current_pos = found_pos + old.len();
                countdown -= 1;
            } else {
                break;
            }
        }

        if starting_points.is_empty() {
            return Ok(0);
        }

        let replacements = starting_points.len();
        let (replacement, replacement_offset, _) = new.raw_data_ref();
        if old.len() == new.len() {
            let target_offset = self.storage_head_offset();
            let target = self.data.as_raw_mut_slice();
            for &pos in &starting_points {
                copy_bits(
                    target,
                    target_offset + pos,
                    replacement,
                    replacement_offset,
                    new.len(),
                );
            }
            return Ok(replacements);
        }

        let retained_bits = self.len() - old.len() * replacements;
        let result_len = new
            .len()
            .checked_mul(replacements)
            .and_then(|inserted| retained_bits.checked_add(inserted))
            .ok_or_else(|| PyOverflowError::new_err("The replacement result is too large."))?;
        let mut result = BitConcat::with_bit_capacity(result_len);
        let source_offset = self.storage_head_offset();
        let source = self.data.as_raw_slice();
        let mut source_pos = 0;
        for &pos in &starting_points {
            let unchanged = pos - source_pos;
            let absolute = source_offset + source_pos;
            result.push_run(&source[absolute / 8..], absolute % 8, unchanged);
            result.push_run(replacement, replacement_offset, new.len());
            source_pos = pos + old.len();
        }
        let tail = self.len() - source_pos;
        let absolute = source_offset + source_pos;
        result.push_run(&source[absolute / 8..], absolute % 8, tail);

        self.data = result.into_bitvec();
        debug_assert_eq!(self.len(), result_len);
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
        let (value_bytes, value_offset, _) = value.raw_data_ref();
        let (mask_bytes, mask_offset, _) = mask.raw_data_ref();
        let bits_offset = self.storage_head_offset();
        let bits_len = self.len();
        deposit_masked_bytes(
            self.data.as_raw_mut_slice(),
            bits_offset,
            value_bytes,
            value_offset,
            value.len(),
            mask_bytes,
            mask_offset,
            bits_len,
        );
        Ok(())
    }

    pub(crate) fn apply_insert_bits(&mut self, mut pos: isize, bs: &Tibs) {
        if bs.is_empty() {
            return;
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
        let (bytes, offset, _) = bs.raw_data_ref();
        self.splice_raw(insert_pos, insert_pos, bytes, offset, bs.len());
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
    #[pyo3(signature = (auto = None, /), text_signature = "(auto=None, /)")]
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
    pub fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        // `cast` rather than `extract::<PyRef<_>>`, which builds and discards a
        // Python exception when the other side is the class not tried first,
        // and which would fold a failed borrow into "not equal".
        if let Ok(other) = other.cast::<Tibs>() {
            // `Tibs` is frozen, so `get` needs no borrow of its own.
            return with_locked(slf, |m| Ok(m.bits_equal(other.get())));
        }
        if let Ok(other) = other.cast::<Mutibs>() {
            // Comparing an object with itself is trivially true, and skips a
            // section that `with_critical_section2` would collapse anyway.
            if other.as_ptr() == slf.as_ptr() {
                return Ok(true);
            }
            // Both operands, together. Locking only the receiver would leave a
            // thread writing to `other` refused by the borrow held here, and
            // locking them in turn would suspend the first.
            return with_locked2(slf, other, |m, other| Ok(m.bits_equal(other)));
        }
        Ok(false)
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    /// Return string representations for printing.
    pub fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        with_locked(slf, |m| Ok(m.to_string()))
    }

    /// Return representation that could be used to recreate the instance.
    pub fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        with_locked(slf, |m| Ok(m.repr_string()))
    }

    /// Return a string formatted according to the Python format mini-language.
    ///
    /// The type codes ``b``, ``o``, ``x`` and ``X`` give the bit representation, and so
    /// keep any leading zeros. They are equivalent to the :attr:`~Mutibs.bin`,
    /// :attr:`~Mutibs.oct` and :attr:`~Mutibs.hex` properties. The type codes ``u`` and
    /// ``i`` give the unsigned and signed integer interpretations, and ``e``, ``f`` and
    /// ``g`` (with their uppercase forms) show the IEEE float value using Python's
    /// scientific, fixed-point and general presentations; a float needs a length of 16,
    /// 32 or 64 bits. All of these read the bits big-endian, the same as the matching
    /// properties. To interpret them in another byte or bit order, format a view such as
    /// ``self.le`` instead.
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
    pub fn __format__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        format_spec: &str,
    ) -> PyResult<String> {
        with_locked(slf, |m| {
            helpers::format_bit_collection(py, m, format_spec, "Mutibs")
        })
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
        slf: &Bound<'_, Self>,
        byte_order: Option<ByteOrder>,
        bit_order: Option<BitOrder>,
    ) -> PyResult<MutableView> {
        let byte_order = byte_order.unwrap_or(ByteOrder::Unspecified);
        let bit_order = bit_order.unwrap_or(BitOrder::Msb0);
        let len = with_locked(slf, |m| Ok(m.len()))?;
        View::validate_layout(len, byte_order, bit_order)?;
        Ok(MutableView::from_mutibs(
            slf.clone().unbind(),
            byte_order,
            bit_order,
        ))
    }

    /// Return a little-endian byte-order view.
    ///
    /// Equivalent to ``view(byte_order=ByteOrder.Little)``.
    ///
    /// The ``Mutibs`` length must be a whole number of bytes.
    ///
    #[getter]
    pub fn le(slf: &Bound<'_, Self>) -> PyResult<MutableView> {
        let len = with_locked(slf, |m| Ok(m.len()))?;
        View::validate_layout(len, ByteOrder::Little, BitOrder::Msb0)?;
        Ok(MutableView::from_mutibs(
            slf.clone().unbind(),
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
    pub fn be(slf: &Bound<'_, Self>) -> PyResult<MutableView> {
        let len = with_locked(slf, |m| Ok(m.len()))?;
        View::validate_layout(len, ByteOrder::Big, BitOrder::Msb0)?;
        Ok(MutableView::from_mutibs(
            slf.clone().unbind(),
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
    pub fn lsb0(slf: &Bound<'_, Self>) -> PyResult<MutableView> {
        let len = with_locked(slf, |m| Ok(m.len()))?;
        View::validate_layout(len, ByteOrder::Unspecified, BitOrder::Lsb0)?;
        Ok(MutableView::from_mutibs(
            slf.clone().unbind(),
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
    // No lock: MSB0 is the default layout, so there is nothing to validate and
    // nothing to read from the source.
    pub fn msb0(slf: &Bound<'_, Self>) -> MutableView {
        MutableView::from_mutibs(slf.clone().unbind(), ByteOrder::Unspecified, BitOrder::Msb0)
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
    #[pyo3(signature = (a, b, /), text_signature = "($self, a, b, /)")]
    pub fn field(slf: &Bound<'_, Self>, py: Python<'_>, a: i64, b: i64) -> PyResult<MutableView> {
        MutableView::from_mutibs(slf.clone().unbind(), ByteOrder::Unspecified, BitOrder::Msb0)
            .field(py, a, b)
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
    pub fn to_bin(
        slf: &Bound<'_, Self>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<String> {
        // A read needs the section too: a shared borrow does not conflict with
        // another reader, but it does lose to a writer.
        with_locked(slf, |m| {
            m.map_slice(start, end, |bits| Ok(BitCollection::to_binary(bits)))
        })
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
    pub fn write_bin(slf: &Bound<'_, Self>, s: &str) -> PyResult<()> {
        let bv = bv_from_bin(s)?;
        with_locked_mut(slf, |m| {
            m.replace_with_bv(bv);
            Ok(())
        })
    }

    /// Property of the binary representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_bin` with no parameters.
    /// Assigning is equivalent to using :meth:`~write_bin` and can change the length.
    ///
    /// :return: The binary representation.
    #[getter]
    fn bin(slf: &Bound<'_, Self>) -> PyResult<String> {
        with_locked(slf, |m| Ok(BitCollection::to_binary(m)))
    }

    #[setter(bin)]
    fn write_bin_property(slf: &Bound<'_, Self>, s: &str) -> PyResult<()> {
        Self::write_bin(slf, s)
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
    pub fn to_oct(
        slf: &Bound<'_, Self>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<String> {
        with_locked(slf, |m| m.map_slice(start, end, BitCollection::to_octal))
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
    pub fn write_oct(slf: &Bound<'_, Self>, s: &str) -> PyResult<()> {
        let bv = bv_from_oct(s)?;
        with_locked_mut(slf, |m| {
            m.replace_with_bv(bv);
            Ok(())
        })
    }

    /// Property of the octal representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_oct` with no parameters.
    /// Assigning is equivalent to using :meth:`~write_oct` and can change the length.
    ///
    /// :return: The octal representation.
    /// :raises ValueError: if the length is not a multiple of 3.
    #[getter]
    fn oct(slf: &Bound<'_, Self>) -> PyResult<String> {
        with_locked(slf, BitCollection::to_octal)
    }

    #[setter(oct)]
    fn write_oct_property(slf: &Bound<'_, Self>, s: &str) -> PyResult<()> {
        Self::write_oct(slf, s)
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
    pub fn to_hex(
        slf: &Bound<'_, Self>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<String> {
        with_locked(slf, |m| {
            m.map_slice(start, end, BitCollection::to_hexadecimal)
        })
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
    pub fn write_hex(slf: &Bound<'_, Self>, s: &str) -> PyResult<()> {
        let bv = bv_from_hex(s)?;
        with_locked_mut(slf, |m| {
            m.replace_with_bv(bv);
            Ok(())
        })
    }

    /// Property of the hexadecimal representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_hex` with no parameters.
    /// Assigning is equivalent to using :meth:`~write_hex` and can change the length.
    ///
    /// :return: The hexadecimal representation.
    /// :raises ValueError: if the length is not a multiple of 4.
    #[getter]
    fn hex(slf: &Bound<'_, Self>) -> PyResult<String> {
        with_locked(slf, BitCollection::to_hexadecimal)
    }

    #[setter(hex)]
    fn write_hex_property(slf: &Bound<'_, Self>, s: &str) -> PyResult<()> {
        Self::write_hex(slf, s)
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
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<PyBytes>> {
        with_locked(slf, |m| {
            m.map_slice(start, end, |bits| BitCollection::to_py_bytes(bits, py))
        })
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
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<PyBytes>> {
        with_locked(slf, |m| {
            m.map_slice(start, end, |bits| {
                BitCollection::to_padded_py_bytes(bits, py)
            })
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
    pub fn write_bytes(slf: &Bound<'_, Self>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        // `bytes_like_to_vec` goes through the buffer protocol, which is Python.
        let bv = bv_from_bytes_slice(bytes_like_to_vec(data)?, None, None)?;
        with_locked_mut(slf, |m| {
            m.replace_with_bv(bv);
            Ok(())
        })
    }

    /// Property of the ``bytes`` representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_bytes` with no parameters.
    /// Assigning is equivalent to using :meth:`~write_bytes` and can change the length.
    ///
    /// :return: The bytes representation.
    /// :raises ValueError: if the length is not a multiple of 8.
    #[getter]
    fn bytes(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        with_locked(slf, |m| BitCollection::to_py_bytes(m, py))
    }

    #[setter(bytes)]
    fn write_bytes_property(slf: &Bound<'_, Self>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::write_bytes(slf, data)
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
    pub fn to_raw_data(slf: &Bound<'_, Self>) -> PyResult<(Vec<u8>, usize, usize)> {
        with_locked(slf, |m| Ok(m.raw_data()))
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
    pub fn as_raw_data(slf: &Bound<'_, Self>) -> PyResult<(Vec<u8>, usize, usize)> {
        with_locked_mut(slf, |m| {
            let offset = m.storage_head_offset();
            let len = m.len();
            let bv = std::mem::take(&mut *m.as_mut_bitvec_ref());
            Ok((bv.into_vec(), offset, len))
        })
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
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        with_locked(slf, |m| {
            m.map_slice(start, end, |bits| BitCollection::to_uint(bits, py, false))
        })
    }

    /// Write the current bits from an unsigned integer without changing the length.
    ///
    /// :param int u: An unsigned integer.
    /// :return: None
    ///
    /// :raises ValueError: if the current length is zero.
    /// :raises ValueError: if the integer doesn't fit in the current length.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs.from_zeros(8)
    ///     >>> m.write_u(15)
    ///     >>> m
    ///     Mutibs('0x0f')
    ///
    #[pyo3(signature = (u, /), text_signature = "($self, u, /)")]
    pub fn write_u(slf: &Bound<'_, Self>, u: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::assign_u(slf, u)
    }

    /// Property of the unsigned integer representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_u` with no parameters. Assigning
    /// is equivalent to using :meth:`~write_u`.
    ///
    /// :return: The value as an unsigned integer.
    #[getter]
    fn u<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Self::to_u(slf, py, None, None)
    }

    #[setter(u)]
    fn write_u_property(slf: &Bound<'_, Self>, u: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::assign_u(slf, u)
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
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        with_locked(slf, |m| {
            m.map_slice(start, end, |bits| BitCollection::to_int(bits, py, false))
        })
    }

    /// Write the current bits from a signed integer without changing the length.
    ///
    /// :param int i: A signed integer.
    /// :return: None
    ///
    /// :raises ValueError: if the current length is zero.
    /// :raises ValueError: if the integer doesn't fit in the current length.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs.from_zeros(4)
    ///     >>> m.write_i(-2)
    ///     >>> m
    ///     Mutibs('0xe')
    ///
    #[pyo3(signature = (i, /), text_signature = "($self, i, /)")]
    pub fn write_i(slf: &Bound<'_, Self>, i: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::assign_i(slf, i)
    }

    /// Property of the signed integer representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_i` with no parameters. Assigning
    /// is equivalent to using :meth:`~write_i`.
    ///
    /// :return: The value as a signed integer.
    #[getter]
    fn i<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Self::to_i(slf, py, None, None)
    }

    #[setter(i)]
    fn write_i_property(slf: &Bound<'_, Self>, i: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::assign_i(slf, i)
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
    pub fn to_f(slf: &Bound<'_, Self>, start: Option<isize>, end: Option<isize>) -> PyResult<f64> {
        with_locked(slf, |m| {
            m.map_slice(start, end, |bits| BitCollection::to_f64(bits, false))
        })
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
    pub fn write_f(slf: &Bound<'_, Self>, f: f64) -> PyResult<()> {
        Self::assign_f(slf, f)
    }

    /// Property of the floating point representation of the Mutibs.
    ///
    /// Reading is equivalent to using :meth:`~to_f` with no parameters. Assigning
    /// is equivalent to using :meth:`~write_f`.
    ///
    /// :return: The value as a Python float.
    #[getter]
    fn f(slf: &Bound<'_, Self>) -> PyResult<f64> {
        Self::to_f(slf, None, None)
    }

    #[setter(f)]
    fn write_f_property(slf: &Bound<'_, Self>, f: f64) -> PyResult<()> {
        Self::assign_f(slf, f)
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
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<PyList>> {
        with_locked(slf, |m| {
            let (start, end) = validate_slice(m.len(), start, end)?;
            helpers::bitslice_to_bool_list(py, &m.as_bitslice()[start..end])
        })
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
        seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let bv = bv_from_random(length, secure, seed)?;
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
            Some(offset) => Some(validate_offset(offset)?),
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
    pub fn __len__(slf: &Bound<'_, Self>) -> PyResult<usize> {
        with_locked(slf, |m| Ok(m.len()))
    }

    /// Whether the Mutibs has any bits.
    pub fn __bool__(slf: &Bound<'_, Self>) -> PyResult<bool> {
        with_locked(slf, |m| Ok(!m.as_bitvec_ref().is_empty()))
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
    #[pyo3(signature = (dtype, /, start = None, end = None), text_signature = "($self, dtype, /, start=None, end=None)")]
    pub fn to_values(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        // `extract_dtype` parses a spec string, so it is Python and belongs
        // outside the lock.
        let dtype = extract_dtype(dtype)?;
        // Validate against the whole container, then copy out only the
        // selected bits. Snapshotting through `to_tibs` first would copy every
        // bit on every call, making a loop of windowed reads over a large
        // `Mutibs` quadratic.
        let window = with_locked(slf, |m| {
            let (start, end) = validate_slice(m.len(), start, end)?;
            Ok(m.window(start, end - start))
        })?;
        py_values_from_range(py, &window, &dtype, None, None)
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
    #[pyo3(signature = (dtype, /, start = None, end = None), text_signature = "($self, dtype, /, start=None, end=None)")]
    pub fn to_value(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<PyAny>> {
        let dtype = extract_dtype(dtype)?;
        // Only the selected bits are copied; see the note in `to_values`.
        let value = with_locked(slf, |m| {
            let (start, end) = validate_slice(m.len(), start, end)?;
            Ok(m.window(start, end - start))
        })?;
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
    pub fn __getitem__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let key = read_item_key(key);
        with_locked(slf, |m| m.get_key(py, key))
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
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let key = read_item_key(key);
        // The value is read as a bit for an index key and promoted to bits for
        // a slice key, both Python, so both happen before the lock. A value
        // that *is* the receiver is snapshotted inside instead.
        let value_is_self = value.as_ptr() == slf.as_ptr();
        let bits = match (&key, value_is_self) {
            (ItemKey::Slice(_), false) => Some(Tibs::extract(value.as_borrowed())?),
            _ => None,
        };
        let bit = match key {
            ItemKey::Index(_) => Some(value.is_truthy()?),
            _ => None,
        };
        with_locked_mut(slf, |m: &mut Self| m.set_key(py, key, bit, bits))
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
    pub fn __delitem__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let key = read_item_key(key);
        with_locked_mut(slf, |m: &mut Self| m.delete_key(py, key))
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
    #[pyo3(signature = (prefix, /), text_signature = "($self, prefix, /)")]
    pub fn starts_with(slf: &Bound<'_, Self>, prefix: Tibs) -> PyResult<bool> {
        with_locked(slf, |m| {
            Ok(<Mutibs as BitCollection>::starts_with(m, prefix))
        })
    }

    /// Return True if b is a sub-sequence of self.
    pub fn __contains__(slf: &Bound<'_, Self>, py: Python<'_>, b: Tibs) -> PyResult<bool> {
        Self::find(slf, py, b, None, None, false, None).map(|found| found.is_some())
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
    #[pyo3(signature = (suffix, /), text_signature = "($self, suffix, /)")]
    pub fn ends_with(slf: &Bound<'_, Self>, suffix: Tibs) -> PyResult<bool> {
        with_locked(slf, |m| Ok(<Mutibs as BitCollection>::ends_with(m, suffix)))
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
    #[pyo3(signature = (needle, /, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, needle, /, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn find(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Option<usize>> {
        with_locked(slf, |m| {
            find_in_bits(
                py,
                m.as_bitslice(),
                &needle,
                SearchParams {
                    start,
                    end,
                    byte_aligned,
                    mask,
                },
                false,
            )
        })
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
    #[pyo3(signature = (needle, /, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, needle, /, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn find_all(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Vec<u64>> {
        with_locked(slf, |m| {
            find_all_in_bits(
                py,
                m.as_bitslice(),
                &needle,
                SearchParams {
                    start,
                    end,
                    byte_aligned,
                    mask,
                },
            )
        })
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
    #[pyo3(signature = (chunk_size, /, count = None), text_signature = "($self, chunk_size, /, count=None)")]
    pub fn chunks(
        slf: &Bound<'_, Self>,
        chunk_size: i64,
        count: Option<i64>,
    ) -> PyResult<Vec<Self>> {
        with_locked(slf, |m| BitCollection::collect_chunks(m, chunk_size, count))
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
    pub fn split_at(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        pos: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        // Positions first, since reading them is Python; the split itself then
        // runs under the lock with no copy of the source.
        let positions = read_split_positions(pos)?;
        let pieces = with_locked(slf, |m| m.split_at_positions(&positions))?;
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
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn count_and(slf: &Bound<'_, Self>, other: Tibs) -> PyResult<usize> {
        with_locked(slf, |m| {
            validate_logical_op_lengths(m.len(), other.len())?;
            Ok(m.pairwise_count(&other, LogicalOp::And))
        })
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
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn count_or(slf: &Bound<'_, Self>, other: Tibs) -> PyResult<usize> {
        with_locked(slf, |m| {
            validate_logical_op_lengths(m.len(), other.len())?;
            Ok(m.pairwise_count(&other, LogicalOp::Or))
        })
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
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn count_xor(slf: &Bound<'_, Self>, other: Tibs) -> PyResult<usize> {
        with_locked(slf, |m| {
            validate_logical_op_lengths(m.len(), other.len())?;
            Ok(m.pairwise_count(&other, LogicalOp::Xor))
        })
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
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn count_andnot(slf: &Bound<'_, Self>, other: Tibs) -> PyResult<usize> {
        with_locked(slf, |m| {
            validate_logical_op_lengths(m.len(), other.len())?;
            Ok(m.pairwise_count(&other, LogicalOp::AndNot))
        })
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
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn intersects(slf: &Bound<'_, Self>, other: Tibs) -> PyResult<bool> {
        with_locked(slf, |m| {
            validate_logical_op_lengths(m.len(), other.len())?;
            Ok(m.pairwise_any(&other, LogicalOp::And))
        })
    }

    /// Return whether no bit is set in both this Mutibs and another.
    ///
    /// The negation of :meth:`intersects`. Equivalent to ``not (self & other).any()``,
    /// but stops at the first bit set in both instead of building the
    /// intermediate object.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: ``True`` if no position is set in both, otherwise ``False``.
    /// :raises ValueError: if the two lengths differ.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b1100').is_disjoint('0b0011')
    ///     True
    ///     >>> Mutibs('0b1100').is_disjoint('0b1010')
    ///     False
    ///
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn is_disjoint(slf: &Bound<'_, Self>, other: Tibs) -> PyResult<bool> {
        with_locked(slf, |m| {
            validate_logical_op_lengths(m.len(), other.len())?;
            Ok(!m.pairwise_any(&other, LogicalOp::And))
        })
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
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn is_subset_of(slf: &Bound<'_, Self>, other: Tibs) -> PyResult<bool> {
        with_locked(slf, |m| {
            validate_logical_op_lengths(m.len(), other.len())?;
            Ok(!m.pairwise_any(&other, LogicalOp::AndNot))
        })
    }

    /// Return whether every bit set in another is also set in this Mutibs.
    ///
    /// The mirror of :meth:`is_subset_of`. Equivalent to ``(self & other) == other``,
    /// but stops at the first bit set there and not here.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: ``True`` if every position set in the other is set here, otherwise ``False``.
    /// :raises ValueError: if the two lengths differ.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0b1010').is_superset_of('0b1000')
    ///     True
    ///     >>> Mutibs('0b1010').is_superset_of('0b1100')
    ///     False
    ///
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn is_superset_of(slf: &Bound<'_, Self>, other: Tibs) -> PyResult<bool> {
        with_locked(slf, |m| {
            validate_logical_op_lengths(m.len(), other.len())?;
            // `and not` with the operands the other way round: the first bit
            // present in `other` and missing here ends the walk.
            Ok(!other.pairwise_any(m, LogicalOp::AndNot))
        })
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
    ///     >>> Mutibs('0b11010110').extracted('0b10110000')
    ///     Mutibs('0b101')
    ///
    // Named `extract_field` because `Mutibs::extract` is the FromPyObject
    // promotion method used throughout the crate.
    #[pyo3(name = "extracted", signature = (mask, /), text_signature = "($self, mask, /)")]
    pub fn extract_field(slf: &Bound<'_, Self>, mask: Tibs) -> PyResult<Self> {
        with_locked(slf, |m| {
            validate_logical_op_lengths(m.len(), mask.len())?;
            Ok(Self::from_bv(m.extract_masked(&mask)))
        })
    }

    /// Write a scattered bit field into the Mutibs in place.
    ///
    /// This is the inverse of :meth:`extracted`, and writes a scattered field the
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
    #[pyo3(signature = (value, mask, /), text_signature = "($self, value, mask, /)")]
    pub fn deposit(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>, mask: Tibs) -> PyResult<()> {
        // Snapshot value first, both to avoid re-borrowing self when value is
        // self, and so a self-deposit reads the pre-write bits.
        if value.as_ptr() == slf.as_ptr() {
            return with_locked_mut(slf, |m| {
                let value = Tibs::from_bv(m.to_bitvec());
                m.apply_deposit(&value, &mask)
            });
        }
        let value = Tibs::extract(value.as_borrowed())?;
        with_locked_mut(slf, |m| m.apply_deposit(&value, &mask))
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
    #[pyo3(signature = (value, mask, /), text_signature = "($self, value, mask, /)")]
    pub fn deposited(
        slf: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        mask: Tibs,
    ) -> PyResult<Self> {
        let value = Tibs::extract(value.as_borrowed())?;
        with_locked(slf, |m| {
            let mut out = m.clone();
            out.apply_deposit(&value, &mask)?;
            Ok(out)
        })
    }

    /// Bit-wise 'and' between two Mutibs. Returns new Mutibs.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: A new Mutibs.
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __and__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        with_locked(slf, |m| {
            validate_logical_op_lengths(m.len(), other.len())?;
            Ok(m.logical_op(&other, LogicalOp::And))
        })
    }

    /// Bit-wise 'or' between two Mutibs. Returns new Mutibs.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: A new Mutibs.
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __or__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        with_locked(slf, |m| {
            validate_logical_op_lengths(m.len(), other.len())?;
            Ok(m.logical_op(&other, LogicalOp::Or))
        })
    }

    /// Bit-wise 'xor' between two Mutibs. Returns new Mutibs.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: A new Mutibs.
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __xor__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        with_locked(slf, |m| {
            validate_logical_op_lengths(m.len(), other.len())?;
            Ok(m.logical_op(&other, LogicalOp::Xor))
        })
    }

    /// Reverse bit-wise 'and' between two Mutibs. Returns new Mutibs.
    ///
    /// This method is used when the RHS is a Mutibs and the LHS is not, but can be converted to one.
    ///
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __rand__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::__and__(slf, other)
    }

    /// Reverse bit-wise 'or' between two Mutibs. Returns new Mutibs.
    ///
    /// This method is used when the RHS is a Mutibs and the LHS is not, but can be converted to one.
    ///
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __ror__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::__or__(slf, other)
    }

    /// Reverse bit-wise 'xor' between two Mutibs. Returns new Mutibs.
    ///
    /// This method is used when the RHS is a Mutibs and the LHS is not, but can be converted to one.
    ///
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __rxor__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::__xor__(slf, other)
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
    #[pyo3(signature = (n, /, start=None, end=None), text_signature = "($self, n, /, start=None, end=None)")]
    pub fn rotate_left(
        slf: &Bound<'_, Self>,
        n: i64,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<()> {
        with_locked_mut(slf, |m| m.apply_rotation(n, start, end, true))
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
    #[pyo3(signature = (n, /, start=None, end=None), text_signature = "($self, n, /, start=None, end=None)")]
    pub fn rotate_right(
        slf: &Bound<'_, Self>,
        n: i64,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<()> {
        with_locked_mut(slf, |m| m.apply_rotation(n, start, end, false))
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
    #[pyo3(signature = (n, /, start=None, end=None), text_signature = "($self, n, /, start=None, end=None)")]
    pub fn rotated_left(
        slf: &Bound<'_, Self>,
        n: i64,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Self> {
        with_locked(slf, |m| {
            let mut out = m.clone();
            out.apply_rotation(n, start, end, true)?;
            Ok(out)
        })
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
    #[pyo3(signature = (n, /, start=None, end=None), text_signature = "($self, n, /, start=None, end=None)")]
    pub fn rotated_right(
        slf: &Bound<'_, Self>,
        n: i64,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Self> {
        with_locked(slf, |m| {
            let mut out = m.clone();
            out.apply_rotation(n, start, end, false)?;
            Ok(out)
        })
    }

    /// Create a Mutibs by decoding bytes created via `encode()`.
    ///
    /// :param bytes | bytearray b: The encoded bytes to decode.
    /// :return: A new Mutibs.
    /// :raises tibs.DecodeError: for badly formed, truncated or extended input bytes.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.decode(Mutibs('0b101').encode())
    ///     Mutibs('0b101')
    ///
    #[classmethod]
    #[pyo3(signature = (b, /), text_signature = "(cls, b, /)")]
    pub fn decode(_cls: &Bound<'_, PyType>, b: &Bound<'_, PyAny>) -> PyResult<Self> {
        tibs_codec::decode_bytes::<Mutibs>(b.py(), bytes_like_to_vec(b)?)
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
    pub fn encode(slf: &Bound<'_, Self>, codec: Option<Codec>) -> PyResult<Vec<u8>> {
        with_locked(slf, |m| tibs_codec::encode(m, codec))
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
    #[pyo3(signature = (pos, /), text_signature = "($self, pos, /)")]
    pub fn set(slf: &Bound<'_, Self>, pos: &Bound<'_, PyAny>) -> PyResult<()> {
        let positions = Self::read_positions(Some(pos))?;
        with_locked_mut(slf, |m| m.apply_set_positions(true, &positions))
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
    #[pyo3(signature = (pos, /), text_signature = "($self, pos, /)")]
    pub fn unset(slf: &Bound<'_, Self>, pos: &Bound<'_, PyAny>) -> PyResult<()> {
        let positions = Self::read_positions(Some(pos))?;
        with_locked_mut(slf, |m| m.apply_set_positions(false, &positions))
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
    #[pyo3(signature = (pos, /), text_signature = "($self, pos, /)")]
    pub fn set_at(slf: &Bound<'_, Self>, pos: &Bound<'_, PyAny>) -> PyResult<Self> {
        let positions = Self::read_positions(Some(pos))?;
        with_locked(slf, |m| {
            let mut out = m.clone();
            out.apply_set_positions(true, &positions)?;
            Ok(out)
        })
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
    #[pyo3(signature = (pos, /), text_signature = "($self, pos, /)")]
    pub fn unset_at(slf: &Bound<'_, Self>, pos: &Bound<'_, PyAny>) -> PyResult<Self> {
        let positions = Self::read_positions(Some(pos))?;
        with_locked(slf, |m| {
            let mut out = m.clone();
            out.apply_set_positions(false, &positions)?;
            Ok(out)
        })
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
    /// When ``value`` is a multi-bit pattern, overlapping occurrences are all counted,
    /// just as they are by :meth:`find_all`. ``byte_aligned`` also applies to single-bit
    /// counts: ``count(1, byte_aligned=True)`` counts the set bits that land on a byte
    /// boundary.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0xef').count()
    ///     7
    ///     >>> Mutibs('0xef').count(1, 0, 4)
    ///     3
    ///     >>> Mutibs('0xff00ff').count([1, 1, 1])  # overlapping
    ///     12
    ///     >>> Mutibs('0x80ff00').count(1, byte_aligned=True)
    ///     2
    ///
    #[pyo3(signature = (value=None, /, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, value=None, /, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn count(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        value: Option<&Bound<'_, PyAny>>,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<usize> {
        // Resolved before the lock: this is the conversion that made `count`
        // impossible to wrap until `count_in_bits` was split.
        let target = resolve_count_target(value)?;
        with_locked(slf, |m| {
            count_in_bits(
                py,
                m.as_bitslice(),
                &target,
                SearchParams {
                    start,
                    end,
                    byte_aligned,
                    mask,
                },
            )
        })
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
    pub fn all(slf: &Bound<'_, Self>) -> PyResult<bool> {
        with_locked(slf, |m| Ok(<Self as BitCollection>::all_set(m)))
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
    pub fn any(slf: &Bound<'_, Self>) -> PyResult<bool> {
        with_locked(slf, |m| Ok(<Self as BitCollection>::any_set(m)))
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
    #[pyo3(signature = (needle, /, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, needle, /, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn rfind(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Option<usize>> {
        with_locked(slf, |m| {
            find_in_bits(
                py,
                m.as_bitslice(),
                &needle,
                SearchParams {
                    start,
                    end,
                    byte_aligned,
                    mask,
                },
                true,
            )
        })
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
    #[pyo3(signature = (pos = None, /), text_signature = "($self, pos=None, /)")]
    pub fn invert(slf: &Bound<'_, Self>, pos: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let positions = Self::read_positions(pos)?;
        with_locked_mut(slf, |m| m.apply_invert_positions(&positions))
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
    #[pyo3(signature = (pos = None, /), text_signature = "($self, pos=None, /)")]
    pub fn inverted(slf: &Bound<'_, Self>, pos: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let positions = Self::read_positions(pos)?;
        with_locked(slf, |m| {
            let mut out = m.clone();
            out.apply_invert_positions(&positions)?;
            Ok(out)
        })
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
    pub fn reverse(slf: &Bound<'_, Self>) -> PyResult<()> {
        with_locked_mut(slf, |m| {
            helpers::reverse_bitvec_in_place(m.as_mut_bitvec_ref());
            Ok(())
        })
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
    pub fn reversed(slf: &Bound<'_, Self>) -> PyResult<Self> {
        with_locked(slf, |m| Ok(BitCollection::reverse_copy(m)))
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
        slf: &Bound<'_, Self>,
        byte_length: Option<i64>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<()> {
        with_locked_mut(slf, |m| m.apply_byte_swap(byte_length, start, end))
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
        slf: &Bound<'_, Self>,
        byte_length: Option<i64>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Mutibs> {
        with_locked(slf, |m| {
            let mut out = m.clone();
            out.apply_byte_swap(byte_length, start, end)?;
            Ok(out)
        })
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
    pub fn __invert__(slf: &Bound<'_, Self>) -> PyResult<Self> {
        with_locked(slf, |m| Ok(BitCollection::invert_copy(m)))
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
    pub fn __lshift__(slf: &Bound<'_, Self>, n: i64) -> PyResult<Self> {
        with_locked(slf, |m| {
            let shift = validate_shift(m, n)?;
            Ok(m.lshift(shift))
        })
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
    pub fn __rshift__(slf: &Bound<'_, Self>, n: i64) -> PyResult<Self> {
        with_locked(slf, |m| {
            let shift = validate_shift(m, n)?;
            Ok(m.rshift(shift))
        })
    }

    /// Return a new copy of the Mutibs for the copy module.
    pub fn __copy__(slf: &Bound<'_, Self>) -> PyResult<Self> {
        with_locked(slf, |m| Ok(Mutibs::from_bv(m.to_bitvec())))
    }

    /// Return the callable and arguments that recreate the Mutibs.
    ///
    /// Used by :mod:`pickle` and by :func:`copy.deepcopy`. The Mutibs that comes
    /// back is a new object, independent of the one that was pickled or copied.
    ///
    /// :return: A tuple of :meth:`Mutibs.decode` and the encoded bytes to pass to it.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> import pickle
    ///     >>> pickle.loads(pickle.dumps(Mutibs('0b110101')))
    ///     Mutibs('0b110101')
    ///
    pub fn __reduce__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<(Py<PyAny>, (Py<PyBytes>,))> {
        // Codec::Raw rather than the Codec::Auto default of `encode`: pickling
        // and deep copying should cost about what copying costs, and Auto
        // measures the alternative codecs and compresses on every call.
        let encoded = PyBytes::new(
            py,
            &with_locked(slf, |m| tibs_codec::encode(m, Some(Codec::Raw)))?,
        );
        let decode = py.get_type::<Self>().getattr("decode")?;
        Ok((decode.unbind(), (encoded.unbind(),)))
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
    pub fn to_tibs(slf: &Bound<'_, Self>) -> PyResult<Tibs> {
        with_locked(slf, |m| Ok(Tibs::from_bv(m.to_bitvec())))
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
    pub fn as_tibs(slf: &Bound<'_, Self>) -> PyResult<Tibs> {
        with_locked_mut(slf, |m| {
            let mut data = std::mem::take(&mut *m.as_mut_bitvec_ref());
            data.shrink_to_fit();
            Ok(Tibs::from_bv(data))
        })
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
    pub fn clear(slf: &Bound<'_, Self>) -> PyResult<()> {
        with_locked_mut(slf, |m| {
            m.as_mut_bitvec_ref().clear();
            Ok(())
        })
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
    pub fn capacity(slf: &Bound<'_, Self>) -> PyResult<usize> {
        with_locked(slf, |m| Ok(m.as_bitvec_ref().capacity()))
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
    #[pyo3(signature = (additional, /), text_signature = "($self, additional, /)")]
    pub fn reserve(slf: &Bound<'_, Self>, additional: usize) -> PyResult<()> {
        with_locked_mut(slf, |m| {
            m.as_mut_bitvec_ref().reserve(additional);
            Ok(())
        })
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
    pub fn __add__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // We accept the PyAny and convert manually here because if we instead
        // accept a Tibs, then correct types with wrong values (e.g. a malformed string)
        // will fail and return a TypeError instead of ValueError which we can't control.
        let other = Tibs::extract(other.as_borrowed())?;
        with_locked(slf, |m| {
            Ok(Mutibs::from_bv(concatenate_bitcollections(m, &other)))
        })
    }

    /// Concatenate Mutibs and return a new Mutibs.
    ///
    /// :param object other: The bits to prepend. This can be anything promotable to ``Tibs``.
    /// :return: A new Mutibs.
    ///
    pub fn __radd__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        with_locked(slf, |m| {
            Ok(Mutibs::from_bv(concatenate_bitcollections(&other, m)))
        })
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
    pub fn __iadd__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::extend(slf, other)
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
    #[pyo3(signature = (bit, /), text_signature = "($self, bit, /)")]
    pub fn append(slf: &Bound<'_, Self>, bit: &Bound<'_, PyAny>) -> PyResult<()> {
        // The argument is resolved to a plain `bool` first. Doing it inside the
        // closure would run Python (`__bool__`, `__index__`) with the critical
        // section held, which suspends it. See `helpers::locking`.
        let Some(b) = helpers::convert_to_bool(bit) else {
            return Err(PyTypeError::new_err(
                "Only True, False, 0 or 1 can be appended.",
            ));
        };
        with_locked_mut(slf, |m| {
            m.as_mut_bitvec_ref().push(b);
            Ok(())
        })
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
    pub fn pop<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<pyo3::Borrowed<'py, 'py, PyBool>> {
        let bit = with_locked_mut(slf, |m| {
            m.as_mut_bitvec_ref()
                .pop()
                .ok_or_else(|| PyIndexError::new_err("pop from empty Mutibs."))
        })?;
        // `PyBool::new` is a borrowed singleton, so building it outside the
        // closure costs nothing and keeps the section free of Python calls.
        Ok(PyBool::new(py, bit))
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
    pub fn extend(slf: &Bound<'_, Self>, bs: &Bound<'_, PyAny>) -> PyResult<()> {
        // Self-extension reads and writes the one object, so it stays entirely
        // inside a single critical section.
        if bs.as_ptr() == slf.as_ptr() {
            return with_locked_mut(slf, |m| {
                let bits_clone = m.to_bitvec();
                m.append_run(bits_clone.as_raw_slice(), 0, bits_clone.len());
                Ok(())
            });
        }
        // Everything that can run Python is done first: the type checks here
        // and, in the fallback, the whole iterator protocol inside
        // `promote_to_bv`. Only settled Rust data crosses into the closure.
        // A `Tibs` or `Mutibs` operand is borrowed rather than materialised,
        // which is why this is not simply `promote_to_bv` for every input. A
        // `Tibs` is frozen, so only the receiver needs locking; a `Mutibs`
        // operand needs its own section, below.
        if let Ok(tibs) = bs.extract::<PyRef<Tibs>>() {
            return with_locked_mut(slf, |m| {
                m.append_collection(&*tibs);
                Ok(())
            });
        }
        if let Ok(mutibs) = bs.cast::<Mutibs>() {
            // Both objects, together: the operand is read in place rather than
            // copied, so its section has to be held too. Self-extension was
            // handled above, so the two are known to be different here.
            return with_locked_mut2(slf, mutibs, |m, other| {
                m.append_collection(other);
                Ok(())
            });
        }
        let bits = promote_to_bv(bs)?;
        with_locked_mut(slf, |m| {
            if m.is_empty() {
                // For an empty receiver, move the promoted BitVec into place
                // rather than copying it into another allocation.
                *m.as_mut_bitvec_ref() = bits;
            } else {
                m.append_run(
                    bits.as_raw_slice(),
                    head_bit_offset(bits.as_bitslice()),
                    bits.len(),
                );
            }
            Ok(())
        })
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
    pub fn extend_left(slf: &Bound<'_, Self>, bs: &Bound<'_, PyAny>) -> PyResult<()> {
        // Prepending has to build a new buffer either way, so both arms lay
        // the two runs down over bytes rather than growing a BitVec by bits.
        if bs.as_ptr() == slf.as_ptr() {
            return with_locked_mut(slf, |m| {
                let doubled = concatenate_bitcollections(&*m, &*m);
                *m.as_mut_bitvec_ref() = doubled;
                Ok(())
            });
        }
        let to_prepend = Tibs::extract(bs.as_borrowed())?;
        if to_prepend.is_empty() {
            return Ok(());
        }
        with_locked_mut(slf, |m| {
            let joined = concatenate_bitcollections(&to_prepend, &*m);
            *m.as_mut_bitvec_ref() = joined;
            Ok(())
        })
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
    #[pyo3(signature = (old, new, /, start=None, end=None, count=None, byte_aligned=false, mask=None), text_signature = "($self, old, new, /, start=None, end=None, count=None, byte_aligned=False, mask=None)")]
    pub fn replace(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        old: &Bound<'_, PyAny>,
        new: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
        count: Option<i64>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<usize> {
        // Both operands are promoted before the section is entered. A self
        // operand is snapshotted inside it instead, so that it reads the
        // pre-write bits and does not need a second borrow.
        let old_is_self = old.as_ptr() == slf.as_ptr();
        let new_is_self = new.as_ptr() == slf.as_ptr();
        let old_outer = if old_is_self {
            None
        } else {
            Some(Tibs::extract(old.as_borrowed())?)
        };
        let new_outer = if new_is_self {
            None
        } else {
            Some(Tibs::extract(new.as_borrowed())?)
        };
        with_locked_mut(slf, |m: &mut Self| {
            let old = old_outer.unwrap_or_else(|| Tibs::from_bv(m.to_bitvec()));
            if old.is_empty() {
                return Err(PyValueError::new_err("No bits were provided to replace."));
            }
            let new = new_outer.unwrap_or_else(|| Tibs::from_bv(m.to_bitvec()));
            m.apply_replace_bits(py, old, new, start, end, count, byte_aligned, mask)
        })
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
    #[pyo3(signature = (old, new, /, start=None, end=None, count=None, byte_aligned=false, mask=None), text_signature = "($self, old, new, /, start=None, end=None, count=None, byte_aligned=False, mask=None)")]
    pub fn replaced(
        slf: &Bound<'_, Self>,
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
        with_locked(slf, |m| {
            let mut out = m.clone();
            let _ = out.apply_replace_bits(py, old, new, start, end, count, byte_aligned, mask)?;
            Ok(out)
        })
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
    pub fn insert(slf: &Bound<'_, Self>, pos: isize, bs: &Bound<'_, PyAny>) -> PyResult<()> {
        // Self-insertion reads and writes the one object, so it is snapshotted
        // inside the section rather than promoted outside it.
        if bs.as_ptr() == slf.as_ptr() {
            return with_locked_mut(slf, |m| {
                let bs = Tibs::from_bv(m.to_bitvec());
                m.apply_insert_bits(pos, &bs);
                Ok(())
            });
        }
        // `Tibs::extract` promotes an arbitrary object, so it runs first.
        let bs = Tibs::extract(bs.as_borrowed())?;
        with_locked_mut(slf, |m| {
            m.apply_insert_bits(pos, &bs);
            Ok(())
        })
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
    pub fn inserted(slf: &Bound<'_, Self>, pos: isize, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bs = Tibs::extract(bs.as_borrowed())?;
        with_locked(slf, |m| {
            let mut out = m.clone();
            out.apply_insert_bits(pos, &bs);
            Ok(out)
        })
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
    pub fn __ilshift__(slf: &Bound<'_, Self>, n: i64) -> PyResult<()> {
        with_locked_mut(slf, |m| {
            // Shifting by the whole length or more zero-fills, matching __lshift__.
            let shift = validate_shift(&*m, n)?.min(m.len());
            m.shift_in_place(shift, true);
            Ok(())
        })
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
    pub fn __irshift__(slf: &Bound<'_, Self>, n: i64) -> PyResult<()> {
        with_locked_mut(slf, |m| {
            // Shifting by the whole length or more zero-fills, matching __rshift__.
            let shift = validate_shift(&*m, n)?.min(m.len());
            m.shift_in_place(shift, false);
            Ok(())
        })
    }

    /// Return the Mutibs as a bytes object.
    ///
    /// :return: The bytes representation.
    /// :raises ValueError: if the length is not a multiple of 8.
    pub fn __bytes__(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        with_locked(slf, |m| BitCollection::to_py_bytes(m, py))
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
    pub fn __mul__(slf: &Bound<'_, Self>, n: i64) -> PyResult<Self> {
        if n < 0 {
            return Err(PyValueError::new_err(
                "Cannot multiply by a negative integer.",
            ));
        }
        with_locked(slf, |m| Ok(m.multiply(n as usize)))
    }

    /// Return Mutibs consisting of n concatenations of self.
    ///
    /// Called for expressions of the form 'a = 3*b'.
    ///
    /// :param int n: The number of concatenations. Must be >= 0.
    /// :return: A new Mutibs.
    /// :raises ValueError: if n < 0.
    ///
    pub fn __rmul__(slf: &Bound<'_, Self>, n: i64) -> PyResult<Self> {
        Self::__mul__(slf, n)
    }

    /// In-place bit-wise 'and'.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: None
    /// :raises ValueError: if the two bit sequences have differing lengths.
    ///
    pub fn __iand__(slf: &Bound<'_, Self>, other: Tibs) -> PyResult<()> {
        with_locked_mut(slf, |m| m.iand(&other))
    }

    /// In-place bit-wise 'or'.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: None
    /// :raises ValueError: if the two bit sequences have differing lengths.
    ///
    pub fn __ior__(slf: &Bound<'_, Self>, other: Tibs) -> PyResult<()> {
        with_locked_mut(slf, |m| m.ior(&other))
    }

    /// In-place bit-wise 'xor'.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: None
    /// :raises ValueError: if the two bit sequences have differing lengths.
    ///
    pub fn __ixor__(slf: &Bound<'_, Self>, other: Tibs) -> PyResult<()> {
        with_locked_mut(slf, |m| m.ixor(&other))
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
    pub fn __imul__(slf: &Bound<'_, Self>, n: i64) -> PyResult<()> {
        with_locked_mut(slf, |m| match n {
            i if i < 0 => Err(PyValueError::new_err(
                "Cannot multiply by a negative integer.",
            )),
            0 => {
                m.data.clear();
                Ok(())
            }
            1 => Ok(()),
            i => {
                let repeated = repeat_bitcollection(&*m, i as usize);
                m.data = repeated;
                Ok(())
            }
        })
    }

    // Supply some more helpful errors for things which aren't supported for Mutibs, but are for Tibs.
    // `&Bound` rather than `&self`: these only ever raise, so taking the borrow
    // at all would let a concurrent writer turn an AttributeError into
    // `RuntimeError: Already borrowed`.
    pub fn __iter__(_slf: &Bound<'_, Self>) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "'Mutibs' objects are not iterable. You can use '.to_tibs()' or '.as_tibs()' to convert to a 'Tibs' object that does support iteration.",
        ))
    }

    pub fn __getattr__(_slf: &Bound<'_, Self>, name: String) -> PyResult<()> {
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
