use crate::codec as tibs_codec;
use crate::core::{BitCollection, concatenate_bitcollections, read_split_positions};
use crate::dtype::{Dtype, DtypeRepr, RecordField, RecordLayout, SingleDtype, extract_dtype};
use crate::enums::{BitOrder, ByteOrder, Codec, DtypeKind};
use crate::helpers;
use crate::helpers::{
    BS, BV, LogicalOp, bv_from_bin, bv_from_bools, bv_from_bytes_slice, bv_from_f64, bv_from_hex,
    bv_from_int, bv_from_oct, bv_from_ones, bv_from_random, bv_from_uint, bv_from_zeros,
    bytes_like_to_vec, find_bitvec_aligned, promote_to_bv, rfind_bitvec_aligned, str_to_bv,
    validate_index, validate_length, validate_logical_op_lengths, validate_offset, validate_shift,
    validate_slice, with_locked,
};
use crate::iterator::{BoolIterator, ChunksIterator, FindAllIterator, ValuesIterator};
use crate::mutibs::Mutibs;
use crate::view::View;
use bitvec::field::BitField;
use half::{bf16, f16};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyBufferError, PyIndexError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyInt, PyList, PySlice, PyTuple, PyType};
use std::collections::hash_map::DefaultHasher;
use std::ffi::{c_int, c_void};
use std::hash::{Hash, Hasher};
use std::ptr;
use std::sync::Arc;

/// Check a search mask against the length of the bits being searched for.
///
/// Returns `None` when there is no mask, or when every bit of it is set and so
/// the faster unmasked search paths give the same answer.
pub(crate) fn prepare_mask(mask: Option<Tibs>, needle_len: usize) -> PyResult<Option<Tibs>> {
    let Some(mask) = mask else {
        return Ok(None);
    };
    if mask.len() != needle_len {
        return Err(PyValueError::new_err(format!(
            "The mask length of {} does not match the length of the bits to find ({needle_len}).",
            mask.len()
        )));
    }
    Ok(if mask.all() { None } else { Some(mask) })
}

pub(crate) struct SearchParams {
    pub(crate) start: Option<isize>,
    pub(crate) end: Option<isize>,
    pub(crate) byte_aligned: bool,
    pub(crate) mask: Option<Tibs>,
}

pub(crate) fn find_in_bits(
    py: Python<'_>,
    haystack: &BS,
    needle: &Tibs,
    params: SearchParams,
    reverse: bool,
) -> PyResult<Option<usize>> {
    if needle.is_empty() {
        return Err(PyValueError::new_err("No bits were provided to find."));
    }
    let mask = prepare_mask(params.mask, needle.len())?;
    let (start, end) = validate_slice(haystack.len(), params.start, params.end)?;
    let alignment_mod8 = if params.byte_aligned { Some(0) } else { None };

    match (&mask, reverse) {
        (Some(mask), _) => helpers::find_bitvec_masked_aligned(
            py,
            haystack,
            needle.as_bitslice(),
            mask.as_bitslice(),
            start,
            end,
            alignment_mod8,
            reverse,
        ),
        (None, false) => find_bitvec_aligned(
            py,
            haystack,
            needle.as_bitslice(),
            start,
            end,
            alignment_mod8,
        ),
        (None, true) => rfind_bitvec_aligned(
            py,
            haystack,
            needle.as_bitslice(),
            start,
            end,
            alignment_mod8,
        ),
    }
}

pub(crate) fn find_all_in_bits(
    py: Python<'_>,
    haystack: &BS,
    needle: &Tibs,
    params: SearchParams,
) -> PyResult<Vec<u64>> {
    if needle.is_empty() {
        return Err(PyValueError::new_err("No bits were provided to find."));
    }
    let mask = prepare_mask(params.mask, needle.len())?;
    let (start, end) = validate_slice(haystack.len(), params.start, params.end)?;

    match mask {
        Some(mask) => helpers::collect_find_all_positions_masked(
            py,
            haystack,
            needle.as_bitslice(),
            mask.as_bitslice(),
            start,
            end,
            params.byte_aligned,
        ),
        None => helpers::collect_find_all_positions(
            py,
            haystack,
            needle.as_bitslice(),
            start,
            end,
            params.byte_aligned,
        ),
    }
}

/// What `count` was asked to count, with every Python conversion already done.
pub(crate) enum CountTarget {
    /// A single bit value. This is also the no-value case, which counts ones.
    Bit(bool),
    /// A pattern to count non-overlapping occurrences of.
    Pattern(Tibs),
}

/// Turn `count`'s `value` argument into a [`CountTarget`].
///
/// This is separated from [`count_in_bits`] so that a `Mutibs` count can do its
/// Python work - `__bool__`, `__index__`, promotion to `Tibs` - *before* taking
/// the critical section that the count itself runs under. Nothing here may move
/// into `count_in_bits` without breaking that. See [`crate::helpers::locking`].
pub(crate) fn resolve_count_target(value: Option<&Bound<'_, PyAny>>) -> PyResult<CountTarget> {
    let Some(value) = value else {
        return Ok(CountTarget::Bit(true));
    };
    if let Some(bit) = helpers::convert_to_bool(value) {
        return Ok(CountTarget::Bit(bit));
    }
    match Tibs::extract(value.as_borrowed()) {
        Ok(pattern) => Ok(CountTarget::Pattern(pattern)),
        Err(_) if value.is_instance_of::<PyInt>() => Err(PyValueError::new_err(
            "Cannot convert value to 0, 1 or a Tibs",
        )),
        Err(error) => Err(error),
    }
}

/// Count over already-resolved arguments. Runs no Python beyond signal checks.
pub(crate) fn count_in_bits(
    py: Python<'_>,
    haystack: &BS,
    target: &CountTarget,
    params: SearchParams,
) -> PyResult<usize> {
    let (start, end) = validate_slice(haystack.len(), params.start, params.end)?;

    match target {
        // A mask over a single bit leaves nothing to compare, so every
        // candidate position matches and only the positions are counted.
        CountTarget::Bit(bit) => Ok(match prepare_mask(params.mask, 1)? {
            Some(_) => helpers::count_candidate_positions(start, end, params.byte_aligned),
            None => helpers::count_single_bit(haystack, *bit, start, end, params.byte_aligned),
        }),
        CountTarget::Pattern(pattern) => match prepare_mask(params.mask, pattern.len())? {
            Some(mask) => helpers::count_bitvec_masked(
                py,
                haystack,
                pattern.as_bitslice(),
                mask.as_bitslice(),
                start,
                end,
                params.byte_aligned,
            ),
            None => helpers::count_bitvec(
                py,
                haystack,
                pattern.as_bitslice(),
                start,
                end,
                params.byte_aligned,
            ),
        },
    }
}

impl Hash for Tibs {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);

        // The left aligned bytes with the trailing padding cleared, which two
        // equal runs share however their storage happens to be offset - the
        // same normalisation `PartialEq` compares over. Handing the hasher the
        // whole run in one call replaces a walk that assembled it 64 bits at a
        // time through `load_be` and fed the result out a word at a time.
        state.write(&self.padded_byte_data_cow());
    }
}

// ---- Tibs private helper methods. Not part of the Python interface. ----

impl Tibs {
    #[inline]
    pub(crate) fn from_bv(bv: BV) -> Self {
        let length = bv.len();
        if length <= helpers::FAST_INT_BITS {
            let offset = helpers::head_bit_offset(bv.as_bitslice());
            let mut bytes = [0u8; helpers::FAST_INT_BITS / 8];
            let byte_length = length.div_ceil(8);
            if offset == 0 {
                bytes[..byte_length].copy_from_slice(&bv.as_raw_slice()[..byte_length]);
                helpers::mask_padding_bits(&mut bytes[..byte_length], length);
            } else {
                helpers::copy_unaligned_padded_bytes(
                    bv.as_raw_slice(),
                    offset,
                    length,
                    &mut bytes[..byte_length],
                );
            }
            return Self::from_inline_bytes(bytes, length);
        }
        Tibs {
            data: TibsData::Shared(Arc::new(bv)),
            offset: 0,
            length,
        }
    }

    #[inline]
    pub(crate) fn from_inline_bytes(
        bytes: [u8; helpers::FAST_INT_BITS / 8],
        length: usize,
    ) -> Self {
        debug_assert!(length <= helpers::FAST_INT_BITS);
        Tibs {
            data: TibsData::Inline(bytes),
            offset: 0,
            length,
        }
    }

    #[inline]
    fn small_mask(length: usize) -> u64 {
        if length == 0 {
            0
        } else {
            u64::MAX << (helpers::FAST_INT_BITS - length)
        }
    }

    #[inline]
    fn from_padded_word(word: u64, length: usize) -> Self {
        debug_assert!(length <= helpers::FAST_INT_BITS);
        Self::from_inline_bytes((word & Self::small_mask(length)).to_be_bytes(), length)
    }

    /// Return a short run left-aligned in a word, with padding cleared.
    fn padded_word(&self) -> Option<u64> {
        if self.len() > helpers::FAST_INT_BITS {
            return None;
        }
        if self.offset == 0
            && let TibsData::Inline(bytes) = &self.data
        {
            return Some(u64::from_be_bytes(*bytes) & Self::small_mask(self.len()));
        }
        let (bytes, offset, _) = self.raw_data_ref();
        let mut padded = [0u8; helpers::FAST_INT_BITS / 8];
        let byte_length = self.len().div_ceil(8);
        if offset == 0 {
            padded[..byte_length].copy_from_slice(bytes);
            helpers::mask_padding_bits(&mut padded[..byte_length], self.len());
        } else {
            helpers::copy_unaligned_padded_bytes(
                bytes,
                offset,
                self.len(),
                &mut padded[..byte_length],
            );
        }
        Some(u64::from_be_bytes(padded))
    }

    fn shifted_copy(&self, shift: usize, left: bool) -> Self {
        let Some(word) = self.padded_word() else {
            return if left {
                self.lshift(shift)
            } else {
                self.rshift(shift)
            };
        };
        if shift == 0 {
            return self.clone();
        }
        let shifted = if shift >= self.len() {
            0
        } else if left {
            word << shift
        } else {
            word >> shift
        };
        Self::from_padded_word(shifted, self.len())
    }

    fn inverted_copy(&self) -> Self {
        match self.padded_word() {
            Some(word) => Self::from_padded_word(!word, self.len()),
            None => BitCollection::invert_copy(self),
        }
    }

    fn concatenated(&self, other: &Self) -> Self {
        let length = self.len() + other.len();
        if length <= helpers::FAST_INT_BITS {
            let left = self.padded_word().expect("length bounds both operands");
            let right = other.padded_word().expect("length bounds both operands");
            return Self::from_padded_word(left | (right >> self.len()), length);
        }
        Self::from_bv(concatenate_bitcollections(self, other))
    }

    fn repeated(&self, count: usize) -> Self {
        let length = self.len();
        if length == 0 || count == 0 {
            return Self::from_inline_bytes([0; helpers::FAST_INT_BITS / 8], 0);
        }
        if count <= helpers::FAST_INT_BITS / length {
            let word = self
                .padded_word()
                .expect("result length is at most 64 bits");
            let mut repeated = 0;
            for index in 0..count {
                repeated |= word >> (index * length);
            }
            return Self::from_padded_word(repeated, length * count);
        }
        self.multiply(count)
    }

    pub(crate) fn get_slice_unchecked(&self, offset: usize, length: usize) -> Self {
        match &self.data {
            TibsData::Shared(data) => Tibs {
                data: TibsData::Shared(data.clone()),
                offset: self.offset + offset,
                length,
            },
            TibsData::Inline(bytes) => Tibs {
                data: TibsData::Inline(*bytes),
                offset: self.offset + offset,
                length,
            },
        }
    }

    #[inline]
    fn shares_view_with(&self, other: &Self) -> bool {
        ptr::eq(self, other)
            || matches!(
                (&self.data, &other.data),
                (TibsData::Shared(left), TibsData::Shared(right))
                    if Arc::ptr_eq(left, right)
                        && self.offset == other.offset
                        && self.length == other.length
            )
    }

    #[inline]
    fn logical_op_from_python(&self, other: &Bound<'_, PyAny>, op: LogicalOp) -> PyResult<Self> {
        if let Ok(other) = other.cast::<Tibs>() {
            return self.logical_op_with_tibs(other.get(), op);
        }
        if let Ok(other) = other.cast::<Mutibs>() {
            // See `__eq__`: the operand is locked, `self` is frozen.
            return with_locked(other, |other| {
                validate_logical_op_lengths(self.len(), other.len())?;
                Ok(self.logical_op(other, op))
            });
        }
        let other = Tibs::extract(other.as_borrowed())?;
        self.logical_op_with_tibs(&other, op)
    }

    #[inline]
    fn logical_op_with_tibs(&self, other: &Self, op: LogicalOp) -> PyResult<Self> {
        if self.shares_view_with(other) {
            return Ok(match op {
                LogicalOp::And | LogicalOp::Or => self.clone(),
                LogicalOp::Xor | LogicalOp::AndNot => Self::from_bv(bv_from_zeros(self.len())),
            });
        }
        validate_logical_op_lengths(self.len(), other.len())?;
        if let (Some(left), Some(right)) = (self.padded_word(), other.padded_word()) {
            return Ok(Self::from_padded_word(op.word(left, right), self.len()));
        }
        Ok(self.logical_op(other, op))
    }

    #[inline]
    pub(crate) fn as_bitslice(&self) -> &BS {
        match &self.data {
            TibsData::Shared(data) => &data[self.offset..self.offset + self.length],
            TibsData::Inline(bytes) => {
                &BS::from_slice(bytes)[self.offset..self.offset + self.length]
            }
        }
    }

    /// Fast single-bit read that bypasses bitvec's per-access pointer decoding.
    ///
    /// SAFETY: `index` must be less than `self.length`.
    #[inline(always)]
    pub(crate) unsafe fn bit_at_unchecked(&self, index: usize) -> bool {
        // The backing BitVec's storage may not start at bit 0 of its first
        // byte (see raw_data_ref), so include its head offset.
        let (bytes, offset, _) = self.raw_data_ref();
        let abs = offset + index;
        let byte = unsafe { *bytes.get_unchecked(abs >> 3) };
        // Msb0 ordering: semantic bit i within a byte is physical bit (7 - i).
        (byte >> (7 - (abs & 7))) & 1 != 0
    }

    #[inline]
    pub(crate) fn to_bitvec(&self) -> BV {
        let mut result = BV::from_vec(<Self as BitCollection>::to_padded_byte_data(self));
        result.truncate(self.length);
        result
    }

    #[inline]
    pub(crate) fn raw_data_ref(&self) -> (&[u8], usize, usize) {
        match &self.data {
            TibsData::Shared(data) => {
                let physical_start = helpers::head_bit_offset(data.as_bitslice()) + self.offset;
                let byte_start = physical_start / 8;
                let bit_offset = physical_start % 8;
                let byte_len = (bit_offset + self.length).div_ceil(8);
                (
                    &data.as_raw_slice()[byte_start..byte_start + byte_len],
                    bit_offset,
                    self.length,
                )
            }
            TibsData::Inline(bytes) => {
                let byte_start = self.offset / 8;
                let bit_offset = self.offset % 8;
                let byte_len = (bit_offset + self.length).div_ceil(8);
                (
                    &bytes[byte_start..byte_start + byte_len],
                    bit_offset,
                    self.length,
                )
            }
        }
    }

    fn copy_with_mutation(&self, f: impl FnOnce(&mut Mutibs) -> PyResult<()>) -> PyResult<Self> {
        let mut out = self.to_mutibs();
        f(&mut out)?;
        Ok(out.tibs_copy())
    }
}

///     An immutable container of binary data.
///
///     The constructor is a convenient way to delegate to the ``from_string``,
///     ``from_bytes`` or ``from_bools`` builder methods, depending on the type of ``auto``.
///
///     * ``Tibs('0x13')`` - Equivalent to ``Tibs.from_string('0x13')``.
///     * ``Tibs([1, 0])`` - Equivalent to ``Tibs.from_bools([1, 0])``.
///     * ``Tibs(b'hello')`` - Equivalent to ``Tibs.from_bytes(b'hello')``.
///
///     Otherwise, to construct use a builder 'from' method:
///
///     * ``Tibs.from_bin(s)`` - Create from a binary string, optionally starting with '0b'.
///     * ``Tibs.from_oct(s)`` - Create from an octal string, optionally starting with '0o'.
///     * ``Tibs.from_hex(s)`` - Create from a hex string, optionally starting with '0x'.
///     * ``Tibs.from_u(u, length, [byte_order])`` - Create from an unsigned int to a given length.
///     * ``Tibs.from_i(i, length, [byte_order])`` - Create from a signed int to a given length.
///     * ``Tibs.from_f(f, length, [byte_order])`` - Create from an IEEE float to a 16, 32 or 64 bit length.
///     * ``Tibs.from_bytes(b)`` - Create directly from a ``bytes``, ``bytearray`` or ``memoryview`` object.
///     * ``Tibs.from_string(s)`` - Use a formatted string.
///     * ``Tibs.from_bools(iterable)`` - Convert each element in ``iterable`` to a bool.
///     * ``Tibs.from_zeros(length)`` - Initialise with ``length`` ``0`` bits.
///     * ``Tibs.from_ones(length)`` - Initialise with ``length`` ``1`` bits.
///     * ``Tibs.from_random(length, [secure, seed])`` - Initialise with ``length`` randomly set bits.
///     * ``Tibs.from_joined(iterable)`` - Concatenate an iterable of objects.
///
/// Small values live in the Python object itself, avoiding the separate
/// `Arc` and `Vec` allocations that otherwise dominate scalar operations.
/// Larger values retain shared storage so slicing them remains constant-time.
#[derive(Clone)]
enum TibsData {
    Shared(Arc<BV>),
    Inline([u8; helpers::FAST_INT_BITS / 8]),
}

#[derive(Clone)]
#[pyclass(frozen, sequence, skip_from_py_object, module = "tibs")]
pub struct Tibs {
    data: TibsData,
    offset: usize,
    length: usize,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Tibs {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(tibs_ref) = obj.extract::<PyRef<Tibs>>() {
            return Ok(tibs_ref.clone());
        }
        if let Ok(mutibs_ref) = obj.extract::<PyRef<Mutibs>>() {
            return Ok(mutibs_ref.tibs_copy());
        }
        let bv = promote_to_bv(&obj)?;
        Ok(Tibs::from_bv(bv))
    }
}

/// The narrow numeric encoding selected by a public dtype kind.
///
/// This mapping lives at the dtype/Python boundary rather than in `helpers` so
/// the pure codec does not depend on the public PyO3 enum.
#[inline]
fn narrow_float_format(kind: DtypeKind) -> Option<helpers::NarrowFloatFormat> {
    Some(match kind {
        DtypeKind::Binary8P3 => helpers::NarrowFloatFormat::Binary8P3,
        DtypeKind::Binary8P4 => helpers::NarrowFloatFormat::Binary8P4,
        DtypeKind::OcpE4M3Saturate => helpers::NarrowFloatFormat::OcpE4M3Saturate,
        DtypeKind::OcpE4M3Overflow => helpers::NarrowFloatFormat::OcpE4M3Overflow,
        DtypeKind::OcpE5M2Saturate => helpers::NarrowFloatFormat::OcpE5M2Saturate,
        DtypeKind::OcpE5M2Overflow => helpers::NarrowFloatFormat::OcpE5M2Overflow,
        DtypeKind::OcpE3M2 => helpers::NarrowFloatFormat::OcpE3M2,
        DtypeKind::OcpE2M3 => helpers::NarrowFloatFormat::OcpE2M3,
        DtypeKind::OcpE2M1 => helpers::NarrowFloatFormat::OcpE2M1,
        DtypeKind::OcpE8M0 => helpers::NarrowFloatFormat::OcpE8M0,
        DtypeKind::OcpInt8 => helpers::NarrowFloatFormat::OcpInt8,
        _ => return None,
    })
}

fn narrow_float_error(
    value: f64,
    format: helpers::NarrowFloatFormat,
    error: helpers::NarrowFloatEncodeError,
) -> PyErr {
    PyValueError::new_err(format!(
        "Cannot encode {value:?} as '{}': {error}.",
        format.name()
    ))
}

#[inline]
fn encode_narrow_float(value: f64, format: helpers::NarrowFloatFormat) -> PyResult<u8> {
    helpers::encode_narrow_float(value, format)
        .map_err(|error| narrow_float_error(value, format, error))
}

fn bv_from_narrow_float(value: f64, format: helpers::NarrowFloatFormat) -> PyResult<BV> {
    let length = format.bit_length();
    let raw = encode_narrow_float(value, format)?;
    let mut bv = BV::repeat(false, length);
    bv.store_be(raw);
    Ok(bv)
}

fn bv_from_single_value(dtype: &SingleDtype, value: &Bound<'_, PyAny>) -> PyResult<BV> {
    if let Some(format) = narrow_float_format(dtype.kind) {
        debug_assert_eq!(dtype.length, format.bit_length());
        return bv_from_narrow_float(value.extract::<f64>()?, format);
    }
    match dtype.kind {
        DtypeKind::Float => {
            let is_little_endian = dtype.byte_order == ByteOrder::Little;
            bv_from_f64(value.extract::<f64>()?, dtype.length, is_little_endian)
        }
        DtypeKind::BFloat => {
            let is_little_endian = dtype.byte_order == ByteOrder::Little;
            Ok(helpers::bv_from_bf16(
                value.extract::<f64>()?,
                is_little_endian,
            ))
        }
        DtypeKind::Uint => {
            let is_little_endian = dtype.byte_order == ByteOrder::Little;
            bv_from_uint(value, dtype.length, is_little_endian)
        }
        DtypeKind::Int => {
            let is_little_endian = dtype.byte_order == ByteOrder::Little;
            bv_from_int(value, dtype.length, is_little_endian)
        }
        DtypeKind::Bool => match helpers::convert_to_bool(value) {
            Some(bit) => {
                let mut bv = BV::with_capacity(1);
                bv.push(bit);
                Ok(bv)
            }
            None => Err(PyTypeError::new_err(
                "bool dtype values must be True, False, 0 or 1.",
            )),
        },
        DtypeKind::Bits => validate_dtype_value_length(dtype, value.extract::<Tibs>()?.to_bitvec()),
        DtypeKind::Bytes => validate_dtype_value_length(
            dtype,
            bv_from_bytes_slice(bytes_like_to_vec(value)?, None, None)?,
        ),
        DtypeKind::Bin => {
            validate_dtype_value_length(dtype, bv_from_bin(&value.extract::<String>()?)?)
        }
        DtypeKind::Oct => {
            validate_dtype_value_length(dtype, bv_from_oct(&value.extract::<String>()?)?)
        }
        DtypeKind::Hex => {
            validate_dtype_value_length(dtype, bv_from_hex(&value.extract::<String>()?)?)
        }
        DtypeKind::Binary8P3
        | DtypeKind::Binary8P4
        | DtypeKind::OcpE4M3Saturate
        | DtypeKind::OcpE4M3Overflow
        | DtypeKind::OcpE5M2Saturate
        | DtypeKind::OcpE5M2Overflow
        | DtypeKind::OcpE3M2
        | DtypeKind::OcpE2M3
        | DtypeKind::OcpE2M1
        | DtypeKind::OcpE8M0
        | DtypeKind::OcpInt8 => unreachable!("narrow numeric dtypes dispatch before the match"),
    }
}

fn validate_dtype_value_length(dtype: &SingleDtype, bv: BV) -> PyResult<BV> {
    let value_length = bv.len();
    if value_length != dtype.length {
        return Err(PyValueError::new_err(format!(
            "Dtype length is {} bits, but {} value produced {} bits.",
            dtype.length,
            dtype.kind.repr_name(),
            value_length
        )));
    }
    Ok(bv)
}

fn add_value_path(py: Python<'_>, error: PyErr, path: &str) -> PyErr {
    let message = format!("At value{path}: {error}");
    if error.value(py).setattr("args", (message,)).is_err() {
        let _ = error
            .value(py)
            .call_method1("add_note", (format!("At value{path}."),));
    }
    error
}

fn add_value_note(py: Python<'_>, error: PyErr, path: &str) -> PyErr {
    let _ = error
        .value(py)
        .call_method1("add_note", (format!("At value{path}."),));
    error
}

fn append_dtype_value(
    py: Python<'_>,
    repr: &DtypeRepr,
    value: &Bound<'_, PyAny>,
    out: &mut BV,
    path: &str,
) -> PyResult<()> {
    match repr {
        DtypeRepr::Single(dtype) => {
            out.extend(
                bv_from_single_value(dtype, value)
                    .map_err(|error| add_value_path(py, error, path))?,
            );
            Ok(())
        }
        DtypeRepr::Array { dtype, count } => append_dtype_items(
            py,
            std::slice::from_ref(dtype.as_ref()),
            Some(*count),
            value,
            out,
            path,
        ),
        DtypeRepr::Tuple(dtypes) => append_dtype_items(py, dtypes, None, value, out, path),
    }
}

fn append_dtype_items(
    py: Python<'_>,
    dtypes: &[DtypeRepr],
    repeat: Option<usize>,
    value: &Bound<'_, PyAny>,
    out: &mut BV,
    path: &str,
) -> PyResult<()> {
    let expected = repeat.unwrap_or(dtypes.len());
    let mut items = value
        .try_iter()
        .map_err(|error| add_value_path(py, error, path))?;
    for index in 0..expected {
        let item = items.next().ok_or_else(|| {
            PyValueError::new_err(format!(
                "At value{path}: expected {expected} items, but received {index}."
            ))
        })??;
        let dtype = if repeat.is_some() {
            &dtypes[0]
        } else {
            &dtypes[index]
        };
        let item_path = format!("{path}[{index}]");
        append_dtype_value(py, dtype, &item, out, &item_path)?;
    }
    if let Some(item) = items.next() {
        item?;
        return Err(PyValueError::new_err(format!(
            "At value{path}: expected exactly {expected} items."
        )));
    }
    Ok(())
}

pub(crate) fn bv_from_value(dtype: &Dtype, value: &Bound<'_, PyAny>) -> PyResult<BV> {
    if let Some(single) = dtype.single() {
        return bv_from_single_value(single, value);
    }
    if let Some(layout) = &dtype.record_layout {
        if let Some(bv) = bv_from_value_record(layout, value)? {
            return Ok(bv);
        }
    }
    let mut out = BV::with_capacity(dtype.length);
    append_dtype_value(value.py(), &dtype.repr, value, &mut out, "")?;
    Ok(out)
}

/// How the byte-wise path in [`bv_from_values_iter`] packs one value.
///
/// Only numeric dtypes that are a whole number of bytes long qualify, so the
/// byte order and the sign are settled once for the whole sequence rather than
/// per value.
#[derive(Clone, Copy)]
enum BytewisePacker {
    Int {
        byte_length: usize,
        is_little_endian: bool,
        signed: bool,
    },
    Float {
        byte_length: usize,
        is_little_endian: bool,
    },
    /// bfloat16, which is always two bytes and so carries no byte length.
    /// Kept apart from `Float` because that variant picks its conversion by
    /// byte length, and two bytes does not distinguish `f16` from `bf16`.
    BFloat { is_little_endian: bool },
    /// A fixed-width narrow format whose complete code point is one byte.
    NarrowFloat { format: helpers::NarrowFloatFormat },
}

impl BytewisePacker {
    /// Decide whether `dtype` qualifies, without touching the values. This has
    /// to be settled up front, because returning `None` after consuming part
    /// of a one-shot iterable would lose those items.
    fn for_parts(kind: DtypeKind, length: usize, byte_order: ByteOrder) -> Option<Self> {
        debug_assert!(length > 0);
        if let Some(format) = narrow_float_format(kind) {
            debug_assert_eq!(length, format.bit_length());
            return (length == 8).then_some(BytewisePacker::NarrowFloat { format });
        }
        if !length.is_multiple_of(8) {
            return None;
        }
        let byte_length = length / 8;
        if !matches!(
            kind,
            DtypeKind::Uint | DtypeKind::Int | DtypeKind::Float | DtypeKind::BFloat
        ) {
            return None;
        }
        let is_little_endian = byte_order == ByteOrder::Little;
        Some(match kind {
            DtypeKind::Float => BytewisePacker::Float {
                byte_length,
                is_little_endian,
            },
            DtypeKind::BFloat => BytewisePacker::BFloat { is_little_endian },
            _ => BytewisePacker::Int {
                byte_length,
                is_little_endian,
                signed: kind == DtypeKind::Int,
            },
        })
    }

    fn for_dtype(dtype: &SingleDtype) -> Option<Self> {
        Self::for_parts(dtype.kind, dtype.length, dtype.byte_order)
    }

    fn byte_length(&self) -> usize {
        match *self {
            BytewisePacker::Int { byte_length, .. } | BytewisePacker::Float { byte_length, .. } => {
                byte_length
            }
            BytewisePacker::BFloat { .. } => 2,
            BytewisePacker::NarrowFloat { .. } => 1,
        }
    }

    /// The type [`push`](Self::push) converts without running any Python code.
    /// See [`for_each_value`].
    fn plain_type(&self) -> *mut ffi::PyTypeObject {
        match *self {
            BytewisePacker::Int { .. } => &raw mut ffi::PyLong_Type,
            BytewisePacker::Float { .. }
            | BytewisePacker::BFloat { .. }
            | BytewisePacker::NarrowFloat { .. } => &raw mut ffi::PyFloat_Type,
        }
    }

    fn push(&self, out: &mut Vec<u8>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match *self {
            BytewisePacker::Int {
                byte_length,
                is_little_endian,
                signed,
            } => helpers::push_int_bytes(out, value, byte_length, is_little_endian, signed),
            BytewisePacker::Float {
                byte_length,
                is_little_endian,
            } => {
                helpers::push_f64_bytes(
                    out,
                    value.extract::<f64>()?,
                    byte_length,
                    is_little_endian,
                );
                Ok(())
            }
            BytewisePacker::BFloat { is_little_endian } => {
                helpers::push_bf16_bytes(out, value.extract::<f64>()?, is_little_endian);
                Ok(())
            }
            BytewisePacker::NarrowFloat { format } => {
                out.push(encode_narrow_float(value.extract::<f64>()?, format)?);
                Ok(())
            }
        }
    }
}

/// Classify every field of a flat tuple record, or bail with `None` on the
/// first field that isn't a whole-byte-length numeric/float dtype (e.g. a
/// sub-byte int, `bool`, or a `bits`/`bytes`/`hex`/`oct`/`bin` field). Settled
/// up front, once per call, the same as [`BytewisePacker::for_dtype`].
fn classify_tuple_packers(fields: &[RecordField]) -> Option<Vec<BytewisePacker>> {
    let mut packers = Vec::with_capacity(fields.len());
    for field in fields {
        match BytewisePacker::for_parts(field.kind, field.length, field.byte_order) {
            Some(packer) => packers.push(packer),
            None => return None,
        }
    }
    Some(packers)
}

/// Pack one record's fields, in order, straight into `bytes` — no `BV`
/// allocated per field, unlike the generic `append_dtype_value`/
/// `append_dtype_items` path this replaces for [`RecordLayout::Tuple`].
fn push_record_fields(
    py: Python<'_>,
    packers: impl ExactSizeIterator<Item = BytewisePacker>,
    record: &Bound<'_, PyAny>,
    path: &str,
    bytes: &mut Vec<u8>,
) -> PyResult<()> {
    let expected = packers.len();
    let mut items = record
        .try_iter()
        .map_err(|error| add_value_path(py, error, path))?;
    for (field_index, packer) in packers.enumerate() {
        let item = items.next().ok_or_else(|| {
            PyValueError::new_err(format!(
                "At value{path}: expected {expected} items, but received {field_index}."
            ))
        })??;
        packer
            .push(bytes, &item)
            .map_err(|error| add_value_path(py, error, &format!("{path}[{field_index}]")))?;
    }
    if let Some(item) = items.next() {
        item?;
        return Err(PyValueError::new_err(format!(
            "At value{path}: expected exactly {expected} items."
        )));
    }
    Ok(())
}

/// Fast path for [`bv_from_value`] when `dtype` has a [`RecordLayout`] and
/// every field qualifies for [`BytewisePacker`]. Returns `None` when some
/// field doesn't (a sub-byte length, or a `bits`/`bytes`/`hex`/`oct`/`bin`
/// kind), so the caller can fall back to `append_dtype_value` unchanged.
fn bv_from_value_record(layout: &RecordLayout, value: &Bound<'_, PyAny>) -> PyResult<Option<BV>> {
    let py = value.py();
    match layout {
        RecordLayout::Tuple(fields) => {
            let Some(packers) = classify_tuple_packers(fields) else {
                return Ok(None);
            };
            let record_byte_length: usize = packers.iter().map(BytewisePacker::byte_length).sum();
            let mut bytes = Vec::with_capacity(record_byte_length);
            push_record_fields(py, packers.into_iter(), value, "", &mut bytes)?;
            Ok(Some(BV::from_vec(bytes)))
        }
        RecordLayout::Array { element, count } => {
            let Some(packer) =
                BytewisePacker::for_parts(element.kind, element.length, element.byte_order)
            else {
                return Ok(None);
            };
            let mut bytes = Vec::with_capacity(packer.byte_length() * count);
            push_record_fields(
                py,
                std::iter::repeat_n(packer, *count),
                value,
                "",
                &mut bytes,
            )?;
            Ok(Some(BV::from_vec(bytes)))
        }
    }
}

/// Fast path for [`bv_from_values_iter`] when `dtype` has a [`RecordLayout`]
/// and every field qualifies for [`BytewisePacker`] — the `struct`-`">hhl"`-
/// style shape. Every record is written straight into one growing `Vec<u8>`,
/// with no `BitAccumulator` and no per-field `BV` involved, mirroring what
/// `BytewisePacker` already does for a homogeneous single-dtype sequence.
/// Returns `None` when some field doesn't qualify, so the caller falls back
/// to the existing generic per-record path unchanged.
fn bv_from_values_iter_record(
    py: Python<'_>,
    layout: &RecordLayout,
    iterable: &Bound<'_, PyAny>,
    hint: Option<usize>,
) -> PyResult<Option<BV>> {
    match layout {
        RecordLayout::Tuple(fields) => {
            let Some(packers) = classify_tuple_packers(fields) else {
                return Ok(None);
            };
            let record_byte_length: usize = packers.iter().map(BytewisePacker::byte_length).sum();
            let capacity = hint.and_then(|len| len.checked_mul(record_byte_length));
            let mut bytes = capacity.map_or_else(Vec::new, Vec::with_capacity);
            let mut record_index = 0usize;
            // `plain` is deliberately a pointer no real object's type can
            // equal, so `for_each_value` always takes its owned (incref'd)
            // branch here: unlike a bare scalar value, extracting a record's
            // *fields* can run arbitrary Python code (e.g. `__index__` on a
            // non-exact-int field), which could otherwise invalidate a
            // borrowed reference to the record itself mid-visit.
            for_each_value(py, iterable, ptr::null_mut(), |record| {
                let path = format!("[{record_index}]");
                push_record_fields(py, packers.iter().copied(), record, &path, &mut bytes)?;
                record_index += 1;
                Ok(())
            })?;
            Ok(Some(BV::from_vec(bytes)))
        }
        RecordLayout::Array { element, count } => {
            let Some(packer) =
                BytewisePacker::for_parts(element.kind, element.length, element.byte_order)
            else {
                return Ok(None);
            };
            let record_byte_length = packer.byte_length() * count;
            let capacity = hint.and_then(|len| len.checked_mul(record_byte_length));
            let mut bytes = capacity.map_or_else(Vec::new, Vec::with_capacity);
            let mut record_index = 0usize;
            for_each_value(py, iterable, ptr::null_mut(), |record| {
                let path = format!("[{record_index}]");
                push_record_fields(
                    py,
                    std::iter::repeat_n(packer, *count),
                    record,
                    &path,
                    &mut bytes,
                )?;
                record_index += 1;
                Ok(())
            })?;
            Ok(Some(BV::from_vec(bytes)))
        }
    }
}

/// How the bit-wise path in [`bv_from_values_iter`] packs one value.
///
/// The counterpart to [`BytewisePacker`] for the dtypes it turns down because
/// their length is not a whole number of bytes, so that values have to be
/// packed end to end across byte boundaries rather than written out whole.
#[derive(Clone, Copy)]
enum BitwisePacker {
    Int { length: usize, signed: bool },
    Bool,
    NarrowFloat { format: helpers::NarrowFloatFormat },
}

impl BitwisePacker {
    /// Decide whether `dtype` qualifies, under the same up-front rule as
    /// [`BytewisePacker::for_dtype`].
    ///
    /// Unlike that one this needs no byte order: a length that isn't a whole
    /// number of bytes cannot carry an explicit byte order, because `Dtype`
    /// rejects the combination when it is built, so these are all big-endian.
    fn for_dtype(dtype: &SingleDtype) -> Option<Self> {
        if let Some(format) = narrow_float_format(dtype.kind) {
            debug_assert_eq!(dtype.length, format.bit_length());
            return (dtype.length < 8).then_some(BitwisePacker::NarrowFloat { format });
        }
        // A bool dtype is one bit long, likewise by construction.
        if dtype.kind == DtypeKind::Bool {
            return Some(BitwisePacker::Bool);
        }
        debug_assert!(dtype.length > 0);
        if dtype.length.is_multiple_of(8) {
            return None;
        }
        match dtype.kind {
            DtypeKind::Uint => Some(BitwisePacker::Int {
                length: dtype.length,
                signed: false,
            }),
            DtypeKind::Int => Some(BitwisePacker::Int {
                length: dtype.length,
                signed: true,
            }),
            _ => None,
        }
    }

    fn length(&self) -> usize {
        match *self {
            BitwisePacker::Int { length, .. } => length,
            BitwisePacker::Bool => 1,
            BitwisePacker::NarrowFloat { format } => format.bit_length(),
        }
    }

    /// The type [`push`](Self::push) converts without running any Python code.
    /// See [`for_each_value`]. A bool dtype takes `True` and `False` too, but
    /// they are not of exactly this type and so go the owned route.
    fn plain_type(&self) -> *mut ffi::PyTypeObject {
        match self {
            BitwisePacker::Int { .. } | BitwisePacker::Bool => &raw mut ffi::PyLong_Type,
            BitwisePacker::NarrowFloat { .. } => &raw mut ffi::PyFloat_Type,
        }
    }

    fn push(&self, out: &mut helpers::BitAccumulator, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match *self {
            BitwisePacker::Int { length, signed } => {
                helpers::push_int_bits(out, value, length, signed)
            }
            BitwisePacker::Bool => match helpers::convert_to_bool(value) {
                Some(bit) => {
                    out.push(u64::from(bit), 1);
                    Ok(())
                }
                None => Err(PyTypeError::new_err(
                    "bool dtype values must be True, False, 0 or 1.",
                )),
            },
            BitwisePacker::NarrowFloat { format } => {
                let raw = encode_narrow_float(value.extract::<f64>()?, format)?;
                out.push(u64::from(raw), format.bit_length());
                Ok(())
            }
        }
    }
}

/// Hand every value of `iterable` to `visit`, checking for interrupts as it
/// goes.
///
/// A `list` or `tuple` is read by index instead of through the iterator
/// protocol, which saves an interpreter round trip per value: under the limited
/// ABI even a reference count change is a call into the interpreter, and that
/// is a large share of the work when converting one value is a handful of
/// instructions. The standard library's own packing pays neither, because it is
/// handed its values as a C array.
///
/// Items read this way are borrowed rather than owned, which is only sound
/// while `visit` cannot run Python code that might drop them. `plain` is the
/// type it converts without running any: an item of exactly that type is
/// borrowed, and anything else is handed over owned.
fn for_each_value(
    py: Python<'_>,
    iterable: &Bound<'_, PyAny>,
    plain: *mut ffi::PyTypeObject,
    mut visit: impl FnMut(&Bound<'_, PyAny>) -> PyResult<()>,
) -> PyResult<()> {
    let mut check_at = helpers::SIGNAL_CHECK_INTERVAL;
    let mut interrupt_check = |index: usize| -> PyResult<()> {
        if index >= check_at {
            py.check_signals()?;
            check_at = index.saturating_add(helpers::SIGNAL_CHECK_INTERVAL);
        }
        Ok(())
    };

    // Exact types only. A subclass can define its own `__iter__`, and walking
    // one by index would not give the sequence that iterating it does.
    let is_type = |ty| unsafe { ffi::Py_IS_TYPE(iterable.as_ptr(), ty) } != 0;
    let item_at = if is_type(&raw mut ffi::PyList_Type) {
        ffi::PyList_GetItem
    } else if is_type(&raw mut ffi::PyTuple_Type) {
        ffi::PyTuple_GetItem
    } else {
        for (index, item) in iterable.try_iter()?.enumerate() {
            interrupt_check(index)?;
            visit(&item?)?;
        }
        return Ok(());
    };

    for index in 0..iterable.len()? {
        interrupt_check(index)?;
        // SAFETY: the index is in bounds of the length read above, and the
        // sequence holds the item for as long as the pointer is used here.
        let item = unsafe { item_at(iterable.as_ptr(), index as ffi::Py_ssize_t) };
        if item.is_null() {
            // A list that has run short can only mean an earlier value was
            // converted by Python code that removed items from it. Stop here,
            // as iterating it would.
            unsafe { ffi::PyErr_Clear() };
            return Ok(());
        }
        if unsafe { ffi::Py_IS_TYPE(item, plain) } != 0 {
            // SAFETY: converting a value of exactly this type runs no Python
            // code, so nothing can drop the item while the borrow is live.
            let borrowed = unsafe { Borrowed::from_ptr(py, item) };
            visit(&borrowed)?;
        } else {
            let owned = unsafe { Bound::from_borrowed_ptr(py, item) };
            visit(&owned)?;
        }
    }
    Ok(())
}

pub(crate) fn bv_from_values_iter(
    py: Python<'_>,
    dtype: &Dtype,
    iterable: &Bound<'_, PyAny>,
) -> PyResult<BV> {
    let hint = iterable.len().ok();

    if let Some(layout) = &dtype.record_layout {
        if let Some(bv) = bv_from_values_iter_record(py, layout, iterable, hint)? {
            return Ok(bv);
        }
    }

    let Some(single) = dtype.single() else {
        let capacity = hint.and_then(|len| len.checked_mul(dtype.length));
        let mut bv = capacity.map_or_else(BV::new, BV::with_capacity);
        let mut check_at = helpers::SIGNAL_CHECK_INTERVAL;
        for (index, item) in iterable.try_iter()?.enumerate() {
            if index >= check_at {
                py.check_signals()?;
                check_at = index.saturating_add(helpers::SIGNAL_CHECK_INTERVAL);
            }
            let item = item?;
            append_dtype_value(py, &dtype.repr, &item, &mut bv, &format!("[{index}]"))?;
        }
        return Ok(bv);
    };
    // A dtype that packs into whole bytes can be built as a byte buffer and
    // converted once at the end. The general path below allocates a `BitVec`
    // per value and appends it a bit at a time, which costs far more than the
    // conversion itself for a long sequence.
    if let Some(packer) = BytewisePacker::for_dtype(single) {
        let byte_length = packer.byte_length();
        let capacity = hint.and_then(|len| len.checked_mul(byte_length));
        let mut bytes = capacity.map_or_else(Vec::new, Vec::with_capacity);
        let mut index = 0;
        for_each_value(py, iterable, packer.plain_type(), |item| {
            packer
                .push(&mut bytes, item)
                .map_err(|error| add_value_note(py, error, &format!("[{index}]")))?;
            index += 1;
            Ok(())
        })?;
        return Ok(BV::from_vec(bytes));
    }

    // The same trick for a dtype that straddles byte boundaries: the values go
    // through a bit accumulator that writes out whole bytes, so this too ends
    // in a single conversion rather than a per-value allocate-and-append.
    if let Some(packer) = BitwisePacker::for_dtype(single) {
        let capacity = hint.and_then(|len| len.checked_mul(packer.length()));
        let mut out = helpers::BitAccumulator::with_bit_capacity(capacity);
        let mut index = 0;
        for_each_value(py, iterable, packer.plain_type(), |item| {
            packer
                .push(&mut out, item)
                .map_err(|error| add_value_note(py, error, &format!("[{index}]")))?;
            index += 1;
            Ok(())
        })?;
        return Ok(out.into_bitvec());
    }

    let capacity = hint.and_then(|len| len.checked_mul(dtype.length));
    let mut bv = capacity.map_or_else(BV::new, BV::with_capacity);
    let mut check_at = helpers::SIGNAL_CHECK_INTERVAL;
    for (index, item) in iterable.try_iter()?.enumerate() {
        if index >= check_at {
            py.check_signals()?;
            check_at = index.saturating_add(helpers::SIGNAL_CHECK_INTERVAL);
        }
        bv.extend(
            bv_from_single_value(single, &item?)
                .map_err(|error| add_value_note(py, error, &format!("[{index}]")))?,
        );
    }
    Ok(bv)
}

/// Which reading a [`BytewiseUnpacker`] gives the raw bits it has loaded.
#[derive(Clone, Copy)]
enum NumericReading {
    Uint,
    Int,
    Float,
    /// bfloat16. `Float` picks its conversion from the byte length, which
    /// cannot tell a two-byte `bf16` from a two-byte `f16`, so the two
    /// readings have to be distinct here rather than share a decoder.
    BFloat,
    NarrowFloat(helpers::NarrowFloatFormat),
}

/// How the byte-wise path in [`py_from_value_parts`] decodes one value.
///
/// The counterpart to [`BytewisePacker`], and for the same reason: a numeric
/// dtype that is a whole number of bytes long and fits in a `u64` can be
/// assembled straight from the backing bytes, so the byte order and the sign
/// are settled once and `bitvec`'s slice-then-load-bit-by-bit drops out. Over a
/// sequence that setup is hoisted out of the loop as well.
#[derive(Clone, Copy)]
struct BytewiseUnpacker {
    reading: NumericReading,
    byte_length: usize,
    is_little_endian: bool,
}

impl BytewiseUnpacker {
    /// Decide whether a dtype qualifies, under the same rule as
    /// [`BytewisePacker::for_dtype`] plus a `u64` ceiling: anything longer
    /// needs the big-integer path that the general route already has.
    fn for_parts(
        dtype_kind: DtypeKind,
        dtype_length: usize,
        byte_order: ByteOrder,
    ) -> Option<Self> {
        debug_assert!(dtype_length > 0);
        if dtype_length > helpers::FAST_INT_BITS || !dtype_length.is_multiple_of(8) {
            return None;
        }
        let byte_length = dtype_length / 8;
        let reading = if let Some(format) = narrow_float_format(dtype_kind) {
            debug_assert_eq!(dtype_length, format.bit_length());
            NumericReading::NarrowFloat(format)
        } else {
            match dtype_kind {
                DtypeKind::Uint => NumericReading::Uint,
                DtypeKind::Int => NumericReading::Int,
                DtypeKind::Float => {
                    debug_assert!(matches!(byte_length, 2 | 4 | 8));
                    NumericReading::Float
                }
                DtypeKind::BFloat => {
                    debug_assert_eq!(byte_length, 2);
                    NumericReading::BFloat
                }
                _ => return None,
            }
        };
        let is_little_endian = byte_order == ByteOrder::Little;
        Some(Self {
            reading,
            byte_length,
            is_little_endian,
        })
    }

    fn for_dtype(dtype: &SingleDtype) -> Option<Self> {
        Self::for_parts(dtype.kind, dtype.length, dtype.byte_order)
    }

    /// The raw bits at absolute byte offset `start`, right-aligned in a
    /// `u64`. [`Self::load`] is the homogeneous-repeated-value spelling of
    /// this (`start = index * self.byte_length`); a [`RecordLayout`] field
    /// needs an arbitrary offset instead, since a heterogeneous record's
    /// fields don't share one stride.
    #[inline]
    fn load_at(&self, bytes: &[u8], shift: u32, start: usize) -> u64 {
        let mut buf = [0u8; 8];
        // A big-endian value goes at the top of the buffer and a little-endian
        // one at the bottom, so that either `from_*_bytes` reads the untouched
        // bytes as leading zeros and leaves the value right-aligned.
        let dest = if self.is_little_endian {
            &mut buf[..self.byte_length]
        } else {
            &mut buf[8 - self.byte_length..]
        };
        if shift == 0 {
            dest.copy_from_slice(&bytes[start..start + self.byte_length]);
        } else {
            for (offset, slot) in dest.iter_mut().enumerate() {
                let position = start + offset;
                // The trailing byte is there whenever the shift needs it,
                // because the span was sized to cover every bit of the range.
                let next = bytes.get(position + 1).copied().unwrap_or(0);
                *slot = (bytes[position] << shift) | (next >> (8 - shift));
            }
        }
        if self.is_little_endian {
            u64::from_le_bytes(buf)
        } else {
            u64::from_be_bytes(buf)
        }
    }

    /// The raw bits of the value at `index`, right-aligned in a `u64`.
    ///
    /// `bytes` holds the values end to end, the first of them starting `shift`
    /// bits into `bytes[0]`. Every value is a whole number of bytes long, so
    /// that shift is the same for all of them.
    #[inline]
    fn load(&self, bytes: &[u8], shift: u32, index: usize) -> u64 {
        self.load_at(bytes, shift, index * self.byte_length)
    }

    #[inline]
    fn value_from_raw<'py>(&self, py: Python<'py>, raw: u64) -> PyResult<Bound<'py, PyAny>> {
        match self.reading {
            // Anything narrower than 64 bits fits an `i64`, and that is the
            // cheaper of the two integer objects to build.
            NumericReading::Uint if self.byte_length == 8 => raw.into_bound_py_any(py),
            NumericReading::Uint => (raw as i64).into_bound_py_any(py),
            NumericReading::Int => {
                let pad = helpers::FAST_INT_BITS - self.byte_length * 8;
                (((raw << pad) as i64) >> pad).into_bound_py_any(py)
            }
            NumericReading::Float => match self.byte_length {
                8 => f64::from_bits(raw).into_bound_py_any(py),
                4 => (f32::from_bits(raw as u32) as f64).into_bound_py_any(py),
                _ => f16::from_bits(raw as u16).to_f64().into_bound_py_any(py),
            },
            NumericReading::BFloat => bf16::from_bits(raw as u16).to_f64().into_bound_py_any(py),
            NumericReading::NarrowFloat(format) => {
                helpers::decode_narrow_float(raw as u8, format).into_bound_py_any(py)
            }
        }
    }

    /// The value at `index` as a Python object.
    #[inline]
    fn value<'py>(
        &self,
        py: Python<'py>,
        bytes: &[u8],
        shift: u32,
        index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.value_from_raw(py, self.load(bytes, shift, index))
    }

    /// The value at absolute byte offset `start` as a Python object — the
    /// [`RecordLayout`] field counterpart of [`Self::value`].
    #[inline]
    fn value_at<'py>(
        &self,
        py: Python<'py>,
        bytes: &[u8],
        shift: u32,
        start: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.value_from_raw(py, self.load_at(bytes, shift, start))
    }
}

/// How a [`SubByteUnpacker`] interprets the raw field it extracts.
#[derive(Clone, Copy)]
enum SubByteReading {
    Int { signed: bool },
    NarrowFloat(helpers::NarrowFloatFormat),
}

/// Unpacker for numeric dtypes narrower than a byte.
///
/// [`BytewiseUnpacker`] needs whole bytes, so `u1`..`u7` and their signed twins
/// fell through to the general route, which rebuilds a `Tibs` window and
/// re-dispatches on the dtype for every value. A field this narrow spans at
/// most two bytes, so one shift and a mask lift it out.
#[derive(Clone, Copy)]
struct SubByteUnpacker {
    length: usize,
    reading: SubByteReading,
}

impl SubByteUnpacker {
    fn for_parts(dtype_kind: DtypeKind, dtype_length: usize) -> Option<Self> {
        debug_assert!(dtype_length > 0);
        if dtype_length >= 8 {
            return None;
        }
        let reading = if let Some(format) = narrow_float_format(dtype_kind) {
            debug_assert_eq!(dtype_length, format.bit_length());
            SubByteReading::NarrowFloat(format)
        } else {
            match dtype_kind {
                DtypeKind::Uint => SubByteReading::Int { signed: false },
                DtypeKind::Int => SubByteReading::Int { signed: true },
                _ => return None,
            }
        };
        Some(Self {
            length: dtype_length,
            reading,
        })
    }

    fn for_dtype(dtype: &SingleDtype) -> Option<Self> {
        Self::for_parts(dtype.kind, dtype.length)
    }

    /// The value at `index` as a Python object.
    ///
    /// `bytes` holds the values end to end, the first starting `shift` bits
    /// into `bytes[0]`.
    #[inline]
    fn value<'py>(
        &self,
        py: Python<'py>,
        bytes: &[u8],
        shift: usize,
        index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let bit = shift + index * self.length;
        let byte = bit >> 3;
        let offset = bit & 7;
        // `Msb0` puts the field's first bit at the top, and the field is short
        // enough that two bytes always cover it.
        let window = ((bytes[byte] as u16) << 8) | bytes.get(byte + 1).copied().unwrap_or(0) as u16;
        let raw = (window >> (16 - offset - self.length)) & ((1u16 << self.length) - 1);
        match self.reading {
            SubByteReading::Int { signed: true } => {
                let pad = 16 - self.length;
                ((((raw << pad) as i16) >> pad) as i64).into_bound_py_any(py)
            }
            SubByteReading::Int { signed: false } => (raw as i64).into_bound_py_any(py),
            SubByteReading::NarrowFloat(format) => {
                helpers::decode_narrow_float(raw as u8, format).into_bound_py_any(py)
            }
        }
    }
}

pub(crate) fn py_from_value_parts(
    py: Python<'_>,
    dtype_kind: DtypeKind,
    dtype_length: usize,
    byte_order: ByteOrder,
    value: &Tibs,
) -> PyResult<Py<PyAny>> {
    if value.len() != dtype_length {
        return Err(PyValueError::new_err(format!(
            "Cannot convert {} bits using a dtype with length {} bits.",
            value.len(),
            dtype_length
        )));
    }

    if let Some(unpacker) = BytewiseUnpacker::for_parts(dtype_kind, dtype_length, byte_order) {
        let (bytes, shift, _) = value.raw_data_ref();
        return Ok(unpacker.value(py, bytes, shift as u32, 0)?.unbind());
    }
    if let Some(unpacker) = SubByteUnpacker::for_parts(dtype_kind, dtype_length) {
        let (bytes, shift, _) = value.raw_data_ref();
        return Ok(unpacker.value(py, bytes, shift, 0)?.unbind());
    }

    match dtype_kind {
        DtypeKind::Float => unreachable!("validated float dtypes unpack bytewise"),
        DtypeKind::BFloat => unreachable!("bf16 is two bytes and unpacks bytewise"),
        DtypeKind::Uint => {
            let is_little_endian = byte_order == ByteOrder::Little;
            Ok(BitCollection::to_uint(value, py, is_little_endian)?.unbind())
        }
        DtypeKind::Int => {
            let is_little_endian = byte_order == ByteOrder::Little;
            Ok(BitCollection::to_int(value, py, is_little_endian)?.unbind())
        }
        DtypeKind::Bool => value.as_bitslice()[0].into_py_any(py),
        DtypeKind::Bits => {
            let py_obj = Py::new(py, value.clone())?.into_pyobject(py)?;
            Ok(py_obj.into())
        }
        DtypeKind::Bytes => BitCollection::to_byte_data(value)?.into_py_any(py),
        DtypeKind::Bin => BitCollection::to_binary(value).into_py_any(py),
        DtypeKind::Oct => BitCollection::to_octal(value)?.into_py_any(py),
        DtypeKind::Hex => BitCollection::to_hexadecimal(value)?.into_py_any(py),
        DtypeKind::Binary8P3
        | DtypeKind::Binary8P4
        | DtypeKind::OcpE4M3Saturate
        | DtypeKind::OcpE4M3Overflow
        | DtypeKind::OcpE5M2Saturate
        | DtypeKind::OcpE5M2Overflow
        | DtypeKind::OcpE3M2
        | DtypeKind::OcpE2M3
        | DtypeKind::OcpE2M1
        | DtypeKind::OcpE8M0
        | DtypeKind::OcpInt8 => unreachable!("narrow numeric dtypes unpack through fast paths"),
    }
}

pub(crate) fn py_from_value(py: Python<'_>, dtype: &Dtype, value: &Tibs) -> PyResult<Py<PyAny>> {
    if let Some(dtype) = dtype.single() {
        return py_from_value_parts(py, dtype.kind, dtype.length, dtype.byte_order, value);
    }
    if value.len() != dtype.length {
        return Err(PyValueError::new_err(format!(
            "Cannot convert {} bits using a dtype with length {} bits.",
            value.len(),
            dtype.length
        )));
    }
    if let Some(layout) = &dtype.record_layout {
        let (bytes, shift, _) = value.raw_data_ref();
        if let Some(mut values) = py_values_from_range_record(py, layout, bytes, shift as u32, 1)? {
            return Ok(values.pop().expect("count = 1 produces exactly one value"));
        }
    }
    py_from_dtype_repr(py, &dtype.repr, value, 0)
}

fn py_from_dtype_repr(
    py: Python<'_>,
    repr: &DtypeRepr,
    value: &Tibs,
    start: usize,
) -> PyResult<Py<PyAny>> {
    match repr {
        DtypeRepr::Single(dtype) => {
            let bits = value.get_slice_unchecked(start, dtype.length);
            py_from_value_parts(py, dtype.kind, dtype.length, dtype.byte_order, &bits)
        }
        DtypeRepr::Array { dtype, count } => {
            let item_length = dtype.length()?;
            let mut values = Vec::with_capacity(*count);
            for index in 0..*count {
                values.push(py_from_dtype_repr(
                    py,
                    dtype,
                    value,
                    start + index * item_length,
                )?);
            }
            Ok(PyTuple::new(py, values)?.into_any().unbind())
        }
        DtypeRepr::Tuple(dtypes) => {
            let mut values = Vec::with_capacity(dtypes.len());
            let mut position = start;
            for dtype in dtypes {
                values.push(py_from_dtype_repr(py, dtype, value, position)?);
                position += dtype.length()?;
            }
            Ok(PyTuple::new(py, values)?.into_any().unbind())
        }
    }
}

/// Fast path for [`py_values_from_range`] (and, with `count = 1`,
/// [`py_from_value`]) when `dtype` has a [`RecordLayout`] and every field
/// qualifies for [`BytewiseUnpacker`]. `bytes`/`shift` are the raw backing
/// store and its bit offset, fetched once by the caller via `raw_data_ref`.
/// Returns `None` when some field doesn't qualify (a sub-byte length, or a
/// `bits`/`bytes`/`hex`/`oct`/`bin` kind), so the caller falls back to the
/// existing `py_from_dtype_repr` recursive walk unchanged.
fn py_values_from_range_record(
    py: Python<'_>,
    layout: &RecordLayout,
    bytes: &[u8],
    shift: u32,
    count: usize,
) -> PyResult<Option<Vec<Py<PyAny>>>> {
    match layout {
        RecordLayout::Tuple(fields) => {
            let mut unpackers = Vec::with_capacity(fields.len());
            for field in fields {
                let Some(unpacker) =
                    BytewiseUnpacker::for_parts(field.kind, field.length, field.byte_order)
                else {
                    return Ok(None);
                };
                // Every field here is a whole number of bytes (a precondition
                // of `BytewiseUnpacker` classification), so a prefix sum of
                // them is too: `bit_offset` divides evenly by 8.
                unpackers.push((unpacker, field.bit_offset / 8));
            }
            let record_byte_length: usize = unpackers.iter().map(|(u, _)| u.byte_length).sum();
            let mut values = Vec::with_capacity(count);
            let mut check_at = helpers::SIGNAL_CHECK_INTERVAL;
            for index in 0..count {
                if index >= check_at {
                    py.check_signals()?;
                    check_at = index.saturating_add(helpers::SIGNAL_CHECK_INTERVAL);
                }
                let record_start = index * record_byte_length;
                let mut fields_out = Vec::with_capacity(unpackers.len());
                for (unpacker, field_byte_offset) in &unpackers {
                    fields_out.push(
                        unpacker
                            .value_at(py, bytes, shift, record_start + field_byte_offset)?
                            .unbind(),
                    );
                }
                values.push(PyTuple::new(py, fields_out)?.into_any().unbind());
            }
            Ok(Some(values))
        }
        RecordLayout::Array {
            element,
            count: element_count,
        } => {
            let Some(unpacker) =
                BytewiseUnpacker::for_parts(element.kind, element.length, element.byte_order)
            else {
                return Ok(None);
            };
            let record_byte_length = unpacker.byte_length * element_count;
            let mut values = Vec::with_capacity(count);
            let mut check_at = helpers::SIGNAL_CHECK_INTERVAL;
            for index in 0..count {
                if index >= check_at {
                    py.check_signals()?;
                    check_at = index.saturating_add(helpers::SIGNAL_CHECK_INTERVAL);
                }
                let record_start = index * record_byte_length;
                let mut fields_out = Vec::with_capacity(*element_count);
                for element_index in 0..*element_count {
                    let field_start = record_start + element_index * unpacker.byte_length;
                    fields_out.push(unpacker.value_at(py, bytes, shift, field_start)?.unbind());
                }
                values.push(PyTuple::new(py, fields_out)?.into_any().unbind());
            }
            Ok(Some(values))
        }
    }
}

pub(crate) fn py_values_from_range(
    py: Python<'_>,
    bits: &Tibs,
    dtype: &Dtype,
    start: Option<isize>,
    end: Option<isize>,
) -> PyResult<Vec<Py<PyAny>>> {
    let (start, end) = validate_slice(bits.len(), start, end)?;
    let selected_len = end - start;
    if !selected_len.is_multiple_of(dtype.length) {
        return Err(PyValueError::new_err(format!(
            "Cannot convert to values - selected length of {selected_len} bits is not a multiple of dtype length {} bits.",
            dtype.length
        )));
    }

    let count = selected_len / dtype.length;
    if count == 0 {
        return Ok(Vec::new());
    }

    if let Some(layout) = &dtype.record_layout {
        let window = bits.get_slice_unchecked(start, selected_len);
        let (bytes, shift, _) = window.raw_data_ref();
        if let Some(values) = py_values_from_range_record(py, layout, bytes, shift as u32, count)? {
            return Ok(values);
        }
    }

    let mut values = Vec::with_capacity(count);
    let mut check_at = helpers::SIGNAL_CHECK_INTERVAL;

    // A dtype that reads out of whole bytes takes them from the backing store
    // directly, with the byte order and the sign settled once for the whole
    // sequence. See `BytewiseUnpacker`.
    if let Some(unpacker) = dtype.single().and_then(BytewiseUnpacker::for_dtype) {
        let window = bits.get_slice_unchecked(start, selected_len);
        let (bytes, shift, _) = window.raw_data_ref();
        let shift = shift as u32;
        for index in 0..count {
            if index >= check_at {
                py.check_signals()?;
                check_at = index.saturating_add(helpers::SIGNAL_CHECK_INTERVAL);
            }
            values.push(unpacker.value(py, bytes, shift, index)?.unbind());
        }
        return Ok(values);
    }

    // Integer dtypes narrower than a byte read out of the same backing store,
    // one shift and mask each.
    if let Some(unpacker) = dtype.single().and_then(SubByteUnpacker::for_dtype) {
        let window = bits.get_slice_unchecked(start, selected_len);
        let (bytes, shift, _) = window.raw_data_ref();
        for index in 0..count {
            if index >= check_at {
                py.check_signals()?;
                check_at = index.saturating_add(helpers::SIGNAL_CHECK_INTERVAL);
            }
            values.push(unpacker.value(py, bytes, shift, index)?.unbind());
        }
        return Ok(values);
    }

    // One window is reused for the whole sequence, with only its offset moving
    // between values. Taking a fresh slice each time would clone the `Arc` once
    // per value for a read that never outlives the loop.
    let mut window = bits.get_slice_unchecked(start, dtype.length);
    let base_offset = window.offset;
    for index in 0..count {
        if index >= check_at {
            py.check_signals()?;
            check_at = index.saturating_add(helpers::SIGNAL_CHECK_INTERVAL);
        }
        window.offset = base_offset + index * dtype.length;
        values.push(py_from_value(py, dtype, &window)?);
    }
    Ok(values)
}

/// Public Python-facing methods.
#[pymethods]
impl Tibs {
    #[new]
    #[pyo3(signature = (auto = None, /), text_signature = "(auto=None, /)")]
    pub fn py_new(auto: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(auto) = auto else {
            return Ok(BitCollection::empty());
        };
        Tibs::extract(auto.as_borrowed())
    }

    /// Return a new instance with the bits reversed.
    ///
    /// :return: Tibs
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Tibs('0b00011')
    ///     >>> a.reversed()
    ///     Tibs('0b11000')
    ///
    fn reversed(&self) -> Self {
        BitCollection::reverse_copy(self)
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
    /// :return: Tibs
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Tibs('0x12345678')
    ///     >>> b = a.byte_swapped(2)
    ///     >>> b
    ///     Tibs('0x34127856')
    ///
    #[pyo3(signature = (byte_length = None, start=None, end=None), text_signature = "($self, byte_length=None, start=None, end=None)")]
    pub fn byte_swapped(
        &self,
        byte_length: Option<i64>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Tibs> {
        self.copy_with_mutation(|out| out.apply_byte_swap(byte_length, start, end))
    }

    /// Return a copy of the raw byte information.
    ///
    /// This returns the underlying byte data and can contain leading and trailing
    /// bits that are not considered part of the object's data. Usually using
    /// :meth:`~to_bytes` is what you really need.
    ///
    /// :return: A tuple of the raw bytes, the bit offset and the bit length.
    ///
    /// .. code-block:: python
    ///
    ///     raw_bytes, offset, length = t.to_raw_data()
    ///     assert t == Tibs.from_bytes(raw_bytes, offset=offset, length=length)
    ///
    pub fn to_raw_data(&self) -> (Vec<u8>, usize, usize) {
        self.raw_data()
    }

    /// Export a read-only buffer (the ``buffer protocol``), for e.g. ``memoryview(t)``.
    ///
    /// This is only possible when the underlying storage starts on a byte
    /// boundary; otherwise a :class:`BufferError` is raised, in which case
    /// :meth:`~to_bytes` or :meth:`~to_padded_bytes` can be used to get an
    /// owned copy instead. As with the raw byte data exposed by
    /// :meth:`~to_raw_data`, bits beyond the logical length in the final byte
    /// are not masked to zero.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err(
                "Tibs is immutable and cannot export a writable buffer.",
            ));
        }
        let (data_ptr, data_len) = {
            // Infallible: Tibs is a frozen pyclass, so it has no borrow flag
            // to contend for and this cannot fail on a free-threaded build.
            let bits = slf.borrow();
            let Some(bytes) = BitCollection::byte_aligned_raw_data(&*bits) else {
                return Err(PyBufferError::new_err(
                    "Cannot export a buffer for this Tibs: its data does not start on a byte \
                     boundary. Use to_bytes() or to_padded_bytes() to get an owned copy instead.",
                ));
            };
            (bytes.as_ptr(), bytes.len())
        };
        // Safety: `data_ptr` points into the Arc<BV> owned by `slf`. Storing `slf`
        // itself in `view.obj` keeps that Arc (and so this pointer) alive for as
        // long as the buffer is exported. Tibs is frozen and its Arc<BV> is never
        // mutated in place, so the pointer stays valid without export tracking.
        unsafe {
            (*view).obj = slf.into_any().into_ptr();
            (*view).buf = data_ptr as *mut c_void;
            (*view).len = data_len as isize;
            (*view).readonly = 1;
            (*view).itemsize = 1;
            // A 'static format string, so there is nothing to free on release and
            // no allocation on export. This is what CPython's own PyBuffer_FillInfo
            // does; consumers treat the field as read-only despite the *mut.
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                c"B".as_ptr().cast_mut()
            } else {
                ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                &mut (*view).len
            } else {
                ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                &mut (*view).itemsize
            } else {
                ptr::null_mut()
            };
            (*view).suboffsets = ptr::null_mut();
            (*view).internal = ptr::null_mut();
        }
        Ok(())
    }

    // No __releasebuffer__: the exported view owns nothing that needs freeing.
    // CPython drops the reference stored in `view.obj` for us.

    /// Return string representations for printing.
    pub fn __str__(&self) -> String {
        self.to_string()
    }

    /// Return representation that could be used to recreate the instance.
    pub fn __repr__(&self) -> String {
        if self.is_empty() {
            "Tibs()".to_string()
        } else {
            format!("Tibs('{}')", self.__str__())
        }
    }

    /// Return a string formatted according to the Python format mini-language.
    ///
    /// The type codes ``b``, ``o``, ``x`` and ``X`` give the bit representation, and so
    /// keep any leading zeros. They are equivalent to the :attr:`~Tibs.bin`,
    /// :attr:`~Tibs.oct` and :attr:`~Tibs.hex` properties. The type codes ``u`` and ``i``
    /// give the unsigned and signed integer interpretations, and ``e``, ``f`` and ``g``
    /// (with their uppercase forms) show the IEEE float value using Python's scientific,
    /// fixed-point and general presentations; a float needs a length of 16, 32 or 64
    /// bits. All of these read the bits big-endian, the same as the matching properties.
    /// To interpret them in another byte or bit order, format a view such as ``self.le``
    /// instead.
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
    ///     >>> f"{Tibs('0xac804f4b'):#_.2x}"
    ///     '0xac_80_4f_4b'
    ///     >>> f"{Tibs('0x0f'):b}"
    ///     '00001111'
    ///
    #[pyo3(signature = (format_spec, /), text_signature = "($self, format_spec, /)")]
    pub fn __format__(&self, py: Python<'_>, format_spec: &str) -> PyResult<String> {
        helpers::format_bit_collection(py, self, format_spec, "Tibs")
    }

    /// Return a view with interpretation settings.
    ///
    /// A view does not change the underlying bits. It changes how operations such
    /// as integer conversion, byte conversion and field extraction interpret those
    /// bits.
    ///
    /// Byte-oriented views must have a whole-byte length. This applies when using
    /// little-endian or big-endian byte order, or when using ``BitOrder.Lsb0``.
    ///
    /// :param ByteOrder byte_order: The byte order used when interpreting whole-byte values. Defaults to ``ByteOrder.Unspecified``.
    /// :param BitOrder bit_order: The bit numbering order used for field labels. Defaults to ``BitOrder.Msb0``.
    /// :return: A new :class:`View`.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0x0100').view(byte_order=ByteOrder.Little).u
    ///     1
    ///
    #[pyo3(signature = (byte_order = ByteOrder::Unspecified, bit_order = BitOrder::Msb0), text_signature = "($self, byte_order=None, bit_order=None)")]
    pub fn view(
        slf: PyRef<'_, Self>,
        byte_order: Option<ByteOrder>,
        bit_order: Option<BitOrder>,
    ) -> PyResult<View> {
        let byte_order = byte_order.unwrap_or(ByteOrder::Unspecified);
        let bit_order = bit_order.unwrap_or(BitOrder::Msb0);
        View::validate_layout(slf.len(), byte_order, bit_order)?;
        Ok(View::from_tibs(slf.clone(), byte_order, bit_order))
    }

    /// Return a little-endian byte-order view.
    ///
    /// Equivalent to ``view(byte_order=ByteOrder.Little)``.
    ///
    /// The ``Tibs`` length must be a whole number of bytes.
    ///
    #[getter]
    pub fn le(slf: PyRef<'_, Self>) -> PyResult<View> {
        View::validate_layout(slf.len(), ByteOrder::Little, BitOrder::Msb0)?;
        Ok(View::from_tibs(
            slf.clone(),
            ByteOrder::Little,
            BitOrder::Msb0,
        ))
    }

    /// Return a big-endian byte-order view.
    ///
    /// Equivalent to ``view(byte_order=ByteOrder.Big)``.
    ///
    /// The ``Tibs`` length must be a whole number of bytes.
    ///
    #[getter]
    pub fn be(slf: PyRef<'_, Self>) -> PyResult<View> {
        View::validate_layout(slf.len(), ByteOrder::Big, BitOrder::Msb0)?;
        Ok(View::from_tibs(slf.clone(), ByteOrder::Big, BitOrder::Msb0))
    }

    /// Return an LSB0 bit-order view.
    ///
    /// ``BitOrder.Lsb0`` means that field labels are counted from the least
    /// significant bit of each byte. The ``Tibs`` length must be a whole number of
    /// bytes.
    ///
    /// Equivalent to ``view(bit_order=BitOrder.Lsb0)``.
    ///
    #[getter]
    pub fn lsb0(slf: PyRef<'_, Self>) -> PyResult<View> {
        View::validate_layout(slf.len(), ByteOrder::Unspecified, BitOrder::Lsb0)?;
        Ok(View::from_tibs(
            slf.clone(),
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
    pub fn msb0(slf: PyRef<'_, Self>) -> View {
        View::from_tibs(slf.clone(), ByteOrder::Unspecified, BitOrder::Msb0)
    }

    /// Extract a field using inclusive MSB0 bit labels.
    ///
    /// ``a`` and ``b`` must be zero or positive bit labels. The two endpoints
    /// are inclusive and may be provided in either order. This is equivalent to
    /// ``self.msb0.field(a, b)``.
    ///
    /// :param int a: One non-negative inclusive field endpoint.
    /// :param int b: The other non-negative inclusive field endpoint.
    /// :return: A new :class:`View`.
    ///
    #[pyo3(signature = (a, b, /), text_signature = "($self, a, b, /)")]
    pub fn field(slf: PyRef<'_, Self>, a: i64, b: i64) -> PyResult<View> {
        View::from_tibs(slf.clone(), ByteOrder::Unspecified, BitOrder::Msb0).field(a, b)
    }

    /// Iterate over the bits of the Tibs, yielding each bit as a boolean.
    ///
    /// :return: An iterator yielding bool values.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b101'))
    ///     [True, False, True]
    ///
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<BoolIterator>> {
        let py = slf.py();
        let length = slf.len();
        Py::new(
            py,
            BoolIterator {
                bits: slf.clone(),
                index: 0,
                length,
            },
        )
    }

    /// Return a list of Tibs by cutting into chunks.
    ///
    /// :param int chunk_size: The size in bits of the chunks to create.
    /// :param int | None count: If specified, at most count items are created. Default is to cut as many times as possible.
    /// :return: A list of Tibs chunks.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b110011').chunks(2)
    ///     [Tibs('0b11'), Tibs('0b00'), Tibs('0b11')]
    ///
    #[pyo3(signature = (chunk_size, /, count = None), text_signature = "($self, chunk_size, /, count=None)")]
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
    /// The returned pieces are normal ``Tibs`` slices. They share storage with
    /// the original ``Tibs`` when possible.
    ///
    /// :param int | Iterable[int] pos: The bit position or positions where the split should occur.
    /// :return: A tuple of ``Tibs`` pieces.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b101100').split_at(3)
    ///     (Tibs('0b101'), Tibs('0b100'))
    ///     >>> Tibs('0b101100').split_at([2, 5])
    ///     (Tibs('0b10'), Tibs('0b110'), Tibs('0b0'))
    ///
    #[pyo3(signature = (pos, /), text_signature = "($self, pos, /)")]
    pub fn split_at(&self, py: Python<'_>, pos: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
        let pieces = self.split_at_positions(&read_split_positions(pos)?)?;
        Ok(PyTuple::new(py, pieces)?.unbind())
    }

    /// Return an iterator by cutting into Tibs chunks.
    ///
    /// :param int chunk_size: The size in bits of the chunks to generate.
    /// :param int | None count: If specified, at most count items are generated. Default is to cut as many times as possible.
    /// :return: A generator yielding Tibs chunks.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b110011').chunks_iter(2))
    ///     [Tibs('0b11'), Tibs('0b00'), Tibs('0b11')]
    ///
    #[pyo3(signature = (chunk_size, /, count = None), text_signature = "($self, chunk_size, /, count=None)")]
    pub fn chunks_iter(
        slf: PyRef<'_, Self>,
        chunk_size: i64,
        count: Option<i64>,
    ) -> PyResult<Py<ChunksIterator>> {
        if chunk_size <= 0 {
            return Err(PyValueError::new_err(format!(
                "Cannot create chunk generator - chunk_size of {chunk_size} given, but it must be > 0."
            )));
        }
        let max_chunks = match count {
            Some(c) => {
                if c < 0 {
                    return Err(PyValueError::new_err(format!(
                        "Cannot create chunk generator - count of {c} given, but it must be >= 0 if present."
                    )));
                }
                c as usize
            }
            None => usize::MAX,
        };

        let py = slf.py();
        let bits_len = slf.len();
        let iter = ChunksIterator {
            bits_object: slf.into(),
            chunk_size: chunk_size as usize,
            max_chunks,
            current_pos: 0,
            chunks_generated: 0,
            bits_len,
            is_reverse: false,
        };
        Py::new(py, iter)
    }

    /// Return a reverse iterator by cutting into Tibs chunks, starting from the end.
    ///
    /// :param int chunk_size: The size in bits of the chunks to generate.
    /// :param int | None count: If specified, at most count items are generated. Default is to cut as many times as possible.
    /// :return: A generator yielding Tibs chunks.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b1100111').rchunks_iter(3))
    ///     [Tibs('0b111'), Tibs('0b100'), Tibs('0b11')]
    ///
    #[pyo3(signature = (chunk_size, /, count = None), text_signature = "($self, chunk_size, /, count=None)")]
    pub fn rchunks_iter(
        slf: PyRef<'_, Self>,
        chunk_size: i64,
        count: Option<i64>,
    ) -> PyResult<Py<ChunksIterator>> {
        if chunk_size <= 0 {
            return Err(PyValueError::new_err(format!(
                "Cannot create chunk generator - chunk_size of {chunk_size} given, but it must be > 0."
            )));
        }
        let max_chunks = match count {
            Some(c) => {
                if c < 0 {
                    return Err(PyValueError::new_err(format!(
                        "Cannot create chunk generator - count of {c} given, but it must be >= 0 if present."
                    )));
                }
                c as usize
            }
            None => usize::MAX,
        };

        let py = slf.py();
        let bits_len = slf.len();
        let iter = ChunksIterator {
            bits_object: slf.into(),
            chunk_size: chunk_size as usize,
            max_chunks,
            current_pos: bits_len,
            chunks_generated: 0,
            bits_len,
            is_reverse: true,
        };
        Py::new(py, iter)
    }

    /// Return True if two Tibs have the same binary representation.
    ///
    /// Equality is only defined against :class:`Tibs` and :class:`Mutibs`.
    ///
    /// >>> Tibs('0b1110') == Tibs('0xe')
    /// True
    ///
    pub fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        // `cast` rather than `extract::<PyRef<_>>`, which builds and discards a
        // Python exception when the other side is the class not tried first.
        if let Ok(other) = other.cast::<Tibs>() {
            return Ok(self.bits_equal(other.get()));
        }
        if let Ok(other) = other.cast::<Mutibs>() {
            // The operand needs its own section, or a thread writing to it is
            // refused by the borrow held here. `self` is a frozen `Tibs`, so
            // there is nothing to lock on this side and one section is enough.
            return with_locked(other, |other| Ok(self.bits_equal(other)));
        }
        Ok(false)
    }

    /// Return a hash of the logical bit sequence.
    pub fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let hash = hasher.finish() as isize;
        // Python reserves -1 as the error return value from tp_hash.
        if hash == -1 { -2 } else { hash }
    }

    /// Find all occurrences of a bit sequence.
    ///
    /// :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position of the slice to search. Defaults to 0.
    /// :param int | None end: The end bit position of the slice to search. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries. Defaults to ``False``.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: A list of bit positions.
    ///
    /// :raises ValueError: if needle is empty, if start or end are out of range, if end is before start
    ///     or if the mask length doesn't match the needle length.
    ///
    /// All occurrences of needle are found, even if they overlap.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b10111011').find_all('0b11')
    ///     [2, 3, 6]
    ///     >>> Tibs('0x1f2f3a').find_all('0x0f', mask='0x0f', byte_aligned=True)
    ///     [0, 8]
    ///
    #[pyo3(signature = (needle, /, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, needle, /, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn find_all(
        &self,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Vec<u64>> {
        find_all_in_bits(
            py,
            self.as_bitslice(),
            &needle,
            SearchParams {
                start,
                end,
                byte_aligned,
                mask,
            },
        )
    }

    /// Find all occurrences of a bit sequence, returning an iterator of bit positions.
    ///
    /// :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position of the slice to search. Defaults to 0.
    /// :param int | None end: The end bit position of the slice to search. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries. Defaults to ``False``.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: A generator yielding bit positions.
    ///
    /// :raises ValueError: if needle is empty, if start or end are out of range, if end is before start
    ///     or if the mask length doesn't match the needle length.
    ///
    /// All occurrences of needle are found, even if they overlap.
    ///
    /// Note that this method is not available for :class:`Mutibs` as its value could change while the
    /// generator is still active. For that case, convert to a :class:`Tibs` first with
    /// :meth:`Mutibs.to_tibs`, or use :meth:`Mutibs.as_tibs` if you no longer need the mutable object.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b10111011').find_all_iter('0b11'))
    ///     [2, 3, 6]
    ///
    #[pyo3(signature = (needle, /, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, needle, /, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn find_all_iter(
        slf: PyRef<'_, Self>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Py<FindAllIterator>> {
        FindAllIterator::new(slf, needle, start, end, byte_aligned, mask, false)
    }

    /// Find all occurrences of a bit sequence in reverse, returning an iterator of bit positions.
    ///
    /// :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position of the slice to search. Defaults to 0.
    /// :param int | None end: The end bit position of the slice to search. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries. Defaults to ``False``.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: A generator yielding bit positions.
    ///
    /// :raises ValueError: if needle is empty, if start or end are out of range, if end is before start
    ///     or if the mask length doesn't match the needle length.
    ///
    /// All occurrences of needle are found, even if they overlap.
    ///
    /// Note that this method is not available for :class:`Mutibs` as its value could change while the
    /// generator is still active. For that case, convert to a :class:`Tibs` first with
    /// :meth:`Mutibs.to_tibs`, or use :meth:`Mutibs.as_tibs` if you no longer need the mutable object.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0b10111011').rfind_all_iter('0b11'))
    ///     [6, 3, 2]
    ///
    #[pyo3(signature = (needle, /, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, needle, /, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn rfind_all_iter(
        slf: PyRef<'_, Self>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Py<FindAllIterator>> {
        FindAllIterator::new(slf, needle, start, end, byte_aligned, mask, true)
    }

    /// The bit length of the Tibs.
    #[inline]
    pub fn __len__(&self) -> usize {
        self.len()
    }

    /// Create a new instance with all bits set to '0'.
    ///
    /// :param int length: The number of bits to set.
    /// :return: A Tibs object with all bits set to zero.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_zeros(500)  # 500 zero bits
    ///
    #[classmethod]
    #[pyo3(signature = (length, /), text_signature = "(cls, length, /)")]
    pub fn from_zeros(_cls: &Bound<'_, PyType>, length: i64) -> PyResult<Self> {
        let length = validate_length(length)?;
        Ok(Self::from_bv(bv_from_zeros(length)))
    }

    /// Create a new instance by encoding one Python value with a dtype.
    ///
    /// :param Dtype | str dtype: The value encoding to use.
    /// :param object value: The value to encode.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_value("u8", 15)
    ///     Tibs('0x0f')
    ///
    #[classmethod]
    #[pyo3(signature = (dtype, value, /), text_signature = "(cls, dtype, value, /)")]
    pub fn from_value(
        _cls: &Bound<'_, PyType>,
        dtype: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let dtype = extract_dtype(dtype)?;
        Ok(Tibs::from_bv(bv_from_value(&dtype, value)?))
    }

    /// Create a new instance by encoding and concatenating values with a dtype.
    ///
    /// :param Dtype | str dtype: The value encoding to use for each item.
    /// :param Iterable iterable: The values to encode.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_values("u8", [1, 2, 3])
    ///     Tibs('0x010203')
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
        Ok(Tibs::from_bv(bv_from_values_iter(py, &dtype, iterable)?))
    }

    /// Return an iterator over values decoded with a dtype.
    ///
    /// The selected range must be a whole number of dtype values.
    ///
    /// :param Dtype | str dtype: The value encoding to use for each yielded item.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    /// :return: An iterator yielding decoded Python values.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> list(Tibs('0x010203').to_values_iter("u8"))
    ///     [1, 2, 3]
    ///
    #[pyo3(signature = (dtype, /, start = None, end = None), text_signature = "($self, dtype, /, start=None, end=None)")]
    pub fn to_values_iter(
        slf: PyRef<'_, Self>,
        dtype: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<ValuesIterator>> {
        let dtype = extract_dtype(dtype)?;
        let (start, end) = validate_slice(slf.len(), start, end)?;
        let py = slf.py();
        ValuesIterator::new(py, slf.into(), dtype, start, end)
    }

    /// Return a list of values decoded with a dtype.
    ///
    /// The selected range must be a whole number of dtype values.
    ///
    /// :param Dtype | str dtype: The value encoding to use for each item.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    /// :return: A list of decoded Python values.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0x010203').to_values("u8")
    ///     [1, 2, 3]
    ///
    #[pyo3(signature = (dtype, /, start = None, end = None), text_signature = "($self, dtype, /, start=None, end=None)")]
    pub fn to_values(
        &self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let dtype = extract_dtype(dtype)?;
        py_values_from_range(py, self, &dtype, start, end)
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
    ///     >>> Tibs('0x0f').to_value("u8")
    ///     15
    ///
    #[pyo3(signature = (dtype, /, start = None, end = None), text_signature = "($self, dtype, /, start=None, end=None)")]
    pub fn to_value(
        &self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<PyAny>> {
        let dtype = extract_dtype(dtype)?;
        let (start, end) = validate_slice(self.len(), start, end)?;
        let value = self.get_slice_unchecked(start, end - start);
        py_from_value(py, &dtype, &value)
    }

    /// Create a new instance with all bits set to '1'.
    ///
    /// :param int length: The number of bits to set.
    /// :return: A Tibs object with all bits set to one.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_ones(5)
    ///     Tibs('0b11111')
    ///
    #[classmethod]
    #[pyo3(signature = (length, /), text_signature = "(cls, length, /)")]
    pub fn from_ones(_cls: &Bound<'_, PyType>, length: i64) -> PyResult<Self> {
        let length = validate_length(length)?;
        Ok(Tibs::from_bv(bv_from_ones(length)))
    }

    /// Create a new instance from a formatted string.
    ///
    /// :param str s: The formatted string to convert. This can begin with '0b', '0o' or '0x' to indicate binary, octal or hexadecimal, and commas can be used to separate items.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_string("0xff01")
    ///     b = Tibs.from_string("0o775, 0b1")
    ///
    /// The ``__init__`` method can also redirect to ``from_string``:
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs("0xff01")
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_string(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        let bv = str_to_bv(s)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Create a new instance from an unsigned integer.
    ///
    /// :param int u: An unsigned integer.
    /// :param int length: The bit length to create. Can be any positive number of bits.
    /// :param ByteOrder byte_order: The byte order used to store the integer. Defaults to ByteOrder.Unspecified.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// :raises ValueError: if the integer doesn't fit in the length given.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_u(15, length=8)
    ///     Tibs('0x0f')
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
        Ok(Tibs::from_bv(bv_from_uint(u, length, is_little_endian)?))
    }

    /// Return the unsigned integer representation of the Tibs.
    ///
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    ///
    /// :return: The value as an unsigned integer.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0x0f').to_u()
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

    /// Read-only property of the unsigned integer representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_u` with no parameters.
    ///
    /// :return: The value as an unsigned integer.
    #[getter]
    fn u<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.to_u(py, None, None)
    }

    /// Create a new instance from a signed integer.
    ///
    /// :param int i: A signed integer.
    /// :param int length: The bit length to create. Can be any positive number of bits.
    /// :param ByteOrder byte_order: The byte order used to store the integer. Defaults to ByteOrder.Unspecified.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// :raises ValueError: if the integer doesn't fit in the length given.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_i(-2, length=4)
    ///     Tibs('0xe')
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
        Ok(Tibs::from_bv(bv_from_int(i, length, is_little_endian)?))
    }

    /// Return the signed integer representation of the Tibs.
    ///
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    ///
    /// :return: The value as a signed integer.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0xe').to_i()
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

    /// Read-only property of the signed integer representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_i` with no parameters.
    ///
    /// :return: The value as a signed integer.
    #[getter]
    fn i<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.to_i(py, None, None)
    }

    /// Create a new instance from a floating point number.
    ///
    /// :param float f: A floating point value.
    /// :param int length: The bit length to create. Must be 16, 32 or 64.
    /// :param ByteOrder byte_order: The byte order used to store the float. Defaults to ByteOrder.Unspecified.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_f(1.5, length=32)
    ///     Tibs('0x3fc00000')
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
        Ok(Tibs::from_bv(bv))
    }

    /// Return the floating point representation of the Tibs.
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
    ///     >>> Tibs('0x3fc00000').to_f()
    ///     1.5
    ///
    #[pyo3(signature = (start = None, end = None), text_signature = "($self, start=None, end=None)")]
    pub fn to_f(&self, start: Option<isize>, end: Option<isize>) -> PyResult<f64> {
        self.map_slice(start, end, |bits| BitCollection::to_f64(bits, false))
    }

    /// Read-only property of the floating point representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_f` with no parameters.
    ///
    /// :return: The value as a Python float.
    #[getter]
    fn f(&self) -> PyResult<f64> {
        self.to_f(None, None)
    }

    /// Create a new instance from a binary string.
    ///
    /// :param str s: A string of ``0`` and ``1`` s, optionally preceded with ``0b`` and optionally containing underscores.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_bin("0000_1111_0101")
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_bin(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        let bv = bv_from_bin(s)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Return the binary representation of the Tibs as a string.
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

    /// Read-only property of the binary representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_bin` with no parameters.
    ///
    /// :return: The binary representation.
    #[getter]
    fn bin(&self) -> String {
        BitCollection::to_binary(self)
    }

    /// Create a new instance from an octal string.
    ///
    /// :param str s: A string of octal digits, optionally preceded with ``0o`` and optionally containing underscores.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_oct("17")
    ///     Tibs('0b001111')
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_oct(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        let bv = bv_from_oct(s)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Return the octal representation of the Tibs as a string.
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

    /// Read-only property of the octal representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_oct` with no parameters.
    ///
    /// :return: The octal representation.
    /// :raises ValueError: if the length is not a multiple of 3.
    #[getter]
    fn oct(&self) -> PyResult<String> {
        BitCollection::to_octal(self)
    }

    /// Create a new instance from a hexadecimal string.
    ///
    /// :param str s: A string of hexadecimal digits, optionally preceded with ``0x`` and optionally containing underscores.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_hex("0f")
    ///     Tibs('0x0f')
    ///
    #[classmethod]
    #[pyo3(signature = (s, /), text_signature = "(cls, s, /)")]
    pub fn from_hex(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        let bv = bv_from_hex(s)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Return the hexadecimal representation of the Tibs as a string.
    ///
    /// Equivalent to using the ``hex`` property when called with no parameters.
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

    /// Read-only property of the hexadecimal representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_hex` with no parameters.
    ///
    /// :return: The hexadecimal representation.
    /// :raises ValueError: if the length is not a multiple of 4.
    #[getter]
    fn hex(&self) -> PyResult<String> {
        BitCollection::to_hexadecimal(self)
    }

    /// Create a new instance from a bytes object.
    ///
    /// :param bytes | bytearray | memoryview data: The bytes, bytearray or memoryview object to convert to a :class:`Tibs`.
    /// :param int | None offset: The bit offset from the start. Defaults to zero.
    /// :param int | None length: The bit length to use. Defaults to the whole of the data.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_bytes(b"some_bytes_maybe_from_a_file")
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

    /// Create a new instance from an iterable by converting each element to a bool.
    ///
    /// :param Iterable iterable: The iterable to convert to a :class:`Tibs`.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_bools([False, 0, 1, "Steven"])  # binary 0011
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /), text_signature = "(cls, iterable, /)")]
    pub fn from_bools(_cls: &Bound<'_, PyType>, iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bv = bv_from_bools(iterable)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Return the bits as a list of bools.
    ///
    /// This is much faster than using ``list()`` on the Tibs, which iterates bit by bit.
    ///
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to len(self).
    /// :return: A list of bools.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b101').to_bools()
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
    /// :return: A newly constructed ``Tibs`` with random data.
    ///
    /// The 'secure' option uses the OS's random data source, so will be slower and could potentially
    /// fail.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_random(1000000)  # A million random bits
    ///     b = Tibs.from_random(100, seed=b'a_seed')
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
        Ok(Tibs::from_bv(bv))
    }

    /// Create a new instance by concatenating a sequence of Tibs objects.
    ///
    /// This method concatenates a sequence of Tibs objects into a single Tibs object.
    ///
    /// :param Iterable iterable: An iterable to concatenate. Items can be anything that can be promoted to a Tibs.
    /// :return: A newly constructed ``Tibs``.
    ///
    /// .. code-block:: python
    ///
    ///     a = Tibs.from_joined(['0x01', [1, 0], b'some_bytes'])
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /), text_signature = "(cls, iterable, /)")]
    pub fn from_joined(_cls: &Bound<'_, PyType>, iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Build the immutable result directly; going through Mutibs::as_tibs
        // would move through an unnecessary mutable wrapper.
        Ok(Tibs::from_bv(Mutibs::joined_bv_from_iterable(iterable)?))
    }

    /// Return the Tibs as a bytes object.
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

    /// Return the Tibs as a bytes object, padding the right-hand side with zero bits.
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

    /// Read-only property of the ``bytes`` representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_bytes` with no parameters.
    ///
    /// :return: The bytes representation.
    /// :raises ValueError: if the length is not a multiple of 8.
    #[getter]
    fn bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        BitCollection::to_py_bytes(self, py)
    }

    /// Find first occurrence of a bit sequence.
    ///
    /// Returns the bit position if found, or None if not found.
    ///
    /// :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: The bit position if found, or None if not found.
    ///
    /// :raises ValueError: if ``needle`` is empty, if the slice parameters are invalid, or if the
    ///     mask length doesn't match the needle length.
    ///
    /// The ``mask`` must be the same length as ``needle``. Only the bits set in it are compared, so
    /// the bits of ``needle`` under a zero mask bit are ignored and can be anything.
    ///
    /// .. code-block:: pycon
    ///
    ///      >>> Tibs('0xc3e').find('0b1111')
    ///      6
    ///      >>> Tibs('0x3a5f').find('0x0f', mask='0x0f', byte_aligned=True)
    ///      8
    ///
    #[pyo3(signature = (needle, /, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, needle, /, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn find(
        &self,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Option<usize>> {
        find_in_bits(
            py,
            self.as_bitslice(),
            &needle,
            SearchParams {
                start,
                end,
                byte_aligned,
                mask,
            },
            false,
        )
    }

    /// Return True if b is a sub-sequence of self.
    pub fn __contains__(&self, py: Python<'_>, b: Tibs) -> PyResult<bool> {
        self.find(py, b, None, None, false, None)
            .map(|found| found.is_some())
    }

    /// As Tibs is immutable, this returns the same instance.
    pub fn __copy__(slf: PyRef<'_, Self>) -> Py<Self> {
        slf.into()
    }

    /// Return the callable and arguments that recreate the Tibs.
    ///
    /// Used by :mod:`pickle` and by :func:`copy.deepcopy`.
    ///
    /// :return: A tuple of :meth:`Tibs.decode` and the encoded bytes to pass to it.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> import pickle
    ///     >>> pickle.loads(pickle.dumps(Tibs('0b110101')))
    ///     Tibs('0b110101')
    ///
    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (Py<PyBytes>,))> {
        // Codec::Raw rather than the Codec::Auto default of `encode`: pickling
        // and deep copying should cost about what copying costs, and Auto
        // measures the alternative codecs and compresses on every call.
        let encoded = PyBytes::new(py, &self.encode(Some(Codec::Raw))?);
        let decode = py.get_type::<Self>().getattr("decode")?;
        Ok((decode.unbind(), (encoded.unbind(),)))
    }

    /// Find last occurrence of a bit sequence.
    ///
    /// Returns the bit position if found, or None if not found.
    ///
    /// :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: The bit position if found, or None if not found.
    ///
    /// :raises ValueError: if ``needle`` is empty, if the slice parameters are invalid, or if the
    ///     mask length doesn't match the needle length.
    ///
    /// .. code-block:: pycon
    ///
    ///      >>> Tibs('0b10111011').rfind('0b11')
    ///      6
    ///      >>> Tibs('0b10111011').rfind('0b00', mask='0b10')
    ///      5
    ///
    #[pyo3(signature = (needle, /, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, needle, /, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn rfind(
        &self,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Option<usize>> {
        find_in_bits(
            py,
            self.as_bitslice(),
            &needle,
            SearchParams {
                start,
                end,
                byte_aligned,
                mask,
            },
            true,
        )
    }

    /// Return whether the current Tibs starts with prefix.
    ///
    /// :param object prefix: The bits to search for. This can be anything promotable to ``Tibs``.
    /// :return: True if the Tibs starts with the prefix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b101100').starts_with('0b101')
    ///     True
    ///     >>> Tibs('0b101100').starts_with('0b100')
    ///     False
    ///
    #[pyo3(signature = (prefix, /), text_signature = "($self, prefix, /)")]
    pub fn starts_with(&self, prefix: Tibs) -> bool {
        <Tibs as BitCollection>::starts_with(self, prefix)
    }

    /// Return whether the current Tibs ends with suffix.
    ///
    /// :param object suffix: The bits to search for. This can be anything promotable to ``Tibs``.
    /// :return: True if the Tibs ends with the suffix, otherwise False.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b101100').ends_with('0b100')
    ///     True
    ///     >>> Tibs('0b101100').ends_with('0b101')
    ///     False
    ///
    #[pyo3(signature = (suffix, /), text_signature = "($self, suffix, /)")]
    pub fn ends_with(&self, suffix: Tibs) -> bool {
        <Tibs as BitCollection>::ends_with(self, suffix)
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
    ///     >>> Tibs('0xef').count()
    ///     7
    ///     >>> Tibs('0xef').count(1, 0, 4)
    ///     3
    ///     >>> Tibs.from_bin('0011010101100').count('0b01')
    ///     4
    ///     >>> Tibs('0b1111111').count('0b11')  # overlapping
    ///     6
    ///     >>> Tibs('0x80ff00').count(1, byte_aligned=True)
    ///     2
    ///
    #[pyo3(signature = (value=None, /, start=None, end=None, byte_aligned=false, mask=None), text_signature = "($self, value=None, /, start=None, end=None, byte_aligned=False, mask=None)")]
    pub fn count(
        &self,
        py: Python<'_>,
        value: Option<&Bound<'_, PyAny>>,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<usize> {
        count_in_bits(
            py,
            self.as_bitslice(),
            &resolve_count_target(value)?,
            SearchParams {
                start,
                end,
                byte_aligned,
                mask,
            },
        )
    }

    /// Return True if all bits are equal to 1, otherwise return False.
    ///
    /// :return: ``True`` if all bits are 1, otherwise ``False``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b1111').all()
    ///     True
    ///     >>> Tibs('0b1011').all()
    ///     False
    ///
    #[inline]
    pub fn all(&self) -> bool {
        <Self as BitCollection>::all_set(self)
    }

    /// Return True if any bits are equal to 1, otherwise return False.
    ///
    /// :return: ``True`` if any bits are 1, otherwise ``False``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b0000').any()
    ///     False
    ///     >>> Tibs('0b1000').any()
    ///     True
    ///
    #[inline]
    pub fn any(&self) -> bool {
        <Self as BitCollection>::any_set(self)
    }

    /// Return a new Tibs with one or many bits set to 1.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.set`.
    ///
    /// :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
    /// :return: A new Tibs.
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_zeros(5).set_at([1, 3])
    ///     Tibs('0b01010')
    ///
    #[pyo3(signature = (pos, /), text_signature = "($self, pos, /)")]
    pub fn set_at(&self, pos: &Bound<'_, PyAny>) -> PyResult<Self> {
        let positions = Mutibs::read_positions(Some(pos))?;
        self.copy_with_mutation(|out| out.apply_set_positions(true, &positions))
    }

    /// Return a new Tibs with one or many bits set to 0.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.unset`.
    ///
    /// :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
    /// :return: A new Tibs.
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.from_ones(5).unset_at([1, 3])
    ///     Tibs('0b10101')
    ///
    #[pyo3(signature = (pos, /), text_signature = "($self, pos, /)")]
    pub fn unset_at(&self, pos: &Bound<'_, PyAny>) -> PyResult<Self> {
        let positions = Mutibs::read_positions(Some(pos))?;
        self.copy_with_mutation(|out| out.apply_set_positions(false, &positions))
    }

    /// Return a new Tibs with selected bits inverted.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.invert`.
    ///
    /// :param int | Iterable[int] | None pos: Either a single bit position, an iterable of bit positions,
    ///   or None to invert every bit. Defaults to None.
    /// :return: A new Tibs.
    /// :raises IndexError: if pos < -len(self) or pos >= len(self).
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b10110').inverted([0, 2])
    ///     Tibs('0b00010')
    ///
    #[pyo3(signature = (pos = None, /), text_signature = "($self, pos=None, /)")]
    pub fn inverted(&self, pos: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if pos.is_none() {
            return Ok(self.inverted_copy());
        }
        let positions = Mutibs::read_positions(pos)?;
        self.copy_with_mutation(|out| out.apply_invert_positions(&positions))
    }

    /// Insert bits at position pos and return a new Tibs.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.insert`.
    ///
    /// :param int pos: The bit position to insert at. Clips to the start or end if out of range.
    /// :param object bs: The bits to insert. This can be anything promotable to ``Tibs``.
    /// :return: A new Tibs.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b1011').inserted(2, '0b00')
    ///     Tibs('0b100011')
    ///
    #[pyo3(signature = (pos, bs, /), text_signature = "($self, pos, bs, /)")]
    pub fn inserted(&self, pos: isize, bs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bs = Tibs::extract(bs.as_borrowed())?;
        self.copy_with_mutation(|out| {
            out.apply_insert_bits(pos, &bs);
            Ok(())
        })
    }

    /// Search and replace and return a new Tibs.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.replace`.
    ///
    /// :param object old: The bits to search for. This can be anything promotable to ``Tibs``.
    /// :param object new: The bits to replace with. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param int | None count: If present, the maximum number of replacements to make.
    /// :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: A new Tibs.
    /// :raises ValueError: if old is empty, count is negative, the slice parameters are invalid or
    ///     the mask length doesn't match the length of old.
    ///
    /// The ``mask`` affects only which bits have to match; the whole of each match is still
    /// replaced by ``new``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b00010010').replaced([0, 1], [1, 1, 1])
    ///     Tibs('0b0011101110')
    ///     >>> Tibs('0x1f2e3f').replaced('0x0f', '0x00', mask='0x0f', byte_aligned=True)
    ///     Tibs('0x002e00')
    ///
    #[pyo3(signature = (old, new, /, start=None, end=None, count=None, byte_aligned=false, mask=None), text_signature = "($self, old, new, /, start=None, end=None, count=None, byte_aligned=False, mask=None)")]
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
        self.copy_with_mutation(move |out| {
            out.apply_replace_bits(py, old, new, start, end, count, byte_aligned, mask)?;
            Ok(())
        })
    }

    /// Create and return a mutable copy of the Tibs as a Mutibs instance.
    ///
    /// :return: A new Mutibs with the same bit data.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> t = Tibs.from_hex('abc')
    ///     >>> m = t.to_mutibs()
    ///     >>> m *= 4
    ///     >>> print(t.hex)
    ///     abc
    ///     >>> print(m.hex)
    ///     abcabcabcabc
    ///
    pub fn to_mutibs(&self) -> Mutibs {
        Mutibs::from_bv(self.to_bitvec())
    }

    #[inline]
    /// Get a bit or a slice of bits.
    ///
    /// :param int | slice key: The index or slice to get.
    /// :return: A bool for a single index, or a new Tibs for a slice.
    /// :raises IndexError: If the index is out of range.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> t = Tibs('0b101100')
    ///     >>> t[0]
    ///     True
    ///     >>> t[1:4]
    ///     Tibs('0b011')
    ///
    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = key.py();
        // Handle integer indexing
        // Fast path for exact int keys via direct ffi. The bool singleton is
        // chosen by indexing, not branching: with random data an if/else here
        // mispredicts ~50% of the time, costing ~10ns per read.
        unsafe {
            if ffi::PyLong_Check(key.as_ptr()) != 0 {
                let index = ffi::PyLong_AsSsize_t(key.as_ptr());
                if index == -1 && !ffi::PyErr_Occurred().is_null() {
                    let err = PyErr::fetch(py);
                    return Err(if err.is_instance_of::<PyOverflowError>(py) {
                        PyIndexError::new_err(format!(
                            "Index is out of range for length of {}",
                            self.length
                        ))
                    } else {
                        err
                    });
                }
                let index = validate_index(index, self.length)?;
                // SAFETY: validate_index guarantees index < self.length.
                let value = self.bit_at_unchecked(index);
                // select_unpredictable compiles to a conditional move: a plain
                // if/else on random bit data mispredicts ~50% of the time,
                // which costs ~10ns per read.
                let obj = std::hint::select_unpredictable(value, ffi::Py_True(), ffi::Py_False());
                ffi::Py_INCREF(obj);
                return Ok(Bound::from_owned_ptr(py, obj).unbind());
            }
        }
        // Handle slice indexing. This is checked before the general integer
        // extraction below because that extraction raises and discards a Python
        // exception for a key it cannot convert, which costs more than the
        // slice itself.
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(self.len() as isize)?;
            let (start, stop, step) = (indices.start, indices.stop, indices.step);

            let result = if step == 1 {
                if start < stop {
                    self.get_slice_unchecked(start as usize, (stop - start) as usize)
                } else {
                    Tibs::empty()
                }
            } else {
                self.get_slice_with_step(start, stop, step)?
            };
            let py_obj = Py::new(py, result)?.into_pyobject(py)?;
            return Ok(py_obj.into());
        }

        // Anything else that can still act as an index, such as a NumPy
        // integer, which the `PyLong_Check` above does not accept.
        if let Ok(index) = key.extract::<isize>() {
            let index = validate_index(index, self.length)?;
            // SAFETY: validate_index guarantees index < self.length.
            let value = unsafe { self.bit_at_unchecked(index) };
            let py_value = PyBool::new(py, value);
            return Ok(py_value.to_owned().into());
        }

        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    /// Return new Tibs shifted by n to the left.
    ///
    /// :param int n: The number of bits to shift. Must be >= 0.
    /// :return: A new Tibs.
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b001100') << 2
    ///     Tibs('0b110000')
    ///
    pub fn __lshift__(&self, n: i64) -> PyResult<Self> {
        let shift = validate_shift(self, n)?;
        Ok(self.shifted_copy(shift, true))
    }

    /// Return new Tibs shifted by n to the right.
    ///
    /// :param int n: The number of bits to shift. Must be >= 0.
    /// :return: A new Tibs.
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b001100') >> 2
    ///     Tibs('0b000011')
    ///
    pub fn __rshift__(&self, n: i64) -> PyResult<Self> {
        let shift = validate_shift(self, n)?;
        Ok(self.shifted_copy(shift, false))
    }

    /// Concatenates two Tibs and return a newly constructed Tibs.
    ///
    /// :param object other: The bits to append. This can be anything promotable to ``Tibs``.
    /// :return: A new Tibs.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b10') + '0b1'
    ///     Tibs('0b101')
    ///
    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        Ok(self.concatenated(&other))
    }

    /// Concatenates two Tibs and return a newly constructed Tibs.
    ///
    /// :param object other: The bits to prepend. This can be anything promotable to ``Tibs``.
    /// :return: A new Tibs.
    ///
    pub fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        Ok(other.concatenated(self))
    }

    /// Count the bits set in both this Tibs and another.
    ///
    /// Equivalent to ``(self & other).count(1)``, but without building the
    /// intermediate ``Tibs``.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: The number of positions set in both.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b1100').count_and('0b1010')
    ///     1
    ///
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn count_and(&self, other: Tibs) -> PyResult<usize> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(self.pairwise_count(&other, LogicalOp::And))
    }

    /// Count the bits set in either this Tibs or another.
    ///
    /// Equivalent to ``(self | other).count(1)``, but without building the
    /// intermediate ``Tibs``.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: The number of positions set in either.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b1100').count_or('0b1010')
    ///     3
    ///
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn count_or(&self, other: Tibs) -> PyResult<usize> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(self.pairwise_count(&other, LogicalOp::Or))
    }

    /// Count the bits that differ between this Tibs and another.
    ///
    /// This is the Hamming distance. Equivalent to ``(self ^ other).count(1)``,
    /// but without building the intermediate ``Tibs``.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: The number of positions where the two differ.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b1100').count_xor('0b1010')
    ///     2
    ///
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn count_xor(&self, other: Tibs) -> PyResult<usize> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(self.pairwise_count(&other, LogicalOp::Xor))
    }

    /// Count the bits set in this Tibs but not in another.
    ///
    /// Equivalent to ``self.count(1) - self.count_and(other)``, but in a single pass.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: The number of positions set here but not in the other.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b1100').count_andnot('0b1010')
    ///     1
    ///
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn count_andnot(&self, other: Tibs) -> PyResult<usize> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(self.pairwise_count(&other, LogicalOp::AndNot))
    }

    /// Return whether any bit is set in both this Tibs and another.
    ///
    /// Equivalent to ``(self & other).any()``, but stops at the first bit set in
    /// both instead of building the intermediate ``Tibs``.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: ``True`` if some position is set in both, otherwise ``False``.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b1100').intersects('0b1010')
    ///     True
    ///     >>> Tibs('0b1100').intersects('0b0011')
    ///     False
    ///
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn intersects(&self, other: Tibs) -> PyResult<bool> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(self.pairwise_any(&other, LogicalOp::And))
    }

    /// Return whether no bit is set in both this Tibs and another.
    ///
    /// The negation of :meth:`intersects`. Equivalent to ``not (self & other).any()``,
    /// but stops at the first bit set in both instead of building the
    /// intermediate ``Tibs``.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: ``True`` if no position is set in both, otherwise ``False``.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b1100').is_disjoint('0b0011')
    ///     True
    ///     >>> Tibs('0b1100').is_disjoint('0b1010')
    ///     False
    ///
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn is_disjoint(&self, other: Tibs) -> PyResult<bool> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(!self.pairwise_any(&other, LogicalOp::And))
    }

    /// Return whether every bit set in this Tibs is also set in another.
    ///
    /// Equivalent to ``(self & other) == self``, but stops at the first bit set
    /// here and not there.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: ``True`` if every position set here is set in the other, otherwise ``False``.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b1000').is_subset_of('0b1010')
    ///     True
    ///     >>> Tibs('0b1100').is_subset_of('0b1010')
    ///     False
    ///
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn is_subset_of(&self, other: Tibs) -> PyResult<bool> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(!self.pairwise_any(&other, LogicalOp::AndNot))
    }

    /// Return whether every bit set in another is also set in this Tibs.
    ///
    /// The mirror of :meth:`is_subset_of`. Equivalent to ``(self & other) == other``,
    /// but stops at the first bit set there and not here.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: ``True`` if every position set in the other is set here, otherwise ``False``.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b1010').is_superset_of('0b1000')
    ///     True
    ///     >>> Tibs('0b1010').is_superset_of('0b1100')
    ///     False
    ///
    #[pyo3(signature = (other, /), text_signature = "($self, other, /)")]
    pub fn is_superset_of(&self, other: Tibs) -> PyResult<bool> {
        validate_logical_op_lengths(self.len(), other.len())?;
        // `and not` with the operands the other way round: the first bit
        // present in `other` and missing here ends the walk.
        Ok(!other.pairwise_any(self, LogicalOp::AndNot))
    }

    /// Read the bits at the positions set in a mask, packed together.
    ///
    /// This reads a bit field whose bits are scattered through the Tibs by the
    /// mask, the way :meth:`field` reads a contiguous one. The result has one
    /// bit for each set bit of the mask, in order.
    ///
    /// :param object mask: The mask selecting which bits to read. This can be anything promotable to ``Tibs``, and must be the same length as ``self``.
    /// :return: A new Tibs of length ``mask.count()``.
    /// :raises ValueError: if the mask length doesn't match the length of ``self``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b11010110').extracted('0b10110000')
    ///     Tibs('0b101')
    ///
    // Named `extract_field` because `Tibs::extract` is the FromPyObject
    // promotion method used throughout the crate.
    #[pyo3(name = "extracted", signature = (mask, /), text_signature = "($self, mask, /)")]
    pub fn extract_field(&self, mask: Tibs) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), mask.len())?;
        Ok(Self::from_bv(self.extract_masked(&mask)))
    }

    /// Return a new Tibs with a scattered bit field written into it.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.deposit`, and the
    /// inverse of :meth:`extracted`: the bits of ``value`` are written into the
    /// positions set in ``mask``, and the other bits are copied unchanged.
    ///
    /// :param object value: The bits to deposit. This can be anything promotable to ``Tibs``, and must be ``mask.count()`` bits long.
    /// :param object mask: The mask selecting which positions to write. This can be anything promotable to ``Tibs``, and must be the same length as ``self``.
    /// :return: A new Tibs.
    /// :raises ValueError: if the mask length doesn't match the length of ``self``, or ``value`` is not ``mask.count()`` bits long.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b11010110').deposited('0b111', '0b10110000').bin
    ///     '11110110'
    ///
    #[pyo3(signature = (value, mask, /), text_signature = "($self, value, mask, /)")]
    pub fn deposited(&self, value: &Bound<'_, PyAny>, mask: Tibs) -> PyResult<Self> {
        let value = Tibs::extract(value.as_borrowed())?;
        self.copy_with_mutation(|out| out.apply_deposit(&value, &mask))
    }

    /// Bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: A new Tibs.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.logical_op_from_python(other, LogicalOp::And)
    }

    /// Bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: A new Tibs.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.logical_op_from_python(other, LogicalOp::Or)
    }

    /// Bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: A new Tibs.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.logical_op_from_python(other, LogicalOp::Xor)
    }

    /// Reverse bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__and__(other)
    }

    /// Reverse bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__or__(other)
    }

    /// Reverse bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// This method is used when the RHS is a Tibs and the LHS is not, but can be converted to one.
    ///
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__xor__(other)
    }

    /// Return a new Tibs with the bits rotated to the left.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.rotate_left`.
    ///
    /// :param int n: The number of bits to rotate by.
    /// :param int | None start: Start of slice to rotate. Defaults to 0.
    /// :param int | None end: End of slice to rotate. Defaults to len(self).
    /// :return: A new Tibs.
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b10110').rotated_left(2)
    ///     Tibs('0b11010')
    ///
    #[pyo3(signature = (n, /, start=None, end=None), text_signature = "($self, n, /, start=None, end=None)")]
    pub fn rotated_left(&self, n: i64, start: Option<isize>, end: Option<isize>) -> PyResult<Self> {
        self.copy_with_mutation(|out| out.apply_rotation(n, start, end, true))
    }

    /// Return a new Tibs with the bits rotated to the right.
    ///
    /// This is the immutable equivalent of :meth:`Mutibs.rotate_right`.
    ///
    /// :param int n: The number of bits to rotate by.
    /// :param int | None start: Start of slice to rotate. Defaults to 0.
    /// :param int | None end: End of slice to rotate. Defaults to len(self).
    /// :return: A new Tibs.
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b10110').rotated_right(1)
    ///     Tibs('0b01011')
    ///
    #[pyo3(signature = (n, /, start=None, end=None), text_signature = "($self, n, /, start=None, end=None)")]
    pub fn rotated_right(
        &self,
        n: i64,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Self> {
        self.copy_with_mutation(|out| out.apply_rotation(n, start, end, false))
    }

    /// Create a Tibs by decoding bytes created via Tibs.encode()
    ///
    /// :param bytes | bytearray b: The encoded bytes to decode.
    /// :return: A new Tibs.
    /// :raises tibs.DecodeError: for badly formed, truncated or extended input bytes.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.decode(Tibs('0b101').encode())
    ///     Tibs('0b101')
    ///
    #[classmethod]
    #[pyo3(signature = (b, /), text_signature = "(cls, b, /)")]
    pub fn decode(_cls: &Bound<'_, PyType>, b: &Bound<'_, PyAny>) -> PyResult<Tibs> {
        tibs_codec::decode_bytes::<Tibs>(b.py(), bytes_like_to_vec(b)?)
    }

    /// Encode the tibs as a bytes instance.
    ///
    /// The bit length and the bit indexing are stored in the encoded bytes.
    ///
    /// The bytes instance can be used to recreate the Tibs exactly -
    /// see :meth:`Tibs.decode`.
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
    ///     >>> t = Tibs('0b101')
    ///     >>> b = t.encode()
    ///     >>> b
    ///     b'\x8d'
    ///     >>> Tibs.decode(b)
    ///     Tibs('0b101')
    ///
    #[pyo3(signature = (codec=Codec::Auto), text_signature = "($self, codec=None)")]
    pub fn encode(&self, codec: Option<Codec>) -> PyResult<Vec<u8>> {
        tibs_codec::encode(self, codec)
    }

    /// Return the instance with every bit inverted.
    ///
    /// :return: A new Tibs.
    ///
    /// Inverting an empty Tibs gives an empty Tibs, as :meth:`inverted` does.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> ~Tibs('0b10110')
    ///     Tibs('0b01001')
    ///
    pub fn __invert__(&self) -> Self {
        self.inverted_copy()
    }

    /// Return the Tibs as a bytes object.
    ///
    /// :return: The bytes representation.
    /// :raises ValueError: if the length is not a multiple of 8.
    pub fn __bytes__(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        BitCollection::to_py_bytes(self, py)
    }

    /// Return new Tibs consisting of n concatenations of self.
    ///
    /// Called for expression of the form 'a = b*3'.
    ///
    /// :param int n: The number of concatenations. Must be >= 0.
    /// :return: A new Tibs.
    /// :raises ValueError: if n < 0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b10') * 3
    ///     Tibs('0b101010')
    ///
    pub fn __mul__(&self, n: i64) -> PyResult<Self> {
        if n < 0 {
            return Err(PyValueError::new_err(
                "Cannot multiply by a negative integer.",
            ));
        }
        Ok(self.repeated(n as usize))
    }

    /// Return Tibs consisting of n concatenations of self.
    ///
    /// Called for expressions of the form 'a = 3*b'.
    ///
    /// :param int n: The number of concatenations. Must be >= 0.
    /// :return: A new Tibs.
    /// :raises ValueError: if n < 0.
    ///
    pub fn __rmul__(&self, n: i64) -> PyResult<Self> {
        self.__mul__(n)
    }

    /// Item assignment is not supported for immutable Tibs objects.
    pub fn __setitem__(&self, _key: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "Tibs objects do not support item assignment. Did you mean to use the Mutibs class? Call to_mutibs() to convert to a Mutibs.",
        ))
    }

    /// Item deletion is not supported for immutable Tibs objects.
    pub fn __delitem__(&self, _key: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "Tibs objects do not support item deletion. Did you mean to use the Mutibs class? Call to_mutibs() to convert to a Mutibs.",
        ))
    }
}
