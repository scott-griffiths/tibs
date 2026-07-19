use crate::codec as tibs_codec;
use crate::core::{BitCollection, concatenate_bitcollections, count_bitslice};
use crate::dtype::{Dtype, extract_dtype};
use crate::enums::{BitOrder, ByteOrder, Codec, DtypeKind};
use crate::helpers;
use crate::helpers::{
    BS, BV, bv_from_bin, bv_from_bools, bv_from_bytes_slice, bv_from_f64, bv_from_hex,
    bv_from_i128, bv_from_oct, bv_from_ones, bv_from_random, bv_from_u128, bv_from_zeros,
    bytes_like_to_vec, find_bitvec_aligned, promote_to_bv, rfind_bitvec_aligned, str_to_bv,
    validate_length, validate_logical_op_lengths, validate_shift, validate_slice,
};
use crate::iterator::{BoolIterator, ChunksIterator, FindAllIterator, ValuesIterator};
use crate::mutibs::Mutibs;
use crate::view::View;
use bitvec::prelude::*;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyList, PySlice, PyTuple, PyType};
use std::collections::hash_map::DefaultHasher;
use std::ffi::{CString, c_int, c_void};
use std::hash::{Hash, Hasher};
use std::ptr;
use std::sync::Arc;

impl Hash for Tibs {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);

        let bits = self.to_bitslice();

        let mut words = bits.chunks_exact(64);
        for chunk in words.by_ref() {
            state.write_u64(chunk.load_be::<u64>());
        }

        let mut bytes = words.remainder().chunks_exact(8);
        for chunk in bytes.by_ref() {
            state.write_u8(chunk.load_be::<u8>());
        }

        let tail = bytes.remainder();
        if !tail.is_empty() {
            let mut last = 0u8;
            for bit in tail {
                last = (last << 1) | (*bit as u8);
            }
            last <<= 8 - tail.len();
            state.write_u8(last);
        }
    }
}

// ---- Tibs private helper methods. Not part of the Python interface. ----

impl Tibs {
    pub(crate) fn from_bv(bv: BV) -> Self {
        let length = bv.len();
        Tibs {
            data: Arc::new(bv),
            offset: 0,
            length,
        }
    }

    pub(crate) fn get_slice_unchecked(&self, offset: usize, length: usize) -> Self {
        Tibs {
            data: self.data.clone(),
            offset: self.offset + offset,
            length,
        }
    }

    #[inline]
    fn shares_view_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
            && self.offset == other.offset
            && self.length == other.length
    }

    #[inline]
    pub(crate) fn as_bitslice(&self) -> &BS {
        &self.data[self.offset..self.offset + self.length]
    }

    #[inline]
    pub(crate) fn to_bitvec(&self) -> BV {
        let mut result = BV::from_vec(<Self as BitCollection>::to_padded_byte_data(self));
        result.truncate(self.length);
        result
    }

    #[inline]
    pub(crate) fn to_bitslice(&self) -> &BS {
        self.as_bitslice()
    }

    #[inline]
    pub(crate) fn raw_bytes(&self) -> Vec<u8> {
        let bit_offset = match self.as_bitslice().domain() {
            bitvec::domain::Domain::Enclave(elem) => elem.head().into_inner() as usize,
            bitvec::domain::Domain::Region {
                head: Some(elem), ..
            } => elem.head().into_inner() as usize,
            _ => 0,
        };
        let physical_start = self.offset;
        let byte_start = physical_start / 8;
        let byte_len = (bit_offset + self.length).div_ceil(8);
        self.data.as_raw_slice()[byte_start..byte_start + byte_len].to_vec()
    }

    #[inline]
    pub(crate) fn raw_data_ref(&self) -> Option<(&[u8], usize, usize)> {
        let data_head_offset = match self.data.as_bitslice().domain() {
            bitvec::domain::Domain::Enclave(elem) => elem.head().into_inner() as usize,
            bitvec::domain::Domain::Region {
                head: Some(elem), ..
            } => elem.head().into_inner() as usize,
            _ => 0,
        };
        if data_head_offset != 0 {
            return None;
        }

        let physical_start = self.offset;
        let byte_start = physical_start / 8;
        let bit_offset = physical_start % 8;
        let byte_len = (bit_offset + self.length).div_ceil(8);
        Some((
            &self.data.as_raw_slice()[byte_start..byte_start + byte_len],
            bit_offset,
            self.length,
        ))
    }

    pub(crate) fn find_impl(
        &self,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        reverse: bool,
    ) -> PyResult<Option<usize>> {
        if needle.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to find."));
        }
        let (start, end) = validate_slice(self.len(), start, end)?;
        let alignment_mod8 = if byte_aligned { Some(0) } else { None };

        let found = if !reverse {
            find_bitvec_aligned(
                py,
                self.to_bitslice(),
                needle.as_bitslice(),
                start,
                end,
                alignment_mod8,
            )?
        } else {
            rfind_bitvec_aligned(
                py,
                self.to_bitslice(),
                needle.as_bitslice(),
                start,
                end,
                alignment_mod8,
            )?
        };
        Ok(found)
    }

    fn copy_with_mutation(&self, f: impl FnOnce(&mut Mutibs) -> PyResult<()>) -> PyResult<Self> {
        let mut out = self.to_mutibs();
        f(&mut out)?;
        Ok(out.to_tibs())
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
#[derive(Clone)]
#[pyclass(frozen, sequence, skip_from_py_object, module = "tibs")]
pub struct Tibs {
    data: Arc<BV>,
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
            return Ok(mutibs_ref.to_tibs());
        }
        let bv = promote_to_bv(&obj)?;
        Ok(Tibs::from_bv(bv))
    }
}

pub(crate) fn bv_from_value(dtype: &Dtype, value: &Bound<'_, PyAny>) -> PyResult<BV> {
    match dtype.kind {
        DtypeKind::Float => {
            let is_little_endian =
                ByteOrder::is_little_endian(Some(dtype.byte_order), dtype.length)?;
            bv_from_f64(value.extract::<f64>()?, dtype.length, is_little_endian)
        }
        DtypeKind::Uint => {
            let is_little_endian =
                ByteOrder::is_little_endian(Some(dtype.byte_order), dtype.length)?;
            bv_from_u128(value.extract::<u128>()?, dtype.length, is_little_endian)
        }
        DtypeKind::Int => {
            let is_little_endian =
                ByteOrder::is_little_endian(Some(dtype.byte_order), dtype.length)?;
            bv_from_i128(value.extract::<i128>()?, dtype.length, is_little_endian)
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
    }
}

fn validate_dtype_value_length(dtype: &Dtype, bv: BV) -> PyResult<BV> {
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

pub(crate) fn bv_from_values_iter(
    py: Python<'_>,
    dtype: &Dtype,
    iterable: &Bound<'_, PyAny>,
) -> PyResult<BV> {
    let capacity = iterable
        .len()
        .ok()
        .and_then(|len| len.checked_mul(dtype.length));
    let mut bv = capacity.map_or_else(BV::new, BV::with_capacity);
    let mut check_at = helpers::SIGNAL_CHECK_INTERVAL;
    for (index, item) in iterable.try_iter()?.enumerate() {
        if index >= check_at {
            py.check_signals()?;
            check_at = index.saturating_add(helpers::SIGNAL_CHECK_INTERVAL);
        }
        bv.extend(bv_from_value(dtype, &item?)?);
    }
    Ok(bv)
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

    match dtype_kind {
        DtypeKind::Float => {
            let is_little_endian = ByteOrder::is_little_endian(Some(byte_order), dtype_length)?;
            BitCollection::to_f64(value, is_little_endian)?.into_py_any(py)
        }
        DtypeKind::Uint => {
            let is_little_endian = ByteOrder::is_little_endian(Some(byte_order), dtype_length)?;
            BitCollection::to_u128(value, is_little_endian)?.into_py_any(py)
        }
        DtypeKind::Int => {
            let is_little_endian = ByteOrder::is_little_endian(Some(byte_order), dtype_length)?;
            BitCollection::to_i128(value, is_little_endian)?.into_py_any(py)
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
    }
}

pub(crate) fn py_from_value(py: Python<'_>, dtype: &Dtype, value: &Tibs) -> PyResult<Py<PyAny>> {
    py_from_value_parts(py, dtype.kind, dtype.length, dtype.byte_order, value)
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
    let mut values = Vec::with_capacity(count);
    let mut check_at = helpers::SIGNAL_CHECK_INTERVAL;
    for index in 0..count {
        if index >= check_at {
            py.check_signals()?;
            check_at = index.saturating_add(helpers::SIGNAL_CHECK_INTERVAL);
        }
        let value = bits.get_slice_unchecked(start + index * dtype.length, dtype.length);
        values.push(py_from_value(py, dtype, &value)?);
    }
    Ok(values)
}

/// Public Python-facing methods.
#[pymethods]
impl Tibs {
    #[new]
    #[pyo3(signature = (auto = None), text_signature = "(auto=None)")]
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
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                CString::new("B").unwrap().into_raw()
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

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe {
            if !(*view).format.is_null() {
                drop(CString::from_raw((*view).format));
            }
        }
    }

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
    pub fn msb0(slf: PyRef<'_, Self>) -> PyResult<View> {
        Ok(View::from_tibs(
            slf.clone(),
            ByteOrder::Unspecified,
            BitOrder::Msb0,
        ))
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
    #[pyo3(signature = (a, b), text_signature = "($self, a, b)")]
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
        let pieces = BitCollection::collect_split_at(self, pos)?;
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
    #[pyo3(signature = (chunk_size, count = None), text_signature = "($self, chunk_size, count=None)")]
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
    #[pyo3(signature = (chunk_size, count = None), text_signature = "($self, chunk_size, count=None)")]
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
                        "Cannot create chunk generator - count of {c} given, but it must be > 0 if present."
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
        if let Ok(other) = other.extract::<PyRef<'_, Tibs>>() {
            return Ok(self.as_bitslice() == other.as_bitslice());
        }
        if let Ok(other) = other.extract::<PyRef<'_, Mutibs>>() {
            return Ok(self.as_bitslice() == other.as_bitslice());
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
    /// :return: A list of bit positions.
    ///
    /// :raises ValueError: if needle is empty, if start or end are out of range or if end is before start.
    ///
    /// All occurrences of needle are found, even if they overlap.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b10111011').find_all('0b11')
    ///     [2, 3, 6]
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn find_all(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
    ) -> PyResult<Vec<u64>> {
        if needle.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to find."));
        }

        let haystack_len = slf.len();
        let (start, end) = validate_slice(haystack_len, start, end)?;

        helpers::collect_find_all_positions(
            py,
            slf.as_bitslice(),
            needle.as_bitslice(),
            start,
            end,
            byte_aligned,
        )
    }

    /// Find all occurrences of a bit sequence, returning an iterator of bit positions.
    ///
    /// :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position of the slice to search. Defaults to 0.
    /// :param int | None end: The end bit position of the slice to search. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries. Defaults to ``False``.
    /// :return: A generator yielding bit positions.
    ///
    /// :raises ValueError: if needle is empty, if start or end are out of range or if end is before start.
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
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn find_all_iter(
        slf: PyRef<'_, Self>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
    ) -> PyResult<Py<FindAllIterator>> {
        FindAllIterator::new(slf, needle, start, end, byte_aligned, false)
    }

    /// Find all occurrences of a bit sequence in reverse, returning an iterator of bit positions.
    ///
    /// :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position of the slice to search. Defaults to 0.
    /// :param int | None end: The end bit position of the slice to search. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries. Defaults to ``False``.
    /// :return: A generator yielding bit positions.
    ///
    /// :raises ValueError: if needle is empty, if start or end are out of range or end is before start.
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
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn rfind_all_iter(
        slf: PyRef<'_, Self>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
    ) -> PyResult<Py<FindAllIterator>> {
        FindAllIterator::new(slf, needle, start, end, byte_aligned, true)
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
    #[pyo3(signature = (dtype, start = None, end = None), text_signature = "($self, dtype, start=None, end=None)")]
    pub fn to_values_iter(
        slf: PyRef<'_, Self>,
        dtype: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<ValuesIterator>> {
        let dtype = extract_dtype(dtype)?;
        let (start, end) = validate_slice(slf.len(), start, end)?;
        let selected_len = end - start;
        let chunk_size = dtype.length;
        if !selected_len.is_multiple_of(chunk_size) {
            return Err(PyValueError::new_err(format!(
                "Cannot create values iterator - selected length of {selected_len} bits is not a multiple of dtype length {} bits.",
                dtype.length
            )));
        }

        let py = slf.py();
        Py::new(
            py,
            ValuesIterator {
                bits_object: slf.into(),
                dtype_kind: dtype.kind,
                dtype_length: dtype.length,
                byte_order: dtype.byte_order,
                chunk_size,
                current_pos: start,
                end_pos: end,
            },
        )
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
    #[pyo3(signature = (dtype, start = None, end = None), text_signature = "($self, dtype, start=None, end=None)")]
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
    #[pyo3(signature = (dtype, start = None, end = None), text_signature = "($self, dtype, start=None, end=None)")]
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
    pub fn from_string(_cls: &Bound<'_, PyType>, s: String) -> PyResult<Self> {
        let bv = str_to_bv(s)?;
        Ok(Tibs::from_bv(bv))
    }

    /// Create a new instance from an unsigned integer.
    ///
    /// :param int u: An unsigned integer.
    /// :param int length: The bit length to create. Can be up to 128.
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
        u: u128,
        length: i64,
        byte_order: Option<ByteOrder>,
    ) -> PyResult<Self> {
        let length = validate_length(length)?;
        let is_little_endian = ByteOrder::is_little_endian(byte_order, length)?;
        Ok(Tibs::from_bv(bv_from_u128(u, length, is_little_endian)?))
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
    pub fn to_u(&self, start: Option<isize>, end: Option<isize>) -> PyResult<u128> {
        self.map_slice(start, end, |bits| BitCollection::to_u128(bits, false))
    }

    /// Read-only property of the unsigned integer representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_u` with no parameters.
    ///
    /// :return: The value as an unsigned integer.
    #[getter]
    fn u(&self) -> PyResult<u128> {
        self.to_u(None, None)
    }

    /// Create a new instance from a signed integer.
    ///
    /// :param int i: A signed integer.
    /// :param int length: The bit length to create. Can be up to 128.
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
        i: i128,
        length: i64,
        byte_order: Option<ByteOrder>,
    ) -> PyResult<Self> {
        let length = validate_length(length)?;
        let is_little_endian = ByteOrder::is_little_endian(byte_order, length)?;
        Ok(Tibs::from_bv(bv_from_i128(i, length, is_little_endian)?))
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
    pub fn to_i(&self, start: Option<isize>, end: Option<isize>) -> PyResult<i128> {
        self.map_slice(start, end, |bits| BitCollection::to_i128(bits, false))
    }

    /// Read-only property of the signed integer representation of the Tibs.
    ///
    /// Equivalent to using :meth:`~to_i` with no parameters.
    ///
    /// :return: The value as a signed integer.
    #[getter]
    fn i(&self) -> PyResult<i128> {
        self.to_i(None, None)
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
            Some(offset) => Some(validate_length(offset)?),
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
        seed: Option<Vec<u8>>,
    ) -> PyResult<Self> {
        let bv = bv_from_random(length, secure, &seed)?;
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
    /// :return: The bit position if found, or None if not found.
    ///
    /// :raises ValueError: if ``needle`` is empty, or if the slice parameters are invalid.
    ///
    /// .. code-block:: pycon
    ///
    ///      >>> Tibs('0xc3e').find('0b1111')
    ///      6
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn find(
        &self,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
    ) -> PyResult<Option<usize>> {
        self.find_impl(py, needle, start, end, byte_aligned, false)
    }

    /// Return True if b is a sub-sequence of self.
    pub fn __contains__(&self, py: Python<'_>, b: Tibs) -> PyResult<bool> {
        self.find(py, b, None, None, false)
            .map(|found| found.is_some())
    }

    /// As Tibs is immutable, this returns the same instance.
    pub fn __copy__(slf: PyRef<'_, Self>) -> Py<Self> {
        slf.into()
    }

    /// Find last occurrence of a bit sequence.
    ///
    /// Returns the bit position if found, or None if not found.
    ///
    /// :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries.
    /// :return: The bit position if found, or None if not found.
    ///
    /// :raises ValueError: if ``needle`` is empty, or if the slice parameters are invalid.
    ///
    /// .. code-block:: pycon
    ///
    ///      >>> Tibs('0b10111011').rfind('0b11')
    ///      6
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn rfind(
        &self,
        py: Python<'_>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
    ) -> PyResult<Option<usize>> {
        self.find_impl(py, needle, start, end, byte_aligned, true)
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
    pub fn starts_with(&self, prefix: Tibs) -> PyResult<bool> {
        Ok(<Tibs as BitCollection>::starts_with(self, prefix))
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
    pub fn ends_with(&self, suffix: Tibs) -> PyResult<bool> {
        Ok(<Tibs as BitCollection>::ends_with(self, suffix))
    }

    /// Counts the total number of occurrences of a bit pattern.
    ///
    /// :param object value: Either something that can be converted to a ``Tibs``, or a single bit (one of ``0``, ``1``, ``False`` or ``True``).
    /// :param int | None start: The start of the slice to count within. Defaults to 0.
    /// :param int | None end: The end of the slice to count within. Defaults to len(self).
    ///
    /// :return: The number of times the bit pattern is found.
    /// :raises ValueError: if the slice parameters are invalid.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0xef').count(1)
    ///     7
    ///     >>> Tibs('0xef').count(1, 0, 4)
    ///     3
    ///     >>> Tibs.from_bin('0011010101100').count('0b01')
    ///     4
    ///
    #[pyo3(signature = (value, start=None, end=None), text_signature = "($self, value, start=None, end=None)")]
    pub fn count(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<usize> {
        let (start, end) = validate_slice(self.len(), start, end)?;
        let haystack = &self.as_bitslice()[start..end];

        if let Some(b) = helpers::convert_to_bool(value) {
            return Ok(count_bitslice(haystack, b));
        }

        match Tibs::extract(value.as_borrowed()) {
            Ok(v) => {
                if v.len() == 1 {
                    Ok(count_bitslice(haystack, v.get_index(0)?))
                } else {
                    helpers::count_bitvec(py, haystack, v.as_bitslice())
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
    ///     >>> Tibs('0b1111').all()
    ///     True
    ///     >>> Tibs('0b1011').all()
    ///     False
    ///
    #[inline]
    pub fn all(&self) -> bool {
        self.to_bitslice().all()
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
        self.to_bitslice().any()
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
    pub fn set_at(&self, pos: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.copy_with_mutation(|out| out.apply_set_positions(true, pos))
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
    pub fn unset_at(&self, pos: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.copy_with_mutation(|out| out.apply_set_positions(false, pos))
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
    #[pyo3(signature = (pos = None), text_signature = "($self, pos=None)")]
    pub fn inverted(&self, pos: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        self.copy_with_mutation(|out| out.apply_invert_positions(pos))
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
        self.copy_with_mutation(|out| out.apply_insert_bits(pos, &bs))
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
    /// :return: A new Tibs.
    /// :raises ValueError: if old is empty, count is negative or the slice parameters are invalid.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0b00010010').replaced([0, 1], [1, 1, 1])
    ///     Tibs('0b0011101110')
    ///
    #[pyo3(signature = (old, new, start=None, end=None, count=None, byte_aligned=false), text_signature = "($self, old, new, start=None, end=None, count=None, byte_aligned=False)")]
    pub fn replaced(
        &self,
        py: Python<'_>,
        old: &Bound<'_, PyAny>,
        new: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
        count: Option<i64>,
        byte_aligned: bool,
    ) -> PyResult<Self> {
        let old = Tibs::extract(old.as_borrowed())?;
        let new = Tibs::extract(new.as_borrowed())?;
        self.copy_with_mutation(move |out| {
            out.apply_replace_bits(py, old, new, start, end, count, byte_aligned)?;
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
        Ok(self.lshift(shift))
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
        Ok(self.rshift(shift))
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
        Ok(Tibs::from_bv(concatenate_bitcollections(self, &other)))
    }

    /// Concatenates two Tibs and return a newly constructed Tibs.
    ///
    /// :param object other: The bits to prepend. This can be anything promotable to ``Tibs``.
    /// :return: A new Tibs.
    ///
    pub fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        Ok(Tibs::from_bv(concatenate_bitcollections(&other, self)))
    }

    /// Bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: A new Tibs.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        if self.shares_view_with(&other) {
            return Ok(self.clone());
        }
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_and(self, &other))
    }

    /// Bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: A new Tibs.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        if self.shares_view_with(&other) {
            return Ok(self.clone());
        }
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_or(self, &other))
    }

    /// Bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// :param object other: The other bits. This can be anything promotable to ``Tibs``.
    /// :return: A new Tibs.
    /// :raises ValueError: if the two Tibs have differing lengths.
    ///
    pub fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;

        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_xor(self, &other))
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
    #[pyo3(signature = (n, start=None, end=None), text_signature = "($self, n, start=None, end=None)")]
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
    #[pyo3(signature = (n, start=None, end=None), text_signature = "($self, n, start=None, end=None)")]
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
    /// :raises ValueError: for badly formed, truncated or extended input bytes.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs.decode(Tibs('0b101').encode())
    ///     Tibs('0b101')
    ///
    #[classmethod]
    #[pyo3(signature = (b, /), text_signature = "(cls, b, /)")]
    pub fn decode(_cls: &Bound<'_, PyType>, b: Vec<u8>) -> PyResult<Tibs> {
        tibs_codec::decode_bytes::<Tibs>(b)
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
    /// :raises ValueError: if the Tibs is empty.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> ~Tibs('0b10110')
    ///     Tibs('0b01001')
    ///
    pub fn __invert__(&self) -> PyResult<Self> {
        if self.to_bitslice().is_empty() {
            return Err(PyValueError::new_err("Cannot invert empty Tibs."));
        }
        Ok(BitCollection::invert_copy(self))
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
        Ok(self.multiply(n as usize))
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
