use crate::core::BitCollection;
use crate::enums::{BitIndexing, Codec, Endianness};
use crate::helpers::{
    BS, BV, bv_from_bin, bv_from_bools, bv_from_bytes_slice, bv_from_f64, bv_from_hex,
    bv_from_i128, bv_from_oct, bv_from_ones, bv_from_random, bv_from_u128, bv_from_zeros,
    byte_aligned_physical_offset, find_bitvec, find_bitvec_aligned, logical_range_to_physical,
    physical_match_to_logical_start, promote_to_bv, str_to_bv, validate_index,
    validate_logical_op_lengths, validate_shift, validate_slice,
};
use crate::tibs_::Tibs;

use crate::helpers;
use pyo3::exceptions::{PyAttributeError, PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PySlice, PyType};
use std::ops::{Deref, Not};

///     A mutable container of binary data.
///
///     To construct, use a builder 'from' method:
///
///     * ``Mutibs.from_bin(s)`` - Create from a binary string, optionally starting with '0b'.
///     * ``Mutibs.from_oct(s)`` - Create from an octal string, optionally starting with '0o'.
///     * ``Mutibs.from_hex(s)`` - Create from a hex string, optionally starting with '0x'.
///     * ``Mutibs.from_u(u, length, [endianness])`` - Create from an unsigned int to a given length.
///     * ``Mutibs.from_i(i, length, [endianness])`` - Create from a signed int to a given length.
///     * ``Mutibs.from_f(f, length, [endianness])`` - Create from an IEEE float to a 16, 32 or 64 bit length.
///     * ``Mutibs.from_bytes(b)`` - Create directly from a ``bytes`` or ``bytearray`` object.
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
#[derive(Clone)]
pub struct Mutibs {
    pub data: BV,
    pub msb0: bool,
}

// Internal methods, not exported to Python
impl Mutibs {
    pub(crate) fn from_bv(bv: BV, msb0: bool) -> Self {
        Mutibs { data: bv, msb0 }
    }

    #[inline]
    pub(crate) fn as_bitvec_ref(&self) -> &BV {
        &self.data
    }

    #[inline]
    pub(crate) fn as_bitslice(&self) -> &BS {
        self.as_bitvec_ref().as_bitslice()
    }

    #[inline]
    pub(crate) fn to_bitvec(&self) -> BV {
        // Materialize a single owned copy of the current logical view.
        self.as_bitvec_ref().to_bitvec()
    }

    #[inline]
    pub(crate) fn as_mut_bitvec_ref(&mut self) -> &mut BV {
        &mut self.data
    }

    #[inline]
    pub(crate) fn raw_bytes(&self) -> Vec<u8> {
        self.data.as_raw_slice().to_vec()
    }

    #[inline]
    pub fn set_index(&mut self, index: isize) -> PyResult<()> {
        self.set_from_sequence(true, vec![index])
    }

    #[inline]
    pub fn unset_index(&mut self, index: isize) -> PyResult<()> {
        self.set_from_sequence(false, vec![index])
    }

    pub(crate) fn set_slice(&mut self, start: usize, end: usize, value: &BS) {
        if start >= end {
            // This is an insertion in Python
            let tail = self.as_mut_bitvec_ref().split_off(start);
            self.as_mut_bitvec_ref().extend_from_bitslice(value);
            self.as_mut_bitvec_ref().extend_from_bitslice(&tail);
        } else if end - start == value.len() {
            // This is an overwrite, so no need to move data around.
            self.as_mut_bitvec_ref()[start..start + value.len()].copy_from_bitslice(value);
        } else {
            let tail = self.as_mut_bitvec_ref().split_off(end);
            self.as_mut_bitvec_ref().truncate(start);
            self.as_mut_bitvec_ref().extend_from_bitslice(value);
            self.as_mut_bitvec_ref().extend_from_bitslice(&tail);
        }
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

    pub(crate) fn set_from_sequence(&mut self, value: bool, indices: Vec<isize>) -> PyResult<()> {
        let mut validated = Vec::with_capacity(indices.len());
        for idx in indices {
            validated.push(validate_index(idx, self.len(), self.msb0)?);
        }
        for idx in validated {
            self.as_mut_bitvec_ref().set(idx, value);
        }
        Ok(())
    }

    pub(crate) fn apply_set_positions(
        &mut self,
        value: bool,
        pos: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if let Ok(index) = pos.extract::<isize>() {
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
            let indices = pos.extract::<Vec<isize>>()?;
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
        if n < 0 {
            return Err(PyValueError::new_err("Cannot rotate by a negative amount."));
        }

        let (start, end) = validate_slice(self.len(), start, end)?;
        if start != end {
            let n = (n % (end as i64 - start as i64)) as usize;
            if rotate_left {
                self.as_mut_bitvec_ref()[start..end].rotate_left(n);
            } else {
                self.as_mut_bitvec_ref()[start..end].rotate_right(n);
            }
        }
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
                    let pos: usize = validate_index(pos, self.len(), self.msb0)?;
                    let value = self.as_bitvec_ref()[pos];
                    self.as_mut_bitvec_ref().set(pos, !value);
                } else if let Ok(pos_list) = p.extract::<Vec<isize>>() {
                    for pos in pos_list {
                        let pos: usize = validate_index(pos, self.len(), self.msb0)?;
                        let value = self.as_bitvec_ref()[pos];
                        self.as_mut_bitvec_ref().set(pos, !value);
                    }
                } else {
                    return Err(PyTypeError::new_err(
                        "invert() argument must be an integer, an iterable of ints, or None",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn apply_replace_bits(
        &mut self,
        old: Tibs,
        new: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        count: Option<i64>,
        byte_aligned: bool,
    ) -> PyResult<()> {
        if old.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to replace."));
        }

        let len = self.len();
        let (start, end) = validate_slice(len, start, end)?;
        let (search_old, replace_new) = if self.msb0 {
            (old, new)
        } else {
            (
                BitCollection::reverse_copy(&old),
                BitCollection::reverse_copy(&new),
            )
        };
        let (search_start, search_end) = logical_range_to_physical(len, start, end, self.msb0);
        let mut countdown = count.unwrap_or(i64::MAX);
        if countdown < 0 {
            return Err(PyValueError::new_err(format!(
                "The count in replace() should not be negative. Received {}.",
                countdown
            )));
        }

        let mut starting_points: Vec<usize> = Vec::new();
        if self.msb0 {
            let mut current_pos = search_start;
            while current_pos < search_end && countdown > 0 {
                if let Some(found_pos) = find_bitvec(
                    self.as_bitvec_ref(),
                    search_old.as_bitslice(),
                    current_pos,
                    search_end,
                    byte_aligned,
                ) {
                    starting_points.push(found_pos);
                    current_pos = found_pos + search_old.len();
                    countdown -= 1;
                } else {
                    break;
                }
            }
        } else {
            let mut current_end = search_end;
            while current_end > search_start && countdown > 0 {
                if let Some(found_pos) = helpers::rfind_bitvec(
                    self.as_bitvec_ref(),
                    search_old.as_bitslice(),
                    search_start,
                    current_end,
                    byte_aligned,
                ) {
                    starting_points.push(found_pos);
                    current_end = found_pos;
                    countdown -= 1;
                } else {
                    break;
                }
            }
        }

        if starting_points.is_empty() {
            return Ok(());
        }

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
        let logical_pos = pos as usize;
        let insert_pos = if self.msb0 {
            logical_pos
        } else {
            self.len() - logical_pos
        };
        if bs.len() == 1 {
            self.as_mut_bitvec_ref()
                .insert(insert_pos, bs.as_bitslice()[0]);
            return Ok(());
        }
        let tail = self.as_mut_bitvec_ref().split_off(insert_pos);
        self.as_mut_bitvec_ref()
            .extend_from_bitslice(bs.as_bitslice());
        self.as_mut_bitvec_ref().extend_from_bitslice(&tail);
        Ok(())
    }

    pub(crate) fn find_impl(
        &self,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        reverse: bool,
    ) -> PyResult<Option<usize>> {
        if needle.is_empty() {
            return Err(PyValueError::new_err("No bits were provided to find."));
        }
        let len = self.len();
        let (start, end) = validate_slice(len, start, end)?;
        let needle = if self.msb0 {
            needle
        } else {
            BitCollection::reverse_copy(&needle)
        };
        let (start, end) = logical_range_to_physical(len, start, end, self.msb0);
        let alignment_mod8 = if byte_aligned {
            Some(byte_aligned_physical_offset(len, needle.len(), self.msb0))
        } else {
            None
        };

        let use_find = self.msb0 ^ reverse;
        let found = if use_find {
            find_bitvec_aligned(
                self.as_bitvec_ref(),
                needle.as_bitslice(),
                start,
                end,
                alignment_mod8,
            )
        } else {
            helpers::rfind_bitvec_aligned(
                self.as_bitvec_ref(),
                needle.as_bitslice(),
                start,
                end,
                alignment_mod8,
            )
        };
        Ok(found.map(|pos| physical_match_to_logical_start(len, needle.len(), pos, self.msb0)))
    }

    pub(crate) fn set_from_slice(
        &mut self,
        value: bool,
        start: isize,
        stop: isize,
        step: isize,
    ) -> PyResult<()> {
        let len = self.len() as isize;
        if len == 0 {
            return Ok(());
        }
        let positive_start = if start < 0 { start + len } else { start };
        // For negative steps, Python ranges use stop=-1 as an exclusive sentinel
        // so that index 0 is included (e.g. range(3, -1, -1) -> 3,2,1,0).
        let positive_stop = if step < 0 && stop == -1 {
            -1
        } else if stop < 0 {
            stop + len
        } else {
            stop
        };
        if positive_start < 0 || positive_start >= len {
            return Err(PyIndexError::new_err("Start of slice out of bounds."));
        }
        if (step > 0 && (positive_stop < 0 || positive_stop > len))
            || (step < 0 && (positive_stop < -1 || positive_stop >= len))
        {
            return Err(PyIndexError::new_err("End of slice out of bounds."));
        }
        if step == 0 {
            return Err(PyValueError::new_err("Step cannot be zero."));
        }
        // after your existing start/stop/step validation:
        let len_isize = self.len() as isize;
        let len_usize = self.len();
        let msb0 = self.msb0;
        let mut i = positive_start;

        // Contiguous fast paths
        if step == 1 {
            let bv = self.as_mut_bitvec_ref();
            if msb0 {
                bv[positive_start as usize..positive_stop as usize].fill(value);
            } else {
                // logical [start, stop) -> physical [len-stop, len-start)
                let a = len_usize - positive_stop as usize;
                let b = len_usize - positive_start as usize;
                bv[a..b].fill(value);
            }
            return Ok(());
        }
        if step == -1 {
            // logical i = start, start-1, ..., stop+1
            let bv = self.as_mut_bitvec_ref();
            if msb0 {
                bv[(positive_stop + 1) as usize..(positive_start + 1) as usize].fill(value);
            } else {
                // mapped contiguous region in physical space
                let a = len_usize - (positive_start as usize) - 1;
                let b = len_usize - (positive_stop as usize) - 1;
                bv[a..b].fill(value);
            }
            return Ok(());
        }
        // General strided path
        let bv = self.as_mut_bitvec_ref();
        if step > 0 {
            while i < positive_stop {
                debug_assert!(i >= 0 && i < len_isize);
                let p = if msb0 {
                    i as usize
                } else {
                    len_usize - 1 - i as usize
                };
                unsafe { bv.set_unchecked(p, value) };
                i += step;
            }
        } else {
            while i > positive_stop {
                debug_assert!(i >= 0 && i < len_isize);
                debug_assert!(step < 0);
                let p = if msb0 {
                    i as usize
                } else {
                    len_usize - 1 - i as usize
                };
                unsafe { bv.set_unchecked(p, value) };
                i += step; // step < 0
            }
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
        // Default to msb0 when creating from other types.
        let bv = promote_to_bv(&obj)?;
        Ok(Mutibs::from_bv(bv, true))
    }
}

#[pymethods]
impl Mutibs {
    #[new]
    #[pyo3(signature = (auto = None, bit_indexing = BitIndexing::Msb0), text_signature = "(auto=None, bit_indexing=BitIndexing.Msb0)")]
    pub fn py_new(
        auto: Option<&Bound<'_, PyAny>>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let Some(auto) = auto else {
            return Ok(BitCollection::empty(msb0));
        };
        let mut mutibs = Mutibs::extract(auto.as_borrowed())?;
        mutibs.msb0 = msb0;
        Ok(mutibs)
    }

    /// Whether the bits are indexed from the most significant bit (BitIndexing.Msb0, the default) or from the
    /// least significant bit (BitIndexing.Lsb0). This doesn't affect the actual data stored, just how it's
    /// accessed.
    #[getter]
    pub fn bit_indexing(&self) -> BitIndexing {
        if self.msb0 {
            BitIndexing::Msb0
        } else {
            BitIndexing::Lsb0
        }
    }

    #[setter]
    pub fn set_bit_indexing(&mut self, val: BitIndexing) {
        self.msb0 = match val {
            BitIndexing::Msb0 => true,
            BitIndexing::Lsb0 => false,
        }
    }

    /// Return True if two Mutibs have the same binary representation.
    ///
    /// The right hand side will be promoted to a Mutibs if needed and possible.
    ///
    /// >>> Mutibs('0xf2') == '0b11110010'
    /// True
    ///
    pub fn __eq__(&self, other: Tibs) -> bool {
        *self.as_bitvec_ref() == *other.as_bitslice()
    }

    /// Return string representations for printing.
    pub fn __str__(&self) -> String {
        self.to_string()
    }

    /// Return representation that could be used to recreate the instance.
    pub fn __repr__(&self) -> String {
        if self.is_empty() {
            let bit_indexing = if self.msb0 {
                "".to_string()
            } else {
                "bit_indexing=BitIndexing.Lsb0".to_string()
            };
            format!("Mutibs({})", bit_indexing)
        } else {
            let bit_indexing = if self.msb0 {
                "".to_string()
            } else {
                ", BitIndexing.Lsb0".to_string()
            };
            format!("Mutibs('{}'{})", self.__str__(), bit_indexing)
        }
    }

    /// Create a new instance from a formatted string.
    ///
    /// This method initializes a new instance of :class:`Mutibs` using a formatted string.
    ///
    /// :param str s: The formatted string to convert. This can begin with '0b', '0o' or '0x' to indicate binary, octal or hexadecimal, and commas can be used to separate items.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
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
    #[pyo3(signature = (s, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, s, /, bit_indexing=BitIndexing.Msb0)"
    )]
    pub fn from_string(
        _cls: &Bound<'_, PyType>,
        s: String,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = str_to_bv(s)?;
        Ok(Mutibs::from_bv(bv, msb0))
    }

    /// Create a new instance from a binary string.
    ///
    /// :param str s: A string of ``0`` and ``1`` s, optionally preceded with ``0b`` and optionally containing underscores.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_bin("0000_1111_0101")
    ///
    #[classmethod]
    #[pyo3(signature = (s, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, s, /, bit_indexing=BitIndexing.Msb0)"
    )]
    pub fn from_bin(
        _cls: &Bound<'_, PyType>,
        s: String,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = bv_from_bin(&s)?;
        Ok(Mutibs::from_bv(bv, msb0))
    }

    /// Return the binary representation of the Mutibs as a string.
    ///
    /// Equivalent to using the ``bin`` property.
    ///
    /// :return: The binary representation.
    pub fn to_bin(&self) -> String {
        BitCollection::to_binary(self)
    }

    /// Read-only property of the binary representation of the Mutibs.
    ///
    /// Equivalent to using :meth:`~to_bin`.
    ///
    /// :return: The binary representation.
    #[getter]
    fn bin(&self) -> String {
        BitCollection::to_binary(self)
    }

    /// Create a new instance from an octal string.
    ///
    /// :param str s: A string of octal digits, optionally preceded with ``0o`` and optionally containing underscores.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    #[classmethod]
    #[pyo3(signature = (s, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, s, /, bit_indexing=BitIndexing.Msb0)"
    )]
    pub fn from_oct(
        _cls: &Bound<'_, PyType>,
        s: String,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = bv_from_oct(&s)?;
        Ok(Mutibs::from_bv(bv, msb0))
    }

    /// Return the octal representation of the Mutibs as a string.
    ///
    /// Equivalent to using the ``oct`` property.
    ///
    /// :return: The octal representation.
    /// :raises ValueError: if the length is not a multiple of 3.
    pub fn to_oct(&self) -> PyResult<String> {
        BitCollection::to_octal(self)
    }

    /// Read-only property of the octal representation of the Mutibs.
    ///
    /// Equivalent to using :meth:`~to_oct`.
    ///
    /// :return: The octal representation.
    /// :raises ValueError: if the length is not a multiple of 3.
    #[getter]
    fn oct(&self) -> PyResult<String> {
        BitCollection::to_octal(self)
    }

    /// Create a new instance from a hexadecimal string.
    ///
    /// Equivalent to using the ``hex`` property.
    ///
    /// :param str s: A string of hexadecimal digits, optionally preceded with ``0x`` and optionally containing underscores.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    #[classmethod]
    #[pyo3(signature = (s, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, s, /, bit_indexing=BitIndexing.Msb0)"
    )]
    pub fn from_hex(
        _cls: &Bound<'_, PyType>,
        s: String,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = bv_from_hex(&s)?;
        Ok(Mutibs::from_bv(bv, msb0))
    }

    /// Return the hexadecimal representation of the Mutibs as a string.
    ///
    /// :return: The hexadecimal representation.
    /// :raises ValueError: if the length is not a multiple of 4.
    pub fn to_hex(&self) -> PyResult<String> {
        BitCollection::to_hexadecimal(self)
    }

    /// Read-only property of the hexadecimal representation of the Mutibs.
    ///
    /// Equivalent to using :meth:`~to_hex`.
    ///
    /// :return: The hexadecimal representation.
    /// :raises ValueError: if the length is not a multiple of 4.
    #[getter]
    fn hex(&self) -> PyResult<String> {
        BitCollection::to_hexadecimal(self)
    }

    /// Return the Mutibs as a bytes object.
    ///
    /// :return: The bytes representation.
    /// :raises ValueError: if the length is not a multiple of 8.
    pub fn to_bytes(&self) -> PyResult<Vec<u8>> {
        BitCollection::to_byte_data(self)
    }

    /// Read-only property of the ``bytes`` representation of the Mutibs.
    ///
    /// Equivalent to using :meth:`~to_bytes`.
    ///
    /// :return: The bytes representation.
    /// :raises ValueError: if the length is not a multiple of 8.
    #[getter]
    fn bytes(&self) -> PyResult<Vec<u8>> {
        BitCollection::to_byte_data(self)
    }

    /// Return a copy of the raw byte information.
    ///
    /// This returns the underlying byte data and can contain leading and trailing
    /// bits that are not considered part of the object's value. Usually using
    /// :meth:`~to_bytes` is what you really need.
    ///
    /// The way that the data is stored is not considered part of the public interface
    /// and so the output of this method may change between point releases, and even
    /// during the running of a program.
    ///
    /// See also :meth:`~as_raw_data` which moves the byte data instead of copying it.
    ///
    /// :return: A tuple of the raw bytes, the bit offset and the bit length.
    ///
    /// .. code-block:: python
    ///
    ///     raw_bytes, offset, length = t.to_raw_data()
    ///     assert t == Mutibs.from_bytes(raw_bytes)[offset:offset + length]
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
    /// The way that the data is stored is not considered part of the public interface
    /// and so the output of this method may change between point releases, and even
    /// during the running of a program.
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
    /// :param int length: The bit length to create. Can be up to 128.
    /// :param Endianness endianness: The byte endianness used to store the integer. Defaults to Endianness.Unspecified.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// :raises ValueError: if the integer doesn't fit in the length given.
    ///
    #[classmethod]
    #[pyo3(signature = (u, /, length, endianness = Endianness::Unspecified, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, u, /, length, endianness=Endianness.Unspecified, bit_indexing=BitIndexing.Msb0)")]
    pub fn from_u(
        _cls: &Bound<'_, PyType>,
        u: u128,
        length: i64,
        endianness: Option<Endianness>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let is_little_endian = Endianness::is_little_endian(endianness, length as usize)?;
        let bv = bv_from_u128(u, length, is_little_endian)?;
        Ok(Mutibs::from_bv(bv, msb0))
    }

    /// Return the unsigned integer representation of the Mutibs.
    ///
    /// :param Endianness endianness: The byte endianness used to interpret the integer. Defaults to Endianness.Unspecified.
    #[pyo3(signature = (endianness = Endianness::Unspecified), text_signature = "($self, endianness=Endianness.Unspecified)")]
    pub fn to_u(&self, endianness: Option<Endianness>) -> PyResult<u128> {
        let is_little_endian = Endianness::is_little_endian(endianness, self.len())?;
        BitCollection::to_u128(self, is_little_endian)
    }

    /// Create a new instance from a signed integer.
    ///
    /// :param int i: A signed integer.
    /// :param int length: The bit length to create. Can be up to 128.
    /// :param Endianness endianness: The byte endianness used to store the integer. Defaults to Endianness.Unspecified.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// :raises ValueError: if the integer doesn't fit in the length given.
    ///
    #[classmethod]
    #[pyo3(signature = (i, /, length, endianness = Endianness::Unspecified, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, i, /, length, endianness=Endianness.Unspecified, bit_indexing=BitIndexing.Msb0)")]
    pub fn from_i(
        _cls: &Bound<'_, PyType>,
        i: i128,
        length: i64,
        endianness: Option<Endianness>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let is_little_endian = Endianness::is_little_endian(endianness, length as usize)?;
        let bv = bv_from_i128(i, length, is_little_endian)?;
        Ok(Mutibs::from_bv(bv, msb0))
    }

    /// Return the signed integer representation of the Mutibs.
    ///
    /// :param Endianness endianness: The byte endianness used to interpret the integer. Defaults to Endianness.Unspecified.
    #[pyo3(signature = (endianness = Endianness::Unspecified), text_signature = "($self, endianness=Endianness.Unspecified)")]
    pub fn to_i(&self, endianness: Option<Endianness>) -> PyResult<i128> {
        let is_little_endian = Endianness::is_little_endian(endianness, self.len())?;
        BitCollection::to_i128(self, is_little_endian)
    }

    /// Create a new instance from a floating point number.
    ///
    /// :param float f: A floating point value.
    /// :param int length: The bit length to create. Must be 16, 32 or 64.
    /// :param Endianness endianness: The byte endianness used to store the float. Defaults to Endianness.Unspecified.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    #[classmethod]
    #[pyo3(signature = (f, /, length, endianness = Endianness::Unspecified, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, f, /, length, endianness=Endianness.Unspecified, bit_indexing=BitIndexing.Msb0)")]
    pub fn from_f(
        _cls: &Bound<'_, PyType>,
        f: f64,
        length: i64,
        endianness: Option<Endianness>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let is_little_endian = Endianness::is_little_endian(endianness, length as usize)?;
        let bv = bv_from_f64(f, length, is_little_endian)?;
        Ok(Mutibs::from_bv(bv, msb0))
    }

    /// Return the floating point representation of the Mutibs.
    ///
    /// The length must be 16, 32 or 64.
    ///
    /// :param Endianness endianness: The byte endianness used to interpret the float. Defaults to Endianness.Unspecified.
    #[pyo3(signature = (endianness = Endianness::Unspecified), text_signature = "($self, endianness=Endianness.Unspecified)")]
    pub fn to_f(&self, endianness: Option<Endianness>) -> PyResult<f64> {
        let is_little_endian = Endianness::is_little_endian(endianness, self.len())?;
        BitCollection::to_f64(self, is_little_endian)
    }

    /// Create a new instance with all bits set to zero.
    ///
    /// :param int length: The number of bits to set.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    /// :return: A Mutibs object with all bits set to zero.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_zeros(500)  # 500 zero bits
    ///
    #[classmethod]
    #[pyo3(signature = (length, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, length, /, bit_indexing=BitIndexing.Msb0)")]
    pub fn from_zeros(
        _cls: &Bound<'_, PyType>,
        length: i64,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        if length < 0 {
            return Err(PyValueError::new_err(format!(
                "Negative bit length given: {}.",
                length
            )));
        }
        Ok(Self::from_bv(bv_from_zeros(length as usize), msb0))
    }

    /// Create a new instance with all bits set to one.
    ///
    /// :param int length: The number of bits to set.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs.from_ones(5)
    ///     Mutibs('0b11111')
    ///
    #[classmethod]
    #[pyo3(signature = (length, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, length, /, bit_indexing=BitIndexing.Msb0)")]
    pub fn from_ones(
        _cls: &Bound<'_, PyType>,
        length: i64,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        if length < 0 {
            return Err(PyValueError::new_err(format!(
                "Negative bit length given: {}.",
                length
            )));
        }
        Ok(Mutibs::from_bv(bv_from_ones(length as usize), msb0))
    }

    /// Create a new instance from an iterable by converting each element to a bool.
    ///
    /// :param Iterable iterable: The iterable to convert to a :class:`Mutibs`.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_bools([False, 0, 1, "Steven"])  # binary 0011
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, iterable, /, bit_indexing=BitIndexing.Msb0)")]
    pub fn from_bools(
        _cls: &Bound<'_, PyType>,
        iterable: &Bound<'_, PyAny>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = bv_from_bools(iterable)?;
        Ok(Mutibs::from_bv(bv, msb0))
    }

    /// Create a new instance with all bits randomly set.
    ///
    /// :param int length: The number of bits to set. Must be positive.
    /// :param bool secure: If ``True``, use the OS's cryptographically secure generator. Default is ``False``.
    /// :param bytes | bytearray | None seed: A bytes or bytearray to use as an optional seed, only if ``secure`` is ``False``.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
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
    #[pyo3(signature = (length, /, secure=false, seed=None, bit_indexing = BitIndexing::Msb0), text_signature="(cls, length, /, secure=False, seed=None, bit_indexing=BitIndexing.Msb0)")]
    pub fn from_random(
        _cls: &Bound<'_, PyType>,
        length: i64,
        secure: bool,
        seed: Option<Vec<u8>>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = bv_from_random(length, secure, &seed)?;
        Ok(Mutibs::from_bv(bv, msb0))
    }

    /// Create a new instance from a bytes object.
    ///
    /// :param bytes | bytearray | memoryview data: The bytes, bytearray or memoryview object to convert to a :class:`Mutibs`.
    /// :param int | None offset: The bit offset from the start. Defaults to zero.
    /// :param int | None length: The bit length to use. Defaults to the whole of the data.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_bytes(b"some_bytes_maybe_from_a_file")
    ///
    #[classmethod]
    #[inline]
    #[pyo3(signature = (data, /, offset=None, length=None, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, data, /, offset=None, length=None, bit_indexing=BitIndexing.Msb0)")]
    pub fn from_bytes(
        _cls: &Bound<'_, PyType>,
        data: Vec<u8>,
        offset: Option<i64>,
        length: Option<i64>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        let bv = bv_from_bytes_slice(data, offset, length)?;
        Ok(Self::from_bv(bv, msb0))
    }

    /// Create a new instance by concatenating a sequence of Mutibs objects.
    ///
    /// This method concatenates a sequence of Mutibs objects into a single Mutibs object.
    ///
    /// :param Iterable iterable: An iterable to concatenate. Items can be anything that can be promoted to a :class:`Mutibs`.
    /// :param BitIndexing bit_indexing: The bit indexing mode. Defaults to BitIndexing.Msb0.
    ///
    /// .. code-block:: python
    ///
    ///     a = Mutibs.from_joined(['0x01', [1, 0], b'some_bytes'])
    ///
    #[classmethod]
    #[pyo3(signature = (iterable, /, bit_indexing = BitIndexing::Msb0), text_signature = "(cls, iterable, /, bit_indexing=BitIndexing.Msb0)")]
    pub fn from_joined(
        _cls: &Bound<'_, PyType>,
        iterable: &Bound<'_, PyAny>,
        bit_indexing: Option<BitIndexing>,
    ) -> PyResult<Self> {
        let msb0 = BitIndexing::is_msb0(bit_indexing);
        // Collect Tibs handles first so we can preallocate once without BV temporaries.
        let iter = iterable.try_iter()?;
        let mut parts: Vec<Tibs> = Vec::new();
        let mut total_len: usize = 0;
        for item in iter {
            let obj = item?;
            let tibs = Tibs::extract(obj.as_borrowed())?;
            total_len += tibs.len();
            parts.push(tibs);
        }

        let mut bv = BV::with_capacity(total_len);
        for part in parts {
            bv.extend_from_bitslice(part.as_bitslice());
        }
        Ok(Mutibs::from_bv(bv, msb0))
    }

    /// The bit length of the Mutibs.
    pub fn __len__(&self) -> usize {
        self.len()
    }

    /// Get a bit or a slice of bits.
    ///
    /// :param int | slice key: The index or slice to get.
    /// :return: A bool for a single index, or a new Mutibs for a slice.
    /// :raises IndexError: If the index is out of range.
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
            let (start, stop, step) = (
                isize::try_from(indices.start)?,
                isize::try_from(indices.stop)?,
                isize::try_from(indices.step)?,
            );

            let result = if step == 1 {
                if start < stop {
                    self.get_slice(start as usize, (stop - start) as usize)?
                } else {
                    Mutibs::empty(self.msb0)
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
    /// Examples:
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
                Tibs::from_bv(slf.to_bitvec(), slf.msb0)
            } else {
                Tibs::extract(value.as_borrowed())?
            };

            let indices = slice.indices(length as isize)?;
            let start: isize = indices.start.try_into()?;
            let stop: isize = indices.stop.try_into()?;
            let step: isize = indices.step.try_into()?;

            if step == 1 {
                debug_assert!(start >= 0);
                debug_assert!(stop >= 0);
                if slf.msb0 {
                    slf.set_slice(start as usize, stop as usize, tibs.as_bitslice());
                } else {
                    slf.set_slice(
                        length - stop as usize,
                        length - start as usize,
                        tibs.as_bitslice(),
                    );
                }
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
                    // TODO: This validate_index call is overkill just to do the msb0/lsb0.
                    positions.push(validate_index(i, length, slf.msb0)?);
                    i += step;
                }
            } else {
                // TODO: with a negative step I think start or stop could be -1.
                let mut i = start;
                while i > stop {
                    positions.push(validate_index(i, length, slf.msb0)?);
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
            if self.msb0 {
                self.as_mut_bitvec_ref().remove(index as usize);
            } else {
                self.as_mut_bitvec_ref()
                    .remove((length as i64 - index - 1) as usize);
            }
            return Ok(());
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(length as isize)?;
            let start: i64 = indices.start.try_into()?;
            let stop: i64 = indices.stop.try_into()?;
            let step: i64 = indices.step.try_into()?;

            if step == 1 {
                if stop > start {
                    if self.msb0 {
                        self.as_mut_bitvec_ref()
                            .drain(start as usize..stop as usize);
                    } else {
                        self.as_mut_bitvec_ref()
                            .drain(length - start as usize..length - stop as usize);
                    }
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
                // Remove from end of underlying bitvec for both MSB0 and LSB0.
                if self.msb0 {
                    for i in to_remove.into_iter().rev() {
                        self.as_mut_bitvec_ref().remove(i);
                    }
                } else {
                    for i in to_remove.into_iter() {
                        self.as_mut_bitvec_ref().remove(length - i - 1);
                    }
                }
            }
            return Ok(());
        }
        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    /// Return whether the current Mutibs starts with prefix.
    ///
    /// :param Tibs prefix: The bits to search for.
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
    pub fn __contains__(&self, b: Tibs) -> bool {
        match self.find(b, None, None, false) {
            Ok(Some(_)) => true,
            _ => false,
        }
    }

    /// Return whether the current Mutibs ends with suffix.
    ///
    /// :param Tibs suffix: The bits to search for.
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
    /// :param Tibs needle: The Tibs to find.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
    /// :return: The bit position if found, or None if not found.
    ///
    /// .. code-block:: pycon
    ///
    ///      >>> Mutibs('0xc3e').find('0b1111')
    ///      6
    ///
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn find(
        &self,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
    ) -> PyResult<Option<usize>> {
        self.find_impl(needle, start, end, byte_aligned, false)
    }

    /// Bit-wise 'and' between two Mutibs. Returns new Mutibs.
    ///
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_and(self, &other))
    }

    /// Bit-wise 'or' between two Mutibs. Returns new Mutibs.
    ///
    /// :raises ValueError: if the two Mutibs have differing lengths.
    ///
    pub fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_or(self, &other))
    }

    /// Bit-wise 'xor' between two Mutibs. Returns new Mutibs.
    ///
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
    ///     >>> a = Mutibs('0b1011')
    ///     >>> a.rotate_left(2)
    ///     >>> a
    ///     Mutibs('0b1110')
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
    ///     >>> a = Mutibs('0b1011')
    ///     >>> a.rotate_right(1)
    ///     >>> a
    ///     Mutibs('0b1101')
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
    #[pyo3(signature = (n, start=None, end=None), text_signature = "($self, n, start=None, end=None)")]
    pub fn rotated_left(&self, n: i64, start: Option<isize>, end: Option<isize>) -> PyResult<Self> {
        let mut out = self.clone();
        out.apply_rotation(n, start, end, true)?;
        Ok(out)
    }

    /// Return a new Mutibs with the bits rotated to the right.
    ///
    /// This is the non-inplace version of :meth:`rotate_right`.
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
    /// :return: A new Mutibs.
    /// :raises ValueError: for badly formed, truncated or extended input bytes.
    #[classmethod]
    #[pyo3(signature = (b, /), text_signature = "(cls, b, /)")]
    pub fn decode(_cls: &Bound<'_, PyType>, b: Vec<u8>) -> PyResult<Self> {
        <Mutibs as BitCollection>::decode_bytes(b)
    }

    /// Encode the Mutibs as a bytes instance.
    ///
    /// The bytes instance can be used to recreate the Mutibs exactly with :meth:`decode`.
    #[pyo3(signature = (codec=Codec::Auto), text_signature = "($self, codec=Codec.Auto)")]
    pub fn encode(&self, codec: Option<Codec>) -> Vec<u8> {
        <Mutibs as BitCollection>::encode(self, codec)
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
    pub fn unset_at(&self, pos: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut out = self.clone();
        out.apply_set_positions(false, pos)?;
        Ok(out)
    }

    /// Counts the total number of occurrences of a bit pattern.
    ///
    /// :param object value: Either something that can be converted to a ``Tibs``, or a single bit (one of ``0``, ``1``, ``False`` or ``True``).
    ///
    /// :return: The number of times the bit pattern is found.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Mutibs('0xef').count(1)
    ///     7
    ///     >>> Mutibs('0xff00ff').count([1, 1, 1])
    ///     12
    ///
    pub fn count(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        match Tibs::extract(value.as_borrowed()) {
            Ok(v) => {
                if v.len() == 1 {
                    Ok(<Mutibs as BitCollection>::count(self, v.get_index(0)?))
                } else {
                    Ok(helpers::count_bitvec(self.as_bitslice(), v.as_bitslice()))
                }
            }
            Err(_) => {
                let count_ones = helpers::convert_to_bool(value);
                match count_ones {
                    Some(b) => Ok(<Mutibs as BitCollection>::count(self, b)),
                    None => Err(PyValueError::new_err(
                        "Cannot convert value to 0, 1 or a Tibs",
                    )),
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
    /// :param Tibs needle: The bits to find.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
    /// :return: The bit position if found, or None if not found.
    #[pyo3(signature = (needle, start=None, end=None, byte_aligned=false), text_signature = "($self, needle, start=None, end=None, byte_aligned=False)")]
    pub fn rfind(
        &self,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
    ) -> PyResult<Option<usize>> {
        self.find_impl(needle, start, end, byte_aligned, true)
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
    ///     >>> a = Mutibs('0b1011')
    ///     >>> a.reverse()
    ///     >>> a
    ///     Mutibs('0b1101')
    ///
    pub fn reverse(mut slf: PyRefMut<'_, Self>) {
        slf.as_mut_bitvec_ref().reverse();
    }

    /// Return a new instance with the bits reversed.
    ///
    /// :return: Mutibs
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0b00011')
    ///     >>> a.reversed()
    ///     >>> Mutibs('0b11000')
    ///
    pub fn reversed(&self) -> Self {
        BitCollection::reverse_copy(self)
    }

    /// Change the byte endianness in-place.
    ///
    /// The whole of the Mutibs will be byte-swapped. It must be a multiple
    /// of byte_length long.
    ///
    /// :param int | None byte_length: An int giving the number of bytes in each swap, or None (the default)
    ///   to do a single reverse over the whole data.
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0x12345678')
    ///     >>> a.byte_swap(2)
    ///     >>> a
    ///     Mutibs('0x34127856')
    ///
    #[pyo3(signature = (byte_length = None), text_signature = "($self, byte_length=None)")]
    pub fn byte_swap(mut slf: PyRefMut<'_, Self>, byte_length: Option<i64>) -> PyResult<()> {
        // We create a new Mutibs and replace rather than explicitly doing this in-place.
        // If we add a start / end later then this should be made properly in-place.
        *slf = BitCollection::byte_swap_copy(slf.deref(), byte_length)?;
        Ok(())
    }

    /// Return a new instance with the byte endianness swapped.
    ///
    /// The whole of the data will be byte-swapped. It must be a multiple
    /// of byte_length long.
    ///
    /// :param int | None byte_length: An int giving the number of bytes in each swap, or None (the default)
    ///   to do a single reverse over the whole data.
    /// :return: Mutibs
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> a = Mutibs('0x12345678')
    ///     >>> b = a.byte_swapped(2)
    ///     >>> b
    ///     Mutibs('0x34127856')
    ///
    #[pyo3(signature = (byte_length = None), text_signature = "($self, byte_length=None)")]
    pub fn byte_swapped(&self, byte_length: Option<i64>) -> PyResult<Mutibs> {
        Ok(BitCollection::byte_swap_copy(self, byte_length)?)
    }

    /// Return the instance with every bit inverted.
    ///
    /// :raises ValueError: if the Mutibs is empty.
    ///
    pub fn __invert__(&self) -> PyResult<Self> {
        if self.as_bitvec_ref().is_empty() {
            return Err(PyValueError::new_err("Cannot invert empty Mutibs."));
        }
        Ok(Mutibs::from_bv(self.to_bitvec().not(), self.msb0))
    }

    /// Return new Mutibs shifted by n to the left.
    ///
    /// n -- the number of bits to shift. Must be >= 0.
    ///
    pub fn __lshift__(&self, n: i64) -> PyResult<Self> {
        let shift = validate_shift(self, n)?;
        Ok(self.lshift(shift))
    }

    /// Return new Mutibs shifted by n to the right.
    ///
    /// n -- the number of bits to shift. Must be >= 0.
    ///
    pub fn __rshift__(&self, n: i64) -> PyResult<Self> {
        let shift = validate_shift(self, n)?;
        Ok(self.rshift(shift))
    }

    /// Return a new copy of the Mutibs for the copy module.
    pub fn __copy__(&self) -> Self {
        Mutibs::from_bv(self.to_bitvec(), self.msb0)
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
    ///     >>> a = Mutibs('0b1011')
    ///     >>> b = a.to_tibs()
    ///     >>> a
    ///     Mutibs('0b1011')
    ///     >>> b
    ///     Tibs('0b1011')
    ///
    pub fn to_tibs(&self) -> Tibs {
        Tibs::from_bv(self.to_bitvec(), self.msb0)
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
    ///     >>> a = Mutibs('0b1011')
    ///     >>> b = a.as_tibs()
    ///     >>> a
    ///     Mutibs()
    ///     >>> b
    ///     Tibs('0b1011')
    ///
    pub fn as_tibs(&mut self) -> Tibs {
        let mut data = std::mem::take(&mut *self.as_mut_bitvec_ref());
        data.shrink_to_fit();
        Tibs::from_bv(data, self.msb0)
    }

    /// Clear all bits, making the Mutibs empty.
    ///
    /// This doesn't change the allocated capacity, so won't free up any memory.
    ///
    /// :return: None.
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
    pub fn reserve(&mut self, additional: usize) {
        self.as_mut_bitvec_ref().reserve(additional);
    }

    /// Concatenate Mutibs and return a new Mutibs.
    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // We accept the PyAny and convert manually here because if we instead
        // accept a Tibs, then correct types with wrong values (e.g. a malformed string)
        // will fail and return a TypeError instead of ValueError which we can't control.
        let other = Tibs::extract(other.as_borrowed())?;
        let mut data = BV::with_capacity(self.len() + other.len());
        data.extend_from_bitslice(self.as_bitvec_ref());
        data.extend_from_bitslice(other.as_bitslice());
        Ok(Mutibs::from_bv(data, self.msb0))
    }

    /// Concatenate Mutibs and return a new Mutibs.
    pub fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = Tibs::extract(other.as_borrowed())?;
        let mut data = BV::with_capacity(self.len() + other.len());
        data.extend_from_bitslice(other.as_bitslice());
        data.extend_from_bitslice(self.as_bitvec_ref());
        Ok(Mutibs::from_bv(data, self.msb0))
    }

    /// Concatenate in-place.
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
    pub fn pop<'a>(mut slf: PyRefMut<'a, Self>) -> PyResult<bool> {
        match slf.as_mut_bitvec_ref().pop() {
            Some(bit) => Ok(bit),
            None => Err(PyIndexError::new_err("pop from empty Mutibs.")),
        }
    }

    /// Extend the current Mutibs in-place.
    ///
    /// :param Tibs bs: The bits to extend with.
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
            slf.as_mut_bitvec_ref().extend_from_bitslice(&bits_clone);
        } else {
            let bs = Tibs::extract(bs.as_borrowed())?;
            slf.as_mut_bitvec_ref()
                .extend_from_bitslice(bs.as_bitslice());
        }
        Ok(())
    }

    /// Extend the current Mutibs in-place from the start.
    ///
    /// This is broadly equivalent to ``self = bs + self``.
    /// Note that this method is inherently slower than :meth:`extend` and
    /// should be avoided in performance critical code. See also :meth:`from_joined`.
    ///
    /// :param Tibs bs: The bits to prepend to the current Mutibs.
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
        if bs.as_ptr() == slf.as_ptr() {
            let mut new_data = slf.to_bitvec();
            new_data.extend_from_bitslice(slf.as_bitvec_ref());
            *slf.as_mut_bitvec_ref() = new_data;
        } else {
            let to_prepend = Tibs::extract(bs.as_borrowed())?;
            if to_prepend.is_empty() {
                return Ok(());
            }
            let mut new_data = BV::with_capacity(to_prepend.len() + slf.len());
            new_data.extend_from_bitslice(to_prepend.as_bitslice());
            new_data.extend_from_bitslice(slf.as_bitvec_ref());
            *slf.as_mut_bitvec_ref() = new_data;
        }
        Ok(())
    }

    /// Search and replace in-place.
    ///
    /// :param Tibs old: The bits to search for.
    /// :param Tibs new: The bits to replace with.
    /// :param int | None start: The starting bit position. Defaults to 0.
    /// :param int | None end: The end position. Defaults to len(self).
    /// :param int | None count: If present, the maximum number of replacements to make.
    /// :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
    /// :return: None
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> m = Mutibs('0b00010010')
    ///     >>> m.replace([0, 1], [1, 1, 1])
    ///     >>> m
    ///     Mutibs('0b0011101110')
    ///
    #[pyo3(signature = (old, new, start=None, end=None, count=None, byte_aligned=false), text_signature = "($self, old, new, start=None, end=None, count=None, byte_aligned=False)")]
    pub fn replace<'a>(
        mut slf: PyRefMut<'a, Self>,
        old: &Bound<'_, PyAny>,
        new: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
        count: Option<i64>,
        byte_aligned: bool,
    ) -> PyResult<()> {
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
        slf.apply_replace_bits(old, new, start, end, count, byte_aligned)
    }

    /// Search and replace and return a new Mutibs.
    ///
    /// This is the non-inplace version of :meth:`replace`.
    #[pyo3(signature = (old, new, start=None, end=None, count=None, byte_aligned=false), text_signature = "($self, old, new, start=None, end=None, count=None, byte_aligned=False)")]
    pub fn replaced(
        &self,
        old: &Bound<'_, PyAny>,
        new: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
        count: Option<i64>,
        byte_aligned: bool,
    ) -> PyResult<Self> {
        let old = Tibs::extract(old.as_borrowed())?;
        let new = Tibs::extract(new.as_borrowed())?;
        let mut out = self.clone();
        out.apply_replace_bits(old, new, start, end, count, byte_aligned)?;
        Ok(out)
    }

    /// Insert bits at position pos.
    ///
    /// Clips to start or end if insert position is out of range.
    ///
    /// :param int pos: The bit position to insert at.
    /// :param Tibs bs: The bits to insert.
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
        let shift = validate_shift(&*slf, n)?;
        slf.as_mut_bitvec_ref().shift_left(shift);
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
        let shift = validate_shift(&*slf, n)?;
        slf.as_mut_bitvec_ref().shift_right(shift);
        Ok(())
    }

    /// Return the Mutibs as a bytes object.
    ///
    /// :raises ValueError: if the length is not a multiple of 8.
    pub fn __bytes__(&self) -> PyResult<Vec<u8>> {
        self.to_bytes()
    }

    /// Return new Mutibs consisting of n concatenations of self.
    ///
    /// Called for expression of the form 'a = b*3'.
    ///
    /// n -- The number of concatenations. Must be >= 0.
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
    /// n -- The number of concatenations. Must be >= 0.
    ///
    pub fn __rmul__(&self, n: i64) -> PyResult<Self> {
        self.__mul__(n)
    }

    /// In-place bit-wise 'and'.
    pub fn __iand__(mut slf: PyRefMut<'_, Self>, other: Tibs) -> PyResult<()> {
        slf.iand(other.as_bitslice())
    }

    /// In-place bit-wise 'or'.
    pub fn __ior__(mut slf: PyRefMut<'_, Self>, other: Tibs) -> PyResult<()> {
        slf.ior(other.as_bitslice())
    }

    /// In-place bit-wise 'xor'.
    pub fn __ixor__(mut slf: PyRefMut<'_, Self>, other: Tibs) -> PyResult<()> {
        slf.ixor(other.as_bitslice())
    }

    /// In-place multiplication by a non-negative integer.
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
                let orig_data = slf.to_bitvec();
                let len = slf.len();
                slf.reserve(len * (n - 1));
                let mut mul = 1;
                while mul * 2 <= n {
                    // Double the length
                    let current = slf.to_bitvec();
                    slf.as_mut_bitvec_ref().extend_from_bitslice(&current);
                    mul *= 2;
                }
                while mul < n {
                    slf.as_mut_bitvec_ref().extend_from_bitslice(&orig_data);
                    mul += 1;
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
        if name == "find_all" || name == "rfind_all" || name == "chunks" {
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
