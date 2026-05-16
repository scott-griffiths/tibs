use crate::core::BitCollection;
use crate::enums::{BitOrder, Endianness};
use crate::helpers::{BS, BV, bv_from_f64, bv_from_i128, bv_from_u128};
use crate::mutibs::Mutibs;
use crate::tibs_::Tibs;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

fn reverse_bits_within_bytes(source: &BS) -> BV {
    let mut viewed = BV::with_capacity(source.len());
    for byte in source.chunks(8) {
        for bit in byte.iter().rev() {
            viewed.push(*bit);
        }
    }
    viewed
}

fn view_bits_from_physical_bits(
    source: &BS,
    byte_order: Endianness,
    bit_order: BitOrder,
) -> PyResult<BV> {
    let bits = match bit_order {
        BitOrder::Msb0 => source.to_bitvec(),
        BitOrder::Lsb0 => reverse_bits_within_bytes(source),
    };

    if byte_order == Endianness::Little {
        BitCollection::byte_swap_copy(&Tibs::from_bv(bits), None).map(|tibs| tibs.to_bitvec())
    } else {
        Ok(bits)
    }
}

fn physical_bits_from_view_bits(
    viewed: BV,
    byte_order: Endianness,
    bit_order: BitOrder,
) -> PyResult<BV> {
    let bits = if byte_order == Endianness::Little {
        BitCollection::byte_swap_copy(&Tibs::from_bv(viewed), None)?.to_bitvec()
    } else {
        viewed
    };

    Ok(match bit_order {
        BitOrder::Msb0 => bits,
        BitOrder::Lsb0 => reverse_bits_within_bytes(bits.as_bitslice()),
    })
}

fn physical_index_for_label(bit_order: BitOrder, label: usize) -> usize {
    match bit_order {
        BitOrder::Msb0 => label,
        BitOrder::Lsb0 => (label / 8) * 8 + (7 - (label % 8)),
    }
}

fn validate_field_labels(len: usize, a: usize, b: usize) -> PyResult<(usize, usize)> {
    if len == 0 {
        return Err(PyValueError::new_err(
            "Cannot extract a field from an empty view.",
        ));
    }
    if a >= len || b >= len {
        return Err(PyValueError::new_err(format!(
            "Field labels must be in the range 0..{}. Received {a} and {b}.",
            len - 1
        )));
    }

    Ok((a.min(b), a.max(b)))
}

fn field_source_indices(bit_order: BitOrder, low: usize, high: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(high - low + 1);
    match bit_order {
        BitOrder::Msb0 => {
            for label in low..=high {
                indices.push(physical_index_for_label(bit_order, label));
            }
        }
        BitOrder::Lsb0 => {
            for label in (low..=high).rev() {
                indices.push(physical_index_for_label(bit_order, label));
            }
        }
    }
    indices
}

///     A view of a :class:`Tibs` with different interpretation settings.
///
///     A ``View`` does not change the underlying bits. It records how operations such as
///     integer conversion, byte conversion and field extraction should interpret those
///     bits.
///
///     Views are usually created from :class:`Tibs` instances using the
///     :attr:`~Tibs.le`, :attr:`~Tibs.be`, :attr:`~Tibs.lsb0`, :attr:`~Tibs.msb0`
///     or :meth:`~Tibs.view` helpers.
///
///     Passing a :class:`Mutibs` to the direct ``View`` constructor stores a
///     :class:`Tibs` snapshot. Later changes to the original :class:`Mutibs` are
///     not reflected in the view. Use :class:`MutableView` for a live mutable view.
///
///     .. code-block:: pycon
///
///         >>> t = Tibs('0x0100')
///         >>> t.le.u
///         1
///         >>> t.lsb0.hex
///         '8000'
///
#[pyclass(module = "tibs", frozen)]
pub struct View {
    pub(crate) source: Tibs,
    pub(crate) byte_order: Endianness,
    pub(crate) bit_order: BitOrder,
}

impl View {
    pub(crate) fn validate_layout(
        len: usize,
        byte_order: Endianness,
        bit_order: BitOrder,
    ) -> PyResult<()> {
        let is_byte_oriented = byte_order != Endianness::Unspecified || bit_order != BitOrder::Msb0;
        if is_byte_oriented && !len.is_multiple_of(8) {
            return Err(PyValueError::new_err(format!(
                "Cannot create a byte-oriented view with a length of {len} bits. It must be a whole number of bytes long."
            )));
        }
        Ok(())
    }

    pub(crate) fn from_tibs(tibs: Tibs, byte_order: Endianness, bit_order: BitOrder) -> Self {
        View {
            source: tibs,
            byte_order,
            bit_order,
        }
    }

    fn with_layout(&self, byte_order: Endianness, bit_order: BitOrder) -> PyResult<Self> {
        Self::validate_layout(self.source.len(), byte_order, bit_order)?;
        Ok(View {
            source: self.source.clone(),
            byte_order,
            bit_order,
        })
    }

    fn to_tibs_view(&self) -> PyResult<Tibs> {
        if self.bit_order == BitOrder::Msb0 && self.byte_order != Endianness::Little {
            return Ok(self.source.clone());
        }

        let tibs = if self.bit_order == BitOrder::Lsb0 {
            Tibs::from_bv(reverse_bits_within_bytes(self.source.to_bitslice()))
        } else {
            self.source.clone()
        };

        if self.byte_order == Endianness::Little {
            BitCollection::byte_swap_copy(&tibs, None)
        } else {
            Ok(tibs)
        }
    }
}

#[derive(Clone)]
enum MutableSelection {
    Whole,
    Field { indices: Vec<usize> },
}

impl MutableSelection {
    fn len(&self, source_len: usize) -> PyResult<usize> {
        match self {
            MutableSelection::Whole => Ok(source_len),
            MutableSelection::Field { indices } => {
                if indices.iter().any(|&index| index >= source_len) {
                    return Err(PyValueError::new_err(
                        "MutableView source is too short for this field.",
                    ));
                }
                Ok(indices.len())
            }
        }
    }

    fn source_indices(&self, source_len: usize) -> PyResult<Vec<usize>> {
        match self {
            MutableSelection::Whole => Ok((0..source_len).collect()),
            MutableSelection::Field { indices } => {
                self.len(source_len)?;
                Ok(indices.clone())
            }
        }
    }
}

///     A live mutable view of a :class:`Mutibs` with different interpretation settings.
///
///     ``MutableView`` records how operations such as integer conversion, byte
///     conversion and field extraction should interpret the source bits. Unlike
///     :class:`View`, it keeps a live reference to the source ``Mutibs``.
///
///     Assigning through ``u``, ``i`` or ``f`` mutates the source ``Mutibs`` without
///     changing its length.
///
#[pyclass(module = "tibs")]
pub struct MutableView {
    pub(crate) source: Py<Mutibs>,
    pub(crate) byte_order: Endianness,
    pub(crate) bit_order: BitOrder,
    selection: MutableSelection,
}

impl MutableView {
    pub(crate) fn from_mutibs(
        source: Py<Mutibs>,
        byte_order: Endianness,
        bit_order: BitOrder,
    ) -> Self {
        MutableView {
            source,
            byte_order,
            bit_order,
            selection: MutableSelection::Whole,
        }
    }

    fn from_parts(
        source: Py<Mutibs>,
        byte_order: Endianness,
        bit_order: BitOrder,
        selection: MutableSelection,
    ) -> Self {
        MutableView {
            source,
            byte_order,
            bit_order,
            selection,
        }
    }

    fn with_layout(
        &self,
        py: Python<'_>,
        byte_order: Endianness,
        bit_order: BitOrder,
    ) -> PyResult<Self> {
        let source = self.source.borrow(py);
        let len = self.selection.len(source.len())?;
        View::validate_layout(len, byte_order, bit_order)?;
        Ok(Self::from_parts(
            self.source.clone_ref(py),
            byte_order,
            bit_order,
            self.selection.clone(),
        ))
    }

    fn current_len(&self, source_len: usize) -> PyResult<usize> {
        self.selection.len(source_len)
    }

    fn validate_current_layout(&self, source_len: usize) -> PyResult<usize> {
        let len = self.current_len(source_len)?;
        View::validate_layout(len, self.byte_order, self.bit_order)?;
        Ok(len)
    }

    fn selected_source_bits(&self, source: &Mutibs) -> PyResult<BV> {
        match &self.selection {
            MutableSelection::Whole => Ok(source.to_bitvec()),
            MutableSelection::Field { indices } => {
                self.selection.len(source.len())?;
                let source_bits = source.as_bitslice();
                let mut selected = BV::with_capacity(indices.len());
                for &index in indices {
                    selected.push(source_bits[index]);
                }
                Ok(selected)
            }
        }
    }

    fn to_tibs_view(&self, py: Python<'_>) -> PyResult<Tibs> {
        let source = self.source.borrow(py);
        self.validate_current_layout(source.len())?;
        let source_bits = self.selected_source_bits(&source)?;
        if self.bit_order == BitOrder::Msb0 && self.byte_order != Endianness::Little {
            return Ok(Tibs::from_bv(source_bits));
        }

        Ok(Tibs::from_bv(view_bits_from_physical_bits(
            source_bits.as_bitslice(),
            self.byte_order,
            self.bit_order,
        )?))
    }

    fn assign_from_view_bits(&self, py: Python<'_>, viewed: BV) -> PyResult<()> {
        let mut source = self.source.borrow_mut(py);
        let len = self.validate_current_layout(source.len())?;
        let physical = physical_bits_from_view_bits(viewed, self.byte_order, self.bit_order)?;
        debug_assert_eq!(len, physical.len());

        match &self.selection {
            MutableSelection::Whole => {
                source
                    .as_mut_bitvec_ref()
                    .copy_from_bitslice(physical.as_bitslice());
            }
            MutableSelection::Field { indices } => {
                debug_assert_eq!(indices.len(), physical.len());
                for (bit_index, &source_index) in indices.iter().enumerate() {
                    source
                        .as_mut_bitvec_ref()
                        .set(source_index, physical[bit_index]);
                }
            }
        }
        Ok(())
    }

    fn assign_u(&self, py: Python<'_>, u: u128) -> PyResult<()> {
        let source = self.source.borrow(py);
        let len = self.validate_current_layout(source.len())?;
        drop(source);

        let viewed = bv_from_u128(u, len as i64, false)?;
        self.assign_from_view_bits(py, viewed)
    }

    fn assign_i(&self, py: Python<'_>, i: i128) -> PyResult<()> {
        let source = self.source.borrow(py);
        let len = self.validate_current_layout(source.len())?;
        drop(source);

        let viewed = bv_from_i128(i, len as i64, false)?;
        self.assign_from_view_bits(py, viewed)
    }

    fn assign_f(&self, py: Python<'_>, f: f64) -> PyResult<()> {
        let source = self.source.borrow(py);
        let len = self.validate_current_layout(source.len())?;
        drop(source);

        let viewed = bv_from_f64(f, len as i64, false)?;
        self.assign_from_view_bits(py, viewed)
    }
}

#[pymethods]
impl MutableView {
    /// Create a live mutable view from a :class:`Mutibs`.
    ///
    /// ``byte_order`` controls byte-wise interpretation for whole-byte values.
    /// ``bit_order`` controls how bit labels are interpreted within each byte.
    ///
    /// Byte-oriented views must have a whole-byte length. This applies when using
    /// little-endian or big-endian byte order, or when using ``BitOrder.Lsb0``.
    ///
    #[new]
    #[pyo3(signature = (source, byte_order = Endianness::Unspecified, bit_order = BitOrder::Msb0), text_signature = "(source, byte_order=Endianness.Unspecified, bit_order=BitOrder.Msb0)")]
    pub fn py_new(
        source: PyRef<'_, Mutibs>,
        byte_order: Option<Endianness>,
        bit_order: Option<BitOrder>,
    ) -> PyResult<Self> {
        let byte_order = byte_order.unwrap_or(Endianness::Unspecified);
        let bit_order = bit_order.unwrap_or(BitOrder::Msb0);
        View::validate_layout(source.len(), byte_order, bit_order)?;
        Ok(Self::from_mutibs(source.into(), byte_order, bit_order))
    }

    /// Return a mutable view with updated interpretation settings.
    ///
    /// Any setting left as ``None`` keeps its current value.
    ///
    #[pyo3(signature = (byte_order = None, bit_order = None), text_signature = "($self, byte_order=None, bit_order=None)")]
    pub fn view(
        &self,
        py: Python<'_>,
        byte_order: Option<Endianness>,
        bit_order: Option<BitOrder>,
    ) -> PyResult<Self> {
        self.with_layout(
            py,
            byte_order.unwrap_or(self.byte_order),
            bit_order.unwrap_or(self.bit_order),
        )
    }

    /// Return a little-endian byte-order mutable view.
    #[getter]
    pub fn le(&self, py: Python<'_>) -> PyResult<Self> {
        self.with_layout(py, Endianness::Little, self.bit_order)
    }

    /// Return a big-endian byte-order mutable view.
    #[getter]
    pub fn be(&self, py: Python<'_>) -> PyResult<Self> {
        self.with_layout(py, Endianness::Big, self.bit_order)
    }

    /// Return an LSB0 bit-order mutable view.
    #[getter]
    pub fn lsb0(&self, py: Python<'_>) -> PyResult<Self> {
        self.with_layout(py, self.byte_order, BitOrder::Lsb0)
    }

    /// Return an MSB0 bit-order mutable view.
    #[getter]
    pub fn msb0(&self, py: Python<'_>) -> PyResult<Self> {
        self.with_layout(py, self.byte_order, BitOrder::Msb0)
    }

    /// Return the current number of source bits in the view.
    pub fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        let source = self.source.borrow(py);
        self.current_len(source.len())
    }

    /// Interpret the viewed bits as an unsigned integer.
    pub fn to_u(&self, py: Python<'_>) -> PyResult<u128> {
        let tibs = self.to_tibs_view(py)?;
        BitCollection::to_u128(&tibs, false)
    }

    /// Interpret the viewed bits as a signed integer.
    pub fn to_i(&self, py: Python<'_>) -> PyResult<i128> {
        let tibs = self.to_tibs_view(py)?;
        BitCollection::to_i128(&tibs, false)
    }

    /// Interpret the viewed bits as an IEEE floating point value.
    pub fn to_f(&self, py: Python<'_>) -> PyResult<f64> {
        let tibs = self.to_tibs_view(py)?;
        BitCollection::to_f64(&tibs, false)
    }

    /// Set the viewed bits from an unsigned integer without changing the source length.
    #[pyo3(signature = (u, /), text_signature = "($self, u, /)")]
    pub fn set_u(&self, py: Python<'_>, u: u128) -> PyResult<()> {
        self.assign_u(py, u)
    }

    /// Set the viewed bits from a signed integer without changing the source length.
    #[pyo3(signature = (i, /), text_signature = "($self, i, /)")]
    pub fn set_i(&self, py: Python<'_>, i: i128) -> PyResult<()> {
        self.assign_i(py, i)
    }

    /// Set the viewed bits from a floating point number without changing the source length.
    #[pyo3(signature = (f, /), text_signature = "($self, f, /)")]
    pub fn set_f(&self, py: Python<'_>, f: f64) -> PyResult<()> {
        self.assign_f(py, f)
    }

    /// Return the viewed bits as a binary string.
    pub fn to_bin(&self, py: Python<'_>) -> PyResult<String> {
        Ok(BitCollection::to_binary(&self.to_tibs_view(py)?))
    }

    /// Return the viewed bits as an octal string.
    pub fn to_oct(&self, py: Python<'_>) -> PyResult<String> {
        BitCollection::to_octal(&self.to_tibs_view(py)?)
    }

    /// Return the viewed bits as a hexadecimal string.
    pub fn to_hex(&self, py: Python<'_>) -> PyResult<String> {
        BitCollection::to_hexadecimal(&self.to_tibs_view(py)?)
    }

    /// Return the viewed bits as bytes.
    pub fn to_bytes(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        BitCollection::to_byte_data(&self.to_tibs_view(py)?)
    }

    /// Materialize the current view as a new :class:`Tibs`.
    pub fn to_tibs(&self, py: Python<'_>) -> PyResult<Tibs> {
        self.to_tibs_view(py)
    }

    /// Materialize the current view as a new :class:`Mutibs`.
    pub fn to_mutibs(&self, py: Python<'_>) -> PyResult<Mutibs> {
        Ok(Mutibs::from_bv(self.to_tibs_view(py)?.to_bitvec()))
    }

    /// Extract a field using inclusive bit labels.
    ///
    /// The returned ``MutableView`` is a live view onto the selected source bits.
    pub fn field(&self, py: Python<'_>, a: usize, b: usize) -> PyResult<Self> {
        let source = self.source.borrow(py);
        let current_len = self.validate_current_layout(source.len())?;
        let (low, high) = validate_field_labels(current_len, a, b)?;
        let field_len = high - low + 1;
        let byte_order = if field_len.is_multiple_of(8) {
            self.byte_order
        } else {
            Endianness::Unspecified
        };
        let source_indices = self.selection.source_indices(source.len())?;
        let indices = field_source_indices(self.bit_order, low, high)
            .into_iter()
            .map(|index| source_indices[index])
            .collect();

        Ok(Self::from_parts(
            self.source.clone_ref(py),
            byte_order,
            BitOrder::Msb0,
            MutableSelection::Field { indices },
        ))
    }

    /// Interpret the viewed bits as an unsigned integer.
    #[getter]
    fn u(&self, py: Python<'_>) -> PyResult<u128> {
        self.to_u(py)
    }

    #[setter(u)]
    fn set_u_property(&self, py: Python<'_>, u: u128) -> PyResult<()> {
        self.assign_u(py, u)
    }

    /// Interpret the viewed bits as a signed integer.
    #[getter]
    fn i(&self, py: Python<'_>) -> PyResult<i128> {
        self.to_i(py)
    }

    #[setter(i)]
    fn set_i_property(&self, py: Python<'_>, i: i128) -> PyResult<()> {
        self.assign_i(py, i)
    }

    /// Interpret the viewed bits as an IEEE floating point value.
    #[getter]
    fn f(&self, py: Python<'_>) -> PyResult<f64> {
        self.to_f(py)
    }

    #[setter(f)]
    fn set_f_property(&self, py: Python<'_>, f: f64) -> PyResult<()> {
        self.assign_f(py, f)
    }

    /// Return the viewed bits as a binary string.
    #[getter]
    fn bin(&self, py: Python<'_>) -> PyResult<String> {
        self.to_bin(py)
    }

    /// Return the viewed bits as an octal string.
    #[getter]
    fn oct(&self, py: Python<'_>) -> PyResult<String> {
        self.to_oct(py)
    }

    /// Return the viewed bits as a hexadecimal string.
    #[getter]
    fn hex(&self, py: Python<'_>) -> PyResult<String> {
        self.to_hex(py)
    }

    /// Return the viewed bits as bytes.
    #[getter]
    fn bytes(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        self.to_bytes(py)
    }

    pub fn __repr__(&self, py: Python<'_>) -> String {
        let source = self.source.borrow(py);
        let mut parts = vec![source.__repr__()];
        if self.byte_order != Endianness::Unspecified {
            parts.push(format!("byte_order={}", self.byte_order.repr_name()));
        }
        if self.bit_order != BitOrder::Msb0 {
            parts.push(format!("bit_order={}", self.bit_order.repr_name()));
        }
        format!("MutableView({})", parts.join(", "))
    }
}

#[pymethods]
impl View {
    /// Create a new view from a :class:`Tibs` or :class:`Mutibs`.
    ///
    /// The ``source`` must be a :class:`Tibs` or :class:`Mutibs` instance. A
    /// :class:`Tibs` source is cloned cheaply, while a :class:`Mutibs` source is
    /// copied into an immutable snapshot.
    ///
    /// ``byte_order`` controls byte-wise interpretation for whole-byte values.
    /// ``bit_order`` controls how bit labels are interpreted within each byte.
    ///
    /// Byte-oriented views must have a whole-byte length. This applies when using
    /// little-endian or big-endian byte order, or when using ``BitOrder.Lsb0``.
    ///
    /// :param source: The :class:`Tibs` or :class:`Mutibs` to view.
    /// :param Endianness byte_order: The byte order used when interpreting whole-byte values. Defaults to ``Endianness.Unspecified``.
    /// :param BitOrder bit_order: The bit numbering order used for field labels. Defaults to ``BitOrder.Msb0``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> View(Tibs('0x1234'), Endianness.Little).hex
    ///     '3412'
    ///
    #[new]
    #[pyo3(signature = (source, byte_order = Endianness::Unspecified, bit_order = BitOrder::Msb0), text_signature = "(source, byte_order=Endianness.Unspecified, bit_order=BitOrder.Msb0)")]
    pub fn py_new(
        source: &Bound<'_, PyAny>,
        byte_order: Option<Endianness>,
        bit_order: Option<BitOrder>,
    ) -> PyResult<Self> {
        let byte_order = byte_order.unwrap_or(Endianness::Unspecified);
        let bit_order = bit_order.unwrap_or(BitOrder::Msb0);

        if let Ok(tibs) = source.extract::<PyRef<'_, Tibs>>() {
            Self::validate_layout(tibs.len(), byte_order, bit_order)?;
            return Ok(View::from_tibs(tibs.clone(), byte_order, bit_order));
        }

        if let Ok(mutibs) = source.extract::<PyRef<'_, Mutibs>>() {
            Self::validate_layout(mutibs.len(), byte_order, bit_order)?;
            return Ok(View::from_tibs(mutibs.to_tibs(), byte_order, bit_order));
        }

        Err(PyTypeError::new_err(
            "View source must be a Tibs or Mutibs instance.",
        ))
    }

    /// Return a view with updated interpretation settings.
    ///
    /// Any setting left as ``None`` keeps its current value.
    ///
    /// Byte-oriented views must have a whole-byte length. This applies when using
    /// little-endian or big-endian byte order, or when using ``BitOrder.Lsb0``.
    ///
    /// :param Endianness byte_order: The byte order to use, or ``None`` to keep the current byte order.
    /// :param BitOrder bit_order: The bit order to use, or ``None`` to keep the current bit order.
    /// :return: A new ``View``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0x0100').view(byte_order=Endianness.Little).u
    ///     1
    ///
    #[pyo3(signature = (byte_order = None, bit_order = None), text_signature = "($self, byte_order=None, bit_order=None)")]
    pub fn view(
        &self,
        byte_order: Option<Endianness>,
        bit_order: Option<BitOrder>,
    ) -> PyResult<Self> {
        self.with_layout(
            byte_order.unwrap_or(self.byte_order),
            bit_order.unwrap_or(self.bit_order),
        )
    }

    /// Return a little-endian byte-order view.
    ///
    /// Equivalent to ``view(byte_order=Endianness.Little)``.
    ///
    /// The view length must be a whole number of bytes.
    ///
    #[getter]
    pub fn le(&self) -> PyResult<Self> {
        self.with_layout(Endianness::Little, self.bit_order)
    }

    /// Return a big-endian byte-order view.
    ///
    /// Equivalent to ``view(byte_order=Endianness.Big)``.
    ///
    /// The view length must be a whole number of bytes.
    ///
    #[getter]
    pub fn be(&self) -> PyResult<Self> {
        self.with_layout(Endianness::Big, self.bit_order)
    }

    /// Return an LSB0 bit-order view.
    ///
    /// ``BitOrder.Lsb0`` means that field labels are counted from the least
    /// significant bit of each byte. The view length must be a whole number of
    /// bytes.
    ///
    /// Equivalent to ``view(bit_order=BitOrder.Lsb0)``.
    ///
    #[getter]
    pub fn lsb0(&self) -> PyResult<Self> {
        self.with_layout(self.byte_order, BitOrder::Lsb0)
    }

    /// Return an MSB0 bit-order view.
    ///
    /// ``BitOrder.Msb0`` means that field labels are counted from the most
    /// significant bit of each byte. This is the default bit order.
    ///
    /// Equivalent to ``view(bit_order=BitOrder.Msb0)``.
    ///
    #[getter]
    pub fn msb0(&self) -> PyResult<Self> {
        self.with_layout(self.byte_order, BitOrder::Msb0)
    }

    /// Return the number of source bits in the view.
    pub fn __len__(&self) -> usize {
        self.source.len()
    }

    /// Interpret the viewed bits as an unsigned integer.
    ///
    /// :return: The unsigned integer value.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0x0100').le.to_u()
    ///     1
    ///
    pub fn to_u(&self) -> PyResult<u128> {
        let tibs = self.to_tibs_view()?;
        BitCollection::to_u128(&tibs, false)
    }

    /// Interpret the viewed bits as a signed integer.
    ///
    /// :return: The signed integer value.
    ///
    pub fn to_i(&self) -> PyResult<i128> {
        let tibs = self.to_tibs_view()?;
        BitCollection::to_i128(&tibs, false)
    }

    /// Interpret the viewed bits as an IEEE floating point value.
    ///
    /// The viewed length must be 16, 32 or 64 bits.
    ///
    /// :return: The floating point value.
    ///
    pub fn to_f(&self) -> PyResult<f64> {
        let tibs = self.to_tibs_view()?;
        BitCollection::to_f64(&tibs, false)
    }

    /// Return the viewed bits as a binary string.
    ///
    /// :return: The binary representation as a string.
    ///
    pub fn to_bin(&self) -> PyResult<String> {
        Ok(BitCollection::to_binary(&self.to_tibs_view()?))
    }

    /// Return the viewed bits as an octal string.
    ///
    /// :return: The octal representation as a string.
    ///
    pub fn to_oct(&self) -> PyResult<String> {
        BitCollection::to_octal(&self.to_tibs_view()?)
    }

    /// Return the viewed bits as a hexadecimal string.
    ///
    /// :return: The hexadecimal representation as a string.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Tibs('0x0100').le.to_hex()
    ///     '0001'
    ///
    pub fn to_hex(&self) -> PyResult<String> {
        BitCollection::to_hexadecimal(&self.to_tibs_view()?)
    }

    /// Return the viewed bits as bytes.
    ///
    /// The viewed length must be a whole number of bytes.
    ///
    /// :return: A ``bytes`` value.
    ///
    pub fn to_bytes(&self) -> PyResult<Vec<u8>> {
        BitCollection::to_byte_data(&self.to_tibs_view()?)
    }

    /// Materialize the view as a new :class:`Tibs`.
    ///
    /// :return: A :class:`Tibs` containing the viewed bits.
    ///
    pub fn to_tibs(&self) -> PyResult<Tibs> {
        self.to_tibs_view()
    }

    /// Materialize the view as a new :class:`Mutibs`.
    ///
    /// :return: A :class:`Mutibs` containing the viewed bits.
    ///
    pub fn to_mutibs(&self) -> PyResult<Mutibs> {
        Ok(Mutibs::from_bv(self.to_tibs_view()?.to_bitvec()))
    }

    /// Interpret the viewed bits as an unsigned integer.
    ///
    /// Equivalent to using :meth:`~to_u`.
    ///
    #[getter]
    fn u(&self) -> PyResult<u128> {
        self.to_u()
    }

    /// Interpret the viewed bits as a signed integer.
    ///
    /// Equivalent to using :meth:`~to_i`.
    ///
    #[getter]
    fn i(&self) -> PyResult<i128> {
        self.to_i()
    }

    /// Interpret the viewed bits as an IEEE floating point value.
    ///
    /// Equivalent to using :meth:`~to_f`.
    ///
    #[getter]
    fn f(&self) -> PyResult<f64> {
        self.to_f()
    }

    /// Return the viewed bits as a binary string.
    ///
    /// Equivalent to using :meth:`~to_bin`.
    ///
    #[getter]
    fn bin(&self) -> PyResult<String> {
        self.to_bin()
    }

    /// Return the viewed bits as an octal string.
    ///
    /// Equivalent to using :meth:`~to_oct`.
    ///
    #[getter]
    fn oct(&self) -> PyResult<String> {
        self.to_oct()
    }

    /// Return the viewed bits as a hexadecimal string.
    ///
    /// Equivalent to using :meth:`~to_hex`.
    ///
    #[getter]
    fn hex(&self) -> PyResult<String> {
        self.to_hex()
    }

    /// Return the viewed bits as bytes.
    ///
    /// Equivalent to using :meth:`~to_bytes`.
    ///
    #[getter]
    fn bytes(&self) -> PyResult<Vec<u8>> {
        self.to_bytes()
    }

    /// Extract a field using inclusive bit labels.
    ///
    /// This is intended for specifications that describe fields using inclusive
    /// bit labels such as ``31:26``. The endpoints may be provided in either
    /// order.
    ///
    /// For an LSB0 view, labels are interpreted within each byte with bit 0 at the
    /// least significant bit. For an MSB0 view, labels match normal Python slice
    /// positions.
    ///
    /// The returned view has ``BitOrder.Msb0`` because the field has been
    /// materialized into normal value order. The current byte order is kept for
    /// whole-byte fields and dropped for non-whole-byte fields.
    ///
    /// :param int a: One inclusive field endpoint.
    /// :param int b: The other inclusive field endpoint.
    /// :return: A new ``View`` containing the field.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> t = Tibs('0x88040410')
    ///     >>> t.lsb0.field(31, 26).u
    ///     4
    ///
    pub fn field(&self, a: usize, b: usize) -> PyResult<Self> {
        let len = self.source.len();
        let (low, high) = validate_field_labels(len, a, b)?;
        let field_len = high - low + 1;
        let byte_order = if field_len.is_multiple_of(8) {
            self.byte_order
        } else {
            Endianness::Unspecified
        };

        let source = self.source.to_bitslice();
        let mut field = BV::with_capacity(field_len);
        for index in field_source_indices(self.bit_order, low, high) {
            field.push(source[index]);
        }

        Ok(View::from_tibs(
            Tibs::from_bv(field),
            byte_order,
            BitOrder::Msb0,
        ))
    }

    pub fn __repr__(&self) -> String {
        let mut parts = vec![self.source.__repr__()];
        if self.byte_order != Endianness::Unspecified {
            parts.push(format!("byte_order={}", self.byte_order.repr_name()));
        }
        if self.bit_order != BitOrder::Msb0 {
            parts.push(format!("bit_order={}", self.bit_order.repr_name()));
        }
        format!("View({})", parts.join(", "))
    }
}
