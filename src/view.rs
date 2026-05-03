use crate::core::BitCollection;
use crate::enums::{BitOrder, Endianness};
use crate::helpers::BV;
use crate::mutibs::Mutibs;
use crate::tibs_::Tibs;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PySlice};

///     A view of a :class:`Tibs` with different interpretation settings.
///
///     A ``View`` does not change the underlying bits. It records how operations such as
///     integer conversion, byte conversion and field extraction should interpret those
///     bits.
///
///     Views are usually created from :class:`Tibs` or :class:`Mutibs` instances using
///     the :attr:`~Tibs.le`, :attr:`~Tibs.be`, :attr:`~Tibs.lsb0`, :attr:`~Tibs.msb0`
///     or :meth:`~Tibs.view` helpers.
///
///     A view created from a :class:`Mutibs` stores a :class:`Tibs` snapshot. Later
///     changes to the original :class:`Mutibs` are not reflected in the view.
///
///     .. code-block:: pycon
///
///         >>> t = Tibs('0x0100')
///         >>> t.le.u
///         1
///         >>> t.lsb0.hex
///         '8000'
///
#[pyclass(module = "tibs")]
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
        let bv = self.source.to_bitvec();
        let tibs = if self.bit_order == BitOrder::Msb0 {
            Tibs::from_bv(bv)
        } else {
            let mut viewed = BV::with_capacity(bv.len());
            for byte in bv.chunks(8) {
                for bit in byte.iter().rev() {
                    viewed.push(*bit);
                }
            }
            Tibs::from_bv(viewed)
        };

        if Endianness::is_little_endian(Some(self.byte_order), tibs.len())? {
            BitCollection::byte_swap_copy(&tibs, None)
        } else {
            Ok(tibs)
        }
    }

    fn physical_index_for_label(&self, label: usize) -> usize {
        match self.bit_order {
            BitOrder::Msb0 => label,
            BitOrder::Lsb0 => (label / 8) * 8 + (7 - (label % 8)),
        }
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

    /// Return a copy of the raw byte information after applying the view.
    ///
    /// This returns the underlying byte data for the materialized viewed value and
    /// can contain leading and trailing bits that are not considered part of the
    /// viewed data. Usually using :meth:`~to_bytes` is what you really need.
    ///
    /// :return: A tuple of the raw bytes, the bit offset and the bit length.
    ///
    pub fn to_raw_data(&self) -> PyResult<(Vec<u8>, usize, usize)> {
        Ok(self.to_tibs_view()?.raw_data())
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
    /// :return: A string beginning with ``0b``.
    ///
    pub fn to_bin(&self) -> PyResult<String> {
        Ok(BitCollection::to_binary(&self.to_tibs_view()?))
    }

    /// Return the viewed bits as an octal string.
    ///
    /// :return: A string beginning with ``0o``.
    ///
    pub fn to_oct(&self) -> PyResult<String> {
        BitCollection::to_octal(&self.to_tibs_view()?)
    }

    /// Return the viewed bits as a hexadecimal string.
    ///
    /// :return: A string beginning with ``0x``.
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

    /// Return a bit or slice from the materialized view.
    ///
    /// Indexing with an integer returns a ``bool``. Slicing returns a new ``View``
    /// with default interpretation settings over the sliced bits.
    ///
    /// :param key: An integer index or slice.
    /// :return: A ``bool`` or ``View``.
    ///
    pub fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let tibs = self.to_tibs_view()?;

        if let Ok(index) = key.extract::<isize>() {
            let value = tibs.get_index(index)?;
            let py_value = PyBool::new(py, value);
            return Ok(py_value.to_owned().into());
        }

        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(tibs.len() as isize)?;
            let (start, stop, step) = (
                isize::try_from(indices.start)?,
                isize::try_from(indices.stop)?,
                isize::try_from(indices.step)?,
            );

            let result = if step == 1 {
                if start < stop {
                    tibs.get_slice(start as usize, (stop - start) as usize)?
                } else {
                    Tibs::empty()
                }
            } else {
                tibs.get_slice_with_step(start, stop, step)?
            };
            let result = View::from_tibs(result, Endianness::Unspecified, BitOrder::Msb0);
            return Ok(Py::new(py, result)?.into_pyobject(py)?.into());
        }

        Err(PyTypeError::new_err("Index must be an integer or a slice."))
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
    /// materialized into normal value order. The current byte order is kept, so
    /// extracting a non-whole-byte field from a little-endian or big-endian view is
    /// an error.
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
        if a >= len || b >= len {
            return Err(PyValueError::new_err(format!(
                "Field labels must be in the range 0..{}. Received {a} and {b}.",
                len.saturating_sub(1)
            )));
        }

        let high = a.max(b);
        let low = a.min(b);
        let field_len = high - low + 1;
        Self::validate_layout(field_len, self.byte_order, BitOrder::Msb0)?;

        let source = self.source.to_bitvec();
        let mut field = BV::with_capacity(field_len);

        match self.bit_order {
            BitOrder::Msb0 => {
                for label in low..=high {
                    field.push(source[self.physical_index_for_label(label)]);
                }
            }
            BitOrder::Lsb0 => {
                for label in (low..=high).rev() {
                    field.push(source[self.physical_index_for_label(label)]);
                }
            }
        }

        Ok(View::from_tibs(
            Tibs::from_bv(field),
            self.byte_order,
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
