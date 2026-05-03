use crate::core::BitCollection;
use crate::enums::{BitOrder, Endianness};
use crate::helpers::BV;
use crate::mutibs::Mutibs;
use crate::tibs_::Tibs;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PySlice};

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

    #[getter]
    pub fn le(&self) -> PyResult<Self> {
        self.with_layout(Endianness::Little, self.bit_order)
    }

    #[getter]
    pub fn be(&self) -> PyResult<Self> {
        self.with_layout(Endianness::Big, self.bit_order)
    }

    #[getter]
    pub fn lsb0(&self) -> PyResult<Self> {
        self.with_layout(self.byte_order, BitOrder::Lsb0)
    }

    #[getter]
    pub fn msb0(&self) -> PyResult<Self> {
        self.with_layout(self.byte_order, BitOrder::Msb0)
    }

    pub fn __len__(&self) -> usize {
        self.source.len()
    }

    pub fn to_raw_data(&self) -> PyResult<(Vec<u8>, usize, usize)> {
        Ok(self.to_tibs_view()?.raw_data())
    }

    pub fn to_u(&self) -> PyResult<u128> {
        let tibs = self.to_tibs_view()?;
        BitCollection::to_u128(&tibs, false)
    }

    pub fn to_i(&self) -> PyResult<i128> {
        let tibs = self.to_tibs_view()?;
        BitCollection::to_i128(&tibs, false)
    }

    pub fn to_f(&self) -> PyResult<f64> {
        let tibs = self.to_tibs_view()?;
        BitCollection::to_f64(&tibs, false)
    }

    pub fn to_bin(&self) -> PyResult<String> {
        Ok(BitCollection::to_binary(&self.to_tibs_view()?))
    }

    pub fn to_oct(&self) -> PyResult<String> {
        BitCollection::to_octal(&self.to_tibs_view()?)
    }

    pub fn to_hex(&self) -> PyResult<String> {
        BitCollection::to_hexadecimal(&self.to_tibs_view()?)
    }

    pub fn to_bytes(&self) -> PyResult<Vec<u8>> {
        BitCollection::to_byte_data(&self.to_tibs_view()?)
    }

    pub fn to_tibs(&self) -> PyResult<Tibs> {
        self.to_tibs_view()
    }

    pub fn to_mutibs(&self) -> PyResult<Mutibs> {
        Ok(Mutibs::from_bv(self.to_tibs_view()?.to_bitvec()))
    }

    #[getter]
    fn u(&self) -> PyResult<u128> {
        self.to_u()
    }

    #[getter]
    fn i(&self) -> PyResult<i128> {
        self.to_i()
    }

    #[getter]
    fn f(&self) -> PyResult<f64> {
        self.to_f()
    }

    #[getter]
    fn bin(&self) -> PyResult<String> {
        self.to_bin()
    }

    #[getter]
    fn oct(&self) -> PyResult<String> {
        self.to_oct()
    }

    #[getter]
    fn hex(&self) -> PyResult<String> {
        self.to_hex()
    }

    #[getter]
    fn bytes(&self) -> PyResult<Vec<u8>> {
        self.to_bytes()
    }

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
