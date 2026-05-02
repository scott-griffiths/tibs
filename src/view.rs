use crate::core::BitCollection;
use crate::enums::{BitOrder, Endianness};
use crate::helpers::BV;
use crate::mutibs::Mutibs;
use crate::tibs_::Tibs;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PySlice};

pub(crate) enum ViewSource {
    Tibs(Py<Tibs>),
    Mutibs(Py<Mutibs>),
}

impl ViewSource {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            ViewSource::Tibs(tibs) => ViewSource::Tibs(tibs.clone_ref(py)),
            ViewSource::Mutibs(mutibs) => ViewSource::Mutibs(mutibs.clone_ref(py)),
        }
    }

    fn repr(&self, py: Python<'_>) -> String {
        match self {
            ViewSource::Tibs(tibs) => tibs.borrow(py).__repr__(),
            ViewSource::Mutibs(mutibs) => mutibs.borrow(py).__repr__(),
        }
    }

    fn len(&self, py: Python<'_>) -> usize {
        match self {
            ViewSource::Tibs(tibs) => tibs.borrow(py).len(),
            ViewSource::Mutibs(mutibs) => mutibs.borrow(py).len(),
        }
    }

    fn to_bitvec(&self, py: Python<'_>) -> BV {
        match self {
            ViewSource::Tibs(tibs) => tibs.borrow(py).to_bitvec(),
            ViewSource::Mutibs(mutibs) => mutibs.borrow(py).to_bitvec(),
        }
    }
}

#[pyclass(module = "tibs")]
pub struct View {
    pub(crate) source: ViewSource,
    pub(crate) byte_order: Endianness,
    pub(crate) bit_order: BitOrder,
}

impl View {
    pub(crate) fn from_tibs(tibs: Py<Tibs>, byte_order: Endianness, bit_order: BitOrder) -> Self {
        View {
            source: ViewSource::Tibs(tibs),
            byte_order,
            bit_order,
        }
    }

    pub(crate) fn from_mutibs(
        mutibs: Py<Mutibs>,
        byte_order: Endianness,
        bit_order: BitOrder,
    ) -> Self {
        View {
            source: ViewSource::Mutibs(mutibs),
            byte_order,
            bit_order,
        }
    }

    fn with_layout(&self, py: Python<'_>, byte_order: Endianness, bit_order: BitOrder) -> Self {
        View {
            source: self.source.clone_ref(py),
            byte_order,
            bit_order,
        }
    }

    fn to_tibs_view(&self, py: Python<'_>) -> PyResult<Tibs> {
        let bv = self.source.to_bitvec(py);
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
}

#[pymethods]
impl View {
    #[pyo3(signature = (byte_order = None, bit_order = None), text_signature = "($self, byte_order=None, bit_order=None)")]
    pub fn view(
        &self,
        py: Python<'_>,
        byte_order: Option<Endianness>,
        bit_order: Option<BitOrder>,
    ) -> Self {
        self.with_layout(
            py,
            byte_order.unwrap_or(self.byte_order),
            bit_order.unwrap_or(self.bit_order),
        )
    }

    #[getter]
    pub fn le(&self, py: Python<'_>) -> Self {
        self.with_layout(py, Endianness::Little, self.bit_order)
    }

    #[getter]
    pub fn be(&self, py: Python<'_>) -> Self {
        self.with_layout(py, Endianness::Big, self.bit_order)
    }

    #[getter]
    pub fn lsb0(&self, py: Python<'_>) -> Self {
        self.with_layout(py, self.byte_order, BitOrder::Lsb0)
    }

    #[getter]
    pub fn msb0(&self, py: Python<'_>) -> Self {
        self.with_layout(py, self.byte_order, BitOrder::Msb0)
    }

    pub fn __len__(&self, py: Python<'_>) -> usize {
        self.source.len(py)
    }

    pub fn to_raw_data(&self, py: Python<'_>) -> PyResult<(Vec<u8>, usize, usize)> {
        Ok(self.to_tibs_view(py)?.raw_data())
    }

    pub fn to_u(&self, py: Python<'_>) -> PyResult<u128> {
        let tibs = self.to_tibs_view(py)?;
        BitCollection::to_u128(&tibs, false)
    }

    pub fn to_i(&self, py: Python<'_>) -> PyResult<i128> {
        let tibs = self.to_tibs_view(py)?;
        BitCollection::to_i128(&tibs, false)
    }

    pub fn to_f(&self, py: Python<'_>) -> PyResult<f64> {
        let tibs = self.to_tibs_view(py)?;
        BitCollection::to_f64(&tibs, false)
    }

    pub fn to_bin(&self, py: Python<'_>) -> PyResult<String> {
        Ok(BitCollection::to_binary(&self.to_tibs_view(py)?))
    }

    pub fn to_oct(&self, py: Python<'_>) -> PyResult<String> {
        BitCollection::to_octal(&self.to_tibs_view(py)?)
    }

    pub fn to_hex(&self, py: Python<'_>) -> PyResult<String> {
        BitCollection::to_hexadecimal(&self.to_tibs_view(py)?)
    }

    pub fn to_bytes(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        BitCollection::to_byte_data(&self.to_tibs_view(py)?)
    }

    pub fn to_tibs(&self, py: Python<'_>) -> PyResult<Tibs> {
        self.to_tibs_view(py)
    }

    pub fn to_mutibs(&self, py: Python<'_>) -> PyResult<Mutibs> {
        Ok(Mutibs::from_bv(self.to_tibs_view(py)?.to_bitvec()))
    }

    #[getter]
    fn u(&self, py: Python<'_>) -> PyResult<u128> {
        self.to_u(py)
    }

    #[getter]
    fn i(&self, py: Python<'_>) -> PyResult<i128> {
        self.to_i(py)
    }

    #[getter]
    fn f(&self, py: Python<'_>) -> PyResult<f64> {
        self.to_f(py)
    }

    #[getter]
    fn bin(&self, py: Python<'_>) -> PyResult<String> {
        self.to_bin(py)
    }

    #[getter]
    fn oct(&self, py: Python<'_>) -> PyResult<String> {
        self.to_oct(py)
    }

    #[getter]
    fn hex(&self, py: Python<'_>) -> PyResult<String> {
        self.to_hex(py)
    }

    #[getter]
    fn bytes(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        self.to_bytes(py)
    }

    pub fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let tibs = self.to_tibs_view(py)?;

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
            let result = View::from_tibs(
                Py::new(py, result)?,
                Endianness::Unspecified,
                BitOrder::Msb0,
            );
            return Ok(Py::new(py, result)?.into_pyobject(py)?.into());
        }

        Err(PyTypeError::new_err("Index must be an integer or a slice."))
    }

    pub fn __repr__(&self, py: Python<'_>) -> String {
        let mut parts = vec![self.source.repr(py)];
        if self.byte_order != Endianness::Unspecified {
            parts.push(format!("byte_order={}", self.byte_order.repr_name()));
        }
        if self.bit_order != BitOrder::Msb0 {
            parts.push(format!("bit_order={}", self.bit_order.repr_name()));
        }
        format!("View({})", parts.join(", "))
    }
}
