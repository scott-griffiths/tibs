use crate::core::BitCollection;
use crate::helpers;
use crate::tibs_::Tibs;
use pyo3::prelude::*;
use pyo3::PyResult;

#[pyclass]
pub struct BoolIterator {
    pub(crate) bits: Py<Tibs>,
    pub(crate) index: usize,
    pub(crate) length: usize,
}

#[pymethods]
impl BoolIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<bool>> {
        if self.index < self.length {
            let bits = self.bits.borrow(py);
            let result = bits._getindex(self.index as i64);
            self.index += 1;
            result.map(Some)
        } else {
            Ok(None)
        }
    }
}

#[pyclass]
pub struct FindAllIterator {
    pub haystack: Py<Tibs>, // Py<T> keeps the Python object alive
    pub needle: Py<Tibs>,
    pub start: usize,
    pub end: usize,
    pub byte_aligned: bool,
    pub step: usize,
    pub current_pos: usize,
}

#[pymethods]
impl FindAllIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<usize>> {
        let py = slf.py();

        // Read values from slf that are needed for the find logic
        // or for updating state *after* the find.
        let current_pos = slf.current_pos;
        let byte_aligned = slf.byte_aligned;
        let step = slf.step; // Needed to update slf.current_pos later

        // This block limits the scope of haystack_rs and needle_rs.
        // The immutable borrows of slf (to access slf.haystack and slf.needle)
        // will end when this block finishes.
        let find_result = {
            let haystack_rs = slf.haystack.borrow(py);
            let needle_rs = slf.needle.borrow(py);

            let needle_len = needle_rs.len();
            if needle_len == 0 {
                // If needle is empty, stop iteration.
                return Ok(None);
            }

            let haystack_len = haystack_rs.len();
            if current_pos >= haystack_len || haystack_len.saturating_sub(current_pos) < needle_len
            {
                return Ok(None); // No space left for the needle or already past the end
            }
            helpers::find_bitvec(&haystack_rs, &needle_rs, current_pos, slf.end, byte_aligned)
        };

        // Now, `slf` can be mutably accessed without conflicting with the previous borrows.
        match find_result {
            Some(pos) => {
                slf.current_pos = pos + step;
                Ok(Some(pos))
            }
            None => Ok(None),
        }
    }
}

#[pyclass]
pub struct ChunksIterator {
    pub(crate) bits_object: Py<Tibs>,
    pub(crate) chunk_size: usize,
    pub(crate) max_chunks: usize,
    pub(crate) current_pos: usize,
    pub(crate) chunks_generated: usize,
    pub(crate) bits_len: usize,
}

#[pymethods]
impl ChunksIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Tibs>> {
        if slf.chunks_generated >= slf.max_chunks || slf.current_pos >= slf.bits_len {
            return Ok(None);
        }

        let chunk_size = slf.chunk_size;
        let start = slf.current_pos;
        let remaining = slf.bits_len - start;
        let take = if remaining > chunk_size {
            chunk_size
        } else {
            remaining
        };
        let end = start + take;

        // Borrow only long enough to copy out the bits slice
        let chunk_bits = {
            let bits = slf.bits_object.borrow(slf.py());
            let slice = &bits.data[start..end];
            Tibs::new(slice.to_bitvec())
        };

        slf.current_pos = end;
        slf.chunks_generated += 1;

        Ok(Some(chunk_bits))
    }
}
