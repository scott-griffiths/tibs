use crate::core::BitCollection;
use crate::helpers;
use crate::tibs_::Tibs;
use pyo3::prelude::*;

#[pyclass]
pub struct BoolIterator {
    pub(crate) bits: Py<Tibs>,
    pub(crate) index: isize,
    pub(crate) length: isize,
}

#[pymethods]
impl BoolIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<bool>> {
        if self.index < self.length {
            // It's probably pretty inefficient borrowing on each iterator.
            // It may make more sense to buffer some values in advance.
            let bits = self.bits.borrow(py);
            let result = bits.get_index(self.index);
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
    pub needle: Tibs,
    pub start: usize,
    pub end: usize,
    pub byte_aligned: bool,
    pub step: usize,
    pub current_pos: usize,
    pub lps: Vec<usize>,
    pub is_reverse: bool,
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
        let needle_len = slf.needle.len();
        if needle_len == 0 {
            return Ok(None);
        }

        // This block limits the scope of haystack_rs and needle_rs.
        // The immutable borrows of slf (to access slf.haystack and slf.needle)
        // will end when this block finishes.
        let (find_result, haystack_len, haystack_msb0) = {
            let haystack_rs = slf.haystack.borrow(py);
            let lps = &slf.lps;
            let haystack_len = haystack_rs.len();
            let haystack_msb0 = haystack_rs.msb0;

            let result = if slf.is_reverse {
                if current_pos <= slf.start || current_pos > slf.end {
                    return Ok(None);
                }
                helpers::rfind_bitvec_with_lps(
                    haystack_rs.to_bitslice(),
                    slf.needle.to_bitslice(),
                    lps,
                    slf.start,
                    current_pos,
                    byte_aligned,
                )
            } else {
                if current_pos >= haystack_len
                    || haystack_len.saturating_sub(current_pos) < needle_len
                {
                    return Ok(None); // No space left for the needle or already past the end
                }
                helpers::find_bitvec_with_lps(
                    haystack_rs.to_bitslice(),
                    slf.needle.to_bitslice(),
                    lps,
                    current_pos,
                    slf.end,
                    byte_aligned,
                )
            };
            (result, haystack_len, haystack_msb0)
        };

        // Now, `slf` can be mutably accessed without conflicting with the previous borrows.
        match find_result {
            Some(pos) => {
                if slf.is_reverse {
                    slf.current_pos = pos + needle_len.saturating_sub(step);
                } else {
                    slf.current_pos = pos + step;
                }
                Ok(Some(helpers::physical_match_to_logical_start(
                    haystack_len,
                    needle_len,
                    pos,
                    haystack_msb0,
                )))
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
        let take = std::cmp::min(slf.chunk_size, slf.bits_len - slf.current_pos);
        // Create a cheap slice without copying the underlying data.
        let chunk_bits = {
            let bits = slf.bits_object.borrow(slf.py());
            bits.get_slice_unchecked(slf.current_pos, take)
        };
        slf.current_pos += take;
        slf.chunks_generated += 1;

        Ok(Some(chunk_bits))
    }
}
