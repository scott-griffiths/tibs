use crate::core::BitCollection;
use crate::enums::{ByteOrder, DtypeKind};
use crate::helpers;
use crate::helpers::MaskedMatcher;
use crate::tibs_::{Tibs, prepare_mask, py_from_value_parts};
use memchr::memmem;
use pyo3::prelude::*;

#[pyclass]
pub struct BoolIterator {
    pub(crate) bits: Tibs,
    pub(crate) index: usize,
    pub(crate) length: usize,
}

#[pymethods]
impl BoolIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<bool> {
        if self.index < self.length {
            // SAFETY: index < length == bits.len(), and the Tibs is immutable.
            let result = unsafe { *self.bits.as_bitslice().get_unchecked(self.index) };
            self.index += 1;
            Some(result)
        } else {
            None
        }
    }

    fn __length_hint__(&self) -> usize {
        self.length - self.index
    }
}

#[pyclass]
pub struct FindAllIterator {
    pub haystack: Py<Tibs>, // Py<T> keeps the Python object alive
    pub search_needle: Tibs,
    pub start: usize,
    pub end: usize,
    pub byte_aligned: bool,
    pub step: usize,
    pub current_pos: usize,
    pub lps: Vec<usize>,
    pub is_reverse: bool,
    pub byte_haystack: Option<Vec<u8>>,
    pub byte_needle: Option<Vec<u8>>,
    pub byte_base: usize,
    pub byte_current: usize,
    /// Prepared once when searching with a mask, in place of the lps and the
    /// byte search, neither of which can cope with don't-care bits.
    pub(crate) matcher: Option<MaskedMatcher>,
}

impl FindAllIterator {
    pub(crate) fn new(
        slf: PyRef<'_, Tibs>,
        needle: Tibs,
        start: Option<isize>,
        end: Option<isize>,
        byte_aligned: bool,
        mask: Option<Tibs>,
        is_reverse: bool,
    ) -> PyResult<Py<Self>> {
        if needle.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No bits were provided to find.",
            ));
        }
        let mask = prepare_mask(mask, needle.len())?;

        let (start, end) = helpers::validate_slice(slf.len(), start, end)?;
        let step = if byte_aligned { 8 } else { 1 };
        let alignment_mod8 = if byte_aligned { Some(0) } else { None };
        let py = slf.py();

        if let Some(mask) = mask {
            let matcher = MaskedMatcher::new(needle.as_bitslice(), mask.as_bitslice(), is_reverse);
            let iter_obj = Self {
                haystack: slf.into(),
                search_needle: needle,
                lps: Vec::new(),
                start,
                end,
                byte_aligned,
                step,
                current_pos: if is_reverse { end } else { start },
                is_reverse,
                byte_haystack: None,
                byte_needle: None,
                byte_base: 0,
                byte_current: 0,
                matcher: Some(matcher),
            };
            return Py::new(py, iter_obj);
        }

        let (byte_haystack, byte_needle, byte_base) = helpers::byte_search_prep(
            slf.as_bitslice(),
            needle.as_bitslice(),
            start,
            end,
            alignment_mod8,
        )
        .map_or((None, None, 0), |(haystack, needle, base)| {
            (Some(haystack.into_owned()), Some(needle.into_owned()), base)
        });

        let using_byte_search = byte_haystack.is_some();
        let using_small_search = needle.len() <= 64;
        let (search_needle, lps) = if using_byte_search {
            (needle, Vec::new())
        } else if is_reverse {
            let reversed_needle =
                Tibs::from_bv(needle.as_bitslice().iter().by_vals().rev().collect());
            let lps = if using_small_search {
                Vec::new()
            } else {
                helpers::compute_lps(py, reversed_needle.as_bitslice())?
            };
            (reversed_needle, lps)
        } else if using_small_search {
            (needle, Vec::new())
        } else {
            let lps = helpers::compute_lps(py, needle.as_bitslice())?;
            (needle, lps)
        };

        let iter_obj = Self {
            haystack: slf.into(),
            search_needle,
            lps,
            start,
            end,
            byte_aligned,
            step,
            current_pos: if is_reverse { end } else { start },
            is_reverse,
            byte_haystack,
            byte_needle,
            byte_base,
            byte_current: if is_reverse { end / 8 - byte_base } else { 0 },
            matcher: None,
        };
        Py::new(py, iter_obj)
    }
}

#[pymethods]
impl FindAllIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<usize>> {
        let needle_len = slf.search_needle.len();
        if needle_len == 0 {
            return Ok(None);
        }
        if let (Some(byte_haystack), Some(byte_needle)) = (&slf.byte_haystack, &slf.byte_needle) {
            let found = if slf.is_reverse {
                if slf.byte_current == 0 {
                    None
                } else {
                    memmem::rfind(&byte_haystack[..slf.byte_current], byte_needle)
                }
            } else if slf.byte_current >= byte_haystack.len() {
                None
            } else {
                memmem::find(&byte_haystack[slf.byte_current..], byte_needle)
                    .map(|pos| pos + slf.byte_current)
            };

            return match found {
                Some(byte_pos) => {
                    let absolute_byte_pos = slf.byte_base + byte_pos;
                    if slf.is_reverse {
                        slf.byte_current = byte_pos + byte_needle.len().saturating_sub(1);
                    } else {
                        slf.byte_current = byte_pos + 1;
                    }
                    Ok(Some(absolute_byte_pos * 8))
                }
                None => Ok(None),
            };
        }

        // Read values from slf that are needed for the find logic
        // or for updating state *after* the find.
        let current_pos = slf.current_pos;
        let byte_aligned = slf.byte_aligned;
        let step = slf.step; // Needed to update slf.current_pos later

        // This block limits the scope of haystack_rs and search_needle.
        // The immutable borrows of slf (to access slf.haystack and slf.search_needle)
        // will end when this block finishes.
        let find_result = {
            let haystack_rs = slf.haystack.borrow(py);
            let lps = &slf.lps;
            let alignment_mod8 = if byte_aligned { Some(0) } else { None };

            if let Some(matcher) = &slf.matcher {
                if slf.is_reverse {
                    if current_pos <= slf.start || current_pos > slf.end {
                        return Ok(None);
                    }
                    matcher.find(
                        py,
                        haystack_rs.as_bitslice(),
                        slf.start,
                        current_pos,
                        alignment_mod8,
                    )?
                } else {
                    if slf.end.saturating_sub(current_pos) < needle_len {
                        return Ok(None);
                    }
                    matcher.find(
                        py,
                        haystack_rs.as_bitslice(),
                        current_pos,
                        slf.end,
                        alignment_mod8,
                    )?
                }
            } else if slf.is_reverse {
                if current_pos <= slf.start || current_pos > slf.end {
                    return Ok(None);
                }
                helpers::rfind_bitvec_with_reversed_lps_aligned(
                    py,
                    haystack_rs.as_bitslice(),
                    slf.search_needle.as_bitslice(),
                    lps,
                    slf.start,
                    current_pos,
                    alignment_mod8,
                )?
            } else {
                // A byte-aligned step can push current_pos past `end`, so the
                // bound must be checked against `end`, not the haystack length.
                if slf.end.saturating_sub(current_pos) < needle_len {
                    return Ok(None); // No space left for the needle or already past the end
                }
                helpers::find_bitvec_with_lps_aligned(
                    py,
                    haystack_rs.as_bitslice(),
                    slf.search_needle.as_bitslice(),
                    lps,
                    current_pos,
                    slf.end,
                    alignment_mod8,
                )?
            }
        };

        // Now, `slf` can be mutably accessed without conflicting with the previous borrows.
        match find_result {
            Some(pos) => {
                if slf.is_reverse {
                    slf.current_pos = pos + needle_len.saturating_sub(step);
                } else {
                    slf.current_pos = pos + step;
                }
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
    pub is_reverse: bool,
}

#[pymethods]
impl ChunksIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Tibs>> {
        if slf.chunks_generated >= slf.max_chunks {
            return Ok(None);
        }

        if slf.is_reverse {
            if slf.current_pos == 0 {
                return Ok(None);
            }
        } else if slf.current_pos >= slf.bits_len {
            return Ok(None);
        }

        let take = if slf.is_reverse {
            std::cmp::min(slf.chunk_size, slf.current_pos)
        } else {
            std::cmp::min(slf.chunk_size, slf.bits_len - slf.current_pos)
        };
        let start = if slf.is_reverse {
            slf.current_pos - take
        } else {
            slf.current_pos
        };

        // Create a cheap slice without copying the underlying data.
        let chunk_bits = {
            let bits = slf.bits_object.borrow(slf.py());
            bits.get_slice_unchecked(start, take)
        };
        if slf.is_reverse {
            slf.current_pos -= take;
        } else {
            slf.current_pos += take;
        }
        slf.chunks_generated += 1;

        Ok(Some(chunk_bits))
    }
}

#[pyclass]
pub struct ValuesIterator {
    pub(crate) bits_object: Py<Tibs>,
    pub(crate) dtype_kind: DtypeKind,
    pub(crate) dtype_length: usize,
    pub(crate) byte_order: ByteOrder,
    pub(crate) chunk_size: usize,
    pub(crate) current_pos: usize,
    pub(crate) end_pos: usize,
}

#[pymethods]
impl ValuesIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if slf.current_pos >= slf.end_pos {
            return Ok(None);
        }

        let value = {
            let bits = slf.bits_object.borrow(py);
            bits.get_slice_unchecked(slf.current_pos, slf.chunk_size)
        };
        slf.current_pos += slf.chunk_size;

        py_from_value_parts(py, slf.dtype_kind, slf.dtype_length, slf.byte_order, &value).map(Some)
    }
}
