use super::bits::{BS, BV};
use super::raw_bytes::byte_search_prep;
use memchr::memmem;
use pyo3::prelude::*;

pub(crate) const SIGNAL_CHECK_INTERVAL: usize = 65_536;

// An implementation of the KMP algorithm for bit slices.
pub(crate) fn compute_lps(py: Python<'_>, pattern: &BS) -> PyResult<Vec<usize>> {
    let len = pattern.len();
    let mut lps = vec![0; len];
    let mut i = 1;
    let mut len_prev = 0;
    let mut check_at = SIGNAL_CHECK_INTERVAL.min(len);

    while i < len {
        while i < check_at {
            match pattern[i] == pattern[len_prev] {
                true => {
                    len_prev += 1;
                    lps[i] = len_prev;
                    i += 1;
                }
                false if len_prev != 0 => len_prev = lps[len_prev - 1],
                false => {
                    lps[i] = 0;
                    i += 1;
                }
            }
        }
        if i < len {
            py.check_signals()?;
            check_at = i.saturating_add(SIGNAL_CHECK_INTERVAL).min(len);
        }
    }
    Ok(lps)
}

pub(crate) fn find_bitvec(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    byte_aligned: bool,
) -> PyResult<Option<usize>> {
    debug_assert!(end >= start);
    debug_assert!(end <= haystack.len());
    let lps = compute_lps(py, needle)?;
    let alignment_mod8 = if byte_aligned { Some(0) } else { None };
    find_bitvec_with_lps_aligned(py, haystack, needle, &lps, start, end, alignment_mod8)
}

pub(crate) fn find_bitvec_aligned(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<Option<usize>> {
    debug_assert!(end >= start);
    debug_assert!(end <= haystack.len());
    let lps = compute_lps(py, needle)?;
    find_bitvec_with_lps_aligned(py, haystack, needle, &lps, start, end, alignment_mod8)
}

pub(crate) fn find_bitvec_with_lps_aligned(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    lps: &[usize],
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<Option<usize>> {
    debug_assert!(end >= start);
    debug_assert!(end <= haystack.len());
    if let Some(found) = try_find_byte_search(haystack, needle, start, end, alignment_mod8, false) {
        return Ok(found);
    }
    find_bitvec_impl_with_lps_aligned(py, haystack, needle, lps, start, end, alignment_mod8)
}

pub(crate) fn rfind_bitvec_aligned(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<Option<usize>> {
    debug_assert!(end >= start);
    debug_assert!(end <= haystack.len());
    if let Some(found) = try_find_byte_search(haystack, needle, start, end, alignment_mod8, true) {
        return Ok(found);
    }
    let reversed_needle: BV = needle.iter().by_vals().rev().collect();
    let reversed_lps = compute_lps(py, reversed_needle.as_bitslice())?;
    rfind_bitvec_with_reversed_lps_aligned(
        py,
        haystack,
        reversed_needle.as_bitslice(),
        &reversed_lps,
        start,
        end,
        alignment_mod8,
    )
}

pub(crate) fn collect_find_all_positions(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    byte_aligned: bool,
) -> PyResult<Vec<u64>> {
    debug_assert!(!needle.is_empty());
    debug_assert!(end >= start);
    debug_assert!(end <= haystack.len());

    let alignment_mod8 = if byte_aligned { Some(0) } else { None };
    if needle.len() == 1 {
        return collect_single_bit_positions(py, haystack, needle[0], start, end, alignment_mod8);
    }

    if let Some((byte_haystack, byte_needle, byte_base)) =
        byte_search_prep(haystack, needle, start, end, alignment_mod8)
    {
        let mut matches = Vec::new();
        let mut byte_current = 0;
        let mut check_at = SIGNAL_CHECK_INTERVAL;

        loop {
            if matches.len() >= check_at {
                py.check_signals()?;
                check_at = matches.len().saturating_add(SIGNAL_CHECK_INTERVAL);
            }
            let found = if byte_current >= byte_haystack.len() {
                None
            } else {
                memmem::find(&byte_haystack[byte_current..], &byte_needle)
                    .map(|pos| pos + byte_current)
            };

            let Some(byte_pos) = found else {
                break;
            };
            matches.push(((byte_base + byte_pos) * 8) as u64);
            byte_current = byte_pos + 1;
        }

        return Ok(matches);
    }

    collect_find_all_positions_kmp(py, haystack, needle, start, end, alignment_mod8)
}

fn collect_single_bit_positions(
    py: Python<'_>,
    haystack: &BS,
    value: bool,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<Vec<u64>> {
    let mut matches = Vec::new();
    let mut check_at = SIGNAL_CHECK_INTERVAL;

    if let Some(required) = alignment_mod8 {
        let start_mod = start & 7;
        let adjustment = (required + 8 - start_mod) & 7;
        let mut pos = start.saturating_add(adjustment);
        while pos < end {
            if haystack[pos] == value {
                matches.push(pos as u64);
                if matches.len() >= check_at {
                    py.check_signals()?;
                    check_at = matches.len().saturating_add(SIGNAL_CHECK_INTERVAL);
                }
            }
            pos += 8;
        }
        return Ok(matches);
    }

    let slice = &haystack[start..end];
    if value {
        for pos in slice.iter_ones() {
            matches.push((start + pos) as u64);
            if matches.len() >= check_at {
                py.check_signals()?;
                check_at = matches.len().saturating_add(SIGNAL_CHECK_INTERVAL);
            }
        }
    } else {
        for pos in slice.iter_zeros() {
            matches.push((start + pos) as u64);
            if matches.len() >= check_at {
                py.check_signals()?;
                check_at = matches.len().saturating_add(SIGNAL_CHECK_INTERVAL);
            }
        }
    }

    Ok(matches)
}

fn collect_find_all_positions_kmp(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<Vec<u64>> {
    if needle.len() > end - start {
        return Ok(Vec::new());
    }

    let lps = compute_lps(py, needle)?;
    let mut matches = Vec::new();
    let needle_len = needle.len();
    let mut i = start;
    let mut j = 0;
    let mut check_at = start.saturating_add(SIGNAL_CHECK_INTERVAL).min(end);

    while i < end {
        while i < check_at {
            if needle[j] == haystack[i] {
                i += 1;
                j += 1;

                if j == needle_len {
                    let match_pos = i - j;
                    if matches_alignment(match_pos, alignment_mod8) {
                        matches.push(match_pos as u64);
                    }
                    j = lps[j - 1];
                }
            } else if j != 0 {
                j = lps[j - 1];
            } else {
                i += 1;
            }
        }
        if i < end {
            py.check_signals()?;
            check_at = i.saturating_add(SIGNAL_CHECK_INTERVAL).min(end);
        }
    }

    Ok(matches)
}

#[inline]
fn matches_alignment(match_pos: usize, alignment_mod8: Option<usize>) -> bool {
    match alignment_mod8 {
        Some(required) => (match_pos & 7) == required,
        None => true,
    }
}

fn try_find_byte_search(
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
    reverse: bool,
) -> Option<Option<usize>> {
    let (search_bytes, needle_bytes, start_byte) =
        byte_search_prep(haystack, needle, start, end, alignment_mod8)?;
    let found = if reverse {
        memmem::rfind(search_bytes.as_ref(), needle_bytes.as_ref())
    } else {
        memmem::find(search_bytes.as_ref(), needle_bytes.as_ref())
    };
    Some(found.map(|index| (start_byte + index) * 8))
}

pub(crate) fn rfind_bitvec_with_reversed_lps_aligned(
    py: Python<'_>,
    haystack: &BS,
    reversed_needle: &BS,
    reversed_lps: &[usize],
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<Option<usize>> {
    if reversed_needle.is_empty() || reversed_needle.len() > end - start {
        return Ok(None);
    }

    let needle_len = reversed_needle.len();
    let search_len = end - start;
    let mut i = 0;
    let mut j = 0;
    let mut check_at = SIGNAL_CHECK_INTERVAL.min(search_len);

    while i < search_len {
        while i < check_at {
            if reversed_needle[j] == haystack[end - 1 - i] {
                i += 1;
                j += 1;

                if j == needle_len {
                    let reversed_match_pos = i - j;
                    let match_pos = end - needle_len - reversed_match_pos;
                    if matches_alignment(match_pos, alignment_mod8) {
                        return Ok(Some(match_pos));
                    }
                    j = reversed_lps[j - 1];
                }
            } else if j != 0 {
                j = reversed_lps[j - 1];
            } else {
                i += 1;
            }
        }
        if i < search_len {
            py.check_signals()?;
            check_at = i.saturating_add(SIGNAL_CHECK_INTERVAL).min(search_len);
        }
    }

    Ok(None)
}

fn find_bitvec_impl_with_lps_aligned(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    lps: &[usize],
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<Option<usize>> {
    if needle.is_empty() || needle.len() > end - start {
        return Ok(None);
    }
    let needle_len = needle.len();
    let mut i = start;
    let mut j = 0;
    let mut check_at = start.saturating_add(SIGNAL_CHECK_INTERVAL).min(end);

    while i < end {
        while i < check_at {
            if needle[j] == haystack[i] {
                i += 1;
                j += 1;

                if j == needle_len {
                    let match_pos = i - j;
                    if matches_alignment(match_pos, alignment_mod8) {
                        return Ok(Some(match_pos));
                    }
                    // Continue searching for a byte-aligned match
                    j = lps[j - 1];
                }
            } else if j != 0 {
                j = lps[j - 1];
            } else {
                i += 1;
            }
        }
        if i < end {
            py.check_signals()?;
            check_at = i.saturating_add(SIGNAL_CHECK_INTERVAL).min(end);
        }
    }
    Ok(None)
}

/// Count the number of occurrences of needle in haystack.
pub(crate) fn count_bitvec(py: Python<'_>, haystack: &BS, needle: &BS) -> PyResult<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Ok(0);
    }
    let lps = compute_lps(py, needle)?;
    let needle_len = needle.len();
    let mut i = 0; // The start
    let mut j = 0;
    let end = haystack.len();
    let mut count = 0;
    let mut check_at = SIGNAL_CHECK_INTERVAL.min(end);
    while i < end {
        while i < check_at {
            if needle[j] == haystack[i] {
                i += 1;
                j += 1;

                if j == needle_len {
                    count += 1;
                    // Continue searching
                    j = lps[j - 1];
                }
            } else if j != 0 {
                j = lps[j - 1];
            } else {
                i += 1;
            }
        }
        if i < end {
            py.check_signals()?;
            check_at = i.saturating_add(SIGNAL_CHECK_INTERVAL).min(end);
        }
    }
    Ok(count)
}
