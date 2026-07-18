use super::bits::{BS, BV};
use super::raw_bytes::byte_search_prep;
use bitvec::domain::Domain;
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
    let alignment_mod8 = if byte_aligned { Some(0) } else { None };
    find_bitvec_aligned(py, haystack, needle, start, end, alignment_mod8)
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
    if let Some(found) = try_find_byte_search(haystack, needle, start, end, alignment_mod8, false) {
        return Ok(found);
    }
    if needle.len() <= 64 {
        return find_bitvec_small(py, haystack, needle, start, end, alignment_mod8);
    }
    match find_large_prefix_scan(py, haystack, needle, start, end, alignment_mod8)? {
        PrefixScan::Found(pos) => Ok(Some(pos)),
        PrefixScan::NotFound => Ok(None),
        PrefixScan::Fallback(resume) => {
            let lps = compute_lps(py, needle)?;
            find_bitvec_impl_with_lps_aligned(py, haystack, needle, &lps, resume, end, alignment_mod8)
        }
    }
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
    if needle.len() <= 64 {
        return find_bitvec_small(py, haystack, needle, start, end, alignment_mod8);
    }
    match find_large_prefix_scan(py, haystack, needle, start, end, alignment_mod8)? {
        PrefixScan::Found(pos) => Ok(Some(pos)),
        PrefixScan::NotFound => Ok(None),
        PrefixScan::Fallback(resume) => {
            find_bitvec_impl_with_lps_aligned(py, haystack, needle, lps, resume, end, alignment_mod8)
        }
    }
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
    if needle.len() <= 64 {
        return rfind_bitvec_small(
            py,
            haystack,
            SmallPattern::reversed(needle),
            start,
            end,
            alignment_mod8,
        );
    }
    let reversed_needle: BV = needle.iter().by_vals().rev().collect();
    match rfind_large_prefix_scan(
        py,
        haystack,
        reversed_needle.as_bitslice(),
        start,
        end,
        alignment_mod8,
    )? {
        PrefixScan::Found(pos) => Ok(Some(pos)),
        PrefixScan::NotFound => Ok(None),
        PrefixScan::Fallback(new_end) => {
            let reversed_lps = compute_lps(py, reversed_needle.as_bitslice())?;
            rfind_kmp_reversed(
                py,
                haystack,
                reversed_needle.as_bitslice(),
                &reversed_lps,
                start,
                new_end,
                alignment_mod8,
            )
        }
    }
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

    if needle.len() <= 64 {
        return collect_find_all_positions_small(py, haystack, needle, start, end, alignment_mod8);
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

#[derive(Clone, Copy)]
struct SmallPattern {
    target: u64,
    mask: u64,
    len: usize,
}

impl SmallPattern {
    fn new(needle: &BS) -> Self {
        Self::from_bits(needle.iter().by_vals())
    }

    fn reversed(needle: &BS) -> Self {
        Self::from_bits(needle.iter().by_vals().rev())
    }

    fn from_bits(bits: impl ExactSizeIterator<Item = bool>) -> Self {
        let len = bits.len();
        debug_assert!(len <= 64);
        let target = bits.fold(0, |value, bit| (value << 1) | bit as u64);
        let mask = u64::MAX.checked_shr((64 - len) as u32).unwrap_or(0);
        Self { target, mask, len }
    }
}

/// Streaming matcher state for patterns of up to 64 bits.
///
/// Bits are fed oldest-first in groups of up to 8, right-aligned in `bits`.
/// Forward scans feed ascending haystack positions; reverse scans feed
/// descending positions with a correspondingly reversed pattern.
struct SmallScanner<F: FnMut(usize) -> bool> {
    pattern: SmallPattern,
    window: u128,
    fed: usize,
    reverse: bool,
    /// Forward: haystack position of the first fed bit. Reverse: one past the last.
    origin: usize,
    alignment_mod8: Option<usize>,
    on_match: F,
}

impl<F: FnMut(usize) -> bool> SmallScanner<F> {
    /// Feed `n` bits and test the candidate positions they complete.
    /// Returns false if `on_match` requested a stop.
    #[inline(always)]
    fn feed(&mut self, bits: u8, n: usize) -> bool {
        self.window = (self.window << n) | bits as u128;
        self.fed += n;
        let len = self.pattern.len;
        if self.fed < len {
            return true;
        }
        // Candidate windows end at bit counts `first..=self.fed`.
        let first = self.fed - n + 1;
        let j_start = len.saturating_sub(first);
        for j in j_start..n {
            let value = ((self.window >> (n - 1 - j)) as u64) & self.pattern.mask;
            if value == self.pattern.target {
                let e = first + j;
                let match_pos = if self.reverse {
                    self.origin - e
                } else {
                    self.origin + e - len
                };
                if matches_alignment(match_pos, self.alignment_mod8) && !(self.on_match)(match_pos)
                {
                    return false;
                }
            }
        }
        true
    }
}

/// Right-align the live bits of a partial edge element that starts at bit `head`.
#[inline(always)]
fn live_bits_forward(value: u8, head: usize, live: usize) -> u8 {
    value >> (8 - head - live)
}

/// As `live_bits_forward`, but with the live bits in reversed (descending) order.
#[inline(always)]
fn live_bits_reverse(value: u8, head: usize) -> u8 {
    value.reverse_bits() >> head
}

fn scan_groups_forward<F: FnMut(usize) -> bool>(
    py: Python<'_>,
    hs: &BS,
    scanner: &mut SmallScanner<F>,
) -> PyResult<()> {
    match hs.domain() {
        Domain::Enclave(elem) => {
            let head = elem.head().into_inner() as usize;
            scanner.feed(live_bits_forward(elem.load_value(), head, hs.len()), hs.len());
            Ok(())
        }
        Domain::Region { head, body, tail } => {
            let live_head = match &head {
                Some(elem) => 8 - elem.head().into_inner() as usize,
                None => 0,
            };
            if let Some(elem) = head
                && !scanner.feed(elem.load_value(), live_head)
            {
                return Ok(());
            }
            let mut until_check = SIGNAL_CHECK_INTERVAL / 8;
            for &byte in body {
                if !scanner.feed(byte, 8) {
                    return Ok(());
                }
                until_check -= 1;
                if until_check == 0 {
                    py.check_signals()?;
                    until_check = SIGNAL_CHECK_INTERVAL / 8;
                }
            }
            let live_tail = hs.len() - live_head - body.len() * 8;
            if live_tail > 0
                && let Some(elem) = tail
            {
                scanner.feed(live_bits_forward(elem.load_value(), 0, live_tail), live_tail);
            }
            Ok(())
        }
    }
}

fn scan_groups_reverse<F: FnMut(usize) -> bool>(
    py: Python<'_>,
    hs: &BS,
    scanner: &mut SmallScanner<F>,
) -> PyResult<()> {
    match hs.domain() {
        Domain::Enclave(elem) => {
            let head = elem.head().into_inner() as usize;
            scanner.feed(live_bits_reverse(elem.load_value(), head), hs.len());
            Ok(())
        }
        Domain::Region { head, body, tail } => {
            let live_head = match &head {
                Some(elem) => 8 - elem.head().into_inner() as usize,
                None => 0,
            };
            let live_tail = hs.len() - live_head - body.len() * 8;
            if live_tail > 0
                && let Some(elem) = tail
                && !scanner.feed(live_bits_reverse(elem.load_value(), 0), live_tail)
            {
                return Ok(());
            }
            let mut until_check = SIGNAL_CHECK_INTERVAL / 8;
            for &byte in body.iter().rev() {
                if !scanner.feed(byte.reverse_bits(), 8) {
                    return Ok(());
                }
                until_check -= 1;
                if until_check == 0 {
                    py.check_signals()?;
                    until_check = SIGNAL_CHECK_INTERVAL / 8;
                }
            }
            if let Some(elem) = head {
                let head_offset = 8 - live_head;
                scanner.feed(live_bits_reverse(elem.load_value(), head_offset), live_head);
            }
            Ok(())
        }
    }
}

fn for_each_small_match<F>(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
    on_match: F,
) -> PyResult<()>
where
    F: FnMut(usize) -> bool,
{
    for_each_small_match_forward(
        py,
        haystack,
        SmallPattern::new(needle),
        start,
        end,
        alignment_mod8,
        on_match,
    )
}

fn for_each_small_match_forward<F>(
    py: Python<'_>,
    haystack: &BS,
    pattern: SmallPattern,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
    on_match: F,
) -> PyResult<()>
where
    F: FnMut(usize) -> bool,
{
    if pattern.len == 0 || pattern.len > end - start {
        return Ok(());
    }
    let mut scanner = SmallScanner {
        pattern,
        window: 0,
        fed: 0,
        reverse: false,
        origin: start,
        alignment_mod8,
        on_match,
    };
    scan_groups_forward(py, &haystack[start..end], &mut scanner)
}

/// Reverse counterpart of `for_each_small_match_forward`. The pattern must be
/// built from the bit-reversed needle; matches are reported at the position of
/// their first (lowest) bit, in descending order.
fn for_each_small_match_reverse<F>(
    py: Python<'_>,
    haystack: &BS,
    pattern: SmallPattern,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
    on_match: F,
) -> PyResult<()>
where
    F: FnMut(usize) -> bool,
{
    if pattern.len == 0 || pattern.len > end - start {
        return Ok(());
    }
    let mut scanner = SmallScanner {
        pattern,
        window: 0,
        fed: 0,
        reverse: true,
        origin: end,
        alignment_mod8,
        on_match,
    };
    scan_groups_reverse(py, &haystack[start..end], &mut scanner)
}

fn find_bitvec_small(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<Option<usize>> {
    debug_assert!(needle.len() <= 64);
    let mut found = None;
    for_each_small_match(py, haystack, needle, start, end, alignment_mod8, |pos| {
        found = Some(pos);
        false
    })?;
    Ok(found)
}

fn rfind_bitvec_small(
    py: Python<'_>,
    haystack: &BS,
    pattern: SmallPattern,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<Option<usize>> {
    let mut found = None;
    for_each_small_match_reverse(py, haystack, pattern, start, end, alignment_mod8, |pos| {
        found = Some(pos);
        false
    })?;
    Ok(found)
}

/// Number of leading (or trailing, for reverse searches) needle bits used as a
/// fast filter when the needle is too long for a `SmallPattern`.
const PREFIX_FILTER_BITS: usize = 64;
/// After this many failed candidate verifications the filter gives up and the
/// caller falls back to KMP, keeping the worst case linear.
const PREFIX_FILTER_BUDGET: usize = 64;

enum PrefixScan {
    Found(usize),
    NotFound,
    /// Verification budget exhausted. Forward scans resume KMP from the
    /// contained start position; reverse scans rescan with it as the new end.
    Fallback(usize),
}

/// Search for a needle longer than 64 bits by scanning for its first 64 bits
/// and verifying the remainder at each candidate position.
fn find_large_prefix_scan(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<PrefixScan> {
    debug_assert!(needle.len() > PREFIX_FILTER_BITS);
    let needle_len = needle.len();
    if needle_len > end - start {
        return Ok(PrefixScan::NotFound);
    }
    let rest = &needle[PREFIX_FILTER_BITS..];
    let mut found = None;
    let mut fallback = None;
    let mut failures = 0usize;
    for_each_small_match(
        py,
        haystack,
        &needle[..PREFIX_FILTER_BITS],
        start,
        end,
        alignment_mod8,
        |pos| {
            if pos + needle_len > end {
                return false;
            }
            if haystack[pos + PREFIX_FILTER_BITS..pos + needle_len] == rest[..] {
                found = Some(pos);
                return false;
            }
            failures += 1;
            if failures >= PREFIX_FILTER_BUDGET {
                fallback = Some(pos + 1);
                return false;
            }
            true
        },
    )?;
    Ok(match (found, fallback) {
        (Some(pos), _) => PrefixScan::Found(pos),
        (None, Some(resume)) => PrefixScan::Fallback(resume),
        (None, None) => PrefixScan::NotFound,
    })
}

/// Reverse counterpart of `find_large_prefix_scan`, operating on the
/// bit-reversed needle: scan backwards for the needle's last 64 bits and
/// verify the leading remainder at each candidate.
fn rfind_large_prefix_scan(
    py: Python<'_>,
    haystack: &BS,
    reversed_needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<PrefixScan> {
    debug_assert!(reversed_needle.len() > PREFIX_FILTER_BITS);
    let needle_len = reversed_needle.len();
    if needle_len > end - start {
        return Ok(PrefixScan::NotFound);
    }
    let rest_len = needle_len - PREFIX_FILTER_BITS;
    // The needle bits preceding the suffix block, restored to original order.
    let orig_rest: BV = reversed_needle[PREFIX_FILTER_BITS..]
        .iter()
        .by_vals()
        .rev()
        .collect();
    // The scanner filters on the suffix block's position, which sits
    // `rest_len` bits after the start of a full match.
    let suffix_alignment = alignment_mod8.map(|required| (required + rest_len) & 7);
    let pattern = SmallPattern::new(&reversed_needle[..PREFIX_FILTER_BITS]);
    let mut found = None;
    let mut fallback = None;
    let mut failures = 0usize;
    for_each_small_match_reverse(py, haystack, pattern, start, end, suffix_alignment, |suffix| {
        if suffix < start + rest_len {
            return false;
        }
        let pos = suffix - rest_len;
        if haystack[pos..suffix] == orig_rest[..] {
            found = Some(pos);
            return false;
        }
        failures += 1;
        if failures >= PREFIX_FILTER_BUDGET {
            fallback = Some((pos + needle_len - 1).min(end));
            return false;
        }
        true
    })?;
    Ok(match (found, fallback) {
        (Some(pos), _) => PrefixScan::Found(pos),
        (None, Some(new_end)) => PrefixScan::Fallback(new_end),
        (None, None) => PrefixScan::NotFound,
    })
}

fn collect_find_all_positions_small(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<Vec<u64>> {
    debug_assert!((2..=64).contains(&needle.len()));
    let mut matches = Vec::new();
    for_each_small_match(py, haystack, needle, start, end, alignment_mod8, |pos| {
        matches.push(pos as u64);
        true
    })?;
    Ok(matches)
}

fn count_bitvec_small(py: Python<'_>, haystack: &BS, needle: &BS) -> PyResult<usize> {
    debug_assert!((1..=64).contains(&needle.len()));
    let mut count = 0;
    for_each_small_match(py, haystack, needle, 0, haystack.len(), None, |_| {
        count += 1;
        true
    })?;
    Ok(count)
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
    if reversed_needle.len() <= 64 {
        return rfind_bitvec_small(
            py,
            haystack,
            SmallPattern::new(reversed_needle),
            start,
            end,
            alignment_mod8,
        );
    }
    match rfind_large_prefix_scan(py, haystack, reversed_needle, start, end, alignment_mod8)? {
        PrefixScan::Found(pos) => Ok(Some(pos)),
        PrefixScan::NotFound => Ok(None),
        PrefixScan::Fallback(new_end) => rfind_kmp_reversed(
            py,
            haystack,
            reversed_needle,
            reversed_lps,
            start,
            new_end,
            alignment_mod8,
        ),
    }
}

fn rfind_kmp_reversed(
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
    if needle.len() <= 64 {
        return count_bitvec_small(py, haystack, needle);
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
