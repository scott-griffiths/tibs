use super::bits::{BS, BV};
use super::bitwise::count_bitslice;
use super::raw_bytes::{bitslice_storage, byte_search_prep};
use super::splice::copy_bits;
use bitvec::domain::Domain;
use memchr::memmem;
use pyo3::prelude::*;
use std::borrow::Cow;

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

// --- Searching over bytes at any bit offset -----------------------------
//
// `memmem` matches whole bytes, so a search that may land part way through one
// looks like it cannot use it. It can. A needle placed at bit offset `r` still
// covers a run of whole bytes in the middle, and that run is what `memmem`
// looks for; the handful of bits hanging off either end confirm each hit.
// There are only eight offsets, so eight scans cover every position a match
// could take, and eight vectorised substring searches still beat walking the
// haystack a bit at a time by a wide margin.
//
// A byte-aligned search needs only the one offset its alignment allows, which
// is why `byte_aligned=True` no longer cares whether the needle is a whole
// number of bytes.

/// The needle's whole bytes when it begins `head` bits after a byte boundary.
struct PhasePattern {
    /// Needle bits before the first whole byte. Also the distance back from
    /// the core to the start of the match.
    head: usize,
    core: Vec<u8>,
}

/// A needle prepared for byte-wise searching at every bit offset it could
/// match at.
struct BytewiseSearch<'a> {
    hay: Cow<'a, [u8]>,
    /// Bit offset of `haystack`'s first bit within `hay`.
    hay_offset: usize,
    needle: &'a BS,
    needle_len: usize,
    patterns: Vec<PhasePattern>,
    start: usize,
    end: usize,
}

impl<'a> BytewiseSearch<'a> {
    /// `None` when the needle is too short to cover a whole byte at one of the
    /// offsets wanted, which leaves those cases to the bit-at-a-time scanners.
    fn new(
        haystack: &'a BS,
        needle: &'a BS,
        start: usize,
        end: usize,
        alignment_mod8: Option<usize>,
    ) -> Option<Self> {
        let needle_len = needle.len();
        if needle_len == 0 || needle_len > end.saturating_sub(start) {
            return None;
        }
        let (hay, hay_offset) = bitslice_storage(haystack);
        let (needle_bytes, needle_offset) = bitslice_storage(needle);

        // A match at haystack position `p` puts its core at byte
        // `(hay_offset + p + head) / 8`, so `p` is congruent to
        // `-(head + hay_offset)` mod 8 and an alignment requirement picks out
        // exactly one head.
        let heads: Vec<usize> = match alignment_mod8 {
            Some(required) => vec![(8 - ((required + hay_offset) % 8)) % 8],
            None => (0..8).collect(),
        };

        let mut patterns = Vec::with_capacity(heads.len());
        for head in heads {
            if needle_len < head + 8 {
                return None;
            }
            let core_len = (needle_len - head) / 8;
            let mut core = vec![0u8; core_len];
            copy_bits(
                &mut core,
                0,
                &needle_bytes,
                needle_offset + head,
                core_len * 8,
            );
            patterns.push(PhasePattern { head, core });
        }

        Some(BytewiseSearch {
            hay,
            hay_offset,
            needle,
            needle_len,
            patterns,
            start,
            end,
        })
    }

    /// The match position implied by a core found at byte `core_byte`, once
    /// the bits outside the core and the bounds have been checked.
    fn confirm(
        &self,
        pattern: &PhasePattern,
        core_byte: usize,
        lo: usize,
        hi: usize,
    ) -> Option<usize> {
        let position = (8 * core_byte).checked_sub(pattern.head + self.hay_offset)?;
        if position < lo || position + self.needle_len > hi {
            return None;
        }
        let base = self.hay_offset + position;
        let after_core = pattern.head + pattern.core.len() * 8;
        for index in (0..pattern.head).chain(after_core..self.needle_len) {
            if bit_at(&self.hay, base + index) != self.needle[index] {
                return None;
            }
        }
        Some(position)
    }

    /// Report every match at one bit offset that starts in `lo..hi`, in order.
    /// `on_match` returns false to stop early.
    ///
    /// Narrowing `lo`/`hi` shrinks the bytes `memmem` is given, which is what
    /// makes the later offsets cheap once an earlier one has found something.
    fn scan<F>(&self, pattern: &PhasePattern, reverse: bool, lo: usize, hi: usize, mut on_match: F)
    where
        F: FnMut(usize) -> bool,
    {
        let core_len = pattern.core.len();
        if core_len == 0 || hi < lo + self.needle_len {
            return;
        }
        // Only the bytes a match starting in `lo..hi` could reach into.
        let low = (self.hay_offset + lo) / 8;
        let high = (self.hay_offset + hi).div_ceil(8).min(self.hay.len());
        if high <= low || high - low < core_len {
            return;
        }
        let window = &self.hay[low..high];

        if reverse {
            let mut upto = window.len();
            while let Some(found) = memmem::rfind(&window[..upto], &pattern.core) {
                if let Some(position) = self.confirm(pattern, low + found, lo, hi)
                    && !on_match(position)
                {
                    return;
                }
                // `found + core_len <= upto`, so this always shrinks.
                upto = found + core_len - 1;
                if upto < core_len {
                    return;
                }
            }
        } else {
            let mut from = 0;
            while from + core_len <= window.len() {
                let Some(found) = memmem::find(&window[from..], &pattern.core) else {
                    return;
                };
                let at = from + found;
                if let Some(position) = self.confirm(pattern, low + at, lo, hi)
                    && !on_match(position)
                {
                    return;
                }
                from = at + 1;
            }
        }
    }
}

#[inline]
fn bit_at(bytes: &[u8], bit: usize) -> bool {
    (bytes[bit >> 3] >> (7 - (bit & 7))) & 1 == 1
}

/// First (or last) match, searching over bytes at every offset a match could
/// take. The outer `None` means the needle was unsuitable and the caller
/// should fall back.
fn try_find_bytewise(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
    reverse: bool,
) -> PyResult<Option<Option<usize>>> {
    let Some(search) = BytewiseSearch::new(haystack, needle, start, end, alignment_mod8) else {
        return Ok(None);
    };
    // Each offset's own scan stops at its first hit; the answer is the best of
    // the eight, since the offsets interleave. Once one has been found, the
    // rest only have to beat it, so the searched range keeps shrinking.
    let mut best: Option<usize> = None;
    let (mut lo, mut hi) = (search.start, search.end);
    for pattern in &search.patterns {
        py.check_signals()?;
        let mut first = None;
        search.scan(pattern, reverse, lo, hi, |position| {
            first = Some(position);
            false
        });
        if let Some(position) = first {
            let better = match best {
                None => true,
                Some(current) if reverse => position > current,
                Some(current) => position < current,
            };
            if better {
                best = Some(position);
                if reverse {
                    lo = position;
                } else {
                    hi = position + search.needle_len;
                }
            }
        }
    }
    Ok(Some(best))
}

/// Every match position, ascending.
fn try_find_all_bytewise(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<Option<Vec<u64>>> {
    let Some(search) = BytewiseSearch::new(haystack, needle, start, end, alignment_mod8) else {
        return Ok(None);
    };
    let mut matches = Vec::new();
    for pattern in &search.patterns {
        py.check_signals()?;
        search.scan(pattern, false, search.start, search.end, |position| {
            matches.push(position as u64);
            true
        });
    }
    // Each offset produced an ascending run; the offsets interleave.
    matches.sort_unstable();
    Ok(Some(matches))
}

fn try_count_bytewise(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<Option<usize>> {
    let Some(search) = BytewiseSearch::new(haystack, needle, start, end, alignment_mod8) else {
        return Ok(None);
    };
    let mut count = 0;
    for pattern in &search.patterns {
        py.check_signals()?;
        search.scan(pattern, false, search.start, search.end, |_| {
            count += 1;
            true
        });
    }
    Ok(Some(count))
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
    if let Some(found) = try_find_bytewise(py, haystack, needle, start, end, alignment_mod8, false)?
    {
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
            find_bitvec_impl_with_lps_aligned(
                py,
                haystack,
                needle,
                &lps,
                resume,
                end,
                alignment_mod8,
            )
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
    if let Some(found) = try_find_bytewise(py, haystack, needle, start, end, alignment_mod8, false)?
    {
        return Ok(found);
    }
    if needle.len() <= 64 {
        return find_bitvec_small(py, haystack, needle, start, end, alignment_mod8);
    }
    match find_large_prefix_scan(py, haystack, needle, start, end, alignment_mod8)? {
        PrefixScan::Found(pos) => Ok(Some(pos)),
        PrefixScan::NotFound => Ok(None),
        PrefixScan::Fallback(resume) => find_bitvec_impl_with_lps_aligned(
            py,
            haystack,
            needle,
            lps,
            resume,
            end,
            alignment_mod8,
        ),
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
    if let Some(found) = try_find_bytewise(py, haystack, needle, start, end, alignment_mod8, true)?
    {
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

    if let Some(matches) = try_find_all_bytewise(py, haystack, needle, start, end, alignment_mod8)?
    {
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

    /// As `new`, but only the bits set in `mask` take part in the comparison.
    fn masked(needle: &BS, mask: &BS) -> Self {
        Self::from_masked_bits(needle.iter().by_vals(), mask.iter().by_vals())
    }

    /// As `reversed`, but only the bits set in `mask` take part in the comparison.
    fn masked_reversed(needle: &BS, mask: &BS) -> Self {
        Self::from_masked_bits(needle.iter().by_vals().rev(), mask.iter().by_vals().rev())
    }

    fn from_bits(bits: impl ExactSizeIterator<Item = bool>) -> Self {
        let len = bits.len();
        debug_assert!(len <= 64);
        let target = bits.fold(0, |value, bit| (value << 1) | bit as u64);
        let mask = u64::MAX.checked_shr((64 - len) as u32).unwrap_or(0);
        Self { target, mask, len }
    }

    fn from_masked_bits(
        bits: impl ExactSizeIterator<Item = bool>,
        mask_bits: impl ExactSizeIterator<Item = bool>,
    ) -> Self {
        let len = bits.len();
        debug_assert_eq!(len, mask_bits.len());
        debug_assert!(len <= 64);
        let (target, mask) = bits
            .zip(mask_bits)
            .fold((0u64, 0u64), |(target, mask), (bit, m)| {
                ((target << 1) | (bit & m) as u64, (mask << 1) | m as u64)
            });
        let len_mask = u64::MAX.checked_shr((64 - len) as u32).unwrap_or(0);
        Self {
            target,
            mask: mask & len_mask,
            len,
        }
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
            scanner.feed(
                live_bits_forward(elem.load_value(), head, hs.len()),
                hs.len(),
            );
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
                scanner.feed(
                    live_bits_forward(elem.load_value(), 0, live_tail),
                    live_tail,
                );
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
    for_each_small_match_reverse(
        py,
        haystack,
        pattern,
        start,
        end,
        suffix_alignment,
        |suffix| {
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
        },
    )?;
    Ok(match (found, fallback) {
        (Some(pos), _) => PrefixScan::Found(pos),
        (None, Some(new_end)) => PrefixScan::Fallback(new_end),
        (None, None) => PrefixScan::NotFound,
    })
}

/// A contiguous run of needle bits that the mask requires to match, sitting
/// outside the filter window and so checked only at candidate positions.
struct MaskedRun {
    offset: usize,
    bits: BV,
}

/// A needle with don't-care bits, prepared for searching.
///
/// Neither `memmem` nor KMP can be used once wildcards are involved: masked
/// equality is not transitive, so a failure function built from it is unsound.
/// Instead a 64-bit window of the needle is used as a filter through the same
/// `SmallScanner` machinery as unmasked searches, and the required bits outside
/// that window are verified at each candidate position. Needles of up to 64
/// bits are entirely covered by the filter and need no verification at all.
pub(crate) struct MaskedMatcher {
    len: usize,
    reverse: bool,
    filter: SmallPattern,
    /// Position of the filter window within the needle.
    offset: usize,
    runs: Vec<MaskedRun>,
}

impl MaskedMatcher {
    /// Prepare a search for `needle`, comparing only the bits set in `mask`.
    ///
    /// `mask` must be the same length as `needle`. An all-ones mask gives the
    /// same results as an unmasked search but not its speed, so callers should
    /// route that case to the unmasked functions.
    pub(crate) fn new(needle: &BS, mask: &BS, reverse: bool) -> Self {
        debug_assert_eq!(needle.len(), mask.len());
        let len = needle.len();
        if len <= PREFIX_FILTER_BITS {
            let filter = match reverse {
                true => SmallPattern::masked_reversed(needle, mask),
                false => SmallPattern::masked(needle, mask),
            };
            return Self {
                len,
                reverse,
                filter,
                offset: 0,
                runs: Vec::new(),
            };
        }
        let offset = best_filter_offset(mask);
        let window = offset..offset + PREFIX_FILTER_BITS;
        let filter = match reverse {
            true => SmallPattern::masked_reversed(&needle[window.clone()], &mask[window.clone()]),
            false => SmallPattern::masked(&needle[window.clone()], &mask[window]),
        };
        let mut runs = Vec::new();
        for (from, to) in [(0, offset), (offset + PREFIX_FILTER_BITS, len)] {
            let mut i = from;
            while i < to {
                if !mask[i] {
                    i += 1;
                    continue;
                }
                let mut j = i + 1;
                while j < to && mask[j] {
                    j += 1;
                }
                runs.push(MaskedRun {
                    offset: i,
                    bits: needle[i..j].to_bitvec(),
                });
                i = j;
            }
        }
        Self {
            len,
            reverse,
            filter,
            offset,
            runs,
        }
    }

    fn verify(&self, haystack: &BS, pos: usize) -> bool {
        self.runs.iter().all(|run| {
            let from = pos + run.offset;
            haystack[from..from + run.bits.len()] == run.bits[..]
        })
    }

    fn for_each<F>(
        &self,
        py: Python<'_>,
        haystack: &BS,
        start: usize,
        end: usize,
        alignment_mod8: Option<usize>,
        mut on_match: F,
    ) -> PyResult<()>
    where
        F: FnMut(usize) -> bool,
    {
        debug_assert!(end >= start);
        debug_assert!(end <= haystack.len());
        if self.len == 0 || self.len > end - start {
            return Ok(());
        }
        // The scanner reports the filter window, which starts `self.offset`
        // bits into the match, so both the alignment requirement and the
        // reported position have to be shifted back by it.
        let filter_alignment = alignment_mod8.map(|required| (required + self.offset) & 7);
        let tail = self.len - self.offset;
        if self.reverse {
            // Candidate positions descend: no room after the window is a skip,
            // no room before it means nothing further can match.
            for_each_small_match_reverse(
                py,
                haystack,
                self.filter,
                start,
                end,
                filter_alignment,
                |window| {
                    if window + tail > end {
                        return true;
                    }
                    if window < start + self.offset {
                        return false;
                    }
                    let pos = window - self.offset;
                    !self.verify(haystack, pos) || on_match(pos)
                },
            )
        } else {
            for_each_small_match_forward(
                py,
                haystack,
                self.filter,
                start,
                end,
                filter_alignment,
                |window| {
                    if window < start + self.offset {
                        return true;
                    }
                    if window + tail > end {
                        return false;
                    }
                    let pos = window - self.offset;
                    !self.verify(haystack, pos) || on_match(pos)
                },
            )
        }
    }

    /// The first match in scan order, so the last one in the haystack when the
    /// matcher was built for a reverse search.
    pub(crate) fn find(
        &self,
        py: Python<'_>,
        haystack: &BS,
        start: usize,
        end: usize,
        alignment_mod8: Option<usize>,
    ) -> PyResult<Option<usize>> {
        let mut found = None;
        self.for_each(py, haystack, start, end, alignment_mod8, |pos| {
            found = Some(pos);
            false
        })?;
        Ok(found)
    }

    pub(crate) fn collect(
        &self,
        py: Python<'_>,
        haystack: &BS,
        start: usize,
        end: usize,
        alignment_mod8: Option<usize>,
    ) -> PyResult<Vec<u64>> {
        let mut matches = Vec::new();
        self.for_each(py, haystack, start, end, alignment_mod8, |pos| {
            matches.push(pos as u64);
            true
        })?;
        Ok(matches)
    }

    pub(crate) fn count(
        &self,
        py: Python<'_>,
        haystack: &BS,
        start: usize,
        end: usize,
        alignment_mod8: Option<usize>,
    ) -> PyResult<usize> {
        let mut count = 0;
        self.for_each(py, haystack, start, end, alignment_mod8, |_| {
            count += 1;
            true
        })?;
        Ok(count)
    }
}

/// The start of the 64-bit window of `mask` with the most bits set, which makes
/// the most selective filter. Ties keep the earliest window.
fn best_filter_offset(mask: &BS) -> usize {
    debug_assert!(mask.len() > PREFIX_FILTER_BITS);
    let last = mask.len() - PREFIX_FILTER_BITS;
    let mut count = mask[..PREFIX_FILTER_BITS].count_ones();
    let mut best_count = count;
    let mut best = 0;
    for offset in 1..=last {
        if best_count == PREFIX_FILTER_BITS {
            break;
        }
        count += mask[offset + PREFIX_FILTER_BITS - 1] as usize;
        count -= mask[offset - 1] as usize;
        if count > best_count {
            best_count = count;
            best = offset;
        }
    }
    best
}

/// Find the first masked match, or the last one if `reverse` is set.
pub(crate) fn find_bitvec_masked_aligned(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    mask: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
    reverse: bool,
) -> PyResult<Option<usize>> {
    MaskedMatcher::new(needle, mask, reverse).find(py, haystack, start, end, alignment_mod8)
}

pub(crate) fn collect_find_all_positions_masked(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    mask: &BS,
    start: usize,
    end: usize,
    byte_aligned: bool,
) -> PyResult<Vec<u64>> {
    let alignment_mod8 = if byte_aligned { Some(0) } else { None };
    MaskedMatcher::new(needle, mask, false).collect(py, haystack, start, end, alignment_mod8)
}

/// Count the masked occurrences of needle in haystack, including overlaps.
pub(crate) fn count_bitvec_masked(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    mask: &BS,
    start: usize,
    end: usize,
    byte_aligned: bool,
) -> PyResult<usize> {
    let alignment_mod8 = if byte_aligned { Some(0) } else { None };
    MaskedMatcher::new(needle, mask, false).count(py, haystack, start, end, alignment_mod8)
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

fn count_bitvec_small(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<usize> {
    debug_assert!((2..=64).contains(&needle.len()));
    let mut count = 0;
    for_each_small_match(py, haystack, needle, start, end, alignment_mod8, |_| {
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

/// Count the occurrences of needle in `haystack[start..end]`, including overlaps.
///
/// This mirrors the dispatch of [`collect_find_all_positions`], so it reaches the
/// same fast paths — the byte-oriented SIMD search when the needle can be treated
/// as bytes, and the windowed bit scan otherwise — instead of always falling back
/// to a bit-by-bit scan.
pub(crate) fn count_bitvec(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    byte_aligned: bool,
) -> PyResult<usize> {
    debug_assert!(end >= start);
    debug_assert!(end <= haystack.len());

    // An empty needle matches nothing to count (and would trap `memmem` below).
    if needle.is_empty() || needle.len() > end - start {
        return Ok(0);
    }

    let alignment_mod8 = if byte_aligned { Some(0) } else { None };

    if needle.len() == 1 {
        return Ok(count_single_bit(
            haystack,
            needle[0],
            start,
            end,
            byte_aligned,
        ));
    }

    if let Some((byte_haystack, byte_needle, _byte_base)) =
        byte_search_prep(haystack, needle, start, end, alignment_mod8)
    {
        let mut count = 0;
        let mut byte_current = 0;
        let mut check_at = SIGNAL_CHECK_INTERVAL;
        loop {
            if count >= check_at {
                py.check_signals()?;
                check_at = count.saturating_add(SIGNAL_CHECK_INTERVAL);
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
            count += 1;
            byte_current = byte_pos + 1;
        }
        return Ok(count);
    }

    if let Some(count) = try_count_bytewise(py, haystack, needle, start, end, alignment_mod8)? {
        return Ok(count);
    }

    if needle.len() <= 64 {
        return count_bitvec_small(py, haystack, needle, start, end, alignment_mod8);
    }

    count_bitvec_kmp(py, haystack, needle, start, end, alignment_mod8)
}

/// Count bits equal to `value` in `haystack[start..end]`, honouring an optional
/// byte-alignment requirement on the matching positions.
pub(crate) fn count_single_bit(
    haystack: &BS,
    value: bool,
    start: usize,
    end: usize,
    byte_aligned: bool,
) -> usize {
    if !byte_aligned {
        return count_bitslice(&haystack[start..end], value);
    }
    let mut pos = first_aligned(start, 0);
    let mut count = 0;
    while pos < end {
        if haystack[pos] == value {
            count += 1;
        }
        pos += 8;
    }
    count
}

/// The number of candidate positions in `[start, end)`: every position, or only
/// those on a byte boundary when `byte_aligned`. This is what a single-bit count
/// returns when the mask matches regardless of value.
pub(crate) fn count_candidate_positions(start: usize, end: usize, byte_aligned: bool) -> usize {
    if !byte_aligned {
        return end - start;
    }
    end.saturating_sub(first_aligned(start, 0)).div_ceil(8)
}

/// The first position at or after `start` whose bit index mod 8 is `required`.
#[inline]
fn first_aligned(start: usize, required: usize) -> usize {
    let start_mod = start & 7;
    let adjustment = (required + 8 - start_mod) & 7;
    start.saturating_add(adjustment)
}

fn count_bitvec_kmp(
    py: Python<'_>,
    haystack: &BS,
    needle: &BS,
    start: usize,
    end: usize,
    alignment_mod8: Option<usize>,
) -> PyResult<usize> {
    let lps = compute_lps(py, needle)?;
    let needle_len = needle.len();
    let mut i = start;
    let mut j = 0;
    let mut count = 0;
    let mut check_at = start.saturating_add(SIGNAL_CHECK_INTERVAL).min(end);
    while i < end {
        while i < check_at {
            if needle[j] == haystack[i] {
                i += 1;
                j += 1;

                if j == needle_len {
                    let match_pos = i - j;
                    if matches_alignment(match_pos, alignment_mod8) {
                        count += 1;
                    }
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
