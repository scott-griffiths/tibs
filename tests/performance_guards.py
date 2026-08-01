"""Guards against operations that quietly take the bit-at-a-time route.

Almost every performance bug found in tibs so far has the same shape: an
operation reaches for one of bitvec's bit-domain APIs (``extend_from_bitslice``,
``copy_from_bitslice``, ``BitVec::clone``, ``shift_start``, ``rotate_left``,
``BitSlice``'s ``PartialEq``) when the same work could go through the byte
representation and ``BV::from_vec``. The bit route runs at 2-6 Gbit/s and the
byte route at 80-250 Gbit/s, so the mistake costs one to two orders of
magnitude while leaving every correctness test green.

A plain benchmark cannot catch that on its own: a number only means something
next to a baseline from another commit, and somebody has to read it. So these
are assertions instead. Each one pairs the suspect operation with a *reference*
operation that moves the same bits on the same machine in the same process, and
fails if the ratio between them exceeds a limit. That makes them self-checking
and machine-independent - the ratio survives a slower laptop, a debug build or a
noisy CI runner, none of which a raw timing does.

Ground rules, borrowed from ``performance_scaling.py``:

* Where a pair produces identical output, that is asserted before anything is
  timed, so a case cannot pass by doing less work. Where the two sides
  legitimately differ (``rotate`` against ``>>``, ``==`` against ``count_and``),
  a comment says why the pair is still fair - in every such case both sides
  read or write the same number of bits.
* The two sides are timed alternately, best-of-N. Alternating matters: run one
  side to completion and then the other, and a thermal or scheduler wobble lands
  entirely on whichever went second.
* Limits are set to what the byte route should achieve, with enough slack that
  they are not tripped by noise. They are deliberately looser than the gaps they
  were written against - the point is to catch an order of magnitude, not to
  pin down a percentage.

This module is intentionally not named ``test_*.py``, so the normal suite does
not collect it and CI does not run it. Run it directly for a report::

    .venv/bin/python tests/performance_guards.py

or as assertions::

    .venv/bin/python -m pytest tests/performance_guards.py

These began as the worklist for the audit of 2026-07-27, when all of them
failed. Fixed rows remain here as regression guards, and newly discovered rows
are added while they still fail so the report also serves as the current
optimization worklist. Each one names the call site it is about.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import pytest

from tibs import Codec, Mutibs, Tibs

# 1M bits is 125 KB: far past the point where per-call overhead matters, and
# small enough that a guard is a few milliseconds rather than a few seconds.
BITS = 1_000_000
HALF = BITS // 2

BIG_T = Tibs.from_random(BITS, seed=b"tibs-guards")
BIG_M = Mutibs(BIG_T)
OTHER_T = Tibs.from_random(BITS, seed=b"tibs-guards-other")
# A separate object with the same bits, so equality has to compare them rather
# than notice that it was handed the same object twice.
BIG_T_EQUAL = Tibs.from_bytes(BIG_T.to_bytes())
assert BIG_T_EQUAL is not BIG_T

HALF_T = Tibs.from_random(HALF, seed=b"tibs-guards-half")
# The left operand of a concatenation takes a byte-wide path only when its
# length is a whole number of bytes; this one deliberately is not.
HALF_T_UNALIGNED = Tibs.from_random(HALF + 3, seed=b"tibs-guards-odd")

REPEAT_SMALL_PATTERN = Tibs("0b1011001011")
REPEAT_LARGE_PATTERN = REPEAT_SMALL_PATTERN * 100

ALL_ONES = Tibs.from_ones(BITS)
ALL_ZEROS = Tibs.from_zeros(BITS)

# Large integer conversion is the only current public path known to create a
# BitVec whose live data starts part way through its first storage byte. Keep a
# byte-realigned copy of exactly the same value to expose operations that fall
# back to bitvec only because of that storage shape.
ODD_BITS = BITS - 3
ODD_VALUE = int.from_bytes(BIG_T.to_bytes(), "big") >> 3
ODD_OTHER_VALUE = int.from_bytes(OTHER_T.to_bytes(), "big") >> 3
ODD_NUMERIC_T = Tibs.from_u(ODD_VALUE, ODD_BITS)
ODD_NUMERIC_OTHER = Tibs.from_u(ODD_OTHER_VALUE, ODD_BITS)
ODD_REALIGNED_T = Tibs(Mutibs(ODD_NUMERIC_T))
ODD_REALIGNED_OTHER = Tibs(Mutibs(ODD_NUMERIC_OTHER))
assert ODD_NUMERIC_T == ODD_REALIGNED_T
assert ODD_NUMERIC_OTHER == ODD_REALIGNED_OTHER

BIG_BYTES = BIG_T.to_bytes()
OTHER_BYTES = OTHER_T.to_bytes()
BIG_U = int.from_bytes(BIG_BYTES, "big")
RAW_ENCODED = BIG_T.encode(Codec.Raw)

# Reusable mutation targets. Both sides of each pair repeatedly write the same
# result, so every timed call performs the same work without measuring setup.
DEPOSIT_SLOW_TARGET = Mutibs(BIG_T)
DEPOSIT_FAST_TARGET = Mutibs(BIG_T)
VIEW_WRITE_SLOW_TARGET = Mutibs(BIG_T)
VIEW_WRITE_FAST_TARGET = Mutibs(BIG_T)
WHOLE_MUTABLE_VIEW = VIEW_WRITE_SLOW_TARGET.view()
WRITE_U_TARGET = Mutibs.from_zeros(BITS)

REPLACE_OLD_BYTES = b"\x00"
REPLACE_NEW_BYTES = b"\xff"
REPLACE_OLD = Tibs.from_bytes(REPLACE_OLD_BYTES)
REPLACE_NEW = Tibs.from_bytes(REPLACE_NEW_BYTES)

# One megabit either way. The multi-token spelling exercises the parser's
# repeated BitVec extension without changing the amount of text or bit data.
PARSE_HEX_PIECE = "ab" * 1_250
PARSE_MULTI = ",".join(["0x" + PARSE_HEX_PIECE] * 100)
PARSE_SINGLE = "0x" + PARSE_HEX_PIECE * 100
PARSE_TINY_HEX_PIECE = "ab" * 5
PARSE_TINY_MULTI = ",".join(["0x" + PARSE_TINY_HEX_PIECE] * 25_000)
PARSE_TINY_SINGLE = "0x" + PARSE_TINY_HEX_PIECE * 25_000

ONE_ZERO = Tibs("0b0")
ONE_ONE = Tibs("0b1")

# A one-bit needle, and a container holding few enough of them that collecting
# their positions is a measurement of the scan rather than of list building.
ONE_BIT = Tibs("0b1")
SPARSE_T = Tibs.from_zeros(BITS).set_at(range(0, BITS, BITS // 100))
assert ALL_ZEROS.find(ONE_BIT) is None, "the single-bit search guards must miss"
assert SPARSE_T.count() == 100

# Needles that are absent, so a search has to scan the whole haystack and both
# sides of a search pair return the same answer. 40 bits is a whole number of
# bytes and 41 is not, which is the only difference between them that matters.
NEEDLE_40 = Tibs("0b" + "0" * 39 + "1")
NEEDLE_41 = Tibs("0b" + "0" * 40 + "1")
assert BIG_T.find(NEEDLE_40) is None, "needle must be absent for the search guards"
assert BIG_T.find(NEEDLE_41) is None, "needle must be absent for the search guards"

SET_POSITIONS_RANGE = range(0, BITS, 8)
SET_POSITIONS_LIST = list(SET_POSITIONS_RANGE)
SET_POSITIONS_TUPLE = tuple(SET_POSITIONS_RANGE)

# Targets for the guards whose operation mutates in place. Each is used by one
# guard only, so repeated timed calls cannot interfere with another case.
# Rotating, shifting, byte-swapping and setting the same positions again all
# repeat the same work on every call, so timing many calls stays honest.
BIG_M_SHIFT = Mutibs(BIG_T)
BIG_M_ROTATE = Mutibs(BIG_T)
BIG_M_SWAP = Mutibs(BIG_T)
BIG_M_SET_RANGE = Mutibs.from_zeros(BITS)
BIG_M_SET_LIST = Mutibs.from_zeros(BITS)
BIG_M_SET_TUPLE = Mutibs.from_zeros(BITS)

# The range, list and tuple spellings must land on the same bits, or the
# guards below would be timing two different amounts of work.
_probe_range, _probe_list, _probe_tuple = (Mutibs.from_zeros(BITS) for _ in range(3))
_probe_range.set(SET_POSITIONS_RANGE)
_probe_list.set(SET_POSITIONS_LIST)
_probe_tuple.set(SET_POSITIONS_TUPLE)
assert _probe_range == _probe_list == _probe_tuple

# One round is aimed at this long. Short enough to keep the whole module quick,
# long enough that perf_counter's resolution is not part of the answer.
_ROUND_SECONDS = 0.01
_ROUNDS = 7


def _time_reps(fn: Callable[[], object], reps: int) -> float:
    """Seconds per call, averaged over `reps` back-to-back calls."""
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) / reps


def _calibrate(fn: Callable[[], object]) -> int:
    """Smallest power-of-two repeat count that fills a round."""
    reps = 1
    while reps < 1 << 24:
        if _time_reps(fn, reps) * reps >= _ROUND_SECONDS:
            return reps
        reps *= 2
    return reps


def _compare(
    slow: Callable[[], object], fast: Callable[[], object]
) -> tuple[float, float]:
    """Best-of-N seconds per call for each function, timed alternately."""
    slow_reps = _calibrate(slow)
    fast_reps = _calibrate(fast)
    slow_best = math.inf
    fast_best = math.inf
    for _ in range(_ROUNDS):
        slow_best = min(slow_best, _time_reps(slow, slow_reps))
        fast_best = min(fast_best, _time_reps(fast, fast_reps))
    return slow_best, fast_best


@dataclass(frozen=True)
class Guard:
    """One suspect operation, its reference, and the ratio limit between them."""

    name: str
    site: str
    slow: Callable[[], object]
    fast: Callable[[], object]
    limit: float
    # Set when both sides must produce the same value, which is checked before
    # timing. Left False where the pair is matched on bits moved, not on result.
    same_result: bool = True


def _extend_in_place() -> Mutibs:
    # Both sides of the extend pair build the same fresh Mutibs from a Tibs
    # first, so that construction cancels out of the ratio. It is a byte-wide
    # copy and costs about 2% of the extend, so leaving it in only makes the
    # measured gap smaller than the real one.
    result = Mutibs(HALF_T)
    result.extend(HALF_T)
    return result


def _concat_for_extend() -> Mutibs:
    result = Mutibs(HALF_T)
    return result + HALF_T


def _shift_in_place() -> Mutibs:
    BIG_M_SHIFT.__irshift__(7)
    return BIG_M_SHIFT


def _and_in_place() -> Mutibs:
    result = Mutibs(BIG_T)
    result &= OTHER_T
    return result


def _deposit_all() -> Mutibs:
    DEPOSIT_SLOW_TARGET.deposit(OTHER_T, ALL_ONES)
    return DEPOSIT_SLOW_TARGET


def _write_all_bytes() -> Mutibs:
    DEPOSIT_FAST_TARGET.write_bytes(OTHER_BYTES)
    return DEPOSIT_FAST_TARGET


def _write_through_whole_view() -> Mutibs:
    WHOLE_MUTABLE_VIEW.write_bytes(OTHER_BYTES)
    return VIEW_WRITE_SLOW_TARGET


def _write_directly() -> Mutibs:
    VIEW_WRITE_FAST_TARGET.write_bytes(OTHER_BYTES)
    return VIEW_WRITE_FAST_TARGET


def _write_u() -> Mutibs:
    WRITE_U_TARGET.write_u(BIG_U)
    return WRITE_U_TARGET


GUARDS: list[Guard] = [
    # ---- 1. rotate ------------------------------------------------------
    # Not the same result as a shift, but exactly the same bits moved: both
    # rewrite every bit of a 1M-bit container.
    Guard(
        name="rotate_right(7) vs >> 7",
        site="mutibs.rs:517 apply_rotation -> bitvec rotate_right",
        slow=lambda: BIG_M_ROTATE.rotate_right(7),
        fast=lambda: BIG_T >> 7,
        limit=6.0,
        same_result=False,
    ),
    # ---- 2. encode ------------------------------------------------------
    # Raw encoding is to_bytes plus a six-bit header and a varint length, so
    # the two differ by a constant, not by a factor.
    Guard(
        name="encode(Raw) vs to_bytes()",
        site="codec.rs:121 encode_as_raw -> bv.extend(bits.to_bitvec())",
        slow=lambda: BIG_T.encode(Codec.Raw),
        fast=lambda: BIG_T.to_bytes(),
        limit=25.0,
        same_result=False,
    ),
    # ---- 3. reverse via slice -------------------------------------------
    Guard(
        name="b[::-1] vs b.reversed()",
        site="core.rs:192 get_slice_with_step - step -1 never reaches reverse_copy",
        slow=lambda: BIG_T[::-1],
        fast=lambda: BIG_T.reversed(),
        limit=4.0,
    ),
    # ---- 4. copying a Mutibs --------------------------------------------
    Guard(
        name="Mutibs(Mutibs) vs Mutibs(Tibs)",
        site="mutibs.rs:77 to_bitvec = data.clone(); tibs_.rs:123 uses BV::from_vec",
        slow=lambda: Mutibs(BIG_M),
        fast=lambda: Mutibs(BIG_T),
        limit=4.0,
    ),
    # ---- 4b. slicing a Mutibs -------------------------------------------
    # Slicing a Mutibs copies (unlike Tibs, which shares storage and is close
    # to free), so a whole-object copy is the right reference: both allocate
    # and fill about a megabit. The mid-byte case additionally needs a shift
    # pass, which is what the looser limit allows for.
    Guard(
        name="M[8:] vs Mutibs(M)",
        site="core.rs get_slice_unchecked for Mutibs -> copied_range",
        slow=lambda: BIG_M[8:],
        fast=lambda: Mutibs(BIG_M),
        limit=3.0,
        same_result=False,
    ),
    Guard(
        name="M[3:] unaligned vs Mutibs(M)",
        site="core.rs get_slice_unchecked for Mutibs -> copied_range",
        slow=lambda: BIG_M[3:],
        fast=lambda: Mutibs(BIG_M),
        limit=5.0,
        same_result=False,
    ),
    # ---- 5. the concatenation family ------------------------------------
    Guard(
        name="from_joined([a, b]) vs a + b",
        site="mutibs.rs:176 join_parts -> copy_from_bitslice",
        slow=lambda: Tibs.from_joined([HALF_T, HALF_T]),
        fast=lambda: HALF_T + HALF_T,
        limit=5.0,
    ),
    Guard(
        name="a * 2 vs a + a",
        site="core.rs repeat_bitcollection -> BitConcat.push_repeated_run",
        slow=lambda: HALF_T * 2,
        fast=lambda: HALF_T + HALF_T,
        limit=5.0,
    ),
    # These produce the same one-megabit result. Runtime should be determined
    # by the result size, not by whether it contains 100,000 or 1,000 copies.
    Guard(
        name="10-bit repeat vs 1000-bit repeat",
        site="helpers/bitwise.rs BitConcat.push_repeated_run",
        slow=lambda: REPEAT_SMALL_PATTERN * 100_000,
        fast=lambda: REPEAT_LARGE_PATTERN * 1_000,
        limit=5.0,
    ),
    Guard(
        name="Mutibs.extend vs +",
        site="mutibs.rs:3336 extend -> extend_from_bitslice",
        slow=_extend_in_place,
        fast=_concat_for_extend,
        limit=5.0,
    ),
    # ---- 6. concatenation onto an unaligned left operand -----------------
    # Different data, same volume. An unaligned left operand does need a shift
    # pass that the aligned case skips, which is what the limit allows for.
    Guard(
        name="unaligned a + b vs aligned a + b",
        site="core.rs:745 concat - byte fast path needs left.len() % 8 == 0",
        slow=lambda: HALF_T_UNALIGNED + HALF_T,
        fast=lambda: HALF_T + HALF_T,
        limit=8.0,
        same_result=False,
    ),
    # ---- 7. in-place shifts ---------------------------------------------
    # An in-place shift should beat the copying one, never mind trail it.
    Guard(
        name="m >>= 7 vs b >> 7",
        site="mutibs.rs:3579 __irshift__ -> bitvec shift_end",
        slow=_shift_in_place,
        fast=lambda: BIG_T >> 7,
        limit=4.0,
        same_result=False,
    ),
    # ---- 7b. in-place logical operations -------------------------------
    # Building the mutable copy is included, so this does more memory traffic
    # than the immutable reference. It should still stay within a small factor;
    # bit-at-a-time assignment makes it roughly fifty times slower.
    Guard(
        name="Mutibs(a) &= b vs a & b",
        site="mutibs.rs apply_logical_op -> logical_op_assign_bytes",
        slow=_and_in_place,
        fast=lambda: BIG_T & OTHER_T,
        limit=4.0,
    ),
    # ---- 8. search off a byte boundary ----------------------------------
    # Both miss, so both scan everything and return None. The byte-aligned side
    # is one memmem pass; the unaligned side needs at most eight of them, which
    # is what the limit leaves room for.
    Guard(
        name="find() vs find(byte_aligned=True)",
        site="search.rs:1074 try_find_byte_search - bit KMP fallback otherwise",
        slow=lambda: BIG_T.find(NEEDLE_40),
        fast=lambda: BIG_T.find(NEEDLE_40, byte_aligned=True),
        limit=30.0,
    ),
    # ---- 9. byte-aligned search with a needle that is not a whole byte ---
    Guard(
        name="find(41-bit, aligned) vs find(40-bit, aligned)",
        site="search.rs:1074 - needle not a byte multiple loses memmem entirely",
        slow=lambda: BIG_T.find(NEEDLE_41, byte_aligned=True),
        fast=lambda: BIG_T.find(NEEDLE_40, byte_aligned=True),
        limit=15.0,
    ),
    # ---- 10. equality ---------------------------------------------------
    # count_and reads both operands in full, exactly as == must, and returns
    # a different kind of answer from the same traffic.
    Guard(
        name="a == b vs a.count_and(b)",
        site="core.rs:842 PartialEq -> bitvec sp_eq (chunks(64) + load_be)",
        slow=lambda: BIG_T == BIG_T_EQUAL,
        fast=lambda: BIG_T.count_and(OTHER_T),
        limit=4.0,
        same_result=False,
    ),
    # ---- 11. extract ----------------------------------------------------
    # An all-ones mask makes extract an identity copy, so the reference is a
    # copy of the same bits and the two results are equal.
    Guard(
        name="extract(all-ones) vs to_mutibs()",
        site="core.rs:97 extract_masked - push per set bit",
        slow=lambda: BIG_T.extracted(ALL_ONES),
        fast=lambda: BIG_T.to_mutibs(),
        limit=20.0,
    ),
    # ---- 12. byte_swap --------------------------------------------------
    # Both rewrite every byte in place.
    Guard(
        name="byte_swap() vs reverse()",
        site="mutibs.rs:536 apply_byte_swap -> copy_from_bitslice",
        slow=lambda: BIG_M_SWAP.byte_swap(4),
        fast=lambda: BIG_M_SWAP.reverse(),
        limit=6.0,
        same_result=False,
    ),
    # ---- 13. set() from a sequence of positions --------------------------
    # The reference is Python summing the *same list*, because that is the
    # honest floor here: both have to read 125,000 integer objects, and no
    # amount of work on the bit side removes that. This guard first compared
    # against `set(range)`, which touches no Python object at all, and so
    # measured a cost the list form cannot avoid rather than a defect - the
    # fast path already reads the list quicker than `sum` does.
    #
    # It still catches the regression that matters. Dropping the borrowed
    # reference walk for pyo3's generic extraction takes this from 0.9x to
    # 1.9x, and the fully generic iterator path to 4.1x.
    Guard(
        name="set(list) vs sum(list)",
        site="mutibs.rs set_from_list -> validate_sequence_indices",
        slow=lambda: BIG_M_SET_LIST.set(SET_POSITIONS_LIST),
        fast=lambda: sum(SET_POSITIONS_LIST),
        limit=1.4,
        same_result=False,
    ),
    # Tuples take the same borrowed-reference path as lists, not the generic
    # one, so the two spellings should cost the same.
    Guard(
        name="set(tuple) vs set(list)",
        site="mutibs.rs set_from_tuple -> validate_sequence_indices",
        slow=lambda: BIG_M_SET_TUPLE.set(SET_POSITIONS_TUPLE),
        fast=lambda: BIG_M_SET_LIST.set(SET_POSITIONS_LIST),
        limit=1.3,
        same_result=False,
    ),
    # ---- 14. to_values('u1') --------------------------------------------
    # Same data, same Python list of 1M objects; u1 just takes a slower road.
    Guard(
        name="to_values('u1') vs to_bools()",
        site="dtype.rs u1 unpack path",
        slow=lambda: BIG_T.to_values("u1"),
        fast=lambda: BIG_T.to_bools(),
        limit=2.5,
    ),
    # ---- 15. finding a single bit ---------------------------------------
    # A one-bit needle is too short to cover a whole byte at any offset, so it
    # can never reach the byte-wise scanners and used to fall all the way
    # through to the windowed scan that steps one bit at a time. Both sides
    # here read every byte of the same container and neither builds anything:
    # the find misses, and the count has to look at all of it regardless.
    Guard(
        name="find(single bit) vs count()",
        site="search.rs find_bitvec_aligned - find_single_bit fast path",
        slow=lambda: ALL_ZEROS.find(ONE_BIT),
        fast=lambda: ALL_ZEROS.count(),
        limit=4.0,
        same_result=False,
    ),
    Guard(
        name="rfind(single bit) vs count()",
        site="search.rs rfind_bitvec_aligned - find_single_bit fast path",
        slow=lambda: ALL_ZEROS.rfind(ONE_BIT),
        fast=lambda: ALL_ZEROS.count(),
        limit=4.0,
        same_result=False,
    ),
    # ---- 16. collecting the positions of a sparse set --------------------
    # `iter_ones` on a `u8` store steps a byte at a time whether or not the
    # byte holds anything, so a sparse container spends everything on empty
    # storage. With a hundred set bits in a million the returned list is small
    # enough that this is a scan measurement, matched against a count of the
    # same bits.
    Guard(
        name="find_all(single bit, sparse) vs count()",
        site="search.rs collect_single_bit_positions - byte skipping",
        slow=lambda: SPARSE_T.find_all(ONE_BIT),
        fast=lambda: SPARSE_T.count(),
        limit=4.0,
        same_result=False,
    ),
    # ---- 17. non-canonical large-integer storage -----------------------
    # The two operands in each pair contain exactly the same bits. Only the
    # construction differs: from_u currently retains the leading pad as a
    # BitVec head offset, while the Mutibs round-trip realigns the bytes.
    Guard(
        name="unaligned from_u & vs realigned &",
        site="numeric.rs bv_from_big_int -> bits[pad..].to_bitvec",
        slow=lambda: ODD_NUMERIC_T & ODD_NUMERIC_OTHER,
        fast=lambda: ODD_REALIGNED_T & ODD_REALIGNED_OTHER,
        limit=6.0,
    ),
    Guard(
        name="unaligned from_u bytes vs realigned bytes",
        site="core.rs to_padded_byte_data -> extend_from_bitslice fallback",
        slow=lambda: ODD_NUMERIC_T.to_padded_bytes(),
        fast=lambda: ODD_REALIGNED_T.to_padded_bytes(),
        limit=4.0,
    ),
    # Both sides create the same Python int. The reference uses the padded
    # bytes that Tibs already exposes, then applies the three-bit alignment
    # shift directly instead of rebuilding a padded BitVec.
    Guard(
        name="unaligned to_u vs from_bytes + shift",
        site="core.rs to_big_int -> extend_from_bitslice",
        slow=lambda: ODD_REALIGNED_T.to_u(),
        fast=lambda: int.from_bytes(ODD_REALIGNED_T.to_padded_bytes(), "big") >> 3,
        limit=2.0,
    ),
    # ---- 18. deposit ----------------------------------------------------
    # An all-ones mask makes deposit exactly a whole-value assignment. Both
    # sides repeatedly overwrite a mutable megabit target with OTHER_T.
    Guard(
        name="deposit(all ones) vs write_bytes()",
        site="bitwise.rs deposit_masked -> iter_ones + set",
        slow=_deposit_all,
        fast=_write_all_bytes,
        limit=20.0,
    ),
    # ---- 19. whole mutable-view writes ---------------------------------
    Guard(
        name="view.write_bytes vs Mutibs.write_bytes",
        site="view.rs assign_from_view_bits -> copy_from_bitslice",
        slow=_write_through_whole_view,
        fast=_write_directly,
        limit=10.0,
    ),
    # ---- 20. write_u ----------------------------------------------------
    # Both sides run the same large-int conversion. write_u additionally
    # copies the resulting BitVec bit by bit into storage it already owns.
    Guard(
        name="write_u vs from_u",
        site="mutibs.rs assign_from_bv -> copy_from_bitslice",
        slow=_write_u,
        fast=lambda: Mutibs.from_u(BIG_U, BITS),
        limit=2.0,
    ),
    # ---- 21. raw codec decode ------------------------------------------
    Guard(
        name="decode(Raw) vs from_bytes()",
        site="codec.rs decode_raw_payload -> to_bitvec",
        slow=lambda: Tibs.decode(RAW_ENCODED),
        fast=lambda: Tibs.from_bytes(BIG_BYTES),
        limit=4.0,
    ),
    # ---- 22. replacement rebuild ---------------------------------------
    # A one-byte needle cannot overlap itself, so bytes.replace and
    # replaced(..., byte_aligned=True) have identical replacement semantics.
    Guard(
        name="aligned replace vs bytes.replace",
        site="mutibs.rs apply_replace_bits -> memchr_iter",
        slow=lambda: BIG_T.replaced(REPLACE_OLD, REPLACE_NEW, byte_aligned=True),
        fast=lambda: Tibs.from_bytes(
            BIG_BYTES.replace(REPLACE_OLD_BYTES, REPLACE_NEW_BYTES)
        ),
        limit=2.0,
    ),
    # ---- 23. multi-token parsing ---------------------------------------
    Guard(
        name="parse 100 tokens vs one token",
        site="helpers/parse.rs try_bv_from_hex_tokens",
        slow=lambda: Tibs.from_string(PARSE_MULTI),
        fast=lambda: Tibs.from_string(PARSE_SINGLE),
        limit=2.0,
    ),
    Guard(
        name="parse 25000 hex tokens vs one token",
        site="helpers/parse.rs try_bv_from_hex_tokens",
        slow=lambda: Tibs.from_string(PARSE_TINY_MULTI),
        fast=lambda: Tibs.from_string(PARSE_TINY_SINGLE),
        limit=3.0,
    ),
    # ---- 24. all/any early exit ----------------------------------------
    # The first bit decides each result. Running the same predicate on a
    # one-bit object preserves Python-call overhead while exposing a scan that
    # unnecessarily depends on the million-bit input length.
    Guard(
        name="all immediate exit vs one bit",
        site="mutibs.rs/tibs_.rs all -> bitvec count_zeros",
        slow=lambda: ALL_ZEROS.all(),
        fast=lambda: ONE_ZERO.all(),
        limit=4.0,
    ),
    Guard(
        name="any immediate exit vs one bit",
        site="mutibs.rs/tibs_.rs any -> bitvec count_ones",
        slow=lambda: ALL_ONES.any(),
        fast=lambda: ONE_ONE.any(),
        limit=4.0,
    ),
]


def _check(guard: Guard) -> tuple[float, float, float]:
    """Verify the pair agrees where it should, then time it. Returns
    (slow seconds, fast seconds, ratio)."""
    if guard.same_result:
        left, right = guard.slow(), guard.fast()
        assert left == right, (
            f"{guard.name}: the two sides must produce the same value, "
            f"otherwise the comparison is not measuring the same work"
        )
    slow_t, fast_t = _compare(guard.slow, guard.fast)
    return slow_t, fast_t, slow_t / fast_t


@pytest.mark.parametrize("guard", GUARDS, ids=lambda g: g.name)
def test_operation_takes_the_byte_path(guard: Guard) -> None:
    slow_t, fast_t, ratio = _check(guard)
    assert ratio < guard.limit, (
        f"{guard.name} is {ratio:.0f}x slower than its reference "
        f"({slow_t * 1e6:.1f}us vs {fast_t * 1e6:.1f}us), limit {guard.limit:g}x. "
        f"See {guard.site}."
    )


# --- Shape guards -------------------------------------------------------
#
# The two below are not ratios between operations but assertions about how one
# operation scales. bitvec's rotate moves at most 64 bits per pass and does a
# full-width copy_within on every pass, so it costs O(len * n) where it should
# cost O(len). No single-size measurement can see that, however slow the number
# looks, which is why these are separate.

# Kept well under a megabit: at these sizes a quadratic rotate is tens of
# milliseconds, where at 1M bits it would be seconds.
_ROTATE_SMALL = 25_000
_ROTATE_LARGE = 100_000


def _rotate_timer(length: int, amount: int) -> Callable[[], object]:
    target = Mutibs.from_random(length, seed=b"tibs-guards-rotate")
    return lambda: target.rotate_right(amount)


def test_rotate_cost_does_not_grow_with_the_rotate_amount() -> None:
    """Rotating by half the length must cost about what rotating by 1 costs.

    Both rewrite every bit exactly once. A rotate built on reversals or on a
    shift pair is flat in the amount; bitvec's is linear in it.
    """
    length = _ROTATE_LARGE
    by_half, by_one = _compare(
        _rotate_timer(length, length // 2), _rotate_timer(length, 1)
    )
    ratio = by_half / by_one
    assert ratio < 4.0, (
        f"rotate_right({length // 2}) is {ratio:.0f}x the cost of rotate_right(1) "
        f"({by_half * 1e6:.1f}us vs {by_one * 1e6:.1f}us): cost scales with the "
        f"rotate amount, so rotate is O(len * n). See mutibs.rs:517."
    )


def test_rotate_scales_linearly_with_length() -> None:
    """Worst-case rotate at 4x the length must cost about 4x, not about 16x."""
    small, large = _ROTATE_SMALL, _ROTATE_LARGE
    growth = large / small
    large_t, small_t = _compare(
        _rotate_timer(large, large // 2), _rotate_timer(small, small // 2)
    )
    ratio = large_t / small_t
    # Linear would be `growth`; O(len * n) is growth squared. Halfway between
    # the two, on a log scale, separates them without being noise-sensitive.
    limit = growth ** 1.5
    assert ratio < limit, (
        f"rotate at {large} bits costs {ratio:.0f}x rotate at {small} bits "
        f"({large_t * 1e6:.1f}us vs {small_t * 1e6:.1f}us) for a {growth:g}x "
        f"length increase; linear would be about {growth:g}x. rotate is "
        f"quadratic in the worst case. See mutibs.rs:517."
    )


def _main() -> int:
    print(f"{'guard':<44}{'slow':>11}{'reference':>11}{'ratio':>8}{'limit':>8}")
    print("-" * 88)
    failures = 0
    for guard in GUARDS:
        slow_t, fast_t, ratio = _check(guard)
        ok = ratio < guard.limit
        failures += not ok
        print(
            f"{guard.name:<44}{slow_t * 1e6:>9.1f}us{fast_t * 1e6:>9.1f}us"
            f"{ratio:>7.0f}x{guard.limit:>7g}x  {'ok' if ok else 'FAIL'}"
        )
        if not ok:
            print(f"{'':<44}{guard.site}")

    print()
    for shape in (
        test_rotate_cost_does_not_grow_with_the_rotate_amount,
        test_rotate_scales_linearly_with_length,
    ):
        try:
            shape()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {shape.__name__}\n     {exc}")
        else:
            print(f"ok   {shape.__name__}")

    print(f"\n{failures} of {len(GUARDS) + 2} guards failing.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
