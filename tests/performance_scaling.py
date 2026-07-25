#!/usr/bin/env python3
"""How tibs and bitarray compare on realistic jobs, across a range of sizes.

This is a deliberately neutral comparison, and a different question from
``performance_comparison.py``. That script collects the cases that have been
useful for tuning tibs, at one size. This one asks how the two libraries behave
on ordinary work as the data grows: every workload runs at three or four sizes
spanning several decades, so the answer is a curve rather than a number.

Ground rules, so that the numbers mean something:

* Both sides get the idiomatic spelling for their own library, using the fastest
  route that library documents for the job - not a transliteration of the tibs
  code. Where the choice was not obvious, the reason is in a comment next to it.
* Both sides start from identical bits, prepared once outside the timed region.
* Every pair of results is checked for equality before anything is timed, so a
  case cannot win by doing less work.
* The two sides are timed alternately and the median sample is reported, which
  keeps a background hiccup from landing entirely on one library.

Run it with no arguments for the console table. ``--markdown`` writes a table
whose rows link back to the code that produced them, ``--svg`` draws time
against size, and ``--json`` gives the raw numbers.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import inspect
import json
import math
import operator
import os
import platform
import random
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tibs
from tibs import Mutibs, Tibs

try:
    import bitarray
    from bitarray import bitarray as Bitarray, decodetree
    from bitarray.util import (
        any_and,
        count_and,
        count_or,
        count_xor,
        int2ba,
        ones,
        subset,
        zeros,
    )
except ImportError:  # pragma: no cover - reported properly in main()
    bitarray = None


THIS_FILE = os.path.basename(__file__)
DEFAULT_REPO = "https://github.com/scott-griffiths/tibs"

# Sizes are decimal throughout: a megabit is 10**6 bits, and a megabyte 10**6
# bytes, so that the ladder is made of clean decades and the log axis of the
# chart is evenly spaced.
BIT_SIZES = (100, 10_000, 1_000_000, 100_000_000)
BYTE_SIZES = (1_000, 1_000_000, 100_000_000)
VALUE_SIZES = (100, 10_000, 1_000_000)
ITEM_SIZES = (10, 1_000, 100_000)


# ---------------------------------------------------------------------------
# Shared input preparation
# ---------------------------------------------------------------------------


def deterministic_bytes(count: int, seed: str) -> bytes:
    """Reproducible pseudo-random bytes, so a rerun measures the same data."""
    return random.Random(seed).randbytes(count)


def make_bitarray(data: bytes, bit_count: int | None = None) -> Bitarray:
    """A big-endian bitarray holding ``data``, truncated to ``bit_count`` bits.

    Big-endian is pinned because it is the bit order tibs uses, so the two
    containers hold the same bits in the same order and ``tobytes`` on one can
    be compared with ``to_bytes`` on the other.
    """
    bits = Bitarray(endian="big")
    bits.frombytes(data)
    if bit_count is not None:
        del bits[bit_count:]
    return bits


def bit_pair(bit_count: int, seed: str) -> tuple[Bitarray, Tibs]:
    """The same pseudo-random bits, once as a bitarray and once as a Tibs."""
    data = deterministic_bytes((bit_count + 7) // 8, seed)
    return make_bitarray(data, bit_count), Tibs.from_bytes(data)[:bit_count]


def same_bits(bitarray_result: Any, tibs_result: Any) -> bool:
    """Equality across the two container types: same length, same bits."""
    return len(bitarray_result) == len(tibs_result) and (
        bitarray_result.tobytes() == tibs_result.to_padded_bytes()
    )


# ---------------------------------------------------------------------------
# Workload registry
# ---------------------------------------------------------------------------


Builder = Callable[[int], tuple[Callable[[], Any], Callable[[], Any]]]


@dataclass(frozen=True)
class Workload:
    """One job, with a builder that prepares it at a given size.

    ``build(size)`` returns the two functions to time. Everything they need is
    captured in their closures, so dropping them frees the inputs - which
    matters when a single case holds a hundred megabytes twice over.
    """

    name: str
    group: str
    unit: str
    sizes: tuple[int, ...]
    build: Builder
    check: Callable[[Any, Any], bool]
    line: int

    @property
    def source_url(self) -> str:
        return f"{THIS_FILE}#L{self.line}"


WORKLOADS: list[Workload] = []


def workload(
    name: str,
    *,
    group: str,
    unit: str = "bits",
    sizes: Sequence[int] = BIT_SIZES,
    check: Callable[[Any, Any], bool] = operator.eq,
) -> Callable[[Builder], Builder]:
    """Register a builder as a workload, remembering where it is defined.

    The line number is what lets a rendered table link each row back to the code
    that produced it, and it stays correct as this file is edited.
    """

    def register(build: Builder) -> Builder:
        WORKLOADS.append(
            Workload(
                name=name,
                group=group,
                unit=unit,
                sizes=tuple(sizes),
                build=build,
                check=check,
                line=inspect.getsourcelines(build)[1],
            )
        )
        return build

    return register


# ---------------------------------------------------------------------------
# Packing fixed-width values
# ---------------------------------------------------------------------------

PACK_WIDTH = 12


def u12_code() -> dict[int, Bitarray]:
    """A 12-bit fixed-width code table for bitarray's encode/decode.

    bitarray writes a stream of same-width values through a code dictionary,
    the same machinery it uses for Huffman codes. Building the table depends
    only on the width, so it is hoisted out of the timed region the way a
    reused ``struct.Struct`` would be. It is by far bitarray's fastest route:
    the obvious ``for value in values: out.extend(int2ba(value, length=12))``
    loop is around 17x slower. The catch is that the table has one entry per
    representable value, so this works for u12 and not for, say, u32.
    """
    return {value: int2ba(value, length=PACK_WIDTH, endian="big") for value in range(1 << PACK_WIDTH)}


def u12_values(count: int) -> list[int]:
    generator = random.Random("pack")
    return [generator.randrange(1 << PACK_WIDTH) for _ in range(count)]


@workload("pack u12 values", group="Packing values", unit="values", sizes=VALUE_SIZES, check=same_bits)
def pack_u12(count: int):
    values = u12_values(count)
    code = u12_code()

    def with_bitarray():
        out = Bitarray(endian="big")
        out.encode(code, values)
        return out

    def with_tibs():
        return Tibs.from_values("u12", values)

    return with_bitarray, with_tibs


@workload("unpack u12 values", group="Packing values", unit="values", sizes=VALUE_SIZES)
def unpack_u12(count: int):
    values = u12_values(count)
    # decode() accepts the raw dictionary too, but then it compiles it into a
    # decode tree on every call - a quarter of a millisecond that would swamp
    # the small sizes. decodetree is bitarray's answer to that, so it is built
    # once here alongside the values.
    tree = decodetree(u12_code())
    packed_tibs = Tibs.from_values("u12", values)
    packed_bits = make_bitarray(packed_tibs.to_padded_bytes(), len(packed_tibs))

    def with_bitarray():
        # decode() yields symbols, so the list is built here to match to_values.
        return list(packed_bits.decode(tree))

    def with_tibs():
        return packed_tibs.to_values("u12")

    return with_bitarray, with_tibs


# ---------------------------------------------------------------------------
# Bytes in and out
# ---------------------------------------------------------------------------


@workload("from_bytes", group="Bytes conversion", unit="bytes", sizes=BYTE_SIZES, check=same_bits)
def from_bytes(byte_count: int):
    data = deterministic_bytes(byte_count, "bytes")

    def with_bitarray():
        # frombytes copies, which is what Tibs.from_bytes does. bitarray(buffer=data)
        # avoids the copy but aliases the source and is read-only, so it is a
        # different operation rather than a faster spelling of this one.
        bits = Bitarray(endian="big")
        bits.frombytes(data)
        return bits

    def with_tibs():
        return Tibs.from_bytes(data)

    return with_bitarray, with_tibs


@workload("to_bytes", group="Bytes conversion", unit="bytes", sizes=BYTE_SIZES)
def to_bytes(byte_count: int):
    bits, bit_string = bit_pair(byte_count * 8, "bytes")

    def with_bitarray():
        return bits.tobytes()

    def with_tibs():
        return bit_string.to_bytes()

    return with_bitarray, with_tibs


# ---------------------------------------------------------------------------
# Counting and set algebra
# ---------------------------------------------------------------------------


@workload("population count", group="Counting and set algebra")
def population_count(bit_count: int):
    bits, bit_string = bit_pair(bit_count, "count")

    def with_bitarray():
        return bits.count(1)

    def with_tibs():
        return bit_string.count(1)

    return with_bitarray, with_tibs


@workload("Hamming distance", group="Counting and set algebra")
def hamming_distance(bit_count: int):
    left_bits, left_string = bit_pair(bit_count, "left")
    right_bits, right_string = bit_pair(bit_count, "right")

    def with_bitarray():
        return count_xor(left_bits, right_bits)

    def with_tibs():
        return left_string.count_xor(right_string)

    return with_bitarray, with_tibs


@workload("subset check", group="Counting and set algebra")
def subset_check(bit_count: int):
    # The candidate is the superset masked down, so it really is a subset and
    # the answer is True. A False answer would let either library stop at the
    # first offending bit, which would measure the data rather than the code.
    superset_bits, superset_string = bit_pair(bit_count, "superset")
    mask_bits, mask_string = bit_pair(bit_count, "mask")
    candidate_bits = superset_bits & mask_bits
    candidate_string = superset_string & mask_string

    def with_bitarray():
        return subset(candidate_bits, superset_bits)

    def with_tibs():
        return candidate_string.is_subset_of(superset_string)

    return with_bitarray, with_tibs


@workload("intersects check", group="Counting and set algebra")
def intersects_check(bit_count: int):
    # Complementary inputs are disjoint by construction, so the answer is False
    # and both libraries have to reach the end to know it. Overlapping inputs
    # would stop at the first shared bit, typically within the first word.
    bits, bit_string = bit_pair(bit_count, "intersects")
    other_bits = ~bits
    other_string = ~bit_string

    def with_bitarray():
        return any_and(bits, other_bits)

    def with_tibs():
        return bit_string.intersects(other_string)

    return with_bitarray, with_tibs


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------

NEEDLE_BITS = 32


def haystack_and_needle(bit_count: int, offset: int, seed: str):
    """A haystack, plus a needle lifted from three quarters of the way in.

    Taking the needle out of the haystack guarantees a hit, and taking it from
    late on means most of the data has to be scanned to find it. ``offset`` is
    added to the byte-aligned position, so a caller can ask for a needle that
    does or does not start on a byte boundary.
    """
    bits, bit_string = bit_pair(bit_count, seed)
    start = (bit_count * 3 // 4) // 8 * 8 + offset
    length = min(NEEDLE_BITS, bit_count - start)
    return bits, bit_string, bits[start : start + length], bit_string[start : start + length]


@workload("find, byte-aligned", group="Searching")
def find_byte_aligned(bit_count: int):
    bits, bit_string, needle_bits, needle_string = haystack_and_needle(bit_count, 0, "haystack")
    # bytes.find works in whole bytes, which only trims the needle at the
    # smallest size, where there are fewer than 32 bits left to take it from.
    whole_bytes = len(needle_bits) // 8 * 8
    needle_bits = needle_bits[:whole_bytes]
    needle_string = needle_string[:whole_bytes]
    needle_bytes = needle_bits.tobytes()

    def with_bitarray():
        # bitarray has no byte-aligned search mode, and its general find() walks
        # the haystack a bit at a time: it is around a hundred times slower here
        # than handing the bytes to bytes.find, which is what someone parsing a
        # byte-aligned stream would actually write. The tobytes() copy is timed
        # with it because converting is part of doing the job that way.
        return bits.tobytes().find(needle_bytes) * 8

    def with_tibs():
        return bit_string.find(needle_string, byte_aligned=True)

    return with_bitarray, with_tibs


@workload("find, bit-aligned", group="Searching")
def find_bit_aligned(bit_count: int):
    # The needle starts three bits into a byte, so neither library can restrict
    # itself to byte boundaries: this is the general case of the search above.
    bits, bit_string, needle_bits, needle_string = haystack_and_needle(bit_count, 3, "haystack")

    def with_bitarray():
        return bits.find(needle_bits)

    def with_tibs():
        return bit_string.find(needle_string)

    return with_bitarray, with_tibs


@workload("find_all, few hits", group="Searching")
def find_all_sparse(bit_count: int):
    # A 32-bit needle planted at a handful of spread out positions. Random data
    # effectively never contains it by accident, so the hit count stays small
    # however large the haystack gets and the time is all scanning.
    needle_string = Tibs("0x9e3779b9")
    scratch = Mutibs.from_bytes(deterministic_bytes((bit_count + 7) // 8, "sparse"))[:bit_count]
    plant_count = max(1, min(8, bit_count // 1_000))
    for index in range(plant_count):
        position = (index + 1) * bit_count // (plant_count + 1)
        scratch[position : position + len(needle_string)] = needle_string
    bit_string = scratch.to_tibs()
    bits = make_bitarray(scratch.to_padded_bytes(), bit_count)
    needle_bits = make_bitarray(needle_string.to_bytes())

    def with_bitarray():
        return len(list(bits.search(needle_bits)))

    def with_tibs():
        return len(bit_string.find_all(needle_string))

    return with_bitarray, with_tibs


@workload("find_all, many hits", group="Searching")
def find_all_dense(bit_count: int):
    # An 8-bit needle in random data turns up about once every 256 bits, so the
    # number of hits grows with the haystack: a few thousand at a megabit, a few
    # hundred thousand at a hundred megabits. Both libraries report overlapping
    # matches here, so the counts agree.
    bits, bit_string = bit_pair(bit_count, "dense")
    needle_string = Tibs("0b10101010")
    needle_bits = make_bitarray(needle_string.to_bytes())

    def with_bitarray():
        # search() is an iterator; find_all returns a list, so one is built here
        # too rather than comparing a materialised list against a generator.
        return len(list(bits.search(needle_bits)))

    def with_tibs():
        return len(bit_string.find_all(needle_string))

    return with_bitarray, with_tibs


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------


@workload("insert and delete a bit", group="Editing", check=same_bits)
def insert_and_delete(bit_count: int):
    bits, bit_string = bit_pair(bit_count, "editing")
    editable_bits = bits.copy()
    editable_string = bit_string.to_mutibs()
    one = Tibs("0b1")
    # Each call inserts one bit and deletes another, so the length is invariant
    # and a long timing run stays a memmove benchmark instead of turning into a
    # growth benchmark. The positions are fixed rather than drawn per call: the
    # two sides are called a different number of times each, so an advancing
    # position would leave them holding different data.
    insert_at = bit_count // 3
    delete_at = bit_count * 2 // 3

    def with_bitarray():
        editable_bits.insert(insert_at, 1)
        del editable_bits[delete_at]
        return editable_bits

    def with_tibs():
        editable_string.insert(insert_at, one)
        del editable_string[delete_at]
        return editable_string

    return with_bitarray, with_tibs


# ---------------------------------------------------------------------------
# Complete workloads
# ---------------------------------------------------------------------------


@workload("sieve of Eratosthenes", group="Complete workloads")
def sieve(limit: int):
    # Both sides clear multiples with a strided bulk operation rather than a
    # Python loop over indices, which is how each library's own documentation
    # writes a sieve.

    def with_bitarray():
        is_prime = ones(limit, endian="big")
        is_prime[:2] = False
        for candidate in range(2, math.isqrt(limit) + 1):
            if is_prime[candidate]:
                is_prime[candidate * candidate :: candidate] = False
        return is_prime.count()

    def with_tibs():
        is_prime = Mutibs.from_ones(limit)
        is_prime.unset([0, 1])
        for candidate in range(2, math.isqrt(limit) + 1):
            if is_prime[candidate]:
                is_prime.unset(range(candidate * candidate, limit, candidate))
        return is_prime.count()

    return with_bitarray, with_tibs


FEATURE_BITS = 1024
FEATURES_PER_ITEM = 40
REQUIRED_FEATURES = (11, 407, 900)


@workload("fingerprint screen and score", group="Complete workloads", unit="fingerprints", sizes=ITEM_SIZES)
def fingerprint_screen(count: int):
    """Build a fingerprint per item, screen it, and score the survivors.

    This is the shape of Bloom filter and chemical similarity work: bits go in
    by feature number, then the set questions are asked of the result. Mixing
    the writes and the reads is the point - it is a more honest profile than
    timing either half on its own.
    """
    rng = random.Random("fingerprints")
    feature_lists = []
    for index in range(count):
        features = rng.sample(range(FEATURE_BITS), FEATURES_PER_ITEM)
        if index % 16 == 0:
            # Guarantee a steady trickle through the screen, so the scoring path
            # is exercised at every size rather than left to chance.
            features.extend(REQUIRED_FEATURES)
        feature_lists.append(features)

    reference_features = rng.sample(range(FEATURE_BITS), FEATURES_PER_ITEM)
    required_bits = zeros(FEATURE_BITS, endian="big")
    required_bits[list(REQUIRED_FEATURES)] = 1
    reference_bits = zeros(FEATURE_BITS, endian="big")
    reference_bits[reference_features] = 1
    required_string = Tibs.from_zeros(FEATURE_BITS).set_at(REQUIRED_FEATURES)
    reference_string = Tibs.from_zeros(FEATURE_BITS).set_at(reference_features)

    def with_bitarray():
        matched = 0
        score = 0.0
        for features in feature_lists:
            fingerprint = zeros(FEATURE_BITS, endian="big")
            fingerprint[features] = 1
            if subset(required_bits, fingerprint):
                matched += 1
                score += count_and(reference_bits, fingerprint) / count_or(reference_bits, fingerprint)
        return matched, round(score, 6)

    def with_tibs():
        matched = 0
        score = 0.0
        for features in feature_lists:
            fingerprint = Mutibs.from_zeros(FEATURE_BITS)
            fingerprint.set(features)
            if required_string.is_subset_of(fingerprint):
                matched += 1
                score += reference_string.count_and(fingerprint) / reference_string.count_or(fingerprint)
        return matched, round(score, 6)

    return with_bitarray, with_tibs


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

# A sample has to last long enough that clock resolution and scheduler noise are
# a small part of it. The fastest calls here take well under a microsecond.
MIN_SAMPLE_SECONDS = 0.005
MAX_CALLS_PER_SAMPLE = 200_000


def calls_per_sample(function: Callable[[], Any]) -> int:
    """How many calls to fold into one sample to get clear of the noise."""
    started = time.perf_counter()
    function()
    elapsed = time.perf_counter() - started
    if elapsed >= MIN_SAMPLE_SECONDS:
        return 1
    if elapsed <= 0:
        return MAX_CALLS_PER_SAMPLE
    return min(math.ceil(MIN_SAMPLE_SECONDS / elapsed), MAX_CALLS_PER_SAMPLE)


def median_times(
    baseline: Callable[[], Any], candidate: Callable[[], Any], repeats: int
) -> tuple[float, float]:
    """Time both functions, alternating which one goes first.

    Each side gets its own call count, so a case where one side is orders of
    magnitude faster still has both measured over a usable interval. The
    reported time is per call either way.
    """
    functions = {"baseline": baseline, "candidate": candidate}
    counts = {name: calls_per_sample(function) for name, function in functions.items()}
    times: dict[str, list[float]] = {"baseline": [], "candidate": []}
    for repeat in range(repeats):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for name in order:
            function = functions[name]
            calls = counts[name]
            started = time.perf_counter()
            for _ in range(calls):
                function()
            times[name].append((time.perf_counter() - started) / calls)
    return statistics.median(times["baseline"]), statistics.median(times["candidate"])


@dataclass(frozen=True)
class Measurement:
    workload: Workload
    size: int
    bitarray_time: float
    tibs_time: float

    @property
    def speedup(self) -> float:
        """How many times faster tibs was; below one means bitarray was faster."""
        return self.bitarray_time / self.tibs_time if self.tibs_time else float("inf")


def run(workloads: Iterable[Workload], repeats: int, progress: bool) -> list[Measurement]:
    measurements = []
    for item in workloads:
        for size in item.sizes:
            if progress:
                label = f"timing {item.name} at {format_size(size, item.unit)}"
                print(f"\r{label:<60}", end="", file=sys.stderr, flush=True)
            with_bitarray, with_tibs = item.build(size)
            bitarray_result = with_bitarray()
            tibs_result = with_tibs()
            if not item.check(bitarray_result, tibs_result):
                raise AssertionError(
                    f"{item.name} at {format_size(size, item.unit)} disagreed: "
                    f"bitarray={summarise(bitarray_result)}, tibs={summarise(tibs_result)}"
                )
            del bitarray_result, tibs_result
            bitarray_time, tibs_time = median_times(with_bitarray, with_tibs, repeats)
            measurements.append(Measurement(item, size, bitarray_time, tibs_time))
            # Dropping the closures releases the prepared inputs, which for the
            # largest cases is a few hundred megabytes.
            del with_bitarray, with_tibs
    if progress:
        print(f"\r{'':<60}\r", end="", file=sys.stderr, flush=True)
    return measurements


def summarise(result: Any) -> str:
    if isinstance(result, (Tibs, Mutibs) if bitarray is None else (Bitarray, Tibs, Mutibs)):
        return f"{type(result).__name__}(len={len(result):,}, ones={result.count(1):,})"
    if isinstance(result, list) and len(result) > 10:
        return f"list(len={len(result):,}, first={result[0]!r}, last={result[-1]!r})"
    return repr(result)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

SIZE_SUFFIXES = {
    "bits": ((1_000_000_000, "Gbit"), (1_000_000, "Mbit"), (1_000, "kbit")),
    "bytes": ((1_000_000_000, "GB"), (1_000_000, "MB"), (1_000, "KB")),
}


def format_size(size: int, unit: str) -> str:
    for limit, suffix in SIZE_SUFFIXES.get(unit, ()):
        if size >= limit and size % limit == 0:
            return f"{size // limit} {suffix}"
    return f"{size:,} {unit}"


def short_number(value: int) -> str:
    for limit, suffix in ((1_000_000_000, "G"), (1_000_000, "M"), (1_000, "k")):
        if value >= limit and value % limit == 0:
            return f"{value // limit}{suffix}"
    return f"{value:,}"


TIME_UNITS = ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns"))


def format_time(seconds: float) -> str:
    for scale, suffix in TIME_UNITS:
        if seconds >= scale:
            return f"{seconds / scale:.3g} {suffix}"
    return f"{seconds / 1e-9:.3g} ns"


def format_verdict(speedup: float) -> str:
    """Which library was faster, and by how much, or a dash for a tie.

    Anything within five percent is called a tie: repeated runs of this script
    move by about that much, so a smaller difference is not worth reporting.
    """
    if 0.95 <= speedup <= 1.05:
        return "-"
    if speedup > 1:
        return f"tibs {speedup:.2f}×"
    return f"bitarray {1 / speedup:.2f}×"


def format_overall(speedup: float) -> str:
    """Like format_verdict, but a summary always names a direction."""
    side = "tibs" if speedup >= 1 else "bitarray"
    factor = speedup if speedup >= 1 else 1 / speedup
    if 0.95 <= speedup <= 1.05:
        return f"{factor:.2f}× to {side}, which is level"
    return f"{side} {factor:.2f}×"


def geometric_mean(values: Sequence[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def summary_lines(measurements: Sequence[Measurement]) -> list[str]:
    """The two summary statistics, which are ratios of ratios - handle with care.

    A single number cannot say which library is faster: these workloads are not
    weighted by how often anyone runs them, and the spread between the best and
    worst case is three orders of magnitude. The per-case rows are the result;
    these lines are only a coarse check that nothing has moved wholesale.
    """
    speedups = [measurement.speedup for measurement in measurements]
    return [
        f"{len(measurements)} measurements, {environment()}",
        f"Geometric mean of the per-case ratios: {format_overall(geometric_mean(speedups))}",
        f"Median of the per-case ratios: {format_overall(statistics.median(speedups))}",
    ]


def environment() -> str:
    return (
        f"{platform.python_implementation()} {platform.python_version()} · "
        f"tibs {tibs.__version__} · bitarray {bitarray.__version__} · "
        f"{platform.system()} {platform.machine()}"
    )


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

NAME_WIDTH = 30
SIZE_WIDTH = 21
TIME_WIDTH = 12
VERDICT_WIDTH = 16


def print_table(measurements: Sequence[Measurement]) -> None:
    header = (
        f"{'workload':<{NAME_WIDTH}}{'size':>{SIZE_WIDTH}}"
        f"{'bitarray':>{TIME_WIDTH}}{'tibs':>{TIME_WIDTH}}  {'faster':<{VERDICT_WIDTH}}"
    )
    rule = "-" * len(header.rstrip())
    group = None
    for measurement in measurements:
        item = measurement.workload
        if item.group != group:
            group = item.group
            print()
            print(group)
            print(header.rstrip())
            print(rule)
        print(
            (
                f"{item.name:<{NAME_WIDTH}}"
                f"{format_size(measurement.size, item.unit):>{SIZE_WIDTH}}"
                f"{format_time(measurement.bitarray_time):>{TIME_WIDTH}}"
                f"{format_time(measurement.tibs_time):>{TIME_WIDTH}}"
                f"  {format_verdict(measurement.speedup)}"
            ).rstrip()
        )
    print()
    for line in summary_lines(measurements):
        print(line)


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------


def markdown_table(measurements: Sequence[Measurement], repo: str, ref: str) -> str:
    """A table for the README, with every row linked to the code behind it."""
    lines = [
        "| Workload | Size | bitarray | tibs | Faster |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    anchors: dict[str, str] = {}
    for measurement in measurements:
        item = measurement.workload
        anchor = item.name.replace(" ", "-").replace(",", "")
        anchors[anchor] = f"{repo}/blob/{ref}/tests/{item.source_url}"
        lines.append(
            f"| [{item.name}][{anchor}] "
            f"| {format_size(measurement.size, item.unit)} "
            f"| {format_time(measurement.bitarray_time)} "
            f"| {format_time(measurement.tibs_time)} "
            f"| {format_verdict(measurement.speedup)} |"
        )
    lines.append("")
    lines.extend(f"{line}  " for line in summary_lines(measurements))
    lines.append("")
    lines.extend(f"[{anchor}]: {url}" for anchor, url in anchors.items())
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

# Two series, so two categorical hues, each also carrying a marker shape: colour
# alone should not be the only thing telling them apart.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "panel": "#ffffff",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e3e2df",
        "bitarray": "#2a78d6",
        "tibs": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "panel": "#232322",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a38",
        "bitarray": "#3987e5",
        "tibs": "#d95926",
    },
}

PANEL_WIDTH = 250
PANEL_HEIGHT = 175
PANEL_COLUMNS = 3
PLOT_LEFT = 52
PLOT_RIGHT = 12
PLOT_TOP = 30
PLOT_BOTTOM = 40
CHART_MARGIN = 18
HEADER_HEIGHT = 84


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def log_position(value: float, low: float, high: float, start: float, end: float) -> float:
    if high <= low:
        return (start + end) / 2
    fraction = (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))
    return start + fraction * (end - start)


def decade_ticks(low: float, high: float, limit: int = 4) -> list[float]:
    """Decade gridlines across a range, thinned out if there are too many."""
    first = math.floor(math.log10(low))
    last = math.ceil(math.log10(high))
    exponents = list(range(first, last + 1))
    stride = max(1, math.ceil(len(exponents) / limit))
    return [10.0**exponent for exponent in exponents[::stride]]


def marker(shape: str, x: float, y: float, colour: str, surface: str) -> str:
    """A data point, ringed in the surface colour so overlaps stay readable."""
    if shape == "square":
        return (
            f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1.5" '
            f'fill="{colour}" stroke="{surface}" stroke-width="2" />'
        )
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colour}" stroke="{surface}" stroke-width="2" />'


def panel_svg(item: Workload, points: Sequence[Measurement], origin_x: float, origin_y: float, colours: dict[str, str], repo: str, ref: str) -> str:
    left = origin_x + PLOT_LEFT
    right = origin_x + PANEL_WIDTH - PLOT_RIGHT
    top = origin_y + PLOT_TOP
    bottom = origin_y + PANEL_HEIGHT - PLOT_BOTTOM

    sizes = [point.size for point in points]
    times = [time for point in points for time in (point.bitarray_time, point.tibs_time)]
    size_low, size_high = min(sizes), max(sizes)
    time_low = 10 ** math.floor(math.log10(min(times)))
    time_high = 10 ** math.ceil(math.log10(max(times)))
    if time_high <= time_low:
        time_high = time_low * 10

    parts = [
        f'<rect x="{origin_x}" y="{origin_y}" width="{PANEL_WIDTH}" height="{PANEL_HEIGHT}" '
        f'rx="6" fill="{colours["panel"]}" />'
    ]
    title = escape(item.name)
    url = f"{repo}/blob/{ref}/tests/{item.source_url}"
    parts.append(
        f'<a href="{escape(url)}"><text x="{origin_x + 12}" y="{origin_y + 19}" class="title">{title}</text></a>'
    )

    for tick in decade_ticks(time_low, time_high):
        y = log_position(tick, time_low, time_high, bottom, top)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{colours["grid"]}" stroke-width="1" />'
        )
        parts.append(f'<text x="{left - 8}" y="{y + 3.5:.1f}" class="tick end">{escape(format_time(tick))}</text>')

    for size in sizes:
        x = log_position(size, size_low, size_high, left, right) if size_high > size_low else (left + right) / 2
        parts.append(f'<text x="{x:.1f}" y="{bottom + 16}" class="tick mid">{escape(short_number(size))}</text>')
    parts.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{colours["grid"]}" stroke-width="1" />'
    )
    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="{bottom + 30}" class="axis mid">{escape(item.unit)}</text>'
    )

    for key, shape in (("bitarray", "circle"), ("tibs", "square")):
        coordinates = []
        for point in points:
            x = log_position(point.size, size_low, size_high, left, right) if size_high > size_low else (left + right) / 2
            value = point.bitarray_time if key == "bitarray" else point.tibs_time
            coordinates.append((x, log_position(value, time_low, time_high, bottom, top)))
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
        parts.append(
            f'<polyline points="{path}" fill="none" stroke="{colours[key]}" stroke-width="2" '
            'stroke-linejoin="round" stroke-linecap="round" />'
        )
        parts.extend(marker(shape, x, y, colours[key], colours["panel"]) for x, y in coordinates)
    return "".join(parts)


def chart_svg(measurements: Sequence[Measurement], theme: str, repo: str, ref: str) -> str:
    colours = THEMES[theme]
    grouped: dict[str, list[Measurement]] = {}
    for measurement in measurements:
        grouped.setdefault(measurement.workload.name, []).append(measurement)
    panels = list(grouped.values())
    rows = math.ceil(len(panels) / PANEL_COLUMNS)
    width = CHART_MARGIN * 2 + PANEL_COLUMNS * PANEL_WIDTH
    height = HEADER_HEIGHT + rows * PANEL_HEIGHT + CHART_MARGIN + 28

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="'
        f'system-ui, -apple-system, Segoe UI, Roboto, sans-serif">',
        "<style>"
        f'.title{{font-size:12px;font-weight:600;fill:{colours["text"]}}}'
        f'.tick{{font-size:9.5px;fill:{colours["muted"]}}}'
        f'.axis{{font-size:10px;fill:{colours["muted"]}}}'
        f'.heading{{font-size:19px;font-weight:600;fill:{colours["text"]}}}'
        f'.sub{{font-size:11.5px;fill:{colours["muted"]}}}'
        f'.legend{{font-size:12px;fill:{colours["text"]}}}'
        ".end{text-anchor:end}.mid{text-anchor:middle}"
        "a{text-decoration:none}"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="{colours["surface"]}" />',
        f'<text x="{CHART_MARGIN}" y="30" class="heading">tibs and bitarray: time against size</text>',
        f'<text x="{CHART_MARGIN}" y="49" class="sub">Median time for one call. Both axes are logarithmic, '
        f'so a straight line is a power law and a lower line is faster.</text>',
    ]

    legend_x = CHART_MARGIN
    for key, shape, label in (("bitarray", "circle", "bitarray"), ("tibs", "square", "tibs")):
        parts.append(
            f'<line x1="{legend_x}" y1="68" x2="{legend_x + 26}" y2="68" stroke="{colours[key]}" stroke-width="2" />'
        )
        parts.append(marker(shape, legend_x + 13, 68, colours[key], colours["surface"]))
        parts.append(f'<text x="{legend_x + 34}" y="72" class="legend">{label}</text>')
        legend_x += 110

    for index, points in enumerate(panels):
        origin_x = CHART_MARGIN + (index % PANEL_COLUMNS) * PANEL_WIDTH
        origin_y = HEADER_HEIGHT + (index // PANEL_COLUMNS) * PANEL_HEIGHT
        parts.append(panel_svg(points[0].workload, points, origin_x, origin_y, colours, repo, ref))

    parts.append(f'<text x="{CHART_MARGIN}" y="{height - 14}" class="sub">{escape(environment())}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def selected_workloads(patterns: Sequence[str], quick: bool) -> list[Workload]:
    chosen = [
        item
        for item in WORKLOADS
        if not patterns or any(pattern.lower() in item.name.lower() for pattern in patterns)
    ]
    if quick:
        # Dropping the top size takes the run from minutes to seconds, at the
        # cost of the part of the curve where the libraries differ most.
        chosen = [
            Workload(**{**item.__dict__, "sizes": item.sizes[:-1] or item.sizes})
            for item in chosen
        ]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=5, help="timing samples per case (default 5)")
    parser.add_argument("--filter", action="append", default=[], metavar="TEXT", help="only workloads whose name contains TEXT; repeatable")
    parser.add_argument("--quick", action="store_true", help="skip the largest size of every workload")
    parser.add_argument("--markdown", metavar="PATH", help="write a linked markdown table ('-' for stdout)")
    parser.add_argument("--svg", metavar="PATH", help="write a time against size chart")
    parser.add_argument("--json", metavar="PATH", help="write the raw measurements ('-' for stdout)")
    parser.add_argument("--theme", choices=tuple(THEMES), default="light", help="chart colours (default light)")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="repository the source links point into")
    parser.add_argument("--ref", default="main", help="branch or tag the source links point at (default main)")
    parser.add_argument("--list", action="store_true", help="list the workloads and exit")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    chosen = selected_workloads(args.filter, args.quick)
    if not chosen:
        parser.error("no workloads matched --filter")

    if args.list:
        for item in chosen:
            sizes = ", ".join(format_size(size, item.unit) for size in item.sizes)
            print(f"{item.name:<32}{item.group:<28}{sizes}")
        return

    if bitarray is None:
        raise SystemExit("bitarray is not installed; install it to run this comparison.")

    measurements = run(chosen, args.repeats, sys.stderr.isatty())

    print_table(measurements)

    if args.markdown:
        write(args.markdown, markdown_table(measurements, args.repo, args.ref))
    if args.json:
        payload = {
            "environment": environment(),
            "measurements": [
                {
                    "workload": measurement.workload.name,
                    "group": measurement.workload.group,
                    "source": measurement.workload.source_url,
                    "size": measurement.size,
                    "unit": measurement.workload.unit,
                    "bitarray_seconds": measurement.bitarray_time,
                    "tibs_seconds": measurement.tibs_time,
                }
                for measurement in measurements
            ],
        }
        write(args.json, json.dumps(payload, indent=2) + "\n")
    if args.svg:
        write(args.svg, chart_svg(measurements, args.theme, args.repo, args.ref))


def write(path: str, text: str) -> None:
    if path == "-":
        print()
        print(text, end="")
        return
    with open(path, "w", encoding="utf-8") as output:
        output.write(text)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
