#!/usr/bin/env python3
"""Local performance comparison between tibs and other ways of doing the job.

Two separate tables, answering two different questions:

* against bitarray: is tibs competitive with the other bit library?
* against the standard library: do you need a bit library at all?

Each case uses equivalent prepared inputs and idiomatic operations on both
sides. Results are checked for equivalence before timing. This is useful for
finding optimization opportunities and regressions, but is not an exhaustive
overall score for anything.
"""

import argparse
import array
from collections.abc import Callable
from dataclasses import dataclass
import math
import operator
import os
import pickle
import random
import statistics
import struct
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tibs import Dtype, Mutibs, Tibs

try:
    from bitarray import bitarray
    from bitarray.util import ba2int, int2ba, ones, random_p, zeros
except ImportError:
    bitarray = None


def deterministic_bytes(size, seed):
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(size))


@dataclass(frozen=True)
class ComparisonCase:
    name: str
    baseline_fn: Callable[[], Any]
    tibs_fn: Callable[[], Any]
    equivalent: Callable[[Any, Any], bool] = operator.eq


# A single sample has to last long enough that clock resolution and scheduler
# noise are a small part of it. The fastest cases here take a few microseconds,
# where timing one call at a time gives swings of 2x in either direction.
MIN_SAMPLE_SECONDS = 0.002
MAX_CALLS_PER_SAMPLE = 100_000


def calls_per_sample(fn):
    """How many calls to fold into one timed sample to get clear of the noise."""
    started = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - started
    if elapsed >= MIN_SAMPLE_SECONDS:
        return 1
    if elapsed <= 0:
        return MAX_CALLS_PER_SAMPLE
    calls = math.ceil(MIN_SAMPLE_SECONDS / elapsed)
    return min(calls, MAX_CALLS_PER_SAMPLE)


def median_times(baseline_fn, tibs_fn, repeats):
    """Time both functions while alternating which one runs first.

    Each side gets its own call count, so a case where one side is orders of
    magnitude faster still has both sides measured over a usable interval. The
    reported time is per call either way.
    """
    functions = {"baseline": baseline_fn, "tibs": tibs_fn}
    counts = {name: calls_per_sample(fn) for name, fn in functions.items()}
    times = {"baseline": [], "tibs": []}
    for repeat in range(repeats):
        order = ("baseline", "tibs") if repeat % 2 == 0 else ("tibs", "baseline")
        for name in order:
            function = functions[name]
            calls = counts[name]
            started = time.perf_counter()
            for _ in range(calls):
                function()
            times[name].append((time.perf_counter() - started) / calls)
    return statistics.median(times["baseline"]), statistics.median(times["tibs"])


def make_bitarray(data):
    # frombytes copies, which matches Tibs.from_bytes. bitarray(buffer=data) is
    # a zero-copy alternative, but it aliases the source and is read-only, so it
    # is not the same operation.
    bits = bitarray(endian="big")
    bits.frombytes(data)
    return bits


def same_bits(bitarray_result, tibs_result):
    return (
            len(bitarray_result) == len(tibs_result)
            and bitarray_result.tobytes() == tibs_result.to_padded_bytes()
    )


def same_bytes(baseline_result, tibs_result):
    return baseline_result == tibs_result.to_bytes()


RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
BOLD = "\033[1m"

SLOW_THRESHOLD = 0.8
FAST_THRESHOLD = 1.2

NAME_WIDTH = 26
TIME_WIDTH = 13
SPEEDUP_WIDTH = 9
PAIR_WIDTH = TIME_WIDTH + SPEEDUP_WIDTH


def colourise(text, colour, enabled):
    return f"{colour}{text}{RESET}" if enabled and colour else text


def speedup_colour(speedup):
    if speedup < SLOW_THRESHOLD:
        return RED
    if speedup > FAST_THRESHOLD:
        return GREEN
    return ""


def relative_factor(speedup):
    """How many times faster the faster side is, whichever side that is."""
    if speedup >= 1:
        return speedup
    return 1 / speedup if speedup else float("inf")


def relative_cells(speedup):
    """Format both relative columns, leaving the slower side blank."""
    factor = relative_factor(speedup)
    if factor < 1.005:
        # Would print as 1.00x, which reads as a win it didn't have.
        return "", ""
    text = f"{factor:.2f}x"
    return ("", text) if speedup >= 1 else (text, "")


def relative_cell(text, cell_colour, colour):
    return colourise(text.rjust(SPEEDUP_WIDTH), cell_colour if text else "", colour)


TIME_UNITS = ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns"))


def format_time(seconds):
    """Use a readable unit even for scalar cases that take under a microsecond."""
    for scale, suffix in TIME_UNITS:
        if seconds >= scale:
            return f"{seconds / scale:.3g} {suffix}"
    return f"{seconds / 1e-9:.3g} ns"


def result_summary(result):
    bit_types = (Tibs, Mutibs) if bitarray is None else (bitarray, Tibs, Mutibs)
    if isinstance(result, bit_types):
        return f"{type(result).__name__}(len={len(result):,}, ones={result.count(1):,})"
    if isinstance(result, list) and len(result) > 10:
        return f"list(len={len(result):,}, first={result[0]!r}, last={result[-1]!r})"
    return repr(result)


def build_bitarray_cases(byte_count, value_count):
    search_bytes = deterministic_bytes(byte_count, "comparison-search")
    other_bytes = deterministic_bytes(byte_count, "comparison-other")
    search_tibs = Tibs.from_bytes(search_bytes)
    other_tibs = Tibs.from_bytes(other_bytes)
    search_bits = make_bitarray(search_bytes)
    other_bits = make_bitarray(other_bytes)
    bit_count = len(search_bits)
    bit_list = list(search_tibs[: min(byte_count * 8, 80_000)])

    all_zero_bits = zeros(bit_count, endian="big")
    all_one_bits = ones(bit_count, endian="big")
    all_zero_tibs = Tibs.from_zeros(bit_count)
    all_one_tibs = Tibs.from_ones(bit_count)

    deposit_mask_bits = other_bits
    deposit_mask_tibs = other_tibs
    deposit_value_length = deposit_mask_bits.count(1)
    deposit_value_bits = search_bits[:deposit_value_length]
    deposit_value_tibs = search_tibs[:deposit_value_length]

    search_pattern_bits = bitarray("101010111100", endian="big")
    search_pattern_tibs = Tibs("0xabc")

    find_pattern_len = min(137, bit_count)
    find_pattern_start = (bit_count - find_pattern_len) // 3
    find_pattern_bits = search_bits[
        find_pattern_start: find_pattern_start + find_pattern_len
    ]
    find_pattern_tibs = search_tibs[
        find_pattern_start: find_pattern_start + find_pattern_len
    ]

    slice_start = min(3, bit_count)
    slice_end = max(slice_start, bit_count - 4)

    mutation_width = min(10_000, max(1, bit_count // 8))
    mutation_start = (bit_count - mutation_width) // 2
    mutation_end = mutation_start + mutation_width
    mutation_bits = bitarray(mutation_width, endian="big")
    mutation_bits.setall(1)
    mutation_tibs = Tibs.from_ones(mutation_width)

    rng = random.Random("comparison-index-reads")
    read_positions = [rng.randrange(bit_count) for _ in range(value_count)]

    rng = random.Random("comparison-values")
    value_words = [rng.randrange(1 << 16) for _ in range(value_count)]
    value_bytes = Tibs.from_values("u16", value_words).to_bytes()

    bulk_set_bit_count = min(byte_count * 8, 2_000_000)
    bulk_set_width = 8
    rng = random.Random("comparison-bulk-set")
    bulk_set_positions = [
        [rng.randrange(bulk_set_bit_count) for _ in range(bulk_set_width)]
        for _ in range(value_count)
    ]

    sieve_limit = min(bit_count, 2_000_000)
    random_bit_count = min(bit_count, 2_000_000)
    random_repeats = 10

    pop_count = min(bit_count, 200_000)
    pop_bits = search_bits[:pop_count]
    pop_tibs = search_tibs[:pop_count]

    ba_piece = bitarray("10101", endian="big")
    tibs_piece = Tibs("0b10101")
    ba_pieces = [ba_piece] * 50_000
    tibs_pieces = [tibs_piece] * 50_000
    repeat_pattern_bits = bitarray("1011001011", endian="big")
    repeat_pattern_tibs = Tibs("0b1011001011")
    repeat_count = bit_count // len(repeat_pattern_bits)
    small_and_widths = (1, 7, 13, 31, 47, 63, 64)
    small_and_values = [
        (
            0x123456789ABCDEF0 & ((1 << width) - 1),
            0xFEDCBA9876543210 & ((1 << width) - 1),
        )
        for width in small_and_widths
    ]
    small_and_bits = [
        (
            int2ba(left, length=width, endian="big"),
            int2ba(right, length=width, endian="big"),
        )
        for width, (left, right) in zip(
            small_and_widths, small_and_values, strict=True
        )
    ]
    small_and_tibs = [
        (Tibs.from_u(left, width), Tibs.from_u(right, width))
        for width, (left, right) in zip(
            small_and_widths, small_and_values, strict=True
        )
    ]
    chunk_target_bits = bitarray("11111", endian="big")
    chunk_target_tibs = Tibs("0b11111")

    def ba_find_all():
        return len(list(search_bits.search(search_pattern_bits)))

    def tibs_find_all():
        return len(search_tibs.find_all(search_pattern_tibs))

    def ba_find():
        return search_bits.find(find_pattern_bits)

    def tibs_find():
        return search_tibs.find(find_pattern_tibs)

    def ba_rfind():
        return search_bits.find(find_pattern_bits, 0, len(search_bits), right=True)

    def tibs_rfind():
        return search_tibs.rfind(find_pattern_tibs)

    def ba_reverse_find_all():
        return len(list(search_bits.search(search_pattern_bits, right=True)))

    def tibs_reverse_find_all():
        return len(list(search_tibs.rfind_all_iter(search_pattern_tibs)))

    def ba_bitops():
        end = min(byte_count * 8, 500_000)
        for _ in range(50):
            combined = search_bits | other_bits
            result = combined[10:end] & other_bits[9: end - 1]
        return result

    def tibs_bitops():
        end = min(byte_count * 8, 500_000)
        for _ in range(50):
            combined = search_tibs | other_tibs
            result = combined[10:end] & other_tibs[9: end - 1]
        return result

    def ba_small_ands():
        return [left & right for left, right in small_and_bits]

    def tibs_small_ands():
        return [left & right for left, right in small_and_tibs]

    def same_small_bits(bitarray_results, tibs_results):
        return len(bitarray_results) == len(tibs_results) and all(
            same_bits(bitarray_result, tibs_result)
            for bitarray_result, tibs_result in zip(
                bitarray_results, tibs_results, strict=True
            )
        )

    # Start from a fresh mutable copy on each call, so repeated samples do the
    # same work instead of repeatedly ANDing an already-filtered result.
    def ba_inplace_and():
        out = search_bits.copy()
        out &= other_bits
        return out

    def tibs_inplace_and():
        out = search_tibs.to_mutibs()
        out &= other_tibs
        return out

    def ba_deposit():
        out = search_bits.copy()
        out[deposit_mask_bits] = deposit_value_bits
        return out

    def tibs_deposit():
        out = search_tibs.to_mutibs()
        out.deposit(deposit_value_tibs, deposit_mask_tibs)
        return out

    def ba_whole_view_write():
        out = search_bits.copy()
        out[:] = other_bits
        return out

    def tibs_whole_view_write():
        out = search_tibs.to_mutibs()
        out.view().write_bytes(other_bytes)
        return out

    # bitarray short-circuits these predicates. The immediate-exit rows put
    # the deciding bit first; the full-scan rows make every bit agree so that
    # both implementations have to inspect the whole input.
    def ba_all_immediate():
        return all_zero_bits.all()

    def tibs_all_immediate():
        return all_zero_tibs.all()

    def ba_any_immediate():
        return all_one_bits.any()

    def tibs_any_immediate():
        return all_one_tibs.any()

    def ba_all_full_scan():
        return all_one_bits.all()

    def tibs_all_full_scan():
        return all_one_tibs.all()

    def ba_any_full_scan():
        return all_zero_bits.any()

    def tibs_any_full_scan():
        return all_zero_tibs.any()

    def ba_count():
        return sum(search_bits.count(1) for _ in range(10))

    def tibs_count():
        return sum(search_tibs.count(1) for _ in range(10))

    def ba_join_small_pieces():
        out = bitarray(endian="big")
        for item in ba_pieces:
            out.extend(item)
        return out

    def tibs_join_small_pieces():
        return Tibs.from_joined(tibs_pieces)

    def ba_extend_bool_list():
        out = bitarray(endian="big")
        out.extend(bit_list)
        return out

    def tibs_extend_bool_list():
        out = Mutibs()
        out.extend(bit_list)
        return out

    def ba_append_bits():
        out = bitarray(endian="big")
        append = out.append
        for bit in bit_list:
            append(bit)
        return out

    def tibs_append_bits():
        out = Mutibs()
        # Tibs documents reserve as the way to avoid growth reallocations when
        # constructing incrementally, so give it the known final size.
        out.reserve(len(bit_list))
        append = out.append
        for bit in bit_list:
            append(bit)
        return out

    def ba_bulk_index_set():
        out = zeros(bulk_set_bit_count, endian="big")
        for positions in bulk_set_positions:
            out[positions] = 1
        return out

    def tibs_bulk_index_set():
        out = Mutibs.from_zeros(bulk_set_bit_count)
        for positions in bulk_set_positions:
            out.set(positions)
        return out

    def ba_bool_construction():
        return bitarray(bit_list, endian="big")

    def tibs_bool_construction():
        return Tibs.from_bools(bit_list)

    def ba_from_bytes():
        return make_bitarray(search_bytes)

    def tibs_from_bytes():
        return Tibs.from_bytes(search_bytes)

    def ba_to_bytes():
        return search_bits.tobytes()

    def tibs_to_bytes():
        return search_tibs.to_bytes()

    def ba_slice():
        for _ in range(100):
            result = search_bits[slice_start:slice_end]
        return result

    def tibs_slice():
        for _ in range(100):
            result = search_tibs[slice_start:slice_end]
        return result

    def ba_concat():
        return search_bits + other_bits

    def tibs_concat():
        return search_tibs + other_tibs

    def ba_repeat_pattern():
        return repeat_pattern_bits * repeat_count

    def tibs_repeat_pattern():
        return repeat_pattern_tibs * repeat_count

    def ba_invert():
        return ~search_bits

    def tibs_invert():
        return ~search_tibs

    # Both libraries reverse in place, so this measures the reverse itself
    # rather than a copy. The two buffers are reversed the same number of
    # times, so they stay in step with each other.
    reverse_bits = search_bits.copy()
    reverse_tibs = search_tibs.to_mutibs()

    def ba_reverse():
        reverse_bits.reverse()
        return reverse_bits

    def tibs_reverse():
        reverse_tibs.reverse()
        return reverse_tibs

    def ba_shift_left():
        return search_bits << 13

    def tibs_shift_left():
        return search_tibs << 13

    def ba_index_reads():
        # This row isolates scalar indexing on both sides. The separate
        # gather-and-count row below lets bitarray use its bulk indexing API.
        return sum(search_bits[pos] for pos in read_positions)

    def tibs_index_reads():
        return sum(search_tibs[pos] for pos in read_positions)

    def ba_gather_count():
        return search_bits[read_positions].count(1)

    def tibs_gather_count():
        # Tibs has no fancy-indexing read, so scalar indexing is its idiomatic
        # route for gathering arbitrary positions.
        return sum(search_tibs[pos] for pos in read_positions)

    def ba_to_bool_list():
        return search_bits.tolist()

    def tibs_to_bool_list():
        return search_tibs.to_bools()

    def ba_copy_slice_set():
        out = search_bits.copy()
        out[mutation_start:mutation_end] = mutation_bits
        return out

    def tibs_copy_slice_set():
        out = search_tibs.to_mutibs()
        out[mutation_start:mutation_end] = mutation_tibs
        return out

    def ba_copy_slice_delete():
        out = search_bits.copy()
        del out[mutation_start:mutation_end]
        return out

    def tibs_copy_slice_delete():
        out = search_tibs.to_mutibs()
        del out[mutation_start:mutation_end]
        return out

    def ba_slice_count():
        total = 0
        limit = min(byte_count * 8 - 257, 75_000)
        for start in range(0, limit, 5):
            total += search_bits.count(1, start, start + 257)
        return total

    def tibs_slice_count():
        total = 0
        limit = min(byte_count * 8 - 257, 75_000)
        for start in range(0, limit, 5):
            total += search_tibs.count(1, start, start + 257)
        return total

    def ba_pack_u16():
        # struct.pack + frombytes is ~50x faster than this loop, but struct is
        # equally available to tibs, so using it would compare struct with
        # itself rather than the two libraries. int2ba is bitarray's own route,
        # and it is the only route for a non-byte-aligned width such as u12.
        out = bitarray(endian="big")
        for value in value_words:
            out.extend(int2ba(value, length=16, endian="big"))
        return out

    def tibs_pack_u16():
        return Tibs.from_values("u16", value_words)

    def ba_unpack_u16():
        # As with packing, struct.unpack is excluded as it is available to both.
        bits = make_bitarray(value_bytes)
        return [ba2int(bits[index: index + 16]) for index in range(0, len(bits), 16)]

    def tibs_unpack_u16():
        return Tibs.from_bytes(value_bytes).to_values("u16")

    buffer_view_repeats = 2_000

    def ba_buffer_view():
        total = 0
        for _ in range(buffer_view_repeats):
            total += len(memoryview(search_bits))
        return total

    def tibs_buffer_view():
        total = 0
        for _ in range(buffer_view_repeats):
            total += len(memoryview(search_tibs))
        return total

    def ba_primes():
        is_prime = ones(sieve_limit)
        is_prime[:2] = False
        for i in range(2, math.isqrt(sieve_limit) + 1):
            if is_prime[i]:
                is_prime[i * i:: i] = False
        # bitarray counts non-overlapping occurrences, so it misses one of the
        # two overlapping "101" hits in 3, 5, 7 - the only prime triple there is.
        return is_prime.count(bitarray("101")) + 1

    def tibs_primes():
        is_prime = Mutibs.from_ones(sieve_limit)
        is_prime.unset([0, 1])
        for i in range(2, math.isqrt(sieve_limit) + 1):
            if is_prime[i]:
                is_prime.unset(range(i * i, sieve_limit, i))
        return is_prime.count([1, 0, 1])

    def ba_random():
        for _ in range(random_repeats):
            out = random_p(random_bit_count)
        return len(out)

    def tibs_random():
        for _ in range(random_repeats):
            out = Mutibs.from_random(random_bit_count)
        return len(out)

    def ba_pop():
        out = pop_bits.copy()
        pop = out.pop
        total = 0
        while out:
            total += pop()
        return total

    def tibs_pop():
        out = pop_tibs.to_mutibs()
        pop = out.pop
        total = 0
        while out:
            total += pop()
        return total

    def ba_chunks():
        # bitarray has no chunk iterator, and count(sub_bitarray) counts
        # non-overlapping hits at any offset rather than on a 5-bit stride,
        # so the slice-compare loop is the equivalent spelling.
        count = 0
        for index in range(0, len(search_bits), 5):
            if search_bits[index: index + 5] == chunk_target_bits:
                count += 1
        return count

    def tibs_chunks():
        return sum(
            1 for chunk in search_tibs.chunks_iter(5) if chunk == chunk_target_tibs
        )

    return [
        ComparisonCase("find_all 12-bit", ba_find_all, tibs_find_all),
        ComparisonCase("find", ba_find, tibs_find),
        ComparisonCase("rfind", ba_rfind, tibs_rfind),
        ComparisonCase(
            "reverse find_all iter", ba_reverse_find_all, tibs_reverse_find_all
        ),
        ComparisonCase("bit ops sliced", ba_bitops, tibs_bitops, same_bits),
        ComparisonCase(
            "1-64-bit and",
            ba_small_ands,
            tibs_small_ands,
            same_small_bits,
        ),
        ComparisonCase(
            "copy + in-place and", ba_inplace_and, tibs_inplace_and, same_bits
        ),
        ComparisonCase("masked bit deposit", ba_deposit, tibs_deposit, same_bits),
        ComparisonCase(
            "copy + whole view write",
            ba_whole_view_write,
            tibs_whole_view_write,
            same_bits,
        ),
        ComparisonCase("all immediate exit", ba_all_immediate, tibs_all_immediate),
        ComparisonCase("any immediate exit", ba_any_immediate, tibs_any_immediate),
        ComparisonCase("all full scan", ba_all_full_scan, tibs_all_full_scan),
        ComparisonCase("any full scan", ba_any_full_scan, tibs_any_full_scan),
        ComparisonCase("count ones", ba_count, tibs_count),
        ComparisonCase(
            "join small pieces", ba_join_small_pieces, tibs_join_small_pieces, same_bits
        ),
        ComparisonCase(
            "extend bool list", ba_extend_bool_list, tibs_extend_bool_list, same_bits
        ),
        ComparisonCase("append bits", ba_append_bits, tibs_append_bits, same_bits),
        ComparisonCase("bulk index set", ba_bulk_index_set, tibs_bulk_index_set, same_bits),
        ComparisonCase(
            "bool construction", ba_bool_construction, tibs_bool_construction, same_bits
        ),
        ComparisonCase("from bytes", ba_from_bytes, tibs_from_bytes, same_bits),
        ComparisonCase("to bytes", ba_to_bytes, tibs_to_bytes),
        ComparisonCase("unaligned slices x100", ba_slice, tibs_slice, same_bits),
        ComparisonCase("concatenate", ba_concat, tibs_concat, same_bits),
        ComparisonCase(
            "repeat 10-bit pattern",
            ba_repeat_pattern,
            tibs_repeat_pattern,
            same_bits,
        ),
        ComparisonCase("invert", ba_invert, tibs_invert, same_bits),
        ComparisonCase("reverse in place", ba_reverse, tibs_reverse, same_bits),
        ComparisonCase("shift left", ba_shift_left, tibs_shift_left, same_bits),
        ComparisonCase("random index reads", ba_index_reads, tibs_index_reads),
        ComparisonCase("gather + count", ba_gather_count, tibs_gather_count),
        ComparisonCase("to bool list", ba_to_bool_list, tibs_to_bool_list),
        ComparisonCase(
            "copy + slice set", ba_copy_slice_set, tibs_copy_slice_set, same_bits
        ),
        ComparisonCase(
            "copy + slice delete", ba_copy_slice_delete, tibs_copy_slice_delete, same_bits
        ),
        ComparisonCase("slice count", ba_slice_count, tibs_slice_count),
        ComparisonCase("pack u16", ba_pack_u16, tibs_pack_u16, same_bits),
        ComparisonCase("unpack u16", ba_unpack_u16, tibs_unpack_u16),
        ComparisonCase("prime sieve", ba_primes, tibs_primes),
        ComparisonCase("random generation", ba_random, tibs_random),
        ComparisonCase("pop all bits", ba_pop, tibs_pop),
        ComparisonCase("chunks_iter", ba_chunks, tibs_chunks),
        ComparisonCase("repeated buffer view", ba_buffer_view, tibs_buffer_view),
    ]


def build_stdlib_cases(byte_count, value_count):
    """Cases where the standard library can do the same job as tibs.

    Each case is modelled on an example from the Python documentation for the
    module concerned, so that the standard library side is doing the job the way
    its own docs present it. struct and array stop at byte boundaries and ints
    have no length, so the overlap with tibs is only partial.
    """
    search_bytes = deterministic_bytes(byte_count, "comparison-search")
    other_bytes = deterministic_bytes(byte_count, "comparison-other")
    search_tibs = Tibs.from_bytes(search_bytes)
    other_tibs = Tibs.from_bytes(other_bytes)
    search_int = int.from_bytes(search_bytes, "big")
    other_int = int.from_bytes(other_bytes, "big")
    bit_count = byte_count * 8
    width_mask = (1 << bit_count) - 1
    unaligned_bit_count = bit_count - 3
    unaligned_search_int = search_int >> 3
    unaligned_other_int = other_int >> 3
    # This construction currently leaves the backing BitVec starting part way
    # through a byte, which reactivates several bit-at-a-time fallbacks.
    unaligned_search_tibs = Tibs.from_u(unaligned_search_int, unaligned_bit_count)
    unaligned_other_tibs = Tibs.from_u(unaligned_other_int, unaligned_bit_count)

    rng = random.Random("comparison-values")
    value_words = [rng.randrange(1 << 16) for _ in range(value_count)]
    value_bytes = Tibs.from_values("u16", value_words).to_bytes()
    value_floats = [rng.random() for _ in range(value_count)]

    # struct's documented first example is pack('hhl', 1, 2, 3), a mixed-width
    # record. Byte order is pinned so the sizes are standard rather than native.
    rng = random.Random("comparison-records")
    record_count = max(1, value_count // 4)
    records = [
        (
            rng.randrange(-(1 << 15), 1 << 15),
            rng.randrange(-(1 << 15), 1 << 15),
            rng.randrange(-(1 << 31), 1 << 31),
        )
        for _ in range(record_count)
    ]
    # The docs recommend a compiled Struct when a format is reused, so the tibs
    # side hoists a DtypeTuple to match rather than rebuilding dtypes per field.
    # DtypeTuple("(i16, i16, i32)") is the mixed-width record dtype added in
    # the Dtype expansion, so a whole record round-trips in one pack/unpack
    # call rather than one call per field.
    record_struct = struct.Struct(">hhl")
    record_bytes = b"".join(record_struct.pack(*record) for record in records)
    record_dtype = Dtype("(i16, i16, i32)")
    scalar_record = records[0]
    scalar_record_bytes = record_struct.pack(*scalar_record)
    scalar_record_tibs = Tibs.from_bytes(scalar_record_bytes)

    # bytes.join and Tibs.from_joined both accept prepared pieces. Five bytes is
    # small enough for per-piece overhead to matter but is still a plausible
    # fixed-width field or short protocol fragment.
    byte_pieces = [
        search_bytes[index: index + 5] for index in range(0, byte_count, 5)
    ]
    tibs_byte_pieces = [Tibs.from_bytes(piece) for piece in byte_pieces]
    spaced_hex_pieces = " ".join(piece.hex() for piece in byte_pieces)
    tibs_hex_pieces = ",".join(f"0x{piece.hex()}" for piece in byte_pieces)

    u64_value = 0x123456789ABCDEF0
    u64_other_value = 0xFEDCBA9876543210
    u64_bytes = u64_value.to_bytes(8, "big")
    small_and_widths = (1, 7, 13, 31, 47, 63, 64)
    small_and_values = [
        (
            u64_value & ((1 << width) - 1),
            u64_other_value & ((1 << width) - 1),
        )
        for width in small_and_widths
    ]
    small_and_tibs = [
        (Tibs.from_u(left, width), Tibs.from_u(right, width))
        for width, (left, right) in zip(
            small_and_widths, small_and_values, strict=True
        )
    ]

    replace_old = search_bytes[byte_count // 2: byte_count // 2 + 1]
    replace_new = bytes([replace_old[0] ^ 0xFF])
    replace_old_tibs = Tibs.from_bytes(replace_old)
    replace_new_tibs = Tibs.from_bytes(replace_new)

    hex_string = search_bytes.hex()
    prefixed_hex = "0x" + hex_string

    # Taken from two thirds of the way in so both sides have to scan for it. A
    # pattern near the start would flatter whichever side exits earliest.
    aligned_pattern_start = byte_count * 2 // 3
    aligned_pattern = search_bytes[aligned_pattern_start: aligned_pattern_start + 4]
    aligned_pattern_tibs = Tibs.from_bytes(aligned_pattern)

    u16_format = f">{len(value_words)}H"
    f32_format = f">{len(value_floats)}f"

    def struct_pack_records():
        # struct docs: pack('hhl', 1, 2, 3)
        return b"".join(record_struct.pack(*record) for record in records)

    def tibs_pack_records():
        # DtypeTuple.pack_values (via Tibs.from_values) packs every field of
        # every record in one call rather than one call per field. Dtype
        # caches a flat per-field byte-offset layout (RecordLayout, in
        # dtype.rs) for a tuple/array of scalar fields that are all whole
        # bytes wide, so this writes straight into one Vec<u8> - no BV
        # allocated per field the way the pre-2.0 hand-rolled version needed.
        return Tibs.from_values(record_dtype, records).to_bytes()

    def struct_unpack_records():
        # struct docs: iter_unpack, "iteratively unpack from the buffer"
        return list(record_struct.iter_unpack(record_bytes))

    def tibs_unpack_records():
        # DtypeTuple.unpack_values (via Tibs.to_values) mirrors iter_unpack:
        # one call decodes every record instead of slicing each field by hand.
        return Tibs.from_bytes(record_bytes).to_values(record_dtype)

    def struct_pack_scalar_record():
        # The bulk row above is appropriate when all records are known at once.
        # This row covers independently arriving records, with both the Struct
        # and Dtype compiled and reused outside the timed region.
        return record_struct.pack(*scalar_record)

    def tibs_pack_scalar_record():
        return record_dtype.pack(scalar_record).to_bytes()

    def struct_unpack_scalar_record():
        return record_struct.unpack(scalar_record_bytes)

    def tibs_unpack_scalar_record():
        return record_dtype.unpack(scalar_record_tibs)

    def struct_pack_u16():
        # struct docs: a format character may be preceded by a repeat count,
        # so '4h' means the same as 'hhhh'.
        return struct.pack(u16_format, *value_words)

    def tibs_pack_u16():
        return Tibs.from_values("u16", value_words).to_bytes()

    def struct_unpack_u16():
        return list(struct.unpack(u16_format, value_bytes))

    def tibs_unpack_u16():
        return Tibs.from_bytes(value_bytes).to_values("u16")

    def struct_pack_f32():
        return struct.pack(f32_format, *value_floats)

    def tibs_pack_f32():
        return Tibs.from_values("f32", value_floats).to_bytes()

    def struct_pack_u16_le():
        return struct.pack(f"<{len(value_words)}H", *value_words)

    def tibs_pack_u16_le():
        return Dtype("u16_le").pack_values(value_words).to_bytes()

    def array_to_bytes():
        # array is native-endian, so a byteswap is needed to match the
        # big-endian convention used throughout this script.
        out = array.array("H", value_words)
        out.byteswap()
        return out.tobytes()

    def array_from_bytes():
        out = array.array("H")
        out.frombytes(value_bytes)
        out.byteswap()
        return out.tolist()

    def int_popcount():
        # int docs: bit_count(), "also known as the population count"
        return search_int.bit_count()

    def tibs_popcount():
        return search_tibs.count(1)

    def same_value(int_result, tibs_result):
        # Each side is left holding its own natural type, so the conversion
        # happens here in the equivalence check rather than inside the timing.
        return int_result == tibs_result.to_u()

    def int_and():
        return search_int & other_int

    def tibs_and():
        return search_tibs & other_tibs

    def int_small_ands():
        return [left & right for left, right in small_and_values]

    def tibs_small_ands():
        return [left & right for left, right in small_and_tibs]

    def same_small_values(int_results, tibs_results):
        return int_results == [result.to_u() for result in tibs_results]

    def int_unaligned_and():
        return unaligned_search_int & unaligned_other_int

    def tibs_unaligned_and():
        return unaligned_search_tibs & unaligned_other_tibs

    def int_shift_left():
        # An int has no length, so it grows on a left shift where tibs drops the
        # bits shifted off the top. Masking back to width is the work a user has
        # to do to get fixed-width behaviour from an int, so it counts.
        return (search_int << 13) & width_mask

    def tibs_shift_left():
        return search_tibs << 13

    def int_from_bytes():
        # int docs: int.from_bytes(b'\x00\x10', byteorder='big')
        return int.from_bytes(search_bytes, "big")

    def tibs_from_bytes_to_u():
        return Tibs.from_bytes(search_bytes).to_u()

    def int_unaligned_to_u():
        return int.from_bytes(search_bytes, "big") >> 3

    def tibs_unaligned_to_u():
        return search_tibs[:-3].to_u()

    def int_u64_to_bytes():
        # This is deliberately small: fixed-width integer fields in headers and
        # protocol records do not amortize conversion overhead over a large buffer.
        return u64_value.to_bytes(8, "big")

    def tibs_u64_to_bytes():
        return Tibs.from_u(u64_value, 64).to_bytes()

    def int_bytes_to_u64():
        return int.from_bytes(u64_bytes, "big")

    def tibs_bytes_to_u64():
        return Tibs.from_bytes(u64_bytes).to_u()

    def bytes_pickle_round_trip():
        return pickle.loads(pickle.dumps(search_bytes, protocol=5))

    def tibs_pickle_round_trip():
        return pickle.loads(pickle.dumps(search_tibs, protocol=5))

    def bytes_join_pieces():
        return b"".join(byte_pieces)

    def tibs_join_byte_pieces():
        return Tibs.from_joined(tibs_byte_pieces).to_bytes()

    def bytes_parse_hex_pieces():
        # bytes.fromhex accepts ASCII whitespace between byte groups.
        return bytes.fromhex(spaced_hex_pieces)

    def tibs_parse_hex_pieces():
        return Tibs.from_string(tibs_hex_pieces).to_bytes()

    def bytes_replace_aligned():
        return search_bytes.replace(replace_old, replace_new)

    def tibs_replace_aligned():
        return search_tibs.replaced(
            replace_old_tibs, replace_new_tibs, byte_aligned=True
        )

    def bytes_to_hex():
        # bytes docs: b'\xf0\xf1\xf2'.hex()
        return search_bytes.hex()

    def tibs_to_hex():
        return search_tibs.to_hex()

    def bytes_from_hex():
        # bytes docs: bytes.fromhex('2Ef0 F1f2  ')
        return bytes.fromhex(hex_string)

    def tibs_from_hex():
        return Tibs.from_string(prefixed_hex).to_bytes()

    def bytes_to_binary():
        # Converting the whole buffer to one int and formatting that is much
        # faster than joining format(byte, '08b') per byte, which spends its
        # time allocating one small string object per byte rather than on the
        # conversion. int -> str is linear for base 2 (only decimal is
        # superlinear, and only decimal is capped by int_max_str_digits). The
        # explicit width is needed because an int has no length of its own.
        return format(int.from_bytes(search_bytes, "big"), f"0{bit_count}b")

    def tibs_to_binary():
        return search_tibs.to_bin()

    def bytes_aligned_find():
        return search_bytes.find(aligned_pattern) * 8

    def tibs_aligned_find():
        return search_tibs.find(aligned_pattern_tibs, byte_aligned=True)

    def bytes_aligned_count():
        # bytes.count is non-overlapping and find_all overlaps. A 4-byte pattern
        # drawn from random data effectively never overlaps itself, so the two
        # agree here, but they would not for a self-similar pattern.
        return search_bytes.count(aligned_pattern)

    def tibs_aligned_count():
        return len(search_tibs.find_all(aligned_pattern_tibs, byte_aligned=True))

    return [
        ComparisonCase("struct: pack hhl", struct_pack_records, tibs_pack_records),
        ComparisonCase("struct: iter_unpack hhl", struct_unpack_records, tibs_unpack_records),
        ComparisonCase(
            "struct: pack one hhl", struct_pack_scalar_record, tibs_pack_scalar_record
        ),
        ComparisonCase(
            "struct: unpack one hhl",
            struct_unpack_scalar_record,
            tibs_unpack_scalar_record,
        ),
        ComparisonCase("struct: pack u16", struct_pack_u16, tibs_pack_u16),
        ComparisonCase("struct: unpack u16", struct_unpack_u16, tibs_unpack_u16),
        ComparisonCase("struct: pack f32", struct_pack_f32, tibs_pack_f32),
        ComparisonCase("struct: pack u16 le", struct_pack_u16_le, tibs_pack_u16_le),
        ComparisonCase("array: u16 to bytes", array_to_bytes, tibs_pack_u16),
        ComparisonCase("array: bytes to u16", array_from_bytes, tibs_unpack_u16),
        ComparisonCase("int: popcount", int_popcount, tibs_popcount),
        ComparisonCase("int: bitwise and", int_and, tibs_and, same_value),
        ComparisonCase(
            "int: 1-64-bit and",
            int_small_ands,
            tibs_small_ands,
            same_small_values,
        ),
        ComparisonCase(
            "int: unaligned and", int_unaligned_and, tibs_unaligned_and, same_value
        ),
        ComparisonCase("int: shift left", int_shift_left, tibs_shift_left, same_value),
        ComparisonCase("int: from bytes", int_from_bytes, tibs_from_bytes_to_u),
        ComparisonCase(
            "int: unaligned to int", int_unaligned_to_u, tibs_unaligned_to_u
        ),
        ComparisonCase("int: u64 to bytes", int_u64_to_bytes, tibs_u64_to_bytes),
        ComparisonCase("int: 8 bytes to u64", int_bytes_to_u64, tibs_bytes_to_u64),
        ComparisonCase(
            "pickle: round trip",
            bytes_pickle_round_trip,
            tibs_pickle_round_trip,
            same_bytes,
        ),
        ComparisonCase(
            "bytes: join 5-byte pieces", bytes_join_pieces, tibs_join_byte_pieces
        ),
        ComparisonCase(
            "bytes: parse hex pieces", bytes_parse_hex_pieces, tibs_parse_hex_pieces
        ),
        ComparisonCase(
            "bytes: aligned replace",
            bytes_replace_aligned,
            tibs_replace_aligned,
            same_bytes,
        ),
        ComparisonCase("bytes: to hex", bytes_to_hex, tibs_to_hex),
        ComparisonCase("bytes: from hex", bytes_from_hex, tibs_from_hex),
        ComparisonCase("bytes: to binary str", bytes_to_binary, tibs_to_binary),
        ComparisonCase("bytes: aligned find", bytes_aligned_find, tibs_aligned_find),
        ComparisonCase("bytes: aligned count", bytes_aligned_count, tibs_aligned_count),
    ]


NOTE = """\
Note: This is not a complete or impartial comparison of the two libraries, and it
is not meant as a competition. The cases here are the ones that have been useful
for tuning tibs, so treat the results as a regression check and a performance
goal rather than an overall score for either library. Where tibs is markedly
slower than bitarray on a case, that points at something worth optimizing.

I have tried to be fair and idiomatic in using bitarray - any inefficiencies
are my fault and I'd be happy to correct them.

The standard library table asks a different question: for a job that doesn't
need bit-level addressing, is a dedicated library worth it at all? Each case is
modelled on an example from the Python documentation for the module concerned,
so the standard library is doing the job the way its own docs present it. Only
the byte-aligned part of tibs overlaps here - struct and array stop at byte
boundaries, and ints have no length.
"""


def time_cases(cases, baseline_label, repeats):
    """Time every case, checking equivalence first, and return one row each.

    The whole table has to be timed before any of it can be printed, since the
    rows are ordered by the result. Progress goes to stderr so that it stays out
    of the way when stdout is redirected.
    """
    progress = sys.stderr.isatty()
    # Padded so that a short case name does not leave part of a longer one behind.
    width = max(len(case.name) for case in cases) + 8
    rows = []
    for case in cases:
        if progress:
            message = f"timing {case.name}".ljust(width)
            print(f"\r{message}", end="", file=sys.stderr, flush=True)
        baseline_result = case.baseline_fn()
        tibs_result = case.tibs_fn()
        if not case.equivalent(baseline_result, tibs_result):
            raise AssertionError(
                f"{case.name} returned different results: "
                f"{baseline_label}={result_summary(baseline_result)}, "
                f"tibs={result_summary(tibs_result)}"
            )
        del baseline_result, tibs_result

        baseline_time, tibs_time = median_times(
            case.baseline_fn, case.tibs_fn, repeats
        )
        speedup = baseline_time / tibs_time if tibs_time else float("inf")
        rows.append((speedup, case.name, baseline_time, tibs_time))
    if progress:
        print("\r" + " " * width + "\r", end="", file=sys.stderr, flush=True)
    rows.sort(key=lambda row: row[0], reverse=True)
    return rows


def run_table(title, baseline_label, cases, repeats, colour):
    """Time every case, print the table, and return the speedups."""
    rows = time_cases(cases, baseline_label, repeats)

    print()
    print(colourise(title, BOLD, colour))
    print("Each case shows how many times faster the faster of the two was,")
    print("with the best tibs result first.")
    print()

    header = (
        f"{'case':<{NAME_WIDTH}}{baseline_label:^{PAIR_WIDTH}}{'tibs':^{PAIR_WIDTH}}"
    )
    rule = "-" * len(header)
    print(colourise(header.rstrip(), BOLD, colour))
    print(rule)

    speedups = []
    for speedup, name, baseline_time, tibs_time in rows:
        if baseline_time > 0 and tibs_time > 0:
            speedups.append(speedup)
        # The winner is green whichever library it is: the column it lands in
        # already says who won, so a second colour would only add noise.
        baseline_relative, tibs_relative = relative_cells(speedup)
        cell_colour = GREEN if relative_factor(speedup) > FAST_THRESHOLD else ""
        print(
            (
                f"{name:<{NAME_WIDTH}}"
                f"{format_time(baseline_time):>{TIME_WIDTH}}"
                f"{relative_cell(baseline_relative, cell_colour, colour)}"
                f"{format_time(tibs_time):>{TIME_WIDTH}}"
                f"{relative_cell(tibs_relative, cell_colour, colour)}"
            ).rstrip()
        )

    if speedups:
        print(rule)
        for label, value in (
                ("Geometric mean", math.prod(speedups) ** (1 / len(speedups))),
                ("Median", statistics.median(speedups)),
        ):
            comparison = (
                f"{value:.2f}x faster" if value >= 1 else f"{1 / value:.2f}x slower"
            )
            summary = f"Tibs is {comparison}".rjust(2 * PAIR_WIDTH)
            print(
                f"{label:<{NAME_WIDTH}}"
                f"{colourise(summary, speedup_colour(value), colour)}"
            )
    return speedups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=250_000, help="random data bytes")
    parser.add_argument("--values", type=int, default=20_000, help="u16 value count")
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats per case")
    parser.add_argument("--no-color", action="store_true", help="disable coloured output")
    parser.add_argument(
        "--table",
        choices=("bitarray", "stdlib", "both"),
        default="both",
        help="which comparison table to run",
    )
    args = parser.parse_args()

    colour = not args.no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    if args.bytes < 2:
        parser.error("--bytes must be at least 2")
    if args.values < 0:
        parser.error("--values must not be negative")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    wants_bitarray = args.table in ("bitarray", "both")
    if wants_bitarray and bitarray is None:
        if args.table == "bitarray":
            raise SystemExit(
                "bitarray is not installed; install it to run this local comparison."
            )
        wants_bitarray = False

    print(NOTE)
    print(
        f"Running local comparison with {args.bytes:,} bytes, "
        f"{args.values:,} u16 values, {args.repeats} repeats."
    )
    print("Lower times are better.")
    if args.table != "stdlib" and not wants_bitarray:
        print("Skipping the bitarray table: bitarray is not installed.")

    if wants_bitarray:
        run_table(
            "tibs vs bitarray",
            "bitarray",
            build_bitarray_cases(args.bytes, args.values),
            args.repeats,
            colour,
        )
    if args.table in ("stdlib", "both"):
        run_table(
            "tibs vs the standard library",
            "stdlib",
            build_stdlib_cases(args.bytes, args.values),
            args.repeats,
            colour,
        )


if __name__ == "__main__":
    main()
