#!/usr/bin/env python3
"""Local performance comparison between tibs and bitarray.

Each case uses equivalent prepared inputs and idiomatic operations for both
libraries. Results are checked for equivalence before timing. This is useful for
finding optimization opportunities and regressions, but is not an exhaustive
overall score for either library.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import math
import operator
import os
import random
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tibs import Mutibs, Tibs

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
    bitarray_fn: Callable[[], Any]
    tibs_fn: Callable[[], Any]
    equivalent: Callable[[Any, Any], bool] = operator.eq


def median_times(bitarray_fn, tibs_fn, repeats):
    """Time both functions while alternating which library runs first."""
    times = {"bitarray": [], "tibs": []}
    functions = {"bitarray": bitarray_fn, "tibs": tibs_fn}
    for repeat in range(repeats):
        order = ("bitarray", "tibs") if repeat % 2 == 0 else ("tibs", "bitarray")
        for name in order:
            started = time.perf_counter()
            functions[name]()
            times[name].append(time.perf_counter() - started)
    return statistics.median(times["bitarray"]), statistics.median(times["tibs"])


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


RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
BOLD = "\033[1m"

SLOW_THRESHOLD = 0.8
FAST_THRESHOLD = 1.2

NAME_WIDTH = 24
TIME_WIDTH = 13
SPEEDUP_WIDTH = 10


def colourise(text, colour, enabled):
    return f"{colour}{text}{RESET}" if enabled and colour else text


def speedup_colour(speedup):
    if speedup < SLOW_THRESHOLD:
        return RED
    if speedup > FAST_THRESHOLD:
        return GREEN
    return ""


def result_summary(result):
    if isinstance(result, (bitarray, Tibs, Mutibs)):
        return f"{type(result).__name__}(len={len(result):,}, ones={result.count(1):,})"
    if isinstance(result, list) and len(result) > 10:
        return f"list(len={len(result):,}, first={result[0]!r}, last={result[-1]!r})"
    return repr(result)


def build_cases(byte_count, value_count):
    search_bytes = deterministic_bytes(byte_count, "comparison-search")
    other_bytes = deterministic_bytes(byte_count, "comparison-other")
    search_tibs = Tibs.from_bytes(search_bytes)
    other_tibs = Tibs.from_bytes(other_bytes)
    search_bits = make_bitarray(search_bytes)
    other_bits = make_bitarray(other_bytes)
    bit_count = len(search_bits)
    bit_list = list(search_tibs[: min(byte_count * 8, 80_000)])

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
        # Kept as a per-index loop to match the tibs side. bitarray can also
        # gather in bulk with `search_bits[read_positions].count(1)`, which is
        # ~2.3x faster here; tibs has no equivalent fancy-indexing read.
        return sum(search_bits[pos] for pos in read_positions)

    def tibs_index_reads():
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
        total = 0
        while out:
            total += out.pop()
        return total

    def tibs_pop():
        out = pop_tibs.to_mutibs()
        # About half the time here is in the method lookup, which isn't cached in
        # the same way as with the bitarray C extension. Hoisting it out of the
        # loop with `pop = out.pop` speeds this up a lot, but the point of the
        # case is the idiomatic version.
        total = 0
        while out:
            total += out.pop()
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
        ComparisonCase("count ones", ba_count, tibs_count),
        ComparisonCase(
            "join small pieces", ba_join_small_pieces, tibs_join_small_pieces, same_bits
        ),
        ComparisonCase(
            "extend bool list", ba_extend_bool_list, tibs_extend_bool_list, same_bits
        ),
        ComparisonCase("bulk index set", ba_bulk_index_set, tibs_bulk_index_set, same_bits),
        ComparisonCase(
            "bool construction", ba_bool_construction, tibs_bool_construction, same_bits
        ),
        ComparisonCase("from bytes", ba_from_bytes, tibs_from_bytes, same_bits),
        ComparisonCase("to bytes", ba_to_bytes, tibs_to_bytes),
        ComparisonCase("unaligned slices x100", ba_slice, tibs_slice, same_bits),
        ComparisonCase("concatenate", ba_concat, tibs_concat, same_bits),
        ComparisonCase("invert", ba_invert, tibs_invert, same_bits),
        ComparisonCase("reverse in place", ba_reverse, tibs_reverse, same_bits),
        ComparisonCase("shift left", ba_shift_left, tibs_shift_left, same_bits),
        ComparisonCase("random index reads", ba_index_reads, tibs_index_reads),
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


NOTE = """\
Note: This is not a complete or impartial comparison of the two libraries, and it
is not meant as a competition. The cases here are the ones that have been useful
for tuning tibs, so treat the results as a regression check and a performance
goal rather than an overall score for either library. Where tibs is markedly
slower than bitarray on a case, that points at something worth optimizing.

I have tried to be fair and idiomatic in using bitarray - any inefficiencies
are my fault and I'd be happy to correct them.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=250_000, help="random data bytes")
    parser.add_argument("--values", type=int, default=20_000, help="u16 value count")
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats per case")
    parser.add_argument("--no-color", action="store_true", help="disable coloured output")
    args = parser.parse_args()

    colour = not args.no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    if args.bytes < 2:
        parser.error("--bytes must be at least 2")
    if args.values < 0:
        parser.error("--values must not be negative")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    if bitarray is None:
        raise SystemExit("bitarray is not installed; install it to run this local comparison.")

    print(NOTE)
    print(
        f"Running local comparison with {args.bytes:,} bytes, "
        f"{args.values:,} u16 values, {args.repeats} repeats."
    )
    print("Lower times are better. Speedup is bitarray_time / tibs_time.")
    print()

    header = f"{'case':<{NAME_WIDTH}}{'bitarray':>{TIME_WIDTH}}{'tibs':>{TIME_WIDTH}}{'speedup':>{SPEEDUP_WIDTH}}"
    rule = "-" * len(header)
    print(colourise(header, BOLD, colour))
    print(rule)

    speedups = []
    for case in build_cases(args.bytes, args.values):
        bitarray_result = case.bitarray_fn()
        tibs_result = case.tibs_fn()
        if not case.equivalent(bitarray_result, tibs_result):
            raise AssertionError(
                f"{case.name} returned different results: "
                f"bitarray={result_summary(bitarray_result)}, "
                f"tibs={result_summary(tibs_result)}"
            )
        del bitarray_result, tibs_result

        bitarray_time, tibs_time = median_times(
            case.bitarray_fn, case.tibs_fn, args.repeats
        )
        speedup = bitarray_time / tibs_time if tibs_time else float("inf")
        if bitarray_time > 0 and tibs_time > 0:
            speedups.append(speedup)
        speedup_text = f"{speedup:.2f}x".rjust(SPEEDUP_WIDTH)
        print(
            f"{case.name:<{NAME_WIDTH}}"
            f"{f'{bitarray_time * 1e3:.3f} ms':>{TIME_WIDTH}}"
            f"{f'{tibs_time * 1e3:.3f} ms':>{TIME_WIDTH}}"
            f"{colourise(speedup_text, speedup_colour(speedup), colour)}"
        )

    if speedups:
        print(rule)
        geometric_mean = math.prod(speedups) ** (1 / len(speedups))
        median_speedup = statistics.median(speedups)
        arithmetic_mean = statistics.fmean(speedups)
        for label, value in (
                ("Arithmetic mean", arithmetic_mean),
                ("Geometric mean", geometric_mean),
                ("Median", median_speedup),
        ):
            comparison = (
                f"{value:.2f}x faster" if value >= 1 else f"{1 / value:.2f}x slower"
            )
            summary = f"Tibs is {comparison}".rjust(2 * TIME_WIDTH + SPEEDUP_WIDTH)
            print(
                f"{label:<{NAME_WIDTH}}"
                f"{colourise(summary, speedup_colour(value), colour)}"
            )


if __name__ == "__main__":
    main()
