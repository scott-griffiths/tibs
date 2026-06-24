#!/usr/bin/env python3
"""Small local performance comparison between tibs and bitarray.

This is a diagnostic tool meant to rank optimization
opportunities by showing where Tibs is materially slower than a mature bit
library on similar workloads.

This helps track regressions and improvements, but it is not intended
to be a balanced comparison between the two libraries.

"""

import argparse
import math
import os
import random
import statistics
import sys
import time


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tibs import Mutibs, Tibs

try:
    from bitarray import bitarray
    from bitarray.util import ba2int, int2ba
except ImportError:
    bitarray = None


def deterministic_bytes(size, seed):
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(size))


def median_time(fn, repeats):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def make_bitarray(data):
    bits = bitarray(endian="big")
    bits.frombytes(data)
    return bits


def build_cases(byte_count, value_count):
    search_bytes = deterministic_bytes(byte_count, "comparison-search")
    other_bytes = deterministic_bytes(byte_count, "comparison-other")
    search_tibs = Tibs.from_bytes(search_bytes)
    other_tibs = Tibs.from_bytes(other_bytes)
    search_bits = make_bitarray(search_bytes)
    other_bits = make_bitarray(other_bytes)
    bit_list = list(search_tibs[: min(byte_count * 8, 80_000)])

    rng = random.Random("comparison-values")
    value_words = [rng.randrange(1 << 16) for _ in range(value_count)]
    value_bytes = Tibs.from_values("u16", value_words).to_bytes()

    def ba_find_all():
        pattern = bitarray("101010111100", endian="big")
        return len(list(search_bits.search(pattern)))

    def tibs_find_all():
        return len(Tibs.from_bytes(search_bytes).find_all("0xabc"))

    def ba_find_all_byte_aligned():
        pattern = bitarray("1010101111001101", endian="big")
        return len([pos for pos in search_bits.search(pattern) if pos % 8 == 0])

    def tibs_find_all_byte_aligned():
        return len(Tibs.from_bytes(search_bytes).find_all("0xabcd", byte_aligned=True))

    def ba_bitops():
        total = 0
        end = min(byte_count * 8, 500_000)
        for _ in range(50):
            combined = search_bits | other_bits
            result = combined[10:end] & other_bits[9 : end - 1]
            total += len(result)
        return total

    def tibs_bitops():
        total = 0
        end = min(byte_count * 8, 500_000)
        for _ in range(50):
            combined = search_tibs | other_tibs
            result = combined[10:end] & other_tibs[9 : end - 1]
            total += len(result)
        return total

    def ba_count():
        return sum(search_bits.count(1) for _ in range(10))

    def tibs_count():
        return sum(search_tibs.count(1) for _ in range(10))

    def ba_join_small_pieces():
        out = bitarray(endian="big")
        piece = bitarray("10101", endian="big")
        for item in [piece] * 50_000:
            out.extend(item)
        return len(out)

    def tibs_join_small_pieces():
        piece = Tibs("0b10101")
        return len(Tibs.from_joined([piece] * 50_000))

    def ba_extend_bool_list():
        out = bitarray(endian="big")
        out.extend(bit_list)
        return len(out)

    def tibs_extend_bool_list():
        out = Mutibs()
        out.extend(bit_list)
        return len(out)

    def ba_bool_construction():
        return len(bitarray(bit_list, endian="big"))

    def tibs_bool_construction():
        return len(Tibs.from_bools(bit_list))

    def ba_slice_count():
        total = 0
        limit = min(byte_count * 8 - 257, 75_000)
        for start in range(0, limit, 5):
            total += search_bits[start : start + 257].count(1)
        return total

    def tibs_slice_count():
        total = 0
        limit = min(byte_count * 8 - 257, 75_000)
        for start in range(0, limit, 5):
            total += search_tibs.count(1, start, start + 257)
        return total

    def ba_pack_u16():
        out = bitarray(endian="big")
        for value in value_words:
            out.extend(int2ba(value, length=16, endian="big"))
        return len(out)

    def tibs_pack_u16():
        return len(Tibs.from_values("u16", value_words))

    def ba_unpack_u16():
        bits = make_bitarray(value_bytes)
        return [ba2int(bits[index : index + 16]) for index in range(0, len(bits), 16)]

    def tibs_unpack_u16():
        return Tibs.from_bytes(value_bytes).to_values("u16")

    def ba_chunks():
        target = bitarray("11111", endian="big")
        count = 0
        for index in range(0, len(search_bits), 5):
            if search_bits[index : index + 5] == target:
                count += 1
        return count

    def tibs_chunks():
        target = Tibs("0b11111")
        return sum(1 for chunk in search_tibs.chunks_iter(5) if chunk == target)

    return [
        ("find_all 12-bit", ba_find_all, tibs_find_all),
        ("find_all byte aligned", ba_find_all_byte_aligned, tibs_find_all_byte_aligned),
        ("bit ops sliced", ba_bitops, tibs_bitops),
        ("count ones", ba_count, tibs_count),
        ("join small pieces", ba_join_small_pieces, tibs_join_small_pieces),
        ("extend bool list", ba_extend_bool_list, tibs_extend_bool_list),
        ("bool construction", ba_bool_construction, tibs_bool_construction),
        ("slice count", ba_slice_count, tibs_slice_count),
        ("pack u16", ba_pack_u16, tibs_pack_u16),
        ("unpack u16", ba_unpack_u16, tibs_unpack_u16),
        ("chunks_iter", ba_chunks, tibs_chunks),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=250_000, help="random data bytes")
    parser.add_argument("--values", type=int, default=20_000, help="u16 value count")
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats per case")
    args = parser.parse_args()

    if bitarray is None:
        raise SystemExit("bitarray is not installed; install it to run this local comparison.")

    print(
        f"Running local comparison with {args.bytes:,} bytes, "
        f"{args.values:,} u16 values, {args.repeats} repeats."
    )
    print("Lower times are better. Speedup is bitarray_time / tibs_time.")
    print()

    speedups = []
    for name, bitarray_fn, tibs_fn in build_cases(args.bytes, args.values):
        bitarray_result = bitarray_fn()
        tibs_result = tibs_fn()
        if bitarray_result != tibs_result:
            raise AssertionError(
                f"{name} returned different results: "
                f"bitarray={bitarray_result!r}, tibs={tibs_result!r}"
            )

        bitarray_time = median_time(bitarray_fn, args.repeats)
        tibs_time = median_time(tibs_fn, args.repeats)
        speedup = bitarray_time / tibs_time if tibs_time else float("inf")
        if bitarray_time > 0 and tibs_time > 0:
            speedups.append(speedup)
        print(
            f"{name:24s} bitarray={bitarray_time * 1e3:8.3f} ms "
            f"tibs={tibs_time * 1e3:8.3f} ms speedup={speedup:6.2f}x"
        )

    if speedups:
        geometric_mean = math.prod(speedups) ** (1 / len(speedups))
        median_speedup = statistics.median(speedups)
        print()
        if geometric_mean >= 1:
            print(f"Geometric mean: Tibs is {geometric_mean:.2f}x faster.")
        else:
            print(
                "Geometric mean: "
                f"Tibs is {1 / geometric_mean:.2f}x slower."
            )
        if median_speedup >= 1:
            print(f"Median: Tibs is {median_speedup:.2f}x faster.")
        else:
            print(f"Median: Tibs is {1 / median_speedup:.2f}x slower.")


if __name__ == "__main__":
    main()
