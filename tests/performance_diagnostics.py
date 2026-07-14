"""Diagnostic performance benchmarks for local optimization work.

This module is intentionally not named ``test_*.py`` so the normal test suite
doesn't collect it. Run it explicitly with pytest-benchmark when investigating
performance opportunities:

    .venv/bin/python -m pytest tests/performance_diagnostics.py --benchmark-only
"""

import itertools
import random

from tibs import Mutibs, Tibs


def _deterministic_bytes(size, seed):
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(size))


SEARCH_BYTES = _deterministic_bytes(128_000, "diagnostic-search")
OTHER_BYTES = _deterministic_bytes(128_000, "diagnostic-other")

SEARCH_TIBS = Tibs.from_bytes(SEARCH_BYTES)
OTHER_TIBS = Tibs.from_bytes(OTHER_BYTES)
OFFSET_SEARCH_TIBS = SEARCH_TIBS[3:-5]

DENSE_SINGLE_BIT = Tibs.from_ones(200_000)
SPARSE_SINGLE_BIT = Mutibs.from_zeros(200_000)
SPARSE_SINGLE_BIT.set(range(0, 200_000, 997))
SPARSE_SINGLE_BIT = SPARSE_SINGLE_BIT.to_tibs()

CHUNK_SOURCE = Tibs.from_joined(
    itertools.repeat(Tibs("0xef1356a6200b3, 0b0"), 2_000)
)
BOOL_LIST = list(SEARCH_TIBS[:40_000])

VALUE_RNG = random.Random("diagnostic-values")
U8_VALUES = [VALUE_RNG.randrange(1 << 8) for _ in range(30_000)]
U16_VALUES = [VALUE_RNG.randrange(1 << 16) for _ in range(20_000)]
U32_VALUES = [VALUE_RNG.randrange(1 << 32) for _ in range(10_000)]
U8_TIBS = Tibs.from_values("u8", U8_VALUES)
U16_TIBS = Tibs.from_values("u16", U16_VALUES)
U32_TIBS = Tibs.from_values("u32", U32_VALUES)


def test_search_unaligned_sparse_pattern(benchmark):
    def run():
        return len(SEARCH_TIBS.find_all("0xabc"))

    assert benchmark(run) >= 0


def test_find_unaligned_sparse_pattern(benchmark):
    def run():
        return SEARCH_TIBS.find("0xabc")

    assert benchmark(run) is not None


def test_find_all_iter_unaligned_sparse_pattern(benchmark):
    def run():
        return len(list(SEARCH_TIBS.find_all_iter("0xabc")))

    assert benchmark(run) >= 0


def test_count_unaligned_sparse_pattern(benchmark):
    def run():
        return SEARCH_TIBS.count("0xabc")

    assert benchmark(run) >= 0


def test_search_byte_aligned_pattern(benchmark):
    def run():
        return len(SEARCH_TIBS.find_all("0xabcd", byte_aligned=True))

    assert benchmark(run) >= 0


def test_search_unaligned_offset_slice(benchmark):
    def run():
        return len(OFFSET_SEARCH_TIBS.find_all("0xabc"))

    assert benchmark(run) >= 0


def test_search_dense_overlapping_pattern(benchmark):
    dense = Tibs.from_ones(60_000)

    def run():
        return len(dense.find_all("0b11111"))

    assert benchmark(run) == 59_996


def test_search_single_bit_dense(benchmark):
    def run():
        return len(DENSE_SINGLE_BIT.find_all("0b1"))

    assert benchmark(run) == len(DENSE_SINGLE_BIT)


def test_search_single_bit_sparse(benchmark):
    def run():
        return len(SPARSE_SINGLE_BIT.find_all("0b1"))

    assert benchmark(run) == 201


def test_reverse_search_iterator(benchmark):
    def run():
        return len(list(SEARCH_TIBS.rfind_all_iter("0xdeade")))

    assert benchmark(run) >= 0


def test_logical_ops_aligned(benchmark):
    def run():
        result = SEARCH_TIBS & OTHER_TIBS
        result = result | SEARCH_TIBS
        result = result ^ OTHER_TIBS
        return len(result)

    assert benchmark(run) == len(SEARCH_TIBS)


def test_logical_ops_unaligned_slices(benchmark):
    left = SEARCH_TIBS[3:400_003]
    right = OTHER_TIBS[5:400_005]

    def run():
        result = left & right
        result = result | left
        result = result ^ right
        return len(result)

    assert benchmark(run) == len(left)


def test_logical_ops_mutibs_inplace(benchmark):
    other = OTHER_TIBS[:400_000]

    def run():
        result = Mutibs(SEARCH_TIBS[:400_000])
        result &= other
        result |= other
        result ^= other
        return len(result)

    assert benchmark(run) == len(other)


def test_from_values_u8(benchmark):
    result = benchmark(lambda: Tibs.from_values("u8", U8_VALUES))
    assert len(result) == 8 * len(U8_VALUES)


def test_from_values_u16(benchmark):
    result = benchmark(lambda: Tibs.from_values("u16", U16_VALUES))
    assert len(result) == 16 * len(U16_VALUES)


def test_from_values_u32(benchmark):
    result = benchmark(lambda: Tibs.from_values("u32", U32_VALUES))
    assert len(result) == 32 * len(U32_VALUES)


def test_to_values_u8(benchmark):
    assert benchmark(lambda: U8_TIBS.to_values("u8")) == U8_VALUES


def test_to_values_u16(benchmark):
    assert benchmark(lambda: U16_TIBS.to_values("u16")) == U16_VALUES


def test_to_values_u32(benchmark):
    assert benchmark(lambda: U32_TIBS.to_values("u32")) == U32_VALUES


def test_chunks_iter_small_chunks(benchmark):
    target = Tibs("0b11111")

    def run():
        return sum(1 for chunk in CHUNK_SOURCE.chunks_iter(5) if chunk == target)

    assert benchmark(run) >= 0


def test_to_values_iter_u16(benchmark):
    def run():
        return sum(U16_TIBS.to_values_iter("u16"))

    assert benchmark(run) == sum(U16_VALUES)


def test_bool_iteration(benchmark):
    def run():
        return sum(SEARCH_TIBS[:40_000])

    assert benchmark(run) >= 0


def test_mutation_contiguous_set_unset(benchmark):
    def run():
        result = Mutibs.from_zeros(200_000)
        result.set(range(10_000, 150_000))
        result.unset(range(60_000, 90_000))
        return result.count(1)

    assert benchmark(run) == 110_000


def test_mutation_strided_set_unset(benchmark):
    def run():
        result = Mutibs.from_zeros(200_000)
        result.set(range(0, 200_000, 3))
        result.unset(range(0, 200_000, 9))
        return result.count(1)

    assert benchmark(run) > 0


def test_mutation_insert_delete_positions(benchmark):
    payload = Tibs.from_random(512, seed=b"diagnostic-insert")

    def run():
        result = Mutibs.from_random(100_000, seed=b"diagnostic-base")
        result.insert(0, payload)
        result.insert(len(result) // 2, payload)
        del result[:512]
        del result[-512:]
        return len(result)

    assert benchmark(run) == 100_000
