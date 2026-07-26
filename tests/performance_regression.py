"""Tibs-only performance benchmarks for CI regression reporting.

This module is intentionally not named ``test_*.py`` so the normal test suite
doesn't collect it. CI invokes it explicitly with pytest-benchmark and compares
the generated JSON for the pull request against its base commit.
"""

import itertools
import math
import random

from tibs import Mutibs, Tibs


def _deterministic_bytes(size, seed):
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(size))


SEARCH_BYTES = _deterministic_bytes(250_000, "tibs-search")
OTHER_BYTES = _deterministic_bytes(250_000, "tibs-other")
COUNT_BYTES = _deterministic_bytes(1_000_000, "tibs-count")
POP_BYTES = SEARCH_BYTES[:6_250]

SEARCH_TIBS = Tibs.from_bytes(SEARCH_BYTES)
OTHER_TIBS = Tibs.from_bytes(OTHER_BYTES)
COUNT_TIBS = Tibs.from_bytes(COUNT_BYTES)
# Counting reads the storage a 64-bit word at a time, and these are the two
# shapes that stop the run being a clean array of words: a length that leaves a
# part-word at the end, and a start that is not on a word boundary. COUNT_TIBS
# itself is a whole number of words either way, so on its own it cannot tell a
# word-at-a-time scan from a byte-at-a-time one.
COUNT_TIBS_PART_WORD_TAIL = COUNT_TIBS[:-1]
COUNT_TIBS_UNALIGNED_START = COUNT_TIBS[8:]
COUNT_TIBS_MINUS_THREE = COUNT_TIBS[:-3]
COUNT_BYTES_MINUS_THREE_PADDED = COUNT_BYTES[:-1] + bytes([COUNT_BYTES[-1] & 0b1111_1000])
COUNT_BITS_MINUS_SIX = len(COUNT_BYTES) * 8 - 6
BIT_LIST = list(SEARCH_TIBS[:50_000])

CHUNK_SOURCE = Tibs.from_joined(
    itertools.repeat(Tibs("0xef1356a6200b3, 0b0"), 4_000)
)
_value_rng = random.Random(0xB17)
VALUE_WORDS = [_value_rng.randrange(1 << 16) for _ in range(20_000)]
VALUE_BYTES = Tibs.from_values("u16", VALUE_WORDS).to_bytes()
_bulk_set_rng = random.Random("tibs-bulk-set")
BULK_SET_POSITIONS = [
    [_bulk_set_rng.randrange(len(SEARCH_TIBS)) for _ in range(8)]
    for _ in range(20_000)
]


def test_find_all_bits(benchmark):
    def find_all():
        return len(Tibs.from_bytes(SEARCH_BYTES).find_all("0xabc"))

    count = benchmark(find_all)
    assert count >= 0


def test_find_all_iter_bits(benchmark):
    def find_all_iter():
        return len(list(SEARCH_TIBS.find_all_iter("0xabc")))

    count = benchmark(find_all_iter)
    assert count >= 0


def test_reverse_find_all_iter_bits(benchmark):
    def reverse_find_all_iter():
        return len(list(SEARCH_TIBS.rfind_all_iter("0xdeade")))

    count = benchmark(reverse_find_all_iter)
    assert count >= 0


def test_count_bit_pattern(benchmark):
    def count_bit_pattern():
        return SEARCH_TIBS.count("0xabc")

    count = benchmark(count_bit_pattern)
    assert count >= 0


def test_find_all_byte_aligned(benchmark):
    def find_all_byte_aligned():
        return len(Tibs.from_bytes(SEARCH_BYTES).find_all("0xabcd", byte_aligned=True))

    count = benchmark(find_all_byte_aligned)
    assert count >= 0


def test_reverse_find_pattern(benchmark):
    def reverse_find():
        return len(Tibs.from_bytes(SEARCH_BYTES).find_all("0xdeade"))

    count = benchmark(reverse_find)
    assert count >= 0


def test_bit_operations(benchmark):
    def bit_operations():
        result_length = 0
        for _ in range(50):
            combined = SEARCH_TIBS | OTHER_TIBS
            result = combined[10:500_000] & OTHER_TIBS[9:499_999]
            result_length += len(result)
        return result_length

    result_length = benchmark(bit_operations)
    assert result_length == 24_999_500


def test_concatenation(benchmark):
    result = benchmark(lambda: SEARCH_TIBS + OTHER_TIBS)
    assert len(result) == len(SEARCH_TIBS) + len(OTHER_TIBS)


def test_invert_all(benchmark):
    result = benchmark(lambda: ~SEARCH_TIBS)
    assert result.count(1) == len(SEARCH_TIBS) - SEARCH_TIBS.count(1)


def test_reverse_bits(benchmark):
    result = benchmark(SEARCH_TIBS.reversed)
    assert result[0] == SEARCH_TIBS[-1]
    assert result[-1] == SEARCH_TIBS[0]


def test_shift_left(benchmark):
    result = benchmark(lambda: SEARCH_TIBS << 13)
    assert result[:-13] == SEARCH_TIBS[13:]
    assert result[-13:] == Tibs.from_zeros(13)


def test_bool_list_conversion(benchmark):
    result = benchmark(list, SEARCH_TIBS)
    assert len(result) == len(SEARCH_TIBS)
    assert sum(result) == SEARCH_TIBS.count(1)


def test_copy_and_slice_set(benchmark):
    replacement = Tibs.from_ones(10_000)
    start = (len(SEARCH_TIBS) - len(replacement)) // 2

    def copy_and_set():
        result = SEARCH_TIBS.to_mutibs()
        result[start : start + len(replacement)] = replacement
        return result

    result = benchmark(copy_and_set)
    assert result[start : start + len(replacement)].all()


def test_copy_and_slice_delete(benchmark):
    start = (len(SEARCH_TIBS) - 10_000) // 2

    def copy_and_delete():
        result = SEARCH_TIBS.to_mutibs()
        del result[start : start + 10_000]
        return result

    result = benchmark(copy_and_delete)
    assert len(result) == len(SEARCH_TIBS) - 10_000


def test_bulk_index_set(benchmark):
    def bulk_index_set():
        result = Mutibs.from_zeros(len(SEARCH_TIBS))
        for positions in BULK_SET_POSITIONS:
            result.set(positions)
        return result

    result = benchmark(bulk_index_set)
    assert result.count(1) > 0


def test_joined_construction(benchmark):
    def joined_construction():
        piece = Tibs("0b10101")
        return Tibs.from_joined([piece] * 50_000)

    result = benchmark(joined_construction)
    assert len(result) == 250_000


def test_counting(benchmark):
    def counting():
        total = 0
        for _ in range(10):
            total += COUNT_TIBS.count(1)
        return total

    total = benchmark(counting)
    assert total > 0


def test_counting_part_word_tail(benchmark):
    def counting():
        total = 0
        for _ in range(10):
            total += COUNT_TIBS_PART_WORD_TAIL.count(1)
        return total

    total = benchmark(counting)
    assert total > 0


def test_counting_unaligned_start(benchmark):
    def counting():
        total = 0
        for _ in range(10):
            total += COUNT_TIBS_UNALIGNED_START.count(1)
        return total

    total = benchmark(counting)
    assert total > 0


def test_to_bytes_aligned(benchmark):
    result = benchmark(COUNT_TIBS.to_bytes)
    assert result == COUNT_BYTES


def test_to_padded_bytes_unaligned(benchmark):
    result = benchmark(COUNT_TIBS_MINUS_THREE.to_padded_bytes)
    assert result == COUNT_BYTES_MINUS_THREE_PADDED


def test_from_bytes_unaligned(benchmark):
    result = benchmark(Tibs.from_bytes, COUNT_BYTES, 3, COUNT_BITS_MINUS_SIX)
    assert result == COUNT_TIBS[3:-3]


def test_random_generation(benchmark):
    def random_generation():
        return Mutibs.from_random(2_000_000, seed=b"tibs-ci-benchmark")

    result = benchmark(random_generation)
    assert len(result) == 2_000_000


def test_chunks(benchmark):
    target = Tibs("0b11111")

    def chunks():
        count = 0
        for chunk in CHUNK_SOURCE.chunks_iter(5):
            if chunk == target:
                count += 1
        return count

    count = benchmark(chunks)
    assert count >= 0


def test_extending_bits(benchmark):
    def extending_bits():
        result = Mutibs()
        result.extend(BIT_LIST)
        return result

    result = benchmark(extending_bits)
    assert len(result) == len(BIT_LIST)


def test_pop(benchmark):
    def pop_bits():
        result = Mutibs.from_bytes(POP_BYTES)
        while result:
            result.pop()
        return len(result)

    length = benchmark(pop_bits)
    assert length == 0


def test_bool_construction(benchmark):
    def bool_construction():
        return Tibs.from_bools(BIT_LIST)

    result = benchmark(bool_construction)
    assert len(result) == len(BIT_LIST)


def test_slice_count(benchmark):
    def slice_count():
        total = 0
        for start in range(0, 75_000, 5):
            total += SEARCH_TIBS.count(1, start, start + 257)
        return total

    total = benchmark(slice_count)
    assert total >= 0


def test_pack_u16_values(benchmark):
    def pack_u16_values():
        return Tibs.from_values("u16", VALUE_WORDS)

    result = benchmark(pack_u16_values)
    assert len(result) == 16 * len(VALUE_WORDS)


def test_unpack_u16_values(benchmark):
    def unpack_u16_values():
        return Tibs.from_bytes(VALUE_BYTES).to_values("u16")

    values = benchmark(unpack_u16_values)
    assert values == VALUE_WORDS


def test_primes(benchmark):
    def primes():
        limit = 200_000
        is_prime = Mutibs.from_ones(limit)
        is_prime.unset([0, 1])
        for i in range(2, math.isqrt(limit) + 1):
            if is_prime[i]:
                is_prime.unset(range(i * i, limit, i))
        return is_prime.count([1, 0, 1])

    twin_primes = benchmark(primes)
    assert twin_primes > 0
