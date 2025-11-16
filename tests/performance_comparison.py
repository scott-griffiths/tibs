# Comparison of performance between bitarray and tibs.
# This isn't meant as a competition, but more of a sanity check.
# If the tibs speed for a task is significantly less than the speed that bitarray can
# do the same task, then that points to an area that needs to be optimized.


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import timeit
import random
import math
from math import isqrt
from random import randrange
from tibs import Tibs, Mutibs
from bitarray.util import random_p, ones
from bitarray import bitarray
from bitarray.util import int2ba, ba2int, pprint

ba_rand = random_p(1_000_000_000)
tibs_rand = Tibs.from_random(1_000_000_000)

some_bytes = Tibs.from_random(10_000_000, seed=b'a').to_bytes()
other_bytes = Tibs.from_random(10_000_000, seed=b'b').to_bytes()


def test_findall_tibs():
    t = Tibs.from_bytes(some_bytes)
    x = list(t.find_all('0xabc'))


def test_findall_bitarray():
    b = bitarray()
    b.frombytes(some_bytes)
    pattern = bitarray('101010111100')
    x = list(b.search(pattern))


def test_bitops_tibs():
    t1 = Tibs.from_bytes(some_bytes)
    t2 = Tibs.from_bytes(other_bytes)
    for _ in range(100):
        t3 = t1 | t2
        t4 = t3[10:1_000_000] & t2[9:999_999]
    print(t4.count(1))


def test_bitops_bitarray():
    b1 = bitarray()
    b1.frombytes(some_bytes)
    b2 = bitarray()
    b2.frombytes(other_bytes)
    for _ in range(100):
        b3 = b1 | b2
        b4 = b3[10:1_000_000] & b2[9:999_999]
    print(b4.count())


def test_construction_bitarray():
    b = bitarray()
    x = bitarray('10101')
    for _ in range(1000000):
        b.extend(x)
    assert len(b) == 5 * 1000000


def test_construction_tibs():
    t = Mutibs()
    x = Tibs('0b10101')
    for _ in range(1000000):
        t += x
    assert len(t) == 5 * 1000000


def test_counting_bitarray():
    for _ in range(100):
        _ = ba_rand.count(1)


def test_counting_tibs():
    for _ in range(100):
        _ = tibs_rand.count(1)


def test_rand_bitarray():
    s = random_p(1_000_000_000)


def test_rand_tibs():
    s = Mutibs.from_random(1_000_000_000)


def test_primes_bitarray():
    limit = 50_000_000
    is_prime = ones(limit)
    is_prime[:2] = False

    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i * i:: i] = False
    x = is_prime.count(bitarray("101")) + 1
    assert x == 239101


def test_primes_tibs():
    limit = 50_000_000
    is_prime = Mutibs.from_ones(limit)
    is_prime.set(False, [0, 1])
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime.set(False, range(i * i, limit, i))
    twin_primes = len(list(is_prime.as_tibs().find_all('0b101')))
    assert twin_primes == 239101


class FunctionPairs:
    def __init__(self, name, bitarray_func, tibs_func):
        self.name = name
        self.bitarray_func = bitarray_func
        self.tibs_func = tibs_func
        self.bf_time = None
        self.bs_time = None
        self.ratio = 1.0

    def run(self):
        self.ba_time = timeit.timeit(self.bitarray_func, number=5)
        self.t_time = timeit.timeit(self.tibs_func, number=5)
        self.ratio = self.ba_time / self.t_time


class TestSuite:
    def __init__(self, pairs):
        self.pairs = pairs

    def run(self):
        for pair in self.pairs:
            pair.run()

    def print_results(self):
        for pair in self.pairs:
            if pair.ratio > 1.0:
                extra = ""
            else:
                extra = f"({1 / pair.ratio:.2f}⨉ slower)"
            print(
                f'{pair.name}: {pair.ratio:.2f}⨉ faster {extra} bitarray: {pair.ba_time:.2f}s vs tibs: {pair.t_time:.2f}s')
        # For ratios we use a geometric mean
        average = math.prod(r.ratio for r in self.pairs) ** (1 / len(self.pairs))
        print(f"AVERAGE: {average:.2f}⨉ faster")


def main():
    fn_pairs = [

        FunctionPairs("Primes", test_primes_bitarray, test_primes_tibs),
        FunctionPairs("Counting", test_counting_bitarray, test_counting_tibs),
        FunctionPairs("Random Generation", test_rand_bitarray, test_rand_tibs),
        FunctionPairs("Construction", test_construction_bitarray, test_construction_tibs),
        FunctionPairs("Find all", test_findall_bitarray, test_findall_tibs),
        FunctionPairs("Bit ops", test_bitops_bitarray, test_bitops_tibs),
    ]
    ts = TestSuite(fn_pairs)
    ts.run()
    ts.print_results()


if __name__ == "__main__":
    main()
