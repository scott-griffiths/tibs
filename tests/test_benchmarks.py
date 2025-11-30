import sys
import pytest

sys.path.insert(0, "..")
from tibs import Tibs, Mutibs
import random
import math
import itertools


def test_chunking(benchmark):
    def chunks():
        s = Tibs.from_string("0xef1356a6200b3, 0b0")
        s = Tibs.from_joined(itertools.repeat(s, 6000))
        c = 0
        for triplet in s.chunks(3):
            if triplet == "0b001":
                c += 1
        return c

    c = benchmark(chunks)
    assert c == 12000, c


def test_count(benchmark):
    def count():
        s = Mutibs.from_zeros(100000000)
        s = s.set(1, [10, 100, 1000, 10000000])
        return s.count(1)

    c = benchmark(count)
    assert c == 4


def test_token_parsing(benchmark):
    def token_parsing():
        s = Mutibs()
        for i in range(10000):
            s += "0x3e44f, 0b11011, 0o75523"
            s += Tibs.from_bools([0, 1, 2, 0, 0, 1, 2, 0, -1, 0, "hello"])
            s += Tibs.from_zeros(104)

    benchmark(token_parsing)


def test_find_all(benchmark):
    def finding():
        s = Tibs.from_random(20000000, b"99")
        for ss in [
            "0b11010010101",
            "0xabcdef1234, 0b000101111010101010011010100100101010101",
            "0x4321"
        ]:
            x = len(list(s.find_all(ss)))
        return x

    c = benchmark(finding)
    assert c == 305


def test_primes(benchmark):
    def primes():
        limit = 1000000
        is_prime = Mutibs.from_ones(limit)
        # Manually set 0 and 1 to be not prime.
        is_prime.set(False, [0, 1])
        # For every other integer, if it's set as prime then unset all of its multiples
        for i in range(2, math.ceil(math.sqrt(limit))):
            if is_prime[i]:
                is_prime.set(False, range(i * i, limit, i))
        twin_primes = len(list(is_prime.to_tibs().find_all("0b101")))
        return twin_primes

    c = benchmark(primes)
    assert c == 8169
