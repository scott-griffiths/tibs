import sys
import pytest

sys.path.insert(0, "..")
from tibs import Reader, Tibs, Mutibs
import random
import math
import itertools


def test_chunking(benchmark):
    def chunks():
        s = Tibs.from_string("0xef1356a6200b3, 0b0")
        s = Tibs.from_joined(itertools.repeat(s, 6000))
        c = 0
        v = Tibs('0b001')
        for triplet in s.chunks(3):
            if triplet == v:
                c += 1
        return c

    c = benchmark(chunks)
    assert c == 12000, c


def test_count(benchmark):
    def count():
        s = Mutibs.from_zeros(100000000)
        s.set([10, 100, 1000, 10000000])
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
        s = Tibs.from_random(20000000, seed=b"99")
        for ss in [
            "0b11010010101",
            "0xabcdef1234, 0b000101111010101010011010100100101010101",
            "0x4321"
        ]:
            x = len(s.find_all(ss))
        return x

    c = benchmark(finding)
    assert c == 305


def test_pairwise_counts(benchmark):
    def counting():
        a = Tibs.from_random(2000000, seed=b"99")
        b = Tibs.from_random(2000000, seed=b"98")
        # Each of these would otherwise build a 2 million bit intermediate.
        return (a.count_and(b), a.count_or(b), a.count_xor(b), a.count_andnot(b))

    c = benchmark(counting)
    assert c == (500480, 1500548, 1000068, 500584)


def test_find_all_masked(benchmark):
    def finding():
        s = Tibs.from_random(2000000, seed=b"99")
        # Every byte whose low nibble is 1111. Masked searches can't use the
        # byte-oriented fast path, so this is a bit-by-bit scan.
        return len(s.find_all("0x0f", mask="0x0f", byte_aligned=True))

    c = benchmark(finding)
    assert c == 15674


def test_primes(benchmark):
    def primes():
        limit = 1000000
        is_prime = Mutibs.from_ones(limit)
        # Manually set 0 and 1 to be not prime.
        is_prime.unset([0, 1])
        # For every other integer, if it's set as prime then unset all of its multiples
        for i in range(2, math.ceil(math.sqrt(limit))):
            if is_prime[i]:
                is_prime.unset(range(i * i, limit, i))
        twin_primes = len(is_prime.find_all("0b101"))
        return twin_primes

    c = benchmark(primes)
    assert c == 8169


def test_reader_scan(benchmark):
    def scan():
        # A stream of length-prefixed records read with a cursor: the shape a
        # Reader exists for, where the alternative is threading a bit position
        # through the loop by hand.
        s = Mutibs()
        for i in range(20000):
            payload = i % 60 + 4
            s += Tibs.from_u(payload, length=8)
            s += Tibs.from_zeros(payload)
        r = Reader(s.to_tibs())
        total = 0
        while not r.at_end:
            payload = r.read_value("u8")
            r.read_bits(payload)
            total += payload
        return total

    c = benchmark(scan)
    assert c == 669600


def test_reader_bookmark(benchmark):
    def bookmarking():
        r = Reader(Tibs.from_random(80000, seed=b"99"))
        total = 0
        while not r.at_end:
            # Look ahead further than a single value, then read for real.
            with r.bookmark():
                total += r.read_value("u8") + r.read_value("u8")
            r.read_value("u16")
        return total

    c = benchmark(bookmarking)
    assert c == 1282283
