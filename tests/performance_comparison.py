# Comparison of performace between bitformat and bitstring.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import bitstring
import bitformat
import timeit
import random
import math
from math import isqrt
from random import randrange


from bitarray.util import ones
from bitarray import bitarray
from bitarray.util import int2ba, ba2int, pprint

# This is copied from bitarray
class SmallIntArray:
    """
    Class which allows efficiently storing an array of integers
    represented by a specified number of bits.
    For example, an array with 1000 5 bit integers can be created,
    allowing each element in the array to take values form 0 to 31,
    while the size of the object is 625 (5000/8) bytes.
    """
    def __init__(self, N, k):
        self.N = N  # number of integers
        self.k = k  # bits for each integer
        self.array = bitarray(N * k)

    def slice_i(self, i):
        assert 0 <= i < self.N
        return slice(self.k * i, self.k * (i + 1))

    def __getitem__(self, i):
        return ba2int(self.array[self.slice_i(i)])

    def __setitem__(self, i, v):
        self.array[self.slice_i(i)] = int2ba(v, self.k)

def test_small_ints_bitarray():
    # define array of integers, each represented by 5 bits
    a = SmallIntArray(100000, 5)

    for i in range(100000):
        v = randrange(32)
        a[i] = v

def test_small_ints_bitformat():
    a = bitformat.Array.from_zeros('u5', 100000)
    for i in range(100000):
        v = randrange(32)
        a[i] = v


def test_cutting_bitstring():
    s = bitstring.Bits('0xef1356a6200b3, 0b0')
    s *= 6000
    c = 0
    for triplet in s.cut(3):
        if triplet == '0b001':
            c += 1
    return c

def test_cutting_bitformat():
    s = bitformat.Bits('0xef1356a6200b3, 0b0')
    s *= 6000
    c = 0
    for triplet in s.chunks(3):
        if triplet == '0b001':
            c += 1
    return c

def test_primes_bitstring():
    limit = 50_000_000
    is_prime = bitstring.BitArray(limit)
    is_prime.set(True)
    # Manually set 0 and 1 to be not prime.
    is_prime.set(False, [0, 1])
    # For every other integer, if it's set as prime then unset all of its multiples
    for i in range(2, math.ceil(math.sqrt(limit))):
        if is_prime[i]:
            is_prime.set(False, range(i * i, limit, i))
    twin_primes = len(list(is_prime.findall('0b101')))
    return twin_primes

def test_primes_bitarray():
    limit = 50_000_000
    is_prime = ones(limit)
    is_prime[:2] = False

    # Perform sieve
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:  # i is prime, so all multiples are not
            is_prime[i * i :: i] = False
    x = is_prime.count(bitarray("101")) + 1
    assert x == 239101

def test_primes_bitformat():
    limit = 50_000_000
    is_prime = bitformat.MutableBits.from_ones(limit)
    # Manually set 0 and 1 to be not prime.
    is_prime.set(False, [0, 1])
    # For every other integer, if it's set as prime then unset all of its multiples
    for i in range(2, math.ceil(math.sqrt(limit))):
        if is_prime[i]:
            is_prime.set(False, range(i * i, limit, i))
    twin_primes = len(list(is_prime.as_bits().find_all('0b101')))
    assert twin_primes == 239101
    return twin_primes

def test_token_parsing_mutating_bitformat():
    b = bitformat.MutableBits()
    b.reserve(10000*200)
    for i in range(10000):
        b += "u12=244, f32=0.4"
        b += "0x3e44f, 0b11011, 0o75523"
        b += [0, 1, 2, 0, 0, 1, 2, 0, -1, 0, "hello"]
        b += bitformat.Bits.from_zeros(104)
    return b

def test_token_parsing_mutating_bitstring():
    s = bitstring.BitArray()
    for i in range(10000):
        s += 'uint:12=244, float:32=0.4'
        s += '0x3e44f, 0b11011, 0o75523'
        s += [0, 1, 2, 0, 0, 1, 2, 0, -1, 0, 'hello']
        s += bitstring.BitArray(104)
    return s

def test_token_parsing_joining_bitformat():
    s = []
    for i in range(10000):
        s.append(bitformat.Bits.from_string('u12=244, f32=0.4'))
        s.append(bitformat.Bits.from_string('0x3e44f, 0b11011, 0o75523'))
        s.append(bitformat.Bits.from_bools([0, 1, 2, 0, 0, 1, 2, 0, -1, 0, 'hello']))
        s.append(bitformat.Bits.from_zeros(104))
    return bitformat.Bits.from_joined(s)

def test_token_parsing_joining_bitstring():
    s = []
    for i in range(10000):
        s.append(bitstring.Bits.fromstring('uint:12=244, float:32=0.4'))
        s.append(bitstring.Bits.fromstring('0x3e44f, 0b11011, 0o75523'))
        s.append(bitstring.Bits([0, 1, 2, 0, 0, 1, 2, 0, -1, 0, 'hello']))
        s.append(bitstring.Bits(104))
    return bitstring.Bits().join(s)


def test_count_bitstring():
    s = bitstring.BitArray(1_000_000_000)
    s.set(1, range(0, 1_000_000_000, 7))
    return s.count(1)

def test_count_bitformat():
    s = bitformat.MutableBits.from_zeros(1_000_000_000)
    s.set(1, range(0, 1_000_000_000, 7))
    return s.count(1)

def test_finding_bitstring():
    random.seed(999)
    i = random.randrange(0, 2 ** 20000000)
    s = bitstring.BitArray(uint=i, length=20000000)
    for ss in ['0b11010010101', '0xabcdef1234, 0b000101111010101010011010100100101010101', '0x4321']:
        x = len(list(s.findall(ss, count=100)))
    return x

def test_finding_bitformat():
    random.seed(999)
    i = random.randrange(0, 2 ** 20000000)
    s = bitformat.Bits.from_dtype('u20000000', i)
    for ss in ['0b11010010101', '0xabcdef1234, 0b000101111010101010011010100100101010101', '0x4321']:
        x = len(list(s.find_all(ss, count=100)))
    return x

class FunctionPairs:
    def __init__(self, name, bitstring_func, bitformat_func):
        self.name = name
        self.bitstring_func = bitstring_func
        self.bitformat_func = bitformat_func
        self.bf_time = None
        self.bs_time = None
        self.ratio = 1.0

    def run(self):
        self.bs_time = timeit.timeit(self.bitstring_func, number=5)
        self.bf_time = timeit.timeit(self.bitformat_func, number=5)
        self.ratio = self.bs_time / self.bf_time

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
                extra = f"({1/pair.ratio:.2f}⨉ slower)"
            print(f'{pair.name}: {pair.ratio:.2f}⨉ faster {extra} bs: {pair.bs_time:.2f}s vs bf: {pair.bf_time:.2f}s')
        # For ratios we use a geometric mean
        average = math.prod(r.ratio for r in self.pairs) ** (1 / len(self.pairs))
        print(f"AVERAGE: {average:.2f}⨉ faster")

def main():
    fn_pairs = [
        FunctionPairs("Cutting", test_cutting_bitstring, test_cutting_bitformat),
        FunctionPairs("Token parsing mutating", test_token_parsing_mutating_bitstring, test_token_parsing_mutating_bitformat),
        FunctionPairs("Token parsing joining", test_token_parsing_joining_bitstring, test_token_parsing_joining_bitformat),
        FunctionPairs("Count", test_count_bitstring, test_count_bitformat),
        FunctionPairs("Finding", test_finding_bitstring, test_finding_bitformat),

        # These are tested against examples provided by bitarray.
        FunctionPairs("Primes", test_primes_bitarray, test_primes_bitformat),
        FunctionPairs("Small ints", test_small_ints_bitarray, test_small_ints_bitformat)
    ]
    ts = TestSuite(fn_pairs)
    ts.run()
    ts.print_results()


if __name__ == "__main__":
    main()