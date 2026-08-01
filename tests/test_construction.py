#!/usr/bin/env python
import array
import io
import sys

import pytest
from hypothesis import given
import hypothesis.strategies as st
from tibs import Tibs, Mutibs
from typing import Iterable


class TestCreation:
    def test_creation_from_bytes(self):
        s = Tibs.from_bytes(b"\xa0\xff")
        assert (len(s), s.hex) == (16, "a0ff")

    @given(st.binary())
    def test_creation_from_bytes_roundtrip(self, data):
        s = Tibs.from_bytes(data)
        assert s.to_bytes() == data

    def test_creation_from_hex(self):
        s = Tibs.from_hex("0xA0ff")
        assert (len(s), s.hex) == (16, "a0ff")

    def test_creation_from_byte_aligned_hex_tokens(self):
        assert Tibs.from_string("0xab_cd,,0x,0x0123,0X45").hex == "abcd012345"

    def test_creation_from_odd_width_hex_tokens(self):
        assert Tibs.from_string("0xa,0xb,0xc,0xd").hex == "abcd"


class TestInitialisation:
    def test_empty_init(self):
        a = Tibs()
        assert a == Tibs()

    def test_find(self):
        a = Tibs.from_string("0xabcd")
        r = a.find("0xbc")
        assert r == 4
        r = a.find("0x23462346246", byte_aligned=True)
        assert r is None

    def test_rfind(self):
        a = Tibs.from_string("0b11101010010010")
        b = a.rfind("0b010")
        assert b == 11

    def test_find_all(self):
        a = Tibs("0b0010011")
        b = a.find_all('0b1')
        assert b == [2, 5, 6]
        t = Tibs("0b10")
        tp = t.find_all("0b1")
        assert tp == [0]


class TestCut:
    def test_cut(self):
        s = Tibs().from_joined(["0b000111"] * 10)
        for t in s.chunks(6):
            assert t == Tibs('0b000111')


def test_unorderable():
    a = Tibs("0b000111")
    b = Tibs("0b000111")
    with pytest.raises(TypeError):
        _ = a < b
    with pytest.raises(TypeError):
        _ = a > b
    with pytest.raises(TypeError):
        _ = a <= b
    with pytest.raises(TypeError):
        _ = a >= b


class TestPadToken:
    def test_creation(self):
        with pytest.raises(ValueError):
            _ = Tibs.from_string("pad10")
        with pytest.raises(ValueError):
            _ = Tibs.from_string("pad")


def test_adding():
    a = Tibs.from_string("0b0")
    b = Tibs.from_string("0b11")
    c = a + b
    assert c == Tibs('0b011')
    assert a == Tibs('0b0')
    assert b == Tibs('0b11')


class TestContainsBug:
    def test_contains(self):
        a = Tibs.from_string("0b1, 0x0001dead0001")
        assert "0xdead" in a
        assert "0xfeed" not in a

        assert "0b1" in Tibs.from_string("0xf")
        assert "0b0" not in Tibs.from_string("0xf")


class TestUnderscoresInLiterals:
    def test_hex_creation(self):
        a = Tibs.from_hex("ab_cd__ef")
        assert a.to_hex() == "abcdef"
        b = Tibs.from_string("0x0102_0304")
        assert b.to_hex() == "01020304"

    def test_binary_creation(self):
        a = Tibs.from_bin("0000_0001_0010")
        assert a.bin == "000000010010"
        b = Tibs.from_string("0b0011_1100_1111_0000")
        assert b.to_bin() == "0011110011110000"

    def test_octal_creation(self):
        a = Tibs.from_oct("0011_2233_4455_6677")
        assert a.oct == "0011223344556677"
        b = Tibs.from_string("0o123_321_123_321")
        assert b.to_oct() == "123321123321"


def test_from_iterable():
    with pytest.raises(TypeError):
        _ = Tibs.from_bools()
    a = Tibs.from_bools([])
    assert a == Tibs()
    a = Tibs.from_bools([1, 0, 1, 1])
    assert a == Tibs('0b1011')
    a = Tibs.from_bools((True,))
    assert a.to_bin() == "1"


def test_constructor_strict_bit_pattern_promotion():
    for cls in (Tibs, Mutibs):
        assert cls([True, False, 1, 0]) == Tibs("0b1010")
        assert cls((True, False, 1, 0)) == Tibs("0b1010")


def test_constructor_rejects_ambiguous_iterables():
    for cls in (Tibs, Mutibs):
        with pytest.raises(TypeError, match="from_values"):
            cls([1, 2, 3])

        iterator = iter([1, 0, 1])
        with pytest.raises(TypeError, match="from_bools"):
            cls(iterator)
        assert cls.from_bools(iter([1, 0, 1])) == Tibs("0b101")

        stream = io.BytesIO(b"\x01\x02")
        with pytest.raises(TypeError, match="from_bytes"):
            cls(stream)
        assert stream.tell() == 0

        byte_array = array.array("B", [1, 2, 3])
        with pytest.raises(TypeError, match="from_bytes"):
            cls(byte_array)
        assert cls.from_bytes(memoryview(byte_array)) == Tibs("0x010203")


def test_mul_by_zero():
    a = Tibs.from_string("0b1010")
    b = a * 0
    assert b == Tibs()
    b = a * 1
    assert b == a
    b = a * 2
    assert b == a + a


def test_from_ones():
    a = Tibs.from_ones(0)
    assert a == Tibs()
    a = Tibs.from_ones(1)
    assert a == Tibs("0b1")
    with pytest.raises(ValueError):
        _ = Tibs.from_ones(-1)


def test_from_zeros():
    a = Tibs.from_zeros(0)
    assert a == Tibs()
    a = Tibs.from_zeros(1)
    assert a == Tibs("0b0")
    with pytest.raises(ValueError):
        _ = Tibs.from_zeros(-1)


def test_bits_slicing():
    a = Tibs('0b1010101010101010')
    b = a[-5:-8:1]
    assert b == Tibs()

    assert a[::2] == Tibs('0xff')
    assert a[1::2] == Tibs('0x00')


def test_from_random():
    a = Tibs.from_random(0)
    assert a == Tibs()
    a = Tibs.from_random(1)
    assert a == Tibs('0b1') or a == Tibs('0b0')
    a = Tibs.from_random(10000, seed=b'a_seed')
    b = Tibs.from_random(10000, seed=b'a_seed')
    assert a == b
    b = Tibs.from_random(10000,
                         seed=b'a different seed this time - quite long to test if this makes a difference or not. It shouldnt really, but who knows?')
    assert a != b
    c = Mutibs.from_random(10000, seed=b'a_seed')
    assert a == c


def test_strict_equality_and_hashing():
    assert Tibs("0xf") == Tibs("0b1111")
    assert Tibs("0xf") == Mutibs("0xf")
    assert Mutibs("0xf") == Tibs("0xf")

    assert Tibs("0xf") != "0xf"
    assert "0xf" != Tibs("0xf")
    assert Tibs("0xf") != b"\x0f"
    assert Tibs("0b101") != [1, 0, 1]
    assert Mutibs("0xf") != "0xf"

    a = Tibs("0xabcd")
    b = Tibs("0xabcd")
    c = Tibs("0x00abcd")[8:]
    d = Tibs("0b11001101")

    assert len({a, b, c}) == 1
    assert hash(a) == hash(b) == hash(c)
    assert len({Tibs("0x0f"), Tibs("0b1111"), d}) == 3

    with pytest.raises(TypeError, match="unhashable"):
        hash(Mutibs("0xf"))


def test_is_things():
    a = Tibs('0b1010101010101010')
    b = Mutibs('0b1')
    assert isinstance(a, Iterable)
    assert isinstance(b, Iterable)


def test_bits_from_bytes_string():
    a = Tibs.from_bytes(b'ABC')
    assert a.bytes == b'ABC'


def test_bool_conversion():
    a = Tibs()
    b = Tibs('0b0')
    c = Tibs('0b1')
    assert not a
    assert b
    assert c


def test_find_all():
    a = Tibs(' 0 B 0 0 01011')
    g = a.find_all_iter('0b1')
    assert next(g) == 3
    assert next(g) == 5
    assert next(g) == 6
    with pytest.raises(StopIteration):
        _ = next(g)


def test_repr():
    a = Tibs()
    assert repr(a) == "Tibs()"
    a = Tibs('')
    assert repr(a) == "Tibs()"
    a = Tibs(" 0b 1")
    assert repr(a) == "Tibs('0b1')"


def test_bits_not_orderable():
    a = Tibs.from_string("0b0")
    b = Tibs.from_string("0b1")
    with pytest.raises(TypeError):
        _ = a < b
    with pytest.raises(TypeError):
        _ = a <= b
    with pytest.raises(TypeError):
        _ = a > b
    with pytest.raises(TypeError):
        _ = a >= b


def test_bools_from_iterable():
    v = [1, 0, 0, 1]
    i = iter(v)
    b = Tibs.from_bools(i)
    assert b == Tibs('0b1001')


def test_joined_from_iterable():
    v = [[0], '0b11']
    i = iter(v)
    b = Tibs.from_joined(v)
    assert b == Tibs('0b011')
    assert Tibs.from_joined(["0b1", [0, 1], b"\xff"]) == Tibs("0b10111111111")


def test_joined_repeated_bit_containers():
    expected = Tibs('0b101101101101')
    for cls in (Tibs, Mutibs):
        assert cls.from_joined([Tibs('0b101')] * 4) == expected
        assert cls.from_joined([Mutibs('0b101')] * 4) == expected

    # Equal but distinct objects use the general list path.
    assert Tibs.from_joined([Tibs('0b101') for _ in range(4)]) == expected


def test_promotion_from_mutibs():
    m = Mutibs('0x123')
    t = Tibs(m)
    assert isinstance(t, Tibs)
    assert m == t
    m2 = Mutibs(t)
    assert isinstance(m2, Mutibs)
    assert m2 == t
    m3 = Mutibs(m)
    assert isinstance(m3, Mutibs)
    assert m3 == t


def test_reversed():
    a = Tibs('0b1100')
    b = a.reversed()
    assert b == Tibs('0b0011')

    m1 = Mutibs('0b11100')
    m2 = m1.reversed()
    assert m2 == Tibs('0b00111')
    m3 = Mutibs.from_random(1_000_000)
    m4 = m3.reversed()
    m4.reverse()
    assert m3 == m4


@pytest.mark.parametrize('length', [0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1001])
def test_reversed_matches_bit_string(length):
    bits = ''.join('01101'[i % 5] for i in range(length))
    for cls in (Tibs, Mutibs):
        a = cls('0b' + bits) if bits else cls()
        assert a.reversed().bin == bits[::-1]
        # The original must be left alone by the copying version.
        assert a.bin == bits


@pytest.mark.parametrize('offset', range(9))
def test_reversed_with_storage_starting_mid_byte(offset):
    source = ''.join('0110100011110000101'[i % 19] for i in range(offset + 37))
    for cls in (Tibs, Mutibs):
        a = cls('0b' + source)[offset:]
        assert a.reversed().bin == source[offset:][::-1]


class TestCapacityLimit:
    """A length past what the platform can hold must raise, not panic.

    bitvec addresses a bit with a ``usize`` and spends three of those bits on
    the position within an element, so a container holds at most 2**61 - 1 bits
    on a 64-bit build but only 2**29 - 1 (about 64 MB) on a 32-bit one - which
    the x86 wheels are. Before this was guarded, exceeding it panicked inside
    bitvec, and pyo3 turns a panic into ``PanicException``, which derives from
    ``BaseException`` so that it tears down the interpreter rather than being
    caught by ordinary error handling.

    Nothing here allocates: every length used is past what any build accepts,
    so the check has to reject it before reaching the allocator.
    """

    # 2**61 - 1 on a 64-bit build, 2**29 - 1 on a 32-bit one.
    CAP = 2 ** ((sys.maxsize.bit_length() + 1) - 3) - 1

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    @pytest.mark.parametrize(
        "name, args",
        [
            ("from_zeros", ()),
            ("from_ones", ()),
            ("from_random", ()),
        ],
    )
    def test_length_constructors_raise(self, cls, name, args):
        with pytest.raises(MemoryError, match="supports at most"):
            getattr(cls, name)(self.CAP + 1, *args)

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    @pytest.mark.parametrize("name, value", [("from_u", 1), ("from_i", 1), ("from_f", 1.0)])
    def test_numeric_constructors_raise(self, cls, name, value):
        with pytest.raises(MemoryError, match="supports at most"):
            getattr(cls, name)(value, self.CAP + 1)

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    def test_the_error_is_catchable_as_exception(self, cls):
        # The whole point: PanicException derives from BaseException, so this
        # would not have caught it.
        try:
            cls.from_zeros(self.CAP + 1)
        except Exception as e:
            assert isinstance(e, MemoryError)
        else:
            pytest.fail("expected the capacity limit to be reported")

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    def test_a_length_wider_than_usize_is_not_truncated(self, cls):
        # Only a 32-bit build can truncate, where usize is narrower than the
        # i64 arriving from Python: 'length as usize' would turn 2**32 + 100
        # into a silently-wrong 100-bit container. The check runs before the
        # cast, so the length is rejected instead. On a 64-bit build the same
        # length is merely large and legal, so only the over-cap value applies.
        lengths = [2**63 - 1]
        if self.CAP < 2**32:
            lengths.append(2**32 + 100)
        for length in lengths:
            with pytest.raises(MemoryError):
                cls.from_zeros(length)

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    def test_negative_lengths_still_report_as_value_errors(self, cls):
        # The capacity check must not have swallowed the existing negative case.
        with pytest.raises(ValueError, match="Negative bit length"):
            cls.from_zeros(-1)
        with pytest.raises(ValueError, match="Negative bit length"):
            cls.from_random(-1)

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    def test_the_limit_itself_is_not_rejected(self, cls):
        # Guards an off-by-one that would cap the container one bit short. The
        # only way to observe the boundary is to allocate it, so this runs only
        # where that is 64 MB rather than an impossible 2**58 bytes.
        if self.CAP > 2**32:
            pytest.skip("allocating 2**61 bits is not a test")
        try:
            container = cls.from_zeros(self.CAP)
        except MemoryError as e:
            # Distinguish our own refusal, which is the bug being tested for,
            # from the machine genuinely not having 64 MB to spare - otherwise
            # this would be a flaky failure on a small 32-bit runner.
            if "supports at most" in str(e):
                pytest.fail(f"the capacity limit itself was rejected: {e}")
            pytest.skip("not enough memory to allocate the limit")
        assert len(container) == self.CAP
