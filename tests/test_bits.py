#!/usr/bin/env python
import pytest
import re
from hypothesis import given
import hypothesis.strategies as st
from tibs import Tibs, Mutibs
from typing import Iterable, Sequence


def remove_unprintable(s: str) -> str:
    colour_escape = re.compile(r"(?:\x1B[@-_])[0-?]*[ -/]*[@-~]")
    return colour_escape.sub("", s)


class TestCreation:
    def test_creation_from_bytes(self):
        s = Tibs.from_bytes(b"\xa0\xff")
        assert (len(s), s.to_hex()) == (16, "a0ff")

    @given(st.binary())
    def test_creation_from_bytes_roundtrip(self, data):
        s = Tibs.from_bytes(data)
        assert s.to_bytes() == data

    def test_creation_from_hex(self):
        s = Tibs.from_hex("0xA0ff")
        assert (len(s), s.to_hex()) == (16, "a0ff")


class TestInitialisation:
    def test_empty_init(self):
        a = Tibs()
        assert a == ""

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
        b = list(a.find_all('0b1'))
        assert b == [2, 5, 6]
        t = Tibs("0b10")
        tp = list(t.find_all("0b1"))
        assert tp == [0]


class TestCut:
    def test_cut(self):
        s = Tibs().from_joined(["0b000111"] * 10)
        for t in s.chunks(6):
            assert t == "0b000111"


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

    @pytest.mark.skip
    def test_unpack(self):
        s = Tibs.from_string("0b111000111")
        x, y = s.unpack(["bits3", "pad3", "bits3"])
        assert (x, y.unpack("u")) == ("0b111", 7)
        x, y = s.unpack(["bits2", "pad2", "bin5"])
        assert (x.unpack(["u2"])[0], y) == (3, "00111")
        x = s.unpack(["pad1", "pad2", "pad3"])
        assert x == ()


def test_adding():
    a = Tibs.from_string("0b0")
    b = Tibs.from_string("0b11")
    c = a + b
    assert c == "0b011"
    assert a == "0b0"
    assert b == "0b11"


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
        assert a.to_bin() == "000000010010"
        b = Tibs.from_string("0b0011_1100_1111_0000")
        assert b.to_bin() == "0011110011110000"

    def test_octal_creation(self):
        a = Tibs.from_oct("0011_2233_4455_6677")
        assert a.to_oct() == "0011223344556677"
        b = Tibs.from_string("0o123_321_123_321")
        assert b.to_oct() == "123321123321"


def test_from_iterable():
    with pytest.raises(TypeError):
        _ = Tibs.from_bools()
    a = Tibs.from_bools([])
    assert a == Tibs()
    a = Tibs.from_bools([1, 0, 1, 1])
    assert a == "0b1011"
    a = Tibs.from_bools((True,))
    assert a.to_bin() == "1"


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

    assert a[::2] == '0xff'
    assert a[1::2] == '0x00'


def test_from_random():
    a = Tibs.from_random(0)
    assert a == Tibs()
    a = Tibs.from_random(1)
    assert a == '0b1' or a == '0b0'
    a = Tibs.from_random(10000, b'a_seed')
    b = Tibs.from_random(10000, b'a_seed')
    assert a == b
    b = Tibs.from_random(10000,
                         b'a different seed this time - quite long to test if this makes a difference or not. It shouldnt really, but who knows?')
    assert a != b
    c = Mutibs.from_random(10000, b'a_seed')
    assert a == c


@pytest.mark.skip
def test_is_things():
    a = Tibs('0b1010101010101010')
    assert isinstance(a, Iterable)
    assert isinstance(a, Sequence)


def test_bits_from_bytes_string():
    a = Tibs.from_bytes(b'ABC')
    assert a.to_bytes() == b'ABC'


def test_bool_conversion():
    a = Tibs()
    b = Tibs('0b0')
    c = Tibs('0b1')
    assert not a
    assert b
    assert c


def test_find_all():
    a = Tibs(' 0 B 0 0 01011')
    g = a.find_all('0b1')
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
