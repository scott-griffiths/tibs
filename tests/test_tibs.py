#!/usr/bin/env python
import pytest
from tibs import Tibs, Mutibs


def test_from_bin():
    a = Tibs.from_bin('010')
    b = Tibs.from_string('0b010')
    c = Mutibs.from_bin('0b010')
    d = Tibs('0b010')
    assert a == b == c == d


def test_to_bin():
    a = Tibs('0b1001')
    assert a.to_bin() == '1001'
    assert a.to_mutibs().to_bin() == '1001'


def test_from_oct():
    a = Tibs.from_oct('12')
    b = Tibs.from_string('0o12')
    c = Mutibs.from_oct('0o12')
    d = Tibs('0o12')
    assert a == b == c == d


def test_to_oct():
    a = Tibs('0b001100')
    assert a.to_oct() == '14'
    assert a.to_mutibs().to_oct() == '14'


def test_from_hex():
    a = Tibs.from_hex('A')
    b = Tibs.from_string('0xA')
    c = Mutibs.from_hex('0xA')
    d = Tibs('0xA')
    assert a == b == c == d


def test_to_hex():
    a = Tibs('0b1010')
    assert a.to_hex() == 'a'
    assert a.to_mutibs().to_hex() == 'a'


def test_rfind():
    a = Mutibs()
    a += '0b1110001110'
    b = a.rfind('0b111')
    assert b == 6


def test_count_large():
    a = Tibs('0b' + '1' * 72)
    b = a[:65]
    assert b.count(1) == 65


def test_from_u():
    a = Tibs.from_u(15, 8)
    assert a == '0b00001111'
    b = Mutibs.from_u(15, 8)
    assert a == b
    c = a.to_u()
    assert c == 15


def test_from_u_errors():
    with pytest.raises(ValueError):
        _ = Tibs.from_u(0, -1)
    with pytest.raises(ValueError):
        _ = Tibs.from_u(0, 0)
    with pytest.raises(OverflowError):
        _ = Tibs.from_u(-1, 5)


def test_from_i():
    a = Tibs.from_i(-9, 100)
    b = Mutibs.from_i(-9, 100)
    assert a == b
    assert len(a) == 100
    assert a.to_i() == -9
    assert b.to_i() == -9


def test_from_i_errors():
    with pytest.raises(OverflowError):
        _ = Tibs.from_i(4, 2)


def test_from_large_ints():
    with pytest.raises(ValueError):
        _ = Tibs.from_i(-1, 1000)
    a = Tibs.from_i(-1, 128)
    assert a.all()
    with pytest.raises(ValueError):
        _ = Mutibs.from_i(-1, 1000)
    b = Mutibs.from_u(0, 128)
    assert not b.any()
    assert a.to_i() == -1
    assert b.to_u() == 0


def test_from_f():
    a = Tibs.from_f(0.25, 16)
    b = Tibs.from_f(0.25, 32)
    c = Tibs.from_f(0.25, 64)
    a2 = Mutibs.from_f(0.25, 16)
    b2 = Mutibs.from_f(0.25, 32)
    c2 = Mutibs.from_f(0.25, 64)
    assert a == a2
    assert len(a) == 16
    assert len(b) == 32
    assert len(c) == 64
    assert b == b2
    assert c == c2
    f = a.to_f()
    g = b.to_f()
    h = c.to_f()
    f2 = a2.to_f()
    g2 = b2.to_f()
    h2 = c2.to_f()
    assert f == g == h == f2 == g2 == h2 == 0.25

def test_raw_bytes_and_offset():
    a = Tibs('0xff00ff')
    raw_bytes, offset, length = a.to_raw_data()
    assert raw_bytes == b'\xff\x00\xff'
    assert offset == 0
    b = a[4:20]
    raw_bytes, offset, length = b.to_raw_data()
    assert offset == 4
    assert raw_bytes == b'\xff\x00\xff'
    assert Tibs.from_bytes(raw_bytes) & '0x0ffff0' == '0x0f00f0'

def test_mutibs_raw_bytes_and_offset():
    a = Mutibs('0xff')
    b = a[4:]
    b += '0x77'
    assert b == '0xf77'
    raw_bytes, offset, length = b.to_raw_data()
    assert Tibs.from_bytes(raw_bytes) & '0x0fff' == '0x0f77'
    assert offset == 4
    assert b == '0xf77'
    raw_bytes, offset, length = b.as_raw_data()
    assert Tibs.from_bytes(raw_bytes) & '0x0fff' == '0x0f77'
    assert offset == 4
    assert length == 12
    assert b == []

def test_from_bytes_offsets():
    x = b'\xff\x00\xee\x11'
    a = Tibs.from_bytes(x)
    assert a == '0xff00ee11'
    b = Tibs.from_bytes(x, None,16)
    assert b == '0xff00'
    c = Tibs.from_bytes(x, offset=16)
    assert c == '0xee11'
    d = Tibs.from_bytes(x, 4, 12)
    assert d == '0xf00'
    e = Mutibs.from_bytes(x, length=4, offset=28)
    assert e == '0x1'
    f = Mutibs.from_bytes(x, 0, 32)
    assert f == a
    g = Mutibs.from_bytes(x, 0, 0)
    assert g == []


def test_from_bytes_errors():
    x = b'\xff\x00\xee\x11'
    with pytest.raises(ValueError):
        _ = Tibs.from_bytes(x, length=33)
    with pytest.raises(ValueError):
        _ = Tibs.from_bytes(x, None, -1)
    with pytest.raises(ValueError):
        _ = Tibs.from_bytes(x, offset=-1)
    with pytest.raises(ValueError):
        _ = Tibs.from_bytes(x, length=-1)
    with pytest.raises(ValueError):
        _ = Tibs.from_bytes(x, offset=28, length=5)


def test_bit_ops_alignments():
    a = Tibs('0x00ff00')
    b = a[4:20]
    c = a[2:18]
    assert b & c == '0b0000001111110000'

    a = Mutibs('0x00ff00')
    b = a[4:20]
    c = a[2:18]
    assert b & c == '0b0000001111110000'


def test_raw_data_bug():
    a = Mutibs.from_bytes(b'hello')
    b = a[8:]
    assert a.to_raw_data() == (b'hello', 0, 40)
    assert b.to_raw_data() == (b'ello', 0, 32)

    a = Tibs.from_bytes(b'hello')
    b = a[8:]
    assert a.to_raw_data() == (b'hello', 0, 40)
    assert b.to_raw_data() == (b'ello', 0, 32)

def test_from_bools_generator():
    bits = [1, 0, 0, 1, 0]
    generator = (y for y in bits)
    t = Tibs.from_bools(generator)
    assert list(t) == bits