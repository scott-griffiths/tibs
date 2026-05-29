#!/usr/bin/env python
import pytest
from tibs import Tibs, Mutibs, Endianness, Dtype, DtypeKind

def test_creation():
    d = Dtype(DtypeKind.Uint, 8)
    assert d.kind is DtypeKind.Uint
    assert d.length == 8


def test_uint_constructor():
    d = Dtype.u(8, Endianness.Little)
    assert d.kind is DtypeKind.Uint
    assert d.length == 8
    assert d.byte_order is Endianness.Little


def test_dtype_float():
    d = Dtype(DtypeKind.Float, 16, Endianness.Little)
    t = Tibs.from_dtype(d, 14.5)
    t2 = Tibs.from_f(14.5, 16, Endianness.Little)
    assert t == t2


def test_dtype_uint():
    d = Dtype(DtypeKind.Uint, 9)
    t = Tibs.from_dtype(d, 17)
    assert t == Tibs.from_u(17, 9)


def test_dtype_bin():
    d = Dtype(DtypeKind.Bin, 4)
    t = Tibs.from_dtype(d, "0b1010")
    assert t == Tibs.from_bin("0b1010")


def test_dtype_oct():
    d = Dtype(DtypeKind.Oct, 6)
    t = Tibs.from_dtype(d, "17")
    assert t == Tibs.from_oct("17")


def test_dtype_hex():
    d = Dtype(DtypeKind.Hex, 8)
    t = Tibs.from_dtype(d, "0f")
    assert t == Tibs.from_hex("0f")


def test_to_dtype_float():
    d = Dtype(DtypeKind.Float, 16, Endianness.Little)
    t = Tibs.from_dtype(d, 14.5)
    assert t.to_dtype(d) == 14.5


def test_to_dtype_uint():
    d = Dtype(DtypeKind.Uint, 16, Endianness.Little)
    t = Tibs.from_dtype(d, 0x0102)
    assert t.to_dtype(d) == 0x0102


def test_to_dtype_int():
    d = Dtype(DtypeKind.Int, 8)
    t = Tibs.from_dtype(d, -2)
    assert t.to_dtype(d) == -2


def test_to_dtype_bytes():
    d = Dtype(DtypeKind.Bytes, 16)
    t = Tibs.from_dtype(d, b"ab")
    assert t.to_dtype(d) == b"ab"


def test_to_dtype_bin():
    d = Dtype(DtypeKind.Bin, 4)
    t = Tibs.from_dtype(d, "0b1010")
    assert t.to_dtype(d) == "1010"


def test_to_dtype_oct():
    d = Dtype(DtypeKind.Oct, 6)
    t = Tibs.from_dtype(d, "17")
    assert t.to_dtype(d) == "17"


def test_to_dtype_hex():
    d = Dtype(DtypeKind.Hex, 8)
    t = Tibs.from_dtype(d, "0f")
    assert t.to_dtype(d) == "0f"


def test_to_dtype_slice():
    d = Dtype(DtypeKind.Hex, 8)
    t = Tibs.from_hex("aa0fbb")
    assert t.to_dtype(d, 8, 16) == "0f"


def test_to_dtype_requires_matching_length():
    d = Dtype(DtypeKind.Uint, 8)
    with pytest.raises(ValueError, match="dtype with length 8 bits"):
        Tibs.from_hex("0f0f").to_dtype(d)


def test_from_dtype_iter_uint():
    d = Dtype(DtypeKind.Uint, 8)
    t = Tibs.from_dtype_iter(d, [1, 2, 3])
    assert t == Tibs.from_bytes(b"\x01\x02\x03")


def test_from_dtype_iter_little_endian_uint():
    d = Dtype(DtypeKind.Uint, 16, Endianness.Little)
    t = Tibs.from_dtype_iter(d, [0x0102, 0x0304])
    assert t == Tibs.from_hex("02010403")


def test_from_dtype_iter_generator():
    d = Dtype(DtypeKind.Hex, 8)
    t = Tibs.from_dtype_iter(d, (x for x in ["aa", "bb", "cc"]))
    assert t == Tibs.from_hex("aabbcc")


def test_from_dtype_iter_empty():
    d = Dtype(DtypeKind.Uint, 8)
    assert Tibs.from_dtype_iter(d, []) == Tibs()


def test_from_dtype_iter_propagates_item_errors():
    d = Dtype(DtypeKind.Uint, 8)
    with pytest.raises(OverflowError, match="does not fit"):
        Tibs.from_dtype_iter(d, [1, 256])


def test_to_dtype_iter_uint():
    d = Dtype(DtypeKind.Uint, 8)
    t = Tibs.from_dtype_iter(d, [1, 2, 3])
    assert list(t.to_dtype_iter(d)) == [1, 2, 3]


def test_to_dtype_iter_little_endian_uint():
    d = Dtype(DtypeKind.Uint, 16, Endianness.Little)
    t = Tibs.from_hex("02010403")
    assert list(t.to_dtype_iter(d)) == [0x0102, 0x0304]


def test_to_dtype_iter_strings():
    d = Dtype(DtypeKind.Hex, 8)
    t = Tibs.from_hex("aabbcc")
    assert list(t.to_dtype_iter(d)) == ["aa", "bb", "cc"]


def test_to_dtype_iter_slice():
    d = Dtype(DtypeKind.Uint, 8)
    t = Tibs.from_bytes(b"\x00\x01\x02\x03")
    assert list(t.to_dtype_iter(d, 8, 24)) == [1, 2]


def test_to_dtype_iter_empty():
    d = Dtype(DtypeKind.Uint, 8)
    assert list(Tibs().to_dtype_iter(d)) == []


def test_to_dtype_iter_rejects_zero_length_dtype():
    with pytest.raises(ValueError, match="greater than zero"):
        Dtype(DtypeKind.Bin, 0)


def test_to_dtype_iter_requires_multiple_of_dtype_length():
    d = Dtype(DtypeKind.Uint, 8)
    with pytest.raises(ValueError, match="not a multiple"):
        Tibs.from_bin("1010").to_dtype_iter(d)
