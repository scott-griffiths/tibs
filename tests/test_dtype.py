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


@pytest.mark.parametrize(
    "constructor,kind",
    [
        (Dtype.i, DtypeKind.Int),
        (Dtype.f, DtypeKind.Float),
    ],
)
def test_endian_dtype_kind_constructors(constructor, kind):
    d = constructor(8, Endianness.Little)
    assert d.kind is kind
    assert d.length == 8
    assert d.byte_order is Endianness.Little


@pytest.mark.parametrize(
    "constructor,kind",
    [
        (Dtype.bytes, DtypeKind.Bytes),
        (Dtype.bin, DtypeKind.Bin),
        (Dtype.oct, DtypeKind.Oct),
        (Dtype.hex, DtypeKind.Hex),
    ],
)
def test_dtype_kind_constructors(constructor, kind):
    d = constructor(8)
    assert d.kind is kind
    assert d.length == 8
    assert d.byte_order is Endianness.Unspecified


def test_from_value_float():
    d = Dtype(DtypeKind.Float, 16, Endianness.Little)
    t = Tibs.from_value(d, 14.5)
    t2 = Tibs.from_f(14.5, 16, Endianness.Little)
    assert t == t2


def test_from_value_uint():
    d = Dtype(DtypeKind.Uint, 9)
    t = Tibs.from_value(d, 17)
    assert t == Tibs.from_u(17, 9)


def test_from_value_bin():
    d = Dtype(DtypeKind.Bin, 4)
    t = Tibs.from_value(d, "0b1010")
    assert t == Tibs.from_bin("0b1010")


def test_from_value_oct():
    d = Dtype(DtypeKind.Oct, 6)
    t = Tibs.from_value(d, "17")
    assert t == Tibs.from_oct("17")


def test_from_value_hex():
    d = Dtype(DtypeKind.Hex, 8)
    t = Tibs.from_value(d, "0f")
    assert t == Tibs.from_hex("0f")


def test_to_value_float():
    d = Dtype(DtypeKind.Float, 16, Endianness.Little)
    t = Tibs.from_value(d, 14.5)
    assert t.to_value(d) == 14.5


def test_to_value_uint():
    d = Dtype(DtypeKind.Uint, 16, Endianness.Little)
    t = Tibs.from_value(d, 0x0102)
    assert t.to_value(d) == 0x0102


def test_to_value_int():
    d = Dtype(DtypeKind.Int, 8)
    t = Tibs.from_value(d, -2)
    assert t.to_value(d) == -2


def test_to_value_bytes():
    d = Dtype(DtypeKind.Bytes, 16)
    t = Tibs.from_value(d, b"ab")
    assert t.to_value(d) == b"ab"


def test_to_value_bin():
    d = Dtype(DtypeKind.Bin, 4)
    t = Tibs.from_value(d, "0b1010")
    assert t.to_value(d) == "1010"


def test_to_value_oct():
    d = Dtype(DtypeKind.Oct, 6)
    t = Tibs.from_value(d, "17")
    assert t.to_value(d) == "17"


def test_to_value_hex():
    d = Dtype(DtypeKind.Hex, 8)
    t = Tibs.from_value(d, "0f")
    assert t.to_value(d) == "0f"


def test_to_value_slice():
    d = Dtype(DtypeKind.Hex, 8)
    t = Tibs.from_hex("aa0fbb")
    assert t.to_value(d, 8, 16) == "0f"


def test_to_value_requires_matching_length():
    d = Dtype(DtypeKind.Uint, 8)
    with pytest.raises(ValueError, match="dtype with length 8 bits"):
        Tibs.from_hex("0f0f").to_value(d)


def test_from_values_uint():
    d = Dtype(DtypeKind.Uint, 8)
    t = Tibs.from_values(d, [1, 2, 3])
    assert t == Tibs.from_bytes(b"\x01\x02\x03")


def test_from_values_little_endian_uint():
    d = Dtype(DtypeKind.Uint, 16, Endianness.Little)
    t = Tibs.from_values(d, [0x0102, 0x0304])
    assert t == Tibs.from_hex("02010403")


def test_from_values_generator():
    d = Dtype(DtypeKind.Hex, 8)
    t = Tibs.from_values(d, (x for x in ["aa", "bb", "cc"]))
    assert t == Tibs.from_hex("aabbcc")


def test_from_values_empty():
    d = Dtype(DtypeKind.Uint, 8)
    assert Tibs.from_values(d, []) == Tibs()


def test_from_values_propagates_item_errors():
    d = Dtype(DtypeKind.Uint, 8)
    with pytest.raises(OverflowError, match="does not fit"):
        Tibs.from_values(d, [1, 256])


def test_to_values_iter_uint():
    d = Dtype(DtypeKind.Uint, 8)
    t = Tibs.from_values(d, [1, 2, 3])
    assert list(t.to_values_iter(d)) == [1, 2, 3]


def test_to_values_uint():
    d = Dtype(DtypeKind.Uint, 8)
    t = Tibs.from_values(d, [1, 2, 3])
    assert t.to_values(d) == [1, 2, 3]


def test_to_values_iter_little_endian_uint():
    d = Dtype(DtypeKind.Uint, 16, Endianness.Little)
    t = Tibs.from_hex("02010403")
    assert list(t.to_values_iter(d)) == [0x0102, 0x0304]


def test_to_values_iter_strings():
    d = Dtype(DtypeKind.Hex, 8)
    t = Tibs.from_hex("aabbcc")
    assert list(t.to_values_iter(d)) == ["aa", "bb", "cc"]


def test_to_values_iter_slice():
    d = Dtype(DtypeKind.Uint, 8)
    t = Tibs.from_bytes(b"\x00\x01\x02\x03")
    assert list(t.to_values_iter(d, 8, 24)) == [1, 2]


def test_to_values_slice():
    d = Dtype(DtypeKind.Uint, 8)
    t = Tibs.from_bytes(b"\x00\x01\x02\x03")
    assert t.to_values(d, 8, 24) == [1, 2]


def test_to_values_iter_empty():
    d = Dtype(DtypeKind.Uint, 8)
    assert list(Tibs().to_values_iter(d)) == []


def test_to_values_empty():
    d = Dtype(DtypeKind.Uint, 8)
    assert Tibs().to_values(d) == []


def test_to_values_iter_rejects_zero_length_dtype():
    with pytest.raises(ValueError, match="greater than zero"):
        Dtype(DtypeKind.Bin, 0)


def test_to_values_iter_requires_multiple_of_dtype_length():
    d = Dtype(DtypeKind.Uint, 8)
    with pytest.raises(ValueError, match="not a multiple"):
        Tibs.from_bin("1010").to_values_iter(d)


def test_to_values_requires_multiple_of_dtype_length():
    d = Dtype(DtypeKind.Uint, 8)
    with pytest.raises(ValueError, match="not a multiple"):
        Tibs.from_bin("1010").to_values(d)


def test_mutibs_from_value():
    d = Dtype(DtypeKind.Uint, 8)
    assert Mutibs.from_value(d, 1) == Mutibs.from_bytes(b"\x01")


def test_mutibs_from_values():
    d = Dtype(DtypeKind.Uint, 8)
    assert Mutibs.from_values(d, [1, 2, 3]) == Mutibs.from_bytes(b"\x01\x02\x03")


def test_mutibs_to_value():
    d = Dtype(DtypeKind.Uint, 8)
    m = Mutibs.from_bytes(b"\x01")
    assert m.to_value(d) == 1


def test_mutibs_to_values():
    d = Dtype(DtypeKind.Uint, 8)
    m = Mutibs.from_bytes(b"\x01\x02\x03")
    values = m.to_values(d)
    m[0] = 0
    assert values == [1, 2, 3]


def test_mutibs_to_values_slice():
    d = Dtype(DtypeKind.Uint, 8)
    m = Mutibs.from_bytes(b"\x00\x01\x02\x03")
    assert m.to_values(d, 8, 24) == [1, 2]


def test_old_dtype_method_names_are_not_exposed():
    for name in [
        "from_dtype",
        "from_dtype_iter",
        "from_dtypes",
        "to_dtype",
        "to_dtype_iter",
        "to_dtypes",
    ]:
        assert not hasattr(Tibs, name)

    for name in ["to_dtypes", "to_values_iter"]:
        assert not hasattr(Mutibs, name)
