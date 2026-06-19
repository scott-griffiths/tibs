#!/usr/bin/env python
import pytest
from tibs import Tibs, Mutibs, Endianness, Dtype, DtypeKind

def test_creation():
    d = Dtype("u8")
    assert d.kind is DtypeKind.Uint
    assert d.length == 8


def test_parse_little_endian_uint():
    d = Dtype("u8_le")
    assert d.kind is DtypeKind.Uint
    assert d.length == 8
    assert d.byte_order is Endianness.Little


@pytest.mark.parametrize(
    "spec,kind",
    [
        ("i8_le", DtypeKind.Int),
        ("f32_be", DtypeKind.Float),
    ],
)
def test_endian_dtype_specs(spec, kind):
    d = Dtype(spec)
    assert d.kind is kind
    assert d.byte_order is not Endianness.Unspecified


@pytest.mark.parametrize(
    "spec,kind",
    [
        ("bytes8", DtypeKind.Bytes),
        ("bin8", DtypeKind.Bin),
        ("oct9", DtypeKind.Oct),
        ("hex8", DtypeKind.Hex),
    ],
)
def test_dtype_kind_specs(spec, kind):
    d = Dtype(spec)
    assert d.kind is kind
    assert d.byte_order is Endianness.Unspecified


def test_from_params():
    d = Dtype.from_params(DtypeKind.Uint, 16, Endianness.Little)
    assert d.kind is DtypeKind.Uint
    assert d.length == 16
    assert d.byte_order is Endianness.Little


def test_repr_is_parseable():
    d = Dtype("u16_le")
    assert repr(d) == "Dtype('u16_le')"
    assert Dtype("u16_le").kind is DtypeKind.Uint


@pytest.mark.parametrize("spec", ["", "u", "unknown8", "u8_xe", "hex8_le", "u7_le", "u0"])
def test_invalid_specs(spec):
    with pytest.raises(ValueError):
        Dtype(spec)


def test_from_value_float():
    d = Dtype("f16_le")
    t = Tibs.from_value(d, 14.5)
    t2 = Tibs.from_f(14.5, 16, Endianness.Little)
    assert t == t2


def test_from_value_uint():
    d = Dtype("u9")
    t = Tibs.from_value(d, 17)
    assert t == Tibs.from_u(17, 9)


def test_from_value_accepts_dtype_string():
    assert Tibs.from_value("u16_le", 0x0102) == Tibs.from_hex("0201")


def test_from_value_bin():
    d = Dtype("bin4")
    t = Tibs.from_value(d, "0b1010")
    assert t == Tibs.from_bin("0b1010")


def test_from_value_oct():
    d = Dtype("oct6")
    t = Tibs.from_value(d, "17")
    assert t == Tibs.from_oct("17")


def test_from_value_hex():
    d = Dtype("hex8")
    t = Tibs.from_value(d, "0f")
    assert t == Tibs.from_hex("0f")


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("bin4", "101"),
        ("bin4", "10101"),
        ("oct6", "7"),
        ("oct6", "777"),
        ("hex8", "f"),
        ("hex8", "fff"),
    ],
)
def test_from_value_textual_requires_dtype_length(dtype, value):
    with pytest.raises(ValueError, match="Dtype length"):
        Tibs.from_value(dtype, value)


def test_mutibs_from_value_textual_requires_dtype_length():
    with pytest.raises(ValueError, match="Dtype length"):
        Mutibs.from_value("hex8", "f")


def test_from_values_textual_requires_each_item_to_match_dtype_length():
    with pytest.raises(ValueError, match="Dtype length"):
        Tibs.from_values("hex8", ["0f", "f"])


def test_to_value_float():
    d = Dtype("f16_le")
    t = Tibs.from_value(d, 14.5)
    assert t.to_value(d) == 14.5


def test_to_value_uint():
    d = Dtype("u16_le")
    t = Tibs.from_value(d, 0x0102)
    assert t.to_value(d) == 0x0102


def test_to_value_accepts_dtype_string():
    t = Tibs.from_hex("0201")
    assert t.to_value("u16_le") == 0x0102


def test_to_value_int():
    d = Dtype("i8")
    t = Tibs.from_value(d, -2)
    assert t.to_value(d) == -2


def test_to_value_bytes():
    d = Dtype("bytes16")
    t = Tibs.from_value(d, b"ab")
    assert t.to_value(d) == b"ab"


def test_to_value_bin():
    d = Dtype("bin4")
    t = Tibs.from_value(d, "0b1010")
    assert t.to_value(d) == "1010"


def test_to_value_oct():
    d = Dtype("oct6")
    t = Tibs.from_value(d, "17")
    assert t.to_value(d) == "17"


def test_to_value_hex():
    d = Dtype("hex8")
    t = Tibs.from_value(d, "0f")
    assert t.to_value(d) == "0f"


def test_to_value_slice():
    d = Dtype("hex8")
    t = Tibs.from_hex("aa0fbb")
    assert t.to_value(d, 8, 16) == "0f"


def test_to_value_requires_matching_length():
    d = Dtype("u8")
    with pytest.raises(ValueError, match="dtype with length 8 bits"):
        Tibs.from_hex("0f0f").to_value(d)


def test_from_values_uint():
    d = Dtype("u8")
    t = Tibs.from_values(d, [1, 2, 3])
    assert t == Tibs.from_bytes(b"\x01\x02\x03")


def test_from_values_accepts_dtype_string():
    t = Tibs.from_values("u16_le", [0x0102, 0x0304])
    assert t == Tibs.from_hex("02010403")


def test_from_values_little_endian_uint():
    d = Dtype("u16_le")
    t = Tibs.from_values(d, [0x0102, 0x0304])
    assert t == Tibs.from_hex("02010403")


def test_from_values_generator():
    d = Dtype("hex8")
    t = Tibs.from_values(d, (x for x in ["aa", "bb", "cc"]))
    assert t == Tibs.from_hex("aabbcc")


def test_from_values_empty():
    d = Dtype("u8")
    assert Tibs.from_values(d, []) == Tibs()


def test_from_values_propagates_item_errors():
    d = Dtype("u8")
    with pytest.raises(OverflowError, match="does not fit"):
        Tibs.from_values(d, [1, 256])


def test_to_values_iter_uint():
    d = Dtype("u8")
    t = Tibs.from_values(d, [1, 2, 3])
    assert list(t.to_values_iter(d)) == [1, 2, 3]


def test_to_values_iter_accepts_dtype_string():
    t = Tibs.from_hex("02010403")
    assert list(t.to_values_iter("u16_le")) == [0x0102, 0x0304]


def test_to_values_uint():
    d = Dtype("u8")
    t = Tibs.from_values(d, [1, 2, 3])
    assert t.to_values(d) == [1, 2, 3]


def test_to_values_accepts_dtype_string():
    t = Tibs.from_hex("02010403")
    assert t.to_values("u16_le") == [0x0102, 0x0304]


def test_to_values_iter_little_endian_uint():
    d = Dtype("u16_le")
    t = Tibs.from_hex("02010403")
    assert list(t.to_values_iter(d)) == [0x0102, 0x0304]


def test_to_values_iter_strings():
    d = Dtype("hex8")
    t = Tibs.from_hex("aabbcc")
    assert list(t.to_values_iter(d)) == ["aa", "bb", "cc"]


def test_to_values_iter_slice():
    d = Dtype("u8")
    t = Tibs.from_bytes(b"\x00\x01\x02\x03")
    assert list(t.to_values_iter(d, 8, 24)) == [1, 2]


def test_to_values_slice():
    d = Dtype("u8")
    t = Tibs.from_bytes(b"\x00\x01\x02\x03")
    assert t.to_values(d, 8, 24) == [1, 2]


def test_to_values_iter_empty():
    d = Dtype("u8")
    assert list(Tibs().to_values_iter(d)) == []


def test_to_values_empty():
    d = Dtype("u8")
    assert Tibs().to_values(d) == []


def test_to_values_iter_rejects_zero_length_dtype():
    with pytest.raises(ValueError, match="greater than zero"):
        Dtype("bin0")


def test_to_values_iter_requires_multiple_of_dtype_length():
    d = Dtype("u8")
    with pytest.raises(ValueError, match="not a multiple"):
        Tibs.from_bin("1010").to_values_iter(d)


def test_to_values_requires_multiple_of_dtype_length():
    d = Dtype("u8")
    with pytest.raises(ValueError, match="not a multiple"):
        Tibs.from_bin("1010").to_values(d)


def test_mutibs_from_value():
    d = Dtype("u8")
    assert Mutibs.from_value(d, 1) == Mutibs.from_bytes(b"\x01")


def test_mutibs_from_value_accepts_dtype_string():
    assert Mutibs.from_value("u16_le", 0x0102) == Mutibs.from_hex("0201")


def test_mutibs_from_values():
    d = Dtype("u8")
    assert Mutibs.from_values(d, [1, 2, 3]) == Mutibs.from_bytes(b"\x01\x02\x03")


def test_mutibs_from_values_accepts_dtype_string():
    assert Mutibs.from_values("u16_le", [0x0102, 0x0304]) == Mutibs.from_hex("02010403")


def test_mutibs_to_value():
    d = Dtype("u8")
    m = Mutibs.from_bytes(b"\x01")
    assert m.to_value(d) == 1


def test_mutibs_to_value_accepts_dtype_string():
    m = Mutibs.from_hex("0201")
    assert m.to_value("u16_le") == 0x0102


def test_mutibs_to_values():
    d = Dtype("u8")
    m = Mutibs.from_bytes(b"\x01\x02\x03")
    values = m.to_values(d)
    m[0] = 0
    assert values == [1, 2, 3]


def test_mutibs_to_values_accepts_dtype_string():
    m = Mutibs.from_hex("02010403")
    assert m.to_values("u16_le") == [0x0102, 0x0304]


def test_mutibs_to_values_slice():
    d = Dtype("u8")
    m = Mutibs.from_bytes(b"\x00\x01\x02\x03")
    assert m.to_values(d, 8, 24) == [1, 2]


def test_dtype_string_errors_are_reported_by_value_methods():
    with pytest.raises(ValueError, match="Cannot parse Dtype spec"):
        Tibs.from_value("unknown8", 1)
    with pytest.raises(TypeError, match="dtype must be a Dtype instance or dtype string"):
        Tibs.from_value(object(), 1)


def test_old_dtype_method_names_are_not_exposed():
    for name in ["u", "i", "f", "bytes", "bin", "oct", "hex"]:
        assert not hasattr(Dtype, name)

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
