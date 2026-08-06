#!/usr/bin/env python
import math
import re

import pytest
from tibs import Tibs, Mutibs, ByteOrder, Dtype, DtypeKind, DtypeSingle

def test_creation():
    d = Dtype("u8")
    assert type(d) is DtypeSingle
    assert d.kind is DtypeKind.Uint
    assert d.length == 8


def test_parse_little_endian_uint():
    d = Dtype("u8_le")
    assert d.kind is DtypeKind.Uint
    assert d.length == 8
    assert d.byte_order is ByteOrder.Little


@pytest.mark.parametrize(
    "spec,kind",
    [
        ("i8_le", DtypeKind.Int),
        ("f32_be", DtypeKind.Float),
        ("bf16_le", DtypeKind.BFloat),
    ],
)
def test_byte_order_dtype_specs(spec, kind):
    d = Dtype(spec)
    assert d.kind is kind
    assert d.byte_order is not ByteOrder.Unspecified


@pytest.mark.parametrize(
    "spec,kind",
    [
        ("bytes8", DtypeKind.Bytes),
        ("bool", DtypeKind.Bool),
        ("bits4", DtypeKind.Bits),
        ("bin8", DtypeKind.Bin),
        ("oct9", DtypeKind.Oct),
        ("hex8", DtypeKind.Hex),
        ("bf16", DtypeKind.BFloat),
    ],
)
def test_dtype_kind_specs(spec, kind):
    d = Dtype(spec)
    assert d.kind is kind
    assert d.byte_order is ByteOrder.Unspecified


def test_dtype_single_from_params():
    d = DtypeSingle.from_params(DtypeKind.Uint, 16, ByteOrder.Little)
    assert d.kind is DtypeKind.Uint
    assert d.length == 16
    assert d.byte_order is ByteOrder.Little


def test_dtype_single_bool_from_params_requires_length_one():
    d = DtypeSingle.from_params(DtypeKind.Bool, 1)
    assert d.kind is DtypeKind.Bool
    assert d.length == 1

    with pytest.raises(ValueError, match="length 1"):
        DtypeSingle.from_params(DtypeKind.Bool, 2)


def test_dtype_single_bits_from_params():
    d = DtypeSingle.from_params(DtypeKind.Bits, 3)
    assert d.kind is DtypeKind.Bits
    assert d.length == 3
    assert repr(d) == "DtypeSingle('bits3')"


def test_repr_is_parseable():
    d = Dtype("u16_le")
    assert repr(d) == "DtypeSingle('u16_le')"
    assert Dtype("u16_le").kind is DtypeKind.Uint


@pytest.mark.parametrize("spec", ["bool", "bits7", "bf16", "bf16_le", "bf16_be"])
def test_new_dtype_repr_is_parseable(spec):
    assert repr(Dtype(spec)) == f"DtypeSingle('{spec}')"


def test_dtype_equality_and_hashing_are_by_value():
    assert Dtype("u8") == Dtype("u8")
    assert Dtype("u8") != Dtype("u16")
    assert Dtype("u8") != "u8"
    assert "u8" != Dtype("u8")
    assert {Dtype("u8"), Dtype("u8"), Dtype("u16")} == {Dtype("u8"), Dtype("u16")}
    assert {Dtype("u8"): "value"}[Dtype("u8")] == "value"


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "u",
        "unknown8",
        "u8_xe",
        "hex8_le",
        "u7_le",
        "u0",
        "bool1",
        "bool0",
        "bool_le",
        "bits",
        "bits0",
        "bits8_le",
        "bf",
        "bf16_xe",
        "bf0",
    ],
)
def test_invalid_specs(spec):
    with pytest.raises(ValueError):
        Dtype(spec)


@pytest.mark.parametrize(
    "kind,length",
    [
        (DtypeKind.Bool, 1),
        (DtypeKind.Bits, 8),
        (DtypeKind.Bytes, 8),
        (DtypeKind.Bin, 8),
        (DtypeKind.Oct, 9),
        (DtypeKind.Hex, 8),
    ],
)
def test_byte_order_rejected_for_non_numeric_kinds(kind, length):
    with pytest.raises(ValueError, match="byte order"):
        DtypeSingle.from_params(kind, length, ByteOrder.Little)


# bf16 is bfloat16: 1 sign bit, 8 exponent bits and 7 mantissa bits, which is
# the top half of the f32 encoding rather than an IEEE 16-bit float. It needs a
# kind of its own because (DtypeKind.Float, 16) already means binary16, so the
# length cannot pick the decoder on its own. That is what most of these check.


def test_bf16_is_a_kind_of_its_own_not_a_float_of_length_16():
    d = Dtype("bf16")

    assert d.kind is DtypeKind.BFloat
    assert d.kind is not DtypeKind.Float
    assert d.length == 16
    assert d.byte_order is ByteOrder.Unspecified
    assert repr(d) == "DtypeSingle('bf16')"

    assert Dtype("bf16") != Dtype("f16")
    assert len({Dtype("bf16"), Dtype("f16")}) == 2


def test_bf16_from_params_round_trips_through_spec():
    d = DtypeSingle.from_params(DtypeKind.BFloat, 16, ByteOrder.Little)

    assert d.kind is DtypeKind.BFloat
    assert d.length == 16
    assert d.byte_order is ByteOrder.Little
    assert str(d) == "bf16_le"
    assert Dtype(str(d)) == d


@pytest.mark.parametrize("length", [8, 15, 17, 32, 64])
def test_bf16_is_the_only_bfloat_length(length):
    with pytest.raises(ValueError, match="length 16"):
        DtypeSingle.from_params(DtypeKind.BFloat, length)
    with pytest.raises(ValueError, match="length 16"):
        Dtype(f"bf{length}")


@pytest.mark.parametrize("spec", ["bfloat", "bfloat16", "bfloat16_le", "BFloat16"])
def test_bfloat_spellings_are_rejected_with_a_pointer_to_bf16(spec):
    # One canonical spelling, but the obvious wrong turns say where to go.
    with pytest.raises(ValueError, match="did you mean 'bf16'"):
        Dtype(spec)


# 1.0 is 0x3f800000 as an f32, and bf16 is its top half.
BF16_PATTERNS = [
    (0.0, "0000"),
    (-0.0, "8000"),
    (1.0, "3f80"),
    (-2.0, "c000"),
    (-2.5, "c020"),
    (0.125, "3e00"),
    (256.0, "4380"),
    (1.0078125, "3f81"),
    (float("inf"), "7f80"),
    (float("-inf"), "ff80"),
]


@pytest.mark.parametrize("value,hex_pattern", BF16_PATTERNS)
def test_bf16_known_bit_patterns_round_trip(value, hex_pattern):
    packed = Tibs.from_value("bf16", value)

    assert packed.hex == hex_pattern
    assert len(packed) == 16
    assert packed.to_value("bf16") == value
    assert math.copysign(1.0, packed.to_value("bf16")) == math.copysign(1.0, value)


def test_bf16_and_f16_read_the_same_bits_as_different_numbers():
    # The whole reason bf16 cannot be a 16-bit DtypeKind.Float.
    t = Tibs("0x3f80")

    assert t.to_value("bf16") == 1.0
    assert t.to_value("f16") == 1.875

    assert Tibs.from_value("bf16", 1.0).hex == "3f80"
    assert Tibs.from_value("f16", 1.0).hex == "3c00"
    assert Tibs.from_value("f32", 1.0).hex == "3f800000"


def test_bf16_keeps_f32_range_and_loses_mantissa_precision():
    # 8 exponent bits reach what f16 flushes to zero; 7 mantissa bits cannot
    # hold what f16's 10 can.
    assert Tibs.from_value("bf16", 1e-8).to_value("bf16") == pytest.approx(1e-8, rel=1e-2)
    assert Tibs.from_value("f16", 1e-8).to_value("f16") == 0.0

    assert Tibs.from_value("bf16", 1.001).to_value("bf16") == 1.0
    assert Tibs.from_value("f16", 1.001).to_value("f16") == 1.0009765625


def test_bf16_nan_round_trips_as_nan():
    assert math.isnan(Tibs.from_value("bf16", float("nan")).to_value("bf16"))


def test_bf16_byte_order_swaps_the_two_bytes():
    assert Tibs.from_value("bf16", 1.0).hex == "3f80"
    assert Tibs.from_value("bf16_be", 1.0).hex == "3f80"
    assert Tibs.from_value("bf16_le", 1.0).hex == "803f"

    assert Tibs("0x803f").to_value("bf16_le") == 1.0
    assert Tibs("0x3f80").to_value("bf16_be") == 1.0


def test_bf16_values_pack_and_unpack_in_bulk():
    values = [1.0, -2.0, 0.125, 256.0]

    packed = Tibs.from_values("bf16", values)

    assert packed.hex == "3f80c0003e004380"
    assert packed.to_values("bf16") == values
    assert list(packed.to_values_iter("bf16")) == values


def test_bf16_rejects_values_that_are_not_numbers():
    for bad in ("nope", None, [1]):
        with pytest.raises(TypeError):
            Tibs.from_value("bf16", bad)


def test_from_value_float():
    d = Dtype("f16_le")
    t = Tibs.from_value(d, 14.5)
    t2 = Tibs.from_f(14.5, 16, ByteOrder.Little)
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
    "value,expected_bits",
    [
        (True, "1"),
        (False, "0"),
        (1, "1"),
        (0, "0"),
    ],
)
def test_from_value_bool(value, expected_bits):
    d = Dtype("bool")
    expected = Tibs.from_bin(expected_bits)

    assert d.pack(value) == expected
    assert Tibs.from_value(d, value) == expected
    assert Mutibs.from_value(d, value) == Mutibs(expected)


@pytest.mark.parametrize("value", [2, -1, 1.0, "true", None, []])
def test_from_value_bool_rejects_non_boollike_values(value):
    with pytest.raises(TypeError, match="bool dtype"):
        Tibs.from_value("bool", value)


@pytest.mark.parametrize(
    "value",
    [
        Tibs.from_bin("101"),
        Mutibs.from_bin("101"),
        "0b101",
        [True, False, True],
    ],
)
def test_from_value_bits(value):
    assert Tibs.from_value("bits3", value) == Tibs.from_bin("101")


def test_from_value_bits_accepts_bytes():
    assert Tibs.from_value("bits8", b"\xff") == Tibs.from_hex("ff")


@pytest.mark.parametrize("value", ["0b10", Tibs.from_bin("1010"), True])
def test_from_value_bits_requires_promoted_dtype_length(value):
    with pytest.raises((TypeError, ValueError)):
        Tibs.from_value("bits3", value)


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("u9", 17),
        ("i8", -2),
        ("f16_le", 14.5),
        ("bool", True),
        ("bits3", Tibs.from_bin("101")),
        ("bytes16", b"ab"),
        ("bin4", "0b1010"),
        ("oct6", "17"),
        ("hex8", "0f"),
    ],
)
def test_dtype_pack_matches_from_value(dtype, value):
    d = Dtype(dtype)
    assert d.pack(value) == Tibs.from_value(d, value)


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("u9", 17),
        ("i8", -2),
        ("f16_le", 14.5),
        ("bool", True),
        ("bits3", Tibs.from_bin("101")),
        ("bytes16", b"ab"),
        ("bin4", "0b1010"),
        ("oct6", "17"),
        ("hex8", "0f"),
    ],
)
def test_dtype_unpack_matches_to_value(dtype, value):
    d = Dtype(dtype)
    bits = d.pack(value)
    assert d.unpack(bits) == bits.to_value(d)


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


@pytest.mark.parametrize(
    "bits,expected",
    [
        ("0b1", True),
        ("0b0", False),
    ],
)
def test_to_value_bool(bits, expected):
    d = Dtype("bool")
    assert d.unpack(bits) is expected
    assert Tibs(bits).to_value(d) is expected


def test_to_value_bits_returns_tibs():
    value = Tibs.from_bin("101")
    decoded = value.to_value("bits3")

    assert decoded == value
    assert type(decoded) is Tibs


def test_mutibs_to_value_bits_returns_immutable_tibs_snapshot():
    m = Mutibs.from_bin("101")
    decoded = m.to_value("bits3")
    m[0] = False

    assert decoded == Tibs.from_bin("101")
    assert type(decoded) is Tibs


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


def test_from_values_bool():
    d = Dtype("bool")
    values = [True, 0, 1, False]

    assert Tibs.from_values(d, values) == Tibs.from_bin("1010")
    assert Mutibs.from_values(d, values) == Mutibs.from_bin("1010")


def test_from_values_bits_validates_each_item_length():
    assert Tibs.from_values("bits3", ["0b101", Mutibs.from_bin("010")]) == Tibs.from_bin("101010")

    with pytest.raises(ValueError, match="Dtype length"):
        Tibs.from_values("bits3", ["0b101", "0b10"])


def test_dtype_pack_values_matches_from_values():
    d = Dtype("u8")
    assert d.pack_values([1, 2, 3]) == Tibs.from_values(d, [1, 2, 3])


def test_dtype_pack_values_accepts_empty_iterables_and_generators():
    d = Dtype("hex8")

    assert d.pack_values([]) == Tibs()
    assert d.pack_values(x for x in ["aa", "bb", "cc"]) == Tibs.from_hex("aabbcc")


def test_from_values_propagates_item_errors():
    d = Dtype("u8")
    with pytest.raises(ValueError, match="does not fit"):
        Tibs.from_values(d, [1, 256])


# from_values packs a byte-aligned numeric dtype through a bulk byte path
# rather than one BitVec per value. These check it agrees with from_value,
# which always takes the general path.

BYTE_ALIGNED_SPECS = [
    f"{kind}{length}{suffix}"
    for kind in ("u", "i")
    for length in (8, 16, 24, 32, 40, 64, 72, 128)
    for suffix in ("", "_be", "_le")
] + [
    f"f{length}{suffix}" for length in (16, 32, 64) for suffix in ("", "_be", "_le")
] + [f"bf16{suffix}" for suffix in ("", "_be", "_le")]

FLOAT_PREFIXES = ("f", "bf")


def _sample_values(spec):
    if spec.startswith(FLOAT_PREFIXES):
        return [0.0, -0.0, 1.0, -2.5, 0.125, float("inf"), float("-inf")]
    length = int(spec[1:].split("_")[0])
    if spec.startswith("u"):
        top = (1 << length) - 1
        return [0, 1, top, top - 1, top // 2, top // 3]
    low, high = -(1 << (length - 1)), (1 << (length - 1)) - 1
    return [0, 1, -1, low, high, low + 1, high - 1]


@pytest.mark.parametrize("spec", BYTE_ALIGNED_SPECS)
def test_from_values_bulk_path_matches_from_value(spec):
    values = _sample_values(spec)
    expected = Tibs.from_joined([Tibs.from_value(spec, v) for v in values])

    assert Tibs.from_values(spec, values) == expected
    assert Mutibs.from_values(spec, values) == Mutibs(expected)
    assert Tibs.from_values(spec, iter(values)) == expected
    assert Tibs.from_values(spec, tuple(values)) == expected
    assert Tibs.from_values(spec, []) == Tibs()


@pytest.mark.parametrize("spec", BYTE_ALIGNED_SPECS)
def test_from_values_bulk_path_round_trips(spec):
    values = _sample_values(spec)
    decoded = Tibs.from_values(spec, values).to_values(spec)
    if spec.startswith(FLOAT_PREFIXES):
        # f16, bf16 and f32 round to their own precision, so compare via
        # from_value.
        assert decoded == [Tibs.from_value(spec, v).to_value(spec) for v in values]
    else:
        assert decoded == values


@pytest.mark.parametrize("spec", ["u8", "u16_le", "i32", "u64", "u72", "f32"])
def test_from_values_bulk_path_reports_bad_items_like_from_value(spec):
    for bad in ("nope", None, [1]):
        with pytest.raises(TypeError) as bulk:
            Tibs.from_values(spec, [bad])
        with pytest.raises(TypeError) as single:
            Tibs.from_value(spec, bad)
        assert str(bulk.value) == str(single.value)


@pytest.mark.parametrize("spec", ["u8", "u16", "u16_le", "i16", "i32", "u64", "i64", "u72"])
def test_from_values_bulk_path_reports_range_errors_like_from_value(spec):
    length = int(spec[1:].split("_")[0])
    if spec.startswith("u"):
        out_of_range = [(1 << length), -1]
    else:
        out_of_range = [1 << (length - 1), -(1 << (length - 1)) - 1]

    for value in out_of_range:
        with pytest.raises(ValueError) as bulk:
            Tibs.from_values(spec, [0, value])
        with pytest.raises(ValueError) as single:
            Tibs.from_value(spec, value)
        assert str(bulk.value) == str(single.value)


def test_from_values_bulk_path_accepts_index_objects():
    class Index:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    assert Tibs.from_values("u8", [Index(1), Index(255)]) == Tibs.from_hex("01ff")
    assert Tibs.from_values("u72", [Index(1)]) == Tibs.from_hex("000000000000000001")
    with pytest.raises(ValueError, match="does not fit"):
        Tibs.from_values("u8", [Index(256)])


def test_from_values_leaves_iterable_alone_when_bulk_path_does_not_apply():
    # Which path to take has to be settled before any item is pulled, or a
    # one-shot iterable would lose the items it had already yielded.
    values = (x for x in ["aa", "bb", "cc"])
    assert Tibs.from_values("hex8", values) == Tibs.from_hex("aabbcc")
    assert list(values) == []


# A numeric dtype that isn't a whole number of bytes long takes a third path,
# packing the values end to end through a bit accumulator. These check it too
# agrees with from_value, across the widths where it changes how a value is
# converted: up to 57 bits a value is shifted into place as a u64, and beyond
# that it goes through int.to_bytes.

BIT_LEVEL_SPECS = [
    f"{kind}{length}"
    for kind in ("u", "i")
    for length in (1, 2, 3, 7, 9, 12, 17, 31, 33, 55, 56, 57, 58, 63, 65, 71, 127, 129)
] + ["bool"]


def _sample_bit_level_values(spec):
    if spec == "bool":
        return [True, False, 0, 1, True, True, False]
    length = int(spec[1:])
    if spec.startswith("u"):
        top = (1 << length) - 1
        return [0, 1, top, top - 1, top // 2, top // 3]
    low, high = -(1 << (length - 1)), (1 << (length - 1)) - 1
    # i1 holds only -1 and 0, so the usual samples have to be filtered.
    candidates = [0, 1, -1, low, high, low + 1, high - 1]
    return [value for value in candidates if low <= value <= high]


@pytest.mark.parametrize("spec", BIT_LEVEL_SPECS)
def test_from_values_bit_path_matches_from_value(spec):
    values = _sample_bit_level_values(spec)
    expected = Tibs.from_joined([Tibs.from_value(spec, v) for v in values])

    assert Tibs.from_values(spec, values) == expected
    assert Mutibs.from_values(spec, values) == Mutibs(expected)
    assert Tibs.from_values(spec, iter(values)) == expected
    assert Tibs.from_values(spec, tuple(values)) == expected
    assert Tibs.from_values(spec, []) == Tibs()


@pytest.mark.parametrize("spec", BIT_LEVEL_SPECS)
def test_from_values_bit_path_round_trips(spec):
    values = _sample_bit_level_values(spec)
    packed = Tibs.from_values(spec, values)

    if spec == "bool":
        assert len(packed) == len(values)
        assert packed.to_values(spec) == [bool(v) for v in values]
    else:
        assert len(packed) == len(values) * int(spec[1:])
        assert packed.to_values(spec) == values


@pytest.mark.parametrize("spec", ["u1", "u12", "i12", "u57", "i57", "u65", "bool"])
def test_from_values_bit_path_reports_bad_items_like_from_value(spec):
    for bad in ("nope", None, [1]):
        with pytest.raises(TypeError) as bulk:
            Tibs.from_values(spec, [bad])
        with pytest.raises(TypeError) as single:
            Tibs.from_value(spec, bad)
        assert str(bulk.value) == str(single.value)


@pytest.mark.parametrize("spec", ["u1", "u3", "u12", "i12", "u57", "i57", "u58", "u65", "i129"])
def test_from_values_bit_path_reports_range_errors_like_from_value(spec):
    length = int(spec[1:])
    if spec.startswith("u"):
        out_of_range = [(1 << length), -1]
    else:
        out_of_range = [1 << (length - 1), -(1 << (length - 1)) - 1]

    for value in out_of_range:
        with pytest.raises(ValueError) as bulk:
            Tibs.from_values(spec, [0, value])
        with pytest.raises(ValueError) as single:
            Tibs.from_value(spec, value)
        assert str(bulk.value) == str(single.value)


def test_from_values_bit_path_accepts_index_objects():
    class Index:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    assert Tibs.from_values("u12", [Index(1), Index(0xFFF)]) == Tibs.from_hex("001fff")
    assert Tibs.from_values("u65", [Index(1)]) == Tibs.from_bin("0" * 64 + "1")
    with pytest.raises(ValueError, match="does not fit"):
        Tibs.from_values("u12", [Index(0x1000)])


def test_from_values_bit_path_carries_across_byte_boundaries():
    # Values that don't divide into bytes have to run straight on from each
    # other, with the last one padded out only as far as the next byte.
    assert Tibs.from_values("u4", [0xA, 0xB, 0xC]) == Tibs.from_bin("101010111100")
    assert Tibs.from_values("u12", [0xABC, 0xDEF]) == Tibs.from_hex("abcdef")
    assert Tibs.from_values("u3", [0b101] * 8) == Tibs.from_bin("101" * 8)
    assert Tibs.from_values("u1", [1, 0, 1]) == Tibs.from_bin("101")
    assert Tibs.from_values("u9", [0x1FF, 0]) == Tibs.from_bin("1" * 9 + "0" * 9)


def test_to_values_reuses_its_window_without_aliasing():
    # to_values walks the sequence with a single moving window, so container
    # dtypes have to come back as independent objects.
    bits = Tibs.from_hex("01020304")
    assert bits.to_values("bits8") == [Tibs.from_hex(h) for h in ("01", "02", "03", "04")]
    assert bits.to_values("bytes8") == [b"\x01", b"\x02", b"\x03", b"\x04"]
    assert bits.to_values("hex8") == ["01", "02", "03", "04"]


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


def test_to_values_iter_bool():
    assert list(Tibs.from_bin("1010").to_values_iter("bool")) == [True, False, True, False]


def test_to_values_bits():
    t = Tibs.from_bin("101010")
    values = t.to_values("bits3")

    assert values == [Tibs.from_bin("101"), Tibs.from_bin("010")]
    assert all(type(value) is Tibs for value in values)


def test_to_values_iter_bits():
    t = Tibs.from_bin("101010")
    values = list(t.to_values_iter("bits3"))

    assert values == [Tibs.from_bin("101"), Tibs.from_bin("010")]
    assert all(type(value) is Tibs for value in values)


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


def test_dtype_unpack_values_matches_to_values():
    d = Dtype("u8")
    bits = Tibs.from_values(d, [1, 2, 3])

    assert d.unpack_values(bits) == bits.to_values(d)
    assert d.unpack_values(bits, 8, 24) == [2, 3]


def test_dtype_unpack_values_iter_accepts_promoted_inputs():
    d = Dtype("u8")

    assert list(d.unpack_values_iter("0x010203")) == [1, 2, 3]
    assert list(d.unpack_values_iter(b"\x01\x02\x03")) == [1, 2, 3]
    assert list(d.unpack_values_iter([0, 0, 0, 0, 0, 0, 0, 1])) == [1]


def test_dtype_unpack_values_iter_snapshots_mutibs():
    d = Dtype("u8")
    bits = Mutibs.from_bytes(b"\x01\x02")
    values = d.unpack_values_iter(bits)

    bits.bytes = b"\xff\xff"

    assert list(values) == [1, 2]


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


def test_mutibs_to_values_bits_returns_immutable_tibs_snapshots():
    m = Mutibs.from_bin("101010")
    values = m.to_values("bits3")
    m[0] = False

    assert values == [Tibs.from_bin("101"), Tibs.from_bin("010")]
    assert all(type(value) is Tibs for value in values)


def test_dtype_string_errors_are_reported_by_value_methods():
    with pytest.raises(ValueError, match="Cannot parse Dtype spec"):
        Tibs.from_value("unknown8", 1)
    with pytest.raises(TypeError, match="dtype must be a Dtype instance"):
        Tibs.from_value(object(), 1)


def test_dtype_pack_and_unpack_error_consistency():
    d = Dtype("u8")

    with pytest.raises(ValueError, match="does not fit"):
        d.pack(256)
    with pytest.raises(ValueError, match="Dtype length"):
        Dtype("hex8").pack("f")
    with pytest.raises(ValueError, match="dtype with length 8 bits"):
        d.unpack("0b101")
    with pytest.raises(ValueError, match="not a multiple"):
        d.unpack_values("0b101")
    with pytest.raises(ValueError, match="not a multiple"):
        list(d.unpack_values_iter("0b101"))
    with pytest.raises(TypeError, match="Cannot promote object"):
        d.unpack(object())


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


SPREAD_BYTES = bytes((index * 37 + 11) % 256 for index in range(64))


@pytest.mark.parametrize("dtype", ["u16", "i16", "u24", "u32", "i32", "f32", "u64", "i64", "f64"])
@pytest.mark.parametrize("offset", range(8))
def test_to_values_from_an_unaligned_start(dtype, offset):
    source = Tibs.from_bytes(SPREAD_BYTES)
    length = Dtype(dtype).length
    values = Dtype(dtype).unpack_values(source[: len(source) // length * length])
    shifted = Tibs.from_zeros(offset) + Dtype(dtype).pack_values(values)
    assert shifted[offset:].to_values(dtype) == values
    assert shifted.to_value(dtype, offset, offset + Dtype(dtype).length) == values[0]


@pytest.mark.parametrize("dtype", ["u16_le", "i32_le", "f64_le"])
@pytest.mark.parametrize("offset", [0, 3, 7])
def test_to_values_little_endian_from_an_unaligned_start(dtype, offset):
    source = Tibs.from_bytes(SPREAD_BYTES)
    length = Dtype(dtype).length
    values = Dtype(dtype).unpack_values(source[: len(source) // length * length])
    shifted = Tibs.from_ones(offset) + Dtype(dtype).pack_values(values)
    assert shifted[offset:].to_values(dtype) == values


def test_to_values_when_the_storage_starts_part_way_into_a_byte():
    a = Mutibs.from_bytes(b"\x01\x02\x03\x04\x05\x06\x07\x08")
    del a[:3]
    assert a.to_values("u16", 0, 48) == Tibs(a)[:48].to_values("u16")
    assert a.to_values("i16", 0, 48) == Tibs(a)[:48].to_values("i16")
    assert a.to_values("f32", 0, 32) == Tibs(a)[:32].to_values("f32")


def test_values_above_the_signed_range_round_trip():
    values = [0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]
    assert Tibs.from_values("u64", values).to_values("u64") == values
    with pytest.raises(ValueError):
        Tibs.from_values("u64", [1 << 64])
    with pytest.raises(ValueError):
        Tibs.from_values("u64", [-1])


@pytest.mark.parametrize(
    "iterable",
    [
        [1, 2, 3],
        (1, 2, 3),
        iter([1, 2, 3]),
        (value for value in [1, 2, 3]),
        range(1, 4),
    ],
)
def test_from_values_accepts_any_iterable(iterable):
    assert Tibs.from_values("u8", iterable) == Tibs("0x010203")


def test_from_values_uses_a_sequence_subclass_iterator():
    class Backwards(list):
        def __iter__(self):
            return reversed([list.__getitem__(self, i) for i in range(len(self))])

    class Fixed(tuple):
        def __iter__(self):
            return iter([9, 9])

    assert Tibs.from_values("u8", Backwards([1, 2, 3])) == Tibs("0x030201")
    assert Tibs.from_values("u8", Fixed((1, 2))) == Tibs("0x0909")


@pytest.mark.parametrize("dtype", ["u8", "u12", "bool"])
def test_from_values_accepts_bools_and_index_objects(dtype):
    class One:
        def __index__(self):
            return 1

    assert Tibs.from_values(dtype, [One(), True, 1]) == Tibs.from_values(dtype, [1, 1, 1])
    assert Tibs.from_values(dtype, [0, False]) == Tibs.from_values(dtype, [0, 0])


@pytest.mark.parametrize("dtype", ["u8", "u12"])
def test_from_values_survives_a_list_edited_while_it_is_read(dtype):
    # Converting a value can run Python code, and that code can reach the list
    # being packed. Whatever it does, packing has to stop at the end of it.
    shrinking = [1, None, 3, 4, 5]

    class Shrink:
        def __index__(self):
            del shrinking[2:]
            return 2

    shrinking[1] = Shrink()
    assert Tibs.from_values(dtype, shrinking) == Tibs.from_values(dtype, [1, 2])

    growing = [None, 2]

    class Grow:
        def __index__(self):
            growing.append(3)
            return 1

    growing[0] = Grow()
    assert Tibs.from_values(dtype, growing) == Tibs.from_values(dtype, [1, 2])


# Wrong dtype specs mostly come from a handful of habits: a tuple written
# without its parentheses, an array written with a comma, or a kind spelled the
# way numpy, struct or C spells it. Each of those should say what the spec
# should have been rather than only where parsing stopped.
@pytest.mark.parametrize(
    ("spec", "suggestion"),
    [
        ("u12, u12", "(u12, u12)"),
        ("u12,u12", "(u12, u12)"),
        (" u8 , bool , i4 ", "(u8, bool, i4)"),
        ("u8 u8", "(u8, u8)"),
        ("u8; 4", "[u8; 4]"),
        ("[u12, u12]", "(u12, u12)"),
        ("[u8, 4]", "[u8; 4]"),
        ("[u8 4]", "[u8; 4]"),
        ("(u8, [u4, 2])", "(u8, [u4; 2])"),
        ("uint12", "u12"),
        ("int8", "i8"),
        ("float32", "f32"),
        ("uint16_le", "u16_le"),
        ("double", "f64"),
        ("<u4", "u32_le"),
        (">i2", "i16_be"),
        ("u16le", "u16_le"),
        ("bool8", "[bool; 8]"),
        ("u1_000", "u1000"),
    ],
)
def test_wrong_specs_suggest_the_right_one(spec, suggestion):
    with pytest.raises(ValueError, match=re.escape(f"'{suggestion}'")):
        Dtype(spec)
    Dtype(suggestion)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        # No suggestion is possible for these, so the message has to be enough
        # on its own.
        ("", "empty spec"),
        ("   ", "empty spec"),
        ("12", "expected a kind"),
        ("8u", "expected a kind"),
        ("u", "expected a bit length after 'u'"),
        ("bytes", "expected a bit length after 'bytes'"),
        ("u12.5", "expected a whole number of bits after 'u', but found '12.5'"),
        ("hex3.5", "expected a whole number of bits after 'hex', but found '3.5'"),
        ("u8_ne", "only '_le' and '_be' are supported"),
        ("=u4", "not a leading character"),
        ("(u8)", "needs a trailing comma"),
        ("(u8 u8)", "expected ',' between the fields"),
        ("(u8, u8", "unterminated tuple dtype"),
        ("[u8; 4", "unterminated array dtype"),
    ],
)
def test_wrong_specs_explain_themselves(spec, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        Dtype(spec)


# The kinds that fix their own bit length, and so describe a complete dtype
# without one being written alongside: the fixed formats, plus Bool and BFloat.
INTRINSIC_LENGTH_KINDS = [
    (DtypeKind.Bool, 1, "bool"),
    (DtypeKind.BFloat, 16, "bf16"),
    (DtypeKind.Binary8P3, 8, "binary8p3"),
    (DtypeKind.Binary8P4, 8, "binary8p4"),
    (DtypeKind.OcpE4M3Saturate, 8, "ocp_e4m3_saturate"),
    (DtypeKind.OcpE4M3Overflow, 8, "ocp_e4m3_overflow"),
    (DtypeKind.OcpE5M2Saturate, 8, "ocp_e5m2_saturate"),
    (DtypeKind.OcpE5M2Overflow, 8, "ocp_e5m2_overflow"),
    (DtypeKind.OcpE3M2, 6, "ocp_e3m2"),
    (DtypeKind.OcpE2M3, 6, "ocp_e2m3"),
    (DtypeKind.OcpE2M1, 4, "ocp_e2m1"),
    (DtypeKind.OcpE8M0, 8, "ocp_e8m0"),
    (DtypeKind.OcpInt8, 8, "ocp_int8"),
]

# The kinds that are a family of widths, so a length is always needed.
SIZED_KINDS = [
    DtypeKind.Uint,
    DtypeKind.Int,
    DtypeKind.Float,
    DtypeKind.Bits,
    DtypeKind.Bin,
    DtypeKind.Oct,
    DtypeKind.Hex,
    DtypeKind.Bytes,
]


@pytest.mark.parametrize("kind, length, spec", INTRINSIC_LENGTH_KINDS)
def test_from_params_infers_an_intrinsic_length(kind, length, spec):
    d = DtypeSingle.from_params(kind)
    assert d.kind is kind
    assert d.length == length
    assert d == Dtype(spec)
    # Passing the length explicitly stays valid and means the same thing.
    assert DtypeSingle.from_params(kind, length) == d


@pytest.mark.parametrize("kind", SIZED_KINDS)
def test_from_params_still_requires_a_length_for_sized_kinds(kind):
    with pytest.raises(ValueError, match="does not determine a length"):
        DtypeSingle.from_params(kind)


@pytest.mark.parametrize("kind", SIZED_KINDS)
def test_missing_length_message_suggests_a_dtype_that_parses(kind):
    with pytest.raises(ValueError) as exc:
        DtypeSingle.from_params(kind)
    # The suggestion has to satisfy that kind's own length rule -- Float admits
    # only 16/32/64, Bytes only multiples of 8 -- so it is not one width with
    # different prefixes.
    suggestion = re.search(r"For example, '([^']+)'", str(exc.value)).group(1)
    assert Dtype(suggestion).kind is kind


@pytest.mark.parametrize("kind, length, spec", INTRINSIC_LENGTH_KINDS)
def test_a_bare_kind_is_accepted_as_a_dtype(kind, length, spec):
    assert Dtype(spec) == DtypeSingle.from_params(kind)

    # A bare kind must mean the same as its spec on every path taking a dtype,
    # not just on to_value.
    zeros = Tibs.from_zeros(length)
    value = zeros.to_value(spec)
    assert zeros.to_value(kind) == value

    t = Tibs.from_values(kind, [value] * 3)
    assert t == Tibs.from_values(spec, [value] * 3)
    assert t.to_values(kind) == t.to_values(spec)
    assert Tibs.from_value(kind, value) == Tibs.from_value(spec, value)
    assert Mutibs.from_value(kind, value).to_value(kind) == value


@pytest.mark.parametrize("kind", SIZED_KINDS)
def test_a_bare_sized_kind_is_rejected_as_a_dtype(kind):
    with pytest.raises(ValueError, match="does not determine a length"):
        Tibs.from_zeros(8).to_value(kind)


def test_dtype_type_error_lists_what_is_accepted():
    with pytest.raises(TypeError, match="Dtype instance, a DtypeKind with a fixed length, or a"):
        Tibs.from_zeros(8).to_value(object())


def test_repeated_specs_parse_to_equal_but_independent_dtypes():
    # Specs are cached internally; the cache must not let a caller observe one
    # dtype object where two equal ones are expected, nor confuse spellings.
    a, b = Dtype("u16"), Dtype("u16")
    assert a == b and hash(a) == hash(b)
    assert Dtype("U16 ") == a
    assert Dtype("u16_le") != a
    # A cached hit must not survive as the wrong dtype for a different spec.
    assert Dtype("u13") != a and Dtype("u13").length == 13


def test_invalid_specs_keep_raising_when_repeated():
    for _ in range(3):
        with pytest.raises(ValueError, match="Cannot parse Dtype spec"):
            Dtype("nonsense9")
