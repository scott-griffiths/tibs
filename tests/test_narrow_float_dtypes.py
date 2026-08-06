#!/usr/bin/env python
"""Format and integration tests for P3109/OCP narrow numeric dtypes."""

from __future__ import annotations

from functools import lru_cache
import math
import struct

import pytest

from tibs import Dtype, Mutibs, Tibs
from tests.narrow_float_reference import (
    FLOAT_FORMAT_NAMES,
    FORMATS,
    decode,
    encode,
    positive_finite_codes,
    positive_finite_values,
    same_float,
    terminal_midpoint,
    terminal_virtual_code,
)


def _raw(spec: str, value: float) -> int:
    return Tibs.from_value(spec, value).u


def _assert_same(actual: float, expected: float, *, context: str = "") -> None:
    assert same_float(actual, expected), (
        f"{context}: expected {expected!r} "
        f"({math.copysign(1.0, expected) if expected == 0 else 'nonzero'}), got {actual!r}"
    )


@pytest.mark.parametrize(
    "spec",
    [
        "ocp_e4m3",
        "ocp_e5m2",
        "ieee8",
    ],
)
def test_noncanonical_and_ambiguous_spellings_are_rejected(spec):
    with pytest.raises(ValueError):
        Dtype(spec)


KNOWN_CODES = [
    ("binary8p3", 0x00, 0.0),
    ("binary8p3", 0x01, 2.0**-17),
    ("binary8p3", 0x04, 2.0**-15),
    ("binary8p3", 0x7E, 49152.0),
    ("binary8p3", 0x7F, math.inf),
    ("binary8p3", 0x80, math.nan),
    ("binary8p3", 0xFF, -math.inf),
    ("binary8p4", 0x01, 2.0**-10),
    ("binary8p4", 0x08, 2.0**-7),
    ("binary8p4", 0x7E, 224.0),
    ("binary8p4", 0x7F, math.inf),
    ("binary8p4", 0x80, math.nan),
    ("binary8p4", 0xFF, -math.inf),
    ("ocp_e4m3_saturate", 0x01, 2.0**-9),
    ("ocp_e4m3_saturate", 0x08, 2.0**-6),
    ("ocp_e4m3_saturate", 0x7E, 448.0),
    ("ocp_e4m3_saturate", 0x7F, math.nan),
    ("ocp_e4m3_saturate", 0x80, -0.0),
    ("ocp_e4m3_saturate", 0xFF, math.nan),
    ("ocp_e5m2_saturate", 0x01, 2.0**-16),
    ("ocp_e5m2_saturate", 0x04, 2.0**-14),
    ("ocp_e5m2_saturate", 0x7B, 57344.0),
    ("ocp_e5m2_saturate", 0x7C, math.inf),
    ("ocp_e5m2_saturate", 0x7D, math.nan),
    ("ocp_e5m2_saturate", 0x80, -0.0),
    ("ocp_e5m2_saturate", 0xFC, -math.inf),
    ("ocp_e3m2", 0x01, 0.0625),
    ("ocp_e3m2", 0x04, 0.25),
    ("ocp_e3m2", 0x1F, 28.0),
    ("ocp_e2m3", 0x01, 0.125),
    ("ocp_e2m3", 0x08, 1.0),
    ("ocp_e2m3", 0x1F, 7.5),
    ("ocp_e2m1", 0x01, 0.5),
    ("ocp_e2m1", 0x02, 1.0),
    ("ocp_e2m1", 0x07, 6.0),
    ("ocp_e8m0", 0x00, 2.0**-127),
    ("ocp_e8m0", 0x7F, 1.0),
    ("ocp_e8m0", 0xFE, 2.0**127),
    ("ocp_e8m0", 0xFF, math.nan),
    ("ocp_int8", 0x00, 0.0),
    ("ocp_int8", 0x01, 1.0 / 64.0),
    ("ocp_int8", 0x7F, 127.0 / 64.0),
    ("ocp_int8", 0x80, -2.0),
    ("ocp_int8", 0xFF, -1.0 / 64.0),
]


@pytest.mark.parametrize("spec,raw,expected", KNOWN_CODES)
def test_known_codes(spec, raw, expected):
    width = FORMATS[spec].width
    actual = Tibs.from_u(raw, width).to_value(spec)
    _assert_same(actual, expected, context=f"{spec} raw 0x{raw:02x}")


@pytest.mark.parametrize("spec", FORMATS)
def test_every_raw_code_decodes_like_the_independent_reference(spec):
    width = FORMATS[spec].width
    for raw in range(1 << width):
        actual = Tibs.from_u(raw, width).to_value(spec)
        _assert_same(actual, decode(spec, raw), context=f"{spec} raw {raw:#x}")


@pytest.mark.parametrize("spec", FORMATS)
def test_every_raw_code_reencodes_to_its_canonical_code(spec):
    width = FORMATS[spec].width
    for raw in range(1 << width):
        value = decode(spec, raw)
        expected = encode(spec, value)
        actual = _raw(spec, value)
        assert actual == expected, (
            f"{spec} raw {raw:#x} decoded as {value!r}: "
            f"expected canonical {expected:#x}, got {actual:#x}"
        )


def _signed_code(spec: str, positive_code: int, negative: bool) -> int:
    fmt = FORMATS[spec]
    if not negative:
        return positive_code
    if positive_code == 0 and not fmt.signed_zero:
        return 0
    return positive_code | fmt.sign_bit


@pytest.mark.parametrize("spec", FLOAT_FORMAT_NAMES)
def test_every_adjacent_finite_pair_rounds_at_the_exact_midpoint(spec):
    codes = positive_finite_codes(spec)
    values = positive_finite_values(spec)
    for lower_code, upper_code, lower, upper in zip(codes, codes[1:], values, values[1:]):
        midpoint = (lower + upper) / 2.0
        tie_code = lower_code if lower_code & 1 == 0 else upper_code

        cases = [
            (math.nextafter(midpoint, -math.inf), lower_code),
            (midpoint, tie_code),
            (math.nextafter(midpoint, math.inf), upper_code),
            (math.nextafter(-midpoint, -math.inf), _signed_code(spec, upper_code, True)),
            (-midpoint, _signed_code(spec, tie_code, True)),
            (math.nextafter(-midpoint, math.inf), _signed_code(spec, lower_code, True)),
        ]
        for value, expected in cases:
            assert _raw(spec, value) == expected, (
                f"{spec} between raw {lower_code:#x} ({lower}) and "
                f"{upper_code:#x} ({upper}) at input {value!r}"
            )


@pytest.mark.parametrize(
    "spec", [name for name in FLOAT_FORMAT_NAMES if terminal_virtual_code(name) is not None]
)
def test_terminal_rounding_boundary_uses_rne_before_special_value_policy(spec):
    fmt = FORMATS[spec]
    midpoint = terminal_midpoint(spec)
    assert midpoint is not None
    max_code = positive_finite_codes(spec)[-1]
    special_code = terminal_virtual_code(spec)
    assert special_code is not None
    positive_special = 0xFF if fmt.family == "ocp_e4m3" else special_code
    negative_special = 0xFF if fmt.family == "ocp_e4m3" else special_code | fmt.sign_bit
    tie_code = max_code if max_code & 1 == 0 else positive_special
    negative_tie = _signed_code(spec, max_code, True) if max_code & 1 == 0 else negative_special

    assert _raw(spec, math.nextafter(midpoint, -math.inf)) == max_code
    assert _raw(spec, midpoint) == tie_code
    assert _raw(spec, math.nextafter(midpoint, math.inf)) == positive_special
    assert _raw(spec, math.nextafter(-midpoint, math.inf)) == _signed_code(spec, max_code, True)
    assert _raw(spec, -midpoint) == negative_tie
    assert _raw(spec, math.nextafter(-midpoint, -math.inf)) == negative_special


@pytest.mark.parametrize(
    "spec",
    [
        "ocp_e4m3_saturate",
        "ocp_e5m2_saturate",
        "ocp_e3m2",
        "ocp_e2m3",
        "ocp_e2m1",
    ],
)
def test_saturating_formats_clamp_finite_values_and_infinities(spec):
    max_code = positive_finite_codes(spec)[-1]
    sign_bit = FORMATS[spec].sign_bit
    maximum = positive_finite_values(spec)[-1]
    for value in [math.nextafter(maximum, math.inf), maximum * 2.0, 1e300, math.inf]:
        assert _raw(spec, value) == max_code
        assert _raw(spec, -value) == max_code | sign_bit


@pytest.mark.parametrize("spec", FLOAT_FORMAT_NAMES)
def test_zero_encoding_and_sign(spec):
    fmt = FORMATS[spec]
    assert _raw(spec, 0.0) == 0
    assert _raw(spec, -0.0) == (fmt.sign_bit if fmt.signed_zero else 0)


@pytest.mark.parametrize("spec", ["ocp_e3m2", "ocp_e2m3", "ocp_e2m1", "ocp_int8"])
def test_formats_without_nan_reject_it(spec):
    with pytest.raises(ValueError, match="NaN|nan|not representable"):
        Tibs.from_value(spec, math.nan)


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("binary8p3", 0x80),
        ("binary8p4", 0x80),
        ("ocp_e4m3_saturate", 0xFF),
        ("ocp_e4m3_overflow", 0xFF),
        ("ocp_e5m2_saturate", 0xFF),
        ("ocp_e5m2_overflow", 0xFF),
        ("ocp_e8m0", 0xFF),
    ],
)
def test_nan_has_a_deterministic_canonical_code(spec, expected):
    assert _raw(spec, math.nan) == expected


def test_ocp_int8_rounds_every_adjacent_pair_ties_to_even():
    code_values = sorted((decode("ocp_int8", raw), raw) for raw in range(256))
    for (lower, lower_code), (upper, upper_code) in zip(code_values, code_values[1:]):
        midpoint = (lower + upper) / 2.0
        tie_code = lower_code if lower_code & 1 == 0 else upper_code
        assert _raw("ocp_int8", math.nextafter(midpoint, -math.inf)) == lower_code
        assert _raw("ocp_int8", midpoint) == tie_code
        assert _raw("ocp_int8", math.nextafter(midpoint, math.inf)) == upper_code


def test_ocp_int8_saturates_asymmetrically_and_rejects_nan():
    assert _raw("ocp_int8", -1e300) == 0x80
    assert _raw("ocp_int8", -math.inf) == 0x80
    assert _raw("ocp_int8", 1e300) == 0x7F
    assert _raw("ocp_int8", math.inf) == 0x7F
    with pytest.raises(ValueError, match="NaN|nan|not representable"):
        Tibs.from_value("ocp_int8", math.nan)


def test_ocp_e8m0_accepts_only_nan_or_exact_positive_in_range_powers_of_two():
    assert _raw("ocp_e8m0", 2.0**-127) == 0x00
    assert _raw("ocp_e8m0", 1.0) == 0x7F
    assert _raw("ocp_e8m0", 2.0**127) == 0xFE
    assert _raw("ocp_e8m0", math.nan) == 0xFF

    invalid = [
        0.0,
        -0.0,
        -1.0,
        math.inf,
        -math.inf,
        math.nextafter(2.0**-127, 0.0),
        math.nextafter(2.0**127, math.inf),
        0.75,
        1.5,
        3.0,
    ]
    for value in invalid:
        with pytest.raises(ValueError):
            Tibs.from_value("ocp_e8m0", value)


def test_every_e8m0_adjacent_midpoint_is_rejected():
    for raw in range(0xFE):
        midpoint = (decode("ocp_e8m0", raw) + decode("ocp_e8m0", raw + 1)) / 2.0
        with pytest.raises(ValueError):
            Tibs.from_value("ocp_e8m0", midpoint)


@lru_cache(maxsize=1)
def _all_binary16_values() -> tuple[float, ...]:
    return tuple(struct.unpack(">e", raw.to_bytes(2, "big"))[0] for raw in range(1 << 16))


@pytest.mark.parametrize("spec", [name for name in FORMATS if name != "ocp_e8m0"])
def test_every_binary16_input_matches_the_independent_reference(spec):
    values = _all_binary16_values()
    accepted = []
    expected = []
    for value in values:
        try:
            raw = encode(spec, value)
        except ValueError:
            continue
        accepted.append(value)
        expected.append(raw)

    packed = Tibs.from_values(spec, accepted)
    actual = packed.to_values(f"u{FORMATS[spec].width}")
    assert actual == expected


SAMPLE_VALUES = {
    "binary8p3": [0.0, 2.0**-17, 1.0, -2.0, 49152.0, math.inf],
    "binary8p4": [0.0, 2.0**-10, 1.0, -2.0, 224.0, -math.inf],
    "ocp_e4m3_saturate": [0.0, -0.0, 2.0**-9, 1.0, -2.0, 448.0],
    "ocp_e4m3_overflow": [0.0, -0.0, 2.0**-9, 1.0, -2.0, 448.0],
    "ocp_e5m2_saturate": [0.0, -0.0, 2.0**-16, 1.0, -2.0, 57344.0],
    "ocp_e5m2_overflow": [0.0, -0.0, 2.0**-16, 1.0, -2.0, math.inf],
    "ocp_e3m2": [0.0, -0.0, 0.0625, 1.0, -2.0, 28.0],
    "ocp_e2m3": [0.0, -0.0, 0.125, 1.0, -2.0, 7.5],
    "ocp_e2m1": [0.0, -0.0, 0.5, 1.0, -2.0, 6.0],
    "ocp_e8m0": [2.0**-127, 0.5, 1.0, 2.0, 2.0**127],
    "ocp_int8": [-2.0, -1.0 / 64.0, 0.0, 1.0 / 64.0, 127.0 / 64.0],
}


@pytest.mark.parametrize("spec", FORMATS)
def test_bulk_pack_unpack_matches_scalar_and_accepts_any_iterable(spec):
    values = SAMPLE_VALUES[spec]
    expected = Tibs.from_joined(Tibs.from_value(spec, value) for value in values)

    assert Tibs.from_values(spec, values) == expected
    assert Tibs.from_values(spec, iter(values)) == expected
    assert Mutibs.from_values(spec, tuple(values)) == Mutibs(expected)
    assert Dtype(spec).pack_values(value for value in values) == expected
    assert Tibs.from_values(spec, []) == Tibs()

    decoded = expected.to_values(spec)
    assert len(decoded) == len(values)
    for index, (actual, value) in enumerate(zip(decoded, values)):
        _assert_same(actual, decode(spec, encode(spec, value)), context=f"{spec} item {index}")
    assert list(expected.to_values_iter(spec)) == decoded
    assert Mutibs(expected).to_values(spec) == decoded


@pytest.mark.parametrize("spec", FORMATS)
@pytest.mark.parametrize("offset", range(8))
def test_bulk_and_scalar_decode_from_every_unaligned_offset(spec, offset):
    values = SAMPLE_VALUES[spec][:4]
    packed = Tibs.from_values(spec, values)
    shifted = Tibs.from_ones(offset) + packed + Tibs.from_zeros(7 - offset)
    end = offset + len(packed)

    decoded = shifted.to_values(spec, offset, end)
    sliced = shifted[offset:end]
    assert sliced.to_values(spec) == decoded
    assert Mutibs(shifted).to_values(spec, offset, end) == decoded
    _assert_same(shifted.to_value(spec, offset, offset + FORMATS[spec].width), decoded[0])
    _assert_same(shifted.view().to_value(spec, offset, offset + FORMATS[spec].width), decoded[0])


def test_subbyte_arrays_and_mixed_compound_dtypes_round_trip():
    array_dtype = Dtype("[ocp_e2m1; 4]")
    array_value = (0.5, -1.0, 3.0, -6.0)
    mixed_dtype = Dtype("(ocp_e2m1, bool, ocp_e3m2, ocp_e8m0, ocp_int8)")
    mixed_value = (-1.5, True, 0.25, 2.0, -0.5)

    array_bits = array_dtype.pack(array_value)
    mixed_bits = mixed_dtype.pack(mixed_value)

    assert len(array_bits) == 16
    assert len(mixed_bits) == 27
    assert array_dtype.unpack(array_bits) == array_value
    assert mixed_dtype.unpack(mixed_bits) == mixed_value
    assert Tibs.from_value(array_dtype, array_value) == array_bits
    assert Mutibs.from_value(mixed_dtype, mixed_value) == Mutibs(mixed_bits)


def test_bytewise_arrays_and_mixed_compound_dtypes_round_trip_unaligned():
    tuple_dtype = Dtype(
        "(binary8p4, ocp_e4m3_overflow, ocp_e8m0, ocp_int8)"
    )
    records = [
        (1.0, 448.0, 0.5, -2.0),
        (-2.0, -1.5, 2.0, 127.0 / 64.0),
    ]
    array_dtype = Dtype("[ocp_e4m3_saturate; 3]")
    array_value = (0.5, -1.0, 448.0)

    one_record = tuple_dtype.pack(records[0])
    packed = tuple_dtype.pack_values(records)
    expected = Tibs.from_joined(tuple_dtype.pack(record) for record in records)
    shifted = Tibs("0b101") + packed

    assert tuple_dtype.unpack(one_record) == records[0]
    assert packed == expected
    assert tuple_dtype.unpack_values(packed) == records
    assert shifted.to_values(tuple_dtype, 3, len(shifted)) == records
    assert Mutibs.from_values(tuple_dtype, records) == Mutibs(packed)
    assert array_dtype.unpack(array_dtype.pack(array_value)) == array_value


def test_repeated_mixed_compound_dtypes_round_trip_unaligned():
    dtype = Dtype("[(ocp_e2m1, ocp_e3m2, bool); 2]")
    values = [
        ((0.5, 0.25, True), (-1.0, -2.0, False)),
        ((3.0, 28.0, False), (-6.0, -0.0625, True)),
    ]

    packed = Tibs.from_values(dtype, values)
    shifted = Tibs("0b101") + packed

    assert packed.to_values(dtype) == values
    assert shifted.to_values(dtype, 3, len(shifted)) == values
    assert dtype.unpack_values(packed) == values
    assert list(dtype.unpack_values_iter(packed)) == values


@pytest.mark.parametrize("spec", FORMATS)
@pytest.mark.parametrize("bad", ["not a number", None, [1.0]])
def test_narrow_dtypes_reject_non_numeric_values(spec, bad):
    with pytest.raises(TypeError):
        Tibs.from_value(spec, bad)


def test_nested_conversion_error_preserves_value_path():
    dtype = Dtype("[(ocp_e2m1, bool); 2]")
    with pytest.raises(ValueError) as exc_info:
        dtype.pack(((0.5, True), (math.nan, False)))
    message = str(exc_info.value)
    assert "1" in message and "0" in message
