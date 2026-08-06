#!/usr/bin/env python

import pytest

from tibs import ByteOrder, Dtype, DtypeKind, DtypeSingle


NARROW_FORMATS = [
    ("p3109_k8p3se", DtypeKind.P3109K8P3SE, 8),
    ("p3109_k8p4se", DtypeKind.P3109K8P4SE, 8),
    ("ocp_e4m3_saturate", DtypeKind.OcpE4M3Saturate, 8),
    ("ocp_e4m3_overflow", DtypeKind.OcpE4M3Overflow, 8),
    ("ocp_e5m2_saturate", DtypeKind.OcpE5M2Saturate, 8),
    ("ocp_e5m2_overflow", DtypeKind.OcpE5M2Overflow, 8),
    ("ocp_e3m2", DtypeKind.OcpE3M2, 6),
    ("ocp_e2m3", DtypeKind.OcpE2M3, 6),
    ("ocp_e2m1", DtypeKind.OcpE2M1, 4),
    ("ocp_e8m0", DtypeKind.OcpE8M0, 8),
    ("ocp_int8", DtypeKind.OcpInt8, 8),
]


def test_existing_dtype_kind_integer_values_remain_stable():
    existing = [
        DtypeKind.Uint,
        DtypeKind.Int,
        DtypeKind.Float,
        DtypeKind.BFloat,
        DtypeKind.Bool,
        DtypeKind.Bits,
        DtypeKind.Bytes,
        DtypeKind.Bin,
        DtypeKind.Oct,
        DtypeKind.Hex,
    ]
    assert [int(kind) for kind in existing] == list(range(10))


@pytest.mark.parametrize(("spec", "kind", "length"), NARROW_FORMATS)
def test_narrow_format_descriptor_round_trips(spec, kind, length):
    parsed = Dtype(spec)
    constructed = DtypeSingle.from_params(kind, length)

    assert type(parsed) is DtypeSingle
    assert parsed.kind is kind
    assert parsed.length == length
    assert parsed.byte_order is ByteOrder.Unspecified
    assert str(parsed) == spec
    assert repr(parsed) == f"DtypeSingle('{spec}')"
    assert constructed == parsed
    assert str(constructed) == spec


@pytest.mark.parametrize(("spec", "kind", "length"), NARROW_FORMATS)
def test_narrow_format_rejects_non_intrinsic_lengths(spec, kind, length):
    for invalid_length in (length - 1, length + 1):
        with pytest.raises(ValueError, match=rf"must have length {length} bits"):
            DtypeSingle.from_params(kind, invalid_length)


@pytest.mark.parametrize(("spec", "kind", "length"), NARROW_FORMATS)
def test_narrow_format_rejects_byte_order(spec, kind, length):
    for byte_order in (ByteOrder.Big, ByteOrder.Little):
        with pytest.raises(ValueError, match="byte order cannot be specified"):
            DtypeSingle.from_params(kind, length, byte_order)

    for suffix in ("_be", "_le"):
        with pytest.raises(ValueError, match="byte order cannot be specified"):
            Dtype(spec + suffix)


@pytest.mark.parametrize(
    "alias",
    [
        "p3binary",
        "p4binary",
        "e4m3mxfp_saturate",
        "e4m3mxfp_overflow",
        "e5m2mxfp_saturate",
        "e5m2mxfp_overflow",
        "e3m2mxfp",
        "e2m3mxfp",
        "e2m1mxfp",
        "e8m0mxfp",
        "mxint",
        # Released Bitstring 4.4 spellings, before the policy names split.
        "e4m3mxfp",
        "e5m2mxfp",
    ],
)
def test_bitstring_aliases_are_not_accepted(alias):
    with pytest.raises(ValueError):
        Dtype(alias)
