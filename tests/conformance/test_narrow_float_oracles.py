#!/usr/bin/env python

"""Live conformance checks for Tibs' low-precision numeric dtypes.

These tests intentionally use optional, exactly pinned dependencies.  The
normal test environment skips this module at collection time. Install the
``conformance`` extra and the reviewed Bitstring 5 source snapshot with
``--no-deps``, as the dedicated workflow does, to run the live comparisons
without replacing the Tibs checkout.

``gfloat`` is the format/rounding oracle.  ``ml_dtypes`` supplies an
independent implementation for the OCP floating formats it supports.
Bitstring 5 is a compatibility oracle, but only for binary16 inputs because
its conversion lookup first narrows arbitrary Python floats to binary16.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import metadata
import math
import os
from pathlib import Path
import struct
from typing import Any

import pytest


# Importing gfloat and ml_dtypes also requires NumPy.  A normal Tibs
# development environment has none of these packages, so missing any oracle
# skips the complete module rather than leaving a misleading partial run.
np = pytest.importorskip("numpy", reason="the narrow-float conformance extra is not installed")
gfloat = pytest.importorskip(
    "gfloat", reason="the narrow-float conformance extra is not installed"
)
ml_dtypes = pytest.importorskip(
    "ml_dtypes", reason="the narrow-float conformance extra is not installed"
)
bitstring = pytest.importorskip(
    "bitstring", reason="the narrow-float conformance extra is not installed"
)

from gfloat.formats import (  # noqa: E402 - imports follow module-level skips
    format_info_ocp_e2m1,
    format_info_ocp_e2m3,
    format_info_ocp_e3m2,
    format_info_ocp_e4m3,
    format_info_ocp_e5m2,
    format_info_ocp_e8m0,
    format_info_ocp_int8,
    format_info_p3109,
)
from tibs import Tibs  # noqa: E402 - imports follow module-level skips


ORACLE_VERSIONS = {
    "gfloat": "0.5.2",
    "ml-dtypes": "0.5.4",
}

BITSTRING_VERSION = "5.0.0_beta1"
BITSTRING_COMMIT = "c336d1e6a6ebf7ccc364840cb12949a7b012d985"

# The P3109 formats are provisional.  This is the public-draft snapshot whose
# K8P3SE and K8P4SE encodings Tibs freezes under the names below.
P3109_PUBLIC_COMMIT = "aa9d236d7a31b38fbe43b703a0bfdfc3d8be5d45"

P3109_VALUE_TABLES = (
    (
        "p3109_k8p3se",
        Path("Value Tables/Hexadecimal/K8/P3/signed/Binary8p3se.csv"),
    ),
    (
        "p3109_k8p4se",
        Path("Value Tables/Hexadecimal/K8/P4/signed/Binary8p4se.csv"),
    ),
)


@dataclass(frozen=True)
class FormatCase:
    tibs_spec: str
    bits: int
    gfloat_info: Any
    bitstring_spec: str
    gfloat_saturate: bool = False
    ml_dtype_name: str | None = None
    canonical_nan: int | None = None
    exact_only: bool = False


P3109_K8P3SE = format_info_p3109(8, 3)
P3109_K8P4SE = format_info_p3109(8, 4)


FORMAT_CASES = (
    FormatCase(
        "p3109_k8p3se",
        8,
        P3109_K8P3SE,
        "p3binary",
        canonical_nan=0x80,
    ),
    FormatCase(
        "p3109_k8p4se",
        8,
        P3109_K8P4SE,
        "p4binary",
        canonical_nan=0x80,
    ),
    FormatCase(
        "ocp_e4m3_saturate",
        8,
        format_info_ocp_e4m3,
        "e4m3mxfp_saturate",
        gfloat_saturate=True,
        ml_dtype_name="float8_e4m3fn",
        canonical_nan=0xFF,
    ),
    FormatCase(
        "ocp_e4m3_overflow",
        8,
        format_info_ocp_e4m3,
        "e4m3mxfp_overflow",
        canonical_nan=0xFF,
    ),
    FormatCase(
        "ocp_e5m2_saturate",
        8,
        format_info_ocp_e5m2,
        "e5m2mxfp_saturate",
        gfloat_saturate=True,
        ml_dtype_name="float8_e5m2",
        canonical_nan=0xFF,
    ),
    FormatCase(
        "ocp_e5m2_overflow",
        8,
        format_info_ocp_e5m2,
        "e5m2mxfp_overflow",
        canonical_nan=0xFF,
    ),
    FormatCase(
        "ocp_e3m2",
        6,
        format_info_ocp_e3m2,
        "e3m2mxfp",
        gfloat_saturate=True,
        ml_dtype_name="float6_e3m2fn",
    ),
    FormatCase(
        "ocp_e2m3",
        6,
        format_info_ocp_e2m3,
        "e2m3mxfp",
        gfloat_saturate=True,
        ml_dtype_name="float6_e2m3fn",
    ),
    FormatCase(
        "ocp_e2m1",
        4,
        format_info_ocp_e2m1,
        "e2m1mxfp",
        gfloat_saturate=True,
        ml_dtype_name="float4_e2m1fn",
    ),
    FormatCase(
        "ocp_e8m0",
        8,
        format_info_ocp_e8m0,
        "e8m0mxfp",
        ml_dtype_name="float8_e8m0fnu",
        canonical_nan=0xFF,
        exact_only=True,
    ),
    FormatCase(
        "ocp_int8",
        8,
        format_info_ocp_int8,
        "mxint",
        gfloat_saturate=True,
    ),
)

GFLOAT_BINARY16_CASES = tuple(case for case in FORMAT_CASES if not case.exact_only)
ML_DTYPES_CASES = tuple(case for case in FORMAT_CASES if case.ml_dtype_name is not None)
ML_DTYPES_ROUNDING_CASES = tuple(case for case in ML_DTYPES_CASES if not case.exact_only)

BINARY16_VALUES = tuple(
    struct.unpack(">e", struct.pack(">H", source_bits))[0]
    for source_bits in range(1 << 16)
)

VALUE_ERROR = object()


def _case_id(case: FormatCase) -> str:
    return case.tibs_spec


def _float_signature(value: Any) -> tuple[str, int | float | None]:
    value = float(value)
    if math.isnan(value):
        return ("nan", None)
    if math.isinf(value):
        return ("infinity", -1 if value < 0.0 else 1)
    if value == 0.0:
        return ("zero", -1 if math.copysign(1.0, value) < 0.0 else 1)
    return ("finite", value)


def _assert_same_float(actual: Any, expected: Any, *, context: str) -> None:
    actual_signature = _float_signature(actual)
    expected_signature = _float_signature(expected)
    assert actual_signature == expected_signature, (
        f"{context}: actual {actual!r} ({actual_signature}) != "
        f"expected {expected!r} ({expected_signature})"
    )


def _parse_p3109_value(value: str, *, path: Path, codepoint: str) -> float:
    special_values = {
        "Inf": math.inf,
        "-Inf": -math.inf,
        "NaN": math.nan,
    }
    if value in special_values:
        return special_values[value]

    try:
        return float.fromhex(value)
    except ValueError as error:
        pytest.fail(
            f"{path}: invalid value {value!r} for codepoint {codepoint!r}: "
            f"{error}"
        )


def _read_p3109_value_table(path: Path) -> list[float]:
    assert path.is_file(), f"P3109 value table does not exist: {path}"
    with path.open(encoding="ascii", newline="") as table_file:
        rows = list(csv.reader(table_file, strict=True))

    assert rows, f"P3109 value table is empty: {path}"
    assert rows[0] == ["codepoint", "value", "subnormal"], (
        f"{path}: unexpected header {rows[0]!r}"
    )
    assert len(rows) == 257, (
        f"{path}: expected a header and 256 values, found {len(rows)} rows"
    )

    values = []
    for expected_raw, row in enumerate(rows[1:]):
        assert len(row) == 3, (
            f"{path}: codepoint index {expected_raw} has {len(row)} columns: {row!r}"
        )
        codepoint, value, subnormal = row
        try:
            actual_raw = int(codepoint, 16)
        except ValueError as error:
            pytest.fail(f"{path}: invalid codepoint {codepoint!r}: {error}")
        assert actual_raw == expected_raw, (
            f"{path}: expected sequential codepoint 0x{expected_raw:02x}, "
            f"found {codepoint!r}"
        )
        assert subnormal.strip() in {"", "*"}, (
            f"{path}: invalid subnormal marker {subnormal!r} for {codepoint}"
        )
        values.append(_parse_p3109_value(value, path=path, codepoint=codepoint))

    return values


def _tibs_decode(case: FormatCase, raw: int) -> float:
    return Tibs.from_value(f"u{case.bits}", raw).to_value(case.tibs_spec)


def _tibs_encode(case: FormatCase, value: float) -> int:
    return Tibs.from_value(case.tibs_spec, value).to_value(f"u{case.bits}")


def _tibs_encode_outcome(case: FormatCase, value: float) -> int | object:
    try:
        return _tibs_encode(case, value)
    except ValueError:
        return VALUE_ERROR


def _gfloat_encode_outcome(case: FormatCase, value: float) -> int | object:
    try:
        rounded = gfloat.round_float(
            case.gfloat_info,
            value,
            sat=case.gfloat_saturate,
        )
    except ValueError:
        return VALUE_ERROR

    if math.isnan(float(rounded)) and case.canonical_nan is not None:
        # gfloat deliberately treats all NaN encodings as equivalent.  Tibs'
        # stable wire spelling follows the standards/Bitstring snapshot.
        return case.canonical_nan
    return int(gfloat.encode_float(case.gfloat_info, rounded))


def _bitstring_encode_outcome(dtype: Any, value: float) -> int | object:
    try:
        return int(dtype.pack(value).uint)
    except ValueError:
        return VALUE_ERROR


def _ml_dtype(case: FormatCase) -> Any:
    assert case.ml_dtype_name is not None
    return getattr(ml_dtypes, case.ml_dtype_name)


def _ml_decode(case: FormatCase, raw: int) -> float:
    # NumPy stores the four- and six-bit dtypes in one byte.  Only the low
    # ``case.bits`` bits carry the OCP encoding; padding bits are not part of
    # the value being compared with Tibs.
    storage = np.asarray([raw], dtype=np.uint8)
    return float(storage.view(_ml_dtype(case))[0])


def _ml_encode(case: FormatCase, value: float) -> int:
    with np.errstate(over="ignore", invalid="ignore"):
        encoded = np.asarray([value], dtype=np.float64).astype(_ml_dtype(case))
    storage_raw = int(encoded.view(np.uint8)[0])
    # ml_dtypes documents the high bits of its byte-backed FP4/FP6 storage as
    # unused.  They are deliberately masked rather than treated as part of the
    # wire encoding.
    return storage_raw & ((1 << case.bits) - 1)


@pytest.mark.parametrize(
    ("distribution", "expected_version"),
    ORACLE_VERSIONS.items(),
)
def test_oracle_version_is_pinned(distribution: str, expected_version: str) -> None:
    actual_version = metadata.version(distribution)
    assert actual_version == expected_version, (
        f"conformance oracle {distribution!r} is {actual_version}, expected "
        f"the reviewed pin {expected_version}; upgrade the pin and expected "
        "semantics together"
    )


def test_bitstring_snapshot_is_pinned() -> None:
    assert bitstring.__version__ == BITSTRING_VERSION, (
        f"Bitstring is {bitstring.__version__}, expected the reviewed "
        f"{BITSTRING_VERSION} snapshot at {BITSTRING_COMMIT}"
    )
    assert BITSTRING_COMMIT == "c336d1e6a6ebf7ccc364840cb12949a7b012d985"


def test_gfloat_p3109_descriptors_match_the_public_draft_snapshot() -> None:
    assert P3109_PUBLIC_COMMIT == "aa9d236d7a31b38fbe43b703a0bfdfc3d8be5d45"
    assert P3109_K8P3SE.name == "p3109_k8p3se", P3109_PUBLIC_COMMIT
    assert P3109_K8P3SE.bits == 8, P3109_PUBLIC_COMMIT
    assert P3109_K8P3SE.precision == 3, P3109_PUBLIC_COMMIT
    assert P3109_K8P3SE.max == 49152.0, P3109_PUBLIC_COMMIT
    assert P3109_K8P4SE.name == "p3109_k8p4se", P3109_PUBLIC_COMMIT
    assert P3109_K8P4SE.bits == 8, P3109_PUBLIC_COMMIT
    assert P3109_K8P4SE.precision == 4, P3109_PUBLIC_COMMIT
    assert P3109_K8P4SE.max == 224.0, P3109_PUBLIC_COMMIT


@pytest.mark.parametrize(
    ("tibs_spec", "relative_path"),
    P3109_VALUE_TABLES,
    ids=("p3109_k8p3se", "p3109_k8p4se"),
)
def test_p3109_public_snapshot_exhaustive_raw_decode(
    tibs_spec: str,
    relative_path: Path,
) -> None:
    snapshot_root = os.environ.get("TIBS_P3109_SNAPSHOT")
    if snapshot_root is None:
        pytest.skip(
            "set TIBS_P3109_SNAPSHOT to the P3109/Public checkout at "
            f"{P3109_PUBLIC_COMMIT} to run the official value-table oracle"
        )

    table_path = Path(snapshot_root).expanduser() / relative_path
    for raw, expected in enumerate(_read_p3109_value_table(table_path)):
        actual = Tibs.from_value("u8", raw).to_value(tibs_spec)
        _assert_same_float(
            actual,
            expected,
            context=f"{tibs_spec}/{relative_path}, raw=0x{raw:02x}",
        )


@pytest.mark.parametrize("case", FORMAT_CASES, ids=_case_id)
def test_gfloat_exhaustive_raw_decode(case: FormatCase) -> None:
    assert case.gfloat_info.bits == case.bits
    for raw in range(1 << case.bits):
        expected = gfloat.decode_float(case.gfloat_info, raw).fval
        actual = _tibs_decode(case, raw)
        _assert_same_float(
            actual,
            expected,
            context=f"{case.tibs_spec}, raw=0x{raw:02x}",
        )


@pytest.mark.parametrize("case", GFLOAT_BINARY16_CASES, ids=_case_id)
def test_gfloat_all_binary16_encodings(case: FormatCase) -> None:
    for source_bits, value in enumerate(BINARY16_VALUES):
        expected = _gfloat_encode_outcome(case, value)
        actual = _tibs_encode_outcome(case, value)
        assert actual == expected, (
            f"{case.tibs_spec}, binary16=0x{source_bits:04x}, value={value!r}: "
            f"Tibs raw={actual!r}, gfloat raw={expected!r}"
        )


@pytest.mark.parametrize("case", ML_DTYPES_CASES, ids=_case_id)
def test_ml_dtypes_exhaustive_raw_decode(case: FormatCase) -> None:
    for raw in range(1 << case.bits):
        expected = _ml_decode(case, raw)
        actual = _tibs_decode(case, raw)
        _assert_same_float(
            actual,
            expected,
            context=f"{case.tibs_spec}/{case.ml_dtype_name}, raw=0x{raw:02x}",
        )


@pytest.mark.parametrize("case", ML_DTYPES_ROUNDING_CASES, ids=_case_id)
def test_ml_dtypes_all_in_range_binary16_encodings(case: FormatCase) -> None:
    for source_bits, value in enumerate(BINARY16_VALUES):
        if (
            not math.isfinite(value)
            or value < case.gfloat_info.min
            or value > case.gfloat_info.max
        ):
            continue

        expected = _gfloat_encode_outcome(case, value)
        ml_raw = _ml_encode(case, value)
        tibs_raw = _tibs_encode(case, value)
        assert ml_raw == expected, (
            f"{case.ml_dtype_name}, binary16=0x{source_bits:04x}, value={value!r}: "
            f"ml_dtypes raw=0x{ml_raw:02x}, gfloat raw={expected!r}"
        )
        assert tibs_raw == ml_raw, (
            f"{case.tibs_spec}, binary16=0x{source_bits:04x}, value={value!r}: "
            f"Tibs raw=0x{tibs_raw:02x}, ml_dtypes raw=0x{ml_raw:02x}"
        )


@pytest.mark.parametrize("case", ML_DTYPES_CASES, ids=_case_id)
def test_ml_dtypes_canonical_values_encode_identically(case: FormatCase) -> None:
    for raw in range(1 << case.bits):
        value = gfloat.decode_float(case.gfloat_info, raw).fval
        if not math.isfinite(float(value)):
            # Payload/sign canonicalization differs between libraries; decode
            # classification is exhaustive above.  Infinity packing is also
            # policy-specific for Tibs' E5M2 saturating dtype.  Those policies
            # are checked against gfloat and Bitstring instead.
            continue
        if int(gfloat.encode_float(case.gfloat_info, value)) != raw:
            continue

        ml_raw = _ml_encode(case, float(value))
        tibs_raw = _tibs_encode(case, float(value))
        assert ml_raw == raw, (
            f"{case.ml_dtype_name}, canonical value={value!r}: "
            f"raw=0x{ml_raw:02x}, expected 0x{raw:02x}"
        )
        assert tibs_raw == raw, (
            f"{case.tibs_spec}, canonical value={value!r}: "
            f"raw=0x{tibs_raw:02x}, expected 0x{raw:02x}"
        )


@pytest.mark.parametrize("case", FORMAT_CASES, ids=_case_id)
def test_bitstring_exhaustive_raw_decode(case: FormatCase) -> None:
    dtype = bitstring.Dtype(case.bitstring_spec)
    for raw in range(1 << case.bits):
        bits = bitstring.Bits(uint=raw, length=case.bits)
        expected = dtype.unpack(bits)
        actual = _tibs_decode(case, raw)
        _assert_same_float(
            actual,
            expected,
            context=f"{case.tibs_spec}/{case.bitstring_spec}, raw=0x{raw:02x}",
        )


@pytest.mark.parametrize("case", FORMAT_CASES, ids=_case_id)
def test_bitstring_all_binary16_encodings(case: FormatCase) -> None:
    dtype = bitstring.Dtype(case.bitstring_spec)
    for source_bits, value in enumerate(BINARY16_VALUES):
        expected = _bitstring_encode_outcome(dtype, value)
        actual = _tibs_encode_outcome(case, value)
        assert actual == expected, (
            f"{case.tibs_spec}/{case.bitstring_spec}, "
            f"binary16=0x{source_bits:04x}, value={value!r}: "
            f"Tibs raw={actual!r}, Bitstring raw={expected!r}"
        )
