"""Small, dependency-free reference model for Tibs' narrow numeric dtypes.

This deliberately favours an enumerable value table over the bit-level
conversion algorithm used by the Rust implementation.  The formats contain at
most 256 encodings, so choosing between adjacent decoded values is both simple
and an independent way to test round-to-nearest, ties-to-even.

The definitions use OCP MX v1.0/OFP8 v1.0 and the P3109 signed, extended K8P3
and K8P4 draft configurations reviewed for this feature. Live
comparisons with external libraries belong in the optional conformance suite;
normal tests must remain dependency-free.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
import math


@dataclass(frozen=True)
class NarrowFormat:
    name: str
    width: int
    family: str
    exponent_bits: int = 0
    mantissa_bits: int = 0
    bias: int = 0
    overflow: str = "saturate"
    signed_zero: bool = True

    @property
    def sign_bit(self) -> int:
        return 1 << (self.width - 1)


FORMATS = {
    fmt.name: fmt
    for fmt in (
        NarrowFormat("binary8p3", 8, "p3109", 5, 2, 16, "extended", False),
        NarrowFormat("binary8p4", 8, "p3109", 4, 3, 8, "extended", False),
        NarrowFormat("ocp_e4m3_saturate", 8, "ocp_e4m3", 4, 3, 7),
        NarrowFormat("ocp_e4m3_overflow", 8, "ocp_e4m3", 4, 3, 7, "overflow"),
        NarrowFormat("ocp_e5m2_saturate", 8, "ocp_e5m2", 5, 2, 15),
        NarrowFormat("ocp_e5m2_overflow", 8, "ocp_e5m2", 5, 2, 15, "overflow"),
        NarrowFormat("ocp_e3m2", 6, "ocp_finite", 3, 2, 3),
        NarrowFormat("ocp_e2m3", 6, "ocp_finite", 2, 3, 1),
        NarrowFormat("ocp_e2m1", 4, "ocp_finite", 2, 1, 1),
        NarrowFormat("ocp_e8m0", 8, "ocp_e8m0"),
        NarrowFormat("ocp_int8", 8, "ocp_int8"),
    )
}

FLOAT_FORMAT_NAMES = tuple(
    name
    for name, fmt in FORMATS.items()
    if fmt.family not in {"ocp_e8m0", "ocp_int8"}
)


def _decode_regular(fmt: NarrowFormat, raw: int) -> float:
    """Decode sign/exponent/mantissa without applying reserved-code rules."""

    sign = bool(raw & fmt.sign_bit)
    exponent_mask = (1 << fmt.exponent_bits) - 1
    mantissa_mask = (1 << fmt.mantissa_bits) - 1
    exponent = (raw >> fmt.mantissa_bits) & exponent_mask
    mantissa = raw & mantissa_mask
    if exponent == 0:
        magnitude = math.ldexp(mantissa / (1 << fmt.mantissa_bits), 1 - fmt.bias)
    else:
        magnitude = math.ldexp(
            1.0 + mantissa / (1 << fmt.mantissa_bits), exponent - fmt.bias
        )
    return math.copysign(magnitude, -1.0 if sign else 1.0)


def decode(name: str, raw: int) -> float:
    """Decode one raw code according to the pinned format definition."""

    fmt = FORMATS[name]
    if not 0 <= raw < 1 << fmt.width:
        raise ValueError(f"raw value {raw} does not fit {fmt.width} bits")

    if fmt.family == "ocp_e8m0":
        return math.nan if raw == 0xFF else math.ldexp(1.0, raw - 127)

    if fmt.family == "ocp_int8":
        signed = raw if raw < 0x80 else raw - 0x100
        return signed / 64.0

    if fmt.family == "p3109":
        if raw == 0x80:
            return math.nan
        if raw == 0x7F:
            return math.inf
        if raw == 0xFF:
            return -math.inf
        return _decode_regular(fmt, raw)

    exponent_mask = (1 << fmt.exponent_bits) - 1
    mantissa_mask = (1 << fmt.mantissa_bits) - 1
    exponent = (raw >> fmt.mantissa_bits) & exponent_mask
    mantissa = raw & mantissa_mask
    if fmt.family == "ocp_e4m3" and exponent == exponent_mask and mantissa == mantissa_mask:
        return math.nan
    if fmt.family == "ocp_e5m2" and exponent == exponent_mask:
        if mantissa:
            return math.nan
        return math.copysign(math.inf, -1.0 if raw & fmt.sign_bit else 1.0)
    return _decode_regular(fmt, raw)


@lru_cache(maxsize=None)
def positive_finite_codes(name: str) -> tuple[int, ...]:
    fmt = FORMATS[name]
    if fmt.family == "p3109" or fmt.family == "ocp_e4m3":
        last = 0x7E
    elif fmt.family == "ocp_e5m2":
        last = 0x7B
    elif fmt.family == "ocp_finite":
        last = fmt.sign_bit - 1
    else:
        raise ValueError(f"{name} is not an S/E/M floating-point format")
    return tuple(range(last + 1))


@lru_cache(maxsize=None)
def positive_finite_values(name: str) -> tuple[float, ...]:
    return tuple(decode(name, raw) for raw in positive_finite_codes(name))


def terminal_virtual_code(name: str) -> int | None:
    """Return the reserved code participating in terminal RNE, if any.

    P3109 extended formats and the OCP ``overflow`` policies round against the
    numerical value that the first reserved code would have had before mapping
    it to infinity or NaN.  Saturating formats never select a terminal code.
    """

    fmt = FORMATS[name]
    if fmt.family == "p3109":
        return 0x7F
    if fmt.family == "ocp_e4m3" and fmt.overflow == "overflow":
        return 0x7F
    if fmt.family == "ocp_e5m2" and fmt.overflow == "overflow":
        return 0x7C
    return None


def terminal_midpoint(name: str) -> float | None:
    code = terminal_virtual_code(name)
    if code is None:
        return None
    values = positive_finite_values(name)
    return (values[-1] + _decode_regular(FORMATS[name], code)) / 2.0


def _terminal_result(fmt: NarrowFormat, negative: bool) -> int:
    if fmt.family == "ocp_e4m3":
        # A deterministic quiet NaN is used for either overflow sign.
        return 0xFF
    code = terminal_virtual_code(fmt.name)
    assert code is not None
    return code | (fmt.sign_bit if negative else 0)


def _apply_sign(fmt: NarrowFormat, code: int, negative: bool) -> int:
    if not negative:
        return code
    if code == 0 and not fmt.signed_zero:
        return 0
    return code | fmt.sign_bit


def _encode_sem_float(fmt: NarrowFormat, value: float) -> int:
    if math.isnan(value):
        if fmt.family == "p3109":
            return 0x80
        if fmt.family in {"ocp_e4m3", "ocp_e5m2"}:
            return 0xFF
        raise ValueError(f"{fmt.name} has no NaN representation")

    negative = math.copysign(1.0, value) < 0.0
    if math.isinf(value):
        if terminal_virtual_code(fmt.name) is not None:
            return _terminal_result(fmt, negative)
        max_code = positive_finite_codes(fmt.name)[-1]
        return _apply_sign(fmt, max_code, negative)

    magnitude = abs(value)
    if magnitude == 0.0:
        return _apply_sign(fmt, 0, negative)

    codes = positive_finite_codes(fmt.name)
    values = positive_finite_values(fmt.name)
    max_code = codes[-1]
    if magnitude > values[-1]:
        terminal_code = terminal_virtual_code(fmt.name)
        if terminal_code is None:
            return _apply_sign(fmt, max_code, negative)
        virtual_value = _decode_regular(fmt, terminal_code)
        lower_distance = magnitude - values[-1]
        upper_distance = virtual_value - magnitude
        if lower_distance < upper_distance:
            return _apply_sign(fmt, max_code, negative)
        if upper_distance < lower_distance:
            return _terminal_result(fmt, negative)
        if max_code & 1:
            return _terminal_result(fmt, negative)
        return _apply_sign(fmt, max_code, negative)

    index = bisect_left(values, magnitude)
    if index < len(values) and values[index] == magnitude:
        return _apply_sign(fmt, codes[index], negative)
    if index == 0:
        chosen = codes[0]
    else:
        lower_code, upper_code = codes[index - 1], codes[index]
        lower_distance = magnitude - values[index - 1]
        upper_distance = values[index] - magnitude
        if lower_distance < upper_distance:
            chosen = lower_code
        elif upper_distance < lower_distance:
            chosen = upper_code
        else:
            chosen = lower_code if lower_code & 1 == 0 else upper_code
    return _apply_sign(fmt, chosen, negative)


def _encode_e8m0(value: float) -> int:
    if math.isnan(value):
        return 0xFF
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("ocp_e8m0 requires a positive, finite power of two")
    fraction, exponent = math.frexp(value)
    if fraction != 0.5:
        raise ValueError("ocp_e8m0 requires an exact power of two")
    raw = exponent - 1 + 127
    if not 0 <= raw <= 0xFE:
        raise ValueError("ocp_e8m0 exponent is outside -127 through 127")
    return raw


def _encode_int8(value: float) -> int:
    if math.isnan(value):
        raise ValueError("ocp_int8 has no NaN representation")
    if value == math.inf:
        return 0x7F
    if value == -math.inf:
        return 0x80
    scaled = round(value * 64.0)
    scaled = min(127, max(-128, scaled))
    return scaled & 0xFF


def encode(name: str, value: float) -> int:
    """Encode using exhaustive-neighbour RNE and the format's special rules."""

    fmt = FORMATS[name]
    value = float(value)
    if fmt.family == "ocp_e8m0":
        return _encode_e8m0(value)
    if fmt.family == "ocp_int8":
        return _encode_int8(value)
    return _encode_sem_float(fmt, value)


def same_float(actual: float, expected: float) -> bool:
    """Compare exact decoded values, including NaN class and zero sign."""

    if math.isnan(expected):
        return math.isnan(actual)
    if actual != expected:
        return False
    if expected == 0.0:
        return math.copysign(1.0, actual) == math.copysign(1.0, expected)
    return True
