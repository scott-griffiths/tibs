use std::fmt;

/// The narrow numeric formats supported by Tibs.
///
/// The OCP E4M3 and E5M2 variants have identical bit-level decoding but distinct
/// overflow behavior when packing. Keeping that policy in the enum prevents a
/// caller from accidentally applying it to a format for which it is meaningless.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NarrowFloatFormat {
    Binary8P3,
    Binary8P4,
    OcpE4M3Saturate,
    OcpE4M3Overflow,
    OcpE5M2Saturate,
    OcpE5M2Overflow,
    OcpE3M2,
    OcpE2M3,
    OcpE2M1,
    OcpE8M0,
    OcpInt8,
}

impl NarrowFloatFormat {
    /// Every variant, in discriminant order, so that `format as usize` indexes
    /// this and any table built from it.
    pub(crate) const ALL: [Self; 11] = [
        Self::Binary8P3,
        Self::Binary8P4,
        Self::OcpE4M3Saturate,
        Self::OcpE4M3Overflow,
        Self::OcpE5M2Saturate,
        Self::OcpE5M2Overflow,
        Self::OcpE3M2,
        Self::OcpE2M3,
        Self::OcpE2M1,
        Self::OcpE8M0,
        Self::OcpInt8,
    ];

    pub(crate) const COUNT: usize = Self::ALL.len();

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Binary8P3 => "binary8p3",
            Self::Binary8P4 => "binary8p4",
            Self::OcpE4M3Saturate => "ocp_e4m3_saturate",
            Self::OcpE4M3Overflow => "ocp_e4m3_overflow",
            Self::OcpE5M2Saturate => "ocp_e5m2_saturate",
            Self::OcpE5M2Overflow => "ocp_e5m2_overflow",
            Self::OcpE3M2 => "ocp_e3m2",
            Self::OcpE2M3 => "ocp_e2m3",
            Self::OcpE2M1 => "ocp_e2m1",
            Self::OcpE8M0 => "ocp_e8m0",
            Self::OcpInt8 => "ocp_int8",
        }
    }

    pub(crate) const fn bit_length(self) -> usize {
        match self {
            Self::OcpE3M2 | Self::OcpE2M3 => 6,
            Self::OcpE2M1 => 4,
            _ => 8,
        }
    }

    const fn binary_format(self) -> Option<BinaryFormat> {
        match self {
            Self::Binary8P3 => Some(BinaryFormat {
                bits: 8,
                exponent_bits: 5,
                mantissa_bits: 2,
                bias: 16,
                max_finite_code: 0x7e,
                has_negative_zero: false,
            }),
            Self::Binary8P4 => Some(BinaryFormat {
                bits: 8,
                exponent_bits: 4,
                mantissa_bits: 3,
                bias: 8,
                max_finite_code: 0x7e,
                has_negative_zero: false,
            }),
            Self::OcpE4M3Saturate | Self::OcpE4M3Overflow => Some(BinaryFormat {
                bits: 8,
                exponent_bits: 4,
                mantissa_bits: 3,
                bias: 7,
                max_finite_code: 0x7e,
                has_negative_zero: true,
            }),
            Self::OcpE5M2Saturate | Self::OcpE5M2Overflow => Some(BinaryFormat {
                bits: 8,
                exponent_bits: 5,
                mantissa_bits: 2,
                bias: 15,
                max_finite_code: 0x7b,
                has_negative_zero: true,
            }),
            Self::OcpE3M2 => Some(BinaryFormat {
                bits: 6,
                exponent_bits: 3,
                mantissa_bits: 2,
                bias: 3,
                max_finite_code: 0x1f,
                has_negative_zero: true,
            }),
            Self::OcpE2M3 => Some(BinaryFormat {
                bits: 6,
                exponent_bits: 2,
                mantissa_bits: 3,
                bias: 1,
                max_finite_code: 0x1f,
                has_negative_zero: true,
            }),
            Self::OcpE2M1 => Some(BinaryFormat {
                bits: 4,
                exponent_bits: 2,
                mantissa_bits: 1,
                bias: 1,
                max_finite_code: 0x07,
                has_negative_zero: true,
            }),
            Self::OcpE8M0 | Self::OcpInt8 => None,
        }
    }

    const fn nan_code(self) -> Option<u8> {
        match self {
            Self::Binary8P3 | Self::Binary8P4 => Some(0x80),
            Self::OcpE4M3Saturate
            | Self::OcpE4M3Overflow
            | Self::OcpE5M2Saturate
            | Self::OcpE5M2Overflow
            | Self::OcpE8M0 => Some(0xff),
            Self::OcpE3M2 | Self::OcpE2M3 | Self::OcpE2M1 | Self::OcpInt8 => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NarrowFloatEncodeError {
    NaNNotSupported,
    ValueNotRepresentable,
}

impl fmt::Display for NarrowFloatEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NaNNotSupported => f.write_str("this format cannot represent NaN"),
            Self::ValueNotRepresentable => {
                f.write_str("the value is not exactly representable in this format")
            }
        }
    }
}

impl std::error::Error for NarrowFloatEncodeError {}

#[derive(Clone, Copy)]
struct BinaryFormat {
    bits: u8,
    exponent_bits: u8,
    mantissa_bits: u8,
    bias: i32,
    /// Largest finite positive code point. All supported binary formats use
    /// sign-magnitude encoding, so the negative counterpart sets `sign_mask`.
    max_finite_code: u8,
    has_negative_zero: bool,
}

impl BinaryFormat {
    const fn sign_mask(self) -> u8 {
        1 << (self.bits - 1)
    }

    const fn exponent_mask(self) -> u8 {
        (1 << self.exponent_bits) - 1
    }

    const fn mantissa_mask(self) -> u8 {
        (1 << self.mantissa_bits) - 1
    }

    const fn min_normal_exponent(self) -> i32 {
        1 - self.bias
    }

    /// The value of the least significant subnormal bit, `2^(1-bias-mantissa)`.
    ///
    /// Every supported format puts this well inside the binary64 normal range,
    /// so the constant is just an exponent field.
    const fn subnormal_quantum(self) -> f64 {
        let exponent = 1 - self.bias - self.mantissa_bits as i32;
        f64::from_bits(((exponent + 1023) as u64) << 52)
    }
}

/// Decode one already-extracted narrow-format code point to an `f64`.
///
/// For four- and six-bit formats, `raw` must not contain bits above the field.
/// Every in-range code point has a defined result, although that result may be
/// a NaN or infinity for formats which provide those special values.
///
/// An unpacker resolves [`narrow_float_decode_table`] once instead of calling
/// this per value, so only the tests below reach it.
#[cfg(test)]
fn decode_narrow_float(raw: u8, format: NarrowFloatFormat) -> f64 {
    narrow_float_decode_table(format)[raw as usize]
}

/// Every code point of `format`, decoded.
///
/// An unpacker resolves this once and indexes it per value, which keeps the
/// one-time `LazyLock` check and the format dispatch out of the loop.
pub(crate) fn narrow_float_decode_table(format: NarrowFloatFormat) -> &'static [f64; 256] {
    &DECODE_TABLES[format as usize]
}

/// Every code point of every format, decoded once.
///
/// A format has at most 256 code points, so decoding is a table lookup rather
/// than a per-value walk through the format's special cases. 22 KB of static
/// data, of which one 2 KB row is touched per unpack.
static DECODE_TABLES: std::sync::LazyLock<[[f64; 256]; NarrowFloatFormat::COUNT]> =
    std::sync::LazyLock::new(|| {
        std::array::from_fn(|format_index| {
            let format = NarrowFloatFormat::ALL[format_index];
            let length = format.bit_length();
            std::array::from_fn(|raw| {
                if length < 8 && raw >= (1 << length) {
                    // Unreachable code points for a four- or six-bit format;
                    // the row is square so the index arithmetic stays uniform.
                    f64::NAN
                } else {
                    decode_narrow_float_uncached(raw as u8, format)
                }
            })
        })
    });

fn decode_narrow_float_uncached(raw: u8, format: NarrowFloatFormat) -> f64 {
    match format {
        NarrowFloatFormat::OcpE8M0 => decode_e8m0(raw),
        NarrowFloatFormat::OcpInt8 => (raw as i8 as f64) * (1.0 / 64.0),
        _ => decode_binary(raw, format, format.binary_format().unwrap()),
    }
}

/// Round and encode an `f64` directly into one narrow-format code point.
///
/// Rounding is round-to-nearest, ties-to-even. Conversion is performed from
/// the exact binary64 significand and exponent and never passes through f16.
#[inline]
pub(crate) fn encode_narrow_float(
    value: f64,
    format: NarrowFloatFormat,
) -> Result<u8, NarrowFloatEncodeError> {
    match format {
        NarrowFloatFormat::OcpE8M0 => encode_e8m0(value),
        NarrowFloatFormat::OcpInt8 => encode_int8(value),
        _ => encode_binary(value, format, format.binary_format().unwrap()),
    }
}

fn decode_binary(raw: u8, format: NarrowFloatFormat, binary: BinaryFormat) -> f64 {
    match format {
        NarrowFloatFormat::Binary8P3 | NarrowFloatFormat::Binary8P4 => {
            if raw == 0x80 {
                return f64::NAN;
            }
            if raw == 0x7f {
                return f64::INFINITY;
            }
            if raw == 0xff {
                return f64::NEG_INFINITY;
            }
        }
        NarrowFloatFormat::OcpE4M3Saturate | NarrowFloatFormat::OcpE4M3Overflow => {
            if raw == 0x7f || raw == 0xff {
                return f64::NAN;
            }
        }
        NarrowFloatFormat::OcpE5M2Saturate | NarrowFloatFormat::OcpE5M2Overflow => {
            let exponent = (raw >> binary.mantissa_bits) & binary.exponent_mask();
            let mantissa = raw & binary.mantissa_mask();
            if exponent == binary.exponent_mask() {
                if mantissa == 0 {
                    return if raw & binary.sign_mask() == 0 {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    };
                }
                return f64::NAN;
            }
        }
        NarrowFloatFormat::OcpE3M2 | NarrowFloatFormat::OcpE2M3 | NarrowFloatFormat::OcpE2M1 => {}
        NarrowFloatFormat::OcpE8M0 | NarrowFloatFormat::OcpInt8 => unreachable!(),
    }

    let negative = raw & binary.sign_mask() != 0;
    let exponent = ((raw >> binary.mantissa_bits) & binary.exponent_mask()) as i32;
    let mantissa = (raw & binary.mantissa_mask()) as u64;
    let magnitude = if exponent == 0 {
        // Subnormals use 0.M * 2^(1-bias), equivalently
        // integer(M) * 2^(1-bias-mantissa_bits). The mantissa is a handful of
        // bits and the quantum a constant, so the product is exact.
        (mantissa as f64) * binary.subnormal_quantum()
    } else {
        // A normal narrow value is 1.M * 2^(exponent-bias), which is a
        // binary64 with the same fraction bits left-aligned and the exponent
        // rebiased. Every supported format's normal range sits inside
        // binary64's, so the assembled exponent field is always in range.
        let biased = (exponent - binary.bias + 1023) as u64;
        f64::from_bits((biased << 52) | (mantissa << (52 - binary.mantissa_bits as u32)))
    };

    if negative { -magnitude } else { magnitude }
}

fn encode_binary(
    value: f64,
    format: NarrowFloatFormat,
    binary: BinaryFormat,
) -> Result<u8, NarrowFloatEncodeError> {
    if value.is_nan() {
        return format
            .nan_code()
            .ok_or(NarrowFloatEncodeError::NaNNotSupported);
    }

    let negative = value.is_sign_negative();
    if value.is_infinite() {
        return Ok(encode_binary_overflow(negative, format, binary));
    }

    if value == 0.0 {
        return Ok(if negative && binary.has_negative_zero {
            binary.sign_mask()
        } else {
            0
        });
    }

    let magnitude = value.abs();
    let magnitude_code = rounded_positive_binary_code(magnitude, binary);
    if magnitude_code > binary.max_finite_code as u32 {
        return Ok(encode_binary_overflow(negative, format, binary));
    }

    // A P3109 format has no negative zero: underflow from a negative input
    // must produce +0 rather than its repurposed 0x80 NaN code point.
    if magnitude_code == 0 && !binary.has_negative_zero {
        return Ok(0);
    }

    Ok((magnitude_code as u8) | if negative { binary.sign_mask() } else { 0 })
}

fn encode_binary_overflow(negative: bool, format: NarrowFloatFormat, binary: BinaryFormat) -> u8 {
    let sign = if negative { binary.sign_mask() } else { 0 };
    match format {
        NarrowFloatFormat::Binary8P3 | NarrowFloatFormat::Binary8P4 => 0x7f | sign,
        NarrowFloatFormat::OcpE4M3Saturate
        | NarrowFloatFormat::OcpE5M2Saturate
        | NarrowFloatFormat::OcpE3M2
        | NarrowFloatFormat::OcpE2M3
        | NarrowFloatFormat::OcpE2M1 => binary.max_finite_code | sign,
        NarrowFloatFormat::OcpE4M3Overflow => 0xff,
        NarrowFloatFormat::OcpE5M2Overflow => 0x7c | sign,
        NarrowFloatFormat::OcpE8M0 | NarrowFloatFormat::OcpInt8 => unreachable!(),
    }
}

/// Return the positive sign-magnitude code after direct binary64-to-target
/// rounding. The returned value may be beyond the format's finite range; the
/// caller applies the format-specific overflow rule.
fn rounded_positive_binary_code(value: f64, binary: BinaryFormat) -> u32 {
    debug_assert!(value.is_finite() && value > 0.0);

    // Express binary64 exactly as significand * 2^source_exponent. For a normal
    // number the significand has 53 bits; for a binary64 subnormal it may have
    // fewer. Keeping this representation integral makes midpoint decisions
    // exact, including cases where an f16 intermediate would double-round.
    let raw = value.to_bits();
    let stored_exponent = ((raw >> 52) & 0x7ff) as i32;
    let fraction = raw & ((1u64 << 52) - 1);
    let (significand, source_exponent) = if stored_exponent == 0 {
        (fraction, -1074)
    } else {
        ((1u64 << 52) | fraction, stored_exponent - 1023 - 52)
    };
    debug_assert_ne!(significand, 0);

    let significand_top_bit = 63 - significand.leading_zeros() as i32;
    let floor_exponent = source_exponent + significand_top_bit;
    let mut target_exponent = floor_exponent.max(binary.min_normal_exponent());
    let quantum_exponent = target_exponent - binary.mantissa_bits as i32;
    let mut rounded_significand =
        round_integer_to_power_of_two(significand, source_exponent, quantum_exponent);

    let hidden_bit = 1u64 << binary.mantissa_bits;
    if rounded_significand == 0 {
        return 0;
    }

    // Values below the minimum normal exponent share its fixed subnormal
    // quantum. Reaching the hidden bit naturally crosses into the minimum
    // normal value.
    if rounded_significand < hidden_bit {
        debug_assert_eq!(target_exponent, binary.min_normal_exponent());
        return rounded_significand as u32;
    }

    // Rounding 1.111... upward carries into the next exponent.
    if rounded_significand == hidden_bit << 1 {
        rounded_significand = hidden_bit;
        target_exponent += 1;
    }
    debug_assert!((hidden_bit..(hidden_bit << 1)).contains(&rounded_significand));

    let biased_exponent = target_exponent + binary.bias;
    debug_assert!(biased_exponent > 0);
    ((biased_exponent as u32) << binary.mantissa_bits) | (rounded_significand - hidden_bit) as u32
}

/// Round `significand * 2^source_exponent / 2^quantum_exponent` to an integer
/// using round-to-nearest, ties-to-even.
fn round_integer_to_power_of_two(
    significand: u64,
    source_exponent: i32,
    quantum_exponent: i32,
) -> u64 {
    let right_shift = quantum_exponent - source_exponent;
    if right_shift <= 0 {
        return significand
            .checked_shl((-right_shift) as u32)
            .expect("target significand shift must fit in u64");
    }
    if right_shift > 64 {
        return 0;
    }
    if right_shift == 64 {
        let halfway = 1u64 << 63;
        return u64::from(significand > halfway);
    }

    let shift = right_shift as u32;
    let truncated = significand >> shift;
    let remainder_mask = (1u64 << shift) - 1;
    let remainder = significand & remainder_mask;
    let halfway = 1u64 << (shift - 1);
    truncated + u64::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
}

fn decode_e8m0(raw: u8) -> f64 {
    if raw == 0xff {
        f64::NAN
    } else {
        // 2^(raw-127) rebiased into binary64: 0x00 gives 2^-127 and 0xfe gives
        // 2^127, both comfortably normal, so this is only an exponent field.
        f64::from_bits((raw as u64 + (1023 - 127)) << 52)
    }
}

fn encode_e8m0(value: f64) -> Result<u8, NarrowFloatEncodeError> {
    if value.is_nan() {
        return Ok(0xff);
    }
    if !value.is_finite() || value <= 0.0 {
        return Err(NarrowFloatEncodeError::ValueNotRepresentable);
    }

    let raw = value.to_bits();
    let stored_exponent = ((raw >> 52) & 0x7ff) as i32;
    let fraction = raw & ((1u64 << 52) - 1);
    // All E8M0 values are normal binary64 powers of two, so an exact value has
    // a zero fraction and an ordinary stored exponent.
    if stored_exponent == 0 || fraction != 0 {
        return Err(NarrowFloatEncodeError::ValueNotRepresentable);
    }
    let exponent = stored_exponent - 1023;
    if !(-127..=127).contains(&exponent) {
        return Err(NarrowFloatEncodeError::ValueNotRepresentable);
    }
    Ok((exponent + 127) as u8)
}

fn encode_int8(value: f64) -> Result<u8, NarrowFloatEncodeError> {
    if value.is_nan() {
        return Err(NarrowFloatEncodeError::NaNNotSupported);
    }
    if value >= 127.0 / 64.0 {
        return Ok(0x7f);
    }
    if value <= -2.0 {
        return Ok(0x80);
    }

    // Multiplication by a power of two is exact throughout the unsaturated
    // range. `floor` plus an explicit half/even decision also handles negative
    // ties: e.g. -1.5 chooses -2 while -2.5 chooses -2.
    let scaled = value * 64.0;
    let lower = scaled.floor();
    let remainder = scaled - lower;
    let lower_integer = lower as i16;
    let rounded =
        lower_integer + i16::from(remainder > 0.5 || (remainder == 0.5 && lower_integer & 1 != 0));
    Ok((rounded as i8) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_FORMATS: [NarrowFloatFormat; 11] = [
        NarrowFloatFormat::Binary8P3,
        NarrowFloatFormat::Binary8P4,
        NarrowFloatFormat::OcpE4M3Saturate,
        NarrowFloatFormat::OcpE4M3Overflow,
        NarrowFloatFormat::OcpE5M2Saturate,
        NarrowFloatFormat::OcpE5M2Overflow,
        NarrowFloatFormat::OcpE3M2,
        NarrowFloatFormat::OcpE2M3,
        NarrowFloatFormat::OcpE2M1,
        NarrowFloatFormat::OcpE8M0,
        NarrowFloatFormat::OcpInt8,
    ];

    const BINARY_FORMATS: [NarrowFloatFormat; 9] = [
        NarrowFloatFormat::Binary8P3,
        NarrowFloatFormat::Binary8P4,
        NarrowFloatFormat::OcpE4M3Saturate,
        NarrowFloatFormat::OcpE4M3Overflow,
        NarrowFloatFormat::OcpE5M2Saturate,
        NarrowFloatFormat::OcpE5M2Overflow,
        NarrowFloatFormat::OcpE3M2,
        NarrowFloatFormat::OcpE2M3,
        NarrowFloatFormat::OcpE2M1,
    ];

    fn is_negative_zero(value: f64) -> bool {
        value == 0.0 && value.is_sign_negative()
    }

    fn next_up(value: f64) -> f64 {
        debug_assert!(value.is_finite() && value >= 0.0);
        f64::from_bits(value.to_bits() + 1)
    }

    fn next_down(value: f64) -> f64 {
        debug_assert!(value.is_finite() && value > 0.0);
        f64::from_bits(value.to_bits() - 1)
    }

    #[test]
    fn bit_lengths_are_intrinsic() {
        for format in ALL_FORMATS {
            let expected = match format {
                NarrowFloatFormat::OcpE2M1 => 4,
                NarrowFloatFormat::OcpE3M2 | NarrowFloatFormat::OcpE2M3 => 6,
                _ => 8,
            };
            assert_eq!(format.bit_length(), expected, "{format:?}");
        }
    }

    #[test]
    fn p3109_known_values_and_specials() {
        let p3 = NarrowFloatFormat::Binary8P3;
        assert_eq!(decode_narrow_float(0x00, p3), 0.0);
        assert_eq!(decode_narrow_float(0x01, p3), 2.0f64.powi(-17));
        assert_eq!(decode_narrow_float(0x40, p3), 1.0);
        assert_eq!(decode_narrow_float(0x7e, p3), 49_152.0);
        assert!(decode_narrow_float(0x80, p3).is_nan());
        assert_eq!(decode_narrow_float(0x7f, p3), f64::INFINITY);
        assert_eq!(decode_narrow_float(0xff, p3), f64::NEG_INFINITY);

        let p4 = NarrowFloatFormat::Binary8P4;
        assert_eq!(decode_narrow_float(0x01, p4), 2.0f64.powi(-10));
        assert_eq!(decode_narrow_float(0x08, p4), 2.0f64.powi(-7));
        assert_eq!(decode_narrow_float(0x40, p4), 1.0);
        assert_eq!(decode_narrow_float(0x7e, p4), 224.0);
        assert!(decode_narrow_float(0x80, p4).is_nan());
        assert_eq!(decode_narrow_float(0x7f, p4), f64::INFINITY);
        assert_eq!(decode_narrow_float(0xff, p4), f64::NEG_INFINITY);
    }

    #[test]
    fn ocp_float_known_values_and_specials() {
        let e4 = NarrowFloatFormat::OcpE4M3Saturate;
        assert_eq!(decode_narrow_float(0x01, e4), 2.0f64.powi(-9));
        assert_eq!(decode_narrow_float(0x38, e4), 1.0);
        assert_eq!(decode_narrow_float(0x7e, e4), 448.0);
        assert!(decode_narrow_float(0x7f, e4).is_nan());
        assert!(decode_narrow_float(0xff, e4).is_nan());

        let e5 = NarrowFloatFormat::OcpE5M2Saturate;
        assert_eq!(decode_narrow_float(0x01, e5), 2.0f64.powi(-16));
        assert_eq!(decode_narrow_float(0x3c, e5), 1.0);
        assert_eq!(decode_narrow_float(0x7b, e5), 57_344.0);
        assert_eq!(decode_narrow_float(0x7c, e5), f64::INFINITY);
        assert_eq!(decode_narrow_float(0xfc, e5), f64::NEG_INFINITY);
        for raw in [0x7d, 0x7e, 0x7f, 0xfd, 0xfe, 0xff] {
            assert!(decode_narrow_float(raw, e5).is_nan(), "raw={raw:#04x}");
        }

        let e3m2 = NarrowFloatFormat::OcpE3M2;
        assert_eq!(decode_narrow_float(0x01, e3m2), 0.0625);
        assert_eq!(decode_narrow_float(0x04, e3m2), 0.25);
        assert_eq!(decode_narrow_float(0x1f, e3m2), 28.0);

        let e2m3 = NarrowFloatFormat::OcpE2M3;
        assert_eq!(decode_narrow_float(0x01, e2m3), 0.125);
        assert_eq!(decode_narrow_float(0x08, e2m3), 1.0);
        assert_eq!(decode_narrow_float(0x1f, e2m3), 7.5);
    }

    #[test]
    fn e2m1_matches_the_complete_ocp_value_table() {
        let format = NarrowFloatFormat::OcpE2M1;
        let positive = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (raw, expected) in positive.into_iter().enumerate() {
            assert_eq!(decode_narrow_float(raw as u8, format), expected);
            let negative = decode_narrow_float((raw as u8) | 0x08, format);
            if raw == 0 {
                assert!(is_negative_zero(negative));
            } else {
                assert_eq!(negative, -expected);
            }
        }
    }

    #[test]
    fn ocp_formats_preserve_signed_zero_but_p3109_has_unique_zero() {
        for format in [
            NarrowFloatFormat::OcpE4M3Saturate,
            NarrowFloatFormat::OcpE4M3Overflow,
            NarrowFloatFormat::OcpE5M2Saturate,
            NarrowFloatFormat::OcpE5M2Overflow,
            NarrowFloatFormat::OcpE3M2,
            NarrowFloatFormat::OcpE2M3,
            NarrowFloatFormat::OcpE2M1,
        ] {
            let sign_mask = 1 << (format.bit_length() - 1);
            assert!(is_negative_zero(decode_narrow_float(
                sign_mask as u8,
                format
            )));
            assert_eq!(encode_narrow_float(-0.0, format), Ok(sign_mask as u8));
        }
        for format in [NarrowFloatFormat::Binary8P3, NarrowFloatFormat::Binary8P4] {
            assert!(decode_narrow_float(0x80, format).is_nan());
            assert_eq!(encode_narrow_float(-0.0, format), Ok(0x00));
            assert_eq!(encode_narrow_float(-f64::MIN_POSITIVE, format), Ok(0x00));
        }
    }

    #[test]
    fn every_binary_code_round_trips_or_canonicalizes_its_nan() {
        for format in BINARY_FORMATS {
            let count = 1u16 << format.bit_length();
            for raw in 0..count {
                let raw = raw as u8;
                let value = decode_narrow_float(raw, format);
                let encoded = encode_narrow_float(value, format);
                if value.is_nan() {
                    if let Some(canonical) = format.nan_code() {
                        assert_eq!(encoded, Ok(canonical), "{format:?}, raw={raw:#04x}");
                    } else {
                        assert_eq!(
                            encoded,
                            Err(NarrowFloatEncodeError::NaNNotSupported),
                            "{format:?}, raw={raw:#04x}"
                        );
                    }
                } else if format == NarrowFloatFormat::OcpE5M2Saturate && value.is_infinite() {
                    let expected = if value.is_sign_negative() { 0xfb } else { 0x7b };
                    assert_eq!(encoded, Ok(expected), "{format:?}, raw={raw:#04x}");
                } else {
                    assert_eq!(encoded, Ok(raw), "{format:?}, raw={raw:#04x}");
                }
            }
        }
    }

    #[test]
    fn every_finite_rounding_boundary_uses_ties_to_even() {
        for format in BINARY_FORMATS {
            let binary = format.binary_format().unwrap();
            for lower_raw in 0..binary.max_finite_code {
                let upper_raw = lower_raw + 1;
                let lower = decode_narrow_float(lower_raw, format);
                let upper = decode_narrow_float(upper_raw, format);
                if !lower.is_finite() || !upper.is_finite() {
                    continue;
                }
                let midpoint = lower + (upper - lower) * 0.5;
                let expected_magnitude = if lower_raw & 1 == 0 {
                    lower_raw
                } else {
                    upper_raw
                };
                assert_eq!(
                    encode_narrow_float(next_down(midpoint), format),
                    Ok(lower_raw),
                    "{format:?}, below midpoint between {lower_raw:#04x} and {upper_raw:#04x}"
                );
                assert_eq!(
                    encode_narrow_float(midpoint, format),
                    Ok(expected_magnitude),
                    "{format:?}, midpoint between {lower_raw:#04x} and {upper_raw:#04x}"
                );
                assert_eq!(
                    encode_narrow_float(next_up(midpoint), format),
                    Ok(upper_raw),
                    "{format:?}, above midpoint between {lower_raw:#04x} and {upper_raw:#04x}"
                );

                let negative_midpoint = -midpoint;
                let negative_lower = if expected_magnitude == 0 && !binary.has_negative_zero {
                    0
                } else {
                    expected_magnitude | binary.sign_mask()
                };
                assert_eq!(
                    encode_narrow_float(negative_midpoint, format),
                    Ok(negative_lower),
                    "{format:?}, negative midpoint between {lower_raw:#04x} and {upper_raw:#04x}"
                );
            }
        }
    }

    #[test]
    fn p3109_overflow_midpoints_match_the_draft_formats() {
        for (format, midpoint, max) in [
            (NarrowFloatFormat::Binary8P3, 53_248.0, 49_152.0),
            (NarrowFloatFormat::Binary8P4, 232.0, 224.0),
        ] {
            assert_eq!(encode_narrow_float(midpoint, format), Ok(0x7e));
            assert_eq!(decode_narrow_float(0x7e, format), max);
            assert_eq!(encode_narrow_float(next_up(midpoint), format), Ok(0x7f));
            assert_eq!(encode_narrow_float(-next_up(midpoint), format), Ok(0xff));
            assert_eq!(encode_narrow_float(f64::INFINITY, format), Ok(0x7f));
            assert_eq!(encode_narrow_float(f64::NEG_INFINITY, format), Ok(0xff));
            assert_eq!(encode_narrow_float(f64::NAN, format), Ok(0x80));
        }
    }

    #[test]
    fn e4m3_overflow_policy_is_part_of_the_format() {
        let saturate = NarrowFloatFormat::OcpE4M3Saturate;
        let overflow = NarrowFloatFormat::OcpE4M3Overflow;
        assert_eq!(encode_narrow_float(464.0, saturate), Ok(0x7e));
        assert_eq!(encode_narrow_float(464.0, overflow), Ok(0x7e));
        assert_eq!(encode_narrow_float(next_up(464.0), saturate), Ok(0x7e));
        assert_eq!(encode_narrow_float(next_up(464.0), overflow), Ok(0xff));
        assert_eq!(encode_narrow_float(-next_up(464.0), saturate), Ok(0xfe));
        assert_eq!(encode_narrow_float(-next_up(464.0), overflow), Ok(0xff));
        assert_eq!(encode_narrow_float(f64::INFINITY, saturate), Ok(0x7e));
        assert_eq!(encode_narrow_float(f64::NEG_INFINITY, saturate), Ok(0xfe));
        assert_eq!(encode_narrow_float(f64::INFINITY, overflow), Ok(0xff));
        assert_eq!(encode_narrow_float(f64::NEG_INFINITY, overflow), Ok(0xff));
        assert_eq!(encode_narrow_float(f64::NAN, saturate), Ok(0xff));
        assert_eq!(encode_narrow_float(f64::NAN, overflow), Ok(0xff));
    }

    #[test]
    fn e5m2_overflow_policy_is_part_of_the_format() {
        let saturate = NarrowFloatFormat::OcpE5M2Saturate;
        let overflow = NarrowFloatFormat::OcpE5M2Overflow;
        assert_eq!(encode_narrow_float(next_down(61_440.0), saturate), Ok(0x7b));
        assert_eq!(encode_narrow_float(next_down(61_440.0), overflow), Ok(0x7b));
        // The lower code at this tie is odd, so ties-to-even rounds upward.
        assert_eq!(encode_narrow_float(61_440.0, saturate), Ok(0x7b));
        assert_eq!(encode_narrow_float(61_440.0, overflow), Ok(0x7c));
        assert_eq!(encode_narrow_float(-61_440.0, saturate), Ok(0xfb));
        assert_eq!(encode_narrow_float(-61_440.0, overflow), Ok(0xfc));
        assert_eq!(encode_narrow_float(f64::INFINITY, saturate), Ok(0x7b));
        assert_eq!(encode_narrow_float(f64::NEG_INFINITY, saturate), Ok(0xfb));
        assert_eq!(encode_narrow_float(f64::INFINITY, overflow), Ok(0x7c));
        assert_eq!(encode_narrow_float(f64::NEG_INFINITY, overflow), Ok(0xfc));
        assert_eq!(encode_narrow_float(f64::NAN, saturate), Ok(0xff));
        assert_eq!(encode_narrow_float(f64::NAN, overflow), Ok(0xff));
    }

    #[test]
    fn smaller_ocp_formats_saturate_and_reject_nan() {
        for (format, positive_max, negative_max) in [
            (NarrowFloatFormat::OcpE3M2, 0x1f, 0x3f),
            (NarrowFloatFormat::OcpE2M3, 0x1f, 0x3f),
            (NarrowFloatFormat::OcpE2M1, 0x07, 0x0f),
        ] {
            assert_eq!(encode_narrow_float(f64::MAX, format), Ok(positive_max));
            assert_eq!(encode_narrow_float(f64::INFINITY, format), Ok(positive_max));
            assert_eq!(encode_narrow_float(-f64::MAX, format), Ok(negative_max));
            assert_eq!(
                encode_narrow_float(f64::NEG_INFINITY, format),
                Ok(negative_max)
            );
            assert_eq!(
                encode_narrow_float(f64::NAN, format),
                Err(NarrowFloatEncodeError::NaNNotSupported)
            );
        }
    }

    #[test]
    fn e8m0_exhaustive_decode_and_exact_encode() {
        let format = NarrowFloatFormat::OcpE8M0;
        for raw in 0u8..=254 {
            let expected = 2.0f64.powi(raw as i32 - 127);
            assert_eq!(decode_narrow_float(raw, format), expected);
            assert_eq!(encode_narrow_float(expected, format), Ok(raw));
        }
        assert!(decode_narrow_float(0xff, format).is_nan());
        assert_eq!(encode_narrow_float(f64::NAN, format), Ok(0xff));

        for invalid in [
            f64::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            f64::from_bits(2.0f64.powi(-127).to_bits() - 1),
            1.5,
            f64::from_bits(2.0f64.powi(127).to_bits() + 1),
            f64::INFINITY,
        ] {
            assert_eq!(
                encode_narrow_float(invalid, format),
                Err(NarrowFloatEncodeError::ValueNotRepresentable),
                "invalid value {invalid:?}"
            );
        }
    }

    #[test]
    fn int8_exhaustive_round_trip_and_ties() {
        let format = NarrowFloatFormat::OcpInt8;
        for raw in 0u8..=255 {
            let expected = (raw as i8 as f64) / 64.0;
            assert_eq!(decode_narrow_float(raw, format), expected);
            assert_eq!(encode_narrow_float(expected, format), Ok(raw));
        }

        assert_eq!(encode_narrow_float(1.0 / 128.0, format), Ok(0x00));
        assert_eq!(encode_narrow_float(3.0 / 128.0, format), Ok(0x02));
        assert_eq!(encode_narrow_float(-1.0 / 128.0, format), Ok(0x00));
        assert_eq!(encode_narrow_float(-3.0 / 128.0, format), Ok(0xfe));
        assert_eq!(encode_narrow_float(-2.0, format), Ok(0x80));
        assert_eq!(encode_narrow_float(-100.0, format), Ok(0x80));
        assert_eq!(encode_narrow_float(100.0, format), Ok(0x7f));
        assert_eq!(encode_narrow_float(f64::NEG_INFINITY, format), Ok(0x80));
        assert_eq!(encode_narrow_float(f64::INFINITY, format), Ok(0x7f));
        assert_eq!(
            encode_narrow_float(f64::NAN, format),
            Err(NarrowFloatEncodeError::NaNNotSupported)
        );
    }

    #[test]
    fn rounding_uses_binary64_bits_beyond_binary16_precision() {
        // 1.0625 is the midpoint between adjacent E4M3 values. The neighboring
        // binary64 values must land on opposite sides; an implementation which
        // first narrowed both to f16 could collapse a wider interval onto the
        // midpoint and choose the wrong even endpoint.
        let format = NarrowFloatFormat::OcpE4M3Saturate;
        assert_eq!(encode_narrow_float(next_down(1.0625), format), Ok(0x38));
        assert_eq!(encode_narrow_float(1.0625, format), Ok(0x38));
        assert_eq!(encode_narrow_float(next_up(1.0625), format), Ok(0x39));

        // Repeat at a subnormal boundary to exercise the fixed subnormal
        // quantum rather than only normal exponent handling.
        let subnormal_midpoint = 3.0 * 2.0f64.powi(-10);
        assert_eq!(
            encode_narrow_float(next_down(subnormal_midpoint), format),
            Ok(0x01)
        );
        assert_eq!(encode_narrow_float(subnormal_midpoint, format), Ok(0x02));
        assert_eq!(
            encode_narrow_float(next_up(subnormal_midpoint), format),
            Ok(0x02)
        );
    }
}
