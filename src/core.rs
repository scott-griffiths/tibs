use crate::helpers::{validate_index, BV};
use crate::mutibs::Mutibs;
use crate::tibs_::Tibs;
use bitvec::prelude::*;
use half::f16;
use lru::LruCache;
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Mutex;

// ---- Rust-only helper methods ----

// Define a static LRU cache.
const BITS_CACHE_SIZE: usize = 1024;
static BITS_CACHE: Lazy<Mutex<LruCache<String, BV>>> =
    Lazy::new(|| Mutex::new(LruCache::new(NonZeroUsize::new(BITS_CACHE_SIZE).unwrap())));

fn string_literal_to_mutibs(s: &str) -> PyResult<Mutibs> {
    match s.get(0..2).map(|p| p.to_ascii_lowercase()).as_deref() {
        Some("0b") => Ok(BitCollection::from_binary(s).map_err(PyValueError::new_err)?),
        Some("0x") => Ok(BitCollection::from_hexadecimal(s).map_err(PyValueError::new_err)?),
        Some("0o") => Ok(BitCollection::from_octal(s).map_err(PyValueError::new_err)?),
        _ => Err(PyValueError::new_err(format!(
            "Can't parse token '{s}'. Did you mean to prefix with '0x', '0b' or '0o'?"
        ))),
    }
}

pub(crate) fn str_to_mutibs(s: String) -> PyResult<Mutibs> {
    // Check cache first
    {
        let mut cache = BITS_CACHE.lock().unwrap();
        if let Some(cached_data) = cache.get(&s) {
            return Ok(Mutibs::new(cached_data.clone()));
        }
    }
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let tokens = s.split(',');
    let mut bits_array = Vec::<Mutibs>::new();
    let mut total_bit_length = 0;
    for token in tokens {
        if token.is_empty() {
            continue;
        }
        let x = string_literal_to_mutibs(token)?;
        total_bit_length += x.len();
        bits_array.push(x);
    }
    if bits_array.is_empty() {
        return Ok(BitCollection::empty());
    }
    // Combine all bits
    let result = if bits_array.len() == 1 {
        bits_array.pop().unwrap()
    } else {
        let mut result = BV::with_capacity(total_bit_length);
        for bits in bits_array {
            result.extend_from_bitslice(bits.data());
        }
        Mutibs::new(result)
    };
    // Update cache with new result
    {
        let mut cache = BITS_CACHE.lock().unwrap();
        cache.put(s, result.data().clone());
    }
    Ok(result)
}

// Trait used for commonality between the Tibs and Mutibs structs.
pub(crate) trait BitCollection: Sized {
    #[inline]
    fn logical_or(&self, other: &impl BitCollection) -> Self {
        debug_assert!(self.len() == other.len());
        let mut result = self.data().clone();
        result |= other.data();
        Self::new(result)
    }

    #[inline]
    fn logical_and(&self, other: &impl BitCollection) -> Self {
        debug_assert!(self.len() == other.len());
        let mut result = self.data().clone();
        result &= other.data();
        Self::new(result)
    }

    #[inline]
    fn logical_xor(&self, other: &impl BitCollection) -> Self {
        debug_assert!(self.len() == other.len());
        let mut result = self.data().clone();
        result ^= other.data();
        Self::new(result)
    }

    #[inline]
    fn from_zeros(length: usize) -> Self {
        Self::new(BV::repeat(false, length))
    }

    #[inline]
    fn from_ones(length: usize) -> Self {
        Self::new(BV::repeat(true, length))
    }

    #[inline]
    fn from_bytes(data: Vec<u8>) -> Self {
        let bv = BV::from_vec(data);
        Self::new(bv)
    }
    fn to_string(&self) -> String {
        if self.is_empty() {
            return "".to_string();
        }
        const MAX_BITS_TO_PRINT: usize = 10000;
        debug_assert!(MAX_BITS_TO_PRINT % 4 == 0);
        if self.len() <= MAX_BITS_TO_PRINT {
            match self.to_hexadecimal() {
                Ok(hex) => format!("0x{}", hex),
                Err(_) => format!("0b{}", self.to_binary()),
            }
        } else {
            format!(
                "0x{}... # length={}",
                self.get_slice_unchecked(0, MAX_BITS_TO_PRINT)
                    .to_hexadecimal()
                    .unwrap(),
                self.len()
            )
        }
    }

    fn starts_with(&self, prefix: impl BitCollection) -> bool {
        let n = prefix.len();
        if n <= self.len() {
            *prefix.data() == self.data()[..n]
        } else {
            false
        }
    }

    #[inline]
    fn empty() -> Self {
        Self::new(BV::new())
    }

    fn ends_with(&self, suffix: impl BitCollection) -> bool {
        let n = suffix.len();
        if n <= self.len() {
            *suffix.data() == self.data()[self.len() - n..]
        } else {
            false
        }
    }

    fn new(bv: BV) -> Self;

    /// Returns the bool value at a given bit index.
    #[inline]
    fn get_index(&self, bit_index: i64) -> PyResult<bool> {
        let index = validate_index(bit_index, self.len())?;
        Ok(self.data()[index])
    }

    fn getslice_with_step(&self, start_bit: i64, end_bit: i64, step: i64) -> PyResult<Self> {
        if step == 0 {
            return Err(PyValueError::new_err("Slice step cannot be zero."));
        }
        // Note that a start_bit or end_bit of -1 means to stop at the beginning when using a negative step.
        // Otherwise they should both be positive indices.
        debug_assert!(start_bit >= -1);
        debug_assert!(end_bit >= -1);
        debug_assert!(step != 0);
        if start_bit < -1 || end_bit < -1 {
            return Err(PyValueError::new_err(
                "Indices less than -1 are not valid values.",
            ));
        }
        if step > 0 {
            if start_bit >= end_bit {
                return Ok(BitCollection::empty());
            }
            if end_bit as usize > self.len() {
                return Err(PyValueError::new_err(
                    "Slice end goes past the end of the container.",
                ));
            }
            Ok(Self::new(
                self.data()[start_bit as usize..end_bit as usize]
                    .iter()
                    .step_by(step as usize)
                    .collect(),
            ))
        } else {
            if start_bit <= end_bit || start_bit == -1 {
                return Ok(BitCollection::empty());
            }
            if start_bit as usize > self.len() {
                return Err(PyValueError::new_err(
                    "Slice start bit is past the end of the container.",
                ));
            }
            // For negative step, the end_bit is inclusive, but the start_bit is exclusive.
            debug_assert!(step < 0);
            let adjusted_end_bit = (end_bit + 1) as usize;
            Ok(Self::new(
                self.data()[adjusted_end_bit..=start_bit as usize]
                    .iter()
                    .rev()
                    .step_by(-step as usize)
                    .collect(),
            ))
        }
    }

    // Unchecked version
    fn get_slice_unchecked(&self, start_bit: usize, length: usize) -> Self {
        Self::new(self.data()[start_bit..start_bit + length].to_bitvec())
    }

    // Checked version
    fn get_slice(&self, start_bit: usize, length: usize) -> PyResult<Self> {
        if length == 0 {
            return Ok(BitCollection::empty());
        }
        if start_bit + length > self.len() {
            return Err(PyValueError::new_err(
                "End bit of the slice goes past the end of the container.",
            ));
        }
        Ok(self.get_slice_unchecked(start_bit, length))
    }

    fn count(&self, count_ones: bool) -> usize {
        let len = self.len();

        let (mut ones, raw) = (0usize, self.data().as_raw_slice());
        if let Ok(words) = bytemuck::try_cast_slice::<u8, usize>(raw) {
            // Considerable speed increase by casting data to usize if possible.
            for word in words {
                ones += word.count_ones() as usize;
            }
            let used_bits = words.len() * usize::BITS as usize;
            if used_bits > len {
                let extra = used_bits - len;
                if let Some(last) = words.last() {
                    ones -= (last & (!0usize >> extra)).count_ones() as usize;
                }
            }
        } else {
            // Fallback to library method
            ones = self.data().count_ones();
        }

        if count_ones {
            ones
        } else {
            len - ones
        }
    }

    fn multiply(&self, n: usize) -> Self {
        let len = self.len();
        if n == 0 || len == 0 {
            return BitCollection::empty();
        }
        let mut bv = BV::with_capacity(len * n);
        bv.extend_from_bitslice(self.data());
        // TODO: This could be done more efficiently with doubling.
        for _ in 1..n {
            bv.extend_from_bitslice(self.data());
        }
        Self::new(bv)
    }

    fn lshift(&self, n: usize) -> Self {
        if n == 0 {
            return Self::new(self.data().clone());
        }
        let len = self.len();
        if n >= len {
            return BitCollection::from_zeros(len);
        }
        let mut result_data = BV::with_capacity(len);
        result_data.extend_from_bitslice(&self.data()[n..]);
        result_data.resize(len, false);
        Self::new(result_data)
    }

    fn rshift(&self, n: usize) -> Self {
        if n == 0 {
            return Self::new(self.data().clone());
        }
        let len = self.len();
        if n >= len {
            return BitCollection::from_zeros(len);
        }
        let mut result_data = BV::repeat(false, n);
        result_data.extend_from_bitslice(&self.data()[..len - n]);
        Self::new(result_data)
    }

    fn data(&self) -> &BV;

    /// Bit-wise 'and' between two Tibs. Returns new Tibs.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    fn and(&self, other: &impl BitCollection) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_and(self, other))
    }

    /// Bit-wise 'or' between two Tibs. Returns new Tibs.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    fn or(&self, other: &impl BitCollection) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_or(self, other))
    }

    /// Bit-wise 'xor' between two Tibs. Returns new Tibs.
    ///
    /// Raises ValueError if the two Tibs have differing lengths.
    fn xor(&self, other: &impl BitCollection) -> PyResult<Self> {
        validate_logical_op_lengths(self.len(), other.len())?;
        Ok(BitCollection::logical_xor(self, other))
    }

    #[inline]
    fn from_binary(binary_string: &str) -> Result<Self, String> {
        // Ignore any leading '0b' or '0B'
        let s = binary_string
            .strip_prefix("0b")
            .or_else(|| binary_string.strip_prefix("0B"))
            .unwrap_or(binary_string);
        let mut b: BV = BV::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '0' => b.push(false),
                '1' => b.push(true),
                '_' => continue,
                c if c.is_whitespace() => continue,
                _ => {
                    return Err(format!(
                        "Cannot convert from bin '{binary_string}: Invalid character '{c}'."
                    ))
                }
            }
        }
        b.set_uninitialized(false);
        Ok(Self::new(b))
    }

    #[inline]
    fn from_octal(octal_string: &str) -> Result<Self, String> {
        // Ignore any leading '0o'
        let s = octal_string
            .strip_prefix("0o")
            .or_else(|| octal_string.strip_prefix("0O"))
            .unwrap_or(octal_string);
        let mut b: BV = BV::with_capacity(s.len() * 3);
        for c in s.chars() {
            match c {
                '0' => b.extend_from_bitslice(bits![0, 0, 0]),
                '1' => b.extend_from_bitslice(bits![0, 0, 1]),
                '2' => b.extend_from_bitslice(bits![0, 1, 0]),
                '3' => b.extend_from_bitslice(bits![0, 1, 1]),
                '4' => b.extend_from_bitslice(bits![1, 0, 0]),
                '5' => b.extend_from_bitslice(bits![1, 0, 1]),
                '6' => b.extend_from_bitslice(bits![1, 1, 0]),
                '7' => b.extend_from_bitslice(bits![1, 1, 1]),
                '_' => continue,
                c if c.is_whitespace() => continue,
                _ => {
                    return Err(format!(
                        "Cannot convert from oct '{octal_string}': Invalid character '{c}'."
                    ))
                }
            }
        }
        Ok(Self::new(b))
    }

    #[inline]
    fn from_hexadecimal(hex: &str) -> Result<Self, String> {
        // Ignore any leading '0x'
        let mut new_hex = hex
            .strip_prefix("0x")
            .or_else(|| hex.strip_prefix("0X"))
            .unwrap_or(hex)
            .to_string();
        // Remove any underscores or whitespace characters
        new_hex.retain(|c| c != '_' && !c.is_whitespace());
        let is_odd_length: bool = new_hex.len() % 2 != 0;
        if is_odd_length {
            new_hex.push('0');
        }
        let data = match hex::decode(new_hex) {
            Ok(d) => d,
            Err(e) => return Err(format!("Cannot convert from hex '{hex}': {}", e)),
        };
        let mut bv = <Self as BitCollection>::from_bytes(data).data().clone();
        if is_odd_length {
            bv.drain(bv.len() - 4..bv.len());
        }
        Ok(Self::new(bv))
    }

    #[inline]
    fn from_u128(value: u128, length: i64) -> Result<Self, String> {
        if length <= 0 || length > 128 {
            return Err(format!(
                "Bit length for unsigned int must be between 1 and 128. Received {length}."
            ));
        }
        if length < 128 && value >= (1u128 << length) {
            return Err(format!("Value {value} does not fit in {length} bits."));
        }
        let mut bv = BV::repeat(false, length as usize);
        bv.store_be(value);
        Ok(Self::new(bv))
    }

    #[inline]
    fn from_i128(value: i128, length: i64) -> Result<Self, String> {
        if length <= 0 || length > 128 {
            return Err(format!(
                "Bit length for signed int must be between 1 and 128. Received {length}."
            ));
        }
        if length < 128 {
            let min_val = -(1i128 << (length - 1));
            let max_val = (1i128 << (length - 1)) - 1;
            if value < min_val || value > max_val {
                return Err(format!(
                    "Value {value} does not fit in {length} signed bits."
                ));
            }
        }
        let repeat_bit = value < 0;
        let mut bv = BV::repeat(repeat_bit, length as usize);
        bv.store_be(value);
        Ok(Self::new(bv))
    }

    fn from_f64(value: f64, length: i64) -> Result<Self, String> {
        let bv = match length {
            64 => {
                let mut bv = BV::repeat(false, 64);
                bv.store_be(value.to_bits());
                bv
            }
            32 => {
                let value_f32 = value as f32;
                let mut bv = BV::repeat(false, 32);
                bv.store_be(value_f32.to_bits());
                bv
            }
            16 => {
                let value_f16 = f16::from_f64(value);
                let mut bv = BV::repeat(false, 16);
                bv.store_be(value_f16.to_bits());
                bv
            }
            _ => {
                return Err(format!(
                    "Unsupported float bit length '{length}'. Only 16, 32 and 64 are supported."
                ));
            }
        };
        Ok(Self::new(bv))
    }

    #[inline]
    fn to_binary(&self) -> String {
        let mut s = String::with_capacity(self.len());
        for bit in self.data().iter() {
            s.push(if *bit { '1' } else { '0' });
        }
        s
    }

    #[inline]
    fn to_octal(&self) -> Result<String, String> {
        let len = self.len();
        if len % 3 != 0 {
            return Err(format!(
                "Cannot interpret as octal - length of {} is not a multiple of 3 bits.",
                len
            ));
        }
        Ok(self.build_oct_string())
    }

    #[inline]
    fn to_hexadecimal(&self) -> Result<String, String> {
        let len = self.len();
        if len % 4 != 0 {
            return Err(format!(
                "Cannot interpret as hex - length of {} is not a multiple of 4 bits.",
                len
            ));
        }
        Ok(self.build_hex_string())
    }

    #[inline]
    fn build_oct_string(&self) -> String {
        debug_assert!(self.len() % 3 == 0);
        let mut s = String::with_capacity(self.len() / 3);
        for chunk in self.data().chunks(3) {
            let tribble = chunk.load_be::<u8>();
            let oct_char = std::char::from_digit(tribble as u32, 8).unwrap();
            s.push(oct_char);
        }
        s
    }

    #[inline]
    fn build_hex_string(&self) -> String {
        debug_assert!(self.len() % 4 == 0);
        let mut s = String::with_capacity(self.len() / 4);
        for chunk in self.data().chunks(4) {
            let nibble = chunk.load_be::<u8>();
            let hex_char = std::char::from_digit(nibble as u32, 16).unwrap();
            s.push(hex_char);
        }
        s
    }

    #[inline]
    fn to_byte_data(&self) -> Result<Vec<u8>, String> {
        if self.is_empty() {
            return Ok(Vec::new());
        }
        let len_bits = self.len();
        if len_bits % 8 != 0 {
            return Err(format!(
                "Cannot interpret as bytes - length of {len_bits} is not a multiple of 8 bits."
            ));
        }
        match self.data().as_bitslice().domain() {
            // Fast path: element-aligned and length is a multiple of 8
            bitvec::domain::Domain::Region {
                head: None,
                body,
                tail: None,
            } => {
                // Already byte-aligned; copy the bytes directly.
                Ok(body.to_vec())
            }
            // Misaligned: repack by extending from the bitslice
            _ => {
                let mut bv = BV::with_capacity(len_bits);
                bv.extend_from_bitslice(self.data());
                let new_len = (len_bits + 7) & !7;
                bv.resize(new_len, false);
                Ok(bv.into_vec())
            }
        }
    }

    #[inline]
    fn to_u128(&self) -> Result<u128, String> {
        let length = self.len();
        if length > 128 {
            return Err(format!(
                "Bit length to convert to unsigned int must be between 1 and 128. Received {length}."
            ));
        }
        let mut padded_bv = BV::new();
        let padding = 128 - length;
        padded_bv.resize(padding, false);
        padded_bv.extend_from_bitslice(self.data());
        Ok(padded_bv.load_be::<u128>())
    }

    #[inline]
    fn to_i128(&self) -> Result<i128, String> {
        let length = self.len();
        if length > 128 {
            return Err(format!(
                "Bit length to convert to unsigned int must be between 1 and 128. Received {length}."
            ));
        }
        let mut padded_bv = BV::new();
        let padding = 128 - length;
        let pad_bit = self.get_bit(0);
        padded_bv.resize(padding, pad_bit);
        padded_bv.extend_from_bitslice(self.data());
        Ok(padded_bv.load_be::<i128>())
    }

    fn to_f64(&self) -> Result<f64, String> {
        let length = self.len();
        match length {
            64 => {
                let bits = self.data().load_be::<u64>();
                Ok(f64::from_bits(bits))
            }
            32 => {
                let bits = self.data().load_be::<u32>();
                Ok(f32::from_bits(bits) as f64)
            }
            16 => {
                let bits = self.data().load_be::<u16>();
                Ok(f16::from_bits(bits).to_f64())
            }
            _ => Err(format!(
                "Unsupported float bit length '{length}'. Only 16, 32 and 64 are supported."
            )),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.data().is_empty()
    }

    #[inline]
    fn get_bit(&self, i: usize) -> bool {
        self.data()[i]
    }

    #[inline]
    fn len(&self) -> usize {
        self.data().len()
    }
}

impl BitCollection for Tibs {
    fn new(bv: BV) -> Self {
        Self::new_from_bv(bv)
    }

    #[inline]
    fn data(&self) -> &BV {
        self.data_to_bv()
    }
}

impl BitCollection for Mutibs {
    fn new(bv: BV) -> Self {
        Self::new_from_bv(bv)
    }

    #[inline]
    fn data(&self) -> &BV {
        self.data_to_bv()
    }
}

impl fmt::Debug for Tibs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len() > 100 {
            return f
                .debug_struct("Tibs")
                .field("hex", &self.get_slice_unchecked(0, 100).to_hex().unwrap())
                .field("length", &self.len())
                .finish();
        }
        if self.len() % 4 == 0 {
            return f
                .debug_struct("Tibs")
                .field("hex", &self.to_hex().unwrap())
                .field("length", &self.len())
                .finish();
        }
        f.debug_struct("Tibs")
            .field("bin", &self.to_bin())
            .field("length", &self.len())
            .finish()
    }
}

impl PartialEq for Tibs {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data()
    }
}

impl PartialEq<Mutibs> for Tibs {
    #[inline]
    fn eq(&self, other: &Mutibs) -> bool {
        self.data() == other.data()
    }
}

impl PartialEq for Mutibs {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data()
    }
}

impl PartialEq<Tibs> for Mutibs {
    #[inline]
    fn eq(&self, other: &Tibs) -> bool {
        self.data() == other.data()
    }
}

pub(crate) fn validate_logical_op_lengths(a: usize, b: usize) -> PyResult<()> {
    if a != b {
        Err(PyValueError::new_err(format!("For logical operations the lengths of both objects must match. Received lengths of {a} and {b} bits.")))
    } else {
        Ok(())
    }
}
