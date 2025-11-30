use crate::helpers::{validate_index, BV};
use crate::mutibs::Mutibs;
use crate::tibs_::Tibs;
use bitvec::bits;
use bitvec::field::BitField;
use bitvec::prelude::Lsb0;
use half::f16;
use lru::LruCache;
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Mutex;

// Trait used for commonality between the Tibs and Mutibs structs.
pub(crate) trait BitCollection: Sized {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn empty() -> Self;
    fn from_zeros(length: usize) -> Self;
    fn from_ones(length: usize) -> Self;
    fn from_bytes(data: Vec<u8>) -> Self;
    fn from_binary(binary_string: &str) -> Result<Self, String>;
    fn from_octal(octal_string: &str) -> Result<Self, String>;
    fn from_hexadecimal(hex_string: &str) -> Result<Self, String>;
    fn from_u128(value: u128, length: usize) -> Result<Self, String>;
    fn from_i128(value: i128, length: usize) -> Result<Self, String>;
    fn from_f64(value: f64, length: i64) -> Result<Self, String>;
    fn logical_or(&self, other: &Tibs) -> Self;
    fn logical_and(&self, other: &Tibs) -> Self;
    fn logical_xor(&self, other: &Tibs) -> Self;

    fn get_bit(&self, i: usize) -> bool;
    fn to_binary(&self) -> String;
    fn to_octal(&self) -> Result<String, String>;
    fn to_hexadecimal(&self) -> Result<String, String>;
    fn to_byte_data(&self) -> Result<Vec<u8>, String>;
    fn to_u128(&self) -> Result<u128, String>;
    fn to_i128(&self) -> Result<i128, String>;
    fn to_f64(&self) -> Result<f64, String>;
    /// Return bytes that can easily be converted to an int in Python
    fn to_int_byte_data(&self, signed: bool) -> Vec<u8>;
}

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
        let x = string_literal_to_mutibs(&token)?;
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
            result.extend_from_bitslice(&bits.inner.data);
        }
        Mutibs::new(result)
    };
    // Update cache with new result
    {
        let mut cache = BITS_CACHE.lock().unwrap();
        cache.put(s, result.inner.data.clone());
    }
    Ok(result)
}

impl BitCollection for Tibs {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    fn empty() -> Self {
        Tibs::new(BV::new())
    }

    #[inline]
    fn from_zeros(length: usize) -> Self {
        Tibs::new(BV::repeat(false, length))
    }

    #[inline]
    fn from_ones(length: usize) -> Self {
        Tibs::new(BV::repeat(true, length))
    }

    #[inline]
    fn from_bytes(data: Vec<u8>) -> Self {
        let bv = BV::from_vec(data);
        Tibs::new(bv)
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
        Ok(Tibs::new(b))
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
        Ok(Tibs::new(b))
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
        let mut bv = <Tibs as BitCollection>::from_bytes(data).data;
        if is_odd_length {
            bv.drain(bv.len() - 4..bv.len());
        }
        Ok(Tibs::new(bv))
    }

    #[inline]
    fn from_u128(value: u128, length: usize) -> Result<Self, String> {
        if length == 0 || length > 128 {
            return Err(format!(
                "Bit length for unsigned int must be between 1 and 128. Received {length}."
            ));
        }
        if length < 128 && value >= (1u128 << length) {
            return Err(format!("Value {value} does not fit in {length} bits."));
        }
        let mut bv = BV::repeat(false, length);
        bv.store_be(value);
        Ok(Tibs::new(bv))
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
        padded_bv.extend_from_bitslice(&self.data);
        Ok(padded_bv.load_be::<u128>())
    }

    #[inline]
    fn from_i128(value: i128, length: usize) -> Result<Self, String> {
        if length == 0 || length > 128 {
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
        let repeat_bit = if value < 0 { true } else { false };
        let mut bv = BV::repeat(repeat_bit, length as usize);
        bv.store_be(value);
        Ok(Tibs::new(bv))
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
        let pad_bit = if self.get_bit(0) { true } else { false };
        padded_bv.resize(padding, pad_bit);
        padded_bv.extend_from_bitslice(&self.data);
        Ok(padded_bv.load_be::<i128>())
    }

    #[inline]
    fn logical_or(&self, other: &Tibs) -> Self {
        debug_assert!(self.len() == other.len());
        let mut result = self.data.clone();
        result |= &other.data;
        Tibs::new(result)
    }

    #[inline]
    fn logical_and(&self, other: &Tibs) -> Self {
        debug_assert!(self.len() == other.len());
        let mut result = self.data.clone();
        result &= &other.data;
        Tibs::new(result)
    }

    #[inline]
    fn logical_xor(&self, other: &Tibs) -> Self {
        debug_assert!(self.len() == other.len());
        let mut result = self.data.clone();
        result ^= &other.data;
        Tibs::new(result)
    }

    #[inline]
    fn get_bit(&self, i: usize) -> bool {
        self.data[i]
    }

    #[inline]
    fn to_binary(&self) -> String {
        self.build_bin_string()
        // self.bin_cache.get_or_init(|| {
        //     self.build_bin_string()
        // }).clone()
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
        // Ok(self.oct_cache.get_or_init(|| {
        //     self.build_oct_string()
        // }).clone())
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
        // Ok(self.hex_cache.get_or_init(|| {
        //     self.build_hex_string()
        // }).clone())
    }

    #[inline]
    fn to_byte_data(&self) -> Result<Vec<u8>, String> {
        if self.data.is_empty() {
            return Ok(Vec::new());
        }
        let len_bits = self.len();
        if len_bits % 8 != 0 {
            return Err(format!(
                "Cannot interpret as bytes - length of {len_bits} is not a multiple of 8 bits."
            ));
        }
        match self.data.as_bitslice().domain() {
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
                bv.extend_from_bitslice(&self.data);
                let new_len = (len_bits + 7) & !7;
                bv.resize(new_len, false);
                Ok(bv.into_vec())
            }
        }
    }

    fn to_int_byte_data(&self, signed: bool) -> Vec<u8> {
        if self.is_empty() {
            return Vec::new();
        }

        let needed_bits = (self.len() + 7) & !7;
        let mut bv = BV::with_capacity(needed_bits);

        let sign_bit = signed && self.data[0];
        let padding = needed_bits - self.len();

        for _ in 0..padding {
            bv.push(sign_bit);
        }
        bv.extend_from_bitslice(&self.data);

        bv.into_vec()
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
        Ok(Tibs::new(bv))
    }

    fn to_f64(&self) -> Result<f64, String> {
        let length = self.len();
        match length {
            64 => {
                let bits = self.data.load_be::<u64>();
                Ok(f64::from_bits(bits))
            }
            32 => {
                let bits = self.data.load_be::<u32>();
                Ok(f32::from_bits(bits) as f64)
            }
            16 => {
                let bits = self.data.load_be::<u16>();
                Ok(f16::from_bits(bits).to_f64())
            }
            _ => Err(format!(
                "Unsupported float bit length '{length}'. Only 16, 32 and 64 are supported."
            )),
        }
    }
}

impl BitCollection for Mutibs {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    fn empty() -> Self {
        Self {
            inner: <Tibs as BitCollection>::empty(),
        }
    }

    #[inline]
    fn from_zeros(length: usize) -> Self {
        Self {
            inner: <Tibs as BitCollection>::from_zeros(length),
        }
    }

    #[inline]
    fn from_ones(length: usize) -> Self {
        Self {
            inner: <Tibs as BitCollection>::from_ones(length),
        }
    }

    #[inline]
    fn from_bytes(data: Vec<u8>) -> Self {
        Self {
            inner: <Tibs as BitCollection>::from_bytes(data),
        }
    }

    #[inline]
    fn from_binary(binary_string: &str) -> Result<Self, String> {
        Ok(Self {
            inner: <Tibs as BitCollection>::from_binary(binary_string)?,
        })
    }

    #[inline]
    fn from_octal(oct: &str) -> Result<Self, String> {
        Ok(Self {
            inner: <Tibs as BitCollection>::from_octal(oct)?,
        })
    }

    #[inline]
    fn from_hexadecimal(hex: &str) -> Result<Self, String> {
        Ok(Self {
            inner: <Tibs as BitCollection>::from_hexadecimal(hex)?,
        })
    }

    #[inline]
    fn from_u128(value: u128, length: usize) -> Result<Self, String> {
        Ok(Self {
            inner: <Tibs as BitCollection>::from_u128(value, length)?,
        })
    }

    #[inline]
    fn from_i128(value: i128, length: usize) -> Result<Self, String> {
        Ok(Self {
            inner: <Tibs as BitCollection>::from_i128(value, length)?,
        })
    }

    #[inline]
    fn to_u128(&self) -> Result<u128, String> {
        self.inner.to_u128()
    }

    #[inline]
    fn to_i128(&self) -> Result<i128, String> {
        self.inner.to_i128()
    }

    #[inline]
    fn logical_or(&self, other: &Tibs) -> Self {
        Self {
            inner: self.inner.logical_or(other),
        }
    }

    #[inline]
    fn logical_and(&self, other: &Tibs) -> Self {
        Self {
            inner: self.inner.logical_and(other),
        }
    }

    #[inline]
    fn logical_xor(&self, other: &Tibs) -> Self {
        Self {
            inner: self.inner.logical_xor(other),
        }
    }

    #[inline]
    fn get_bit(&self, i: usize) -> bool {
        self.inner.data[i]
    }

    #[inline]
    fn to_binary(&self) -> String {
        self.inner.to_binary()
    }

    #[inline]
    fn to_octal(&self) -> Result<String, String> {
        self.inner.to_octal()
    }

    #[inline]
    fn to_hexadecimal(&self) -> Result<String, String> {
        self.inner.to_hexadecimal()
    }

    #[inline]
    fn to_byte_data(&self) -> Result<Vec<u8>, String> {
        self.inner.to_byte_data()
    }

    fn to_int_byte_data(&self, signed: bool) -> Vec<u8> {
        self.inner.to_int_byte_data(signed)
    }

    fn from_f64(value: f64, length: i64) -> Result<Self, String> {
        Ok(Self {
            inner: <Tibs as BitCollection>::from_f64(value, length)?,
        })
    }

    fn to_f64(&self) -> Result<f64, String> {
        self.inner.to_f64()
    }
}

impl fmt::Debug for Tibs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len() > 100 {
            return f
                .debug_struct("Tibs")
                .field("hex", &self.slice(0, 100).to_hex().unwrap())
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
        self.data == other.data
    }
}

impl PartialEq<Mutibs> for Tibs {
    #[inline]
    fn eq(&self, other: &Mutibs) -> bool {
        self.data == other.inner.data
    }
}

impl PartialEq for Mutibs {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner.data == other.inner.data
    }
}

impl PartialEq<Tibs> for Mutibs {
    #[inline]
    fn eq(&self, other: &Tibs) -> bool {
        self.inner.data == other.data
    }
}

// ---- Tibs private helper methods. Not part of the Python interface. ----

impl Tibs {
    pub(crate) fn new(bv: BV) -> Self {
        Tibs { data: bv }
    }

    /// Slice used internally without bounds checking.
    pub(crate) fn slice(&self, start_bit: usize, length: usize) -> Self {
        Tibs::new(self.data[start_bit..start_bit + length].to_bitvec())
    }

    #[inline]
    fn build_bin_string(&self) -> String {
        let mut s = String::with_capacity(self.len());
        for bit in self.data.iter() {
            s.push(if *bit { '1' } else { '0' });
        }
        s
    }

    #[inline]
    fn build_oct_string(&self) -> String {
        debug_assert!(self.len() % 3 == 0);
        let mut s = String::with_capacity(self.len() / 3);
        for chunk in self.data.chunks(3) {
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
        for chunk in self.data.chunks(4) {
            let nibble = chunk.load_be::<u8>();
            let hex_char = std::char::from_digit(nibble as u32, 16).unwrap();
            s.push(hex_char);
        }
        s
    }
}

pub(crate) fn validate_logical_op_lengths(a: usize, b: usize) -> PyResult<()> {
    if a != b {
        Err(PyValueError::new_err(format!("For logical operations the lengths of both objects must match. Received lengths of {a} and {b} bits.")))
    } else {
        Ok(())
    }
}

impl Mutibs {
    pub fn new(bv: BV) -> Self {
        Self {
            inner: Tibs::new(bv),
        }
    }

    pub fn _set_from_sequence(&mut self, value: bool, indices: Vec<i64>) -> PyResult<()> {
        for idx in indices {
            let pos: usize = validate_index(idx, self.inner.len())?;
            self.inner.data.set(pos, value);
        }
        Ok(())
    }
}
