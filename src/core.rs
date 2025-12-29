use crate::helpers::{BV, validate_index, BS, bv_from_zeros};
use crate::mutibs::Mutibs;
use crate::tibs_::Tibs;
use bitvec::prelude::*;
use half::f16;
use std::fmt;

// Trait used for commonality between the Tibs and Mutibs structs.
pub(crate) trait BitCollection: Sized {

    fn from_bv(bv: BV) -> Self;
    fn as_bitvec(&self) -> BV;
    fn as_bitvec_ref(&self) -> &BV;
    fn as_bitslice(&self) -> &BS;
    fn get_slice_unchecked(&self, start_bit: usize, length: usize) -> Self;

    fn get_raw_bytes(&self) -> Vec<u8>;

    fn raw_data(&self) -> (Vec<u8>, usize, usize) {
        let raw_bytes = self.get_raw_bytes();
        let slice = self.as_bitslice();
        let offset = match slice.domain() {
            bitvec::domain::Domain::Enclave(elem) => elem.head().into_inner() as usize,
            bitvec::domain::Domain::Region {
                head: Some(elem), ..
            } => elem.head().into_inner() as usize,
            _ => 0,
        };
        (raw_bytes, offset, self.len())
    }

    #[inline]
    fn logical_or(&self, other: &impl BitCollection) -> Self {
        debug_assert!(self.len() == other.len());

        // TODO: We only have the 150x speedup when both offsets are zero.
        let (lhs, lhs_offset, _) = self.raw_data();
        let (rhs, rhs_offset, _) = other.raw_data();

        if lhs_offset == rhs_offset {
            let data: Vec<u8> = lhs.iter().zip(rhs.iter()).map(|(&a, &b)| a | b).collect();
            let bv = BV::from_vec(data);
            Self::from_bv(bv).get_slice_unchecked(lhs_offset, self.len())
        } else {
            let mut result = self.as_bitvec().clone();
            result |= other.as_bitslice();
            Self::from_bv(result)
        }
    }

    #[inline]
    fn logical_and(&self, other: &impl BitCollection) -> Self {
        debug_assert!(self.len() == other.len());

        let (lhs, lhs_offset, _) = self.raw_data();
        let (rhs, rhs_offset, _) = other.raw_data();

        if lhs_offset == rhs_offset {
            let data: Vec<u8> = lhs.iter().zip(rhs.iter()).map(|(&a, &b)| a & b).collect();
            let bv = BV::from_vec(data);
            Self::from_bv(bv).get_slice_unchecked(lhs_offset, self.len())
        } else {
            let mut result = self.as_bitvec().clone();
            result &= other.as_bitslice();
            Self::from_bv(result)
        }
    }

    // This doesn't work for Mutibs. Not sure why yet.
    // #[inline]
    // fn logical_and(&self, other: &impl BitCollection) -> Self {
    //     debug_assert!(self.len() == other.len());
    //
    //     let (lhs, lhs_offset, _) = self.raw_data();
    //     let (rhs, rhs_offset, _) = other.raw_data();
    //     debug_assert!(lhs_offset < 8);
    //     debug_assert!(rhs_offset < 8);
    //     debug_assert_eq!(lhs.len(), (lhs_offset + self.len() + 7) / 8);
    //     debug_assert_eq!(rhs.len(), (rhs_offset + other.len() + 7) / 8);
    //
    //     let (lhs_byte_data, rhs_byte_data, new_offset) =
    //         match lhs_offset.cmp(&rhs_offset) {
    //             Ordering::Equal => {
    //                 (lhs, rhs, lhs_offset)
    //             },
    //             Ordering::Less => {
    //                 // Shift rhs to the left
    //                 let mut bv = BV::from_vec(rhs);
    //                 bv.shift_left(rhs_offset - lhs_offset);
    //                 bv.set_uninitialized(false);
    //                 (lhs, bv.into_vec(), lhs_offset)
    //             },
    //             Ordering::Greater => {
    //                 let mut bv = BV::from_vec(lhs);
    //                 bv.shift_left(lhs_offset - rhs_offset);
    //                 bv.set_uninitialized(false);
    //                 (bv.into_vec(), rhs, rhs_offset)
    //             }
    //         };
    //
    //     let data: Vec<u8> = lhs_byte_data.iter().zip(rhs_byte_data.iter()).map(|(&a, &b)| a & b).collect();
    //     debug_assert!(new_offset < 8);
    //     let data_length = data.len();
    //     debug_assert_eq!((self.len() + new_offset + 7) / 8, data.len());
    //     debug_assert!(data_length * 8 >= self.len());
    //     let bv = BV::from_vec(data);
    //     debug_assert!(bv.len() == data_length * 8);
    //     Self::new(bv).get_slice_unchecked(new_offset, self.len())
    //
    // }


    #[inline]
    fn logical_xor(&self, other: &impl BitCollection) -> Self {
        debug_assert!(self.len() == other.len());

        let (lhs, lhs_offset, _) = self.raw_data();
        let (rhs, rhs_offset, _) = other.raw_data();

        if lhs_offset == rhs_offset {
            let data: Vec<u8> = lhs.iter().zip(rhs.iter()).map(|(&a, &b)| a ^ b).collect();
            let bv = BV::from_vec(data);
            Self::from_bv(bv).get_slice_unchecked(lhs_offset, self.len())

        } else {
            let mut result = self.as_bitvec().clone();
            result ^= other.as_bitslice();
            Self::from_bv(result)
        }
    }

    #[inline]
    fn from_bytes_slice(
        data: Vec<u8>,
        offset: Option<i64>,
        length: Option<i64>,
    ) -> Result<Self, String> {
        if length.is_none() && offset.is_none() {
            return Ok(Self::from_bv(BV::from_vec(data)));
        }
        let start_bit = offset.unwrap_or(0);
        if start_bit < 0 {
            return Err(format!(
                "Cannot create using a negative offset of {start_bit}."
            ));
        }
        let start_bit = start_bit as usize;
        let data_length = data.len() * 8;
        if start_bit > data_length {
            return Err(format!(
                "Offset of {start_bit} is greater than the data length ({data_length} bits)."
            ));
        }
        let length = length.unwrap_or(data_length as i64 - start_bit as i64);
        if length < 0 {
            return Err(format!("Negative length of {length} bits provided."));
        }
        let length = length as usize;
        if start_bit + length > data_length {
            return Err(format!(
                "Length of {length} with offset of {start_bit} is greater than the data length ({data_length} bits)."
            ));
        }
        let mut bv = BV::from_vec(data);
        bv.drain(..start_bit);
        bv.truncate(length);
        Ok(Self::from_bv(bv))
    }

    fn to_string(&self) -> String {
        if self.is_empty() {
            return "".to_string();
        }
        const MAX_BITS_TO_PRINT: usize = 10000;
        const {
            assert!(MAX_BITS_TO_PRINT.is_multiple_of(4));
        }
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
            *prefix.as_bitslice() == self.as_bitslice()[..n]
        } else {
            false
        }
    }

    #[inline]
    fn empty() -> Self {
        Self::from_bv(BV::new())
    }

    fn ends_with(&self, suffix: impl BitCollection) -> bool {
        let n = suffix.len();
        if n <= self.len() {
            *suffix.as_bitslice() == self.as_bitslice()[self.len() - n..]
        } else {
            false
        }
    }

    /// Returns the bool value at a given bit index.
    #[inline]
    fn get_index(&self, bit_index: i64) -> Result<bool, String> {
        let index = validate_index(bit_index, self.len())?;
        Ok(self.as_bitslice()[index])
    }

    fn get_slice_with_step(&self, start_bit: i64, end_bit: i64, step: i64) -> Result<Self, String> {
        if step == 0 {
            return Err("Slice step cannot be zero.".to_string());
        }
        // Note that a start_bit or end_bit of -1 means to stop at the beginning when using a negative step.
        // Otherwise they should both be positive indices.
        debug_assert!(start_bit >= -1);
        debug_assert!(end_bit >= -1);
        debug_assert!(step != 0);
        if start_bit < -1 || end_bit < -1 {
            return Err("Indices less than -1 are not valid values.".to_string());
        }
        if step > 0 {
            if start_bit >= end_bit {
                return Ok(BitCollection::empty());
            }
            if end_bit as usize > self.len() {
                return Err("Slice end goes past the end of the container.".to_string());
            }
            // TODO: This alternate method might be faster
            // Ok(Self::new(
            //     self.data()[start_bit as usize..end_bit as usize]
            //         .iter()
            //         .step_by(step as usize)
            //         .collect(),
            // ))
            let mut new_data = self.as_bitslice()[start_bit as usize..end_bit as usize].to_bitvec();
            new_data.retain(|idx, _| idx % step as usize == 0);
            Ok(Self::from_bv(new_data))
        } else {
            if start_bit <= end_bit || start_bit == -1 {
                return Ok(BitCollection::empty());
            }
            if start_bit as usize > self.len() {
                return Err("Slice start bit is past the end of the container.".to_string());
            }
            // For negative step, the end_bit is inclusive, but the start_bit is exclusive.
            debug_assert!(step < 0);
            let adjusted_end_bit = (end_bit + 1) as usize;
            Ok(Self::from_bv(
                self.as_bitslice()[adjusted_end_bit..=start_bit as usize]
                    .iter()
                    .rev()
                    .step_by(-step as usize)
                    .collect(),
            ))
        }
    }

    fn count(&self, count_ones: bool) -> usize {
        let len = self.len();
        let r = self.as_bitvec_ref();
        let s = r.as_raw_slice();
        let (mut ones, raw) = (0usize, s);
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
            ones = self.as_bitslice().count_ones();
        }

        if count_ones { ones } else { len - ones }
    }

    fn multiply(&self, n: usize) -> Self {
        let len = self.len();
        if n == 0 || len == 0 {
            return BitCollection::empty();
        }
        let mut bv = BV::with_capacity(len * n);
        bv.extend_from_bitslice(self.as_bitslice());
        // TODO: This could be done more efficiently with doubling.
        for _ in 1..n {
            bv.extend_from_bitslice(self.as_bitslice());
        }
        Self::from_bv(bv)
    }

    fn lshift(&self, n: usize) -> Self {
        if n == 0 {
            return Self::from_bv(self.as_bitvec().clone());
        }
        let len = self.len();
        if n >= len {
            return Self::from_bv(bv_from_zeros(len));
        }
        let mut result_data = BV::with_capacity(len);
        result_data.extend_from_bitslice(&self.as_bitslice()[n..]);
        result_data.resize(len, false);
        Self::from_bv(result_data)
    }

    fn rshift(&self, n: usize) -> Self {
        if n == 0 {
            return Self::from_bv(self.as_bitvec().clone());
        }
        let len = self.len();
        if n >= len {
            return Self::from_bv(bv_from_zeros(len));
        }
        let mut result_data = BV::repeat(false, n);
        result_data.extend_from_bitslice(&self.as_bitslice()[..len - n]);
        Self::from_bv(result_data)
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
                    ));
                }
            }
        }
        b.set_uninitialized(false);
        Ok(Self::from_bv(b))
    }

    #[inline]
    fn from_octal(octal_string: &str) -> Result<Self, String> {
        // Ignore any leading '0o' or '0O'
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
                    ));
                }
            }
        }
        b.set_uninitialized(false);
        Ok(Self::from_bv(b))
    }

    #[inline]
    fn from_hexadecimal(hex: &str) -> Result<Self, String> {
        // Ignore any leading '0x' or '0X'
        let mut new_hex = hex
            .strip_prefix("0x")
            .or_else(|| hex.strip_prefix("0X"))
            .unwrap_or(hex)
            .to_string();
        // Remove any underscores or whitespace characters
        new_hex.retain(|c| c != '_' && !c.is_whitespace());
        let new_hex_length = new_hex.len() as i64;
        if new_hex_length % 2 != 0 {
            new_hex.push('0');
        }
        let data = match hex::decode(&new_hex) {
            Ok(d) => d,
            Err(e) => return Err(format!("Cannot convert from hex '{hex}': {}", e)),
        };
        let t = <Self as BitCollection>::from_bytes_slice(data, None, Some(new_hex_length * 4))?;
        Ok(t)
    }

    #[inline]
    fn from_u128(value: u128, length: i64) -> Result<Self, String> {
        if length <= 0 || length > 128 {
            return Err(format!(
                "Bit length for unsigned int must be between 1 and 128. Received {length}."
            ));
        }
        if value >= (1u128 << length) {
            return Err(format!("Value {value} does not fit in {length} bits."));
        }
        let mut bv = BV::repeat(false, length as usize);
        bv.store_be(value);
        Ok(Self::from_bv(bv))
    }

    #[inline]
    fn from_i128(value: i128, length: i64) -> Result<Self, String> {
        if length <= 0 || length > 128 {
            return Err(format!(
                "Bit length for signed int must be between 1 and 128. Received {length}."
            ));
        }
        let min_val = -(1i128 << (length - 1));
        let max_val = (1i128 << (length - 1)) - 1;
        if value < min_val || value > max_val {
            return Err(format!(
                "Value {value} does not fit in {length} signed bits."
            ));
        }
        let repeat_bit = value < 0;
        let mut bv = BV::repeat(repeat_bit, length as usize);
        bv.store_be(value);
        Ok(Self::from_bv(bv))
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
        Ok(Self::from_bv(bv))
    }

    #[inline]
    fn to_binary(&self) -> String {
        let mut s = String::with_capacity(self.len());
        for bit in self.as_bitslice().iter() {
            s.push(if *bit { '1' } else { '0' });
        }
        s
    }

    #[inline]
    fn to_octal(&self) -> Result<String, String> {
        let len = self.len();
        if !len.is_multiple_of(3) {
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
        if !len.is_multiple_of(4) {
            return Err(format!(
                "Cannot interpret as hex - length of {} is not a multiple of 4 bits.",
                len
            ));
        }
        Ok(self.build_hex_string())
    }

    #[inline]
    fn build_oct_string(&self) -> String {
        debug_assert!(self.len().is_multiple_of(3));
        let mut s = String::with_capacity(self.len() / 3);
        for chunk in self.as_bitslice().chunks(3) {
            let tribble = chunk.load_be::<u8>();
            let oct_char = std::char::from_digit(tribble as u32, 8).unwrap();
            s.push(oct_char);
        }
        s
    }

    #[inline]
    fn build_hex_string(&self) -> String {
        debug_assert!(self.len().is_multiple_of(4));
        let mut s = String::with_capacity(self.len() / 4);
        for chunk in self.as_bitslice().chunks(4) {
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
        if !len_bits.is_multiple_of(8) {
            return Err(format!(
                "Cannot interpret as bytes - length of {len_bits} is not a multiple of 8 bits."
            ));
        }
        match self.as_bitslice().domain() {
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
                bv.extend_from_bitslice(self.as_bitslice());
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
        padded_bv.extend_from_bitslice(self.as_bitslice());
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
        padded_bv.extend_from_bitslice(self.as_bitslice());
        Ok(padded_bv.load_be::<i128>())
    }

    fn to_f64(&self) -> Result<f64, String> {
        let length = self.len();
        match length {
            64 => {
                let bits = self.as_bitslice().load_be::<u64>();
                Ok(f64::from_bits(bits))
            }
            32 => {
                let bits = self.as_bitslice().load_be::<u32>();
                Ok(f32::from_bits(bits) as f64)
            }
            16 => {
                let bits = self.as_bitslice().load_be::<u16>();
                Ok(f16::from_bits(bits).to_f64())
            }
            _ => Err(format!(
                "Unsupported float bit length '{length}'. Only 16, 32 and 64 are supported."
            )),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.as_bitslice().is_empty()
    }

    #[inline]
    fn get_bit(&self, i: usize) -> bool {
        self.as_bitslice()[i]
    }

    #[inline]
    fn len(&self) -> usize {
        self.as_bitslice().len()
    }

    #[inline]
    fn get_slice(&self, start_bit: usize, length: usize) -> Result<Self, String> {
        if length == 0 {
            return Ok(BitCollection::empty());
        }
        if start_bit + length > self.len() {
            return Err("End bit of the slice goes past the end of the container.".to_string());
        }
        Ok(self.get_slice_unchecked(start_bit, length))
    }
}

impl BitCollection for Tibs {
    fn from_bv(bv: BV) -> Self {
        Tibs::from_bv(bv)
    }

    #[inline]
    fn as_bitvec(&self) -> BV {
        self.as_bitslice().to_bitvec()
    }

    #[inline]
    fn as_bitvec_ref(&self) -> &BV {
        Tibs::as_bitvec_ref(&self)
    }

    #[inline]
    fn as_bitslice(&self) -> &BS {
        Tibs::as_bitslice(&self)
    }

    #[inline]
    fn get_slice_unchecked(&self, start_bit: usize, length: usize) -> Self {
        Tibs::from_slice_unchecked(&self, start_bit, length)
    }

    #[inline]
    fn get_raw_bytes(&self) -> Vec<u8> {
        Tibs::raw_bytes(self)
    }


}

impl BitCollection for Mutibs {
    fn from_bv(bv: BV) -> Self {
        Mutibs::from_bv(bv)
    }

    #[inline]
    fn as_bitvec(&self) -> BV {
        Mutibs::as_bitvec_ref(&self).to_bitvec()
    }

    #[inline]
    fn as_bitvec_ref(&self) -> &BV {
        Mutibs::as_bitvec_ref(&self)
    }

    #[inline]
    fn as_bitslice(&self) -> &BS {
        Mutibs::as_bitvec_ref(&self).as_bitslice()
    }

    #[inline]
    fn get_slice_unchecked(&self, start_bit: usize, length: usize) -> Self {
        Self::from_bv(self.as_bitslice()[start_bit..start_bit + length].to_bitvec())
    }

    #[inline]
    fn get_raw_bytes(&self) -> Vec<u8> {
        self.raw_bytes()
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
        if self.len().is_multiple_of(4) {
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
        self.as_bitslice() == other.as_bitslice()
    }
}

impl PartialEq<Mutibs> for Tibs {
    #[inline]
    fn eq(&self, other: &Mutibs) -> bool {
        self.as_bitslice() == other.as_bitvec_ref()
    }
}

impl PartialEq for Mutibs {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_bitvec_ref() == other.as_bitvec_ref()
    }
}

impl PartialEq<Tibs> for Mutibs {
    #[inline]
    fn eq(&self, other: &Tibs) -> bool {
        self.as_bitvec_ref() == other.as_bitslice()
    }
}
