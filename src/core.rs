use crate::helpers::{BS, BV, bv_from_zeros, validate_index};
use crate::mutibs::Mutibs;
use crate::tibs_::Tibs;
use bitvec::prelude::*;
use half::f16;
use pyo3::PyResult;
use pyo3::exceptions::{PyIndexError, PyValueError};
use std::fmt;

// Trait used for commonality between the Tibs and Mutibs structs.
pub(crate) trait BitCollection: Sized + Clone {
    fn from_bv(bv: BV, msb0: bool) -> Self;
    fn msb0(&self) -> bool;
    fn to_bitvec(&self) -> BV;
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
            Self::from_bv(bv, self.msb0()).get_slice_unchecked(lhs_offset, self.len())
        } else {
            let mut result = self.to_bitvec();
            result |= other.as_bitslice();
            Self::from_bv(result, self.msb0())
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
            Self::from_bv(bv, self.msb0()).get_slice_unchecked(lhs_offset, self.len())
        } else {
            let mut result = self.to_bitvec();
            result &= other.as_bitslice();
            Self::from_bv(result, self.msb0())
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
    //     let (lhs_byte_data, rhs_byte_data, new_offset) = match lhs_offset.cmp(&rhs_offset) {
    //         Ordering::Equal => (lhs, rhs, lhs_offset),
    //         Ordering::Less => {
    //             // Shift rhs to the left
    //             let mut bv = BV::from_vec(rhs);
    //             bv.shift_left(rhs_offset - lhs_offset);
    //             bv.set_uninitialized(false);
    //             (lhs, bv.into_vec(), lhs_offset)
    //         }
    //         Ordering::Greater => {
    //             let mut bv = BV::from_vec(lhs);
    //             bv.shift_left(lhs_offset - rhs_offset);
    //             bv.set_uninitialized(false);
    //             (bv.into_vec(), rhs, rhs_offset)
    //         }
    //     };
    //
    //     let data: Vec<u8> = lhs_byte_data
    //         .iter()
    //         .zip(rhs_byte_data.iter())
    //         .map(|(&a, &b)| a & b)
    //         .collect();
    //     debug_assert!(new_offset < 8);
    //     let data_length = data.len();
    //     debug_assert_eq!((self.len() + new_offset + 7) / 8, data.len());
    //     debug_assert!(data_length * 8 >= self.len());
    //     let bv = BV::from_vec(data);
    //     debug_assert!(bv.len() == data_length * 8);
    //     Self::from_bv(bv).get_slice_unchecked(new_offset, self.len())
    // }

    #[inline]
    fn logical_xor(&self, other: &impl BitCollection) -> Self {
        debug_assert!(self.len() == other.len());

        let (lhs, lhs_offset, _) = self.raw_data();
        let (rhs, rhs_offset, _) = other.raw_data();

        if lhs_offset == rhs_offset {
            let data: Vec<u8> = lhs.iter().zip(rhs.iter()).map(|(&a, &b)| a ^ b).collect();
            let bv = BV::from_vec(data);
            Self::from_bv(bv, self.msb0()).get_slice_unchecked(lhs_offset, self.len())
        } else {
            let mut result = self.to_bitvec();
            result ^= other.as_bitslice();
            Self::from_bv(result, self.msb0())
        }
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
            if self.msb0() {
                *prefix.as_bitslice() == self.as_bitslice()[..n]
            } else {
                *prefix.as_bitslice() == self.as_bitslice()[self.len() - n..]
            }
        } else {
            false
        }
    }

    #[inline]
    fn empty(msb0: bool) -> Self {
        Self::from_bv(BV::new(), msb0)
    }

    fn ends_with(&self, suffix: impl BitCollection) -> bool {
        let n = suffix.len();
        if n <= self.len() {
            if self.msb0() {
                *suffix.as_bitslice() == self.as_bitslice()[self.len() - n..]
            } else {
                *suffix.as_bitslice() == self.as_bitslice()[..n]
            }
        } else {
            false
        }
    }

    /// Returns the bool value at a given bit index.
    #[inline]
    fn get_index(&self, bit_index: i64) -> PyResult<bool> {
        let index = validate_index(bit_index, self.len(), self.msb0())?;
        Ok(self.as_bitslice()[index])
    }

    fn get_slice_with_step(&self, start_bit: i64, end_bit: i64, step: i64) -> PyResult<Self> {
        if step == 0 {
            return Err(PyValueError::new_err(
                "Slice step cannot be zero.".to_string(),
            ));
        }
        // Note that a start_bit or end_bit of -1 means to stop at the beginning when using a negative step.
        // Otherwise they should both be positive indices.
        debug_assert!(start_bit >= -1);
        debug_assert!(end_bit >= -1);
        debug_assert!(step != 0);
        if start_bit < -1 || end_bit < -1 {
            return Err(PyValueError::new_err(
                "Indices less than -1 are not valid values.".to_string(),
            ));
        }
        if step > 0 {
            if start_bit >= end_bit {
                return Ok(BitCollection::empty(self.msb0()));
            }
            if end_bit as usize > self.len() {
                return Err(PyValueError::new_err(
                    "Slice end goes past the end of the container.".to_string(),
                ));
            }
            if self.msb0() {
                Ok(Self::from_bv(
                    self.as_bitslice()[start_bit as usize..end_bit as usize]
                        .iter()
                        .step_by(step as usize)
                        .collect(),
                    true,
                ))
            } else {
                // For lsb0 we step in reverse and then reverse the
                // collected bits.
                let lsb0_start = self.len() - end_bit as usize;
                let lsb0_end = self.len() - start_bit as usize;
                Ok(Self::from_bv(
                    self.as_bitslice()[lsb0_start..lsb0_end]
                        .iter()
                        .rev()
                        .step_by(step as usize)
                        .collect::<BV>()
                        .into_iter()
                        .rev()
                        .collect(),
                    false,
                ))
            }
        } else {
            if start_bit <= end_bit || start_bit == -1 {
                return Ok(BitCollection::empty(self.msb0()));
            }
            if start_bit as usize > self.len() {
                return Err(PyValueError::new_err(
                    "Slice start bit is past the end of the container.".to_string(),
                ));
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
                self.msb0(),
            ))
        }
    }

    fn count(&self, count_ones: bool) -> usize {
        let slice = self.as_bitslice();
        let mut ones = 0;

        match slice.domain() {
            bitvec::domain::Domain::Region { head, body, tail } => {
                if let Some(h) = head {
                    ones += h.into_bitslice().count_ones() as usize;
                }
                if let Ok(words) = bytemuck::try_cast_slice::<u8, usize>(body) {
                    // Considerable speed increase by casting data to usize if possible.
                    for &word in words {
                        ones += word.count_ones() as usize;
                    }
                    // Handle the remainder not fitting into usize
                    let remainder_start = words.len() * std::mem::size_of::<usize>();
                    for &byte in &body[remainder_start..] {
                        ones += byte.count_ones() as usize;
                    }
                } else {
                    // Fallback for architectures where alignment is strict
                    for &byte in body {
                        ones += byte.count_ones() as usize;
                    }
                }
                if let Some(t) = tail {
                    ones += t.into_bitslice().count_ones() as usize;
                }
            }
            _ => {
                ones = slice.count_ones();
            }
        }

        if count_ones { ones } else { self.len() - ones }
    }

    fn multiply(&self, n: usize) -> Self {
        let len = self.len();
        if n == 0 || len == 0 {
            return BitCollection::empty(self.msb0());
        }
        let mut bv = BV::with_capacity(len * n);
        bv.extend_from_bitslice(self.as_bitslice());
        // TODO: This could be done more efficiently with doubling.
        for _ in 1..n {
            bv.extend_from_bitslice(self.as_bitslice());
        }
        Self::from_bv(bv, self.msb0())
    }

    fn lshift(&self, n: usize) -> Self {
        if n == 0 {
            return self.clone();
        }
        let len = self.len();
        if n >= len {
            return Self::from_bv(bv_from_zeros(len), self.msb0());
        }
        let mut result_data = BV::with_capacity(len);
        result_data.extend_from_bitslice(&self.as_bitslice()[n..]);
        result_data.resize(len, false);
        Self::from_bv(result_data, self.msb0())
    }

    fn rshift(&self, n: usize) -> Self {
        if n == 0 {
            return self.clone();
        }
        let len = self.len();
        if n >= len {
            return Self::from_bv(bv_from_zeros(len), self.msb0());
        }
        let mut result_data = BV::repeat(false, n);
        result_data.extend_from_bitslice(&self.as_bitslice()[..len - n]);
        Self::from_bv(result_data, self.msb0())
    }

    /// Return a bit reversed copy
    fn reverse_copy(&self) -> Self {
        let mut bv = self.to_bitvec();
        bv.reverse();
        Self::from_bv(bv, self.msb0())
    }

    /// Return a byte swapped copy
    fn byte_swap_copy(&self, byte_length: Option<i64>) -> PyResult<Self> {
        let len = self.len();
        if !len.is_multiple_of(8) {
            return Err(PyValueError::new_err(format!(
                "Bit length must be an multiple of 8 to use byte_swap (got length of {len} bits). This error can also be caused by using an endianness modifier on non-whole byte data."
            )));
        }
        let byte_length = byte_length.unwrap_or((len as i64) / 8);
        if byte_length == 0 && len == 0 {
            return Ok(BitCollection::empty(self.msb0()));
        }
        if byte_length <= 0 {
            return Err(PyValueError::new_err(format!(
                "Need a positive byte length for byte_swap. Received '{byte_length}'."
            )));
        }
        let byte_length = byte_length as usize;
        let self_byte_length = len / 8;
        if !self_byte_length.is_multiple_of(byte_length) {
            return Err(PyValueError::new_err(format!(
                "The data to byte_swap is {self_byte_length} bytes long, but it needs to be a multiple of {byte_length} bytes."
            )));
        }

        let mut bytes = self.to_byte_data()?;
        for chunk in bytes.chunks_mut(byte_length) {
            chunk.reverse();
        }
        Ok(BitCollection::from_bv(BV::from_vec(bytes), self.msb0()))
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
    fn to_octal(&self) -> PyResult<String> {
        let len = self.len();
        if !len.is_multiple_of(3) {
            return Err(PyValueError::new_err(format!(
                "Cannot interpret as octal - length of {} is not a multiple of 3 bits.",
                len
            )));
        }
        Ok(self.build_oct_string())
    }

    #[inline]
    fn to_hexadecimal(&self) -> PyResult<String> {
        let len = self.len();
        if !len.is_multiple_of(4) {
            return Err(PyValueError::new_err(format!(
                "Cannot interpret as hex - length of {} is not a multiple of 4 bits.",
                len
            )));
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
    fn to_byte_data(&self) -> PyResult<Vec<u8>> {
        if self.is_empty() {
            return Ok(Vec::new());
        }
        let len_bits = self.len();
        if !len_bits.is_multiple_of(8) {
            return Err(PyValueError::new_err(format!(
                "Cannot interpret as bytes - length of {len_bits} is not a multiple of 8 bits."
            )));
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
    fn to_u128(&self, is_little_endian: bool) -> PyResult<u128> {
        let length = self.len();
        if length == 0 {
            return Err(PyValueError::new_err(
                "Cannot convert to unsigned int when bit length is zero.",
            ));
        }
        if length > 128 {
            return Err(PyValueError::new_err(format!(
                "Bit length to convert to unsigned int must be between 1 and 128. Received {length}."
            )));
        }
        let raw = if is_little_endian {
            self.as_bitslice().load_le::<u128>()
        } else {
            self.as_bitslice().load_be::<u128>()
        };
        Ok(raw)
    }

    #[inline]
    fn to_i128(&self, is_little_endian: bool) -> PyResult<i128> {
        let length = self.len();
        if length == 0 {
            return Err(PyValueError::new_err(
                "Cannot convert to signed int when bit length is zero.",
            ));
        }
        if length > 128 {
            return Err(PyValueError::new_err(format!(
                "Bit length to convert to signed int must be between 1 and 128. Received {length}."
            )));
        }
        let raw = if is_little_endian {
            self.as_bitslice().load_le::<u128>()
        } else {
            self.as_bitslice().load_be::<u128>()
        };

        let shift = 128 - length;
        Ok(((raw << shift) as i128) >> shift)
    }

    fn to_f64(&self, is_little_endian: bool) -> PyResult<f64> {
        let length = self.len();
        match length {
            64 => {
                let bits = if is_little_endian {
                    self.as_bitslice().load_le::<u64>()
                } else {
                    self.as_bitslice().load_be::<u64>()
                };
                Ok(f64::from_bits(bits))
            }
            32 => {
                let bits = if is_little_endian {
                    self.as_bitslice().load_le::<u32>()
                } else {
                    self.as_bitslice().load_be::<u32>()
                };
                Ok(f32::from_bits(bits) as f64)
            }
            16 => {
                let bits = if is_little_endian {
                    self.as_bitslice().load_le::<u16>()
                } else {
                    self.as_bitslice().load_be::<u16>()
                };
                Ok(f16::from_bits(bits).to_f64())
            }
            _ => Err(PyValueError::new_err(format!(
                "Unsupported float bit length '{length}'. Only 16, 32 and 64 are supported."
            ))),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.as_bitslice().is_empty()
    }

    #[inline]
    fn len(&self) -> usize {
        self.as_bitslice().len()
    }

    #[inline]
    fn get_slice(&self, start_bit: usize, length: usize) -> PyResult<Self> {
        if length == 0 {
            return Ok(BitCollection::empty(self.msb0()));
        }
        if start_bit + length > self.len() {
            return Err(PyIndexError::new_err(
                "End bit of the slice goes past the end of the container.".to_string(),
            ));
        }
        Ok(self.get_slice_unchecked(start_bit, length))
    }
}

impl BitCollection for Tibs {
    fn from_bv(bv: BV, msb0: bool) -> Self {
        Tibs::from_bv(bv, msb0)
    }

    fn msb0(&self) -> bool {
        self.msb0
    }

    #[inline]
    fn to_bitvec(&self) -> BV {
        self.to_bitslice().to_bitvec()
    }

    #[inline]
    fn as_bitslice(&self) -> &BS {
        Tibs::to_bitslice(self)
    }

    #[inline]
    fn get_slice_unchecked(&self, start_bit: usize, length: usize) -> Self {
        Tibs::get_slice_unchecked(self, start_bit, length)
    }

    #[inline]
    fn get_raw_bytes(&self) -> Vec<u8> {
        Tibs::raw_bytes(self)
    }
}

impl BitCollection for Mutibs {
    fn from_bv(bv: BV, msb0: bool) -> Self {
        Mutibs::from_bv(bv, msb0)
    }

    fn msb0(&self) -> bool {
        self.msb0
    }

    #[inline]
    fn to_bitvec(&self) -> BV {
        Mutibs::as_bitvec_ref(self).to_bitvec()
    }

    #[inline]
    fn as_bitslice(&self) -> &BS {
        Mutibs::as_bitvec_ref(self).as_bitslice()
    }

    #[inline]
    fn get_slice_unchecked(&self, start_bit: usize, length: usize) -> Self {
        if self.msb0() {
            Self::from_bv(
                self.as_bitslice()[start_bit..start_bit + length].to_bitvec(),
                true,
            )
        } else {
            let start_bit = self.data.len() - length - start_bit;
            Self::from_bv(
                self.as_bitslice()[start_bit..start_bit + length].to_bitvec(),
                false,
            )
        }
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
        self.to_bitslice() == other.to_bitslice()
    }
}

impl PartialEq<Mutibs> for Tibs {
    #[inline]
    fn eq(&self, other: &Mutibs) -> bool {
        self.to_bitslice() == other.as_bitvec_ref()
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
        self.as_bitvec_ref() == other.to_bitslice()
    }
}
