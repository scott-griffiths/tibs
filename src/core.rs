use crate::enums::Codec;
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
    fn get_index(&self, bit_index: isize) -> PyResult<bool> {
        let index = validate_index(bit_index, self.len(), self.msb0())?;
        Ok(self.as_bitslice()[index])
    }

    fn get_slice_with_step(&self, start_bit: isize, end_bit: isize, step: isize) -> PyResult<Self> {
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

    fn collect_chunks(
        &self,
        chunk_size: i64,
        count: Option<i64>,
    ) -> PyResult<Vec<Self>> {
        if chunk_size <= 0 {
            return Err(PyValueError::new_err(format!(
                "Cannot create chunk list - chunk_size of {chunk_size} given, but it must be > 0."
            )));
        }
        let max_chunks = match count {
            Some(c) => {
                if c < 0 {
                    return Err(PyValueError::new_err(format!(
                        "Cannot create chunk list - count of {c} given, but it must be > 0."
                    )));
                }
                c as usize
            }
            None => usize::MAX,
        };

        let bits_len = self.len();
        let chunk_size = chunk_size as usize;
        let mut current_pos = 0;
        let mut chunks_generated = 0;
        let mut chunks = Vec::new();

        while chunks_generated < max_chunks {
            if current_pos >= bits_len {
                break;
            }
            let take = std::cmp::min(chunk_size, bits_len - current_pos);
            let start = current_pos;
            chunks.push(self.get_slice_unchecked(start, take));
            current_pos += take;
            chunks_generated += 1;
        }

        Ok(chunks)
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
                "Bit length must be a multiple of 8 to use byte_swap (got length of {len} bits). This error can also be caused by using an endianness modifier on non-whole byte data."
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

    fn raw_encoded_bit_length(bit_length: usize) -> usize {
        let data_byte_length = bit_length.div_ceil(8);
        5 + Self::encode_varint(data_byte_length as u64).len() + data_byte_length * 8
    }

    fn rice_encode_int(value: usize, k: u8) -> BV {
        let mut out = BV::new();
        let quotient = value >> k;
        for _ in 0..quotient {
            out.push(true);
        }
        out.push(false);
        if k > 0 {
            let remainder_mask = (1usize << k) - 1;
            let remainder = value & remainder_mask;
            for shift in (0..k).rev() {
                out.push(((remainder >> shift) & 1) == 1);
            }
        }
        out
    }

    fn rice_decode_int(bits: &BS, start: usize, k: u8) -> PyResult<(usize, usize)> {
        let mut pos = start;
        while pos < bits.len() && bits[pos] {
            pos += 1;
        }
        if pos >= bits.len() {
            return Err(PyValueError::new_err(
                "The encoded sequence ended unexpectedly.",
            ));
        }
        let quotient = pos - start;
        pos += 1;

        let k_usize = k as usize;
        if bits.len() - pos < k_usize {
            return Err(PyValueError::new_err(
                "The encoded sequence ended unexpectedly.",
            ));
        }
        let remainder = if k == 0 {
            0
        } else {
            bits[pos..pos + k_usize].load_be::<usize>()
        };
        pos += k_usize;

        let base = quotient
            .checked_shl(k as u32)
            .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
        let value = base
            .checked_add(remainder)
            .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
        Ok((value, pos))
    }

    fn zstd_compress_bytes(&self) -> Vec<u8> {
        let bit_length = self.len();
        let data_byte_length = bit_length.div_ceil(8);
        let raw_bit_padding = data_byte_length * 8 - bit_length;

        let mut raw = self.to_bitvec();
        for _ in 0..raw_bit_padding {
            raw.push(false);
        }

        zstd::bulk::compress(&raw.into_vec(), 0).expect("zstd compression failed")
    }

    fn encode_as_zstd_from_compressed(&self, compressed: Vec<u8>) -> BV {
        let mut bv = BV::new();
        bv.push(true);
        bv.push(false);
        let bit_padding = if self.len() % 8 == 0 {
            0
        } else {
            8 - self.len() % 8
        };
        for shift in (0..3).rev() {
            bv.push((bit_padding >> shift) & 1 == 1);
        }
        bv.extend(Self::encode_varint(compressed.len() as u64));
        bv.extend(BV::from_vec(compressed));
        bv
    }

    fn encode_as_zstd(&self) -> BV {
        self.encode_as_zstd_from_compressed(Self::zstd_compress_bytes(self))
    }

    fn encode_as_raw(&self) -> BV {
        let bit_length = self.len();
        let data_byte_length = bit_length.div_ceil(8);
        let bit_padding = data_byte_length * 8 - bit_length;

        let mut bv = BV::new();
        bv.push(false);
        bv.push(false);
        for shift in (0..3).rev() {
            bv.push((bit_padding >> shift) & 1 == 1);
        }
        bv.extend(Self::encode_varint(data_byte_length as u64));
        bv.extend(self.to_bitvec());
        for _ in 0..bit_padding {
            bv.push(false);
        }
        bv
    }

    fn rice_encoded_gaps(bits: &BS, sparse_bit: bool) -> Vec<usize> {
        let mut gaps = Vec::new();

        let mut previous = 0;
        if sparse_bit {
            for p in bits.iter_ones() {
                gaps.push(p - previous);
                previous = p + 1;
            }
        } else {
            for p in bits.iter_zeros() {
                gaps.push(p - previous);
                previous = p + 1;
            }
        }

        if let Some(last) = bits.last() {
            if *last != sparse_bit {
                gaps.push(bits.len() - previous - 1);
            }
        }

        gaps
    }

    fn estimated_rice_k(gaps: &[usize]) -> u8 {
        if gaps.is_empty() {
            return 0;
        }

        let total_gap: usize = gaps.iter().sum();
        if total_gap == 0 {
            return 0;
        }

        let mean_gap = total_gap as f64 / gaps.len() as f64;
        let estimate = (mean_gap * std::f64::consts::LN_2).log2().round();
        estimate.clamp(0.0, 31.0) as u8
    }

    fn rice_payload_bit_length(gaps: &[usize], k: u8) -> usize {
        gaps.iter().map(|gap| (gap >> k) + 1 + k as usize).sum()
    }

    fn rice_encoded_bit_length(&self, sparse_bit: bool) -> usize {
        let gaps = Self::rice_encoded_gaps(self.as_bitslice(), sparse_bit);
        let estimated_k = Self::estimated_rice_k(&gaps);
        let payload_bit_length = Self::rice_payload_bit_length(&gaps, estimated_k);
        let payload_byte_length = payload_bit_length.div_ceil(8);
        5 + Self::encode_varint(payload_byte_length as u64).len() + 8 + payload_byte_length * 8
    }

    fn encode_as_rice(&self, sparse_bit: bool) -> BV {
        let bits = self.as_bitslice();

        let gaps = Self::rice_encoded_gaps(bits, sparse_bit);
        debug_assert!(bits.len() > 0);
        let final_bit = *bits
            .last()
            .expect("Rice encoding not supported for empty Tibs.");
        let estimated_k = Self::estimated_rice_k(&gaps);

        let payload_bit_length = Self::rice_payload_bit_length(&gaps, estimated_k);
        let mut payload = BV::new();
        for gap in &gaps {
            payload.extend(Self::rice_encode_int(*gap, estimated_k));
        }
        debug_assert_eq!(payload.len(), payload_bit_length);
        let payload_byte_length = payload_bit_length.div_ceil(8);
        let bit_padding = payload_byte_length * 8 - payload_bit_length;
        for _ in 0..bit_padding {
            payload.push(false);
        }

        let mut encoded = BV::new();
        encoded.push(false);
        encoded.push(true);
        for shift in (0..3).rev() {
            encoded.push((bit_padding >> shift) & 1 == 1);
        }
        encoded.extend(Self::encode_varint(payload_byte_length as u64));
        for shift in (0..5).rev() {
            encoded.push((estimated_k >> shift) & 1 == 1);
        }
        encoded.push(sparse_bit);
        encoded.push(final_bit);
        encoded.push(false);
        encoded.extend(payload);

        encoded
    }

    fn decode_raw_payload(
        bv: &BS,
        msb0_flag: bool,
        bit_padding: usize,
        data_start: usize,
        data_bits: usize,
    ) -> PyResult<Self> {
        let data_end = data_start
            .checked_add(data_bits)
            .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
        if bv.len() < data_end {
            return Err(PyValueError::new_err(
                "The encoded sequence ended unexpectedly.",
            ));
        }
        if bv.len() != data_end {
            return Err(PyValueError::new_err(
                "The encoded sequence has unexpected trailing bytes.",
            ));
        }
        if bit_padding > data_bits {
            return Err(PyValueError::new_err("The encoded sequence is reserved."));
        }

        let out_end = data_end - bit_padding;
        Ok(Self::from_bv(
            bv[data_start..out_end].to_bitvec(),
            msb0_flag,
        ))
    }

    fn decode_rice_payload(
        bv: &BS,
        msb0_flag: bool,
        bit_padding: usize,
        data_start: usize,
        payload_bits: usize,
    ) -> PyResult<Self> {
        let config_end = data_start
            .checked_add(8)
            .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
        if bv.len() < config_end {
            return Err(PyValueError::new_err(
                "The encoded sequence ended unexpectedly.",
            ));
        }

        let payload_start = config_end;
        let payload_end = payload_start
            .checked_add(payload_bits)
            .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
        if bv.len() < payload_end {
            return Err(PyValueError::new_err(
                "The encoded sequence ended unexpectedly.",
            ));
        }
        if bv.len() != payload_end {
            return Err(PyValueError::new_err(
                "The encoded sequence has unexpected trailing bytes.",
            ));
        }
        if bit_padding > payload_bits {
            return Err(PyValueError::new_err("The encoded sequence is reserved."));
        }

        let config = &bv[data_start..config_end];
        if config[7] {
            return Err(PyValueError::new_err("The encoded sequence is reserved."));
        }
        let k = config[0..5].load_be::<u8>();
        let sparse_bit = config[5];
        let final_bit = config[6];

        let encoded_gaps_end = payload_end - bit_padding;
        let encoded_gaps = &bv[payload_start..encoded_gaps_end];

        let mut decoded = BV::new();
        let mut pos = 0usize;
        while pos < encoded_gaps.len() {
            let (gap, next_pos) = Self::rice_decode_int(encoded_gaps, pos, k)?;
            pos = next_pos;

            for _ in 0..gap {
                decoded.push(!sparse_bit);
            }
            decoded.push(sparse_bit);
        }

        if decoded.is_empty() {
            return Err(PyValueError::new_err("The encoded sequence is reserved."));
        }
        let final_pos = decoded.len() - 1;
        decoded.set(final_pos, final_bit);
        Ok(Self::from_bv(decoded, msb0_flag))
    }

    fn decode_zstd_payload(
        bv: &BS,
        msb0_flag: bool,
        bit_padding: usize,
        data_start: usize,
        payload_bits: usize,
    ) -> PyResult<Self> {
        let payload_end = data_start + payload_bits;
        if bv.len() < payload_end {
            return Err(PyValueError::new_err(
                "The encoded sequence ended unexpectedly.",
            ));
        }
        if bv.len() != payload_end {
            return Err(PyValueError::new_err(
                "The encoded sequence has unexpected trailing bytes.",
            ));
        }

        let compressed = bv[data_start..payload_end].to_bitvec().into_vec();
        let decompressed_size = zstd::zstd_safe::get_frame_content_size(&compressed)
            .map_err(|e| {
                PyValueError::new_err(format!("The zstd payload could not be decoded: {e}"))
            })?
            .ok_or_else(|| {
                PyValueError::new_err("The zstd payload did not include its decompressed size.")
            })?;

        let decompressed = zstd::bulk::decompress(&compressed, decompressed_size as usize)
            .map_err(|e| {
                PyValueError::new_err(format!("The zstd payload could not be decoded: {e}"))
            })?;

        let data_bits = decompressed.len() * 8;
        let out_end = data_bits - bit_padding;
        let decompressed = BV::from_vec(decompressed);
        Ok(Self::from_bv(
            decompressed[..out_end].to_bitvec(),
            msb0_flag,
        ))
    }

    fn encode_varint(mut u: u64) -> BV {
        let mut chunks: Vec<u8> = Vec::new();
        loop {
            chunks.push((u & 0x7f) as u8);
            u >>= 7;
            if u == 0 {
                break;
            }
        }
        chunks.reverse();

        let mut out: BV = BV::with_capacity(chunks.len() * 8);
        for (i, chunk) in chunks.iter().enumerate() {
            let continuation = i + 1 < chunks.len();
            out.push(continuation);
            for shift in (0..7).rev() {
                out.push(((chunk >> shift) & 1) == 1);
            }
        }
        out
    }

    fn decode_varint(bits: &BS) -> PyResult<(usize, usize)> {
        let mut value: usize = 0;
        let mut bits_consumed: usize = 0;
        let mut saw_final = false;

        for byte in bits.chunks(8) {
            if byte.len() < 8 {
                break;
            }
            let continuation = byte[0];
            let payload = byte[1..8].load_be::<u8>() as usize;

            if bits_consumed == 0 && continuation && payload == 0 {
                return Err(PyValueError::new_err("The encoded sequence is reserved."));
            }
            if value > (usize::MAX >> 7) {
                return Err(PyValueError::new_err(
                    "The encoded sequence is too large to decode.",
                ));
            }
            value = (value << 7) | payload;
            bits_consumed += 8;

            if !continuation {
                saw_final = true;
                break;
            }
        }

        if !saw_final {
            return Err(PyValueError::new_err(
                "The encoded sequence ended unexpectedly.",
            ));
        }
        Ok((value, bits_consumed))
    }

    fn decode_bytes(b: Vec<u8>) -> PyResult<Self> {
        if b.is_empty() {
            return Err(PyValueError::new_err("Cannot decode an empty bytes."));
        }
        let bv = BV::from_vec(b);
        let single_byte_flag = bv[0];
        let msb0_flag = bv[1];
        if single_byte_flag {
            if bv.len() != 8 {
                return Err(PyValueError::new_err(
                    "The encoded sequence has unexpected trailing bytes.",
                ));
            }
            for bit_pos in 2..7 {
                if bv[bit_pos] {
                    return Ok(Self::from_bv(bv[bit_pos + 1..].to_bitvec(), msb0_flag));
                }
            }
            if !bv[7] {
                return Err(PyValueError::new_err("The encoded sequence is reserved."));
            }
            return Ok(Self::empty(msb0_flag));
        }
        let short_form_flag = bv[2];
        if short_form_flag {
            let length_minus_6 = bv[3..8].load_be::<u8>() as usize;
            let bit_length = length_minus_6 + 6;
            if bv.len() < bit_length + 8 {
                return Err(PyValueError::new_err(
                    "The encoded sequence ended unexpectedly.",
                ));
            }
            if bv.len() != (1 + (bit_length + 7) / 8) * 8 {
                return Err(PyValueError::new_err(
                    "The encoded sequence has unexpected trailing bytes.",
                ));
            }
            return Ok(Self::from_bv(bv[8..8 + bit_length].to_bitvec(), msb0_flag));
        }

        let codec = bv[3..5].load_be::<u8>();
        let bit_padding = bv[5..8].load_be::<u8>() as usize;

        let (byte_length, varint_bits) = Self::decode_varint(&bv[8..])?;
        let data_start = 8 + varint_bits;
        let data_bits = byte_length
            .checked_mul(8)
            .ok_or_else(|| PyValueError::new_err("The encoded sequence is too large to decode."))?;
        match codec {
            0b00 => Self::decode_raw_payload(&bv, msb0_flag, bit_padding, data_start, data_bits),
            0b01 => Self::decode_rice_payload(&bv, msb0_flag, bit_padding, data_start, data_bits),
            0b10 => Self::decode_zstd_payload(&bv, msb0_flag, bit_padding, data_start, data_bits),
            _ => Err(PyValueError::new_err("The codec value is reserved.")),
        }
    }

    fn encode(&self, codec: Option<Codec>) -> Vec<u8> {
        let bit_length = self.len();
        let mut bv: BV = BV::new();

        // Length of zero treated as a special case and ignores the codec.
        // Uses the Auto codec, and encodes as a single byte.
        if bit_length == 0 {
            bv.push(true);
            bv.push(self.msb0());
            for _ in 0..5 {
                bv.push(false);
            }
            bv.push(true);
            return bv.into_vec();
        }

        match codec.unwrap_or(Codec::Auto) {
            Codec::Auto => match bit_length {
                0..=5 => {
                    bv.push(true);
                    bv.push(self.msb0());
                    let leading_zeros = 5 - bit_length;
                    for _ in 0..leading_zeros {
                        bv.push(false);
                    }
                    bv.push(true);
                    bv.extend_from_bitslice(self.as_bitslice());
                }
                6..=37 => {
                    bv.push(false);
                    bv.push(self.msb0());
                    bv.push(true);
                    let length_minus_6 = (bit_length - 6) as u8;
                    for shift in (0..5).rev() {
                        bv.push((length_minus_6 >> shift) & 1 == 1);
                    }
                    bv.extend(self.to_bitvec());
                    let padding_bits = 8 - bv.len() % 8;
                    if padding_bits != 8 {
                        for _ in 0..padding_bits {
                            bv.push(false);
                        }
                    }
                }
                38.. => {
                    bv.push(false);
                    bv.push(self.msb0());
                    bv.push(false);

                    let raw_bit_length = Self::raw_encoded_bit_length(bit_length);
                    let mut best_codec = Codec::Raw;
                    let mut best_bit_length = raw_bit_length;

                    let ones_count = self.count(true);
                    let sparse_bit = ones_count < bit_length / 2;
                    let sparseness = if sparse_bit {
                        ones_count as f64 / self.len() as f64
                    } else {
                        (self.len() - ones_count) as f64 / self.len() as f64
                    };
                    if bit_length <= 128 || sparseness < 0.25 {
                        let rice_bit_length = self.rice_encoded_bit_length(sparse_bit);
                        if rice_bit_length < best_bit_length {
                            best_codec = Codec::Rice;
                            best_bit_length = rice_bit_length;
                        }
                    }

                    let zstd_compressed = Self::zstd_compress_bytes(self);
                    let zstd_bit_length = 5
                        + Self::encode_varint(zstd_compressed.len() as u64).len()
                        + zstd_compressed.len() * 8;

                    if zstd_bit_length < best_bit_length {
                        best_codec = Codec::Zstd;
                    }
                    match best_codec {
                        Codec::Raw => bv.extend(self.encode_as_raw()),
                        Codec::Rice => bv.extend(self.encode_as_rice(sparse_bit)),
                        Codec::Zstd => {
                            bv.extend(self.encode_as_zstd_from_compressed(zstd_compressed))
                        }
                        Codec::Auto => unreachable!(),
                    }
                }
            },
            Codec::Raw => {
                bv.push(false);
                bv.push(self.msb0());
                bv.push(false);
                bv.extend(self.encode_as_raw());
            }
            Codec::Rice => {
                bv.push(false);
                bv.push(self.msb0());
                bv.push(false);
                let sparse_bit = self.count(true) < self.len() / 2;
                bv.extend(self.encode_as_rice(sparse_bit));
            }
            Codec::Zstd => {
                bv.push(false);
                bv.push(self.msb0());
                bv.push(false);
                bv.extend(self.encode_as_zstd());
            }
        }

        bv.into_vec()
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
        Tibs::to_bitvec(self)
    }

    #[inline]
    fn as_bitslice(&self) -> &BS {
        Tibs::as_bitslice(self)
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
        Mutibs::to_bitvec(self)
    }

    #[inline]
    fn as_bitslice(&self) -> &BS {
        Mutibs::as_bitslice(self)
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
