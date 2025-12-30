#[cfg(test)]
mod tests {
    use crate::core::BitCollection;
    use crate::helpers::{bv_from_bin, bv_from_hex, bv_from_oct, bv_from_ones, bv_from_zeros};
    use crate::mutibs::Mutibs;
    use crate::tibs_::Tibs;

    // #[test]
    // fn from_bytes() {
    //     let data: Vec<u8> = vec![10, 20, 30];
    //     let bits = Tibs::from_bytes(data);
    //     assert_eq!(*bits.to_bytes().unwrap(), vec![10, 20, 30]);
    //     assert_eq!(bits.len(), 24);
    // }

    #[test]
    fn from_hex() {
        let bits = Tibs::from_bv(bv_from_hex("0x0a_14  _1e").unwrap());
        assert_eq!(*bits.to_bytes().unwrap(), vec![10, 20, 30]);
        assert_eq!(bits.len(), 24);
        let bits = Tibs::from_bv(bv_from_hex("").unwrap());
        assert_eq!(bits.len(), 0);
        let bv = bv_from_hex("hello");
        assert!(bv.is_err());
        let bits = Tibs::from_bv(bv_from_hex("1").unwrap());
        assert_eq!(*bits.to_bytes().unwrap(), vec![16]);
        assert_eq!(bits.len(), 4);
    }

    #[test]
    fn from_bin() {
        let bits = Tibs::from_bv(bv_from_bin("00001010").unwrap());
        assert_eq!(*bits.to_bytes().unwrap(), vec![10]);
        assert_eq!(bits.len(), 8);
        let bits = Tibs::from_bv(bv_from_bin("").unwrap());
        assert_eq!(bits.len(), 0);
        let bits = Tibs::from_bv(bv_from_bin("1").unwrap());
        assert_eq!(*bits.to_bytes().unwrap(), vec![128]);
        assert_eq!(bits.len(), 1);
    }

    #[test]
    fn from_zeros() {
        let bits = Tibs::from_bv(bv_from_zeros(8));
        assert_eq!(*bits.to_bytes().unwrap(), vec![0]);
        assert_eq!(bits.len(), 8);
        assert_eq!(bits.to_hexadecimal().unwrap(), "00");
        let bits = Tibs::from_bv(bv_from_zeros(9));
        assert_eq!(*bits.to_bytes().unwrap(), vec![0, 0]);
        assert_eq!(bits.len(), 9);
        let bits = <Tibs as BitCollection>::empty();
        assert_eq!(bits.len(), 0);
    }

    #[test]
    fn from_ones() {
        let bits = Tibs::from_bv(bv_from_ones(8));
        assert_eq!(*bits.to_bytes().unwrap(), vec![255]);
        assert_eq!(bits.len(), 8);
        assert_eq!(bits.to_hexadecimal().unwrap(), "ff");
        let bits = Tibs::from_bv(bv_from_ones(9));
        assert_eq!(bits.to_bin(), "111111111");
        assert_eq!((*bits.to_bytes().unwrap())[0], 0xff);
        assert_eq!((*bits.to_bytes().unwrap())[1] & 0x80, 0x80);
        assert_eq!(bits.len(), 9);
        let bits = Tibs::from_bv(bv_from_ones(0));
        assert_eq!(bits.len(), 0);
    }

    #[test]
    fn get_index() {
        let bits = Tibs::from_bv(bv_from_bin("001100").unwrap());
        assert_eq!(bits.get_index(0).unwrap(), false);
        assert_eq!(bits.get_index(1).unwrap(), false);
        assert_eq!(bits.get_index(2).unwrap(), true);
        assert_eq!(bits.get_index(3).unwrap(), true);
        assert_eq!(bits.get_index(4).unwrap(), false);
        assert_eq!(bits.get_index(5).unwrap(), false);
        assert!(bits.get_index(6).is_err());
        assert!(bits.get_index(60).is_err());
    }

    #[test]
    fn hex_edge_cases() {
        let b1 = Tibs::from_bv(bv_from_hex("0123456789abcdef").unwrap());
        let b2 = b1.get_slice(12, b1.len()).unwrap();
        assert_eq!(b2.to_hexadecimal().unwrap(), "3456789abcdef");
        assert_eq!(b2.len(), 52);
        let t = Tibs::from_bv(bv_from_hex("123").unwrap());
        assert_eq!(t.to_hexadecimal().unwrap(), "123");
    }

    // #[test]
    // fn test_find() {
    //     let b1 = <Tibs as BitCollection>::from_zeros(10);
    //     let b2 = <Tibs as BitCollection>::from_ones(2);
    //     assert_eq!(b1.find(&b2, Some(0), None, false), None);
    //     let b3 = Tibs::from_bin("00001110").unwrap();
    //     let b4 = Tibs::from_bin("01").unwrap();
    //     assert_eq!(b3.find(&b4, None, None, false), Some(3));
    //     assert_eq!(b3.find(&b4, Some(2), None, false), Some(3));
    //
    //     let s = Tibs::from_bin("0000110110000").unwrap();
    //     let f = Tibs::from_bin("11011").unwrap();
    //     let p = s.find(&f, None, None, false).unwrap();
    //     assert_eq!(p, 4);
    //
    //     let s = Tibs::from_hex("010203040102ff").unwrap();
    //     // assert s.find("0x05", bytealigned=True) is None
    //     let f = Tibs::from_hex("02").unwrap();
    //     let p = s.find(&f, None, None, true);
    //     assert_eq!(p, Some(8));
    // }

    // #[test]
    // fn test_rfind() {
    //     let b1 = Tibs::from_hex("00780f0").unwrap();
    //     let b2 = Tibs::from_bin("1111").unwrap();
    //     assert_eq!(b1.rfind(&b2, 0, b1.len(), false), Some(20));
    //     assert_eq!(b1.find(&b2, None, None, false), Some(9));
    // }

    // #[test]
    // fn test_and() {
    //     let a1 = Tibs::from_bv(bv_from_hex("f0f").unwrap();
    //     let a2 = Tibs::from_bv(bv_from_hex("123").unwrap();
    //     let a3 = a1.and(&a2).unwrap();
    //     let b = Tibs::from_bv(bv_from_hex("103").unwrap();
    //     assert_eq!(a3, b);
    //     let a4 = a1
    //         .get_slice_unchecked(4, 8)
    //         .and(&a2.get_slice_unchecked(4, 8))
    //         .unwrap();
    //     assert_eq!(a4, Tibs::from_bv(bv_from_hex("03").unwrap());
    // }

    #[test]
    fn test_set_mutable_slice() {
        let mut a = Mutibs::from_bv(bv_from_hex("0011223344").unwrap());
        let b = Tibs::from_bv(bv_from_hex("ff").unwrap());
        a.set_slice(8, 16, &b.as_bitslice());
        assert_eq!(a.to_hexadecimal().unwrap(), "00ff223344");
    }

    #[test]
    fn test_get_mutable_slice() {
        let a = Tibs::from_bv(bv_from_hex("01ffff").unwrap());
        assert_eq!(a.len(), 24);
        let b = a.get_slice(1, a.len()).unwrap();
        assert_eq!(b.len(), 23);
        let c = b.to_mutibs();
        assert_eq!(c.len(), 23);
    }

    #[test]
    fn test_getslice() {
        let a = Tibs::from_bv(bv_from_bin("00010001").unwrap());
        assert_eq!(a.get_slice(0, 4).unwrap().to_bin(), "0001");
        assert_eq!(a.get_slice(4, 8).unwrap().to_bin(), "0001");
    }

    #[test]
    fn test_all_set() {
        let b = Tibs::from_bv(bv_from_bin("111").unwrap());
        assert!(b.all());
        let c = Tibs::from_bv(bv_from_oct("7777777777").unwrap());
        assert!(c.all());
    }

    #[test]
    fn test_set_index() {
        let mut b = Mutibs::from_bv(bv_from_zeros(10));
        b.set_index(true, 0).unwrap();
        assert_eq!(b.to_binary(), "1000000000");
        b.set_index(true, -1).unwrap();
        assert_eq!(b.to_binary(), "1000000001");
        b.set_index(false, 0).unwrap();
        assert_eq!(b.to_binary(), "0000000001");
    }

    #[test]
    fn test_to_bytes_from_slice() {
        let a = Tibs::from_bv(bv_from_ones(16));
        assert_eq!(a.to_bytes().unwrap(), vec![255, 255]);
        let b = a.get_slice(7, a.len()).unwrap();
        assert_eq!(b.to_bin(), "111111111");
        assert_eq!(b.to_bytes().unwrap(), vec![255, 128]);
    }

    #[test]
    fn test_from_oct() {
        let bits = Tibs::from_bv(bv_from_oct("123").unwrap());
        assert_eq!(bits.to_bin(), "001010011");
        let bits = Tibs::from_bv(bv_from_oct("7").unwrap());
        assert_eq!(bits.to_bin(), "111");
    }

    #[test]
    fn test_from_oct_checked() {
        let bits = Tibs::from_bv(bv_from_oct("123").unwrap());
        assert_eq!(bits.to_bin(), "001010011");
        let bits = Tibs::from_bv(bv_from_oct("0o123").unwrap());
        assert_eq!(bits.to_bin(), "001010011");
        let bits = Tibs::from_bv(bv_from_oct("7").unwrap());
        assert_eq!(bits.to_bin(), "111");
        let bv = bv_from_oct("8");
        assert!(bv.is_err());
    }

    #[test]
    fn test_to_oct() {
        let bits = Tibs::from_bv(bv_from_bin("001010011").unwrap());
        assert_eq!(bits.to_oct().unwrap(), "123");
        let bits = Tibs::from_bv(bv_from_bin("111").unwrap());
        assert_eq!(bits.get_slice(0, 3).unwrap().to_oct().unwrap(), "7");
        let bits = Tibs::from_bv(bv_from_bin("000").unwrap());
        assert_eq!(bits.to_oct().unwrap(), "0");
    }

    #[test]
    fn test_set_from_slice() {
        let mut bits = Mutibs::from_bv(bv_from_bin("00000000").unwrap());
        bits.set_from_slice(true, 1, 7, 2).unwrap();
        assert_eq!(bits.to_binary(), "01010100");
        bits.set_from_slice(true, -7, -1, 2).unwrap();
        assert_eq!(bits.to_binary(), "01010100");
        bits.set_from_slice(false, 1, 7, 2).unwrap();
        assert_eq!(bits.to_binary(), "00000000");
    }

    #[test]
    fn test_any_set() {
        let bits = Tibs::from_bv(bv_from_bin("0000").unwrap());
        assert!(!bits.any());
        let bits = Tibs::from_bv(bv_from_bin("1000").unwrap());
        assert!(bits.any());
    }

    // #[test]
    // fn test_xor() {
    //     let a = Tibs::from_bv(bv_from_bin("1100").unwrap();
    //     let b = Tibs::from_bv(bv_from_bin("1010").unwrap();
    //     let result = a.xor(&b).unwrap();
    //     assert_eq!(result.to_bin(), "0110");
    // }
    //
    // #[test]
    // fn test_or() {
    //     let a = Tibs::from_bv(bv_from_bin("1100").unwrap();
    //     let b = Tibs::from_bv(bv_from_bin("1010").unwrap();
    //     let result = a.or(&b).unwrap();
    //     assert_eq!(result.to_bin(), "1110");
    // }
    //
    // #[test]
    // fn test_and2() {
    //     let a = Tibs::from_bv(bv_from_bin("1100").unwrap();
    //     let b = Tibs::from_bv(bv_from_bin("1010").unwrap();
    //     let result = a.and(&b).unwrap();
    //     assert_eq!(result.to_bin(), "1000");
    // }

    #[test]
    fn test_len() {
        let bits = Tibs::from_bv(bv_from_bin("1100").unwrap());
        assert_eq!(bits.__len__(), 4);
        let bits = Tibs::from_bv(bv_from_bin("101010").unwrap());
        assert_eq!(bits.__len__(), 6);
    }

    #[test]
    fn test_eq() {
        let a = Tibs::from_bv(bv_from_bin("1100").unwrap());
        let b = Tibs::from_bv(bv_from_bin("1100").unwrap());
        assert_eq!(a, b);
        let c = Tibs::from_bv(bv_from_bin("1010").unwrap());
        assert_ne!(a, c);
    }

    #[test]
    fn test_getslice_withstep() {
        let bits = Tibs::from_bv(bv_from_bin("11001100").unwrap());
        let slice = bits.get_slice_with_step(0, 8, 2).unwrap();
        assert_eq!(slice.to_bin(), "1010");
        let slice = bits.get_slice_with_step(7, -1, -2).unwrap();
        assert_eq!(slice.to_bin(), "0101");
        let slice = bits.get_slice_with_step(0, 8, 1).unwrap();
        assert_eq!(slice.to_bin(), "11001100");
        let slice = bits.get_slice_with_step(7, -1, -1).unwrap();
        assert_eq!(slice.to_bin(), "00110011");
        let slice = bits.get_slice_with_step(0, 8, 8).unwrap();
        assert_eq!(slice.to_bin(), "1");
        let slice = bits.get_slice_with_step(0, 8, -8).unwrap();
        assert_eq!(slice.to_bin(), "");
        let slice = bits.get_slice_with_step(0, 8, 3).unwrap();
        assert_eq!(slice.to_bin(), "100");
    }

    #[test]
    fn mutable_from_immutable() {
        let immutable = Tibs::from_bv(bv_from_bin("1010").unwrap());
        let mutable = Mutibs::from_bv(immutable.to_bitvec());
        assert_eq!(mutable.to_binary(), "1010");
    }

    #[test]
    fn freeze_preserves_data() {
        let mutable = Mutibs::from_bv(bv_from_bin("1100").unwrap());
        let immutable = mutable.to_tibs();
        assert_eq!(immutable.to_bin(), "1100");
    }

    #[test]
    fn modify_then_freeze() {
        let mut mutable = Mutibs::from_bv(bv_from_bin("0000").unwrap());
        mutable.set_index(true, 1).unwrap();
        mutable.set_index(true, 2).unwrap();
        let immutable = mutable.to_tibs();
        assert_eq!(immutable.to_bin(), "0110");
    }

    #[test]
    fn mutable_constructors() {
        let m1 = Mutibs::from_bv(bv_from_zeros(4));
        assert_eq!(m1.to_binary(), "0000");

        let m2 = Mutibs::from_bv(bv_from_ones(4));
        assert_eq!(m2.to_binary(), "1111");

        let m3 = Mutibs::from_bv(bv_from_bin("1010").unwrap());
        assert_eq!(m3.to_binary(), "1010");

        let m4 = Mutibs::from_bv(bv_from_hex("a").unwrap());
        assert_eq!(m4.to_binary(), "1010");

        let m5 = Mutibs::from_bv(bv_from_oct("12").unwrap());
        assert_eq!(m5.to_binary(), "001010");
    }

    #[test]
    fn mutable_equality() {
        let m1 = Mutibs::from_bv(bv_from_bin("1100").unwrap());
        let m2 = Mutibs::from_bv(bv_from_bin("1100").unwrap());
        let m3 = Mutibs::from_bv(bv_from_bin("0011").unwrap());

        assert!(m1 == m2);
        assert!(m1 != m3);
    }

    #[test]
    fn mutable_getslice() {
        let m = Mutibs::from_bv(bv_from_bin("11001010").unwrap());

        let slice1 = m.get_slice(2, 6).unwrap();
        assert_eq!(slice1.to_binary(), "0010");
    }

    // #[test]
    // fn mutable_find_operations() {
    //     let haystack = Mutibs::from_bin("00110011").unwrap();
    //     let needle = Tibs::from_bin("11").unwrap();
    //
    //     assert_eq!(haystack._find(&needle, 0, haystack.len(), false), Some(2));
    //     assert_eq!(haystack._find(&needle, 3, haystack.len(), false), Some(6));
    //     assert_eq!(haystack._rfind(&needle, 0, haystack.len(),false), Some(6));
    // }

    #[test]
    fn mutable_set_operations() {
        let mut m = Mutibs::from_bv(bv_from_zeros(8));

        m.set_index(true, 0).unwrap();
        m.set_index(true, 7).unwrap();
        assert_eq!(m.to_binary(), "10000001");

        m.set_from_slice(true, 2, 6, 1).unwrap();
        assert_eq!(m.to_binary(), "10111101");

        m.set_from_sequence(false, vec![0, 3, 7]).unwrap();
        assert_eq!(m.to_binary(), "00101100");
    }

    #[test]
    fn mutable_immutable_interaction() {
        let pattern1 = Mutibs::from_bv(bv_from_bin("1100").unwrap());
        let pattern2 = Tibs::from_bv(bv_from_bin("0011").unwrap());

        let mut m = Mutibs::from_bv(pattern1.as_bitvec_ref().clone());

        m.set_slice(0, 2, &pattern2.as_bitslice());
        assert_eq!(m.to_binary(), "001100");
    }

    #[test]
    fn empty_data_operations() {
        let empty_mutable = <Mutibs as BitCollection>::empty();

        assert_eq!(empty_mutable.len(), 0);
        assert!(!empty_mutable.any());

        assert_eq!(empty_mutable.to_tibs().len(), 0);
    }

    #[test]
    fn mutable_edge_index_operations() {
        let mut m = Mutibs::from_bv(bv_from_bin("1010").unwrap());

        m.set_index(false, 0).unwrap();
        m.set_index(false, 3).unwrap();
        assert_eq!(m.to_binary(), "0010");

        m.set_index(true, -1).unwrap();
        m.set_index(true, -4).unwrap();
        assert_eq!(m.to_binary(), "1011");

        assert!(m.set_index(true, 4).is_err());
        assert!(m.set_index(true, -5).is_err());
    }

    #[test]
    fn set_mutable_slice_with_bits() {
        let mut m = Mutibs::from_bv(bv_from_bin("00000000").unwrap());
        let pattern = Tibs::from_bv(bv_from_bin("1111").unwrap());

        m.set_slice(2, 6, &pattern.as_bitslice());
        assert_eq!(m.to_binary(), "00111100");

        m.set_slice(0, 2, &pattern.as_bitslice());
        assert_eq!(m.to_binary(), "1111111100");

        m.set_slice(6, 8, &pattern.as_bitslice());
        assert_eq!(m.to_binary(), "111111111100");
    }

    #[test]
    fn conversion_round_trip() {
        let original = Tibs::from_bv(bv_from_bin("101010").unwrap());
        let mut mutable = Mutibs::from_bv(original.to_bitvec());
        mutable.set_index(false, 0).unwrap();
        mutable.set_index(true, 1).unwrap();
        let result = mutable.as_tibs();

        assert_eq!(result.to_bin(), "011010");
    }

    // This one causes a panic that stops the other tests.
    // #[test]
    // fn mutable_to_representations() {
    //     let m = MutableBitRust::from_bin_checked("11110000");
    //
    //     assert_eq!(m.to_bin(), "11110000");
    //     assert_eq!(m.to_hex().unwrap(), "f0");
    //     assert_eq!(m.to_oct().unwrap(), "360");
    //     assert_eq!(m.to_bytes().unwrap(), vec![0xF0]);
    // }

    #[test]
    fn mutable_from_checked_constructors() {
        let bin = Mutibs::from_bv(bv_from_bin("1010").unwrap());
        assert_eq!(bin.to_binary(), "1010");

        let hex = Mutibs::from_bv(bv_from_hex("a").unwrap());
        assert_eq!(hex.to_binary(), "1010");

        let oct = Mutibs::from_bv(bv_from_oct("12").unwrap());
        assert_eq!(oct.to_binary(), "001010");

        assert!(bv_from_bin("123").is_err());
        let bv = bv_from_hex("xy");
        assert!(bv.is_err());
        let bv = bv_from_oct("89");
        assert!(bv.is_err());
    }

    #[test]
    fn negative_indexing_in_mutable() {
        let m = Mutibs::from_bv(bv_from_bin("10101010").unwrap());

        assert_eq!(m.get_index(-3).unwrap(), false);
        assert_eq!(m.get_index(-8).unwrap(), true);
        assert!(m.get_index(-9).is_err());
    }

    #[test]
    fn mutable_getslice_edge_cases() {
        let m = Mutibs::from_bv(bv_from_bin("11001010").unwrap());

        let empty = m.get_slice(4, 4).unwrap();
        assert_eq!(empty.to_binary(), "");

        let full = m.get_slice(0, m.len()).unwrap();
        assert_eq!(full.to_binary(), "11001010");

        assert!(m.get_slice(9, 10).is_err());
    }

    // #[test]
    // fn bit_ops_performance() {
    //     let bv1 = crate::helpers::bv_from_random(10_000_000, false, &None).unwrap();
    //     let bv2 = crate::helpers::bv_from_random(10_000_000, false, &None).unwrap();
    //     let b1 = Tibs::new(bv1);
    //     let b2 = Tibs::new(bv2);
    //     for _ in 0..100 {
    //         let _ = b1.or(&b2).unwrap();
    //     }
    // }
}
