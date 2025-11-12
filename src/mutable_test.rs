#[cfg(test)]
mod tests {
    use crate::core::BitCollection;
    use crate::tibs_::Tibs;
    use crate::mutibs::Mutibs;

    #[test]
    fn test_set_and_get_index() {
        let mut mb = <Mutibs as BitCollection>::from_zeros(8);
        mb._set_index(true, 3).unwrap();
        assert_eq!(mb._getindex(3).unwrap(), true);
        mb._set_index(false, 3).unwrap();
        assert_eq!(mb._getindex(3).unwrap(), false);
    }

    #[test]
    fn test_set_slice() {
        let mut mb = <Mutibs as BitCollection>::from_zeros(6);
        let br = <Tibs as BitCollection>::from_ones(2);
        mb._set_slice(2, 4, &br);
        assert_eq!(mb.to_binary(), "001100");
    }

    #[test]
    fn test_overwrite_slice() {
        let mut mb = <Mutibs as BitCollection>::from_zeros(6);
        let br = <Tibs as BitCollection>::from_ones(2);
        mb._set_slice(2, 4, &br);
        assert_eq!(mb.to_binary(), "001100");
    }

    #[test]
    fn test_iand_ior_ixor() {
        let mut mb1 = <Mutibs as BitCollection>::from_ones(4);
        let mb2 = <Mutibs as BitCollection>::from_zeros(4);
        mb1._iand(&mb2).unwrap();
        assert_eq!(mb1.to_binary(), "0000");
        mb1._ior(&<Mutibs as BitCollection>::from_ones(4))
            .unwrap();
        assert_eq!(mb1.to_binary(), "1111");
        mb1._ixor(&<Mutibs as BitCollection>::from_ones(4))
            .unwrap();
        assert_eq!(mb1.to_binary(), "0000");
    }

    #[test]
    fn test_unusual_slice_setting() {
    let mut mb = Mutibs::from_hexadecimal("0x12345678").unwrap();
    let zeros = <Tibs as BitCollection>::from_zeros(8);
    mb._set_slice(0 , 8, &zeros);
    assert_eq!(mb.to_hexadecimal().unwrap(), "00345678");
    }

}
