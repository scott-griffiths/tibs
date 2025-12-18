#[cfg(test)]
mod tests {
    use crate::core::BitCollection;
    use crate::mutibs::Mutibs;
    use crate::tibs_::Tibs;

    #[test]
    fn test_set_and_get_index() {
        let mut mb = <Mutibs as BitCollection>::from_zeros(8);
        mb.set_index(true, 3).unwrap();
        assert_eq!(mb.get_index(3).unwrap(), true);
        mb.set_index(false, 3).unwrap();
        assert_eq!(mb.get_index(3).unwrap(), false);
    }

    #[test]
    fn test_set_slice() {
        let mut mb = <Mutibs as BitCollection>::from_zeros(6);
        let br = <Tibs as BitCollection>::from_ones(2);
        mb.set_slice(2, 4, &br);
        assert_eq!(mb.to_binary(), "001100");
    }

    #[test]
    fn test_overwrite_slice() {
        let mut mb = <Mutibs as BitCollection>::from_zeros(6);
        let br = <Tibs as BitCollection>::from_ones(2);
        mb.set_slice(2, 4, &br);
        assert_eq!(mb.to_binary(), "001100");
    }
    

    #[test]
    fn test_unusual_slice_setting() {
        let mut mb = Mutibs::from_hexadecimal("0x12345678").unwrap();
        let zeros = <Tibs as BitCollection>::from_zeros(8);
        mb.set_slice(0, 8, &zeros);
        assert_eq!(mb.to_hexadecimal().unwrap(), "00345678");
    }
}
