#
# from bitformat.rust import BitRust
# import pytest
#
# def test_creation():
#     b = BitRust.from_zeros(10)
#     assert len(b) == 10
#     assert b.slice_to_bin(0, 10) == '0000000000'
#
#     b2 = BitRust.from_ones(8)
#     assert b2.slice_to_bin(0, 8) == '11111111'
#     assert b2.slice_to_hex(0, 8) == 'ff'
#
# def test_creation_from_bytes():
#     b3 = BitRust.from_bytes(b'hello')
#     assert b3.slice_to_hex(0, len(b3)) == '68656c6c6f'
#     assert b3.to_bytes() == b'hello'
#     b4 = b3._getslice(8, 40)
#     assert b4.slice_to_hex(0, len(b4)) == '656c6c6f'
#     assert b4.to_bytes() == b'ello'
#
# def test_join():
#     a = BitRust.from_zeros(4)
#     b = BitRust.from_ones(4)
#     c = BitRust.from_joined([a, b])
#     assert c.slice_to_bin(0, len(c)) == '00001111'
#     c = c._clone_as_mutable()
#     c.reverse()
#     assert c.slice_to_bin(0, len(c)) == '11110000'
#
# def test_find():
#     a = BitRust.from_bin('00000110001110')
#     b = BitRust.from_bin('11')
#     assert a.find(b, 0, False) == 5
#     assert a.find(b, 0, True) is None
#
# def test_from_oct():
#     a = BitRust.from_oct('776')
#     assert a.slice_to_bin(0, len(a)) == '111111110'
#     with pytest.raises(ValueError):
#         b = BitRust.from_oct('abc')
#     assert a.slice_to_oct(0, len(a)) == "776"
#
# def test_to_bytes():
#     a = BitRust.from_ones(16)
#     assert a.to_bytes() == b"\xff\xff"
#     b = a._getslice(7, len(a))
#     assert b.to_bytes() == b"\xff\x80"
