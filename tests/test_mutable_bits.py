#!/usr/bin/env python
import pytest
from tibs import Tibs, Mutibs


def test_creation():
    a = Mutibs.from_zeros(5)
    b = Mutibs.from_bools([1, 0, 0])
    c = Mutibs.from_bytes(b'123')
    e = Mutibs.from_string('0b1110')
    for x in [a, b, c, e]:
        assert isinstance(x, Mutibs)


def test_append():
    # Basic append functionality
    a = Mutibs('0x0f')
    a.append('0x0a')
    assert a == '0x0f0a'

    # Verify method chaining
    a = Mutibs('0x01')
    result = a.append('0x02').append('0x03')
    assert a == '0x010203'
    assert result is a  # Should return self

    # Different input types
    a = Mutibs('0b1010')
    a.append(Tibs('0b1111'))  # Tibs object
    assert a == '0b10101111'
    a.append(Tibs.from_bools([True, False, True]))
    assert a == '0b10101111101'

    # Empty append
    a = Mutibs('0x42')
    a.append(Tibs())
    assert a == '0x42'


def test_prepend():
    # Basic prepend functionality
    a = Mutibs('0x0f')
    a.prepend('0x0a')
    assert a == '0x0a0f'

    # Verify method chaining
    a = Mutibs('0x03')
    result = a.prepend('0x02').prepend('0x01')
    assert a == '0x010203'
    assert result is a  # Should return self

    # Different input types
    a = Mutibs('0b1010')
    a.prepend(Tibs('0b1111'))  # Tibs object
    assert a == '0b11111010'
    a.prepend(Tibs.from_bools([True, False, True]))  # Boolean list
    assert a == '0b10111111010'

    # Empty prepend
    a = Mutibs('0x42')
    a.prepend(Tibs())
    assert a == '0x42'


def test_append_prepend_together():
    # Test combining both operations
    a = Mutibs('0xAA')
    a.append('0xBB').prepend('0xCC')
    assert a == '0xCCAABB'


def test_setitem_single_bit():
    a = Mutibs('0b0010')
    a[0] = 1
    assert a == '0b1010'
    a[2] = 0
    assert a == '0b1000'
    a[-1] = True
    assert a == '0b1001'
    a[-4] = False
    assert a == '0b0001'
    # Out of range
    with pytest.raises(IndexError):
        a[4] = 1
    with pytest.raises(IndexError):
        a[-5] = 0


def test_setitem_slice():
    a = Mutibs('0b101010')
    a[1:4] = '0b111'
    assert a == '0b111110'
    a[0:2] = Tibs('0b00')
    assert a == '0b001110'
    a[2:5] = Mutibs('0b101')
    assert a == '0b001010'
    # Negative indices
    a[-3:-1] = '0b11'
    assert a == '0b001110'
    # Full slice
    a[:] = '0b000000'
    assert a == '0b000000'
    # Empty slice
    a[2:2] = '0b'
    assert a == '0b000000'
    a[1:3] = '0b1'
    assert a == '0b01000'
    # Stepping is not allowed
    with pytest.raises(ValueError):
        a[::2] = '0b00'
    a[10:12] = '0b00'  # Out of range, so just extends.
    assert a == '0b0100000'


def test_setitem_slice_length_change():
    a = Mutibs('0b1010')
    a[1:3] = '0b111'
    assert a == '0b11110'  # Length increased by 1
    a[0:2] = '0b0'
    assert a == '0b0110'
    a[1:2] = '0b1111'
    assert a == '0b0111110'
    a[0:15] = '0b1'
    assert a == '0b1'
    # Setting to empty
    a[:] = ''
    assert a == ''
    # Setting empty slice to non-empty
    a[0:0] = '0b101'
    assert a == '0b101'


def test_delitem_single_bit():
    # Test deleting single bits
    a = Mutibs('0b1010')
    del a[1]
    assert a == '0b110'

    a = Mutibs('0b1010')
    del a[-1]
    assert a == '0b101'

    # Out of range
    with pytest.raises(IndexError):
        a = Mutibs('0b101')
        del a[3]

    with pytest.raises(IndexError):
        a = Mutibs('0b101')
        del a[-4]


def test_delitem_slice():
    # Test deleting slices
    a = Mutibs('0b101010')
    del a[1:4]
    assert a == '0b110'

    # Negative indices
    a = Mutibs('0b101010')
    del a[-4:-2]
    assert a == '0b1010'

    # Empty slice should do nothing
    a = Mutibs('0b1010')
    del a[2:2]
    assert a == '0b1010'

    # Full slice deletion
    a = Mutibs('0b1010')
    del a[:]
    assert a == ''

    # Partial indices
    a = Mutibs('0b101010')
    del a[2:]  # Delete from index 2 to the end
    assert a == '0b10'

    a = Mutibs('0b101010')
    del a[:2]  # Delete from start to index 2
    assert a == '0b1010'


def test_delitem_with_step():
    # Test slices with step
    a = Mutibs('0b101010')
    del a[::2]  # Delete every other bit
    assert a == '0b000'
    with pytest.raises(ValueError):
        del a[::0]


def test_delitem_edge_cases():
    # Empty bits
    a = Mutibs()
    with pytest.raises(IndexError):
        del a[0]

    a = Mutibs('0b1010')
    del a[10:20]  # Out of range slice, should do nothing
    assert a == '0b1010'

    # Delete last bit
    a = Mutibs('0b1')
    del a[0]
    assert a == ''


def test_inplace_add():
    a = Mutibs('0x123')
    a += '0xff'
    assert a == '0x123ff'


def test_inplace_mul():
    a = Mutibs()
    a *= 10000
    assert a == Mutibs()
    a += '0b10'
    a *= 5
    assert a == '0b1010101010'


def test_or():
    a = Mutibs('0x0f')
    b = Mutibs('0xf0')
    c = a | b
    assert c == '0xff'


def test_ior():
    a = Mutibs('0xf00')
    a |= '0x00a'
    assert a == '0xf0a'


def test_iand():
    a = Mutibs('0b1100')
    a &= '0b1010'
    assert a == '0b1000'
    b = Mutibs('0b1111')
    a &= b
    assert a == '0b1000'
    from tibs import Tibs
    c = Tibs('0b0100')
    a &= c
    assert a == '0b0000'


def test_and():
    a = Mutibs('0b1100')
    b = Mutibs('0b1010')
    c = a & b
    assert c == '0b1000'
    d = Tibs('0b0110')
    e = a & d
    assert e == '0b0100'


def test_ixor():
    a = Mutibs('0b1100')
    a ^= '0b1010'
    assert a == '0b0110'
    b = Mutibs('0b0011')
    a ^= b
    assert a == '0b0101'
    c = Tibs('0b1100')
    a ^= c
    assert a == '0b1001'


def test_xor():
    a = Mutibs('0b1100')
    b = Mutibs('0b1010')
    c = a ^ b
    assert c == '0b0110'
    d = Tibs('0b0110')
    e = a ^ d
    assert e == '0b1010'


# def test_constructors():
#     a = Mutibs.from_dtype('f32', 0.5)
#     b = Mutibs.from_dtype('bytes', b'123')
#     c = Mutibs.from_dtype('bin', '100')
#     d = Mutibs.from_dtype('oct', '7654')
#
#     b.prepend(b)
#     assert b == Mutibs.from_bytes(b'123123')
#
#     c.append(d)
#     assert c == '0o47654'
#     d.append(d)
#     assert d == '0o76547654'

def test_invert():
    a = Mutibs('0b1110')
    b = ~a
    assert b == '0b0001'
    assert a == '0b1110'


def test_insert_basic():
    # Basic insert functionality
    a = Mutibs('0b1010')
    a.insert(2, '0b11')
    assert a == '0b101110'


def test_insert_beginning():
    # Insert at beginning
    a = Mutibs('0b1010')
    a.insert(0, '0b11')
    assert a == '0b111010'


def test_insert_end():
    # Insert at end
    a = Mutibs('0b1010')
    a.insert(4, '0b11')
    assert a == '0b101011'


def test_insert_empty():
    # Insert empty bits
    a = Mutibs('0b1010')
    a.insert(2, '')
    assert a == '0b1010'


def test_insert_from_bits():
    # Insert with Tibs object
    a = Mutibs('0b1010')
    a.insert(2, Tibs('0b11'))
    assert a == '0b101110'


def test_insert_from_mutable_bits():
    # Insert with Mutibs object
    a = Mutibs('0b1010')
    a.insert(2, Mutibs('0b11'))
    assert a == '0b101110'


def test_insert_chaining():
    # Method chaining
    a = Mutibs('0b10')
    result = a.insert(1, '0b1').insert(2, '0b0')
    assert a == '0b1100'
    assert result is a


def test_insert_beyond_length():
    # Position beyond length
    a = Mutibs('0b1010')
    a.insert(5, '0b11')  # Position beyond length
    assert a == '0b101011'  # Just appends - standard Python behaviour


def test_set_single_bit_to_one():
    # Basic set functionality - setting a single bit to 1
    a = Mutibs('0b0000')
    a.set(1, 2)
    assert a == '0b0010'


def test_set_single_bit_to_zero():
    # Setting a single bit to 0
    a = Mutibs('0b1111')
    a.set(0, 2)
    assert a == '0b1101'


def test_set_with_boolean_values():
    # Setting with boolean values
    a = Mutibs('0b0000')
    a.set(True, 1)
    assert a == '0b0100'
    a.set(False, 1)
    assert a == '0b0000'


def test_set_with_negative_index():
    # Setting with negative index
    a = Mutibs('0b0010')
    a.set(1, -1)
    assert a == '0b0011'
    a.set(0, -2)
    assert a == '0b0001'


def test_set_multiple_positions():
    # Setting multiple positions
    a = Mutibs('0b0000')
    a.set(1, [0, 2])
    assert a == '0b1010'


def test_set_mixed_indices():
    # Setting with mixed positive and negative indices
    a = Mutibs('0b0000')
    a.set(1, [1, -1])
    assert a == '0b0101'


def test_set_with_range():
    # Setting with range
    a = Mutibs('0b0000')
    a.set(1, range(4))
    assert a == '0b1111'


def test_set_with_empty_sequence():
    # Setting with an empty sequence
    a = Mutibs('0b1010')
    a.set(0, [])
    assert a == '0b1010'  # Should remain unchanged


def test_set_method_chaining():
    # Method chaining
    a = Mutibs('0b0000')
    result = a.set(1, 0).set(1, 2)
    assert a == '0b1010'
    assert result is a


def test_set_with_non_boolean_values():
    # Testing non-boolean values
    a = Mutibs('0b0000')
    a.set("string", 1)  # Any non-empty string evaluates to True
    assert a == '0b0100'
    a.set(0, 1)  # 0 evaluates to False
    assert a == '0b0000'


def test_set_index_out_of_range():
    # Error cases
    with pytest.raises(IndexError):
        a = Mutibs('0b1010')
        a.set(1, 4)  # Index out of range


def test_set_negative_index_out_of_range():
    with pytest.raises(IndexError):
        a = Mutibs('0b1010')
        a.set(0, -5)  # Negative index out of range


def test_invert_all():
    # Test invert method with no argument (inverts all bits)
    a = Mutibs('0b1010')
    a.invert()
    assert a == '0b0101'


def test_invert_single_bit():
    # Test inverting single bit
    a = Mutibs('0b1010')
    a.invert(1)
    assert a == '0b1110'


def test_invert_with_negative_index():
    # Test with negative index
    a = Mutibs('0b1010')
    a.invert(-1)
    assert a == '0b1011'


def test_invert_multiple_positions():
    # Test with list of positions
    a = Mutibs('0b1010')
    a.invert([0, 2])
    assert a == '0b0000'


def test_invert_mixed_indices():
    # Test with mixed positive and negative indices
    a = Mutibs('0b1010')
    a.invert([0, -2])
    assert a == '0b0000'


def test_invert_with_range():
    # Test with range
    a = Mutibs('0b1010')
    a.invert(range(2))
    assert a == '0b0110'


def test_invert_chaining():
    # Method chaining
    a = Mutibs('0b1010')
    result = a.invert(1).invert(2)
    assert a == '0b1100'
    assert result is a


def test_invert_index_out_of_range():
    # Error cases
    with pytest.raises(IndexError):
        a = Mutibs('0b1010')
        a.invert(4)  # Index out of range


def test_invert_negative_index_out_of_range():
    with pytest.raises(IndexError):
        a = Mutibs('0b1010')
        a.invert(-5)  # Negative index out of range


def test_invert_empty_bits():
    # Empty Mutibs
    a = Mutibs()
    a.invert()  # Inverting empty bits should do nothing
    assert a == ''


#
# def test_replace_basic():
#     # Basic replace functionality
#     a = Mutibs('0b10101010')
#     a.replace('0b10', '0b111')
#     assert a == '0b111111111111'
#
# def test_replace_same_length():
#     # Replace with same length pattern
#     a = Mutibs('0b10101010')
#     a.replace('0b10', '0b00')
#     assert a == '0b00000000'
#
# def test_replace_with_empty():
#     # Replace with empty bits (should effectively delete)
#     a = Mutibs('0b10101010')
#     a.replace('0b10', '')
#     assert a == ''
#
# def test_replace_with_count():
#     # Replace only first occurrences with count parameter
#     a = Mutibs('0b10101010')
#     a.replace('0b10', '0b00', count=2)
#     assert a == '0b00001010'
#
# def test_replace_with_start():
#     # Replace with start parameter
#     a = Mutibs('0b10101010')
#     a.replace('0b10', '0b11', start=2)
#     assert a == '0b10111111'
#
# def test_replace_with_end():
#     # Replace with end parameter
#     a = Mutibs('0b10101010')
#     a.replace('0b10', '0b11', end=4)
#     assert a == '0b11111010'
#
# def test_replace_with_start_end():
#     # Replace with both start and end parameters
#     a = Mutibs('0b10101010')
#     a.replace('0b10', '0b11', start=2, end=6)
#     assert a == '0b10111110'
#
# def test_replace_byte_aligned():
#     # Replace with byte_aligned=True
#     a = Mutibs('0b10101010')
#     a.replace('0b1010', '0b1111', byte_aligned=True)
#     assert a == '0b11111010'
#
# def test_replace_method_chaining():
#     # Method chaining
#     a = Mutibs('0b10101010')
#     result = a.replace('0b10', '0b11').replace('0b11', '0b00')
#     assert a == '0b00000000'
#     assert result is a
#
# def test_replace_different_types():
#     # Replace with different types
#     a = Mutibs('0b10101010')
#     a.replace(Tibs('0b10'), Mutibs('0b11'))
#     assert a == '0b11111111'
#
# def test_replace_empty_pattern():
#     # Empty pattern (should raise error)
#     with pytest.raises(ValueError):
#         a = Mutibs('0b1010')
#         a.replace('', '0b11')
#
# def test_replace_pattern_not_found():
#     # Pattern not found
#     a = Mutibs('0b1010')
#     a.replace('0b11', '0b00')
#     assert a == '0b1010'  # Should remain unchanged
#
# def test_replace_with_count_zero():
#     # Count=0 (should not replace anything)
#     a = Mutibs('0b10101010')
#     a.replace('0b10', '0b11', count=0)
#     assert a == '0b10101010'

def test_reverse_basic():
    # Basic reverse functionality
    a = Mutibs('0b1010')
    a.reverse()
    assert a == '0b0101'


def test_reverse_palindrome():
    # Palindrome should remain the same when reversed
    a = Mutibs('0b1001')
    a.reverse()
    assert a == '0b1001'


def test_reverse_empty():
    # Reverse empty Mutibs
    a = Mutibs()
    a.reverse()
    assert a == ''


def test_reverse_single_bit():
    # Reverse single bit
    a = Mutibs('0b1')
    a.reverse()
    assert a == '0b1'


def test_reverse_hex():
    # Reverse with hex representation
    a = Mutibs('0xAB')
    a.reverse()
    assert a == '0xd5'  # 0xAB = 10101011 -> 11010101 = 0xd5


def test_reverse_method_chaining():
    # Method chaining
    a = Mutibs('0b1100')
    result = a.reverse()
    assert a == '0b0011'
    assert result is a


def test_reverse_idempotence():
    # Reverse twice should give original
    a = Mutibs('0b10110')
    a.reverse().reverse()
    assert a == '0b10110'


def test_rol_basic():
    # Basic rotate left functionality
    a = Mutibs('0b1010')
    a.rol(1)
    assert a == '0b0101'


def test_rol_full_rotation():
    # Rotating by the full length should return the original
    a = Mutibs('0b1010')
    a.rol(4)
    assert a == '0b1010'


def test_rol_wraparound():
    # Rotating by more than length should wrap around
    a = Mutibs('0b1010')
    a.rol(5)
    assert a == '0b0101'  # Same as rol(1)


def test_rol_with_start_end():
    # Rotating with start and end parameters
    a = Mutibs('0b10101100')
    a.rol(2, start=2, end=6)
    assert a == '0b10111000'


def test_rol_method_chaining():
    # Method chaining
    a = Mutibs('0b1010')
    result = a.rol(1)
    assert a == '0b0101'
    assert result is a


def test_rol_negative_amount():
    # Error cases - negative rotation
    with pytest.raises(ValueError):
        a = Mutibs('0b1010')
        a.rol(-1)  # Negative rotation amount


def test_rol_empty_bits():
    # Error cases - empty bits
    with pytest.raises(ValueError):
        a = Mutibs()
        a.rol(1)  # Empty Mutibs


def test_rol_zero_rotation():
    # Zero rotation should not change anything
    a = Mutibs('0b1010')
    a.rol(0)
    assert a == '0b1010'


def test_rol_large_rotation():
    # Large rotation value
    a = Mutibs('0b1010')
    a.rol(1000000)  # Should be equivalent to rol(0) since 1000000 % 4 = 0
    assert a == '0b1010'


def test_ror_basic():
    # Basic rotate right functionality
    a = Mutibs('0b1010')
    a.ror(1)
    assert a == '0b0101'


def test_ror_full_rotation():
    # Rotating by the full length should return the original
    a = Mutibs('0b1010')
    a.ror(4)
    assert a == '0b1010'


def test_ror_wraparound():
    # Rotating by more than length should wrap around
    a = Mutibs('0b1010')
    a.ror(5)
    assert a == '0b0101'  # Same as ror(1)


def test_ror_with_start_end():
    # Rotating with start and end parameters
    a = Mutibs('0b10101100')
    a.ror(2, start=2, end=6)
    assert a == '0b10111000'


def test_ror_method_chaining():
    # Method chaining
    a = Mutibs('0b1010')
    result = a.ror(1)
    assert a == '0b0101'
    assert result is a


def test_rol_ror_cancellation():
    # Rotating left then right should cancel out
    a = Mutibs('0b10110')
    a.rol(2).ror(2)
    assert a == '0b10110'


def test_ror_negative_amount():
    # Error cases - negative rotation
    with pytest.raises(ValueError):
        a = Mutibs('0b1010')
        a.ror(-1)  # Negative rotation amount


def test_ror_empty_bits():
    # Error cases - empty bits
    with pytest.raises(ValueError):
        a = Mutibs()
        a.ror(1)  # Empty Mutibs


def test_ror_zero_rotation():
    # Zero rotation should not change anything
    a = Mutibs('0b1010')
    a.ror(0)
    assert a == '0b1010'


def test_ror_large_rotation():
    # Large rotation value
    a = Mutibs('0b1010')
    a.ror(1000000)  # Should be equivalent to ror(0) since 1000000 % 4 = 0
    assert a == '0b1010'


def test_byte_swap_basic():
    # Basic byte_swap functionality with default parameters
    a = Mutibs('0x1234')
    a.byte_swap()
    assert a == '0x3412'


def test_byte_swap_with_length():
    # Byte swap with specific byte_length parameter
    a = Mutibs('0x12345678')
    a.byte_swap(2)
    assert a == '0x34127856'


def test_byte_swap_single_byte():
    # Byte swap single byte (no change)
    a = Mutibs('0x12')
    a.byte_swap(1)
    assert a == '0x12'


def test_byte_swap_method_chaining():
    # Method chaining
    a = Mutibs('0x1234')
    result = a.byte_swap()
    assert a == '0x3412'
    assert result is a


def test_byte_swap_idempotence():
    # Byte swap twice should return to original
    a = Mutibs('0x12345678')
    a.byte_swap(2).byte_swap(2)
    assert a == '0x12345678'


def test_byte_swap_non_multiple_of_8():
    # Non-multiple of 8 bits
    with pytest.raises(ValueError):
        a = Mutibs('0b10101')
        a.byte_swap()


def test_byte_swap_empty():
    # Empty Mutibs
    a = Mutibs()
    result = a.byte_swap()
    assert a == ''
    assert result == Mutibs()


def test_byte_swap_negative_length():
    # Negative byte length
    with pytest.raises(ValueError):
        a = Mutibs('0x1234')
        a.byte_swap(-1)


def test_byte_swap_zero_length():
    # Zero byte length
    with pytest.raises(ValueError):
        a = Mutibs('0x1234')
        a.byte_swap(0)


def test_byte_swap_not_multiple_of_byte_length():
    # Not a multiple of byte_length
    with pytest.raises(ValueError):
        a = Mutibs('0x123456')  # 3 bytes
        a.byte_swap(2)  # Not a multiple of 2 bytes


def test_to_tibs_basic():
    # Basic conversion
    a = Mutibs('0b1010')
    b = a.to_tibs()
    assert isinstance(b, Tibs)
    assert b == '0b1010'


def test_to_tibs_immutable_copy_operations():
    # Original shouldn't change when immutable copy is modified
    a = Mutibs('0b1010')
    b = a.to_tibs()
    c = ~b
    assert a == '0b1010'  # Original remains unchanged
    assert b == '0b1010'  # Original immutable copy unchanged
    assert c == '0b0101'  # New inverted copy


def test_to_tibs_original_modifications():
    # Changes to original shouldn't affect the immutable copy
    a = Mutibs('0b1010')
    b = a.to_tibs()
    a.invert()
    assert a == '0b0101'  # Original changed
    assert b == '0b1010'  # Immutable copy remains unchanged


def test_to_tibs_empty():
    # Empty Mutibs conversion
    a = Mutibs()
    b = a.to_tibs()
    assert isinstance(b, Tibs)
    assert b == ''
    assert len(b) == 0


def test_mutable_bits_from_bits():
    # Test creating Mutibs from Tibs object
    b = Tibs('0b1010')
    a = b.to_mutibs()
    assert a == '0b1010'
    assert isinstance(a, Mutibs)

    # Modification should not affect original
    a.invert()
    assert a == '0b0101'
    assert b == '0b1010'


def test_setitem_with_bits_object():
    # Test setting slices using Tibs objects
    a = Mutibs('0b1010')
    b = Tibs('0b11')
    a[1:3] = b
    assert a == '0b1110'


def test_iadd_with_bits():
    # Test in-place add with Tibs objects
    a = Mutibs('0x12')
    b = Tibs('0x34')
    a += b
    assert a == '0x1234'


def test_iadd_multiple_types():
    # Test in-place add with various types
    a = Mutibs('0b1010')
    a += '0b11'  # String
    a += Tibs('0b00')  # Tibs object
    a += Mutibs('0b111')  # Another Mutibs
    assert a == '0b10101100111'


def test_imul_repeats():
    # Test in-place multiply
    a = Mutibs('0b101')
    a *= 3
    assert a == '0b101101101'

    # Test with zero
    b = Mutibs('0b111')
    b *= 0
    assert b == ''


def test_delitem_sequence():
    # Test deleting multiple items in sequence
    a = Mutibs('0b10101010')
    del a[0]
    assert a == '0b0101010'
    del a[2]
    assert a == '0b011010'
    del a[-1]
    assert a == '0b01101'


def test_setitem_complex_cases():
    # Test setting a slice with different-length content
    a = Mutibs('0b1010')
    a[1:3] = '0b111'  # Replace 2 bits with 3 bits
    assert a == '0b11110'

    # Replace with empty content (effectively deleting)
    a[2:4] = ''
    assert a == '0b110'

    # Replace everything with shorter content
    a[:] = '0b1'
    assert a == '0b1'


def test_bit_operations_with_bits():
    # Testing bitwise AND with Tibs
    a = Mutibs('0b1100')
    b = Tibs('0b1010')
    a &= b
    assert a == '0b1000'

    # Testing bitwise OR with Tibs
    a = Mutibs('0b1100')
    b = Tibs('0b0011')
    a |= b
    assert a == '0b1111'

    # Testing bitwise XOR with Tibs
    a = Mutibs('0b1100')
    b = Tibs('0b1010')
    a ^= b
    assert a == '0b0110'


def test_equality_with_bits():
    # Test equality comparison with Tibs
    a = Mutibs('0b1010')
    b = Tibs('0b1010')
    assert a == b

    # Test after modification
    a[0] = 0
    assert a != b
    assert a == '0b0010'


def test_interleaved_operations():
    # Test a sequence of interleaved operations
    a = Mutibs('0b1010')
    a[1:3] = '0b00'
    a += '0b11'
    a.invert(0)
    del a[-1]
    assert a == '0b00001'

    # Chain multiple operations
    a = Mutibs('0b101')
    result = a.append('0b010').invert().reverse()
    assert result == a  # Verify chaining returns self
    assert a == '0b101010'  # 101 + 010 -> 101010 -> 010101 (invert) -> 010010 (reverse)


def test_mutable_bits_conversion_roundtrip():
    # Test round-trip conversion between Tibs and Mutibs
    orig = Tibs('0b10101100')
    mutable = orig.to_mutibs()
    mutable.invert(range(4))  # Modify some bits
    back_to_tibs = mutable.to_tibs()

    assert isinstance(back_to_tibs, Tibs)
    assert back_to_tibs == '0b01011100'
    assert orig == '0b10101100'  # Original should be unchanged


def test_inserting_bits_objects():
    # Test inserting Tibs objects at specific positions
    a = Mutibs('0b1010')
    b = Tibs('0b11')
    a.insert(2, b)
    assert a == '0b101110'

    # Insert at beginning
    c = Tibs('0b00')
    a.insert(0, c)
    assert a == '0b00101110'


def test_mixed_representation_operations():
    # Test operations with mixed representations (binary, hex)
    a = Mutibs('0b1010')
    a += '0x3A'
    assert a == '0b1010_0011_1010'

    a[4:8] = '0o7'
    assert a == '0b1010_111_1010'


def test_shifting_inplace():
    # Test in-place shifting operations
    a = Mutibs('0b001010')
    a <<= 2
    assert a == '0b101000'
    a >>= 3
    assert a == '0b000101'
    with pytest.raises(ValueError):
        a <<= -1
    with pytest.raises(ValueError):
        a >>= -1


def test_all_any():
    a = Mutibs('0x00')
    assert not a.any()
    assert not a.all()
    b = Mutibs('0xff')
    assert b.any()
    assert b.all()


def test_shifts():
    a = Mutibs.from_ones(5)
    a += '0b0'
    b = a << 1
    assert b == '0b111100'
    c = b >> 1
    assert c == '0b011110'


def test_str():
    a = Mutibs.from_ones(8)
    assert a.__str__() == '0xff'
    assert a.__repr__() == "Mutibs('0xff')"


def test_logical_op_misc():
    a = Mutibs('0xffff')
    b = Mutibs('0x000')
    try:
        _ = a & b
    except ValueError as e:
        assert "12" in str(e)
        assert "16" in str(e)


def test_auto_conversions():
    a = Mutibs()
    with pytest.raises(TypeError):
        _ = a + None
    with pytest.raises(TypeError):
        _ = a + True
    with pytest.raises(TypeError):
        _ = a + False
    with pytest.raises(TypeError):
        _ = a + 1
    b = a + '0x1'
    assert isinstance(b, Mutibs) and b == '0x1'
    b = a + b'123'
    assert isinstance(b, Mutibs) and b == b'123'
    b = a + [1, 0]
    assert isinstance(b, Mutibs) and b == '0b10'
    b = a + (1, 0, 'steve')
    assert isinstance(b, Mutibs) and b == '0b101'


def test_clear():
    a = Mutibs()
    a.clear()
    assert a == Mutibs()
    assert not a
    a += '0b1'
    assert a
    a.clear()
    assert not a
    assert a == Mutibs()


def test_reserve():
    a = Mutibs()
    assert a.capacity() == 0
    a.reserve(10)
    assert a.capacity() >= 10
    a += Mutibs.from_random(1000000)
    b4 = a.capacity()
    assert b4 >= 1000000
    a.clear()
    assert a.capacity() == b4


def test_insert_slice():
    a = Mutibs('0xff')
    a[0:0] = '0xab'
    assert a == '0xabff'
    a[0:0] = a
    assert a == '0xabffabff'


def test_del_ranges():
    a = Mutibs.from_zeros(10)
    del a[5:3]
    assert len(a) == 10


def test_set_item_with_step():
    a = Mutibs('0b000000')
    a[::2] = '0b110'
    assert a == '0b101000'


def test_iter():
    a = Mutibs('0b110')
    with pytest.raises(TypeError):
        _ = [bool(q) for q in a]
