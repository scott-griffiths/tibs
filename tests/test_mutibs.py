#!/usr/bin/env python
import pytest
from tibs import Tibs, Mutibs, ByteOrder, Codec


def test_creation():
    a = Mutibs.from_zeros(5)
    b = Mutibs.from_bools([1, 0, 0])
    c = Mutibs.from_bytes(b'123')
    e = Mutibs.from_string('0b1110')
    for x in [a, b, c, e]:
        assert isinstance(x, Mutibs)


def test_extend():
    # Basic extend functionality
    a = Mutibs('0x0f')
    a.extend('0x0a')
    assert a == Tibs('0x0f0a')

    a = Mutibs('0x01')
    a.extend('0x02')
    a.extend('0x03')
    assert a == Tibs('0x010203')

    # Different input types
    a = Mutibs('0b1010')
    a.extend(Tibs('0b1111'))  # Tibs object
    assert a == Tibs('0b10101111')
    a.extend(Tibs.from_bools([True, False, True]))
    assert a == Tibs('0b10101111101')

    # Empty extend
    a = Mutibs('0x42')
    a.extend(Tibs())
    assert a == Tibs('0x42')


def test_extend_left():
    # Basic prepend functionality
    a = Mutibs('0x0f')
    a.extend_left('0x0a')
    assert a == Tibs('0x0a0f')

    a = Mutibs('0x03')
    a.extend_left('0x02')
    a.extend_left('0x01')
    assert a == Tibs('0x010203')

    # Different input types
    a = Mutibs('0b1010')
    a.extend_left(Tibs('0b1111'))  # Tibs object
    assert a == Tibs('0b11111010')
    a.extend_left(Tibs.from_bools([True, False, True]))  # Boolean list
    assert a == Tibs('0b10111111010')

    # Empty prepend
    a = Mutibs('0x42')
    a.extend_left(Tibs())
    assert a == Tibs('0x42')


def test_extend_prepend_together():
    # Test combining both operations
    a = Mutibs('0xAA')
    a.extend('0xBB')
    a.extend_left('0xCC')
    assert a == Tibs('0xCCAABB')


def test_setitem_single_bit():
    a = Mutibs('0b0010')
    a[0] = 1
    assert a == Tibs('0b1010')
    a[2] = 0
    assert a == Tibs('0b1000')
    a[-1] = True
    assert a == Tibs('0b1001')
    a[-4] = False
    assert a == Tibs('0b0001')
    # Out of range
    with pytest.raises(IndexError):
        a[4] = 1
    with pytest.raises(IndexError):
        a[-5] = 0


def test_setitem_slice():
    a = Mutibs('0b101010')
    a[1:4] = '0b111'
    assert a == Tibs('0b111110')
    a[0:2] = Tibs('0b00')
    assert a == Tibs('0b001110')
    a[2:5] = Mutibs('0b101')
    assert a == Tibs('0b001010')
    # Negative indices
    a[-3:-1] = '0b11'
    assert a == Tibs('0b001110')
    # Full slice
    a[:] = '0b000000'
    assert a == Tibs('0b000000')
    # Empty slice
    a[2:2] = '0b'
    assert a == Tibs('0b000000')
    a[1:3] = '0b1'
    assert a == Tibs('0b01000')
    # Stepping is not allowed
    with pytest.raises(ValueError):
        a[::2] = '0b00'
    a[10:12] = '0b00'  # Out of range, so just extends.
    assert a == Tibs('0b0100000')


def test_setitem_slice_on_sliced_mutibs():
    # Slicing can leave the underlying storage starting mid-byte; the
    # byte-aligned fast paths must still write to the right place.
    a = Mutibs.from_zeros(24)
    b = a[4:20]
    b[0:8] = Mutibs.from_bytes(b'\xff')
    assert b.to_bin() == '1111111100000000'
    b.set([8, -1])
    assert b.to_bin() == '1111111110000001'
    b.unset(0)
    assert b.to_bin() == '0111111110000001'


def test_delitem_slice_on_sliced_mutibs():
    a = Mutibs.from_string('0b1010') + Mutibs.from_bytes(b'\xf0\x0f')
    b = a[4:20]
    expected = b.to_bin()
    del b[0:8]
    assert b.to_bin() == expected[8:]


def test_setitem_slice_length_change():
    a = Mutibs('0b1010')
    a[1:3] = '0b111'
    assert a == Tibs('0b11110')  # Length increased by 1
    a[0:2] = '0b0'
    assert a == Tibs('0b0110')
    a[1:2] = '0b1111'
    assert a == Tibs('0b0111110')
    a[0:15] = '0b1'
    assert a == Tibs('0b1')
    # Setting to empty
    a[:] = ''
    assert a == Tibs()
    # Setting empty slice to non-empty
    a[0:0] = '0b101'
    assert a == Tibs('0b101')


def test_delitem_single_bit():
    # Test deleting single bits
    a = Mutibs('0b1010')
    del a[1]
    assert a == Tibs('0b110')

    a = Mutibs('0b1010')
    del a[-1]
    assert a == Tibs('0b101')

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
    assert a == Tibs('0b110')

    # Negative indices
    a = Mutibs('0b101010')
    del a[-4:-2]
    assert a == Tibs('0b1010')

    # Empty slice should do nothing
    a = Mutibs('0b1010')
    del a[2:2]
    assert a == Tibs('0b1010')

    # Full slice deletion
    a = Mutibs('0b1010')
    del a[:]
    assert a == Tibs()

    # Partial indices
    a = Mutibs('0b101010')
    del a[2:]  # Delete from index 2 to the end
    assert a == Tibs('0b10')

    a = Mutibs('0b101010')
    del a[:2]  # Delete from start to index 2
    assert a == Tibs('0b1010')


def test_delitem_with_step():
    # Test slices with step
    a = Mutibs('0b101010')
    del a[::2]  # Delete every other bit
    assert a == Tibs('0b000')
    with pytest.raises(ValueError):
        del a[::0]


def test_delitem_edge_cases():
    # Empty bits
    a = Mutibs()
    with pytest.raises(IndexError):
        del a[0]

    a = Mutibs('0b1010')
    del a[10:20]  # Out of range slice, should do nothing
    assert a == Tibs('0b1010')

    # Delete last bit
    a = Mutibs('0b1')
    del a[0]
    assert a == Tibs()


def test_inplace_add():
    a = Mutibs('0x123')
    a += '0xff'
    assert a == Tibs('0x123ff')


def test_inplace_mul():
    a = Mutibs()
    a *= 10000
    assert a == Mutibs()
    a += '0b10'
    a *= 5
    assert a == Tibs('0b1010101010')


def test_find_all():
    a = Mutibs('0b11111')
    assert a.find_all(Tibs('0b1')) == [0, 1, 2, 3, 4]
    assert a.find_all(Tibs('0b11')) == [0, 1, 2, 3]
    assert a.find_all(Tibs('0b10')) == []

    b = Mutibs('0b1001001001001001001')
    assert b.find_all(Tibs('0b1001')) == [0, 3, 6, 9, 12, 15]


def test_chunks():
    a = Mutibs('0x00112233445')
    assert a.chunks(8) == [Tibs('0x00'), Tibs('0x11'), Tibs('0x22'), Tibs('0x33'), Tibs('0x44'), Tibs('0x5')]
    assert a[8:16].chunks(4) == [Tibs('0x1'), Tibs('0x1')]
    assert a[0:44].chunks(4, 4) == [Tibs('0x0'), Tibs('0x0'), Tibs('0x1'), Tibs('0x1')]
    assert Mutibs().chunks(10) == []


def test_split_at_returns_mutibs_pieces():
    m = Mutibs('0b101100')

    pieces = m.split_at([2, -1])

    assert pieces == (Mutibs('0b10'), Mutibs('0b110'), Mutibs('0b0'))
    assert isinstance(pieces, tuple)
    assert all(isinstance(piece, Mutibs) for piece in pieces)

    pieces[0][0] = False
    assert pieces[0] == Tibs('0b00')
    assert m == Tibs('0b101100')


def test_split_at_mutibs_errors():
    m = Mutibs('0b101100')

    with pytest.raises(ValueError, match="out of range"):
        _ = m.split_at(7)
    with pytest.raises(ValueError, match="nondecreasing"):
        _ = m.split_at([4, 3])


def test_or():
    a = Mutibs('0x0f')
    b = Mutibs('0xf0')
    c = a | b
    assert c == Tibs('0xff')


def test_ior():
    a = Mutibs('0xf00')
    a |= '0x00a'
    assert a == Tibs('0xf0a')


def test_iand():
    a = Mutibs('0b1100')
    a &= '0b1010'
    assert a == Tibs('0b1000')
    b = Mutibs('0b1111')
    a &= b
    assert a == Tibs('0b1000')
    c = Tibs('0b0100')
    a &= c
    assert a == Tibs('0b0000')


def test_and():
    a = Mutibs('0b1100')
    b = Mutibs('0b1010')
    c = a & b
    assert c == Tibs('0b1000')
    d = Tibs('0b0110')
    e = a & d
    assert e == Tibs('0b0100')


def test_ixor():
    a = Mutibs('0b1100')
    a ^= '0b1010'
    assert a == Tibs('0b0110')
    b = Mutibs('0b0011')
    a ^= b
    assert a == Tibs('0b0101')
    c = Tibs('0b1100')
    a ^= c
    assert a == Tibs('0b1001')


def test_xor():
    a = Mutibs('0b1100')
    b = Mutibs('0b1010')
    c = a ^ b
    assert c == Tibs('0b0110')
    d = Tibs('0b0110')
    e = a ^ d
    assert e == Tibs('0b1010')


def test_constructors():
    a = Mutibs.from_f(0.5, 32)
    b = Mutibs.from_bytes(b'123')
    c = Mutibs.from_bin('100')
    d = Mutibs.from_oct('7654')

    b.extend_left(b)
    assert b == Mutibs.from_bytes(b'123123')

    c.extend(d)
    assert c == Tibs('0o47654')
    d.extend(d)
    assert d == Tibs('0o76547654')


def test_invert():
    a = Mutibs('0b1110')
    b = ~a
    assert b == Tibs('0b0001')
    assert a == Tibs('0b1110')


def test_insert_basic():
    # Basic insert functionality
    a = Mutibs('0b1010')
    a.insert(2, '0b11')
    assert a == Tibs('0b101110')


def test_insert_beginning():
    # Insert at beginning
    a = Mutibs('0b1010')
    a.insert(0, '0b11')
    assert a == Tibs('0b111010')


def test_insert_end():
    # Insert at end
    a = Mutibs('0b1010')
    a.insert(4, '0b11')
    assert a == Tibs('0b101011')


def test_insert_empty():
    # Insert empty bits
    a = Mutibs('0b1010')
    a.insert(2, '')
    assert a == Tibs('0b1010')


def test_insert_from_bits():
    # Insert with Tibs object
    a = Mutibs('0b1010')
    a.insert(2, Tibs('0b11'))
    assert a == Tibs('0b101110')


def test_insert_from_mutable_bits():
    # Insert with Mutibs object
    a = Mutibs('0b1010')
    a.insert(2, Mutibs('0b11'))
    assert a == Tibs('0b101110')


def test_inserted_returns_new_mutibs():
    a = Mutibs('0b1010')
    b = a.inserted(2, '0b11')
    assert a == Tibs('0b1010')
    assert b == Tibs('0b101110')


def test_insert_chaining():
    a = Mutibs('0b10')
    a.insert(1, '0b1')
    a.insert(2, '0b0')
    assert a == Tibs('0b1100')


def test_insert_beyond_length():
    # Position beyond length
    a = Mutibs('0b1010')
    a.insert(5, '0b11')  # Position beyond length
    assert a == Tibs('0b101011')  # Just extends - standard Python behaviour


def test_set_single_bit_to_one():
    # Basic set functionality - setting a single bit to 1
    a = Mutibs('0b0000')
    a.set(2)
    assert a == Tibs('0b0010')


def test_set_single_bit_to_zero():
    # Setting a single bit to 0
    a = Mutibs('0b1111')
    a.unset(2)
    assert a == Tibs('0b1101')


def test_set_with_boolean_values():
    # Setting with boolean values
    a = Mutibs('0b0000')
    a.set(1)
    assert a == Tibs('0b0100')
    a.unset(1)
    assert a == Tibs('0b0000')


def test_set_with_negative_index():
    # Setting with negative index
    a = Mutibs('0b0010')
    a.set(-1)
    assert a == Tibs('0b0011')
    a.unset(-2)
    assert a == Tibs('0b0001')


def test_set_multiple_positions():
    # Setting multiple positions
    a = Mutibs('0b0000')
    a.set([0, 2])
    assert a == Tibs('0b1010')


def test_set_list_is_atomic_on_invalid_item():
    for positions, error in [([1, 99, 2], IndexError), ([1, 'bad', 2], TypeError)]:
        a = Mutibs('0b0000')
        with pytest.raises(error):
            a.set(positions)
        assert a == Tibs('0b0000')


def test_set_list_accepts_index_objects():
    class Index:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    a = Mutibs('0b00000000')
    a.set([Index(2), Index(-1)])
    assert a == Tibs('0b00100001')


def test_set_long_list():
    a = Mutibs.from_zeros(24)
    a.set(list(range(20)))
    assert a == Tibs.from_ones(20) + Tibs.from_zeros(4)


def test_set_multiple_positions_tuple():
    a = Mutibs('0b0000')
    a.set((0, 2))
    a.unset((2,))
    assert a == Tibs('0b1000')

    b = Mutibs.from_ones(4)
    with pytest.raises(IndexError):
        b.unset((0, 99))
    assert b == Tibs('0b1111')


def test_set_mixed_indices():
    # Setting with mixed positive and negative indices
    a = Mutibs('0b0000')
    a.set([1, -1])
    assert a == Tibs('0b0101')


def test_set_with_range():
    # Setting with range
    a = Mutibs('0b0000')
    a.set(range(4))
    assert a == Tibs('0b1111')


def test_set_with_empty_sequence():
    # Setting with an empty sequence
    a = Mutibs('0b1010')
    a.unset([])
    assert a == Tibs('0b1010')  # Should remain unchanged


def test_set_method_chaining():
    a = Mutibs('0b0000')
    result = a.set(0)
    assert result is None
    a.set(2)
    assert a == Tibs('0b1010')


def test_set_at_returns_new_mutibs():
    a = Mutibs('0b0000')
    b = a.set_at([0, 2])
    assert a == Tibs('0b0000')
    assert b == Tibs('0b1010')
    assert isinstance(b, Mutibs)


def test_unset_at_returns_new_mutibs():
    a = Mutibs('0b1111')
    b = a.unset_at(range(1, 4))
    assert a == Tibs('0b1111')
    assert b == Tibs('0b1000')


def test_set_index_out_of_range():
    # Error cases
    with pytest.raises(IndexError):
        a = Mutibs('0b1010')
        a.set(4)  # Index out of range


def test_set_negative_index_out_of_range():
    with pytest.raises(IndexError):
        a = Mutibs('0b1010')
        a.unset(-5)  # Negative index out of range


def test_invert_all():
    # Test invert method with no argument (inverts all bits)
    a = Mutibs('0b1010')
    a.invert()
    assert a == Tibs('0b0101')


def test_invert_single_bit():
    # Test inverting single bit
    a = Mutibs('0b1010')
    a.invert(1)
    assert a == Tibs('0b1110')


def test_invert_with_negative_index():
    # Test with negative index
    a = Mutibs('0b1010')
    a.invert(-1)
    assert a == Tibs('0b1011')


def test_invert_multiple_positions():
    # Test with list of positions
    a = Mutibs('0b1010')
    a.invert([0, 2])
    assert a == Tibs('0b0000')


def test_invert_mixed_indices():
    # Test with mixed positive and negative indices
    a = Mutibs('0b1010')
    a.invert([0, -2])
    assert a == Tibs('0b0000')


def test_invert_with_range():
    # Test with range
    a = Mutibs('0b1010')
    a.invert(range(2))
    assert a == Tibs('0b0110')


def test_invert_chaining():
    a = Mutibs('0b1010')
    result = a.invert(1)
    assert result is None
    a.invert(2)
    assert a == Tibs('0b1100')


def test_inverted_returns_new_mutibs():
    a = Mutibs('0b1010')
    b = a.inverted([0, -1])
    assert a == Tibs('0b1010')
    assert b == Tibs('0b0011')
    assert isinstance(b, Mutibs)


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
    assert a == Tibs()


def test_replace_basic():
    # Basic replace functionality
    a = Mutibs('0b10101010')
    count = a.replace('0b10', '0b111')
    assert count == 4
    assert a == Tibs('0b111111111111')


def test_replace_same_length():
    # Replace with same length pattern
    a = Mutibs('0b10101010')
    a.replace('0b10', '0b00')
    assert a == Tibs('0b00000000')


def test_replace_with_empty():
    # Replace with empty bits (should effectively delete)
    a = Mutibs('0b10101010')
    a.replace('0b10', '')
    assert a == Tibs()


def test_replace_with_count():
    # Replace only first occurrences with count parameter
    a = Mutibs('0b10101010')
    count = a.replace('0b10', '0b00', count=2)
    assert count == 2
    assert a == Tibs('0b00001010')


def test_replace_with_start():
    # Replace with start parameter
    a = Mutibs('0b10101010')
    a.replace('0b10', '0b11', start=2)
    assert a == Tibs('0b10111111')


def test_replace_with_end():
    # Replace with end parameter
    a = Mutibs('0b10101010')
    a.replace('0b10', '0b11', end=4)
    assert a == Tibs('0b11111010')


def test_replace_with_start_end():
    # Replace with both start and end parameters
    a = Mutibs('0b10101010')
    a.replace('0b10', '0b11', start=2, end=6)
    assert a == Tibs('0b10111110')


def test_replace_byte_aligned():
    # Replace with byte_aligned=True
    a = Mutibs('0b10101010')
    a.replace('0b1010', '0b1111', byte_aligned=True)
    assert a == Tibs('0b11111010')


def test_replace_returns_count():
    a = Mutibs('0b10101010')
    result = a.replace('0b10', '0b11')
    assert result == 4
    result = a.replace('0b11', '0b00')
    assert result == 4
    assert a == Tibs('0b00000000')


def test_replaced_returns_new_mutibs():
    a = Mutibs('0b10101010')
    b = a.replaced('0b10', '0b11', count=2)
    assert a == Tibs('0b10101010')
    assert b == Tibs('0b11111010')


def test_replace_different_types():
    # Replace with different types
    a = Mutibs('0b10101010')
    a.replace(Tibs('0b10'), Mutibs('0b11'))
    assert a == Tibs('0b11111111')


def test_replace_empty_pattern():
    # Empty pattern (should raise error)
    with pytest.raises(ValueError):
        a = Mutibs('0b1010')
        a.replace('', '0b11')


def test_replace_pattern_not_found():
    # Pattern not found
    a = Mutibs('0b1010')
    count = a.replace('0b11', '0b00')
    assert count == 0
    assert a == Tibs('0b1010')  # Should remain unchanged


def test_replace_with_count_zero():
    # Count=0 (should not replace anything)
    a = Mutibs('0b10101010')
    count = a.replace('0b10', '0b11', count=0)
    assert count == 0
    assert a == Tibs('0b10101010')


def test_reverse_basic():
    # Basic reverse functionality
    a = Mutibs('0b1010')
    a.reverse()
    assert a == Tibs('0b0101')


def test_reverse_palindrome():
    # Palindrome should remain the same when reversed
    a = Mutibs('0b1001')
    a.reverse()
    assert a == Tibs('0b1001')


def test_reverse_empty():
    # Reverse empty Mutibs
    a = Mutibs()
    a.reverse()
    assert a == Tibs()


def test_reverse_single_bit():
    # Reverse single bit
    a = Mutibs('0b1')
    a.reverse()
    assert a == Tibs('0b1')


def test_reverse_hex():
    # Reverse with hex representation
    a = Mutibs('0xAB')
    a.reverse()
    assert a == Tibs('0xd5')  # 0xAB = 10101011 -> 11010101 = 0xd5


def test_reverse_method_chaining():
    a = Mutibs('0b1100')
    result = a.reverse()
    assert result is None
    assert a == Tibs('0b0011')


def test_reverse_idempotence():
    # Reverse twice should give original
    a = Mutibs('0b10110')
    a.reverse()
    a.reverse()
    assert a == Tibs('0b10110')


# The reverse is done a word at a time over the raw storage, so the lengths
# that matter are the ones around word and byte boundaries, and the ones that
# leave a partial byte of padding for the shift to mop up.
REVERSE_LENGTHS = [0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 71, 127, 128, 129, 255, 257, 1000, 1001]


@pytest.mark.parametrize('length', REVERSE_LENGTHS)
def test_reverse_matches_bit_string(length):
    bits = ''.join('01101'[i % 5] for i in range(length))
    a = Mutibs('0b' + bits) if bits else Mutibs()
    a.reverse()
    assert a.bin == bits[::-1]
    assert len(a) == length


@pytest.mark.parametrize('length', REVERSE_LENGTHS)
def test_reverse_matches_reversed_copy(length):
    a = Mutibs(Tibs.from_random(length, seed=b'reverse'))
    original = Tibs(a)
    a.reverse()
    assert a == original.reversed()
    a.reverse()
    assert a == original


@pytest.mark.parametrize('offset', range(9))
@pytest.mark.parametrize('length', [0, 1, 5, 8, 13, 64, 100])
def test_reverse_with_storage_starting_mid_byte(offset, length):
    # Slicing gives storage that need not start on a byte boundary, which the
    # offset handling in the reverse has to take into account.
    source = ''.join('0110100011110000101'[i % 19] for i in range(offset + length))
    a = Mutibs('0b' + source)[offset:offset + length]
    assert a.bin == source[offset:]
    a.reverse()
    expected = source[offset:][::-1]
    assert a.bin == expected
    # Reading it back out through the byte-level paths must agree too.
    if expected:
        assert a.to_padded_bytes() == Mutibs('0b' + expected).to_padded_bytes()


def test_reverse_leaves_padding_bits_clear():
    # A non-whole-byte length pads the final byte; the reverse moves that
    # padding to the front, so it has to be cleared rather than carried along.
    a = Mutibs('0b1' + '0' * 11)
    a.reverse()
    assert a.bin == '0' * 11 + '1'
    assert a.to_padded_bytes() == b'\x00\x10'


def test_rol_basic():
    # Basic rotate left functionality
    a = Mutibs('0b1010')
    a.rotate_left(1)
    assert a == Tibs('0b0101')


def test_rol_full_rotation():
    # Rotating by the full length should return the original
    a = Mutibs('0b1010')
    a.rotate_left(4)
    assert a == Tibs('0b1010')


def test_rol_wraparound():
    # Rotating by more than length should wrap around
    a = Mutibs('0b1010')
    a.rotate_left(5)
    assert a == Tibs('0b0101')  # Same as rotate_left(1)


def test_rol_with_start_end():
    # Rotating with start and end parameters
    a = Mutibs('0b10101100')
    a.rotate_left(2, start=2, end=6)
    assert a == Tibs('0b10111000')


def test_rol_method_chaining():
    a = Mutibs('0b1010')
    result = a.rotate_left(1)
    assert a == Tibs('0b0101')
    assert result is None


def test_rotated_left_returns_new_mutibs():
    a = Mutibs('0b1010')
    b = a.rotated_left(1)
    assert a == Tibs('0b1010')
    assert b == Tibs('0b0101')
    assert isinstance(b, Mutibs)


def test_rol_negative_amount():
    # Error cases - negative rotation
    with pytest.raises(ValueError):
        a = Mutibs('0b1010')
        a.rotate_left(-1)  # Negative rotation amount


def test_rol_empty_bits():
    # Error cases - empty bits
    with pytest.raises(ValueError):
        a = Mutibs()
        a.rotate_left(1)  # Empty Mutibs


def test_rol_zero_rotation():
    # Zero rotation should not change anything
    a = Mutibs('0b1010')
    a.rotate_left(0)
    assert a == Tibs('0b1010')


def test_rol_large_rotation():
    # Large rotation value
    a = Mutibs('0b1010')
    a.rotate_left(1000000)  # Should be equivalent to rotate_left(0) since 1000000 % 4 = 0
    assert a == Tibs('0b1010')


def test_ror_basic():
    # Basic rotate right functionality
    a = Mutibs('0b1010')
    a.rotate_right(1)
    assert a == Tibs('0b0101')


def test_ror_full_rotation():
    # Rotating by the full length should return the original
    a = Mutibs('0b1010')
    a.rotate_right(4)
    assert a == Tibs('0b1010')


def test_ror_wraparound():
    # Rotating by more than length should wrap around
    a = Mutibs('0b1010')
    a.rotate_right(5)
    assert a == Tibs('0b0101')  # Same as rotate_right(1)


def test_ror_with_start_end():
    # Rotating with start and end parameters
    a = Mutibs('0b10101100')
    a.rotate_right(2, start=2, end=6)
    assert a == Tibs('0b10111000')


def test_ror_method_chaining():
    a = Mutibs('0b1010')
    result = a.rotate_right(1)
    assert a == Tibs('0b0101')
    assert result is None


def test_rotated_right_returns_new_mutibs():
    a = Mutibs('0b1010')
    b = a.rotated_right(1)
    assert a == Tibs('0b1010')
    assert b == Tibs('0b0101')
    assert isinstance(b, Mutibs)


def test_rol_ror_cancellation():
    # Rotating left then right should cancel out
    a = Mutibs('0b10110')
    a.rotate_left(2)
    a.rotate_right(2)
    assert a == Tibs('0b10110')


def test_ror_negative_amount():
    # Error cases - negative rotation
    with pytest.raises(ValueError):
        a = Mutibs('0b1010')
        a.rotate_right(-1)  # Negative rotation amount


def test_ror_empty_bits():
    # Error cases - empty bits
    with pytest.raises(ValueError):
        a = Mutibs()
        a.rotate_right(1)  # Empty Mutibs


def test_ror_zero_rotation():
    # Zero rotation should not change anything
    a = Mutibs('0b1010')
    a.rotate_right(0)
    assert a == Tibs('0b1010')


def test_ror_large_rotation():
    # Large rotation value
    a = Mutibs('0b1010')
    a.rotate_right(1000000)  # Should be equivalent to rotate_right(0) since 1000000 % 4 = 0
    assert a == Tibs('0b1010')


def test_byte_swap_basic():
    # Basic byte_swap functionality with default parameters
    a = Mutibs('0x1234')
    a.byte_swap()
    assert a == Tibs('0x3412')


def test_byte_swap_with_length():
    # Byte swap with specific byte_length parameter
    a = Mutibs('0x12345678')
    a.byte_swap(2)
    assert a == Tibs('0x34127856')


def test_byte_swap_with_slice():
    a = Mutibs('0x001122334455')
    a.byte_swap(start=8, end=40)
    assert a == Tibs('0x004433221155')


def test_byte_swap_with_length_and_slice():
    a = Mutibs('0x001122334455')
    a.byte_swap(2, start=8, end=40)
    assert a == Tibs('0x002211443355')


def test_byte_swap_with_unaligned_slice():
    a = Mutibs('0b000000001000000101')
    a.byte_swap(start=1, end=17)
    assert a == Tibs('0b000000010000000011')


def test_byte_swap_with_empty_slice():
    a = Mutibs('0x1234')
    a.byte_swap(start=8, end=8)
    assert a == Tibs('0x1234')


def test_byte_swap_single_byte():
    # Byte swap single byte (no change)
    a = Mutibs('0x12')
    a.byte_swap(1)
    assert a == Tibs('0x12')


def test_byte_swap_method_chaining():
    a = Mutibs('0x1234')
    result = a.byte_swap()
    assert a == Tibs('0x3412')
    assert result is None


def test_byte_swap_idempotence():
    # Byte swap twice should return to original
    a = Mutibs('0x12345678')
    a.byte_swap(2)
    a.byte_swap(2)
    assert a == Tibs('0x12345678')


def test_byte_swap_non_multiple_of_8():
    # Non-multiple of 8 bits
    with pytest.raises(ValueError):
        a = Mutibs('0b10101')
        a.byte_swap()


def test_byte_swap_empty():
    # Empty Mutibs
    a = Mutibs()
    a.byte_swap()
    assert a == Tibs()


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


def test_byte_swap_slice_not_multiple_of_byte_length():
    a = Mutibs('0x0011223344')
    with pytest.raises(ValueError):
        a.byte_swap(2, start=8, end=32)


def test_byte_swap_invalid_slice():
    a = Mutibs('0x0011223344')
    with pytest.raises(ValueError):
        a.byte_swap(start=32, end=8)


def test_to_tibs_basic():
    # Basic conversion
    a = Mutibs('0b1010')
    b = a.to_tibs()
    assert isinstance(b, Tibs)
    assert b == Tibs('0b1010')


def test_to_tibs_immutable_copy_operations():
    # Original shouldn't change when immutable copy is modified
    a = Mutibs('0b1010')
    b = a.to_tibs()
    c = ~b
    assert a == Tibs('0b1010')  # Original remains unchanged
    assert b == Tibs('0b1010')  # Original immutable copy unchanged
    assert c == Tibs('0b0101')  # New inverted copy


def test_to_tibs_original_modifications():
    # Changes to original shouldn't affect the immutable copy
    a = Mutibs('0b1010')
    b = a.to_tibs()
    a.invert()
    assert a == Tibs('0b0101')  # Original changed
    assert b == Tibs('0b1010')  # Immutable copy remains unchanged


def test_to_tibs_empty():
    # Empty Mutibs conversion
    a = Mutibs()
    b = a.to_tibs()
    assert isinstance(b, Tibs)
    assert b == Tibs()
    assert len(b) == 0


def test_mutable_bits_from_bits():
    # Test creating Mutibs from Tibs object
    b = Tibs('0b1010')
    a = b.to_mutibs()
    assert a == Tibs('0b1010')
    assert isinstance(a, Mutibs)

    # Modification should not affect original
    a.invert()
    assert a == Tibs('0b0101')
    assert b == Tibs('0b1010')


def test_setitem_with_bits_object():
    # Test setting slices using Tibs objects
    a = Mutibs('0b1010')
    b = Tibs('0b11')
    a[1:3] = b
    assert a == Tibs('0b1110')


def test_iadd_with_bits():
    # Test in-place add with Tibs objects
    a = Mutibs('0x12')
    b = Tibs('0x34')
    a += b
    assert a == Tibs('0x1234')


def test_iadd_multiple_types():
    # Test in-place add with various types
    a = Mutibs('0b1010')
    a += '0b11'  # String
    a += Tibs('0b00')  # Tibs object
    a += Mutibs('0b111')  # Another Mutibs
    assert a == Tibs('0b10101100111')


def test_imul_repeats():
    # Test in-place multiply
    a = Mutibs('0b101')
    a *= 3
    assert a == Tibs('0b101101101')

    # Test with zero
    b = Mutibs('0b111')
    b *= 0
    assert b == Tibs()


def test_delitem_sequence():
    # Test deleting multiple items in sequence
    a = Mutibs('0b10101010')
    del a[0]
    assert a == Tibs('0b0101010')
    del a[2]
    assert a == Tibs('0b011010')
    del a[-1]
    assert a == Tibs('0b01101')


def test_setitem_complex_cases():
    # Test setting a slice with different-length content
    a = Mutibs('0b1010')
    a[1:3] = '0b111'  # Replace 2 bits with 3 bits
    assert a == Tibs('0b11110')

    # Replace with empty content (effectively deleting)
    a[2:4] = ''
    assert a == Tibs('0b110')

    # Replace everything with shorter content
    a[:] = '0b1'
    assert a == Tibs('0b1')


def test_bit_operations_with_bits():
    # Testing bitwise AND with Tibs
    a = Mutibs('0b1100')
    b = Tibs('0b1010')
    a &= b
    assert a == Tibs('0b1000')

    # Testing bitwise OR with Tibs
    a = Mutibs('0b1100')
    b = Tibs('0b0011')
    a |= b
    assert a == Tibs('0b1111')

    # Testing bitwise XOR with Tibs
    a = Mutibs('0b1100')
    b = Tibs('0b1010')
    a ^= b
    assert a == Tibs('0b0110')


def test_equality_with_bits():
    # Test equality comparison with Tibs
    a = Mutibs('0b1010')
    b = Tibs('0b1010')
    assert a == b

    # Test after modification
    a[0] = 0
    assert a != b
    assert a == Tibs('0b0010')


def test_interleaved_operations():
    # Test a sequence of interleaved operations
    a = Mutibs('0b1010')
    a[1:3] = '0b00'
    a += '0b11'
    a.invert(0)
    del a[-1]
    assert a == Tibs('0b00001')

    # Chain multiple operations
    a = Mutibs('0b101')
    result = a.extend('0b010')
    a.invert()
    a.reverse()
    assert result is None
    assert a == Tibs('0b101010')  # 101 + 010 -> 101010 -> 010101 (invert) -> 010010 (reverse)


def test_mutable_bits_conversion_roundtrip():
    # Test round-trip conversion between Tibs and Mutibs
    orig = Tibs('0b10101100')
    mutable = orig.to_mutibs()
    mutable.invert(range(4))  # Modify some bits
    back_to_tibs = mutable.to_tibs()

    assert isinstance(back_to_tibs, Tibs)
    assert back_to_tibs == Tibs('0b01011100')
    assert orig == Tibs('0b10101100')  # Original should be unchanged


def test_inserting_bits_objects():
    # Test inserting Tibs objects at specific positions
    a = Mutibs('0b1010')
    b = Tibs('0b11')
    a.insert(2, b)
    assert a == Tibs('0b101110')

    # Insert at beginning
    c = Tibs('0b00')
    a.insert(0, c)
    assert a == Tibs('0b00101110')


def test_mixed_representation_operations():
    # Test operations with mixed representations (binary, hex)
    a = Mutibs('0b1010')
    a += '0x3A'
    assert a == Tibs('0b1010_0011_1010')

    a[4:8] = '0o7'
    assert a == Tibs('0b1010_111_1010')


def test_shifting_inplace():
    # Test in-place shifting operations
    a = Mutibs('0b001010')
    a <<= 2
    assert a == Tibs('0b101000')
    a >>= 3
    assert a == Tibs('0b000101')
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
    assert b == Tibs('0b111100')
    c = b >> 1
    assert c == Tibs('0b011110')


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
    assert isinstance(b, Mutibs) and b == Tibs('0x1')
    b = a + b'123'
    assert isinstance(b, Mutibs) and b == Tibs(b'123')
    b = a + [1, 0]
    assert isinstance(b, Mutibs) and b == Tibs('0b10')
    b = a + (1, 0, True)
    assert isinstance(b, Mutibs) and b == Tibs('0b101')
    with pytest.raises(TypeError):
        _ = a + (1, 0, 'steve')
    b = a + Mutibs.from_bools((1, 0, 'steve'))
    assert isinstance(b, Mutibs) and b == Tibs('0b101')


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
    assert a == Tibs('0xabff')
    a[0:0] = a
    assert a == Tibs('0xabffabff')


def test_del_ranges():
    a = Mutibs.from_zeros(10)
    del a[5:3]
    assert len(a) == 10


def test_set_item_with_step():
    a = Mutibs('0b000000')
    a[::2] = '0b110'
    assert a == Tibs('0b101000')


def test_iter():
    a = Mutibs('0b110')
    with pytest.raises(TypeError):
        _ = [bool(q) for q in a]


def test_partial_update():
    a = Mutibs.from_ones(10)
    try:
        a.unset([0, 1, 1000])
    except IndexError:
        pass
    assert a == Tibs('0b1111111111')


def test_append():
    a = Mutibs()
    a.append(True)
    a.append(False)
    a.append(True)
    a.append(False)
    a.append(0)
    a.append(1)
    assert a == Tibs('0b101001')
    with pytest.raises(TypeError):
        a.append(0.5)
    with pytest.raises(TypeError):
        a.append("1")
    with pytest.raises(TypeError):
        a.append(2)
    with pytest.raises(TypeError):
        a.append(-1)


def test_pop():
    m = Mutibs()
    with pytest.raises(IndexError):
        _ = m.pop()
    m.append(1)
    x = m.pop()
    assert x is True
    assert not m
    m.extend('0b10100')
    assert m.pop() is False
    assert m.pop() is False
    assert m.pop() is True
    assert m.pop() is False
    assert m.pop() is True
    with pytest.raises(IndexError):
        _ = m.pop()


def test_count_edge_cases():
    m = Mutibs.from_bin('0000_1111_0000')
    assert m.count(1) == 4
    assert m.count(True) == 4
    assert m.count(0) == 8
    assert m.count(1, 2, 10) == m[2:10].count(1)
    assert m.count([1, 1], 2, 10) == m[2:10].count([1, 1])
    assert m.count(0, -4) == m[-4:].count(0)
    assert m.count([1, 1, 1, 1]) == 1
    with pytest.raises(TypeError):
        m.count([1, 2])
    assert m.count(m) == 1


def test_count_byte_aligned():
    m = Mutibs('0xabababab')
    assert m.count('0xab', byte_aligned=True) == 4
    assert m.count('0xab', byte_aligned=True) == len(m.to_tibs().find_all('0xab', byte_aligned=True))
    m = Mutibs('0b1000_0001_1000_0000')
    assert m.count(1, byte_aligned=True) == 2
    assert m.count(m + [0]) == 0
    with pytest.raises(ValueError):
        _ = m.count(2)
    with pytest.raises(ValueError):
        _ = m.count(1, 8, 2)


def test_set_bug():
    m = Mutibs.from_hex('0x001122')
    m[8:0] = '0xff'
    assert m == Tibs('0x00ff1122')


def test_convenience_properties():
    m = Mutibs('0x123')
    assert m.to_hex() == m.hex
    assert m.to_oct() == m.oct
    assert m.to_bin() == m.bin
    assert m[:8].to_bytes() == m[:8].bytes


def test_representation_write_methods_replace_value_and_may_resize():
    m = Mutibs('0xff')

    assert m.write_bin('0b101') is None
    assert m == Mutibs('0b101')
    assert len(m) == 3

    assert m.write_oct('17') is None
    assert m == Mutibs('0b001111')
    assert len(m) == 6

    assert m.write_hex('123') is None
    assert m == Mutibs('0x123')
    assert len(m) == 12

    assert m.write_bytes(b'\xab\xcd') is None
    assert m == Mutibs('0xabcd')
    assert len(m) == 16


def test_representation_property_setters_replace_value_and_may_resize():
    m = Mutibs('0xff')

    m.bin = '10_1'
    assert m == Mutibs('0b101')
    assert len(m) == 3

    m.oct = '7'
    assert m == Mutibs('0b111')
    assert len(m) == 3

    m.hex = '0xabc'
    assert m == Mutibs('0xabc')
    assert len(m) == 12

    m.bytes = bytearray(b'AZ')
    assert m == Mutibs(b'AZ')
    assert len(m) == 16


def test_representation_write_errors_leave_value_unchanged():
    m = Mutibs('0xff')
    original = m.to_tibs()

    with pytest.raises(ValueError):
        m.write_hex('not hex')

    assert m == original

    with pytest.raises(ValueError):
        m.oct = '8'

    assert m == original


def test_byte_swapped():
    a = Mutibs.from_bytes(b'!olleh')
    b = a.byte_swapped()
    assert b == Tibs(b'hello!')


def test_mutibs_byte_swapped_with_slice():
    a = Mutibs('0x001122334455')
    b = a.byte_swapped(start=8, end=40)
    assert a == Tibs('0x001122334455')
    assert b == Tibs('0x004433221155')


def test_tibs_byte_swapped_with_slice():
    a = Tibs('0x001122334455')
    b = a.byte_swapped(start=8, end=40)
    assert a == Tibs('0x001122334455')
    assert b == Tibs('0x004433221155')


def test_from_u_bad_byte_order_type():
    with pytest.raises(TypeError):
        a = Mutibs.from_u(101, 16, "asdf")
    a = Mutibs.from_u(101, 16, ByteOrder.Unspecified)
    assert a.to_u() == 101


def assert_matching_exception(left, right):
    with pytest.raises(Exception) as left_error:
        left()
    with pytest.raises(Exception) as right_error:
        right()

    assert type(left_error.value) is type(right_error.value)
    assert str(left_error.value) == str(right_error.value)


def test_write_u_preserves_length_and_matches_from_u():
    m = Mutibs.from_ones(12)
    result = m.write_u(0x123)

    assert result is None
    assert len(m) == 12
    assert m == Mutibs.from_u(0x123, 12)
    assert m.u == 0x123


def test_write_i_preserves_length_and_matches_from_i():
    m = Mutibs.from_zeros(5)
    result = m.write_i(-3)

    assert result is None
    assert len(m) == 5
    assert m == Mutibs.from_i(-3, 5)
    assert m.i == -3


def test_write_f_preserves_length_and_matches_from_f():
    m = Mutibs.from_ones(32)
    result = m.write_f(3.5)

    assert result is None
    assert len(m) == 32
    assert m == Mutibs.from_f(3.5, 32)
    assert m.f == 3.5


def test_numeric_property_setters_preserve_length():
    m = Mutibs.from_zeros(8)
    m.u = 0x7f

    assert len(m) == 8
    assert m == Mutibs.from_u(0x7f, 8)

    m.i = -1
    assert len(m) == 8
    assert m == Mutibs.from_i(-1, 8)
    assert m.i == -1

    f = Mutibs.from_zeros(32)
    f.f = 1.5

    assert len(f) == 32
    assert f == Mutibs.from_f(1.5, 32)
    assert f.f == 1.5


def test_write_u_errors_match_from_u_and_leave_value_unchanged():
    m = Mutibs.from_zeros(4)
    original = m.to_tibs()

    assert_matching_exception(lambda: m.write_u(16), lambda: Mutibs.from_u(16, 4))
    assert m == original

    empty = Mutibs()
    assert_matching_exception(lambda: empty.write_u(0), lambda: Mutibs.from_u(0, 0))


def test_write_i_errors_match_from_i_and_leave_value_unchanged():
    m = Mutibs.from_zeros(4)
    original = m.to_tibs()

    assert_matching_exception(lambda: m.write_i(8), lambda: Mutibs.from_i(8, 4))
    assert m == original

    assert_matching_exception(lambda: m.write_i(-9), lambda: Mutibs.from_i(-9, 4))
    assert m == original

    empty = Mutibs()
    assert_matching_exception(lambda: empty.write_i(0), lambda: Mutibs.from_i(0, 0))


def test_write_f_errors_match_from_f_and_leave_value_unchanged():
    m = Mutibs.from_zeros(24)
    original = m.to_tibs()

    assert_matching_exception(lambda: m.write_f(1.25), lambda: Mutibs.from_f(1.25, 24))
    assert m == original


def test_numeric_write_methods_do_not_accept_endianness():
    m = Mutibs.from_zeros(16)

    with pytest.raises(TypeError):
        m.write_u(3, ByteOrder.Little)

    with pytest.raises(TypeError):
        m.write_i(-3, ByteOrder.Little)

    with pytest.raises(TypeError):
        m.write_f(1.25, ByteOrder.Little)


def test_contains():
    m = Mutibs('0x12345')
    assert '0x23' in m
    assert '0xff' not in m
    with pytest.raises(ValueError):
        'trevor' not in m


def test_special_method_creation_fails():
    m = Mutibs('0xff')
    with pytest.raises(ValueError):
        _ = 'macdonald' + m
    with pytest.raises(ValueError):
        _ = m & 'grebditch'
    with pytest.raises(ValueError):
        _ = m | 'grebditch'
    with pytest.raises(ValueError):
        _ = m ^ 'grebditch'
    with pytest.raises(ValueError):
        m ^= 'grebditch'
    with pytest.raises(ValueError):
        m |= 'grebditch'
    with pytest.raises(ValueError):
        m &= 'grebditch'


def test_replace_negative_count():
    m = Mutibs.from_random(1_000_000)
    t = m.to_tibs()
    m.replace('0b1', '0b0', count=0)
    assert m == t
    with pytest.raises(ValueError):
        m.replace('0b1', '0b0', count=-1)
    assert m == t

def test_float_endianness():
    m1 = Mutibs.from_f(3.5, 32)
    m2 = Mutibs.from_f(3.5, 32, ByteOrder.Unspecified)
    m3 = Mutibs.from_f(3.5, 32, ByteOrder.Big)
    m4 = Mutibs.from_f(3.5, 32, ByteOrder.Little)
    assert m1.to_f() == m2.to_f() == 3.5
    assert m4.le.f == 3.5
    assert m3.be.to_f() == 3.5
    assert m3.byte_swapped() == m4

def test_encode_decode():
    m = Mutibs.from_zeros(1000)
    m[56] = 1
    b1 = m.encode()
    t = Tibs.decode(b1)
    assert t == m
    m1 = Mutibs.decode(b1)
    assert m == m1

def test_empty_encode():
    m = Mutibs()
    a = m.encode(Codec.Auto)
    z = m.encode(Codec.Zstd)
    w = m.encode(Codec.Raw)
    r = m.encode(Codec.Rice)

    assert Mutibs.decode(a) == m
    assert Mutibs.decode(z) == m
    assert Mutibs.decode(r) == m
    assert Mutibs.decode(w) == m


def test_find_with_mask():
    m = Mutibs('0x1f2e3f')
    assert m.find('0x0f', mask='0x0f', byte_aligned=True) == 0
    assert m.rfind('0x0f', mask='0x0f', byte_aligned=True) == 16
    assert m.find_all('0x0f', mask='0x0f', byte_aligned=True) == [0, 16]
    assert m.count('0x0f', mask='0x0f') == 4
    assert m.find_all('0b11', mask='0b00') == list(range(len(m) - 1))
    with pytest.raises(ValueError):
        m.find('0x0f', mask='0b0')


def test_replace_with_mask():
    m = Mutibs('0x1f2e3f')
    assert m.replace('0x0f', '0x00', mask='0x0f', byte_aligned=True) == 2
    assert m == Mutibs('0x002e00')

    m = Mutibs('0x1f2e3f')
    assert m.replace('0x0f', '0x00', mask='0x0f', byte_aligned=True, count=1) == 1
    assert m == Mutibs('0x002e3f')

    m = Mutibs('0x1f2e3f')
    assert m.replaced('0x0f', '0x00', mask='0x0f', byte_aligned=True) == Mutibs('0x002e00')
    assert m == Mutibs('0x1f2e3f')
    with pytest.raises(ValueError):
        m.replace('0x0f', '0x00', mask='0b1')


def test_replace_with_mask_long_needle():
    # Over 64 bits the masked search uses a filter window plus verification.
    m = Mutibs.from_zeros(100) + Mutibs('0b1') + Mutibs.from_zeros(99)
    old = Mutibs.from_zeros(70)
    # Ignoring bit 30 of the needle lets it straddle the single set bit.
    mask = Mutibs.from_ones(70)
    mask[30] = 0
    assert m.find(old, mask=mask) == 0
    # Either the set bit falls outside the match, or it lands on the ignored bit.
    assert m.find_all(old, mask=mask) == list(range(31)) + [70] + list(range(101, 131))
    assert m.replace(old, '0b1', mask=mask, count=1) == 1
    assert m == Mutibs('0b1') + Mutibs.from_zeros(30) + Mutibs('0b1') + Mutibs.from_zeros(99)


def test_pairwise_operations():
    a, b = Mutibs('0b1100'), Mutibs('0b1010')
    assert a.count_and(b) == 1
    assert a.count_or(b) == 3
    assert a.count_xor(b) == 2
    assert a.count_andnot(b) == 1
    assert a.intersects(b) is True
    assert a.is_disjoint(b) is False
    assert a.is_subset_of(b) is False
    assert a.is_superset_of(b) is False
    assert Mutibs('0b1000').is_subset_of('0b1010') is True
    assert Mutibs('0b1010').is_superset_of('0b1000') is True
    assert Mutibs('0b1100').is_disjoint('0b0011') is True
    # The empty container, all zeros and all ones.
    empty, zeros, ones = Mutibs(''), Mutibs('0b0000'), Mutibs('0b1111')
    assert empty.is_disjoint(empty) is True
    assert empty.is_superset_of(empty) is True
    assert zeros.is_disjoint(ones) is True
    assert ones.is_disjoint(ones) is False
    assert ones.is_superset_of(zeros) is True
    assert zeros.is_superset_of(ones) is False
    # Mutibs and Tibs operands are interchangeable.
    assert a.count_and(Tibs('0b1010')) == 1
    assert Tibs('0b1100').count_and(a) == 2
    assert a.is_superset_of(Tibs('0b0100')) is True
    with pytest.raises(ValueError):
        a.count_and('0b101')
    for call in [lambda: a.is_disjoint('0b101'), lambda: a.is_superset_of('0b101')]:
        with pytest.raises(ValueError):
            call()


def test_pairwise_matches_tibs_when_unaligned():
    parent = Mutibs.from_ones(400)
    other = Mutibs.from_random(400, seed=b'm')
    for offset in range(8):
        for length in [1, 8, 9, 65, 130]:
            a, b = parent[offset:offset + length], other[offset:offset + length]
            ta, tb = Tibs(a), Tibs(b)
            assert a.count_and(b) == ta.count_and(tb) == (ta & tb).count(1)
            assert a.count_xor(b) == (ta ^ tb).count(1)
            assert a.count_andnot(b) == ta.count(1) - (ta & tb).count(1)
            assert a.intersects(b) == (ta & tb).any()
            assert a.is_disjoint(b) == (not (ta & tb).any())
            assert a.is_subset_of(b) == ((ta & tb) == ta)
            assert a.is_superset_of(b) == ((ta & tb) == tb)


def test_invert_empty_special_method():
    assert ~Mutibs() == Mutibs()
    assert ~Tibs() == Tibs()


def test_extract_deposit_mutibs():
    m = Mutibs('0b11010110')
    assert m.extract('0b10110000') == Mutibs('0b101')
    # deposit mutates in place and returns None
    ret = m.deposit('0b111', '0b10110000')
    assert ret is None
    assert m == Mutibs('0b11110110')
    # deposited returns a new object, leaving the original alone
    base = Mutibs('0b11010110')
    assert base.deposited('0b111', '0b10110000') == Mutibs('0b11110110')
    assert base == Mutibs('0b11010110')


def test_deposit_self_value():
    # Depositing a Mutibs into itself must read its pre-write bits.
    m = Mutibs('0b1010')
    m.deposit(m, Tibs.from_ones(4))
    assert m == Tibs('0b1010')


def test_extract_deposit_mutibs_errors():
    m = Mutibs('0b1011')
    with pytest.raises(ValueError):
        m.extract('0b101')
    with pytest.raises(ValueError):
        m.deposit('0b1', '0b101')
    with pytest.raises(ValueError):
        m.deposit('0b1', '0b1100')
