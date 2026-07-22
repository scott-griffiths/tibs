#!/usr/bin/env python
import pytest
from hypothesis import given, strategies as st
from tibs import Tibs, Mutibs


# A 32 bit value used throughout. Its interpretations are:
#   hex 'ac804f4b', bin '10101100100000000100111101001011',
#   u 2894090059, i -1400877237
VALUE = '0xac804f4b'

# A 12 bit value, so that octal (multiple of 3) is available.
OCT_VALUE = '0o7531'


@pytest.fixture(params=[Tibs, Mutibs])
def cls(request):
    return request.param


def test_empty_spec_matches_str(cls):
    for s in (VALUE, OCT_VALUE, '0b101', ''):
        t = cls(s)
        assert format(t, '') == str(t)


def test_empty_spec_truncates_like_str(cls):
    # str() gives up above MAX_BITS_TO_PRINT and appends the length.
    t = cls.from_zeros(10008)
    assert format(t, '') == str(t)
    assert format(t, '').endswith('# length=10008')


def test_type_codes_never_truncate(cls):
    t = cls.from_zeros(10008)
    assert format(t, 'b') == '0' * 10008
    assert format(t, 'x') == '0' * 2502


# Representation codes: b / o / x / X


def test_hex(cls):
    t = cls(VALUE)
    assert format(t, 'x') == 'ac804f4b'
    assert format(t, 'x') == t.hex
    assert format(t, '#x') == '0xac804f4b'


def test_upper_hex(cls):
    t = cls(VALUE)
    assert format(t, 'X') == 'AC804F4B'
    assert format(t, '#X') == '0XAC804F4B'


def test_bin(cls):
    t = cls(VALUE)
    assert format(t, 'b') == '10101100100000000100111101001011'
    assert format(t, 'b') == t.bin
    assert format(t, '#b') == '0b10101100100000000100111101001011'


def test_oct(cls):
    t = cls(OCT_VALUE)
    assert format(t, 'o') == '7531'
    assert format(t, 'o') == t.oct
    assert format(t, '#o') == '0o7531'


def test_leading_zeros_are_preserved(cls):
    # The whole point of the representation codes - these are not int formats.
    assert format(cls('0x0f'), 'b') == '00001111'
    assert format(cls('0x0f'), 'x') == '0f'
    assert format(cls('0x000000'), 'x') == '000000'


def test_representation_is_not_the_numeric_value(cls):
    t = cls('0x0f')
    assert format(t, 'b') != format(t.u, 'b')
    assert format(t, 'x') != format(t.u, 'x')


# Grouping with _


def test_default_group_size_is_four(cls):
    t = cls(VALUE)
    assert format(t, '_b') == '1010_1100_1000_0000_0100_1111_0100_1011'
    assert format(t, '_x') == 'ac80_4f4b'
    assert format(t, '_X') == 'AC80_4F4B'
    assert format(cls(OCT_VALUE), '_o') == '7531'


def test_group_size_from_precision(cls):
    t = cls(VALUE)
    assert format(t, '_.8b') == '10101100_10000000_01001111_01001011'
    assert format(t, '_.2x') == 'ac_80_4f_4b'
    assert format(t, '_.1b') == '1_0_1_0_1_1_0_0_1_0_0_0_0_0_0_0_0_1_0_0_1_1_1_1_0_1_0_0_1_0_1_1'
    assert format(cls(OCT_VALUE), '_.2o') == '75_31'


def test_grouping_runs_left_to_right(cls):
    # Divergence from Python: a Tibs is a sequence starting at bit 0, so the
    # ragged group lands at the end rather than the start.
    t = cls('0b101010101')
    assert format(t, '_b') == '1010_1010_1'
    assert format(t, '_.3b') == '101_010_101'
    assert format(t, '_.8b') == '10101010_1'
    assert format(0b101010101, '_b') == '1_0101_0101'


def test_group_size_larger_than_value(cls):
    t = cls('0b1011')
    assert format(t, '_.8b') == '1011'


def test_no_trailing_separator(cls):
    t = cls('0xff')
    assert format(t, '_.4b') == '1111_1111'
    assert format(t, '_.2x') == 'ff'


def test_grouping_with_prefix(cls):
    t = cls(VALUE)
    assert format(t, '#_x') == '0xac80_4f4b'
    assert format(t, '#_.8b') == '0b10101100_10000000_01001111_01001011'


# Width, fill and alignment


def test_width_defaults_to_right_align_for_type_codes(cls):
    t = cls(VALUE)
    assert format(t, '20x') == '            ac804f4b'
    assert format(t, '>20x') == '            ac804f4b'


def test_width_defaults_to_left_align_with_no_type_code(cls):
    # With no type code the body is str(t), which is string-like.
    t = cls(VALUE)
    assert format(t, '20') == '0xac804f4b          '
    assert format(t, '<20') == '0xac804f4b          '
    assert format(t, '>20') == '          0xac804f4b'


def test_alignment_and_fill(cls):
    t = cls(VALUE)
    assert format(t, '<20x') == 'ac804f4b            '
    assert format(t, '^12x') == '  ac804f4b  '
    assert format(t, '*>12x') == '****ac804f4b'
    assert format(t, '.^12x') == '..ac804f4b..'


def test_width_smaller_than_body_is_ignored(cls):
    t = cls(VALUE)
    assert format(t, '4x') == 'ac804f4b'


def test_padding_that_could_be_mistaken_for_data_is_rejected(cls):
    # Zero padding an int is harmless because leading zeros mean nothing there. Here
    # the length is part of the value, so it would silently change what the string
    # says. Rejected however it is spelled, and whatever the width.
    t = cls('0xf')
    for spec in ['#06x', '06x', '#010x', '0>10x', '0=#10x', '0x', '016_b', '08o']:
        with pytest.raises(ValueError):
            format(t, spec)


def test_any_digit_fill_is_rejected_not_just_zero(cls):
    for spec, value in [('f>10x', '0xff'), ('F>10X', '0xff'), ('a>10x', '0xff'),
                        ('1>10b', '0b1'), ('7>8o', '0o7531')]:
        with pytest.raises(ValueError):
            format(cls(value), spec)


def test_padding_is_rejected_regardless_of_width(cls):
    # The spec is invalid even when the width is too small for any padding to be
    # inserted, so a spec is never valid for one value and invalid for another.
    with pytest.raises(ValueError):
        format(cls('0xac804f4b'), '#010x')
    with pytest.raises(ValueError):
        format(cls('0xac804f4b'), '#02x')


def test_fill_that_cannot_be_data_is_allowed(cls):
    t = cls('0xf')
    assert format(t, '=#6x') == '0x   f'
    assert format(t, '>#6x') == '   0xf'
    assert format(t, '*>6x') == '*****f'
    assert format(t, '^5x') == '  f  '
    # A digit that is not valid for this base cannot be mistaken for the data.
    assert format(cls('0b1111'), '8>6b') == '881111'
    assert format(cls('0o7531'), '9>6o') == '997531'


def test_fill_align_lookahead(cls):
    # '#=6x' is fill '#' with '=' alignment, not the alternate form, because the
    # fill/align pair is consumed first. Same as int formatting.
    t = cls('0xf')
    assert format(t, '#=6x') == '#####f'
    assert format(t, '#=6x') == format(0xf, '#=6x')


def test_padding_is_applied_after_grouping(cls):
    t = cls('0xff')
    assert format(t, '>12_b') == '   1111_1111'


def test_padding_is_never_grouped(cls):
    # Divergence from Python, which groups the padding too. Here the separators stay
    # lined up with bit positions in the actual data, and the padding is just padding.
    t = cls('0xff')
    assert format(t, '*>14_b') == '*****1111_1111'
    assert format(255, '014_b') == '0000_1111_1111'


# Numeric codes: u / i


def test_unsigned(cls):
    t = cls(VALUE)
    assert format(t, 'u') == '2894090059'
    assert format(t, 'u') == str(t.u)


def test_signed(cls):
    t = cls(VALUE)
    assert format(t, 'i') == '-1400877237'
    assert format(t, 'i') == str(t.i)


def test_numeric_codes_follow_int_formatting(cls):
    t = cls(VALUE)
    for spec, int_spec in [('_u', '_d'), (',u', ',d'), ('+u', '+d'),
                           ('>15u', '>15d'), ('015u', '015d'), ('^14u', '^14d'),
                           ('_i', '_d'), (',i', ',d'), ('+i', '+d'), ('015i', '015d')]:
        expected = format(t.u if spec.endswith('u') else t.i, int_spec)
        assert format(t, spec) == expected, spec


def test_numeric_grouping_runs_right_to_left(cls):
    # Unlike the representation codes - these really are numbers.
    t = cls(VALUE)
    assert format(t, '_u') == '2_894_090_059'
    assert format(t, ',u') == '2,894,090,059'
    assert format(t, ',i') == '-1,400,877,237'


def test_numeric_codes_ignore_bit_length(cls):
    assert format(cls('0x0f'), 'u') == '15'
    assert format(cls('0b101'), 'u') == '5'
    assert format(cls('0b101'), 'i') == '-3'


# Empty containers


def test_empty_container(cls):
    t = cls()
    assert format(t, '') == ''
    assert format(t, 'b') == ''
    assert format(t, 'x') == ''
    assert format(t, 'o') == ''
    assert format(t, '_b') == ''
    assert format(t, '_.2x') == ''


def test_empty_container_keeps_the_prefix(cls):
    t = cls()
    assert format(t, '#x') == '0x'
    assert format(t, '#b') == '0b'
    assert format(t, '#_b') == '0b'
    # ...and the prefix alone still parses back to an empty container.
    assert Tibs(format(t, '#x')) == Tibs()


def test_numeric_codes_have_no_length_limit(cls):
    # Both families of code work at any length. 'u' and 'i' were capped at 128
    # bits while the .u and .i properties were.
    assert len(format(cls.from_ones(128), 'b')) == 128
    assert format(cls.from_ones(128), 'u') == str(2 ** 128 - 1)
    assert format(cls.from_ones(129), 'u') == str(2 ** 129 - 1)
    assert format(cls.from_ones(129), 'i') == '-1'
    assert format(cls.from_ones(1000), 'u') == str(2 ** 1000 - 1)


def test_numeric_codes_still_need_some_bits(cls):
    with pytest.raises(ValueError):
        format(cls(), 'u')
    with pytest.raises(ValueError):
        format(cls(), 'i')


def test_multibyte_fill_is_counted_in_code_points(cls):
    assert format(cls('0xff'), '€>6x') == '€€€€ff'
    assert format(cls('0xff'), '€^7x') == '€€ff€€€'


def test_absurd_width_raises_instead_of_aborting(cls):
    # A width too large to allocate must raise a catchable MemoryError rather
    # than aborting the process or raising a Rust PanicException.
    t = cls('0xff')
    for spec in ['1' + '0' * 18 + 'x', '9' * 19 + 'x', '€>' + '9' * 19 + 'x']:
        with pytest.raises(MemoryError):
            format(t, spec)


def test_width_too_large_to_parse(cls):
    with pytest.raises(ValueError):
        format(cls('0xff'), '9' * 20 + 'x')
    with pytest.raises(ValueError):
        format(cls('0xff'), '_.' + '9' * 20 + 'b')


def test_empty_container_has_no_numeric_value(cls):
    with pytest.raises(ValueError):
        format(cls(), 'u')
    with pytest.raises(ValueError):
        format(cls(), 'i')


# Errors


def test_wrong_length_for_type_code(cls):
    with pytest.raises(ValueError):
        format(cls(VALUE), 'o')      # 32 bits is not a multiple of 3
    with pytest.raises(ValueError):
        format(cls('0b10101'), 'x')  # 5 bits is not a multiple of 4
    with pytest.raises(ValueError):
        format(cls('0b10101'), 'X')


def test_comma_grouping_rejected_for_representation_codes(cls):
    # A bit sequence is not a number, so thousands separators make no sense.
    for spec in [',b', ',x', ',X', ',o', '#,x']:
        with pytest.raises(ValueError):
            format(cls(VALUE), spec)


def test_sign_rejected_for_representation_codes(cls):
    for spec in ['+b', '-b', ' b', '+x', '+#x']:
        with pytest.raises(ValueError):
            format(cls(VALUE), spec)


def test_group_size_requires_a_grouping_character(cls):
    for spec in ['.4b', '.2x', '#.2x']:
        with pytest.raises(ValueError):
            format(cls(VALUE), spec)


def test_group_size_must_be_positive(cls):
    with pytest.raises(ValueError):
        format(cls(VALUE), '_.0b')


def test_group_size_rejected_for_numeric_codes(cls):
    for spec in ['_.2u', '.3i', '_.4i']:
        with pytest.raises(ValueError):
            format(cls(VALUE), spec)


def test_unknown_type_codes(cls):
    for spec in ['d', 'f', 's', 'c', 'n', 'e', 'g', 'q', '%', 'B', 'O', 'U', 'I']:
        with pytest.raises(ValueError):
            format(cls(VALUE), spec)


def test_modifiers_rejected_with_no_type_code(cls):
    for spec in ['#', '_', ',', '+', ' ', '.4', '_.4', '#20']:
        with pytest.raises(ValueError):
            format(cls(VALUE), spec)


def test_malformed_specs(cls):
    for spec in ['>>>x', 'xx', 'x4', '_x_', '4.2', '#-x', '_,b', '..2_b']:
        with pytest.raises(ValueError):
            format(cls(VALUE), spec)


# Round-tripping


ROUND_TRIP_SPECS = ['#x', '#X', '#b', '#_x', '#_b', '#_.2x', '#_.8b', '#_.1b']


def test_round_trip(cls):
    t = cls(VALUE)
    for spec in ROUND_TRIP_SPECS:
        assert Tibs(format(t, spec)) == Tibs(VALUE), spec
    oct_t = cls(OCT_VALUE)
    for spec in ['#o', '#_o', '#_.2o']:
        assert Tibs(format(oct_t, spec)) == Tibs(OCT_VALUE), spec


@given(st.binary(max_size=64), st.sampled_from(ROUND_TRIP_SPECS))
def test_round_trip_hypothesis(data, spec):
    t = Tibs.from_bytes(data)
    assert Tibs(format(t, spec)) == t


@given(st.lists(st.booleans(), max_size=200))
def test_bin_round_trip_hypothesis(bools):
    t = Tibs.from_bools(bools)
    assert Tibs(format(t, '#_.5b')) == t


# Views


def test_view_uses_the_view_byte_order():
    v = Tibs('0x0100').le
    assert format(v, 'x') == '0001'
    assert format(v, '#x') == '0x0001'
    assert format(v, 'u') == '1'


def test_mutable_view_matches_view():
    v = Tibs('0x0100').le
    mv = Mutibs('0x0100').le
    for spec in ['x', '#x', 'b', '_b', 'u', 'i', '>12x']:
        assert format(mv, spec) == format(v, spec), spec


def test_view_field_formatting():
    # The eBPF example from the docs: LSB0 field labels on a little-endian word.
    instruction = Tibs.from_bytes(bytes.fromhex('07 01 00 00 44 33 22 11')).lsb0.le
    assert format(instruction.field(63, 32), '#x') == '0x11223344'
    assert format(instruction.field(11, 8), 'u') == '1'


def test_view_empty_spec_is_unchanged():
    # Views have no __str__, so format(v, '') keeps giving the repr.
    v = Tibs('0x0100').le
    assert format(v, '') == repr(v)
    mv = Mutibs('0x0100').le
    assert format(mv, '') == repr(mv)


def test_view_errors():
    v = Tibs('0x0100').le
    with pytest.raises(ValueError):
        format(v, 'o')
    with pytest.raises(ValueError):
        format(v, 'd')


# f-string integration


def test_f_strings(cls):
    t = cls(VALUE)
    assert f'{t:#x}' == '0xac804f4b'
    assert f'{t:_.8b}' == '10101100_10000000_01001111_01001011'
    assert f'{t:u}' == '2894090059'
    assert f'{t}' == str(t)


def test_format_method(cls):
    t = cls(VALUE)
    assert '{:#x}'.format(t) == '0xac804f4b'
    assert '{0:u} / {0:i}'.format(t) == '2894090059 / -1400877237'
