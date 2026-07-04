#!/usr/bin/env python
import pytest
from tibs import Tibs, Mutibs, ByteOrder, Codec
import random

def test_from_bin():
    a = Tibs.from_bin('010')
    b = Tibs.from_string('0b010')
    c = Mutibs.from_bin('0b010')
    d = Tibs('0b010')
    assert a == b == c == d


def test_to_bin():
    a = Tibs('0b1001')
    assert a.to_bin() == '1001'
    assert a.to_mutibs().to_bin() == '1001'


def test_to_base_conversions_with_bit_range():
    for cls in (Tibs, Mutibs):
        bits = cls('0x123456')
        assert bits.to_bin(4, 12) == '00100011'
        assert bits.to_bin(start=4, end=12) == '00100011'
        assert bits.to_bin(-20, -12) == '00100011'
        assert bits.to_bin(None, 4) == '0001'
        assert bits.to_bin(20, None) == '0110'
        assert bits.bin == '000100100011010001010110'

        assert bits.to_oct(4, 16) == '1064'
        assert bits.to_oct(start=4, end=16) == '1064'
        assert bits.oct == '04432126'

        assert bits.to_hex(4, 20) == '2345'
        assert bits.to_hex(start=4, end=20) == '2345'
        assert bits.hex == '123456'

        assert bits.to_bytes(4, 20) == b'\x23\x45'
        assert bits.to_bytes(start=4, end=20) == b'\x23\x45'
        assert bits.bytes == b'\x12\x34\x56'
        assert bits.to_padded_bytes() == b'\x12\x34\x56'
        assert bits.to_padded_bytes(0, 12) == b'\x12\x30'
        assert bits.to_padded_bytes(start=0, end=12) == b'\x12\x30'


def test_to_base_conversions_with_bit_range_errors():
    for cls in (Tibs, Mutibs):
        bits = cls('0x123456')
        with pytest.raises(ValueError, match="Invalid slice positions"):
            bits.to_bin(12, 4)
        with pytest.raises(ValueError, match="Invalid slice positions"):
            bits.to_bytes(-30, 8)
        with pytest.raises(ValueError, match="Invalid slice positions"):
            bits.to_padded_bytes(-30, 8)
        with pytest.raises(ValueError, match="not a multiple of 3"):
            bits.to_oct(0, 4)
        with pytest.raises(ValueError, match="not a multiple of 4"):
            bits.to_hex(0, 6)
        with pytest.raises(ValueError, match="not a multiple of 8"):
            bits.to_bytes(0, 12)
        assert bits.to_padded_bytes(0, 12) == b'\x12\x30'


@pytest.mark.parametrize("cls", (Tibs, Mutibs))
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("", b""),
        ("0b1", b"\x80"),
        ("0b101", b"\xa0"),
        ("0b1010101", b"\xaa"),
        ("0b10101010", b"\xaa"),
        ("0b101010101", b"\xaa\x80"),
    ],
)
def test_to_padded_bytes_zero_pads_rhs(cls, source, expected):
    bits = cls(source)
    assert bits.to_padded_bytes() == expected


def test_to_padded_bytes_handles_offset_slices():
    bits = Tibs.from_bytes(b"\xff")[:5]

    assert bits.to_padded_bytes() == b"\xf8"
    with pytest.raises(ValueError, match="not a multiple of 8"):
        bits.to_bytes()


def test_from_oct():
    a = Tibs.from_oct('12')
    b = Tibs.from_string('0o12')
    c = Mutibs.from_oct('0o12')
    d = Tibs('0o12')
    assert a == b == c == d


def test_to_oct():
    a = Tibs('0b001100')
    assert a.to_oct() == '14'
    assert a.to_mutibs().to_oct() == '14'


def test_from_hex():
    a = Tibs.from_hex('A')
    b = Tibs.from_string('0xA')
    c = Mutibs.from_hex('0xA')
    d = Tibs('0xA')
    assert a == b == c == d


def test_to_hex():
    a = Tibs('0b1010')
    assert a.to_hex() == 'a'
    assert a.to_mutibs().to_hex() == 'a'


def test_rfind():
    a = Mutibs()
    a += '0b1110001110'
    b = a.rfind('0b111')
    assert b == 6


def test_mutibs_find_byte_aligned_whole_bytes():
    a = Mutibs.from_bytes(b"\x11\x22\x33\x22")
    assert a.find("0x22", byte_aligned=True) == 8
    assert a.rfind("0x22", byte_aligned=True) == 24


def test_count_large():
    a = Tibs('0b' + '1' * 72)
    b = a[:65]
    assert b.count(1) == 65


def test_from_u():
    a = Tibs.from_u(15, 8)
    assert a == Tibs('0b00001111')
    b = Mutibs.from_u(15, 8)
    assert a == b
    c = a.to_u()
    assert c == 15


def test_to_numeric_with_bit_range():
    for cls in (Tibs, Mutibs):
        bits = cls('0b11100101000')
        assert bits.to_u(3, 8) == 5
        assert bits.to_u(start=3, end=8) == 5
        assert bits.to_u(-8, -3) == 5
        assert bits.to_u(None, 4) == 14
        assert bits.to_u(3, None) == int('00101000', 2)
        assert bits.u == int('11100101000', 2)

        signed_bits = cls('0b00111000')
        assert signed_bits.to_i(2, 6) == -2
        assert signed_bits.to_i(start=2, end=6) == -2
        assert signed_bits.i == 56

        float_bits = cls.from_joined([
            cls('0b101'),
            cls.from_f(0.25, 32),
            cls('0b00'),
        ])
        assert float_bits.to_f(3, 35) == 0.25
        assert float_bits.to_f(start=3, end=35) == 0.25


def test_to_numeric_with_bit_range_errors():
    for cls in (Tibs, Mutibs):
        bits = cls('0b11100101000')
        with pytest.raises(ValueError, match="Invalid slice positions"):
            bits.to_u(8, 3)
        with pytest.raises(ValueError, match="Invalid slice positions"):
            bits.to_i(-20, 3)
        with pytest.raises(ValueError, match="Cannot convert to unsigned int"):
            bits.to_u(3, 3)
        with pytest.raises(ValueError, match="Cannot convert to signed int"):
            bits.to_i(3, 3)
        with pytest.raises(ValueError, match="Unsupported float bit length"):
            bits.to_f(0, 8)


def test_from_u_errors():
    with pytest.raises(ValueError):
        _ = Tibs.from_u(0, -1)
    with pytest.raises(ValueError):
        _ = Tibs.from_u(0, 0)
    # Windows raises a ValueError instead of an OverflowError. :shrug:
    with pytest.raises((OverflowError, ValueError)):
        _ = Tibs.from_u(-1, 5)


def test_negative_length_with_byte_endianness_reports_negative_length():
    for cls in (Tibs, Mutibs):
        for method_name, value in (
            ("from_u", 0),
            ("from_i", 0),
            ("from_f", 0.0),
        ):
            method = getattr(cls, method_name)
            with pytest.raises(ValueError, match="Negative bit length"):
                method(value, -1, ByteOrder.Big)


def test_from_i():
    a = Tibs.from_i(-9, 100)
    b = Mutibs.from_i(-9, 100)
    assert a == b
    assert len(a) == 100
    assert a.to_i() == -9
    assert b.to_i() == -9


def test_from_i_errors():
    with pytest.raises(OverflowError):
        _ = Tibs.from_i(4, 2)


def test_signed_int_from_large_ints():
    with pytest.raises(ValueError):
        _ = Tibs.from_i(-1, 129)
    a = Tibs.from_i(-1, 128)
    assert a.all()
    assert a.to_i() == -1
    assert not Tibs.from_i(0, 128).any()
    assert Tibs.from_i(17, 128).to_i() == 17
    assert Tibs.from_i(-17, 128).to_i() == -17
    with pytest.raises(ValueError):
        _ = Mutibs.from_i(-1, 129)
    b = Mutibs.from_i(-1, 128)
    assert b.all()
    assert b.to_i() == -1


def test_unsigned_int_from_large_ints():
    with pytest.raises(ValueError):
        _ = Tibs.from_u(0, 129)
    a = Tibs.from_u(0, 128)
    assert not a.any()
    assert a.to_u() == 0
    assert Tibs.from_u((1 << 128) - 1, 128).all()
    assert Tibs.from_u(17, 128).to_u() == 17
    with pytest.raises(ValueError):
        _ = Mutibs.from_u(0, 129)
    b = Mutibs.from_u(0, 128)
    assert not b.any()
    assert b.to_u() == 0
    assert Mutibs.from_u((1 << 128) - 1, 128).all()


def test_from_f():
    a = Tibs.from_f(0.25, 16)
    b = Tibs.from_f(0.25, 32)
    c = Tibs.from_f(0.25, 64)
    a2 = Mutibs.from_f(0.25, 16)
    b2 = Mutibs.from_f(0.25, 32)
    c2 = Mutibs.from_f(0.25, 64)
    assert a == a2
    assert len(a) == 16
    assert len(b) == 32
    assert len(c) == 64
    assert b == b2
    assert c == c2
    f = a.to_f()
    g = b.to_f()
    h = c.to_f()
    f2 = a2.to_f()
    g2 = b2.to_f()
    h2 = c2.to_f()
    assert f == g == h == f2 == g2 == h2 == 0.25


def test_raw_bytes_and_offset():
    a = Tibs('0xff00ff')
    raw_bytes, offset, length = a.to_raw_data()
    assert raw_bytes == b'\xff\x00\xff'
    assert offset == 0
    b = a[4:20]
    raw_bytes, offset, length = b.to_raw_data()
    assert offset == 4
    assert raw_bytes == b'\xff\x00\xff'
    assert Tibs.from_bytes(raw_bytes) & '0x0ffff0' == Tibs('0x0f00f0')


def test_mutibs_raw_bytes_and_offset():
    a = Mutibs('0xff')
    b = a[4:]
    b += '0x77'
    assert b == Tibs('0xf77')
    raw_bytes, offset, length = b.to_raw_data()
    assert Tibs.from_bytes(raw_bytes) & '0x0fff' == Tibs('0x0f77')
    assert offset == 4
    assert b == Tibs('0xf77')
    raw_bytes, offset, length = b.as_raw_data()
    assert Tibs.from_bytes(raw_bytes) & '0x0fff' == Tibs('0x0f77')
    assert offset == 4
    assert length == 12
    assert b == Tibs()


def test_from_bytes_offsets():
    x = b'\xff\x00\xee\x11'
    a = Tibs.from_bytes(x)
    assert a == Tibs('0xff00ee11')
    b = Tibs.from_bytes(x, None, 16)
    assert b == Tibs('0xff00')
    c = Tibs.from_bytes(x, offset=16)
    assert c == Tibs('0xee11')
    d = Tibs.from_bytes(x, 4, 12)
    assert d == Tibs('0xf00')
    e = Mutibs.from_bytes(x, length=4, offset=28)
    assert e == Tibs('0x1')
    f = Mutibs.from_bytes(x, 0, 32)
    assert f == a
    g = Mutibs.from_bytes(x, 0, 0)
    assert g == Tibs()


def test_from_bytes_errors():
    x = b'\xff\x00\xee\x11'
    with pytest.raises(ValueError):
        _ = Tibs.from_bytes(x, length=33)
    with pytest.raises(ValueError):
        _ = Tibs.from_bytes(x, None, -1)
    with pytest.raises(ValueError):
        _ = Tibs.from_bytes(x, offset=-1)
    with pytest.raises(ValueError):
        _ = Tibs.from_bytes(x, length=-1)
    with pytest.raises(ValueError):
        _ = Tibs.from_bytes(x, offset=28, length=5)


def test_bit_ops_alignments():
    a = Tibs('0x00ff00')
    b = a[4:20]
    c = a[2:18]
    assert b & c == Tibs('0b0000001111110000')

    a = Mutibs('0x00ff00')
    b = a[4:20]
    c = a[2:18]
    assert b & c == Tibs('0b0000001111110000')


def test_raw_data_bug():
    a = Mutibs.from_bytes(b'hello')
    b = a[8:]
    assert a.to_raw_data() == (b'hello', 0, 40)
    assert b.to_raw_data() == (b'ello', 0, 32)

    a = Tibs.from_bytes(b'hello')
    b = a[8:]
    assert a.to_raw_data() == (b'hello', 0, 40)
    assert b.to_raw_data() == (b'ello', 0, 32)


def test_from_bools_generator():
    bits = [1, 0, 0, 1, 0]
    generator = (y for y in bits)
    t = Tibs.from_bools(generator)
    assert list(t) == bits


def test_count_expanded():
    a = Tibs('0xaaaa')
    b = a.count([1, 0, 1])
    assert b == 7
    b = a.count([1])
    assert b == 8
    b = a.count([1, 1])
    assert b == 0


def test_count_with_range():
    a = Tibs('0b0011010101100')
    assert a.count(1, 2, 10) == a[2:10].count(1)
    assert a.count([1, 0], 2, 10) == a[2:10].count([1, 0])
    assert a.count(0, -5) == a[-5:].count(0)
    with pytest.raises(ValueError):
        _ = a.count(1, 8, 2)


def test_tibs_set_at_returns_new_instance():
    a = Tibs('0b0000')
    b = a.set_at([0, -1])
    assert a == Tibs('0b0000')
    assert b == Tibs('0b1001')
    assert isinstance(b, Tibs)


def test_tibs_unset_at_returns_new_instance():
    a = Tibs('0b1111')
    b = a.unset_at(range(2))
    assert a == Tibs('0b1111')
    assert b == Tibs('0b0011')


def test_tibs_inverted_returns_new_instance():
    a = Tibs('0b1010')
    b = a.inverted([0, -1])
    assert a == Tibs('0b1010')
    assert b == Tibs('0b0011')


def test_tibs_inserted_returns_new_instance():
    a = Tibs('0b1010')
    b = a.inserted(2, '0b11')
    assert a == Tibs('0b1010')
    assert b == Tibs('0b101110')


def test_tibs_replaced_returns_new_instance():
    a = Tibs('0b10101010')
    b = a.replaced('0b10', '0b11', count=2)
    assert a == Tibs('0b10101010')
    assert b == Tibs('0b11111010')


def test_tibs_rotated_left_returns_new_instance():
    a = Tibs('0b1010')
    b = a.rotated_left(1)
    assert a == Tibs('0b1010')
    assert b == Tibs('0b0101')
    assert isinstance(b, Tibs)


def test_tibs_rotated_right_with_slice():
    a = Tibs('0b10101100')
    b = a.rotated_right(2, start=2, end=6)
    assert a == Tibs('0b10101100')
    assert b == Tibs('0b10111000')


def test_start_and_ends_with():
    a = Tibs.from_bytes(b'xyz')
    assert a.starts_with(b'x')
    assert a.ends_with(b'z')

    b = Mutibs.from_bytes(b'abcde')
    assert b.starts_with(b'a')
    assert b.ends_with(b'e')


def test_special_method_creation_fails():
    m = Tibs('0xff')
    with pytest.raises(ValueError):
        _ = m + 'percy'
    with pytest.raises(ValueError):
        _ = 'percy' + m
    with pytest.raises(ValueError):
        _ = m & 'percy'
    with pytest.raises(ValueError):
        _ = m | 'percy'
    with pytest.raises(ValueError):
        _ = m ^ 'percy'


def test_rfind_all():
    t = Mutibs.from_zeros(100)
    t.set([4, 8, 14, 99])
    a = t.to_tibs().rfind_all_iter([1])
    assert list(a) == [99, 14, 8, 4]
    a = t.to_tibs().rfind_all_iter([1, 0])
    assert list(a) == [14, 8, 4]


def test_endianness_i():
    t1 = Tibs.from_i(3, 16, ByteOrder.Big)
    assert t1.bin == '0000000000000011'
    t2 = Tibs.from_i(3, 16, ByteOrder.Little)
    assert t2.bin == '0000001100000000'
    assert t1.to_i() == 3
    assert t1.be.i == 3
    assert t2.le.i == 3
    assert t2.to_i() == 3 << 8

def test_endianness_u():
    t1 = Tibs.from_u(10001, 32)
    t2 = Tibs.from_u(10001, 32, ByteOrder.Big)
    t3 = Tibs.from_u(10001, 32, ByteOrder.Little)
    assert t1 == t2
    assert t1 != t3
    assert t2.to_u() == 10001
    assert t3.to_u() != 10001
    assert t3.le.to_u() == 10001
    with pytest.raises(ValueError):
        _ = Tibs.from_u(999, 31, ByteOrder.Big)
    with pytest.raises(ValueError):
        _ = Tibs('0x123').be.u


def test_rchunks():
    t = Tibs('0b111')
    for i in range(5):
        t += Tibs.from_u(i, 7)
    c = list(t.rchunks_iter(7))
    assert c[-1] == Tibs('0b111')
    for i in range(5):
        assert c[i].to_u() == 4 - i
        assert len(c[i]) == 7


def test_rchunks_remainder_and_count():
    t = Tibs('0b1010110010')

    # Reverse chunks are yielded from the end of the bitstring.
    all_chunks = list(t.rchunks_iter(4))
    assert [chunk.bin for chunk in all_chunks] == ['0010', '1011', '10']

    # count limits the number of yielded chunks, even in reverse mode.
    limited_chunks = list(t.rchunks_iter(4, count=2))
    assert [chunk.bin for chunk in limited_chunks] == ['0010', '1011']


def test_split_at_single_position():
    t = Tibs('0b101100')

    pieces = t.split_at(3)

    assert pieces == (Tibs('0b101'), Tibs('0b100'))
    assert isinstance(pieces, tuple)
    assert all(isinstance(piece, Tibs) for piece in pieces)


def test_split_at_multiple_positions():
    t = Tibs('0b101100')

    assert t.split_at([2, 5]) == (
        Tibs('0b10'),
        Tibs('0b110'),
        Tibs('0b0'),
    )
    assert t.split_at((0, 2, 2, len(t))) == (
        Tibs(),
        Tibs('0b10'),
        Tibs(),
        Tibs('0b1100'),
        Tibs(),
    )
    assert t.split_at([]) == (t,)
    assert Tibs().split_at(0) == (Tibs(), Tibs())
    assert Tibs().split_at([]) == (Tibs(),)


def test_split_at_negative_positions():
    t = Tibs('0b101100')

    assert t.split_at([2, -1]) == (
        Tibs('0b10'),
        Tibs('0b110'),
        Tibs('0b0'),
    )
    assert t.split_at(-len(t)) == (Tibs(), t)


def test_split_at_errors():
    t = Tibs('0b101100')

    with pytest.raises(ValueError, match="out of range"):
        _ = t.split_at(len(t) + 1)
    with pytest.raises(ValueError, match="out of range"):
        _ = t.split_at(-len(t) - 1)
    with pytest.raises(ValueError, match="nondecreasing"):
        _ = t.split_at([4, 3])


def encode_long_int(u: int) -> Tibs:
    if u <= 127:
        return [0] + Tibs.from_u(u, 7)
    # Work out how many bits long it is
    t = Tibs.from_u(u, 64)
    t = t[t.find([1]):]
    # For each non-final chunk of 7, we want a continuation bit and then the data
    chunks = list(t.rchunks_iter(7))[::-1]
    if len(chunks[0]) < 7:
        chunks[0] = [0]*(7 - len(chunks[0])) + chunks[0]
    m = Mutibs()
    for chunk in chunks[:-1]:
        m += [1] + chunk  # With continuation bit
    m += [0] + chunks[-1]
    assert len(m) % 8 == 0
    return m.as_tibs()


def decode_to_long_int(t: Tibs) -> int:
    assert len(t) > 0 and len(t) % 8 == 0
    if len(t) == 8:
        assert t[0] == 0
        return t[1:].to_u()
    m = Mutibs()
    for byte in t.chunks(8):
        m += byte[1:]
        if byte[0] == 0:
            break
    return m.to_u()


def test_encoding_ints():
    lengths = [0, 1, 2, 3, 14, 55, 101, 1022, 1023123, 12312451251, 86987698138715283]
    for u in lengths:
        t = encode_long_int(u)
        u2 = decode_to_long_int(t)
        assert u == u2


def encode_tibs(t: Tibs) -> bytes:
    n = len(t)

    # Single-byte form: bit0=1, then prefix-coded length/data for 0..6 bits.
    if n <= 6:
        if n == 6:
            e = Mutibs.from_joined([[1, 1], t])
        elif n == 5:
            e = Mutibs.from_joined([[1, 0, 1], t])
        elif n == 4:
            e = Mutibs.from_joined([[1, 0, 0, 1], t])
        elif n == 3:
            e = Mutibs.from_joined([[1, 0, 0, 0, 1], t])
        elif n == 2:
            e = Mutibs.from_joined([[1, 0, 0, 0, 0, 1], t])
        elif n == 1:
            e = Mutibs.from_joined([[1, 0, 0, 0, 0, 0, 1], t])
        else:
            e = Mutibs([1, 0, 0, 0, 0, 0, 0, 1])
        assert len(e) == 8
        return e.to_bytes()

    # Short form: bit0=0, bit1=1, bit2..bit4 = byte_length_minus_1, bit5..bit7 = bit_padding.
    if n <= 64:
        byte_length = (n + 7) // 8
        bit_padding = byte_length * 8 - n
        header = Mutibs.from_joined([
            [0, 1],
            Tibs.from_u(byte_length - 1, 3),
            Tibs.from_u(bit_padding, 3),
        ])
        e = Mutibs.from_joined([header, t])
        if bit_padding:
            e += [0] * bit_padding
        return e.to_bytes()

    # Long form: bit0=0, bit1=0, codec=000(raw), bit_padding(3).
    byte_length = (n + 7) // 8
    bit_padding = byte_length * 8 - n
    header = Mutibs.from_joined([[0, 0, 0, 0, 0], Tibs.from_u(bit_padding, 3)])
    var_length = encode_long_int(byte_length)
    e = Mutibs.from_joined([header, var_length, t])
    if bit_padding:
        e += [0] * bit_padding
    assert len(e) % 8 == 0
    return e.to_bytes()


def decode_tibs(b: bytes) -> Tibs:
    m = Mutibs.from_bytes(b)
    single_byte_flag, short_form_flag = m[0], m[1]

    if single_byte_flag:
        if m[1] == 1:
            m_out = m[2:8]
        elif m[2] == 1:
            m_out = m[3:8]
        elif m[3] == 1:
            m_out = m[4:8]
        elif m[4] == 1:
            m_out = m[5:8]
        elif m[5] == 1:
            m_out = m[6:8]
        elif m[6] == 1:
            m_out = m[7:8]
        else:
            m_out = Mutibs()
        return m_out.as_tibs()

    if short_form_flag:
        byte_length = m[2:5].to_u() + 1
        bit_padding = m[5:8].to_u()
        short_length = byte_length * 8 - bit_padding
        m_out = m[8:8 + short_length]
        return m_out.as_tibs()

    codec = m[2:5].to_u()
    assert codec == 0
    bit_padding = m[5:8].to_u()
    u = Mutibs()
    for byte in m[8:].to_tibs().chunks(8):
        u += byte
        if byte[0] == 0:
            break
    data_start = 8 + len(u)
    byte_length = decode_to_long_int(u.as_tibs())
    m_out = m[data_start: data_start + byte_length * 8]
    if bit_padding:
        m_out = m_out[:-bit_padding]
    return m_out.as_tibs()


def test_encoding():
    for _ in [None]:
        for length in range(400):
            # value = random.randint(0, (1 << length) - 1)
            t = Tibs.from_zeros(length)
            # b = encode_tibs(t)
            b2 = t.encode()
            # assert b == b2
            # t2 = decode_tibs(b)
            t3 = Tibs.decode(b2)
            # assert t == t2
            assert t == t3


def test_encoding_boundaries():
    assert Tibs.from_zeros(0).encode() == bytes.fromhex("81")
    assert Tibs.from_zeros(6).encode() == bytes.fromhex("c0")
    assert Tibs.from_zeros(7).encode() == bytes.fromhex("4100")
    assert Tibs.from_zeros(24).encode() == bytes.fromhex("50000000")
    assert Tibs.from_zeros(25).encode() == bytes.fromhex("0a0124a0")


def test_more_encoding():
    t = Tibs.from_ones(50) + [0] + Tibs.from_ones(50)
    b1 = t.encode()
    b2 = t.encode(Codec.Raw)
    rice = t.encode(Codec.Rice)
    assert b1 == rice
    assert len(b2) > len(b1)
    assert Tibs.decode(b1) == t
    assert Tibs.decode(b2) == t

    t2 = Tibs.from_random(8191, seed=b'x') & Tibs.from_random(8191, seed=b'y')
    b_auto = t2.encode(Codec.Auto)
    b_zstd = t2.encode(Codec.Zstd)
    b_raw = t2.encode(Codec.Raw)
    b_rice = t2.encode(Codec.Rice)

    temp = Tibs.decode(b_zstd)
    assert len(temp) == 8191

    assert b_auto == b_zstd
    assert Tibs.decode(b_zstd) == Tibs.decode(b_raw) == Tibs.decode(b_rice)


def test_raw_encoding_is_stable_key_for_tibs_and_mutibs():
    values = [
        Tibs("0b101"),
        Tibs("0b101"),
        Tibs("0x05"),
        Mutibs("0b101"),
    ]

    keys = {value.encode(Codec.Raw) for value in values}

    assert len(keys) == 2
    assert Tibs("0b101").encode(Codec.Raw) in keys
    assert Mutibs("0b101").encode(Codec.Raw) in keys
    decoded_keys = {Tibs.decode(key).encode(Codec.Raw) for key in keys}
    assert decoded_keys == keys
