#!/usr/bin/env python
import pytest
import io
import re
from hypothesis import given
import hypothesis.strategies as st
import tibs
from tibs import Tibs, Endianness, Mutibs
from typing import Iterable, Sequence

@pytest.mark.skip
def test_build():
    a = Tibs.from_dtype("u12", 104)
    assert a == "u12 = 104"
    b = Tibs.from_dtype("bool", False)
    assert len(b) == 1
    assert b[0] == 0
    c = Tibs.from_dtype(Dtype("f64"), 13.75)
    assert len(c) == 64
    assert c.unpack(["f64"]) == (13.75,)


def remove_unprintable(s: str) -> str:
    colour_escape = re.compile(r"(?:\x1B[@-_])[0-?]*[ -/]*[@-~]")
    return colour_escape.sub("", s)

@pytest.mark.skip
class TestCreation:
    def test_creation_from_bytes(self):
        s = Tibs.from_bytes(b"\xa0\xff")
        assert (len(s), s.unpack("hex")) == (16, "a0ff")

    @given(st.binary())
    def test_creation_from_bytes_roundtrip(self, data):
        s = Tibs.from_dtype("bytes", data)
        assert s.bytes == data

    def test_creation_from_hex(self):
        s = Tibs.from_dtype("hex", "0xA0ff")
        assert (len(s), s.unpack("hex")) == (16, "a0ff")

    def test_creation_from_hex_with_whitespace(self):
        s = Tibs("  \n0x a  4e       \r3  \n")
        assert s.hex == "a4e3"

    @pytest.mark.parametrize("bad_val", ["0xx0", "0xX0", "0Xx0", "-2e"])
    def test_creation_from_hex_errors(self, bad_val: str):
        with pytest.raises(ValueError):
            Tibs.from_dtype("hex", bad_val)

    def test_creation_from_bin(self):
        s = Tibs.from_dtype("bin", "1010000011111111")
        assert (len(s), s.hex) == (16, "a0ff")
        s = Tibs.from_string("0b00")[:1]
        assert s.unpack("bin") == "0"
        s = Tibs.from_dtype("bin", " 0000 \n 0001\r ")
        assert s.bin == "00000001"

    def test_creation_from_uint_errors(self):
        # test = Tibs.pack('u10', -1)

        with pytest.raises(ValueError):
            Tibs.from_dtype("u10", -1)
        with pytest.raises(ValueError):
            Tibs.from_dtype("uint", 12)
        with pytest.raises(ValueError):
            Tibs.from_dtype("uint2", 4)
        with pytest.raises(ValueError):
            Tibs.from_dtype("u0", 1)
        with pytest.raises(ValueError):
            Tibs.from_dtype("u2", 12)

    def test_creation_from_int(self):
        s = Tibs.from_dtype("i4", 0)
        d = Dtype.from_string("bin4")
        _ = s.unpack([d])
        # assert s.unpack([d])[0] == "0000"
        # s = Tibs.from_dtype(Dtype.from_string("i2"), 1)
        # assert s.bin == "01"
        # s = Tibs.from_dtype("i11", -1)
        # assert s.bin == "11111111111"
        # s = Tibs.from_string("i12=7")
        # assert s.bin == "000000000111"
        # assert s.i == 7
        # s = Tibs.from_dtype(Dtype.from_string("i108"), -243)
        # assert (s.unpack(Dtype("i")), len(s)) == (-243, 108)
        # for length in range(6, 10):
        #     for value in range(-17, 17):
        #         s = Tibs.from_dtype(DtypeSingle.from_params(DtypeKind.INT, length), value)
        #         assert (s.i, len(s)) == (value, length)

    @pytest.mark.parametrize("int_, length", [[-1, 0], [12, 0], [4, 3], [-5, 3]])
    def test_creation_from_int_errors(self, int_, length):
        with pytest.raises(ValueError):
            _ = Tibs.from_dtype(DtypeSingle.from_params(DtypeKind.INT, length), int_)

    def test_creation_from_bool(self):
        a = Tibs.from_dtype("bool", False)
        assert a == "0b0"
        b = Tibs.from_string("bool=0")
        assert b == Tibs.from_bools([False])

    def test_creation_from_bool_errors(self):
        with pytest.raises(ValueError):
            _ = Tibs.from_dtype("bool2", 0)

    def test_creation_keyword_error(self):
        with pytest.raises(ValueError):
            Tibs.from_dtype("squirrel", 5)

    def test_creation_from_memoryview(self):
        x = bytes(bytearray(range(20)))
        m = memoryview(x[10:15])
        b = Tibs.from_dtype("bytes", m)
        assert b.unpack("[u8; 5]") == (10, 11, 12, 13, 14)


class TestInitialisation:
    def test_empty_init(self):
        a = Tibs()
        assert a == ""

    def test_find(self):
        a = Tibs.from_string("0xabcd")
        r = a.find("0xbc")
        assert r == 4
        r = a.find("0x23462346246", byte_aligned=True)
        assert r is None

    def test_rfind(self):
        a = Tibs.from_string("0b11101010010010")
        b = a.rfind("0b010")
        assert b == 11

    def test_find_all(self):
        a = Tibs("0b0010011")
        b = list(a.find_all('0b1'))
        assert b == [2, 5, 6]
        t = Tibs("0b10")
        tp = list(t.find_all("0b1"))
        assert tp == [0]


class TestCut:
    def test_cut(self):
        s = Tibs().from_joined(["0b000111"] * 10)
        for t in s.chunks(6):
            assert t == "0b000111"


def test_unorderable():
    a = Tibs("0b000111")
    b = Tibs("0b000111")
    with pytest.raises(TypeError):
        _ = a < b
    with pytest.raises(TypeError):
        _ = a > b
    with pytest.raises(TypeError):
        _ = a <= b
    with pytest.raises(TypeError):
        _ = a >= b


class TestPadToken:
    def test_creation(self):
        with pytest.raises(ValueError):
            _ = Tibs.from_string("pad10")
        with pytest.raises(ValueError):
            _ = Tibs.from_string("pad")


    @pytest.mark.skip
    def test_unpack(self):
        s = Tibs.from_string("0b111000111")
        x, y = s.unpack(["bits3", "pad3", "bits3"])
        assert (x, y.unpack("u")) == ("0b111", 7)
        x, y = s.unpack(["bits2", "pad2", "bin5"])
        assert (x.unpack(["u2"])[0], y) == (3, "00111")
        x = s.unpack(["pad1", "pad2", "pad3"])
        assert x == ()


def test_adding():
    a = Tibs.from_string("0b0")
    b = Tibs.from_string("0b11")
    c = a + b
    assert c == "0b011"
    assert a == "0b0"
    assert b == "0b11"


class TestContainsBug:
    def test_contains(self):
        a = Tibs.from_string("0b1, 0x0001dead0001")
        assert "0xdead" in a
        assert "0xfeed" not in a

        assert "0b1" in Tibs.from_string("0xf")
        assert "0b0" not in Tibs.from_string("0xf")


@pytest.mark.skip
class TestUnderscoresInLiterals:
    def test_hex_creation(self):
        a = Tibs.from_dtype("hex", "ab_cd__ef")
        assert a.hex == "abcdef"
        b = Tibs.from_string("0x0102_0304")
        assert b.u == 0x0102_0304

    def test_binary_creation(self):
        a = Tibs.from_dtype("bin", "0000_0001_0010")
        assert a.bin == "000000010010"
        b = Tibs.from_string("0b0011_1100_1111_0000")
        assert b.bin == "0011110011110000"
        v = 0b1010_0000
        c = Tibs.from_dtype("u8", 0b1010_0000)
        assert c.u == v

    def test_octal_creation(self):
        a = Tibs.from_dtype("oct", "0011_2233_4455_6677")
        assert a.u == 0o001122334455_6677
        b = Tibs.from_string("0o123_321_123_321")
        assert b.u == 0o123_321_123321

@pytest.mark.skip
def test_unpack_array():
    a = Tibs.from_string("0b1010101010101010")
    assert a.unpack(("u8", "u4", "u4")) == (170, 10, 10)
    assert a.unpack(["u4", "u4", "u8"]) == (10, 10, 170)
    assert a.unpack(["u4", "u4", "u4", "u4"]) == (10, 10, 10, 10)

    assert a.unpack(["[u4; 4]"]) == ((10, 10, 10, 10),)
    assert a.unpack("[u4; 3]") == (10, 10, 10)


@pytest.mark.skip
def test_unpack_errors():
    with pytest.raises(ValueError):
        _ = Tibs().unpack("bool")
    with pytest.raises(ValueError):
        _ = Tibs("0b1").unpack("u2")
    a = Tibs.from_string("0b101010101")
    with pytest.raises(ValueError):
        _ = a.unpack(["u4", "u4", "u4"])
    with pytest.raises(ValueError):
        _ = a.unpack("i10")


@pytest.mark.skip
def test_unpack_single():
    a = Tibs("0x12345")
    assert a.unpack("[u4; 5]") == (1, 2, 3, 4, 5)
    assert a.unpack("u8") == 0x12


@pytest.mark.skip
def test_pack_array():
    d = DtypeArray.from_params(DtypeKind.UINT, 33, 5)
    a = Tibs.from_dtype(d, [10, 100, 1000, 32, 1])
    assert a.unpack(d) == (10, 100, 1000, 32, 1)


@pytest.mark.skip
def test_from_iterable():
    with pytest.raises(TypeError):
        _ = Tibs.from_bools()
    a = Tibs.from_bools([])
    assert a == Tibs()
    a = Tibs.from_bools([1, 0, 1, 1])
    assert a == "0b1011"
    a = Tibs.from_bools((True,))
    assert a == "bool=1"


def test_mul_by_zero():
    a = Tibs.from_string("0b1010")
    b = a * 0
    assert b == Tibs()
    b = a * 1
    assert b == a
    b = a * 2
    assert b == a + a


@pytest.mark.skip
def test_uintne():
    s = Tibs.from_dtype("u160_ne", 454)
    t = Tibs("u160_ne=454")
    assert s == t


@pytest.mark.skip
def test_float_endianness():
    a = Tibs("f64_le=12, f64_be=-0.01, f64_ne=3e33")
    x, y, z = a.unpack(["f64_le", "f64_be", "f64_ne"])
    assert x == 12.0
    assert y == -0.01
    assert z == 3e33
    a = Tibs("f16_le=12, f32_be=-0.01, f32_ne=3e33")
    x, y, z = a.unpack(["f16_le", "f32_be", "f32_ne"])
    assert x / 12.0 == pytest.approx(1.0)
    assert y / -0.01 == pytest.approx(1.0)
    assert z / 3e33 == pytest.approx(1.0)


@pytest.mark.skip
def test_non_aligned_float_reading():
    s = Tibs("0b1, f32 = 10.0")
    (y,) = s.unpack(["pad1", "f32"])
    assert y == 10.0
    s = Tibs("0b1, f32_le = 20.0")
    x, y = s.unpack(["bits1", "f32_le"])
    assert y == 20.0

@pytest.mark.skip
def test_float_errors():
    a = Tibs("0x3")
    with pytest.raises(ValueError):
        _ = a.f
    for le in (8, 10, 12, 18, 30, 128, 200):
        with pytest.raises(ValueError):
            _ = Tibs.from_dtype(DtypeSingle.from_params(DtypeKind.FLOAT, le), 1.0)

@pytest.mark.skip
def test_little_endian_uint():
    s = Tibs("u16 = 100")
    assert s.unpack("u_le") == 25600
    assert s.u_le == 25600
    s = Tibs("u16_le=100")
    assert s.u == 25600
    assert s.u_le == 100
    s = Tibs("u32_le=999")
    assert s.u_le == 999
    s = s.to_mutable_bits()
    s = s.byte_swap()
    assert s.u == 999
    s = Tibs.from_dtype("u24_le", 1001)
    assert s.u_le == 1001
    assert len(s) == 24
    assert s.unpack("u_le") == 1001
    with pytest.raises(ValueError):
        s.unpack('i_le_be')


@pytest.mark.skip
def test_little_endian_errors():
    with pytest.raises(ValueError):
        _ = Tibs("uint15_le=10")
    with pytest.raises(ValueError):
        _ = Tibs("i31_le=-999")
    s = Tibs("0xfff")
    with pytest.raises(ValueError):
        _ = s.i_le
    with pytest.raises(ValueError):
        _ = s.u_be
    with pytest.raises(ValueError):
        _ = s.unpack("uint_le")
    with pytest.raises(ValueError):
        _ = s.unpack("i_ne")


@pytest.mark.skip
def test_big_endian_errors():
    with pytest.raises(ValueError):
        _ = Tibs("u15_be = 10")
    b = Tibs("0xabc")
    with pytest.raises(ValueError):
        _ = b.f_be


def test_native_endian_floats():
    if tibs.byteorder == "little":
        a = Tibs.from_dtype("f64_ne", 0.55)
        assert a.unpack("f64_ne") == 0.55
        assert a.f_le == 0.55
        assert a.f_ne == 0.55
        d =Dtype("f64_ne")
        d2 = DtypeSingle.from_params(DtypeKind.FLOAT, 64, endianness=Endianness.NATIVE)
        assert d == d2
        assert d.endianness is Endianness.NATIVE
        d3 = DtypeSingle.from_params(DtypeKind.FLOAT, 64, endianness=Endianness.LITTLE)
        assert d != d3


@pytest.mark.skip
def test_unpack_dtype_tuple():
    f = "(u8, u8, u8, bool)"
    d = DtypeTuple(f)
    b = d.pack([55, 33, 11, 0])
    assert b.unpack(d) == (55, 33, 11, False)
    assert b.unpack(f) == (55, 33, 11, False)

def test_from_ones():
    a = Tibs.from_ones(0)
    assert a == Tibs()
    a = Tibs.from_ones(1)
    assert a == Tibs("0b1")
    with pytest.raises(ValueError):
        _ = Tibs.from_ones(-1)

def test_from_zeros():
    a = Tibs.from_zeros(0)
    assert a == Tibs()
    a = Tibs.from_zeros(1)
    assert a == Tibs("0b0")
    with pytest.raises(ValueError):
        _ = Tibs.from_zeros(-1)

def test_bits_slicing():
    a = Tibs('0b1010101010101010')
    b = a[-5:-8:1]
    assert b == Tibs()

    assert a[::2] == '0xff'
    assert a[1::2] == '0x00'

def test_from_random():
    a = Tibs.from_random(0)
    assert a == Tibs()
    a = Tibs.from_random(1)
    assert a == '0b1' or a == '0b0'
    a = Tibs.from_random(10000, b'a_seed')
    b = Tibs.from_random(10000, b'a_seed')
    assert a == b
    b = Tibs.from_random(10000, b'a different seed this time - quite long to test if this makes a difference or not. It shouldnt really, but who knows?')
    assert a != b
    c = Mutibs.from_random(10000, b'a_seed')
    assert a == c

@pytest.mark.skip
def test_unpacking_array_dtype_with_no_length():
    a = Tibs.from_dtype('[u8; 10]', range(10))
    assert a.unpack('[u8; 10]') == tuple(range(10))
    assert a.unpack('[u8;]') == tuple(range(10))
    b = a + '0b1'
    assert b.unpack('[u8;]') == tuple(range(10))
    assert len(b.unpack('[bool;]')) == 81

@pytest.mark.skip
def test_creating_from_array_dtype_with_no_length():
    a = Tibs.from_dtype('[u8;]', [1, 2, 3, 4])
    assert a.unpack('[u8;]') == (1, 2, 3, 4)
    a += '0b1111111'
    assert a.unpack('[u8;]') == (1, 2, 3, 4)
    a += '0b1'
    assert a.unpack('[u8;]') == (1, 2, 3, 4, 255)


@pytest.mark.skip
def test_array_from_str():
    x = Tibs('[u8; 1]= [1]')
    assert x.unpack('u8') == 1
    y = Tibs('[bool; 4] = [True, False, True, False]')
    assert y.bin == '1010'

@pytest.mark.skip
def test_tuple_from_str():
    x = Tibs('(u8, u6) = (1, 2)')
    assert x == 'u8 = 1, u6 = 2'
    y = Tibs('(bool, bool) = [0, 0]')
    assert y == 'bool = 0, bool = 0'

def test_is_things():
    a = Tibs('0b1010101010101010')
    assert isinstance(a, Iterable)
    assert isinstance(a, Sequence)

@pytest.mark.skip
def test_conversion_to_long_ints():
    for l in [400, 64, 128, 1000]:
        zeros = Tibs.from_zeros(l)
        assert zeros.i == 0
        assert zeros.u == 0
        ones = Tibs.from_ones(l)
        assert ones.u == (1 << l) - 1
        assert ones.i == -1

# TODO: Nice to reinstate this method of creating Tibs.
# def test_bits_from_bytes_string():
#     a = Tibs("b'ABC'")
#     assert a.to_bytes() == b'ABC'

@pytest.mark.skip
def test_unpack_tuple():
    a = Tibs('0X 123 00ff')
    x, y, z = a.unpack(('hex3', 'u8', 'i'))
    assert x == '123'
    assert y == 0
    assert z == -1

def test_bool_conversion():
    a = Tibs()
    b = Tibs('0b0')
    c = Tibs('0b1')
    assert not a
    assert b
    assert c

def test_rfind_all():
    a = Tibs(' 0 B 0 0 01011')
    g = a.rfind_all('0b1')
    assert next(g) == 6
    assert next(g) == 5
    assert next(g) == 3
    with pytest.raises(StopIteration):
        _ = next(g)

def test_repr():
    a = Tibs()
    assert repr(a) == "Tibs()"
    a = Tibs('')
    assert repr(a) == "Tibs()"
    a = Tibs(" 0b 1")
    assert repr(a) == "Tibs('0b1')"

def test_bits_not_orderable():
    a = Tibs.from_string("0b0")
    b = Tibs.from_string("0b1")
    with pytest.raises(TypeError):
        _ = a < b
    with pytest.raises(TypeError):
        _ = a <= b
    with pytest.raises(TypeError):
        _ = a > b
    with pytest.raises(TypeError):
        _ = a >= b

@pytest.mark.skip
def test_unpack_with_range():
    a = Tibs('u12=99, bool=True, hex4=4321')
    x = a.unpack('hex4', start=-16)
    assert x == "4321"
    assert a.unpack('u', end=12) == 99
