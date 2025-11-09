#!/usr/bin/env python

import pytest
import sys
import bitformat
from bitformat import Bits, Dtype, DtypeTuple, MutableBits
import math
import copy

sys.path.insert(0, "..")


class TestAll:
    def test_creation_from_uint(self):
        s = Bits.from_dtype("u6", 15)
        assert s.bin == "001111"
        s = Bits.from_dtype("u1", 0)
        assert s.bin == "0"
        s = Bits.from_zeros(8)
        assert s.u == 0

    def test_creation_from_oct(self):
        s = Bits.from_dtype(Dtype("oct"), "7")
        assert s.oct == "7"
        assert s.bin == "111"
        s += "0o1"
        assert s.bin == "111001"
        s = Bits() + "0o12345670"
        assert len(s) == 24
        assert s.bin == "001010011100101110111000"
        s = Bits.from_string("0o123")
        assert s.oct == "123"


class TestNoPosAttribute:
    def test_replace(self):
        s = MutableBits.from_string("0b01")
        s = s.replace("0b1", "0b11")
        assert s == "0b011"

    def test_delete(self):
        s = MutableBits('0b000000001')
        del s[-1:]
        assert s == '0b00000000'

    def test_insert(self):
        s = MutableBits.from_string("0b00")
        s = s.insert(1, "0xf")
        assert s == "0b011110"

    def test_insert_self(self):
        b = MutableBits.from_string("0b10")
        b = b.insert(0, b)
        assert b == "0b1010"
        c = MutableBits.from_string("0x00ff")
        c = c.insert(8, c)
        assert c == "0x0000ffff"
        a = MutableBits.from_string("0b11100")
        a = a.insert(3, a)
        assert a == "0b1111110000"

    def test_overwrite(self):
        s = MutableBits.from_string("0b01110")
        s[1:4] = "0b000"
        assert s == "0b00000"

    def test_prepend(self):
        s = Bits.from_zeros(1)
        t = Bits.from_bools([1]) + s
        assert s == "0b0"
        assert t == "0b10"

    def test_rol(self):
        s = MutableBits("0b0001")
        t = s.rol(1)
        assert t == "0b0010"

    def test_ror(self):
        s = Bits.from_string("0b1000")
        t = s.to_mutable_bits().ror(1)
        assert s == "0b1000"
        assert t == "0b0100"

    def test_set_item(self):
        s = MutableBits('0b000100')
        s[4:5] = '0xf'
        assert s == '0b000111110'
        s[0:1] = [1]
        assert s == '0b100111110'
        s[5:5] = MutableBits()
        assert s == '0b100111110'

    def test_adding_nonsense(self):
        a = MutableBits.from_bools([0])
        with pytest.raises(ValueError):
            a += '3'
        with pytest.raises(ValueError):
            a += 'se'
        with pytest.raises(ValueError):
            a += 'float:32'

class Testbyte_aligned:
    def test_not_byte_aligned(self):
        a = Bits.from_string("0x00 ff 0f f")
        li = list(a.find_all("0xff"))
        assert li == [8, 20]
        p = a.find("0x0f")
        assert p == 4
        p = a.rfind("0xff")
        assert p == 20
        a = a.to_mutable_bits().replace("0xff", "")
        assert a == "0x000"

    def test_byte_aligned(self):
        bitformat.Options().byte_aligned = True
        a = Bits.from_string("0x00 ff 0f f")
        li = list(a.find_all("0xff"))
        assert li == [8]
        p = a.find("0x0f")
        assert p == 16
        p = a.rfind("0xff")
        assert p == 8
        a = a.to_mutable_bits().replace("0xff", "")
        assert a == "0x000ff"
        bitformat.Options().byte_aligned = False


class TestSliceAssignment:

    def test_slice_assignment_single_bit(self):
        a = MutableBits('0b000')
        a[2] = '0b1'
        assert a.bin == '001'
        a[0] = MutableBits('0b1')
        assert a.bin == '101'
        a[-1] = 0
        assert a.bin == '100'
        a[-3] = 0
        assert a.bin == '000'

    def test_slice_assignment_single_bit_errors(self):
        a = MutableBits('0b000')
        with pytest.raises(IndexError):
            a[-4] = 1
        with pytest.raises(IndexError):
            a[3] = 1

    def test_slice_assignment_muliple_bits(self):
        a = MutableBits('0b0')
        a[0:1] = '0b110'
        assert a.bin == '110'
        a[0:1] = '0b000'
        assert a.bin == '00010'
        a[0:3] = '0b111'
        assert a.bin == '11110'
        a[-2:] = '0b011'
        assert a.bin == '111011'
        a[:] = '0x12345'
        assert a.hex == '12345'
        a[:] = ''
        assert not a

    def test_slice_assignment_multiple_bits_errors(self):
        a = MutableBits()
        with pytest.raises(IndexError):
            a[0] = '0b00'
        a += '0b1'
        a[0:2] = '0b11'
        assert a == '0b11'

    def test_del_slice_step(self):
        a = MutableBits.from_dtype('bin', '100111101001001110110100101')
        del a[::2]
        assert a.bin == '0110010101100'
        del a[3:9:3]
        assert a.bin == '01101101100'
        del a[2:7:1]
        assert a.bin == '011100'
        del a[::99]
        assert a.bin == '11100'
        del a[::1]
        assert a.bin == ''

    def test_del_slice_negative_step(self):
        a = MutableBits('0b0001011101101100100110000001')
        del a[5:23:-3]
        assert a.bin == '0001011101101100100110000001'
        del a[25:3:-3]
        assert a.bin == '00011101010000100001'
        del a[:6:-7]
        assert a.bin == '000111010100010000'
        del a[15::-2]
        assert a.bin == '0010000000'
        del a[::-1]
        assert a.bin == ''

    def test_del_slice_negative_end(self):
        a = MutableBits('0b01001000100001')
        del a[:-5]
        assert a == '0b00001'
        a = MutableBits('0b01001000100001')
        del a[-11:-5]
        assert a == '0b01000001'

    def test_del_slice_errors(self):
        a = MutableBits.from_zeros(10)
        del a[5:3]
        assert a == Bits.from_zeros(10)
        del a[3:5:-1]
        assert a == Bits.from_zeros(10)

    def test_del_single_element(self):
        a = MutableBits('0b0010011')
        del a[-1]
        assert a.bin == '001001'
        del a[2]
        assert a.bin == '00001'
        with pytest.raises(IndexError):
            del a[5]

    def test_set_slice_step(self):
        a = MutableBits.from_dtype('bin', '0000000000')
        a[::2] = '0b11111'
        assert a.bin == '1010101010'
        a[4:9:3] = [0, 0]
        assert a.bin == '1010001010'
        a[7:3:-1] = [1, 1, 1, 0]
        assert a.bin == '1010011110'
        a[7:1:-2] = [0, 0, 1]
        assert a.bin == '1011001010'
        a[::-5] = [1, 1]
        assert a.bin == '1011101011'
        a[::-1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        assert a.bin == '1000000000'

    def test_set_slice_step_with_int(self):
        a = MutableBits.from_zeros(9)
        with pytest.raises(TypeError):
            a[5:8] = -1

    def test_set_slice_errors(self):
        a = MutableBits.from_zeros(8)
        with pytest.raises(ValueError):
            a[::3] = [1]

        class A(object):
            pass

        with pytest.raises(TypeError):
            a[1:2] = A()
        with pytest.raises(ValueError):
            a[1:4:-1] = [1, 2]

# TODO: Should we allow subclassing or now?
# class TestSubclassing:
#
#     def test_is_instance(self):
#         class SubBits(MutableBits):
#             pass
#
#         a = SubBits()
#         assert isinstance(a, SubBits)
#
#     def test_class_type(self):
#         class SubBits(MutableBits):
#             pass
#
#         assert SubBits().__class__ == SubBits

def test_adding():
    a = Bits.from_string("0b0")
    b = Bits.from_string("0b11")
    c = a + b
    assert c == "0b011"
    assert a == "0b0"
    assert b == "0b11"


def test_copy_method():
    s = Bits.from_zeros(9000)
    t = copy.copy(s)
    assert s == t
    assert s is t
    s = s.to_mutable_bits()
    t = copy.copy(s)
    assert s == t
    assert s is not t


class TestRepr:
    def test_standard_repr(self):
        a = Bits.from_string("0o12345")
        assert repr(a).splitlines()[0] == "Bits('0b001010011100101')"


class TestNewProperties:
    def test_getter_length_errors(self):
        a = Bits.from_string("0x123")
        with pytest.raises(ValueError):
            _ = a.f
        b = Bits()
        with pytest.raises(ValueError):
            _ = b.u

    def test_bytes_properties(self):
        a = Bits.from_bytes(b"hello")
        assert a.bytes == b"hello"

def test_bits_conversion_to_bytes():
    a = Bits.from_string("0x41424344, 0b1")
    b = bytes(a)
    assert b == b"ABCD\x80"
    a = Bits()
    assert bytes(a) == b""

def test_mutable_bits_conversion_to_bytes():
    a = MutableBits('0x0001')
    b = bytes(a)
    assert b == b'\x00\x01'
    b = bytes(MutableBits())
    assert b == b''

class TestBFloats:
    @pytest.mark.skip
    def test_creation(self):
        a = BitArray("bfloat=100.5")
        assert a.unpack("bfloat")[0] == 100.5
        b = BitArray(bfloat=20.25)
        assert b.bfloat == 20.25
        b.bfloat = -30.5
        assert b.bfloat == -30.5
        assert len(b) == 16
        fs = [0.0, -6.1, 1.52e35, 0.000001]
        a = bitstring.from_dtype("4*bfloat", *fs)
        fsp = a.unpack("4*bfloat")
        assert len(a) == len(fs) * 16
        for f, fp in zip(fs, fsp):
            assert f == pytest.approx(fp, abs=abs(f / 100))
        a = BitArray(bfloat=13)
        assert a.bfloat == 13
        c = BitArray()
        with pytest.raises(ValueError):
            _ = c.bfloat

    @pytest.mark.skip
    def test_creation_errors(self):
        a = BitArray(bfloat=-0.25, length=16)
        assert len(a) == 16
        with pytest.raises(ValueError):
            _ = BitArray(bfloat=10, length=15)
        with pytest.raises(ValueError):
            _ = BitArray("bfloat:1=0.5")

    @pytest.mark.skip
    def test_little_endian(self):
        a = BitArray.from_string("f32=1000")
        b = BitArray(bfloat=a.f)
        assert a[0:16] == b[0:16]

        a = BitArray("floatle:32=1000")
        b = BitArray(bfloatle=1000)
        assert a[16:32] == b
        assert b.bfloatle == 1000.0
        b.byte_swap()
        assert b.bfloat == 1000.0
        assert b.bfloatbe == 1000.0

        with pytest.raises(ValueError):
            _ = BitArray(bfloatle=-5, length=15)
        c = BitArray()
        with pytest.raises(ValueError):
            _ = c.bfloatle
        with pytest.raises(ValueError):
            _ = c.bfloatne

    @pytest.mark.skip
    def test_more_creation(self):
        a = BitArray("bfloat:16=1.0, bfloat16=2.0, bfloat=3.0")
        x, y, z = a.unpack("3*bfloat16")
        assert (x, y, z) == (1.0, 2.0, 3.0)

    def test_interpret_bug(self):
        a = Bits.from_ones(100)
        with pytest.raises(ValueError):
            _ = a.f

    def test_overflows(self):
        inf16 = Bits.from_dtype("f16", math.inf)
        inf32 = Bits.from_string("f32 = inf")
        inf64 = Dtype.from_string("f64").pack(float("inf"))

        s = Bits.from_string("f64 = 1e400")
        assert s == inf64
        s = Bits.from_string("f32 = 1e60")
        assert s == inf32
        s = Bits.from_string("f16 = 100000")
        assert s == inf16

        ninf16 = Dtype.from_string("f16").pack(float("-inf"))
        ninf32 = Dtype.from_string("f32").pack(float("-inf"))
        ninf64 = Dtype.from_string("f64").pack(float("-inf"))

        assert ninf64 == Bits.from_string("f64 = -1e400")
        assert ninf32 == Bits.from_string("f32 = -1e60")
        assert ninf16 == Bits.from_string("f16 = -100000")


try:
    import numpy as np

    numpy_installed = True
except ImportError:
    numpy_installed = False


class TestNumpy:
    @pytest.mark.skipif(not numpy_installed, reason="numpy not installed.")
    def test_getting(self):
        a = Bits("0b110")
        p = np.int_(1)
        assert a[p] is True
        p = np.short(0)
        assert a[p] is True

    @pytest.mark.skipif(not numpy_installed, reason="numpy not installed.")
    def test_creation(self):
        a = Bits.from_zeros(np.longlong(12))
        assert a.hex == "000"


def test_bytes_from_list():
    s = Bits.from_dtype("bytes", [1, 2])
    assert s == "0x0102"
    s = Bits.from_bytes(bytearray([1, 2]))
    assert s == "0x0102"

def test_from_dtype_tuple():
    a = Bits.from_dtype(DtypeTuple('(u8, bool)'), [50, True])
    b = Bits.from_dtype(' ( u8, bool )', [50, True])
    assert a.unpack("(u8, bool)") == (50, True)
    assert a == b
