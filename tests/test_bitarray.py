#!/usr/bin/env python

import pytest
from tibs import Tibs, Mutibs
import math
import copy


class TestNoPosAttribute:
    def test_replace(self):
        s = Mutibs.from_string("0b01")
        s = s.replace("0b1", "0b11")
        assert s == "0b011"

    def test_delete(self):
        s = Mutibs('0b000000001')
        del s[-1:]
        assert s == '0b00000000'

    def test_insert(self):
        s = Mutibs.from_string("0b00")
        s = s.insert(1, "0xf")
        assert s == "0b011110"

    def test_insert_self(self):
        b = Mutibs.from_string("0b10")
        b = b.insert(0, b)
        assert b == "0b1010"
        c = Mutibs.from_string("0x00ff")
        c = c.insert(8, c)
        assert c == "0x0000ffff"
        a = Mutibs.from_string("0b11100")
        a = a.insert(3, a)
        assert a == "0b1111110000"

    def test_overwrite(self):
        s = Mutibs.from_string("0b01110")
        s[1:4] = "0b000"
        assert s == "0b00000"

    def test_prepend(self):
        s = Tibs.from_zeros(1)
        t = Tibs.from_bools([1]) + s
        assert s == "0b0"
        assert t == "0b10"

    def test_rol(self):
        s = Mutibs("0b0001")
        t = s.rol(1)
        assert t == "0b0010"

    def test_ror(self):
        s = Tibs.from_string("0b1000")
        t = s.to_mutibs().ror(1)
        assert s == "0b1000"
        assert t == "0b0100"

    def test_set_item(self):
        s = Mutibs('0b000100')
        s[4:5] = '0xf'
        assert s == '0b000111110'
        s[0:1] = [1]
        assert s == '0b100111110'
        s[5:5] = Mutibs()
        assert s == '0b100111110'

    def test_adding_nonsense(self):
        a = Mutibs.from_bools([0])
        with pytest.raises(ValueError):
            a += '3'
        with pytest.raises(ValueError):
            a += 'se'
        with pytest.raises(ValueError):
            a += 'float:32'


def test_not_byte_aligned():
    a = Tibs.from_string("0x00 ff 0f f")
    li = list(a.find_all("0xff"))
    assert li == [8, 20]
    p = a.find("0x0f")
    assert p == 4
    p = a.rfind("0xff")
    assert p == 20
    a = a.to_mutibs().replace("0xff", "")
    assert a == "0x000"


class TestSliceAssignment:

    def test_slice_assignment_single_bit_errors(self):
        a = Mutibs('0b000')
        with pytest.raises(IndexError):
            a[-4] = 1
        with pytest.raises(IndexError):
            a[3] = 1

    def test_slice_assignment_multiple_bits(self):
        a = Mutibs('0b0')
        a[0:1] = '0b110'
        assert a.to_bin() == '110'
        a[0:1] = '0b000'
        assert a.to_bin() == '00010'
        a[0:3] = '0b111'
        assert a.to_bin() == '11110'
        a[-2:] = '0b011'
        assert a.to_bin() == '111011'
        a[:] = '0x12345'
        assert a.to_hex() == '12345'
        a[:] = ''
        assert not a

    def test_slice_assignment_multiple_bits_errors(self):
        a = Mutibs()
        with pytest.raises(IndexError):
            a[0] = '0b00'
        a += '0b1'
        a[0:2] = '0b11'
        assert a == '0b11'

    def test_del_slice_step(self):
        a = Mutibs.from_bin('100111101001001110110100101')
        del a[::2]
        assert a.to_bin() == '0110010101100'
        del a[3:9:3]
        assert a.to_bin() == '01101101100'
        del a[2:7:1]
        assert a.to_bin() == '011100'
        del a[::99]
        assert a.to_bin() == '11100'
        del a[::1]
        assert a.to_bin() == ''

    def test_del_slice_negative_step(self):
        a = Mutibs('0b0001011101101100100110000001')
        del a[5:23:-3]
        assert a.to_bin() == '0001011101101100100110000001'
        del a[25:3:-3]
        assert a.to_bin() == '00011101010000100001'
        del a[:6:-7]
        assert a.to_bin() == '000111010100010000'
        del a[15::-2]
        assert a.to_bin() == '0010000000'
        del a[::-1]
        assert a.to_bin() == ''

    def test_del_slice_negative_end(self):
        a = Mutibs('0b01001000100001')
        del a[:-5]
        assert a == '0b00001'
        a = Mutibs('0b01001000100001')
        del a[-11:-5]
        assert a == '0b01000001'

    def test_del_slice_errors(self):
        a = Mutibs.from_zeros(10)
        del a[5:3]
        assert a == Tibs.from_zeros(10)
        del a[3:5:-1]
        assert a == Tibs.from_zeros(10)

    def test_del_single_element(self):
        a = Mutibs('0b0010011')
        del a[-1]
        assert a.to_bin() == '001001'
        del a[2]
        assert a.to_bin() == '00001'
        with pytest.raises(IndexError):
            del a[5]

    def test_set_slice_step(self):
        a = Mutibs.from_bin('0000000000')
        a[::2] = '0b11111'
        assert a.to_bin() == '1010101010'
        a[4:9:3] = [0, 0]
        assert a.to_bin() == '1010001010'
        a[7:3:-1] = [1, 1, 1, 0]
        assert a.to_bin() == '1010011110'
        a[7:1:-2] = [0, 0, 1]
        assert a.to_bin() == '1011001010'
        a[::-5] = [1, 1]
        assert a.to_bin() == '1011101011'
        a[::-1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        assert a.to_bin() == '1000000000'

    def test_set_slice_step_with_int(self):
        a = Mutibs.from_zeros(9)
        with pytest.raises(TypeError):
            a[5:8] = -1

    def test_set_slice_errors(self):
        a = Mutibs.from_zeros(8)
        with pytest.raises(ValueError):
            a[::3] = [1]

        class A(object):
            pass

        with pytest.raises(TypeError):
            a[1:2] = A()
        with pytest.raises(ValueError):
            a[1:4:-1] = [1, 2]


def test_adding():
    a = Tibs.from_string("0b0")
    b = Tibs.from_string("0b11")
    c = a + b
    assert c == "0b011"
    assert a == "0b0"
    assert b == "0b11"


# def test_copy_method():
#     s = Tibs.from_zeros(9000)
#     t = copy.copy(s)
#     assert s == t
#     assert s is t
#     s = s.to_mutibs()
#     t = copy.copy(s)
#     assert s == t
#     assert s is not t


class TestRepr:
    def test_standard_repr(self):
        a = Tibs.from_string("0o12345")
        assert repr(a).splitlines()[0] == "Tibs('0b001010011100101')"


def test_bits_conversion_to_bytes():
    a = Tibs.from_string("0x41424344, 0b1")
    with pytest.raises(ValueError):
        _ = bytes(a)
    b = bytes(a + [0, 0, 0, 0, 0, 0, 0])
    assert b == b"ABCD\x80"
    a = Tibs()
    assert bytes(a) == b""


def test_mutable_bits_conversion_to_bytes():
    a = Mutibs('0x0001')
    b = bytes(a)
    assert b == b'\x00\x01'
    b = bytes(Mutibs())
    assert b == b''


def test_bytes_from_list():
    s = Tibs.from_bytes(bytes([1, 2]))
    assert s == "0x0102"
    s = Tibs.from_bytes(bytearray([1, 2]))
    assert s == "0x0102"
