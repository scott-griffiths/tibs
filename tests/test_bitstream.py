#!/usr/bin/env python

import pytest
import copy
from tibs import Tibs, Mutibs


class TestFind:
    def test_find1(self):
        s = Tibs("0b0000110110000")
        assert s.find(Tibs("0b11011")) == 4

    def test_find_with_offset(self):
        s = Tibs("0x112233")[4:]
        assert s.find("0x23") == 8

    def test_find_corner_cases(self):
        s = Tibs("0b000111000111")
        assert s.find("0b000") == 0
        assert s.find("0b0111000111") == 2
        assert s[2:].find("0b000") == 4

    def test_find_bytes(self):
        s = Tibs.from_string("0x010203040102ff")
        assert s.find("0x05", byte_aligned=True) is None
        assert s.find("0x02", byte_aligned=True) == 8
        assert s[16:].find("0x02", byte_aligned=True) == 24
        assert s[1:].find("0x02", byte_aligned=True) == 0

    def test_find_bytes_aligned_corner_cases(self):
        s = Tibs("0xff")
        assert s.find(s) is not None
        assert s.find(Tibs("0x12")) is None
        assert s.find(Tibs("0xffff")) is None

    def test_find_byte_aligned(self):
        s = Tibs.from_hex("0x12345678")
        assert s.find(Tibs("0x56"), byte_aligned=True) == 16
        assert not s[16:].find(Tibs("0x45"), byte_aligned=True)
        s = Tibs("0x1234")
        assert s.find("0x1234") == 0

    def test_find_byte_aligned_with_offset(self):
        s = Tibs("0x112233")[4:]
        assert s.find("0x23", byte_aligned=True) == 8

    def test_find_byte_aligned_errors(self):
        s = Tibs("0xffff")
        with pytest.raises(ValueError):
            s.find("")
        with pytest.raises(ValueError):
            s.find(Tibs())


class TestRfind:
    def test_rfind(self):
        a = Tibs("0b001001001")
        b = a.rfind("0b001")
        assert b == 6
        big = Tibs.from_zeros(100000) + "0x12" + Tibs.from_zeros(10000)
        found = big.rfind("0x12", byte_aligned=True)
        assert found == 100000

    def test_rfind_byte_aligned(self):
        a = Tibs("0x8888")
        b = a.rfind("0b1", byte_aligned=True)
        assert b == 8

    def test_rfind_startbit(self):
        a = Tibs("0x0000ffffff")
        b = a.rfind("0x0000", start=1, byte_aligned=True)
        assert b is None

    def test_rfind_endbit(self):
        a = Tibs("0x000fff")
        b = a.rfind("0b011", start=0, end=14, byte_aligned=False)
        assert b is not None
        b = a.rfind("0b011", start=0, end=13, byte_aligned=False)
        assert b is None

    def test_rfind_errors(self):
        a = Tibs("0x43234234")
        with pytest.raises(ValueError):
            a.rfind("", byte_aligned=True)


class TestShift:
    def test_shift_left(self):
        s = Tibs.from_string("0b1010")
        t = s << 1
        assert s.to_bin() == "1010"
        assert t.to_bin() == "0100"
        s = t << 0
        assert s == "0b0100"
        t = s << 100
        assert t.to_bin() == "0000"

    def test_shift_left_errors(self):
        s = Tibs()
        with pytest.raises(ValueError):
            _ = s << 1
        s = Tibs("0xf")
        with pytest.raises(ValueError):
            _ = s << -1

    def test_shift_right(self):
        s = Tibs("0b1010")
        t = s >> 1
        assert s.to_bin() == "1010"
        assert t.to_bin() == "0101"
        q = s >> 0
        assert q == "0b1010"
        t = s >> 100
        assert t.to_bin() == "0000"

    def test_shift_right_errors(self):
        s = Tibs()
        with pytest.raises(ValueError):
            _ = s >> 1
        s = Tibs("0xf")
        with pytest.raises(ValueError):
            _ = s >> -1


class TestReplace:
    def test_replace1(self):
        a = Mutibs("0b1")
        a.replace("0b1", "0b0", byte_aligned=True)
        assert a.to_bin() == "0"
        a.replace("0b1", "0b0", byte_aligned=True)
        assert a.to_bin() == "0"

    def test_replace2(self):
        a = Mutibs("0b00001111111")
        a.replace("0b1", "0b0", byte_aligned=True)
        assert a.to_bin() == "00001111011"
        a.replace("0b1", "0b0", byte_aligned=False)
        assert a.to_bin() == "00000000000"

    def test_replace3(self):
        a = Mutibs("0b0")
        a.replace("0b0", "0b110011111", byte_aligned=True)
        assert a.to_bin() == "110011111"
        a.replace("0b11", "", byte_aligned=False)
        assert a.to_bin() == "001"

    def test_replace4(self):
        a = Mutibs("0x00114723ef4732344700")
        a.replace("0x47", "0x00", byte_aligned=True)
        assert a.to_hex() == "00110023ef0032340000"
        a.replace("0x00", "", byte_aligned=True)
        assert a.to_hex() == "1123ef3234"
        a.replace("0x11", "", start=1, byte_aligned=True)
        assert a.to_hex() == "1123ef3234"
        a.replace("0x11", "0xfff", start=7, byte_aligned=True)
        assert a.to_hex() == "1123ef3234"
        a.replace("0x11", "0xfff", start=0, byte_aligned=True)
        assert a.to_hex() == "fff23ef3234"

    def test_replace5(self):
        a = Tibs.from_string("0xab")
        b = Tibs.from_string("0xcd")
        c = Tibs.from_string("0xabef")
        c = c.to_mutibs().replace(a, b)
        assert c == "0xcdef"
        assert a == "0xab"
        assert b == "0xcd"
        a = Mutibs("0x0011223344").replace("0x11", "0xfff", byte_aligned=True)
        assert a == "0x00fff223344"

    def test_replace_with_self(self):
        a = Mutibs("0b11")
        a.replace("0b1", a)
        assert a == "0xf"
        a.replace(a, a)
        assert a == "0xf"

    def test_replace_count(self):
        a = Mutibs("0x223344223344223344")
        a.replace("0x2", "0x0", count=0, byte_aligned=True)
        assert a.to_hex() == "223344223344223344"
        a.replace("0x2", "0x0", count=1, byte_aligned=True)
        assert a.to_hex() == "023344223344223344"
        a.replace("0x33", "", count=2, byte_aligned=True)
        assert a.to_hex() == "02442244223344"
        a.replace("0x44", "0x4444", count=1435, byte_aligned=True)
        assert a.to_hex() == "02444422444422334444"

    @pytest.mark.skip
    def test_replace_errors(self):
        a = Mutibs("0o123415")
        with pytest.raises(ValueError):
            a.replace("", Tibs("0o7"), byte_aligned=True)


def test_empty_bitstring():
    s = Tibs()
    assert s.to_bin() == ""
    assert s.to_hex() == ""
    assert not s


class TestAppend:
    def test_append(self):
        s1 = Tibs("0b00000")
        s1 = s1 + Tibs.from_bools([1])
        assert s1.to_bin() == "000001"
        assert (Tibs("0x0102") + Tibs("0x0304")).to_hex() == "01020304"

    def test_append_same_bitstring(self):
        s1 = Tibs("0xf0")[:6]
        s1 = s1 + s1
        assert s1.to_bin() == "111100111100"


def test_insert():
    s = Mutibs("0x0011")
    s = s.insert(8, Tibs("0x22"))
    assert s.to_hex() == "002211"
    s = Mutibs.from_ones(0)
    s = s.insert(0, "0b101")
    assert s.to_bin() == "101"


class TestSlice:
    def test_byte_aligned_slice(self):
        s = Tibs("0x123456")
        assert s[8:16].to_hex() == "34"
        s = s[8:24]
        assert len(s) == 16
        assert s.to_hex() == "3456"
        s = s[0:8]
        assert s.to_hex() == "34"

    def test_slice(self):
        s = Tibs("0b000001111100000")
        s1 = s[0:5]
        s2 = s[5:10]
        s3 = s[10:15]
        assert s1.to_bin() == "00000"
        assert s2.to_bin() == "11111"
        assert s3.to_bin() == "00000"


class TestInsert:
    def test_insert(self):
        s1 = Mutibs("0x123456")
        s2 = Tibs("0xff")
        s1.insert(8, s2)
        assert s1.to_hex() == "12ff3456"
        s1.insert(24, "0xee")
        assert s1.to_hex() == "12ff34ee56"
        s1.insert(-1000, "0b1")  # Copying standard behavior of insert on lists
        assert s1 == '0b1, 0x12ff34ee56'
        s1.insert(1000, "0b1")  # Copying standard behavior of insert on lists
        assert s1 == '0b1, 0x12ff34ee56, 0b1'

    def test_insert_null(self):
        s = Mutibs("0x123")
        s.insert(3, Tibs())
        assert s.to_hex() == "123"

    def test_insert_bits(self):
        one = Tibs("0b1")
        zero = Tibs("0b0")
        s = Mutibs("0b00").insert(0, one)
        assert s.to_bin() == "100"
        s.insert(0, zero)
        assert s.to_bin() == "0100"
        s.insert(len(s), one)
        assert s.to_bin() == "01001"
        s.insert(2, s)
        assert s.to_bin() == "0101001001"


class TestOverwriting:
    def test_overwrite_bit(self):
        s = Tibs("0b0").to_mutibs()
        s[0:1] = "0b1"
        assert s.to_bin() == "1"

    def test_overwrite_limits(self):
        s = Mutibs.from_bin("0b11111")
        s[0:3] = "0b000"
        assert s.to_bin() == "00011"
        s[2:5] = "0b000"
        assert s.to_bin() == "00000"

    def test_overwrite_null(self):
        s = Mutibs("0x342563fedec")
        s2 = s[:]
        s[23:23] = Tibs()
        assert s.to_bin() == s2.to_bin()

    def test_overwrite_position(self):
        s1 = Mutibs("0x0123456")
        s2 = Tibs("0xff")
        s1[8: 8 + len(s2)] = s2
        assert s1.to_hex() == "01ff456"
        s1[0:8] = "0xff"
        assert s1.to_hex() == "ffff456"

    def test_overwrite_with_self(self):
        s = Mutibs("0x123")
        s[0:len(s)] = s
        assert s == "0x123"


class TestAdding:
    def test_adding(self):
        s1 = Tibs("0x0102")
        s2 = Tibs("0x0304")
        s3 = s1 + s2
        assert s1.to_hex() == "0102"
        assert s2.to_hex() == "0304"
        assert s3.to_hex() == "01020304"
        s3 += s1
        assert s3.to_hex() == "010203040102"
        assert s2[9:16].to_bin() == "0000100"
        assert s1[0:9].to_bin() == "000000010"
        s4 = Tibs.from_bin("000000010") + Tibs("0b0000100")
        assert s4.to_bin() == "0000000100000100"
        s5 = s1[0:9] + s2[9:16]
        assert s5.to_bin() == "0000000100000100"

    def test_more_adding(self):
        s = Tibs("0b00") + Tibs() + Tibs("0b11")
        assert s.to_bin() == "0011"
        s = "0b01"
        s += Tibs("0b11")
        assert s.to_bin() == "0111"
        s = Tibs("0x00")
        t = Tibs("0x11")
        s += t
        assert s.to_hex() == "0011"
        assert t.to_hex() == "11"
        s += s
        assert s.to_hex() == "00110011"

    def test_radd(self):
        s = "0xff" + Tibs("0xee")
        assert s.to_hex() == "ffee"

    def test_overwrite_more(self):
        s = Mutibs("0b11111")
        s[5:6] = "0b0"
        assert s.to_bin() == "111110"
        s[1:] = Tibs("0x00")
        assert s.to_bin() == "100000000"

    def test_get_item_with_positive_position(self):
        s = Tibs("0b1011")
        assert s[0] is True
        assert s[1] is False
        assert s[2] is True
        assert s[3] is True
        with pytest.raises(IndexError):
            _ = s[4]

    def test_get_item_with_negative_position(self):
        s = Tibs("0b1011")
        assert s[-1] is True
        assert s[-2] is True
        assert s[-3] is False
        assert s[-4] is True
        with pytest.raises(IndexError):
            _ = s[-5]

    def test_slicing(self):
        s = Tibs("0x0123456789")
        assert s[0:8].to_hex() == "01"
        assert not s[0:0]
        assert not s[23:20]
        assert s[8:12].to_bin() == "0010"
        assert s[32:80] == "0x89"

    def test_negative_slicing(self):
        s = Tibs("0x012345678")
        assert s[:-8].to_hex() == "0123456"
        assert s[-16:-8].to_hex() == "56"
        assert s[-24:].to_hex() == "345678"
        assert s[-1000:-24] == "0x012"

    def test_len(self):
        s = Tibs()
        assert len(s) == 0
        s = s + "0b001"
        assert len(s) == 3

    def test_join(self):
        s1 = Tibs("0b0")
        s2 = Tibs("0b1")
        s3 = Tibs("0b000")
        s4 = Tibs("0b111")
        strings = [s1, s2, s1, s3, s4]
        s = Tibs.from_joined(strings)
        assert s.to_bin() == "010000111"

    def test_join2(self):
        s1 = Tibs("0x00f1")
        assert s1[4:12].to_hex() == "0f"
        bsl = [s1[0:16], s1[4:12]]
        s = Tibs.from_joined(bsl)
        assert s.to_hex() == "00f10f"

        s1 = Tibs("0x00112233445566778899aabbccddeeff")
        s2 = Tibs("0b000011")
        bsl = [s1[0:32], s1[4:12], s2, s2, s2, s2]
        s = Tibs.from_joined(bsl)
        assert s.to_hex() == "00112233010c30c3"

    def test_join_with_ints(self):
        with pytest.raises(TypeError):
            _ = Tibs.from_joined([1, 2])

    def test_various_things2(self):
        s1 = Tibs("0x1f08")[:13]
        assert s1.to_bin() == "0001111100001"
        s2 = Tibs("0b0101")
        assert s2.to_bin() == "0101"
        s1 += s2
        assert len(s1) == 17
        assert s1.to_bin() == "00011111000010101"
        s1 = s1[3:8]
        assert s1.to_bin() == "11111"

    def test_various_things3(self):
        s1 = Tibs("0x012480ff")[2:27]
        s2 = s1 + s1
        assert len(s2) == 50
        s3 = s2[0:25]
        s4 = s2[25:50]
        assert s3.to_bin() == s4.to_bin()

    def test_insert_using_auto(self):
        s = Mutibs("0xff")
        s = s.insert(4, "0x00")
        assert s.to_hex() == "f00f"

    def test_overwrite_using_auto(self):
        s = Mutibs("0x0110")
        s[0:1] = "0b1"
        assert s.to_hex() == "8110"
        s[0:0] = ""
        assert s.to_hex() == "8110"

    def test_find_using_auto(self):
        s = Tibs("0b000000010100011000")
        assert s.find("0b101") == 7

    def test_findbyte_aligned_using_auto(self):
        s = Tibs("0x00004700")
        assert s.find("0b01000111", byte_aligned=True) == 16

    def test_append_using_auto(self):
        s = Tibs("0b000")
        s = s + "0b111"
        assert s.to_bin() == "000111"
        s = s + "0b0"
        assert s.to_bin() == "0001110"

    def test_prepend(self):
        s = Tibs("0b000")
        s = "0b11" + s
        assert s.to_bin() == "11000"
        s = s + s
        assert s.to_bin() == "1100011000"
        s = "" + s
        assert s.to_bin() == "1100011000"

    def test_null_slice(self):
        s = Tibs("0x111")
        t = s[1:1]
        assert len(t) == 0

    def test_multiple_autos(self):
        s = Tibs("0xa")
        s = "0xf" + s
        s = s + "0xb"
        assert s == "0xfab"
        s = s + s
        s = s + "0x100"
        print(type(s))
        with pytest.raises(TypeError):
            s[4: 8] = "0x5"
        s = s.to_mutibs()
        s[4:8] = "0x5"
        assert s == "0xf5bfab100"

    def test_reverse(self):
        s = Tibs("0b0011")
        s = s.to_mutibs().reverse()
        assert s.to_bin() == "1100"
        s = Mutibs("0b10")
        s.reverse()
        assert s.to_bin() == "01"
        s = Mutibs()
        s.reverse()
        assert s.to_bin() == ""

    def test_init_with_concatenated_strings(self):
        s = Tibs("0xff, 0xee,0xd ,0xcc")
        assert s.to_hex() == "ffeedcc"
        s = Tibs("0b0 ,0b111 ,0b001")
        assert s.to_bin() == "0111001"
        s = Tibs("0xffee")
        assert s.to_hex() == "ffee"
        s = Tibs("  0o123 ,0o7 ,0o1")
        assert s.to_oct() == "12371"
        s += "  0o 332"
        assert s.to_oct() == "12371332"

    def test_equals(self):
        s1 = Tibs("0b01010101")
        s2 = Tibs("0b01010101")
        assert s1 == s2
        s3 = Tibs()
        s4 = Tibs()
        assert s3 == s4
        assert not s3 != s4

    def test_large_equals(self):
        s1 = Tibs.from_zeros(1000000)
        s2 = Mutibs.from_zeros(1000000)
        s1 = s1.to_mutibs().set(True, [-1, 55, 53214, 534211, 999999])
        s2.set(True, [-1, 55, 53214, 534211, 999999])
        assert s1 == s2
        s1 = s1.set(True, 800000)
        assert s1 != s2

    def test_not_equals(self):
        s1 = Tibs("0b0")
        s2 = Tibs("0b1")
        assert s1 != s2
        assert not s1 != Tibs("0b0")

    def test_equality_with_auto_initialised(self):
        a = Tibs("0b00110111")
        assert a == "0b00110111"
        assert a == "0x37"
        assert "0b0011 0111" == a
        assert "0x37" == a
        assert not a == "0b11001000"
        assert not "0x3737" == a

    def test_invert_special_method(self):
        s = Tibs("0b00011001")
        assert (~s).to_bin() == "11100110"
        assert (~Tibs("0b0")).to_bin() == "1"
        assert (~Tibs("0b1")).to_bin() == "0"
        assert ~~s == s

    def test_invert_special_method_errors(self):
        s = Tibs()
        with pytest.raises(ValueError):
            _ = ~s

    def test_join_with_auto(self):
        s = Tibs.from_joined(["0xf", "0b00", Tibs.from_bin("11")])
        assert s == "0b11110011"


class TestMultiplication:
    def test_multiplication(self):
        a = Tibs("0xff")
        b = a * 8
        assert b == "0xffffffffffffffff"
        b = 4 * a
        assert b == "0xffffffff"
        assert 1 * a == a * 1 == a
        c = a * 0
        assert not c
        a *= 3
        assert a == "0xffffff"
        a *= 0
        assert not a
        one = Tibs("0b1")
        zero = Tibs("0b0")
        mix = one * 2 + 3 * zero + 2 * one * 2
        assert mix == "0b110001111"
        q = Tibs()
        q *= 143
        assert not q
        q += Tibs.from_bools([True, True, False])
        q *= 0
        assert not q

    def test_multiplication_errors(self):
        a = Tibs("0b1")
        b = Tibs("0b0")
        with pytest.raises(ValueError):
            _ = a * -1
        with pytest.raises(ValueError):
            a *= -1
        with pytest.raises(ValueError):
            _ = -1 * a
        with pytest.raises(TypeError):
            _ = a * 1.2
        with pytest.raises(TypeError):
            _ = b * a
        with pytest.raises(TypeError):
            a *= b


class TestBitWise:
    def test_bitwise_and(self):
        a = Tibs("0b01101")
        b = Tibs("0b00110")
        assert (a & b).to_bin() == "00100"
        assert (a & "0b11111") == a
        with pytest.raises(ValueError):
            _ = a & "0b1"
        with pytest.raises(ValueError):
            _ = b & "0b110111111"
        c = Tibs("0b0011011")
        d = c & "0b1111000"
        assert d.to_bin() == "0011000"
        d = "0b1111000" & c
        assert d.to_bin() == "0011000"

    def test_bitwise_or(self):
        a = Tibs("0b111001001")
        b = Tibs("0b011100011")
        c = a | b
        assert c.to_bin() == "111101011"
        assert (a | "0b000000000") == a
        with pytest.raises(ValueError):
            _ = a | "0b0000"
        with pytest.raises(ValueError):
            _ = b | (a + "0b1")
        a = "0xff00" | Tibs("0x00f0")
        assert a.to_hex() == "fff0"

    def test_bitwise_xor(self):
        a = Tibs("0b111001001")
        b = Tibs("0b011100011")
        c = a ^ b
        assert c.to_bin() == "100101010"
        assert (a ^ "0b111100000").to_bin() == "000101001"
        with pytest.raises(ValueError):
            _ = a ^ "0b0000"
        with pytest.raises(ValueError):
            _ = b ^ (a + "0b1")
        a = "0o707" ^ Tibs("0o777")
        assert a.to_oct() == "070"


def test_mutable_bitwise():
    a = '0xf0' & Mutibs('0x0f')
    assert a == '0x00'
    a = '0xf0' | Mutibs('0x0f')
    assert a == '0xff'
    a = '0xf0' & Mutibs('0x0f')
    assert a == '0x00'


class TestManyDifferentThings:
    def test_find_byte_aligned_with_bits(self):
        a = Tibs("0x00112233445566778899")
        x = a.find("0b0001", byte_aligned=True)
        assert x == 8

    def test_find_startbit_not_byte_aligned(self):
        a = Tibs("0b0010000100")
        found = a.find("0b1", start=4)
        assert found == 7
        found = a.find("0b1", start=2)
        assert found == 2
        found = a.find("0b1", start=8, byte_aligned=False)
        assert found is None

    def test_find_endbit_not_byte_aligned(self):
        a = Tibs("0b0010010000")
        found = a.find("0b1", end=2, byte_aligned=False)
        assert found is None
        found = a.find("0b1", end=3)
        assert found == 2
        found = a.find("0b1", start=3, end=5, byte_aligned=False)
        assert found is None
        found = a.find("0b1", start=3, end=6)
        assert found == 5

    def test_find_startbit_byte_aligned(self):
        a = Tibs("0xff001122ff0011ff")
        found = a.find("0x22", start=24, byte_aligned=True)
        assert found == 24
        found = a.find("0b111", start=40, byte_aligned=True)
        assert found == 40 + 16

    def test_find_endbit_byte_aligned(self):
        a = Tibs("0xff001122ff0011ff")
        found = a.find("0x22", start=31, byte_aligned=True)
        assert found is None
        found = a.find("0x22", end=32, byte_aligned=True)
        assert found == 24

    def test_find_all(self):
        a = Tibs("0b11111")
        p = a.find_all("0b1")
        assert list(p) == [0, 1, 2, 3, 4]
        p = a.find_all("0b11")
        assert list(p) == [0, 1, 2, 3]
        p = a.find_all("0b10")
        assert list(p) == []
        a = Tibs("0x4733eeff66554747335832434547")
        p = a.find_all("0x47", byte_aligned=True)
        assert list(p) == [0, 6 * 8, 7 * 8, 13 * 8]
        p = a.find_all("0x4733", byte_aligned=True)
        assert list(p) == [0, 7 * 8]
        a = Tibs("0b1001001001001001001")
        p = a.find_all("0b1001", byte_aligned=False)
        assert list(p) == [0, 3, 6, 9, 12, 15]

    def test_find_all_generator(self):
        a = Tibs("0xff1ff4512345ff1234ff12ff")
        p = a.find_all("0xff", byte_aligned=True)
        assert next(p) == 0
        assert next(p) == 6 * 8
        assert next(p) == 9 * 8
        assert next(p) == 11 * 8
        with pytest.raises(StopIteration):
            _ = next(p)

    def test_contains(self):
        a = Tibs("0b1") + "0x0001dead0001"
        assert "0xdead" in a
        assert "0xfeed" not in a

    def test_repr(self):
        bls = ["", "0b1", "0o5", "0x43412424f41", "0b00101001010101"]
        for bs in bls:
            a = Tibs(bs)
            b = eval(a.__repr__())
            assert a == b
        a = Tibs("0b1")
        assert repr(a).splitlines()[0] == "Tibs('0b1')"
        a += "0b11"
        assert repr(a).splitlines()[0] == "Tibs('0b111')"
        a += "0b1"
        assert repr(a).splitlines()[0] == "Tibs('0xf')"

    def test_iter(self):
        a = Tibs("0b001010")
        b = Tibs()
        for bit in a:
            b = b + Tibs.from_bools([bit])
        assert a == b

    def test_non_zero_bits_at_end(self):
        a = Tibs.from_bytes(b"\xff")[:5]
        b = Tibs("0b00")
        a += b
        assert a == "0b1111100"
        assert (a + [0]).to_bytes() == b"\xf8"
        with pytest.raises(AttributeError):
            _ = a.bytes
        with pytest.raises(ValueError):
            _ = a.to_bytes()

    def test_slice_step(self):
        a = Tibs("0x3")
        b = a[::1]
        assert a == b
        assert a[2:4:1] == "0b11"
        assert a[0:2:1] == "0b00"
        assert a[:3] == "0o1"

        a = Tibs("0x0011223344556677")
        assert a[-8:] == "0x77"
        assert a[:-24] == "0x0011223344"
        assert a[-1000:-24] == "0x0011223344"

    def test_interesting_slice_step(self):
        a = Tibs("0b0011000111")
        assert a[7:3:-1] == "0b1000"
        assert a[9:2:-1] == "0b1110001"
        assert a[8:2:-2] == "0b100"
        assert a[100:-20:-3] == "0b1010"
        assert a[100:-20:-1] == "0b1110001100"
        assert a[10:2:-1] == "0b1110001"
        assert a[100:2:-1] == "0b1110001"

    def test_reverse_with_slice(self):
        a = Tibs("0x0012ff")
        b = a.to_mutibs()
        b.reverse()
        assert a == "0x0012ff"
        assert b == "0xff4800"
        a = a[8:16].to_mutibs()
        a.reverse()
        assert a == "0x48"

    def test_cut(self):
        a = Tibs("0x00112233445")
        b = list(a.chunks(8))
        assert b == ["0x00", "0x11", "0x22", "0x33", "0x44", "0x5"]
        b = list(a[8:16].chunks(4))
        assert b == ["0x1", "0x1"]
        b = list(a[0:44].chunks(4, 4))
        assert b == ["0x0", "0x0", "0x1", "0x1"]
        a = Tibs()
        b = list(a.chunks(10))
        assert not b

    def test_cut_errors(self):
        a = Tibs("0b1")
        with pytest.raises(ValueError):
            _ = a.chunks(0)
        with pytest.raises(ValueError):
            _ = a.chunks(-2)
        with pytest.raises(ValueError):
            _ = a.chunks(1, count=-1)

    def test_cut_problem(self):
        s = Tibs("0x1234")
        for n in list(s.chunks(4)):
            s = n + s
        assert s == "0x43211234"

    def test_join_functions(self):
        a = Tibs.from_joined(["0xa", "0xb", "0b1111"])
        assert a == "0xabf"

    def test_difficult_prepends(self):
        a = Tibs("0b1101011")
        b = Tibs()
        for i in range(10):
            b = a + b
        assert b == a * 10

    def test_reverse_bytes(self):
        a = Mutibs("0x123456")
        a.byte_swap()
        assert a == "0x563412"
        b = a + "0b1"
        with pytest.raises(ValueError):
            b.byte_swap()
        a = Mutibs("0x54")
        a = a.byte_swap()
        assert a == "0x54"
        a = Mutibs()
        a = a.byte_swap()
        assert not a

    def test_startswith(self):
        a = Tibs()
        assert a.starts_with(Tibs())
        assert not a.starts_with("0b0")
        a = Tibs("0x12ff")
        assert a.starts_with("0x1")
        assert a.starts_with("0b0001001")
        assert a.starts_with("0x12ff")
        assert not a.starts_with("0x12ff, 0b1")
        assert not a.starts_with("0x2")

    def test_startswith_start_end(self):
        s = Tibs("0x123456")
        assert s[4:].starts_with("0x234")
        assert not s[:11].starts_with("0x123")
        assert s[:12].starts_with("0x123")
        assert s[8:16].starts_with("0x34")
        assert not s[7:16].starts_with("0x34")
        assert not s[9:16].starts_with("0x34")
        assert not s[8:15].starts_with("0x34")

    def test_endswith(self):
        a = Tibs()
        assert a.ends_with("")
        assert not a.ends_with(Tibs("0b1"))
        a = Tibs("0xf2341")
        assert a.ends_with("0x41")
        assert a.ends_with("0b001")
        assert a.ends_with("0xf2341")
        assert not a.ends_with("0x1f2341")
        assert not a.ends_with("0o34")

    def test_endswith_start_end(self):
        s = Tibs("0x123456")
        assert s[:16].ends_with("0x234")
        assert not s[13:].ends_with("0x456")
        assert s[12:].ends_with("0x456")
        assert s[8:16].ends_with("0x34")
        assert s[7:16].ends_with("0x34")
        assert not s[9:16].ends_with("0x34")
        assert not s[8:15].ends_with("0x34")

    @pytest.mark.skip
    def test_const_bit_stream_hashability(self):
        a = Tibs("0x1")
        b = Tibs("0x2")
        c = Tibs("0x1")
        s = {a, b, c}
        assert len(s) == 2
        assert hash(a) == hash(c)

    @pytest.mark.skip
    def test_hash_edge_cases(self):
        a = Tibs("0xabcd")
        b = Tibs("0xabcd")
        c = b[1:]
        assert hash(a) == hash(b)
        assert hash(a) != hash(c)

    @pytest.mark.skip
    def test_const_bits_copy(self):
        a = Tibs("0xabc")
        b = copy.copy(a)
        assert id(a) == id(b)


class TestSet:
    def test_set(self):
        a = Tibs.from_zeros(16)
        a = a.to_mutibs().set(True, 0)
        assert a == Mutibs("0b10000000 00000000")
        a.set(1, 15)
        assert a == "0b10000000 00000001"
        b = a[4:12]
        b = b.set(True, 1)
        assert b == "0b01000000"
        b = b.set(True, -1)
        assert b == "0b01000001"
        b = b.set(1, -8)
        assert b == "0b11000001"
        with pytest.raises(IndexError):
            _ = b.set(True, -9)
        with pytest.raises(IndexError):
            _ = b.set(True, 8)

    def test_set_negative_index(self):
        a = Mutibs.from_string('0b0110000000')
        a.set(1, -1)
        assert a.to_bin() == "0110000001"
        a.set(1, [-1, -10])
        assert a.to_bin() == "1110000001"
        with pytest.raises(IndexError):
            a.set(1, [-11])

    def test_set_list(self):
        a = Tibs.from_zeros(18)
        b = a.to_mutibs().set(True, range(18))
        assert b.all()
        b.set(False, range(18))
        assert not b.any()

    def test_unset(self):
        a = Mutibs.from_ones(16)
        a.set(False, 0)
        b = ~a
        assert b == "0b10000000 00000000"
        a.set(0, 15)
        assert ~a == "0b10000000 00000001"
        b = a[4:12]
        b.set(False, 1)
        assert ~b == "0b01000000"
        b.set(False, -1)
        assert ~b == "0b01000001"
        b.set(False, -8)
        assert ~b == "0b11000001"
        with pytest.raises(IndexError):
            b.set(False, -9)
        with pytest.raises(IndexError):
            b.set(False, 8)


class TestInvert:
    def test_invert_bits(self):
        a = Mutibs("0b111000")
        a = a.invert(range(len(a)))
        assert a == "0b000111"
        a = a.invert([0, 1, -1])
        assert a == "0b110110"

    def test_invert_whole_bit_stream(self):
        a = Mutibs("0b11011")
        a = a.invert()
        assert a == "0b00100"

    def test_invert_single_bit(self):
        a = Mutibs("0b000001")
        a = a.invert(0)
        assert a.to_bin() == "100001"
        a = a.invert(-1)
        assert a.to_bin() == "100000"

    def test_invert_errors(self):
        a = Mutibs.from_zeros(10)
        with pytest.raises(IndexError):
            _ = a.invert(10)
        with pytest.raises(IndexError):
            _ = a.invert(-11)
        with pytest.raises(IndexError):
            _ = a.invert([1, 2, 10])

    def test_ior(self):
        a = Tibs("0b1101001")
        a |= "0b1110000"
        assert a == "0b1111001"
        b = a[2:]
        c = a[1:-1]
        b |= c
        assert c == "0b11100"
        assert b == "0b11101"

    def test_iand(self):
        a = Tibs("0b0101010101000")
        a &= "0b1111110000000"
        assert a == "0b0101010000000"

    def test_ixor(self):
        a = Tibs("0b11001100110011")
        a ^= "0b11111100000010"
        assert a == "0b00110000110001"

    def test_logical_inplace_errors(self):
        a = Tibs.from_zeros(4)
        with pytest.raises(ValueError):
            a |= "0b111"
        with pytest.raises(ValueError):
            a &= "0b111"
        with pytest.raises(ValueError):
            a ^= "0b111"


class TestAllAndAny:
    def test_all(self):
        a = Tibs("0b111")
        assert a.all()

    def test_any(self):
        a = Tibs("0b10011011")
        assert a.any()

    def test_all_false(self):
        a = Tibs("0b0010011101")
        assert not a.all()

    def test_any_false(self):
        a = Tibs("0b0000")
        assert not a.any()

    def test_any_empty_bitstring(self):
        a = Tibs()
        assert not a.any()  # Python any function returns False for empty iterables.

    def test_all_empty_bit_stream(self):
        a = Tibs()
        assert a.all()  # Python all function returns True for empty iterables.

    def test_any_whole_bitstring(self):
        a = Tibs("0xfff")
        assert a.any()

    def test_all_whole_bitstring(self):
        a = Tibs("0xfff")
        assert a.all()

    ###################


class TestMoreMisc:

    def test_ror(self):
        a = Tibs("0b11001").to_mutibs()
        a.ror(0)
        assert a == "0b11001"
        a.ror(1)
        assert a == "0b11100"
        a.ror(5)
        assert a == "0b11100"
        a.ror(101)
        assert a == "0b01110"
        a = Mutibs("0b1")
        a.ror(1000000)
        assert a == "0b1"

    def test_ror_errors(self):
        a = Mutibs()
        with pytest.raises(ValueError):
            a.ror(0)
        a += "0b001"
        with pytest.raises(ValueError):
            a.ror(-1)

    def test_rol(self):
        a = Mutibs("0b11001")
        a.rol(0)
        assert a == "0b11001"
        a.rol(1)
        assert a == "0b10011"
        a = a.rol(5)
        assert a == "0b10011"
        a.rol(101)
        assert a == "0b00111"
        a = Tibs("0b1")
        a = a.to_mutibs().rol(1000000)
        assert a == "0b1"

    def test_rol_errors(self):
        a = Mutibs()
        with pytest.raises(ValueError):
            a.rol(0)
        a += "0b001"
        with pytest.raises(ValueError):
            a.rol(-1)

    def test_init_with_zeros(self):
        a = Tibs.from_zeros(0)
        assert not a
        a = Tibs.from_zeros(1)
        assert a == "0b0"
        with pytest.raises(ValueError):
            _ = Tibs.from_zeros(-1)
        with pytest.raises(TypeError):
            a += 10

    def test_add_verses_in_place_add(self):
        a1 = Tibs("0xabc")
        b1 = a1
        a1 += "0xdef"
        assert a1 == "0xabcdef"
        assert b1 == "0xabc"

    def test_and_verses_in_place_and(self):
        a1 = Tibs("0xabc")
        b1 = a1
        a1 &= "0xf0f"
        assert a1 == "0xa0c"
        assert b1 == "0xabc"

    def test_or_verses_in_place_or(self):
        a1 = Tibs("0xabc")
        b1 = a1
        a1 |= "0xf0f"
        assert a1 == "0xfbf"
        assert b1 == "0xabc"

    def test_xor_verses_in_place_xor(self):
        a1 = Tibs("0xabc")
        b1 = a1
        a1 ^= "0xf0f"
        assert a1 == "0x5b3"
        assert b1 == "0xabc"

    def test_mul_verses_in_place_mul(self):
        a1 = Tibs("0xabc")
        b1 = a1
        a1 *= 3
        assert a1 == "0xabcabcabc"
        assert b1 == "0xabc"

    def test_lshift_verses_in_place_lshift(self):
        a1 = Tibs("0xabc")
        b1 = a1
        a1 <<= 4
        assert a1 == "0xbc0"
        assert b1 == "0xabc"

    def test_rshift_verses_in_place_rshift(self):
        a1 = Tibs("0xabc")
        b1 = a1
        a1 >>= 4
        assert a1 == "0x0ab"
        assert b1 == "0xabc"


class TestBugs:
    def test_bug_in_replace(self):
        s = Mutibs("0x00112233")
        s = s.replace("0x22", "0xffff", start=8, byte_aligned=True)
        assert s == "0x0011ffff33"
        s = Mutibs("0x0123412341234")
        s = s.replace("0x23", "0xf", start=9, byte_aligned=True)
        assert s == "0x012341f41f4"

    def test_function_negative_indices(self):
        # insert
        s = Mutibs("0b0111")
        s = s.insert(-1, "0b0")
        assert s == "0b01101"

        # overwrite
        t = Mutibs("0x77ab9988c7bf")
        t[-20: -20 + 12] = "0x666"
        assert t == "0x77ab998666bf"

        # find
        found = t.find("0x998", start=-32, byte_aligned=True)
        assert found == 16
        found = t.find("0x988", end=-21, byte_aligned=True)
        assert found is None
        found = t.find("0x998", end=-20, byte_aligned=True)
        assert found == 16

        # find_all
        s = Tibs("0x1234151f")
        li = list(s.find_all("0x1", start=-15))
        assert li == [24]
        li = list(s.find_all("0x1", start=-16, byte_aligned=True))
        assert li == [16, 24]
        li = list(s.find_all("0x1", end=-5, byte_aligned=True))
        assert li == [0, 16]
        li = list(s.find_all("0x1", end=-4, byte_aligned=True))
        assert li == [0, 16, 24]

        # rfind
        found = (s.rfind("0x1f", end=-1))
        assert found is None
        found = s.rfind("0x12", start=-31)
        assert found is None

        # chunks
        s = Tibs("0x12345")
        li = list(s[-12:-4].chunks(4))
        assert li == ["0x3", "0x4"]

        # startswith
        s = Mutibs("0xfe0012fe1200fe")
        assert s[-16:].starts_with("0x00f")
        assert s[:-40].starts_with("0xfe00")
        assert not s[:-41].starts_with("0xfe00")

        # endswith
        assert s[-16:].ends_with("0x00fe")
        assert not s[-15:].ends_with("0x00fe")
        assert not s[-1:].ends_with("0x00fe")
        assert s[:-4].ends_with("0x00f")

        # replace
        s.replace("0xfe", "", end=-1)
        assert s == "0x00121200fe"
        s.replace("0x00", "", start=-24)
        assert s == "0x001212fe"

    def test_rotate_start_and_end(self):
        a = Mutibs("0b110100001")
        a.rol(1, 3, 6)
        assert a == "0b110001001"
        a.ror(1, start=-4)
        assert a == "0b110001100"
        a.rol(202, end=-5)
        assert a == "0b001101100"
        a.ror(3, end=4)
        assert a == "0b011001100"
        with pytest.raises(ValueError):
            a.rol(5, start=-4, end=-6)

    def test_byte_swap_int(self):
        s = Mutibs("0xf234567f")
        s.byte_swap(1)
        assert s == "0xf234567f"
        s.byte_swap(2)
        assert s == "0x34f27f56"
        s.byte_swap(2)
        assert s == "0xf234567f"
        with pytest.raises(ValueError):
            s.byte_swap(3)

    def test_byte_swap_errors(self):
        s = Mutibs("0x0011223344556677")
        with pytest.raises(TypeError):
            s.byte_swap("z")
        with pytest.raises(ValueError):
            s.byte_swap(-1)
        with pytest.raises(TypeError):
            s.byte_swap([-1])
        with pytest.raises(TypeError):
            s.byte_swap(5.4)


def test_count():
    a = Tibs("0xf0f")
    assert a.count(True) == 8
    assert a.count(False) == 4

    b = Tibs()
    assert b.count(True) == 0
    assert b.count(False) == 0

    a = Tibs("0xff0120ff")
    b = a[1:-1]
    assert b.count(1) == 16
    assert b.count(0) == 14


def test_overwrite_with_self():
    s = Mutibs("0b1101")
    s[:] = s
    assert s == "0b1101"


def test_byte_swap():
    b = Tibs.from_bytes(b"\x01\x02\x03\x04")
    c = b.to_mutibs().byte_swap()
    assert c == "0x04030201"


def test_overlapping_bits():
    a = Tibs('0x00fff0')
    zeros = a[0:8]
    x = a[4:16]
    y = x[1:9]
    assert a == "0x00fff0"
    assert zeros == "0x00"
    assert x == "0x0ff"
    assert y == Tibs("0b00011111")
    _ = ~y
    _ = y.to_mutibs().set(0, [0, 1, 2, 3, 4, 5, 6, 7])
    _ = y.to_mutibs().byte_swap()
    _ = y.to_mutibs().ror(1)
    _ = y.to_mutibs().rol(1)
    assert a == "0x00fff0"
    assert zeros == "0x00"
    assert x == "0x0ff"
    assert y == Tibs("0b00011111")
    y = ~y
    assert y == Tibs("0b11100000")
    y = y.to_mutibs().set(0, [2, 3]).to_tibs()
    y = y.to_mutibs().byte_swap().to_tibs()
    y = y.to_mutibs().ror(2)
    y = y.rol(1)
    assert a == "0x00fff0"
    assert zeros == "0x00"
    assert x == "0x0ff"
    assert y == Tibs("0b01100000")


def test_mutable_freeze():
    a = Mutibs('0x0000')
    b = a.to_tibs()
    assert isinstance(b, Tibs)
    assert a == b
    a.set(1, -1)
    assert a == '0x0001'
    assert b.to_hex() == '0000'


def test_del_unavailability():
    a = Tibs('0xff')
    with pytest.raises(TypeError):
        del a[:]
