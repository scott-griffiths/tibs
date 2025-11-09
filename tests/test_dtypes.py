import pytest
import sys

import bitformat
from bitformat import Dtype, Bits, Endianness, DtypeTuple, DtypeSingle, DtypeArray
from bitformat._dtypes import DtypeDefinition, Register
from bitformat._common import DtypeKind, Expression, ExpressionError

sys.path.insert(0, "..")


class TestBasicFunctionality:
    def test_setting_bool(self):
        b = Dtype("bool")
        assert str(b) == "bool"
        assert isinstance(b, DtypeSingle)
        assert b.kind is DtypeKind.BOOL
        assert b.size == 1
        assert b.bit_length == 1

        b2 = Dtype.from_string("bool1")
        assert b == b2
        # self.assertTrue(b is b2)

    def test_setting_with_length(self):
        d = DtypeSingle.from_params(DtypeKind.UINT, 12)
        assert str(d) == "u12"
        assert d.size == 12
        assert d.kind is DtypeKind.UINT

    def test_build_errors(self):
        dtype = Dtype.from_string("u8")
        value = "not_an_integer"
        with pytest.raises(ValueError):
            dtype.pack(value)

    def test_pack(self):
        dtype = Dtype("i88")
        x = dtype.pack(10001)
        assert x.i == 10001

    def test_unpack(self):
        dtype = Dtype("u12")
        x = dtype.unpack("0x3ff")
        assert x == 1023

    def test_immutability(self):
        d = Dtype("f32")
        with pytest.raises(AttributeError):
            d.kind = "uint8"

    def test_building_bits(self):
        d = Dtype.from_string("bits3")
        a = d.pack("0b101")
        assert a == "0b101"
        with pytest.raises(ValueError):
            d.pack("0b1010")

    def test_building_bin(self):
        d = Dtype.from_string("bin9")
        a = d.pack("0b000111000")
        assert a == "0b000111000"
        with pytest.raises(ValueError):
            d.pack("0b0001110000")

    def test_building_ints(self):
        d = Dtype("i3")
        a = d.pack(-3)
        assert a == "0b101"
        with pytest.raises(ValueError):
            d.pack(4)

    def test_dtype_single_creation(self):
        d = Dtype('u8')
        assert isinstance(d, DtypeSingle)
        assert d.kind == DtypeKind.UINT
        assert d.size.const_value == 8
        assert d.bit_length == 8

    def test_dtype_array_creation(self):
        d = Dtype('[u8; 4]')
        assert isinstance(d, DtypeArray)
        assert d.kind == DtypeKind.UINT
        assert d.items == 4
        assert d.bit_length == 32

    def test_dtype_tuple_creation(self):
        d = Dtype('(u8, u16)')
        assert isinstance(d, DtypeTuple)
        assert d.items == 2
        assert d.bit_length == 24

    def test_dtype_single_pack_unpack(self):
        d = Dtype('u8')
        packed = d.pack(255)
        assert packed == Bits('0xff')
        unpacked = d.unpack(packed)
        assert unpacked == 255

    def test_dtype_array_pack_unpack(self):
        d = Dtype('[u8; 4]')
        packed = d.pack([1, 2, 3, 4])
        assert packed == Bits('0x01020304')
        unpacked = d.unpack(packed)
        assert unpacked == (1, 2, 3, 4)

    def test_dtype_tuple_pack_unpack(self):
        d = Dtype('(u8, u16)')
        packed = d.pack([1, 258])
        assert packed == Bits('0x010102')
        unpacked = d.unpack(packed)
        assert unpacked == (1, 258)



# class TestCreatingNewDtypes:
#     def test_new_type(self):
#
#         md = DtypeDefinition("uintr", "A new type", Bits._set_u, Bits._get_u)
#         Register().add_dtype(md)
#         a = Bits("0xf")
#         assert a.uintr == 15
#         a = Bits.from_dtype("uintr4", 1)
#         assert a == "0x1"
#         a += "uintr100=0"
#         assert a == "0x1, 0b" + "0" * 100
#
#     def test_new_type_with_getter(self):
#         def get_fn(bs):
#             return bs.count(1)
#
#         md = DtypeDefinition("counter", "Some sort of counter", None, get_fn)
#         Register().add_dtype(md)
#         a = Bits.from_string("0x010f")
#         assert a.counter == 5
#         with pytest.raises(AttributeError):
#             a.counter = 4

    def test_invalid_dtypes(self):
        with pytest.raises(TypeError):
            _ = Dtype()
        with pytest.raises(ValueError):
            _ = Dtype.from_string("float17")
        with pytest.raises(ValueError):
            _ = Dtype("[u8]")
        with pytest.raises(ValueError):
            _ = Dtype("u8, i8")


def test_len():
    a = Dtype("bytes2")
    assert isinstance(a, DtypeSingle)
    assert a.size == 2
    assert a.bit_length == 16
    a = Dtype("[bytes1; 2]")
    assert isinstance(a, DtypeArray)
    assert a.size == 1
    assert a.items == 2
    assert a.bit_length == 16
    a = DtypeSingle("u8")
    assert a.size == 8
    assert a.bit_length == 8
    a = DtypeSingle("bits8")
    assert a.size == 8
    assert a.bit_length == 8
    a = DtypeSingle("bool")
    assert a.size == 1
    assert a.bit_length == 1
    a = DtypeSingle("bytes4")
    assert a.size == 4
    assert a.bit_length == 32
    a = DtypeSingle("f")
    assert a.size == Expression('{None}')
    assert a.size.is_none()
    assert a.bit_length is None
    a = DtypeArray("[u8; 3]")
    assert a.size == 8
    assert a.bit_length == 24
    assert a.items == 3
    a = DtypeArray("[bytes3;]")
    assert a.size == 3
    assert a.bit_length is None
    assert a.items is None
    a = DtypeArray("[bytes3; 4]")
    assert a.size == 3
    assert a.bit_length == 96
    assert a.items == 4
    a = Dtype("(u8, f16, bool)")
    assert isinstance(a, DtypeTuple)
    assert a.bit_length == 25
    assert a.items == 3
    a = DtypeTuple("([u8; 10], [bool; 0], i20)")
    assert a.bit_length == 100
    assert a.items == 3


def test_len_errors():
    for x in ["u", "[u8;]"]:
        d = Dtype(x)
        with pytest.raises(TypeError):
            _ = len(d)


def test_endianness():
    d_le = DtypeSingle.from_params(DtypeKind.UINT, 16, endianness=Endianness.LITTLE)
    d_be = DtypeSingle.from_params(DtypeKind.UINT, 16, endianness=Endianness.BIG)
    d_ne = DtypeSingle.from_params(DtypeKind.UINT, 16, endianness=Endianness.NATIVE)

    be = d_be.pack(0x1234)
    le = d_le.pack(0x1234)

    assert be.unpack(d_be) == 0x1234
    assert le.unpack(d_le) == 0x1234

    assert be == "0x1234"
    assert le == "0x3412"


def test_endianness_type_str():
    d_le = DtypeSingle.from_params(DtypeKind.UINT, 16, endianness=Endianness.LITTLE)
    d_be = DtypeSingle.from_params(DtypeKind.UINT, 16, endianness=Endianness.BIG)
    d_ne = DtypeSingle.from_params(DtypeKind.UINT, 16, endianness=Endianness.NATIVE)

    d_le2 = Dtype("u16_le")
    d_be2 = Dtype("u16_be")
    d_ne2 = Dtype("u16_ne")

    assert d_le == d_le2
    assert d_be == d_be2
    assert d_ne == d_ne2


def test_endianness_errors():
    with pytest.raises(ValueError):
        _ = DtypeSingle.from_params(DtypeKind.UINT, 15, endianness=Endianness.BIG)
    with pytest.raises(ValueError):
        _ = DtypeSingle.from_params(DtypeKind.BOOL, endianness=Endianness.LITTLE)
    with pytest.raises(ValueError):
        _ = DtypeSingle.from_params(DtypeKind.BYTES, 16, endianness=Endianness.LITTLE)


def test_dtype_tuple_packing():
    d = DtypeTuple("(bool, u8, f16)")
    a = d.pack([1, 254, 0.5])
    assert a == "0b1, 0xfe, 0x3800"
    with pytest.raises(ValueError):
        _ = d.pack([0, 0, 0, 0])
    with pytest.raises(ValueError):
        _ = d.pack([0, 0])


def test_dtype_tuple_unpacking():
    d = Dtype("(bool, u8, f16)")
    a = d.unpack("0b1, 0xfe, 0x3800")
    assert a == (1, 254, 0.5)


def test_dtype_tuple_unpacking_with_pad():
    s = Bits.from_string("0b111000111")
    d = Dtype(" ( bits3 , pad3 , bits3 , ) ")
    x, y = d.unpack(s)
    assert (x, y.unpack("u")) == ("0b111", 7)


def test_dtype_tuple_slicing():
    d = DtypeTuple("(u1, u2, u3, u4, u5)")
    d2 = d[1:4]
    assert d2 == DtypeTuple("(u2, u3, u4)")


def test_dtype_str_with_le():
    d = Dtype("u16_le")
    assert str(d) == "u16_le"
    d = Dtype.from_string("f16_be")
    assert str(d) == "f16_be"
    d = Dtype("i16_ne")
    assert str(d) == "i16_ne"
    assert repr(d) == "DtypeSingle('i16_ne')"


def test_hashing():
    a = Dtype('u8')
    b = Dtype('u8_be')
    c = Dtype('u8_le')
    d = Dtype('u8_ne')
    e = Dtype('i8')
    f = Dtype('[u8; 1]')
    g = Dtype('[u8; 2]')
    h = Dtype('(u8)')
    i = Dtype('(u8, u8)')
    s = {a, b, c, d, e, f, g, h, i}
    assert len(s) == 9

def test_str():
    a = Dtype('u8_le')
    b = Dtype('(bool, [i5; 1])')
    assert str(a) == 'u8_le'
    assert str(b) == '(bool, [i5; 1])'
    assert repr(a) == "DtypeSingle('u8_le')"
    assert repr(b) == "DtypeTuple('(bool, [i5; 1])')"
    # nt = DtypeDefinition(DtypeKind("u"), "A new type", "new", Bits._set_u, Bits._get_u)
    # s = "DtypeDefinition(kind='u', description='A new type', short_description='new', return_type=Any, is_signed=False, allowed_lengths=(), bits_per_character=None)"
    # assert str(nt) == s
    # assert repr(nt) == s

def test_unpacking_dtype_array_with_no_length():
    d = Dtype('[bool;]')
    assert str(d) == '[bool;]'
    assert d.unpack('0b110') == (True, True, False)
    assert Dtype('[u8;]').unpack('0x0001f') == (0, 1)

# def test_unpacking_dtypetuple_array_with_no_length():
#     # We shouldn't even be able to create the dtypetuple with no length array
#     with pytest.raises(ValueError):
#         _ = DtypeTuple('([bool;], u8)')
#     with pytest.raises(ValueError):
#         _ = DtypeTuple('([u8;],)')

def test_creating_dtype_with_no_size():
    d = Dtype('f')
    with pytest.raises(ValueError):
        _ = d.pack(5.0)
    b = Bits.from_dtype('f32', 12.5)
    assert b.unpack(d) == 12.5
    with pytest.raises(ValueError):
        _ = Dtype('[u;4]')
    with pytest.raises(ValueError):
        _ = Dtype('[i;]')

def test_from_string_methods():
    a = Dtype.from_string('u16_le')
    b = Dtype.from_string('[u8; 2]')
    c = Dtype.from_string('(bool, u15)')
    ap = DtypeSingle('u16_le')
    bp = DtypeArray('[u8; 2]')
    cp = DtypeTuple('(bool, u15)')
    assert isinstance(a, DtypeSingle)
    assert isinstance(b, DtypeArray)
    assert isinstance(c, DtypeTuple)
    assert isinstance(ap, DtypeSingle)
    assert isinstance(bp, DtypeArray)
    assert isinstance(cp, DtypeTuple)
    assert a == ap
    assert b == bp
    assert c == cp

def test_expression_dtype_single():
    a = Dtype('u{x}')
    assert isinstance(a, DtypeSingle)
    assert a.kind is DtypeKind.UINT
    assert str(a) == 'u{x}'

def test_dtype_array_from_str():
    a = Dtype('[u8; 2]')
    assert isinstance(a, DtypeArray)
    assert a.size == 8
    assert a.items == 2
    assert a.bit_length == 16
    b = Dtype('[f{x}; {y}]')
    assert isinstance(b, DtypeArray)
    assert b.size == Expression('{x}')
    assert b.items == Expression('{y}')
    assert b.bit_length is None

def test_dtype_str():
    try:
        bitformat.Options().no_color = False
        a = Dtype('u8')
        assert str(a) == 'u8'
        b = Dtype('[u8; 3]')
        assert str(b) == '[u8; 3]'
        c = Dtype('(bool, u8)')
        assert str(c) == '(bool, u8)'
    finally:
        bitformat.Options().no_color = True

def test_evaluate():
    concrete = Dtype('u32')
    e1 = Dtype('u{my_size}')
    e2 = Dtype('[u8; {my_items}]')

    assert e1.evaluate(my_size=32) == concrete
    assert e2.evaluate(my_items=10).bit_length == 80

def test_unpack_dtype_tuple():
    s = Bits('0x100')
    with pytest.raises(ExpressionError):
        _ = s.unpack('u{x}')
    x = s.unpack('u')
    y = s.unpack('(u)')
    assert y[0] == x

def test_unpack_dtype_tuple_with_single_dynamic_type():
    d = Dtype('(u8, [u8;], u8)')
    s = Bits.from_string('0x010203040506')
    x = d.unpack(s)
    assert x == (1, (2, 3, 4, 5), 6)
    x = s.unpack(d)
    assert x == (1, (2, 3, 4, 5), 6)
    x = s.unpack('(u8, u, u8)')
    assert x == (1, 0x02030405, 6)

def test_unpack_dtype_array_with_no_length():
    d = Dtype('[u{x};]')
    b = Bits('0x1234')
    with pytest.raises(ValueError):
        _ = d.unpack(b)

def test_dtype_single_endianness():
    d_le = Dtype("u16_le")
    d_be = Dtype("u16_be")
    val = 0x1234
    packed_le = d_le.pack(val)
    packed_be = d_be.pack(val)
    assert packed_le == Bits("0x3412")
    assert packed_be == Bits("0x1234")
    assert d_le.unpack(packed_le) == val
    assert d_be.unpack(packed_be) == val

def test_dtype_single_invalid_endianness():
    with pytest.raises(ValueError):
        Dtype("u7_le") # Endianness only for whole bytes
    with pytest.raises(ValueError):
        Dtype("bytes2_le") # Bytes type does not support endianness

def test_dtype_single_pack_invalid_value():
    d = Dtype("u8")
    with pytest.raises(ValueError):
        d.pack(256) # Out of range
    with pytest.raises(ValueError):
        d.pack("abc") # Invalid type
    d_f16 = Dtype("f16")
    with pytest.raises(ValueError): # f16 pack expects a float
        d_f16.pack("not a float")

def test_dtype_single_unpack_invalid_length():
    d = Dtype("u16")
    with pytest.raises(ValueError):
        d.unpack(Bits("0x12")) # Too short

def test_dtype_single_dynamic_size_unpack():
    d = Dtype("u") # Unsigned int, dynamic size
    assert d.unpack(Bits("0b1")) == 1
    assert d.unpack(Bits("0xffff")) == 0xffff
    d_bytes = Dtype("bytes")
    assert d_bytes.unpack(Bits("0x010203")) == b"\x01\x02\x03"

def test_dtype_single_evaluate_with_expression():
    d_expr = Dtype("u{size_val}")
    d_concrete = d_expr.evaluate(size_val=16)
    assert isinstance(d_concrete, DtypeSingle)
    assert d_concrete.bit_length == 16
    assert d_concrete.kind == DtypeKind.UINT
    packed = d_concrete.pack(100)
    assert packed == Bits.from_dtype('u16', 100)
    assert d_concrete.unpack(packed) == 100

def test_dtype_single_info():
    d = Dtype("f32_be")
    info_str = d.info()
    assert "32 bit" in info_str
    assert "float" in info_str
    assert "big-endian" in info_str # Based on current DtypeSingle.info() for f32_be
    d_bytes = Dtype("bytes5")
    info_str_bytes = d_bytes.info()
    assert "40 bit (5 characters)" in info_str_bytes
    assert "bytes" in info_str_bytes

def test_dtype_array_endianness():
    d_le = Dtype("[u16_le; 2]")
    d_be = Dtype("[u16_be; 2]")
    val = [0x1234, 0x5678]
    packed_le = d_le.pack(val)
    packed_be = d_be.pack(val)
    assert packed_le == Bits("0x34127856")
    assert packed_be == Bits("0x12345678")
    assert d_le.unpack(packed_le) == tuple(val)
    assert d_be.unpack(packed_be) == tuple(val)

def test_dtype_array_pack_invalid_value():
    d = Dtype("[u8; 2]")
    with pytest.raises(ValueError):
        d.pack([255, 256]) # One item out of range
    with pytest.raises(ValueError):
        d.pack([1, 2, 3]) # Wrong number of items
    with pytest.raises(ValueError):
        d.pack("abc") # Invalid type, expects a sequence

def test_dtype_array_unpack_invalid_length():
    d = Dtype("[u8; 4]")
    with pytest.raises(ValueError):
        d.unpack(Bits("0x010203")) # Too short, needs 32 bits, got 24

def test_dtype_array_dynamic_items():
    d = Dtype("[u8;]") # Dynamic number of items
    assert isinstance(d, DtypeArray) # Added for clarity, though .items access is next
    assert d.items is None
    val = [1, 2, 3, 4]
    # Packing dynamic items array is tricky as pack expects a fixed number of items if not Bits
    # Let's pack manually then unpack
    packed = Bits.from_joined(Dtype("u8").pack(v) for v in val)
    assert packed == Bits("0x01020304")
    unpacked = d.unpack(packed)
    assert unpacked == tuple(val)
    assert d.unpack(Bits("0xfffe")) == (0xff, 0xfe)

def test_dtype_array_evaluate_with_expression():
    d_expr = Dtype("[u{size_val}; {num_items}]")
    d_concrete = d_expr.evaluate(size_val=8, num_items=3)
    assert isinstance(d_concrete, DtypeArray)
    assert d_concrete.bit_length == 24
    assert d_concrete.items == 3
    assert d_concrete.size == 8
    assert d_concrete.kind == DtypeKind.UINT
    val = [10, 20, 30]
    packed = d_concrete.pack(val)
    assert packed == Bits("0x0a141e")
    assert d_concrete.unpack(packed) == tuple(val)

def test_dtype_array_info():
    d = Dtype("[f16_le; 3]")
    info_str = d.info()
    assert "array" in info_str
    assert "16 bit" in info_str # DtypeSingle part
    assert "float" in info_str
    assert "little-endian" in info_str # DtypeSingle part
    assert "3 items" in info_str

def test_dtype_tuple_various_configs():
    # Test different configurations for DtypeTuple
    test_cases = [
        ("(u4, bool)", (0xf, True), 5, 2),
        ("(i8, f16, u1)", (-10, 2.5, 0), 8 + 16 + 1, 3),
        ("([u4;2], bytes1)", ([1,2], b"z"), 8+8, 2)
    ]
    for dtype_str, val, blen, num_items in test_cases:
        d = Dtype(dtype_str)
        assert isinstance(d, DtypeTuple)
        assert d.bit_length == blen
        assert d.items == num_items
        packed = d.pack(val)
        unpacked = d.unpack(packed)
        # Need to handle comparison for nested structures like arrays within tuples
        if isinstance(val[0], list):
            assert unpacked[0] == tuple(val[0])
            assert unpacked[1] == val[1]
        else:
            assert unpacked == val

def test_dtype_tuple_nested():
    d = Dtype("(u8, (bool, i4), [u2;2])")
    val = (10, (True, -3), [1, 2])
    assert isinstance(d, DtypeTuple) # Added for clarity
    assert d.bit_length == 8 + (1 + 4) + (2*2)
    assert d.items == 3
    packed = d.pack(val)
    unpacked = d.unpack(packed)
    assert unpacked[0] == val[0]
    assert unpacked[1] == val[1] # Inner tuple
    assert unpacked[2] == tuple(val[2]) # Inner array becomes tuple

def test_dtype_tuple_pack_invalid_value():
    d = Dtype("(u8, bool)")
    with pytest.raises(ValueError):
        d.pack((256, True)) # First item out of range
    with pytest.raises(ValueError):
        d.pack((10, True, False)) # Wrong number of items
    with pytest.raises(ValueError):
        d.pack("abc") # Invalid type, expects a sequence

def test_dtype_tuple_unpack_invalid_length():
    d = Dtype("(u8, u8)")
    with pytest.raises(ValueError):
        d.unpack(Bits("0x01")) # Too short, needs 16 bits, got 8

def test_dtype_tuple_with_dynamic_element():
    # One dynamic element (array with no fixed items)
    d1 = Dtype("(u8, [bool;], u8)")
    val1 = (0xaa, [True, False, True], 0xbb)
    # Pack each part and join, as direct pack might be tricky with dynamic middle
    p1 = Dtype("u8").pack(val1[0])
    p2 = Dtype("[bool;]").pack(val1[1]) # This pack works for array
    p3 = Dtype("u8").pack(val1[2])
    packed1 = Bits.from_joined([p1, p2, p3])
    unpacked1 = d1.unpack(packed1)
    assert unpacked1[0] == val1[0]
    assert unpacked1[1] == tuple(val1[1])
    assert unpacked1[2] == val1[2]

    # One dynamic element (single dtype with no fixed size)
    d2 = Dtype("(u8, bytes, u8)")
    val2 = (0xcc, b"Hello", 0xdd)
    p1_2 = Dtype("u8").pack(val2[0])
    p2_2 = Dtype("bytes").pack(val2[1])
    p3_2 = Dtype("u8").pack(val2[2])
    packed2 = Bits.from_joined([p1_2, p2_2, p3_2])
    unpacked2 = d2.unpack(packed2)
    assert unpacked2[0] == val2[0]
    assert unpacked2[1] == val2[1]
    assert unpacked2[2] == val2[2]

def test_dtype_tuple_multiple_dynamic_elements_error():
    with pytest.raises(ValueError):
        Dtype("([u8;], [bool;])") # Two arrays with dynamic items
    with pytest.raises(ValueError):
        Dtype("(u, i)") # Two single dtypes with dynamic sizes
    with pytest.raises(ValueError):
        Dtype("(u8, [u4;], u, u8)") # Array and single dynamic

def test_dtype_tuple_evaluate_with_expression():
    d_expr = Dtype("(u{size_a}, [i8; {num_b}])")
    d_concrete = d_expr.evaluate(size_a=4, num_b=2)
    assert isinstance(d_concrete, DtypeTuple)
    assert d_concrete.bit_length == 4 + (8*2)
    assert d_concrete.items == 2
    val = (0b1010, [-10, 20])
    packed = d_concrete.pack(val)
    unpacked = d_concrete.unpack(packed)
    assert unpacked[0] == val[0]
    assert unpacked[1] == tuple(val[1])

def test_dtype_tuple_info():
    d = Dtype("(u16_le, [bool;2])")
    info_str = d.info()
    assert "tuple of" in info_str
    assert "16 bit little-endian unsigned int" in info_str
    assert "array of 1 bit bools with 2 items" in info_str

def test_dtype_general_equality_and_hash():
    d1 = Dtype("u8")
    d2 = Dtype("u8")
    d3 = Dtype("i8")
    d4 = Dtype("[u8;2]")
    d5 = Dtype("[u8;2]")
    d6 = Dtype("[u8;3]")
    d7 = Dtype("(u8, bool)")
    d8 = Dtype("(u8, bool)")
    d9 = Dtype("(u8, i8)")

    assert d1 == d2
    assert d1 != d3
    assert hash(d1) == hash(d2)
    assert hash(d1) != hash(d3) # Usually true

    assert d4 == d5
    assert d4 != d6
    assert hash(d4) == hash(d5)

    assert d7 == d8
    assert d7 != d9
    assert hash(d7) == hash(d8)

    assert d1 != d4
    assert d1 != d7
    assert d4 != d7
