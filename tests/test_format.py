#!/usr/bin/env python

from bitformat import Format, Dtype, DtypeSingle, Bits, Field, Array, FieldType, Repeat, DtypeKind, Expression, While, \
    MutableBits
from hypothesis import given
import pytest
import hypothesis.strategies as st
import random


class TestCreation:
    def test_create_empty(self):
        f = Format()
        f.pack([])
        b = f.to_bits()
        assert len(b) == 0
        assert f.name == ""
        assert len(f) == 0
        g = Format('()')
        assert f == g
        assert len(g) == 0
        assert str(f) == "(\n)"
        assert str(g) == "(\n)"

    def test_create_from_dtype(self):
        d = Dtype.from_string("u12")
        f = Format.from_params([Field.from_params(d)], "x")
        f.pack([1000])
        x = f.to_bits()
        assert f.name == "x"
        assert x == "u12=1000"
        assert len(x) == 12
        assert len(f) == 1

    def test_create_from_bits_string(self):
        f = Format.from_params([Field.from_params("f16", "foo", 12.5)], "x")
        g = Format.from_params(["foo: f16=12.5"], "y")
        assert f.to_bits() == g.to_bits()
        assert f.name == "x"

    def test_create_from_dtype_string(self):
        f1 = Format("(x: f16)")  # No comma
        f2 = Format("(x: f16,)") # With comma
        assert f1 == f2
        assert f1[0].name == "x"
        assert f1[0].dtype == DtypeSingle.from_params(DtypeKind.FLOAT, 16)

    @given(name=st.sampled_from(["f16", "u12", "bool", "f64"]))
    def test_building_field(self, name):
        f = Field(name)
        f.pack(0)
        b = f.to_bits()
        assert b == Bits.from_string(f"{name}=0")

    def test_create_from_bits(self):
        b = Bits.from_string("0xabc")
        field = Field.from_bits(b)
        f = Format.from_params([field])
        x = f.to_bits()
        assert f.name == ""
        assert x == "0xabc"
        assert isinstance(x, Bits)

    def test_create_from_bits_with_name(self):
        f = Format.from_params([Field.from_bits("0xabc", "some_bits")])
        assert f.to_bits() == "0xabc"

    def test_create_from_list(self):
        f = Format.from_params(["const bits = 0xabc", "u5", "u5"])
        f.pack([3, 10])
        x = f.to_bits()
        assert x == "bits = 0xabc, u5=3, u5=10"
        f.parse(x)
        assert isinstance(f, Format)

    def testComplicatedCreation(self):
        a = Field('const bits = 0x1')
        assert a.bit_length == 4
        f = Format.from_params(
            (
                "const bits = 0x000001b3",
                "u12",
                "height:const u12  = 288",
                "flag: const bool  =True",
            ),
            "header",
        )
        g = Format(
            "header: (const bits = 0x000001b3, u12, height: const u12 = 288, flag: const bool = True)"
        )
        assert f == g
        assert f.name == "header"
        f.pack([352])
        assert f.to_bits() == "0x000001b3, u12=352, u12=288, 0b1"
        _ = Format.from_params([f, "bytes5"], "main")

    def test_stretchy_token_at_start(self):
        with pytest.raises(ValueError):
            _ = Format('(hex, u8)')

    def test_nested_formats1(self):
        f1 = Format('(u8)')
        f2 = Format('(u16)')
        f3 = Format.from_params(['u4', f1, f2, 'u32'])
        assert f3.value == [None, [None], [None], None]
        f3.pack([1, [2], [3], 4])
        assert f3.value == [1, [2], [3], 4]
        assert f3.to_bits() == 'u4=1, u8=2, u16=3, u32=4'
        b = f3.to_bits()
        f3.clear()
        assert f3.value == [None, [None], [None], None]
        f3.parse(b)
        assert f3.value == [1, [2], [3], 4]

    def test_nested_formats2(self):
        header = Format.from_params(
            ["const bits = 0x000001b3", "width:u12", "height:u12", "f1:bool", "f2:bool"],
            "header",
        )
        main = Format.from_params(["const bits = 0b1", "v1:i7", "v2:i9"], "main")
        m = Bits("0b1, i7=5, i9=-99, 0x47")
        main.parse(m)
        f = Format.from_params([header, main, "const hex = 0x47"], "all")
        b = Bits("0x000001b3, u12=100, u12=200, 0b1, 0b0, 0b1, i7=5, i9=-99, 0x47, 0x00000000000000")
        f.parse(b)
        print(f)
        t = f["header"]
        assert t["width"].value == 100
        assert f["header"]["width"].value == 100
        assert f["main"]["v2"].value == -99


def test_building():
    f1 = Field("u8")
    f1.pack(9)
    assert f1.value == 9

    f2 = Field("[u8; 3]")
    f2.pack([1, 2, 3])
    assert f2.value == (1, 2, 3)

    f3 = Format.from_params(["u8"])
    f3.pack([4])
    assert f3.value == [4]

    f4 = Format.from_params(["[u8; 3]"])
    f4.pack([[1, 2, 3]])
    assert f4.value == [(1, 2, 3)]

    f5 = Format.from_params(["u8", "[u8; 3]"])
    f5.pack([4, [1, 2, 3]])
    assert f5.value == [4, (1, 2, 3)]

    f6 = Format.from_params(["u8", "[u8; 3]", "u8"])
    f6.pack([4, [1, 2, 3], 5])
    assert f6.value == [4, (1, 2, 3), 5]

    f7 = Format.from_params([f1, f2])
    f7.pack([6, [4, 5, 6]])
    assert f7.value == [6, (4, 5, 6)]

    f8 = Format.from_params([f1, f7])
    f8.pack([7, [8, (9, 10, 11)]])
    assert f8.value == [7, [8, (9, 10, 11)]]


def test_packing_bug():
    f = Format("bug: (u8, (u8, u8))")
    f.pack([10, [20, 30]])
    assert f.value == [10, [20, 30]]


class TestAddition:
    def test_adding_bits(self):
        f = Format()
        f += Field.from_bits("0xff")
        assert f.to_bytes() == b"\xff"
        assert f[0].to_bytes() == b"\xff"
        assert len(f) == 1
        f += Field.from_string("penguin:i9 =-8")
        assert len(f) == 2
        x = f["penguin"]
        assert x.value == -8
        f["penguin"].value += 6
        assert f["penguin"].value == -2


class TestArray:
    def test_simple_array(self):
        array_field = Field.from_params("[u8; 20]", "my_array")
        f = Format.from_params([array_field], "a")
        assert f[0].dtype.items == 20
        f.pack([list(range(20))])

        f2 = Format.from_params(["new_array: [u8;20]"], "b")
        assert f2[0].dtype.items == 20
        assert f2[0].value is None
        with pytest.raises(ValueError):
            f2["new_array"].value = f.to_bits()[2:]
        f2["new_array"].value = f.to_bits()
        assert f.to_bits() == f2.to_bits()

    @given(w=st.integers(1, 5), h=st.integers(1, 5))
    def test_example_with_array(self, w, h):
        f = Format.from_params(
            [
                "signature: const bytes = b'BMP'",
                "width: i8",
                "height: i8",
                "pixels: [u8 ; {width * height}]",
            ],
            "construct_example",
        )
        p = tuple(random.randint(0, 255) for _ in range(w * h))
        f.pack([w, h, p])
        b = f.to_bits()
        f.clear()
        f.parse(b)
        assert f["width"].value == w
        assert f["height"].value == h
        assert f["pixels"].value == p


def test_example_from_docs():
    f = Format("_:(x: u8, y: u{x}, bool)")
    b = Bits.from_string("u8=10, u10=987, bool=1")
    assert f.parse(b) == 19  # Number of parsed bits
    assert f["y"].value == 987

    # f = Format("(sync_byte: const hex2 = 0xff,"
    #            "items: u16,"
    #            "flags: [bool ; {items + 1} ],"
    #            "repeat {items + 1}: "
    #            "    (byte_cluster_size: u4, bytes{byte_cluster_size}),"
    #            "u8)")
    # f.pack([1, [True, False], [[1, b'a'], [2, b'qz']], 255])


def test_items():
    f = Format.from_params(["q:i5", "[u3; {q + 1}]"])
    b = Bits.from_string("i5=1, u3=2, u3=0")
    f.parse(b)
    assert f[0].value == 1
    assert f[1].value == (2, 0)
    f.clear()
    f.pack([1, [2, 0]])
    assert b == f.to_bits()
    f.clear()
    f.pack([3, [1, 2, 3, 4]])
    assert f.to_bits() == Bits.from_string("i5=3, u3=1, u3=2, u3=3, u3=4")


class TestMethods:
    def test_clear(self):
        f = Format.from_params(
            ["const bits = 0x000001b3", "u12", "height:u12", "  flag : bool "], "header"
        )
        f["height"].value = 288
        f.clear()
        g = Format.from_string(
            "header: (const bits = 0x000001b3, u12, height:u12, flag:bool)"
        )
        assert f == g

    def test_get_item(self):
        f = Format.from_params(["f16=7", "bool", "bytes5", "pop :u100  = 144"])
        assert f[0].value == 7
        assert f[1].value is None
        assert f["pop"].value == 144

    def test_set_item(self):
        f = Format.from_params(["const f16=7", "bool", "bytes5", "pop : u100 = 144"])
        with pytest.raises(ValueError):
            f[0].value = 2
        f["pop"].value = 999999
        assert f["pop"].value == 999999


def test_repeating_field():
    f = Repeat.from_params(5, "u8")
    d = Array("u8", [1, 5, 9, 7, 6]).to_bits()
    f.unpack(d)
    assert f.value == [1, 5, 9, 7, 6]


def test_format_get_and_set():
    f = Format("(u8, u8, u8)")
    for field in f:
        field.value = 12
    assert f.value == [12, 12, 12]
    f[0].value = 0
    assert f.value == [0, 12, 12]
    g = Format.from_params(f[:])
    assert g.value == [0, 12, 12]
    f[-1].value = 7
    assert g[-1].value == 12


def test_repeating_from_expression():
    f = Format.from_params(["x: u8", Repeat.from_params(Expression("{2*x}"), "hex1")], "my_little_format")
    f.pack([2, ["a", "b", "c", "d"]])
    assert f.to_bits() == "u8=2, hex1=a, hex1=b, hex1=c, hex1=d"

def test_if_with_const_value():
    f = Format('(if {False}: const u8=255 else: const u8=0)')
    f.pack([])
    assert f.to_bits() == "u8=0"

def test_repeat_with_const_expression():
    f = Format("(the_size: i9, repeat {the_size}: (const u5=0, const bin3=111))")
    f.pack([3])
    assert f.to_bits() == "i9=3, 0x070707"

def test_repeat_with_bits():
    f = Repeat.from_params(3, "const bits=0xab")
    f.pack([])
    b = f.to_bits()
    assert b == "0xababab"
    f2 = Repeat.from_params(2, Field.from_bits(b, const=True))
    f2.pack([])
    b2 = f2.to_bits()
    assert b2 == "0xabababababab"

def test_repeat_with_dtype():
    f = Repeat.from_params(4, "i4")
    f.pack([1, 2, 3, 4])
    assert f.value == [1, 2, 3, 4]

    f = Repeat.from_params(2, "[i8; 2]")
    b = f.pack([[-40, 20], [-100, 4]])
    f.unpack(b)
    assert f.value == [(-40, 20), (-100, 4)]


def test_field_array_str():
    f = Field.from_string("test   :  f32_le = 0.25  ")
    assert str(f) == "test: f32_le = 0.25"
    assert f.to_bits().unpack('f_le') == 0.25
    f = Field.from_string("test: [f32_le ; 3]")
    assert str(f) == "test: [f32_le; 3]"


def test_format_repr_string():
    f = Format.from_params(["x:const u8 = 12", "u:bool = False", "[u3;44]"], "dave")
    r = repr(f)
    assert r == "Format.from_params(['x: const u8 = 12', 'u: bool = False', '[u3; 44]'], name='dave')"


def test_to_bits():
    f1 = Format.from_params(["u8_le", "u8_be", "u8_ne"])
    f = Format("(u8_le, u8_be, u8_ne)")
    assert f == f1
    f.pack([1, 2, 3])
    b = f.to_bits()
    assert b == "u8=1, u8=2, u8=3"
    f[1].clear()
    assert f[0].value == 1
    assert f[1].value is None
    assert f[2].value == 3
    assert f[0].to_bits() == "u8=1"
    with pytest.raises(ValueError):
        _ = f[1].to_bits()
    with pytest.raises(ValueError):
        _ = f.to_bits()


def test_partial_parse():
    f = Format.from_params(["bool", "[f16;3]"])
    b = Bits.from_string("0b1, f16=1.0, f16=2.0, f16=3.0")
    f.parse(b)
    assert f[0].value is True
    assert f[1].value == (1.0, 2.0, 3.0)
    f.clear()
    with pytest.raises(ValueError):
        _ = f.parse(b[:-16])


def test_from_string():
    s = "header: (u8,u4, bool)"
    f = Format.from_string(s)
    assert f.name == "header"
    assert f[0].dtype == Dtype.from_string("u8")
    assert str(f) == str(Format(str(f)))


def test_recursive_from_string():
    s = "header: (u8, u4, bool,body:(u8=23, [u4; 3], bool))"
    f = FieldType.from_string(s)
    assert f.name == "header"
    assert f[3][0].value == 23
    b = f["body"]
    assert b[0].value == 23
    assert str(f) == str(Format(str(s)))
    assert str(b) == str(Format("body: (u8=23, [u4; 3], bool)"))

    fp = eval(repr(f))
    assert fp == f


def test_interesting_types_from_string():
    s = "  (const f32= -3.75e2 , _fred : bytes4 = b'abc\x04',) "
    f = Format.from_string(s)
    assert f[0].value == -375
    assert f["_fred"].value == b"abc\x04"

def test_expression_literals():
    a = Field.from_string("u{4 + 4}")
    a.pack(255)
    assert a.to_bits() == Bits('0xff')
    b = Field.from_string("[bool; {4 + 4}]")
    b.pack([1, 1, 1, 1, 0, 0, 0, 0])
    assert b.to_bits() == Bits('0b11110000')
    c = Field.from_string("[u{5 + 5}; {4 + 4}]")
    assert c.bit_length == 80
    d = Format("(a: u{4 + 4}, b: [u{5 + 5}; {4 + 4}])")
    d.pack([255, [1, 10, 55, 4, 3, 2, 1, 0]])
    assert d.to_bits() == Bits('0xff, u10=1, u10=10, u10=55, u10=4, u10=3, u10=2, u10=1, u10=0')

def test_passed_in_value():
    a = Field.from_string("u{x}")
    a.pack(5, x=10)

def test_expression_dtypes():
    a = Field.from_string('u{testing}')
    assert str(a) == 'u{testing}'
    d = Field.from_string('my_name: [f{4*e}; {a + b}]')
    assert str(d) == 'my_name: [f{4*e}; {a + b}]'
    f = Format('(x: u8, [u{x}; {x + 1}])')
    b = Bits('u8=3, u3=1, u3=2, u3=3, u3=4')
    f.parse(b)
    assert f['x'].value == 3
    assert f[1].value == (1, 2, 3, 4)
    assert f.value == [3, (1, 2, 3, 4)]


def test_unpack():
    f = Format.from_string("header: (u8, u4, bool)")
    b = Bits.from_string("u8=1, u4=2, 0b1")
    assert f.unpack(b) == [1, 2, True]
    f[1].clear()
    assert f.unpack() == [1, None, True]


def test_construction_by_appending():
    f = Format()
    f += "u8"
    f += "i4"
    f += Field("const f16 = 0.25")
    g = Format()
    g.append("u8")
    g.append("i4")
    g += Field("const f16 = 0.25")
    h = Format()
    h.extend(["u8", "i4", "const f16=0.25"])
    i = Format() + "u8"
    i = i + "i4" + "const f16=0.25"
    assert f == g == h == i


f_str = """
sequence_header: (
    sequence_header_code: const hex8 = 0x000001b3,
    horizontal_size_value: u12,
    vertical_size_value: u12 ,
    aspect_ratio_information: u4,
    frame_rate_code: u4,
    bit_rate_value: u18,
    marker_bit: bool,
    vbv_buffer_size_value: u10,
    constrained_parameters_flag: bool,
    load_intra_quantiser_matrix: u1
)
"""


def test_example_format():
    _ = Format(f_str)


def test_format_str_equivalences():
    f1 = Format("abc : ( f16, u5, [bool; 4]) ")
    f2 = Format("abc:(f16,u5,[  bool  ;4]  )  ")
    f3 = Format("""    abc : (
    f16,
    u5,
    
    [bool;4],)
    """)
    assert f1 == f2 == f3
    assert str(f1) == str(f2) == str(f3)
    assert repr(f1) == repr(f2) == repr(f3)
    f4 = eval(repr(f1))
    assert f4 == f1


def test_stretchy_field():
    f = Format("(u8, u)")
    f.unpack("0xff1")
    assert f.value == [255, 1]

    with pytest.raises(ValueError):
        _ = Format("(u, u8)")
    g = Format("(u5, bytes)")
    g.parse(b"hello_world")
    assert g[0].value == 13
    with pytest.raises(ValueError):
        _ = g[1].value


def test_repeated_field_copy():
    i = Field("hex4 = abcd")
    f = Format.from_params([i, i])
    assert f[0].value == "abcd"
    assert f[1].value == "abcd"
    f[0].value = "0xdead"
    assert f[0].value == "dead"
    assert f[1].value == "abcd"


def test_format_copy():
    f = Format("(x: u8 = 10, y: u8 = 20)")
    g = Format.from_params([f, f])
    assert g[0].value == [10, 20]
    f[0].value = 5
    assert f[0].value == 5
    assert g[0].value == [10, 20]
    g[0][0].value = 7
    assert g[0].value == [7, 20]
    assert g[1].value == [10, 20]
    assert f[0].value == 5

s = """
header :(
    x: u8,
    y: u8
    z: u8,
    data: [u8; 3],
    repeat{2}: (
        a: u8
        b: u8
    )
    bool
)
"""


def test_format_with_repeat():
    f = Format(s)
    b = Bits.from_bytes(b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d")
    f.parse(b)
    assert f["x"].value == 1
    assert f["y"].value == 2
    assert f["z"].value == 3
    assert f["data"].value == (4, 5, 6)
    assert f[4].value == [[7, 8], [9, 10]]
    # assert f['a'].value == [7, 9]


s2 = """
x: (i5,
q: u8,
(u3, 
u4
),
u5
)
"""


def test_format_inside_format_from_string():
    test = Format("x: ((u8, u8),)")
    test.pack([[1, 2]])
    assert test.value == [[1, 2]]
    f = Format(s2)
    assert f.bit_length == 25
    assert len(f) == 4
    f.pack([1, 2, [3, 4], 5])
    assert f.value == [1, 2, [3, 4], 5]

# def test_repr_eval_with_repeat():
#     f = Format(s)
#     r = repr(f)
#     f2 = eval(r)
#     assert f == f2


def test_eq():
    f = Format("(u8, u8)")
    assert f == Format("(u8, u8)")
    assert f != Format("(u8, u8, u8)")
    assert f != Format("(u8, const u8 = 10)")
    assert f != Format("(u8, x: u8)")
    assert Format("(u8 = 3)") == Format("(u8 = 3,)")
    assert Format("(u8 = 3)") != Format("(u8 = 4)")

def test_wrong_arguments():
    f = Format("(bool, bool, (i3, q: i3), [f64; 1])")
    f.pack([True, False, [2, -2], [4.5]])
    assert f.value == [True, False, [2, -2], (4.5,)]
    f.clear()
    with pytest.raises(TypeError):
        f.pack(1)
    with pytest.raises(ValueError):
        f.pack([1])

def test_slicing_fields():
    f = Format.from_string("x: (u8, u7, u6, u5, u4, u3, u2, u1)")
    f.pack([8, 7, 6, 5, 4, 3, 2, 1])
    assert f[0].value == 8
    assert f[-1].value == 1
    assert f[:2].value == [8, 7]
    assert f[2::-1].value == [6, 7, 8]
    assert f[::2].value == [8, 6, 4, 2]
    f2 = f[::2]
    assert f2[1].dtype == Dtype("u6")
    assert f2[1].value == 6
    f2[1].value = 7
    assert f2[1].value == 7
    assert f[2].value == 6
    with pytest.raises(IndexError):
        _ = f[8]
    for field in f:
        assert field.value == field.dtype.bit_length

def test_deleting_fields():
    f = Format.from_params(["x: u8", "y: u8", "z: u8"])
    del f[1]
    assert len(f) == 2
    assert f[1].name == "z"
    del f['x']
    assert len(f) == 1
    with pytest.raises(KeyError):
        del f['x']
    assert len(f) == 1

def test_setting_fields():
    f = Format.from_params(["x: u8", "y: u8", "z: u8", "q:i4"])
    g = Format.from_params(["a: u8", "b: u8", "c: u8"])
    f[2] = Field("u15 = 5")
    assert f[2].value == 5
    f[0:2] = g[:]
    assert f[0].name == "a"
    assert len(f) == 5

def test_dtypetuple_in_format():
    h = Format('(tuple(u8, u6))')
    h.pack([[10, 5]])
    assert h[0].value == (10, 5)

def test_set_values():
    f = Format.from_params(["x: u8", "y: u8", "z: u8", "q:i4"])
    f.value = [1, 2, 3, 4]
    assert f.value == [1, 2, 3, 4]
    f.value = [5, 6, 7, -8]
    assert f.value == [5, 6, 7, -8]

def test_bad_names():
    f = Format('()')
    assert f.name == ''
    with pytest.raises(ValueError):
        f.name = 'if'
    with pytest.raises(ValueError):
        f.name = '__with_double_underscore'
    with pytest.raises(TypeError):
        f.name = 5

def test_const_tuple():
    f = Format("(const tuple(bool, u5, u10) = (1, 31, 0))")
    assert f.value == [(True, 31, 0)]

def test_open_ended_array():
    f = Format("([u8;])")
    f.pack([[1, 2, 3]])
    assert f.value == [(1, 2, 3)]
    b = f.to_bits()
    f.parse(b)
    assert f.value == [(1, 2, 3)]

def test_expressions_more():
    f = Format("(a: u8, u{a}, u{a})")
    f['a'].value = 3
    assert f.value[0] == 3

# def test_packing_format_with_const_field():
#     f = Format("(a: u8, b: u{a} = 5)")
#     f.pack([3])
#     assert f.value == [3, 5]
#     assert f.to_bits() == "u8=3, u8=5"
#     f2 = Format.from_params(["a: u8", "b: const u8 = 5"])
#     f2.pack([4])
#     assert f2.value == [4, 5]
#     assert f2.to_bits() == "u8=4, u8=5"

def test_const_variable():
    f = FieldType.from_string('(x: const u8 = 5, [u4; {x}])')
    f.parse('u8=5, u4=1, u4=2, u4=3, u4=4, u4=5')
    assert f.value == [5, (1, 2, 3, 4, 5)]

def test_format_with_simple_fields():
    """Test a format with multiple simple fields"""
    fmt = Format("test_format: (a: u8, b: u16, c: bool)")
    assert len(fmt) == 3
    assert fmt.name == "test_format"

    # Try packing with values
    fmt.pack([42, 1000, True])
    assert fmt.to_bits() == Bits.from_string("u8=42, u16=1000, 0b1")
    assert fmt.value == [42, 1000, 1]

    # Try accessing named fields
    assert fmt["a"].value == 42
    assert fmt["b"].value == 1000
    assert fmt["c"].value == 1


def test_format_with_field_expressions():
    """Test a format with fields that use expressions"""
    fmt = Format("(size: u8, data: bytes{size})")

    # Pack with size 4 and data
    fmt.pack([4, b"test"])
    assert fmt.to_bits() == Bits.from_string("0x0474657374")  # 4, "test"

    # Test unpacking
    data = Bits.from_string("0x0348656c")  # 3, "Hel"
    fmt.parse(data)
    assert fmt.value == [3, b"Hel"]
    assert fmt["size"].value == 3
    assert fmt["data"].value == b"Hel"


def test_format_with_nested_formats():
    """Test a format with nested formats"""
    header = Format("header: (version: u8=1, flags: u8=0)")
    payload = Format("payload: (length: u8, data: bytes{length})")
    full_msg = Format.from_params([header, payload], 'message')
    # TODO: The above is a bit clunky. It would be nice to write something like:
    # full_msg = Format("message: (header, payload)")
    # but there would have to be some way of tagging the header, payload as existing fieldtypes?

    # Pack with length 5 and data
    payload.pack([5, b"hello"])
    full_msg.pack([header.value, payload.value])

    # Check the whole message
    expected = Bits.from_string("0x010005" + "".join(f"{ord(c):02x}" for c in "hello"))
    assert full_msg.to_bits() == expected

    # Test direct access to nested field
    assert full_msg["header"]["version"].value == 1
    assert full_msg["payload"]["length"].value == 5
    assert full_msg["payload"]["data"].value == b"hello"


def test_format_with_repeat():
    """Test a format with a repeat field"""
    # Format with count, followed by that many u8 values
    fmt = Format("(count: u8, repeat{count}: u8)")

    # Pack with count 3 and values
    fmt.pack([3, [10, 20, 30]])
    assert fmt.to_bits() == Bits.from_string("0x030a141e")  # 3, [10, 20, 30]

    # Test unpacking
    data = Bits.from_string("0x0201020304")  # 2, [1, 2], extra bytes 0x0304 ignored
    fmt.parse(data)
    assert fmt.value[0] == 2  # count
    assert fmt.value[1] == [1, 2]  # values

def test_format_with_conditional_fields():
    """Test a format with conditional fields based on a flag"""
    fmt = Format("(has_name: bool, if {has_name}: name: bytes4, id: u16)")

    # Test with name (has_name=True)
    fmt.pack([True, b"ABCD", 1234])
    assert fmt.to_bits() == Bits('0b1') + b'ABCD' + 'u16 = 1234'

    # Test without name (has_name=False)
    fmt.pack([False, 5678])  # No name needed
    assert fmt.to_bits() == Bits.from_string('0b0, u16=5678')

def test_format_with_complex_expressions():
    """Test a format with more complex expressions"""
    fmt = Format("(width: u8, height: u8, repeat{width*height}: pixel: u8)")

    # Pack a 2x3 grid of pixels
    fmt.pack([2, 3, [10, 20, 30, 40, 50, 60]])
    assert fmt.to_bits() == Bits.from_string("0x02030a141e28323c")

    # Unpack
    data = Bits.from_string("0x0102010203")  # 1x2 grid with pixels [1, 2]
    fmt.parse(data)
    assert fmt.value == [1, 2, [1, 2]]

def test_format_with_padding():
    """Test a format with padding fields"""
    fmt = Format("(a: u4, pad4, b: u8)")

    # Pack
    fmt.pack([0xA, 0xBC])
    assert fmt.to_bits() == Bits.from_string("0xA0BC")  # 0xA with 4 bits of padding, then 0xBC

    # Unpack
    data = Bits.from_string("0xF0AB")  # 0xF with 4 bits of padding, then 0xAB
    fmt.parse(data)
    assert fmt.value == [0xF, 0xAB]


def test_format_with_dynamic_dtype():
    """Test a format with a field whose dtype depends on another field"""
    # A format where a type selector determines the type of the second field
    fmt = Format("(type: u2, if {type == 0}: value: u8 else: if {type == 1}: value: i8 else: if {type == 2}: value: f16 else: value: bool)")

    # Test with type 0: u8
    fmt.pack([0, 200])
    assert fmt.to_bits() == Bits('0b00, u8=200')

    # Test with type 1: i8
    fmt.pack([1, -50])
    assert fmt.to_bits() == Bits('0b01, i8=-50')

    # Test with type 2: f16
    fmt.pack([2, 1.5])
    assert fmt.to_bits() == Bits('0b10, f16=1.5')

    # Test with type 3: bool
    fmt.pack([3, True])
    assert fmt.to_bits() == Bits.from_string("0b111")

def test_format_with_expressions_in_if_condition():
    """Test a format with expressions in if condition that reference other fields"""
    fmt = Format("(flags: u8, len: u8, if {flags & 0x01 != 0}: opt1: u8, if {flags & 0x02 != 0}: opt2: u16)")

    # Neither option present
    fmt.pack([0, 5])
    assert len(fmt.to_bits()) == 16  # 8 + 8 bits

    # First option present
    fmt.pack([1, 5, 42])
    assert fmt.to_bits() == Bits.from_string("0x01052A")

    # Second option present
    fmt.pack([2, 5, 1000])
    assert fmt.to_bits() == Bits.from_string("0x020503E8")

    # Both options present
    fmt.pack([3, 5, 42, 1000])
    assert fmt.to_bits() == Bits.from_string("0x03052A03E8")

@pytest.mark.skip  # TODO
def test_format_with_nested_repeat():
    """Test a format with nested repeat structures"""
    # A format representing a list of lists
    fmt = Format("(row_count: u8, repeat{row_count}: row: (col_count: u8, repeat{col_count}: value: u8))")

    # Pack 2 rows, first with 3 columns, second with 2 columns
    row1 = [3, [1, 2, 3]]  # 3 columns: 1, 2, 3
    row2 = [2, [4, 5]]     # 2 columns: 4, 5
    fmt.pack([2, [row1, row2]])

    # Should be [2, [3, [1, 2, 3], 2, [4, 5]]]
    expected = Bits.from_string("0x0203010203020405")
    assert fmt.to_bits() == expected

def test_let():
    f = FieldType.from_string('let x = {2}')
    assert str(f) == 'let x = {2}'
    f = Format("(let x = {5}, u{x+y})")
    f.pack([22], y=3)
    assert f.to_bits() == 'u8=22'
    f.pack([1], y=1000)
    assert f.to_bits() == 'u1005=1'

def test_while():
    f = While("while {x > 0}: (bool, let x = {x - 1})")
    b = f.pack([[1], [0], [1]], x=3)
    assert b == '0b101'

    f.clear()
    x = f.unpack('0x0f01', x=8)
    assert x == [False, False, False, False, True, True, True, True]

def test_more_while():
    f = Format("(let y = {x}, while {x}: (byte: u8, if {byte < 12}: let x = {x-1}))")
    b = '0x01ff02ff03ff04ff'
    v = f.unpack(b, x=1)
    assert v[0] == [1]
    v = f.unpack(b, x=2)
    assert v[0] == [1, 255, 2]

def test_appending_fields():
    f = Format()
    f += "x: u8"
    assert f.parse('0x10') == 8
    x_value = f['x'].value
    assert x_value == 0x10

def test_string_value():
    f = Format("(x: const hex4 = dead, if {x == 'dead'}: u8)")
    b = MutableBits('0xdeadff')
    f.parse(b)
    assert f.value == ['dead', 255]