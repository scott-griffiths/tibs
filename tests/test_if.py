import pytest
from bitformat import Format, If, Bits


def test_creation():
    i = If.from_params("{1 > 0}", "u2", "i2")
    assert i.bit_length == 2
    v = i.parse("0b11")
    assert v == 2
    assert i.value == 3

    assert str(i) == "if {1 > 0}:\n    u2 = 3\nelse:\n    i2"


def test_from_string():
    i = If.from_string("if {1 > 0}: u2 else: i2")
    assert i.bit_length == 2
    assert str(i) == "if {1 > 0}:\n    u2\nelse:\n    i2"

    j = If("if {x < 5}: bool")
    with pytest.raises(ValueError):
        _ = j.bit_length


def test_simple_parse():
    f = Format.from_params(("x: u8", If.from_params("{x > 50}", "u8")))
    b = f.parse("0xabfe")
    assert b == 16
    assert f[0].value == 0xAB
    assert f[1].value == 0xFE
    b = f.parse("0x0044")
    assert b == 8
    assert f[0].value == 0


def test_explicit_pass():
    f = If.from_params("{x > 0}", "pass", "bool = True")
    f.parse(x=2)
    assert f.bit_length == 0
    f.parse("0b1", x=-1)
    assert f.bit_length == 1
    assert f.value is True
    f.parse(x=4)
    assert f.bit_length == 0

def test_if_str():
    f = Format("(if {x}: bool else: if {y}: bool else: if {z}: bool)")
    s = str(f)
    assert s == """(
    if {x}:
        bool
    else:
        if {y}:
            bool
        else:
            if {z}:
                bool
)"""

def test_slightly_more_complex_things():
    f = Format("""my_format: (
        header: hex2 = 0x47,
        flag: bool,
        if {flag}:
            data: [u8; 6]
        else:
            data: bool
        f32
    )""")
    g = Format.from_string(str(f))
    assert f == g
    h = Format.from_params(f, f.name)
    assert f == h
    i = eval(repr(f))
    assert f == i
    b = Bits("0x47, 0b1, 0x050403020100, f32=6.5")
    f.parse(b)
    # assert f.fields[-1].value == 6.5
    # assert f['flag'].value is True
    # assert f['data'].value == [5, 4, 3, 2, 1, 0]
    # assert f['data'].dtype == '[u8;6]'
    #
    # print(f)
    #
    # b = f.pack(['47', True, [5, 4, 3, 2, 1, 0]])
    # assert b == '0x47, 0b1, 0x050403020100'
    # b2 = f.pack(['47', False, [5, 4, 3, 2, 1, 0]])
    # assert b2 == '0x47, 0b0, 0x050403020100'


def test_eq():
    i = If.from_params("{1 > 0}", "u2", "i2")
    assert i == If.from_params("{1 > 0}", "u2", "i2")
    assert i != If.from_params("{1 > 0}", "u2", "i3")
    assert i != If.from_params("{2 > 0}", "u2", "i2")

# def test_expressions():
#     f = Format('(x:bool, if {x is True}: pass else: f32)')
#     f.pack([True])
#     assert f.to_bits() == '0b1'