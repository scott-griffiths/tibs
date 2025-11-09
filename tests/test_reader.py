import pytest
from bitformat import Reader, Field, Bits, Dtype, DtypeTuple, MutableBits


def test_creation():
    r = Reader(Bits())
    assert len(r.bits) == 0
    assert r.pos == 0

    r = Reader(Bits.from_string("0x12345"), 4)
    assert len(r.bits) == 20
    assert r.pos == 4

    r = Reader(Bits.from_bytes(b"hello"))
    assert len(r.bits) == 40

    with pytest.raises(TypeError):
        _ = Reader('0x234')
    with pytest.raises(TypeError):
        _ = Reader(b'hello')

def test_read():
    r = Reader(Bits("0x12345"), 4)
    assert r.pos == 4
    assert r.read("bits4") == "0x2"
    assert r.pos == 8
    assert r.read("u4") == 3
    assert r.pos == 12
    r.pos = 0
    b = r.read(8)
    assert b == '0x12'

def test_peek():
    r = Reader(Bits("0xff000001"))
    assert r.peek('hex2') == 'ff'
    assert r.peek(Dtype('hex2')) == 'ff'
    assert r.pos == 0
    r.pos = len(r) - 1
    assert r.peek(1) == '0b1'
    with pytest.raises(ValueError):
        _ = r.peek(2)
    assert r.peek(1) == '0b1'

def test_parse():
    r = Reader(Bits.from_bytes(b"hello_world"))
    r.pos = 6 * 8
    f = Field("bytes3")
    assert r.parse(f) == 24
    assert f.value == b"wor"
    r.parse(g := Field("bool"))
    assert g.value is False
    with pytest.raises(TypeError):
        _ = r.parse('bin2')

def test_read_tuple():
    r = Reader(Bits('0x00ffff'))
    x = r.peek(DtypeTuple('(hex3, u4)'))
    assert x == ('00f', 15)
    x = r.read('(u8, bool, bool)')
    assert x == (0, True, True)

def test_with_mutable_bits():
    mb = MutableBits('0x00ff')
    r = Reader(mb)
    assert r.pos == 0
    assert r.read('u9') == 1
    assert r.pos == 9
    mb.prepend('0b111111111')
    assert r.pos == 9
    assert r.read('u9') == 1
    x = r.bits
    assert isinstance(x, MutableBits)
    assert x is mb

def test_open_ended_read():
    a = Bits.from_zeros(100)
    r = Reader(a)
    r.pos = 1
    x = r.peek('[u11;]')
    assert x == (0, 0, 0, 0, 0, 0, 0, 0, 0)
    x = r.read('[u11;]')
    assert x == (0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert r.pos == 100