import pytest
from tibs import Tibs, Mutibs, BitIndexing


def test_empty():
    t1 = Tibs()
    t2 = Tibs(bit_indexing=BitIndexing.Msb0)
    t3 = Tibs(bit_indexing=BitIndexing.Lsb0)

    assert repr(t1) == repr(t2) == "Tibs()"
    assert repr(t3) == "Tibs(bit_indexing=BitIndexing.Lsb0)"
    assert t1.bit_indexing is BitIndexing.Msb0
    assert t3.bit_indexing is BitIndexing.Lsb0

    m1 = Mutibs()
    m2 = Mutibs(bit_indexing=BitIndexing.Msb0)
    m3 = Mutibs(bit_indexing=BitIndexing.Lsb0)

    assert repr(m1) == repr(m2) == "Mutibs()"
    assert repr(m3) == "Mutibs(bit_indexing=BitIndexing.Lsb0)"


def test_changing():
    t = Tibs('0b1')
    assert t.bit_indexing == BitIndexing.Msb0
    with pytest.raises(AttributeError):
        t.bit_indexing = BitIndexing.Lsb0
    m = Mutibs('0b1')
    assert m.bit_indexing == BitIndexing.Msb0
    m.bit_indexing = BitIndexing.Lsb0
    assert m.bit_indexing is BitIndexing.Lsb0
    with pytest.raises(TypeError):
        m.bit_indexing = "something_else"


def test_lsb0_bit_indexing():
    t = Tibs('0b11100', bit_indexing=BitIndexing.Lsb0)
    assert t[0] == 0
    assert t[1] == 0
    assert t[2] == 1
    assert t[-1] == 1

    m = Mutibs(bit_indexing=BitIndexing.Lsb0)
    m += '0b1100'
    assert m[0] == False
    assert m[-1] == True


def test_lsb0_slice():
    t = Tibs('0x00ff0', bit_indexing=BitIndexing.Lsb0)
    assert t[4:12] == '0xff'
    assert t[-16:-8] == '0xff'
    assert t[4:] == '0x00ff'

    m = Mutibs.from_oct('775')
    m.bit_indexing = BitIndexing.Lsb0
    assert m[:2] == '0b01'
    assert m[-6:] == '0o77'


def test_lsb0_slice_with_step():
    t = Tibs('0b10101010', BitIndexing.Lsb0)
    assert t[::2] == '0b0000'

    m = Mutibs('0b100100100100')
    m.bit_indexing = BitIndexing.Lsb0
    assert m[2::3] == '0b1111'


def test_lsb0_equality():
    t1 = Tibs.from_bin('1010', BitIndexing.Msb0)
    t2 = Tibs.from_bin('1010', BitIndexing.Lsb0)
    assert t1 == t2
    assert t1.bit_indexing != t2.bit_indexing

    m1 = Mutibs.from_hex('1010', BitIndexing.Msb0)
    m2 = Mutibs.from_hex('1010', BitIndexing.Lsb0)
    assert m1 == m2
    assert m1.bit_indexing != m2.bit_indexing
