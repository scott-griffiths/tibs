#!/usr/bin/env python
import pytest
from tibs import Tibs, Mutibs, Endianness, Dtype, DtypeKind

def test_creation():
    d = Dtype(DtypeKind.Uint, 8)
    assert d.kind is DtypeKind.Uint
    assert d.length == 8


def test_dtype_float():
    d = Dtype(DtypeKind.Float, 16, Endianness.Little)
    t = Tibs.from_dtype(d, 14.5)
    t2 = Tibs.from_f(14.5, 16, Endianness.Little)
    assert t == t2


def test_dtype_uint():
    d = Dtype(DtypeKind.Uint, 9)
    t = Tibs.from_dtype(d, 17)
    assert t == Tibs.from_u(17, 9)


def test_dtype_bin():
    d = Dtype(DtypeKind.Bin, 4)
    t = Tibs.from_dtype(d, "0b1010")
    assert t == Tibs.from_bin("0b1010")


def test_dtype_oct():
    d = Dtype(DtypeKind.Oct, 6)
    t = Tibs.from_dtype(d, "17")
    assert t == Tibs.from_oct("17")


def test_dtype_hex():
    d = Dtype(DtypeKind.Hex, 8)
    t = Tibs.from_dtype(d, "0f")
    assert t == Tibs.from_hex("0f")
