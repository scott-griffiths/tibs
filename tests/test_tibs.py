#!/usr/bin/env python
import pytest
import io
import re
from hypothesis import given
import tibs
from tibs import Tibs, Mutibs
from typing import Iterable, Sequence

def test_from_bin():
    a = Tibs.from_bin('010')
    b = Tibs.from_string('0b010')
    c = Tibs.from_bin('0b010')
    d = Tibs('0b010')
    assert a == b == c == d

def test_to_bin():
    a = Tibs('0b1001')
    assert a.to_bin() == '1001'

def test_from_oct():
    a = Tibs.from_oct('12')
    b = Tibs.from_string('0o12')
    c = Tibs.from_oct('0o12')
    d = Tibs('0o12')
    assert a == b == c == d

def test_to_oct():
    a = Tibs('0b001100')
    assert a.to_oct() == '14'

def test_from_hex():
    a = Tibs.from_hex('A')
    b = Tibs.from_string('0xA')
    c = Tibs.from_hex('0xA')
    d = Tibs('0xA')
    assert a == b == c == d

def test_to_hex():
    a = Tibs('0b1010')
    assert a.to_hex() == 'a'