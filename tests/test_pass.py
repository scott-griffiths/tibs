import pytest
from bitformat import Pass, Format, Bits


def test_pass_creation():
    p = Pass()
    assert p.bit_length == 0
    p.clear()
    assert p.to_bits() == Bits()
    assert p.to_bytes() == b""
    with pytest.raises(AttributeError):
        _ = p.value


def test_singleton():
    a = Pass()
    b = Pass()
    assert a is b


def test_using_in_format():
    f = Format.from_params([Pass(), "u8", Pass(), "i3", Pass(), ""])
    assert len(f) == 2


def test_eq():
    p = Pass()
    assert p == Pass()
