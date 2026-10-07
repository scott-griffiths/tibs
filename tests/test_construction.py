#!/usr/bin/env python
import array
import io
import struct
import sys

import pytest
from hypothesis import given
import hypothesis.strategies as st
from tibs import Tibs, Mutibs
from typing import Iterable


class TestCreation:
    def test_creation_from_bytes(self):
        s = Tibs.from_bytes(b"\xa0\xff")
        assert (len(s), s.hex) == (16, "a0ff")

    @given(st.binary())
    def test_creation_from_bytes_roundtrip(self, data):
        s = Tibs.from_bytes(data)
        assert s.to_bytes() == data

    def test_creation_from_hex(self):
        s = Tibs.from_hex("0xA0ff")
        assert (len(s), s.hex) == (16, "a0ff")

    def test_creation_from_byte_aligned_hex_tokens(self):
        assert Tibs.from_string("0xab_cd,,0x,0x0123,0X45").hex == "abcd012345"

    def test_creation_from_odd_width_hex_tokens(self):
        assert Tibs.from_string("0xa,0xb,0xc,0xd").hex == "abcd"


class TestInitialisation:
    def test_empty_init(self):
        a = Tibs()
        assert a == Tibs()

    def test_find(self):
        a = Tibs.from_string("0xabcd")
        r = a.find("0xbc")
        assert r == 4
        r = a.find("0x23462346246", byte_aligned=True)
        assert r is None

    def test_rfind(self):
        a = Tibs.from_string("0b11101010010010")
        b = a.rfind("0b010")
        assert b == 11

    def test_find_all(self):
        a = Tibs("0b0010011")
        b = a.find_all('0b1')
        assert b == [2, 5, 6]
        t = Tibs("0b10")
        tp = t.find_all("0b1")
        assert tp == [0]


class TestCut:
    def test_cut(self):
        s = Tibs().from_joined(["0b000111"] * 10)
        for t in s.chunks(6):
            assert t == Tibs('0b000111')


def test_unorderable():
    a = Tibs("0b000111")
    b = Tibs("0b000111")
    with pytest.raises(TypeError):
        _ = a < b
    with pytest.raises(TypeError):
        _ = a > b
    with pytest.raises(TypeError):
        _ = a <= b
    with pytest.raises(TypeError):
        _ = a >= b


class TestPadToken:
    def test_creation(self):
        with pytest.raises(ValueError):
            _ = Tibs.from_string("pad10")
        with pytest.raises(ValueError):
            _ = Tibs.from_string("pad")


def test_adding():
    a = Tibs.from_string("0b0")
    b = Tibs.from_string("0b11")
    c = a + b
    assert c == Tibs('0b011')
    assert a == Tibs('0b0')
    assert b == Tibs('0b11')


class TestContainsBug:
    def test_contains(self):
        a = Tibs.from_string("0b1, 0x0001dead0001")
        assert "0xdead" in a
        assert "0xfeed" not in a

        assert "0b1" in Tibs.from_string("0xf")
        assert "0b0" not in Tibs.from_string("0xf")


class TestUnderscoresInLiterals:
    def test_hex_creation(self):
        a = Tibs.from_hex("ab_cd__ef")
        assert a.to_hex() == "abcdef"
        b = Tibs.from_string("0x0102_0304")
        assert b.to_hex() == "01020304"

    def test_binary_creation(self):
        a = Tibs.from_bin("0000_0001_0010")
        assert a.bin == "000000010010"
        b = Tibs.from_string("0b0011_1100_1111_0000")
        assert b.to_bin() == "0011110011110000"

    def test_octal_creation(self):
        a = Tibs.from_oct("0011_2233_4455_6677")
        assert a.oct == "0011223344556677"
        b = Tibs.from_string("0o123_321_123_321")
        assert b.to_oct() == "123321123321"


def test_from_iterable():
    with pytest.raises(TypeError):
        _ = Tibs.from_bools()
    a = Tibs.from_bools([])
    assert a == Tibs()
    a = Tibs.from_bools([1, 0, 1, 1])
    assert a == Tibs('0b1011')
    a = Tibs.from_bools((True,))
    assert a.to_bin() == "1"


def test_constructor_strict_bit_pattern_promotion():
    for cls in (Tibs, Mutibs):
        assert cls([True, False, 1, 0]) == Tibs("0b1010")
        assert cls((True, False, 1, 0)) == Tibs("0b1010")


def test_constructor_rejects_ambiguous_iterables():
    for cls in (Tibs, Mutibs):
        with pytest.raises(TypeError, match="from_values"):
            cls([1, 2, 3])

        iterator = iter([1, 0, 1])
        with pytest.raises(TypeError, match="from_bools"):
            cls(iterator)
        assert cls.from_bools(iter([1, 0, 1])) == Tibs("0b101")

        stream = io.BytesIO(b"\x01\x02")
        with pytest.raises(TypeError, match="from_bytes"):
            cls(stream)
        assert stream.tell() == 0

        byte_array = array.array("B", [1, 2, 3])
        with pytest.raises(TypeError, match="from_bytes"):
            cls(byte_array)
        assert cls.from_bytes(memoryview(byte_array)) == Tibs("0x010203")


# A memoryview is read as the bytes it covers, whatever its format or shape,
# so the result is always bytes(mv). Reading it item by item instead, as tibs
# 2.0.1 did, silently halved an 'H' view and raised for most other formats.
# Multi-byte items are packed in native order, as the buffer holds them.
MEMORYVIEW_FORMATS = {
    "H": (lambda: memoryview(array.array("H", [1, 2])), struct.pack("=HH", 1, 2)),
    "H_over_255": (lambda: memoryview(array.array("H", [1, 258])), struct.pack("=HH", 1, 258)),
    "signed_b": (lambda: memoryview(array.array("b", [-1, 1])), b"\xff\x01"),
    "cast_c": (lambda: memoryview(b"ab").cast("c"), b"ab"),
    "two_dimensional": (lambda: memoryview(b"abcd").cast("B", (2, 2)), b"abcd"),
    "double": (lambda: memoryview(array.array("d", [1.0])), struct.pack("=d", 1.0)),
    "zero_dimensional": (lambda: memoryview(b"a").cast("B", ()), b"a"),
}

# bytes() copies a non-contiguous view in logical order, and so does tibs.
MEMORYVIEW_NON_CONTIGUOUS = {
    "step_2": (lambda: memoryview(b"abcdef")[::2], b"ace"),
    "reversed": (lambda: memoryview(b"abcdef")[::-1], b"fedcba"),
    "H_step_2": (lambda: memoryview(array.array("H", [1, 2, 3, 4]))[::2], struct.pack("=HH", 1, 3)),
    "two_dimensional_rows": (
        lambda: memoryview(bytearray(range(12))).cast("B", (3, 4))[::2],
        bytes([0, 1, 2, 3, 8, 9, 10, 11]),
    ),
}

MEMORYVIEW_CASES = {**MEMORYVIEW_FORMATS, **MEMORYVIEW_NON_CONTIGUOUS}


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
@pytest.mark.parametrize("make_view,expected", MEMORYVIEW_CASES.values(), ids=MEMORYVIEW_CASES.keys())
def test_from_bytes_reads_a_memoryview_as_its_bytes(cls, make_view, expected):
    view = make_view()
    assert bytes(view) == expected
    bits = cls.from_bytes(view)
    assert type(bits) is cls
    assert bits.to_bytes() == expected
    # The buffer was released, or this would raise BufferError.
    view.release()


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
@pytest.mark.parametrize("make_view,expected", MEMORYVIEW_CASES.values(), ids=MEMORYVIEW_CASES.keys())
def test_constructor_promotes_a_memoryview_as_its_bytes(cls, make_view, expected):
    assert cls(make_view()) == Tibs.from_bytes(expected)


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
@pytest.mark.parametrize("make_view,expected", MEMORYVIEW_CASES.values(), ids=MEMORYVIEW_CASES.keys())
def test_from_bytes_offset_and_length_apply_to_a_memoryviews_bytes(cls, make_view, expected):
    expected_bin = "".join(f"{byte:08b}" for byte in expected)
    data_length = len(expected_bin)
    for bit_offset, bit_length in ((0, data_length), (4, data_length - 4), (3, 10), (data_length, 0)):
        if bit_offset + bit_length > data_length:
            continue
        bits = cls.from_bytes(make_view(), bit_offset, bit_length)
        assert bits.to_bin() == expected_bin[bit_offset:bit_offset + bit_length]


@given(
    data=st.binary(max_size=48),
    fmt=st.sampled_from("BbcHhIiLlQqfd?"),
    step=st.sampled_from([1, 2, 3, -1, -2]),
)
def test_from_bytes_memoryview_matches_bytes_for_any_format_and_stride(data, fmt, step):
    itemsize = struct.calcsize(fmt)
    data = data[: len(data) - len(data) % itemsize]
    view = memoryview(data).cast(fmt)[::step]
    for cls in (Tibs, Mutibs):
        assert cls.from_bytes(view).to_bytes() == bytes(view)


def test_byte_memoryview_releases_its_buffer():
    data = bytearray(b"\x01\x02\x03")
    view = memoryview(data)
    assert Tibs.from_bytes(view) == Tibs("0x010203")
    assert Mutibs(view) == Tibs("0x010203")
    view.release()
    data.append(4)  # Resizing raises BufferError while any export is held.


def test_released_memoryview_raises_like_bytes():
    view = memoryview(b"ab")
    view.release()
    with pytest.raises(ValueError, match="released memoryview"):
        bytes(view)
    for cls in (Tibs, Mutibs):
        with pytest.raises(ValueError, match="released memoryview"):
            cls.from_bytes(view)


def test_other_bytes_like_parameters_read_a_memoryview_as_its_bytes():
    def h_view():
        return memoryview(array.array("H", [1, 258]))

    raw = struct.pack("=HH", 1, 258)

    m = Mutibs()
    m.write_bytes(h_view())
    assert m.to_bytes() == raw
    m.bytes = h_view()
    assert m.to_bytes() == raw

    m = Mutibs.from_zeros(32)
    m.view().write_bytes(h_view())
    assert m.to_bytes() == raw

    assert Tibs.from_value("bytes32", h_view()).to_bytes() == raw
    assert Tibs.from_random(256, seed=h_view()) == Tibs.from_random(256, seed=raw)

    encoded = Tibs("0x0123456789")[3:].encode()
    for cls in (Tibs, Mutibs):
        assert cls.decode(memoryview(encoded).cast("c")) == cls.decode(encoded)


def test_mul_by_zero():
    a = Tibs.from_string("0b1010")
    b = a * 0
    assert b == Tibs()
    b = a * 1
    assert b == a
    b = a * 2
    assert b == a + a


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
@pytest.mark.parametrize("n", [2**62, 2**62 + 1, 2**63 - 1])
def test_mul_result_too_long_raises(cls, n):
    # The result length was multiplied unchecked, so it wrapped: 80 bits times
    # 2**62 came back empty, and times 2**62 + 1 came back as the original.
    a = cls("0x0123456789abcdef0123")
    with pytest.raises(MemoryError):
        a * n
    with pytest.raises(MemoryError):
        n * a


@pytest.mark.parametrize("n", [2**62, 2**62 + 1, 2**63 - 1])
def test_imul_result_too_long_raises_and_leaves_the_value(n):
    a = Mutibs("0x0123456789abcdef0123")
    with pytest.raises(MemoryError):
        a *= n
    assert a == Tibs("0x0123456789abcdef0123")


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
def test_mul_of_empty_by_huge_count_is_empty(cls):
    assert cls() * (2**63 - 1) == Tibs()
    a = cls()
    a *= 2**63 - 1
    assert a == Tibs()


def test_from_ones():
    a = Tibs.from_ones(0)
    assert a == Tibs()
    a = Tibs.from_ones(1)
    assert a == Tibs("0b1")
    with pytest.raises(ValueError):
        _ = Tibs.from_ones(-1)


def test_from_zeros():
    a = Tibs.from_zeros(0)
    assert a == Tibs()
    a = Tibs.from_zeros(1)
    assert a == Tibs("0b0")
    with pytest.raises(ValueError):
        _ = Tibs.from_zeros(-1)


def test_bits_slicing():
    a = Tibs('0b1010101010101010')
    b = a[-5:-8:1]
    assert b == Tibs()

    assert a[::2] == Tibs('0xff')
    assert a[1::2] == Tibs('0x00')


def test_from_random():
    a = Tibs.from_random(0)
    assert a == Tibs()
    a = Tibs.from_random(1)
    assert a == Tibs('0b1') or a == Tibs('0b0')
    a = Tibs.from_random(10000, seed=b'a_seed')
    b = Tibs.from_random(10000, seed=b'a_seed')
    assert a == b
    b = Tibs.from_random(10000,
                         seed=b'a different seed this time - quite long to test if this makes a difference or not. It shouldnt really, but who knows?')
    assert a != b
    c = Mutibs.from_random(10000, seed=b'a_seed')
    assert a == c


def test_strict_equality_and_hashing():
    assert Tibs("0xf") == Tibs("0b1111")
    assert Tibs("0xf") == Mutibs("0xf")
    assert Mutibs("0xf") == Tibs("0xf")

    # Equality is the one place that does not promote, and it is deliberate
    # rather than an oversight: a Tibs is hashable, so anything comparing equal
    # to it would have to hash equal to it, and hash(Tibs('0xff')) cannot agree
    # with both hash('0xff') and hash(b'\xff'). The standard library draws the
    # same line with b'a' != 'a'. Do not "fix" these by promoting.
    assert Tibs("0xf") != "0xf"
    assert "0xf" != Tibs("0xf")
    assert Tibs("0xf") != b"\x0f"
    assert Tibs("0xf") != bytearray(b"\x0f")
    assert Tibs("0xf") != memoryview(b"\x0f")
    assert Tibs("0b101") != [1, 0, 1]
    assert Tibs("0b101") != (1, 0, 1)
    assert Mutibs("0xf") != "0xf"
    assert Mutibs("0b101") != [1, 0, 1]

    # ... while every other argument position does promote, which is the
    # asymmetry these assertions exist to hold in place.
    assert Tibs("0xf") + "0xf" == Tibs("0xff")
    assert "0b101" in Tibs("0b11010")
    assert Tibs("0b1010").find([1, 0]) == 0

    a = Tibs("0xabcd")
    b = Tibs("0xabcd")
    c = Tibs("0x00abcd")[8:]
    d = Tibs("0b11001101")

    assert len({a, b, c}) == 1
    assert hash(a) == hash(b) == hash(c)
    assert len({Tibs("0x0f"), Tibs("0b1111"), d}) == 3

    with pytest.raises(TypeError, match="unhashable"):
        hash(Mutibs("0xf"))


def test_is_things():
    a = Tibs('0b1010101010101010')
    b = Mutibs('0b1')
    assert isinstance(a, Iterable)
    # A Mutibs is deliberately not iterable, and says so through the protocol
    # rather than only when iteration is attempted: `Mutibs.__iter__` is None,
    # which `collections.abc` reads as "not implemented".
    assert not isinstance(b, Iterable)
    with pytest.raises(TypeError, match="not iterable"):
        iter(b)


def test_bits_from_bytes_string():
    a = Tibs.from_bytes(b'ABC')
    assert a.bytes == b'ABC'


def test_bool_conversion():
    a = Tibs()
    b = Tibs('0b0')
    c = Tibs('0b1')
    assert not a
    assert b
    assert c


def test_find_all():
    a = Tibs(' 0 B 0 0 01011')
    g = a.find_all_iter('0b1')
    assert next(g) == 3
    assert next(g) == 5
    assert next(g) == 6
    with pytest.raises(StopIteration):
        _ = next(g)


def test_repr():
    a = Tibs()
    assert repr(a) == "Tibs()"
    a = Tibs('')
    assert repr(a) == "Tibs()"
    a = Tibs(" 0b 1")
    assert repr(a) == "Tibs('0b1')"


def test_bits_not_orderable():
    a = Tibs.from_string("0b0")
    b = Tibs.from_string("0b1")
    with pytest.raises(TypeError):
        _ = a < b
    with pytest.raises(TypeError):
        _ = a <= b
    with pytest.raises(TypeError):
        _ = a > b
    with pytest.raises(TypeError):
        _ = a >= b


def test_bools_from_iterable():
    v = [1, 0, 0, 1]
    i = iter(v)
    b = Tibs.from_bools(i)
    assert b == Tibs('0b1001')


def test_joined_from_iterable():
    v = [[0], '0b11']
    i = iter(v)
    b = Tibs.from_joined(v)
    assert b == Tibs('0b011')
    assert Tibs.from_joined(["0b1", [0, 1], b"\xff"]) == Tibs("0b10111111111")


def test_joined_repeated_bit_containers():
    expected = Tibs('0b101101101101')
    for cls in (Tibs, Mutibs):
        assert cls.from_joined([Tibs('0b101')] * 4) == expected
        assert cls.from_joined([Mutibs('0b101')] * 4) == expected

    # Equal but distinct objects use the general list path.
    assert Tibs.from_joined([Tibs('0b101') for _ in range(4)]) == expected


def test_promotion_from_mutibs():
    m = Mutibs('0x123')
    t = Tibs(m)
    assert isinstance(t, Tibs)
    assert m == t
    m2 = Mutibs(t)
    assert isinstance(m2, Mutibs)
    assert m2 == t
    m3 = Mutibs(m)
    assert isinstance(m3, Mutibs)
    assert m3 == t


def test_reversed():
    a = Tibs('0b1100')
    b = a.reversed()
    assert b == Tibs('0b0011')

    m1 = Mutibs('0b11100')
    m2 = m1.reversed()
    assert m2 == Tibs('0b00111')
    m3 = Mutibs.from_random(1_000_000)
    m4 = m3.reversed()
    m4.reverse()
    assert m3 == m4


@pytest.mark.parametrize('length', [0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1001])
def test_reversed_matches_bit_string(length):
    bits = ''.join('01101'[i % 5] for i in range(length))
    for cls in (Tibs, Mutibs):
        a = cls('0b' + bits) if bits else cls()
        assert a.reversed().bin == bits[::-1]
        # The original must be left alone by the copying version.
        assert a.bin == bits


@pytest.mark.parametrize('offset', range(9))
def test_reversed_with_storage_starting_mid_byte(offset):
    source = ''.join('0110100011110000101'[i % 19] for i in range(offset + 37))
    for cls in (Tibs, Mutibs):
        a = cls('0b' + source)[offset:]
        assert a.reversed().bin == source[offset:][::-1]


class TestCapacityLimit:
    """A length past what the platform can hold must raise, not panic.

    bitvec addresses a bit with a ``usize`` and spends three of those bits on
    the position within an element, so a container holds at most 2**61 - 1 bits
    on a 64-bit build but only 2**29 - 1 (about 64 MB) on a 32-bit one - which
    the x86 wheels are. Before this was guarded, exceeding it panicked inside
    bitvec, and pyo3 turns a panic into ``PanicException``, which derives from
    ``BaseException`` so that it tears down the interpreter rather than being
    caught by ordinary error handling.

    Nothing here allocates: every length used is past what any build accepts,
    so the check has to reject it before reaching the allocator.
    """

    # bitvec's REGION_MAX_BITS, which is `usize::MAX >> 3`: an eighth of the
    # usize range, not the whole of it. 2**61 - 1 on a 64-bit build, and
    # 2**29 - 1 = 536,870,911 (64 MiB of data) on a 32-bit one.
    POINTER_BITS = sys.maxsize.bit_length() + 1
    CAP = 2 ** (POINTER_BITS - 3) - 1
    IS_32_BIT = POINTER_BITS == 32

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    @pytest.mark.parametrize(
        "name, args",
        [
            ("from_zeros", ()),
            ("from_ones", ()),
            ("from_random", ()),
        ],
    )
    def test_length_constructors_raise(self, cls, name, args):
        with pytest.raises(MemoryError, match="supports at most"):
            getattr(cls, name)(self.CAP + 1, *args)

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    @pytest.mark.parametrize("name, value", [("from_u", 1), ("from_i", 1), ("from_f", 1.0)])
    def test_numeric_constructors_raise(self, cls, name, value):
        with pytest.raises(MemoryError, match="supports at most"):
            getattr(cls, name)(value, self.CAP + 1)

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    def test_the_error_is_catchable_as_exception(self, cls):
        # The whole point: PanicException derives from BaseException, so this
        # would not have caught it.
        try:
            cls.from_zeros(self.CAP + 1)
        except Exception as e:
            assert isinstance(e, MemoryError)
        else:
            pytest.fail("expected the capacity limit to be reported")

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    def test_a_length_wider_than_usize_is_not_truncated(self, cls):
        # Only a 32-bit build can truncate, where usize is narrower than the
        # i64 arriving from Python: 'length as usize' would turn 2**32 + 100
        # into a silently-wrong 100-bit container. The check runs before the
        # cast, so the length is rejected instead. On a 64-bit build the same
        # length is merely large and legal, so only the over-cap value applies.
        lengths = [2**63 - 1]
        if self.IS_32_BIT:
            lengths.append(2**self.POINTER_BITS + 100)
        for length in lengths:
            with pytest.raises(MemoryError):
                cls.from_zeros(length)

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    def test_negative_lengths_still_report_as_value_errors(self, cls):
        # The capacity check must not have swallowed the existing negative case.
        with pytest.raises(ValueError, match="Negative bit length"):
            cls.from_zeros(-1)
        with pytest.raises(ValueError, match="Negative bit length"):
            cls.from_random(-1)

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    def test_the_limit_itself_is_not_rejected(self, cls):
        # Guards an off-by-one that would cap the container one bit short. The
        # only way to observe the boundary is to allocate it, so this runs only
        # on a 32-bit build, where the limit is 64 MiB rather than 256 PiB.
        if not self.IS_32_BIT:
            pytest.skip("allocating 2**61 bits is not a test")
        try:
            container = cls.from_zeros(self.CAP)
        except MemoryError as e:
            # Distinguish our own refusal, which is the bug being tested for,
            # from the machine genuinely not having 64 MB to spare - otherwise
            # this would be a flaky failure on a small 32-bit runner.
            if "supports at most" in str(e):
                pytest.fail(f"the capacity limit itself was rejected: {e}")
            pytest.skip("not enough memory to allocate the limit")
        assert len(container) == self.CAP

    def test_reserve_past_the_limit_raises(self):
        # bitvec's own reserve panics here.
        a = Mutibs("0x0123")
        with pytest.raises(MemoryError, match="supports at most"):
            a.reserve(self.CAP)
        with pytest.raises(MemoryError, match="supports at most"):
            a.reserve(2 * sys.maxsize + 1)  # usize::MAX, so len + additional overflows
        assert a == Tibs("0x0123")


@pytest.mark.skipif(sys.maxsize < 2**32, reason="2**60 bits is past the 32-bit limit")
class TestAllocationFailure:
    """A length within the limit that cannot be allocated must raise, not abort.

    2**60 bits is under the 64-bit container limit but is 128 PiB, more than any
    machine's address space, so its allocation always fails. An infallible Rust
    allocation aborts the process when that happens, which no ``except`` can
    catch; Python raises ``MemoryError`` for the same request.
    """

    LENGTH = 2**60

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    @pytest.mark.parametrize("name", ["from_zeros", "from_ones", "from_random"])
    def test_length_constructors_raise(self, cls, name):
        with pytest.raises(MemoryError, match="Not enough memory"):
            getattr(cls, name)(self.LENGTH)

    @pytest.mark.parametrize("cls", [Tibs, Mutibs])
    def test_repetition_raises(self, cls):
        a = cls("0b" + "10" * 40)
        with pytest.raises(MemoryError, match="Not enough memory"):
            a * (self.LENGTH // 80)

    def test_reserve_raises_and_keeps_the_value(self):
        a = Mutibs("0x0123")
        with pytest.raises(MemoryError, match="Not enough memory"):
            a.reserve(self.LENGTH)
        assert a == Tibs("0x0123")
        a.append(1)
        assert a == Tibs("0b00000001001000111")


def test_reserve_keeps_bits_stored_from_mid_byte():
    # A Mutibs whose storage starts part way into a byte is moved into place
    # before reserving, and must still hold the same bits afterwards.
    a = Mutibs("0b101" + "0110" * 20)
    del a[:3]
    a.reserve(10_000)
    assert a.capacity >= len(a) + 10_000
    assert a == Tibs("0b" + "0110" * 20)
    a.extend("0b1")
    assert a == Tibs("0b" + "0110" * 20 + "1")


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
@pytest.mark.parametrize(
    "offset, length, match",
    [
        (2**63 - 1, None, "Offset of"),
        (4, 2**63 - 1, "greater than the data length"),
        (0, 25, "greater than the data length"),
        (-1, None, "Negative bit offset"),
        (0, -1, "Negative bit length"),
    ],
)
def test_from_bytes_out_of_range_is_a_value_error(cls, offset, length, match):
    # Offsets and lengths are bounded by the data, so a huge one is out of range
    # of the data rather than out of memory. They used to go through the
    # container size check and come back as MemoryError.
    with pytest.raises(ValueError, match=match):
        cls.from_bytes(b"abc", offset, length)
