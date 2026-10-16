"""Packing and unpacking values least significant bit first, with ``bit_order``.

The reference throughout is ``lsb0.field``: value ``i`` of an ``n``-bit dtype
starting at label ``start`` is ``bits.lsb0.field(start + i*n + n-1, start + i*n)``,
and an array or tuple applies that rule to each scalar field in turn.
"""
import copy
import math
import struct

import pytest
from hypothesis import given, settings, strategies as st

from tibs import BitOrder, Dtype, DtypeArray, DtypeSingle, DtypeTuple, Mutibs, ReadError, Reader, Tibs

LSB0 = BitOrder.Lsb0


def reference_value(dtype, bits, label):
    """Decode one value of ``dtype`` at ``label`` through ``lsb0.field``."""
    if isinstance(dtype, DtypeSingle):
        if dtype.length == 0:
            return dtype.unpack(Tibs())
        view = bits.lsb0.field(label + dtype.length - 1, label)
        return view.to_value(dtype)
    if isinstance(dtype, DtypeArray):
        item = dtype.dtype
        return tuple(reference_value(item, bits, label + i * item.length) for i in range(dtype.count))
    values = []
    for field in dtype.dtypes:
        values.append(reference_value(field, bits, label))
        label += field.length
    return tuple(values)


def reference_values(dtype, bits, start, count):
    return [reference_value(dtype, bits, start + i * dtype.length) for i in range(count)]


def same(a, b):
    """Equality that treats NaN as equal to NaN, at any nesting depth."""
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(map(same, a, b))
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b


def low_labels(data, length):
    """The first ``length`` labels of ``data``, zero padded to whole bytes."""
    value = int.from_bytes(data, "little") & ((1 << length) - 1)
    return value.to_bytes((length + 7) // 8, "little")


# --- the test vectors from the issue ----------------------------------------

VECTORS = [
    ("e4", 2, [0, 1, 2, 3]),
    ("803fe00ff803fe803fe00ff803fe", 7, [0, 127] * 8),
    ("21436587a9cb", 12, [0x321, 0x654, 0x987, 0xCBA]),
    ("3412cdab", 16, [0x1234, 0xABCD]),
]


@pytest.mark.parametrize("data, n, values", VECTORS)
def test_vectors(data, n, values):
    data = bytes.fromhex(data)
    dtype = Dtype(f"u{n}")

    assert dtype.unpack_values(data, bit_order=LSB0) == values
    assert dtype.unpack_values(Tibs.from_bytes(data), bit_order=LSB0) == values
    assert list(dtype.unpack_values_iter(data, bit_order=LSB0)) == values
    assert dtype.pack_values(values, bit_order=LSB0).to_bytes() == data
    assert Reader(data, bit_order=LSB0).read_values(dtype) == values
    assert [dtype.unpack(data, start=i * n, end=(i + 1) * n, bit_order=LSB0)
            for i in range(len(values))] == values


def test_whole_bytes_match_little_endian():
    data = bytes(range(1, 17))
    for n in (16, 32, 64):
        assert Dtype(f"u{n}").unpack_values(data, bit_order=LSB0) == \
            Dtype(f"u{n}_le").unpack_values(data)
        assert Dtype(f"i{n}").unpack_values(data, bit_order=LSB0) == \
            Dtype(f"i{n}_le").unpack_values(data)
    assert Dtype("f32").unpack(struct.pack("<f", 1.5), bit_order=LSB0) == 1.5
    assert Dtype("f64").pack(-2.25, bit_order=LSB0).to_bytes() == struct.pack("<d", -2.25)


def test_msb0_is_the_default_and_can_be_given():
    data = bytes.fromhex("21436587a9cb")
    dtype = Dtype("u12")

    assert dtype.unpack_values(data, bit_order=BitOrder.Msb0) == dtype.unpack_values(data)
    assert dtype.pack_values([1, 2], bit_order=BitOrder.Msb0) == dtype.pack_values([1, 2])
    assert dtype.unpack_values(data, bit_order=None) == dtype.unpack_values(data)
    assert Reader(data).bit_order == BitOrder.Msb0


# --- packing ------------------------------------------------------------------


def test_pack_pads_to_whole_bytes_at_the_high_labels():
    packed = Dtype("u12").pack_values([0x321, 0x654, 0x987], bit_order=LSB0)

    assert len(packed) == 40
    assert packed.to_bytes() == bytes.fromhex("2143658709")
    assert Dtype("u12").pack(0xFFF, bit_order=LSB0) == Tibs("0xff0f")
    assert Dtype("bool").pack(True, bit_order=LSB0) == Tibs("0x01")
    assert Dtype("u3").pack_values([], bit_order=LSB0) == Tibs()


def test_tuple_and_array_apply_the_rule_to_each_field():
    data = bytes.fromhex("21436587")

    assert Dtype("(u12, u4)").unpack_values(data, bit_order=LSB0) == [(0x321, 0x4), (0x765, 0x8)]
    assert Dtype("[u12; 2]").unpack(data, end=24, bit_order=LSB0) == (0x321, 0x654)
    assert Dtype("(u12, u4)").pack_values([(0x321, 0x4), (0x765, 0x8)], bit_order=LSB0) \
        .to_bytes() == data


def test_bytes_and_text_kinds_read_in_value_order():
    # As lsb0.field reads them: the value's most significant bit is the
    # highest label, so whole bytes come out last byte first.
    t = Tibs.from_bytes(b"\x01\x02")

    assert Dtype("bytes16").unpack(t, bit_order=LSB0) == t.lsb0.field(15, 0).bytes == b"\x02\x01"
    assert Dtype("hex16").unpack(t, bit_order=LSB0) == "0201"
    assert Dtype("bits4").unpack(t, end=4, bit_order=LSB0) == Tibs("0b0001")


# --- errors -------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["u16_le", "f32_be", "(u8, u16_le)", "[i16_be; 2]"])
def test_a_dtype_with_a_byte_order_is_refused(spec):
    dtype = Dtype(spec)
    data = bytes(dtype.length // 8)
    calls = [
        lambda: dtype.unpack(data, bit_order=LSB0),
        lambda: dtype.unpack_values(data, bit_order=LSB0),
        lambda: dtype.unpack_values_iter(data, bit_order=LSB0),
        lambda: dtype.pack(dtype.unpack(data), bit_order=LSB0),
        lambda: dtype.pack_values([dtype.unpack(data)], bit_order=LSB0),
        lambda: Reader(data, bit_order=LSB0).read_value(dtype),
        lambda: Reader(data, bit_order=LSB0).peek_value(dtype),
        lambda: Reader(data, bit_order=LSB0).read_values(dtype),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="already decides the byte order"):
            call()


def test_bits_that_are_not_whole_bytes_are_refused():
    t = Tibs("0b101")
    with pytest.raises(ValueError, match="whole number of bytes"):
        Dtype("u1").unpack_values(t, bit_order=LSB0)
    with pytest.raises(ValueError, match="whole number of bytes"):
        Dtype("u3").unpack(t, bit_order=LSB0)
    with pytest.raises(ValueError, match="whole number of bytes"):
        Dtype("u1").unpack_values_iter(t, bit_order=LSB0)
    with pytest.raises(ValueError, match="whole number of bytes"):
        Reader(t, bit_order=LSB0)


def test_ranges_are_checked_as_usual():
    data = bytes(3)
    with pytest.raises(ValueError, match="not a multiple of dtype length"):
        Dtype("u12").unpack_values(data, end=20, bit_order=LSB0)
    with pytest.raises(ValueError, match="Cannot convert 20 bits"):
        Dtype("u12").unpack(data, end=20, bit_order=LSB0)
    with pytest.raises(ValueError):
        Dtype("u12").unpack(data, start=30, bit_order=LSB0)
    with pytest.raises(ValueError, match="length 0"):
        Dtype("bytes0").unpack_values(data, bit_order=LSB0)
    assert Dtype("bytes0").unpack(data, end=0, bit_order=LSB0) == b""
    assert Dtype("u12").unpack_values(data, start=-12, bit_order=LSB0) == [0]


# --- the reference rule, for any length and offset ----------------------------


@settings(max_examples=300)
@given(data=st.binary(min_size=1, max_size=40), n=st.integers(1, 64), start=st.integers(0, 80),
       kind=st.sampled_from(["u", "i", "bits", "bin"]))
def test_unpack_values_agrees_with_field(data, n, start, kind):
    t = Tibs.from_bytes(data)
    start = start % (len(t) + 1)
    count = (len(t) - start) // n
    dtype = Dtype(f"{kind}{n}")
    end = start + count * n

    expected = reference_values(dtype, t, start, count)
    assert dtype.unpack_values(t, start=start, end=end, bit_order=LSB0) == expected
    assert dtype.unpack_values(data, start=start, end=end, bit_order=LSB0) == expected
    assert list(dtype.unpack_values_iter(t, start=start, end=end, bit_order=LSB0)) == expected
    r = Reader(t, start, bit_order=LSB0)
    assert r.read_values(dtype) == expected
    assert r.pos == end
    if count:
        assert dtype.unpack(t, start=start, end=start + n, bit_order=LSB0) == expected[0]
        assert Reader(t, start, bit_order=LSB0).read_bits(n) == t.lsb0.field(start + n - 1, start).to_tibs()


@settings(max_examples=300)
@given(data=st.binary(min_size=1, max_size=40), n=st.integers(1, 64), start=st.integers(0, 80))
def test_pack_values_inverts_unpack_values(data, n, start):
    t = Tibs.from_bytes(data)
    start = start % (len(t) + 1)
    count = (len(t) - start) // n
    dtype = Dtype(f"u{n}")
    values = dtype.unpack_values(t, start=start, end=start + count * n, bit_order=LSB0)

    shifted = (int.from_bytes(data, "little") >> start).to_bytes(len(data), "little")
    assert dtype.pack_values(values, bit_order=LSB0).to_bytes() == low_labels(shifted, count * n)


SPECS = ["f16", "f32", "f64", "bf16", "bool", "hex12", "oct9", "bytes16", "ocp_e2m1",
         "(u12, u4)", "(bool, i7, f16)", "[u3; 5]", "(u5, [i3; 3], (bool, u2))",
         "[(u3, u2); 3]", "(u4, bytes0, [u8; 0], u4)", "bytes72", "u100"]


@settings(max_examples=200)
@given(data=st.binary(min_size=1, max_size=64), spec=st.sampled_from(SPECS),
       start=st.integers(0, 100))
def test_every_kind_agrees_with_field_and_round_trips(data, spec, start):
    t = Tibs.from_bytes(data)
    dtype = Dtype(spec)
    start = start % (len(t) + 1)
    count = (len(t) - start) // dtype.length
    end = start + count * dtype.length

    values = dtype.unpack_values(t, start=start, end=end, bit_order=LSB0)
    assert all(map(same, values, reference_values(dtype, t, start, count)))
    # Not every bit pattern survives a round trip through a value (a NaN's
    # payload, say), so the repacked bits are checked by decoding them again.
    repacked = dtype.pack_values(values, bit_order=LSB0)
    assert len(repacked) == (count * dtype.length + 7) // 8 * 8
    assert all(map(same, dtype.unpack_values(repacked, end=count * dtype.length, bit_order=LSB0),
                   values))


@given(values=st.lists(st.integers(0, 2**12 - 1), max_size=30))
def test_pack_then_unpack(values):
    dtype = Dtype("u12")
    packed = dtype.pack_values(values, bit_order=LSB0)
    assert dtype.unpack_values(packed, end=12 * len(values), bit_order=LSB0) == values
    if values:
        assert dtype.pack(values[0], bit_order=LSB0) == \
            dtype.pack_values(values[:1], bit_order=LSB0)


# --- Reader -------------------------------------------------------------------


def test_reader_reads_in_lsb0_order():
    r = Reader(bytes.fromhex("21436587a9cb"), bit_order=LSB0)

    assert r.bit_order == LSB0
    assert r.peek_value("u12") == 0x321
    assert r.read_value("u12") == 0x321
    assert r.read_value("u4") == 0x4
    assert r.read_value("u8") == 0x65
    assert r.read_values("u12", 1) == [0x987]
    assert r.peek_bits(4) == Tibs("0xa")
    assert r.read_bits(12) == Tibs("0xcba")
    assert r.at_end
    with pytest.raises(ReadError):
        r.read_value("u1")


def test_reader_unaligned_position_selects_one_fat12_entry():
    table = Dtype("u12").pack_values(range(100, 140), bit_order=LSB0)
    for k in (0, 1, 7, 38):
        assert Reader(table, 12 * k, bit_order=LSB0).read_value("u12") == 100 + k


def test_reader_over_a_growing_mutibs():
    m = Mutibs.from_bytes(b"\x21")
    r = Reader(m, bit_order=LSB0)
    assert r.read_value("u4") == 1
    m.extend(b"\x43")
    assert r.read_value("u12") == 0x432
    m.append(1)
    with pytest.raises(ValueError, match="whole number of bytes"):
        r.read_value("u1")


def test_reader_lsb0_refuses_searches():
    r = Reader(b"\x00\xff", bit_order=LSB0)
    for call in (lambda: r.seek_to("0xff"), lambda: r.seek_past("0xff"),
                 lambda: r.read_to("0xff"), lambda: r.read_past("0xff")):
        with pytest.raises(ValueError, match="can't search"):
            call()
    r.pos = 16
    with pytest.raises(ValueError, match="can't search"):
        r.seek_back_to("0xff")
    assert r.pos == 16
    # Moving the cursor around is unaffected.
    r.pos = 3
    assert r.align() == 5


def test_reader_lsb0_copies_and_repr():
    r = Reader(Tibs("0x2143"), 4, bit_order=LSB0)

    assert repr(r) == "Reader(Tibs('0x2143'), 4, bit_order=BitOrder.Lsb0)"
    assert repr(Reader(Tibs("0x2143"), 4)) == "Reader(Tibs('0x2143'), 4)"
    for other in (copy.copy(r), copy.deepcopy(r)):
        assert other.bit_order == LSB0
        assert other.read_value("u12") == 0x432
