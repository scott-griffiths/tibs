#!/usr/bin/env python

# The 'u' and 'i' interpretations used to be capped at 128 bits, because the
# conversions went through Rust's u128 / i128. They are now unbounded: lengths
# up to 64 bits take a fast path through u64, and anything longer goes via
# int.to_bytes / int.from_bytes. These tests pin down the behaviour on both
# sides of that boundary, and check that the join between them is invisible.

import pytest
from hypothesis import given, strategies as st
from tibs import ByteOrder, Dtype, Mutibs, MutableView, Tibs, View


@pytest.fixture(params=[Tibs, Mutibs])
def cls(request):
    return request.param


# Lengths chosen to cover: the u64 fast path, both sides of the 64 bit tier
# boundary, the old 128 bit limit, whole-byte and part-byte lengths above it,
# and something comfortably larger than any machine word.
SMALL_LENGTHS = [1, 2, 7, 8, 31, 32, 63, 64]
LARGE_LENGTHS = [65, 71, 72, 100, 127, 128, 129, 130, 160, 200, 201, 256, 999, 1024]
ALL_LENGTHS = SMALL_LENGTHS + LARGE_LENGTHS


def u_edges(length):
    """Interesting unsigned values that fit in `length` bits."""
    values = {0, 1, (1 << length) - 1, (1 << length) >> 1}
    if length > 1:
        values |= {(1 << (length - 1)) - 1, 1 << (length - 1)}
    if length > 64:
        # Straddle the 64 bit word boundary, where a chunked implementation
        # would be most likely to lose or duplicate bits.
        values |= {(1 << 64) - 1, 1 << 64, (1 << 64) + 1, (1 << 65) - 1}
    return sorted(v for v in values if 0 <= v < (1 << length))


def i_edges(length):
    """Interesting signed values that fit in `length` bits."""
    lo, hi = -(1 << (length - 1)), (1 << (length - 1)) - 1
    values = {0, lo, hi, -1}
    if length > 1:
        values |= {1, -2, lo + 1, hi - 1}
    if length > 65:
        values |= {(1 << 64) - 1, 1 << 64, -(1 << 64), -(1 << 64) - 1}
    return sorted(v for v in values if lo <= v <= hi)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("length", ALL_LENGTHS)
def test_unsigned_round_trip(cls, length):
    for value in u_edges(length):
        t = cls.from_u(value, length)
        assert len(t) == length
        assert t.to_u() == value
        assert t.u == value


@pytest.mark.parametrize("length", ALL_LENGTHS)
def test_signed_round_trip(cls, length):
    for value in i_edges(length):
        t = cls.from_i(value, length)
        assert len(t) == length
        assert t.to_i() == value
        assert t.i == value


@pytest.mark.parametrize("length", LARGE_LENGTHS)
def test_bit_pattern_matches_binary_string(cls, length):
    # Round trips can agree while both directions share the same bug, so check
    # the bits themselves against Python's own formatting.
    for value in u_edges(length):
        t = cls.from_u(value, length)
        assert t.bin == format(value, f"0{length}b")
        assert t == cls.from_bin(format(value, f"0{length}b"))


@pytest.mark.parametrize("length", LARGE_LENGTHS)
def test_signed_bit_pattern_is_twos_complement(cls, length):
    for value in i_edges(length):
        unsigned = value & ((1 << length) - 1)
        assert cls.from_i(value, length) == cls.from_u(unsigned, length)
        assert cls.from_i(value, length).to_u() == unsigned


def test_tier_boundary_is_seamless(cls):
    # Every length either side of the 64 bit fast path boundary, with a value
    # whose bits reach the top of the field.
    for length in range(1, 200):
        value = (1 << length) - 1
        assert cls.from_u(value, length).to_u() == value
        assert cls.from_u(value, length).to_i() == -1
        assert cls.from_i(-1, length).to_u() == value


def test_all_ones_and_all_zeros(cls):
    for length in LARGE_LENGTHS:
        assert cls.from_ones(length).u == (1 << length) - 1
        assert cls.from_ones(length).i == -1
        assert cls.from_zeros(length).u == 0
        assert cls.from_zeros(length).i == 0


def test_to_u_and_to_i_accept_a_range(cls):
    t = cls.from_u(1, 200) + cls.from_u((1 << 200) - 1, 200)
    assert t.to_u(0, 200) == 1
    assert t.to_u(200, 400) == (1 << 200) - 1
    assert t.to_i(200, 400) == -1
    assert t.to_u(-200) == (1 << 200) - 1


# ---------------------------------------------------------------------------
# Overflow and error cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("length", ALL_LENGTHS)
def test_unsigned_out_of_range(cls, length):
    with pytest.raises(ValueError):
        cls.from_u(1 << length, length)
    with pytest.raises(ValueError):
        cls.from_u((1 << length) + 1, length)


@pytest.mark.parametrize("length", ALL_LENGTHS)
def test_signed_out_of_range(cls, length):
    with pytest.raises(ValueError):
        cls.from_i(1 << (length - 1), length)
    with pytest.raises(ValueError):
        cls.from_i(-(1 << (length - 1)) - 1, length)


@pytest.mark.parametrize("length", ALL_LENGTHS)
def test_negative_value_for_unsigned(cls, length):
    # Whatever the length, a negative value is a ValueError rather than the
    # TypeError that a different code path might produce.
    with pytest.raises(ValueError):
        cls.from_u(-1, length)


def test_hugely_oversized_value_reports_the_field_not_the_word_size(cls):
    # This used to be pyo3's "int too big to convert", which described u128
    # rather than the field the user asked for.
    with pytest.raises(ValueError) as excinfo:
        cls.from_u(1 << 200, 8)
    assert "8" in str(excinfo.value)


def test_zero_length_is_still_rejected(cls):
    with pytest.raises(ValueError):
        cls.from_u(0, 0)
    with pytest.raises(ValueError):
        cls.from_i(0, 0)
    with pytest.raises(ValueError):
        cls().to_u()
    with pytest.raises(ValueError):
        cls().to_i()


def test_negative_length_is_still_rejected(cls):
    with pytest.raises(ValueError):
        cls.from_u(0, -1)
    with pytest.raises(ValueError):
        cls.from_i(0, -200)


@pytest.mark.parametrize("length", [8, 200])
def test_non_integers_are_type_errors(cls, length):
    for bad in (1.5, "5", None, [1], b"\x01"):
        with pytest.raises(TypeError):
            cls.from_u(bad, length)
        with pytest.raises(TypeError):
            cls.from_i(bad, length)


@pytest.mark.parametrize("length", [8, 65, 200])
def test_index_protocol_is_honoured(cls, length):
    # Anything with __index__ is an integer as far as Python is concerned, and
    # both tiers should agree. The old u128 path leaked a confusing
    # "unsupported operand type(s) for >>" TypeError here.
    class Indexable:
        def __index__(self):
            return 5

    assert cls.from_u(Indexable(), length).to_u() == 5
    assert cls.from_i(Indexable(), length).to_i() == 5
    assert cls.from_u(True, length).to_u() == 1


# ---------------------------------------------------------------------------
# Byte order
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("length", [72, 128, 200, 256, 1024])
def test_little_endian_matches_int_to_bytes(cls, length):
    nbytes = length // 8
    for value in u_edges(length):
        t = cls.from_u(value, length, byte_order=ByteOrder.Little)
        assert t.bytes == value.to_bytes(nbytes, "little")
        assert cls.from_u(value, length, byte_order=ByteOrder.Big).bytes == \
            value.to_bytes(nbytes, "big")


@pytest.mark.parametrize("length", [72, 200])
def test_little_endian_signed_matches_int_to_bytes(cls, length):
    nbytes = length // 8
    for value in i_edges(length):
        t = cls.from_i(value, length, byte_order=ByteOrder.Little)
        assert t.bytes == value.to_bytes(nbytes, "little", signed=True)


@pytest.mark.parametrize("length", [72, 200, 256])
def test_little_endian_round_trips_through_dtype(length):
    for value in u_edges(length):
        d = Dtype(f"u{length}_le")
        assert d.unpack(d.pack(value)) == value


@pytest.mark.parametrize("length", [65, 129, 201])
def test_byte_order_still_needs_whole_bytes(cls, length):
    with pytest.raises(ValueError):
        cls.from_u(0, length, byte_order=ByteOrder.Little)
    with pytest.raises(ValueError):
        cls.from_i(0, length, byte_order=ByteOrder.Big)


# ---------------------------------------------------------------------------
# Mutibs writers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("length", [64, 65, 200])
def test_mutibs_write_and_property_set(length):
    m = Mutibs.from_zeros(length)
    value = (1 << length) - 3
    m.write_u(value)
    assert m.u == value
    m.u = 7
    assert m.u == 7
    m.write_i(-3)
    assert m.i == -3
    m.i = -(1 << (length - 1))
    assert m.i == -(1 << (length - 1))


@pytest.mark.parametrize("length", [64, 65, 200])
def test_mutibs_out_of_range_write_leaves_value_untouched(length):
    m = Mutibs.from_u(1, length)
    with pytest.raises(ValueError):
        m.write_u(1 << length)
    assert m.u == 1
    with pytest.raises(ValueError):
        m.write_i(1 << (length - 1))
    assert m.u == 1


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("length", [64, 65, 200])
def test_view_reads_large_values(length):
    value = (1 << length) - 5
    v = View(Tibs.from_u(value, length))
    assert v.u == value
    assert v.to_u() == value
    assert v.i == -5
    assert v.to_i() == -5


@pytest.mark.parametrize("length", [64, 65, 200])
def test_mutable_view_writes_large_values(length):
    m = Mutibs.from_zeros(length)
    v = MutableView(m)
    value = (1 << length) - 5
    v.u = value
    assert m.u == value
    v.write_i(-1)
    assert m.all()
    with pytest.raises(ValueError):
        v.u = 1 << length


@pytest.mark.parametrize("length", [72, 200])
def test_view_little_endian_large(length):
    value = (1 << length) - 5
    v = Tibs.from_u(value, length, byte_order=ByteOrder.Little).view().le
    assert v.u == value


# ---------------------------------------------------------------------------
# Dtypes and bulk values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec,length", [("u65", 65), ("i65", 65), ("u200", 200),
                                         ("i200", 200), ("u1000", 1000)])
def test_dtype_pack_unpack(spec, length):
    d = Dtype(spec)
    assert d.length == length
    signed = spec[0] == "i"
    values = i_edges(length) if signed else u_edges(length)
    for value in values:
        packed = d.pack(value)
        assert len(packed) == length
        assert d.unpack(packed) == value


@pytest.mark.parametrize("spec", ["u65", "i65", "u200", "i200"])
def test_from_values_and_to_values(spec):
    d = Dtype(spec)
    length = d.length
    values = i_edges(length) if spec[0] == "i" else u_edges(length)
    t = Tibs.from_values(d, values)
    assert len(t) == length * len(values)
    assert t.to_values(d) == values
    assert list(t.to_values_iter(d)) == values
    assert Tibs.from_values(spec, values) == t


def test_to_value_and_from_value_round_trip():
    value = (1 << 300) - 12345
    t = Tibs.from_value("u300", value)
    assert t.to_value("u300") == value
    assert Tibs.from_value("i300", -1).to_value("i300") == -1


def test_values_of_mixed_sizes_concatenate_cleanly():
    # Small and large fields packed next to each other, so a wrong length in
    # either tier would shift everything after it.
    t = Tibs.from_value("u8", 0xAB) + Tibs.from_value("u200", 1) + Tibs.from_value("u8", 0xCD)
    assert len(t) == 216
    assert t.to_value("u8", 0, 8) == 0xAB
    assert t.to_value("u200", 8, 208) == 1
    assert t.to_value("u8", 208, 216) == 0xCD


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def test_format_codes_have_no_length_limit(cls):
    assert format(cls.from_ones(200), "u") == str(2 ** 200 - 1)
    assert format(cls.from_ones(200), "i") == "-1"
    assert format(cls.from_u(1 << 199, 200), "u") == str(1 << 199)
    assert format(cls.from_zeros(1000), "u") == "0"


def test_format_flags_still_apply_to_large_values(cls):
    t = cls.from_u(1 << 200, 256)
    assert format(t, ",u") == f"{1 << 200:,d}"
    assert format(t, "_u") == f"{1 << 200:_d}"
    assert format(t, ">70u") == f"{1 << 200:>70d}"
    assert format(cls.from_i(-1, 200), "+i") == "-1"


def test_fstring_with_large_values(cls):
    t = cls.from_u(1 << 130, 200)
    assert f"{t:u}" == str(1 << 130)


# ---------------------------------------------------------------------------
# Property based cross-checks against Python's own int conversions
# ---------------------------------------------------------------------------

@given(length=st.integers(min_value=1, max_value=600), data=st.data())
def test_unsigned_agrees_with_python(length, data):
    value = data.draw(st.integers(min_value=0, max_value=(1 << length) - 1))
    t = Tibs.from_u(value, length)
    assert t.to_u() == value
    assert t.bin == format(value, f"0{length}b")
    assert Mutibs.from_u(value, length) == t


@given(length=st.integers(min_value=1, max_value=600), data=st.data())
def test_signed_agrees_with_python(length, data):
    value = data.draw(st.integers(min_value=-(1 << (length - 1)),
                                  max_value=(1 << (length - 1)) - 1))
    t = Tibs.from_i(value, length)
    assert t.to_i() == value
    assert t.to_u() == value & ((1 << length) - 1)
    assert Mutibs.from_i(value, length) == t


@given(nbytes=st.integers(min_value=1, max_value=80), data=st.data())
def test_byte_order_agrees_with_python(nbytes, data):
    value = data.draw(st.integers(min_value=0, max_value=(1 << (nbytes * 8)) - 1))
    length = nbytes * 8
    assert Tibs.from_u(value, length, byte_order=ByteOrder.Little).bytes == \
        value.to_bytes(nbytes, "little")
    assert Tibs.from_u(value, length, byte_order=ByteOrder.Big).bytes == \
        value.to_bytes(nbytes, "big")


@given(data=st.binary(min_size=1, max_size=100))
def test_from_bytes_reads_as_python_int(data):
    t = Tibs.from_bytes(data)
    assert t.u == int.from_bytes(data, "big")
    assert t.i == int.from_bytes(data, "big", signed=True)


# ---------------------------------------------------------------------------
# Reading a field out of the storage bytes
#
# Fields of up to 64 bits are assembled from the bytes they sit in rather than
# walked bit by bit, so the reading has to hold for every combination of where
# the field starts within a byte and how long it is - including when the
# storage itself starts part way into a byte.
# ---------------------------------------------------------------------------

SPREAD = bytes((index * 37 + 11) % 256 for index in range(24))


@pytest.mark.parametrize("offset", range(64))
def test_word_sized_reads_at_every_offset_and_length(cls, offset):
    whole = cls.from_bytes(SPREAD)
    for length in range(1, 65):
        if offset + length > len(whole):
            break
        field = whole[offset: offset + length]
        expected = int(field.bin, 2)
        assert field.to_u() == expected
        assert whole.to_u(offset, offset + length) == expected
        signed = expected - (1 << length) if field[0] else expected
        assert field.to_i() == signed
        assert whole.to_i(offset, offset + length) == signed


@pytest.mark.parametrize("head", range(1, 8))
def test_word_sized_reads_when_the_storage_starts_part_way_into_a_byte(cls, head):
    shifted = (cls.from_ones(head) + cls.from_bytes(SPREAD))[head:]
    assert shifted == cls.from_bytes(SPREAD)
    for offset in range(0, 48, 5):
        for length in range(1, 65):
            expected = int(shifted[offset: offset + length].bin, 2)
            assert shifted.to_u(offset, offset + length) == expected


@pytest.mark.parametrize("length", [16, 32, 64])
@pytest.mark.parametrize("offset", range(8))
def test_float_reads_at_every_offset(cls, length, offset):
    packed = cls.from_f(1.5, length)
    padded = cls.from_ones(offset) + packed
    assert padded.to_f(offset, offset + length) == 1.5
    assert padded[offset:].to_f() == 1.5


@pytest.mark.parametrize("head", range(1, 8))
def test_little_endian_reads_when_the_storage_starts_part_way_into_a_byte(head):
    # Storage that begins mid-byte is the one route by which a little-endian
    # read reaches the general load rather than the byte-wise unpacker.
    for spec, value in [("u16_le", 65535), ("i32_le", -7), ("u64_le", 1 << 63),
                        ("f16_le", 1.5), ("f32_le", -2.5), ("f64_le", 1.5)]:
        dtype = Dtype(spec)
        packed = Mutibs.from_zeros(head) + Mutibs(dtype.pack(value))
        del packed[:head]
        assert dtype.unpack(packed) == value
        assert dtype.unpack_values(packed) == [value]
