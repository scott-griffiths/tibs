import struct

import pytest

from tibs import (
    ByteOrder,
    Dtype,
    DtypeArray,
    DtypeKind,
    DtypeSingle,
    DtypeTuple,
    Mutibs,
    Tibs,
)


@pytest.mark.parametrize(
    "spec, expected_type, canonical",
    [
        ("u8", DtypeSingle, "u8"),
        (" [ u8 ; 4 ] ", DtypeArray, "[u8; 4]"),
        (" ( u8 , u16 , ) ", DtypeTuple, "(u8, u16)"),
        ("(u8,)", DtypeTuple, "(u8,)"),
        (" [ ( u8 , bool ) ; 3 ] ", DtypeArray, "[(u8, bool); 3]"),
    ],
)
def test_dtype_factory_returns_concrete_normalized_subtype(spec, expected_type, canonical):
    dtype = Dtype(spec)

    assert type(dtype) is expected_type
    assert isinstance(dtype, Dtype)
    assert str(dtype) == canonical
    assert repr(dtype) == f"{expected_type.__name__}({canonical!r})"


def test_concrete_dtype_constructors_accept_only_their_own_variant():
    assert DtypeSingle("u8") == Dtype("u8")
    assert DtypeArray("[u8; 2]") == Dtype("[u8; 2]")
    assert DtypeTuple("(u8, bool)") == Dtype("(u8, bool)")

    with pytest.raises(ValueError):
        DtypeSingle("[u8; 2]")
    with pytest.raises(ValueError):
        DtypeArray("(u8, bool)")
    with pytest.raises(ValueError):
        DtypeTuple("u8")


def test_parameter_builders_belong_to_concrete_subtypes():
    assert not hasattr(Dtype, "from_params")

    single = DtypeSingle.from_params(DtypeKind.Uint, 16, ByteOrder.Little)
    array = DtypeArray.from_params(single, 3)
    tuple_ = DtypeTuple.from_params(["bool", array])

    assert type(single) is DtypeSingle
    assert type(array) is DtypeArray
    assert type(tuple_) is DtypeTuple
    assert str(single) == "u16_le"
    assert str(array) == "[u16_le; 3]"
    assert str(tuple_) == "(bool, [u16_le; 3])"


def test_dtype_array_from_params_accepts_any_dtype_or_dtype_string():
    nested_array = DtypeArray.from_params("[u4; 2]", 3)
    record_array = DtypeArray.from_params("(u8, bool)", 2)

    assert nested_array == Dtype("[[u4; 2]; 3]")
    assert record_array == Dtype("[(u8, bool); 2]")


def test_compound_dtype_properties_are_structural_and_immutable():
    single = Dtype("u16_le")
    array = Dtype("[u16_le; 3]")
    tuple_ = Dtype("(bool, [u16_le; 3])")

    assert single.length == 16
    assert single.kind is DtypeKind.Uint
    assert single.byte_order is ByteOrder.Little

    assert array.length == 48
    assert array.dtype == single
    assert array.count == 3
    assert not hasattr(array, "kind")
    assert not hasattr(array, "byte_order")

    assert tuple_.length == 49
    assert tuple_.dtypes == (Dtype("bool"), array)
    assert type(tuple_.dtypes) is tuple
    assert not hasattr(tuple_, "kind")
    assert not hasattr(tuple_, "byte_order")

    for dtype, attribute in [
        (single, "kind"),
        (array, "count"),
        (tuple_, "dtypes"),
    ]:
        with pytest.raises(AttributeError):
            setattr(dtype, attribute, None)


def test_compound_dtype_equality_and_hashing_are_structural():
    dtypes = {
        Dtype("[u8; 2]"),
        DtypeArray.from_params("u8", 2),
        Dtype("(u8, u8)"),
        DtypeTuple.from_params(["u8", "u8"]),
    }

    assert dtypes == {Dtype("[u8; 2]"), Dtype("(u8, u8)")}
    assert Dtype("[u8; 2]") != Dtype("(u8, u8)")
    assert Dtype("[u8; 2]") != "[u8; 2]"


def test_concrete_reprs_round_trip_through_concrete_classes():
    namespace = {
        "DtypeSingle": DtypeSingle,
        "DtypeArray": DtypeArray,
        "DtypeTuple": DtypeTuple,
    }

    for dtype in [
        Dtype("u8"),
        Dtype("[u8; 4]"),
        Dtype("(u8, [bool; 3])"),
    ]:
        assert eval(repr(dtype), namespace) == dtype


@pytest.mark.parametrize(
    "spec",
    [
        "[]",
        "[u8]",
        "[u8;]",
        "[u8; 0]",
        "[u8; -1]",
        "()",
        "(,)",
        "(u8, bool",
        "u8, bool",
    ],
)
def test_invalid_compound_dtype_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        Dtype(spec)


def test_dtype_array_pack_and_unpack():
    dtype = Dtype("[u8; 4]")

    packed = dtype.pack([1, 2, 3, 4])

    assert packed == Tibs("0x01020304")
    assert dtype.unpack(packed) == (1, 2, 3, 4)
    assert Tibs.from_value(dtype, (1, 2, 3, 4)) == packed
    assert packed.to_value(dtype) == (1, 2, 3, 4)


def test_dtype_tuple_pack_and_unpack():
    dtype = Dtype("(u8, u16, bool)")

    packed = dtype.pack((1, 0x0203, True))

    assert packed == Tibs("0x010203, 0b1")
    assert dtype.unpack(packed) == (1, 0x0203, True)
    assert Tibs.from_value("(u8, u16, bool)", [1, 0x0203, True]) == packed
    assert packed.to_value("(u8, u16, bool)") == (1, 0x0203, True)


def test_nested_array_of_tuples_round_trips():
    dtype = Dtype("[(u4, bool); 2]")
    value = ((0xA, True), (0x3, False))

    packed = dtype.pack(value)

    assert packed == Tibs("0b1010100110")
    assert dtype.unpack(packed) == value


def test_nested_array_and_tuple_values_round_trip():
    dtype = Dtype("(u4, [[bool; 2]; 2], (u3, i3))")
    value = (0xA, ((True, False), (False, True)), (5, -2))

    packed = dtype.pack(value)

    assert len(packed) == dtype.length == 14
    assert dtype.unpack(packed) == value


@pytest.mark.parametrize(
    "dtype_spec, struct_spec",
    [
        ("(i16_le, i32_le, i32_le)", "<hll"),
        ("(i16_be, i32_be, i32_be)", ">hll"),
    ],
)
def test_explicit_endian_tuple_matches_standard_struct_layout(dtype_spec, struct_spec):
    values = (1, 2, 3)

    assert Dtype(dtype_spec).pack(values).bytes == struct.pack(struct_spec, *values)


def test_compound_pack_accepts_generators():
    array = Dtype("[u8; 3]")
    tuple_ = Dtype("(u8, u8, u8)")

    assert array.pack(x for x in [1, 2, 3]) == Tibs("0x010203")
    assert tuple_.pack(x for x in [1, 2, 3]) == Tibs("0x010203")


@pytest.mark.parametrize(
    "dtype, value",
    [
        (Dtype("[u8; 3]"), [1, 2]),
        (Dtype("[u8; 3]"), [1, 2, 3, 4]),
        (Dtype("(u8, bool)"), [1]),
        (Dtype("(u8, bool)"), [1, True, 2]),
    ],
)
def test_compound_pack_requires_exact_value_count(dtype, value):
    with pytest.raises(ValueError):
        dtype.pack(value)


def test_nested_pack_errors_report_the_value_path_without_changing_exception_type():
    dtype = Dtype("[(u8, bool); 2]")

    with pytest.raises(ValueError) as exc_info:
        dtype.pack(((1, True), (256, False)))

    message = str(exc_info.value).lower()
    assert "1" in message
    assert "0" in message


def test_compound_unpack_requires_exact_dtype_length():
    dtype = Dtype("(u8, u16)")

    with pytest.raises(ValueError):
        dtype.unpack(Tibs.from_zeros(dtype.length - 1))
    with pytest.raises(ValueError):
        dtype.unpack(Tibs.from_zeros(dtype.length + 1))


def test_compound_value_methods_accept_explicit_ranges():
    dtype = Dtype("(u8, u16)")
    value = (1, 0x0203)
    bits = Tibs("0xff") + dtype.pack(value) + Tibs("0xff")

    assert bits.to_value(dtype, 8, 32) == value


def test_compound_repeated_value_methods_repeat_the_whole_dtype():
    dtype = Dtype("(u8, u16_le)")
    values = [(1, 0x0203), (4, 0x0506)]

    packed = Tibs.from_values(dtype, values)

    assert packed == Tibs("0x010302040605")
    assert dtype.pack_values(iter(values)) == packed
    assert packed.to_values(dtype) == values
    assert list(packed.to_values_iter(dtype)) == values
    assert dtype.unpack_values(packed) == values
    assert list(dtype.unpack_values_iter(packed)) == values


def test_array_pack_values_repeats_arrays_rather_than_flattening_one_array():
    dtype = Dtype("[u4; 2]")
    values = [(1, 2), (3, 4)]

    packed = dtype.pack_values(values)

    assert packed == Tibs("0x1234")
    assert dtype.unpack_values(packed) == values


def test_compound_repeated_value_methods_accept_empty_iterables():
    dtype = Dtype("(u8, bool)")

    assert dtype.pack_values([]) == Tibs()
    assert dtype.unpack_values(Tibs()) == []
    assert list(dtype.unpack_values_iter(Tibs())) == []


def test_compound_repeated_unpack_requires_a_multiple_of_dtype_length():
    dtype = Dtype("(u8, bool)")

    with pytest.raises(ValueError):
        Tibs.from_zeros(dtype.length + 1).to_values(dtype)
    with pytest.raises(ValueError):
        list(Tibs.from_zeros(dtype.length + 1).to_values_iter(dtype))


def test_mutibs_compound_value_methods_match_tibs():
    dtype = Dtype("(u8, [bool; 2])")
    values = [(1, (True, False)), (2, (False, True))]

    mutable = Mutibs.from_values(dtype, values)

    assert mutable == Mutibs(Tibs.from_values(dtype, values))
    assert mutable.to_value(dtype, 0, dtype.length) == values[0]
    assert mutable.to_values(dtype) == values
    assert not hasattr(mutable, "to_values_iter")


# The fast flat-record path (a DtypeTuple of scalar fields, or a DtypeArray of
# one scalar field, all byte-aligned) added to close the `struct: pack hhl`
# gap in performance_comparison.py. These exercise it directly rather than
# through whichever dtype the other tests above happen to use.


@pytest.mark.parametrize("count", [0, 1, 5, 50])
def test_flat_record_pack_values_and_unpack_values_handle_various_counts(count):
    tuple_dtype = Dtype("(i16, i16, i32)")
    array_dtype = Dtype("[u8; 3]")
    tuple_records = [(1, -2, 3) for _ in range(count)]
    array_records = [(1, 2, 3) for _ in range(count)]

    tuple_packed = Tibs.from_values(tuple_dtype, tuple_records)
    array_packed = Tibs.from_values(array_dtype, array_records)

    assert len(tuple_packed) == count * tuple_dtype.length
    assert len(array_packed) == count * array_dtype.length
    assert tuple_packed.to_values(tuple_dtype) == tuple_records
    assert array_packed.to_values(array_dtype) == array_records


def test_single_field_tuple_record_round_trips():
    dtype = Dtype("(u8,)")
    values = [(1,), (2,), (255,)]

    packed = Tibs.from_values(dtype, values)

    assert packed.hex == "0102ff"
    assert packed.to_values(dtype) == values


def test_mixed_byte_order_and_signed_record_matches_struct():
    dtype = Dtype("(u16_le, u16_be, i32_le)")
    values = [(0x1234, 0x1234, -1), (0xFFFF, 0x0001, -1_000_000)]

    packed = Tibs.from_values(dtype, values)

    expected = b"".join(
        struct.pack("<H", a) + struct.pack(">H", b) + struct.pack("<i", c)
        for a, b, c in values
    )
    assert packed.bytes == expected
    assert packed.to_values(dtype) == values


def test_array_and_tuple_of_the_same_scalar_byte_match():
    array_dtype = Dtype("[i16; 3]")
    tuple_dtype = Dtype("(i16, i16, i16)")
    values = (100, -200, 300)

    array_packed = array_dtype.pack(values)
    tuple_packed = tuple_dtype.pack(values)

    assert array_packed == tuple_packed
    assert array_dtype.unpack(array_packed) == tuple_dtype.unpack(tuple_packed)


def test_bf16_records_take_the_fast_flat_record_path():
    # bf16 is byte-aligned and numeric, so it has to qualify for the record
    # packer and unpacker exactly as f16 does, not fall back to the generic
    # field-by-field route.
    array_dtype = Dtype("[bf16; 4]")
    tuple_dtype = Dtype("(u8, bf16, bf16_le)")
    array_records = [(1.0, -2.0, 0.125, 256.0), (0.0, 1.0, 1.0, 1.0)]
    tuple_records = [(7, 1.0, 1.0), (255, -2.0, 0.125)]

    array_packed = Tibs.from_values(array_dtype, array_records)
    tuple_packed = Tibs.from_values(tuple_dtype, tuple_records)

    assert array_packed.hex == "3f80c0003e004380" + "00003f803f803f80"
    assert tuple_packed.hex == "073f80803f" + "ffc000003e"
    assert array_packed.to_values(array_dtype) == array_records
    assert tuple_packed.to_values(tuple_dtype) == tuple_records


def test_bf16_and_f16_fields_are_decoded_independently_in_one_record():
    # A record holding both proves the decoder is picked by kind rather than
    # by the 16-bit field width.
    dtype = Dtype("(bf16, f16)")

    packed = dtype.pack((1.0, 1.0))

    assert packed.hex == "3f803c00"
    assert dtype.unpack(packed) == (1.0, 1.0)
    assert Dtype("(f16, bf16)").unpack(packed) == (1.875, 0.0078125)


def test_record_with_a_non_numeric_field_still_packs_and_unpacks_correctly():
    # A `bits4` field has no fast packer/unpacker, so this forces the record
    # back onto the pre-existing generic path field by field.
    dtype = Dtype("(u8, bits4, u8)")
    values = [(1, Tibs("0b1010"), 2), (3, Tibs("0b0101"), 4)]

    packed = Tibs.from_values(dtype, values)

    assert packed.to_values(dtype) == values


def test_nested_compound_dtypes_pack_values_and_unpack_values_correctly():
    # Nesting one level deeper than a flat record (array-of-tuple, tuple-of-
    # tuple) is out of scope for the fast path and must keep using the
    # generic path, unchanged, for the bulk pack_values/to_values methods.
    array_of_tuple = Dtype("[(u8, bool); 2]")
    tuple_of_tuple = Dtype("((u8, u8), u16)")

    array_values = [((1, True), (2, False)), ((3, False), (4, True))]
    tuple_values = [((1, 2), 3), ((4, 5), 6)]

    assert Tibs.from_values(array_of_tuple, array_values).to_values(array_of_tuple) == array_values
    assert Tibs.from_values(tuple_of_tuple, tuple_values).to_values(tuple_of_tuple) == tuple_values


def test_flat_record_dtype_equality_and_hash_are_unaffected_by_the_cached_layout():
    a = Dtype("(i16, i16, i32)")
    b = Dtype("(i16, i16, i32)")

    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}
    assert {a: "x"}[b] == "x"
