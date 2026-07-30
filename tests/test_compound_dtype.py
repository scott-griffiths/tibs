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

    with pytest.raises(OverflowError) as exc_info:
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
