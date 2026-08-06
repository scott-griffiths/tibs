import re

import pytest

from tibs import BitOrder, ByteOrder, Dtype, Mutibs, MutableView, Tibs, View


def test_view_constructor_accepts_tibs():
    t = Tibs("0x1234")

    assert repr(View(t)) == (
        "View(Tibs('0x1234'), ByteOrder.Unspecified, BitOrder.Msb0)"
    )
    assert repr(View(t, byte_order=ByteOrder.Little)) == (
        "View(Tibs('0x1234'), ByteOrder.Little, BitOrder.Msb0)"
    )
    assert repr(View(t, bit_order=BitOrder.Lsb0)) == (
        "View(Tibs('0x1234'), ByteOrder.Unspecified, BitOrder.Lsb0)"
    )


def test_view_constructor_rejects_promotable_non_tibs_sources():
    with pytest.raises(TypeError):
        _ = View("0xff")
    with pytest.raises(TypeError):
        _ = View(b"\xff")
    with pytest.raises(TypeError):
        _ = View([1, 0, 1])


def test_view_constructor_validates_byte_oriented_layout():
    t = Tibs("0b101")

    assert isinstance(View(t), View)

    with pytest.raises(ValueError):
        _ = View(t, byte_order=ByteOrder.Little)
    with pytest.raises(ValueError):
        _ = View(t, bit_order=BitOrder.Lsb0)


def test_view_from_indices_accepts_ranges_and_iterables():
    t = Tibs("0b01101001")

    assert View.from_indices(t, range(0, 8, 2)).bin == "0110"
    assert View.from_indices(t, range(7, -1, -2)).bin == "1001"
    assert View.from_indices(t, (i for i in [1, 3, 5])).bin == "100"


def test_view_from_indices_rejects_mutibs_source():
    m = Mutibs("0b01101001")

    with pytest.raises(TypeError, match="must be a Tibs"):
        View.from_indices(m, range(0, 8, 2))

    # The explicit copy is accepted, and does not track the Mutibs.
    v = View.from_indices(m.to_tibs(), range(0, 8, 2))
    assert v.bin == "0110"
    m[2] = False
    assert v.bin == "0110"


def test_view_from_indices_validates_source_indices():
    t = Tibs("0b0110")

    with pytest.raises(ValueError, match="too short"):
        _ = View.from_indices(t, [4])

    with pytest.raises(ValueError, match="duplicates"):
        _ = View.from_indices(t, [0, 0])


def test_view_equality_uses_type_source_and_layout():
    t = Tibs("0x12")
    m = Mutibs("0x12")

    assert View(t) == View(m.to_tibs())
    assert View.from_indices(Tibs("0xf0"), range(0, 4)) == View(Tibs("0xf"))

    assert View(t) != t
    assert View(t) != MutableView(m)
    assert View(t) != View(Tibs("0x13"))
    assert View(t) != View(t, byte_order=ByteOrder.Big)
    assert View(t) != View(t, bit_order=BitOrder.Lsb0)


def test_tibs_view_aliases_create_views():
    t = Tibs("0x1234")

    assert isinstance(t.view(), View)
    assert repr(t.view()) == (
        "View(Tibs('0x1234'), ByteOrder.Unspecified, BitOrder.Msb0)"
    )
    assert t.view().byte_order == ByteOrder.Unspecified
    assert t.view().bit_order == BitOrder.Msb0
    assert repr(t.le) == "View(Tibs('0x1234'), ByteOrder.Little, BitOrder.Msb0)"
    assert t.le.byte_order == ByteOrder.Little
    assert t.le.bit_order == BitOrder.Msb0
    assert repr(t.be) == "View(Tibs('0x1234'), ByteOrder.Big, BitOrder.Msb0)"
    assert t.be.byte_order == ByteOrder.Big
    assert repr(t.lsb0) == (
        "View(Tibs('0x1234'), ByteOrder.Unspecified, BitOrder.Lsb0)"
    )
    assert t.lsb0.byte_order == ByteOrder.Unspecified
    assert t.lsb0.bit_order == BitOrder.Lsb0
    assert repr(t.msb0) == (
        "View(Tibs('0x1234'), ByteOrder.Unspecified, BitOrder.Msb0)"
    )
    assert len(t.le) == len(t)


def test_view_layout_properties_are_read_only():
    v = Tibs("0x1234").le.lsb0

    assert v.byte_order == ByteOrder.Little
    assert v.bit_order == BitOrder.Lsb0

    with pytest.raises(AttributeError):
        v.byte_order = ByteOrder.Big
    with pytest.raises(AttributeError):
        v.bit_order = BitOrder.Msb0


def test_mutibs_view_aliases_create_views():
    m = Mutibs("0xaa")

    assert isinstance(m.view(), MutableView)
    assert m.view().byte_order == ByteOrder.Unspecified
    assert m.view().bit_order == BitOrder.Msb0
    assert repr(m.le) == "MutableView(Mutibs('0xaa'), ByteOrder.Little, BitOrder.Msb0)"
    assert m.le.byte_order == ByteOrder.Little
    assert m.le.bit_order == BitOrder.Msb0
    assert repr(m.lsb0) == (
        "MutableView(Mutibs('0xaa'), ByteOrder.Unspecified, BitOrder.Lsb0)"
    )
    assert m.lsb0.byte_order == ByteOrder.Unspecified
    assert m.lsb0.bit_order == BitOrder.Lsb0
    assert len(m.lsb0) == len(m)


def test_mutable_view_layout_properties_are_read_only():
    v = Mutibs("0x1234").le.lsb0

    assert v.byte_order == ByteOrder.Little
    assert v.bit_order == BitOrder.Lsb0

    with pytest.raises(AttributeError):
        v.byte_order = ByteOrder.Big
    with pytest.raises(AttributeError):
        v.bit_order = BitOrder.Msb0


def test_mutable_view_constructor_rejects_source_indices():
    m = Mutibs("0xaa")

    with pytest.raises(TypeError):
        _ = MutableView(m, source_indices=range(0, 8))


def test_mutable_view_from_indices_accepts_ranges_and_iterables():
    m = Mutibs("0x00")
    v = MutableView.from_indices(m, range(0, 8, 2))

    assert repr(v) == (
        "MutableView.from_indices(Mutibs('0x00'), range(0, 8, 2), "
        "ByteOrder.Unspecified, BitOrder.Msb0)"
    )
    assert v.bin == "0000"

    v.bin = "1111"
    assert m == Mutibs("0xaa")

    reverse = MutableView.from_indices(m, (i for i in [7, 5, 3, 1]))
    assert reverse.bin == "0000"
    reverse.bin = "1111"
    assert m == Mutibs("0xff")


def test_mutable_view_from_indices_validates_source_indices():
    m = Mutibs("0b0110")

    with pytest.raises(ValueError, match="too short"):
        _ = MutableView.from_indices(m, [4])

    with pytest.raises(ValueError, match="duplicates"):
        _ = MutableView.from_indices(m, [0, 0])


def test_mutable_view_equality_uses_type_source_layout_and_source_indices():
    m1 = Mutibs("0xff")
    m2 = Mutibs("0xff")

    assert MutableView(m1) == MutableView(m2)
    assert MutableView(m1) == MutableView.from_indices(m1, range(len(m1)))

    assert MutableView(m1) != View(m1.to_tibs())
    assert MutableView(m1) != m1
    assert MutableView(m1) != MutableView(Mutibs("0x00"))
    assert MutableView(m1) != MutableView(m1, byte_order=ByteOrder.Big)
    assert MutableView(m1) != MutableView(m1, bit_order=BitOrder.Lsb0)
    assert MutableView.from_indices(m1, range(0, 4)) != (
        MutableView.from_indices(m1, range(4, 8))
    )

    v1 = MutableView(m1)
    v2 = MutableView(m2)
    assert v1 == v2

    m2[0] = False
    assert v1 != v2


def test_view_chaining_preserves_source_and_updates_layout():
    t = Tibs("0xabcd")

    v = t.le.lsb0
    assert repr(v) == (
        "View(Tibs('0xabcd'), ByteOrder.Little, BitOrder.Lsb0)"
    )

    assert repr(v.be) == (
        "View(Tibs('0xabcd'), ByteOrder.Big, BitOrder.Lsb0)"
    )
    assert repr(v.msb0) == "View(Tibs('0xabcd'), ByteOrder.Little, BitOrder.Msb0)"


def test_view_method_can_set_both_layout_fields():
    t = Tibs("0xff")

    v = t.view(byte_order=ByteOrder.Little, bit_order=BitOrder.Lsb0)
    assert repr(v) == (
        "View(Tibs('0xff'), ByteOrder.Little, BitOrder.Lsb0)"
    )

    assert repr(v.view(byte_order=ByteOrder.Big)) == (
        "View(Tibs('0xff'), ByteOrder.Big, BitOrder.Lsb0)"
    )


def test_byte_oriented_view_requires_whole_byte_source():
    t = Tibs("0b101")
    m = Mutibs("0b101")

    assert isinstance(t.view(), View)
    assert isinstance(t.msb0, View)
    assert isinstance(m.view(), MutableView)
    assert isinstance(m.msb0, MutableView)

    with pytest.raises(ValueError):
        _ = t.le
    with pytest.raises(ValueError):
        _ = t.be
    with pytest.raises(ValueError):
        _ = t.lsb0
    with pytest.raises(ValueError):
        _ = t.view(byte_order=ByteOrder.Little)
    with pytest.raises(ValueError):
        _ = t.view(bit_order=BitOrder.Lsb0)
    with pytest.raises(ValueError):
        _ = m.le
    with pytest.raises(ValueError):
        _ = m.lsb0


def test_view_to_methods_use_byte_order_for_numeric_interpretation():
    t = Tibs.from_u(100, 16, ByteOrder.Little)

    assert t.view(ByteOrder.Little).to_u() == 100
    assert t.le.to_u() == 100
    assert t.le.u == 100
    assert t.le.to_i() == 100
    assert t.le.i == 100
    assert t.be.u == t.view(ByteOrder.Big).to_u()
    assert Tibs("0x0100").le.to_tibs() == Tibs("0x0001")

    f = Tibs.from_f(1.5, 32, ByteOrder.Little)
    assert f.le.to_f() == 1.5
    assert f.le.f == 1.5


def test_view_to_methods_use_lsb0_value_order():
    single_byte = Tibs("0b00010010").lsb0

    assert single_byte.to_bin() == "00010010"
    assert single_byte.to_hex() == "12"
    assert single_byte.to_u() == 0x12

    t = Tibs("0x1234")
    v = t.lsb0

    assert v.to_bin() == "0011010000010010"
    assert v.bin == "0011010000010010"
    assert v.to_hex() == "3412"
    assert v.hex == "3412"
    assert v.to_bytes() == b"\x34\x12"
    assert bytes(v) == b"\x34\x12"
    assert v.bytes == b"\x34\x12"
    assert v.to_u() == 0x3412
    assert v.to_tibs() == Tibs("0x3412")
    assert v.to_mutibs() == Mutibs("0x3412")

    assert Tibs("0x123456").lsb0.to_oct() == Tibs("0x563412").to_oct()
    assert Tibs("0x123456").lsb0.oct == Tibs("0x563412").oct


def test_view_to_value_reaches_dtypes_the_privileged_kinds_do_not():
    # The point of to_value on a view: any dtype, not just the nine
    # interpretations that happen to have their own view method.
    t = Tibs("0x0000803f")

    assert t.le.to_value("f32") == 1.0
    assert t.le.to_value(Dtype("f32")) == 1.0
    assert t.le.to_value("[u8; 4]") == (0x3F, 0x80, 0x00, 0x00)
    assert t.le.to_value("(u8, u24)") == (0x3F, 0x800000)
    assert t.lsb0.to_value("bits32") == Tibs("0x3f800000")
    assert Tibs("0b1").view().to_value("bool") is True


def test_view_to_value_reaches_bf16_which_has_no_view_property():
    # to_f is IEEE only and picks its format from the length, so a view can
    # only reach bfloat16 through a dtype.
    t = Tibs("0x803f")

    assert t.le.to_value("bf16") == 1.0
    assert t.to_value("bf16") != 1.0
    assert t.le.to_f() != 1.0

    with pytest.raises(ValueError, match="through a view that specifies its own byte order"):
        t.le.to_value("bf16_le")

    m = Mutibs.from_zeros(16)
    m.le.write_value("bf16", 1.0)
    assert m.hex == "803f"
    assert m.le.to_value("bf16") == 1.0
    assert m.to_value("bf16") != 1.0


@pytest.mark.parametrize("spec", ["0x0100", "0x23a11234", "0xdeadbeefcafe"])
@pytest.mark.parametrize("byte_order", [ByteOrder.Unspecified, ByteOrder.Little, ByteOrder.Big])
@pytest.mark.parametrize("bit_order", [BitOrder.Msb0, BitOrder.Lsb0])
def test_view_to_value_matches_materializing_the_view_first(spec, byte_order, bit_order):
    # The governing rule for every view conversion: the layout is applied
    # first, and start/end are positions in the value the view denotes.
    view = Tibs(spec).view(byte_order, bit_order)
    materialized = view.to_tibs()

    for dtype in ("u16", "i16", "hex4", "bits8", "bool"):
        length = Dtype(dtype).length
        for start in range(0, len(view) - length + 1, 4):
            assert view.to_value(dtype, start, start + length) == materialized.to_value(
                dtype, start, start + length
            )


def test_view_to_value_takes_byte_order_from_one_place_only():
    # The view's layout is applied first, so a '_le' dtype inside an 'le' view
    # would be a second byte order and would swap twice. Say it once.
    t = Tibs("0x0100")

    assert t.to_value("u16") == 256
    assert t.to_value("u16_le") == 1
    assert t.le.to_value("u16") == 1

    with pytest.raises(ValueError, match="through a view that specifies its own byte order"):
        t.le.to_value("u16_le")


@pytest.mark.parametrize("dtype", ["u16_le", "u16_be", "bf16_le", "(u8, u8_le)", "[u16_le; 1]"])
@pytest.mark.parametrize("view_name", ["le", "be", "lsb0"])
def test_byte_ordered_view_refuses_a_byte_ordered_dtype_at_any_nesting_depth(view_name, dtype):
    view = getattr(Tibs("0x0100"), view_name)
    mutable = getattr(Mutibs("0x0100"), view_name)

    with pytest.raises(ValueError, match=f"Cannot use the dtype '{re.escape(str(Dtype(dtype)))}'"):
        view.to_value(dtype)
    with pytest.raises(ValueError, match="through a view that specifies its own byte order"):
        mutable.to_value(dtype)
    with pytest.raises(ValueError, match="through a view that specifies its own byte order"):
        mutable.write_value(dtype, 1 if "(" not in dtype and "[" not in dtype else (1, 1))


def test_a_view_that_states_no_layout_still_passes_a_byte_ordered_dtype_through():
    # A plain view() makes no claim about byte order, so it stays a
    # pass-through to the reading Tibs.to_value would give.
    t = Tibs("0x0100")

    assert t.view().to_value("u16_le") == t.to_value("u16_le") == 1
    assert t.msb0.to_value("(u8, u8)") == (1, 0)
    assert Mutibs("0x0100").view().to_value("u16_le") == 1

    m = Mutibs.from_zeros(16)
    m.view().write_value("u16_le", 1)
    assert m.hex == "0100"


def test_field_views_inherit_the_byte_order_rule_from_their_parent():
    # field() keeps the parent byte order for whole-byte fields and drops it
    # otherwise, and the dtype rule has to follow that either way.
    header = Tibs("0x23a11234").lsb0.le

    assert header.field(31, 16).byte_order == ByteOrder.Little
    with pytest.raises(ValueError, match="through a view that specifies its own byte order"):
        header.field(31, 16).to_value("u16_le")

    narrow = header.field(11, 0)
    assert narrow.byte_order == ByteOrder.Unspecified
    assert narrow.to_value("u12") == 291


def test_view_to_value_rejects_a_range_that_is_not_the_dtype_length():
    with pytest.raises(ValueError, match="dtype with length"):
        Tibs("0xff").le.to_value("u4")
    with pytest.raises(ValueError, match="Invalid slice positions"):
        Tibs("0xff").le.to_value("u4", 0, 99)
    with pytest.raises(TypeError, match="dtype must be a Dtype instance"):
        Tibs("0xff").view().to_value(8)


def test_view_whole_value_matches_full_width_field():
    t = Tibs("0x0100")
    views = [t.view(), t.le, t.lsb0, t.lsb0.le]

    for view in views:
        field = view.field(0, len(view) - 1)
        assert view.bin == field.bin
        assert view.hex == field.hex
        assert view.u == field.u


def test_lsb0_little_endian_bit_zero_displays_as_value_lsb():
    m = Mutibs.from_zeros(32)
    v = m.lsb0.le

    v.field(0, 0).u = 1

    assert v.bin == "00000000000000000000000000000001"
    assert v.u == 1
    assert m.hex == "01000000"


@pytest.mark.parametrize("view_name", ["lsb0", "lsb0.le"])
def test_mutable_lsb0_whole_view_write_matches_bit_zero_field(view_name):
    via_field = Mutibs.from_zeros(32)
    via_whole = Mutibs.from_zeros(32)

    field_view = via_field.lsb0.le if view_name == "lsb0.le" else via_field.lsb0
    whole_view = via_whole.lsb0.le if view_name == "lsb0.le" else via_whole.lsb0

    field_view.field(0, 0).u = 1
    whole_view.u = 1

    assert whole_view.bin == "00000000000000000000000000000001"
    assert via_whole == via_field == Mutibs("0x01000000")


def test_views_do_not_have_to_padded_bytes():
    assert not hasattr(Tibs("0x12").lsb0, "to_padded_bytes")
    assert not hasattr(Mutibs("0x12").lsb0, "to_padded_bytes")


def test_mutable_view_reflects_current_source_value():
    m = Mutibs("0x12")
    v = m.lsb0

    assert v.to_hex() == "12"

    m[0] = True
    assert v.to_bin() == "10010010"


def test_explicit_view_constructor_rejects_mutibs_source():
    m = Mutibs("0x12")

    with pytest.raises(TypeError, match="must be a Tibs"):
        View(m, bit_order=BitOrder.Lsb0)

    # The explicit copy is accepted, and does not track the Mutibs.
    v = View(m.to_tibs(), bit_order=BitOrder.Lsb0)
    assert v.to_hex() == "12"

    m[0] = True
    assert v.to_bin() == "00010010"


def test_mutable_view_write_u_uses_view_layout():
    m = Mutibs.from_u(99, 16, ByteOrder.Little)

    assert m.le.u == 99

    result = m.le.write_u(45)

    assert result is None
    assert len(m) == 16
    assert m.le.u == 45
    assert m == Mutibs.from_u(45, 16, ByteOrder.Little)


def test_mutable_view_numeric_property_setters_use_view_layout():
    m = Mutibs.from_zeros(16)

    m.le.u = 0x1234
    assert m.le.u == 0x1234
    assert m == Mutibs.from_u(0x1234, 16, ByteOrder.Little)

    m.le.i = -300
    assert m.le.i == -300
    assert m == Mutibs.from_i(-300, 16, ByteOrder.Little)

    f = Mutibs.from_zeros(32)
    f.le.f = 1.5
    assert f.le.f == 1.5
    assert f == Mutibs.from_f(1.5, 32, ByteOrder.Little)


def test_mutable_view_lsb0_write_u_uses_bit_order_layout():
    m = Mutibs.from_zeros(8)

    m.lsb0.u = 0x12

    assert m.lsb0.u == 0x12
    assert m == Mutibs("0x12")


def test_mutable_view_combined_layout_write_u_roundtrips():
    m = Mutibs.from_zeros(16)

    m.lsb0.le.u = 0x1234

    assert m.lsb0.le.u == 0x1234
    assert m == Mutibs("0x3412")


def test_mutable_view_representation_setters_preserve_length_and_use_layout():
    m = Mutibs.from_zeros(16)

    assert m.le.write_hex("abcd") is None
    assert len(m) == 16
    assert m.le.hex == "abcd"
    assert m == Mutibs("0xcdab")

    assert m.le.write_bytes(b"\x12\x34") is None
    assert len(m) == 16
    assert bytes(m.le) == b"\x12\x34"
    assert m.le.bytes == b"\x12\x34"
    assert m == Mutibs("0x3412")

    bits = Mutibs.from_zeros(8)
    bits.lsb0.bin = "00010010"
    assert len(bits) == 8
    assert bytes(bits.lsb0) == b"\x12"
    assert bits.lsb0.bin == "00010010"
    assert bits == Mutibs("0x12")

    octal = Mutibs.from_zeros(6)
    octal.view().oct = "17"
    assert len(octal) == 6
    assert octal == Mutibs("0b001111")


def test_mutable_view_representation_setters_reject_width_changes():
    m = Mutibs("0xffff")
    original = m.to_tibs()

    with pytest.raises(ValueError, match="Cannot change the length of a MutableView"):
        m.view().bin = "101"

    assert m == original

    with pytest.raises(ValueError, match="Cannot change the length of a MutableView"):
        m.le.hex = "123"

    assert m == original

    with pytest.raises(ValueError, match="Cannot change the length of a MutableView"):
        m.lsb0.bytes = b"\x00"

    assert m == original


def test_mutable_view_value_methods_round_trip_through_the_view_layout():
    m = Mutibs.from_zeros(32)

    assert m.le.write_value("f32", 1.0) is None
    assert m.hex == "0000803f"
    assert m.le.to_value("f32") == 1.0
    assert m.le.to_value("[u8; 4]") == (0x3F, 0x80, 0x00, 0x00)
    assert m.to_value("f32") != 1.0

    m.lsb0.le.write_value("(u16, u16)", (0x1122, 0x3344))
    assert m.hex == "44332211"
    assert m.lsb0.le.to_value("(u16, u16)") == (0x1122, 0x3344)


def test_mutable_view_write_value_writes_a_field_in_place():
    m = Mutibs.from_bytes(bytes.fromhex("07 01 00 00 44 33 22 11"))
    field = m.lsb0.le.field(63, 32)

    assert field.to_value("u32") == 0x11223344

    field.write_value("u32", 0xDEADBEEF)

    assert len(m) == 64
    assert m.hex == "07010000efbeadde"
    assert m.lsb0.le.field(63, 32).to_value("u32") == 0xDEADBEEF


def test_mutable_view_write_value_is_fixed_width_and_leaves_the_source_alone():
    m = Mutibs("0xffff")
    original = m.to_tibs()

    with pytest.raises(ValueError, match="Cannot change the length of a MutableView"):
        m.le.write_value("u8", 1)

    assert m == original

    with pytest.raises(ValueError):
        m.le.write_value("u16", 1 << 16)

    assert m == original

    with pytest.raises(TypeError, match="dtype must be a Dtype instance"):
        m.le.write_value(16, 1)

    assert m == original


def test_mutable_view_field_returns_mutable_view_and_writes_whole_byte_field():
    m = Mutibs("0x23a11234")
    field = m.lsb0.le.field(31, 16)

    assert isinstance(field, MutableView)
    assert len(field) == 16
    assert field.u == 0x3412

    field.u = 0x5678

    assert field.u == 0x5678
    assert m == Mutibs("0x23a17856")
    assert m.lsb0.le.field(31, 16).u == 0x5678


def test_mutable_view_field_repr_includes_source_selection():
    m = Mutibs("0x000fff")
    v = m.be.field(0, 11)
    namespace = {
        "BitOrder": BitOrder,
        "ByteOrder": ByteOrder,
        "Mutibs": Mutibs,
        "MutableView": MutableView,
        "range": range,
    }

    assert v.hex == "000"
    assert repr(v) == (
        "MutableView.from_indices(Mutibs('0x000fff'), range(0, 12), "
        "ByteOrder.Unspecified, BitOrder.Msb0)"
    )
    recreated = eval(repr(v), namespace)
    assert isinstance(recreated, MutableView)
    assert recreated.hex == v.hex
    assert len(recreated) == len(v)


def test_mutable_view_field_representation_setters_are_fixed_width():
    m = Mutibs("0x23a11234")
    field = m.lsb0.le.field(31, 16)

    field.hex = "5678"

    assert field.hex == "5678"
    assert field.u == 0x5678
    assert m == Mutibs("0x23a17856")

    original = m.to_tibs()
    with pytest.raises(ValueError, match="Cannot change the length of a MutableView"):
        field.hex = "123"

    assert m == original


def test_mutable_view_field_endpoint_order_does_not_change_write_mapping():
    m = Mutibs("0x23a11234")

    m.lsb0.le.field(16, 31).write_u(0x5678)

    assert m == Mutibs("0x23a17856")
    assert m.lsb0.le.field(31, 16).u == 0x5678


def test_mutable_view_field_writes_non_whole_byte_field():
    m = Mutibs("0x88040410")
    field = m.lsb0.field(31, 26)

    assert len(field) == 6
    assert field.u == 4

    field.u = 42

    assert field.u == 42
    assert m == Mutibs("0x880404a8")
    assert m.lsb0.field(31, 26).u == 42


def test_mutable_view_lsb0_field_writes_value_order():
    m = Mutibs.from_zeros(8)
    field = m.lsb0.field(0, 3)

    field.bin = "0001"

    assert field.bin == "0001"
    assert field.u == 1
    assert m == Mutibs("0x01")

    field.u = 8

    assert field.bin == "1000"
    assert field.u == 8
    assert m == Mutibs("0x08")


def test_mutable_view_field_write_failure_leaves_source_unchanged():
    m = Mutibs("0x88040410")
    original = m.to_tibs()
    field = m.lsb0.field(31, 26)

    with pytest.raises(ValueError):
        field.u = 64

    assert m == original


def test_mutable_view_set_errors_leave_value_unchanged():
    m = Mutibs.from_zeros(4)
    original = m.to_tibs()

    with pytest.raises(ValueError):
        m.view().write_u(16)

    assert m == original

    f = Mutibs.from_zeros(24)
    original = f.to_tibs()

    with pytest.raises(ValueError):
        f.view().write_f(1.25)

    assert f == original


def test_mutable_view_revalidates_layout_after_source_length_change():
    m = Mutibs.from_zeros(8)
    v = m.le

    m.append(1)

    with pytest.raises(ValueError):
        _ = v.u

    with pytest.raises(ValueError):
        v.write_u(1)


def test_view_field_extracts_lsb0_spec_labels():
    literal = Tibs("0x01").lsb0

    assert literal.field(0, 3).bin == "0001"
    assert literal.field(0, 3).u == 1

    v = Tibs("0x88040410").lsb0

    assert v.field(1, 0).u == 0
    assert v.field(6, 2).u == 2
    assert v.field(9, 7).u == 1
    assert v.field(12, 10).u == 1
    assert v.field(17, 13).u == 0
    assert v.field(23, 18).u == 1
    assert v.field(25, 24).u == 0
    assert v.field(31, 26).u == 4


def test_view_decodes_published_lsb0_little_endian_ebpf_instruction():
    # Linux's eBPF ISA documentation gives these bytes as a little-endian
    # instruction decoded as "r1 += 0x11223344".
    # https://docs.kernel.org/bpf/standardization/instruction-set.html
    instruction = Tibs.from_bytes(
        bytes.fromhex("07 01 00 00 44 33 22 11")
    ).lsb0.le

    assert instruction.field(2, 0).u == 0x7  # BPF_ALU64 instruction class.
    assert instruction.field(3, 3).u == 0x0  # BPF_K source mode.
    assert instruction.field(7, 4).u == 0x0  # BPF_ADD operation code.
    assert instruction.field(11, 8).u == 1  # dst_reg.
    assert instruction.field(15, 12).u == 0  # src_reg.
    assert instruction.field(31, 16).i == 0
    assert instruction.field(63, 32).u == 0x11223344


def test_view_field_endpoint_order_does_not_change_value():
    v = Tibs("0x88040410").lsb0

    assert v.field(31, 26).u == v.field(26, 31).u == 4


def test_view_field_preserves_byte_order_for_whole_byte_fields():
    v = Tibs("0x0102").lsb0.le
    field = v.field(15, 0)

    assert repr(field) == "View(Tibs('0x0102'), ByteOrder.Little, BitOrder.Msb0)"
    assert field.u == 0x0201


def test_view_field_drops_byte_order_for_non_whole_byte_fields():
    v = Tibs("0x88040410").lsb0.le
    field = v.field(31, 26)

    assert repr(field) == (
        "View(Tibs('0b000100'), ByteOrder.Unspecified, BitOrder.Msb0)"
    )
    assert field.u == 4
    assert field.u == Tibs("0x88040410").lsb0.field(31, 26).u

    payload = Tibs("0x23a11234").lsb0.le.field(11, 0)
    assert repr(payload) == (
        "View(Tibs('0x123'), ByteOrder.Unspecified, BitOrder.Msb0)"
    )
    assert payload.u == 0x123


def test_view_field_uses_msb0_labels_by_default():
    v = Tibs("0x88")

    assert v.view().field(7, 4).bin == "1000"
    assert v.view().field(3, 0).bin == "1000"


def test_tibs_and_mutibs_field_methods_use_msb0_labels():
    t = Tibs("0x0180")

    field = t.field(0, 8)

    assert isinstance(field, View)
    assert field.bin == "000000011"
    assert field == t.msb0.field(0, 8)

    m = Mutibs("0x0000")
    field = m.field(8, 15)

    assert isinstance(field, MutableView)
    assert field.bin == "00000000"

    field.hex = "f0"

    assert field.bin == "11110000"
    assert m == Mutibs("0x00f0")
    assert m.field(8, 15) == m.msb0.field(8, 15)


def test_view_field_validates_labels():
    v = Tibs("0xff").lsb0

    with pytest.raises(ValueError):
        _ = v.field(8, 0)


def test_view_field_rejects_empty_view():
    v = Tibs().view()

    with pytest.raises(ValueError, match="empty view"):
        _ = v.field(0, 0)


def test_mutibs_views():
    m = Mutibs('0x1234')
    with pytest.raises(TypeError, match="must be a Tibs"):
        View(m, ByteOrder.Little)

    v = View(m.to_tibs(), ByteOrder.Little)
    assert v.hex == '3412'
    m += '0x5'
    assert v.hex == '3412'

    # The live view of the same Mutibs does track it.
    assert len(m.view()) == 20


def test_bin_views():
    t = Tibs('0x0100')
    assert t.bin == '0000000100000000'
    assert t.msb0.bin == '0000000100000000'
    assert t.lsb0.bin == '0000000000000001'
    assert t.le.bin == '0000000000000001'
    assert t.le.lsb0.bin == '0000000000000001'


def test_bin_field_views():
    t = Tibs('0x0180')
    assert t.msb0.field(0, 7).bin == '00000001'
    assert t.lsb0.field(0, 7).bin == '00000001'
    assert t.le.field(0, 7).bin == '00000001'
    assert t.msb0.field(0, 8).bin == '000000011'
    assert t.lsb0.field(0, 8).bin == '000000001'

    t = Tibs('0x0000f000')
    assert t.msb0.field(16, 31).bin == '1111000000000000'
    assert t.lsb0.field(16, 31).bin == '0000000011110000'
    assert t.msb0.le.field(16, 31).bin == '0000000011110000'
    assert t.msb0.le.lsb0.field(16, 31).bin == '0000000011110000'

    m = Mutibs('0x0000f000')
    assert m.msb0.field(16, 31).bin == '1111000000000000'
    assert m.lsb0.field(16, 31).bin == '0000000011110000'
    assert m.msb0.le.field(16, 31).bin == '0000000011110000'
    assert m.msb0.le.lsb0.field(16, 31).bin == '0000000011110000'


def test_field_value_errors():
    t = Tibs.from_random(100)
    with pytest.raises(ValueError):
        _ = t.field(-1, 1)
    with pytest.raises(ValueError):
        _ = t.to_mutibs().field(5, -11)
    with pytest.raises(ValueError):
        _ = t.le.field(-4, -6)
