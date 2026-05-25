import pytest

from tibs import BitOrder, Endianness, Mutibs, MutableView, Tibs, View


def test_view_constructor_accepts_tibs_and_mutibs():
    t = Tibs("0x1234")
    m = Mutibs("0x1234")

    assert repr(View(t)) == (
        "View(Tibs('0x1234'), Endianness.Unspecified, BitOrder.Msb0)"
    )
    assert repr(View(m)) == (
        "View(Tibs('0x1234'), Endianness.Unspecified, BitOrder.Msb0)"
    )
    assert repr(View(t, byte_order=Endianness.Little)) == (
        "View(Tibs('0x1234'), Endianness.Little, BitOrder.Msb0)"
    )
    assert repr(View(t, bit_order=BitOrder.Lsb0)) == (
        "View(Tibs('0x1234'), Endianness.Unspecified, BitOrder.Lsb0)"
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
        _ = View(t, byte_order=Endianness.Little)
    with pytest.raises(ValueError):
        _ = View(t, bit_order=BitOrder.Lsb0)


def test_view_from_indices_accepts_ranges_and_iterables():
    t = Tibs("0b01101001")

    assert View.from_indices(t, range(0, 8, 2)).bin == "0110"
    assert View.from_indices(t, range(7, -1, -2)).bin == "1001"
    assert View.from_indices(t, (i for i in [1, 3, 5])).bin == "100"


def test_view_from_indices_snapshots_mutibs_source():
    m = Mutibs("0b01101001")
    v = View.from_indices(m, range(0, 8, 2))

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

    assert View(t) == View(m)
    assert View.from_indices(Tibs("0xf0"), range(0, 4)) == View(Tibs("0xf"))

    assert View(t) != t
    assert View(t) != MutableView(m)
    assert View(t) != View(Tibs("0x13"))
    assert View(t) != View(t, byte_order=Endianness.Big)
    assert View(t) != View(t, bit_order=BitOrder.Lsb0)


def test_tibs_view_aliases_create_views():
    t = Tibs("0x1234")

    assert isinstance(t.view(), View)
    assert repr(t.view()) == (
        "View(Tibs('0x1234'), Endianness.Unspecified, BitOrder.Msb0)"
    )
    assert t.view().byte_order == Endianness.Unspecified
    assert t.view().bit_order == BitOrder.Msb0
    assert repr(t.le) == "View(Tibs('0x1234'), Endianness.Little, BitOrder.Msb0)"
    assert t.le.byte_order == Endianness.Little
    assert t.le.bit_order == BitOrder.Msb0
    assert repr(t.be) == "View(Tibs('0x1234'), Endianness.Big, BitOrder.Msb0)"
    assert t.be.byte_order == Endianness.Big
    assert repr(t.lsb0) == (
        "View(Tibs('0x1234'), Endianness.Unspecified, BitOrder.Lsb0)"
    )
    assert t.lsb0.byte_order == Endianness.Unspecified
    assert t.lsb0.bit_order == BitOrder.Lsb0
    assert repr(t.msb0) == (
        "View(Tibs('0x1234'), Endianness.Unspecified, BitOrder.Msb0)"
    )
    assert len(t.le) == len(t)


def test_view_layout_properties_are_read_only():
    v = Tibs("0x1234").le.lsb0

    assert v.byte_order == Endianness.Little
    assert v.bit_order == BitOrder.Lsb0

    with pytest.raises(AttributeError):
        v.byte_order = Endianness.Big
    with pytest.raises(AttributeError):
        v.bit_order = BitOrder.Msb0


def test_mutibs_view_aliases_create_views():
    m = Mutibs("0xaa")

    assert isinstance(m.view(), MutableView)
    assert m.view().byte_order == Endianness.Unspecified
    assert m.view().bit_order == BitOrder.Msb0
    assert repr(m.le) == "MutableView(Mutibs('0xaa'), Endianness.Little, BitOrder.Msb0)"
    assert m.le.byte_order == Endianness.Little
    assert m.le.bit_order == BitOrder.Msb0
    assert repr(m.lsb0) == (
        "MutableView(Mutibs('0xaa'), Endianness.Unspecified, BitOrder.Lsb0)"
    )
    assert m.lsb0.byte_order == Endianness.Unspecified
    assert m.lsb0.bit_order == BitOrder.Lsb0
    assert len(m.lsb0) == len(m)


def test_mutable_view_layout_properties_are_read_only():
    v = Mutibs("0x1234").le.lsb0

    assert v.byte_order == Endianness.Little
    assert v.bit_order == BitOrder.Lsb0

    with pytest.raises(AttributeError):
        v.byte_order = Endianness.Big
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
        "Endianness.Unspecified, BitOrder.Msb0)"
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

    assert MutableView(m1) != View(m1)
    assert MutableView(m1) != m1
    assert MutableView(m1) != MutableView(Mutibs("0x00"))
    assert MutableView(m1) != MutableView(m1, byte_order=Endianness.Big)
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
        "View(Tibs('0xabcd'), Endianness.Little, BitOrder.Lsb0)"
    )

    assert repr(v.be) == (
        "View(Tibs('0xabcd'), Endianness.Big, BitOrder.Lsb0)"
    )
    assert repr(v.msb0) == "View(Tibs('0xabcd'), Endianness.Little, BitOrder.Msb0)"


def test_view_method_can_set_both_layout_fields():
    t = Tibs("0xff")

    v = t.view(byte_order=Endianness.Little, bit_order=BitOrder.Lsb0)
    assert repr(v) == (
        "View(Tibs('0xff'), Endianness.Little, BitOrder.Lsb0)"
    )

    assert repr(v.view(byte_order=Endianness.Big)) == (
        "View(Tibs('0xff'), Endianness.Big, BitOrder.Lsb0)"
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
        _ = t.view(byte_order=Endianness.Little)
    with pytest.raises(ValueError):
        _ = t.view(bit_order=BitOrder.Lsb0)
    with pytest.raises(ValueError):
        _ = m.le
    with pytest.raises(ValueError):
        _ = m.lsb0


def test_view_to_methods_use_byte_order_for_numeric_interpretation():
    t = Tibs.from_u(100, 16, Endianness.Little)

    assert t.view(Endianness.Little).to_u() == 100
    assert t.le.to_u() == 100
    assert t.le.u == 100
    assert t.le.to_i() == 100
    assert t.le.i == 100
    assert t.be.u == t.view(Endianness.Big).to_u()
    assert Tibs("0x0100").le.to_tibs() == Tibs("0x0001")

    f = Tibs.from_f(1.5, 32, Endianness.Little)
    assert f.le.to_f() == 1.5
    assert f.le.f == 1.5


def test_view_to_methods_use_bit_order_for_materialized_bits():
    t = Tibs("0b00010010")
    v = t.lsb0

    assert v.to_bin() == "01001000"
    assert v.bin == "01001000"
    assert v.to_hex() == "48"
    assert v.hex == "48"
    assert v.to_bytes() == b"\x48"
    assert v.bytes == b"\x48"
    assert v.to_u() == 0x48
    assert v.to_tibs() == Tibs("0x48")
    assert v.to_mutibs() == Mutibs("0x48")

    assert Tibs("0x123456").lsb0.to_oct() == Tibs("0x482c6a").to_oct()
    assert Tibs("0x123456").lsb0.oct == Tibs("0x482c6a").oct


def test_mutable_view_reflects_current_source_value():
    m = Mutibs("0x12")
    v = m.lsb0

    assert v.to_hex() == "48"

    m[0] = True
    assert v.to_bin() == "01001001"


def test_explicit_view_constructor_snapshots_mutibs_source():
    m = Mutibs("0x12")
    v = View(m, bit_order=BitOrder.Lsb0)

    assert v.to_hex() == "48"

    m[0] = True
    assert v.to_bin() == "01001000"


def test_mutable_view_write_u_uses_view_layout():
    m = Mutibs.from_u(99, 16, Endianness.Little)

    assert m.le.u == 99

    result = m.le.write_u(45)

    assert result is None
    assert len(m) == 16
    assert m.le.u == 45
    assert m == Mutibs.from_u(45, 16, Endianness.Little)


def test_mutable_view_numeric_property_setters_use_view_layout():
    m = Mutibs.from_zeros(16)

    m.le.u = 0x1234
    assert m.le.u == 0x1234
    assert m == Mutibs.from_u(0x1234, 16, Endianness.Little)

    m.le.i = -300
    assert m.le.i == -300
    assert m == Mutibs.from_i(-300, 16, Endianness.Little)

    f = Mutibs.from_zeros(32)
    f.le.f = 1.5
    assert f.le.f == 1.5
    assert f == Mutibs.from_f(1.5, 32, Endianness.Little)


def test_mutable_view_lsb0_write_u_uses_bit_order_layout():
    m = Mutibs.from_zeros(8)

    m.lsb0.u = 0x12

    assert m.lsb0.u == 0x12
    assert m == Mutibs("0x48")


def test_mutable_view_combined_layout_write_u_roundtrips():
    m = Mutibs.from_zeros(16)

    m.lsb0.le.u = 0x1234

    assert m.lsb0.le.u == 0x1234
    assert m == Mutibs("0x2c48")


def test_mutable_view_representation_setters_preserve_length_and_use_layout():
    m = Mutibs.from_zeros(16)

    assert m.le.write_hex("abcd") is None
    assert len(m) == 16
    assert m.le.hex == "abcd"
    assert m == Mutibs("0xcdab")

    assert m.le.write_bytes(b"\x12\x34") is None
    assert len(m) == 16
    assert m.le.bytes == b"\x12\x34"
    assert m == Mutibs("0x3412")

    bits = Mutibs.from_zeros(8)
    bits.lsb0.bin = "00010010"
    assert len(bits) == 8
    assert bits.lsb0.bin == "00010010"
    assert bits == Mutibs("0x48")

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


def test_mutable_view_field_returns_mutable_view_and_writes_whole_byte_field():
    m = Mutibs("0x23a11234")
    field = m.lsb0.le.field(31, 16)

    assert isinstance(field, MutableView)
    assert len(field) == 16
    assert field.u == 0x2c48

    field.u = 0x5678

    assert field.u == 0x5678
    assert m == Mutibs("0x23a11e6a")
    assert m.lsb0.le.field(31, 16).u == 0x5678


def test_mutable_view_field_repr_includes_source_selection():
    m = Mutibs("0x000fff")
    v = m.be.field(0, 11)
    namespace = {
        "BitOrder": BitOrder,
        "Endianness": Endianness,
        "Mutibs": Mutibs,
        "MutableView": MutableView,
        "range": range,
    }

    assert v.hex == "000"
    assert repr(v) == (
        "MutableView.from_indices(Mutibs('0x000fff'), range(0, 12), "
        "Endianness.Unspecified, BitOrder.Msb0)"
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
    assert m == Mutibs("0x23a11e6a")

    original = m.to_tibs()
    with pytest.raises(ValueError, match="Cannot change the length of a MutableView"):
        field.hex = "123"

    assert m == original


def test_mutable_view_field_endpoint_order_does_not_change_write_mapping():
    m = Mutibs("0x23a11234")

    m.lsb0.le.field(16, 31).write_u(0x5678)

    assert m == Mutibs("0x23a11e6a")
    assert m.lsb0.le.field(31, 16).u == 0x5678


def test_mutable_view_field_writes_non_whole_byte_field():
    m = Mutibs("0x88040410")
    field = m.lsb0.field(31, 26)

    assert len(field) == 6
    assert field.u == 8

    field.u = 42

    assert field.u == 42
    assert m.lsb0.field(31, 26).u == 42


def test_mutable_view_field_write_failure_leaves_source_unchanged():
    m = Mutibs("0x88040410")
    original = m.to_tibs()
    field = m.lsb0.field(31, 26)

    with pytest.raises(OverflowError):
        field.u = 64

    assert m == original


def test_mutable_view_set_errors_leave_value_unchanged():
    m = Mutibs.from_zeros(4)
    original = m.to_tibs()

    with pytest.raises(OverflowError):
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
    v = Tibs("0x88040410").lsb0

    assert v.field(1, 0).u == 0
    assert v.field(6, 2).u == 8
    assert v.field(9, 7).u == 4
    assert v.field(12, 10).u == 4
    assert v.field(17, 13).u == 0
    assert v.field(23, 18).u == 32
    assert v.field(25, 24).u == 0
    assert v.field(31, 26).u == 8


def test_view_field_endpoint_order_does_not_change_value():
    v = Tibs("0x88040410").lsb0

    assert v.field(31, 26).u == v.field(26, 31).u == 8


def test_view_field_preserves_byte_order_for_whole_byte_fields():
    v = Tibs("0x0102").lsb0.le
    field = v.field(15, 0)

    assert repr(field) == "View(Tibs('0x8040'), Endianness.Little, BitOrder.Msb0)"
    assert field.u == Tibs("0x0102").lsb0.field(15, 0).le.u


def test_view_field_drops_byte_order_for_non_whole_byte_fields():
    v = Tibs("0x88040410").lsb0.le
    field = v.field(31, 26)

    assert repr(field) == (
        "View(Tibs('0b001000'), Endianness.Unspecified, BitOrder.Msb0)"
    )
    assert field.u == 8
    assert field.u == Tibs("0x88040410").lsb0.field(31, 26).u


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
    v = View(m, Endianness.Little)
    assert v.hex == '3412'
    m += '0x5'
    assert v.hex == '3412'


def test_bin_views():
    t = Tibs('0x0100')
    assert t.bin == '0000000100000000'
    assert t.msb0.bin == '0000000100000000'
    assert t.lsb0.bin == '1000000000000000'
    assert t.le.bin == '0000000000000001'
    assert t.le.lsb0.bin == '0000000010000000'


def test_bin_field_views():
    t = Tibs('0x0180')
    assert t.msb0.field(0, 7).bin == '00000001'
    assert t.lsb0.field(0, 7).bin == '10000000'
    assert t.le.field(0, 7).bin == '00000001'
    assert t.msb0.field(0, 8).bin == '000000011'
    assert t.lsb0.field(0, 8).bin == '100000000'

    t = Tibs('0x0000f000')
    assert t.msb0.field(16, 31).bin == '1111000000000000'
    assert t.lsb0.field(16, 31).bin == '0000111100000000'
    assert t.msb0.le.field(16, 31).bin == '0000000011110000'
    assert t.msb0.le.lsb0.field(16, 31).bin == '0000000000001111'

    m = Mutibs('0x0000f000')
    assert m.msb0.field(16, 31).bin == '1111000000000000'
    assert m.lsb0.field(16, 31).bin == '0000111100000000'
    assert m.msb0.le.field(16, 31).bin == '0000000011110000'
    assert m.msb0.le.lsb0.field(16, 31).bin == '0000000000001111'


def test_field_value_errors():
    t = Tibs.from_random(100)
    with pytest.raises(ValueError):
        _ = t.field(-1, 1)
    with pytest.raises(ValueError):
        _ = t.to_mutibs().field(5, -11)
    with pytest.raises(ValueError):
        _ = t.le.field(-4, -6)
