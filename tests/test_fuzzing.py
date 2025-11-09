#!/usr/bin/env python

from bitformat import Format, DtypeSingle, DtypeArray, Bits, Field, Register
from bitformat._common import DtypeKind
from hypothesis import given
import hypothesis.strategies as st
import math


def get_allowed_length(dtype_kind, length):
    al = Register().kind_to_def[dtype_kind].allowed_sizes
    if al and al.sizes:
        if al.sizes[-1] is Ellipsis:
            return al.sizes[1] * length
        else:
            return al.sizes[length % len(al.sizes)]
    return length


def compare_fields(f, f2):
    if isinstance(f.value, float) and math.isnan(f.value):
        assert math.isnan(f2.value)
    assert f == f2, f"Fields are not equal: {f} != {f2}"


@given(
    dtype_kind=st.sampled_from(sorted(Register().kind_to_def.keys(), key=lambda x: x.value)),
    length=st.integers(1, 100),
    int_value=st.integers(0, 2**800 - 1),
)
def test_field_consistency(dtype_kind, length, int_value):
    length = get_allowed_length(dtype_kind, length)
    f = Field.from_params(DtypeSingle.from_params(dtype_kind, length))
    f2 = Field.from_string(str(f))
    compare_fields(f, f2)

    # Create some bits of the right length
    bits_per_character = Register().kind_to_def[dtype_kind].bits_per_character
    if bits_per_character is not None:
        length *= bits_per_character
    b = Bits.from_dtype("u800", int_value)[0:length]
    f.parse(b)
    if dtype_kind != DtypeKind.PAD:
        assert f.to_bits() == b
    v = f.value
    if v is not None:
        f2.value = v
    else:
        f2.clear()
    if dtype_kind is not DtypeKind.PAD and not (isinstance(v, float) and math.isnan(v)):
        assert f.to_bits() == f2.to_bits()
        f3 = eval(repr(f))
        compare_fields(f, f3)


@given(
    dtype_kind=st.sampled_from(sorted(Register().kind_to_def.keys(), key=lambda x: x.value)),
    length=st.integers(1, 5),
    int_value=st.integers(0, 2**160 - 1),
    items=st.integers(1, 4),
)
def test_field_array_consistency(dtype_kind, length, int_value, items):
    length = get_allowed_length(dtype_kind, length)

    f = Field.from_params(DtypeArray.from_params(dtype_kind, length, items))
    f2 = Field.from_string(str(f))
    assert f == f2

    # Create some bits of the right length
    bits_per_character = Register().kind_to_def[dtype_kind].bits_per_character
    if bits_per_character is not None:
        length *= bits_per_character
    b = Bits.from_dtype("u320", int_value)[0: length * items]
    f.parse(b)
    assert f.to_bits() == b
    if not isinstance(f.value[0], float) and f.dtype.kind is not DtypeKind.PAD:
        # Can't compare NaN or pad
        f2.pack(f.value)
        assert f.to_bits() == f2.to_bits()
        assert f.value == f2.value


@given(
    dtype_kinds=st.lists(
        st.sampled_from(
            [k for k in sorted(Register().kind_to_def.keys(), key=lambda x: x.value) if k is not DtypeKind.PAD]
        ),
        min_size=5,
        max_size=5,
    ),
    lengths=st.lists(st.integers(1, 5), min_size=5, max_size=5),
)
def test_format_consistency(dtype_kinds, lengths):
    bits_per_characters = [
        Register().kind_to_def[dtype_kind].bits_per_character
        for dtype_kind in dtype_kinds
    ]
    bits_per_characters = [b if b is not None else 1 for b in bits_per_characters]
    als = []
    for al, length in zip(
        [
            Register().kind_to_def[dtype_kinds].allowed_sizes
            for dtype_kinds in dtype_kinds
        ],
        lengths,
    ):
        if al.sizes:
            if al.sizes[-1] is Ellipsis:
                als.append(al.sizes[1] * length)
            else:
                als.append(al.sizes[length % len(al.sizes)])
        else:
            als.append(length)

    zipped = list(zip(dtype_kinds, als, bits_per_characters))
    for i in range(6):
        dtypes = [
            DtypeSingle.from_params(dtype_kind, length * bits_per_character)
            for dtype_kind, length, bits_per_character in zipped[:i]
        ]
        f = Format.from_params([Field.from_params(dtype) for dtype in dtypes])
        f2 = f
        assert f == f2
        # Create some bits of the right length
        b = Bits.from_ones(f.bit_length)
        f.parse(b)
        assert f.to_bits() == b
