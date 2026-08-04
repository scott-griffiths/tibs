#!/usr/bin/env python
"""Contract tests for the shape of the public API rather than its behaviour.

A parameter that can be passed by keyword is part of the API forever, so the
"subject" of each call is positional-only and only the optional modifiers
(``start``, ``end``, ``byte_aligned``, ``mask``, ``byte_order`` and friends)
are keyword-reachable. These tests fail if that slips.
"""
import inspect

import pytest

from tibs import (
    BitOrder,
    ByteOrder,
    Codec,
    Dtype,
    DtypeArray,
    DtypeKind,
    DtypeSingle,
    DtypeTuple,
    MutableView,
    Mutibs,
    Reader,
    Tibs,
    View,
)


# (label, callable) pairs that each pass the subject of the call by keyword.
POSITIONAL_ONLY_CALLS = [
    ("Tibs(auto=)", lambda: Tibs(auto="0xff")),
    ("Mutibs(auto=)", lambda: Mutibs(auto="0xff")),
    ("Tibs.field(a=, b=)", lambda: Tibs("0xff").field(a=7, b=0)),
    ("Mutibs.field(a=, b=)", lambda: Mutibs("0xff").field(a=7, b=0)),
    ("View.field(a=, b=)", lambda: Tibs("0xff").view().field(a=7, b=0)),
    ("MutableView.field(a=, b=)", lambda: Mutibs("0xff").view().field(a=7, b=0)),
    ("Tibs.find(needle=)", lambda: Tibs("0xff").find(needle="0x0f")),
    ("Tibs.rfind(needle=)", lambda: Tibs("0xff").rfind(needle="0x0f")),
    ("Tibs.find_all(needle=)", lambda: Tibs("0xff").find_all(needle="0x0f")),
    ("Tibs.find_all_iter(needle=)", lambda: Tibs("0xff").find_all_iter(needle="0x0f")),
    ("Tibs.rfind_all_iter(needle=)", lambda: Tibs("0xff").rfind_all_iter(needle="0x0f")),
    ("Mutibs.find(needle=)", lambda: Mutibs("0xff").find(needle="0x0f")),
    ("Tibs.count(value=)", lambda: Tibs("0xff").count(value=1)),
    ("Tibs.count_and(other=)", lambda: Tibs("0xff").count_and(other="0xff")),
    ("Tibs.count_or(other=)", lambda: Tibs("0xff").count_or(other="0xff")),
    ("Tibs.count_xor(other=)", lambda: Tibs("0xff").count_xor(other="0xff")),
    ("Tibs.count_andnot(other=)", lambda: Tibs("0xff").count_andnot(other="0xff")),
    ("Tibs.intersects(other=)", lambda: Tibs("0xff").intersects(other="0xff")),
    ("Tibs.is_disjoint(other=)", lambda: Tibs("0xff").is_disjoint(other="0xff")),
    ("Tibs.is_subset_of(other=)", lambda: Tibs("0xff").is_subset_of(other="0xff")),
    ("Tibs.is_superset_of(other=)", lambda: Tibs("0xff").is_superset_of(other="0xff")),
    ("Tibs.extracted(mask=)", lambda: Tibs("0xff").extracted(mask="0x0f")),
    ("Mutibs.extracted(mask=)", lambda: Mutibs("0xff").extracted(mask="0x0f")),
    ("Tibs.deposited(value=, mask=)",
     lambda: Tibs("0xff").deposited(value="0b1111", mask="0x0f")),
    ("Mutibs.deposit(value=, mask=)",
     lambda: Mutibs("0xff").deposit(value="0b1111", mask="0x0f")),
    ("Tibs.to_value(dtype=)", lambda: Tibs("0xff").to_value(dtype="u8")),
    ("Tibs.to_values(dtype=)", lambda: Tibs("0xff").to_values(dtype="u4")),
    ("Tibs.to_values_iter(dtype=)", lambda: Tibs("0xff").to_values_iter(dtype="u4")),
    ("Tibs.chunks(chunk_size=)", lambda: Tibs("0xff").chunks(chunk_size=4)),
    ("Tibs.chunks_iter(chunk_size=)", lambda: Tibs("0xff").chunks_iter(chunk_size=4)),
    ("Tibs.rchunks_iter(chunk_size=)", lambda: Tibs("0xff").rchunks_iter(chunk_size=4)),
    ("Tibs.starts_with(prefix=)", lambda: Tibs("0xff").starts_with(prefix="0xf")),
    ("Tibs.ends_with(suffix=)", lambda: Tibs("0xff").ends_with(suffix="0xf")),
    ("Tibs.replaced(old=, new=)",
     lambda: Tibs("0xff").replaced(old="0x0f", new="0x00")),
    ("Mutibs.replace(old=, new=)",
     lambda: Mutibs("0xff").replace(old="0x0f", new="0x00")),
    ("Tibs.rotated_left(n=)", lambda: Tibs("0xff").rotated_left(n=1)),
    ("Tibs.rotated_right(n=)", lambda: Tibs("0xff").rotated_right(n=1)),
    ("Mutibs.rotate_left(n=)", lambda: Mutibs("0xff").rotate_left(n=1)),
    ("Mutibs.rotate_right(n=)", lambda: Mutibs("0xff").rotate_right(n=1)),
    ("Tibs.set_at(pos=)", lambda: Tibs("0xff").set_at(pos=0)),
    ("Tibs.unset_at(pos=)", lambda: Tibs("0xff").unset_at(pos=0)),
    ("Tibs.inverted(pos=)", lambda: Tibs("0xff").inverted(pos=0)),
    ("Mutibs.set(pos=)", lambda: Mutibs("0xff").set(pos=0)),
    ("Mutibs.unset(pos=)", lambda: Mutibs("0xff").unset(pos=0)),
    ("Mutibs.invert(pos=)", lambda: Mutibs("0xff").invert(pos=0)),
    ("Mutibs.append(bit=)", lambda: Mutibs("0xff").append(bit=1)),
    ("Mutibs.reserve(additional=)", lambda: Mutibs("0xff").reserve(additional=8)),
    ("View(source=)", lambda: View(source=Tibs("0xff"))),
    ("MutableView(source=)", lambda: MutableView(source=Mutibs("0xff"))),
    ("View.from_indices(source=, indices=)",
     lambda: View.from_indices(source=Tibs("0xff"), indices=[0])),
    ("MutableView.from_indices(source=, indices=)",
     lambda: MutableView.from_indices(source=Mutibs("0xff"), indices=[0])),
    ("View.to_value(dtype=)", lambda: Tibs("0xff").view().to_value(dtype="u8")),
    ("MutableView.to_value(dtype=)", lambda: Mutibs("0xff").view().to_value(dtype="u8")),
    ("MutableView.write_value(dtype=, value=)",
     lambda: Mutibs("0xff").view().write_value(dtype="u8", value=1)),
    ("DtypeSingle.from_params(kind=, length=)",
     lambda: DtypeSingle.from_params(kind=DtypeKind.Uint, length=8)),
    ("DtypeArray.from_params(dtype=, count=)",
     lambda: DtypeArray.from_params(dtype="u8", count=2)),
    ("DtypeTuple.from_params(dtypes=)",
     lambda: DtypeTuple.from_params(dtypes=["u8"])),
    ("Dtype.pack(value=)", lambda: Dtype("u8").pack(value=1)),
    ("Dtype.unpack(bits=)", lambda: Dtype("u8").unpack(bits="0xff")),
    ("Reader(source=)", lambda: Reader(source=Tibs("0xff"))),
    ("Reader.read_value(dtype=)", lambda: Reader(Tibs("0xff")).read_value(dtype="u8")),
    ("Reader.read_values(dtype=)", lambda: Reader(Tibs("0xff")).read_values(dtype="u4")),
    ("Reader.read_bits(n=)", lambda: Reader(Tibs("0xff")).read_bits(n=4)),
    ("Reader.peek_value(dtype=)", lambda: Reader(Tibs("0xff")).peek_value(dtype="u8")),
    ("Reader.peek_bits(n=)", lambda: Reader(Tibs("0xff")).peek_bits(n=4)),
    ("Reader.read_to(needle=)", lambda: Reader(Tibs("0xff")).read_to(needle="0x0f")),
    ("Reader.read_past(needle=)", lambda: Reader(Tibs("0xff")).read_past(needle="0x0f")),
    ("Reader.seek_to(needle=)", lambda: Reader(Tibs("0xff")).seek_to(needle="0x0f")),
    ("Reader.seek_past(needle=)", lambda: Reader(Tibs("0xff")).seek_past(needle="0x0f")),
    ("Reader.seek_back_to(needle=)",
     lambda: Reader(Tibs("0xff"), 8).seek_back_to(needle="0x0f")),
    ("Reader.align(boundary=)", lambda: Reader(Tibs("0xff")).align(boundary=8)),
]


@pytest.mark.parametrize("label, call", POSITIONAL_ONLY_CALLS, ids=[c[0] for c in POSITIONAL_ONLY_CALLS])
def test_call_subject_is_positional_only(label, call):
    with pytest.raises(TypeError):
        call()


# The optional modifiers stay keyword-reachable, and are the reason the
# subjects had to be pinned down separately.
def test_optional_modifiers_stay_keyword_reachable():
    t = Tibs("0x1f2e3f")
    assert t.find("0x0f", start=0, end=24, byte_aligned=True, mask="0x0f") == 0
    assert t.count("0x0f", mask="0x0f", byte_aligned=True) == 2
    assert t.replaced("0x0f", "0x00", count=1, byte_aligned=True) is not None
    assert len(t.chunks(8, count=2)) == 2
    assert t.rotated_left(1, start=0, end=8).hex == "3e2e3f"
    assert t.to_u(start=0, end=8) == 0x1F
    assert Tibs.from_u(1, length=8).hex == "01"
    assert Tibs.from_bytes(b"\xff", offset=4, length=4).bin == "1111"
    assert len(Tibs.from_random(8, secure=True)) == 8
    assert t.view(byte_order=ByteOrder.Unspecified, bit_order=BitOrder.Lsb0) is not None
    assert t.le.to_value("u8", start=0, end=8) == 0x3F
    assert t.encode(codec=Codec.Raw) is not None
    assert t.byte_swapped(byte_length=1, start=0, end=24) is not None
    assert Reader(t, pos=8).pos == 8
    assert Reader(t).read_values("u8", count=2) == [0x1F, 0x2E]
    assert Reader(t).seek_to("0x0f", byte_aligned=True, mask="0x0f") is True
    assert str(DtypeSingle.from_params(DtypeKind.Uint, 16, byte_order=ByteOrder.Little)) == "u16_le"


def test_extract_is_named_extracted():
    # The bare verb is reserved for a possible future in-place Mutibs.extract,
    # and the copy-returning form follows the past-participle convention that
    # reversed/inverted/deposited use.
    assert Tibs("0b11010110").extracted("0b10110000") == Tibs("0b101")
    assert Mutibs("0b11010110").extracted("0b10110000") == Mutibs("0b101")
    assert not hasattr(Tibs, "extract")
    assert not hasattr(Mutibs, "extract")


def test_view_requires_a_tibs_and_mutable_view_requires_a_mutibs():
    t, m = Tibs("0xff"), Mutibs("0xff")

    with pytest.raises(TypeError, match="must be a Tibs"):
        View(m)
    with pytest.raises(TypeError, match="must be a Tibs"):
        View.from_indices(m, [0])
    with pytest.raises(TypeError):
        MutableView(t)
    with pytest.raises(TypeError):
        MutableView.from_indices(t, [0])

    # The explicit conversions are how you cross over.
    assert View(m.to_tibs()) == View(t)
    assert isinstance(m.view(), MutableView)


def test_no_public_method_takes_a_single_letter_keyword():
    # Single-letter names are fine positionally but would be permanent API if
    # they were keyword-reachable.
    offenders = []
    for cls in (Tibs, Mutibs, View, MutableView, Reader, Dtype, DtypeSingle, DtypeArray,
                DtypeTuple):
        for name in dir(cls):
            if name.startswith("_"):
                continue
            attr = inspect.getattr_static(cls, name)
            if isinstance(attr, property):
                continue
            try:
                sig = inspect.signature(getattr(cls, name))
            except (TypeError, ValueError):
                continue
            for p in sig.parameters.values():
                if p.kind is p.POSITIONAL_OR_KEYWORD and len(p.name) == 1:
                    offenders.append(f"{cls.__name__}.{name}({p.name}=)")
    assert offenders == []
