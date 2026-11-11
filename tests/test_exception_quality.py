"""Regression tests for misleading or inconsistent exception behaviour.

These tests deliberately describe problems in the current implementation. They
are expected to fail until the corresponding exception paths are fixed.
"""

import pytest

from tibs import (
    DecodeError,
    Dtype,
    DtypeArray,
    DtypeKind,
    DtypeSingle,
    Mutibs,
    MutableView,
    ReadError,
    Reader,
    Tibs,
    View,
)


OVERSIZED_INDEX = 2**100


def _exception_details(error):
    """Return the message and any Python 3.11 exception notes."""
    return "\n".join([str(error), *getattr(error, "__notes__", ())])


def test_module_specific_errors_are_value_errors():
    assert issubclass(ReadError, ValueError)
    assert issubclass(DecodeError, ValueError)


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
def test_split_at_does_not_treat_an_oversized_integer_as_an_iterable(cls):
    bits = cls("0b1")

    # A normal out-of-range split position is a ValueError; claiming that the
    # integer is not iterable is not.
    with pytest.raises(ValueError):
        bits.split_at(OVERSIZED_INDEX)


@pytest.mark.parametrize(
    "cls, method",
    [
        (Tibs, "with_set"),
        (Tibs, "with_unset"),
        (Tibs, "inverted"),
        (Mutibs, "set"),
        (Mutibs, "unset"),
        (Mutibs, "with_set"),
        (Mutibs, "with_unset"),
        (Mutibs, "invert"),
        (Mutibs, "inverted"),
    ],
    ids=[
        "Tibs.with_set",
        "Tibs.with_unset",
        "Tibs.inverted",
        "Mutibs.set",
        "Mutibs.unset",
        "Mutibs.with_set",
        "Mutibs.with_unset",
        "Mutibs.invert",
        "Mutibs.inverted",
    ],
)
def test_position_methods_treat_an_oversized_integer_as_out_of_range(cls, method):
    bits = cls("0b1")

    with pytest.raises(IndexError):
        getattr(bits, method)(OVERSIZED_INDEX)


def test_mutable_subscription_treats_an_oversized_integer_as_out_of_range():
    bits = Mutibs("0b1")

    with pytest.raises(IndexError):
        bits[OVERSIZED_INDEX] = True

    with pytest.raises(IndexError):
        del bits[OVERSIZED_INDEX]


@pytest.mark.parametrize(
    "view_cls, source",
    [(View, Tibs("0b1")), (MutableView, Mutibs("0b1"))],
)
@pytest.mark.parametrize("index", [-1, OVERSIZED_INDEX])
def test_view_source_index_errors_name_the_invalid_index_and_source_length(
    view_cls, source, index
):
    with pytest.raises(ValueError) as exc_info:
        view_cls.from_indices(source, [index])

    message = str(exc_info.value)
    assert str(index) in message
    assert str(len(source)) in message


@pytest.mark.parametrize(
    "view",
    [Tibs("0x0102").le, Mutibs("0x0102").view().lsb0],
    ids=["View", "MutableView"],
)
@pytest.mark.parametrize("promote", [Tibs, Mutibs, Reader])
def test_promoting_a_view_points_at_to_tibs_rather_than_from_bytes(view, promote):
    # A view is not bytes-like, truthy items or numeric values, so the general
    # advice sends the reader somewhere useless. to_tibs() is the answer.
    with pytest.raises(TypeError) as exc_info:
        promote(view)

    message = _exception_details(exc_info.value)
    assert "to_tibs()" in message
    assert "from_bytes" not in message


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
def test_count_rejects_an_unsupported_value_type_with_type_error(cls):
    with pytest.raises(TypeError):
        cls("0b1").count(object())


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
@pytest.mark.parametrize("container", [list, tuple])
def test_decode_rejects_integer_sequences_that_are_not_bytes(cls, container):
    encoded = container(Tibs("0b1").encode())

    with pytest.raises(TypeError):
        cls.decode(encoded)


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
@pytest.mark.parametrize("seed", [[1], (1,)], ids=["list", "tuple"])
def test_random_seed_rejects_integer_sequences_that_are_not_bytes(cls, seed):
    with pytest.raises(TypeError):
        cls.from_random(8, seed=seed)


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
def test_random_seed_type_error_uses_python_api_terminology(cls):
    with pytest.raises(TypeError) as exc_info:
        cls.from_random(8, seed="not bytes")

    message = str(exc_info.value).lower()
    assert "seed" in message
    assert "bytes" in message
    assert "vec" not in message


def test_nested_dtype_context_preserves_the_original_user_exception():
    original = RuntimeError("conversion failed")

    class BadIndex:
        def __index__(self):
            raise original

    with pytest.raises(RuntimeError) as exc_info:
        Dtype("(u8,)").pack((BadIndex(),))

    assert exc_info.value is original
    assert "value[0]" in _exception_details(exc_info.value).lower()


BULK_DTYPE_ERROR_CASES = [
    pytest.param("u8", [1, 2, 256, 4], ValueError, 2, id="bytewise-u8"),
    pytest.param("bool", [True, False, 2, True], TypeError, 2, id="bitwise-bool"),
    pytest.param("hex8", ["00", "11", "f", "22"], ValueError, 2, id="general-hex"),
]


@pytest.mark.parametrize(
    "dtype, values, error_type, failing_index", BULK_DTYPE_ERROR_CASES
)
def test_dtype_pack_values_errors_identify_the_failing_item(
    dtype, values, error_type, failing_index
):
    with pytest.raises(error_type) as exc_info:
        Dtype(dtype).pack_values(values)

    assert f"[{failing_index}]" in _exception_details(exc_info.value)


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
@pytest.mark.parametrize(
    "dtype, values, error_type, failing_index", BULK_DTYPE_ERROR_CASES
)
def test_container_from_values_errors_identify_the_failing_item(
    cls, dtype, values, error_type, failing_index
):
    with pytest.raises(error_type) as exc_info:
        cls.from_values(dtype, values)

    assert f"[{failing_index}]" in _exception_details(exc_info.value)


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
def test_negative_chunk_count_message_states_that_zero_is_allowed(cls):
    with pytest.raises(ValueError, match=r">= 0"):
        cls("0b1").chunks(1, count=-1)


def test_negative_reverse_chunk_count_message_states_that_zero_is_allowed():
    with pytest.raises(ValueError, match=r">= 0"):
        Tibs("0b1").rchunks_iter(1, count=-1)


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
def test_negative_from_bytes_offset_error_names_the_offset_parameter(cls):
    with pytest.raises(ValueError) as exc_info:
        cls.from_bytes(b"\x00", bit_offset=-1)

    message = str(exc_info.value).lower()
    assert "offset" in message
    assert "-1" in message


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
def test_empty_decode_error_uses_grammatical_python_terminology(cls):
    with pytest.raises(DecodeError) as exc_info:
        cls.decode(b"")

    assert str(exc_info.value) == "Cannot decode an empty byte sequence."


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
@pytest.mark.parametrize("method_name", ["from_u", "from_i"])
def test_integer_constructor_docstring_names_value_error(cls, method_name):
    docstring = getattr(cls, method_name).__doc__

    assert ":raises ValueError:" in docstring


@pytest.mark.parametrize(
    "spec,kind_name",
    [("u0", "DtypeKind.Uint ('u')"), ("i0", "DtypeKind.Int ('i')")],
)
def test_zero_length_integer_dtype_message_names_the_kind(spec, kind_name):
    with pytest.raises(ValueError) as exc_info:
        Dtype(spec)
    message = str(exc_info.value)
    assert kind_name in message
    assert "at least 1 bit" in message
    assert "greater than zero" not in message


def test_negative_lengths_and_counts_are_reported_as_negative():
    with pytest.raises(ValueError, match="can't be negative, but received -1"):
        DtypeSingle.from_params(DtypeKind.Bytes, -1)
    with pytest.raises(ValueError, match="can't be negative, but received -1"):
        DtypeArray.from_params("u8", -1)


def test_a_missing_array_count_is_not_described_as_needing_to_be_positive():
    with pytest.raises(ValueError, match="expected an array count") as exc_info:
        Dtype("[u8; -1]")
    assert "positive" not in str(exc_info.value)


@pytest.mark.parametrize(
    "read",
    [
        lambda: Tibs("0x01").to_values("bytes0"),
        lambda: Dtype("[u8; 0]").unpack_values(Tibs()),
        lambda: Reader(Tibs("0x01")).read_values("hex0"),
    ],
)
def test_reading_an_unknown_number_of_zero_length_values_says_why(read):
    with pytest.raises(ValueError, match="has length 0, so any number of its values fit"):
        read()


class _ClaimsToBeHuge:
    """A short iterable whose ``len()`` says otherwise.

    ``len()`` is only a claim: a ``range`` can report 2**45 while holding
    nothing, and any class can define ``__len__``.
    """

    def __init__(self, items, claimed):
        self._items = items
        self._claimed = claimed

    def __len__(self):
        return self._claimed

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index):
        return self._items[index]


# One that fails to allocate, and one past what any allocation can be.
_CLAIMED_LENGTHS = [2**45, 2**62]

_LENGTH_HINT_CALLS = [
    ("Tibs.split_at", lambda pos: Tibs("0xff").split_at(pos)),
    ("Mutibs.split_at", lambda pos: Mutibs("0xff").split_at(pos)),
    ("Tibs.with_set", lambda pos: Tibs("0x00").with_set(pos)),
    ("Tibs.with_unset", lambda pos: Tibs("0xff").with_unset(pos)),
    ("Tibs.inverted", lambda pos: Tibs("0x00").inverted(pos)),
    ("Mutibs.with_set", lambda pos: Mutibs("0x00").with_set(pos)),
    ("Mutibs.set", lambda pos: _mutated(Mutibs("0x00"), "set", pos)),
    ("Mutibs.unset", lambda pos: _mutated(Mutibs("0xff"), "unset", pos)),
    ("Mutibs.invert", lambda pos: _mutated(Mutibs("0x00"), "invert", pos)),
    ("View.from_indices", lambda pos: View.from_indices(Tibs("0xa5"), pos).bin),
    ("MutableView.from_indices", lambda pos: MutableView.from_indices(Mutibs("0xa5"), pos).bin),
    ("Tibs.from_bools", lambda pos: Tibs.from_bools(pos)),
    ("Mutibs.from_bools", lambda pos: Mutibs.from_bools(pos)),
    ("Tibs.from_values", lambda pos: Tibs.from_values("u8", pos)),
    ("Dtype.pack_values", lambda pos: Dtype("u4").pack_values(pos)),
]


def _mutated(m, method, pos):
    getattr(m, method)(pos)
    return m


@pytest.mark.parametrize("claimed", _CLAIMED_LENGTHS)
@pytest.mark.parametrize(
    "call", [call for _, call in _LENGTH_HINT_CALLS], ids=[name for name, _ in _LENGTH_HINT_CALLS]
)
def test_an_iterable_is_not_trusted_about_its_length(call, claimed):
    # Space for `len()` items was reserved before any were read. A claim of
    # 2**45 aborted the process on the failed allocation, and one of 2**62
    # panicked, so a claim must change nothing about the result.
    items = [1, 3, 6]
    assert call(_ClaimsToBeHuge(items, claimed)) == call(items)


def test_a_wrong_length_claim_that_is_too_small_changes_nothing_either():
    items = list(range(0, 300, 3))
    t = Tibs.from_zeros(300)
    assert t.with_set(_ClaimsToBeHuge(items, 0)) == t.with_set(items)
    assert t.split_at(_ClaimsToBeHuge(items, 1)) == t.split_at(items)
    assert View.from_indices(t, _ClaimsToBeHuge(items, 2)).bin == "0" * 100
