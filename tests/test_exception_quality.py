"""Regression tests for misleading or inconsistent exception behaviour.

These tests deliberately describe problems in the current implementation. They
are expected to fail until the corresponding exception paths are fixed.
"""

import pytest

from tibs import Dtype, Mutibs, MutableView, Tibs, View


OVERSIZED_INDEX = 2**100


def _exception_details(error):
    """Return the message and any Python 3.11 exception notes."""
    return "\n".join([str(error), *getattr(error, "__notes__", ())])


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
        (Tibs, "set_at"),
        (Tibs, "unset_at"),
        (Tibs, "inverted"),
        (Mutibs, "set"),
        (Mutibs, "unset"),
        (Mutibs, "set_at"),
        (Mutibs, "unset_at"),
        (Mutibs, "invert"),
        (Mutibs, "inverted"),
    ],
    ids=[
        "Tibs.set_at",
        "Tibs.unset_at",
        "Tibs.inverted",
        "Mutibs.set",
        "Mutibs.unset",
        "Mutibs.set_at",
        "Mutibs.unset_at",
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
        cls.from_bytes(b"\x00", offset=-1)

    message = str(exc_info.value).lower()
    assert "offset" in message
    assert "-1" in message


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
def test_empty_decode_error_uses_grammatical_python_terminology(cls):
    with pytest.raises(ValueError) as exc_info:
        cls.decode(b"")

    message = str(exc_info.value).lower()
    assert "empty byte sequence" in message or "empty bytes object" in message


@pytest.mark.parametrize("cls", [Tibs, Mutibs])
@pytest.mark.parametrize("method_name", ["from_u", "from_i"])
def test_integer_constructor_docstring_names_value_error(cls, method_name):
    docstring = getattr(cls, method_name).__doc__

    assert ":raises ValueError:" in docstring
