from __future__ import annotations

from ._fieldtype import FieldType
from ._common import override, Indenter
from typing import Any
from ._bits import Bits

__all__ = ["Pass"]


class Pass(FieldType):
    """
    An empty placeholder :class:`FieldType`.

    A Pass field has zero bit_length and no value. It can be used in conditionals when no action is required.
    When used elsewhere in a :class:`Format` it may be removed as an optimisation.
    All Pass fields are identical, so they are implemented as a singleton.

    It is usually created implicitly, for example in an :class:`If` field when no ``else`` field is provided.

    .. code-block:: python

        cond = If.from_params('{ x > 0 }', Pass(), 'bool')

    """

    # All Pass fields are the same, so we make it a singleton.
    _instance = None

    def __new__(cls) -> Pass:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    @override
    def from_params(cls, *args, **kwargs) -> FieldType:
        """Returns the singleton Pass instance. The `args` and `kwargs` are ignored."""
        return cls()

    @override
    def _get_bit_length(self) -> int:
        return 0

    @override
    def _pack(self, value: Any, kwargs: dict[str, Any]) -> bool:
        return False

    @override
    def _parse(self, b: Bits, startbit: int, vars_: dict[str, Any]) -> int:
        return 0

    @override
    def _copy(self) -> Pass:
        return self

    @override
    def _info(self, use_colour: bool) -> str:
        return f"pass fieldtype (always empty)."

    @override
    def clear(self) -> None:
        """Clearing a Pass field has no effect."""
        pass

    @override
    def _get_value(self) -> Any:
        raise AttributeError("A Pass field does not have a value.")

    @override
    def _set_value_with_kwargs(self, val: Any, kwargs: dict[str, Any]) -> None:
        raise ValueError("A Pass field cannot be set to a value.")

    @override
    def _str(self, indent: Indenter, use_colour: bool) -> str:
        s = indent("pass")
        return s

    @override
    def _repr(self) -> str:
        return "Pass()"

    @override
    def to_bits(self) -> Bits:
        """Returns an empty :class:`Bits`."""
        return Bits()

    @override
    def has_dynamic_size(self) -> bool:
        """Returns False for a Pass field."""
        return False

    @override
    def is_const(self) -> bool:
        """Returns True for a Pass field."""
        return True

    @override
    def __eq__(self, other) -> bool:
        return isinstance(other, Pass)

    @override
    def _get_name(self) -> None:
        return None

    @override
    def _set_name(self, name: str) -> None:
        raise AttributeError("The Pass field has no 'name' property.")