from __future__ import annotations

from ._fieldtype import FieldType
from ._common import override, Indenter, Expression, ExpressionError
from typing import Any
from ._bits import Bits

__all__ = ["Let"]


class Let(FieldType):
    """
    A :class:`FieldType` that assigns expression results to variables.

    This is only useful within a :class:`Format` where other fields can refer to the variable.



    """

    _name: str
    _expr: Expression


    def __new__(cls, s: str) -> Let:
        return cls.from_string(s)

    @classmethod
    @override
    def from_params(cls, name: str, expr: str | int | Expression) -> Let:
        """
        Create a Let instance.

        :param name: The variable name to assign to.
        :param expr: The Expression to evaluate.
        :return: The Let instance.

        .. code-block:: python

            increment_x = Let.from_params('x', '{x + 1}')

        """
        x = super().__new__(cls)
        x._name = name
        if isinstance(expr, str):
            expr = Expression.from_string(expr)
        if isinstance(expr, int):
            expr = Expression.from_int(expr)
        x._expr = expr
        return x

    @override
    def _get_bit_length(self) -> int:
        return 0

    @override
    def _pack(self, value: Any, kwargs: dict[str, Any]) -> bool:
        try:
            x = self._expr.evaluate(**kwargs)
        except ExpressionError as e:
            raise ValueError(f"Cannot evaluate expression: {e}")
        kwargs[self._name] = x
        return False  # Doesn't consume value

    @override
    def _parse(self, b: Bits, startbit: int, vars_: dict[str, Any]) -> int:
        try:
            x = self._expr.evaluate(**vars_)
        except ExpressionError as e:
            raise ValueError(f"Cannot evaluate expression: {e}")
        vars_[self._name] = x
        return 0

    @override
    def _copy(self) -> Let:
        return self

    @override
    def _info(self, use_colour: bool) -> str:
        return f"let fieldtype (TODO)."

    @override
    def clear(self) -> None:
        """Clearing a Let field has no effect."""
        pass

    @override
    def _get_value(self) -> Any:
        raise AttributeError("A Let field does not have a value.")

    @override
    def _set_value_with_kwargs(self, val: Any, kwargs: dict[str, Any]) -> None:
        raise ValueError("A Let field cannot be set to a value.")

    @override
    def _str(self, indent: Indenter, use_colour: bool) -> str:
        s = indent(f"let {self._name} = {{{self._expr}}}")
        return s

    @override
    def _repr(self) -> str:
        return f"Let({self._name!r}, {self._expr!r})"

    @override
    def to_bits(self) -> Bits:
        """Returns an empty :class:`Bits`."""
        return Bits()

    @override
    def has_dynamic_size(self) -> bool:
        """Returns False for a Let field."""
        return False

    @override
    def is_const(self) -> bool:
        """Returns True for a Let field."""
        return True

    @override
    def __eq__(self, other) -> bool:
        raise NotImplementedError

    @override
    def _get_name(self) -> None:
        return None

    @override
    def _set_name(self, name: str) -> None:
        raise AttributeError("The Let field can't set 'name' property after creation.")