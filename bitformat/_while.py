from __future__ import annotations

from ._fieldtype import FieldType
from ._common import override, Indenter, Expression, ExpressionError
from typing import Sequence, Any
from ._bits import Bits

__all__ = ["While"]


class While(FieldType):

    field: FieldType
    _expr: Expression
    _bits_list: list[Bits]
    _values_list = list[Any]

    def __new__(cls, s: str) -> While:
        return cls.from_string(s)

    @classmethod
    @override
    def from_params(cls, expr: str | Expression, fieldtype: FieldType | str) -> While:
        """
        Create a While instance.

        :param expr: An Expression that must be True for the fieldtype to be used.
        :param fieldtype: The FieldType to use. Almost always a :class:`Format`.
        :return: The While instance.
        """
        x = super().__new__(cls)
        if isinstance(expr, str):
            expr = Expression.from_string(expr)
        x._expr = expr
        x._bits_list = []
        x._values_list = []
        if isinstance(fieldtype, str):
            fieldtype = FieldType.from_string(fieldtype)
        elif not isinstance(fieldtype, FieldType):
            raise ValueError(f"Invalid field of type {type(fieldtype)}.")
        x.field = fieldtype
        return x

    @override
    def _get_bit_length(self) -> int:
        raise ValueError("Bit lengths cannot be calculated for 'While' fieldtypes.")

    @classmethod
    @override
    def from_string(cls, s: str) -> While:
        """
        Create a :class:`While` instance from a string.

        The string should be of the form ``'while {expression}: fieldtype'``.
        The fieldtype is almost always a :class:`Format` to group multiple fields together.

        :param s: The string to parse.
        :return: The While instance.

        .. code-block:: python

            w = While.from_string('while {x > 5}: (u8, let x = {x - 2})')

        """
        x = super().from_string(s)
        if not isinstance(x, While):
            raise ValueError(f"Can't parse While field from '{s}'. Instead got '{x}'.")
        return x

    @override
    def _info(self, use_colour: bool) -> str:
        s = f"while field with condition '{self._expr}'."
        return s

    @override
    def _str(self, indent: Indenter, use_colour: bool) -> str:
        count_str = "{" + self._expr.code_str + "}"
        s = indent(f"while {count_str}:")
        with indent:
            s += '\n' + self.field._str(indent, use_colour)
        return s

    @override
    def _repr(self) -> str:
        s = f"While.from_params({self._expr!r}, "
        s += self.field._repr()
        s += ")"
        return s

    @override
    def _parse(self, b: Bits, startbit: int, kwargs: dict[str, Any]) -> int:
        self._bits_list = []
        self._values_list = []
        pos = startbit
        while self._expr.evaluate(**kwargs):
            pos += self.field._parse(b, pos, kwargs)
            self._bits_list.append(self.field.to_bits())
            self._values_list.extend(self.field.value)
        return pos - startbit

    @override
    def _pack(self, values: Sequence[Any], kwargs: dict[str, Any]) -> bool:
        self._bits_list = []
        self._values_list = []
        value_iter = iter(values)
        while self._expr.evaluate(**kwargs):
            value = next(value_iter)
            self.field._pack(value, kwargs)
            self._bits_list.append(self.field.to_bits())
            self._values_list.extend(value)
        try:
            next(value_iter)
        except StopIteration:
            return False
        return True

    @override
    def _copy(self) -> While:
        x = self.__class__.from_params(self._expr, self.field._copy())
        return x

    @override
    def to_bits(self) -> Bits:
        return Bits.from_joined(self._bits_list)

    @override
    def clear(self) -> None:
        self._bits_list = []
        self._values_list = []

    @override
    def has_dynamic_size(self) -> bool:
        return False

    @override
    def is_const(self) -> bool:
        return False

    @override
    def _get_value(self) -> list[Any] | None:
        if not self._bits_list:
            return None
        return self._values_list

    @override
    def _set_value_with_kwargs(self, val: list[Any], kwargs: dict[str, Any]) -> None:
        self._values = val  # TODO: This is nonsense.

    @override
    def __eq__(self, other) -> bool:
        if not isinstance(other, While):
            return False
        if self._expr != other._expr:
            return False
        if self.field != other.field:
            return False
        return True

    @override
    def _get_name(self) -> None:
        return None

    @override
    def _set_name(self, name: str) -> None:
        raise AttributeError("The While field has no 'name' property.")

    @property
    def expr(self) -> Expression:
        """
        The expression of the While field.

        :return: The expression of the While field.
        """
        return self._expr

    @expr.setter
    def expr(self, value: str| Expression) -> None:
        """
        Set the expression of the While field.

        :param value: The new expression value.
        """
        if isinstance(value, str):
            value = Expression.from_string(value)
        self._expr = value
        self.clear()