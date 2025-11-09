from __future__ import annotations

from ._fieldtype import FieldType
from ._common import override, Indenter, Expression, ExpressionError
from typing import Sequence, Any, Iterable
from ._bits import Bits

__all__ = ["Repeat"]


class Repeat(FieldType):

    field: FieldType
    _count: Expression
    _concrete_count: int | None
    _bits_list: list[Bits]

    def __new__(cls, s: str) -> Repeat:
        return cls.from_string(s)

    @classmethod
    @override
    def from_params(cls, count: int | str | Expression, fieldtype: FieldType | str, value: Iterable[Any] | None = None) -> Repeat:
        """
        Create a Repeat instance.

        :param count: An Expression or int giving the number of repetitions to do.
        :param fieldtype: The FieldType to repeat.
        :param value: The list of values for the FieldType, one for each repetition.
        :return: The Repeat instance.
        """
        x = super().__new__(cls)
        if isinstance(count, str):
            count = Expression.from_string(count)
        if isinstance(count, int):
            count = Expression.from_int(count)
        x._count = count
        x._concrete_count = None
        if x._count.has_const_value:
            if isinstance(x._count.const_value, int):
                x._concrete_count = x._count.const_value
            else:
                raise ValueError(f"Repeat count must be an integer, not {type(x._count.const_value)}.")
        x._bits_list = []
        if isinstance(fieldtype, str):
            fieldtype = FieldType.from_string(fieldtype)
        elif not isinstance(fieldtype, FieldType):
            raise ValueError(f"Invalid field of type {type(fieldtype)}.")
        if not (fieldtype.value is None or (isinstance(fieldtype.value, Sequence) and all(x is None for x in fieldtype.value))):
            if not fieldtype.is_const():
                raise ValueError(f"Non-const Fieldtypes used inside a Repeat must not have a value set. Received '{fieldtype}'.")
        x.field = fieldtype
        if value is not None:
            x.value = value
        return x

    @override
    def _get_bit_length(self) -> int:
        if self._concrete_count is None:
            raise ValueError("Repeat count is not concrete, cannot calculate bit length.")
        try:
            field_const_length = self.field.bit_length
        except ValueError:
            raise ValueError("Field being repeated does not have a concrete bit length.")
        return field_const_length * self._concrete_count

    @classmethod
    @override
    def from_string(cls, s: str) -> Repeat:
        """
        Create a :class:`Repeat` instance from a string.

        The string should be of the form ``'repeat {expression}: fieldtype'``.
        The fieldtype can be a :class:`Format` to group multiple fields together.

        :param s: The string to parse.
        :return: The Repeat instance.

        .. code-block:: python

            r1 = Repeat.from_string('repeat {5}: u8')
            r2 = Repeat.from_string('repeat {3}: f32 = [0.0, 13.5, -5.05]')
            r3 = Repeat.from_string('repeat {x + 1}: (bool, f64)')

        """
        x = super().from_string(s)
        if not isinstance(x, Repeat):
            raise ValueError(f"Can't parse Repeat field from '{s}'. Instead got '{x}'.")
        return x

    @override
    def _info(self, use_colour: bool) -> str:
        s = f"repeat field with count of {self.count}."
        return s

    @override
    def _str(self, indent: Indenter, use_colour: bool) -> str:
        # We don't use str(self._count) here as that strips the {} from const expressions.
        count_str = "{" + self._count.code_str + "}"
        s = indent(f"repeat{count_str}:")
        with indent:
            value_iter = iter(self.value) if self.value else iter([])
            if self._concrete_count is not None:
                for i in range(self._concrete_count):
                    try:
                        value = next(value_iter)
                    except StopIteration:
                        value = None
                    if value is not None:
                        f = self.field._copy()
                        f.value = value
                        s += "\n" + f._str(indent, use_colour)
                    else:
                        s += "\n" + self.field._str(indent, use_colour)
            else:
                s += '\n' + self.field._str(indent, use_colour)
        return s

    @override
    def _repr(self) -> str:
        s = f"Repeat.from_params({self._count!r}, "
        s += self.field._repr()
        if self.value is not None:
            s += f", {self.value}"
        s += ")"
        return s

    @override
    def _parse(self, b: Bits, startbit: int, kwargs: dict[str, Any]) -> int:
        self._bits_list = []
        pos = startbit
        if self._concrete_count is None:
            self._concrete_count = self._count.evaluate(**kwargs)
        for i in range(self._concrete_count):
            pos += self.field._parse(b, pos, kwargs)
            self._bits_list.append(self.field.to_bits())
        return pos - startbit

    @override
    def _pack(self, value: Sequence[Any], kwargs: dict[str, Any]) -> bool:
        if not isinstance(value, Sequence):
            raise TypeError(f"A Repeat field needs a Sequence to pack, but received '{value}' of type {type(value)}. "
                            f"To pack a single item, make a single item list instead.")
        self._bits_list = []
        if self._concrete_count is None:
            try:
                self._concrete_count = self._count.evaluate(**kwargs)
            except ExpressionError as e:
                raise ValueError(f"Cannot evaluate count for Repeat field: {e}")

        self._set_value_with_kwargs(value, kwargs)
        return True

    @override
    def _copy(self) -> Repeat:
        x = self.__class__.from_params(self._count, self.field._copy(), self.value)
        return x

    @override
    def to_bits(self) -> Bits:
        return Bits.from_joined(self._bits_list)

    @override
    def clear(self) -> None:
        self._concrete_count = None
        self._bits_list = []

    @override
    def has_dynamic_size(self) -> bool:
        return False

    @override
    def is_const(self) -> bool:
        return self._count.has_const_value and self.field.is_const()

    @override
    def _get_value(self) -> list[Any] | None:
        if not self._bits_list:
            return None
        values = [self.field.unpack(bits) for bits in self._bits_list]
        self.field.clear()
        return values

    @override
    def _set_value_with_kwargs(self, value: list[Any], kwargs: dict[str, Any]) -> None:
        value_iter = iter(value)
        for i in range(self._concrete_count):
            try:
                next_value = next(value_iter)
            except StopIteration:
                next_value = None
            self.field._pack(next_value, kwargs)
            self._bits_list.append(self.field.to_bits())
            self.field.clear()

    @override
    def __eq__(self, other) -> bool:
        if not isinstance(other, Repeat):
            return False
        if self._count != other._count:
            return False
        if self.field != other.field:
            return False
        return True

    @override
    def _get_name(self) -> None:
        return None

    @override
    def _set_name(self, name: str) -> None:
        raise AttributeError("The Repeat field has no 'name' property.")

    @property
    def count(self) -> Expression:
        """
        The count of the Repeat field.

        :return: The count of the Repeat field.
        """
        return self._count

    @count.setter
    def count(self, value: int | str| Expression) -> None:
        """
        Set the count of the Repeat field.

        :param value: The new count value.
        """
        if isinstance(value, str):
            value = Expression.from_string(value)
        elif isinstance(value, int):
            value = Expression.from_int(value)
        self.clear()
        self._count = value
