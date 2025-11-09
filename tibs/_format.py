from __future__ import annotations

import itertools

from ._bits import Bits
from typing import Sequence, Any, Iterable, Self
import copy
from ._common import override, Indenter, Colour, validate_name
from ._fieldtype import FieldType
from ._pass import Pass
from ._repeat import Repeat
from ._field import Field

__all__ = ["Format"]


class Format(FieldType):
    """
    A sequence of :class:`FieldType` objects, used to group fields together.

    """
    _name: str
    _fields: list[FieldType]
    _field_names: dict[str, FieldType]
    vars: dict[str, Any]

    def __new__(cls, s: str | None = None) -> Self:
        if s is None:
            x = super().__new__(cls)
            x._fields = []
            x.name = ""
            x.vars = {}
            x._field_names = {}
            return x
        return cls.from_string(s)

    @classmethod
    def from_params(cls, fields: Sequence[FieldType | str] | None = None, name: str = "") -> Self:
        """
        Create a Format instance.

        :param fields: The field types to include in the format, optional.
        :param name: The name of the format, optional.
        :return: The Format instance.
        """
        x = super().__new__(cls)
        x._fields = []
        if fields is None:
            fields = []
        x.name = name
        x.vars = {}
        x._field_names = {}
        stretchy_field = ""
        for fieldtype in fields:
            if stretchy_field:
                raise ValueError(f"A Field with unknown length may only occur at the end of a Format. "
                                 f"Field '{stretchy_field}' is before the end.")
            if isinstance(fieldtype, FieldType):
                fieldtype = fieldtype._copy()
            elif isinstance(fieldtype, str):
                fieldtype = FieldType.from_string(fieldtype)
            else:
                raise ValueError(f"Invalid Field of type {type(fieldtype)}.")
            if fieldtype is Pass():
                # Don't bother appending if it's the Pass singleton.
                continue
            if fieldtype.has_dynamic_size():
                stretchy_field = fieldtype
            x._fields.append(fieldtype)
            if fieldtype.name:
                x._field_names[fieldtype.name] = fieldtype
        return x


    @classmethod
    @override
    def from_string(cls, s: str, /) -> Self:
        """
        Create a :class:`Format` instance from a string.

        The string should be of the form ``'(field1, field2, ...)'`` or ``'name: (field1, field2, ...)'``,
        with commas separating strings that will be used to create other :class:`FieldType` instances.

        :param s: The string to parse.
        :return: The Format instance.

        .. code-block:: python

            f1 = Format.from_string('(u8, bool=True, val: i7)')
            f2 = Format.from_string('my_format: (float16,)')
            f3 = Format.from_string('(u16, another_format: ([u8; 64], [bool; 8]))')

        """
        x = super().from_string(s)
        if not isinstance(x, Format):
            raise ValueError(f'Can\'t parse a Format field from "{s}". This looks like a {type(x)}.')
        return x

    @override
    def _get_bit_length(self) -> int:
        """Return the total length of the Format in bits."""
        return sum(f.bit_length for f in self._fields)

    @override
    def _pack(self, values: Sequence[Any] | None, kwargs: dict[str, Any]) -> bool:
        if values is not None and not isinstance(values, Sequence):
            raise TypeError(f"Format.pack needs a sequence to pack, but received {type(values)}.")
        if values is None:
            value_iter = itertools.repeat(None)
        else:
            value_iter = iter(values)
        consumed_all_values = False
        need_next_value = True
        for fieldtype in self._fields:
            # For const fields (and Repeat with const fields), and padding we don't need to use up a value
            if fieldtype.is_const() or (isinstance(fieldtype, Field) and fieldtype.dtype._is_padding()):
                _ = fieldtype._pack(None, kwargs)
                continue
            fieldtype.clear()
            if isinstance(fieldtype, Repeat) and fieldtype.field.is_const():
                _ = fieldtype._pack([], kwargs)
                continue
            try:
                if need_next_value:
                    next_value = next(value_iter)
            except StopIteration:
                # No more values left to pack, but there may still be some constant fields
                consumed_all_values = True
                _ = fieldtype._pack(None, kwargs)
                continue
            else:
                need_next_value = fieldtype._pack(next_value, kwargs)
        if consumed_all_values:
            return True
        try:
            _ = next(value_iter)
        except StopIteration:
            return True
        return False  # Didn't use up all of values

    @override
    def _parse(self, b: Bits, startbit: int, vars_: dict[str, Any]) -> int:
        self.vars = vars_
        pos = startbit
        for fieldtype in self._fields:
            pos += fieldtype._parse(b, pos, self.vars)
        return pos - startbit

    @override
    def _copy(self) -> Format:
        x = Format()
        x.name = self.name
        x._fields = [f._copy() for f in self._fields]
        x._field_names = {}
        for field in x._fields:
            if field.name != "":
                x._field_names[field.name] = field
        return x

    @override
    def clear(self) -> None:
        self.vars = {}
        for fieldtype in self._fields:
            fieldtype.clear()

    @override
    def has_dynamic_size(self) -> bool:
        return False

    @override
    def is_const(self) -> bool:
        return all(fieldtype.is_const() for fieldtype in self._fields)

    @override
    def _get_value(self) -> list[Any]:
        values = []
        for fieldtype in self._fields:
            try:
                value = fieldtype._get_value()
                if isinstance(fieldtype, Field) and fieldtype._dtype._is_padding():
                    pass
                else:
                    values.append(value)
            except AttributeError:
                pass  # This field type doesn't have values, but that's fine in this context.
        return values

    @override
    def _set_value_with_kwargs(self, val: Sequence[Any], kwargs: dict[str, Any]) -> None:
        if len(val) != len(self._fields):
            raise ValueError(f"Can't set {len(self._fields)} fields from {len(val)} values.")
        for fieldtype, v in zip(self._fields, val):
            fieldtype._set_value_with_kwargs(v, kwargs)

    @override
    def to_bits(self) -> Bits:
        return Bits.from_joined(fieldtype.to_bits() for fieldtype in self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __getitem__(self, key: slice | str | int) -> Any:
        if isinstance(key, int):
            return self._fields[key]
        if isinstance(key, slice):
            return self.__class__.from_params(self._fields[key])
        try:
            return self._field_names[key]
        except KeyError:
            raise KeyError(f"Field with name '{key}' not found.")

    def __setitem__(self, key: slice | str | int, value: str | FieldType) -> None:
        if isinstance(value, str):
            try:
                _ = FieldType.from_string(value)
            except ValueError as e:
                raise ValueError(f"Can't set field from string: {e}") from None
        elif isinstance(value, FieldType):
            field = value
        else:
            raise ValueError(f"Can't create and set field from type '{type(value)}'.")
        if isinstance(key, int):
            self._fields[key] = field
        elif isinstance(key, slice):
            self._fields[key] = field
        elif isinstance(key, str):
            for i in range(len(self._fields)):
                if self._fields[i].name == key:
                    self._fields[i] = field
                    return
            raise KeyError(f"Field with name '{key}' not found.")
        else:
            raise TypeError(f"Invalid key type {type(key)}.")

    def __delitem__(self, key: slice | int | str):
        if isinstance(key, int):
            del self._fields[key]
        elif isinstance(key, slice):
            del self._fields[key]
        elif isinstance(key, str):
            for i, field in enumerate(self._fields):
                if field.name == key:
                    del self._fields[i]
                    return
            raise KeyError(f"Field with name '{key}' not found.")
        else:
            raise TypeError(f"Invalid key type {type(key)}.")

    @override
    def _info(self, use_colour: bool) -> str:
        field_count = len(self._fields)
        name_str = "" if self.name == "" else f" '{self.name}'"
        if field_count == 0:
            return f"an empty Format{name_str}."
        s = f"format{name_str} with {len(self._fields)} field{'' if field_count == 1 else 's'}."
        return s

    @override
    def _str(self, indent: Indenter, use_colour: bool) -> str:
        colour = Colour(use_colour)
        name_str = "" if self.name == "" else f"{colour.name}{self.name}{colour.off}: "
        s = ""
        s += indent(f"{name_str}(\n")
        if self._fields:
            with indent:
                for fieldtype in self._fields[:-1]:
                    s += fieldtype._str(indent, use_colour) + ",\n"
                s += self._fields[-1]._str(indent, use_colour) + '\n'
        s += indent(")")
        return s

    @override
    def _repr(self) -> str:
        name_str = "" if self.name == "" else f", name={self.name!r}"
        s = f"{self.__class__.__name__}.from_params(["
        for i, fieldtype in enumerate(self._fields):
            s += fieldtype._repr()
            if i != len(self._fields) - 1:
                s += ", "
        s += f"]{name_str})"
        return s

    def __iadd__(self, other: FieldType | str) -> Format:
        if isinstance(other, str):
            other = FieldType.from_string(other)
        field_copy = copy.copy(other)
        if other.name != "":
            self._field_names[other.name] = field_copy
        self._fields.append(field_copy)
        return self

    def __copy__(self) -> Format:
        x = Format()
        x.name = self.name
        x.vars = copy.deepcopy(self.vars)
        x._fields = [copy.copy(f) for f in self._fields]
        return x

    def __add__(self, other: FieldType | str) -> Format:
        """
        Add a field to a copy of the format.

        :param other: The field to add.
        :return: The updated format.
        """
        x = copy.copy(self)
        x.__iadd__(other)
        return x

    def append(self, value: Any) -> None:
        """
        Append a field to the format.

        :param value: The field to append.
        """
        self.__iadd__(value)

    def extend(self, values: Iterable) -> None:
        """
        Extend the format with multiple fields.

        :param values: The fields to add.
        """
        for value in values:
            self.__iadd__(value)

    @override
    def __eq__(self, other) -> bool:
        if not isinstance(other, Format):
            return False
        if self.name != other.name:
            return False
        if self.vars != other.vars:
            return False
        if self._fields != other._fields:
            return False
        return True

    @override
    def _get_name(self) -> str:
        return self._name

    @override
    def _set_name(self, name: str) -> None:
        self._name = validate_name(name)