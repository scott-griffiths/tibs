from __future__ import annotations
import abc
import sys

from bitformat import Bits
from ._dtypes import DtypeTransformer
from ._bits import BitsType
from ._common import final, Indenter, field_parser
from typing import Any, Sequence, TextIO
from ._options import Options
from lark import UnexpectedInput
import lark
from .rust import bits_from_any

__all__ = ["FieldType"]

fieldtype_classes: dict[str, FieldType] = {}


class FieldTypeTransformer(DtypeTransformer):

    @staticmethod
    def field_name(items) -> str:
        return items[0]

    @staticmethod
    def const_field(items) -> 'Field':
        assert len(items) == 3
        name = items[0] if items[0] is not None else ''
        dtype = items[1]
        value = items[2]
        return fieldtype_classes['Field'].from_params(dtype, name, value, const=True)

    @staticmethod
    def mutable_field(items) -> 'Field':
        assert len(items) == 3
        name = items[0] if items[0] is not None else ''
        dtype = items[1]
        value = items[2]
        return fieldtype_classes['Field'].from_params(dtype, name, value)

    @staticmethod
    def repeat(items) -> 'Repeat':
        count = items[0]
        field = items[1]
        value = items[2]
        return fieldtype_classes['Repeat'].from_params(count, field, value)

    @staticmethod
    def while_(items) -> 'While':
        return fieldtype_classes['While'].from_params(items[0], items[1])

    @staticmethod
    def pass_(items) -> 'Pass':
        assert len(items) == 0
        return fieldtype_classes['Pass'].from_params()

    @staticmethod
    def let(items) -> 'Let':
        return fieldtype_classes['Let'].from_params(items[0], items[1])

    @staticmethod
    def if_(items) -> 'If':
        assert len(items) == 3
        expr = items[0]
        then_ = items[1]
        else_ = items[2]
        return fieldtype_classes['If'].from_params(expr, then_, else_)

    @staticmethod
    def format(items) -> 'Format':
        assert len(items) >= 1
        name = items[0] if items[0] is not None else ''
        fields = [i for i in items[1:] if i is not None]
        return fieldtype_classes['Format'].from_params(fields, name)

field_type_transformer = FieldTypeTransformer()


class FieldType(abc.ABC):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        fieldtype_classes[cls.__name__] = cls

    @final
    def parse(self, b: BitsType | None = None, /, **kwargs) -> int:
        """
        Parse the field type from the supplied bits.

        The parsing is done lazily, so any values might not be calculated at this point, instead waiting until
        they are explicitly asked for.

        :param b: The bits to parse.
        :return: The number of bits used.
        """
        b = bits_from_any(b) if b is not None else Bits()
        self.clear()
        try:
            return self._parse(b, 0, kwargs)
        except ValueError as e:
            raise ValueError(f"Error parsing field {self}: {str(e)}")

    @final
    def pack(self, values: Sequence[Any] | Any, /, **kwargs) -> Bits:
        """
        Pack with values for each non-const field.

        :param values: The values to pack.
        :param kwargs: Additional keyword arguments.

        :return: The packed data as a :class:`Bits`.
        """
        if kwargs is None:
            kwargs = {}
        _ = self._pack(values, kwargs)
        return self.to_bits()

    @final
    def unpack(self, b: BitsType | None = None, **kwargs) -> Any | list[Any]:
        """
        Unpack the field type from bits.

        :param b: The bits to unpack.
        :return: The unpacked value.
        """
        if b is not None:
            self.parse(b, **kwargs)
        v = self.value
        if v is None:
            raise ValueError("Cannot unpack field as it has no value.")
        return v

    @final
    def __str__(self) -> str:
        """
        Return a string representation of the field type.

        :return: The string representation.
        """
        return self._str(Indenter(Options().indent_size), not Options().no_color)

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the field type.

        :return: The detailed string representation.
        """
        return self._repr()

    def pp(self, stream: TextIO = sys.stdout,indent: int | None = None, depth: int | None = None) -> None:
        """
        Pretty-print the fieldtype to a stream (or stdout by default).

        :param stream: The stream to write to.
        :param indent: The number of spaces for each level of indentation. Defaults to Options().indent_size which defaults to 4.
        :param depth: The maximum depth to print, or None for no limit.
        """
        stream.write(self._str(Indenter(indent_size=indent, max_depth=depth), not Options().no_color))
        stream.write("\n")

    @classmethod
    def from_string(cls, s: str) -> FieldType:
        """
        Create a FieldType instance from a string.

        The type is inferred from the string, so it can be a Field, Format, or other FieldType.

        :param s: The string to parse.
        :return: The FieldType instance.
        """
        try:
            tree = field_parser.parse(s)
        except UnexpectedInput as e:
            raise ValueError(f"Unexpected input: {e}")
        try:
            return field_type_transformer.transform(tree)
        except lark.exceptions.VisitError as e:
            raise ValueError(f"Error parsing FieldType: {e}")

    @abc.abstractmethod
    def _parse(self, b: Bits, startbit: int, vars_: dict[str, Any]) -> int:
        """
        Parse the field from the bits, using the vars_ dictionary to resolve any expressions.

        Return the number of bits used.

        """
        ...

    @abc.abstractmethod
    def _pack(self, value: Any | Sequence[Any], kwargs: dict[str, Any]) -> bool:
        """
        Build the field from the values list, starting at index.

        Returns True if it consumed `value`, False otherwise
        """
        ...

    @abc.abstractmethod
    def to_bits(self) -> Bits:
        """
        Return the bits that represent the field.

        :return: The bits that represent the field.
        """
        ...

    @final
    def to_bytes(self) -> bytes:
        """
        Return the bytes that represent the field. Pads with up to 7 zero bits if necessary.

        :return: The bytes that represent the field.
        """
        b = self.to_bits()
        return b.to_bytes()

    @abc.abstractmethod
    def clear(self) -> None:
        """
        Clear the value of the field, unless it is a constant.
        """
        ...

    def info(self) -> str:
        """
        Return a descriptive string with information about the FieldType.

        Note that the output is designed to be helpful to users and is not considered part of the API.
        You should not use the output programmatically as it may change even between point versions.
        """
        return self._info(not Options().no_color)

    @abc.abstractmethod
    def _info(self, use_colour: bool) -> str:
        ...

    @classmethod
    @abc.abstractmethod
    def from_params(cls, *args, **kwargs) -> FieldType:
        ...

    @abc.abstractmethod
    def _str(self, indent: Indenter, use_colour: bool) -> str: ...

    @abc.abstractmethod
    def _repr(self) -> str: ...

    @abc.abstractmethod
    def _copy(self) -> FieldType: ...

    @abc.abstractmethod
    def _get_value(self) -> Any: ...

    @abc.abstractmethod
    def _set_value_with_kwargs(self, value: Any, kwargs: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    def _get_bit_length(self) -> int:
        """
        Return the length of the FieldType in bits.

        :return: The length in bits.

        Raises ValueError if length cannot be calculated or known.
        """
        ...

    @property
    def bit_length(self) -> int:
        return self._get_bit_length()

    @abc.abstractmethod
    def has_dynamic_size(self) -> bool:
        """Returns whether this FieldType can stretch to fit the available data."""
        ...

    @abc.abstractmethod
    def is_const(self) -> bool:
        """Returns whether this FieldType is a constant, so doesn't need any values to be packed."""
        ...

    @abc.abstractmethod
    def __eq__(self, other) -> bool: ...

    def __copy__(self) -> FieldType:
        return self._copy()

    @property
    def value(self) -> Any:
        return self._get_value()

    @value.setter
    def value(self, val: Any) -> None:
        self._set_value_with_kwargs(val, {})

    @abc.abstractmethod
    def _get_name(self) -> str | None:
        ...

    @abc.abstractmethod
    def _set_name(self, name: str) -> None:
        ...

    @property
    def name(self) -> str | None:
        return self._get_name()

    @name.setter
    def name(self, name: str) -> None:
        self._set_name(name)


