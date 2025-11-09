from __future__ import annotations

from ast import literal_eval
from typing import Union, Iterable, Any
from tibs._options import Options
from tibs.rust import Tibs, Mutibs, bits_from_any
from collections.abc import Sequence

__all__ = ["Tibs", "Mutibs", "BitsType"]


# Things that can be converted to Tibs or Mutibs.
BitsType = Union["Tibs", "Mutibs", str, bytearray, bytes, memoryview]


def _validate_slice(length: int, start: int | None, end: int | None) -> tuple[int, int]:
    """Validate start and end and return them as positive bit positions."""
    start = 0 if start is None else (start + length if start < 0 else start)
    end = length if end is None else (end + length if end < 0 else end)
    if not 0 <= start <= end <= length:
        raise ValueError(f"Invalid slice positions for Tibs length {length}: start={start}, end={end}.")
    return start, end

def dtype_token_to_bits(token: str) -> Tibs:
    try:
        dtype_str, value_str = token.split("=", 1)
        dtype = Dtype.from_string(dtype_str)
    except ValueError:
        raise ValueError(f"Can't parse token '{token}'. It should be in the form 'kind[length][_endianness]=value' (e.g. "
                         "'u16_le = 44') or a literal starting with '0b', '0o' or '0x'.")
    if isinstance(dtype, DtypeSingle) and dtype._definition.return_type not in (bool, bytes):
        return dtype.pack(value_str)
    try:
        value = literal_eval(value_str)
    except ValueError:
        raise ValueError(f"Can't parse token '{token}'. The value '{value_str}' can't be converted to the appropriate type.")
    return dtype.pack(value)


class BaseBitsMethods:
    """Not a real class! This contains the common methods for Tibs and Mutibs, and they
are monkey-patched into those classes later. Yes, it would be more normal to use inheritance, but
this is a step to using the Rust classes as the base classes."""
    # ----- Instance Methods -----


    def find(self, bs: BitsType, /, start: int | None = None, end: int | None = None, byte_aligned: bool | None = None) -> int | None:
        """
        Find first occurrence of substring bs.

        Returns the bit position if found, or None if not found.

        :param bs: The Tibs to find.
        :param start: The starting bit position. Defaults to 0.
        :param end: The end position. Defaults to len(self).
        :param byte_aligned: If ``True``, the Tibs will only be found on byte boundaries.
        :return: The bit position if found, or None if not found.

        .. code-block:: pycon

            >>> Tibs.from_string('0xc3e').find('0b1111')
            6

        """
        bs = bits_from_any(bs)
        start, end = _validate_slice(len(self), start, end)
        if len(bs) == 0:
            raise ValueError("Cannot find an empty Tibs.")
        ba = Options().byte_aligned if byte_aligned is None else byte_aligned
        p = self._find(bs, start, end, bytealigned=ba)
        return p


    def rfind(self, bs: BitsType, /, start: int | None = None, end: int | None = None, byte_aligned: bool | None = None) -> int | None:
        """Find final occurrence of substring bs.

        Returns the bit position if found, or None if not found.
        Note that `start` and `end` define a slice in the usual way, so the
        occurrence of `bs` closest to the `end` posistion will be found.


        :param bs: The Tibs to find.
        :param start: The starting bit position of the slice to search. Defaults to 0.
        :param end: The end bit position of the slice to search. Defaults to len(self).
        :param byte_aligned: If True, the Tibs will only be found on byte boundaries.
        :return: The bit position if found, or None if not found.

        Raises ValueError if bs is empty.

        .. code-block:: pycon

            >>> Tibs('0b110110').rfind('0b1')
            4
            >>> Tibs('0b110110').rfind('0b0')
            5

        """
        bs = bits_from_any(bs)
        start, end = _validate_slice(len(self), start, end)
        ba = Options().byte_aligned if byte_aligned is None else byte_aligned
        if len(bs) == 0:
            raise ValueError("Cannot find an empty Tibs.")
        p = self._rfind(bs, start, end, ba)
        return p

    def unpack(self, dtype: Dtype | str | Sequence[Dtype | str], /, start: int | None = None, end: int | None = None) -> Any | list[Any]:
        """
        Interpret the Tibs as a given data type or list of data types.

        If a single Dtype is given then a single value will be returned, otherwise a list of values will be returned.
        A single Dtype with no length can be used to interpret the whole Tibs - in this common case properties
        are provided as a shortcut. For example instead of ``b.unpack('bin')`` you can use ``b.bin``.

        :param dtype: The data type used to interpret the Tibs.
        :param start: The starting bit position. Defaults to 0.
        :param end: The end position. Defaults to len(self).
        :return: The interpreted value(s).

        .. code-block:: pycon

            >>> s = Tibs('0xdeadbeef')
            >>> s.unpack(['bin4', 'u28'])
            ['1101', 246267631]
            >>> s.unpack(['f16', '[u4; 4]'])
            [-427.25, (11, 14, 14, 15)]
            >>> s.unpack('i')
            -559038737
            >>> s.i
            -559038737

        """
        if isinstance(dtype, str):
            dtype = Dtype.from_string(dtype)
        elif isinstance(dtype, Sequence):
            dtype = DtypeTuple.from_params(dtype)
        dtype = dtype.evaluate()

        if start is None and end is None:
            # If we have a known bit_length then try to unpack that much.
            if dtype.bit_length is not None:
                if dtype.bit_length == len(self):
                    return dtype._unpack_no_checks(self)
                elif dtype.bit_length < len(self):
                    return dtype._unpack_no_checks(self[:dtype.bit_length])
                else:
                    raise ValueError(f"Not enough bits to unpack the requested Dtype. {len(self)} bits are available but {dtype} needs {dtype.bit_length} bits.")
            elif isinstance(dtype, DtypeArray) and dtype._dtype_single.bit_length is not None:
                # For DtypeArray, unpack as many items as possible
                assert dtype.items is None
                if dtype._dtype_single.bit_length < len(self):
                    return dtype._unpack_no_checks(self[:len(self) - (len(self) % dtype._dtype_single.bit_length)])
                else:
                    raise ValueError(f"Not enough bits to unpack the requested Dtype. {len(self)} bits are available but {dtype} needs at least {dtype._dtype_single.bit_length} bits.")
            else:
                return dtype._unpack_no_checks(self)
        start, end = _validate_slice(len(self), start, end)
        return dtype.unpack(self[start:end])


    # ----- Other

    def __contains__(self, bs: BitsType, /) -> bool:
        """Return whether bs is contained in the current Tibs.

        bs -- The Tibs to search for.

        """
        found = BaseBitsMethods.find(self, bs, byte_aligned=False)
        return False if found is None else True



class BitsMethods:

    # ----- Class Methods -----

    @classmethod
    def from_dtype(cls, dtype: Dtype | str, value: Any, /) -> Tibs :
        """
        Pack a value according to a data type or data type tuple.

        :param dtype: The data type to pack.
        :param value: A value appropriate for the data type.
        :returns: A newly constructed ``Tibs``.

        .. code-block:: python

            a = Tibs.from_dtype("u8", 17)
            b = Tibs.from_dtype("f16, i4, bool", [2.25, -3, False])

        """
        if isinstance(dtype, str):
            dtype = Dtype.from_string(dtype)
        try:
            xt = dtype.pack(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Can't pack a value of {value} with a Dtype '{dtype}': {str(e)}")
        return xt

    def rfind_all(self, bs: BitsType, start: int | None = None, end: int | None = None,
                  count: int | None = None, byte_aligned: bool | None = None) -> Iterable[int]:
        """Find all occurrences of bs starting at the end. Return generator of bit positions.

        :param bs: The Tibs to find.
        :param start: The starting bit position of the slice to search. Defaults to 0.
        :param end: The end bit position of the slice to search. Defaults to len(self).
        :param count: The maximum number of occurrences to find.
        :param byte_aligned: If True, the Tibs will only be found on byte boundaries.
        :return: A generator yielding bit positions.

        Raises ValueError if bs is empty, if start < 0, if end > len(self) or
        if end < start.

        All occurrences of bs are found, even if they overlap.

        Note that this method is not available for :class:`Mutibs` as its value could change while the
        generator is still active. For that case you should convert to a :class:`Tibs` first with :meth:`Mutibs.to_bits`.

        .. code-block:: pycon

            >>> list(Tibs('0b10111011').rfind_all('0b11'))
            [6, 3, 2]

        """
        # TODO: This is only a (working) placeholder.
        return (p for p in reversed(list(self.find_all(bs, count, start, end, byte_aligned))))

    def find_all(self, bs: BitsType, start: int | None = None, end: int | None = None,
                 count: int | None = None, byte_aligned: bool | None = None) -> Iterable[int]:
        """Find all occurrences of bs. Return generator of bit positions.

        :param bs: The Tibs to find.
        :param start: The starting bit position of the slice to search. Defaults to 0.
        :param end: The end bit position of the slice to search. Defaults to len(self).
        :param count: The maximum number of occurrences to find.
        :param byte_aligned: If True, the Tibs will only be found on byte boundaries.
        :return: A generator yielding bit positions.

        Raises ValueError if bs is empty, if start < 0, if end > len(self) or
        if end < start.

        All occurrences of bs are found, even if they overlap.

        Note that this method is not available for :class:`Mutibs` as its value could change while the
        generator is still active. For that case you should convert to a :class:`Tibs` first with :meth:`Mutibs.to_bits`.

        .. code-block:: pycon

            >>> list(Tibs('0b10111011').find_all('0b11'))
            [2, 3, 6]

        """
        if count is not None and count < 0:
            raise ValueError("In find_all, count must be >= 0.")
        bs = bits_from_any(bs)
        start, end = _validate_slice(len(self), start, end)
        ba = Options().byte_aligned if byte_aligned is None else byte_aligned
        c = 0
        for i in self._findall(bs, start, end, ba):
            if count is not None and c >= count:
                return
            c += 1
            yield i
        return

    def __hash__(self) -> int:
        """Return an integer hash of the object."""
        # Only requirement is that equal Tibs should return the same hash.
        # For equal Tibs the bytes at the start/end will be the same and they will have the same length
        # (need to check the length as there could be zero padding when getting the bytes).
        length = len(self)
        if length <= 2000:
            # Use the whole Tibs.
            return hash((self.to_bytes(), length))
        else:
            # We can't in general hash the whole Tibs (it could take hours!)
            # So instead take some bits from the start and end.
            start = self._getslice(0, 800)
            end = self._getslice(length - 800, 800)
            return hash(((start + end).to_bytes(), length))

    def __getattr__(self, name):
        """Catch attribute errors and provide helpful messages for methods that exist in Mutibs."""
        # Check if the method exists in Mutibs
        if hasattr(Mutibs, name) and callable(getattr(Mutibs, name)) and not name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                f"Did you mean to use the Mutibs class? Or you could replace '.{name}(...)' with '.to_mutable_bits().{name}(...)'."
            )
        # Default behavior
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __copy__(self: Tibs) -> Tibs:
        """Return a new copy of the Tibs for the copy module.

        This can just return self as it's immutable.

        """
        return self


class MutableBitsMethods:

    # ----- Class Methods -----

    @classmethod
    def from_dtype(cls, dtype: Dtype | str, value: Any, /) -> Mutibs:
        """
        Pack a value according to a data type or data type tuple.

        :param dtype: The data type to pack.
        :param value: A value appropriate for the data type.
        :returns: A newly constructed ``Mutibs``.

        .. code-block:: python

            a = Mutibs.from_dtype("u8", 17)
            b = Mutibs.from_dtype("f16, i4, bool", [2.25, -3, False])

        """
        if isinstance(dtype, str):
            dtype = Dtype.from_string(dtype)
        try:
            xt = dtype.pack(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Can't pack a value of {value} with a Dtype '{dtype}': {str(e)}")
        return xt._as_mutable_bits()

    def __getattr__(self, name):
        """Catch attribute errors and provide helpful messages for methods that exist in Tibs."""
        # Check if the method exists in Tibs
        if hasattr(Tibs, name) and callable(getattr(Tibs, name)) and not name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                f"Did you mean to use the Tibs class? Or you could replace '.{name}(...)' with '.to_bits().{name}(...)'."
            )

        # Default behaviour
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


    def replace(self, old: BitsType, new: BitsType, /, start: int | None = None, end: int | None = None,
                count: int | None = None, byte_aligned: bool | None = None) -> Mutibs:
        """Replaces all occurrences of old with new. Returns self.

        :param old: The Tibs or to replace.
        :param new: The replacement Tibs.
        :param start: Any occurrences that start before this bit position will not be replaced.
        :param end: Any occurrences that finish after this bit position will not be replaced.
        :param count: The maximum number of replacements to make. Defaults to all.
        :param byte_aligned: If True, replacements will only be made on byte boundaries.
        :return: self

        :raises ValueError: if old is empty or if start or end are out of range.

        .. code-block:: pycon

            >>> s = Mutibs('0b10011')
            >>> s.replace('0b1', '0xf')
            Mutibs('0b11110011111111')

        """
        if count == 0:
            return self
        old_bits = bits_from_any(old)
        new_bits = bits_from_any(new)
        if len(old_bits) == 0:
            raise ValueError("Empty Tibs cannot be replaced.")
        start, end = _validate_slice(len(self), start, end)
        if byte_aligned is None:
            byte_aligned = Options().byte_aligned
        # First find all the places where we want to do the replacements
        starting_points: list[int] = []
        if byte_aligned:
            start += (8 - start % 8) % 8
        for x in self[start:end].as_bits().find_all(old, byte_aligned=byte_aligned):
            x += start
            if not starting_points:
                starting_points.append(x)
            elif x >= starting_points[-1] + len(old_bits):
                # Can only replace here if it hasn't already been replaced!
                starting_points.append(x)
            if count != 0 and len(starting_points) == count:
                break
        if not starting_points:
            return self
        original = self.as_bits()
        replacement_list = [original._getslice(0, starting_points[0])]
        for i in range(len(starting_points) - 1):
            replacement_list.append(new_bits)
            replacement_list.append(original._getslice(starting_points[i] + len(old_bits), starting_points[i + 1] - starting_points[i] - len(old_bits)))
        # Final replacement
        replacement_list.append(new_bits)
        replacement_list.append(original._getslice(starting_points[-1] + len(old_bits), len(original) - starting_points[-1] - len(old_bits)))
        self[:] = Mutibs.from_joined(replacement_list)
        return self


# Patching on the methods to Tibs and Mutibs to avoid inheritance.
def _patch_classes():
    for name, method in BaseBitsMethods.__dict__.items():
        if isinstance(method, classmethod):
            setattr(Tibs, name, classmethod(method.__func__))
            setattr(Mutibs, name, classmethod(method.__func__))
        elif callable(method):
            setattr(Tibs, name, method)
            setattr(Mutibs, name, method)

    for name, method in BitsMethods.__dict__.items():
        if isinstance(method, classmethod):
            setattr(Tibs, name, classmethod(method.__func__))
        elif callable(method):
            setattr(Tibs, name, method)

    for name, method in MutableBitsMethods.__dict__.items():
        if isinstance(method, classmethod):
            setattr(Mutibs, name, classmethod(method.__func__))
        elif callable(method):
            setattr(Mutibs, name, method)


# The hash method is not available for a ``Mutibs`` object as it is mutable.
Mutibs.__hash__ = None


_patch_classes()

Sequence.register(Tibs)
Sequence.register(Mutibs)
