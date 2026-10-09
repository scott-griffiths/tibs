r"""Fast, bit-level binary data.

``Tibs`` is an immutable sequence of bits, as ``bytes`` is of bytes, and
``Mutibs`` is its mutable counterpart, as ``bytearray`` is. Lengths and
positions are in bits, so nothing needs to be byte aligned. Use them for
packets, registers, instruction formats, file headers, bitsets and streams
whose fields can be any number of bits long.

There are three ways to use them:

* As a container of bits: slice, concatenate, compare, ``find``,
  ``find_all``, ``replace``, ``count``, ``starts_with``, ``split_at`` and
  ``chunks``, all with bit positions.
* As typed fields: build with ``from_u``, ``from_i``, ``from_f``,
  ``from_bytes`` or ``from_value``, and read back with ``.u``, ``.i``,
  ``.f``, ``.hex``, ``.bin`` and ``.bytes``. ``.le`` and ``.be`` give
  byte-order views, ``.lsb0`` numbers bits from the least significant end,
  ``field`` takes inclusive datasheet-style labels, ``Dtype`` describes
  compound records, and ``Reader`` walks through a stream.
* As a set of bits: ``&``, ``|``, ``^`` and ``~``, plus ``count_and``,
  ``count_xor``, ``intersects`` and ``is_subset_of``, which count or test
  without building an intermediate result.

Anywhere a ``Tibs`` is expected, a string such as ``'0xff'``, ``'0b101'``
or ``'0o7, 0b1'``, a ``bytes``-like object or a list of 0s and 1s is
accepted too.

.. code-block:: pycon

    >>> from tibs import Tibs, Mutibs, Reader
    >>> frame = Tibs('0b101') + Tibs.from_u(1000, 12) + b'ok'
    >>> len(frame), frame[3:15].u, frame.find(b'ok')
    (31, 1000, 15)
    >>> Tibs(b'\x34\x12').le.u
    4660
    >>> r = Reader(frame)
    >>> r.read_value('u3'), r.read_value('u12'), r.read_bits(16).bytes
    (5, 1000, b'ok')
    >>> m = Mutibs.from_zeros(8)
    >>> m.set([0, 7])
    >>> m.bin
    '10000001'

The full documentation is at https://tibs.readthedocs.io/.
"""

from __future__ import annotations
from contextlib import AbstractContextManager
from typing import Any, Callable, ClassVar, overload, Iterable, Iterator, SupportsIndex, Union

__version__: str
__author__: str

_BitPattern = list[bool | int] | tuple[bool | int, ...]
_TibsLike = Union["Tibs", "Mutibs", str, bytes, bytearray, memoryview, _BitPattern]
_DtypeLike = Union["Dtype", "DtypeKind", str]
_DtypeScalarValue = Union[int, float, str, bytes, bool, "Tibs"]
_DtypeValue = Union[_DtypeScalarValue, tuple["_DtypeValue", ...]]


class ReadError(ValueError):
    """A requested read could not be completed."""


class DecodeError(ValueError):
    """Encoded Tibs data could not be decoded."""


class ByteOrder:
    """The order of the bytes of a whole-byte value.

    ``Unspecified`` is the default and is interpreted bitwise big-endian, which
    for whole-byte data is the same as ``Big``, but can be used at any length.
    ``Big`` and ``Little`` require a whole number of bytes.
    """

    Unspecified: ClassVar[ByteOrder]
    Big: ClassVar[ByteOrder]
    Little: ClassVar[ByteOrder]


class BitOrder:
    """How bit labels are mapped within each byte by a :class:`View`.

    ``Msb0`` is the default, and matches ordinary indexing: label 0 is the most
    significant bit of the byte. ``Lsb0`` numbers from the least significant bit
    instead, as many hardware manuals do. Only the labels differ, never the
    stored data.
    """

    Msb0: ClassVar[BitOrder]
    Lsb0: ClassVar[BitOrder]


class Codec:
    """The storage strategy used by :meth:`Tibs.encode`.

    ``Auto`` is the default and picks whatever is most compact. ``Raw`` stores the
    bits directly and is the canonical form to use when the exact bytes matter.
    ``Rice`` suits sparse data, and ``Zstd`` larger byte-like data.
    """

    Auto: ClassVar[Codec]
    Raw: ClassVar[Codec]
    Rice: ClassVar[Codec]
    Zstd: ClassVar[Codec]


class DtypeKind:
    """The kind of value a :class:`DtypeSingle` encodes.

    The kind says how bits become a Python value, and the dtype's length says how
    many bits one value takes. ``Uint``, ``Int``, ``Float``, ``Bits``, ``Bin``,
    ``Oct``, ``Hex`` and ``Bytes`` are families of widths, so a dtype using one
    always needs a length. The remaining thirteen - ``Bool``, ``BFloat`` and the
    narrow float formats - have an intrinsic width, so the kind on its own can be
    used wherever a dtype is accepted.

    Every kind, and the dtype string it corresponds to, is listed in the
    documentation for this enum.
    """

    Uint: ClassVar[DtypeKind]
    Int: ClassVar[DtypeKind]
    Float: ClassVar[DtypeKind]
    BFloat: ClassVar[DtypeKind]
    Bool: ClassVar[DtypeKind]
    Bits: ClassVar[DtypeKind]
    Bytes: ClassVar[DtypeKind]
    Bin: ClassVar[DtypeKind]
    Oct: ClassVar[DtypeKind]
    Hex: ClassVar[DtypeKind]
    Binary8P3: ClassVar[DtypeKind]
    Binary8P4: ClassVar[DtypeKind]
    OcpE4M3Saturate: ClassVar[DtypeKind]
    OcpE4M3Overflow: ClassVar[DtypeKind]
    OcpE5M2Saturate: ClassVar[DtypeKind]
    OcpE5M2Overflow: ClassVar[DtypeKind]
    OcpE3M2: ClassVar[DtypeKind]
    OcpE2M3: ClassVar[DtypeKind]
    OcpE2M1: ClassVar[DtypeKind]
    OcpE8M0: ClassVar[DtypeKind]
    OcpInt8: ClassVar[DtypeKind]


class Dtype:
    """The base class for fixed-width value descriptors.

    Constructing ``Dtype`` parses ``spec`` and returns a
    :class:`DtypeSingle`, :class:`DtypeArray` or :class:`DtypeTuple`.

    :param str spec: A scalar, array or tuple dtype specification.
    :return: The corresponding concrete dtype.

    .. code-block:: pycon

        >>> Dtype("u16_le")
        DtypeSingle('u16_le')
        >>> Dtype("[(u8, bool); 2]")
        DtypeArray('[(u8, bool); 2]')
    """

    def __init__(self, spec: str, /) -> None: ...

    @property
    def length(self) -> int:
        """The number of bits used by one complete value."""

    def pack(self, value: Any, /) -> Tibs:
        """Encode one Python value as a :class:`Tibs`.

        :param object value: A scalar or structured value matching this dtype.
        :return: The encoded bits.
        :raises ValueError: if a structured value has the wrong number of items.

        .. code-block:: pycon

            >>> Dtype("(u8, bool)").pack((15, True))
            Tibs('0b000011111')
        """

    def pack_values(self, iterable: Iterable[Any], /) -> Tibs:
        """Encode and concatenate Python values as a :class:`Tibs`.

        :param Iterable iterable: Values matching this dtype.
        :return: The concatenated encoded bits.

        .. code-block:: pycon

            >>> Dtype("u8").pack_values([1, 2, 3])
            Tibs('0x010203')
        """

    def unpack(self, bits: _TibsLike, /, start: int | None = None,
               end: int | None = None) -> _DtypeValue:
        """Decode one complete value from a bit sequence.

        :param object bits: Anything promotable to :class:`Tibs`.
        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to ``len(bits)``.
        :return: One scalar value, or a tuple for an array or tuple dtype.
        :raises ValueError: if the selected range is not exactly :attr:`length` bits.
        """

    def unpack_values(self, bits: _TibsLike, /, start: int | None = None,
                      end: int | None = None) -> list[_DtypeValue]:
        """Decode a list of complete values from a bit sequence.

        :param object bits: Anything promotable to :class:`Tibs`.
        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to ``len(bits)``.
        :return: A list of decoded values.
        :raises ValueError: if the selected range is not a multiple of :attr:`length`.
        """

    def unpack_values_iter(self, bits: _TibsLike, /, start: int | None = None,
                           end: int | None = None) -> Iterator[_DtypeValue]:
        """Lazily decode complete values from a bit sequence.

        :param object bits: Anything promotable to :class:`Tibs`.
        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to ``len(bits)``.
        :return: An iterator of decoded values.
        :raises ValueError: if the selected range is not a multiple of :attr:`length`.
        """

    def __eq__(self, other: Any) -> bool: ...

    def __hash__(self) -> int: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...


class DtypeSingle(Dtype):
    """A scalar dtype with a kind, bit length and optional byte order.

    Construct directly from a scalar specification, use
    :meth:`from_params`, or obtain one from the :class:`Dtype` factory.
    """

    def __init__(self, spec: str, /) -> None: ...

    @classmethod
    def from_params(
            cls,
            kind: DtypeKind,
            length: int | None = None,
            /,
            byte_order: ByteOrder | None = ByteOrder.Unspecified
    ) -> DtypeSingle:
        """Construct a scalar dtype from explicit parameters.

        :param DtypeKind kind: The scalar value kind.
        :param int | None length: The positive bit length. May be omitted for a
            kind that fixes its own length, such as ``DtypeKind.OcpE2M1`` or
            ``DtypeKind.Bool``.
        :param ByteOrder | None byte_order: The byte order. Defaults to unspecified.
        :return: A scalar dtype.

            .. code-block:: pycon

                >>> DtypeSingle.from_params(DtypeKind.Uint, 12)
                DtypeSingle('u12')
                >>> DtypeSingle.from_params(DtypeKind.OcpE2M1)
                DtypeSingle('ocp_e2m1')
        """

    @property
    def kind(self) -> DtypeKind:
        """The scalar value kind."""

    @property
    def byte_order(self) -> ByteOrder:
        """The scalar byte order."""


class DtypeArray(Dtype):
    """A fixed positive number of repetitions of another dtype.

    Array values pack from any iterable with exactly :attr:`count` items
    and unpack to Python tuples.
    """

    def __init__(self, spec: str, /) -> None: ...

    @classmethod
    def from_params(cls, dtype: _DtypeLike, count: int, /) -> DtypeArray:
        """Construct an array dtype from its element dtype and count.

        :param Dtype | str dtype: The element dtype.
        :param int count: The positive number of elements.
        :return: An array dtype.
        """

    @property
    def dtype(self) -> Dtype:
        """The dtype repeated by this array."""

    @property
    def count(self) -> int:
        """The number of values in this array."""


class DtypeTuple(Dtype):
    """An ordered, non-empty tuple of dtypes.

    Tuple values pack from any iterable with exactly one item per field and
    unpack to Python tuples.
    """

    def __init__(self, spec: str, /) -> None: ...

    @classmethod
    def from_params(cls, dtypes: Iterable[_DtypeLike], /) -> DtypeTuple:
        """Construct a tuple dtype from its field dtypes.

        :param Iterable dtypes: A non-empty iterable of dtypes or dtype strings.
        :return: A tuple dtype.
        """

    @property
    def dtypes(self) -> tuple[Dtype, ...]:
        """The field dtypes as an immutable Python tuple."""


class View:
    """A view of a :class:`Tibs` with different interpretation settings.

    A ``View`` does not change the underlying bits. It records how operations such as
    integer conversion, byte conversion and field extraction should interpret those
    bits.

    Views are usually created from :class:`Tibs` instances using the
    :attr:`~Tibs.le`, :attr:`~Tibs.be`, :attr:`~Tibs.lsb0`, :attr:`~Tibs.msb0`
    or :meth:`~Tibs.view` helpers.

    The source must be a :class:`Tibs`, which a view borrows without copying.
    A :class:`Mutibs` is not accepted, because its bits can change underneath
    a view that cannot see the change: use :meth:`Mutibs.view` for a live
    :class:`MutableView`, or :meth:`Mutibs.to_tibs` for an immutable copy to
    view instead.

    .. code-block:: pycon

        >>> t = Tibs('0x0100')
        >>> t.le.u
        1
        >>> t.lsb0.hex
        '0001'
    """

    def __init__(
            self,
            source: Tibs,
            /,
            byte_order: ByteOrder = ByteOrder.Unspecified,
            bit_order: BitOrder = BitOrder.Msb0
    ) -> None: ...

    @classmethod
    def from_indices(
            cls,
            source: Tibs,
            indices: Iterable[int],
            /,
            byte_order: ByteOrder = ByteOrder.Unspecified,
            bit_order: BitOrder = BitOrder.Msb0
    ) -> View:
        """Create a view by materializing selected source bit positions.

        ``indices`` may be a ``range`` or any iterable of integers. It maps
        each viewed bit to a physical bit position in the source, which must be
        a :class:`Tibs`.

        This is a low-level reconstruction API. Use :meth:`~field` for normal
        specification-labelled fields.

        :raises TypeError: if ``source`` is not a :class:`Tibs`.

        .. code-block:: pycon

            >>> View.from_indices(Tibs('0xf0'), range(0, 4)).bin
            '1111'
            >>> View.from_indices(Tibs('0xf0'), [7, 6, 5, 4]).bin
            '0000'
        """

    def view(
            self,
            byte_order: ByteOrder | None = None,
            bit_order: BitOrder | None = None
    ) -> View:
        """Return a view with updated interpretation settings.

        Any setting left as ``None`` keeps its current value.

        Byte-oriented views must have a whole-byte length. This applies when using
        little-endian or big-endian byte order, or when using ``BitOrder.Lsb0``.

        :param ByteOrder byte_order: The byte order to use, or ``None`` to keep the current byte order.
        :param BitOrder bit_order: The bit order to use, or ``None`` to keep the current bit order.
        :return: A new ``View``.

        .. code-block:: pycon

            >>> Tibs('0x0100').view(byte_order=ByteOrder.Little).u
            1
        """

    @property
    def le(self) -> View:
        """Return a little-endian byte-order view.

        Equivalent to ``view(byte_order=ByteOrder.Little)``.

        The view length must be a whole number of bytes.
        """

    @property
    def be(self) -> View:
        """Return a big-endian byte-order view.

        Equivalent to ``view(byte_order=ByteOrder.Big)``.

        The view length must be a whole number of bytes.
        """

    @property
    def lsb0(self) -> View:
        """Return an LSB0 bit-order view.

        ``BitOrder.Lsb0`` means that field labels are counted from the least
        significant bit of each byte. The view length must be a whole number of
        bytes.

        Equivalent to ``view(bit_order=BitOrder.Lsb0)``.
        """

    @property
    def msb0(self) -> View:
        """Return an MSB0 bit-order view.

        ``BitOrder.Msb0`` means that field labels are counted from the most
        significant bit of each byte. This is the default bit order.

        Equivalent to ``view(bit_order=BitOrder.Msb0)``.
        """

    @property
    def byte_order(self) -> ByteOrder:
        """Return the byte-order interpretation setting for this view."""

    @property
    def bit_order(self) -> BitOrder:
        """Return the bit-order interpretation setting for this view."""

    def to_u(self) -> int:
        """Interpret the viewed bits as an unsigned integer.

        :return: The unsigned integer value.

        .. code-block:: pycon

            >>> Tibs('0x0100').le.to_u()
            1
        """

    def to_i(self) -> int:
        """Interpret the viewed bits as a signed integer.

        :return: The signed integer value.
        """

    def to_f(self) -> float:
        """Interpret the viewed bits as an IEEE floating point value.

        The viewed length must be 16, 32 or 64 bits.

        :return: The floating point value.
        """

    def to_bin(self) -> str:
        """Return the viewed bits as a binary string.

        :return: The binary representation as a string.
        """

    def to_oct(self) -> str:
        """Return the viewed bits as an octal string.

        :return: The octal representation as a string.
        """

    def to_hex(self) -> str:
        """Return the viewed bits as a hexadecimal string.

        :return: The hexadecimal representation as a string.

        .. code-block:: pycon

            >>> Tibs('0x0100').le.to_hex()
            '0001'
        """

    def to_bytes(self) -> bytes:
        """Return the viewed bits as bytes.

        The viewed length must be a whole number of bytes.

        :return: A ``bytes`` value.
        """

    def __bytes__(self) -> bytes:
        """Return the viewed bits as bytes."""

    def to_value(self, dtype: _DtypeLike, /, start: int | None = None, end: int | None = None) -> _DtypeValue:
        """Return one value decoded from the viewed bits with a dtype.

        The byte order and bit order of the view are applied first, so the dtype
        decodes the value denoted by the view rather than the source bits in
        storage. This is the same rule the other conversions follow, which makes
        this equivalent to ``view.to_tibs().to_value(dtype, start, end)``.
        ``start`` and ``end`` are therefore positions within the viewed bits.

        Byte order is stated in one place. An ``le``, ``be`` or ``lsb0`` view
        already states it, so a dtype that names its own byte order, such as
        ``"u16_le"``, is refused there rather than applied on top. Give the byte
        order to the view or to the dtype, not to both.

        The selected range must have exactly the dtype length.

        :param Dtype | str dtype: The value encoding to use.
        :param int | None start: Start bit position within the view. Defaults to 0.
        :param int | None end: End bit position within the view. Defaults to len(self).
        :return: The decoded Python value.
        :raises ValueError: if the selected range is not exactly the dtype length, or if the dtype names a byte order and the view does too.

        .. code-block:: pycon

            >>> Tibs('0x0000803f').le.to_value("f32")
            1.0
            >>> Tibs('0x23a11234').lsb0.le.field(31, 16).to_value("u16")
            13330
        """

    def to_tibs(self) -> Tibs:
        """Materialize the view as a new :class:`Tibs`.

        :return: A :class:`Tibs` containing the viewed bits.
        """

    def to_mutibs(self) -> Mutibs:
        """Materialize the view as a new :class:`Mutibs`.

        :return: A :class:`Mutibs` containing the viewed bits.
        """

    @property
    def u(self) -> int:
        """Interpret the viewed bits as an unsigned integer.

        Equivalent to using :meth:`~to_u`.
        """

    @property
    def i(self) -> int:
        """Interpret the viewed bits as a signed integer.

        Equivalent to using :meth:`~to_i`.
        """

    @property
    def f(self) -> float:
        """Interpret the viewed bits as an IEEE floating point value.

        Equivalent to using :meth:`~to_f`.
        """

    @property
    def bin(self) -> str:
        """Return the viewed bits as a binary string.

        Equivalent to using :meth:`~to_bin`.
        """

    @property
    def oct(self) -> str:
        """Return the viewed bits as an octal string.

        Equivalent to using :meth:`~to_oct`.
        """

    @property
    def hex(self) -> str:
        """Return the viewed bits as a hexadecimal string.

        Equivalent to using :meth:`~to_hex`.
        """

    @property
    def bytes(self) -> bytes:
        """Return the viewed bits as bytes.

        Equivalent to using :meth:`~to_bytes`.
        """

    def field(self, a: int, b: int, /) -> View:
        """Extract a field using inclusive bit labels.

        This is intended for specifications that describe fields using inclusive
        bit labels such as ``31:26``. ``a`` and ``b`` must be zero or positive bit
        labels. The endpoints may be provided in either order.

        For an LSB0 view, labels are interpreted within each byte with bit 0 at the
        least significant bit. For an MSB0 view, labels match normal Python slice
        positions.

        Labels are selected in field-value order after endpoint normalization.
        This means LSB0 labels identify the physical bits, while the returned
        field is not bit-reversed. The returned view has ``BitOrder.Msb0``
        because the selected bits have been materialized. The current byte order
        is kept for whole-byte fields and dropped for non-whole-byte fields.

        :param int a: One non-negative inclusive field endpoint.
        :param int b: The other non-negative inclusive field endpoint.
        :return: A new ``View`` containing the field.

        .. code-block:: pycon

            >>> t = Tibs('0x88040410')
            >>> t.lsb0.field(31, 26).u
            4
        """

    def __len__(self) -> int: ...

    def __eq__(self, other: Any) -> bool: ...

    def __repr__(self) -> str: ...

    def __format__(self, format_spec: str, /) -> str:
        """Return a string formatted according to the Python format mini-language.

        The viewed bits are formatted, so the byte order and bit order of the view are
        applied first. See :meth:`Tibs.__format__` for the accepted format specs.

        An empty format spec gives the ``repr``, as :func:`str` of a view does.

        :param str format_spec: The format specification.
        :return: The formatted string.

        :raises ValueError: if the spec cannot be parsed, if a type code is used that needs a length that is a different multiple, or if a sign or comma grouping is used with a bit representation type code.

        .. code-block:: pycon

            >>> f"{Tibs('0x0100').le:#x}"
            '0x0001'
        """


class MutableView:
    """A live mutable view of a :class:`Mutibs` with different interpretation settings.

    ``MutableView`` records how operations such as integer conversion, byte
    conversion and field extraction should interpret the source bits. Unlike
    :class:`View`, it keeps a live reference to the source ``Mutibs``.

    Assigning through ``u``, ``i`` or ``f`` mutates the source ``Mutibs`` without
    changing its length.
    """

    def __init__(
            self,
            source: Mutibs,
            /,
            byte_order: ByteOrder = ByteOrder.Unspecified,
            bit_order: BitOrder = BitOrder.Msb0
    ) -> None: ...

    @classmethod
    def from_indices(
            cls,
            source: Mutibs,
            indices: Iterable[int],
            /,
            byte_order: ByteOrder = ByteOrder.Unspecified,
            bit_order: BitOrder = BitOrder.Msb0
    ) -> MutableView:
        """Create a live mutable view from source bit positions.

        ``indices`` may be a ``range`` or any iterable of integers. It maps
        each viewed bit to a physical bit position in the source ``Mutibs``.

        This is a low-level reconstruction API. Use :meth:`~field` for normal
        specification-labelled fields.

        .. code-block:: pycon

            >>> m = Mutibs('0x00')
            >>> view = MutableView.from_indices(m, range(0, 8, 2))
            >>> view.bin = '1111'
            >>> m.bin
            '10101010'
        """

    def view(
            self,
            byte_order: ByteOrder | None = None,
            bit_order: BitOrder | None = None
    ) -> MutableView:
        """Return a mutable view with updated interpretation settings.

        Any setting left as ``None`` keeps its current value.
        """

    @property
    def le(self) -> MutableView:
        """Return a little-endian byte-order mutable view."""

    @property
    def be(self) -> MutableView:
        """Return a big-endian byte-order mutable view."""

    @property
    def lsb0(self) -> MutableView:
        """Return an LSB0 bit-order mutable view."""

    @property
    def msb0(self) -> MutableView:
        """Return an MSB0 bit-order mutable view."""

    @property
    def byte_order(self) -> ByteOrder:
        """Return the byte-order interpretation setting for this mutable view."""

    @property
    def bit_order(self) -> BitOrder:
        """Return the bit-order interpretation setting for this mutable view."""

    def to_u(self) -> int:
        """Interpret the viewed bits as an unsigned integer."""

    def write_u(self, u: int, /) -> None:
        """Write the viewed bits from an unsigned integer without changing the source length."""

    def to_i(self) -> int:
        """Interpret the viewed bits as a signed integer."""

    def write_i(self, i: int, /) -> None:
        """Write the viewed bits from a signed integer without changing the source length."""

    def to_f(self) -> float:
        """Interpret the viewed bits as an IEEE floating point value."""

    def write_f(self, f: float, /) -> None:
        """Write the viewed bits from a floating point number without changing the source length."""

    def to_bin(self) -> str:
        """Return the viewed bits as a binary string."""

    def write_bin(self, s: str, /) -> None:
        """Write the viewed bits from a binary string without changing the view length."""

    def to_oct(self) -> str:
        """Return the viewed bits as an octal string."""

    def write_oct(self, s: str, /) -> None:
        """Write the viewed bits from an octal string without changing the view length."""

    def to_hex(self) -> str:
        """Return the viewed bits as a hexadecimal string."""

    def write_hex(self, s: str, /) -> None:
        """Write the viewed bits from a hexadecimal string without changing the view length."""

    def to_bytes(self) -> bytes:
        """Return the viewed bits as bytes."""

    def __bytes__(self) -> bytes:
        """Return the viewed bits as bytes."""

    def write_bytes(self, data: bytes | bytearray | memoryview, /) -> None:
        """Write the viewed bits from a bytes-like object without changing the view length."""

    def to_value(self, dtype: _DtypeLike, /, start: int | None = None, end: int | None = None) -> _DtypeValue:
        """Return one value decoded from the viewed bits with a dtype.

        The byte order and bit order of the view are applied first, so the dtype
        decodes the value denoted by the view rather than the source bits in
        storage. An ``le``, ``be`` or ``lsb0`` view already states the byte
        order, so a dtype that names its own is refused. See
        :meth:`View.to_value`.

        :param Dtype | str dtype: The value encoding to use.
        :param int | None start: Start bit position within the view. Defaults to 0.
        :param int | None end: End bit position within the view. Defaults to len(self).
        :return: The decoded Python value.
        :raises ValueError: if the selected range is not exactly the dtype length, or if the dtype names a byte order and the view does too.

        .. code-block:: pycon

            >>> Mutibs('0x0000803f').le.to_value("f32")
            1.0
        """

    def write_value(self, dtype: _DtypeLike, value: Any, /) -> None:
        """Write the viewed bits from one value encoded with a dtype.

        The value is encoded first and the result is then written through the
        view, so the byte order and bit order of the view are applied to the
        encoded bits. This is the write direction of :meth:`~to_value`, and it
        refuses a dtype that names its own byte order for the same reason.

        The dtype length must match the view length, as a ``MutableView`` never
        changes the length of its source.

        :param Dtype | str dtype: The value encoding to use.
        :param object value: The value to encode.
        :raises ValueError: if the dtype length is not the view length, or if the dtype names a byte order and the view does too.

        .. code-block:: pycon

            >>> m = Mutibs.from_zeros(32)
            >>> m.le.write_value("f32", 1.0)
            >>> m.hex
            '0000803f'
        """

    def to_tibs(self) -> Tibs:
        """Materialize the current view as a new :class:`Tibs`."""

    def to_mutibs(self) -> Mutibs:
        """Materialize the current view as a new :class:`Mutibs`."""

    @property
    def u(self) -> int:
        """Interpret the viewed bits as an unsigned integer."""

    @u.setter
    def u(self, value: int) -> None: ...

    @property
    def i(self) -> int:
        """Interpret the viewed bits as a signed integer."""

    @i.setter
    def i(self, value: int) -> None: ...

    @property
    def f(self) -> float:
        """Interpret the viewed bits as an IEEE floating point value."""

    @f.setter
    def f(self, value: float) -> None: ...

    @property
    def bin(self) -> str:
        """Return the viewed bits as a binary string."""

    @bin.setter
    def bin(self, value: str) -> None: ...

    @property
    def oct(self) -> str:
        """Return the viewed bits as an octal string."""

    @oct.setter
    def oct(self, value: str) -> None: ...

    @property
    def hex(self) -> str:
        """Return the viewed bits as a hexadecimal string."""

    @hex.setter
    def hex(self, value: str) -> None: ...

    @property
    def bytes(self) -> bytes:
        """Return the viewed bits as bytes."""

    @bytes.setter
    def bytes(self, value: bytes | bytearray | memoryview) -> None: ...

    def field(self, a: int, b: int, /) -> MutableView:
        """Extract a field using inclusive bit labels.

        ``a`` and ``b`` must be zero or positive bit labels. The two endpoints
        are inclusive and may be provided in either order. The returned
        ``MutableView`` is a live view onto the selected source bits.
        """

    def __len__(self) -> int: ...

    def __eq__(self, other: Any) -> bool: ...

    def __repr__(self) -> str: ...

    def __format__(self, format_spec: str, /) -> str:
        """Return a string formatted according to the Python format mini-language.

        The viewed bits are formatted, so the byte order and bit order of the view are
        applied first. See :meth:`Tibs.__format__` for the accepted format specs.

        An empty format spec gives the ``repr``, as :func:`str` of a view does.

        :param str format_spec: The format specification.
        :return: The formatted string.

        :raises ValueError: if the spec cannot be parsed, if a type code is used that needs a length that is a different multiple, or if a sign or comma grouping is used with a bit representation type code.

        .. code-block:: pycon

            >>> f"{Mutibs('0x0100').le:#x}"
            '0x0001'
        """


class Reader:
    """A cursor for reading a :class:`Tibs` or :class:`Mutibs` in sequence.

    A ``Reader`` pairs a bit container with a bit position, so that values
    can be read one after another without working out ``start`` and ``end``
    for each one. Every method is anchored at :attr:`~Reader.pos`, and the
    reading and seeking methods move it. For a windowed query, use the
    source object directly through :attr:`~Reader.source`.

    The source is not copied, so a ``Mutibs`` that grows after the reader
    was built can be read up to its new length.

    .. code-block:: pycon

        >>> r = Reader(Tibs('0x47ff10'))
        >>> r.read_value('u8')
        71
        >>> r.read_value('(bool, u7)')
        (True, 127)
        >>> r.remaining
        8
    """

    def __init__(self, source: _TibsLike, /, pos: int = 0) -> None: ...

    @property
    def source(self) -> Tibs | Mutibs:
        """The object being read.

        This is the :class:`Tibs` or :class:`Mutibs` that was given to the
        constructor, not a copy, so it can be used for the windowed queries a
        ``Reader`` deliberately does not have: ``r.source.to_value(dtype,
        start, end)`` reads without moving :attr:`~Reader.pos`.

        .. code-block:: pycon

            >>> t = Tibs('0xabcd')
            >>> Reader(t).source is t
            True
        """

    @property
    def pos(self) -> int:
        """The current bit position.

        Setting it moves the cursor anywhere in ``0`` to ``len(self.source)``
        inclusive. Negative positions are not accepted: unlike a ``start`` or
        ``end`` elsewhere in tibs, this is stored state rather than a one-off
        slice bound, so counting from the end would leave
        :attr:`~Reader.remaining` disagreeing with the value just assigned.

        :raises ValueError: if set outside ``0`` to ``len(self.source)``.
        """

    @pos.setter
    def pos(self, value: int) -> None: ...

    @property
    def byte_pos(self) -> int:
        """The current position in bytes.

        :raises ValueError: when read, if :attr:`~Reader.pos` is not a multiple of 8; when set, if the position is out of range.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0x0123'))
            >>> r.byte_pos = 1
            >>> r.pos
            8
        """

    @byte_pos.setter
    def byte_pos(self, value: int) -> None: ...

    @property
    def remaining(self) -> int:
        """The number of bits between :attr:`~Reader.pos` and the end."""

    @property
    def at_end(self) -> bool:
        """Whether there is nothing left to read."""

    def read_value(self, dtype: _DtypeLike, /) -> _DtypeValue:
        """Read one value with a dtype, advancing by the dtype length.

        The value is a scalar for a :class:`DtypeSingle` and a tuple for a
        :class:`DtypeArray` or :class:`DtypeTuple`, exactly as
        :meth:`Tibs.to_value` gives.

        :param Dtype | str dtype: The value encoding to use.
        :return: The decoded Python value.
        :raises tibs.ReadError: if fewer than ``dtype.length`` bits remain, in which case the position does not move.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0x47ff'))
            >>> r.read_value('u8')
            71
            >>> r.read_value('(bool, u7)')
            (True, 127)
        """

    def read_values(self, dtype: _DtypeLike, /, count: int | None = None) -> list[_DtypeValue]:
        """Read a list of values of one dtype, advancing past all of them.

        With ``count`` given, exactly that many values are read and it is an
        error if they do not all fit. With ``count`` left as ``None``, as many
        whole values as fit are read; any partial value at the end is left
        under the cursor rather than being an error, so
        :attr:`~Reader.remaining` afterwards says how many bits were not
        consumed. This is the one place a ``Reader`` is more forgiving than
        :meth:`Tibs.to_values`, which has no cursor to leave a remainder
        behind.

        :param Dtype | str dtype: The value encoding to use for each item.
        :param int | None count: The number of values to read, or ``None`` for as many as fit.
        :return: A list of decoded Python values.
        :raises ValueError: if ``count`` is negative or too large to represent.
        :raises tibs.ReadError: if the requested values do not fit in the bits that remain, in which case the position does not move.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0x0102030405'))
            >>> r.read_values('u8', 2)
            [1, 2]
            >>> r.read_values('u8')
            [3, 4, 5]
        """

    def read_bits(self, n: int, /) -> Tibs:
        """Read the next ``n`` bits, advancing by ``n``.

        The result is always a :class:`Tibs`, including when reading from a
        :class:`Mutibs`: a read takes a value out of the stream, and there is
        nothing to write back through.

        :param int n: The number of bits to read.
        :return: The bits read.
        :raises ValueError: if ``n`` is negative.
        :raises tibs.ReadError: if ``n`` is more than :attr:`~Reader.remaining`, in which case the position does not move.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0xf00f'))
            >>> r.read_bits(4)
            Tibs('0xf')
        """

    def read_to(self, needle: _TibsLike, /, byte_aligned: bool = False,
                mask: _TibsLike | None = None) -> Tibs:
        """Read up to the next occurrence of ``needle``, leaving the cursor on it.

        The returned bits stop where ``needle`` begins, and the cursor is left
        there too, so the needle itself is read by whatever comes next. Use
        :meth:`~read_past` to consume it as part of the read.

        :param object needle: The bit sequence to read up to. This can be anything promotable to ``Tibs``.
        :param bool byte_aligned: If ``True``, only match on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: The bits between the current position and the match.
        :raises ValueError: if ``needle`` is empty.
        :raises tibs.ReadError: if ``needle`` is not found at or after the current position, in which case the position does not move.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0x0000ff12'))
            >>> r.read_to('0xff')
            Tibs('0x0000')
            >>> r.pos
            16
        """

    def read_past(self, needle: _TibsLike, /, byte_aligned: bool = False,
                  mask: _TibsLike | None = None) -> Tibs:
        """Read up to and including the next occurrence of ``needle``.

        The needle is part of the returned bits and the cursor is left just
        after it, so a loop of ``read_past`` calls always makes progress.

        :param object needle: The bit sequence to read past. This can be anything promotable to ``Tibs``.
        :param bool byte_aligned: If ``True``, only match on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: The bits between the current position and the end of the match.
        :raises ValueError: if ``needle`` is empty.
        :raises tibs.ReadError: if ``needle`` is not found at or after the current position, in which case the position does not move.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0x0000ff12'))
            >>> r.read_past('0xff')
            Tibs('0x0000ff')
            >>> r.pos
            24
        """

    def peek_value(self, dtype: _DtypeLike, /) -> _DtypeValue:
        """Read one value with a dtype without moving the cursor.

        :param Dtype | str dtype: The value encoding to use.
        :return: The decoded Python value.
        :raises tibs.ReadError: if fewer than ``dtype.length`` bits remain.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0x47ff'))
            >>> r.peek_value('u8')
            71
            >>> r.pos
            0
        """

    def peek_bits(self, n: int, /) -> Tibs:
        """Read the next ``n`` bits without moving the cursor.

        :param int n: The number of bits to read.
        :return: The bits read.
        :raises ValueError: if ``n`` is negative.
        :raises tibs.ReadError: if ``n`` is more than :attr:`~Reader.remaining`.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0xf00f'))
            >>> r.peek_bits(4)
            Tibs('0xf')
            >>> r.pos
            0
        """

    def bookmark(self) -> Bookmark:
        """Return a context manager that restores the position on exit.

        The ``with`` block gets this same ``Reader``, so anything can be read
        inside it, and the position it had on entry is put back afterwards
        whether the block finished or raised. This is how a ``Reader`` looks
        ahead by more than one value; :meth:`~peek_value` and
        :meth:`~peek_bits` are the single-item shorthands.

        :return: A :class:`Bookmark` yielding this ``Reader``.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0x010203'))
            >>> with r.bookmark():
            ...     r.read_values('u8')
            [1, 2, 3]
            >>> r.pos
            0
        """

    def align(self, boundary: int = 8, /) -> int:
        """Move forward to the next multiple of ``boundary`` bits.

        Nothing moves if the position is already on a boundary. This covers
        byte alignment as ``align()`` and generalises to 16-bit or 32-bit
        boundaries for free.

        :param int boundary: The alignment in bits. Defaults to 8.
        :return: The number of bits skipped.
        :raises ValueError: if ``boundary`` is not positive.
        :raises tibs.ReadError: if aligning would move past the end, in which case the position does not move.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0xffff'), 3)
            >>> r.align()
            5
            >>> r.pos
            8
        """

    def seek_to(self, needle: _TibsLike, /, byte_aligned: bool = False,
                mask: _TibsLike | None = None) -> bool:
        """Move to the next occurrence of ``needle``, leaving the cursor on it.

        The search starts at the current position, so a needle already under
        the cursor is found where it is and nothing moves. That makes ``while
        r.seek_to(x)`` an infinite loop; :meth:`~seek_past` is the one to loop
        on.

        The position of the match is :attr:`~Reader.pos` afterwards. This
        returns a ``bool`` rather than the position so that a match at bit 0
        is not falsy.

        :param object needle: The bit sequence to seek to. This can be anything promotable to ``Tibs``.
        :param bool byte_aligned: If ``True``, only match on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: ``True`` if it was found, otherwise ``False`` with the position unchanged.
        :raises ValueError: if ``needle`` is empty.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0x0000ff12'))
            >>> r.seek_to('0xff')
            True
            >>> r.pos
            16
        """

    def seek_past(self, needle: _TibsLike, /, byte_aligned: bool = False,
                  mask: _TibsLike | None = None) -> bool:
        """Move to just after the next occurrence of ``needle``.

        The cursor always ends up further forward than it started, so this is
        the one to drive a scanning loop with::

            while reader.seek_past(marker):
                handle(reader.read_value('u16'))

        :param object needle: The bit sequence to seek past. This can be anything promotable to ``Tibs``.
        :param bool byte_aligned: If ``True``, only match on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: ``True`` if it was found, otherwise ``False`` with the position unchanged.
        :raises ValueError: if ``needle`` is empty.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0x0000ff12'))
            >>> r.seek_past('0xff')
            True
            >>> r.pos
            24
        """

    def seek_back_to(self, needle: _TibsLike, /, byte_aligned: bool = False,
                     mask: _TibsLike | None = None) -> bool:
        """Move back to the previous occurrence of ``needle``.

        Only matches that end at or before the current position are
        considered, so the cursor always ends up further back than it started
        and ``while r.seek_back_to(x)`` makes progress.

        :param object needle: The bit sequence to seek back to. This can be anything promotable to ``Tibs``.
        :param bool byte_aligned: If ``True``, only match on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: ``True`` if it was found, otherwise ``False`` with the position unchanged.
        :raises ValueError: if ``needle`` is empty.

        .. code-block:: pycon

            >>> r = Reader(Tibs('0x00ff00ff'), 32)
            >>> r.seek_back_to('0xff')
            True
            >>> r.pos
            24
        """

    def __repr__(self) -> str: ...

    def __copy__(self) -> Reader:
        """Return a second cursor at the same position over the same source.

        The source is shared rather than copied, exactly as it is when a
        ``Reader`` is constructed, so a copy of a reader over a ``Mutibs``
        still reads that ``Mutibs`` live. Use :func:`copy.deepcopy` for a
        reader over its own copy of the bits.

        .. code-block:: pycon

            >>> import copy
            >>> r = Reader(Tibs('0x010203'))
            >>> r.read_value('u8')
            1
            >>> ahead = copy.copy(r)
            >>> ahead.read_values('u8')
            [2, 3]
            >>> r.pos
            8
        """

    def __deepcopy__(self, memo: dict[int, Any], /) -> Reader:
        """Return a cursor at the same position over a deep copy of the source.

        Unlike :meth:`~Reader.__copy__`, the two readers no longer share bits:
        a ``Mutibs`` source is copied along with the cursor.

        :param dict memo: The memo dictionary :mod:`copy` passes through.
        """


class Bookmark(AbstractContextManager["Reader"]):
    """The context manager returned by :meth:`Reader.bookmark`.

    Entering gives back the ``Reader`` it came from and saves where its
    cursor is; leaving puts the cursor back there, whether the block
    finished or raised.

    The position is taken on the way in rather than when the bookmark was
    made, so one that is kept and used more than once saves the position
    each block actually starts from. Cannot be constructed directly.

    .. code-block:: pycon

        >>> r = Reader(Tibs('0x010203'))
        >>> saved = r.bookmark()
        >>> r.read_value('u8')
        1
        >>> with saved:
        ...     r.read_value('u8')
        2
        >>> r.pos
        8
    """

    def __enter__(self) -> Reader:
        """Save the reader's position and return the reader.

        :return: The :class:`Reader` this bookmark came from.
        :raises ValueError: if this bookmark is already inside a ``with`` block.
        """

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Restore the saved position and let any exception propagate."""

    def __repr__(self) -> str: ...


class Tibs:
    """An immutable container of binary data.

    The constructor is a convenient way to delegate to the ``from_string`` or
    ``from_bytes`` builder methods, depending on the type of ``auto``. A list
    or tuple is taken as a bit pattern directly, and only ``True``, ``False``,
    ``0`` and ``1`` are accepted in one.

    * ``Tibs('0x13')`` - Equivalent to ``Tibs.from_string('0x13')``.
    * ``Tibs(b'hello')`` - Equivalent to ``Tibs.from_bytes(b'hello')``.
    * ``Tibs([1, 0])`` - A bit pattern given as a list or tuple.

    Otherwise, to construct use a builder 'from' method:

    * ``Tibs.from_bin(s)`` - Create from a binary string, optionally starting with '0b'.
    * ``Tibs.from_oct(s)`` - Create from an octal string, optionally starting with '0o'.
    * ``Tibs.from_hex(s)`` - Create from a hex string, optionally starting with '0x'.
    * ``Tibs.from_u(u, length, [byte_order])`` - Create from an unsigned int to a given length.
    * ``Tibs.from_i(i, length, [byte_order])`` - Create from a signed int to a given length.
    * ``Tibs.from_f(f, length, [byte_order])`` - Create from an IEEE float to a 16, 32 or 64 bit length.
    * ``Tibs.from_bytes(b)`` - Create directly from a ``bytes``, ``bytearray`` or ``memoryview`` object.
    * ``Tibs.from_string(s)`` - Use a formatted string.
    * ``Tibs.from_bools(iterable)`` - Convert each element in ``iterable`` to a bool.
    * ``Tibs.from_zeros(length)`` - Initialise with ``length`` ``0`` bits.
    * ``Tibs.from_ones(length)`` - Initialise with ``length`` ``1`` bits.
    * ``Tibs.from_random(length, [secure, seed])`` - Initialise with ``length`` randomly set bits.
    * ``Tibs.from_joined(iterable)`` - Concatenate an iterable of objects.
    * ``Tibs.from_value(dtype, value)`` / ``Tibs.from_values(dtype, values)`` - Pack typed values.
    """

    def __init__(self, auto: _TibsLike | None = None, /) -> None: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __format__(self, format_spec: str, /) -> str:
        """Return a string formatted according to the Python format mini-language.

        The type codes ``b``, ``o``, ``x`` and ``X`` give the bit representation, and so
        keep any leading zeros. They are equivalent to the :attr:`~Tibs.bin`,
        :attr:`~Tibs.oct` and :attr:`~Tibs.hex` properties. The type codes ``u`` and ``i``
        give the unsigned and signed integer interpretations, and ``e``, ``f`` and ``g``
        (with their uppercase forms) show the IEEE float value using Python's scientific,
        fixed-point and general presentations; a float needs a length of 16, 32 or 64
        bits. All of these read the bits big-endian, the same as the matching properties.
        To interpret them in another byte or bit order, format a view such as ``self.le``
        instead.

        The ``#`` flag adds a ``0b``, ``0o``, ``0x`` or ``0X`` prefix. The ``_`` option
        groups the digits, with the group size taken from the otherwise unused precision
        field and defaulting to 4. Groups are counted from bit zero, so a short group
        comes last. Fill, alignment and width work as they do elsewhere in Python.

        An empty format spec gives the same string as :func:`str`.

        :param str format_spec: The format specification.
        :return: The formatted string.

        :raises ValueError: if the spec cannot be parsed, if a type code is used that needs a length that is a different multiple, or if a sign or comma grouping is used with a bit representation type code.

        .. code-block:: pycon

            >>> f"{Tibs('0xac804f4b'):#_.2x}"
            '0xac_80_4f_4b'
            >>> f"{Tibs('0x0f'):b}"
            '00001111'
        """

    def view(
            self,
            byte_order: ByteOrder = ByteOrder.Unspecified,
            bit_order: BitOrder = BitOrder.Msb0
    ) -> View:
        """Return a view with interpretation settings.

        A view does not change the underlying bits. It changes how operations such
        as integer conversion, byte conversion and field extraction interpret those
        bits.

        Byte-oriented views must have a whole-byte length. This applies when using
        little-endian or big-endian byte order, or when using ``BitOrder.Lsb0``.

        :param ByteOrder byte_order: The byte order used when interpreting whole-byte values. Defaults to ``ByteOrder.Unspecified``.
        :param BitOrder bit_order: The bit numbering order used for field labels. Defaults to ``BitOrder.Msb0``.
        :return: A new :class:`View`.

        .. code-block:: pycon

            >>> Tibs('0x0100').view(byte_order=ByteOrder.Little).u
            1
        """

    @property
    def le(self) -> View:
        """Return a little-endian byte-order view.

        Equivalent to ``view(byte_order=ByteOrder.Little)``.

        The ``Tibs`` length must be a whole number of bytes.
        """

    @property
    def be(self) -> View:
        """Return a big-endian byte-order view.

        Equivalent to ``view(byte_order=ByteOrder.Big)``.

        The ``Tibs`` length must be a whole number of bytes.
        """

    @property
    def lsb0(self) -> View:
        """Return an LSB0 bit-order view.

        ``BitOrder.Lsb0`` means that field labels are counted from the least
        significant bit of each byte. The ``Tibs`` length must be a whole number of
        bytes.

        Equivalent to ``view(bit_order=BitOrder.Lsb0)``.
        """

    @property
    def msb0(self) -> View:
        """Return an MSB0 bit-order view.

        ``BitOrder.Msb0`` means that field labels are counted from the most
        significant bit of each byte. This is the default bit order.

        Equivalent to ``view(bit_order=BitOrder.Msb0)``.
        """

    def field(self, a: int, b: int, /) -> View:
        """Extract a field using inclusive MSB0 bit labels.

        ``a`` and ``b`` must be zero or positive bit labels. The two endpoints
        are inclusive and may be provided in either order. This is equivalent to
        ``self.msb0.field(a, b)``.

        :param int a: One non-negative inclusive field endpoint.
        :param int b: The other non-negative inclusive field endpoint.
        :return: A new :class:`View`.
        """

    def __iter__(self) -> Iterator[bool]: ...

    def chunks(self, chunk_size: int, /, count: int | None = None) -> list[Tibs]:
        """Return a list of Tibs by cutting into chunks.

        :param int chunk_size: The size in bits of the chunks to create.
        :param int | None count: If specified, at most count items are created. Default is to cut as many times as possible.
        :return: A list of Tibs chunks.

        .. code-block:: pycon

            >>> Tibs('0b110011').chunks(2)
            [Tibs('0b11'), Tibs('0b00'), Tibs('0b11')]
        """

    @overload
    def split_at(self, pos: SupportsIndex, /) -> tuple[Tibs, Tibs]:
        """Split at one or more bit positions.

        ``pos`` may be a single integer or an iterable of integers. Negative
        positions count from the end. Positions must be in nondecreasing order
        after normalization, and each position must be in the range
        ``0`` through ``len(self)``, inclusive.

        The returned pieces are normal ``Tibs`` slices. They share storage with
        the original ``Tibs`` when possible.

        :param int | Iterable[int] pos: The bit position or positions where the split should occur.
        :return: A tuple of ``Tibs`` pieces.

        .. code-block:: pycon

            >>> Tibs('0b101100').split_at(3)
            (Tibs('0b101'), Tibs('0b100'))
            >>> Tibs('0b101100').split_at([2, 5])
            (Tibs('0b10'), Tibs('0b110'), Tibs('0b0'))
        """

    @overload
    def split_at(self, pos: Iterable[SupportsIndex], /) -> tuple[Tibs, ...]:
        """Split at one or more bit positions.

        ``pos`` may be a single integer or an iterable of integers. Negative
        positions count from the end. Positions must be in nondecreasing order
        after normalization, and each position must be in the range
        ``0`` through ``len(self)``, inclusive.

        The returned pieces are normal ``Tibs`` slices. They share storage with
        the original ``Tibs`` when possible.

        :param int | Iterable[int] pos: The bit position or positions where the split should occur.
        :return: A tuple of ``Tibs`` pieces.

        .. code-block:: pycon

            >>> Tibs('0b101100').split_at(3)
            (Tibs('0b101'), Tibs('0b100'))
            >>> Tibs('0b101100').split_at([2, 5])
            (Tibs('0b10'), Tibs('0b110'), Tibs('0b0'))
        """

    def chunks_iter(self, chunk_size: int, /, count: int | None = None) -> Iterator[Tibs]:
        """Return an iterator by cutting into Tibs chunks.

        :param int chunk_size: The size in bits of the chunks to generate.
        :param int | None count: If specified, at most count items are generated. Default is to cut as many times as possible.
        :return: A generator yielding Tibs chunks.

        .. code-block:: pycon

            >>> list(Tibs('0b110011').chunks_iter(2))
            [Tibs('0b11'), Tibs('0b00'), Tibs('0b11')]
        """

    def rchunks_iter(self, chunk_size: int, /, count: int | None = None) -> Iterator[Tibs]:
        """Return a reverse iterator by cutting into Tibs chunks, starting from the end.

        :param int chunk_size: The size in bits of the chunks to generate.
        :param int | None count: If specified, at most count items are generated. Default is to cut as many times as possible.
        :return: A generator yielding Tibs chunks.

        .. code-block:: pycon

            >>> list(Tibs('0b1100111').rchunks_iter(3))
            [Tibs('0b111'), Tibs('0b100'), Tibs('0b1')]
        """

    def __eq__(self, other: Any) -> bool: ...

    def __hash__(self) -> int: ...

    def find_all(self, needle: _TibsLike, /, start: int | None = None, end: int | None = None,
                 byte_aligned: bool = False,
                 mask: _TibsLike | None = None) -> list[int]:
        """Find all occurrences of a bit sequence.

        :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
        :param int | None start: The starting bit position of the slice to search. Defaults to 0.
        :param int | None end: The end bit position of the slice to search. Defaults to len(self).
        :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries. Defaults to ``False``.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: A list of bit positions.

        :raises ValueError: if needle is empty, if start or end are out of range, if end is before start
            or if the mask length doesn't match the needle length.

        All occurrences of needle are found, even if they overlap.

        .. code-block:: pycon

            >>> Tibs('0b10111011').find_all('0b11')
            [2, 3, 6]
            >>> Tibs('0x1f2f3a').find_all('0x0f', mask='0x0f', byte_aligned=True)
            [0, 8]
        """

    def find_all_iter(self, needle: _TibsLike, /, start: int | None = None, end: int | None = None,
                      byte_aligned: bool = False, mask: _TibsLike | None = None) -> Iterator[int]:
        """Find all occurrences of a bit sequence, returning an iterator of bit positions.

        :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
        :param int | None start: The starting bit position of the slice to search. Defaults to 0.
        :param int | None end: The end bit position of the slice to search. Defaults to len(self).
        :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries. Defaults to ``False``.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: A generator yielding bit positions.

        :raises ValueError: if needle is empty, if start or end are out of range, if end is before start
            or if the mask length doesn't match the needle length.

        All occurrences of needle are found, even if they overlap.

        Note that this method is not available for :class:`Mutibs` as its value could change while the
        generator is still active. For that case, convert to a :class:`Tibs` first with
        :meth:`Mutibs.to_tibs`, or use :meth:`Mutibs.take_tibs` if you no longer need the mutable object.

        .. code-block:: pycon

            >>> list(Tibs('0b10111011').find_all_iter('0b11'))
            [2, 3, 6]
        """

    def rfind_all_iter(self, needle: _TibsLike, /, start: int | None = None, end: int | None = None,
                       byte_aligned: bool = False, mask: _TibsLike | None = None) -> Iterator[int]:
        """Find all occurrences of a bit sequence in reverse, returning an iterator of bit positions.

        :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
        :param int | None start: The starting bit position of the slice to search. Defaults to 0.
        :param int | None end: The end bit position of the slice to search. Defaults to len(self).
        :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries. Defaults to ``False``.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: A generator yielding bit positions.

        :raises ValueError: if needle is empty, if start or end are out of range, if end is before start
            or if the mask length doesn't match the needle length.

        All occurrences of needle are found, even if they overlap.

        Note that this method is not available for :class:`Mutibs` as its value could change while the
        generator is still active. For that case, convert to a :class:`Tibs` first with
        :meth:`Mutibs.to_tibs`, or use :meth:`Mutibs.take_tibs` if you no longer need the mutable object.

        .. code-block:: pycon

            >>> list(Tibs('0b10111011').rfind_all_iter('0b11'))
            [6, 3, 2]
        """

    def __len__(self) -> int: ...

    @classmethod
    def from_zeros(cls, length: int, /) -> Tibs:
        """Create a new instance with all bits set to '0'.

        :param int length: The number of bits to set.
        :return: A Tibs object with all bits set to zero.

        .. code-block:: python

            a = Tibs.from_zeros(500)  # 500 zero bits
        """

    @classmethod
    def from_ones(cls, length: int, /) -> Tibs:
        """Create a new instance with all bits set to '1'.

        :param int length: The number of bits to set.
        :return: A Tibs object with all bits set to one.

        .. code-block:: pycon

            >>> Tibs.from_ones(5)
            Tibs('0b11111')
        """

    @classmethod
    def from_string(cls, s: str, /) -> Tibs:
        """Create a new instance from a formatted string.

        :param str s: The formatted string to convert. This can begin with '0b', '0o' or '0x' to indicate binary, octal or hexadecimal, and commas can be used to separate items.
        :return: A newly constructed ``Tibs``.

        .. code-block:: python

            a = Tibs.from_string("0xff01")
            b = Tibs.from_string("0o775, 0b1")

        The ``__init__`` method can also redirect to ``from_string``:

        .. code-block:: python

            a = Tibs("0xff01")
        """

    @classmethod
    def from_value(cls, dtype: _DtypeLike, value: Any, /) -> Tibs:
        """Create a new instance by encoding one Python value with a dtype.

        :param Dtype | str dtype: The value encoding to use.
        :param object value: The value to encode.
        :return: A newly constructed ``Tibs``.

        .. code-block:: pycon

            >>> Tibs.from_value("u8", 15)
            Tibs('0x0f')
        """

    @classmethod
    def from_values(cls, dtype: _DtypeLike, iterable: Iterable[Any], /) -> Tibs:
        """Create a new instance by encoding and concatenating values with a dtype.

        :param Dtype | str dtype: The value encoding to use for each item.
        :param Iterable iterable: The values to encode.
        :return: A newly constructed ``Tibs``.

        .. code-block:: pycon

            >>> Tibs.from_values("u8", [1, 2, 3])
            Tibs('0x010203')
        """

    def to_value(self, dtype: _DtypeLike, /, start: int | None = None, end: int | None = None) -> _DtypeValue:
        """Return one value decoded with a dtype.

        The selected range must have exactly the dtype length.

        :param Dtype | str dtype: The value encoding to use.
        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).
        :return: The decoded Python value.

        .. code-block:: pycon

            >>> Tibs('0x0f').to_value("u8")
            15
        """

    def to_values_iter(self, dtype: _DtypeLike, /, start: int | None = None, end: int | None = None) -> Iterator[
        _DtypeValue]:
        """Return an iterator over values decoded with a dtype.

        The selected range must be a whole number of dtype values.

        :param Dtype | str dtype: The value encoding to use for each yielded item.
        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).
        :return: An iterator yielding decoded Python values.

        .. code-block:: pycon

            >>> list(Tibs('0x010203').to_values_iter("u8"))
            [1, 2, 3]
        """

    def to_values(self, dtype: _DtypeLike, /, start: int | None = None, end: int | None = None) -> list[
        _DtypeValue]:
        """Return a list of values decoded with a dtype.

        The selected range must be a whole number of dtype values.

        :param Dtype | str dtype: The value encoding to use for each item.
        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).
        :return: A list of decoded Python values.

        .. code-block:: pycon

            >>> Tibs('0x010203').to_values("u8")
            [1, 2, 3]
        """

    @classmethod
    def from_u(cls, u: int, /, length: int, byte_order: ByteOrder = ByteOrder.Unspecified) -> Tibs:
        """Create a new instance from an unsigned integer.

        :param int u: An unsigned integer.
        :param int length: The bit length to create. Can be any positive number of bits.
        :param ByteOrder byte_order: The byte order used to store the integer. Defaults to ByteOrder.Unspecified.
        :return: A newly constructed ``Tibs``.

        :raises ValueError: if the integer doesn't fit in the length given.

        .. code-block:: pycon

            >>> Tibs.from_u(15, length=8)
            Tibs('0x0f')
        """

    def to_u(self, start: int | None = None, end: int | None = None) -> int:
        """Return the unsigned integer representation of the Tibs.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The value as an unsigned integer.

        .. code-block:: pycon

            >>> Tibs('0x0f').to_u()
            15
        """

    @property
    def u(self) -> int:
        """Read-only property of the unsigned integer representation of the Tibs.

        Equivalent to using :meth:`~to_u` with no parameters.

        :return: The value as an unsigned integer.
        """

    @classmethod
    def from_i(cls, i: int, /, length: int, byte_order: ByteOrder = ByteOrder.Unspecified) -> Tibs:
        """Create a new instance from a signed integer.

        :param int i: A signed integer.
        :param int length: The bit length to create. Can be any positive number of bits.
        :param ByteOrder byte_order: The byte order used to store the integer. Defaults to ByteOrder.Unspecified.
        :return: A newly constructed ``Tibs``.

        :raises ValueError: if the integer doesn't fit in the length given.

        .. code-block:: pycon

            >>> Tibs.from_i(-2, length=4)
            Tibs('0xe')
        """

    def to_i(self, start: int | None = None, end: int | None = None) -> int:
        """Return the signed integer representation of the Tibs.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The value as a signed integer.

        .. code-block:: pycon

            >>> Tibs('0xe').to_i()
            -2
        """

    @property
    def i(self) -> int:
        """Read-only property of the signed integer representation of the Tibs.

        Equivalent to using :meth:`~to_i` with no parameters.

        :return: The value as a signed integer.
        """

    @classmethod
    def from_f(cls, f: float, /, length: int, byte_order: ByteOrder = ByteOrder.Unspecified) -> Tibs:
        """Create a new instance from a floating point number.

        :param float f: A floating point value.
        :param int length: The bit length to create. Must be 16, 32 or 64.
        :param ByteOrder byte_order: The byte order used to store the float. Defaults to ByteOrder.Unspecified.
        :return: A newly constructed ``Tibs``.

        .. code-block:: pycon

            >>> Tibs.from_f(1.5, length=32)
            Tibs('0x3fc00000')
        """

    def to_f(self, start: int | None = None, end: int | None = None) -> float:
        """Return the floating point representation of the Tibs.

        The length must be 16, 32 or 64.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The value as a Python float.

        .. code-block:: pycon

            >>> Tibs('0x3fc00000').to_f()
            1.5
        """

    @property
    def f(self) -> float:
        """Read-only property of the floating point representation of the Tibs.

        Equivalent to using :meth:`~to_f` with no parameters.

        :return: The value as a Python float.
        """

    @classmethod
    def from_bin(cls, s: str, /) -> Tibs:
        """Create a new instance from a binary string.

        :param str s: A string of ``0`` and ``1`` s, optionally preceded with ``0b`` and optionally containing underscores.
        :return: A newly constructed ``Tibs``.

        .. code-block:: python

            a = Tibs.from_bin("0000_1111_0101")
        """

    def to_bin(self, start: int | None = None, end: int | None = None) -> str:
        """Return the binary representation of the Tibs as a string.

        Equivalent to using the ``bin`` property when called with no parameters.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The binary representation.
        """

    @property
    def bin(self) -> str:
        """Read-only property of the binary representation of the Tibs.

        Equivalent to using :meth:`~to_bin` with no parameters.

        :return: The binary representation.
        """

    @classmethod
    def from_oct(cls, s: str, /) -> Tibs:
        """Create a new instance from an octal string.

        :param str s: A string of octal digits, optionally preceded with ``0o`` and optionally containing underscores.
        :return: A newly constructed ``Tibs``.

        .. code-block:: pycon

            >>> Tibs.from_oct("17")
            Tibs('0b001111')
        """

    def to_oct(self, start: int | None = None, end: int | None = None) -> str:
        """Return the octal representation of the Tibs as a string.

        Equivalent to using the ``oct`` property when called with no parameters.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The octal representation.
        :raises ValueError: if the length is not a multiple of 3.
        """

    @property
    def oct(self) -> str:
        """Read-only property of the octal representation of the Tibs.

        Equivalent to using :meth:`~to_oct` with no parameters.

        :return: The octal representation.
        :raises ValueError: if the length is not a multiple of 3.
        """

    @classmethod
    def from_hex(cls, s: str, /) -> Tibs:
        """Create a new instance from a hexadecimal string.

        :param str s: A string of hexadecimal digits, optionally preceded with ``0x`` and optionally containing underscores.
        :return: A newly constructed ``Tibs``.

        .. code-block:: pycon

            >>> Tibs.from_hex("0f")
            Tibs('0x0f')
        """

    def to_hex(self, start: int | None = None, end: int | None = None) -> str:
        """Return the hexadecimal representation of the Tibs as a string.

        Equivalent to using the ``hex`` property when called with no parameters.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The hexadecimal representation.
        :raises ValueError: if the length is not a multiple of 4.
        """

    @property
    def hex(self) -> str:
        """Read-only property of the hexadecimal representation of the Tibs.

        Equivalent to using :meth:`~to_hex` with no parameters.

        :return: The hexadecimal representation.
        :raises ValueError: if the length is not a multiple of 4.
        """

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview, /, bit_offset: int | None = None,
                   bit_length: int | None = None) -> Tibs:
        """Create a new instance from a bytes object.

        :param bytes | bytearray | memoryview data: The bytes, bytearray or memoryview object to convert to a :class:`Tibs`.
        :param int | None bit_offset: How many bits to skip from the start. Defaults to zero.
        :param int | None bit_length: How many bits to use. Defaults to the whole of the data.
        :return: A newly constructed ``Tibs``.

        Both are counts of bits, not bytes, so to skip a four byte header use
        ``bit_offset=32``.

        .. code-block:: python

            a = Tibs.from_bytes(b"some_bytes_maybe_from_a_file")
        """

    @classmethod
    def from_bools(cls, iterable: Iterable[Any], /) -> Tibs:
        """Create a new instance from an iterable by converting each element to a bool.

        :param Iterable iterable: The iterable to convert to a :class:`Tibs`.
        :return: A newly constructed ``Tibs``.

        .. code-block:: python

            a = Tibs.from_bools([False, 0, 1, "Steven"])  # binary 0011
        """

    def to_bools(self, start: int | None = None, end: int | None = None) -> list[bool]:
        """Return the bits as a list of bools.

        This is much faster than using ``list()`` on the Tibs, which iterates bit by bit.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).
        :return: A list of bools.

        .. code-block:: pycon

            >>> Tibs('0b101').to_bools()
            [True, False, True]
        """

    @classmethod
    def from_random(cls, length: int, /, secure: bool = False, seed: bytes | bytearray | None = None) -> Tibs:
        """Create a new instance with all bits randomly set.

        :param int length: The number of bits to set. Must be non-negative.
        :param bool secure: If ``True``, use the OS's cryptographically secure generator. Default is ``False``.
        :param bytes | bytearray | None seed: A bytes or bytearray to use as an optional seed, only if ``secure`` is ``False``.
        :return: A newly constructed ``Tibs`` with random data.

        The 'secure' option uses the OS's random data source, so will be slower and could potentially
        fail.

        .. code-block:: python

            a = Tibs.from_random(1000000)  # A million random bits
            b = Tibs.from_random(100, seed=b'a_seed')
        """

    @classmethod
    def from_joined(cls, iterable: Iterable[_TibsLike], /) -> Tibs:
        """Create a new instance by concatenating a sequence of Tibs objects.

        This method concatenates a sequence of Tibs objects into a single Tibs object.

        :param Iterable iterable: An iterable to concatenate. Items can be anything that can be promoted to a Tibs.
        :return: A newly constructed ``Tibs``.

        .. code-block:: python

            a = Tibs.from_joined(['0x01', [1, 0], b'some_bytes'])
        """

    @classmethod
    def decode(cls, b: bytes, /) -> Tibs:
        """Create a Tibs by decoding bytes created via Tibs.encode()

        :param bytes | bytearray b: The encoded bytes to decode.
        :return: A new Tibs.
        :raises tibs.DecodeError: for badly formed, truncated or extended input bytes.

        .. code-block:: pycon

            >>> Tibs.decode(Tibs('0b101').encode())
            Tibs('0b101')
        """

    def to_bytes(self, start: int | None = None, end: int | None = None) -> bytes:
        """Return the Tibs as a bytes object.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The bytes representation.
        :raises ValueError: if the length is not a multiple of 8.
        """

    def to_padded_bytes(self, start: int | None = None, end: int | None = None) -> bytes:
        """Return the Tibs as a bytes object, padding the right-hand side with zero bits.

        This appends 0 to 7 zero bits to the end of the selected bit sequence so
        the returned value has a whole number of bytes. If the selected length is
        already a multiple of 8, this is equivalent to :meth:`~to_bytes`.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The padded bytes representation.
        """

    @property
    def bytes(self) -> bytes:
        """Read-only property of the ``bytes`` representation of the Tibs.

        Equivalent to using :meth:`~to_bytes` with no parameters.

        :return: The bytes representation.
        :raises ValueError: if the length is not a multiple of 8.
        """

    def find(self, needle: _TibsLike, /, start: int | None = None, end: int | None = None,
             byte_aligned: bool = False, mask: _TibsLike | None = None) -> int | None:
        """Find first occurrence of a bit sequence.

        Returns the bit position if found, or None if not found.

        :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
        :param int | None start: The starting bit position. Defaults to 0.
        :param int | None end: The end position. Defaults to len(self).
        :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: The bit position if found, or None if not found.

        :raises ValueError: if ``needle`` is empty, if the slice parameters are invalid, or if the
            mask length doesn't match the needle length.

        The ``mask`` must be the same length as ``needle``. Only the bits set in it are compared, so
        the bits of ``needle`` under a zero mask bit are ignored and can be anything.

        .. code-block:: pycon

             >>> Tibs('0xc3e').find('0b1111')
             6
             >>> Tibs('0x3a5f').find('0x0f', mask='0x0f', byte_aligned=True)
             8
        """

    def __contains__(self, b: _TibsLike) -> bool: ...

    def __copy__(self) -> Tibs:
        """As Tibs is immutable, this returns the same instance."""

    def __reduce__(self) -> tuple[Callable[[bytes], Tibs], tuple[bytes]]:
        """Return the callable and arguments that recreate the Tibs.

        Used by :mod:`pickle` and by :func:`copy.deepcopy`.

        :return: A tuple of :meth:`Tibs.decode` and the encoded bytes to pass to it.

        .. code-block:: pycon

            >>> import pickle
            >>> pickle.loads(pickle.dumps(Tibs('0b110101')))
            Tibs('0b110101')
        """

    def rfind(self, needle: _TibsLike, /, start: int | None = None, end: int | None = None,
              byte_aligned: bool = False, mask: _TibsLike | None = None) -> int | None:
        """Find last occurrence of a bit sequence.

        Returns the bit position if found, or None if not found.

        :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
        :param int | None start: The starting bit position. Defaults to 0.
        :param int | None end: The end position. Defaults to len(self).
        :param bool byte_aligned: If ``True``, the Tibs will only be found on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: The bit position if found, or None if not found.

        :raises ValueError: if ``needle`` is empty, if the slice parameters are invalid, or if the
            mask length doesn't match the needle length.

        .. code-block:: pycon

             >>> Tibs('0b10111011').rfind('0b11')
             6
             >>> Tibs('0b10111011').rfind('0b00', mask='0b10')
             5
        """

    def starts_with(self, prefix: _TibsLike, /) -> bool:
        """Return whether the current Tibs starts with prefix.

        :param object prefix: The bits to search for. This can be anything promotable to ``Tibs``.
        :return: True if the Tibs starts with the prefix, otherwise False.

        .. code-block:: pycon

            >>> Tibs('0b101100').starts_with('0b101')
            True
            >>> Tibs('0b101100').starts_with('0b100')
            False
        """

    def ends_with(self, suffix: _TibsLike, /) -> bool:
        """Return whether the current Tibs ends with suffix.

        :param object suffix: The bits to search for. This can be anything promotable to ``Tibs``.
        :return: True if the Tibs ends with the suffix, otherwise False.

        .. code-block:: pycon

            >>> Tibs('0b101100').ends_with('0b100')
            True
            >>> Tibs('0b101100').ends_with('0b101')
            False
        """

    def count(self, value: Any = None, /, start: int | None = None, end: int | None = None,
              byte_aligned: bool = False, mask: _TibsLike | None = None) -> int:
        """Counts the total number of occurrences of a bit pattern.

        :param object | None value: Either something that can be converted to a ``Tibs``, or a single bit (one of ``0``, ``1``, ``False`` or ``True``). Defaults to counting the set bits.
        :param int | None start: The start of the slice to count within. Defaults to 0.
        :param int | None end: The end of the slice to count within. Defaults to len(self).
        :param bool byte_aligned: If ``True``, only occurrences on byte boundaries are counted. Defaults to ``False``.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.

        :return: The number of times the bit pattern is found.
        :raises ValueError: if the slice parameters are invalid, or if the mask length doesn't match
            the length of the value.

        With no ``value`` this counts the set bits, so ``count()`` is the same as ``count(1)``.

        When ``value`` is a multi-bit pattern, overlapping occurrences are all counted,
        just as they are by :meth:`find_all`. ``byte_aligned`` also applies to single-bit
        counts: ``count(1, byte_aligned=True)`` counts the set bits that land on a byte
        boundary.

        .. code-block:: pycon

            >>> Tibs('0xef').count()
            7
            >>> Tibs('0xef').count(1, 0, 4)
            3
            >>> Tibs.from_bin('0011010101100').count('0b01')
            4
            >>> Tibs('0b1111111').count('0b11')  # overlapping
            6
            >>> Tibs('0x80ff00').count(1, byte_aligned=True)
            2
        """

    def count_and(self, other: _TibsLike, /) -> int:
        """Count the bits set in both this Tibs and another.

        Equivalent to ``(self & other).count(1)``, but without building the
        intermediate ``Tibs``.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: The number of positions set in both.
        :raises ValueError: if the two Tibs have differing lengths.

        .. code-block:: pycon

            >>> Tibs('0b1100').count_and('0b1010')
            1
        """

    def count_or(self, other: _TibsLike, /) -> int:
        """Count the bits set in either this Tibs or another.

        Equivalent to ``(self | other).count(1)``, but without building the
        intermediate ``Tibs``.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: The number of positions set in either.
        :raises ValueError: if the two Tibs have differing lengths.

        .. code-block:: pycon

            >>> Tibs('0b1100').count_or('0b1010')
            3
        """

    def count_xor(self, other: _TibsLike, /) -> int:
        """Count the bits that differ between this Tibs and another.

        This is the Hamming distance. Equivalent to ``(self ^ other).count(1)``,
        but without building the intermediate ``Tibs``.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: The number of positions where the two differ.
        :raises ValueError: if the two Tibs have differing lengths.

        .. code-block:: pycon

            >>> Tibs('0b1100').count_xor('0b1010')
            2
        """

    def count_andnot(self, other: _TibsLike, /) -> int:
        """Count the bits set in this Tibs but not in another.

        Equivalent to ``self.count(1) - self.count_and(other)``, but in a single pass.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: The number of positions set here but not in the other.
        :raises ValueError: if the two Tibs have differing lengths.

        .. code-block:: pycon

            >>> Tibs('0b1100').count_andnot('0b1010')
            1
        """

    def intersects(self, other: _TibsLike, /) -> bool:
        """Return whether any bit is set in both this Tibs and another.

        Equivalent to ``(self & other).any()``, but stops at the first bit set in
        both instead of building the intermediate ``Tibs``.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: ``True`` if some position is set in both, otherwise ``False``.
        :raises ValueError: if the two Tibs have differing lengths.

        .. code-block:: pycon

            >>> Tibs('0b1100').intersects('0b1010')
            True
            >>> Tibs('0b1100').intersects('0b0011')
            False
        """

    def is_disjoint(self, other: _TibsLike, /) -> bool:
        """Return whether no bit is set in both this Tibs and another.

        The negation of :meth:`intersects`. Equivalent to ``not (self & other).any()``,
        but stops at the first bit set in both instead of building the
        intermediate ``Tibs``.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: ``True`` if no position is set in both, otherwise ``False``.
        :raises ValueError: if the two Tibs have differing lengths.

        .. code-block:: pycon

            >>> Tibs('0b1100').is_disjoint('0b0011')
            True
            >>> Tibs('0b1100').is_disjoint('0b1010')
            False
        """

    def is_subset_of(self, other: _TibsLike, /) -> bool:
        """Return whether every bit set in this Tibs is also set in another.

        Equivalent to ``(self & other) == self``, but stops at the first bit set
        here and not there.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: ``True`` if every position set here is set in the other, otherwise ``False``.
        :raises ValueError: if the two Tibs have differing lengths.

        .. code-block:: pycon

            >>> Tibs('0b1000').is_subset_of('0b1010')
            True
            >>> Tibs('0b1100').is_subset_of('0b1010')
            False
        """

    def is_superset_of(self, other: _TibsLike, /) -> bool:
        """Return whether every bit set in another is also set in this Tibs.

        The mirror of :meth:`is_subset_of`. Equivalent to ``(self & other) == other``,
        but stops at the first bit set there and not here.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: ``True`` if every position set in the other is set here, otherwise ``False``.
        :raises ValueError: if the two Tibs have differing lengths.

        .. code-block:: pycon

            >>> Tibs('0b1010').is_superset_of('0b1000')
            True
            >>> Tibs('0b1010').is_superset_of('0b1100')
            False
        """

    def extracted(self, mask: _TibsLike, /) -> Tibs:
        """Read the bits at the positions set in a mask, packed together.

        This reads a bit field whose bits are scattered through the Tibs by the
        mask, the way :meth:`field` reads a contiguous one. The result has one
        bit for each set bit of the mask, in order.

        :param object mask: The mask selecting which bits to read. This can be anything promotable to ``Tibs``, and must be the same length as ``self``.
        :return: A new Tibs of length ``mask.count()``.
        :raises ValueError: if the mask length doesn't match the length of ``self``.

        .. code-block:: pycon

            >>> Tibs('0b11010110').extracted('0b10110000')
            Tibs('0b101')
        """

    def deposited(self, value: _TibsLike, mask: _TibsLike, /) -> Tibs:
        """Return a new Tibs with a scattered bit field written into it.

        This is the immutable equivalent of :meth:`Mutibs.deposit`, and the
        inverse of :meth:`extracted`: the bits of ``value`` are written into the
        positions set in ``mask``, and the other bits are copied unchanged.

        :param object value: The bits to deposit. This can be anything promotable to ``Tibs``, and must be ``mask.count()`` bits long.
        :param object mask: The mask selecting which positions to write. This can be anything promotable to ``Tibs``, and must be the same length as ``self``.
        :return: A new Tibs.
        :raises ValueError: if the mask length doesn't match the length of ``self``, or ``value`` is not ``mask.count()`` bits long.

        .. code-block:: pycon

            >>> Tibs('0b11010110').deposited('0b111', '0b10110000').bin
            '11110110'
        """

    def all(self) -> bool:
        """Return True if all bits are equal to 1, otherwise return False.

        :return: ``True`` if all bits are 1, otherwise ``False``.

        .. code-block:: pycon

            >>> Tibs('0b1111').all()
            True
            >>> Tibs('0b1011').all()
            False
        """

    def any(self) -> bool:
        """Return True if any bits are equal to 1, otherwise return False.

        :return: ``True`` if any bits are 1, otherwise ``False``.

        .. code-block:: pycon

            >>> Tibs('0b0000').any()
            False
            >>> Tibs('0b1000').any()
            True
        """

    def with_set(self, pos: SupportsIndex | Iterable[SupportsIndex] | range, /) -> Tibs:
        """Return a new Tibs with one or many bits set to 1.

        This is the immutable equivalent of :meth:`Mutibs.set`.

        :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
        :return: A new Tibs.
        :raises IndexError: if pos < -len(self) or pos >= len(self).

        .. code-block:: pycon

            >>> Tibs.from_zeros(5).with_set([1, 3])
            Tibs('0b01010')
        """

    def with_unset(self, pos: SupportsIndex | Iterable[SupportsIndex] | range, /) -> Tibs:
        """Return a new Tibs with one or many bits set to 0.

        This is the immutable equivalent of :meth:`Mutibs.unset`.

        :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
        :return: A new Tibs.
        :raises IndexError: if pos < -len(self) or pos >= len(self).

        .. code-block:: pycon

            >>> Tibs.from_ones(5).with_unset([1, 3])
            Tibs('0b10101')
        """

    def inverted(self, pos: SupportsIndex | Iterable[SupportsIndex] | None = None, /) -> Tibs:
        """Return a new Tibs with selected bits inverted.

        This is the immutable equivalent of :meth:`Mutibs.invert`.

        :param int | Iterable[int] | None pos: Either a single bit position, an iterable of bit positions,
          or None to invert every bit. Defaults to None.
        :return: A new Tibs.
        :raises IndexError: if pos < -len(self) or pos >= len(self).

        .. code-block:: pycon

            >>> Tibs('0b10110').inverted([0, 2])
            Tibs('0b00010')
        """

    def inserted(self, pos: int, bs: _TibsLike, /) -> Tibs:
        """Insert bits at position pos and return a new Tibs.

        This is the immutable equivalent of :meth:`Mutibs.insert`.

        :param int pos: The bit position to insert at. Clips to the start or end if out of range.
        :param object bs: The bits to insert. This can be anything promotable to ``Tibs``.
        :return: A new Tibs.

        .. code-block:: pycon

            >>> Tibs('0b1011').inserted(2, '0b00')
            Tibs('0b100011')
        """

    def replaced(self, old: _TibsLike, new: _TibsLike, /, start: int | None = None, end: int | None = None,
                 count: int | None = None, byte_aligned: bool = False,
                 mask: _TibsLike | None = None) -> Tibs:
        """Search and replace and return a new Tibs.

        This is the immutable equivalent of :meth:`Mutibs.replace`.

        :param object old: The bits to search for. This can be anything promotable to ``Tibs``.
        :param object new: The bits to replace with. This can be anything promotable to ``Tibs``.
        :param int | None start: The starting bit position. Defaults to 0.
        :param int | None end: The end position. Defaults to len(self).
        :param int | None count: If present, the maximum number of replacements to make.
        :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: A new Tibs.
        :raises ValueError: if old is empty, count is negative, the slice parameters are invalid or
            the mask length doesn't match the length of old.

        The ``mask`` affects only which bits have to match; the whole of each match is still
        replaced by ``new``.

        Matches cannot overlap, since replacing one consumes the bits the next would have
        matched, so fewer replacements can be made than :meth:`count` finds occurrences.

        .. code-block:: pycon

            >>> Tibs('0b00010010').replaced([0, 1], [1, 1, 1])
            Tibs('0b0011101110')
            >>> Tibs('0x1f2e3f').replaced('0x0f', '0x00', mask='0x0f', byte_aligned=True)
            Tibs('0x002e00')
        """

    def rotated_left(self, n: int, /, start: int | None = None, end: int | None = None) -> Tibs:
        """Return a new Tibs with the bits rotated to the left.

        This is the immutable equivalent of :meth:`Mutibs.rotate_left`.

        :param int n: The number of bits to rotate by.
        :param int | None start: Start of slice to rotate. Defaults to 0.
        :param int | None end: End of slice to rotate. Defaults to len(self).
        :return: A new Tibs.
        :raises ValueError: if n < 0.

        .. code-block:: pycon

            >>> Tibs('0b10110').rotated_left(2)
            Tibs('0b11010')
        """

    def rotated_right(self, n: int, /, start: int | None = None, end: int | None = None) -> Tibs:
        """Return a new Tibs with the bits rotated to the right.

        This is the immutable equivalent of :meth:`Mutibs.rotate_right`.

        :param int n: The number of bits to rotate by.
        :param int | None start: Start of slice to rotate. Defaults to 0.
        :param int | None end: End of slice to rotate. Defaults to len(self).
        :return: A new Tibs.
        :raises ValueError: if n < 0.

        .. code-block:: pycon

            >>> Tibs('0b10110').rotated_right(1)
            Tibs('0b01011')
        """

    def reversed(self) -> Tibs:
        """Return a new instance with the bits reversed.

        :return: Tibs

        .. code-block:: pycon

            >>> a = Tibs('0b00011')
            >>> a.reversed()
            Tibs('0b11000')
        """

    def byte_swapped(self, byte_length: int | None = None, start: int | None = None,
                     end: int | None = None) -> Tibs:
        """Return a new instance with the byte order swapped.

        The selected slice will be byte-swapped. It must be a multiple of
        byte_length long.

        :param int | None byte_length: An int giving the number of bytes in each swap, or None (the default)
          to do a single reverse over the selected slice.
        :param int | None start: Start of slice to byte-swap. Defaults to 0.
        :param int | None end: End of slice to byte-swap. Defaults to len(self).
        :return: Tibs

        .. code-block:: pycon

            >>> a = Tibs('0x12345678')
            >>> b = a.byte_swapped(2)
            >>> b
            Tibs('0x34127856')
        """

    def to_mutibs(self) -> Mutibs:
        """Create and return a mutable copy of the Tibs as a Mutibs instance.

        :return: A new Mutibs with the same bit data.

        .. code-block:: pycon

            >>> t = Tibs.from_hex('abc')
            >>> m = t.to_mutibs()
            >>> m *= 4
            >>> print(t.hex)
            abc
            >>> print(m.hex)
            abcabcabcabc
        """

    def encode(self, codec: Codec = Codec.Auto) -> bytes:
        r"""Encode the tibs as a bytes instance.

        The bit length and the bit indexing are stored in the encoded bytes.

        The bytes instance can be used to recreate the Tibs exactly -
        see :meth:`Tibs.decode`.

        Use ``Codec.Raw`` when the encoded bytes themselves need to be a stable,
        canonical representation. The default ``Codec.Auto`` chooses a valid
        encoding for compactness and may produce different bytes for the same
        value in a future release.

        :param Codec codec: The codec to use. Defaults to Codec.Auto.
        :return: The encoded bytes.

        .. code-block:: pycon

            >>> t = Tibs('0b101')
            >>> b = t.encode()
            >>> b
            b'\x8d'
            >>> Tibs.decode(b)
            Tibs('0b101')
        """

    @overload
    def __getitem__(self, key: SupportsIndex) -> bool: ...

    @overload
    def __getitem__(self, key: slice) -> Tibs: ...

    def __getitem__(self, key: SupportsIndex | slice) -> bool | Tibs: ...

    def __lshift__(self, n: int) -> Tibs: ...

    def __rshift__(self, n: int) -> Tibs: ...

    def __add__(self, bs: _TibsLike) -> Tibs: ...

    def __radd__(self, bs: _TibsLike) -> Tibs: ...

    def __and__(self, bs: _TibsLike) -> Tibs: ...

    def __or__(self, bs: _TibsLike) -> Tibs: ...

    def __xor__(self, bs: _TibsLike) -> Tibs: ...

    def __rand__(self, bs: _TibsLike) -> Tibs: ...

    def __ror__(self, bs: _TibsLike) -> Tibs: ...

    def __rxor__(self, bs: _TibsLike) -> Tibs: ...

    def __invert__(self) -> Tibs: ...

    def __bytes__(self) -> bytes:
        """Return the Tibs as a bytes object.

        :return: The bytes representation.
        :raises ValueError: if the length is not a multiple of 8.
        """

    def __buffer__(self, flags: int, /) -> memoryview: ...

    def __mul__(self, n: int) -> Tibs: ...

    def __rmul__(self, n: int) -> Tibs: ...


class Mutibs:
    """A mutable container of binary data.

    To construct, use a builder 'from' method:

    * ``Mutibs.from_bin(s)`` - Create from a binary string, optionally starting with '0b'.
    * ``Mutibs.from_oct(s)`` - Create from an octal string, optionally starting with '0o'.
    * ``Mutibs.from_hex(s)`` - Create from a hex string, optionally starting with '0x'.
    * ``Mutibs.from_u(u, length, [byte_order])`` - Create from an unsigned int to a given length.
    * ``Mutibs.from_i(i, length, [byte_order])`` - Create from a signed int to a given length.
    * ``Mutibs.from_f(f, length, [byte_order])`` - Create from an IEEE float to a 16, 32 or 64 bit length.
    * ``Mutibs.from_bytes(b)`` - Create directly from a ``bytes``, ``bytearray`` or ``memoryview`` object.
    * ``Mutibs.from_string(s)`` - Use a formatted string.
    * ``Mutibs.from_bools(iterable)`` - Convert each element in ``iterable`` to a bool.
    * ``Mutibs.from_zeros(length)`` - Initialise with ``length`` ``0`` bits.
    * ``Mutibs.from_ones(length)`` - Initialise with ``length`` ``1`` bits.
    * ``Mutibs.from_random(length, [secure, seed])`` - Initialise with ``length`` randomly set bits.
    * ``Mutibs.from_joined(iterable)`` - Concatenate an iterable of objects.
    * ``Mutibs.from_value(dtype, value)`` / ``Mutibs.from_values(dtype, values)`` - Pack typed values.

    Using ``Mutibs(auto)`` delegates to ``from_string`` or ``from_bytes``, and takes a
    list or tuple as a bit pattern directly. Only ``True``, ``False``, ``0`` and ``1``
    are accepted in one; use ``from_bools`` to convert other truthy values.

    A ``Mutibs`` is not iterable, as the bits could change underneath an
    iterator. Convert first with ``to_tibs()`` for a copy, or ``take_tibs()``
    if the mutable object is no longer needed.
    """

    def __init__(self, auto: _TibsLike | None = None, /) -> None: ...

    def __eq__(self, other: Any) -> bool: ...

    # The typeshed idiom for an unhashable class; pyright reports it as an
    # incompatible override of object.__hash__ regardless.
    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]

    def __bool__(self) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __format__(self, format_spec: str, /) -> str:
        """Return a string formatted according to the Python format mini-language.

        The type codes ``b``, ``o``, ``x`` and ``X`` give the bit representation, and so
        keep any leading zeros. They are equivalent to the :attr:`~Mutibs.bin`,
        :attr:`~Mutibs.oct` and :attr:`~Mutibs.hex` properties. The type codes ``u`` and
        ``i`` give the unsigned and signed integer interpretations, and ``e``, ``f`` and
        ``g`` (with their uppercase forms) show the IEEE float value using Python's
        scientific, fixed-point and general presentations; a float needs a length of 16,
        32 or 64 bits. All of these read the bits big-endian, the same as the matching
        properties. To interpret them in another byte or bit order, format a view such as
        ``self.le`` instead.

        The ``#`` flag adds a ``0b``, ``0o``, ``0x`` or ``0X`` prefix. The ``_`` option
        groups the digits, with the group size taken from the otherwise unused precision
        field and defaulting to 4. Groups are counted from bit zero, so a short group
        comes last. Fill, alignment and width work as they do elsewhere in Python.

        An empty format spec gives the same string as :func:`str`.

        :param str format_spec: The format specification.
        :return: The formatted string.

        :raises ValueError: if the spec cannot be parsed, if a type code is used that needs a length that is a different multiple, or if a sign or comma grouping is used with a bit representation type code.

        .. code-block:: pycon

            >>> f"{Mutibs('0xac804f4b'):#_.2x}"
            '0xac_80_4f_4b'
            >>> f"{Mutibs('0x0f'):b}"
            '00001111'
        """

    def view(
            self,
            byte_order: ByteOrder = ByteOrder.Unspecified,
            bit_order: BitOrder = BitOrder.Msb0
    ) -> MutableView:
        """Return a mutable view with interpretation settings.

        A mutable view keeps a live reference to the source ``Mutibs``. Later
        changes to the ``Mutibs`` are reflected in the view, and assignment through
        the view mutates the source.

        Byte-oriented views must have a whole-byte length. This applies when using
        little-endian or big-endian byte order, or when using ``BitOrder.Lsb0``.

        :param ByteOrder byte_order: The byte order used when interpreting whole-byte values. Defaults to ``ByteOrder.Unspecified``.
        :param BitOrder bit_order: The bit numbering order used for field labels. Defaults to ``BitOrder.Msb0``.
        :return: A new :class:`MutableView`.

        .. code-block:: pycon

            >>> m = Mutibs('0x0100')
            >>> v = m.le
            >>> v.write_u(2)
            >>> m
            Mutibs('0x0200')
        """

    @property
    def le(self) -> MutableView:
        """Return a little-endian byte-order view.

        Equivalent to ``view(byte_order=ByteOrder.Little)``.

        The ``Mutibs`` length must be a whole number of bytes.
        """

    @property
    def be(self) -> MutableView:
        """Return a big-endian byte-order view.

        Equivalent to ``view(byte_order=ByteOrder.Big)``.

        The ``Mutibs`` length must be a whole number of bytes.
        """

    @property
    def lsb0(self) -> MutableView:
        """Return an LSB0 bit-order view.

        ``BitOrder.Lsb0`` means that field labels are counted from the least
        significant bit of each byte. The ``Mutibs`` length must be a whole number
        of bytes.

        Equivalent to ``view(bit_order=BitOrder.Lsb0)``.
        """

    @property
    def msb0(self) -> MutableView:
        """Return an MSB0 bit-order view.

        ``BitOrder.Msb0`` means that field labels are counted from the most
        significant bit of each byte. This is the default bit order.

        Equivalent to ``view(bit_order=BitOrder.Msb0)``.
        """

    def field(self, a: int, b: int, /) -> MutableView:
        """Extract a mutable field using inclusive MSB0 bit labels.

        ``a`` and ``b`` must be zero or positive bit labels. The two endpoints
        are inclusive and may be provided in either order. This is equivalent to
        ``self.msb0.field(a, b)``.

        :param int a: One non-negative inclusive field endpoint.
        :param int b: The other non-negative inclusive field endpoint.
        :return: A new :class:`MutableView`.
        """

    @classmethod
    def from_string(cls, s: str, /) -> Mutibs:
        """Create a new instance from a formatted string.

        This method initializes a new instance of :class:`Mutibs` using a formatted string.

        :param str s: The formatted string to convert. This can begin with '0b', '0o' or '0x' to indicate binary, octal or hexadecimal, and commas can be used to separate items.
        :return: A newly constructed ``Mutibs``.

        .. code-block:: python

            a = Mutibs.from_string("0xff01")
            b = Mutibs.from_string("0b1")

        The ``__init__`` method for ``Mutibs`` can also redirect to ``from_string``:

        .. code-block:: python

            a = Mutibs("0xff01")
        """

    @classmethod
    def from_bin(cls, s: str, /) -> Mutibs:
        """Create a new instance from a binary string.

        :param str s: A string of ``0`` and ``1`` s, optionally preceded with ``0b`` and optionally containing underscores.
        :return: A newly constructed ``Mutibs``.

        .. code-block:: python

            a = Mutibs.from_bin("0000_1111_0101")
        """

    def to_bin(self, start: int | None = None, end: int | None = None) -> str:
        """Return the binary representation of the Mutibs as a string.

        Equivalent to using the ``bin`` property when called with no parameters.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The binary representation.
        """

    def write_bin(self, s: str, /) -> None:
        """Replace the current bits from a binary string.

        This can change the length of the ``Mutibs``.

        :param str s: A string of ``0`` and ``1`` s, optionally preceded with ``0b`` and optionally containing underscores.
        :return: None

        .. code-block:: pycon

            >>> m = Mutibs()
            >>> m.write_bin('101')
            >>> m
            Mutibs('0b101')
        """

    @property
    def bin(self) -> str:
        """Property of the binary representation of the Mutibs.

        Reading is equivalent to using :meth:`~to_bin` with no parameters.
        Assigning is equivalent to using :meth:`~write_bin` and can change the length.

        :return: The binary representation.
        """

    @bin.setter
    def bin(self, value: str) -> None: ...

    @classmethod
    def from_oct(cls, s: str, /) -> Mutibs:
        """Create a new instance from an octal string.

        :param str s: A string of octal digits, optionally preceded with ``0o`` and optionally containing underscores.
        :return: A newly constructed ``Mutibs``.

        .. code-block:: pycon

            >>> Mutibs.from_oct("17")
            Mutibs('0b001111')
        """

    def to_oct(self, start: int | None = None, end: int | None = None) -> str:
        """Return the octal representation of the Mutibs as a string.

        Equivalent to using the ``oct`` property when called with no parameters.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The octal representation.
        :raises ValueError: if the length is not a multiple of 3.
        """

    def write_oct(self, s: str, /) -> None:
        """Replace the current bits from an octal string.

        This can change the length of the ``Mutibs``.

        :param str s: A string of octal digits, optionally preceded with ``0o`` and optionally containing underscores.
        :return: None

        .. code-block:: pycon

            >>> m = Mutibs()
            >>> m.write_oct('17')
            >>> m
            Mutibs('0b001111')
        """

    @property
    def oct(self) -> str:
        """Property of the octal representation of the Mutibs.

        Reading is equivalent to using :meth:`~to_oct` with no parameters.
        Assigning is equivalent to using :meth:`~write_oct` and can change the length.

        :return: The octal representation.
        :raises ValueError: if the length is not a multiple of 3.
        """

    @oct.setter
    def oct(self, value: str) -> None: ...

    @classmethod
    def from_hex(cls, s: str, /) -> Mutibs:
        """Create a new instance from a hexadecimal string.

        Equivalent to using the ``hex`` property.

        :param str s: A string of hexadecimal digits, optionally preceded with ``0x`` and optionally containing underscores.
        :return: A newly constructed ``Mutibs``.

        .. code-block:: pycon

            >>> Mutibs.from_hex("0f")
            Mutibs('0x0f')
        """

    def to_hex(self, start: int | None = None, end: int | None = None) -> str:
        """Return the hexadecimal representation of the Mutibs as a string.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The hexadecimal representation.
        :raises ValueError: if the length is not a multiple of 4.
        """

    def write_hex(self, s: str, /) -> None:
        """Replace the current bits from a hexadecimal string.

        This can change the length of the ``Mutibs``.

        :param str s: A string of hexadecimal digits, optionally preceded with ``0x`` and optionally containing underscores.
        :return: None

        .. code-block:: pycon

            >>> m = Mutibs()
            >>> m.write_hex('0f')
            >>> m
            Mutibs('0x0f')
        """

    @property
    def hex(self) -> str:
        """Property of the hexadecimal representation of the Mutibs.

        Reading is equivalent to using :meth:`~to_hex` with no parameters.
        Assigning is equivalent to using :meth:`~write_hex` and can change the length.

        :return: The hexadecimal representation.
        :raises ValueError: if the length is not a multiple of 4.
        """

    @hex.setter
    def hex(self, value: str) -> None: ...

    def to_bytes(self, start: int | None = None, end: int | None = None) -> bytes:
        """Return the Mutibs as a bytes object.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The bytes representation.
        :raises ValueError: if the length is not a multiple of 8.
        """

    def to_padded_bytes(self, start: int | None = None, end: int | None = None) -> bytes:
        """Return the Mutibs as a bytes object, padding the right-hand side with zero bits.

        This appends 0 to 7 zero bits to the end of the selected bit sequence so
        the returned value has a whole number of bytes. If the selected length is
        already a multiple of 8, this is equivalent to :meth:`~to_bytes`.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The padded bytes representation.
        """

    def write_bytes(self, data: bytes | bytearray | memoryview, /) -> None:
        """Replace the current bits from a bytes-like object.

        This can change the length of the ``Mutibs``.

        :param bytes data: A bytes-like object.
        :return: None

        .. code-block:: pycon

            >>> m = Mutibs()
            >>> m.write_bytes(b'A')
            >>> m
            Mutibs('0x41')
        """

    @property
    def bytes(self) -> bytes:
        """Property of the ``bytes`` representation of the Mutibs.

        Reading is equivalent to using :meth:`~to_bytes` with no parameters.
        Assigning is equivalent to using :meth:`~write_bytes` and can change the length.

        :return: The bytes representation.
        :raises ValueError: if the length is not a multiple of 8.
        """

    @bytes.setter
    def bytes(self, value: bytes | bytearray | memoryview) -> None: ...

    def take_raw_data(self) -> tuple[bytes, int, int]:
        """Return the raw bytes and offset information, leaving the Mutibs empty.

        This returns the underlying byte data using a move rather than a copy, and can contain
        leading and trailing bits that are not considered part of the object's value. Usually using
        :meth:`~to_bytes` is what you really need.

        :return: A tuple of the raw bytes, the bit offset and the bit length.

        .. code-block:: python

            raw_bytes, offset, length = t.take_raw_data()
            assert t == []
        """

    @classmethod
    def from_u(cls, u: int, /, length: int, byte_order: ByteOrder = ByteOrder.Unspecified) -> Mutibs:
        """Create a new instance from an unsigned integer.

        :param int u: An unsigned integer.
        :param int length: The bit length to create. Can be any positive number of bits.
        :param ByteOrder byte_order: The byte order used to store the integer. Defaults to ByteOrder.Unspecified.
        :return: A newly constructed ``Mutibs``.

        :raises ValueError: if the integer doesn't fit in the length given.

        .. code-block:: pycon

            >>> Mutibs.from_u(15, length=8)
            Mutibs('0x0f')
        """

    def to_u(self, start: int | None = None, end: int | None = None) -> int:
        """Return the unsigned integer representation of the Mutibs.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The value as an unsigned integer.

        .. code-block:: pycon

            >>> Mutibs('0x0f').to_u()
            15
        """

    def write_u(self, u: int, /) -> None:
        """Write the current bits from an unsigned integer without changing the length.

        :param int u: An unsigned integer.
        :return: None

        :raises ValueError: if the current length is zero.
        :raises ValueError: if the integer doesn't fit in the current length.

        .. code-block:: pycon

            >>> m = Mutibs.from_zeros(8)
            >>> m.write_u(15)
            >>> m
            Mutibs('0x0f')
        """

    @property
    def u(self) -> int:
        """Property of the unsigned integer representation of the Mutibs.

        Reading is equivalent to using :meth:`~to_u` with no parameters. Assigning
        is equivalent to using :meth:`~write_u`.

        :return: The value as an unsigned integer.
        """

    @u.setter
    def u(self, value: int) -> None: ...

    @classmethod
    def from_i(cls, i: int, /, length: int, byte_order: ByteOrder = ByteOrder.Unspecified) -> Mutibs:
        """Create a new instance from a signed integer.

        :param int i: A signed integer.
        :param int length: The bit length to create. Can be any positive number of bits.
        :param ByteOrder byte_order: The byte order used to store the integer. Defaults to ByteOrder.Unspecified.
        :return: A newly constructed ``Mutibs``.

        :raises ValueError: if the integer doesn't fit in the length given.

        .. code-block:: pycon

            >>> Mutibs.from_i(-2, length=4)
            Mutibs('0xe')
        """

    def to_i(self, start: int | None = None, end: int | None = None) -> int:
        """Return the signed integer representation of the Mutibs.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The value as a signed integer.

        .. code-block:: pycon

            >>> Mutibs('0xe').to_i()
            -2
        """

    def write_i(self, i: int, /) -> None:
        """Write the current bits from a signed integer without changing the length.

        :param int i: A signed integer.
        :return: None

        :raises ValueError: if the current length is zero.
        :raises ValueError: if the integer doesn't fit in the current length.

        .. code-block:: pycon

            >>> m = Mutibs.from_zeros(4)
            >>> m.write_i(-2)
            >>> m
            Mutibs('0xe')
        """

    @property
    def i(self) -> int:
        """Property of the signed integer representation of the Mutibs.

        Reading is equivalent to using :meth:`~to_i` with no parameters. Assigning
        is equivalent to using :meth:`~write_i`.

        :return: The value as a signed integer.
        """

    @i.setter
    def i(self, value: int) -> None: ...

    @classmethod
    def from_f(cls, f: float, /, length: int, byte_order: ByteOrder = ByteOrder.Unspecified) -> Mutibs:
        """Create a new instance from a floating point number.

        :param float f: A floating point value.
        :param int length: The bit length to create. Must be 16, 32 or 64.
        :param ByteOrder byte_order: The byte order used to store the float. Defaults to ByteOrder.Unspecified.
        :return: A newly constructed ``Mutibs``.

        .. code-block:: pycon

            >>> Mutibs.from_f(1.5, length=32)
            Mutibs('0x3fc00000')
        """

    def to_f(self, start: int | None = None, end: int | None = None) -> float:
        """Return the floating point representation of the Mutibs.

        The length must be 16, 32 or 64.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).

        :return: The value as a Python float.

        .. code-block:: pycon

            >>> Mutibs('0x3fc00000').to_f()
            1.5
        """

    def write_f(self, f: float, /) -> None:
        """Write the current bits from a floating point number without changing the length.

        The current length must be 16, 32 or 64 bits.

        :param float f: A floating point value.
        :return: None

        .. code-block:: pycon

            >>> m = Mutibs.from_zeros(32)
            >>> m.write_f(1.5)
            >>> m
            Mutibs('0x3fc00000')
        """

    @property
    def f(self) -> float:
        """Property of the floating point representation of the Mutibs.

        Reading is equivalent to using :meth:`~to_f` with no parameters. Assigning
        is equivalent to using :meth:`~write_f`.

        :return: The value as a Python float.
        """

    @f.setter
    def f(self, value: float) -> None: ...

    @classmethod
    def from_zeros(cls, length: int, /) -> Mutibs:
        """Create a new instance with all bits set to zero.

        :param int length: The number of bits to set.
        :return: A Mutibs object with all bits set to zero.

        .. code-block:: python

            a = Mutibs.from_zeros(500)  # 500 zero bits
        """

    @classmethod
    def from_ones(cls, length: int, /) -> Mutibs:
        """Create a new instance with all bits set to one.

        :param int length: The number of bits to set.
        :return: A Mutibs object with all bits set to one.

        .. code-block:: pycon

            >>> Mutibs.from_ones(5)
            Mutibs('0b11111')
        """

    @classmethod
    def from_bools(cls, iterable: Iterable[Any], /) -> Mutibs:
        """Create a new instance from an iterable by converting each element to a bool.

        :param Iterable iterable: The iterable to convert to a :class:`Mutibs`.
        :return: A newly constructed ``Mutibs``.

        .. code-block:: python

            a = Mutibs.from_bools([False, 0, 1, "Steven"])  # binary 0011
        """

    def to_bools(self, start: int | None = None, end: int | None = None) -> list[bool]:
        """Return the bits as a list of bools.

        This is much faster than using ``list()`` on the Mutibs, which iterates bit by bit.

        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).
        :return: A list of bools.

        .. code-block:: pycon

            >>> Mutibs('0b101').to_bools()
            [True, False, True]
        """

    @classmethod
    def from_random(cls, length: int, /, secure: bool = False, seed: bytes | bytearray | None = None) -> Mutibs:
        """Create a new instance with all bits randomly set.

        :param int length: The number of bits to set. Must be non-negative.
        :param bool secure: If ``True``, use the OS's cryptographically secure generator. Default is ``False``.
        :param bytes | bytearray | None seed: A bytes or bytearray to use as an optional seed, only if ``secure`` is ``False``.
        :return: A newly constructed ``Mutibs`` with random data.

        The 'secure' option uses the OS's random data source, so will be slower and could potentially
        fail.

        .. code-block:: python

            a = Mutibs.from_random(1000000)  # A million random bits
            b = Mutibs.from_random(100, seed=b'a_seed')
        """

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview, /, bit_offset: int | None = None,
                   bit_length: int | None = None) -> Mutibs:
        """Create a new instance from a bytes object.

        :param bytes | bytearray | memoryview data: The bytes, bytearray or memoryview object to convert to a :class:`Mutibs`.
        :param int | None bit_offset: How many bits to skip from the start. Defaults to zero.
        :param int | None bit_length: How many bits to use. Defaults to the whole of the data.
        :return: A newly constructed ``Mutibs``.

        Both are counts of bits, not bytes, so to skip a four byte header use
        ``bit_offset=32``.

        .. code-block:: python

            a = Mutibs.from_bytes(b"some_bytes_maybe_from_a_file")
        """

    @classmethod
    def from_joined(cls, iterable: Iterable[_TibsLike], /) -> Mutibs:
        """Create a new instance by concatenating a sequence of Mutibs objects.

        This method concatenates a sequence of Mutibs objects into a single Mutibs object.

        :param Iterable iterable: An iterable to concatenate. Items can be anything that can be promoted to a :class:`Mutibs`.
        :return: A newly constructed ``Mutibs``.

        .. code-block:: python

            a = Mutibs.from_joined(['0x01', [1, 0], b'some_bytes'])
        """

    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, key: SupportsIndex) -> bool: ...

    @overload
    def __getitem__(self, key: slice) -> Mutibs: ...

    def __getitem__(self, key: SupportsIndex | slice) -> bool | Mutibs: ...

    @overload
    def __setitem__(self, key: SupportsIndex, value: Any) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: _TibsLike) -> None: ...

    def __setitem__(self, key: SupportsIndex | slice, value: Any | _TibsLike) -> None: ...

    def __delitem__(self, key: SupportsIndex | slice) -> None: ...

    def chunks(self, chunk_size: int, /, count: int | None = None) -> list[Mutibs]:
        """Return a list of Mutibs by cutting into chunks.

        :param int chunk_size: The size in bits of the chunks to create.
        :param int | None count: If specified, at most count items are created. Default is to cut as many times as possible.
        :return: A list of Mutibs chunks.

        .. code-block:: pycon

            >>> Mutibs('0b110011').chunks(2)
            [Mutibs('0b11'), Mutibs('0b00'), Mutibs('0b11')]
        """

    @overload
    def split_at(self, pos: SupportsIndex, /) -> tuple[Mutibs, Mutibs]:
        """Split at one or more bit positions.

        ``pos`` may be a single integer or an iterable of integers. Negative
        positions count from the end. Positions must be in nondecreasing order
        after normalization, and each position must be in the range
        ``0`` through ``len(self)``, inclusive.

        The returned pieces are new ``Mutibs`` objects, matching normal
        ``Mutibs`` slice behavior.

        :param int | Iterable[int] pos: The bit position or positions where the split should occur.
        :return: A tuple of ``Mutibs`` pieces.

        .. code-block:: pycon

            >>> Mutibs('0b101100').split_at(3)
            (Mutibs('0b101'), Mutibs('0b100'))
            >>> Mutibs('0b101100').split_at([2, 5])
            (Mutibs('0b10'), Mutibs('0b110'), Mutibs('0b0'))
        """

    @overload
    def split_at(self, pos: Iterable[SupportsIndex], /) -> tuple[Mutibs, ...]:
        """Split at one or more bit positions.

        ``pos`` may be a single integer or an iterable of integers. Negative
        positions count from the end. Positions must be in nondecreasing order
        after normalization, and each position must be in the range
        ``0`` through ``len(self)``, inclusive.

        The returned pieces are new ``Mutibs`` objects, matching normal
        ``Mutibs`` slice behavior.

        :param int | Iterable[int] pos: The bit position or positions where the split should occur.
        :return: A tuple of ``Mutibs`` pieces.

        .. code-block:: pycon

            >>> Mutibs('0b101100').split_at(3)
            (Mutibs('0b101'), Mutibs('0b100'))
            >>> Mutibs('0b101100').split_at([2, 5])
            (Mutibs('0b10'), Mutibs('0b110'), Mutibs('0b0'))
        """

    @classmethod
    def from_value(cls, dtype: _DtypeLike, value: Any, /) -> Mutibs:
        """Create a new instance by encoding one value with a dtype.

        :param Dtype | str dtype: The value encoding to use.
        :param object value: The value to encode.
        :return: A newly constructed ``Mutibs``.

        .. code-block:: pycon

            >>> Mutibs.from_value("u8", 15)
            Mutibs('0x0f')
        """

    @classmethod
    def from_values(cls, dtype: _DtypeLike, iterable: Iterable[Any], /) -> Mutibs:
        """Create a new instance by encoding and concatenating values with a dtype.

        :param Dtype | str dtype: The value encoding to use for each item.
        :param Iterable iterable: The values to encode.
        :return: A newly constructed ``Mutibs``.

        .. code-block:: pycon

            >>> Mutibs.from_values("u8", [1, 2, 3])
            Mutibs('0x010203')
        """

    def to_value(self, dtype: _DtypeLike, /, start: int | None = None, end: int | None = None) -> _DtypeValue:
        """Return one value decoded with a dtype.

        The selected range must have exactly the dtype length.

        :param Dtype | str dtype: The value encoding to use.
        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).
        :return: The decoded Python value.

        .. code-block:: pycon

            >>> Mutibs('0x0f').to_value("u8")
            15
        """

    def to_values(self, dtype: _DtypeLike, /, start: int | None = None, end: int | None = None) -> list[
        _DtypeValue]:
        """Return a list of values decoded with a dtype.

        The selected range must be a whole number of dtype values. The values are
        decoded from the current contents when the method is called.

        :param Dtype | str dtype: The value encoding to use for each item.
        :param int | None start: Start bit position. Defaults to 0.
        :param int | None end: End bit position. Defaults to len(self).
        :return: A list of decoded Python values.

        .. code-block:: pycon

            >>> Mutibs('0x010203').to_values("u8")
            [1, 2, 3]
        """

    def starts_with(self, prefix: _TibsLike, /) -> bool:
        """Return whether the current Mutibs starts with prefix.

        :param object prefix: The bits to search for. This can be anything promotable to ``Tibs``.
        :return: True if the Mutibs starts with the prefix, otherwise False.

        .. code-block:: pycon

            >>> Mutibs('0b101100').starts_with('0b101')
            True
            >>> Mutibs('0b101100').starts_with('0b100')
            False
        """

    def __contains__(self, b: _TibsLike) -> bool: ...

    def ends_with(self, suffix: _TibsLike, /) -> bool:
        """Return whether the current Mutibs ends with suffix.

        :param object suffix: The bits to search for. This can be anything promotable to ``Tibs``.
        :return: True if the Mutibs ends with the suffix, otherwise False.

        .. code-block:: pycon

            >>> Mutibs('0b101100').ends_with('0b100')
            True
            >>> Mutibs('0b101100').ends_with('0b101')
            False
        """

    def find(self, needle: _TibsLike, /, start: int | None = None, end: int | None = None,
             byte_aligned: bool = False, mask: _TibsLike | None = None) -> int | None:
        """Find first occurrence of a bit sequence.

        Returns the bit position if found, or None if not found.

        :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
        :param int | None start: The starting bit position. Defaults to 0.
        :param int | None end: The end position. Defaults to len(self).
        :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: The bit position if found, or None if not found.
        :raises ValueError: if ``needle`` is empty, if the slice parameters are invalid, or if the
            mask length doesn't match the needle length.

        .. code-block:: pycon

             >>> Mutibs('0xc3e').find('0b1111')
             6
             >>> Mutibs('0x3a5f').find('0x0f', mask='0x0f', byte_aligned=True)
             8
        """

    def find_all(self, needle: _TibsLike, /, start: int | None = None, end: int | None = None,
                 byte_aligned: bool = False, mask: _TibsLike | None = None) -> list[int]:
        """Find all occurrences of a bit sequence.

        :param object needle: The bit sequence to find. This can be anything promotable to ``Tibs``.
        :param int | None start: The starting bit position. Defaults to 0.
        :param int | None end: The end position. Defaults to len(self).
        :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: A list of bit positions.
        :raises ValueError: if ``needle`` is empty, if the slice parameters are invalid, or if the
            mask length doesn't match the needle length.

        All occurrences of needle are found, even if they overlap.

        .. code-block:: pycon

             >>> Mutibs('0xc3e').find_all('0b1111')
             [6, 7]
        """

    def __and__(self, bs: _TibsLike) -> Mutibs: ...

    def __or__(self, bs: _TibsLike) -> Mutibs: ...

    def __xor__(self, bs: _TibsLike) -> Mutibs: ...

    def __rand__(self, bs: _TibsLike) -> Mutibs: ...

    def __ror__(self, bs: _TibsLike) -> Mutibs: ...

    def __rxor__(self, bs: _TibsLike) -> Mutibs: ...

    def rotate_left(self, n: int, /, start: int | None = None, end: int | None = None) -> None:
        """Rotates bit pattern to the left in-place.

        :param int n: The number of bits to rotate by.
        :param int | None start: Start of slice to rotate. Defaults to 0.
        :param int | None end: End of slice to rotate. Defaults to len(self).
        :return: None

        :raises ValueError: if n < 0.

        .. code-block:: pycon

            >>> a = Mutibs('0b10110')
            >>> a.rotate_left(2)
            >>> a
            Mutibs('0b11010')
        """

    def rotate_right(self, n: int, /, start: int | None = None, end: int | None = None) -> None:
        """Rotates bit pattern to the right in-place.

        :param int n: The number of bits to rotate by.
        :param int | None start: Start of slice to rotate. Defaults to 0.
        :param int | None end: End of slice to rotate. Defaults to len(self).
        :return: None

        :raises ValueError: if n < 0.

        .. code-block:: pycon

            >>> a = Mutibs('0b10110')
            >>> a.rotate_right(1)
            >>> a
            Mutibs('0b01011')
        """

    def rotated_left(self, n: int, /, start: int | None = None, end: int | None = None) -> Mutibs:
        """Return a new Mutibs with the bits rotated to the left.

        This is the non-inplace version of :meth:`rotate_left`.

        :param int n: The number of bits to rotate by.
        :param int | None start: Start of slice to rotate. Defaults to 0.
        :param int | None end: End of slice to rotate. Defaults to len(self).
        :return: A new Mutibs.
        :raises ValueError: if n < 0.

        .. code-block:: pycon

            >>> Mutibs('0b10110').rotated_left(2)
            Mutibs('0b11010')
        """

    def rotated_right(self, n: int, /, start: int | None = None, end: int | None = None) -> Mutibs:
        """Return a new Mutibs with the bits rotated to the right.

        This is the non-inplace version of :meth:`rotate_right`.

        :param int n: The number of bits to rotate by.
        :param int | None start: Start of slice to rotate. Defaults to 0.
        :param int | None end: End of slice to rotate. Defaults to len(self).
        :return: A new Mutibs.
        :raises ValueError: if n < 0.

        .. code-block:: pycon

            >>> Mutibs('0b10110').rotated_right(1)
            Mutibs('0b01011')
        """

    @classmethod
    def decode(cls, b: bytes, /) -> Mutibs:
        """Create a Mutibs by decoding bytes created via `encode()`.

        :param bytes | bytearray b: The encoded bytes to decode.
        :return: A new Mutibs.
        :raises tibs.DecodeError: for badly formed, truncated or extended input bytes.

        .. code-block:: pycon

            >>> Mutibs.decode(Mutibs('0b101').encode())
            Mutibs('0b101')
        """

    def encode(self, codec: Codec = Codec.Auto) -> bytes:
        """Encode the Mutibs as a bytes instance.

        The bytes instance can be used to recreate the Mutibs exactly with :meth:`decode`.

        Use ``Codec.Raw`` when the encoded bytes themselves need to be a stable,
        canonical representation. The default ``Codec.Auto`` chooses a valid
        encoding for compactness and may produce different bytes for the same
        value in a future release.

        :param Codec codec: The codec to use. Defaults to Codec.Auto.
        :return: The encoded bytes.

        .. code-block:: pycon

            >>> Mutibs.decode(Mutibs('0b101').encode())
            Mutibs('0b101')
        """

    def set(self, pos: SupportsIndex | Iterable[SupportsIndex] | range, /) -> None:
        """Set one or many bits set to 1.

        :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
        :return: None
        :raises IndexError: if pos < -len(self) or pos >= len(self).

        See also :meth:`unset`.

        .. code-block:: pycon

            >>> a = Mutibs.from_zeros(10)
            >>> a.set(5)
            >>> a
            Mutibs('0b0000010000')
            >>> a.set([-1, -2])
            >>> a
            Mutibs('0b0000010011')
        """

    def unset(self, pos: SupportsIndex | Iterable[SupportsIndex] | range, /) -> None:
        """Set one or many bits set to 0.

        :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
        :return: None
        :raises IndexError: if pos < -len(self) or pos >= len(self).

        See also :meth:`set`.

        .. code-block:: pycon

            >>> a = Mutibs.from_ones(10)
            >>> a.unset(5)
            >>> a
            Mutibs('0b1111101111')
            >>> a.unset([-1, -2])
            >>> a
            Mutibs('0b1111101100')
        """

    def with_set(self, pos: SupportsIndex | Iterable[SupportsIndex] | range, /) -> Mutibs:
        """Return a new Mutibs with one or many bits set to 1.

        This is the non-inplace version of :meth:`set`.

        :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
        :return: A new Mutibs.
        :raises IndexError: if pos < -len(self) or pos >= len(self).

        .. code-block:: pycon

            >>> Mutibs.from_zeros(5).with_set([1, 3])
            Mutibs('0b01010')
        """

    def with_unset(self, pos: SupportsIndex | Iterable[SupportsIndex] | range, /) -> Mutibs:
        """Return a new Mutibs with one or many bits set to 0.

        This is the non-inplace version of :meth:`unset`.

        :param int | Iterable[int] pos: Either a single bit position or an iterable of bit positions.
        :return: A new Mutibs.
        :raises IndexError: if pos < -len(self) or pos >= len(self).

        .. code-block:: pycon

            >>> Mutibs.from_ones(5).with_unset([1, 3])
            Mutibs('0b10101')
        """

    def count(self, value: Any = None, /, start: int | None = None, end: int | None = None,
              byte_aligned: bool = False, mask: _TibsLike | None = None) -> int:
        """Counts the total number of occurrences of a bit pattern.

        :param object | None value: Either something that can be converted to a ``Tibs``, or a single bit (one of ``0``, ``1``, ``False`` or ``True``). Defaults to counting the set bits.
        :param int | None start: The start of the slice to count within. Defaults to 0.
        :param int | None end: The end of the slice to count within. Defaults to len(self).
        :param bool byte_aligned: If ``True``, only occurrences on byte boundaries are counted. Defaults to ``False``.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.

        :return: The number of times the bit pattern is found.
        :raises ValueError: if the slice parameters are invalid, or if the mask length doesn't match
            the length of the value.

        With no ``value`` this counts the set bits, so ``count()`` is the same as ``count(1)``.

        When ``value`` is a multi-bit pattern, overlapping occurrences are all counted,
        just as they are by :meth:`find_all`. ``byte_aligned`` also applies to single-bit
        counts: ``count(1, byte_aligned=True)`` counts the set bits that land on a byte
        boundary.

        .. code-block:: pycon

            >>> Mutibs('0xef').count()
            7
            >>> Mutibs('0xef').count(1, 0, 4)
            3
            >>> Mutibs('0xff00ff').count([1, 1, 1])  # overlapping
            12
            >>> Mutibs('0x80ff00').count(1, byte_aligned=True)
            2
        """

    def count_and(self, other: _TibsLike, /) -> int:
        """Count the bits set in both this Mutibs and another.

        Equivalent to ``(self & other).count(1)``, but without building the
        intermediate object.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: The number of positions set in both.
        :raises ValueError: if the two lengths differ.

        .. code-block:: pycon

            >>> Mutibs('0b1100').count_and('0b1010')
            1
        """

    def count_or(self, other: _TibsLike, /) -> int:
        """Count the bits set in either this Mutibs or another.

        Equivalent to ``(self | other).count(1)``, but without building the
        intermediate object.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: The number of positions set in either.
        :raises ValueError: if the two lengths differ.

        .. code-block:: pycon

            >>> Mutibs('0b1100').count_or('0b1010')
            3
        """

    def count_xor(self, other: _TibsLike, /) -> int:
        """Count the bits that differ between this Mutibs and another.

        This is the Hamming distance. Equivalent to ``(self ^ other).count(1)``,
        but without building the intermediate object.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: The number of positions where the two differ.
        :raises ValueError: if the two lengths differ.

        .. code-block:: pycon

            >>> Mutibs('0b1100').count_xor('0b1010')
            2
        """

    def count_andnot(self, other: _TibsLike, /) -> int:
        """Count the bits set in this Mutibs but not in another.

        Equivalent to ``self.count(1) - self.count_and(other)``, but in a single pass.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: The number of positions set here but not in the other.
        :raises ValueError: if the two lengths differ.

        .. code-block:: pycon

            >>> Mutibs('0b1100').count_andnot('0b1010')
            1
        """

    def intersects(self, other: _TibsLike, /) -> bool:
        """Return whether any bit is set in both this Mutibs and another.

        Equivalent to ``(self & other).any()``, but stops at the first bit set in
        both instead of building the intermediate object.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: ``True`` if some position is set in both, otherwise ``False``.
        :raises ValueError: if the two lengths differ.

        .. code-block:: pycon

            >>> Mutibs('0b1100').intersects('0b1010')
            True
            >>> Mutibs('0b1100').intersects('0b0011')
            False
        """

    def is_disjoint(self, other: _TibsLike, /) -> bool:
        """Return whether no bit is set in both this Mutibs and another.

        The negation of :meth:`intersects`. Equivalent to ``not (self & other).any()``,
        but stops at the first bit set in both instead of building the
        intermediate object.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: ``True`` if no position is set in both, otherwise ``False``.
        :raises ValueError: if the two lengths differ.

        .. code-block:: pycon

            >>> Mutibs('0b1100').is_disjoint('0b0011')
            True
            >>> Mutibs('0b1100').is_disjoint('0b1010')
            False
        """

    def is_subset_of(self, other: _TibsLike, /) -> bool:
        """Return whether every bit set in this Mutibs is also set in another.

        Equivalent to ``(self & other) == self``, but stops at the first bit set
        here and not there.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: ``True`` if every position set here is set in the other, otherwise ``False``.
        :raises ValueError: if the two lengths differ.

        .. code-block:: pycon

            >>> Mutibs('0b1000').is_subset_of('0b1010')
            True
            >>> Mutibs('0b1100').is_subset_of('0b1010')
            False
        """

    def is_superset_of(self, other: _TibsLike, /) -> bool:
        """Return whether every bit set in another is also set in this Mutibs.

        The mirror of :meth:`is_subset_of`. Equivalent to ``(self & other) == other``,
        but stops at the first bit set there and not here.

        :param object other: The other bits. This can be anything promotable to ``Tibs``.
        :return: ``True`` if every position set in the other is set here, otherwise ``False``.
        :raises ValueError: if the two lengths differ.

        .. code-block:: pycon

            >>> Mutibs('0b1010').is_superset_of('0b1000')
            True
            >>> Mutibs('0b1010').is_superset_of('0b1100')
            False
        """

    def extracted(self, mask: _TibsLike, /) -> Mutibs:
        """Read the bits at the positions set in a mask, packed together.

        This reads a bit field whose bits are scattered through the Mutibs by the
        mask, the way :meth:`field` reads a contiguous one. The result has one
        bit for each set bit of the mask, in order.

        :param object mask: The mask selecting which bits to read. This can be anything promotable to ``Tibs``, and must be the same length as ``self``.
        :return: A new Mutibs of length ``mask.count()``.
        :raises ValueError: if the mask length doesn't match the length of ``self``.

        .. code-block:: pycon

            >>> Mutibs('0b11010110').extracted('0b10110000')
            Mutibs('0b101')
        """

    def deposit(self, value: _TibsLike, mask: _TibsLike, /) -> None:
        """Write a scattered bit field into the Mutibs in place.

        This is the inverse of :meth:`extracted`, and writes a scattered field the
        way slice assignment writes a contiguous one: the bits of ``value`` are
        written into the positions set in ``mask``, and the other bits are left
        unchanged.

        :param object value: The bits to deposit. This can be anything promotable to ``Tibs``, and must be ``mask.count()`` bits long.
        :param object mask: The mask selecting which positions to write. This can be anything promotable to ``Tibs``, and must be the same length as ``self``.
        :return: None
        :raises ValueError: if the mask length doesn't match the length of ``self``, or ``value`` is not ``mask.count()`` bits long.

        .. code-block:: pycon

            >>> m = Mutibs('0b11010110')
            >>> m.deposit('0b111', '0b10110000')
            >>> m.bin
            '11110110'
        """

    def deposited(self, value: _TibsLike, mask: _TibsLike, /) -> Mutibs:
        """Return a new Mutibs with a scattered bit field written into it.

        This is the non-inplace version of :meth:`deposit`.

        :param object value: The bits to deposit. This can be anything promotable to ``Tibs``, and must be ``mask.count()`` bits long.
        :param object mask: The mask selecting which positions to write. This can be anything promotable to ``Tibs``, and must be the same length as ``self``.
        :return: A new Mutibs.
        :raises ValueError: if the mask length doesn't match the length of ``self``, or ``value`` is not ``mask.count()`` bits long.

        .. code-block:: pycon

            >>> Mutibs('0b11010110').deposited('0b111', '0b10110000').bin
            '11110110'
        """

    def all(self) -> bool:
        """Return True if all bits are equal to 1, otherwise return False.

        :return: ``True`` if all bits are 1, otherwise ``False``.

        .. code-block:: pycon

            >>> Mutibs('0b1111').all()
            True
            >>> Mutibs('0b1011').all()
            False
        """

    def any(self) -> bool:
        """Return True if any bits are equal to 1, otherwise return False.

        :return: ``True`` if any bits are 1, otherwise ``False``.

        .. code-block:: pycon

            >>> Mutibs('0b0000').any()
            False
            >>> Mutibs('0b1000').any()
            True
        """

    def rfind(self, needle: _TibsLike, /, start: int | None = None, end: int | None = None,
              byte_aligned: bool = False, mask: _TibsLike | None = None) -> int | None:
        """Find last occurrence of a bit sequence.

        Returns the bit position if found, or None if not found.

        :param object needle: The bits to find. This can be anything promotable to ``Tibs``.
        :param int | None start: The starting bit position. Defaults to 0.
        :param int | None end: The end position. Defaults to len(self).
        :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: The bit position if found, or None if not found.
        :raises ValueError: if ``needle`` is empty, if the slice parameters are invalid, or if the
            mask length doesn't match the needle length.

        .. code-block:: pycon

             >>> Mutibs('0b10111011').rfind('0b11')
             6
             >>> Mutibs('0b10111011').rfind('0b00', mask='0b10')
             5
        """

    def invert(self, pos: SupportsIndex | Iterable[SupportsIndex] | None = None, /) -> None:
        """Invert one or many bits in place.

        :param int | Iterable[int] | None pos: Either a single bit position or an iterable of bit positions.
        :return: None

        :raises IndexError: if pos < -len(self) or pos >= len(self).

        .. code-block:: pycon

            >>> a = Mutibs('0b10111')
            >>> a.invert(1)
            >>> a
            Mutibs('0b11111')
            >>> a.invert([0, 2])
            >>> a
            Mutibs('0b01011')
            >>> a.invert()
            >>> a
            Mutibs('0b10100')
        """

    def inverted(self, pos: SupportsIndex | Iterable[SupportsIndex] | None = None, /) -> Mutibs:
        """Return a new Mutibs with selected bits inverted.

        This is the non-inplace version of :meth:`invert`.

        :param int | Iterable[int] | None pos: Either a single bit position, an iterable of bit positions,
          or None to invert every bit. Defaults to None.
        :return: A new Mutibs.
        :raises IndexError: if pos < -len(self) or pos >= len(self).

        .. code-block:: pycon

            >>> Mutibs('0b10110').inverted([0, 2])
            Mutibs('0b00010')
        """

    def reverse(self) -> None:
        """Reverse bits in-place.

        :return: None

        .. code-block:: pycon

            >>> a = Mutibs('0b10110')
            >>> a.reverse()
            >>> a
            Mutibs('0b01101')
        """

    def reversed(self) -> Mutibs:
        """Return a new instance with the bits reversed.

        :return: Mutibs

        .. code-block:: pycon

            >>> a = Mutibs('0b00011')
            >>> a.reversed()
            Mutibs('0b11000')
        """

    def byte_swap(self, byte_length: int | None = None, start: int | None = None,
                  end: int | None = None) -> None:
        """Swap byte order in-place.

        The selected slice will be byte-swapped. It must be a multiple of
        byte_length long.

        :param int | None byte_length: An int giving the number of bytes in each swap, or None (the default)
          to do a single reverse over the selected slice.
        :param int | None start: Start of slice to byte-swap. Defaults to 0.
        :param int | None end: End of slice to byte-swap. Defaults to len(self).
        :return: None

        .. code-block:: pycon

            >>> a = Mutibs('0x12345678')
            >>> a.byte_swap(2)
            >>> a
            Mutibs('0x34127856')
        """

    def byte_swapped(self, byte_length: int | None = None, start: int | None = None,
                     end: int | None = None) -> Mutibs:
        """Return a new instance with the byte order swapped.

        The selected slice will be byte-swapped. It must be a multiple of
        byte_length long.

        :param int | None byte_length: An int giving the number of bytes in each swap, or None (the default)
          to do a single reverse over the selected slice.
        :param int | None start: Start of slice to byte-swap. Defaults to 0.
        :param int | None end: End of slice to byte-swap. Defaults to len(self).
        :return: Mutibs

        .. code-block:: pycon

            >>> a = Mutibs('0x12345678')
            >>> b = a.byte_swapped(2)
            >>> b
            Mutibs('0x34127856')
        """

    def __invert__(self) -> Mutibs: ...

    def __lshift__(self, n: int) -> Mutibs: ...

    def __rshift__(self, n: int) -> Mutibs: ...

    def __copy__(self) -> Mutibs:
        """Return a new copy of the Mutibs for the copy module."""

    def __reduce__(self) -> tuple[Callable[[bytes], Mutibs], tuple[bytes]]:
        """Return the callable and arguments that recreate the Mutibs.

        Used by :mod:`pickle` and by :func:`copy.deepcopy`. The Mutibs that comes
        back is a new object, independent of the one that was pickled or copied.

        :return: A tuple of :meth:`Mutibs.decode` and the encoded bytes to pass to it.

        .. code-block:: pycon

            >>> import pickle
            >>> pickle.loads(pickle.dumps(Mutibs('0b110101')))
            Mutibs('0b110101')
        """

    def to_tibs(self) -> Tibs:
        """Create and return a Tibs instance from a copy of the Mutibs data.

        This copies the underlying binary data, giving a new independent Tibs object.
        If you no longer need the Mutibs, consider using :meth:`take_tibs` instead to avoid the copy.

        :return: A new Tibs instance with the same bit data.

        .. code-block:: pycon

            >>> a = Mutibs('0b10110')
            >>> b = a.to_tibs()
            >>> a
            Mutibs('0b10110')
            >>> b
            Tibs('0b10110')
        """

    def take_tibs(self) -> Tibs:
        """Create and return a Tibs instance by moving the Mutibs data.

        The data is moved to the new Tibs, so the Mutibs will be empty after the operation.
        This is more efficient than :meth:`to_tibs` if you no longer need the Mutibs.

        It will try to reclaim any excess memory capacity that the Mutibs may have had.

        :return: A Tibs instance with the same bit data.

        .. code-block:: pycon

            >>> a = Mutibs('0b10110')
            >>> b = a.take_tibs()
            >>> a
            Mutibs()
            >>> b
            Tibs('0b10110')
        """

    def clear(self) -> None:
        """Clear all bits, making the Mutibs empty.

        This doesn't change the allocated capacity, so won't free up any memory.

        :return: None.

        .. code-block:: pycon

            >>> m = Mutibs('0b1011')
            >>> m.clear()
            >>> m
            Mutibs()
        """

    @property
    def capacity(self) -> int:
        """Return the number of bits the Mutibs can hold without reallocating memory.

        The capacity is always equal to or greater than the current length of the Mutibs.
        If the length ever exceeds the capacity then memory will have to be reallocated, and the
        capacity will increase.

        It can be helpful as a performance optimization to reserve enough capacity before
        constructing a large Mutibs incrementally. See also :meth:`reserve`.

        :return: The current capacity in bits.

        .. code-block:: pycon

            >>> m = Mutibs()
            >>> m.capacity >= len(m)
            True
        """

    def reserve(self, additional: int, /) -> None:
        """Reserve memory for at least `additional` more bits to be appended to the Mutibs.

        This can be helpful as a performance optimization to avoid multiple memory reallocations when
        constructing a large Mutibs incrementally. If enough memory is already reserved then
        this method will have no effect. See also :attr:`capacity`.

        :param int additional: The number of bits that can be appended without any further memory reallocations.
        :return: None.
        :raises MemoryError: if the memory cannot be reserved.

        .. code-block:: python

            m = Mutibs()
            m.reserve(1000)
        """

    def __add__(self, bs: _TibsLike) -> Mutibs: ...

    def __radd__(self, bs: _TibsLike) -> Mutibs: ...

    def __iadd__(self, bs: _TibsLike) -> Mutibs: ...

    def append(self, bit: bool | int, /) -> None:
        """Append a single bit to the current Mutibs in-place.

        :param bool | int bit: Either ``0``, ``1``, ``True`` or ``False`` to append.
        :return: None

        .. code-block:: pycon

            >>> a = Mutibs()
            >>> a.append(True)
            >>> a
            Mutibs('0b1')
        """

    def pop(self) -> bool:
        """Remove and return the final bit.

        :return: bool
        :raises IndexError: if the Mutibs is empty.

        .. code-block:: pycon

            >>> a = Mutibs('0b101')
            >>> a.pop()
            True
            >>> a
            Mutibs('0b10')
        """

    def extend(self, bs: _TibsLike, /) -> None:
        """Extend the current Mutibs in-place.

        :param object bs: The bits to extend with. This can be anything promotable to ``Tibs``.
        :return: None

        .. code-block:: pycon

            >>> a = Mutibs('0x0f')
            >>> a.extend('0x0a')
            >>> a
            Mutibs('0x0f0a')
        """

    def extend_left(self, bs: _TibsLike, /) -> None:
        """Extend the current Mutibs in-place from the start.

        This is broadly equivalent to ``self = bs + self``.
        Note that this method is inherently slower than :meth:`extend` and
        should be avoided in performance critical code. See also :meth:`from_joined`.

        :param object bs: The bits to prepend to the current Mutibs. This can be anything promotable to ``Tibs``.
        :return: None

        .. code-block:: pycon

            >>> a = Mutibs('0x0f')
            >>> a.extend_left('0x0a')
            >>> a
            Mutibs('0x0a0f')
        """

    def replace(self, old: _TibsLike, new: _TibsLike, /, start: int | None = None, end: int | None = None,
                count: int | None = None, byte_aligned: bool = False,
                mask: _TibsLike | None = None) -> int:
        """Search and replace in-place.

        :param object old: The bits to search for. This can be anything promotable to ``Tibs``.
        :param object new: The bits to replace with. This can be anything promotable to ``Tibs``.
        :param int | None start: The starting bit position. Defaults to 0.
        :param int | None end: The end position. Defaults to len(self).
        :param int | None count: If present, the maximum number of replacements to make.
        :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: The number of replacements made.
        :raises ValueError: if old is empty, count is negative, the slice parameters are invalid or
            the mask length doesn't match the length of old.

        The ``mask`` affects only which bits have to match; the whole of each match is still
        replaced by ``new``.

        Matches cannot overlap, since replacing one consumes the bits the next would have
        matched, so this can make fewer replacements than :meth:`count` finds occurrences.

        .. code-block:: pycon

            >>> m = Mutibs('0b00010010')
            >>> m.replace([0, 1], [1, 1, 1])
            2
            >>> m
            Mutibs('0b0011101110')
        """

    def replaced(self, old: _TibsLike, new: _TibsLike, /, start: int | None = None, end: int | None = None,
                 count: int | None = None, byte_aligned: bool = False,
                 mask: _TibsLike | None = None) -> Mutibs:
        """Search and replace and return a new Mutibs.

        This is the non-inplace version of :meth:`replace`.

        :param object old: The bits to search for. This can be anything promotable to ``Tibs``.
        :param object new: The bits to replace with. This can be anything promotable to ``Tibs``.
        :param int | None start: The starting bit position. Defaults to 0.
        :param int | None end: The end position. Defaults to len(self).
        :param int | None count: If present, the maximum number of replacements to make.
        :param bool byte_aligned: If ``True``, the bits will only be found on byte boundaries.
        :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
        :return: A new Mutibs.
        :raises ValueError: if old is empty, count is negative, the slice parameters are invalid or
            the mask length doesn't match the length of old.

        .. code-block:: pycon

            >>> Mutibs('0b00010010').replaced([0, 1], [1, 1, 1])
            Mutibs('0b0011101110')
            >>> Mutibs('0x1f2e3f').replaced('0x0f', '0x00', mask='0x0f', byte_aligned=True)
            Mutibs('0x002e00')
        """

    def insert(self, pos: int, bs: _TibsLike, /) -> None:
        """Insert bits at position pos.

        Clips to start or end if insert position is out of range.

        :param int pos: The bit position to insert at.
        :param object bs: The bits to insert. This can be anything promotable to ``Tibs``.
        :return: None

        .. code-block:: pycon

            >>> a = Mutibs('0b1011')
            >>> a.insert(2, '0b00')
            >>> a
            Mutibs('0b100011')
        """

    def inserted(self, pos: int, bs: _TibsLike, /) -> Mutibs:
        """Insert bits at position pos and return a new Mutibs.

        This is the non-inplace version of :meth:`insert`.

        :param int pos: The bit position to insert at. Clips to the start or end if out of range.
        :param object bs: The bits to insert. This can be anything promotable to ``Tibs``.
        :return: A new Mutibs.

        .. code-block:: pycon

            >>> Mutibs('0b1011').inserted(2, '0b00')
            Mutibs('0b100011')
        """

    def __ilshift__(self, n: int) -> Mutibs: ...

    def __irshift__(self, n: int) -> Mutibs: ...

    def __bytes__(self) -> bytes:
        """Return the Mutibs as a bytes object.

        :return: The bytes representation.
        :raises ValueError: if the length is not a multiple of 8.
        """

    def __mul__(self, n: int) -> Mutibs: ...

    def __rmul__(self, n: int) -> Mutibs: ...

    __iter__: ClassVar[None]

    def __iand__(self, bs: _TibsLike) -> Mutibs: ...

    def __ior__(self, bs: _TibsLike) -> Mutibs: ...

    def __ixor__(self, bs: _TibsLike) -> Mutibs: ...

    def __imul__(self, n: int) -> Mutibs: ...
