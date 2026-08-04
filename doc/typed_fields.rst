.. currentmodule:: tibs

Typed fields
------------

Bits can be read and written as typed values — unsigned and
signed integers, floats, and string or byte representations — of any bit length,
without hand-rolling shifts and masks. Nothing is copied into another type: the
value is decoded from, or encoded into, the same underlying bits.

This chapter covers the default (big-endian, MSB0) interpretations. When a format
uses little-endian byte order or LSB0 bit labels, wrap the value in a
:doc:`view <views>`; to render a value as text, see :doc:`formatting`.

Every method here locates its field with a ``start`` and an ``end``. When the
fields come one after another instead, a :doc:`Reader <reading>` keeps the
position for you.


Numbers in and out
^^^^^^^^^^^^^^^^^^

Three constructors build a value from a number to a given length, and three
matching interpretations read it back:

* :meth:`Tibs.from_u` / :meth:`Tibs.to_u` / :attr:`Tibs.u` — unsigned integer.
* :meth:`Tibs.from_i` / :meth:`Tibs.to_i` / :attr:`Tibs.i` — signed (two's complement) integer.
* :meth:`Tibs.from_f` / :meth:`Tibs.to_f` / :attr:`Tibs.f` — IEEE float; length must be 16, 32 or 64.

Integers can be any positive number of bits long; floats must be 16, 32 or 64::

    # From a signed integer. The length can be any positive number of bits.
    e = Tibs.from_i(-384, 20)

    # From an unsigned integer. For whole-byte lengths a byte order can be used.
    f = Tibs.from_u(3, 32, byte_order=ByteOrder.Little)

    # Floating point values need to have a length of 16, 32 or 64.
    g = Tibs.from_f(-0.125, 16)

.. note::
    These `f` floating point types are IEEE 754 only. Other floating point types
    like bfloat16 are specified with an explicity Dtype, for example `Tibs.from_value("bf16", -0.125)`.

The ``to_`` methods accept optional ``start`` and ``end`` bit positions when you
only want to interpret part of the data. With no parameters, the ``u``, ``i`` and
``f`` properties are a convenient alias, so instead of ``t.to_u()`` you can write
``t.u`` for the whole value. On ``Tibs`` these properties are read-only.

Unlike the lossless representations below, these interpretations can have a
many-to-one relationship: there are many ways for a ``Tibs`` to be constructed
from the unsigned integer 3::

    >>> u1 = Tibs.from_u(3, 5)   # binary 00011
    >>> u2 = Tibs.from_u(3, 16)  # binary 00000000_00000011
    >>> u3 = Tibs.from_u(3, 16, ByteOrder.Little)  # binary 00000011_00000000

These are three different ``Tibs``, but they can all have equal interpretations::

    >>> set([u1, u2, u3])
    {Tibs('0b00011'), Tibs('0x0003'), Tibs('0x0300')}
    >>> set([u1.u, u2.u, u3.le.u])
    {3}

For the value stored in ``u3`` a little-endian :class:`View` was used — see
:doc:`views`.


Lossless representations
^^^^^^^^^^^^^^^^^^^^^^^^

A second family of interpretations returns a lossless representation of the exact
bit sequence — as a string, ``bytes``, or list of bools. These start with ``to_``
and, like the numeric ones, have property aliases for the whole value:

* :meth:`Tibs.to_bin` / :attr:`Tibs.bin` — a string of ``0`` and ``1`` characters. Always available.
* :meth:`Tibs.to_oct` / :attr:`Tibs.oct` — an octal string. Length must be a multiple of 3.
* :meth:`Tibs.to_hex` / :attr:`Tibs.hex` — a hexadecimal string. Length must be a multiple of 4.
* :meth:`Tibs.to_bytes` / :attr:`Tibs.bytes` — a ``bytes`` object. Length must be a multiple of 8.
* :meth:`Tibs.to_bools` — a list of ``bool`` values. Always available, and much faster than iterating bit by bit.

Because they are lossless, you can always reconstruct the original value from one
of these representations — there is a 1:1 relationship, so ``t == Tibs.from_bin(t.bin)``
is always true.

Several of them need the length to be a correct multiple, for example ``bytes``
needs a multiple of 8::

    >>> t = Tibs('0x4145c')
    >>> len(t)
    20
    >>> t.bin
    '01000001010001011100'
    >>> t.bytes
    Traceback (most recent call last):
    ...
    ValueError: Cannot interpret as bytes - length of 20 is not a multiple of 8 bits.

To convert to a ``bytes`` object we need to change the length, for example by
extending it with four ``0`` bits::

    >>> (t + '0x0').bytes
    b'AE\xc0'

This is common enough that :meth:`Tibs.to_padded_bytes` is provided, which appends
0 to 7 zero bits on the right before converting::

    >>> t.to_padded_bytes()
    b'AE\xc0'

If you're rendering rather than converting, these representations are also
available through Python's format mini-language, so ``f"{t:#x}"`` and ``f"{t:_.8b}"``
do what you'd expect. See :ref:`formatting` for the details.

A ``Tibs`` also exports the buffer protocol, so where you want to hand the bytes
to something else rather than own a copy of them you can pass it directly to
anything that takes a bytes-like object::

    >>> t = Tibs('0xff00')
    >>> bytes(memoryview(t))
    b'\xff\x00'

That covers ``memoryview``, ``array.array``, ``numpy.frombuffer``, and writing to
a socket or a file, none of which then need a copy of the data. The buffer is
read-only and always covers whole bytes, so for a length that isn't a multiple of
8 the final byte includes the padding bits, which are not masked to zero. It also
needs the underlying storage to start on a byte boundary, which isn't the case
for every ``Tibs`` — one made by slicing at a bit offset, for example::

    >>> memoryview(Tibs('0xffff')[3:])
    Traceback (most recent call last):
    ...
    BufferError: Cannot export a buffer for this Tibs: its data does not start on a byte boundary. Use to_bytes() or to_padded_bytes() to get an owned copy instead.

Use :meth:`Tibs.to_bytes` or :meth:`Tibs.to_padded_bytes` for an owned copy when
that happens.

``Mutibs`` does not export a buffer. Its storage is reallocated and shifted as the
container is edited, so a borrowed view of it could not be kept valid.


Fixed-width and structured values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When every item uses the same fixed-width encoding, scalar :class:`Dtype`
strings make the intent explicit and avoid writing a construction loop
yourself. The most common forms are unsigned integers such as ``"u8"`` or
``"u12"``, signed integers such as ``"i16"``, floats such as ``"f32"``, and
string or byte representations such as ``"hex16"`` and ``"bytes32"``. Use
``"bool"`` for a single Python boolean bit, or ``"bitsN"`` when each value is
itself a fixed-size bit sequence decoded as :class:`Tibs`.

Use :meth:`Tibs.from_value` for one value, or :meth:`Tibs.from_values` for an
iterable of values::

    >>> Tibs.from_value("u8", 15)
    Tibs('0x0f')
    >>> samples = Tibs.from_values("u12", [0, 103, 2048, 4095])
    >>> samples.hex
    '000067800fff'

The matching interpretation methods decode values back from a bit sequence::

    >>> samples.to_values("u12")
    [0, 103, 2048, 4095]

See :doc:`example_sensor_samples` for packing and unpacking a stream of samples,
and :doc:`example_construct` for driving a whole header from a table of dtypes.

An array dtype describes a fixed number of values with the same dtype, while a
tuple dtype combines fields with different dtypes::

    >>> flags = Tibs.from_value("[bool; 4]", [True, False, True, True])
    >>> flags.bin
    '1011'
    >>> header = Tibs.from_value("(u8, u16_le)", (1, 0x0203))
    >>> header.hex
    '010302'
    >>> header.to_value("(u8, u16_le)")
    (1, 515)

Under the hood, a dtype string is parsed into one of three concrete, immutable
classes: :class:`DtypeSingle` for a scalar value, :class:`DtypeArray` for a
fixed-size repeat of another dtype, and :class:`DtypeTuple` for a fixed
sequence of possibly different dtypes. The string forms above are usually all
you need, but the classes are there directly when you want to build or inspect
a dtype programmatically::

    >>> Dtype("u8")
    DtypeSingle('u8')
    >>> Dtype("[u8; 4]")
    DtypeArray('[u8; 4]')
    >>> Dtype("(u8, bool)")
    DtypeTuple('(u8, bool)')

Arrays and tuples can be nested. Their decoded values are Python tuples, so the
shape of the dtype remains visible in the result::

    >>> dtype = Dtype("[(u4, bool); 2]")
    >>> value = ((10, True), (3, False))
    >>> dtype.unpack(dtype.pack(value))
    ((10, True), (3, False))

The singular methods operate on one complete structured value. The plural
methods repeat the complete dtype::

    >>> records = [(1, 0x0203), (4, 0x0506)]
    >>> packed = Tibs.from_values("(u8, u16_le)", records)
    >>> packed.to_values("(u8, u16_le)")
    [(1, 515), (4, 1286)]

For whole-byte numeric values, append ``_le`` or ``_be`` to the dtype string
when byte order matters. These suffixes mean little-endian and big-endian byte
order respectively::

    >>> Tibs.from_values("u16_le", [0x1234, 0xabcd]).hex
    '3412cdab'

See :doc:`dtype` for the full dtype grammar, hierarchy and programmatic
constructors.


Writing typed values
^^^^^^^^^^^^^^^^^^^^

On a ``Mutibs`` you can also write a typed value back into the existing bits. The
``write_u``, ``write_i`` and ``write_f`` methods replace the current bits with a
new value while preserving the existing bit length::

    >>> m = Mutibs.from_zeros(8)
    >>> m.write_u(15)
    >>> m
    Mutibs('0x0f')
    >>> len(m)
    8

The ``u``, ``i`` and ``f`` properties are settable shortcuts for the same default
interpretations::

    >>> m.u = 42
    >>> m.u
    42
    >>> m.i = -1
    >>> m
    Mutibs('0xff')

The value must fit in the current length. Floating-point assignment uses the
current length too, so it is only available for 16, 32 and 64-bit ``Mutibs``.

For little-endian or LSB0 interpretations, assign through a mutable view instead::

    >>> m = Mutibs.from_u(99, 16, ByteOrder.Little)
    >>> m.le.u
    99
    >>> m.le.u = 45
    >>> m.le.u
    45

A view isn't limited to those interpretations either. :meth:`View.to_value` and
:meth:`MutableView.write_value` take any dtype and apply the byte order and bit
order of the view to it::

    >>> m.le.write_value("(u8, u8)", (1, 2))
    >>> m
    Mutibs('0x0201')

Views are covered next, in :doc:`views`.
