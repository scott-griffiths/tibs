.. currentmodule:: tibs

Byte and Bit Order
------------------

Binary formats often need two separate ordering conventions:

* **Byte order** says how whole bytes are arranged when several bytes represent
  one value.
* **Bit order** says how bit labels are assigned within each byte.

These ideas are easy to mix up, but they answer different questions. Byte
order is about byte-sized pieces of a larger value. Bit order is about labels
inside each byte.


Why ordering terms exist
^^^^^^^^^^^^^^^^^^^^^^^^

Bytes have a natural order in memory, files and network packets. If a file
contains two bytes, one byte comes first and the other comes second.

Integers and floats also have significance. In the two byte hexadecimal value ``0x0005``,
``0x00`` is the more significant byte and ``0x05`` is the less significant byte.
A format must choose which of those bytes is stored first.

If you are not dealing only with whole bytes, you also need to label the bits in a byte.
One specification might call the leftmost bit "bit 0"; another might call the rightmost bit "bit 0".
The stored byte can be identical, but the labels used to describe fields are different.


Byte order
^^^^^^^^^^

Byte order is also called endianness. It only matters when a value is a whole number of bytes and more
than one byte.

For the 16-bit value ``0x0005`` the two bytes are ``0x00`` and ``0x05``.
Big-endian order stores the most significant byte first::

    value:          0x0005
    stored bytes:   00                 05
                    more significant   less significant

Little-endian order stores the least significant byte first::

    value:          0x0005
    stored bytes:   05                 00
                    less significant   more significant

The bytes are not changed. The interpretation of the byte sequence is what
changes::

    >>> Tibs.from_hex('0005').u
    5
    >>> Tibs.from_hex('0500').le.u
    5

In both cases the interpreted integer is ``0x0005``.
The first example has the bytes in big-endian order and uses the ``u`` property to interpret as an unsigned integer.

The second example creates the ``Tibs`` from a little-endian order. It then uses the ``le`` property to create a little-endian
view on the data before using ``u`` to get the unsigned integer interpretation.

For a one-byte value there is no byte order question. For a non-whole-byte value,
there are no complete bytes to reorder.


Bit order
^^^^^^^^^

Bit order is a labelling convention. It says which physical bit in a byte gets
which label.

The default convention in Python-like bit strings is MSB0: label 0 means the
most significant bit of the byte, which is drawn on the left::

    byte bits:   0 0 0 1 0 0 1 0
    MSB0 label:  0 1 2 3 4 5 6 7

Some hardware manuals and protocol specifications use LSB0 instead: label 0
means the least significant bit of the byte, which is drawn on the right::

    byte bits:   0 0 0 1 0 0 1 0
    LSB0 label:  7 6 5 4 3 2 1 0

The byte is still ``00010010``. Only the labels have changed.

This matters when a specification says something like "bits 0..3". With MSB0
labels that means the left-hand nibble. With LSB0 labels that means the
right-hand nibble::

    >>> t = Tibs('0b00010010')
    >>> t.field(0, 3).bin
    '0001'
    >>> t.lsb0.field(0, 3).bin
    '0010'

Bit order is not the same as reversing the byte. It is a way to interpret field
labels from a specification.


How they combine
^^^^^^^^^^^^^^^^

Byte order and bit order can appear together. This is common in hardware
registers.

A register might be stored little-endian, so the lowest-addressed byte is the
least significant byte. The same specification might label bits LSB0, so bit
label 0 is the least significant bit of that byte.

For example, a 32-bit register value may be stored as four bytes::

    stored bytes:  23 a1 12 34
    byte order:    little-endian
    bit labels:    LSB0 within each byte

The byte order tells you how to interpret whole-byte fields. The bit order tells
you how to find fields described by bit labels.

With those rules, the stored bytes and field labels line up like this::

    stored byte:       2 3         a 1         1 2         3 4
    stored bits:    0010 0011   1010 0001   0001 0010   0011 0100
    LSB0 labels:    7  ..   0   15  ..  8   23 ..  16   31 ..  24

    field(31, 16):                         [0001 0010] [0011 0100]
    field(15, 12):             [1010]

``field(31, 16)`` selects the source bytes ``12 34``.
``field(15, 12)`` selects labels 15 through 12, which are the high nibble
``1010`` of the stored byte ``a1``.

Many standards draw the same 32-bit register in significance order instead of
storage order. That view puts the most significant byte on the left and the
least significant byte on the right::

    register view:     34          12          a1          23
    register bits:  0011 0100   0001 0010   1010 0001   0010 0011
    bit labels:     31 ..  24   23 ..  16   15  ..  8   7  ..   0

This is the same data, not a different stored byte sequence. It is a common
reason for LSB0 labels: when the register is drawn this way, field labels are
contiguous across the page, while bit label 0 is still the least significant
bit of the lowest-addressed byte. In that diagram, ``field(31, 16)`` visibly
covers the register bytes ``34 12``; in stored order those bytes are still
``12 34``.

Note that a specification can be big-endian with MSB0
labels, little-endian with LSB0 labels, or another combination. It's important
to read the specification carefully to identify both conventions.


How this maps to tibs
^^^^^^^^^^^^^^^^^^^^^

Normal ``Tibs`` and ``Mutibs`` indexing is always source-order.

For ``Tibs`` and ``Mutibs`` instances, element access and slicing using ``[]`` are
always MSB0, and it follows the usual Python conventions for slicing, indexing and extended slices.
Index ``0`` is the first stored bit, slices run left to right.

For ``View`` and ``MutableView`` instance the ``field`` method is used for access instead of ``[]``.
This has quite different semantics - the start and end points can't be negative, and they are inclusive.
This means that ``.field(a, b)`` is the same as ``field(b, a)``.

Use views when a specification needs a different interpretation:

* ``.be`` and ``.le`` choose big-endian or little-endian byte interpretation.
* ``.msb0`` and ``.lsb0`` choose how field labels are mapped within bytes.
* ``field(a, b)`` selects bits by inclusive specification labels.

For example::

    >>> header = Tibs('0x23a11234').lsb0.le
    >>> header.field(31, 16).hex
    '3412'
    >>> header.field(31, 16).u
    13330
    >>> header.field(15, 12).bin
    '1010'

``field(31, 16)`` selects the stored bytes ``12 34``. Since the field is
interpreted little-endian, ``0x12`` is the low byte and ``0x34`` is the high
byte. That is why ``.hex`` returns ``'3412'`` and ``.u`` returns ``13330``::

    0x12 + (0x34 << 8) = 0x3412 = 13330

The view does not mutate the original data. It supplies the interpretation
needed for labelled fields and whole-byte values. For practical details and
mutable views, see :doc:`views`.
