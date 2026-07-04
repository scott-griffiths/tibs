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

Integers and floats also have significance. In the hexadecimal value ``0x1234``,
``0x12`` is the more significant byte and ``0x34`` is the less significant byte.
A format must choose which of those bytes is stored first.

Specifications may also label bits in a byte. One specification might call the
leftmost bit "bit 0"; another might call the rightmost bit "bit 0". The stored
byte can be identical, but the labels used to describe fields are different.


Byte order
^^^^^^^^^^

Byte order is also called endianness. It only matters when a values are a whole number of bytes and more
than one byte.

For the 16-bit value ``0x1234`` the two bytes are ``0x12`` and ``0x34``.
Big-endian order stores the most significant byte first::

    value:          0x1234
    stored bytes:   12                 34
                    more significant   less significant

Little-endian order stores the least significant byte first::

    value:          0x1234
    stored bytes:   34                 12
                    less significant   more significant

The bytes are not changed. The interpretation of the byte sequence is what
changes::

    >>> Tibs.from_hex('1234').u
    4660
    >>> Tibs.from_hex('3412').le.u
    4660

In both cases the interpreted integer is ``0x1234``. The first example stores
the bytes in big-endian order. The second stores them in little-endian order and
uses a little-endian interpretation.

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

Those two rules are independent. A specification can be big-endian with MSB0
labels, little-endian with LSB0 labels, or another combination. The important
step is to read the specification carefully and identify both conventions.


How this maps to tibs
^^^^^^^^^^^^^^^^^^^^^

Normal ``Tibs`` and ``Mutibs`` indexing is always source-order. Index ``0`` is
the first stored bit, slices run left to right, and the underlying data is not
rearranged just because a different interpretation is useful.

Use views when a specification needs a different interpretation:

* ``.be`` and ``.le`` choose big-endian or little-endian byte interpretation.
* ``.msb0`` and ``.lsb0`` choose how field labels are mapped within bytes.
* ``field(a, b)`` selects bits by inclusive specification labels.

For example::

    >>> header = Tibs('0x23a11234').lsb0.le
    >>> header.field(31, 16).u
    13330
    >>> header.field(15, 12).bin
    '1010'

The view does not mutate the original data. It supplies the interpretation
needed for labelled fields and whole-byte values. For practical details and
mutable views, see :doc:`views`.
