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


.. _four-conventions:

The four combinations
^^^^^^^^^^^^^^^^^^^^^

Two choices of byte order and two of bit order make four conventions. Here are
the labels each one gives the two stored bytes ``12 34``::

    stored byte:       0x12                 0x34
    stored bits:    0 0 0 1 0 0 1 0      0 0 1 1 0 1 0 0

    BE + MSB0:      0 1 2 3 4 5 6 7      8 9 ...     15    value 0x1234
    BE + LSB0:     15 14 ...       8     7 6 ...      0    value 0x1234
    LE + LSB0:      7 6 ...        0    15 14 ...     8    value 0x3412
    LE + MSB0:      0 1 2 3 4 5 6 7      8 9 ...     15    value 0x3412

LSB0 numbers the bits by significance, so in both LSB0 rows label ``k`` is worth
``2**k`` in the value, and the rows differ only in which stored byte holds the
value's least significant end. MSB0 numbers the bits in the order they are
stored. For big-endian data that is also significance order, counted from the
other end. For little-endian data it isn't: labels 0 to 7 are bits 7 to 0 of
the value and labels 8 to 15 are bits 15 to 8, so a field that crosses from one
byte to the next is not one piece of the value.

In tibs each combination is a view, and labels ``3..0`` and ``15..8`` pick out
different bits in each::

    >>> t = Tibs('0x1234')
    >>> t.be.field(0, 3).hex, t.be.field(8, 15).hex          # BE + MSB0
    ('1', '34')
    >>> t.lsb0.be.field(3, 0).hex, t.lsb0.be.field(15, 8).hex  # BE + LSB0
    ('4', '12')
    >>> t.lsb0.le.field(3, 0).hex, t.lsb0.le.field(15, 8).hex  # LE + LSB0
    ('2', '34')
    >>> t.le.field(0, 3).hex, t.le.field(8, 15).hex          # LE + MSB0
    ('1', '34')

Where you're likely to meet each one:

* **BE + MSB0** is how IETF RFCs draw packets, with bit 0 on the left, and how
  IBM and PowerPC documentation and the MPEG and H.264 bitstream syntaxes number
  bits. It's also ordinary Python and ``Tibs`` indexing, so it needs no view
  unless you want ``field``.
* **LE + LSB0** is used by Intel and ARM manuals, USB descriptors, eBPF and
  DEFLATE, and is how C compilers lay out bit-fields on those machines. It's
  probably the most common convention in hardware documentation.
* **BE + LSB0** is used by Motorola 68k manuals and many other big-endian
  processor manuals, and by most SPI and I²C sensor datasheets, which describe
  a 16-bit register as bits 15 to 0 and send its most significant byte first.
* **LE + MSB0** is rare as a deliberate choice. It mostly appears where a
  specification draws each byte with bit 0 on the left and also stores
  multi-byte integers little-endian.


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
* ``.msb0`` and ``.lsb0`` choose how field labels are mapped.
* ``field(a, b)`` selects bits by inclusive specification labels.

MSB0 labels count through the bits in the order they are stored. LSB0 labels
count by significance instead: label ``k`` is bit ``k`` of the value the view
denotes, worth ``2**k``. So with LSB0 the byte order decides where label 0 is:
in the first byte for ``.lsb0.le``, and in the last for ``.lsb0.be``. Within a
single byte there's nothing to choose, so plain ``.lsb0`` is enough. For
anything longer it is an error to use an ``lsb0`` view without saying which
byte order the specification uses.

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

A big-endian specification with LSB0 labels, as in Motorola 68k manuals and
most SPI and I²C sensor datasheets, draws a register with bit 31 on the left
and bit 0 on the right, and sends its most significant byte first. Label 0 is
then the least significant bit of the *last* stored byte, and ``.lsb0.be``
reads it::

    >>> register = Tibs('0x12345678').lsb0.be
    >>> register.field(31, 16).hex
    '1234'
    >>> register.field(15, 12).u
    5

The four combinations of byte order and bit order are compared in full, with
where each one is used, in :ref:`four-conventions`.

The view does not mutate the original data. It supplies the interpretation
needed for labelled fields and whole-byte values. For practical details and
mutable views, see :doc:`views`.


.. _lsb0-values:

Values packed least significant bit first
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bit order also decides how a run of values is packed. Most formats pack MSB
first, which is how ``Tibs`` reads by default: each value starts with its most
significant bit, and fills bytes from the left. Plenty of others pack LSB first
instead. Each byte fills from its least significant bit, and each value goes in
least significant bit first, so a 12-bit value takes all of one byte and the
low half of the next. FAT12 cluster maps, the little-endian ``Mono12Packed``
camera format, 1- and 2-bit pixel formats such as Qt's ``MonoLSB``, and 4-bit
tensor formats all work like this.

In little-endian LSB0 labels the rule is simple: an ``n``-bit value at label
``L`` is ``lsb0.le.field(L + n - 1, L)``. One value can be read that way through a view, but for
many of them the methods that pack and unpack values take a ``bit_order``:
``from_value``, ``from_values``, ``to_value`` and ``to_values`` on ``Tibs`` and
``Mutibs``, the :class:`Dtype` pack and unpack methods, and :class:`Reader`.
``pp`` takes one too, to show them (see :ref:`formatting`)::

    >>> table = Tibs.from_bytes(bytes.fromhex('21436587a9cb'))
    >>> [hex(x) for x in table.to_values('u12', bit_order=BitOrder.Lsb0)]
    ['0x321', '0x654', '0x987', '0xcba']
    >>> Tibs.from_values('u12', [0x321, 0x654, 0x987, 0xcba], bit_order=BitOrder.Lsb0).hex
    '21436587a9cb'

Here ``start`` and ``end`` are labels too, so reading entry ``k`` of a FAT12 table
doesn't need the whole table decoded first::

    >>> k = 2
    >>> hex(table.to_value('u12', start=12 * k, end=12 * k + 12, bit_order=BitOrder.Lsb0))
    '0x987'

The :class:`Dtype` methods do the same with a ``Dtype`` you already have, and
accept ``bytes`` without making a ``Tibs`` first::

    >>> [hex(x) for x in Dtype('u12').unpack_values(table.bytes, bit_order=BitOrder.Lsb0)]
    ['0x321', '0x654', '0x987', '0xcba']

A view's own ``to_value`` and ``write_value`` don't take a ``bit_order``, and an
``lsb0`` view is not a substitute for one. The two agree on every single value,
but they mean different things by an array or tuple:

* ``bit_order=BitOrder.Lsb0`` reads a *stream* of fields. Each field starts at
  the lowest label not yet used, so the first field takes the least significant
  bits. That's how LSB-first formats lay out pixels, table entries and header
  fields, and how C compilers on little-endian machines lay out bit-fields.
* A view reads *one value*, the one it denotes, and an array or tuple splits
  that value most significant field first, as a register diagram draws it.
  That's the same rule ``bin`` and ``hex`` follow on a view, so
  ``view.to_value(dtype)`` is always ``view.to_tibs().to_value(dtype)``.

For four 2-bit pixels packed LSB first, only the stream gives them in order::

    >>> pixels = Tibs('0xe4')
    >>> pixels.to_value('[u2; 4]', bit_order=BitOrder.Lsb0)
    (0, 1, 2, 3)
    >>> pixels.lsb0.to_value('[u2; 4]')
    (3, 2, 1, 0)

Use a view to pick out fields by their labels, and ``bit_order`` to read values
packed one after another.

A few things follow from the rule:

* Every value kind works, because each value's bits are found first and decoded
  after. Packing LSB first makes a value little-endian, just as packing MSB
  first makes it big-endian, so an LSB-first ``u16`` or ``f32`` that starts on a
  byte boundary is the same as an ordinary ``u16_le`` or ``f32_le``.
* A dtype can still name its own byte order, which then applies to that value
  as it does with MSB-first packing. ``'u16_le'`` is ``'u16'`` spelt out, and
  ``'u16_be'`` reads the first byte in the stream as the most significant, for
  the occasional big-endian field in an LSB-first format::

      >>> Tibs('0x1234').to_value('u16', bit_order=BitOrder.Lsb0) == 0x3412
      True
      >>> Tibs('0x1234').to_value('u16_le', bit_order=BitOrder.Lsb0) == 0x3412
      True
      >>> Tibs('0x1234').to_value('u16_be', bit_order=BitOrder.Lsb0) == 0x1234
      True

* An array or tuple dtype applies the rule to each of its fields in turn, so
  ``'(u12, u4)'`` is a 12-bit value then a 4-bit one, each LSB first.
* The values are read in significance order, just as ``field`` reads them. For
  ``bytes``, ``hex`` and the other text kinds that means the last byte comes
  first: ``Tibs('0x0102').to_value('bytes16', bit_order=BitOrder.Lsb0)`` is
  ``b'\x02\x01'``.
* LSB0 labels only exist in whole bytes, so the bits read must be a whole number
  of bytes long, and the packing methods pad what they produce with zero bits
  up to the next byte, so ``Tibs.from_values('u12', [1, 2, 3],
  bit_order=BitOrder.Lsb0)`` is 40 bits long, not 36. The padding takes the
  highest labels, so it falls at the end, where it does in the formats
  themselves.
