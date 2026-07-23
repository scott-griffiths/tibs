.. currentmodule:: tibs

Views
-----

Views wrap a ``Tibs`` or a ``Mutibs`` to allow the bits inside it to be
interpreted in a different way. This allows different byte orders to be used,
as well as different bit numbering methods when interpreting the data.

Both ``Tibs`` and ``Mutibs`` act like a Python container for bits - indexing and
slicing always have the usual meaning, with indices running from left to right.
Interpretations of the bit data also have the usual meanings in Python::

    >>> int.from_bytes(b'xyz')
    7895418
    >>> Tibs(b'xyz').u
    7895418

    >>> bytes.fromhex('abcd')
    b'\xab\xcd'
    >>> Tibs.from_hex('abcd').bytes
    b'\xab\xcd'


It's not uncommon though for a file format or protocol to specify a different byte order or bit numbering
convention from the Python default. For these cases we can create a ``View`` or a ``MutableView``.
For a slower background explanation of byte order and bit labels, see :doc:`byte_and_bit_order`.

For example, let's create a four-byte ``Tibs`` and interpret it as an unsigned int::

    >>> t = Tibs('0x01000000')
    >>> t.to_u()
    16777216

This is the byte-wise and bit-wise big-endian interpretation, which corresponds to
the standard Python indexing convention where bit zero is the most significant bit.

As this is a whole number of bytes long we can also consider its byte order.
A little-endian interpretation essentially reverses the byte order, so the least
significant byte is the first one. Without changing the data at all, we can create
a ``View`` which wraps it, and then use the interpretation on that ``View`` ::

    >>> v = t.view(ByteOrder.Little)
    >>> v
    View(Tibs('0x01000000'), ByteOrder.Little, BitOrder.Msb0)
    >>> v.to_hex()
    '00000001'
    >>> v.to_u()
    1

This is all quite a lot of typing, so a more convenient way to create a view from a
``Tibs`` or ``Mutibs`` is to use properties. The available view properties are:

* :attr:`Tibs.le` / :attr:`Mutibs.le`: little-endian byte order.
* :attr:`Tibs.be` / :attr:`Mutibs.be`: big-endian byte order.
* :attr:`Tibs.lsb0` / :attr:`Mutibs.lsb0`: LSB0 bit numbering within each byte.
* :attr:`Tibs.msb0` / :attr:`Mutibs.msb0`: MSB0 bit numbering within each byte.

These can be combined. For example ``t.lsb0.le`` means that bit labels are LSB0,
and whole-byte values should be interpreted as little-endian.

For the example above, the :attr:`Tibs.u` property makes the little-endian interpretation more concise::

    >>> t.view(ByteOrder.Little).to_u()
    1
    >>> t.le.u  # Same thing, but using properties
    1


Views and data
^^^^^^^^^^^^^^

A view created from a ``Tibs`` is immutable: it keeps the same immutable data and
adds interpretation settings. A :class:`MutableView` created from a ``Mutibs``
keeps a live reference to the ``Mutibs``, so later changes to the ``Mutibs`` are
reflected in the view.

The direct :class:`View` constructor is intentionally stricter than ``Tibs``.
It accepts a ``Tibs`` or ``Mutibs`` object, but not other types that could be
promoted.

Passing a ``Mutibs`` to the direct :class:`View` constructor still creates an
immutable snapshot. Use :class:`MutableView` or the ``Mutibs`` view helpers when
you want live mutable behavior.

Views are intended as interpretation wrappers rather than as another way
to construct binary data.

Byte order
^^^^^^^^^^

Byte order only applies to whole-byte values. When you construct a value from an
integer or float you can choose the byte order used to store it::

    >>> t = Tibs.from_u(666, 16, byte_order=ByteOrder.Little)
    >>> t
    Tibs('0x9a02')

If we read the little-endian bytes with the default interpretation, we don't get
the value we created it with::

    >>> t.u
    39426

The little-endian view gives the intended interpretation::

    >>> t.le.u
    666

This works for floats and bytes too::

    >>> x = Tibs.from_f(1984.3, 64, byte_order=ByteOrder.Little)
    >>> x.f
    4.667261455589845e-62
    >>> x.le.f
    1984.3

The default byte order is ``ByteOrder.Unspecified``. For whole-byte data this is
the same interpretation as ``ByteOrder.Big``, but it can also be used for
non-whole-byte data. The explicit ``be`` and ``le`` views require a whole number
of bytes.

Bit order
^^^^^^^^^

The ``msb0`` and ``lsb0`` views control how bit labels are interpreted within
each byte.

``msb0`` is the default convention used by normal indexing and slicing, where
the most significant bit of the byte (the leftmost bit) is bit 0, with the rightmost
bit being bit 7. For ``lsb0``, which is used in some specifications, the least significant
bit of the byte (the rightmost bit) is bit 0, and the leftmost bit is bit 7.

The easiest way to see the difference is to extract labelled fields from one
byte::

    >>> t = Tibs('0x12')
    >>> t.bin
    '00010010'
    >>> t.field(0, 3).bin
    '0001'
    >>> t.lsb0.field(0, 3).bin
    '0010'
    >>> t.lsb0.field(4, 7).bin
    '0001'

The plain :meth:`~Tibs.field` call uses MSB0 labels, so labels ``0..3`` select
the left-hand nibble. The ``lsb0`` view uses LSB0 labels, so labels ``0..3``
select the right-hand nibble. In both cases the returned field is displayed as a
normal value, with the most significant bit on the left.

The same value-display rule applies to whole views. Representation methods such
as :attr:`~View.bin`, :attr:`~View.hex` and :meth:`~View.to_tibs` show the value
denoted by the view, not a physical traversal of the source bits. For a whole
view, this is the same ordering you would get from
``view.field(0, len(view) - 1)``::

    >>> word = Tibs('0x0100')
    >>> word.bin
    '0000000100000000'
    >>> word.lsb0.bin
    '0000000000000001'
    >>> word.lsb0.field(0, len(word) - 1).bin
    '0000000000000001'

Format specs follow the same rule, so a view can be dropped straight into an
f-string and the layout is applied for you (see :ref:`formatting`)::

    >>> f"{word.le:#x}"
    '0x0001'
    >>> instruction = Tibs.from_bytes(bytes.fromhex("07 01 00 00 44 33 22 11")).lsb0.le
    >>> f"{instruction.field(11, 8):u}"
    '1'
    >>> f"{instruction.field(63, 32):#x}"
    '0x11223344'

The one exception is an empty format spec, which still gives you the ``repr`` of the
view rather than a value, because that's what ``str()`` of a view has always done.

For full-width multi-byte values, ``lsb0`` can therefore look like a
little-endian value display: bit label 0 is the least significant bit of the
whole value, so it appears on the RHS. Use the original ``Tibs`` or ``Mutibs``
when you want ordinary source-order indexing and slicing; use ``field()`` when
you want specification labels.

The physical storage is still unchanged unless you write through a
:class:`MutableView`. Combining ``lsb0`` and ``le`` is common for register and
packet specifications that number bits from the least significant bit and store
multi-byte fields little-endian. Setting bit label 0 in that view makes it
appear on the RHS of the interpreted value, even though the source bit lives in
the first stored byte::

    >>> m = Mutibs.from_zeros(32)
    >>> v = m.lsb0.le
    >>> v.field(0, 0).u = 1
    >>> v.bin
    '00000000000000000000000000000001'
    >>> m.hex
    '01000000'


For ordinary Python indexing and slicing, use the ``Tibs`` or ``Mutibs`` directly.
Views don't provide their own slicing interface, as that would make it too easy
to confuse normal Python slices with specification field labels.


Mutable views
^^^^^^^^^^^^^

A :class:`MutableView` can also write interpreted values back into the source
``Mutibs`` without changing its length. The view supplies the layout::

    >>> m = Mutibs.from_u(99, 16, byte_order=ByteOrder.Little)
    >>> m.le.u
    99
    >>> m.le.write_u(45)
    >>> m.le.u
    45
    >>> m
    Mutibs('0x2d00')

The ``u``, ``i`` and ``f`` properties are settable too::

    >>> m.le.u = 123
    >>> m.le.u
    123

For default layout, the whole ``Mutibs`` also has ``write_u``, ``write_i`` and
``write_f`` methods and settable ``u``, ``i`` and ``f`` properties. Use a mutable
view when byte order or bit order matters.

The representation properties ``bin``, ``oct``, ``hex`` and ``bytes`` are also
settable on a :class:`MutableView`, but the assigned value must have the same bit
length as the view::

    >>> m = Mutibs('0x0000')
    >>> m.le.hex = 'abcd'
    >>> m.le.hex
    'abcd'
    >>> m
    Mutibs('0xcdab')

If you need to change the length, assign to the source ``Mutibs`` or use slice
assignment. Views are fixed mappings onto their current source bits.

Fields
^^^^^^

Some standards describe fields using inclusive bit labels, such as ``31:28``.
In Python slicing that would usually be written as ``[28:32]``, but for LSB0
formats that still doesn't describe the right physical bits. The important detail
is that the specification is giving bit labels, not Python slice positions.

For this case we can use :meth:`Tibs.field`, which uses the default MSB0
labels, or :meth:`View.field` for a different view such as ``lsb0``. The two
endpoints are inclusive and can be given in either order. Both endpoints must be
zero or positive bit labels::

    >>> t = Tibs('0x23a11234')
    >>> t.field(0, 7).hex
    '23'
    >>> t.lsb0.field(31, 28).u
    3
    >>> t.lsb0.field(28, 31).u
    3

When the source is mutable, :meth:`Mutibs.field` and :meth:`MutableView.field`
return live ``MutableView`` objects over the selected bits. Assigning through
that field writes back to the original ``Mutibs``::

    >>> m = Mutibs('0x23a11234')
    >>> m.field(0, 7).hex = '42'
    >>> m
    Mutibs('0x42a11234')
    >>> m.lsb0.le.field(31, 16).u
    13330
    >>> m.lsb0.le.field(31, 16).u = 0x5678
    >>> m.lsb0.le.field(31, 16).u
    22136
    >>> m
    Mutibs('0x42a17856')

For low-level reconstruction from physical source bit positions, use
:meth:`View.from_indices` or :meth:`MutableView.from_indices`.
The ``indices`` argument may be a ``range`` or any iterable of integers.
For ordinary fields described by a specification, prefer :meth:`View.field` or
:meth:`MutableView.field`.

As a more complete example, suppose a format starts with a 32-bit little-endian
header. The first byte in the file contains bits 7:0, the next byte contains bits
15:8, and so on, but the standard draws the complete word with bit 31 on the
left and bit 0 on the right::

    31                                  16 15      12 11                 0
    +--------------------------------------+----------+-------------------+
    | message_id                           | flags    | payload_length    |
    +--------------------------------------+----------+-------------------+

If the four bytes from the file are ``23 a1 12 34``, this header can be decoded
directly from the specification labels. We use both ``lsb0`` and ``le`` because
the standard uses LSB0 bit labels and the whole-byte values are little-endian::

    >>> header = Tibs('0x23a11234').lsb0.le
    >>> header.field(31, 16).u  # message_id
    13330
    >>> header.field(15, 12).bin  # flags
    '1010'
    >>> header.field(11, 0).u   # payload_length
    291

Byte order and field extraction are separate ideas. ``field()`` uses the current
bit order to find the labelled bits, then returns those bits in field-value
order. For LSB0 labels, label 0 is the least-significant bit of the field, so
the extracted value is not bit-reversed.

The ``message_id`` field is 16 bits long, so it keeps the little-endian byte
order from the header. The selected field bytes are ``12 34``, and the integer
interpretation is ``0x3412``::

    >>> header.field(31, 16)
    View(Tibs('0x1234'), ByteOrder.Little, BitOrder.Msb0)
    >>> header.field(31, 16).u
    13330

If the extracted field is not a whole number of bytes, byte order no longer has a
meaning and is dropped::

    >>> header.field(11, 0)
    View(Tibs('0x123'), ByteOrder.Unspecified, BitOrder.Msb0)
    >>> header.field(11, 0).u
    291

Scattered fields
^^^^^^^^^^^^^^^^

:meth:`~Tibs.field` reads a *contiguous* run of bits. Some fields aren't
contiguous — an instruction immediate split across the word, flags interleaved
with data — and for those a mask picks out the bits that belong to the field.

:meth:`Tibs.extract` reads the bits selected by a mask and packs them together.
It's the bit-level version of the PEXT instruction::

    >>> word = Tibs('0b11010110')
    >>> word.extract('0b10110000')   # the bits at positions 0, 2 and 3
    Tibs('0b101')

The mask must be the same length as the container, and the result has one bit for
each set bit of the mask.

.. _deposit:

The inverse, writing such a field, is :meth:`Mutibs.deposit`: the bits of the
value are placed at the positions the mask selects, and the rest of the container
is left alone. It is the bit-level version of the PDEP instruction::

    >>> m = Mutibs('0b11010110')
    >>> m.deposit('0b111', '0b10110000')   # write 3 bits into positions 0, 2, 3
    >>> m.bin
    '11110110'

The value must be exactly ``mask.count()`` bits long. The non-mutating
:meth:`Tibs.deposited` returns a new container instead of writing in place. See
:doc:`example_scattered_field` for a worked register example.

Materializing a view
^^^^^^^^^^^^^^^^^^^^

Most of the time you can use a view directly with the same interpretation
properties used by ``Tibs``::

    >>> t = Tibs('0x0100')
    >>> t.le.hex
    '0001'
    >>> t.le.bytes
    b'\x00\x01'

If you need the viewed bits as a new object, use :meth:`View.to_tibs` or
:meth:`View.to_mutibs` (also available on :class:`MutableView`)::

    >>> t.le.to_tibs()
    Tibs('0x0001')

This makes any byte order or bit order transformation explicit before you go back
to normal indexing, slicing or mutation.
