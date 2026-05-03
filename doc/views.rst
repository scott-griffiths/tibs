.. currentmodule:: tibs

Views
-----

A ``Tibs`` stores a sequence of bits. A :class:`View` doesn't change those stored
bits, but changes how they are interpreted.

The most common reason to create a view is that a file format or protocol specifies
values using a byte order or bit numbering convention. In those cases the same
stored bits might need to be read as a little-endian integer, or decoded using
LSB0 field labels from a standard.

Views are usually created with properties on ``Tibs`` and ``Mutibs``::

    >>> t = Tibs('0x01000000')
    >>> t.le.u
    1

The properties are just short-cuts for :meth:`Tibs.view`::

    >>> t.view(byte_order=Endianness.Little).u
    1

The available view properties are:

* :attr:`Tibs.le` / :attr:`Mutibs.le`: little-endian byte order.
* :attr:`Tibs.be` / :attr:`Mutibs.be`: big-endian byte order.
* :attr:`Tibs.lsb0` / :attr:`Mutibs.lsb0`: LSB0 bit numbering within each byte.
* :attr:`Tibs.msb0` / :attr:`Mutibs.msb0`: MSB0 bit numbering within each byte.

These can be combined. For example ``t.lsb0.le`` means that bit labels are LSB0,
and whole-byte values should be interpreted as little-endian.

Views and data
^^^^^^^^^^^^^^

A view created from a ``Tibs`` is cheap: it keeps the same immutable data and adds
interpretation settings. A view created from a ``Mutibs`` stores an immutable
snapshot, so later changes to the ``Mutibs`` won't affect the view::

    >>> m = Mutibs('0x0100')
    >>> v = m.le
    >>> m[0] = True
    >>> v.u
    1
    >>> m
    Mutibs('0x8100')

The direct :class:`View` constructor is intentionally stricter than ``Tibs``.
It accepts a ``Tibs`` or ``Mutibs`` object, but not strings or bytes::

    >>> View(Tibs('0x0100'), Endianness.Little).u
    1

This keeps ``View`` as an interpretation wrapper rather than another way to
construct binary data.

Byte order
^^^^^^^^^^

Byte order only applies to whole-byte values. When you construct a value from an
integer or float you can choose the byte order used to store it::

    >>> Tibs.from_u(1, 32, Endianness.Big)
    Tibs('0x00000001')
    >>> Tibs.from_u(1, 32, Endianness.Little)
    Tibs('0x01000000')

If we read the little-endian bytes with the default interpretation, we don't get
the original value::

    >>> t = Tibs('0x01000000')
    >>> t.u
    16777216

The little-endian view gives the intended interpretation::

    >>> t.le.u
    1

This works for floats and bytes too::

    >>> f = Tibs.from_f(1984.3, 64, Endianness.Little)
    >>> f.f
    4.667261455589845e-62
    >>> f.le.f
    1984.3

The default byte order is ``Endianness.Unspecified``. For whole-byte data this is
the same interpretation as ``Endianness.Big``, but it can also be used for
non-whole-byte data. The explicit ``be`` and ``le`` views require a whole number
of bytes.

Bit order
^^^^^^^^^

The ``msb0`` and ``lsb0`` views control how bit labels are interpreted within
each byte.

``msb0`` is the default convention used by normal indexing and slicing. In a byte,
label 0 is the most significant bit. ``lsb0`` is common in some specifications:
label 0 is the least significant bit of each byte.

One way to see the difference is to materialize the view::

    >>> Tibs('0x12').bin
    '00010010'
    >>> Tibs('0x12').lsb0.bin
    '01001000'

This is mostly useful when the view is used for interpreting field labels.
For ordinary Python indexing and slicing, use the ``Tibs`` or ``Mutibs`` directly.
Views don't provide their own slicing interface, as that would make it too easy
to confuse normal Python slices with specification field labels.

Fields
^^^^^^

Some standards describe fields using inclusive bit labels, such as ``31:28``.
In Python slicing that would usually be written as ``[28:32]``, but for LSB0
formats that still doesn't describe the right physical bits. The important detail
is that the specification is giving bit labels, not Python slice positions.

Use :meth:`View.field` for this case. The two endpoints are inclusive, and can be
given in either order::

    >>> t = Tibs('0x23a11234')
    >>> t.lsb0.field(31, 28).u
    3
    >>> t.lsb0.field(28, 31).u
    3

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
    4660
    >>> header.field(15, 12).u  # flags
    10
    >>> header.field(11, 0).u   # payload_length
    291

Byte order and field extraction are separate ideas. ``field()`` uses the current
bit order to find the labelled bits. The result is then a normal MSB0 value.

The ``message_id`` field is 16 bits long, so it keeps the little-endian byte
order from the header. The stored field bytes are ``34 12``, but the integer
interpretation is ``0x1234``::

    >>> header.field(31, 16)
    View(Tibs('0x3412'), byte_order=Endianness.Little)
    >>> header.field(31, 16).u
    4660

If the extracted field is not a whole number of bytes, byte order no longer has a
meaning and is dropped::

    >>> header.field(11, 0)
    View(Tibs('0x123'))
    >>> header.field(11, 0).u
    291

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
:meth:`View.to_mutibs`::

    >>> t.le.to_tibs()
    Tibs('0x0001')

This makes any byte order or bit order transformation explicit before you go back
to normal indexing, slicing or mutation.
