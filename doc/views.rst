.. currentmodule:: tibs

Views
-----

Views wrap ``Tibs`` or ``Mutibs`` data to allow the bits inside it to be
interpreted in a different way. This allows different endiannesses to be used,
as well as different bit numbering methods when interpreting the data.

Immutable ``Tibs`` values use :class:`View`. Mutable ``Mutibs`` values use
:class:`MutableView` when created through ``m.view()``, ``m.le``, ``m.be``,
``m.lsb0`` or ``m.msb0``. A mutable view reads from and writes to the original
``Mutibs``.

The most common reason to create a view is that a file format or protocol specifies
values using a different byte order or bit numbering convention to the Python default.

For example, let's create a four-byte ``Tibs`` and interpret it as an unsigned int::

    >>> t = Tibs('0x01000000')
    >>> t.to_u()
    16777216

This is the byte-wise and bit-wise big-endian interpretation, which corresponds to
the standard Python indexing convention where bit zero is the most significant bit.

As this is a whole number of bytes long we can also consider its byte endianness.
A little-endian interpretation essentially reverses the byte order, so the least
significant byte is the first one. Without changing the data at all, we can create
a ``View`` which wraps it, and then use the interpretation on that ``View`` ::

    >>> v = View(t, Endianness.Little)
    >>> v
    View(Tibs('0x01000000'), byte_order=Endianness.Little)
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

So for the above example, if we also use the :attr:`Tibs.u` property instead of ``.to_u()``, we
get the much more convenient ::

    >>> t.le.u
    1


Views and data
^^^^^^^^^^^^^^

A view created from a ``Tibs`` is immutable: it keeps the same immutable data and
adds interpretation settings. A view created from a ``Mutibs`` using
``m.view()``, ``m.le``, ``m.be``, ``m.lsb0`` or ``m.msb0`` is a
:class:`MutableView`. It keeps a live reference to the ``Mutibs``, so later
changes to the ``Mutibs`` are reflected in the view.

The direct :class:`View` constructor is intentionally stricter than ``Tibs``.
It accepts a ``Tibs`` or ``Mutibs`` object, but not other types that could be
promoted.

Passing a ``Mutibs`` to the direct :class:`View` constructor still creates an
immutable snapshot. Use :class:`MutableView` or the ``Mutibs`` view helpers when
you want live mutable behavior.

This emphasises that views are interpretation wrappers rather than another way
to construct binary data.

Mutable views
^^^^^^^^^^^^^

A :class:`MutableView` can also write interpreted values back into the source
``Mutibs`` without changing its length. The view supplies the layout::

    >>> m = Mutibs.from_u(99, 16, Endianness.Little)
    >>> m.le.u
    99
    >>> m.le.set_u(45)
    >>> m.le.u
    45
    >>> m
    Mutibs('0x2d00')

The ``u``, ``i`` and ``f`` properties are settable too::

    >>> m.le.u = 123
    >>> m.le.u
    123

For default layout, the whole ``Mutibs`` also has ``set_u``, ``set_i`` and
``set_f`` methods and settable ``u``, ``i`` and ``f`` properties. Use a mutable
view when byte order or bit order matters.

Byte order
^^^^^^^^^^

Byte order only applies to whole-byte values. When you construct a value from an
integer or float you can choose the byte order used to store it::

    >>> t = Tibs.from_u(666, 16, Endianness.Little)
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

    >>> x = Tibs.from_f(1984.3, 64, Endianness.Little)
    >>> x.f
    4.667261455589845e-62
    >>> x.le.f
    1984.3

The default byte order is ``Endianness.Unspecified``. For whole-byte data this is
the same interpretation as ``Endianness.Big``, but it can also be used for
non-whole-byte data. The explicit ``be`` and ``le`` views require a whole number
of bytes.

Bit order
^^^^^^^^^

The ``msb0`` and ``lsb0`` views control how bit labels are interpreted within
each byte.

``msb0`` is the default convention used by normal indexing and slicing, where
the most significant bit of the byte (the left-most bit) is bit 0, with the right-most
bit being bit 7. For ``lsb0``, which is used in some specifications, the least significant
bit of the byte (the right-most bit) is bit 0, and the left-most bit is bit 7.

One way to see the difference is to materialize the view::

    >>> t = Tibs('0x0100')
    >>> t.bin
    '0000000100000000'
    >>> t.le.bin
    '0000000000000001'
    >>> t.lsb0.bin
    '1000000000000000'
    >>> t.lsb0.le.bin
    '0000000010000000'

Let's go through these one at a time:

* ``t.bin`` -> ``00000001_00000000``. This is the standard Python indexing view. The bit
  indices are just counting up from 0 on the LHS to 15 on the RHS.
* ``t.le.bin`` -> ``00000000_00000001``. The byte-wise little-endian view. The byte
  order is swapped by the view, so the right-most byte has the most significant bits and the
  left-most byte as the least significant bits, but
  the bits within each byte are unchanged.
* ``t.lsb0.bin`` -> ``10000000_00000000``. The Least Significant Bit Zero (LSB0) view. Here
  bit zero of each byte (the left-most bit) is the least significant bit, rather than the
  more usual most significant bit. Note that this view doesn't change the byte order - it's
  like traversing the bytes from left to right, but taking the bits in each byte from
  right to left.
* ``t.lsb0.le.bin`` -> ``00000000_10000000``. Finally we can combine them (in either order)
  to both traverse the bytes from right to left and the bits in the byte from right to left.
  The overall effect is to reverse the bit order.


For ordinary Python indexing and slicing, use the ``Tibs`` or ``Mutibs`` directly.
Views don't provide their own slicing interface, as that would make it too easy
to confuse normal Python slices with specification field labels.

Fields
^^^^^^

Some standards describe fields using inclusive bit labels, such as ``31:28``.
In Python slicing that would usually be written as ``[28:32]``, but for LSB0
formats that still doesn't describe the right physical bits. The important detail
is that the specification is giving bit labels, not Python slice positions.

For this case we can use :meth:`View.field`. The two endpoints are inclusive, and can be
given in either order::

    >>> t = Tibs('0x23a11234')
    >>> t.lsb0.field(31, 28).u
    3
    >>> t.lsb0.field(28, 31).u
    3

When the source is mutable, :meth:`MutableView.field` returns another live
``MutableView`` over the selected bits. Assigning through that field writes back
to the original ``Mutibs``::

    >>> m = Mutibs('0x23a11234')
    >>> m.lsb0.le.field(31, 16).u
    4660
    >>> m.lsb0.le.field(31, 16).u = 0x5678
    >>> m.lsb0.le.field(31, 16).u
    22136
    >>> m
    Mutibs('0x23a15678')

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
    >>> header.field(15, 12).bin  # flags
    '1010'
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
:meth:`View.to_mutibs` (also available on :class:`MutableView`)::

    >>> t.le.to_tibs()
    Tibs('0x0001')

This makes any byte order or bit order transformation explicit before you go back
to normal indexing, slicing or mutation.
