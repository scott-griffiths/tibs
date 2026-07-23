.. currentmodule:: tibs

Manipulations
-------------

Mutating and copy methods
^^^^^^^^^^^^^^^^^^^^^^^^^

``Mutibs`` has many mutating methods, which change the value in-place and return ``None``.
Many of these methods have siblings which do the same task but don't modify the instance and
instead return a new copy. These 'copy' methods are also available on the immutable ``Tibs``.

For example, :meth:`Mutibs.reverse` changes the existing object and returns
``None``::

    >>> m = Mutibs('0b101100')
    >>> result = m.reverse()
    >>> result is None
    True
    >>> m
    Mutibs('0b001101')

The copy-returning form leaves the original value alone::

    >>> t = Tibs('0b101100')
    >>> r = t.reversed()
    >>> t
    Tibs('0b101100')
    >>> r
    Tibs('0b001101')

The same copy-returning methods are also available on ``Mutibs`` when you want a
new mutable value without changing the original::

    >>> m = Mutibs('0b101100')
    >>> r = m.reversed()
    >>> m
    Mutibs('0b101100')
    >>> r
    Mutibs('0b001101')

.. csv-table::
   :header: "Mutibs mutating methods", "Tibs/Mutibs copy equivalent"

   ":meth:`~Mutibs.byte_swap`", ":meth:`~Mutibs.byte_swapped`"
   ":meth:`~Mutibs.insert`", ":meth:`~Mutibs.inserted`"
   ":meth:`~Mutibs.invert`", ":meth:`~Mutibs.inverted`"
   ":meth:`~Mutibs.replace`", ":meth:`~Mutibs.replaced`"
   ":meth:`~Mutibs.reverse`", ":meth:`~Mutibs.reversed`"
   ":meth:`~Mutibs.rotate_left`", ":meth:`~Mutibs.rotated_left`"
   ":meth:`~Mutibs.rotate_right`", ":meth:`~Mutibs.rotated_right`"
   ":meth:`~Mutibs.set`", ":meth:`~Mutibs.set_at`"
   ":meth:`~Mutibs.unset`", ":meth:`~Mutibs.unset_at`"


The linguistic oddities here are ``set_at()`` and ``unset_at()``, as the past-participle of 'set' is
also 'set', so the naming pattern failed (English is annoying sometimes).

Other Mutibs methods
^^^^^^^^^^^^^^^^^^^^

Not all mutating methods have a copy equivalent - things like ``clear()`` don't make sense for a
``Tibs``, and you can use the ``+`` operator to do non-mutating extensions.

Appending and extending
=======================

Use :meth:`Mutibs.append` for a single bit, :meth:`Mutibs.extend` to add bits on
the right, and :meth:`Mutibs.extend_left` to add bits on the left::

    >>> m = Mutibs('0b101')
    >>> m.append(0)
    >>> m.extend('0b11')
    >>> m
    Mutibs('0b101011')
    >>> m.extend_left('0b00')
    >>> m.bin
    '00101011'

Extending to the left is almost certain to need memory reallocation so should be avoided
in performance critical code.

For immutable data, use ``+`` instead::

    >>> Tibs('0b101') + '0b011'
    Tibs('0b101011')

Inserting, replacing and deleting
=================================

:meth:`Mutibs.insert` inserts bits at a position without removing anything::

    >>> m = Mutibs('0b1001')
    >>> m.insert(2, '0b11')
    >>> m
    Mutibs('0b101101')

:meth:`Mutibs.replace` searches for one bit pattern and replaces it with another.
It accepts the same ``start``, ``end``, ``count``, ``byte_aligned`` and ``mask``
options as the non-mutating :meth:`Tibs.replaced` method, and returns the number
of replacements made::

    >>> m = Mutibs('0b100100100')
    >>> m.replace('0b100', '0b11', count=2)
    2
    >>> m
    Mutibs('0b1111100')

A ``mask`` says which bits of the pattern have to match, as described in
:ref:`searching with a mask <searching_with_a_mask>`. It affects only the
matching - the whole of each match is still replaced by the new bits::

    >>> m = Mutibs('0x1f2e3f')
    >>> m.replace('0x0f', '0x00', mask='0x0f', byte_aligned=True)
    2
    >>> m
    Mutibs('0x002e00')

Slices can also be assigned to, or deleted, using normal Python syntax::

    >>> m = Mutibs('0b11110000')
    >>> m[2:6] = '0b01'
    >>> m
    Mutibs('0b110100')
    >>> del m[1:3]
    >>> m.bin
    '1100'

.. _deposit:

Writing a scattered field
=========================

Slice assignment writes a contiguous field. To write one whose bits are
scattered through the container by a mask, use :meth:`Mutibs.deposit`: the bits of
the value are placed at the positions the mask selects, and the rest of the
container is left alone. It is the inverse of :meth:`~Tibs.extract` (see
:ref:`scattered_fields`) and the bit-level version of the PDEP instruction::

    >>> m = Mutibs('0b11010110')
    >>> m.deposit('0b111', '0b10110000')   # write 3 bits into positions 0, 2, 3
    >>> m.bin
    '11110110'

The value must be exactly ``mask.count()`` bits long. The non-mutating
:meth:`Tibs.deposited` returns a new container instead of writing in place.

Setting and unsetting bits
==========================

:meth:`Mutibs.set`, :meth:`Mutibs.unset` and :meth:`Mutibs.invert` can operate on
one position or many positions. Passing a ``range`` is the usual way to update a
regular pattern efficiently::

    >>> m = Mutibs.from_zeros(12)
    >>> m.set([0, 3, 4])
    >>> m.bin
    '100110000000'
    >>> m.invert(range(0, 12, 2))
    >>> m.bin
    '001100101010'

The copy-returning forms are useful when you want expression-style code with an
immutable ``Tibs``::

    >>> Tibs.from_zeros(8).set_at([1, 6])
    Tibs('0x42')

Assigning interpreted values
============================

The ``write_u``, ``write_i`` and ``write_f`` methods replace the current bits with a
new unsigned integer, signed integer or floating-point value while preserving
the existing bit length::

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

For little-endian or LSB0 interpretations, assign through a mutable view instead
(we'll cover :class:`MutableView` later)::

    >>> m = Mutibs.from_u(99, 16, ByteOrder.Little)
    >>> m.le.u
    99
    >>> m.le.u = 45
    >>> m.le.u
    45


Reordering bits
===============

Use :meth:`Mutibs.reverse` to reverse the full sequence.
Use :meth:`Mutibs.rotate_left` and :meth:`Mutibs.rotate_right` to rotate either the
whole sequence or a selected range::

    >>> m = Mutibs('0b10110010')
    >>> m.rotate_left(2, start=1, end=7)
    >>> m.bin
    '11001010'
    >>> m.reverse()
    >>> m.bin
    '01010011'

:meth:`Mutibs.byte_swap` reverses byte groups. With no argument it reverses the
order of all selected bytes; with a byte length it reverses each group of that
size. Use ``start`` and ``end`` to byte-swap only part of a sequence::

    >>> m = Mutibs('0x12345678')
    >>> m.byte_swap()
    >>> m
    Mutibs('0x78563412')
    >>> m.byte_swap(2)
    >>> m
    Mutibs('0x56781234')
    >>> m.byte_swap(start=8, end=24)
    >>> m
    Mutibs('0x56127834')

Stack-like operations and capacity
==================================

:meth:`Mutibs.pop` removes and returns the final bit::

    >>> m = Mutibs('0b101')
    >>> m.pop()
    True
    >>> m
    Mutibs('0b10')

:meth:`Mutibs.clear` removes all bits while keeping the object available for
reuse. :meth:`Mutibs.reserve` asks for space for additional bits, and
:meth:`Mutibs.capacity` reports the current allocated capacity. These capacity
methods are only performance hints; normal code usually does not need them.
