.. currentmodule:: tibs

Manipulations
-------------

Mutating and copy methods
^^^^^^^^^^^^^^^^^^^^^^^^^

``Mutibs`` has many mutating methods, which change the value in-place and return ``None``.
Many of these methods have siblings which do the same task but don't modify the instance and
instead return a new copy. These 'copy' methods are also available on the immutable ``Tibs``.

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
It accepts the same ``start``, ``end``, ``count`` and ``byte_aligned`` options as
the non-mutating :meth:`Tibs.replaced` method::

    >>> m = Mutibs('0b100100100')
    >>> m.replace('0b100', '0b11', count=2)
    >>> m
    Mutibs('0b1111100')

Slices can also be assigned to, or deleted, using normal Python syntax::

    >>> m = Mutibs('0b11110000')
    >>> m[2:6] = '0b01'
    >>> m
    Mutibs('0b110100')
    >>> del m[1:3]
    >>> m.bin
    '1100'

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

The ``set_u``, ``set_i`` and ``set_f`` methods replace the current bits with a
new unsigned integer, signed integer or floating-point value while preserving
the existing bit length::

    >>> m = Mutibs.from_zeros(8)
    >>> m.set_u(15)
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

    >>> m = Mutibs.from_u(99, 16, Endianness.Little)
    >>> m.le.u
    99
    >>> m.le.u = 45
    >>> m.le.u
    45

Mutable views can also select labelled fields and assign through the same
properties. The field endpoints are inclusive and interpreted using the view's
current bit order::

    >>> m = Mutibs('0x23a11234')
    >>> m.lsb0.le.field(31, 16).u = 0x5678
    >>> m
    Mutibs('0x23a15678')

Reordering bits
===============

Use :meth:`Mutibs.reverse` to reverse the full sequence, or
:meth:`Mutibs.rotate_left` and :meth:`Mutibs.rotate_right` to rotate either the
whole sequence or a selected range::

    >>> m = Mutibs('0b10110010')
    >>> m.rotate_left(2, start=1, end=7)
    >>> m.bin
    '11001010'
    >>> m.reverse()
    >>> m.bin
    '01010011'

:meth:`Mutibs.byte_swap` reverses byte groups. With no argument it reverses the
order of all bytes; with a byte length it reverses each group of that size::

    >>> m = Mutibs('0x12345678')
    >>> m.byte_swap()
    >>> m
    Mutibs('0x78563412')
    >>> m.byte_swap(2)
    >>> m
    Mutibs('0x56781234')

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
