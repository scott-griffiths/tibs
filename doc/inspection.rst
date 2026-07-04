.. currentmodule:: tibs

Inspections
-----------

Once you've created a ``Tibs`` you can treat it very like an array of bits, with standard
Python indexing and slicing, as well as finding, counting and other useful methods.

Indexing and slicing
^^^^^^^^^^^^^^^^^^^^

The default behaviour for indexing and slicing should hold no surprises.
Indexing returns a bool, slicing returns a new ``Tibs``::

    >>> t = Tibs('0x0f')
    >>> t[0]
    False
    >>> t[-1]
    True
    >>> t[:4]
    Tibs('0x0')
    >>> t[-4:]
    Tibs('0xf')

You can also use extended slices::

    >>> t[::-1].bin
    '11110000'
    >>> t[::2].bin
    '0011'

A ``Mutibs`` can also have bits and slices set::

    >>> m = Mutibs('0xff01')
    >>> m[0] = 0
    >>> m[-4:] = '0xbeef'
    >>> m
    Mutibs('0x7f0beef')


Information methods
^^^^^^^^^^^^^^^^^^^

count
=====

To count the number of times a bit value or sequence of bits occurs use the :meth:`Tibs.count` method::

    >>> t = Tibs.from_random(100_000_000)
    >>> t.count(1)
    49996739
    >>> t.count([1, 0, 1])
    12503821
    >>> Tibs('0xef').count(1, 0, 4)
    3

Counting should be very fast, especially when just counting the number of ``1`` or ``0`` bits.

find / rfind
============

Use :meth:`Tibs.find` to find the first occurrence of a bit pattern, and
:meth:`Tibs.rfind` to search from the right. Both methods return the bit index of
the match, or ``None`` if no match is found::

    >>> t = Tibs('0b0011010101100')
    >>> t.find('0b101')
    3
    >>> t.rfind('0b101')
    7
    >>> t.find('0b111')
    None

The pattern can be anything that can be promoted to a ``Tibs`` - a binary string,
bytes, a list of bool-like values, or another ``Tibs``.

The optional ``start`` and ``end`` arguments restrict the search to a slice of
the data. If you know the
pattern can only start on a byte boundary, set ``byte_aligned=True``. This is
often faster for scanning binary files or network frames::

    >>> capture = Tibs('0x00ffaa551122aa553344')
    >>> capture.find('0xaa55', byte_aligned=True)
    16

The ``in`` operator is a convenient shorthand when you only care whether the
pattern exists::

    >>> '0xaa55' in capture
    True

find_all / find_all_iter
========================

Use :meth:`Tibs.find_all` to get every matching start position::

    >>> t = Tibs('0b10100101')
    >>> t.find_all('0b101')
    [0, 5]

Matches may overlap. This is useful when searching for bit patterns rather than
tokens::

    >>> Tibs('0b1111').find_all('0b11')
    [0, 1, 2]

For large inputs, :meth:`Tibs.find_all_iter` avoids building the whole list up
front::

    >>> t = Tibs('0b10100101')
    >>> for pos in t.find_all_iter('0b101'):
    ...     print(pos)
    0
    5

There is also :meth:`Tibs.rfind_all_iter`, which yields matches from right to
left. Iterator forms are only available on ``Tibs``. If you have a ``Mutibs``,
use :meth:`Mutibs.to_tibs` to make an immutable copy, or :meth:`Mutibs.as_tibs`
to move the data if you no longer need the mutable object.

starts_with / ends_with
=======================

The :meth:`Tibs.starts_with` and :meth:`Tibs.ends_with` methods test prefixes and
suffixes without spelling out slice boundaries::

    >>> packet = Tibs('0xaa551234')
    >>> packet.starts_with('0xaa55')
    True
    >>> packet.ends_with('0x1234')
    True

any / all
=========

The :meth:`Tibs.any` and :meth:`Tibs.all` methods mirror Python's built-in
``any()`` and ``all()``, but operate directly on the stored bits::

    >>> Tibs('0b0001').any()
    True
    >>> Tibs('0b0001').all()
    False
    >>> Tibs.from_ones(8).all()
    True

They are most useful when the bit sequence itself is the data, for example when
checking whether a mask has any flags set, or whether every flag in a required
set is present.


Splitting at positions
^^^^^^^^^^^^^^^^^^^^^^

Sometimes instead of using slices, if you want to partition a value at one or more bit
positions use :meth:`Tibs.split_at`::

    >>> t = Tibs('0b101100')
    >>> head, tail = t.split_at(3)
    >>> head, tail
    (Tibs('0b101'), Tibs('0b100'))
    >>> flags, length, payload = t.split_at([2, 5])
    >>> flags, length, payload
    (Tibs('0b10'), Tibs('0b110'), Tibs('0b0'))

The positions use normal bit offsets. Negative positions count from the end,
and duplicate positions create empty pieces. The positions must be in
nondecreasing order after negative positions are normalized.
