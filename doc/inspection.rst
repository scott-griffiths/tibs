.. currentmodule:: tibs

Inspections
-----------

Once you've created a ``Tibs`` you can treat it very like an array of bits.

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

Counting should be fast, especially when just counting the number ``1`` or ``0``.

find / rfind
============

find_all / rfind_all
====================

starts_with / ends_with
=======================

any / all
=========