.. currentmodule:: tibs

A set of bits
-------------

This chapter treats the container as a *set of positions* — a ``1`` means the
position is "present" and a ``0`` means "absent". Under this reading the natural
operations are bitwise algebra, cardinalities and set predicates. Flags, masks,
permission sets, Bloom filters and feature fingerprints are all this shape, and a
large ``Mutibs`` doubles as an efficient mutable bitset (see :doc:`example_sieve`).


Bitwise operators
^^^^^^^^^^^^^^^^^

The ``&``, ``|`` and ``^`` operators combine two equal-length containers bit by
bit, and ``~`` inverts every bit. Each builds a new container::

    >>> a, b = Tibs('0b1100'), Tibs('0b1010')
    >>> (a & b).bin, (a | b).bin, (a ^ b).bin
    ('1000', '1110', '0110')
    >>> (~a).bin
    '0011'

The two operands must be the same length, just as they must be for the
comparison methods below. The right-hand side is promoted, so ``a & '0b1010'``
works. On a ``Mutibs`` the in-place forms ``&=``, ``|=`` and ``^=`` combine
without building an intermediate.


Counting
^^^^^^^^

To count the number of times a bit value or sequence of bits occurs use the
:meth:`Tibs.count` method. With no argument it counts the set bits::

    >>> t = Tibs.from_random(100_000_000)
    >>> t.count()
    49996739
    >>> t.count(0)
    50003261
    >>> t.count([1, 0, 1])
    12503821
    >>> Tibs('0xef').count(1, 0, 4)
    3

Counting a multi-bit pattern counts overlapping occurrences, the same way
:meth:`~Tibs.find_all` reports them. The ``byte_aligned`` argument restricts the
count to byte boundaries and also works when counting a single bit, so
``count(1, byte_aligned=True)`` counts just the set bits that land on a byte
boundary::

    >>> Tibs('0b1111111').count('0b11')
    6
    >>> Tibs('0x80ff00').count(1, byte_aligned=True)
    2

Counting should be very fast, especially when just counting the number of ``1`` or ``0`` bits.

.. _comparing_two_containers:

Comparing two containers
^^^^^^^^^^^^^^^^^^^^^^^^

The bit-wise operators ``&``, ``|`` and ``^`` build a new container. When all you
want is a count or a yes/no answer, these methods give it to you without
building anything in between, which is typically several times faster:

.. list-table::
   :header-rows: 1

   * - Method call
     - Equivalent to
   * - ``a.count_and(b)``
     - ``(a & b).count()``
   * - ``a.count_or(b)``
     - ``(a | b).count()``
   * - ``a.count_xor(b)``
     - ``(a ^ b).count()`` — the Hamming distance
   * - ``a.count_andnot(b)``
     - ``a.count() - a.count_and(b)``
   * - ``a.intersects(b)``
     - ``(a & b).any()``
   * - ``a.is_disjoint(b)``
     - ``not (a & b).any()``
   * - ``a.is_subset_of(b)``
     - ``(a & b) == a``
   * - ``a.is_superset_of(b)``
     - ``(a & b) == b``

The four counting methods are :meth:`~Tibs.count_and`, :meth:`~Tibs.count_or`,
:meth:`~Tibs.count_xor` and :meth:`~Tibs.count_andnot`; the four predicates are
:meth:`~Tibs.intersects`, :meth:`~Tibs.is_disjoint`, :meth:`~Tibs.is_subset_of`
and :meth:`~Tibs.is_superset_of`. Both containers must be the same length, as
they must be for ``&``, ``|`` and ``^``.

It helps to think of a container as the *set of positions where the bit is set* -
so a ``1`` means "present" and a ``0`` means "absent", rather than being a second
kind of value that could match. That is what makes the predicates asymmetric between
``1`` and ``0``::

    >>> a, b = Tibs('0b1100'), Tibs('0b1010')
    >>> a.count_and(b)     # only position 0 is set in both
    1
    >>> a.count_xor(b)     # positions 1 and 2 differ
    2
    >>> a.intersects(b)
    True
    >>> a.is_subset_of(b)  # position 1 is set in a but not in b
    False

Flags are the case where this reads most naturally::

    >>> granted = Tibs('0b1011')
    >>> Tibs('0b1010').is_subset_of(granted)
    True
    >>> granted.is_superset_of('0b1010')
    True
    >>> granted.is_disjoint('0b0100')
    True

The predicates stop as soon as they know the answer, so on large containers
where the answer comes early they finish in a fraction of the time that building
``a & b`` would take. See :doc:`example_fingerprints` for a worked example.


Setting, unsetting and inverting bits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

These modify a ``Mutibs`` in place. The copy-returning forms
:meth:`~Tibs.set_at`, :meth:`~Tibs.unset_at` and :meth:`~Tibs.inverted` are
available on both types, and are useful when you want expression-style code with
an immutable ``Tibs``::

    >>> Tibs.from_zeros(8).set_at([1, 6])
    Tibs('0x42')

(The naming is ``set_at`` / ``unset_at`` rather than ``set`` / ``unset`` because
the past participle of "set" is also "set"; see :doc:`tibs_vs_mutibs`.)


any / all
^^^^^^^^^

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


.. _getting_the_positions_out:

Getting the positions out
^^^^^^^^^^^^^^^^^^^^^^^^^

Counting tells you how many positions are in the set. Sooner or later you want
to know *which*, and there is no separate method for that - the searching
methods do it, given a one-bit pattern to look for::

    >>> t = Tibs('0b10110001')
    >>> t.find_all([1])
    [0, 2, 3, 7]
    >>> t.find_all([0])
    [1, 4, 5, 6]

The ``[1]`` is a one-bit pattern, and ``'0b1'`` says the same thing if you
prefer.

When you only want to walk the positions, :meth:`Tibs.find_all_iter` yields them
one at a time without building the list::

    >>> for pos in Tibs('0b10110001').find_all_iter([1]):
    ...     print(pos)
    0
    2
    3
    7

These all work on a ``Mutibs`` too, except the iterator forms, which are
``Tibs``-only because the contents could change while the iterator was live.

There is more on all of these in :ref:`Searching <searching>`, including
``start``/``end`` bounds, ``byte_aligned`` and searching with a mask. For a set of
positions being built and then read back, see :doc:`example_sieve`.
