.. currentmodule:: tibs

A set of bits
-------------

The second lens treats the container as a *set of positions* — a ``1`` means the
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
   * - ``a.is_subset_of(b)``
     - ``(a & b) == a``

The four counting methods are :meth:`~Tibs.count_and`, :meth:`~Tibs.count_or`,
:meth:`~Tibs.count_xor` and :meth:`~Tibs.count_andnot`; the two predicates are
:meth:`~Tibs.intersects` and :meth:`~Tibs.is_subset_of`. Both containers must be
the same length, as they must be for ``&``, ``|`` and ``^``.

It helps to think of a container as the *set of positions where the bit is set* -
so a ``1`` means "present" and a ``0`` means "absent", rather than being a second
kind of value that could match. That is what makes the last two asymmetric between
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

The two predicates stop as soon as they know the answer, so on large containers
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
