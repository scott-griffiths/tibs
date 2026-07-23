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

To count the number of times a bit value or sequence of bits occurs use the :meth:`Tibs.count` method.
With no argument it counts the set bits::

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
========================

The bit-wise operators ``&``, ``|`` and ``^`` build a new container. When all you
want is a count or a yes/no answer, these methods give it to you without
building anything in between, which is typically several times faster:

.. list-table::
   :header-rows: 1

   * - Method
     - Equivalent to
   * - :meth:`~Tibs.count_and`
     - ``(a & b).count(1)``
   * - :meth:`~Tibs.count_or`
     - ``(a | b).count(1)``
   * - :meth:`~Tibs.count_xor`
     - ``(a ^ b).count(1)`` — the Hamming distance
   * - :meth:`~Tibs.count_andnot`
     - ``a.count(1) - a.count_and(b)``
   * - :meth:`~Tibs.intersects`
     - ``(a & b).any()``
   * - :meth:`~Tibs.is_subset_of`
     - ``(a & b) == a``

Both containers must be the same length, as they must be for ``&``, ``|`` and ``^``.

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
``a & b`` would take.

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
    >>> t.find('0b111') is None
    True

The pattern can be anything that can be promoted to a ``Tibs`` - a binary string,
bytes, a strict list/tuple bit pattern, or another ``Tibs``.

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

.. _searching_with_a_mask:

Searching with a mask
=====================

Sometimes only part of a pattern is fixed. Pass a ``mask`` of the same length as
the pattern and only the bits set in it have to match - the rest are don't-cares,
so whatever the pattern has under them is ignored::

    >>> t = Tibs('0x1f2e3f')
    >>> t.find_all('0x0f', mask='0x0f', byte_aligned=True)
    [0, 16]

That finds every byte whose low nibble is ``1111``, whatever its high nibble.
Instruction encodings are the classic case: mask out the register and immediate
fields and search for the opcode bits alone.

The ``mask`` argument works the same way on :meth:`~Tibs.find`,
:meth:`~Tibs.rfind`, :meth:`~Tibs.find_all`, :meth:`~Tibs.find_all_iter`,
:meth:`~Tibs.rfind_all_iter`, :meth:`~Tibs.count` and
:meth:`~Tibs.replaced`, and combines with ``start``, ``end`` and
``byte_aligned`` as usual.

A mask with every bit set is just an ordinary search, and one with no bits set
matches at every position. Masked searches can't use the byte-oriented fast paths
that plain searches do, so they are slower - and for patterns longer than 64 bits
with only a few bits masked in, considerably so.

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


.. _formatting:

Formatting
^^^^^^^^^^

When you just want to see the data, ``Tibs``, ``Mutibs``, ``View`` and ``MutableView``
all support Python's format mini-language, so they work directly in f-strings::

    >>> packet = Tibs('0xac804f4b')
    >>> f"{packet:#x}"
    '0xac804f4b'
    >>> f"{packet:_.8b}"
    '10101100_10000000_01001111_01001011'
    >>> f"{packet:u}"
    '2894090059'

With no format spec you get the same thing as ``str()``, which is what you'd see if you
just printed it::

    >>> f"{packet}"
    '0xac804f4b'

Type codes
==========

There are two families of type code. The first are *representation* codes, which show
you the bits themselves and are exactly equivalent to the ``bin``, ``oct`` and ``hex``
properties:

* ``b`` — binary. Always available.
* ``o`` — octal. Length must be a multiple of 3.
* ``x`` — hexadecimal. Length must be a multiple of 4.
* ``X`` — upper case hexadecimal. Length must be a multiple of 4.

The second are *interpretation* codes, which decode the bits as a number. These borrow
the letters used by :class:`Dtype` rather than Python's ``d``, because a ``Tibs`` has
both a signed and an unsigned reading and there's no sensible way to guess which you
meant:

* ``u`` — the unsigned integer value, as given by :attr:`Tibs.u`.
* ``i`` — the two's complement signed value, as given by :attr:`Tibs.i`.

Both families work at any length, though the interpretation codes need at least one bit
to interpret.

So the same 32 bits can be shown four different ways::

    >>> f"{packet:x}, {packet:b}"
    'ac804f4b, 10101100100000000100111101001011'
    >>> f"{packet:u}, {packet:i}"
    '2894090059, -1400877237'
    >>> f"{Tibs('0o7531'):o}"
    '7531'

.. note::

    The representation codes are not integer formats, so leading zeros are kept and the
    length of the output always tells you the length of the data. ``f"{Tibs('0x0f'):b}"``
    is ``'00001111'``, whereas ``f"{15:b}"`` is ``'1111'``. If you want the number, ask
    for the number with ``u`` or ``i``.

Prefixes and grouping
=====================

The ``#`` flag adds the usual ``0x``, ``0X``, ``0b`` or ``0o`` prefix, and ``_`` inserts
separators between groups of digits. Both are useful for reading long values, and
because the string constructors ignore underscores and understand the prefixes, a value
formatted with ``#``, with or without grouping, can be fed straight back in::

    >>> f"{packet:#_x}"
    '0xac80_4f4b'
    >>> Tibs(f"{packet:#_x}") == packet
    True

.. warning::

    If you combine ``#`` with an alignment, the ``#`` must come *after* it. A ``#``
    immediately followed by an alignment character is read as a fill character
    instead, which is easy to miss::

        >>> f"{packet:>#12x}"
        '  0xac804f4b'
        >>> f"{packet:#>12x}"
        '####ac804f4b'

    This is how the mini-language works for any type, but it bites more often here
    because a prefix is so often what you want.

Python fixes the group size at four digits. That's often the wrong size for binary data,
so tibs lets you set it using the precision field, which the standard mini-language
leaves unused for these types::

    >>> f"{packet:_b}"
    '1010_1100_1000_0000_0100_1111_0100_1011'
    >>> f"{packet:_.8b}"
    '10101100_10000000_01001111_01001011'
    >>> f"{packet:_.2x}"
    'ac_80_4f_4b'

Grouping a ``u`` or ``i`` value works the way it does for any other Python integer, so
you get three-digit groups and ``,`` is available as well::

    >>> f"{packet:_u}"
    '2_894_090_059'
    >>> f"{packet:,i}"
    '-1,400,877,237'

Width and alignment
===================

Fill, alignment and width all behave as they do elsewhere in Python, which is handy for
lining up columns of registers or packet fields::

    >>> for name, value in [('ctrl', Tibs('0x0f')), ('status', Tibs('0xbeef'))]:
    ...     print(f"{name:>8}  {value:>12x}")
        ctrl            0f
      status          beef

.. note::

    Two details differ from integer formatting, both because a ``Tibs`` is a sequence
    rather than a number.

    Groups are counted from bit zero, so it's the *last* group that comes up short
    rather than the first. ``f"{Tibs('0b101010101'):_b}"`` gives ``'1010_1010_1'``,
    where the equivalent integer format gives ``'1_0101_0101'``.

    Padding is added after grouping and is never itself grouped, so the separators stay
    lined up with real bit positions.

The fill character has to be something that can't be mistaken for the data, so zero
padding is not available for ``b``, ``o``, ``x`` and ``X``::

    >>> f"{Tibs('0xf'):#06x}"
    Traceback (most recent call last):
    ...
    ValueError: Zero padding is not allowed with the 'x' format type, because the padding could not be told apart from the data and would change its apparent length. Align with '<', '>' or '^' to pad with spaces instead, or use the 'u' or 'i' type code for a numeric interpretation.

Zero padding an integer is harmless, because leading zeros don't change what an integer
is. Here they would: a 4-bit value padded to ``'0x000f'`` reads as a 16-bit one, and
comes back as a 16-bit one if you feed it in again. The same goes for any other fill
that is a valid digit for the type, such as ``f`` for hex or ``1`` for binary. Digits
that can't appear in that base are fine, as is anything else::

    >>> f"{Tibs('0xf'):*>6x}"
    '*****f'
    >>> f"{Tibs('0b1111'):8>6b}"
    '881111'

If you want a fixed number of bits rather than a fixed number of characters, change the
data rather than its presentation, for example with :meth:`Tibs.from_u`.

A few other things that are meaningful for numbers are also rejected for the
representation codes: the sign characters ``+``, ``-`` and space, and the ``,``
separator. All of them are accepted by ``u`` and ``i``, which really are numbers.

Long values
===========

``str()`` gives up on very long values and shows a truncated version with the length
appended, and a format spec with no type code does the same. An explicit type code
never truncates::

    >>> t = Tibs.from_zeros(50_000)
    >>> f"{t}".endswith('... # length=50000')
    True
    >>> len(f"{t:b}")
    50000
