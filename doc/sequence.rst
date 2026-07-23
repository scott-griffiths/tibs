.. currentmodule:: tibs

The sequence of bits
--------------------

At heart a ``Tibs`` is a sequence of bits, much like ``bytes`` but with the bit as
the unit and no restriction on length. You can treat it very like an array of
bits, with standard Python indexing and slicing, as well as searching, splitting,
editing and reordering. Everything in this chapter works on the bits as raw
material; reading typed values out of them is covered in :doc:`typed_fields`.

The examples use ``Tibs``, but the constructors and non-mutating methods apply
equally to ``Mutibs``.

.. _construction:

Building from raw bits
^^^^^^^^^^^^^^^^^^^^^^

A range of ``from_`` constructors build a value directly from raw bit material:

* :meth:`Tibs.from_bin`: Create from a binary string, optionally starting with '0b'.
* :meth:`Tibs.from_oct`: Create from an octal string, optionally starting with '0o'.
* :meth:`Tibs.from_hex`: Create from a hex string, optionally starting with '0x'.
* :meth:`Tibs.from_bytes`: Create directly from a ``bytes``, ``bytearray`` or ``memoryview`` object.
* :meth:`Tibs.from_string`: Create from a formatted string.
* :meth:`Tibs.from_bools`: Convert each element in an iterable to a bool.
* :meth:`Tibs.from_zeros`: Initialise with ``0`` bits.
* :meth:`Tibs.from_ones`: Initialise with ``1`` bits.
* :meth:`Tibs.from_random`: Initialise with randomly set bits.
* :meth:`Tibs.from_joined`: Concatenate an iterable of objects.

Some examples::

    # Five bits from a binary string
    a = Tibs.from_bin('11001')

    # Directly from bytes, bytearray or a memoryview. Useful if creating from a file.
    b = Tibs.from_bytes(b'some_bytes')

    # Create bits from the truthiness of any iterator.
    c = Tibs.from_bools([1, 0, 1, 1, 1])

    # Optionally seeded random bits. There's also an option to use the OS's secure generator.
    d = Tibs.from_random(1000, seed=b'a_seed')

    # Hex, binary and octal strings can be parsed.
    h = Tibs.from_string('0xff01, 0b101')

    # An efficient way to join many other Tibs together.
    i = Tibs.from_joined([a, b, c, d, h])

The constructors that encode numeric and other typed values — :meth:`Tibs.from_u`,
:meth:`Tibs.from_i`, :meth:`Tibs.from_f`, :meth:`Tibs.from_value` and
:meth:`Tibs.from_values` — are covered in :doc:`typed_fields`.

Calling ``Tibs`` (or ``Mutibs``) directly promotes strings, bytes-like objects and
strict list/tuple bit patterns for you, as described under :ref:`promotion`. A few
finer points are worth knowing.

The list and tuple shorthand is intentionally strict: only ``True``, ``False``,
``0`` and ``1`` are accepted. Use :meth:`Tibs.from_bools` when you want to convert
arbitrary truthy values or an arbitrary iterator to bits. File-like objects such
as ``io.BytesIO`` are not read implicitly; pass the bytes you want explicitly, for
example ``Tibs.from_bytes(stream.getvalue())``, ``Tibs.from_bytes(stream.read())``
or ``Tibs.from_bytes(stream.getbuffer())``. Similarly, pass ``memoryview(arr)`` to
:meth:`Tibs.from_bytes` for an ``array.array`` only when that raw byte
representation is intended.

.. note::

    ``Tibs`` can be arbitrary sizes, so lengths are always given in bits and not bytes.


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

Indexing and slicing always run from left to right, with bit zero on the left.
When a specification numbers bits differently, or a multi-byte value is stored
little-endian, use a :doc:`view <views>` rather than reaching for arithmetic.


Searching
^^^^^^^^^

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


Splitting and chunking
^^^^^^^^^^^^^^^^^^^^^^

Instead of using slices, if you want to partition a value at one or more bit
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

To cut a value into equal-sized pieces use :meth:`Tibs.chunks`, which returns a
list, or :meth:`Tibs.chunks_iter` / :meth:`Tibs.rchunks_iter`, which yield the
pieces lazily. The iterator forms are only available on ``Tibs`` (see
:doc:`tibs_vs_mutibs`)::

    >>> [c.hex for c in Tibs('0xdeadbeef').chunks(8)]
    ['de', 'ad', 'be', 'ef']


Concatenating, repeating and shifting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``+`` operator concatenates, and ``*`` repeats, just as they do for ``bytes``.
The right-hand side of ``+`` is promoted, so a string or bytes value can be joined
directly::

    >>> Tibs('0b101') + '0b011'
    Tibs('0b101011')
    >>> Tibs('0b10') * 3
    Tibs('0b101010')

The ``<<`` and ``>>`` operators shift the bits, keeping the length the same and
filling with zeros::

    >>> Tibs('0b0011').bin, (Tibs('0b0011') << 1).bin, (Tibs('0b0011') >> 1).bin
    ('0011', '0110', '0001')

To move bits around the ends without losing them, use the rotations described
below. When building a long value from many pieces, prefer
:meth:`Tibs.from_joined` or a ``Mutibs`` over repeated ``+`` — see
:doc:`tibs_vs_mutibs`.


Editing in place
^^^^^^^^^^^^^^^^

The methods in this section change a ``Mutibs`` in place and return ``None``. Many
of them have copy-returning siblings (``insert``/``inserted``,
``replace``/``replaced``, and so on) that are also available on the immutable
``Tibs``; the naming convention is described in :doc:`tibs_vs_mutibs`.

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
in performance critical code. For immutable data, use ``+`` instead.

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

Slice assignment writes a *contiguous* field. To write one whose bits are
scattered through the container by a mask, use :meth:`Mutibs.deposit` — see
:ref:`scattered fields <deposit>`.


Reordering bits
^^^^^^^^^^^^^^^

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

Each of these has a copy-returning form (:meth:`~Tibs.reversed`,
:meth:`~Tibs.rotated_left`, :meth:`~Tibs.rotated_right`,
:meth:`~Tibs.byte_swapped`) available on both ``Tibs`` and ``Mutibs``.


Stack-like operations and capacity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
