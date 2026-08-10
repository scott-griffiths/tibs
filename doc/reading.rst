.. currentmodule:: tibs

Reading in sequence
-------------------

The methods in :doc:`typed_fields` each take a ``start`` and an ``end``, which
is the right shape when you know where a field is. When you are working through
a stream, one field after another, you end up threading a bit position through
the code by hand::

    >>> frame = Tibs('0x47ff10')
    >>> pos = 0
    >>> version = frame.to_value('u8', pos, pos + 8)
    >>> pos += 8
    >>> flag, count = frame.to_value('(bool, u7)', pos, pos + 8)
    >>> pos += 8

A :class:`Reader` holds that position for you. It wraps a ``Tibs`` or
``Mutibs`` and moves a cursor as it reads::

    >>> r = Reader(frame)
    >>> r.read_value('u8')
    71
    >>> r.read_value('(bool, u7)')
    (True, 127)
    >>> r.pos
    16

Every dtype has a known bit length, so the reader always knows how far to
advance. What it removes is the bookkeeping, and the off-by-one errors that come
with it.

The source is not copied, and is always reachable::

    >>> r.source is frame
    True

That matters because no ``Reader`` method takes ``start`` or ``end``. For a
query that shouldn't disturb the cursor, ask the source instead::

    >>> r.source.to_value('u8', 0, 8)
    71
    >>> r.pos
    16


What each method returns
^^^^^^^^^^^^^^^^^^^^^^^^

There is no bare ``read()``: each method's name says what it gives back.

.. list-table::
   :header-rows: 1

   * - Method
     - Returns
   * - :meth:`~Reader.read_value`, :meth:`~Reader.peek_value`
     - one decoded value
   * - :meth:`~Reader.read_values`
     - a list of decoded values
   * - :meth:`~Reader.read_bits`, :meth:`~Reader.peek_bits`, :meth:`~Reader.read_to`, :meth:`~Reader.read_past`
     - a :class:`Tibs`
   * - :meth:`~Reader.seek_to`, :meth:`~Reader.seek_past`, :meth:`~Reader.seek_back_to`
     - ``True`` or ``False``
   * - :meth:`~Reader.align`
     - the number of bits skipped

:meth:`~Reader.read_value` and :meth:`~Reader.read_values` mirror
:meth:`Tibs.to_value` and :meth:`Tibs.to_values`, and differ from each other in
exactly the same way: one value or many, a tuple or a list::

    >>> r = Reader(Tibs('0x0102030405'))
    >>> r.read_value('[u8; 2]')
    (1, 2)
    >>> r.read_values('u8', 2)
    [3, 4]

With no count, :meth:`~Reader.read_values` reads as many whole values as fit and
leaves any partial one under the cursor, where :meth:`Tibs.to_values` would
raise::

    >>> r = Reader(Tibs('0b1111111111'))   # ten bits
    >>> r.read_values('u4')
    [15, 15]
    >>> r.remaining
    2


Where the cursor is
^^^^^^^^^^^^^^^^^^^

:attr:`~Reader.pos` is the bit position, and can be set to anything from ``0``
to the length of the source. :attr:`~Reader.byte_pos` is the same thing in
bytes, and refuses to answer if the cursor is not byte aligned::

    >>> r = Reader(Tibs('0x0102030405'))
    >>> r.byte_pos = 3
    >>> r.pos, r.remaining, r.at_end
    (24, 16, False)

A ``Reader`` has no length of its own: :attr:`~Reader.remaining` is what is left
to read, and ``len(reader.source)`` is how long the whole thing is. Test for the
end with :attr:`~Reader.at_end` rather than ``while reader``, as a reader that
has been read to the end is still a perfectly ordinary object::

    >>> r = Reader(Tibs('0xff'), 8)
    >>> r.at_end, r.remaining, len(r.source)
    (True, 0, 8)

:meth:`~Reader.align` moves forward to the next boundary and says how far it
went, which is how you skip padding without counting it::

    >>> r = Reader(Tibs('0x4142'))
    >>> r.read_bits(3)                     # a 3-bit header, then padding
    Tibs('0b010')
    >>> r.align()
    5
    >>> r.byte_pos
    1

The boundary defaults to 8, but any positive number works, so 16-bit and 32-bit
alignment come for free.


Searching from the cursor
^^^^^^^^^^^^^^^^^^^^^^^^^

The seeks look for a bit pattern at or after the cursor and move to it. They
report a miss as ``False`` rather than raising, because looking for something
that turns out not to be there is a normal outcome::

    >>> r = Reader(Tibs('0x0000ff12'))
    >>> r.seek_to('0xff')
    True
    >>> r.pos
    16

``to`` leaves the cursor at the start of the match; ``past`` leaves it just
after. That distinction decides which one you can loop on: a needle already under
the cursor is found where it is, so ``while r.seek_to(x)`` never gets anywhere,
while :meth:`~Reader.seek_past` always makes progress::

    >>> r = Reader(Tibs('0x00ff00ff00'))
    >>> starts = []
    >>> while r.seek_past('0xff'):
    ...     starts.append(r.pos - 8)
    ...
    >>> starts
    [8, 24]

:meth:`~Reader.seek_back_to` scans the other way, considering only matches that
end at or before the cursor, so a loop over it also terminates.

The reading forms return what they passed over. :meth:`~Reader.read_to` stops
where the match begins, and :meth:`~Reader.read_past` swallows it::

    >>> r = Reader(Tibs('0x0000ff12'))
    >>> r.read_to('0xff')
    Tibs('0x0000')
    >>> r.pos
    16

All five take ``byte_aligned`` and ``mask``, exactly as :meth:`Tibs.find` does.
There is no ``find`` on a ``Reader``: use the seeks, or :meth:`Tibs.find` on the
source.


Looking ahead
^^^^^^^^^^^^^

:meth:`~Reader.peek_value` and :meth:`~Reader.peek_bits` read without moving,
which covers the usual case of dispatching on a tag before deciding how to read
the rest::

    >>> r = Reader(Tibs('0x02c0ffee'))
    >>> if r.peek_value('u8') == 2:
    ...     kind, payload = r.read_value('(u8, u24)')
    ...
    >>> hex(payload)
    '0xc0ffee'

For anything larger than a single value, :meth:`~Reader.bookmark` gives a
context manager that puts the cursor back on the way out, whether the block
finished or raised::

    >>> r = Reader(Tibs('0x010203'))
    >>> with r.bookmark():
    ...     r.read_values('u8')
    ...
    [1, 2, 3]
    >>> r.pos
    0

The :class:`Bookmark` saves the position when the block is entered, not when it
was made, so one kept in a variable restores whatever each block started from::

    >>> saved = r.bookmark()
    >>> r.read_value('u8')
    1
    >>> with saved:
    ...     r.read_values('u8')
    ...
    [2, 3]
    >>> r.pos
    8

To keep a position *and* go on reading past it, take a copy of the reader
instead. The copy shares the source and starts where the original is, so the
two cursors move independently over the same bits::

    >>> import copy
    >>> ahead = copy.copy(r)
    >>> ahead.read_values('u8')
    [2, 3]
    >>> r.pos
    8


When things go wrong
^^^^^^^^^^^^^^^^^^^^

Reads raise and searches return ``False``: running out of bits means the value
you asked for was not there, while a search that finds nothing is a normal
outcome::

    >>> r = Reader(Tibs('0x0f'), 4)
    >>> r.read_value('u8')
    Traceback (most recent call last):
        ...
    tibs.ReadError: Cannot read 8 bits at position 4: only 4 of the 8 bits are left.

Whenever a method raises, the cursor is exactly where it was, so there is nothing
to unwind::

    >>> r.pos
    4
    >>> r.remaining
    4


Reading a Mutibs
^^^^^^^^^^^^^^^^

A ``Reader`` over a :class:`Mutibs` reads it live, so bits appended after the
reader was built are there to be read::

    >>> m = Mutibs('0x01')
    >>> r = Reader(m)
    >>> r.read_value('u8')
    1
    >>> r.at_end
    True
    >>> m += '0x0203'
    >>> r.at_end
    False
    >>> r.read_values('u8')
    [2, 3]

:attr:`~Reader.pos` is checked against the length the source has at the time it
is assigned, so when you are writing ahead of the cursor, append first and then
move. Reads always return a :class:`Tibs`, even from a ``Mutibs``.
