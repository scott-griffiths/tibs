.. currentmodule:: tibs


Reading a record stream
-----------------------

When records vary in both kind and length, there is no ``start`` and ``end`` to
compute ahead of time: where the next record begins depends on what the last one
turned out to be. A :class:`Reader` keeps that position, so the loop only has to
decide what it is looking at.

.. literalinclude:: ../examples/record_stream.py
   :language: python

:meth:`Reader.peek_value` reads the tag without consuming it, so the record can
be read whole once its shape is known. :meth:`Reader.seek_past` skips the noise
before the sync word, and :meth:`Reader.align` steps over the padding in front
of a byte-aligned name — both report where they got to rather than leaving the
caller to add it up.
