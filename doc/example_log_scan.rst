.. currentmodule:: tibs


Scanning a binary log
---------------------

When a file or stream uses a byte-aligned sync marker, :meth:`Tibs.find_all_iter`
can scan for candidate records without first splitting the data into Python
bytes. The surrounding fields can then be sliced directly from the bit sequence.

.. literalinclude:: ../examples/log_scan.py
   :language: python

Using ``byte_aligned=True`` matters here: it avoids treating the same bit pattern
inside a payload as a sync marker unless it starts on a byte boundary.
