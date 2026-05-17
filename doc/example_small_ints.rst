.. currentmodule:: tibs


Packing small integers
----------------------

If every value is known to fit in a small fixed width, a ``Tibs`` can store the
values densely without allocating a Python object per value. This is useful for
compact lookup tables, quantized levels, or small counters.

.. literalinclude:: ../examples/small_ints.py
   :language: python

The packed value in this example is only 30 bits long. It does not need to be a
whole number of bytes until you choose to serialize it.
