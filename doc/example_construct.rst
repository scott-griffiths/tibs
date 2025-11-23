.. currentmodule:: tibs


Constructing from a format
--------------------------

Let's say you have a binary format specification and you'd like to generate data from it.

.. literalinclude:: ../examples/construct.py
   :language: python

While this is a useful example, using Tibs for this doesn't scale well to complex formats so I'd recommend
a more specialist library.
See for example construct, or the bitformat and bitstring libraries which both use tibs internally.
