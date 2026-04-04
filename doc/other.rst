.. currentmodule:: tibs

Enums
-----

The ``bit_indexing`` property of instances of both ``Tibs`` and ``Mutibs`` is used to specify
in which order bits are indexed. The default is MSB0 where the most significant bit is given an
index of 0 (Most Significant Bit 0). This is the usual case where the left-most bit is considered the first.

The alternative is LSB0 (Least Significant Bit 0) where the right-most bit is bit zero, which is used in some
binary format specifications.

.. autoclass:: tibs.BitIndexing
   :members:
   :member-order: groupwise
   :undoc-members:



.. autoclass:: tibs.Endianness
   :members:
   :member-order: groupwise
   :undoc-members: