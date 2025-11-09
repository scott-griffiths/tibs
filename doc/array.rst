.. currentmodule:: bitformat
.. _array:

Array
=====

The :class:`Array` class is used as a container for contiguously allocated `Bits` objects with the same `Dtype`.

``Array`` instances act very like an ordinary Python array, but with each element being a fixed-length dtype.
They are mutable, so values can be changed after creation.

----

.. autoclass:: bitformat.Array
   :members:
   :member-order: groupwise
   :undoc-members: