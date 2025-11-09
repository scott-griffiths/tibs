.. currentmodule:: bitformat
.. _bits:

Bits
====

The ``Bits`` class represents an immutable sequence of bits, similar to how the built-in ``bytes`` is an immutable sequence of bytes, and a ``str`` is an immutable sequence of characters.

If you need to modify the data after creation there is also a :class:`MutableBits` class, but the more efficient :class:`Bits` should be preferred if the extra capabilities are not needed.

You can switch between :class:`Bits` and :class:`MutableBits` using the :meth:`Bits.to_mutable_bits` and :meth:`MutableBits.to_bits` methods.


----

.. autoclass:: bitformat.Bits
   :members:
   :member-order: groupwise
   :undoc-members:
   :inherited-members:
