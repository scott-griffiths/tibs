.. currentmodule:: bitformat
.. _mutable_bits:

MutableBits
===========

The ``MutableBits`` class is the mutable version of :class:`Bits`.

It has almost all of the methods and properties of :class:`Bits` and adds capabilities for changing the bits in-place.

The new methods are:

* :meth:`~MutableBits.append`
* :meth:`~MutableBits.byte_swap`
* :meth:`~MutableBits.clear`
* :meth:`~MutableBits.insert`
* :meth:`~MutableBits.invert`
* :meth:`~MutableBits.prepend`
* :meth:`~MutableBits.replace`
* :meth:`~MutableBits.reverse`
* :meth:`~MutableBits.rol`
* :meth:`~MutableBits.ror`
* :meth:`~MutableBits.set`

Some methods from :class:`Bits` that return a generator are not allowed as the underlying binary value could change
while the generator is still live. For example the :meth:`Bits.find_all` method is not available on ``MutableBits``, but you can use
``.to_bits().find_all()`` instead.

You can switch between :class:`MutableBits` and :class:`Bits` using the :meth:`MutableBits.to_bits` and :meth:`Bits.to_mutable_bits` methods.f

The :meth:`~MutableBits.reserve` and :meth:`~MutableBits.capacity` methods can be used to manage the capacity of the :class:`MutableBits` to avoid unneccesary reallocations, but their use is purely for performance optimization.

----

.. autoclass:: bitformat.MutableBits
   :members:
   :member-order: groupwise
   :undoc-members:
   :inherited-members:
