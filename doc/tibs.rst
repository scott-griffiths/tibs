.. currentmodule:: tibs

Tibs
----

The Tibs class is an immutable container for binary data.
The class's methods are detailed below. Python protocol methods do not render
especially well through autodoc, so they are summarized here:

* ``Tibs(...)`` promotes strings, bytes-like objects and bool iterables to a ``Tibs``.
* ``str(t)`` and ``repr(t)`` return compact string representations.
* ``len(t)`` returns the bit length.
* ``bytes(t)`` is equivalent to :meth:`Tibs.to_bytes`.
* ``copy.copy(t)`` returns the same immutable instance.
* ``for bit in t`` iterates over Python ``bool`` values.
* ``[]`` indexes bits as ``bool`` values and slices as ``Tibs`` instances.
* ``in`` searches for a promoted bit pattern.
* ``==`` and ``!=`` compare by bit value, and ``Tibs`` instances are hashable.
* ``+`` and reversed ``+`` concatenate promoted bit sequences.
* ``*`` and reversed ``*`` repeat the bit sequence.
* ``<<``: Left bit shift, e.g. ``s = t << 3``
* ``>>``: Right bit shift, e.g. ``s = t >> 3``
* ``&`` and reversed ``&`` perform bitwise AND.
* ``|`` and reversed ``|`` perform bitwise OR.
* ``^`` and reversed ``^`` perform bitwise XOR.
* ``~`` inverts every bit.


.. autoclass:: tibs.Tibs
   :members:
   :member-order: groupwise
   :undoc-members:
