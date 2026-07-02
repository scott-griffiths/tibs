.. currentmodule:: tibs

Tibs
----

The Tibs class is an immutable container for binary data.
The class's methods are detailed below. Python protocol methods do not render
especially well through autodoc, so they are summarized here:

* ``[]`` indexes bits as ``bool`` values and slices as ``Tibs`` instances.
* ``in`` searches for a promoted bit pattern.
* ``==`` and ``!=`` compare by bit value.
* ``+`` concatenate promoted bit sequences.
* ``*`` repeat the bit sequence.
* ``<<``: Left bit shift, e.g. ``s = t << 3``
* ``>>``: Right bit shift, e.g. ``s = t >> 3``
* ``&`` bitwise AND.
* ``|`` bitwise OR.
* ``^`` bitwise XOR.
* ``~`` inverts every bit.


.. autoclass:: tibs.Tibs
   :members:
   :member-order: groupwise
   :undoc-members:
