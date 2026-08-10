.. currentmodule:: tibs

Tibs
----

The Tibs class is an immutable container for binary data.
The class's methods are detailed below. Python protocol methods do not render
especially well through autodoc, so they are summarized here:

* ``[]`` indexes bits as ``bool`` values and slices as ``Tibs`` instances.
* ``len()`` gives the length in bits, so an empty ``Tibs`` is the only falsy one.
* ``in`` searches for a promoted bit pattern.
* ``==`` and ``!=`` compare by bit value. There is no ordering comparison.
* ``+`` concatenates promoted bit sequences, and ``*`` repeats.
* ``<<`` and ``>>`` shift left and right, keeping the length.
* ``&``, ``|`` and ``^`` are bitwise AND, OR and XOR; ``~`` inverts every bit.
* ``hash()`` is by bit value, so a ``Tibs`` works as a dict key or set member.
* ``str()`` and ``format()`` render the bits — see :doc:`formatting`.
* ``bytes()``, ``memoryview()``, ``copy.copy()`` and ``pickle`` are all supported.


.. autoclass:: tibs.Tibs
   :members:
   :member-order: groupwise
   :undoc-members:
