.. currentmodule:: tibs

Reader
------

A ``Reader`` pairs a :class:`Tibs` or :class:`Mutibs` with a bit position, so
that values can be read one after another without working out ``start`` and
``end`` for each one.

Every method is anchored at :attr:`~Reader.pos`, and no method takes ``start``
or ``end``. A windowed query stays on the wrapped object, which is always
reachable as :attr:`~Reader.source`.

See :doc:`reading` in the manual for how the pieces fit together.

.. autoclass:: tibs.Reader
   :members:
   :member-order: groupwise
   :undoc-members:
