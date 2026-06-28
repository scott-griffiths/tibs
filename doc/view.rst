.. currentmodule:: tibs

View and MutableView
--------------------

Views wrap bit data with interpretation settings such as byte order and bit
order. They do not change the meaning of normal ``Tibs`` or ``Mutibs`` indexing;
they only affect conversions such as integer, float, bytes and labelled field
access.

Python protocol methods are intentionally small for views: ``len(view)`` returns
the viewed bit length, ``bytes(view)`` is equivalent to ``view.to_bytes()``,
``repr(view)`` shows the source and layout, and equality compares source value
and layout. Views do not support indexing or slicing; use the source object or
the labelled :meth:`View.field` APIs instead.

See :doc:`views` in the manual for more details.

View
^^^^

.. autoclass:: tibs.View
   :members:
   :member-order: groupwise
   :undoc-members:

MutableView
^^^^^^^^^^^

.. autoclass:: tibs.MutableView
   :members:
   :member-order: groupwise
   :undoc-members:
