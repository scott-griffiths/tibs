.. currentmodule:: tibs

View and MutableView
--------------------

Views wrap bit data with interpretation settings such as byte order and bit
order. They do not change the meaning of normal ``Tibs`` or ``Mutibs`` indexing;
they only affect conversions such as integer, float, bytes and labelled field
access.

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
