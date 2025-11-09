.. currentmodule:: bitformat
.. _misc:

Miscellaneous
=============

A collection of items that don't fit into any other category.

* :class:`Expression`: Allows values parsed to be used in an f-string like way elsewhere in a :class:`Format`.
* :class:`Options`: Used to set and get global options for the module.
* :class:`Register`: Stores information about what data types are supported.
* :class:`DtypeKind`: The kind of data type.
* :class:`Endianness`: The endianness of the data type.


----

.. _expression:

.. autoclass:: bitformat.Expression
   :members:
   :undoc-members:
   :member-order: groupwise

----

.. _options:

.. autoclass:: bitformat.Options
   :members:
   :undoc-members:
   :member-order: groupwise

----

.. _register:

.. autoclass:: bitformat.Register
   :members:
   :undoc-members:
   :member-order: groupwise

----

.. autoenum:: bitformat.DtypeKind
   :members:

----

.. autoenum:: bitformat.Endianness
   :members:
