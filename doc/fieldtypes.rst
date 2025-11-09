.. currentmodule:: bitformat
.. _fieldtypes:

FieldTypes
==========

The :class:`FieldType` class is an abstract base class for 'field-like' types that must support some basic operations to be used in a :class:`Format`.

You shouldn't usually need to deal with this type directly, but its methods are available for all of the other field types.

The concrete field-type classes are:

* :class:`Field`: An amount of binary data with a single data type, and optionally a name.
* :class:`Format`: A sequence of :class:`FieldType` objects, such as :class:`Field` or other :class:`Format` instances.
* :class:`If`: Uses a condition to choose between a pair of :class:`FieldType` objects.
* :class:`Repeat`: Used to repeat another :class:`FieldType` a number of times.
* :class:`Let`: Creates or modifies a variable.
* :class:`Pass`: An empty :class:`FieldType` object, used as a placeholder.


----

.. _fieldtype:

.. autoclass:: bitformat.FieldType
   :members:
   :undoc-members:
   :member-order: groupwise
   :show-inheritance:

----

.. _field:

.. autoclass:: bitformat.Field
   :members:
   :inherited-members:
   :undoc-members:
   :member-order: groupwise
   :show-inheritance:

----

.. _format:

.. autoclass:: bitformat.Format
   :members:
   :undoc-members:
   :inherited-members:
   :member-order: groupwise
   :show-inheritance:

----

.. _if:

.. autoclass:: bitformat.If
   :members:
   :undoc-members:
   :member-order: groupwise
   :show-inheritance:


----

.. _repeat:

.. autoclass:: bitformat.Repeat
   :members:
   :undoc-members:
   :member-order: groupwise
   :show-inheritance:

----

.. _let:

.. autoclass:: bitformat.Let
   :members:
   :undoc-members:
   :member-order: groupwise
   :show-inheritance:

----

.. _pass:

.. autoclass:: bitformat.Pass
   :members:
   :undoc-members:
   :member-order: groupwise
   :show-inheritance:
