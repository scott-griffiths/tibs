.. currentmodule:: bitstring

.. _quick_reference:

Quick Reference
===============

This section gives a summary of the `bitformat` module's classes, functions and attributes.


Class Overview
--------------

Objects in `bitformat` that can be parsed and built are constructed from the `FieldType` objects that are listed in the table below.
Some of these types will usually be used as part of a larger structure, where they can be optionally constructed from strings - for these types a string example is given in the description.

.. |nbsp| unicode:: 0xa0
   :trim:

.. list-table::
   :header-rows: 1

   * - Class
     - String format?
     - Description

   * - :ref:`fieldtype_quick_reference`
     - n/a
     - The abstract base class for the other classes listed here.
   * - :ref:`field_quick_reference` (Single)
     - ✓
     - A single data type, with optional name and value.
   * -
     -
     - ``'name: dtype = value'``
   * - :ref:`field_quick_reference` (Array)
     - ✓
     - An array of a single fixed-length data type, with optional name and value.
   * -
     -
     - ``'name: [dtype; items] = value'``
   * - :ref:`format_quick_reference`
     - ×
     - A sequence of other FieldTypes, with optional name.
   * -
     -
     - ``Format([field1, field2, ...], 'name'])``
   * - :ref:`repeat_quick_reference`
     - ×
     - Repeat another field type a number of times, with optional name.
   * -
     -
     - ``Repeat(10, 'float64', 'a_name')``


.. _fieldtype_quick_reference:


FieldType
---------

A ``FieldType`` is an abstract base class for all of the other classes in this section.
It could represent a single piece of data, it could be a container for other `FieldType` objects or it could represent an action or decision.
You shouldn't need to deal with this type directly but its methods are available for all of the other field types.

Methods
^^^^^^^

* :meth:`~FieldType.build`: Given positional and keyword values, fill in the any empty field(s) and build a `Bits` object.
* :meth:`~FieldType.parse`: Takes a `Bits` object, parses it according to the field structure and returns the number of bits used.
* :meth:`~FieldType.flatten`: Removes any nesting of fields and returns a flat list of FieldsTypes.
* :meth:`~FieldType.to_bits`: Converts the contents to a `Bits` bit literal.
* :meth:`~FieldType.to_bytes`: Converts the contents to a `bytes` object.
* :meth:`~FieldType.vars`: Returns the positional and keyword values that are contained in the field.
* :meth:`~FieldType.clear`: Sets the `value` of everything that is not a marked as const to `None`.

Properties
^^^^^^^^^^
* :attr:`~FieldType.name`: A string that can be used to refer to the `FieldType`.
* :attr:`~FieldType.value`: A property to get and set the value of the field.


.. _field_quick_reference:

Field
-----

The `Field` is the fundamental building block in `bitformat`.
It represents a well-defined amount of binary data with a single data type.

``Field(dtype: Dtype | str, name: str = '', value: Any = None, const: bool = False)``

Additional methods
^^^^^^^^^^^^^^^^^^

* :meth:`~Field.from_bits`: Construct from ``bytes``, ``bytearray``, a ``Bits`` object or a string that can be used to construct a `Bits` object.
* :meth:`~Field.from_bytes`: Construct from a ``bytes`` or ``bytearray`` object.
* :meth:`~Field.from_string`: Construct from a formatted string to set the `dtype`, `name`, `value` and `const` parameters.

Additional properties
^^^^^^^^^^^^^^^^^^^^^

* :attr:`~Field.dtype`: The data type of the field.
* :attr:`~Field.const`: Whether the field is a constant bit literal.


.. _format_quick_reference:

Format
------

The `Format` type is central to creating useful objects in `bitformat`.
It contains a sequence of other `FieldType` objects, which can include nested `Format` objects.

``Format(fieldtypes: Sequence[FieldType | str], name: str = '')``

Additional methods
^^^^^^^^^^^^^^^^^^

* :meth:`~Format.flatten`: Returns a flat list of `FieldType` objects.
* :meth:`~Format.append`: Add a `FieldType` object to the end of the `Format`.
* :meth:`~Format.extend`: Add a sequence of `FieldType` objects to the end of the `Format`.

Special methods
^^^^^^^^^^^^^^^

* :meth:`~Format.__getitem__`: Get a `FieldType` object by index.
* :meth:`~Format.__setitem__`: Set a `FieldType` object by index.
* :meth:`~Format.__iadd__`: Add a `FieldType` object to the end of the `Format`.


.. _repeat_quick_reference:

Repeat
------

A `Repeat` field repeats another field type a given number of times.

``Repeat(count: int | Iterable[int] | str, fieldtype: FieldType | str | Dtype | Bits | None = None, name: str = '')``




