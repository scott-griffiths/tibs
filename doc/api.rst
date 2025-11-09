.. currentmodule:: bitformat
.. _api:

API Reference
=============

This reference guide is mostly built from the documentation strings in the library, but some introductory notes are also
included in each section.

For a more structured introduction to the library see the `Tour of bitformat <https://nbviewer.org/github/scott-griffiths/bitformat/blob/main/doc/bitformat_tour.ipynb>`_.


The Basics
----------
The :class:`Bits` and :class:`Dtype` classes are the most fundamental ones to use and understand.

* :ref:`Bits <bits>` -- An immutable container for storing binary data.
* :ref:`MutableBits <mutable_bits>` -- A mutable version of :class:`Bits`.
* :ref:`Dtype <dtype>` -- An abstract base class for data types used to interpret binary data.
* :ref:`DtypeSingle <dtypesingle>` -- A :class:`Dtype` representing a single value.
* :ref:`DtypeArray <dtypearray>` -- A :class:`Dtype` representing a sequence of objects of the same type.
* :ref:`DtypeTuple <dtypetuple>` -- A :class:`Dtype` representing a sequence of objects of different types.
* :ref:`Array <array>` -- A mutable container for contiguously allocated objects with the same :class:`Dtype`.
* :ref:`Reader <reader>` -- Read and parse :class:`Bits` as a bit stream with a bit position.

.. mermaid::

    ---
    title: Basic bitformat classes
    config:
        class:
            hideEmptyMembersBox: true
    ---
    classDiagram
        direction BT
        class MutableBits {
            + append()
            + byte_swap()
            + replace()
            + ...()
        }
        class Bits {
            + from_string()
            + from_dtype()
            + from_bytes()
            + ...()
            + count()
            + find()
            + to_bytes()
            + unpack()
            + ...()
        }
        class Reader {
            + Bits bits
            + int pos
            + read()
            + peek()
            + parse()
        }
        Reader --* "1" Bits : contains
        MutableBits --* Bits : extends


.. mermaid::

    ---
    title: Dtypes and Array
    config:
        class:
            hideEmptyMembersBox: true
    ---
    classDiagram
        direction BT
        class MutableBits
        class Array {
            + int item_size
            + Dtype dtype
            + MutableBits data
            + from_iterable()
            + from_bytes()
            + ...()
            + append()
            + as_type()
            + to_bytes()
            + unpack()
            + ...()
        }
        class Dtype {
            <<abstract>>
            + from_string()
            + from_params()
            + pack()
            + unpack()
        }
        class DtypeSingle {
            + DtypeKind kind
            + Endianness endianness
            + int size
        }
        class DtypeArray {
            + DtypeKind kind
            + Endianness endianness
            + int size
            + int items
        }
        class DtypeTuple {
            + List[Dtype] dtypes
        }
        DtypeSingle --|> Dtype
        DtypeArray --|> Dtype
        DtypeTuple --|> Dtype
        Array --* "1" Dtype : contains
        Array --* "1" MutableBits : contains


Field Types
-----------
These classes build upon those above to provide richer and more complex data structures.

* :ref:`FieldType <fieldtype>` -- The abstract base class for the other classes in this section.
* :ref:`Format <format>` -- A sequence of :class:`FieldType` objects, such as :class:`Field` or other :class:`Format` instances.
* :ref:`Field <field>` -- A well-defined amount of binary data with a single data type, and optionally a name.
* :ref:`If <if>` -- A pair of :class:`FieldType` obejcts, one of which is selected based on a condition.
* :ref:`Repeat <repeat>` -- Used to repeat another :class:`FieldType` a number of times.
* :ref:`Let <let>` - Creates or modifies a variable using an :class:`Expression`.
* :ref:`Pass <pass>` -- An empty :class:`FieldType` used as a placeholder.


.. mermaid::

    ---
    title: Field types
    config:
        class:
            hideEmptyMembersBox: true
    ---
    classDiagram
        direction BT
        class Bits
        class Dtype
        class FieldType{
            <<abstract>>
            + str name
        }
        class Field{
            + Dtype dtype
        }
        class If{
            + Expression condition
            + FieldType then_
            + FieldType else_
        }
        class Repeat{
            + int n
            + FieldType field
        }
        class Format{
            + List[FieldType] fields
        }
        Field --|> FieldType
        Format --|> FieldType
        If --|> FieldType
        Repeat --|> FieldType
        Field --* "1" Dtype : contains
        Field --* "0..1" Bits : contains


Miscellaneous
-------------
Other classes and enums are :ref:`listed here. <misc>` Some of the more important ones are:

* :ref:`Expression <expression>` -- Use Field values elsewhere for conditionals and other logic.
* :ref:`Options <options>` -- Getting and setting module-wide options.


-------------


.. toctree::
    :maxdepth: 1
    :hidden:

    dtypes
    bits
    mutable_bits
    array
    reader
    fieldtypes
    parser_ref
    misc

