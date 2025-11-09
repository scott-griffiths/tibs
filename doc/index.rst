
.. currentmodule:: bitformat

.. image:: bitformat_logo.png
   :width: 400px

A Python library for creating and parsing binary formats.

----

* Efficiently store and manipulate binary data in idiomatic Python using the :ref:`bits`, :ref:`mutable_bits` and :ref:`array` classes.
* Powerful binary packing and unpacking functions.
* Bit-level slicing, joining, searching, replacing and more.
* A wide array of data types is supported.  Want to use a 13-bit integer or a 16-bit float? Fine - there are no special hoops to jump through.
* Several field types are available to build up a :class:`Format`, which can then be used to :meth:`~Format.pack`, :meth:`~Format.unpack` or :meth:`~Format.parse` data:

  * The simplest is just a :class:`Field` which contains an optionally named value with a data type.
  * A :class:`Format` contains a list of other fields. These can be nested to any depth.
  * Field types like :class:`Repeat` and :class:`If` can be used to add more logical structure.
* The values of fields can be used in later calculations via an f-string-like expression syntax.
* Data is always stored efficiently as a contiguous array of bits, with the core of the library written in Rust.


It is from the author of the `bitstring <https://github.com/scott-griffiths/bitstring>`_ library.

----

Documentation
-------------

The introduction covers installation and a few brief examples, but doesn't go into much depth.

.. toctree::
    :maxdepth: 2

    introduction

We then explore many of the features of the library via some common use cases.

.. toctree::
    :maxdepth: 2

    use_cases

The API reference is more comprehensive and is mostly generated from the docstrings in the library. Some introductory notes are included in each section.

.. toctree::
    :maxdepth: 2

    api


There is also a notebook with a tour of the features of bitformat:

* `A Tour of bitformat <https://nbviewer.org/github/scott-griffiths/bitformat/blob/main/doc/bitformat_tour.ipynb>`_


----

.. raw:: html

   <style>
       .small-font {
           font-size: 0.9em;
       }
   </style>
   <div class="small-font">
       These docs are styled using the <a href="https://github.com/piccolo-orm/piccolo_theme">Piccolo theme</a>.
   </div>
