.. currentmodule:: tibs

User Manual
-----------

.. note::

    This part of the documentation is under construction.
    For now, see the API docs for the most complete information.

``tibs`` is a Python library for creating, manipulating and interpreting binary data.
It's designed to be light-weight and easy to use, and is written in Rust for efficiency.

The two classes available in the tibs module are:

* :doc:`tibs` — An immutable sequence of bits.
* :doc:`mutibs` — A mutable sequence of bits.

These classes efficiently hold arbitrary length binary data; they can be constructed
from bytes, ints, floats, binary and hex strings, random data, and bools. They can then be
sliced, indexed, searched, replaced, rotated, reversed, byte-swapped, set, unset,
appended, extended, counted, concatenated, cut into chunks, and joined back together,
before being interpreted as bytes, ints, floats, binary and hex strings, and bools.


.. toctree::
    :maxdepth: 1
    :hidden:

    creation
    manipulation
    tibs_vs_mutibs
    misc