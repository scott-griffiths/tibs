.. currentmodule:: tibs

API
---

The API docs are generated from the docstrings, so are also available via the ``help()``
function in a Python interpreter session.

The public API is grouped into:

* :doc:`tibs` — An immutable sequence of bits.
* :doc:`mutibs` — A mutable sequence of bits.
* :doc:`dtype` — Fixed-width data type descriptions used for packing and
  unpacking values.
* :doc:`view` — Immutable and mutable interpretation wrappers for byte order,
  bit order and labelled fields.
* :doc:`other` — Enum classes for byte order, bit order, dtypes and codecs.

.. toctree::
    :maxdepth: 1
    :hidden:

    tibs
    mutibs
    dtype
    view
    other
