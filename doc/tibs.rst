.. currentmodule:: tibs

Tibs
----

The Tibs class is an immutable container for binary data.
The classes methods are detailed below, but what's not listed (due to a bug using autodoc on pyo3 created classes) are
the special methods. These can be seen using the ``help()`` function in Python, but I'll briefly list them here too:

* ``[]`` : Slicing, e.g. ``s = t[4:16]``
* ``+``: Concatenation, e.g. ``s = t + '0b1'``
* ``*``: Concatenation of self, e.g. ``s = t * 100``
* ``<<``: Left bit shift, e.g. ``s = t << 3``
* ``>>``: Right bit shift, e.g. ``s = t >> 3``
* ``&``: Bit-wise AND
* ``|``: Bit-wise OR
* ``^``: Bit-wise XOR
* ``~``: Bit inversion


.. autoclass:: tibs.Tibs
   :members:
   :member-order: groupwise
   :undoc-members: