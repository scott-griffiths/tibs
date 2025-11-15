
.. currentmodule:: tibs

.. image:: tibs.png

A fast and sleek bit library

Introduction
------------

The two main classes are:

* `Tibs`: An immutable sequence of bits.
* `Mutibs`: A mutable sequence of bits.

They are created by class methods starting with ``from_``, for example::

    >>> a = Tibs.from_bin('0110')
    >>> b = Tibs.from_string('0xabc, 0b11')
    >>> c = Tibs.from_bytes(b'some_byte_data')
    >>> d = Mutibs.from_random(1000)
    >>> e = Mutibs.from_joined([a, b, c, d])

and can be interpreted by methods starting with ``to_`` ::

    >>> a.to_hex()
    '6'
    >>> b.to_bin()

API
---

.. autoclass:: tibs.Tibs
   :members:
   :special-members: __and__, __or__, __xor__, __invert__, __add__, __eq__, __str__, __iter__, __hash__, __len__,
    __contains__, __getitem__, __lshift__, __rshift__, __bytes__, __mul__
   :member-order: groupwise
   :undoc-members:

.. autoclass:: tibs.Mutibs
   :members:
   :special-members: __and__, __or__, __xor__, __invert__, __add__, __eq__, __str__, __iter__, __len__,
    __contains__, __getitem__, __lshift__, __rshift__, __bytes__, __mul__, __setitem__, __delitem__, __iadd__,
    __ilshift__, __irshift__, __iand__, __ior__, __ixor__, __imul__
   :member-order: groupwise
   :undoc-members:

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
