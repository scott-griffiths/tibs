
.. currentmodule:: tibs

.. image:: tibs.png

A sleek Python library for your binary data

Introduction
------------

``tibs`` is a simple but powerful Python library for creating, interpreting and manipulating binary data.
It is 100% written in Rust, and from the same author as the bitstring library.

It's called 'tibs' as that's (more or less) 'bits' backwards, and I needed a distinctive name.

The two main classes are:

* `Tibs`: An immutable sequence of bits.
* `Mutibs`: A mutable sequence of bits.

They are created by class methods starting with ``from_``, for example::

    >>> a = Tibs.from_bin('0110')
    >>> b = Tibs.from_string('0xabc, 0b11')
    >>> c = Mutibs.from_bytes(b'some_byte_data')
    >>> d = Mutibs.from_random(1000)
    >>> e = Mutibs.from_joined([a, b, c, d])

and can be interpreted by methods starting with ``to_`` ::

    >>> a.to_hex()
    '6'
    >>> b.to_bin()
    '10101011110011'

You can do everything you'd expect with these classes - slicing, boolean operations, shifting, rotating, finding, replacing, setting, reversing etc.

The project is currently in alpha. For now, instead of a user manual, here are the auto-generated API docs.

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
