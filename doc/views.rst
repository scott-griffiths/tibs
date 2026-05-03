.. currentmodule:: tibs

.. note::

    This part of the documentation is under construction.
    For now, see the API docs for the most complete information.

Views
-----

A view is a wrapper around a `Tibs` which changes how values are interpreted.
This can change the byte endianness, or how the bit indices are numbered.

The simplest use of views is to change endianness, so let's take a look at that case first.

Endianness
^^^^^^^^^^

Byte-wise endianness is available for constructing various whole-byte values.
The endianness isn't a property of the ``Tibs``, but affect both how it's constructed from a value
and how it's reinterpreted as a value. ::

    >>> Tibs.from_u(1, 32, Endianness.Big)
    Tibs('0x00000001')
    >>> Tibs.from_u(1, 32, Endianness.Little)
    Tibs('0x01000000')

The default is ``Endianness.Unspecified`` which is bit-wise big endian. The difference between ``Unspecified``
and ``Big`` is that the latter will complain if it tries to construct or interpret a non whole-byte value. ::

If we take the little-endian value above and interpret it as an unsigned int, it will use the default endianness
and not give the value we expect::

    >>> t = Tibs('0x01000000')
    >>> t.to_u()
    16777216

To get the little-endian value we need to create a view::

    >>> v = t.view(Endianness.Little)
    >>> v.to_u()
    1

Instead of calling :meth:`Tibs.view` directly you can use the :attr:`Tibs.le` property, and combining this with
the :attr:`Tibs.u` property short-cut we get the more compact form ::

    >>> t.le.u
    1

This works when interpreting any whole-byte value. The big-endian view looks identical to the default
``Endianness.Unspecified`` for whole-byte values, but will raise a ``ValueError`` if used on non-whole-byte values.

    >>> m = Mutibs.from_f(1984.3, 64, Endianness.Little)
    >>> m
    Mutibs('0x3333333333019f40')
    >>> m.f
    4.667261455589845e-62
    >>> m.be.f
    4.667261455589845e-62
    >>> m.le.f
    1984.3
