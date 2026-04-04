.. currentmodule:: tibs

Creation, Views and Interpretations
-----------------------------------

Creation
^^^^^^^^

Tibs and Mutibs can be constructed from a number of different types. Their constructors are identical, so I'll
use Tibs in this section, but it all applies equally well to Mutibs.

Several ``from_`` constructor methods are provided:

* :meth:`Tibs.from_bin`
* :meth:`Tibs.from_oct`
* :meth:`Tibs.from_hex`
* :meth:`Tibs.from_bytes`
* :meth:`Tibs.from_string`
* :meth:`Tibs.from_bools`
* :meth:`Tibs.from_zeros`
* :meth:`Tibs.from_ones`
* :meth:`Tibs.from_random`
* :meth:`Tibs.from_u`
* :meth:`Tibs.from_i`
* :meth:`Tibs.from_f`
* :meth:`Tibs.from_joined`


The ``__init__`` method can also be called directly, which is often more convenient, if ever so slightly slower.
This will look at the type of object its been given and try to promote it to a Tibs by delegating to :meth:`Tibs.from_string`,
:meth:`Tibs.from_bools` or :meth:`Tibs.from_bytes` for strings, iterables and bytes types respectively.
So for example ::

    s = Tibs('0xabc')     # Same as Tibs.from_string('0xabc')
    t = Tibs([1, 0, 1])   # Same as Tibs.from_bools([1, 0, 1])
    u = Tibs(b'hello')    # Same as Tibs.from_bytes(b'hello')

The automatic promotion of these types to Tibs is quite pervasive in the library. Roughly speaking, anywhere that
requires a Tibs will also accept another type it can promote in this way. So, for example, if you want to count
how many times the bit pattern `101` appears in a random bit sequence you could write::

    t = Tibs.from_random(1_000_000)  # A million random bits
    c = t.count(Tibs.from_bools([1, 0, 1]))

but it's more natural to use automatic promotion ::

    c = t.count([1, 0, 1])


Data views
^^^^^^^^^^

When a Tibs has been created there are multiple ways to interpret the data. These methods start with ``to_``.

A subset of these methods are simply views on the data, and they either convert to a string or a bytes object.

* :meth:`Tibs.to_bin()` / :attr:`Tibs.bin`. Converts to a string of ``0`` and ``1`` characters. Always available.
* :meth:`Tibs.to_oct()` / :attr:`Tibs.oct`. Converts to an octal string. Length must be a multiple of 3.
* :meth:`Tibs.to_hex()` / :attr:`Tibs.hex`. Converts to a hexadecimal string. Length must be a multiple of 4.
* :meth:`Tibs.to_bytes()` / :attr:`Tibs.bytes`. Converts to a ``bytes`` object. Length must be a multiple of 8.

The ``to_`` methods here don't accept any parameters, so read-only properties are provided as a convenient alias.

When you have a view, you can always reconstruct the original Tibs - there is a 1:1 relationship.
So ``t == Tibs.from_bin(t.to_bin())`` will always be true.


Data interpretations
^^^^^^^^^^^^^^^^^^^^

Unlike the data views, the interpretations can have many-to-one relationships in both directions.
For example there are many ways for a Tibs to be constructed from the unsigned integer 3::

    u1 = Tibs.from_u(3, 5)   # binary 00011
    u2 = Tibs.from_u(3, 16)  # binary 00000000_00000011
    u3 = Tibs.from_u(3, 16, Endianness.Little)  # binary 00000011_00000000

These are three different Tibs, but they all can have equal interpretations::

    u1.to_u() == u2.to_u() == u3.to_u(Endianness.Little) == 3  # True