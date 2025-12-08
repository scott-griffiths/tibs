.. currentmodule:: tibs


Getting started
---------------

To install use ::

    pip install tibs


There are pre-built wheels for most configurations - if there are issues then please let me know.
Tibs works with Python 3.8 and later.

One way to get to know the library is to start a Python interactive session, import the two main
classes, and experiment with some of the example code in the rest of this document. ::

    >>> from tibs import Tibs, Mutibs


A quick tour
------------

The two main classes are:

* :class:`Tibs`: An immutable sequence of bits.
* :class:`Mutibs`: A mutable sequence of bits (pronounced 'mew-tibs').

They are created by class methods starting with ``from_``, for example::

    >>> a = Tibs.from_bin('0110')
    >>> b = Tibs.from_hex('abc')
    >>> c = Tibs.from_string('0xfee, 0b11001')
    >>> d = Tibs.from_bytes(b'some_byte_data')
    >>> e = Tibs.from_random(1000)  # 1000 random bits
    >>> f = Tibs.from_u(76, 25)  # Unsigned int stored in 25 bits
    >>> g = Tibs.from_f(-0.125, 16)  # A float stored in 16 bits
    >>> h = Tibs.from_bools([1, 0, 0])
    >>> i = Tibs.from_joined([a, b, c, d, e, f, g, h])

Once created they are just binary data, stored efficiently, and they don't retain any information about how they were created.

The ``Tibs`` constructor can also be used to create new instances, and it will automatically delegate to :meth:`~Tibs.from_string`, :meth:`~Tibs.from_bytes` or :meth:`~Tibs.from_bools`.
This is often more convenient::

    >>> a = Tibs('0b0110')
    >>> b = Tibs('0xabc')
    >>> c = Tibs('0xfee, 0b11001')
    >>> d = Tibs(b'some_byte_data')
    >>> h = Tibs([1, 0, 0])

Anything that works in the constructor can also be used in other places where a ``Tibs`` is needed.
For example, instead of writing ::

    x = b & Tibs.from_hex('0xff0')
    if x.starts_with(Tibs.from_bin('0b11')):
        x += Tibs.from_bools([0, 1, 1])

you can write just ::

    x = b & '0xff0'
    if x.starts_with('0b11'):
        x += [0, 1, 1]


Note that the binary and hex strings need the ``0b`` and ``0x`` prefixes when not called via :meth:`~Tibs.from_bin` and :meth:`~Tibs.from_hex`.

To get the data out of the ``Tibs`` there are similar methods starting with ``to_``::

    >>> a.to_bin()
    '0110'
    >>> b.to_hex()
    'abc'
    >>> d.to_bytes()
    b'some_byte_data'
    >>> f.to_u()
    76
    >>> g.to_f()
    -0.125

There isn't a ``to_bools`` method, but creating a ``list`` from the ``Tibs`` instance will have the same effect.
You can also use ``Tibs`` instances as iterators of bits.

Instances of ``Tibs`` are immutable, so once created they can't change in value, much like the Python ``bytes`` and ``str`` types.
This allows them to be hashed, stored in sets, used as dictionary keys etc., and also allows various optimizations to be used to make them more efficient.
They should be used by default if values don't need to be changed.

This means that the standard pieces of advice for working with things like Python strings do apply, and why something like this line::

    i = Tibs()
    for t in [a, b, c, d, e, f, g, h]:
        i += t  # NOT RECOMMENDED!

is an anti-pattern to avoid, as it will create a new instance every time it appends. Use :meth:`Tibs.from_joined` instead.

For the times when you do need a mutable container use :class:`Mutibs`.
This can do almost everything that ``Tibs`` can do, except that it's not hashable, so can't be used as a dictionary key, in sets etc.
It also has several extra methods that will mutate the value in-place. ::

    >>> m = Mutibs()
    >>> m.append('0xabde')
    Mutibs('0xabde')
    >>> m.replace([1], [0, 1, 0])
    Mutibs('0b01000100010001001001001000100100100100')


Note that mutating methods like :meth:`~Mutibs.append` and :meth:`~Mutibs.replace` also return the modified ``Mutibs`` instance.
This perhaps isn't the most Pythonic of interfaces, but it allows methods to be chained::

    >>> m[:32].byte_swap().reverse().to_f()
    2.1993814317305072e-18


You can do everything you'd expect with these classes - slicing, boolean operations, shifting, rotating, finding, replacing, setting, reversing etc.



But why is it called tibs?
^^^^^^^^^^^^^^^^^^^^^^^^^^

Because Tibs is Bits backwards (almost), it's distinctive, and the name was available on PyPI.

It's got nothing to do with Ethiopian stew.

