
.. currentmodule:: bitformat
.. _bitstring_comparison:

Comparison with bitstring
==========================

``bitformat`` vs. ``bitstring``

bitformat is from the same author as the `bitstring <https://github.com/scott-griffiths/bitstring>`_ package, which is widely used and has been actively maintained since 2006.
It covers much of the same ground, but is designed to have a stronger emphasis on performance, a simpler API and a more expressive syntax for binary formats.

``bitstring``

* Simple and flexible syntax for binary data manipulation.
* Reasonable performance, but difficult to improve further.
* Very mature and stable - maintained since 2006.
* Hundreds of dependant projects and millions of downloads per month.


``bitformat``

* Expressive syntax for complex binary formats.
* Emphasis on performance.
* Several major features still to be added.
* In beta stage - mostly complete, but still in development.

I am hoping that ``bitformat`` will become a worthy successor, but even if ``bitformat`` is successful I plan to support ``bitstring`` indefinitely - at the time of writing their respective download counts are 88 million for bitstring and 882 for bitformat!

There are many similarities between bitformat and bitstring, but there has been no attempt to make them compatible.
Much of the reason for making a new package was to revisit many of the design decisions that were made almost two
decades ago when George W. Bush was president, the Nintendo Wii was the latest must-have tech, and Python 2.4 was the latest version.

Below is a non-exhaustive list of some of the differences between the two packages.

Classes
-------

``bitstring`` uses a class hierarchy with the base ``Bits`` being immutable with no bit position, and then having the ``BitArray``, ``ConstBitStream`` and ``BitStream`` use it as a base class to add mutating methods and a bit position.

In ``bitformat`` there are just the :class:`Bits` class, which is immutable, and :class:`MutableBits` which adds methods such as :meth:`~MutableBits.insert` and :meth:`~MutableBits.replace`.
Using :class:`Bits` when possible allows for many simplications and efficiencies in the code when it knows that once the object is created it will never change.

As bit positions can be very useful, a :class:`Reader` class has been provided in ``bitformat`` which wraps a :class:`Bits` and provided reading methods.

The way that data types are dealt with is more sophisticated in ``bitformat`` than in ``bitstring``. In ``bitformat`` the :class:`Dtype` class is a base class that can be used to create the :class:`DtypeSingle`, :class:`DtypeArray` and :class:`DtypeTuple` classes. The dtypes in ``bitstring`` are really just the ``DtypeSingle`` used in ``bitformat``.

The :class:`Array` classes in ``bitformat`` and ``bitstring`` are broadly similar.

Performance
-----------

The ``bitstring`` library was pure Python for a very long time, but eventually switched to using the ``bitarray`` package to do lots of the lowest level bit manipulation. This is an external package written in C that is very fast, but ultimately the speed of ``bitstring`` is limited by what methods are available in the ``bitarray`` package.

``bitformat`` has a custom backend written in Rust, which has the potential to be much more performant as it can be tailored to the exact needs and design of the library. Currently the ``bitformat`` performance mostly matches or exceeds that of ``bitarray``, with some significant exceptions that I'm working on! Much of the fundamental design of ``bitformat`` is led by an understanding of why it was hard to make ``bitstring`` any faster.

Fields and Formats
------------------

I had originally planned to make the addition of more descriptive fields and formats a new feature of ``bitstring``.
It was quite a big new addition though, and I realised that I could instead make a new package that could use ``bitstring`` as a dependency and then add the new features on top of that.
This would allow the development to be less constrained, as one issue with developing a very mature library is that there are lots of people who become quite vocal if you make a change that breaks their code!
A new library could run fast and break the API as much as it wanted. If things worked out then perhaps the new features could be added to ``bitstring`` at a later date.

What I found when I started with this approach was that I was continually wanting to rewrite parts of ``bitstring``, updating design decisions that were made by a much younger version of myself working with a much older version of Python. I ended up importing a large chunk of the bitstring code and rewriting so much of it that it became clear I was writing a replacement (or competitor) rather than an extension.

The :class:`Format` and related classes are the most immature part of ``bitformat`` at present, with some notable features still not implemented.