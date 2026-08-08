.. currentmodule:: tibs


.. raw:: html

   <div style="display: flex; justify-content: left; margin: 0 0 1rem 0;">
     <div style="display: flex; align-items: flex-end; gap: 1rem;">
       <img src="_static/tibs_cat.png" alt="Tibs" style="width: 95px; height: auto;"/>
       <div style="display: flex; flex-direction: column; align-items: center;">
         <img src="_static/tibs.png" alt="tibs" style="width: 240px; height: auto;"/>
         <div>A sleek Python library for binary data</div>
       </div>
     </div>
   </div>

|github| |license| |pepy-downloads| |monthly-downloads|


.. |github| image:: https://img.shields.io/static/v1?label=&message=GitHub&logo=github&logoColor=white&labelColor=blue&color=blue
   :target: https://github.com/scott-griffiths/tibs
   :alt: GitHub

.. |license| image:: https://img.shields.io/pypi/l/tibs?labelColor=blue&color=blue
   :alt: PyPI - License

.. |pepy-downloads| image:: https://img.shields.io/pepy/dt/tibs?logo=python&logoColor=white&labelColor=blue&color=blue
   :target: https://www.pepy.tech/projects/tibs
   :alt: Pepy Total Downloads

.. |monthly-downloads| image:: https://img.shields.io/pypi/dm/tibs?label=%40&logoColor=white&labelColor=blue&color=blue
   :target: https://pypistats.org/packages/tibs
   :alt: PyPI - Downloads

User Manual
-----------

``tibs`` is a Rust-backed Python library for creating, manipulating and
interpreting binary data. It's designed to be lightweight and easy to use, and
does not assume everything fits neatly into bytes: fields can have many different
interpretations and be any number of bits long.

The two most important classes in the module are:

* :doc:`tibs` — an immutable sequence of bits ('tibs' is 'bits' rearranged).
* :doc:`mutibs` — a mutable sequence of bits ('mutibs' is a mutable tibs).

They relate to each other just as ``bytes`` relates to ``bytearray``: the
immutable one gives stable values, hashing and cheap slices; the mutable one
gives in-place edits. The constructors and most methods are shared, so throughout
the manual an example using one usually applies equally to the other.


Overview
^^^^^^^^

A ``Tibs`` is a **sequence of bits** — like ``bytes``, but the unit is the bit and
the length can be anything. ``Tibs`` provides an interface very similar to
``bytes`` and other Python containers: you can slice it, concatenate it, search
and replace inside it at bit granularity, or pin searches to byte boundaries for
stream parsing, all in a familiar way.

This 'container of bits' mental model might be all that you need, but the
library also gives you two broad views of the binary data:

* **Typed fields.** Pull integers, floats, strings, hex or binary of any bit
  length straight out of the bits, without hand-rolling shifts and masks.
  Little-endian ordering and LSB0 field labels are handled elegantly so you
  don't reshuffle data yourself.
* **A set of bits.** Bitwise algebra, cardinalities and set predicates, with no
  intermediate object built along the way. ``Mutibs`` can be used as a large
  mutable bitset.

These aren't separate modes or types — it's one object, with a rich interface
to ask different questions of the same bits.


A Taster
^^^^^^^^

**As a container of bits.** ``Tibs`` works like ``bytes``, except that the unit
is the bit instead of the byte. ``Mutibs`` is its mutable counterpart, for
patching in place::

    >>> # A 5-bit header, a message, then 3 bits of padding: nothing is byte aligned.
    >>> frame = Tibs('0b10110') + b'the cat rarely blinked' + [0, 0, 0]
    >>> bytes(frame).find(b'cat')      # as bytes, the message has been scrambled
    -1
    >>> pos = frame.find(b'cat')       # but the tibs still knows where it is
    >>> pos, frame[pos:pos + 24].bytes
    (37, b'cat')

    >>> patched = frame.to_mutibs()
    >>> patched[pos:pos + 24] = b'squirrel'
    >>> patched[5:-3].bytes
    b'the squirrel rarely blinked'
    >>> len(frame), len(patched)       # 40 bits longer, spliced in at bit 37
    (184, 224)

**As typed fields.** Read and write integers, floats and strings of any bit
length, with a view taking care of byte order and bit numbering — the sort of
job that gets awkward quickly with plain bytes and masks::

    >>> # What's inside a float? A sign bit, an 8-bit exponent and a 23-bit fraction.
    >>> x = Tibs.from_f(-118.625, 32)
    >>> f"{x:_.8b}"                    # grouped into bytes to make it readable
    '11000010_11101101_01000000_00000000'
    >>> sign, exponent, fraction = x.split_at([1, 9])
    >>> (-1) ** sign.u * 2 ** (exponent.u - 127) * (1 + fraction.u / 2 ** 23)
    -118.625

    >>> Tibs(b'\x00\x40\xed\xc2').le.f     # the same value, from a little-endian file
    -118.625
    >>> Tibs.from_u(x.u + 1, 32).f         # the adjacent float32, one bit away
    -118.62500762939453

**As a set of bits.** Bitwise algebra and cardinalities over millions of bits,
without building an intermediate object just to count it::

    >>> from math import isqrt
    >>> # A sieve of Eratosthenes over ten million numbers, one bit each.
    >>> limit = 10_000_000
    >>> sieve = Mutibs.from_ones(limit)
    >>> sieve.unset([0, 1])
    >>> for p in range(2, isqrt(limit) + 1):
    ...     if sieve[p]:
    ...         sieve.unset(range(p * p, limit, p))
    ...
    >>> sieve.count(1)                     # primes below ten million
    664579

    >>> # Counting twin, cousin and sexy primes: pairs 2, 4 and 6 apart:
    >>> [sieve.count_and(sieve >> d) for d in (2, 4, 6)]
    [58980, 58622, 117207]

The :doc:`examples` work through larger versions of the same ideas.


Getting started
^^^^^^^^^^^^^^^

To install use ::

    pip install tibs


There are pre-built wheels for most configurations - if there are issues then please let me know.
Tibs works with Python 3.11 and later.


Chapters
^^^^^^^^

The chapters follow the bits and the two ways of reading them.

*The bits as a container*

* :doc:`sequence` — construction, promotion, indexing, searching, splitting, editing and reordering.

*Reading them as typed fields*

* :doc:`typed_fields` — reading and writing typed values out of the bits.
* :doc:`views` — byte order, bit labels and labelled fields.
* :doc:`reading` — reading fields in sequence with a cursor.

*Reading them as a set of bits*

* :doc:`bitset` — bitwise algebra, cardinalities and set predicates.

*Cross-cutting*

* :doc:`serialization` — round-tripping arbitrary-length values through bytes.
* :doc:`tibs_vs_mutibs` — choosing between the immutable and mutable types.

The manual covers the major features but is not exhaustive — see the :doc:`api`
docs for every method and parameter, and the :doc:`appendices` for background on
byte and bit order, rendering values for display, the encoded byte format, the
eight-bit and smaller floating-point formats, and using tibs from several
threads.


.. toctree::
    :maxdepth: 1
    :hidden:

    sequence
    typed_fields
    views
    reading
    bitset
    serialization
    tibs_vs_mutibs
    credits
