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


What Tibs is
^^^^^^^^^^^^

A ``Tibs`` is a **sequence of bits** — like ``bytes``, but the unit is the bit and
the length can be anything. That sequence is the whole object: you slice it,
concatenate it, search and replace inside it at bit granularity, or pin searches
to byte boundaries for stream parsing.

On top of those bits, the same object gives you **two lenses** — two ways of
reading and writing the sequence without ever copying it into another type:

* **Typed fields.** Read and write integers, floats and strings of any bit length,
  with views handling byte order and LSB0 bit labels for you.
* **A set of bits.** Bitwise algebra, cardinalities and set predicates, with no
  intermediate object built along the way.

These aren't separate modes or separate types — it's one object, and the lenses
are just different questions you ask of the same bits.


Quick taste
^^^^^^^^^^^

Build a packet from mixed pieces, split it into fields, and read each field as a
typed value::

    >>> packet = Tibs.from_joined(["0b1010", Tibs.from_u(3200, 12), b"OK"])
    >>> packet
    Tibs('0xac804f4b')
    >>> flags, size, payload = packet.split_at([4, 16])
    >>> flags.bin, size.u, payload.bytes
    ('1010', 3200, b'OK')

The same object also answers set-style questions about its bits — here, whether
every required capability bit is granted::

    >>> required, granted = Tibs('0b0010_1000'), Tibs('0b1010_1100')
    >>> required.is_subset_of(granted)
    True

That is the sequence, the typed-fields lens and the set-of-bits lens, all on one
value. The :doc:`examples` work through larger versions of the same ideas.


Constructors at a glance
^^^^^^^^^^^^^^^^^^^^^^^^^

Both classes offer a wide range of ``from_`` constructors. Those that build from
raw bit material are covered in :doc:`sequence`; those that encode typed values are
covered in :doc:`typed_fields`.

.. list-table::
   :header-rows: 1

   * - Raw bit material
     - Typed values
   * - :meth:`~Tibs.from_zeros`, :meth:`~Tibs.from_ones`, :meth:`~Tibs.from_random`
     - :meth:`~Tibs.from_u`, :meth:`~Tibs.from_i`, :meth:`~Tibs.from_f`
   * - :meth:`~Tibs.from_bin`, :meth:`~Tibs.from_oct`, :meth:`~Tibs.from_hex`, :meth:`~Tibs.from_string`
     - :meth:`~Tibs.from_value`, :meth:`~Tibs.from_values`
   * - :meth:`~Tibs.from_bytes`, :meth:`~Tibs.from_bools`, :meth:`~Tibs.from_joined`
     -


.. _promotion:

Automatic promotion
^^^^^^^^^^^^^^^^^^^^

Rather than calling a constructor explicitly, you can call ``Tibs`` (or ``Mutibs``)
directly and let it work out what you gave it. It delegates to
:meth:`Tibs.from_string` or :meth:`Tibs.from_bytes` for strings and bytes-like
types, and accepts strict list/tuple bit patterns directly::

    s = Tibs('0xabc')     # Same as Tibs.from_string('0xabc')
    t = Tibs([1, 0, 1])   # Strict list/tuple bit-pattern shorthand
    u = Tibs(b'hello')    # Same as Tibs.from_bytes(b'hello')

This promotion is pervasive: most methods that take a bit sequence will also
accept any of these types and promote it for you. So instead of ::

    t = Tibs.from_random(1_000_000)
    c = t.count(Tibs.from_bools([1, 0, 1]))

it's more natural to write::

    c = t.count([1, 0, 1])

Using promotion is generally recommended for conciseness and clarity. The one
exception is performance-critical code, where avoiding the small overhead of
examining and dispatching on the type can matter — there an explicit ``from_``
method is preferred. See :ref:`construction` for the exact promotion rules.


Getting started
^^^^^^^^^^^^^^^

To install use ::

    pip install tibs


There are pre-built wheels for most configurations - if there are issues then please let me know.
Tibs works with Python 3.11 and later.


How the manual is organised
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The chapters follow the substrate and its two lenses.

*The bits as a container*

* :doc:`sequence` — construction, indexing, searching, splitting, editing and reordering.

*Reading them as typed fields*

* :doc:`typed_fields` — reading and writing typed values out of the bits.
* :doc:`views` — byte order, bit labels and labelled fields.
* :doc:`formatting` — rendering values with Python's format mini-language.

*Reading them as a set of bits*

* :doc:`bitset` — bitwise algebra, cardinalities and set predicates.

*Cross-cutting*

* :doc:`serialization` — round-tripping arbitrary-length values through bytes.
* :doc:`tibs_vs_mutibs` — choosing between the immutable and mutable types.

The manual covers the major features but is not exhaustive — see the :doc:`api`
docs for every method and parameter, and the :doc:`appendices` for background on
byte and bit order and the encoded byte format.


.. toctree::
    :maxdepth: 1
    :hidden:

    sequence
    typed_fields
    views
    formatting
    bitset
    serialization
    tibs_vs_mutibs
    credits
