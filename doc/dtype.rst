.. currentmodule:: tibs

Dtypes
------

Data types describe how Python values are converted to fixed-width bit
sequences and back again. Every dtype has a positive :attr:`Dtype.length`,
measured in bits, and can therefore describe either one value or a repeated
sequence of values.

``Dtype`` is the base class and the convenient parsing factory. It returns one
of three concrete, immutable classes:

* :class:`DtypeSingle` describes one scalar value.
* :class:`DtypeArray` repeats another dtype a fixed number of times.
* :class:`DtypeTuple` combines a fixed sequence of possibly different dtypes.

For example::

    >>> type(Dtype("u8")) is DtypeSingle
    True
    >>> type(Dtype("[u8; 4]")) is DtypeArray
    True
    >>> type(Dtype("(u8, u16)")) is DtypeTuple
    True

The concrete class is shown by ``repr()``, while ``str()`` returns the
canonical dtype string::

    >>> dtype = Dtype(" ( u8, [ bool ; 3 ] ) ")
    >>> dtype
    DtypeTuple('(u8, [bool; 3])')
    >>> str(dtype)
    '(u8, [bool; 3])'

All dtypes provide :meth:`~Dtype.pack`, :meth:`~Dtype.unpack`,
:meth:`~Dtype.pack_values`, :meth:`~Dtype.unpack_values` and
:meth:`~Dtype.unpack_values_iter`. The packing methods return immutable
:class:`Tibs` instances. Use :meth:`Mutibs.from_value` or
:meth:`Mutibs.from_values` when mutable output is needed directly.

Single dtypes
^^^^^^^^^^^^^

A single dtype starts with a kind and usually ends with the number of bits used
by one value. The ``bool`` dtype is always exactly one bit long and has no
length suffix. The named OCP and P3109 formats documented below also have
intrinsic widths, so their names do not take a separate length suffix.

.. csv-table::
   :header: "Form", "Meaning", "Example"

   ``uN``, "Unsigned integer", ``u12``
   ``iN``, "Signed integer", ``i16``
   ``fN``, "IEEE floating-point value", ``f32``
   ``bf16``, "bfloat16 floating-point value, always 16 bits", ``bf16``
   ``bool``, "Python bool using one bit", ``bool``
   ``bitsN``, "A bit sequence with exactly N bits, decoded as Tibs", ``bits5``
   ``binN``, "Binary string with exactly N bits", ``bin5``
   ``octN``, "Octal string with exactly N bits", ``oct12``
   ``hexN``, "Hex string with exactly N bits", ``hex16``
   ``bytesN``, "Bytes value using N bits", ``bytes32``

``bool`` values can be packed from ``True``, ``False``, ``0`` or ``1``, and
are unpacked as Python ``bool`` objects. ``bitsN`` values use normal Tibs
promotion, so they can be packed from :class:`Tibs`, :class:`Mutibs`, strings,
bytes-like objects, or strict list/tuple bit patterns. They are always unpacked
as immutable :class:`Tibs` objects.

.. note::
    Lengths are consistently in bits throughout Tibs. In particular,
    ``"bytes32"`` is four bytes long, not 32 bytes long.

For the generic ``uN``, ``iN``, ``fN`` and ``bf16`` dtypes, append ``_le`` or
``_be`` to specify the byte order of a whole-byte value::

    >>> Dtype("u16_le")
    DtypeSingle('u16_le')
    >>> Tibs.from_values("u16_le", [0x1234, 0xabcd]).hex
    '3412cdab'

Byte order cannot be used with ``bool``, ``bits``, ``bin``, ``oct``, ``hex``
or ``bytes`` dtypes. Floating-point values support the IEEE widths 16, 32 and
64 bits. The named narrow formats below likewise reject ``_le`` and ``_be``:
their bit layout is intrinsic to the encoding, and repeated sub-byte values
are packed consecutively in normal Tibs bit order.

``bf16`` is a second 16-bit float, and not an IEEE one. It spends 8 bits on the
exponent and 7 on the mantissa, where ``f16`` spends 5 and 10, which makes it
exactly the top half of the ``f32`` encoding::

    >>> Tibs.from_value("f32", 1.0).hex
    '3f800000'
    >>> Tibs.from_value("bf16", 1.0).hex
    '3f80'
    >>> Tibs.from_value("f16", 1.0).hex
    '3c00'

The two 16-bit formats divide the same sixteen bits differently, so neither
replaces the other and the same bits mean different numbers in each::

    >>> Tibs("0x3f80").to_value("bf16")
    1.0
    >>> Tibs("0x3f80").to_value("f16")
    1.875

``bf16`` keeps the whole ``f32`` range, reaching values that ``f16`` flushes to
zero, and pays for it in precision: ``f16``'s roughly three significant decimal
digits become roughly two::

    >>> Tibs.from_value("bf16", 1e-8).to_value("bf16")
    1.0011717677116394e-08
    >>> Tibs.from_value("f16", 1e-8).to_value("f16")
    0.0
    >>> Tibs.from_value("bf16", 1.001).to_value("bf16")
    1.0
    >>> Tibs.from_value("f16", 1.001).to_value("f16")
    1.0009765625

Only ``bf16`` is accepted; there is no ``bf8`` or ``bf32``. It takes ``_le``
and ``_be`` like the other numeric dtypes, and has its own kind::

    >>> Dtype("bf16_le")
    DtypeSingle('bf16_le')
    >>> Dtype("bf16").kind
    DtypeKind.BFloat

Narrow OCP and P3109 numeric formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tibs also supports the scalar elements used by the Open Compute Project's
microscaling formats and two eight-bit formats from the draft IEEE P3109 work.
Their names include the specification family because a width such as four,
six or eight bits does not identify one floating-point encoding::

    >>> Dtype("ocp_e2m1")
    DtypeSingle('ocp_e2m1')
    >>> values = [0.5, 1.0, 3.0, 6.0]
    >>> Tibs.from_values("ocp_e2m1", values).hex
    '1257'
    >>> Tibs("0x1257").to_values("ocp_e2m1")
    [0.5, 1.0, 3.0, 6.0]

These are the available encodings. ``S/E/M`` gives the number of sign,
exponent and stored mantissa bits; the implicit leading significand bit is not
included in ``M``.

.. list-table:: Narrow numeric dtype reference
   :header-rows: 1
   :widths: 28 8 12 20 28 27

   * - Dtype
     - Bits
     - Layout
     - Finite range
     - Special values
     - Packing outside the finite range
   * - ``p3109_k8p3se``
     - 8
     - S/E/M 1/5/2, bias 16
     - ±2⁻¹⁷ to ±49,152
     - One zero, one NaN, ±infinity
     - Round to ±infinity
   * - ``p3109_k8p4se``
     - 8
     - S/E/M 1/4/3, bias 8
     - ±2⁻¹⁰ to ±224
     - One zero, one NaN, ±infinity
     - Round to ±infinity
   * - ``ocp_e4m3_saturate``
     - 8
     - S/E/M 1/4/3, bias 7
     - ±2⁻⁹ to ±448
     - Signed zero and two NaN codes
     - Clamp to ±448
   * - ``ocp_e4m3_overflow``
     - 8
     - S/E/M 1/4/3, bias 7
     - ±2⁻⁹ to ±448
     - Signed zero and two NaN codes
     - Convert terminal overflow to NaN
   * - ``ocp_e5m2_saturate``
     - 8
     - S/E/M 1/5/2, bias 15
     - ±2⁻¹⁶ to ±57,344
     - Signed zero, ±infinity and six NaN codes
     - Clamp to ±57,344
   * - ``ocp_e5m2_overflow``
     - 8
     - S/E/M 1/5/2, bias 15
     - ±2⁻¹⁶ to ±57,344
     - Signed zero, ±infinity and six NaN codes
     - Convert terminal overflow to ±infinity
   * - ``ocp_e3m2``
     - 6
     - S/E/M 1/3/2, bias 3
     - ±2⁻⁴ to ±28
     - Signed zero; no NaN or infinity
     - Clamp to ±28
   * - ``ocp_e2m3``
     - 6
     - S/E/M 1/2/3, bias 1
     - ±2⁻³ to ±7.5
     - Signed zero; no NaN or infinity
     - Clamp to ±7.5
   * - ``ocp_e2m1``
     - 4
     - S/E/M 1/2/1, bias 1
     - ±2⁻¹ to ±6
     - Signed zero; no NaN or infinity
     - Clamp to ±6
   * - ``ocp_e8m0``
     - 8
     - Unsigned exponent, bias 127
     - 2⁻¹²⁷ to 2¹²⁷
     - One NaN; no zero or infinity
     - Exact powers of two only
   * - ``ocp_int8``
     - 8
     - Signed integer × 2⁻⁶
     - -2 to 127/64
     - No NaN or infinity
     - Clamp to the asymmetric finite range

All ordinary conversions round directly from the Python ``float`` value using
round-to-nearest, ties-to-even. In particular, Tibs does not first reduce a
value to ``f16``; values immediately around a target midpoint therefore do not
suffer a second rounding. The ``saturate`` and ``overflow`` E4M3/E5M2 dtypes
decode the same bits, but make the packing policy part of the dtype's stable
identity instead of relying on a global option::

    >>> Tibs.from_value("ocp_e4m3_saturate", 1000.0).hex
    '7e'
    >>> Tibs.from_value("ocp_e4m3_overflow", 1000.0).hex
    'ff'

OCP E4M3 and E5M2 accept Python NaNs and write the deterministic canonical
code ``0xff``; decoding accepts every NaN code defined by the format. The
smaller E2M1, E2M3 and E3M2 formats, and ``ocp_int8``, reject NaN because they
have no NaN encoding. They saturate infinities. ``ocp_int8`` includes the
optional most-negative OCP value ``0x80``, which decodes as ``-2.0``.

``ocp_e8m0`` is intentionally strict: it packs a positive, in-range, exact
power of two or NaN, and rejects zero, negative values, infinity and values
between powers of two::

    >>> Tibs.from_values("ocp_e8m0", [0.5, 1.0, 2.0]).hex
    '7e7f80'

These dtypes describe **raw scalar elements only**. They do not store a shared
scale, choose one automatically, associate elements into fixed-size blocks, or
implement scaled block arithmetic. The OCP specification reserves names such
as MXFP4 for the combination of a scale and a block of E2M1 elements; calling
the scalar dtype ``ocp_e2m1`` avoids implying that Tibs has implemented that
block behaviour. ``ocp_e8m0`` is exposed because its raw scale-element encoding
is useful independently, but it is not attached to another dtype.

The OCP definitions are frozen to `OCP Microscaling Formats v1.0`_ (September
2023) and `OCP OFP8 revision 1.0`_ (including its December 2023 correction).
The P3109 names are deliberately draft-labelled: Tibs freezes K8P3SE and
K8P4SE to `P3109 public repository commit aa9d236`_ from 29 July 2026. The
`IEEE P3109 project`_ remains an active project rather than a published
standard, and this support is not a claim of formal IEEE conformance. A later
incompatible draft would require a new dtype name; Tibs will not silently
reinterpret bits written with these names.

For migration from Bitstring 5, the corresponding names are ``p3binary``,
``p4binary``, ``e4m3mxfp_saturate``, ``e4m3mxfp_overflow``,
``e5m2mxfp_saturate``, ``e5m2mxfp_overflow``, ``e3m2mxfp``, ``e2m3mxfp``,
``e2m1mxfp``, ``e8m0mxfp`` and ``mxint`` respectively. Bitstring 4.4 used one
unsuffixed E4M3 and E5M2 name plus the global
``bitstring.options.mxfp_overflow`` setting; that global policy is being
removed in Bitstring 5. The Bitstring spellings are not aliases in Tibs: the
explicit ``ocp_`` and ``p3109_`` prefixes keep the source definition visible.

.. _OCP Microscaling Formats v1.0: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
.. _OCP OFP8 revision 1.0: https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
.. _P3109 public repository commit aa9d236: https://github.com/P3109/Public/tree/aa9d236d7a31b38fbe43b703a0bfdfc3d8be5d45
.. _IEEE P3109 project: https://standards.ieee.org/ieee/3109/11165/

A scalar dtype can also be built without parsing a string::

    >>> DtypeSingle.from_params(DtypeKind.Uint, 16, ByteOrder.Little)
    DtypeSingle('u16_le')

The general :class:`Dtype` base class has no ``from_params`` method because the
three concrete variants require different parameters.

Array dtypes
^^^^^^^^^^^^

An array dtype uses ``[dtype; count]``. It represents one structured value
containing exactly ``count`` values of its element dtype::

    >>> flags = Dtype("[bool; 4]")
    >>> flags.length
    4
    >>> flags.pack([True, True, False, True])
    Tibs('0xd')
    >>> flags.unpack("0b1101")
    (True, True, False, True)

The element may itself be any dtype, so arrays can be nested and can contain
tuple records::

    >>> samples = Dtype("[(u4, bool); 2]")
    >>> samples.length
    10
    >>> samples.unpack(samples.pack(((10, True), (3, False))))
    ((10, True), (3, False))

Use :attr:`DtypeArray.dtype` for the element dtype and
:attr:`DtypeArray.count` for the number of elements. The programmatic builder
takes the same two pieces::

    >>> DtypeArray.from_params(Dtype("(u8, bool)"), 3)
    DtypeArray('[(u8, bool); 3]')

An array's own :attr:`~Dtype.length` is the total encoded length of all its
elements, not of one element - for that, use ``array.dtype.length``.

Counts must be greater than zero. An array unpacks to an immutable Python
tuple, including when its input was a list or another iterable.

Tuple dtypes
^^^^^^^^^^^^

A tuple dtype uses ``(dtype, ...)``. It represents one structured value whose
fields may use different dtypes::

    >>> header = Dtype("(u8, u16_le, bool)")
    >>> header.length
    25
    >>> packed = header.pack((1, 0x0203, True))
    >>> header.unpack(packed)
    (1, 515, True)

Tuple fields can be single, array or tuple dtypes. A one-field tuple is written
``(u8,)``; empty tuples are not valid. :attr:`DtypeTuple.dtypes` returns the
fields as an immutable tuple.

The equivalent programmatic form is::

    >>> DtypeTuple.from_params(["u8", Dtype("[bool; 3]")])
    DtypeTuple('(u8, [bool; 3])')

One value and repeated values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A compound dtype describes one structured value. Therefore
:meth:`Tibs.from_value` takes one array or tuple value::

    >>> record = Dtype("(u8, u16_le)")
    >>> packed = Tibs.from_value(record, (1, 0x0203))
    >>> packed.hex
    '010302'
    >>> packed.to_value(record)
    (1, 515)

The plural methods repeat the *whole* dtype rather than treating a tuple's
fields as the repeated values::

    >>> records = [(1, 0x0203), (4, 0x0506)]
    >>> packed = Tibs.from_values(record, records)
    >>> packed.hex
    '010302040605'
    >>> packed.to_values(record)
    [(1, 515), (4, 1286)]

The selected range for :meth:`Tibs.to_value` or :meth:`Dtype.unpack` must have
exactly the dtype length. Repeated decoding requires a whole number of dtype
values. Empty input is valid for the repeated methods because it contains zero
complete values.

Relationship to ``struct``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Tuple dtypes can express the explicit-width, no-padding layouts supported by
the standard-size forms of Python's ``struct`` module. For example,
``"(i16_le, i32_le, i32_le)"`` has the same layout as ``"<hll"``.

Compound dtypes concatenate their children exactly. They do not provide native
C type widths, native alignment or implicit padding, so a bare ``"hll"``
``struct`` format may have a different size and layout.

Immutability and identity
^^^^^^^^^^^^^^^^^^^^^^^^^

Every dtype is immutable, compares structurally and can be used as a dictionary
key or set member. Dtypes do not compare equal to their string specifications.
Arrays and tuples have no scalar ``kind`` or ``byte_order``; those properties
belong only to :class:`DtypeSingle`.

.. autoclass:: tibs.Dtype
   :members:
   :member-order: groupwise
   :undoc-members:

.. autoclass:: tibs.DtypeSingle
   :members:
   :member-order: groupwise
   :show-inheritance:
   :undoc-members:

.. autoclass:: tibs.DtypeArray
   :members:
   :member-order: groupwise
   :show-inheritance:
   :undoc-members:

.. autoclass:: tibs.DtypeTuple
   :members:
   :member-order: groupwise
   :show-inheritance:
   :undoc-members:
