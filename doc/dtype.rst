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
length suffix. The named OCP and P3109 formats documented in
:ref:`exotic-floats` also have intrinsic widths, so their names do not take a
separate length suffix.

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
exactly the top half of the ``f32`` encoding — the whole ``f32`` range, traded
against roughly two significant decimal digits instead of three. The two divide
the same sixteen bits differently, so neither replaces the other and the same
bits mean different numbers in each::

    >>> Tibs.from_value("f32", 1.0).hex
    '3f800000'
    >>> Tibs.from_value("bf16", 1.0).hex
    '3f80'
    >>> Tibs("0x3f80").to_value("f16")
    1.875

Only ``bf16`` is accepted; there is no ``bf8`` or ``bf32``. It takes ``_le``
and ``_be`` like the other numeric dtypes, and has its own kind, because a
length alone cannot say which of the two 16-bit floats was meant::

    >>> Dtype("bf16_le")
    DtypeSingle('bf16_le')
    >>> Dtype("bf16").kind
    DtypeKind.BFloat

Exotic floating point formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tibs also supports the scalar elements used by the Open Compute Project's
microscaling formats and two eight-bit formats from the draft IEEE P3109 work.

See :ref:`exotic-floats` for the complete list of dtypes, bit layouts,
ranges, special values and conversion rules.

.. _kinds-as-dtypes:

Fixed width dtypes
^^^^^^^^^^^^^^^^^^

Some dtype kinds fix their own bit length: ``bool`` is always one bit, ``bf16``
always sixteen, and each of the exotic float formats has an intrinsic width. For those,
the :class:`DtypeKind` on its own already says everything the dtype does, so the
length may be omitted and the kind used directly wherever a dtype is accepted::

    >>> DtypeSingle.from_params(DtypeKind.Bool)
    DtypeSingle('bool')
    >>> Tibs("0x1257").to_values(DtypeKind.OcpE2M1)
    [0.5, 1.0, 3.0, 6.0]

This is worth knowing mainly because :class:`DtypeKind` is a complete list of
every kind tibs supports, so it is a way of finding the narrow formats without
having to know their spelling in advance.

The remaining kinds - ``u``, ``i``, ``f``, ``bits``, ``bin``, ``oct``, ``hex``
and ``bytes`` - are families of widths rather than single formats, so a length
is always needed and a bare kind is rejected::

    >>> Dtype("u12").kind
    DtypeKind.Uint
    >>> DtypeSingle.from_params(DtypeKind.Uint)
    Traceback (most recent call last):
        ...
    ValueError: DtypeKind.Uint does not determine a length on its own, so one must be given. For example, 'u12'.

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
