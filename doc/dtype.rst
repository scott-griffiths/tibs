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
by one value. The ``bool`` dtype is the exception: it is always exactly one bit
long and has no length suffix.

.. csv-table::
   :header: "Form", "Meaning", "Example"

   ``uN``, "Unsigned integer", ``u12``
   ``iN``, "Signed integer", ``i16``
   ``fN``, "IEEE floating-point value", ``f32``
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

For integer and floating-point dtypes, append ``_le`` or ``_be`` to specify
the byte order of a whole-byte value::

    >>> Dtype("u16_le")
    DtypeSingle('u16_le')
    >>> Tibs.from_values("u16_le", [0x1234, 0xabcd]).hex
    '3412cdab'

Byte order cannot be used with ``bool``, ``bits``, ``bin``, ``oct``, ``hex``
or ``bytes`` dtypes. Floating-point values support the IEEE widths 16, 32 and
64 bits.

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
    Tibs('0b1101')
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

Tibs 2.0 migration
^^^^^^^^^^^^^^^^^^

In Tibs 2.0, ``Dtype`` becomes the base class and parsing factory:

* ``Dtype("u8")`` returns a :class:`DtypeSingle`, rather than an object whose
  concrete type is exactly ``Dtype``.
* ``repr(Dtype("u8"))`` becomes ``DtypeSingle('u8')``.
* ``Dtype.from_params(kind, length, byte_order)`` moves to
  :meth:`DtypeSingle.from_params`.
* The common ``length`` property is retained. For an array or tuple it is the
  total encoded length of one structured value; an array element's length is
  available as ``array.dtype.length``.

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
