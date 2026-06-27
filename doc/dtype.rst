.. currentmodule:: tibs

Dtype
-----

Data types control how bits are converted into values, and how values are converted into bits.

They are often used implicitly when creating from or interpreting to integers, floats and other types, but
can also be used explicitly in methods like :meth:`Tibs.from_values`.
Once created, a ``Dtype`` can also pack and unpack values itself::

    >>> d = Dtype("u12")
    >>> packed = d.pack_values([0, 103, 2048, 4095])
    >>> packed.hex
    '000067800fff'
    >>> d.unpack_values(packed)
    [0, 103, 2048, 4095]

The ``pack`` and ``pack_values`` methods return immutable :class:`Tibs`
instances. If mutable output is needed directly, use :meth:`Mutibs.from_value`
or :meth:`Mutibs.from_values`.

Dtype strings
^^^^^^^^^^^^^

The compact dtype string usually starts with a kind and ends with the bit length
of one value. The ``bool`` dtype is the exception: it is always exactly one bit
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

``bool`` values can be packed from ``True``, ``False``, ``0`` or ``1``, and are
unpacked as Python ``bool`` objects. ``bitsN`` values use normal Tibs promotion,
so they can be packed from :class:`Tibs`, :class:`Mutibs`, strings, bytes-like
objects, or bool iterables, and are unpacked as immutable :class:`Tibs` objects.

For integer and floating-point dtypes, append ``_le`` or ``_be`` to specify
byte order for whole-byte values::

    >>> Dtype("u16_le")
    Dtype('u16_le')
    >>> Tibs.from_values("u16_le", [0x1234, 0xabcd]).hex
    '3412cdab'

Byte order cannot be used with ``bool``, ``bits``, ``bin``, ``oct``, ``hex`` or
``bytes`` dtypes.
Float values are encoded using the supported IEEE widths: 16, 32 and 64 bits.
The selected bit range for :meth:`Tibs.to_values` and
:meth:`Dtype.unpack_values` must be a whole number of dtype values.


.. autoclass:: tibs.Dtype
   :members:
   :member-order: groupwise
   :undoc-members:
