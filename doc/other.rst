.. currentmodule:: tibs

Enums
-----

.. autoclass:: tibs.ByteOrder
   :members:
   :member-order: groupwise
   :undoc-members:

This specifies the byte order to use when creating or interpreting some whole-byte values.

The default is generally ``ByteOrder.Unspecified`` which means that values are considered bitwise big-endian.
This is equivalent to ``ByteOrder.Big`` for whole-byte values, but can be used for any length.

Floats and integer values can be constructed and interpreted as big- or little-endian byte-order values by using the appropriate
enum value with ``from_`` and ``to_`` methods, for example see :meth:`Tibs.from_f` and :meth:`Mutibs.to_u`.
The :attr:`Tibs.le` and :attr:`Tibs.be` view properties are usually the most convenient way to interpret an
existing value.


----

.. autoclass:: tibs.BitOrder
   :members:
   :member-order: groupwise
   :undoc-members:

This specifies how bit labels are mapped inside each byte when using :class:`View`.

``BitOrder.Msb0`` is the default convention: within each byte, label 0 refers to the most significant bit.
This matches normal ``Tibs`` indexing and slicing.

``BitOrder.Lsb0`` is common in hardware manuals and protocol specifications: within each byte, label 0 refers to
the least significant bit. Use the :attr:`Tibs.lsb0` view property when a specification uses this numbering.

Bit order is about labels, not about the stored data changing. For a longer introduction and examples of field
extraction, see :doc:`views`.


----

.. autoclass:: tibs.DtypeKind
   :members:
   :member-order: groupwise
   :undoc-members:

Each :class:`Dtype` instance has a 'kind' which controls how to create and interpret values.
For example ``DtypeKind.Float`` is used for floating-point data types.

Unless you are creating or dealing with data types programmatically, you probably won't need to
use this enum directly.

----

.. autoclass:: tibs.Codec
   :members:
   :member-order: groupwise
   :undoc-members:

Different encoding strategies can be specified when using :meth:`Tibs.encode`.
Usually the default ``Codec.Auto`` should be used. It uses compact inline forms for short sequences and tries to
pick a small representation for longer sequences. Use ``Codec.Raw`` when the encoded bytes themselves are part of
an external contract, such as persistent cache keys or serialized hashes.

``Codec.Raw`` stores the bits directly with length metadata. ``Codec.Rice`` is intended for sparse data, where one
bit value occurs much less often than the other. ``Codec.Zstd`` uses Zstandard compression and is often better for
larger byte-like data.

The encoded byte format stores enough length information to decode one value exactly. Future versions should
continue to decode complete values written by earlier stable versions, but the exact bytes produced by
``Codec.Auto`` may change between releases. See :doc:`byte_format` for the format details.
