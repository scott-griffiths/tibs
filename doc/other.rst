.. currentmodule:: tibs

Enums and exceptions
--------------------

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
existing value. For more information on byte order and bit labels, see :doc:`byte_and_bit_order`.


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
extraction, see :doc:`byte_and_bit_order` and :doc:`views`.


----

.. autoclass:: tibs.DtypeKind
   :members:
   :member-order: groupwise

   **Families of widths.** A dtype using one of these always needs a length,
   which is given in bits.

   .. py:attribute:: Uint

      An unsigned integer, written ``uN`` — for example ``u12``. Any length.

   .. py:attribute:: Int

      A two's complement signed integer, written ``iN``. Any length.

   .. py:attribute:: Float

      An IEEE 754 float, written ``fN``. The length must be 16, 32 or 64.

   .. py:attribute:: Bits

      A bit sequence, written ``bitsN`` and decoded as a :class:`Tibs`. Any length.

   .. py:attribute:: Bin

      A binary string, written ``binN``. Any length.

   .. py:attribute:: Oct

      An octal string, written ``octN``. The length must be a multiple of 3.

   .. py:attribute:: Hex

      A hexadecimal string, written ``hexN``. The length must be a multiple of 4.

   .. py:attribute:: Bytes

      A ``bytes`` value, written ``bytesN``. The length must be a multiple of 8,
      and is in bits, so ``bytes32`` is four bytes long.

   **Fixed widths.** These carry their own length, so the kind alone is a
   complete dtype. The narrow float layouts, ranges and rounding rules are in
   :ref:`exotic-floats`.

   .. py:attribute:: Bool

      A Python ``bool`` in a single bit, written ``bool``.

   .. py:attribute:: BFloat

      A 16-bit bfloat16 value, written ``bf16``. Not an IEEE format, and not
      interchangeable with ``f16``.

   .. py:attribute:: Binary8P3

      The 8-bit draft IEEE P3109 ``binary8p3`` format, favouring range.

   .. py:attribute:: Binary8P4

      The 8-bit draft IEEE P3109 ``binary8p4`` format, favouring precision.

   .. py:attribute:: OcpE5M2Saturate

      The 8-bit OCP ``ocp_e5m2_saturate`` format. Out-of-range values, and
      infinities, clamp to the largest finite value.

   .. py:attribute:: OcpE5M2Overflow

      The 8-bit OCP ``ocp_e5m2_overflow`` format. Out-of-range values become
      infinity.

   .. py:attribute:: OcpE4M3Saturate

      The 8-bit OCP ``ocp_e4m3_saturate`` format. Out-of-range values clamp to
      the largest finite value.

   .. py:attribute:: OcpE4M3Overflow

      The 8-bit OCP ``ocp_e4m3_overflow`` format. Out-of-range values become
      NaN, as E4M3 has no infinity encoding.

   .. py:attribute:: OcpE3M2

      The 6-bit OCP ``ocp_e3m2`` format. No NaN or infinity.

   .. py:attribute:: OcpE2M3

      The 6-bit OCP ``ocp_e2m3`` format. No NaN or infinity.

   .. py:attribute:: OcpE2M1

      The 4-bit OCP ``ocp_e2m1`` format. No NaN or infinity.

   .. py:attribute:: OcpE8M0

      The 8-bit OCP ``ocp_e8m0`` scale format: an unsigned power of two, or NaN.
      Values between powers of two are rejected rather than rounded.

   .. py:attribute:: OcpInt8

      The 8-bit OCP ``ocp_int8`` format: a signed integer with an implicit
      factor of 2⁻⁶, so it decodes to a Python ``float``.

Each :class:`DtypeSingle` instance has a ``kind`` which controls how to create
and interpret its scalar value. Array and tuple dtypes instead describe their
children through :attr:`DtypeArray.dtype` and :attr:`DtypeTuple.dtypes`.

Unless you are creating or dealing with data types programmatically, you probably won't need to
use this enum directly — though it is a complete list of every format tibs supports, which is
one way to find the narrow ones without knowing their spelling in advance.

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

Exceptions
----------

.. autoexception:: tibs.ReadError

This Tibs-specific exception is a subclass of :class:`ValueError`. It is raised when a :class:`Reader` cannot
complete a requested move from the bits that remain — a read, or an :meth:`Reader.align` that would step past
the end — or cannot find the delimiter requested by :meth:`Reader.read_to` or :meth:`Reader.read_past`.

.. autoexception:: tibs.DecodeError

This Tibs-specific exception is a subclass of :class:`ValueError`. It is raised when :meth:`Tibs.decode` or
:meth:`Mutibs.decode` receives malformed, truncated or extended encoded data.
