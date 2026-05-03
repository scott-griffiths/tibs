.. currentmodule:: tibs

Enums
-----

.. autoclass:: tibs.Endianness
   :members:
   :member-order: groupwise
   :undoc-members:

This specifies the byte-wise endianness to use when creating or interpreting some whole-byte values.

The default is generally ``Endianness.Unspecified`` which means that values are considered bit-wise big-endian.
This is equivalent to ``Endianness.Big`` if it is whole-byte, but can be used for any lengths.

Floats and integer values can be constructed and interpreted as big or little endian by using the appropriate
enum value when construct with ``from_`` and ``to_`` methods, for example see :meth:`Tibs.from_f` and :meth:`Mutibs.to_u`.


----

.. autoclass:: tibs.BitOrder
   :members:
   :member-order: groupwise
   :undoc-members:




----

.. autoclass:: tibs.Codec
   :members:
   :member-order: groupwise
   :undoc-members:

Different encoding strategies can be specified when using :meth:`Tibs.encode`.
Usually the default ``Codec.Auto`` should be used, which will try to pick the best codec for longer sequences.
