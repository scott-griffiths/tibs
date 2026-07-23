.. currentmodule:: tibs

Serialization
-------------

You can use :meth:`Tibs.to_bytes` and :meth:`Tibs.from_bytes` to store and retrieve
whole-byte bit sequences. If the data isn't a whole number of bytes then
:meth:`Tibs.to_padded_bytes` will add zero bits to the end to make one up, but this
then can't be round-tripped back to a ``Tibs`` as the bit length information has
been lost.

For an arbitrary-length ``Tibs`` that needs to round-trip through bytes by itself,
use :meth:`Tibs.encode` and :meth:`Tibs.decode`::

    >>> t = Tibs('0b001110')
    >>> encoded = t.encode()
    >>> encoded
    b'\xce'
    >>> Tibs.decode(encoded)
    Tibs('0b001110')

The encoded form stores the bit length as well as the data, so it can reconstruct
values whose length is not a multiple of eight. Various compression and storage
codec options are provided, and it is very efficient at storing short sequences and
compressing sparse data::

    >>> m = Mutibs.from_zeros(1_000_000_000)
    >>> m.set([4, 876, 999_999_999])
    >>> encoded = m.encode()
    >>> encoded
    b'\x0e\x0c\xe6\x00\x00\x00 \x00\x00\xd9\xfa\xe6\xb1\xa4\x80'
    >>> t = Tibs.decode(encoded)
    >>> t.find_all([1])
    [4, 876, 999999999]

The codec can be chosen explicitly via the :class:`Codec` enum; with the default
``Codec.Auto`` a suitable one is picked for you. The detailed byte format is
described in :doc:`byte_format`.
