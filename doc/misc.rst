.. currentmodule:: tibs

Miscellaneous
-------------

Endianness
^^^^^^^^^^

Byte-wise endianness is available for constructing and interpreting various whole-byte values.
The endianness isn't a property of the ``Tibs``, but affect both how it's constructed from a value
and how it's reinterpreted as a value. ::

    >>> Tibs.from_u(511, 32, Endianness.Big)
    Tibs('0x000001ff')
    >>> Tibs.from_u(511, 32, Endianness.Little)
    Tibs('0xff010000')

The default is ``Endianness.Unspecified`` which is bit-wise big endian. The difference between ``Unspecified``
and ``Big`` is that the latter will complain if it tries to construct or interpret a non whole-byte value. ::


    >>> m = Mutibs.from_f(1984, 64)
    >>> m.to_f()
    1984.0
    >>> m.to_f(Endianness.Little)
    2.0142e-319
    >>> m.byte_swap()
    >>> m.to_f(Endianness.Little)
    1984.0


Bit indexing
^^^^^^^^^^^^

Two bit indexing methods are supported, MSB0 (most significant bit 0) and LSB0 (least significant bit 0).

The default MSB0 bit numbering is done from 'left' to 'right'.
That is, from bit ``0`` at the start of the data to bit ``n - 1`` at the end.
This allows a ``Tibs`` to be treated like an ordinary Python container that is only allowed to contain single bits.

The LSB0 bit numbering means the right-most bit in the bitstring will
be bit 0, and the left-most bit will be bit (n-1), rather than the
other way around. LSB0 is a more natural numbering
system in some fields.

The ``bit_indexing`` parameter on creation methods sets which numbering is used.
It is a property of the ``Tibs`` or ``Mutibs``, and can be changed after creation only for a ``Mutibs`` instance.
It does not affect equality operations. ::

    >>> t = Tibs('0xabc', bit_indexing=BitIndexing.Lsb0)
    >>> t
    Tibs('0xabc', BitIndexing.Lsb0)
    >>> m = Mutibs('0xabc')
    >>> m == t
    True
    >>> m.bit_indexing == t.bit_indexing
    False


For example, if you set a ``Tibs`` to be the binary ``010001111`` it will be stored in the same way for MSB0 and LSB0,
but slicing, reading, unpacking etc. will all behave differently.

.. list-table:: MSB0 →
   :header-rows: 1

   * - bit index
     - 0
     - 1
     - 2
     - 3
     - 4
     - 5
     - 6
     - 7
     - 8
   * - value
     - ``0``
     - ``1``
     - ``0``
     - ``0``
     - ``0``
     - ``1``
     - ``1``
     - ``1``
     - ``1``

In MSB0 everything behaves like an ordinary Python container.
Bit zero is the left-most bit and reads/slices happen from left to right.

.. list-table:: ← LSB0
   :header-rows: 1

   * - bit index
     - 8
     - 7
     - 6
     - 5
     - 4
     - 3
     - 2
     - 1
     - 0
   * - value
     - ``0``
     - ``1``
     - ``0``
     - ``0``
     - ``0``
     - ``1``
     - ``1``
     - ``1``
     - ``1``

In LSB0 the final, right-most bit is labelled as bit zero. Reads and slices happen from right to left.

When ``Tibs`` are interpreted as integers and other types the left-most bit is always considered as the most significant bit.
It's important to note that this is the case irrespective of whether the first or last bit is considered the bit zero,
so for example if you were to interpret a whole ``Tibs`` as an integer, its value would be the same irrespective
of the ``bit_indexing`` value.

To illustrate this, for the example above this means that the bin and int representations would be ``010001111`` and ``143`` respectively
for both MSB0 and LSB0 bit numbering.

Slicing is still done with the start bit smaller than the end bit.
For example:

    >>> s = Tibs('0b010001111', bit_indexing=BitIndexing.Lsb0)
    >>> s[0:5]  # LSB0 so this is the right-most five bits
    Tibs('0b01111')
    >>> s[0]
    True


Negative indices work as you'd expect, with the first stored
bit being ``s[-1]`` and the final stored bit being ``s[-n]``.



Byte encoding format
^^^^^^^^^^^^^^^^^^^^


The :meth:`Tibs.encode` method stores an arbitrary Tibs as a sequence of bytes which can
be used to reconstruct the Tibs via :meth:`~Tibs.decode`.

There are different codec that are used to compress the data, both a general use Zstandard codec
and a Rice codec, which is particularly good at sparse data.

The raw encoding is also very efficient, and all the encoded sequences contains the bit length, which
means that they can be safely concatenated without losing any information.

The base implementation does a good job at the smaller bit sequences that compression
algorithms would be very inefficient at storing, for example all bit sequences up to 5 bits long are encoded
into a single byte. For longer sequences the raw codec overhead is still small.

The mutable nature of ``Tibs`` and ``Mutibs`` is not part of the encoded data, so
if a ``Tibs`` and ``Mutibs`` are equal (and have the same ``BitIndexing``) they will encode
to the same ``bytes``.

.. csv-table::
   :header: "Tibs length", "Raw encoded byte overhead"

   0 to 5 bits, +0 bytes
   6 to 37 bits, +1 byte
   38 to 1016 bits, +2 bytes
   1017 to 131064 bits, +3 bytes
   ... , ...
   1 MiB, +4 bytes

As an example of using the Rice codec, which is very good at sparse data, let's compress the sparsest data possible -
ten billion zero bits::

    >>> b = Tibs.from_zeros(10_000_000_000).encode(Codec.Rice)
    >>> b
    b'L\x05\xfc\xf5@\xbe?\xf0'
    >>> t = Tibs.decode(b)
    >>> t.count(0)
    10000000000

Obviously compressing ten billion bits into eight bytes is an edge case, but for comparison the Zstandard compression
(with default parameters) would use 38 kB for this sequence.

This section gives a format specification, and although it isn't a formal spec, it should
allow other implementations to encode and decode.

Overview
""""""""

In this section the notation ``a..b`` include both endpoints ``a`` and ``b``.

Each encoded Tibs is in one of three forms, determined by its bit length:

1. Single byte form (only 0..5 bits)
2. Short form (only 6..37 bits)
3. Long form (any length)

The single byte and short forms can only be used to encode 0..5 bits and 6..37 bits respectively.

The long form can be used for any length, but is required for lengths >37 bits.

The encoding and decoding methods are symmetric.
Note that when decoding, any illegal or reserved values encountered are considered errors.

The bit length of the Tibs determines which of the three encodings can be used:

1. Single byte (0..5 bits)
""""""""""""""""""""""""""

The first bit must be set. The second bit records the MSB0 flag from the Tibs.
The remaining bits of the byte decode the data as follows::

    bit0: single_byte_flag = 1
    bit1: msb0_flag
    bit2: is_five_bits_flag
    if is_five_bits_flag:
        bit3..bit7: bit_data
    else:
        bit3: is_four_bits_flag
        if is_four_bits_flag:
            bit4..bit7: bit_data
        else:
            bit4: is_three_bits_flag:
            if is_three_bits_flag:
                bit5..bit7: bit_data
            else:
                bit5: is_two_bits_flag:
                if is_two_bits_flag:
                    bit6..bit7: bit_data
                else:
                    bit6: is_one_bit_flag:
                    if is_one_bit_flag:
                        bit7: bit_data
                    else:
                        bit7: 1  # Zero bit length

For this single byte, decoding the ``bit_data`` will give a sequences of zero to five bits.

The values of ``10000000`` and ``11000000`` do not correspond to a valid encoding and are reserved.

As an example, the byte ``11001110`` would be decoded as::

    1: single_byte_flag
    1: msb0_flag
    0: is_five_bits_flag
    0: is_four_bits_flag
    1: is_three_bits_flag
    110: bit_data

so this represents a 3-bit MSB0 sequence with the value ``110``.

2. Short form (6..37 bits)
""""""""""""""""""""""""""

For short bit sequences ``bit0`` will be unset and ``bit2`` will be set.
The rest of the byte gives the bit length::

    bit0: single_byte_flag = 0
    bit1: msb0_flag
    bit2: short_form_flag = 1
    bit3..bit7: length_minus_6

The ``length_minus_6`` will be in the range 0..31, and so will be used for bit lengths of 6 to 37.
The data for this is then stored in the next 1 to 5 bytes, left aligned.

For example, the byte ``00100011`` would be decoded as::

    0: single_byte_flag
    0: msb0_flag
    1: short_form_flag
    00011: length_minus_6

``length_minus_6`` is ``3``, so this will be followed by 9 bits of data, padded to the next byte,
so including the header byte the sequence ``00100011_11100011_10000000`` represents a 9-bit
LSB0 sequence with the value ``111000111``.

3. Long form (38+ bits)
"""""""""""""""""""""""

The long form is required for encoding 38 bits or greater. Although it can be used on shorter
sequences it is not recommended as it will be less efficient than the specialised encodings.

For long form sequences, both ``bit0`` and ``bit2`` will be unset.
The first byte's format will be::

    bit0: single_byte_flag = 0
    bit1: msb0_flag
    bit2: short_form_flag = 0
    bit3..bit4: codec
    bit5..bit7: bit_padding

There are 2 bits to specify the codec.

.. csv-table::
   :header: "``codec``", "Byte codec"

   ``00``, Raw
   ``01``, Rice
   ``10``, Zstd
   ``11``, Reserved

The ``bit_padding`` decodes as an unsigned integer in the range 0..7. This gives the number
of bits to truncate from the end of the decoded bytes, so allows all bit lengths to be stored.

After this header, a variable-length integer (varint) ``byte_length`` is decoded from one
or more bytes::

    bit0: continuation_flag
    bit1..bit7: length_data

Varint rules:

- Each varint byte contribute 7 data bits (``length_data``).
- These are concatenated in the order they are encountered.
- ``continuation_flag == 1`` means another varint byte follows.
- ``continuation_flag == 0`` marks the final varint byte.

A first varint byte equal to ``10000000`` is not permitted and is reserved.

Raw decoding
""""""""""""

If the ``codec`` is 'raw', this is then followed by ``byte_length`` bytes. The raw bit sequence are these bytes with ``bit_padding`` bits at the end removed.

This bit sequence is just the bits of the Tibs.

Rice decoding
"""""""""""""

If the ``codec`` is 'rice', next comes a configuration byte::

    bit0..4: k (unsigned int, range 0 - 31)
    bit5: sparse_bit
    bit6: final_bit
    bit7: reserved

This is then followed by ``byte_length`` bytes, with ``bit_padding`` bits at the end removed in the same way as for raw.
This bit sequence is then decoded as follows:

- The bit sequence is a concatenation of Rice-coded unsigned integers using the configured ``k`` value.
- Each integer is decoded in the usual Rice form:

  1. Count a unary prefix of ``1`` bits up to and excluding the next ``0`` bit. This count is the quotient ``q``.
  2. Consume that ``0`` separator bit.
  3. Consume the next ``k`` bits as an unsigned integer remainder ``r``. If ``k == 0``, then no remainder bits are consumed and ``r = 0``.
  4. The decoded gap value is ``gap = q * 2**k + r``.

- The decoded gaps describe runs of the opposite bit value associated with occurrences of ``sparse_bit``.
- If ``sparse_bit == 1``, each gap gives the number of ``0`` bits before the next ``1`` bit.
- If ``sparse_bit == 0``, each gap gives the number of ``1`` bits before the next ``0`` bit.
- For each decoded gap:

  1. Append ``gap`` copies of the opposite bit.
  2. Append one ``sparse_bit``.

- After all gaps have been decoded, replace the final decoded bit with ``final_bit``.

This means the encoded gaps always reconstruct a sequence ending in ``sparse_bit``. The
``final_bit`` field then overwrites that last bit so that the decoded sequence can end in
either bit value.

----

For example, let's decode the four byte sequence ``b'I\x01.\xbe'``.
This corresponds to the binary ``01001001_00000001_00101110_10111110``

byte ``01001001`` ::

    0: single_byte_flag
    1: msb0_flag
    0: short_form_flag
    01: codec (rice)
    001: bit_padding (=1)

byte ``00000001`` ::

    0: length_continuation_flag
    0000001: length_data (=1)

So this is Rice encoded with 1 byte of data, with the final bit ignored (so 7 bits of encoded data).

The next byte is the Rice configuration byte ``00101110`` ::

    00101: k (=5)
    1: sparse_bit
    1: final_bit
    0: reserved

and finally the encoded data, which we now know is just 7 bits ``1011111`` ::

    1: prefix count
    0: end of prefix => q = 1
    11111: r = 31

There is a single 1 bit before the first 0 bit, so the count of these bits gives us ``q=1``.

After the 0 bit we read the next ``k`` bits to get the unsigned integer ``r=31``.

This gives us a decoded gap of ``gap = q * 2**k + r = 1 * 2**5 + 31 = 63``.

The ``sparse_bit`` is a ``1``, so the gaps are made of ``0`` bits. And the ``final_bit`` is a ``1``, so
we just have 63 zero bits followed by a one bit and the decoded sequence is ::

    00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001

----

Zstd decoding
"""""""""""""

Larger chunks of binary data are often best compressed with a more general algorithm, and the Zstandard is a
modern, effective and fast option. The main byte payload is compressed, and the extra metadata needed for
the Tibs is still stored in the header.

----

For a final example let's encode rather than decode. Let's say we have a 50 bit MSB0 sequence of all ``1``.
This could be efficiently coded with Rice encoding, but let's use Raw just to demonstrate.

First the header byte is::

    0: single_byte_flag
    1: msb0_flag
    0: short_form_flag
    00: codec (raw)
    110: bit_padding (6)

Then we encode the byte length. We need 7 bytes to store our 50 bits, so we encode the number 7::

    0: length_continuation_flag
    0000111: length_data (7)

We don't need any further bytes to store the byte length, so the ``length_continuation_flag`` is set to ``0`` and
no further length bytes are encoded.

Finally we pad the data with ``0`` bits up the the byte boundary and store it::

    11111111_11111111_11111111_11111111_11111111_11111111_11000000

so the final sequence is ::

    01000110_00000111_11111111_11111111_11111111_11111111_11111111_11111111_11000000
    header   byte_len bit_data                                                padding

Notes
"""""

Encoding is self-delimiting. A decoder can return both:

- The decoded Tibs value
- The exact number of bytes consumed

This enables safe concatenation of multiple encoded Tibs values in one byte stream.



But why is it called tibs?
^^^^^^^^^^^^^^^^^^^^^^^^^^

Because 'tibs' is (almost) 'bits' backwards. It's also distinctive, and the name was available on PyPI.

It's got nothing to do with Ethiopian stew. Or cats.

.. raw:: html

   <div style="display: flex; justify-content: left; margin: 0 0 1rem 0;">
     <div style="display: flex; align-items: flex-end; gap: 1rem;">
       <img src="_static/tibs_white_sleeping.png" alt="Tibs" style="width: 130px; height: auto;"/>
     </div>
   </div>