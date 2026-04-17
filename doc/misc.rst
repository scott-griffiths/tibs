.. currentmodule:: tibs

Miscellaneous
-------------

Endianness
^^^^^^^^^^

TODO

Bit indexing
^^^^^^^^^^^^

TODO


Byte encoding format
^^^^^^^^^^^^^^^^^^^^


The :meth:`~Tibs.encode` method stores an arbitrary Tibs as a sequence of bytes which can
be used to reconstruct the Tibs via :meth:`~Tibs.decode`.

The raw encoding is very efficient, and the encoded sequence contains the bit length, which
means that they can be safely concatenated without losing any information.

It is planned that extra codecs will be added that can compress the data at a later date,
but the base raw implementation does a good job at the smaller bit sequences that compression
algorithms would be very inefficient at storing. For longer sequences the raw codec overhead
is still small.

The mutable nature of ``Tibs`` and ``Mutibs`` is not part of the encoded data, so
``Mutibs.decode(t.encode())`` is equivalent to ``t.to_mutibs()`` for example, though of
course the more efficient :meth:`Tibs.to_mutibs` method is preferred in this case.

.. csv-table::
   :header: "Tibs length", "Raw encoded byte overhead"

   0 to 5 bits, +0 bytes
   6 to 37 bits, +1 byte
   38 to 1016 bits, +2 bytes
   1017 to 131064 bits, +3 bytes
   ... , ...
   1 MiB, +4 bytes

This section gives a format specification, and although it isn't a formal spec, it should
allow other implementations to encode and decode.

Overview
""""""""

In this section the notation a..b include both endpoints a and b.

Each encoded Tibs is in one of three forms, determined by its bit length:

1. Single byte form (only 0..5 bits)
2. Short form (only 6..37 bits)
3. Long form (any length)

The single byte and short forms can only be used to encode 0..5 bits and 6..37 bits respectively.

The long form can be used for any length, but is required for lengths >37 bits. It is not
recommended to use the long form for lengths <=37 bits as it will be less efficient.

The encoding and decoding methods are symmetric.
Note that when decoding, any illegal or reserved values encountered are considered errors.

The bit length of the Tibs determines which of the three encodings to use:

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

There are 2 bits to specify the codec. Currently ``00`` is used for the implemented 'raw'
encoding and ``01`` is assigned to Rice encoding for a future implementation.

.. csv-table::
   :header: "``codec``", "Byte codec"

   ``00``, raw
   ``01``, rice
   ``10``, reserved
   ``11``, reserved

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
