.. currentmodule:: tibs


Byte encoding format
--------------------


The :meth:`Tibs.encode` method stores an arbitrary Tibs as a sequence of bytes which can
be used to reconstruct the Tibs via :meth:`~Tibs.decode`.

There are different codecs that are used to compress the data, both a general use Zstandard codec
and a Rice codec, which is particularly good at sparse data.

The raw encoding is also very efficient, and all the encoded sequences contain the bit length. This
means a stream parser can concatenate encoded values without losing boundary information. The public
``decode`` methods still expect one complete value at a time and reject trailing bytes.

The compatibility contract has two parts:

* Future versions of Tibs will continue to decode complete values written by any earlier stable version.
* ``encode(Codec.Raw)`` is the canonical byte-for-byte representation of a logical bit sequence.

So use ``Codec.Raw`` for persistent keys, serialized hashes, and protocols where the exact bytes are
part of the contract. The default ``Codec.Auto`` picks whatever is most compact, and may produce
different bytes for the same bit sequence in a future release. Forward compatibility is not
guaranteed either way: older versions may reject encodings introduced later.

Whether a value was a ``Tibs`` or a ``Mutibs`` is not part of the encoded data, and the same bytes
can be decoded as either class.

The compact inline forms do a good job at the smaller bit sequences that compression
algorithms would be very inefficient at storing. For example, all bit sequences up
to 6 bits long are encoded into a single byte by the default ``Codec.Auto`` path.
For longer sequences the explicit raw codec overhead is still small.

.. csv-table::
   :header: "Tibs length", "``encode(Codec.Raw)`` overhead"

   0 bits, +1 byte
   1 to 1016 bits, +2 bytes
   1017 to 131064 bits, +3 bytes
   ... , ...
   1 MiB, +4 bytes

As an example of using the Rice codec, which is very good at sparse data, let's compress the sparsest data possible -
ten billion zero bits::

    >>> b = Tibs.from_zeros(10_000_000_000).encode(Codec.Rice)
    >>> b
    b'\x0c\x05\xfc\xf5@\xbe?\xf0'
    >>> t = Tibs.decode(b)
    >>> t.count(0)
    10000000000

Obviously compressing ten billion bits into eight bytes is an edge case, but for comparison the Zstandard compression
(with default parameters) would use 38 kB for this sequence.

This section gives a format specification, and although it isn't a formal spec, it should
allow other implementations to encode and decode.

Overview
""""""""

In this section the notation ``a..b`` includes both endpoints ``a`` and ``b``.

Each encoded Tibs is in one of three forms:

1. Single byte form (only 0..6 bits)
2. Short form (only 7..64 bits)
3. Long form (any length)

The single byte and short forms can only be used to encode 0..6 bits and 7..64 bits respectively.

The long form can be used for any length, but is required for lengths >64 bits.

The encoding and decoding methods are symmetric.
Note that when decoding, any illegal or reserved values encountered are considered errors.

The bit length of the Tibs determines which of the three encodings can be used:

1. Single byte (0..6 bits)
""""""""""""""""""""""""""

The first bit must be set. The remaining bits of the byte decode the data as follows::

    bit0: single_byte_flag = 1
    bit1: is_six_bits_flag
    if is_six_bits_flag:
        bit2..bit7: bit_data
    else:
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

For this single byte, decoding the ``bit_data`` will give a sequence of zero to six bits.

The value ``10000000`` does not correspond to a valid encoding and is reserved.

As an example, the byte ``10001110`` would be decoded as::

    1: single_byte_flag
    0: is_six_bits_flag
    0: is_five_bits_flag
    0: is_four_bits_flag
    1: is_three_bits_flag
    110: bit_data

so this represents a 3-bit sequence with the value ``110``.

2. Short form (7..64 bits)
""""""""""""""""""""""""""

For short bit sequences ``bit0`` will be unset and ``bit1`` will be set.
The rest of the byte gives the byte length and padding::

    bit0: single_byte_flag = 0
    bit1: short_form_flag = 1
    bit2..bit4: byte_length_minus_1
    bit5..bit7: bit_padding

The ``byte_length_minus_1`` will be in the range 0..7, and so will be used for data lengths of
1 to 8 bytes. The ``bit_padding`` value is the number of bits to truncate from the end of the
data bytes, and bit lengths of 1 to 6 are reserved because they must use the single byte form.
The data for this is then stored in the next 1 to 8 bytes, left aligned.

For example, the byte ``01001111`` would be decoded as::

    0: single_byte_flag
    1: short_form_flag
    001: byte_length_minus_1
    111: bit_padding

``byte_length_minus_1`` is ``1`` and ``bit_padding`` is ``7``, so this will be followed by two
bytes of data with the final seven bits ignored. Including the header byte, the sequence
``01001111_11100011_10000000`` represents a 9-bit
sequence with the value ``111000111``.

3. Long form (65+ bits)
"""""""""""""""""""""""

The long form is required for encoding 65 bits or greater. Although it can be used on shorter
sequences it is not recommended as it will be less efficient than the specialised encodings.

For long form sequences, both ``bit0`` and ``bit1`` will be unset.
The first byte's format will be::

    bit0: single_byte_flag = 0
    bit1: short_form_flag = 0
    bit2..bit4: codec
    bit5..bit7: bit_padding

There are 3 bits to specify the codec.

.. csv-table::
   :header: "``codec``", "Byte codec"

   ``000``, Raw
   ``001``, Rice
   ``010``, Zstd
   ``011`` to ``111``, Reserved

The ``bit_padding`` decodes as an unsigned integer in the range 0..7. This gives the number
of bits to truncate from the end of the decoded bytes, so allows all bit lengths to be stored.

After this header, a variable-length integer (varint) ``byte_length`` is decoded from one
or more bytes::

    bit0: continuation_flag
    bit1..bit7: length_data

Varint rules:

- Each varint byte contributes 7 data bits (``length_data``).
- These are concatenated in the order they are encountered.
- ``continuation_flag == 1`` means another varint byte follows.
- ``continuation_flag == 0`` marks the final varint byte.

A first varint byte equal to ``10000000`` is not permitted and is reserved.

Raw decoding
""""""""""""

If the ``codec`` is 'Raw', this is then followed by ``byte_length`` bytes. The raw bit sequence is formed from these bytes with ``bit_padding`` bits at the end removed.

This bit sequence is just the bits of the Tibs.

Rice decoding
"""""""""""""

If the ``codec`` is 'Rice', next comes a configuration byte::

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

For example, let's decode the four byte sequence ``b'\x09\x01.\xbe'``.
This corresponds to the binary ``00001001_00000001_00101110_10111110``

byte ``00001001`` ::

    0: single_byte_flag
    0: short_form_flag
    001: codec (Rice)
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

Larger chunks of binary data are often best compressed with a more general algorithm, and Zstandard is a
modern, effective and fast option. The main byte payload is compressed, and the extra metadata needed for
the Tibs is still stored in the header.

----

For a final example let's encode rather than decode. Let's say we have a 50 bit sequence of all ``1``.
This could be efficiently coded with Rice encoding, but let's use Raw just to demonstrate.

First the header byte is::

    0: single_byte_flag
    0: short_form_flag
    000: codec (Raw)
    110: bit_padding (6)

Then we encode the byte length. We need 7 bytes to store our 50 bits, so we encode the number 7::

    0: length_continuation_flag
    0000111: length_data (7)

We don't need any further bytes to store the byte length, so the ``length_continuation_flag`` is set to ``0`` and
no further length bytes are encoded.

Finally we pad the data with ``0`` bits up to the byte boundary and store it::

    11111111_11111111_11111111_11111111_11111111_11111111_11000000

so the final sequence is ::

    00000110_00000111_11111111_11111111_11111111_11111111_11111111_11111111_11000000
    header   byte_len bit_data                                                padding

Notes
"""""

The public decoder expects a single complete encoded Tibs value. It raises
:exc:`DecodeError` if additional bytes remain after that value.
