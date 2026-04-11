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

(Note this isn't implemented yet)

The :meth:`~Tibs.encode` method stores an arbitrary Tibs as a sequence of bytes which can
be used to reconstruct the Tibs via :meth:`~Tibs.decode`.

The raw encoding is very efficient, and the encoded sequence contains the bit length, which means that they can be safely concatenated without
losing any information.

.. csv-table::
   :header: "Tibs length", "Encoded size"

   TODO, UPDATE
   0 to 4 bits, 1 byte
   5 to 8 bits, 2 bytes
   9 to 16 bits, 3 bytes
   ... , ...
   8 bytes, 10 bytes
   1 MiB, 1MiB + 5 bytes


The first bit determines whether the bits are stored in a single bit. If this bit is set the format of
the byte is::

    bit0: single_byte_flag = 1
    bit1: msb0_flag
    bit2: is_five_bits_flag
    if is_five_bits_flag:
        bit3 - bit7: bit_data
    else:
        bit3: is_four_bits_flag
        if is_four_bits_flag:
            bit4 - bit7: bit_data
            else:
                bit4: is_three_bits_flag:
                if is_three_bits_flag:
                    bit5 - bit7: bit_data
                else:
                    bit5: is_two_bits_flag:
                    if is_two_bits_flag:
                        bit6 - bit7: bit_data
                    else:
                        bit6: is_one_bit_flag:
                        if is_one_bit_flag:
                            bit7: bit_data
                        else:
                            is_empty()

For this single byte encoding the ``bit_data`` will be a sequences of length zero to five bits.
For example, the byte ``11001110`` would be decoded as::

    1: single_byte_flag
    1: msb0_flag
    0: is_five_bits_flag
    0: is_four_bits_flag
    1: is_three_bits_flag
    110: bit_data

so this decodes as a 3-bit MSB0 sequence with the value ``110``.


Bit sequences longer than five bits will need to be stored in at least two bytes.
For short bit sequences bit2 will be set, and then the rest of the byte gives the bit length::

    bit0: single_byte_flag = 0
    bit1: msb0_flag
    bit2: is_short_flag = 1
    bit3 - bit7: length_minus_6

The ``length_minus_6`` will be in the range 0 - 31, and so will be used for bit lengths of 6 to 37.
The data for this is then stored in the next 1 to 5 bytes, left aligned.

For sequences of 38 bits or longer we have the first byte's format of::

    bit0: single_byte_flag = 0
    bit1: msb0_flag
    bit2: is_short_flag = 0
    bit3 - bit4: codec
    bit5 - bit7: bit_padding

There are 2 bits to specify the codec. Currently only ``00`` for the 'raw' encoding is supported.

The ``bit_padding`` decodes as an unsigned integer in the range 0 to 7. This allows the specification
of any bit length, rather than just a byte length.

After this header, a byte length is specified by a sequence of 1 or more bytes::

    do:
        bit0: length_continuation_flag
        bit1 - bit7: length_data
        while(length_continuation_flag)

Thus each byte contains 7 bits of data which is concatenated and interpreted as an unsigned integer  to get a value ``byte_length``.

This is then followed by ``byte_length`` bytes, which is decoded according to the codec to a sequence of bytes.
Finally ``bit_padding`` bits at the end of the decoded bytes are removed to get the final data.

For the raw codec the bytes are just stored literally.


.. mermaid::

   block-beta
      columns 8
      b0["0\nsingle_byte_flag = 1"] b1["1\nmsb0_flag"] b2["2\nsmall_flag"] b3["3-7\nbyte0_data"]:5

If the ``small_flag`` is set, it means that the Tibs is small enough to fit in the ``byte0_data`` section


But why is it called tibs?
^^^^^^^^^^^^^^^^^^^^^^^^^^

Because 'tibs' is (almost) 'bits' backwards. It's also distinctive, and the name was available on PyPI.

It's got nothing to do with Ethiopian stew.