.. currentmodule:: tibs

.. _formatting:

Formatting
----------

When you just want to see the data, ``Tibs``, ``Mutibs``, ``View`` and ``MutableView``
all support Python's format mini-language, so they work directly in f-strings::

    >>> packet = Tibs('0xac804f4b')
    >>> f"{packet:#x}"
    '0xac804f4b'
    >>> f"{packet:_.8b}"
    '10101100_10000000_01001111_01001011'
    >>> f"{packet:u}"
    '2894090059'

With no format spec you get the same thing as ``str()``, which is what you'd see if you
just printed it::

    >>> f"{packet}"
    '0xac804f4b'

Type codes
^^^^^^^^^^

There are two families of type code. The first are *representation* codes, which show
you the bits themselves and are exactly equivalent to the ``bin``, ``oct`` and ``hex``
properties:

* ``b`` — binary. Always available.
* ``o`` — octal. Length must be a multiple of 3.
* ``x`` — hexadecimal. Length must be a multiple of 4.
* ``X`` — upper case hexadecimal. Length must be a multiple of 4.

The second are *interpretation* codes, which decode the bits as a number. These borrow
the letters used by :class:`Dtype` rather than Python's ``d``, because a ``Tibs`` has
both a signed and an unsigned reading and there's no sensible way to guess which you
meant:

* ``u`` — the unsigned integer value, as given by :attr:`Tibs.u`.
* ``i`` — the two's complement signed value, as given by :attr:`Tibs.i`.

Both families work at any length, though the interpretation codes need at least one bit
to interpret.

So the same 32 bits can be shown four different ways::

    >>> f"{packet:x}, {packet:b}"
    'ac804f4b, 10101100100000000100111101001011'
    >>> f"{packet:u}, {packet:i}"
    '2894090059, -1400877237'
    >>> f"{Tibs('0o7531'):o}"
    '7531'

.. note::

    The representation codes are not integer formats, so leading zeros are kept and the
    length of the output always tells you the length of the data. ``f"{Tibs('0x0f'):b}"``
    is ``'00001111'``, whereas ``f"{15:b}"`` is ``'1111'``. If you want the number, ask
    for the number with ``u`` or ``i``.

Prefixes and grouping
^^^^^^^^^^^^^^^^^^^^^

The ``#`` flag adds the usual ``0x``, ``0X``, ``0b`` or ``0o`` prefix, and ``_`` inserts
separators between groups of digits. Both are useful for reading long values, and
because the string constructors ignore underscores and understand the prefixes, a value
formatted with ``#``, with or without grouping, can be fed straight back in::

    >>> f"{packet:#_x}"
    '0xac80_4f4b'
    >>> Tibs(f"{packet:#_x}") == packet
    True

.. warning::

    If you combine ``#`` with an alignment, the ``#`` must come *after* it. A ``#``
    immediately followed by an alignment character is read as a fill character
    instead, which is easy to miss::

        >>> f"{packet:>#12x}"
        '  0xac804f4b'
        >>> f"{packet:#>12x}"
        '####ac804f4b'

    This is how the mini-language works for any type, but it bites more often here
    because a prefix is so often what you want.

Python fixes the group size at four digits. That's often the wrong size for binary data,
so tibs lets you set it using the precision field, which the standard mini-language
leaves unused for these types::

    >>> f"{packet:_b}"
    '1010_1100_1000_0000_0100_1111_0100_1011'
    >>> f"{packet:_.8b}"
    '10101100_10000000_01001111_01001011'
    >>> f"{packet:_.2x}"
    'ac_80_4f_4b'

Grouping a ``u`` or ``i`` value works the way it does for any other Python integer, so
you get three-digit groups and ``,`` is available as well::

    >>> f"{packet:_u}"
    '2_894_090_059'
    >>> f"{packet:,i}"
    '-1,400,877,237'

Width and alignment
^^^^^^^^^^^^^^^^^^^

Fill, alignment and width all behave as they do elsewhere in Python, which is handy for
lining up columns of registers or packet fields::

    >>> for name, value in [('ctrl', Tibs('0x0f')), ('status', Tibs('0xbeef'))]:
    ...     print(f"{name:>8}  {value:>12x}")
        ctrl            0f
      status          beef

.. note::

    Two details differ from integer formatting, both because a ``Tibs`` is a sequence
    rather than a number.

    Groups are counted from bit zero, so it's the *last* group that comes up short
    rather than the first. ``f"{Tibs('0b101010101'):_b}"`` gives ``'1010_1010_1'``,
    where the equivalent integer format gives ``'1_0101_0101'``.

    Padding is added after grouping and is never itself grouped, so the separators stay
    lined up with real bit positions.

The fill character has to be something that can't be mistaken for the data, so zero
padding is not available for ``b``, ``o``, ``x`` and ``X``::

    >>> f"{Tibs('0xf'):#06x}"
    Traceback (most recent call last):
    ...
    ValueError: Zero padding is not allowed with the 'x' format type, because the padding could not be told apart from the data and would change its apparent length. Align with '<', '>' or '^' to pad with spaces instead, or use the 'u' or 'i' type code for a numeric interpretation.

Zero padding an integer is harmless, because leading zeros don't change what an integer
is. Here they would: a 4-bit value padded to ``'0x000f'`` reads as a 16-bit one, and
comes back as a 16-bit one if you feed it in again. The same goes for any other fill
that is a valid digit for the type, such as ``f`` for hex or ``1`` for binary. Digits
that can't appear in that base are fine, as is anything else::

    >>> f"{Tibs('0xf'):*>6x}"
    '*****f'
    >>> f"{Tibs('0b1111'):8>6b}"
    '881111'

If you want a fixed number of bits rather than a fixed number of characters, change the
data rather than its presentation, for example with :meth:`Tibs.from_u`.

A few other things that are meaningful for numbers are also rejected for the
representation codes: the sign characters ``+``, ``-`` and space, and the ``,``
separator. All of them are accepted by ``u`` and ``i``, which really are numbers.

Long values
^^^^^^^^^^^

``str()`` gives up on very long values and shows a truncated version with the length
appended, and a format spec with no type code does the same. An explicit type code
never truncates::

    >>> t = Tibs.from_zeros(50_000)
    >>> f"{t}".endswith('... # length=50000')
    True
    >>> len(f"{t:b}")
    50000
