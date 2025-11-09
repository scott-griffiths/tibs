.. currentmodule:: bitformat
.. _use_case_construction:

Constructing binary data
------------------------

You have some values of a particular type, say an integer or a floating point format, and you want to store them as binary data.

a) A simple Dtype
^^^^^^^^^^^^^^^^^

The simplest data types are just a type and a length. Things like ``'i16'`` for a 16 bit signed integer, or ``'f32'`` for a 32 bit floating point number.
Actually some types are even simpler as they don't need the length - a ``'bool'`` is always a single bit. ::

    >>> b = Bits.from_dtype('u12', 160)
    >>> b.u
    160
    >>> b.bin
    '000010100000'
    >>> b.hex
    '0a0'

Here we create a :class:`Bits` object using a :class:`Dtype` constructed from the ``'u12'`` string, and give it the value of 160.
There are some convenient properties of the ``Bits`` object that allow it to be converted back into a value by interpreting it with a new data type.
In this case we first used the ``u`` property to check that when interpreted as an unsigned integer it is indeed the value we expect. Then the ``bin`` and ``hex`` properties emphasis that this is indeed an object representing a 12-bit sequence.

Once the ``Bits`` is created it is just a collection of bits, and has no way of knowing how it was created, so all of these interpretations are equally true and valid.

The properties used above are useful and convenient but not very general. The more general way of seeing the value would be to use the :meth:`Bits.unpack` method. ::

    >>> b.unpack('u')
    160

You can also construct from literal binary, octal or hexadecimal values::

    >>> p = Bits.from_dtype('bin', '001')
    >>> q = Bits.from_dtype('hex', 'beef')

There are several class constructor methods available to create ``Bits`` objects. As well as :meth:`Bits.from_dtype` new instances can be constructed from the :meth:`~Bits.from_bytes`, :meth:`~Bits.from_bools`, :meth:`~Bits.from_joined`, :meth:`~Bits.from_zeros`, :meth:`~Bits.from_ones` and :meth:`~Bits.from_string` class methods .

Creating from a formatted string is often very convenient::

    >>> b = Bits.from_string('u12 = 160')
    >>> p = Bits.from_string('bin = 001')
    >>> q = Bits.from_string('hex = beef')

It's so frequently used that the default constructor for ``Bits`` is just an alias for the :meth:`~Bits.from_string` method. Add to that the short-cut of using a ``'0b'`` prefix for binary and ``'0x'`` for hexadecimal, you can instead write ::

    >>> b = Bits('u12 = 160')
    >>> p = Bits('0b001')
    >>> q = Bits('0xbeef')


To get back from the ``Bits`` object to a Python-native object you can use the :meth:`Bits.to_bytes` method::

    >>> q.to_bytes()
    b'\xbe\xef'

Note that if your ``Bits`` is not a whole number of bytes long then this method will add up to seven zero bits to make it a whole-byte quantity.
If you are doing lots of bit manipulation work then converting to bytes is often the final stage.


b) An array Dtype
^^^^^^^^^^^^^^^^^

You can also have a data type that represents an fixed-size array of simple data types. For example you might want to have a group of boolean flags, or a chunk of binary data as ``'u8'`` bytes. Note that this shouldn't be confused with the :class:`Array` class, which is a higher level mutable container that we'll come to later.

The format strings for these are borrowed from the Rust programming language::

    >>> a = Bits.from_dtype('[u8; 5]', (104, 101, 108, 108, 111))
    >>> f = Bits.from_dtype('[bool; 4]', [True, True, False, True])

The :class:`DtypeArray` made from the string ``'[u8; 5]'`` represents an array of five 8-bit unsigned integers, and ``'[bool; 4]'`` is an array of four single bit booleans.
Of course these array data types also work when unpacking::

    >>> a.unpack('[i4; 10]')
    (6, -8, 6, 5, 6, -4, 6, -4, 6, -1)

where we have chosen to unpack the 40 bits of data as ten signed 4 bit integers.

c) A sequence of Dtypes
^^^^^^^^^^^^^^^^^^^^^^^

The array Dtype above can only be used for a sequence of data types that are the same. You can also mix and match data types in a :class:`DtypeTuple`. ::

    >>> dt = Dtype('[bool; 4], u12, u12')
    >>> t = Bits.from_dtype(dt, [[1, 1, 0, 1], 160, 120])
    >>> t.bin
    '1101000010100000000001111000'

We've been a bit more explicit when creating the ``Dtype`` here, as we could have just supplied the initialisation string to the ``from_dtype`` method and it would have worked just as well. We are going to use the data type again though, so creating the ``Dtype`` object means it won't have to parse the string more than once ::

    >>> t.unpack(dt)
    [(True, True, False, True), 160, 120]


d) A format specification
^^^^^^^^^^^^^^^^^^^^^^^^^

For more complex needs the :class:`Format` class allows a rich specification language that we'll only touch upon in this section.

Combining some of our earlier creations we could make this format::

    >>> fmt = Format("header: (const hex4 = 0x0147, flags: [bool; 4], w: u12, h: u12)")
    >>> print(fmt)
    header: (
        const hex4 = 0147
        flags: [bool; 4]
        w: u12
        h: u12
    )

Here we have introduced named fields and const fields. It's then easy to set and get the named fields::

    >>> fmt['flags'].value = [1, 1, 0, 1]
    >>> fmt['w'].value = 160
    >>> fmt['h'].value = 120
    >>> print(fmt)
    header: (
        const hex4 = 0147
        flags: [bool; 4] = (True, True, False, True)
        w: u12 = 160
        h: u12 = 120
    )
    >>> fmt.unpack()
    ['0147', (True, True, False, True), 160, 120]
    >>> fmt.to_bytes()
    b'\x01G\xd0\xa0\x07\x80'


Another way to create using the format is via the :meth:`Format.pack` method::

    >>> fmt.clear()
    >>> fmt.pack([[0, 0, 0, 0], 999, 5])
    >>> print(fmt)
    header: (
        const hex4 = 0147
        flags: [bool; 4] = (False, False, False, False)
        w: u12 = 999
        h: u12 = 5
    )
