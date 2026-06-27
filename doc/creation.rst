.. currentmodule:: tibs

Creation and Interpretation
---------------------------

Tibs and Mutibs can be constructed from a number of different types. The constructors for both types are identical, so I'll
use Tibs in this section, but it all applies equally well to Mutibs.

A wide range of ``from_`` constructor methods are provided:

* :meth:`Tibs.from_bin`: Create from a binary string, optionally starting with '0b'.
* :meth:`Tibs.from_oct`: Create from an octal string, optionally starting with '0o'.
* :meth:`Tibs.from_hex`: Create from a hex string, optionally starting with '0x'.
* :meth:`Tibs.from_bytes`: Create directly from a ``bytes``, ``bytearray`` or ``memoryview`` object.
* :meth:`Tibs.from_string`: Create from a formatted string.
* :meth:`Tibs.from_bools`: Convert each element in an iterable to a bool.
* :meth:`Tibs.from_zeros`: Initialise with ``0`` bits.
* :meth:`Tibs.from_ones`: Initialise with ``1`` bits.
* :meth:`Tibs.from_random`: Initialise with randomly set bits.
* :meth:`Tibs.from_u`: Create from an unsigned int to a given length.
* :meth:`Tibs.from_i`: Create from a signed int to a given length.
* :meth:`Tibs.from_f`: Create from an IEEE float to a 16, 32 or 64 bit length.
* :meth:`Tibs.from_value`: Create one fixed-width value from a dtype.
* :meth:`Tibs.from_values`: Create many fixed-width values from a dtype.
* :meth:`Tibs.from_joined`: Concatenate an iterable of objects.

Some examples::

    # Five bits from a binary string
    a = Tibs.from_bin('11001')

    # Directly from bytes, bytearray or a memoryview. Useful if creating from a file.
    b = Tibs.from_bytes(b'some_bytes')

    # Create bits from the truthiness of any iterator.
    c = Tibs.from_bools([1, 0, 1, 1, 1])

    # Optionally seeded random bits. There's also an option to use the OS's secure generator.
    d = Tibs.from_random(1000, seed=b'a_seed')

    # From a signed integer. The length can be any value up to 128 bits.
    e = Tibs.from_i(-384, 20)

    # From an unsigned integer. For whole-byte lengths a byte order can be used.
    f = Tibs.from_u(3, 32, byte_order=Endianness.Little)

    # Floating point values need to have a length of 16, 32 or 64.
    g = Tibs.from_f(-0.125, 16)

    # Hex, binary and octal strings can be parsed.
    h = Tibs.from_string('0xff01, 0b101')

    # An efficient way to join many other Tibs together.
    i = Tibs.from_joined([a, b, c, d, e, f, g, h])

Promotion to Tibs
^^^^^^^^^^^^^^^^^

The ``__init__`` method can also be called directly, which is often more convenient, if ever so slightly slower.
This will look at the type of object it's been given and try to promote it to a Tibs by delegating to :meth:`Tibs.from_string`,
:meth:`Tibs.from_bools` or :meth:`Tibs.from_bytes` for strings, iterables and bytes-like types respectively.
So for example ::

    s = Tibs('0xabc')     # Same as Tibs.from_string('0xabc')
    t = Tibs([1, 0, 1])   # Same as Tibs.from_bools([1, 0, 1])
    u = Tibs(b'hello')    # Same as Tibs.from_bytes(b'hello')

These types (string, iterables and bytes/bytearray/memoryview) can also be automatically promoted to ``Tibs``.
Most methods that
take a bit sequence will also accept another type they can promote in this way. So, for example, if you want to count
how many times the bit pattern ``101`` appears in a random bit sequence you could write::

    t = Tibs.from_random(1_000_000)  # A million random bits
    c = t.count(Tibs.from_bools([1, 0, 1]))

but it's more natural to use automatic promotion ::

    c = t.count([1, 0, 1])

This automatic promotion of these types to Tibs is quite pervasive in the library, and is generally recommended
for conciseness and clarity.
Equality is the main exception: ``Tibs`` and ``Mutibs`` compare equal only to other ``Tibs`` or ``Mutibs``
instances, not to strings, bytes or iterables.
Another exception is when performance is critical and not having the small overhead of examining the type and dispatching to
another method is significant — in this rare case using an explicit ``from_`` method for construction is preferred.


Data representations
^^^^^^^^^^^^^^^^^^^^

When a Tibs has been created there are multiple ways to interpret the data. These methods start with ``to_``.

A subset of these methods return lossless representations of the exact bit sequence, as a string or bytes.

* :meth:`Tibs.to_bin()` / :attr:`Tibs.bin`. Converts to a string of ``0`` and ``1`` characters. Always available.
* :meth:`Tibs.to_oct()` / :attr:`Tibs.oct`. Converts to an octal string. Length must be a multiple of 3.
* :meth:`Tibs.to_hex()` / :attr:`Tibs.hex`. Converts to a hexadecimal string. Length must be a multiple of 4.
* :meth:`Tibs.to_bytes()` / :attr:`Tibs.bytes`. Converts to a ``bytes`` object. Length must be a multiple of 8.

There is also :meth:`Tibs.to_padded_bytes()`, which appends 0 to 7 zero bits on the right before converting
to ``bytes``.

These ``to_`` methods accept optional ``start`` and ``end`` bit positions when you only want to convert part of
the data. With no parameters, the properties are provided as a convenient alias.
So instead of using ``t.to_bin()`` you can use just ``t.bin`` when you want the whole value.
For ``Tibs`` instances these properties are read-only.

Many of these representations need the data to have a length that's a correct multiple, for example ``bytes``
needs the data length to be a multiple of 8::

    >>> t = Tibs('0x4145c')
    >>> len(t)
    20
    >>> t.bin
    '01000001010001011100'
    >>> t.bytes
    ValueError: Cannot interpret as bytes - length of 20 is not a multiple of 8 bits.

.. note::

    ``Tibs`` can be arbitrary sizes, so lengths are always given in bits and not bytes.

To convert to a ``bytes`` object we need to change the length, for example by extending it with four ``0`` bits::

    >>> (t + '0x0').bytes
    b'AE\xc0'

This is a common enough operations that the :meth:`Tibs.to_padded_bytes` method is provided::

    >>> t.to_padded_bytes()
    b'AE\xc0'

Here we used the hex string ``'0x0'`` where a ``Tibs`` was expected, so it was promoted to a 4-bit ``Tibs``
before being used to create a 24-bit value that we could interpret as ``bytes``.

When you have one of these lossless representations, you can always reconstruct the original Tibs - there is a 1:1 relationship.
So ``t == Tibs.from_bin(t.to_bin())`` will always be true.


Data interpretations
^^^^^^^^^^^^^^^^^^^^

There are also a number of data interpretations that complement the data representations:

* :meth:`Tibs.to_u()` / :attr:`Tibs.u`. Interprets as an unsigned integer.
* :meth:`Tibs.to_i()` / :attr:`Tibs.i`. Interprets as a signed integer.
* :meth:`Tibs.to_f()` / :attr:`Tibs.f`. Converts to Python float. Length must be 16, 32 or 64.
* :meth:`Tibs.to_value`. Interprets one fixed-width value using a dtype.
* :meth:`Tibs.to_values` / :meth:`Tibs.to_values_iter`. Interprets repeated fixed-width values using a dtype.

Unlike the data representations, the interpretations can have a many-to-one relationship.
For example there are many ways for a ``Tibs`` to be constructed from the unsigned integer 3::

    u1 = Tibs.from_u(3, 5)   # binary 00011
    u2 = Tibs.from_u(3, 16)  # binary 00000000_00000011
    u3 = Tibs.from_u(3, 16, Endianness.Little)  # binary 00000011_00000000

These are three different ``Tibs``, but they all can have equal interpretations::

    >>> set([u1, u2, u3])
    {Tibs('0b00011'), Tibs('0x0003'), Tibs('0x0300')}
    >>> set([u1.u, u2.u, u3.le.u])
    {3}

For the value stored in ``u3`` a little-endian :class:`View` was used - we'll cover that later.


Repeated fixed-width values
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When every item uses the same fixed-width encoding, :class:`Dtype` strings make
the intent explicit and avoid writing a construction loop yourself. The most
common dtype forms are unsigned integers such as ``"u8"`` or ``"u12"``, signed
integers such as ``"i16"``, floats such as ``"f32"``, and string or byte
representations such as ``"hex16"`` and ``"bytes32"``. Use ``"bool"`` for a
single Python boolean bit, or ``"bitsN"`` when each value is itself a fixed-size
bit sequence decoded as :class:`Tibs`.

Use :meth:`Tibs.from_value` for one value, or :meth:`Tibs.from_values` for an
iterable of values::

    >>> Tibs.from_value("u8", 15)
    Tibs('0x0f')
    >>> samples = Tibs.from_values("u12", [0, 103, 2048, 4095])
    >>> samples.hex
    '000067800fff'

The matching interpretation methods decode values back from a bit sequence::

    >>> samples.to_values("u12")
    [0, 103, 2048, 4095]
    >>> samples.to_value("u12", 12, 24)
    103

Boolean and bit-sequence dtypes are useful when records mix flags with fields
that should stay as bits::

    >>> flags = Tibs.from_values("bool", [True, False, 1, 0])
    >>> flags.bin
    '1010'
    >>> Tibs.from_values("bits3", ["0b101", "0b010"]).to_values("bits3")
    [Tibs('0b101'), Tibs('0b010')]

If the dtype will be reused, creating it once can make the code read more naturally::

    >>> sample_dtype = Dtype("u12")
    >>> samples = sample_dtype.pack_values([0, 103, 2048, 4095])
    >>> sample_dtype.unpack(samples, 12, 24)
    103

For whole-byte numeric values, append ``_le`` or ``_be`` to the dtype string
when byte order matters::

    >>> Tibs.from_values("u16_le", [0x1234, 0xabcd]).hex
    '3412cdab'


Switching between Tibs and Mutibs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The other ``to_`` methods are for changing from the immutable ``Tibs`` to the mutable ``Mutibs`` and
vice versa.

If you have a ``Tibs`` but want to use one of the in-place modifying methods like :meth:`Mutibs.reverse`, then
you can first use :meth:`Tibs.to_mutibs` to create a mutable copy::

    >>> t = Tibs.from_i(-99, 16)
    >>> t.bin
    '1111111110011101'
    >>> m = t.to_mutibs()
    >>> m.reverse()
    >>> m.bin
    '1011100111111111'

In this simple case it's better to use the :meth:`Tibs.reversed` method, which creates and returns a new reversed
``Tibs``. There are a number of these ``reverse`` / ``reversed`` method pairs which either modify in place (for ``Mutibs``
only) or return a new instance.

There are also some methods that are only available on ``Tibs``, which take advantage of its immutable nature.
For example the :meth:`Tibs.chunks_iter` method, which returns an iterator over equal sized chunks of the data, is
not available for ``Mutibs`` as its data could change while the iterator is active. The list-returning
``chunks`` method is available on both ``Tibs`` and ``Mutibs``, while for the iterator form we can use
:meth:`Mutibs.to_tibs`::

    >>> m = Mutibs('0xb2')
    >>> m *= 3
    >>> for c in m.chunks(12): print(c.hex)
    b2b
    2b2
    >>> for c in m.to_tibs().chunks_iter(12): print(c.hex)
    ...
    b2b
    2b2

There is also the :meth:`Mutibs.as_tibs` method, which *moves* the data to a ``Tibs`` instead of making a copy.
This is more efficient if you don't need to use the ``Mutibs`` any more (as it will be empty after the move).
