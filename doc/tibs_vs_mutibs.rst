.. currentmodule:: tibs

Tibs vs Mutibs
--------------

``Tibs`` and ``Mutibs`` share almost all of their behaviour; the difference is that
a ``Tibs`` is immutable while a ``Mutibs`` can be changed in place. That has a few
consequences:

* Tibs instances cannot change after they are created. This lets you use them as keys in dictionaries,
  they can be hashed and used in sets.
* Direct iteration (``for bit in x``) and methods that return iterators over the data are available for
  Tibs, but not Mutibs. This is because for a ``Mutibs`` the data could change while the iterator is live.
  To iterate over a ``Mutibs`` first convert to a ``Tibs`` with :meth:`Mutibs.to_tibs` or :meth:`Mutibs.as_tibs`.


Mutating and copy methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

``Mutibs`` has many mutating methods, which change the value in-place and return ``None``.
Many of these methods have siblings which do the same task but don't modify the instance and
instead return a new copy. These 'copy' methods are also available on the immutable ``Tibs``.

For example, :meth:`Mutibs.reverse` changes the existing object and returns
``None``::

    >>> m = Mutibs('0b101100')
    >>> result = m.reverse()
    >>> result is None
    True
    >>> m
    Mutibs('0b001101')

The copy-returning form leaves the original value alone::

    >>> t = Tibs('0b101100')
    >>> r = t.reversed()
    >>> t
    Tibs('0b101100')
    >>> r
    Tibs('0b001101')

The same copy-returning methods are also available on ``Mutibs`` when you want a
new mutable value without changing the original::

    >>> m = Mutibs('0b101100')
    >>> r = m.reversed()
    >>> m
    Mutibs('0b101100')
    >>> r
    Mutibs('0b001101')

.. csv-table::
   :header: "Mutibs mutating methods", "Tibs/Mutibs copy equivalent"

   ":meth:`~Mutibs.byte_swap`", ":meth:`~Mutibs.byte_swapped`"
   ":meth:`~Mutibs.deposit`", ":meth:`~Mutibs.deposited`"
   ":meth:`~Mutibs.insert`", ":meth:`~Mutibs.inserted`"
   ":meth:`~Mutibs.invert`", ":meth:`~Mutibs.inverted`"
   ":meth:`~Mutibs.replace`", ":meth:`~Mutibs.replaced`"
   ":meth:`~Mutibs.reverse`", ":meth:`~Mutibs.reversed`"
   ":meth:`~Mutibs.rotate_left`", ":meth:`~Mutibs.rotated_left`"
   ":meth:`~Mutibs.rotate_right`", ":meth:`~Mutibs.rotated_right`"
   ":meth:`~Mutibs.set`", ":meth:`~Mutibs.set_at`"
   ":meth:`~Mutibs.unset`", ":meth:`~Mutibs.unset_at`"


The linguistic oddities here are ``set_at()`` and ``unset_at()``, as the past-participle of 'set' is
also 'set', so the naming pattern failed (English is annoying sometimes).

Not all mutating methods have a copy equivalent - things like ``clear()`` don't make sense for a
``Tibs``, and you can use the ``+`` operator to do non-mutating extensions.


Switching between Tibs and Mutibs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``to_`` methods for changing from the immutable ``Tibs`` to the mutable ``Mutibs`` and
vice versa let you move between the two when you need a capability the other type has.

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
``Tibs``.

There are also some methods that are only available on ``Tibs``, which take advantage of its immutable nature.
For example the :meth:`Tibs.chunks_iter` method, which returns an iterator over equal sized chunks of the data, is
not available for ``Mutibs`` as its data could change while the iterator is active. The list-returning
``chunks`` method is available on both ``Tibs`` and ``Mutibs``, while for the iterator form we can use
:meth:`Mutibs.to_tibs`::

    >>> m = Mutibs('0xb2')
    >>> m *= 3
    >>> for c in m.chunks(12):
    ...     print(c.hex)
    b2b
    2b2
    >>> for c in m.to_tibs().chunks_iter(12):
    ...     print(c.hex)
    b2b
    2b2

There is also the :meth:`Mutibs.as_tibs` method, which *moves* the data to a ``Tibs`` instead of making a copy.
This is more efficient if you don't need to use the ``Mutibs`` any more (as it will be empty after the move).


Efficiency
^^^^^^^^^^

The ``tibs`` module is pretty well optimized, so most of the time you shouldn't need to worry about which
type to use - just use the one that's most convenient and use :meth:`Mutibs.to_tibs` and :meth:`Tibs.to_mutibs`
if you ever need to convert between them.

But if you do need to get the most speed, here are some guidelines:

Use Tibs if you don't need to mutate after creation
"""""""""""""""""""""""""""""""""""""""""""""""""""

There are lots of assumptions that can be made when the code knows that a value can't
change, and this can speed up operations.
It can also allow multiple instances to share underlying data storage, which
can be a big win. For example if slices are taken from a ``Tibs`` they don't need to
copy any data::

    t = Tibs.from_bytes(b'a_large_amount_of_data')
    first_half, second_half = t.split_at(len(t)//2)

The two different halves don't own a separate copy of the data here - they share storage with ``t`` so
this split is very quick and doesn't depend on the size of the data.
If we had used a ``Mutibs`` then a data copy would have had to happen.

For another example of shared storage, in the following code each chunk reuses
the data in ``t``::

    t = Tibs.from_random(8_000_000_000)
    for chunk in t.chunks_iter(8000):
        ...

No data needs to be copied as it knows the original can't change in value.

Mutating can also be a big win
""""""""""""""""""""""""""""""

If you are making modifications to the data and don't need to keep intermediate values
then it's better to use a ``Mutibs``. In the :doc:`example_sieve` example we have a loop
where we are setting multiples of prime numbers to be not prime::

    limit = 100_000_000
    is_prime = Mutibs.from_ones(limit)
    is_prime.unset([0, 1])
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime.unset(range(i * i, limit, i))  # Mutates is_prime

Each call to :meth:`Mutibs.unset` modifies it
in place. We could have instead used a ``Tibs``, and returned a new value on each iteration::

    limit = 100_000_000
    is_prime = Tibs.from_ones(limit)
    is_prime = is_prime.unset_at([0, 1])
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime = is_prime.unset_at(range(i * i, limit, i))  # No mutation

The :meth:`Tibs.unset_at` method returns a brand new ``Tibs`` on each iteration. Most
of the time in this version is spent allocating memory for each new ``Tibs``,
which is only being used to generate the next value before being destroyed.

On my laptop the mutating version runs about 30x faster; calculating the primes below
100 million in about half a second.

Don't incrementally build a Tibs
""""""""""""""""""""""""""""""""

There is a well-known anti-pattern in Python for constructing strings where you incrementally
build up a long string::

    s = ""
    for word in ["This ", "is ", "not ", "good"]:
        s += word

Python strings are immutable, so every stage creates a brand new string with an extra word
and throws the old string away. The way to avoid this is to use the string's ``join`` method ::

    s = "".join(["This ", "is ", "much ", "better"])

which constructs the string in one go.

You can have a very similar problem when creating a ``Tibs`` or a ``Mutibs``.
Constructing incrementally isn't very efficient as it throws away all the intermediates::

    t = Tibs()
    for u in range(256):
        t = t + Tibs.from_u(u, 8)  # Not recommended

We have a few ways to fix this. Either use a ``Mutibs`` and append to it::

    m = Mutibs()
    for u in range(256):
        m += Tibs.from_u(u, 8)

or use the :meth:`Tibs.from_joined` constructor to create it all in one go::

    t = Tibs.from_joined((Tibs.from_u(u, 8) for u in range(256)))

or in this particular case as the data types are all the same we can use simply::

    t = Tibs.from_values("u8", range(256))
