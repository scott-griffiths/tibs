.. currentmodule:: tibs

Tibs vs Mutibs
--------------

* Tibs instances cannot change after they are created. This lets you use them as keys in dictionaries,
  they can be hashed and used in sets.
* Methods that return iterators over the data are available for Tibs, but not Mutibs. This is because for
  a ``Mutibs`` the data could change while the iterator is live. To use these methods on a ``Mutibs`` first
  convert to a ``Tibs``.


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

The two different halves here don't own a separate copy of the data here - they share storage with ``t`` so
this split is very quick and doesn't depend on the size of the data.
If we had used a ``Mutibs`` then a data copy would have had to happen.

For another example of shared storage, in the following code each chunk reuses
the data in ``t``::

    t = Tibs.from_random(8_000_000_000)
    for chunk in t.chunks(8000):
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