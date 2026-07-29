.. currentmodule:: tibs


Sieve of Eratosthenes
---------------------

Somehow this one feels like the 'hello world' of bit manipulation libraries.
It's a very, very inefficient method of finding prime numbers by repeatedly
setting all multiples of the prime numbers it finds to False.

This code calculates the primes below one hundred million, counts them, and then counts
the number of twin primes by searching for all ``101`` sequences. Then it asks for
the primes themselves, which for this container means asking where the set bits are.


.. literalinclude:: ../examples/sieve.py
   :language: python

The output is the first few primes, then the first prime at or after ninety-nine
million together with the largest one below the limit, then the start and end of
the whole sequence as hex strings::

    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, ... 79, 83, 89, 97]
    99000007 99999989
    3514510504510414114110404 ... 0000000100400010010000400

The prime values data only uses 1 bit per value, and this code runs in under half a second on my laptop.
