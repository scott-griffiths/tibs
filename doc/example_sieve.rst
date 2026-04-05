.. currentmodule:: tibs


Sieve of Eratosthenes
---------------------

Somehow this one feels like the 'hello world' of bit manipulation libraries.
It's a very, very inefficient method of finding prime numbers by repeatedly
setting all multiples of the prime numbers it finds to False.

This code calculates the first hundred million primes, counts them, and then counts
the number of twin primes by searching for all ``101`` sequences.


.. literalinclude:: ../examples/sieve.py
   :language: python

This will print out the start and end of the prime sequence as a hex string::

    3514510504510414114110404 ... 0000000100400010010000400

The prime values data only uses 1 bit per value, and this code runs in under half a second on my laptop.

