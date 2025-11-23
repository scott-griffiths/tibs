.. currentmodule:: tibs


Sieve of Eratosthenes
---------------------

Somehow this one feels like the 'hello world' of bit manipulation libraries.
It's a very, very inefficient method of finding prime numbers by repeatedly
setting all multiples of the prime numbers it finds to False.

This code calculates the first billion primes, counts them, and then counts
the number of twin primes by searching for all ``101`` sequences.


.. literalinclude:: ../examples/sieve.py
   :language: python

The prime values data only uses 1 bit per value, and this code runs in under 6 seconds on my laptop.
