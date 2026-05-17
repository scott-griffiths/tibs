.. currentmodule:: tibs


Reading little-endian registers
-------------------------------

Hardware register dumps and binary control protocols often store numeric values
least-significant byte first. You can keep the bytes exactly as they arrived and
use a little-endian view only when interpreting each register value.

.. literalinclude:: ../examples/little_endian_registers.py
   :language: python

The important part is that the raw byte order stays visible in the example:
``34 12`` is still the stored data, while ``register.le.u`` is the integer
interpretation of those two bytes.
