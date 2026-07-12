.. currentmodule:: tibs


Reading little-endian registers
-------------------------------

Hardware register dumps and binary control protocols often store numeric values
least-significant byte first. The ``u16_le`` dtype lets you decode and encode
those values while keeping the bytes exactly as they arrived.

.. literalinclude:: ../examples/little_endian_registers.py
   :language: python

The important part is that the raw byte order stays visible in the example:
``34 12`` is still the stored data, while the decoded integer is ``0x1234``.
