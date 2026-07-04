.. currentmodule:: tibs


Building a binary header
------------------------

When a format specification gives field widths in bits, ``Tibs.from_joined`` is
a direct way to assemble a header while keeping each field visible. This example
builds and decodes the fixed part of an MPEG sequence header.

.. literalinclude:: ../examples/construct.py
   :language: python

This is the scale where direct bit assembly is useful: compact headers, control
words, and test vectors. For a complete container or protocol parser, use a
higher-level parser for the structure and keep ``Tibs`` for the fields that need
bit-accurate construction or interpretation.
