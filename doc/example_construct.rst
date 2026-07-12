.. currentmodule:: tibs


Building a binary header
------------------------

When a format specification gives field widths in bits, a small table of
:class:`Dtype` objects can keep the field layout in one place. This example
uses that table to build and decode the fixed part of an MPEG sequence header.

.. literalinclude:: ../examples/construct.py
   :language: python

This is the scale where format-driven bit assembly is useful: compact headers,
control words, and test vectors. For a complete container or protocol parser,
use a higher-level parser for the structure and keep ``Tibs`` for the fields
that need bit-accurate construction or interpretation.
