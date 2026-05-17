.. currentmodule:: tibs


Patching a packed config word
-----------------------------

Packed configuration words are a common place where immutable and mutable bit
strings are useful together. Build or receive the word as a ``Tibs``, promote it
to ``Mutibs`` while editing individual fields, then keep the patched value as
ordinary binary data.

.. literalinclude:: ../examples/patch_config.py
   :language: python

This example uses normal Python slices for fields whose positions are counted
from the left. For specifications that label fields from the least significant
bit, use the view helpers described in :doc:`views`.
