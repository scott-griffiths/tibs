.. currentmodule:: tibs


Building a Bloom filter
-----------------------

A Bloom filter is a compact probabilistic set: inserted values should always be
reported as present, while values that were never inserted may occasionally be
reported as present. The core structure is just a bit array plus several hash
positions per item.

.. literalinclude:: ../examples/bloom_filter.py
   :language: python

This is intentionally small and deterministic. A production Bloom filter would
size ``BIT_COUNT`` and ``HASH_COUNT`` from the expected number of values and the
acceptable false-positive rate.
