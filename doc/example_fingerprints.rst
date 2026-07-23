.. currentmodule:: tibs


Comparing fingerprints
----------------------

Sometimes the bits aren't a sequence to be read but a *set* to be compared. An
item is reduced to a fixed-length fingerprint in which bit ``i`` is set if the
item has feature ``i``, and questions about the items become questions about
which bits they share. Chemical similarity searching, near-duplicate detection
and Bloom filters are all this shape.

The comparisons could all be written with ``&``, ``|`` and ``^``, but each of
those builds a whole new fingerprint only to count its bits and throw it away.
The methods used here give the same answers without the object in the middle:

.. list-table::
   :header-rows: 1

   * - Method
     - Question it answers
   * - :meth:`~Tibs.count`
     - How many features does this item have?
   * - :meth:`~Tibs.count_and`
     - How many do these two share?
   * - :meth:`~Tibs.count_or`
     - How many do they have between them?
   * - :meth:`~Tibs.count_xor`
     - How many do they disagree about? (the Hamming distance)
   * - :meth:`~Tibs.count_andnot`
     - How many does the first have that the second lacks?
   * - :meth:`~Tibs.intersects`
     - Do they have anything in common at all?
   * - :meth:`~Tibs.is_subset_of`
     - Does this item have all the required features?

.. literalinclude:: ../examples/fingerprints.py
   :language: python

The last two matter most on large fingerprints, because they stop as soon as the
answer is settled rather than examining every bit. Screening a library with
:meth:`~Tibs.is_subset_of` before scoring anything is the usual pattern, and the
screen is far cheaper than the score.

See :ref:`comparing_two_containers` for the full set of these methods.
