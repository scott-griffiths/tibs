.. currentmodule:: tibs


Reading a scattered register field
----------------------------------

Register layouts often split a field around fixed bits that sit in the middle of
it — a status flag or a reserved bit wedged between the high and low halves of a
value. The value's bits stay in order, they just aren't adjacent, so a plain
slice can't read them.

:meth:`~Tibs.extracted` gathers the bits picked out by a mask and packs them
together; :meth:`Mutibs.deposit` writes a value back into those positions,
leaving everything else untouched. They are the mask-driven, order-preserving
counterparts of the x86 PEXT/PDEP instructions, and the bit-level analogue of
reading and writing a contiguous field with :meth:`~Tibs.field`.

This example treats a 16-bit ADC result register whose 10-bit sample is
interrupted by a 2-bit status field, reads the sample, increments it, and writes
it back without disturbing the status bits.

.. literalinclude:: ../examples/scattered_field.py
   :language: python

Note that this only works because the sample's bits stay in order. ``extracted``
and ``deposit`` never reorder — for a field whose bits are permuted as well as
scattered (a RISC-V immediate, say) you would gather with a mask and then apply
the permutation separately, or select the bits by explicit index with
:meth:`View.from_indices`.
