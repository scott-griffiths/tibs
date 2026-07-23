.. currentmodule:: tibs


Scanning for an instruction pattern
-----------------------------------

Instruction encodings are mostly fixed bits with a few variable fields punched
through them, which makes them awkward to search for: the opcode you want is
always the same, but the register it operates on is not.

The ``mask`` parameter takes care of that. Only the bits set in the mask have to
match, so the register nibble can simply be left out of it and every instruction
using that opcode is found regardless of which register it targets.

This example scans a compiled eBPF program for every instruction that adds an
immediate to a register, then reads the destination register and the immediate
out of each hit using an ``lsb0.le`` view. Searching and field decoding are the
same object seen two ways.

Standard reference:
https://docs.kernel.org/bpf/standardization/instruction-set.html

.. literalinclude:: ../examples/instruction_scan.py
   :language: python

Note the alignment check. ``byte_aligned=True`` restricts matches to byte
boundaries, which is not the same as instruction boundaries for an ISA with
8-byte instructions, so a match part way through one is possible. Filtering on
``pos % 64 == 0`` keeps only the real ones.
