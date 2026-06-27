.. currentmodule:: tibs


Decoding an eBPF instruction
----------------------------

The Linux eBPF ISA documentation defines instruction fields using low-order bit
labels and gives a concrete little-endian byte sequence for the instruction
``r1 += 0x11223344``.

This example keeps the original bytes visible, then uses ``lsb0.le`` to read
the fields using the same interpretation as the standard.

Standard reference:
https://docs.kernel.org/bpf/standardization/instruction-set.html

.. literalinclude:: ../examples/ebpf_instruction.py
   :language: python
