from tibs import Tibs


# Scanning a compiled eBPF program for every "add an immediate to a register"
# instruction, whatever register it happens to target.
#
# Each eBPF instruction is 8 bytes: an opcode byte, a byte holding the source
# register in its high nibble and the destination register in its low nibble,
# a 16-bit offset and a 32-bit immediate.
#
# https://docs.kernel.org/bpf/standardization/instruction-set.html
program = Tibs(bytes.fromhex(
    "b7 01 00 00 01 00 00 00"   # r1 = 1
    "07 01 00 00 01 00 00 00"   # r1 += 1
    "b7 02 00 00 02 00 00 00"   # r2 = 2
    "07 02 00 00 ff 00 00 00"   # r2 += 255
    "0f 12 00 00 00 00 00 00"   # r2 += r1
    "95 00 00 00 00 00 00 00"   # exit
))

# The opcode must be 0x07 exactly, and the source register must be zero (that is
# what makes the operand an immediate). The destination register is a don't-care,
# so its nibble is left out of the mask.
NEEDLE = Tibs("0x0700")
MASK = Tibs("0xfff0")

# Without a mask this would only find additions to r0.
assert program.find_all("0x0700", byte_aligned=True) == []

matches = program.find_all(NEEDLE, mask=MASK, byte_aligned=True)
assert matches == [64, 192]

# A byte-aligned search can match part way through an instruction, so for a
# fixed-width ISA keep only the hits that land on an instruction boundary.
starts = [pos for pos in matches if pos % 64 == 0]
assert starts == matches

# Having found them, read the fields the way the standard labels them.
additions = []
for start in starts:
    instruction = program[start:start + 64].lsb0.le
    additions.append((instruction.field(11, 8).u, instruction.field(63, 32).u))

assert additions == [(1, 1), (2, 255)]
assert [f"r{reg} += {value}" for reg, value in additions] == ["r1 += 1", "r2 += 255"]
