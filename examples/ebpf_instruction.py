from tibs import Tibs


# Linux's eBPF ISA documentation gives this little-endian instruction as:
#     r1 += 0x11223344
#
# The standard lays out the instruction fields using bit labels from the least
# significant end of the encoded word, so use an LSB0 view. The multi-byte
# offset and immediate fields are little-endian, so combine it with an LE view.
#
# https://docs.kernel.org/bpf/standardization/instruction-set.html
instruction_bytes = bytes.fromhex("07 01 00 00 44 33 22 11")
instruction = Tibs.from_bytes(instruction_bytes).lsb0.le

instruction_class = instruction.field(2, 0).u
source_mode = instruction.field(3, 3).u
operation_code = instruction.field(7, 4).u
dst_reg = instruction.field(11, 8).u
src_reg = instruction.field(15, 12).u
offset = instruction.field(31, 16).i
immediate = instruction.field(63, 32).u

assert instruction_class == 0x7   # BPF_ALU64.
assert source_mode == 0x0         # BPF_K.
assert operation_code == 0x0      # BPF_ADD.
assert dst_reg == 1
assert src_reg == 0
assert offset == 0
assert immediate == 0x11223344

assert f"r{dst_reg} += 0x{immediate:08x}" == "r1 += 0x11223344"
