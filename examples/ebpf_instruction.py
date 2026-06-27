from tibs import Tibs


# Linux's eBPF ISA documentation gives this little-endian instruction as:
#     r1 += 0x11223344
#
# The standard lays out the instruction fields using bit labels from the least
# significant end of the encoded word, so use an LSB0 view. The multi-byte
# offset and immediate fields are little-endian, so combine it with an LE view.
#
# https://docs.kernel.org/bpf/standardization/instruction-set.html
INSTRUCTION_BYTES = bytes.fromhex("07 01 00 00 44 33 22 11")

BPF_ALU64 = 0x7
BPF_K = 0x0
BPF_ADD = 0x0


def decode_instruction(data):
    instruction = Tibs.from_bytes(data).lsb0.le

    return {
        "instruction_class": instruction.field(2, 0).u,
        "source_mode": instruction.field(3, 3).u,
        "operation_code": instruction.field(7, 4).u,
        "dst_reg": instruction.field(11, 8).u,
        "src_reg": instruction.field(15, 12).u,
        "offset": instruction.field(31, 16).i,
        "immediate": instruction.field(63, 32).u,
    }


def format_alu64_add_immediate(fields):
    assert fields["instruction_class"] == BPF_ALU64
    assert fields["source_mode"] == BPF_K
    assert fields["operation_code"] == BPF_ADD
    assert fields["src_reg"] == 0
    assert fields["offset"] == 0

    return f"r{fields['dst_reg']} += 0x{fields['immediate']:08x}"


fields = decode_instruction(INSTRUCTION_BYTES)

assert fields == {
    "instruction_class": BPF_ALU64,
    "source_mode": BPF_K,
    "operation_code": BPF_ADD,
    "dst_reg": 1,
    "src_reg": 0,
    "offset": 0,
    "immediate": 0x11223344,
}
assert format_alu64_add_immediate(fields) == "r1 += 0x11223344"
