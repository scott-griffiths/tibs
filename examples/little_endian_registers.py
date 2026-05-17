from tibs import Endianness, Tibs


REGISTER_NAMES = [
    "device_id",
    "status",
    "sample_rate_hz",
]


def read_registers(register_dump):
    bits = Tibs.from_bytes(register_dump)
    return {
        name: register.le.u
        for name, register in zip(REGISTER_NAMES, bits.chunks(16))
    }


def write_registers(registers):
    return Tibs.from_joined(
        Tibs.from_u(registers[name], 16, Endianness.Little)
        for name in REGISTER_NAMES
    ).bytes


register_dump = bytes.fromhex("34 12 cd ab 2a 00")

registers = read_registers(register_dump)
assert registers == {
    "device_id": 0x1234,
    "status": 0xabcd,
    "sample_rate_hz": 42,
}

updated = dict(registers)
updated["sample_rate_hz"] = 1000

assert write_registers(updated).hex() == "3412cdabe803"
