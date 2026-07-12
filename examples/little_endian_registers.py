from tibs import Tibs


register_dump = bytes.fromhex("34 12 cd ab 2a 00")

device_id, status, sample_rate_hz = Tibs.from_bytes(register_dump).to_values("u16_le")

assert device_id == 0x1234
assert status == 0xabcd
assert sample_rate_hz == 42

updated = Tibs.from_values("u16_le", [device_id, status, 1000])

assert updated.hex == "3412cdabe803"
