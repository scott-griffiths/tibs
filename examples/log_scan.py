from tibs import Tibs


SYNC = Tibs("0xaa55")


log = (
    b"\x00\xff"
    + bytes.fromhex("aa 55 11 02 c0 01")
    + b"\x7f"
    + bytes.fromhex("aa 55 22 01 40")
    + b"\xaa"
)

bits = Tibs(log)
records = []

for start in bits.find_all_iter(SYNC, byte_aligned=True):
    header = bits[start:start + 32]

    record_type = header[16:24].u
    payload_length = header[24:32].u
    payload = bits[start + 32:start + 32 + payload_length * 8]

    if len(payload) == payload_length * 8:
        records.append((record_type, payload.bytes))

assert records == [
    (0x11, b"\xc0\x01"),
    (0x22, b"\x40"),
]
