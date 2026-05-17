from tibs import Tibs


SYNC = Tibs("0xaa55")
HEADER_BITS = 32


def scan_records(log_bytes):
    bits = Tibs.from_bytes(log_bytes)

    for start in bits.find_all_iter(SYNC, byte_aligned=True):
        header = bits[start:start + HEADER_BITS]
        if len(header) != HEADER_BITS:
            continue

        record_type = header[16:24].u
        payload_length = header[24:32].u
        payload_start = start + HEADER_BITS
        payload_end = payload_start + payload_length * 8

        if payload_end <= len(bits):
            yield record_type, bits[payload_start:payload_end].bytes


log = (
    b"\x00\xff"
    + bytes.fromhex("aa 55 11 02 c0 01")
    + b"\x7f"
    + bytes.fromhex("aa 55 22 01 40")
    + b"\xaa"
)

records = list(scan_records(log))
assert records == [
    (0x11, b"\xc0\x01"),
    (0x22, b"\x40"),
]
