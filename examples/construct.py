from tibs import Dtype, Tibs


SEQUENCE_HEADER = [
    ("start_code", Dtype("hex32"), "000001b3"),
    ("width", Dtype("u12"), 352),
    ("height", Dtype("u12"), 288),
    ("aspect_ratio", Dtype("u4"), 1),
    ("frame_rate", Dtype("u4"), 3),
    ("bit_rate_value", Dtype("u18"), 5040),  # Stored in 400 bit/s units.
    ("marker_bit", Dtype("bool"), True),
    ("vbv_buffer_size", Dtype("u10"), 20),
    ("constrained_parameters", Dtype("bool"), False),
]

header = Tibs.from_joined(
    dtype.pack(value)
    for _, dtype, value in SEQUENCE_HEADER
)

assert len(header) == 94

# The header is 94 bits, so byte serialization pads two zero bits on the right.
assert header.to_padded_bytes() == bytes.fromhex("000001b31601201304ec20a0")

fields = {}
offset = 0

for name, dtype, _ in SEQUENCE_HEADER:
    fields[name] = dtype.unpack(header, offset, offset + dtype.length)
    offset += dtype.length

assert offset == len(header)
assert fields == {
    name: value
    for name, _, value in SEQUENCE_HEADER
}
