from tibs import Tibs

SEQUENCE_HEADER_CODE = "0x000001b3"
FIELD_BOUNDARIES = [32, 44, 56, 60, 64, 82, 83, 93]


def build_sequence_header(width, height, aspect_ratio, frame_rate, bit_rate_value):
    return Tibs.from_joined([
        SEQUENCE_HEADER_CODE,
        Tibs.from_u(width, 12),
        Tibs.from_u(height, 12),
        Tibs.from_u(aspect_ratio, 4),
        Tibs.from_u(frame_rate, 4),
        Tibs.from_u(bit_rate_value, 18),
        Tibs.from_u(1, 1),    # MPEG marker bit.
        Tibs.from_u(20, 10),  # VBV buffer size.
        Tibs.from_u(0, 1),    # Constrained parameters flag.
    ])


def parse_sequence_header(header):
    (
        start_code,
        width,
        height,
        aspect_ratio,
        frame_rate,
        bit_rate_value,
        marker_bit,
        vbv_buffer_size,
        constrained_parameters,
    ) = header.split_at(FIELD_BOUNDARIES)

    if start_code.hex != SEQUENCE_HEADER_CODE[2:]:
        raise ValueError("not an MPEG sequence header")
    if marker_bit.u != 1:
        raise ValueError("invalid MPEG marker bit")

    return {
        "width": width.u,
        "height": height.u,
        "aspect_ratio": aspect_ratio.u,
        "frame_rate": frame_rate.u,
        "bit_rate_bps": bit_rate_value.u * 400,
        "vbv_buffer_size": vbv_buffer_size.u,
        "constrained_parameters": constrained_parameters.u == 1,
    }


header = build_sequence_header(
    width=352,
    height=288,
    aspect_ratio=1,
    frame_rate=3,
    bit_rate_value=5040,
)

assert len(header) == 94
assert header.to_padded_bytes().hex() == "000001b31601201304ec20a0"
assert parse_sequence_header(header) == {
    "width": 352,
    "height": 288,
    "aspect_ratio": 1,
    "frame_rate": 3,
    "bit_rate_bps": 2_016_000,
    "vbv_buffer_size": 20,
    "constrained_parameters": False,
}
