from tibs import Tibs


def pack_values(values, width):
    max_value = 1 << width
    for value in values:
        if not 0 <= value < max_value:
            raise ValueError("value does not fit in the requested width")
    return Tibs.from_joined(Tibs.from_u(value, width) for value in values)


def unpack_values(bits, width):
    return [chunk.u for chunk in bits.chunks(width)]


levels = [3, 17, 31, 0, 12, 4]
packed = pack_values(levels, width=5)

assert len(packed) == 30
assert packed.bin == "000111000111111000000110000100"
assert unpack_values(packed, width=5) == levels

try:
    pack_values([32], width=5)
except ValueError:
    pass
else:
    raise AssertionError("expected a value-width error")
