from tibs import Tibs

start_code = '0x000001b3'
width = 352
height = 288
bit_rate = 5040
flags = [1, 1, 0]

t = Tibs.from_joined([
    start_code,
    Tibs.from_u(width, 12)
    Tibs.from_u(height, 12)
    Tibs.from_u(bit_rate, 13)
    flags])

assert t[0:32].to_hex() == '000001b3'
assert t[32:44].to_u() == width
assert t[-3:] == flags