from tibs import Tibs


config = Tibs.from_joined([
    [True, False],       # Enabled, debug.
    Tibs.from_u(1, 3),  # Mode.
    Tibs.from_u(3, 4),  # Retry limit.
    Tibs.from_u(65, 7),  # Timeout in seconds.
])

assert config.hex == "89c1"
assert config[0] is True
assert config[1] is False
assert config[2:5].u == 1
assert config[5:9].u == 3
assert config[9:16].u == 65

patched = config.to_mutibs()
patched.unset(0)
patched.set(1)
patched[2:5] = Tibs.from_u(4, 3)
patched[5:9] = Tibs.from_u(10, 4)

assert patched.hex == "6541"
assert patched[0] is False
assert patched[1] is True
assert patched[2:5].u == 4
assert patched[5:9].u == 10
assert patched[9:16].u == 65
