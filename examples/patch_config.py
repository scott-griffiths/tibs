from tibs import Tibs


ENABLED_BIT = 0
DEBUG_BIT = 1
MODE = slice(2, 5)
RETRY_LIMIT = slice(5, 9)
TIMEOUT_SECONDS = slice(9, 16)


def build_config(enabled, debug, mode, retry_limit, timeout_seconds):
    return Tibs.from_joined([
        [enabled, debug],
        Tibs.from_u(mode, 3),
        Tibs.from_u(retry_limit, 4),
        Tibs.from_u(timeout_seconds, 7),
    ])


def parse_config(config):
    return {
        "enabled": config[ENABLED_BIT],
        "debug": config[DEBUG_BIT],
        "mode": config[MODE].u,
        "retry_limit": config[RETRY_LIMIT].u,
        "timeout_seconds": config[TIMEOUT_SECONDS].u,
    }


config = build_config(
    enabled=True,
    debug=False,
    mode=1,
    retry_limit=3,
    timeout_seconds=65,
)

assert config.hex == "89c1"
assert parse_config(config) == {
    "enabled": True,
    "debug": False,
    "mode": 1,
    "retry_limit": 3,
    "timeout_seconds": 65,
}

patched = config.to_mutibs()
patched.unset(ENABLED_BIT)
patched.set(DEBUG_BIT)
patched[MODE] = Tibs.from_u(4, 3)
patched[RETRY_LIMIT] = Tibs.from_u(10, 4)

assert patched.hex == "6541"
assert parse_config(patched) == {
    "enabled": False,
    "debug": True,
    "mode": 4,
    "retry_limit": 10,
    "timeout_seconds": 65,
}
