from hashlib import blake2b

from tibs import Mutibs


BIT_COUNT = 128
HASH_COUNT = 4


def positions(value):
    digest = blake2b(
        value.encode("utf-8"),
        digest_size=HASH_COUNT * 2,
        person=b"tibs-demo",
    ).digest()

    for offset in range(0, len(digest), 2):
        yield int.from_bytes(digest[offset:offset + 2], "big") % BIT_COUNT


def build_filter(values):
    bits = Mutibs.from_zeros(BIT_COUNT)
    for value in values:
        bits.set(list(positions(value)))
    return bits.to_tibs()


def maybe_contains(bits, value):
    return all(bits[position] for position in positions(value))


services = ["auth", "billing", "search", "metrics", "uploads"]
service_filter = build_filter(services)

assert 0 < service_filter.count(1) < BIT_COUNT

for service in services:
    assert maybe_contains(service_filter, service)

assert not maybe_contains(service_filter, "checkout")
