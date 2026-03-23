from tibs import Mutibs
from math import isqrt

# Create a billion True bits
limit = 1_000_000_000
is_prime = Mutibs.from_ones(limit)

# Zero and one aren't prime, so set these to 0
is_prime.unset([0, 1])

# Set all bits that are a multiple of the lowest known prime to 0
for i in range(2, isqrt(limit) + 1):
    if is_prime[i]:
        is_prime.unset(range(i * i, limit, i))

# We can now use it to count the primes less than a billion
primes_count = is_prime.count(1)
assert primes_count == 50_847_534

# Let's also see how many twin primes there are (primes that differ by 2).
twin_primes = is_prime.count([1, 0, 1])
assert twin_primes == 239_101
