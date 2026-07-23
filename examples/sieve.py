from tibs import Mutibs
from math import isqrt

# Create a hundred million True bits
limit = 100_000_000
is_prime = Mutibs.from_ones(limit)

# Zero and one aren't prime, so set these to 0
is_prime.unset([0, 1])

# Set all bits that are a multiple of the lowest known prime to 0
for i in range(2, isqrt(limit) + 1):
    if is_prime[i]:
        is_prime.unset(range(i * i, limit, i))

# We can now use it to count how many primes.
primes_count = is_prime.count()
assert primes_count == 5_761_455

# Let's also see how many twin primes there are (primes that differ by 2).
twin_primes = is_prime.count([1, 0, 1])
assert twin_primes == 440_312

# Print the start and the end as hexadecimal strings
print(f"{is_prime[0:100].hex} ... {is_prime[-100:].hex}")
