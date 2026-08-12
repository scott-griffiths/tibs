import sys
import sysconfig
import timeit
from concurrent.futures import ThreadPoolExecutor

from tibs import Tibs

# Averaging 12 million 12-bit samples, split across a varying number of threads.
VALUES = 12_000_000
data = Tibs.from_random(VALUES * 12, seed=b"tibs")


def average_value(threads):
    """Average up every sample in `data`, split across threads."""
    size = (VALUES // threads) * 12

    def total(i):
        # Tibs are immutable and slices share their storage.
        # No data is copied here, and threads can't lock each other.
        return sum(data[i * size:(i + 1) * size].to_values("u12"))

    with ThreadPoolExecutor(threads) as pool:
        return sum(pool.map(total, range(threads))) / VALUES


build = "free-threaded" if sysconfig.get_config_var("Py_GIL_DISABLED") else "GIL-enabled"
gil = "on" if getattr(sys, "_is_gil_enabled", lambda: True)() else "off"
print(f"{build} build, GIL {gil}")

baseline = None
for threads in (1, 2, 4, 8):
    # Best of five runs to reduce noise.
    elapsed = min(timeit.repeat(lambda: average_value(threads), repeat=5, number=1))
    if threads == 1:
        baseline = elapsed
    print(f"{threads} threads: {elapsed * 1000:6.1f} ms   {baseline / elapsed:.2f}x")
