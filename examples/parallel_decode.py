import sys
import sysconfig
import threading
import time

from tibs import Tibs

# Adding up 2.4 million 12-bit samples, split across a varying number of threads.
VALUES = 2_400_000
data = Tibs.from_random(VALUES * 12, seed=b"tibs")


def timed_total(threads):
    """Add up every sample in `data`, split across `threads` threads."""
    size = (VALUES // threads) * 12
    totals = [0] * threads

    def work(i):
        def go():
            # Slicing a Tibs shares its storage rather than copying it, and a
            # Tibs is immutable, so the threads take no locks against each other and never wait.
            piece = data[i * size:(i + 1) * size]
            totals[i] = sum(piece.to_values("u12"))

        return go

    workers = [threading.Thread(target=work(i)) for i in range(threads)]
    start = time.perf_counter()
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    return sum(totals), time.perf_counter() - start


free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
# Being a free-threaded build is not the same as having the GIL switched off:
# importing an extension that doesn't declare it can turn the GIL back on for
# the whole process, and then none of this goes any faster. `_is_gil_enabled`
# is the one that actually answers the question, and arrived in 3.13.
gil_on = getattr(sys, "_is_gil_enabled", lambda: True)()
print(f"{'free-threaded' if free_threaded else 'GIL-enabled'} build, "
      f"GIL {'on' if gil_on else 'off'}")

baseline = None
for thread_count in (1, 2, 4, 8):
    # Best of five.
    total, elapsed = min(
        (timed_total(thread_count) for _ in range(5)), key=lambda r: r[1]
    )
    baseline = baseline if baseline is not None else elapsed
    print(f"{thread_count} threads: {elapsed * 1000:6.1f} ms   {baseline / elapsed:.2f}x")
