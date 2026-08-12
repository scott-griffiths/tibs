import sys
import sysconfig
import threading
import time

from tibs import Tibs

# Averaging 24 million 12-bit samples, split across a varying number of threads.
VALUES = 24_000_000
data = Tibs.from_random(VALUES * 12, seed=b"tibs")


def average_value(data, threads):
    """Average up every sample in `data`, split across threads."""
    size = (VALUES // threads) * 12
    totals = [0] * threads

    def work(i):
        def go():
            # Tibs are immutable and slices share their storage.
            # No data is copied here, and threads can't lock each other.
            piece = data[i * size:(i + 1) * size]
            totals[i] = sum(piece.to_values("u12"))

        return go

    workers = [threading.Thread(target=work(i)) for i in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    return sum(totals) / VALUES


free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
# A free-threaded build can still have the GIL switched on by importing other
# extensions that don't support free threading.
gil_on = getattr(sys, "_is_gil_enabled", lambda: True)()
print(f"{'free-threaded' if free_threaded else 'GIL-enabled'} build, "
      f"GIL {'on' if gil_on else 'off'}")

baseline, baseline_average = None, None
for threads in (1, 2, 4, 8):
    start = time.perf_counter()
    average = average_value(data, threads)
    elapsed = time.perf_counter() - start
    if thread_count == 1:
        baseline = elapsed
        baseline_average = average
    assert average == baseline_average
    print(f"{threads} threads: {elapsed * 1000:6.1f} ms   {baseline / elapsed:.2f}x")
