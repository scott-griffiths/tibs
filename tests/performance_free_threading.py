#!/usr/bin/env python
"""Rough free-threading scaling benchmark for tibs.

Fixed total work, split across T threads. Run the same file with a GIL-enabled
interpreter and a free-threaded one and compare the speedup columns.

    .venv/bin/python        tests/performance_free_threading.py
    .venv-3.14t/bin/python  tests/performance_free_threading.py
"""

import sys
import sysconfig
import threading
import time

from tibs import Mutibs, Tibs

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
THREAD_COUNTS = [1, 2, 4, 8]
REPEATS = 5  # min of this many, so a thermal blip doesn't become the result

TOTAL_BITS = 800_000_000  # 100 MB of data


def run_threads(jobs):
    """Run each job on its own thread, released together; return wall seconds."""
    barrier = threading.Barrier(len(jobs) + 1)

    def target(job):
        def go():
            barrier.wait()
            job()

        return go

    threads = [threading.Thread(target=target(j)) for j in jobs]
    for t in threads:
        t.start()
    barrier.wait()  # everything is spawned; start the clock with the workers
    start = time.perf_counter()
    for t in threads:
        t.join()
    return time.perf_counter() - start


def measure(make_jobs, threads):
    """Best of REPEATS for one thread count."""
    best = float("inf")
    for _ in range(REPEATS):
        jobs = make_jobs(threads)  # rebuilt each time: some jobs consume state
        best = min(best, run_threads(jobs))
    return best


def scaling(name, make_jobs, note=""):
    print(f"\n{name}")
    if note:
        print(f"  ({note})")
    baseline = None
    for threads in THREAD_COUNTS:
        seconds = measure(make_jobs, threads)
        if baseline is None:
            baseline = seconds
        print(
            f"  {threads} thread{'s ' if threads != 1 else '  '}"
            f"  {seconds * 1000:8.1f} ms   {baseline / seconds:5.2f}x"
        )


def pieces(source, n):
    """`n` equal slices of `source`. O(1) each - a Tibs slice shares storage."""
    size = len(source) // n
    return [source[i * size:(i + 1) * size] for i in range(n)]


# ---------------------------------------------------------------- workloads

DATA = Tibs.from_random(TOTAL_BITS, seed=b"bench")
# A needle that is not there, so every search scans its whole piece.
MISSING = Tibs.from_zeros(64)
BYTES_DATA = Tibs.from_random(80_000_000, seed=b"bench")  # 10 MB, for to_values


def counting(threads):
    return [lambda p=p: p.count(1) for p in pieces(DATA, threads)]


def searching(threads):
    return [lambda p=p: p.find(MISSING) for p in pieces(DATA, threads)]


def unpacking(threads):
    # u32 rather than u8: values above 256 are not in CPython's small-int
    # cache, so this really does allocate 2.5M objects per run.
    return [lambda p=p: p.to_values("u32") for p in pieces(BYTES_DATA, threads)]


def separate_mutibs(threads):
    # Each thread owns its container: the case the library is actually built for.
    per_thread = 2_000_000 // threads
    containers = [Mutibs() for _ in range(threads)]

    def work(m):
        def go():
            for _ in range(per_thread):
                m.append(True)

        return go

    return [work(m) for m in containers]


def shared_mutibs(threads):
    # One container, every thread writing to it: the serialised case.
    per_thread = 2_000_000 // threads
    m = Mutibs()

    def go():
        for _ in range(per_thread):
            m.append(True)

    return [go for _ in range(threads)]


def shared_tibs_reads(threads):
    # One immutable object read by every thread with start/end, rather than a
    # slice each. No lock is taken at all, so it should scale like the split
    # version - this is the control for "is slicing doing the work?".
    size = TOTAL_BITS // threads

    def work(start):
        def go():
            DATA.count(1, start, start + size)

        return go

    return [work(i * size) for i in range(threads)]


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]}  free-threaded={FREE_THREADED}", end="")
    if FREE_THREADED:
        print(f"  gil_enabled={sys._is_gil_enabled()}")
    else:
        print()
    print(f"best of {REPEATS}, fixed total work split across the threads")

    scaling("count(1) over 800 Mbit", counting, "popcount; memory-bandwidth bound")
    scaling("find() over 800 Mbit, no match", searching, "scans every bit")
    scaling("to_values('u32') over 80 Mbit", unpacking, "allocates 2.5M Python ints")
    scaling("append x2M to one Mutibs per thread", separate_mutibs, "no sharing")
    scaling("append x2M to one shared Mutibs", shared_mutibs, "fully serialised")
    scaling("count() on one shared Tibs", shared_tibs_reads, "same object, no slicing")
