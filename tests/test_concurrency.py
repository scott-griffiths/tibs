#!/usr/bin/env python

"""Concurrency stress tests.

Tibs advertises ``Programming Language :: Python :: Free Threading :: 3 - Stable``.
Rust rules out data races *inside* a single call, so these tests probe the layer
above that:

* whether frozen types (``Tibs``, ``Dtype``, ``View``) can really be shared,
* whether a mutating call on a shared ``Mutibs`` is all-or-nothing, so that
  concurrent mutation cannot lose an update or expose a half-finished resize,
* whether objects that alias a ``Mutibs`` (``MutableView``, ``Reader``) stay safe
  when the source is resized underneath them,
* whether process-wide state (the byte to list-of-bools table in
  ``src/helpers/python.rs``) initialises safely when raced.

The tests run on GIL-enabled builds too, where they are a cheap smoke test for
re-entrancy and deadlock, but they only really bite on a free-threaded build::

    .venv-3.14t/bin/python -m pytest tests/test_concurrency.py

What counts as a failure
------------------------

PyO3 gives every non-frozen ``#[pyclass]`` an atomic borrow flag rather than a
lock, so on a free-threaded build two threads that touch the same ``Mutibs`` at
once do not serialise: the loser is refused. The refusal arrives as
``RuntimeError: Already borrowed`` from a generated method, or as a Rust panic
from one of the explicit ``Py::borrow``/``borrow_mut`` calls in ``view.rs``,
``reader.rs`` and ``iterator.rs``. Both are refusals *before* any mutation runs,
so they cost throughput and ergonomics, not correctness.

The tests below therefore split into two kinds:

* Correctness tests, which tolerate a refusal but not a wrong answer. A refused
  call must leave no trace, and a call that succeeds must see a coherent
  snapshot. These must always pass.
* Two ``xfail(strict=True)`` tests that state what tibs would need for the
  Free Threading classifier to mean what a caller would assume: mutation that
  serialises instead of refusing, and refusals that raise instead of panicking.
  They pass normally on GIL-enabled builds. If either starts passing on a
  free-threaded build, the limitation is gone and the marker should go with it.

The value invariant used throughout is that a ``Mutibs`` of all ones stays all
ones: appending ones, extending with ones, popping, deleting, inserting ones,
reversing, rotating and byte swapping all preserve it. A zero bit appearing
anywhere means a read or a write saw a buffer that was mid-resize.
"""

import subprocess
import sys
import sysconfig
import threading

import pytest

from tibs import Dtype, Mutibs, Reader, Tibs

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

THREADS = 8
# On a free-threaded build roughly 70% of the contended calls are refused, so
# the iteration count has to be well above the number of racing calls actually
# wanted. This costs under two seconds for the file.
ITERATIONS = 500

# The shared object is kept large enough that a single call takes long enough to
# overlap with the other threads. At this size the mixed read/write tests refuse
# several hundred of their calls, which is the sign that they really are racing.
BITS = 65536

# Races pick indices and lengths that another thread may have invalidated by the
# time the call runs. Those raise, and that is correct behaviour rather than a
# failure. ValueError covers ReadError and DecodeError, which subclass it.
EXPECTED_RACE_ERRORS = (IndexError, ValueError, TypeError, OverflowError)


def is_borrow_refusal(e):
    """True for PyO3 refusing simultaneous access to one non-frozen pyclass.

    Two shapes, one cause: a generated method returns ``PyBorrowError`` as
    ``RuntimeError``, while the hand-written ``Py::borrow`` calls panic. Neither
    has mutated anything by the time it gives up.
    """
    if isinstance(e, RuntimeError) and "borrow" in str(e).lower():
        return True
    # PanicException inherits from BaseException, and pyo3_runtime is only
    # importable once a panic has actually happened, so match on the name.
    return type(e).__name__ == "PanicException" and "borrow" in str(e).lower()


def run_concurrently(*workers, timeout=60.0):
    """Run each worker on its own thread, released together, and return their errors.

    Errors are returned rather than raised so that a caller can decide which are
    acceptable. ``BaseException`` is caught deliberately: a Rust panic becomes
    ``pyo3_runtime.PanicException``, which does not inherit from ``Exception``
    and would otherwise vanish silently with the thread.
    """
    barrier = threading.Barrier(len(workers))
    errors = [None] * len(workers)

    def make_target(index, worker):
        def target():
            barrier.wait()
            try:
                worker()
            except BaseException as e:  # noqa: BLE001 - see docstring
                errors[index] = e

        return target

    threads = [
        threading.Thread(target=make_target(i, w), name=f"worker-{i}")
        for i, w in enumerate(workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    alive = [t.name for t in threads if t.is_alive()]
    assert not alive, f"threads did not finish within {timeout}s (deadlock?): {alive}"
    return errors


def assert_no_failures(errors):
    """Fail on any error at all, for races that must be completely clean."""
    for e in errors:
        if e is not None:
            raise AssertionError(
                f"unexpected {type(e).__name__} from a worker thread: {e!r}"
            ) from e


def assert_only_race_errors(errors):
    """Fail on anything worse than a lost race or a refused borrow.

    Refused borrows are tolerated here because they are the subject of
    ``TestFreeThreadedGaps`` rather than of the invariant under test. What is not
    tolerated is a wrong answer, a MemoryError, a SystemError, or a panic from
    anywhere other than a borrow.
    """
    for e in errors:
        if e is None or isinstance(e, EXPECTED_RACE_ERRORS) or is_borrow_refusal(e):
            continue
        raise AssertionError(
            f"unacceptable {type(e).__name__} from a worker thread: {e!r}"
        ) from e


def tolerant(fn, count=ITERATIONS):
    """Worker that repeats `fn`, absorbing refusals so every worker runs to the end."""

    def worker():
        for _ in range(count):
            try:
                fn()
            except BaseException as e:  # noqa: BLE001
                if not is_borrow_refusal(e) and not isinstance(e, EXPECTED_RACE_ERRORS):
                    raise

    return worker


def counted(fn, count=ITERATIONS):
    """Worker that repeats `fn` and records how many attempts actually took effect.

    Returns ``(worker, tally)`` where ``tally`` is a one-element list holding the
    number of calls that were not refused. Each worker owns its own tally, so no
    synchronisation is needed.
    """
    tally = [0]

    def worker():
        for _ in range(count):
            try:
                fn()
            except BaseException as e:  # noqa: BLE001
                if not is_borrow_refusal(e):
                    raise
            else:
                tally[0] += 1

    return worker, tally


class TestBuildAssumptions:
    """The free-threading claim in pyproject.toml has to actually hold."""

    @pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded build")
    def test_import_does_not_re_enable_the_gil(self):
        # An extension module that does not declare Py_MOD_GIL_NOT_USED makes
        # CPython switch the GIL back on for the whole process. That would make
        # every other test in this file vacuous, and would silently invalidate
        # the Free Threading classifier.
        assert not sys._is_gil_enabled()


# Run in a subprocess so the once-lock is genuinely cold. In-process the table is
# built by whichever test touches bools first.
_BOOL_CHUNKS_RACE = """
import threading
from tibs import Tibs

source = Tibs.from_bytes(bytes(range(256)))
expected = source.to_bools()

barrier = threading.Barrier(16)
results = [None] * 16

def work(i):
    barrier.wait()
    results[i] = source.to_bools()

threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
for t in threads:
    t.start()
for t in threads:
    t.join(30)
    assert not t.is_alive(), "to_bools deadlocked"

assert all(r == expected for r in results), "to_bools raced its lookup table"
print("ok")
"""


class TestSharedProcessState:
    def test_bool_chunks_table_initialises_under_a_race(self):
        # BOOL_CHUNKS in src/helpers/python.rs is a process-wide PyOnceLock
        # holding 256 pre-built lists. Every thread here reaches it cold.
        result = subprocess.run(
            [sys.executable, "-c", _BOOL_CHUNKS_RACE],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"cold-start to_bools race failed:\n{result.stdout}\n{result.stderr}"
        )
        assert result.stdout.strip() == "ok"


class TestImmutableSharing:
    """Tibs, Dtype and View are `frozen` pyclasses, so sharing them must be exact."""

    def test_shared_tibs_reads_agree(self):
        source = Tibs.from_bytes(bytes(range(256)) * 4)

        def snapshot():
            return (
                source.to_hex(),
                source.count(1),
                source.find("0x5a"),
                source.to_bools()[:64],
                source[100:200].to_bin(),
                source.to_values("u8", 0, 256),
                source.le.u if len(source) % 8 == 0 else None,
            )

        expected = snapshot()
        results = [None] * THREADS

        def read(index):
            def worker():
                for _ in range(20):
                    results[index] = snapshot()

            return worker

        errors = run_concurrently(*(read(i) for i in range(THREADS)))
        assert_no_failures(errors)
        assert all(r == expected for r in results)

    def test_shared_dtype_packs_identically(self):
        dtype = Dtype("[u8; 4]")
        expected = Tibs.from_value(dtype, [1, 2, 3, 4])
        results = [None] * THREADS

        def pack(index):
            def worker():
                for _ in range(50):
                    packed = Tibs.from_value(dtype, [1, 2, 3, 4])
                    assert packed.to_value(dtype) == (1, 2, 3, 4)
                    results[index] = packed

            return worker

        errors = run_concurrently(*(pack(i) for i in range(THREADS)))
        assert_no_failures(errors)
        assert all(r == expected for r in results)


class TestMutationIsAllOrNothing:
    """A mutating call must either happen completely or not at all.

    These count the calls that were not refused and check the object against that
    count exactly. A short result is a lost update and a long one is a duplicated
    update; either would be corruption rather than an acceptable race.
    """

    def test_appends_are_never_lost_or_duplicated(self):
        m = Mutibs()
        workers, tallies = zip(
            *(counted(lambda: m.append(True)) for _ in range(THREADS))
        )
        errors = run_concurrently(*workers)
        assert_no_failures(errors)
        applied = sum(t[0] for t in tallies)
        assert len(m) == applied
        assert m.count(1) == applied, "an append wrote a bit it was not given"

    def test_extends_are_never_torn(self):
        m = Mutibs()
        chunk = Tibs("0b1011")
        workers, tallies = zip(
            *(counted(lambda: m.extend(chunk)) for _ in range(THREADS))
        )
        errors = run_concurrently(*workers)
        assert_no_failures(errors)
        applied = sum(t[0] for t in tallies)
        assert len(m) == applied * 4, "an extend applied part of its argument"
        assert m.count(1) == applied * 3

    def test_extends_from_python_iterables_are_never_torn(self):
        # Unlike the Tibs case this pulls its bits through the Python iterator
        # protocol, so the mutation is interleaved with arbitrary Python
        # execution while the Mutibs is being resized.
        m = Mutibs()
        workers, tallies = zip(
            *(
                counted(lambda: m.extend([True, False, True, True]))
                for _ in range(THREADS)
            )
        )
        errors = run_concurrently(*workers)
        assert_no_failures(errors)
        applied = sum(t[0] for t in tallies)
        assert len(m) == applied * 4
        assert m.count(1) == applied * 3

    def test_appends_and_pops_settle_correctly(self):
        m = Mutibs.from_ones(THREADS * ITERATIONS)
        pushes, push_tallies = zip(
            *(counted(lambda: m.append(True)) for _ in range(THREADS // 2))
        )
        pops, pop_tallies = zip(
            *(counted(lambda: m.pop()) for _ in range(THREADS // 2))
        )
        errors = run_concurrently(*pushes, *pops)
        assert_no_failures(errors)
        pushed = sum(t[0] for t in push_tallies)
        popped = sum(t[0] for t in pop_tallies)
        assert len(m) == THREADS * ITERATIONS + pushed - popped
        assert m.count(0) == 0, "a push or a pop left a zero bit behind"


class TestConcurrentMutationAndReading:
    """Readers must never observe a buffer caught mid-resize."""

    def test_all_ones_stays_all_ones(self):
        m = Mutibs.from_ones(BITS)
        one = Tibs("0b1")

        def grow():
            m.append(True)
            m.extend("0b111")
            m.insert(0, one)

        def shrink():
            if len(m) > BITS // 2:
                m.pop()
            if len(m) > BITS // 2:
                del m[0]

        def reorder():
            m.reverse()
            m.rotate_left(3)
            m.byte_swap()  # ValueError when not a whole number of bytes

        def read():
            # Each of these is a single call, so it sees one coherent snapshot
            # however the length is moving around it.
            assert m.count(0) == 0, "count saw a zero bit"
            assert "0" not in m.to_bin(), "to_bin saw a zero bit"
            assert m.find("0b0") is None, "find located a zero bit"
            assert m[:64].count(0) == 0, "a slice contained a zero bit"
            assert m.to_padded_bytes().count(0) == 0, "a byte read as zero"

        errors = run_concurrently(
            tolerant(grow),
            tolerant(shrink),
            tolerant(reorder),
            tolerant(read),
            tolerant(read),
            tolerant(read),
        )
        assert_only_race_errors(errors)
        assert m.count(0) == 0
        assert len(m) == len(m.to_bin())

    def test_logical_operations_against_a_moving_operand(self):
        stable = Tibs.from_ones(BITS)
        m = Mutibs.from_ones(BITS)

        def resize():
            m.append(True)
            m.pop()

        def combine():
            # A length mismatch raises, which is fine; a torn read would instead
            # give a wrong count.
            assert (m & stable).count(0) == 0
            assert m.count_and(stable) == BITS
            assert m.intersects(stable)

        errors = run_concurrently(
            tolerant(resize),
            tolerant(resize),
            *(tolerant(combine) for _ in range(4)),
        )
        assert_only_race_errors(errors)


class TestMutableViewAliasing:
    """MutableView holds a live Py<Mutibs>, so the source can move under it."""

    def test_views_read_and_write_coherently_while_the_source_resizes(self):
        m = Mutibs.from_ones(BITS)
        views = [m.field(i * 8, i * 8 + 7) for i in range(THREADS)]
        full = (1 << 8) - 1

        def resize():
            m.extend("0xff")
            if len(m) > BITS // 2:
                del m[-8:]

        def use_view(view):
            def step():
                assert len(view) == 8
                assert view.u == full, "a field view read a zero bit"
                view.u = full
                assert view.to_bin() == "1" * 8

            return tolerant(step)

        errors = run_concurrently(
            tolerant(resize),
            tolerant(resize),
            *(use_view(v) for v in views[: THREADS - 2]),
        )
        assert_only_race_errors(errors)
        assert m.count(0) == 0

    def test_views_onto_a_shrinking_source_fail_cleanly(self):
        # Here the source really does shrink out from under the views, so the
        # point is only that they refuse rather than read stale or freed bits.
        m = Mutibs.from_ones(BITS)
        edge = BITS // 2
        views = [m.field(edge + i * 8, edge + i * 8 + 7) for i in range(THREADS)]

        def truncate():
            if len(m) > 512:
                del m[-64:]

        def use_view(view):
            def step():
                # IndexError or ValueError once the field is off the end.
                assert view.u == 255

            return tolerant(step)

        errors = run_concurrently(
            tolerant(truncate), *(use_view(v) for v in views)
        )
        assert_only_race_errors(errors)


class TestReaderConcurrency:
    def test_independent_readers_over_one_source(self):
        source = Tibs.from_values("u8", list(range(256)) * 4)
        expected = list(range(256)) * 4

        def read():
            for _ in range(10):
                reader = Reader(source)
                assert reader.read_values("u8") == expected
                assert reader.at_end

        errors = run_concurrently(*(read for _ in range(THREADS)))
        assert_no_failures(errors)

    def test_shared_reader_stays_coherent(self):
        # One Reader driven by several threads. Which thread gets which value is
        # a race, but every value must be one that is really in the source and
        # the position must stay inside it.
        source = Tibs.from_values("u8", [0xAA] * 4096)
        reader = Reader(source)

        def step():
            value = reader.read_value("u8")
            assert value == 0xAA, f"shared reader produced {value:#x}"
            assert 0 <= reader.pos <= len(source)

        errors = run_concurrently(
            *(tolerant(step, count=4096 // THREADS) for _ in range(THREADS))
        )
        assert_only_race_errors(errors)

    def test_reader_over_a_mutating_source(self):
        m = Mutibs.from_values("u8", [0xFF] * (BITS // 8))

        def mutate():
            m.extend("0xff")
            if len(m) > BITS // 2:
                del m[-8:]

        def read():
            reader = Reader(m)
            while not reader.at_end:
                assert reader.read_value("u8") == 0xFF, "reader saw a zero byte"

        errors = run_concurrently(
            tolerant(mutate),
            tolerant(mutate),
            *(tolerant(read, count=20) for _ in range(4)),
        )
        assert_only_race_errors(errors)


class TestConstructionUnderLoad:
    """Allocation-heavy paths, run together, to shake out shared allocator state."""

    def test_parallel_construction_and_encoding(self):
        expected_bytes = bytes(range(256))

        def build():
            for _ in range(20):
                t = Tibs.from_bytes(expected_bytes)
                m = Mutibs.from_random(4096, seed=b"tibs")
                assert Tibs.decode(t.encode()) == t
                assert Mutibs.decode(m.encode()) == m
                assert Tibs.from_joined([t, t, t]).to_bytes() == expected_bytes * 3
                assert t.to_bytes() == expected_bytes

        errors = run_concurrently(*(build for _ in range(THREADS)))
        assert_no_failures(errors)


class TestFreeThreadedGaps:
    """What the Free Threading classifier would have to mean, but does not yet.

    Both of these pass on a GIL-enabled build. On a free-threaded build they are
    strict xfails, so if either starts passing the limitation has been fixed and
    the marker should be removed along with the note in the module docstring.
    """

    @pytest.mark.xfail(
        FREE_THREADED,
        strict=True,
        reason="PyO3 gives a non-frozen pyclass an atomic borrow flag, not a lock, "
        "so concurrent mutation of one Mutibs is refused rather than serialised",
    )
    def test_concurrent_mutation_serialises_instead_of_refusing(self):
        m = Mutibs()
        errors = run_concurrently(
            *(tolerant(lambda: m.append(True)) for _ in range(THREADS))
        )
        assert_no_failures(errors)
        assert len(m) == THREADS * ITERATIONS

    @pytest.mark.xfail(
        FREE_THREADED,
        strict=True,
        reason="view.rs, reader.rs and iterator.rs reach the source through the "
        "panicking Py::borrow/borrow_mut; contention should raise, not panic",
    )
    @pytest.mark.parametrize("subject", ["view", "reader"])
    def test_borrow_contention_raises_instead_of_panicking(self, subject):
        m = Mutibs.from_ones(BITS)
        target = m.field(0, 7) if subject == "view" else Reader(m)

        def resize():
            m.extend("0xff")
            if len(m) > BITS // 2:
                del m[-8:]

        def touch():
            # Deliberately not `tolerant`: a RuntimeError refusal is the outcome
            # this test wants, so it is swallowed here and a panic is left to
            # escape to the error list.
            for _ in range(ITERATIONS):
                try:
                    if subject == "view":
                        assert len(target) == 8
                        assert target.u == 255
                    else:
                        target.pos = 0
                        assert target.read_value("u8") == 255
                except (RuntimeError, *EXPECTED_RACE_ERRORS):
                    pass

        errors = run_concurrently(
            tolerant(resize),
            tolerant(resize),
            *(touch for _ in range(THREADS - 2)),
        )
        panics = [e for e in errors if type(e).__name__ == "PanicException"]
        assert not panics, f"Rust panicked on borrow contention: {panics[0]!r}"
