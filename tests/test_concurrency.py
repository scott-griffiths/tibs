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
lock, so two threads reaching the same ``Mutibs`` do not serialise on their own:
the loser is refused with ``RuntimeError: Already borrowed``. Every ``Mutibs``
method therefore runs inside CPython's per-object critical section, which makes
the second thread *wait* instead. See ``src/helpers/locking.rs``.

The tests below split into two kinds:

* Correctness tests, which tolerate a refusal but not a wrong answer. A refused
  call must leave no trace, and a call that succeeds must see a coherent
  snapshot.
* Serialisation tests, which assert that calls are not refused at all.

The second kind only applies to a free-threaded build. `with_critical_section`
compiles to a direct call when the GIL is present, so there is no serialisation
to assert there - the borrow flag behaves exactly as it did before any of this.

Nor can it be made absolute even on a free-threaded build. The interpreter may
suspend a critical section - for a GC pause, or a signal check inside a long
search - while the borrow is still held, and a thread entering the section then
finds the borrow taken. That is rare, costs a refusal rather than correctness,
and is why ``test_no_public_attribute_refuses`` counts refusals per attribute
rather than requiring zero: an unwrapped method loses a large share of its own
calls, while a suspended section costs one call of whatever was running.

A refusal must arrive as a Python exception. ``view.rs``, ``reader.rs``,
``tibs_.rs`` and ``mutibs.rs`` reach their source through ``try_borrow`` for
exactly this reason, and ``test_borrow_contention_raises_instead_of_panicking``
holds them to it: a Rust panic would arrive as ``pyo3_runtime.PanicException``,
which does not inherit from ``Exception`` and so escapes a caller's ``except``.

The value invariant used throughout is that a ``Mutibs`` of all ones stays all
ones: appending ones, extending with ones, popping, deleting, inserting ones,
reversing, rotating and byte swapping all preserve it. A zero bit appearing
anywhere means a read or a write saw a buffer that was mid-resize.
"""

import collections
import copy
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

    The refusal is a ``RuntimeError`` carrying ``PyBorrowError``, raised before
    the call has mutated anything. A panic is matched too, so that the tests here
    stay honest about where refusals come from rather than passing by accident if
    a ``try_borrow`` is ever lost; the panic case is what
    ``test_borrow_contention_raises_instead_of_panicking`` fails on.
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


def repeat_op(target, op, count=ITERATIONS):
    """Worker that applies `op` to `target` `count` times, tolerating nothing."""

    def worker():
        for _ in range(count):
            op(target)

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
            encoding="utf-8",
            errors="replace",
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

    def test_reader_methods_never_refuse(self):
        # A Reader over a Mutibs holds two objects at once: its own cursor and
        # the source it reads through. Both sections are entered together, so a
        # thread writing the source directly queues behind a read in flight
        # rather than being refused by the borrow it is holding.
        m = Mutibs.from_values("u8", [0xFF] * 64)
        reader = Reader(m)
        needle = Tibs("0xff")

        def read():
            for _ in range(ITERATIONS):
                reader.pos = 0
                assert reader.read_value("u8") == 0xFF
                assert reader.read_bits(8) == needle
                assert reader.peek_value("u8") == 0xFF
                assert reader.peek_bits(8) == needle
                reader.align(8)
                reader.byte_pos = 1
                assert reader.seek_to(needle)
                assert not reader.at_end
                assert reader.remaining <= len(reader.source)
                repr(reader)
                with reader.bookmark():
                    reader.read_bits(8)
                assert copy.copy(reader).read_bits(8) == needle

        def write():
            for _ in range(ITERATIONS):
                m.write_bytes(b"\xff" * 64)

        errors = run_concurrently(read, read, write, write)
        assert_no_failures(errors)

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
    """How a refused call has to behave, and the one thing still missing."""

    # Methods wrapped in the critical section, and the bits each call adds.
    # Extend this list as the conversion proceeds: an entry here asserts that
    # the method never refuses, which is stronger than merely never corrupting.
    SERIALISED = [
        ("append", lambda m: m.append(True), 1),
        ("extend_tibs", lambda m: m.extend(Tibs("0b1011")), 4),
        ("extend_list", lambda m: m.extend([True, False, True, True]), 4),
    ]

    @pytest.mark.parametrize(
        "op,per_call", [(op, n) for _, op, n in SERIALISED], ids=[i for i, _, _ in SERIALISED]
    )
    def test_converted_mutation_serialises(self, op, per_call):
        m = Mutibs()
        errors = run_concurrently(*(repeat_op(m, op) for _ in range(THREADS)))
        assert_no_failures(errors)
        # No refusals at all, so every call landed: the length is exact rather
        # than merely consistent with a tally of survivors.
        assert len(m) == THREADS * ITERATIONS * per_call

    def test_converted_reads_never_refuse_under_writers(self):
        m = Mutibs.from_ones(BITS)
        needle = Tibs("0b11111111")

        def read():
            for _ in range(ITERATIONS):
                len(m)
                bool(m)
                assert "0" not in m.to_bin()
                assert m.count(0) == 0
                assert m.find("0b0") is None
                assert m.find(needle) == 0
                assert needle in m
                assert m.find_all("0b0") == []
                # The conversion family, method and property forms.
                assert "0" not in m.hex.replace("f", "")
                assert m.to_hex(0, 8) == "ff"
                assert m.bytes.count(0) == 0
                assert m.to_padded_bytes(0, 16) == b"\xff\xff"
                assert m.to_u(0, 8) == 255
                assert m.to_i(0, 8) == -1
                assert all(m.to_bools(0, 16))
                assert m.to_value("u8", 0, 8) == 255
                assert m.to_values("u8", 0, 16) == [255, 255]
                assert m._raw_data()[0].count(0) == 0

        def write():
            # Only converted methods, and only byte-aligned ones. An unconverted
            # writer such as `del m[-8:]` takes its exclusive borrow without
            # entering the critical section, and a reader inside the section
            # still loses to it - the guarantee holds only once every path goes
            # through the same gate. Byte alignment keeps `hex` and `bytes`
            # legal throughout; their length rules are the library's, not a
            # concurrency effect.
            for _ in range(ITERATIONS):
                if len(m) > BITS // 2:
                    m.write_bytes(b"\xff" * (BITS // 16))
                else:
                    m.extend("0xff")

        errors = run_concurrently(*(read for _ in range(6)), write, write)
        assert_no_failures(errors)
        assert m.count(0) == 0

    def test_two_operand_calls_never_refuse(self):
        # Every other serialisation test writes to the *receiver*, so a method
        # that locks `self` but reaches its operand unprotected passes them all.
        # Here the writer targets the operand instead.
        receiver = Mutibs.from_ones(1024)
        operand = Mutibs.from_ones(1024)

        def compare():
            for _ in range(ITERATIONS):
                assert receiver == operand
                assert operand == receiver
                assert Tibs.from_ones(1024) == operand
                assert (Tibs.from_ones(1024) & operand).all()

        def extend():
            for _ in range(ITERATIONS):
                target = Mutibs()
                target.extend(operand)
                assert len(target) == 1024

        def write_operand():
            for _ in range(ITERATIONS):
                operand.write_bytes(b"\xff" * 128)

        errors = run_concurrently(
            compare, compare, extend, extend, write_operand, write_operand
        )
        assert_no_failures(errors)

    def test_converted_writers_never_refuse(self):
        # Every writer installs the same all-ones value at a fixed length, so a
        # refusal, a torn write or a length change would all show up. `write_u`
        # in particular builds its value outside the lock and confirms the
        # length before installing it.
        m = Mutibs.from_ones(64)
        ones = (1 << 64) - 1

        def by_int():
            for _ in range(ITERATIONS):
                m.write_u(ones)
                m.u = ones

        def by_text():
            for _ in range(ITERATIONS):
                m.write_hex("ff" * 8)
                m.hex = "ff" * 8

        def by_bytes():
            for _ in range(ITERATIONS):
                m.write_bytes(b"\xff" * 8)
                m.bytes = b"\xff" * 8

        def check():
            for _ in range(ITERATIONS):
                assert m.count(0) == 0, "a writer installed a zero bit"
                assert len(m) == 64, "a length-preserving write changed the length"

        errors = run_concurrently(
            by_int, by_int, by_text, by_text, by_bytes, by_bytes, check, check
        )
        assert_no_failures(errors)
        assert len(m) == 64
        assert m.count(0) == 0

    def test_converted_inspection_and_mutation_never_refuse(self):
        # Every mutation here preserves "all ones at a fixed length", so a
        # refusal, a torn write or a stray zero would all show. `set` and
        # `invert` read their positions out of Python before locking, which is
        # the part that made them impossible to wrap.
        m = Mutibs.from_ones(256)
        indices = list(range(0, 256, 3))

        def by_positions():
            for _ in range(ITERATIONS):
                m.set(indices)
                m.set(range(0, 256, 5))
                m.set(7)
                m.set(tuple(indices[:8]))

        def by_whole():
            # No `invert()` here: a pair of them returns to all ones, but the
            # all-zeros state in between is a real state that the checkers are
            # entitled to observe. Per-call atomicity does not make a pair of
            # calls atomic - that is the contract, not a gap in it.
            for _ in range(ITERATIONS):
                m.reverse()
                m.rotate_left(3)
                m.byte_swap()

        def by_operator():
            # `nonlocal` because `m |= x` is an assignment to `m` as far as
            # Python scoping is concerned, even though it mutates in place.
            nonlocal m
            ones = Tibs.from_ones(256)
            for _ in range(ITERATIONS):
                m |= ones
                m &= ones

        def check():
            for _ in range(ITERATIONS):
                assert m.all(), "a converted mutation left a zero bit"
                assert m.any()
                assert len(m) == 256
                assert m.reversed().all()
                assert m.rotated_left(1).all()
                assert m.byte_swapped().all()
                assert m.with_set(3).all()
                assert not m.inverted().any()

        errors = run_concurrently(
            by_positions, by_positions, by_whole, by_operator, check, check
        )
        assert_no_failures(errors)
        assert m.all()
        assert len(m) == 256

    def test_no_public_attribute_refuses(self):
        # The whole class goes through the critical section now, so sweep it
        # rather than listing methods by hand: anything still taking its borrow
        # directly shows up here, including a method added later without a
        # wrapper. 24 bits so that oct, hex and bytes are all legal.
        ones = Tibs.from_ones(24)
        argument_sets = ((), (ones,), (0,), (0, ones), (ones, ones))
        names = [n for n in dir(Mutibs) if not n.startswith("_")]
        ROUNDS = 20

        refusals = collections.Counter()

        def exercise(m, name):
            """Touch one attribute, returning whether it did anything.

            Refusals are recorded with the attribute named rather than raised.
            They cannot be asserted away entirely: the interpreter may suspend a
            critical section - for a GC pause, or a signal check inside a long
            search - while the borrow is still held, and the thread that then
            enters the section finds the borrow taken. That is rare and costs a
            refusal, never correctness. A method that was never wrapped refuses
            on a large fraction of its calls instead, which is what the rate
            check below distinguishes.
            """
            try:
                attribute = getattr(m, name)
            except (TypeError, ValueError, IndexError):
                return False  # a property this receiver cannot produce
            except RuntimeError:
                refusals[name] += 1
                return True
            if not callable(attribute):
                return True
            for args in argument_sets:
                try:
                    attribute(*args)
                except (TypeError, ValueError, IndexError):
                    continue
                except RuntimeError:
                    refusals[name] += 1
                return True
            return False

        reachable = [n for n in names if exercise(Mutibs.from_ones(24), n)]
        # A floor, so a sweep that silently stops reaching anything still fails.
        assert len(reachable) > 40, f"only reached {len(reachable)} attributes"

        m = Mutibs.from_ones(24)

        def touch_all():
            for _ in range(ROUNDS):
                for name in reachable:
                    exercise(m, name)

        def write():
            # Counted, not raised: the writer can lose a call to a suspended
            # section just as a reader can.
            for _ in range(ROUNDS * len(reachable)):
                try:
                    m.write_bytes(b"\xff" * 3)
                except RuntimeError:
                    refusals["write_bytes"] += 1

        errors = run_concurrently(touch_all, touch_all, write, write)
        assert_no_failures(errors)
        if not FREE_THREADED:
            # `with_critical_section` compiles to a direct call on a GIL-enabled
            # build, so there is no serialisation to assert: two threads that
            # interleave at a GIL release point contend on the borrow flag
            # exactly as they did before any of this. Reaching every attribute
            # without a crash or a panic is all this proves here.
            return
        # Counted per attribute rather than in total, which is what separates
        # the two causes. An unwrapped method loses a large share of its own
        # calls - `field` refused over a thousand times in a two-second probe
        # before it was wrapped - while a suspended section costs one call of
        # whichever attribute happened to be running.
        per_attribute = 2 * ROUNDS
        worst = refusals.most_common(1)
        assert not worst or worst[0][1] <= per_attribute // 4, (
            f"{worst[0][0]} refused {worst[0][1]} of its {per_attribute} calls; "
            f"all refusals: {dict(refusals)}"
        )

    @pytest.mark.parametrize("subject", ["view", "reader", "equality"])
    def test_borrow_contention_raises_instead_of_panicking(self, subject):
        m = Mutibs.from_ones(BITS)
        targets = {
            "view": lambda: m.field(0, 7),
            "reader": lambda: Reader(m),
            "equality": lambda: (Tibs.from_ones(BITS), m.field(0, 7)),
        }
        target = targets[subject]()

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
                    elif subject == "reader":
                        target.pos = 0
                        assert target.read_value("u8") == 255
                    else:
                        # Tibs.__eq__, Mutibs.__eq__ and MutableView.__eq__ each
                        # borrow the Mutibs on the other side.
                        stable, view = target
                        # Whether this one is True depends on the racing length,
                        # so only the absence of a panic is asserted.
                        stable == m  # noqa: B015
                        # These two hold both borrows within the one call, so a
                        # resize cannot land between them.
                        assert m == m
                        assert view == view
                except (RuntimeError, *EXPECTED_RACE_ERRORS):
                    pass

        errors = run_concurrently(
            tolerant(resize),
            tolerant(resize),
            *(touch for _ in range(THREADS - 2)),
        )
        panics = [e for e in errors if type(e).__name__ == "PanicException"]
        assert not panics, f"Rust panicked on borrow contention: {panics[0]!r}"
