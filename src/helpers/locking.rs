//! Serialising access to a non-frozen `#[pyclass]` across threads.
//!
//! PyO3 gives a non-frozen `#[pyclass]` an atomic borrow flag, not a lock. Two
//! threads that reach the same object at once do not queue: the loser is
//! refused with `RuntimeError: Already borrowed`. That is safe but useless -
//! `m.append(x)` in a worker thread would need wrapping in `try`, which no
//! Python container asks of its callers.
//!
//! CPython's per-object critical section is the missing lock. Entering one
//! takes the object's `PyMutex`, so the second thread *blocks* and then finds
//! the borrow flag free. `with_critical_section` is a no-op on GIL-enabled
//! builds, compiling to a direct call, so the abi3 wheel is unaffected.
//!
//! # The rule these helpers cannot enforce
//!
//! **No Python may run inside the closure.** A critical section is suspended by
//! any call back into the interpreter, at which point another thread can enter
//! and mutate. Nothing is corrupted - the borrow flag still holds - but the
//! call may be refused, and the object may change under a suspended thread.
//!
//! So convert every Python argument to owned Rust data *before* calling these:
//! consume iterables, resolve `__index__`, extract buffers. Dropping a
//! `Py<PyAny>` inside the closure counts too, since a decref can run `__del__`.
//!
//! One exception is accepted deliberately. The search helpers call
//! `Python::check_signals` every `SIGNAL_CHECK_INTERVAL` bits so that Ctrl-C
//! interrupts a long scan, and a pending Python signal handler is Python. A
//! search over more than that many bits can therefore have its section
//! suspended, and another thread may then be refused. Refusal, not corruption:
//! the borrow is still held throughout. Responsiveness to Ctrl-C is worth more
//! than removing a refusal that only arises on multi-kilobit searches.
//!
//! # What this does and does not promise
//!
//! One call becomes atomic. A *sequence* of calls does not, exactly as for
//! `list`: `if len(m): m.pop()` can still race, because the two calls are the
//! caller's transaction and only the caller knows where it begins. That is the
//! same contract every Python container offers, and the same one the GIL gave -
//! it never made check-then-act atomic either.

use pyo3::PyClass;
use pyo3::prelude::*;
use pyo3::pyclass::boolean_struct::False;
use pyo3::sync::critical_section::{with_critical_section, with_critical_section2};

/// Run `f` with shared access to `slf`, serialised against other threads.
///
/// Use for every read reached from Python. A shared borrow does not conflict
/// with another shared borrow, but it does conflict with a writer, so a read
/// left unwrapped is refused whenever a write is in flight.
#[inline]
pub(crate) fn with_locked<T, R>(
    slf: &Bound<'_, T>,
    f: impl FnOnce(&T) -> PyResult<R>,
) -> PyResult<R>
where
    T: PyClass,
{
    // Entered unconditionally. Trying the borrow first and falling back to the
    // section only on failure looks like a free optimisation - the borrow flag
    // is atomic, so a borrow that succeeds is already safe - but it defeats the
    // purpose: a thread waiting inside the section still loses to one that
    // skipped it, and measured refusals went from 0% back to 6%. Mutual
    // exclusion only holds if every caller goes through the same gate. This is
    // what CPython's own containers do.
    with_critical_section(slf.as_any(), || f(&*slf.try_borrow()?))
}

/// Run `f` with exclusive access to `slf`, serialised against other threads.
///
/// Use for every mutation reached from Python.
#[inline]
pub(crate) fn with_locked_mut<T, R>(
    slf: &Bound<'_, T>,
    f: impl FnOnce(&mut T) -> PyResult<R>,
) -> PyResult<R>
where
    T: PyClass<Frozen = False>,
{
    with_critical_section(slf.as_any(), || f(&mut *slf.try_borrow_mut()?))
}

/// Run `f` with shared access to two objects at once, serialised against other
/// threads.
///
/// For the comparisons and set operations that read a second container: taking
/// the two sections one after another would leave the first suspended while the
/// second was entered, so the pair has to be acquired together. Two is the
/// limit the C API offers; a third operand has to be snapshotted instead.
#[inline]
pub(crate) fn with_locked2<T, U, R>(
    a: &Bound<'_, T>,
    b: &Bound<'_, U>,
    f: impl FnOnce(&T, &U) -> PyResult<R>,
) -> PyResult<R>
where
    T: PyClass,
    U: PyClass,
{
    with_critical_section2(a.as_any(), b.as_any(), || {
        // Both borrows are shared, so `a` and `b` being the same object is fine.
        f(&*a.try_borrow()?, &*b.try_borrow()?)
    })
}

/// Run `f` with `a` locked for writing and `b` for reading, both together.
///
/// For a mutation that reads a second container in place rather than copying
/// it, such as `Mutibs.extend(other_mutibs)`. Locking only the receiver would
/// leave a thread writing to `b` refused by the borrow held on it here, and
/// locking them in turn would suspend the first section.
///
/// `a` and `b` must be different objects: one exclusive and one shared borrow
/// of the same object conflict. Callers reaching this with a possible self
/// operand have to special-case that first, as `extend` does.
#[inline]
pub(crate) fn with_locked_mut2<T, U, R>(
    a: &Bound<'_, T>,
    b: &Bound<'_, U>,
    f: impl FnOnce(&mut T, &U) -> PyResult<R>,
) -> PyResult<R>
where
    T: PyClass<Frozen = False>,
    U: PyClass,
{
    debug_assert!(!std::ptr::eq(a.as_ptr(), b.as_ptr()));
    with_critical_section2(a.as_any(), b.as_any(), || {
        f(&mut *a.try_borrow_mut()?, &*b.try_borrow()?)
    })
}
