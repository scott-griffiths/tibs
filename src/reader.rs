use crate::ReadError;
use crate::core::BitCollection;
use crate::dtype::extract_dtype;
use crate::mutibs::Mutibs;
use crate::tibs_::{SearchParams, Tibs, find_in_bits, py_from_value, py_values_from_range};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::critical_section::{with_critical_section, with_critical_section2};
use std::sync::atomic::{AtomicUsize, Ordering};

/// The bit container a [`Reader`] reads from, kept as the Python object rather
/// than a snapshot of its bits.
///
/// Holding `Py<T>` rather than a cloned `Tibs` is what makes `Reader.source`
/// able to hand back the object the caller passed in, and what makes a
/// `Mutibs` source live: bits appended to it after the reader was built are
/// there to be read.
enum ReaderSource {
    Immutable(Py<Tibs>),
    Mutable(Py<Mutibs>),
}

/// How many bits are readable from `pos`, saturating at zero.
///
/// A `Mutibs` source can be truncated under a reader that is already past the
/// new end, so this subtraction genuinely can go negative and must not wrap.
#[inline]
fn remaining_bits(pos: usize, len: usize) -> usize {
    len.saturating_sub(pos)
}

/// Narrow a caller-supplied cursor position against a source length.
///
/// Negative positions are refused rather than counted from the end. `pos` is
/// stored state, not a one-off slice bound, so `-1` would have to mean either
/// "one before the end" or "one before the end *as it is now*", and either
/// reading leaves `remaining` disagreeing with what was just assigned.
fn validate_pos(pos: i64, len: usize) -> PyResult<usize> {
    if pos < 0 || pos as u64 > len as u64 {
        return Err(PyValueError::new_err(format!(
            "Position of {pos} is out of range for a source of {len} bits."
        )));
    }
    Ok(pos as usize)
}

/// Check that `needed` bits are readable at `pos`, returning the end position.
///
/// This is the single place a short read is turned into an error, so every
/// read reports the same way and none of them can advance `pos` first: the
/// caller has to hold the returned end position to move the cursor at all.
fn end_of_read(pos: usize, needed: usize, len: usize) -> PyResult<usize> {
    match pos.checked_add(needed) {
        Some(end) if end <= len => Ok(end),
        _ => Err(ReadError::new_err(format!(
            "Cannot read {needed} bits at position {pos}: only {} of the {len} bits are left.",
            remaining_bits(pos, len)
        ))),
    }
}

///     A cursor for reading a :class:`Tibs` or :class:`Mutibs` in sequence.
///
///     A ``Reader`` pairs a bit container with a bit position, so that values
///     can be read one after another without working out ``start`` and ``end``
///     for each one. Every method is anchored at :attr:`~Reader.pos`, and the
///     reading and seeking methods move it. For a windowed query, use the
///     source object directly through :attr:`~Reader.source`.
///
///     The source is not copied, so a ``Mutibs`` that grows after the reader
///     was built can be read up to its new length.
///
///     .. code-block:: pycon
///
///         >>> r = Reader(Tibs('0x47ff10'))
///         >>> r.read_value('u8')
///         71
///         >>> r.read_value('(bool, u7)')
///         (True, 127)
///         >>> r.remaining
///         8
///
// `frozen` so that the reader has no borrow flag of its own. Reaching `source`
// otherwise needed a borrow taken outside any critical section, which is the
// very thing being eliminated: a thread merely asking which source to lock
// would be refused by a read already in flight.
#[pyclass(module = "tibs", frozen)]
pub struct Reader {
    source: ReaderSource,
    /// Atomic only to satisfy `Sync` for a frozen pyclass. Every access is
    /// already inside the reader's critical section, which is what makes a
    /// whole method atomic, so `Relaxed` is all the ordering needed.
    pos: AtomicUsize,
}

impl Reader {
    /// The current length of the source, which a `Mutibs` can change.
    ///
    /// The two arms differ deliberately, and every other source access below
    /// follows the same split. `Tibs` is a frozen pyclass, so it has no borrow
    /// flag: `get` reaches it with no check and no `Result`. `Mutibs` is not
    /// frozen, so another thread can hold it on a free-threaded build, and
    /// `try_borrow` is what turns that into a Python exception. `borrow` would
    /// panic instead, and a panic arrives as `pyo3_runtime.PanicException`,
    /// which does not inherit from `Exception` and so escapes a caller's
    /// `except`.
    fn source_len(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(match &self.source {
            ReaderSource::Immutable(tibs) => tibs.get().len(),
            ReaderSource::Mutable(mutibs) => mutibs.try_borrow(py)?.len(),
        })
    }

    /// The `length` bits of the source starting at `start`, as a `Tibs`.
    ///
    /// A `Tibs` source shares its storage, so this is O(1) there, and a copy
    /// of just the window for a `Mutibs`.
    ///
    /// `start + length` must be within the source length.
    fn window(&self, py: Python<'_>, start: usize, length: usize) -> PyResult<Tibs> {
        Ok(match &self.source {
            ReaderSource::Immutable(tibs) => tibs.get().get_slice_unchecked(start, length),
            ReaderSource::Mutable(mutibs) => mutibs.try_borrow(py)?.window(start, length),
        })
    }

    /// Search the source, shared by the seeking and reading-to methods.
    fn search(
        &self,
        py: Python<'_>,
        needle: Tibs,
        params: SearchParams,
        reverse: bool,
    ) -> PyResult<Option<usize>> {
        match &self.source {
            ReaderSource::Immutable(tibs) => {
                find_in_bits(py, tibs.get().as_bitslice(), &needle, params, reverse)
            }
            ReaderSource::Mutable(mutibs) => {
                let source = mutibs.try_borrow(py)?;
                find_in_bits(py, source.as_bitslice(), &needle, params, reverse)
            }
        }
    }

    /// Find `needle` at or after `pos`, for `read_to`, `read_past`, `seek_to`
    /// and `seek_past`.
    fn find_from_pos(
        &self,
        py: Python<'_>,
        needle: Tibs,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Option<usize>> {
        // A `pos` beyond a shrunken source would fail `validate_slice` inside
        // the search with a message about slice positions, which says nothing
        // about the cursor. Nothing can be found ahead of the end anyway.
        if self.position() > self.source_len(py)? {
            return Ok(None);
        }
        self.search(
            py,
            needle,
            SearchParams {
                start: Some(self.position() as isize),
                end: None,
                byte_aligned,
                mask,
            },
            false,
        )
    }

    /// Validate a new cursor position against the current source length.
    fn checked_pos(&self, py: Python<'_>, pos: i64) -> PyResult<usize> {
        validate_pos(pos, self.source_len(py)?)
    }

    /// The cursor position. See the note on the field.
    #[inline]
    fn position(&self) -> usize {
        self.pos.load(Ordering::Relaxed)
    }

    #[inline]
    fn set_position(&self, pos: usize) {
        self.pos.store(pos, Ordering::Relaxed);
    }

    /// Run `f` with the reader locked, and its source locked too when that is a
    /// `Mutibs`.
    ///
    /// A `Tibs` source is frozen, so nothing can be mutating it and the
    /// reader's own section is enough. A `Mutibs` source can be written
    /// directly by another thread, so its section has to be held for as long as
    /// the reader reads through it - and entered together with the reader's,
    /// since taking the second on its own would suspend the first.
    fn with_reader<R>(slf: &Bound<'_, Self>, f: impl FnOnce(&Self) -> PyResult<R>) -> PyResult<R> {
        // `get` rather than a borrow: the class is frozen, so this is a plain
        // pointer read and cannot be refused.
        let reader = slf.get();
        match &reader.source {
            ReaderSource::Mutable(mutibs) => {
                with_critical_section2(slf.as_any(), mutibs.bind(slf.py()).as_any(), || f(reader))
            }
            ReaderSource::Immutable(_) => with_critical_section(slf.as_any(), || f(reader)),
        }
    }
}

#[pymethods]
impl Reader {
    /// Create a reader over a bit container.
    ///
    /// A :class:`Tibs` or :class:`Mutibs` ``source`` is kept as it is rather
    /// than copied, so :attr:`~Reader.source` gives back the same object.
    /// Anything else promotable to a ``Tibs``, such as a string or a
    /// ``bytes``, is converted once and the reader holds the result.
    ///
    /// :param object source: The bits to read. A :class:`Tibs` or :class:`Mutibs` is used directly; anything else promotable to ``Tibs`` is converted.
    /// :param int pos: The initial bit position. Defaults to 0.
    /// :raises ValueError: if ``pos`` is negative or beyond the end of ``source``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Reader(Tibs('0x0123'), 8).read_value('u8')
    ///     35
    ///
    #[new]
    #[pyo3(signature = (source, /, pos = 0), text_signature = "(source, /, pos=0)")]
    pub fn py_new(source: &Bound<'_, PyAny>, pos: i64) -> PyResult<Self> {
        let (source, len) = if let Ok(tibs) = source.extract::<Py<Tibs>>() {
            let len = tibs.get().len();
            (ReaderSource::Immutable(tibs), len)
        } else if let Ok(mutibs) = source.extract::<Py<Mutibs>>() {
            let len = mutibs.try_borrow(source.py())?.len();
            (ReaderSource::Mutable(mutibs), len)
        } else {
            let tibs = source.extract::<Tibs>()?;
            let len = tibs.len();
            (ReaderSource::Immutable(Py::new(source.py(), tibs)?), len)
        };
        Ok(Reader {
            source,
            pos: AtomicUsize::new(validate_pos(pos, len)?),
        })
    }

    /// The object being read.
    ///
    /// This is the :class:`Tibs` or :class:`Mutibs` that was given to the
    /// constructor, not a copy, so it can be used for the windowed queries a
    /// ``Reader`` deliberately does not have: ``r.source.to_value(dtype,
    /// start, end)`` reads without moving :attr:`~Reader.pos`.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> t = Tibs('0xabcd')
    ///     >>> Reader(t).source is t
    ///     True
    ///
    #[getter]
    // No section: the field is fixed at construction, so a momentary borrow is
    // all this needs and there is nothing for another thread to change.
    pub fn source(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match &slf.try_borrow()?.source {
            ReaderSource::Immutable(tibs) => tibs.clone_ref(py).into_any(),
            ReaderSource::Mutable(mutibs) => mutibs.clone_ref(py).into_any(),
        })
    }

    /// The current bit position.
    ///
    /// Setting it moves the cursor anywhere in ``0`` to ``len(self)``
    /// inclusive. Negative positions are not accepted: unlike a ``start`` or
    /// ``end`` elsewhere in tibs, this is stored state rather than a one-off
    /// slice bound, so counting from the end would leave
    /// :attr:`~Reader.remaining` disagreeing with the value just assigned.
    ///
    /// :raises ValueError: if set outside ``0`` to ``len(self)``.
    ///
    #[getter]
    pub fn pos(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Self::with_reader(slf, |r| Ok(r.position()))
    }

    #[setter(pos)]
    pub fn set_pos(slf: &Bound<'_, Self>, py: Python<'_>, pos: i64) -> PyResult<()> {
        Self::with_reader(slf, |r| {
            r.set_position(r.checked_pos(py, pos)?);
            Ok(())
        })
    }

    /// The current position in bytes.
    ///
    /// :raises ValueError: when read, if :attr:`~Reader.pos` is not a multiple of 8; when set, if the position is out of range.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0x0123'))
    ///     >>> r.byte_pos = 1
    ///     >>> r.pos
    ///     8
    ///
    #[getter]
    pub fn byte_pos(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Self::with_reader(slf, |r| {
            if !r.position().is_multiple_of(8) {
                return Err(PyValueError::new_err(format!(
                    "Position of {} bits is not byte aligned, so it has no byte position.",
                    r.position()
                )));
            }
            Ok(r.position() / 8)
        })
    }

    #[setter(byte_pos)]
    pub fn set_byte_pos(slf: &Bound<'_, Self>, py: Python<'_>, byte_pos: i64) -> PyResult<()> {
        let pos = byte_pos.checked_mul(8).ok_or_else(|| {
            PyValueError::new_err(format!("Byte position of {byte_pos} is too large."))
        })?;
        Self::with_reader(slf, |r| {
            r.set_position(r.checked_pos(py, pos)?);
            Ok(())
        })
    }

    /// The number of bits between :attr:`~Reader.pos` and the end.
    #[getter]
    pub fn remaining(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<usize> {
        Self::with_reader(slf, |r| Ok(remaining_bits(r.position(), r.source_len(py)?)))
    }

    /// Whether there is nothing left to read.
    #[getter]
    pub fn at_end(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<bool> {
        Self::with_reader(slf, |r| Ok(r.position() >= r.source_len(py)?))
    }

    /// Read one value with a dtype, advancing by the dtype length.
    ///
    /// The value is a scalar for a :class:`DtypeSingle` and a tuple for a
    /// :class:`DtypeArray` or :class:`DtypeTuple`, exactly as
    /// :meth:`Tibs.to_value` gives.
    ///
    /// :param Dtype | str dtype: The value encoding to use.
    /// :return: The decoded Python value.
    /// :raises tibs.ReadError: if fewer than ``dtype.length`` bits remain, in which case the position does not move.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0x47ff'))
    ///     >>> r.read_value('u8')
    ///     71
    ///     >>> r.read_value('(bool, u7)')
    ///     (True, 127)
    ///
    #[pyo3(signature = (dtype, /), text_signature = "($self, dtype, /)")]
    pub fn read_value(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        // `extract_dtype` parses a spec string and `py_from_value` builds the
        // result object; both are Python, so both stay outside the section.
        let dtype = extract_dtype(dtype)?;
        let window = Self::with_reader(slf, |r| {
            let end = end_of_read(r.position(), dtype.length, r.source_len(py)?)?;
            let window = r.window(py, r.position(), dtype.length)?;
            r.set_position(end);
            Ok(window)
        })?;
        py_from_value(py, &dtype, &window)
    }

    /// Read a list of values of one dtype, advancing past all of them.
    ///
    /// With ``count`` given, exactly that many values are read and it is an
    /// error if they do not all fit. With ``count`` left as ``None``, as many
    /// whole values as fit are read; any partial value at the end is left
    /// under the cursor rather than being an error, so
    /// :attr:`~Reader.remaining` afterwards says how many bits were not
    /// consumed. This is the one place a ``Reader`` is more forgiving than
    /// :meth:`Tibs.to_values`, which has no cursor to leave a remainder
    /// behind.
    ///
    /// :param Dtype | str dtype: The value encoding to use for each item.
    /// :param int | None count: The number of values to read, or ``None`` for as many as fit.
    /// :return: A list of decoded Python values.
    /// :raises ValueError: if ``count`` is negative or too large to represent.
    /// :raises tibs.ReadError: if the requested values do not fit in the bits that remain, in which case the position does not move.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0x0102030405'))
    ///     >>> r.read_values('u8', 2)
    ///     [1, 2]
    ///     >>> r.read_values('u8')
    ///     [3, 4, 5]
    ///
    #[pyo3(signature = (dtype, /, count = None), text_signature = "($self, dtype, /, count=None)")]
    pub fn read_values(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        count: Option<i64>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let dtype = extract_dtype(dtype)?;
        let window = Self::with_reader(slf, |r| {
            let len = r.source_len(py)?;
            let bits = match count {
            None => remaining_bits(r.position(), len) / dtype.length * dtype.length,
            Some(count) if count < 0 => {
                return Err(PyValueError::new_err(format!(
                    "Cannot read a negative number of values ({count})."
                )));
            }
            Some(count) => (count as usize).checked_mul(dtype.length).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Cannot read {count} values of {} bits: the total length is too large to represent.",
                    dtype.length
                ))
            })?,
        };
            let end = end_of_read(r.position(), bits, len)?;
            let window = r.window(py, r.position(), bits)?;
            r.set_position(end);
            Ok(window)
        })?;
        py_values_from_range(py, &window, &dtype, None, None)
    }

    /// Read the next ``n`` bits, advancing by ``n``.
    ///
    /// The result is always a :class:`Tibs`, including when reading from a
    /// :class:`Mutibs`: a read takes a value out of the stream, and there is
    /// nothing to write back through.
    ///
    /// :param int n: The number of bits to read.
    /// :return: The bits read.
    /// :raises ValueError: if ``n`` is negative.
    /// :raises tibs.ReadError: if ``n`` is more than :attr:`~Reader.remaining`, in which case the position does not move.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0xf00f'))
    ///     >>> r.read_bits(4)
    ///     Tibs('0xf')
    ///
    #[pyo3(signature = (n, /), text_signature = "($self, n, /)")]
    pub fn read_bits(slf: &Bound<'_, Self>, py: Python<'_>, n: i64) -> PyResult<Tibs> {
        let n = validate_read_length(n)?;
        Self::with_reader(slf, |r| {
            let end = end_of_read(r.position(), n, r.source_len(py)?)?;
            let bits = r.window(py, r.position(), n)?;
            r.set_position(end);
            Ok(bits)
        })
    }

    /// Read up to the next occurrence of ``needle``, leaving the cursor on it.
    ///
    /// The returned bits stop where ``needle`` begins, and the cursor is left
    /// there too, so the needle itself is read by whatever comes next. Use
    /// :meth:`~read_past` to consume it as part of the read.
    ///
    /// :param object needle: The bit sequence to read up to. This can be anything promotable to ``Tibs``.
    /// :param bool byte_aligned: If ``True``, only match on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: The bits between the current position and the match.
    /// :raises ValueError: if ``needle`` is empty.
    /// :raises tibs.ReadError: if ``needle`` is not found at or after the current position, in which case the position does not move.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0x0000ff12'))
    ///     >>> r.read_to('0xff')
    ///     Tibs('0x0000')
    ///     >>> r.pos
    ///     16
    ///
    #[pyo3(signature = (needle, /, byte_aligned = false, mask = None), text_signature = "($self, needle, /, byte_aligned=False, mask=None)")]
    pub fn read_to(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        needle: Tibs,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Tibs> {
        Self::with_reader(slf, |r| {
            let found = r.require_found(r.find_from_pos(py, needle, byte_aligned, mask)?)?;
            let bits = r.window(py, r.position(), found - r.position())?;
            r.set_position(found);
            Ok(bits)
        })
    }

    /// Read up to and including the next occurrence of ``needle``.
    ///
    /// The needle is part of the returned bits and the cursor is left just
    /// after it, so a loop of ``read_past`` calls always makes progress.
    ///
    /// :param object needle: The bit sequence to read past. This can be anything promotable to ``Tibs``.
    /// :param bool byte_aligned: If ``True``, only match on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: The bits between the current position and the end of the match.
    /// :raises ValueError: if ``needle`` is empty.
    /// :raises tibs.ReadError: if ``needle`` is not found at or after the current position, in which case the position does not move.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0x0000ff12'))
    ///     >>> r.read_past('0xff')
    ///     Tibs('0x0000ff')
    ///     >>> r.pos
    ///     24
    ///
    #[pyo3(signature = (needle, /, byte_aligned = false, mask = None), text_signature = "($self, needle, /, byte_aligned=False, mask=None)")]
    pub fn read_past(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        needle: Tibs,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<Tibs> {
        Self::with_reader(slf, |r| {
            let needle_len = needle.len();
            let found = r.require_found(r.find_from_pos(py, needle, byte_aligned, mask)?)?;
            let end = found + needle_len;
            let bits = r.window(py, r.position(), end - r.position())?;
            r.set_position(end);
            Ok(bits)
        })
    }

    /// Read one value with a dtype without moving the cursor.
    ///
    /// :param Dtype | str dtype: The value encoding to use.
    /// :return: The decoded Python value.
    /// :raises tibs.ReadError: if fewer than ``dtype.length`` bits remain.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0x47ff'))
    ///     >>> r.peek_value('u8')
    ///     71
    ///     >>> r.pos
    ///     0
    ///
    #[pyo3(signature = (dtype, /), text_signature = "($self, dtype, /)")]
    pub fn peek_value(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let dtype = extract_dtype(dtype)?;
        let window = Self::with_reader(slf, |r| {
            end_of_read(r.position(), dtype.length, r.source_len(py)?)?;
            r.window(py, r.position(), dtype.length)
        })?;
        py_from_value(py, &dtype, &window)
    }

    /// Read the next ``n`` bits without moving the cursor.
    ///
    /// :param int n: The number of bits to read.
    /// :return: The bits read.
    /// :raises ValueError: if ``n`` is negative.
    /// :raises tibs.ReadError: if ``n`` is more than :attr:`~Reader.remaining`.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0xf00f'))
    ///     >>> r.peek_bits(4)
    ///     Tibs('0xf')
    ///     >>> r.pos
    ///     0
    ///
    #[pyo3(signature = (n, /), text_signature = "($self, n, /)")]
    pub fn peek_bits(slf: &Bound<'_, Self>, py: Python<'_>, n: i64) -> PyResult<Tibs> {
        let n = validate_read_length(n)?;
        Self::with_reader(slf, |r| {
            end_of_read(r.position(), n, r.source_len(py)?)?;
            r.window(py, r.position(), n)
        })
    }

    /// Return a context manager that restores the position on exit.
    ///
    /// The ``with`` block gets this same ``Reader``, so anything can be read
    /// inside it, and the position it had on entry is put back afterwards
    /// whether the block finished or raised. This is how a ``Reader`` looks
    /// ahead by more than one value; :meth:`~peek_value` and
    /// :meth:`~peek_bits` are the single-item shorthands.
    ///
    /// :return: A context manager yielding this ``Reader``.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0x010203'))
    ///     >>> with r.bookmark():
    ///     ...     r.read_values('u8')
    ///     [1, 2, 3]
    ///     >>> r.pos
    ///     0
    ///
    pub fn bookmark(slf: &Bound<'_, Self>) -> PyResult<Bookmark> {
        let pos = Self::with_reader(slf, |r| Ok(r.position()))?;
        Ok(Bookmark {
            pos,
            reader: slf.clone().unbind(),
        })
    }

    /// Move forward to the next multiple of ``boundary`` bits.
    ///
    /// Nothing moves if the position is already on a boundary. This covers
    /// byte alignment as ``align()`` and generalises to 16-bit or 32-bit
    /// boundaries for free.
    ///
    /// :param int boundary: The alignment in bits. Defaults to 8.
    /// :return: The number of bits skipped.
    /// :raises ValueError: if ``boundary`` is not positive, or if aligning would move past the end, in which case the position does not move.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0xffff'), 3)
    ///     >>> r.align()
    ///     5
    ///     >>> r.pos
    ///     8
    ///
    #[pyo3(signature = (boundary = 8, /), text_signature = "($self, boundary=8, /)")]
    pub fn align(slf: &Bound<'_, Self>, py: Python<'_>, boundary: i64) -> PyResult<usize> {
        if boundary <= 0 {
            return Err(PyValueError::new_err(format!(
                "Alignment boundary must be greater than zero, but received {boundary}."
            )));
        }
        let boundary = boundary as u64;
        Self::with_reader(slf, |r| {
            let skip = ((boundary - r.position() as u64 % boundary) % boundary) as usize;
            let len = r.source_len(py)?;
            if r.position() + skip > len {
                return Err(PyValueError::new_err(format!(
                    "Cannot align to {boundary} bits from position {}: it would move past the end of the {len} bits.",
                    r.position()
                )));
            }
            r.set_position(r.position() + skip);
            Ok(skip)
        })
    }

    /// Move to the next occurrence of ``needle``, leaving the cursor on it.
    ///
    /// The search starts at the current position, so a needle already under
    /// the cursor is found where it is and nothing moves. That makes ``while
    /// r.seek_to(x)`` an infinite loop; :meth:`~seek_past` is the one to loop
    /// on.
    ///
    /// The position of the match is :attr:`~Reader.pos` afterwards. This
    /// returns a ``bool`` rather than the position so that a match at bit 0
    /// is not falsy.
    ///
    /// :param object needle: The bit sequence to seek to. This can be anything promotable to ``Tibs``.
    /// :param bool byte_aligned: If ``True``, only match on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: ``True`` if it was found, otherwise ``False`` with the position unchanged.
    /// :raises ValueError: if ``needle`` is empty.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0x0000ff12'))
    ///     >>> r.seek_to('0xff')
    ///     True
    ///     >>> r.pos
    ///     16
    ///
    #[pyo3(signature = (needle, /, byte_aligned = false, mask = None), text_signature = "($self, needle, /, byte_aligned=False, mask=None)")]
    pub fn seek_to(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        needle: Tibs,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<bool> {
        Self::with_reader(slf, |r| {
            match r.find_from_pos(py, needle, byte_aligned, mask)? {
                Some(found) => {
                    r.set_position(found);
                    Ok(true)
                }
                None => Ok(false),
            }
        })
    }

    /// Move to just after the next occurrence of ``needle``.
    ///
    /// The cursor always ends up further forward than it started, so this is
    /// the one to drive a scanning loop with::
    ///
    ///     while reader.seek_past(marker):
    ///         handle(reader.read_value('u16'))
    ///
    /// :param object needle: The bit sequence to seek past. This can be anything promotable to ``Tibs``.
    /// :param bool byte_aligned: If ``True``, only match on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: ``True`` if it was found, otherwise ``False`` with the position unchanged.
    /// :raises ValueError: if ``needle`` is empty.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0x0000ff12'))
    ///     >>> r.seek_past('0xff')
    ///     True
    ///     >>> r.pos
    ///     24
    ///
    #[pyo3(signature = (needle, /, byte_aligned = false, mask = None), text_signature = "($self, needle, /, byte_aligned=False, mask=None)")]
    pub fn seek_past(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        needle: Tibs,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<bool> {
        let needle_len = needle.len();
        Self::with_reader(slf, |r| {
            match r.find_from_pos(py, needle, byte_aligned, mask)? {
                Some(found) => {
                    r.set_position(found + needle_len);
                    Ok(true)
                }
                None => Ok(false),
            }
        })
    }

    /// Move back to the previous occurrence of ``needle``.
    ///
    /// Only matches that end at or before the current position are
    /// considered, so the cursor always ends up further back than it started
    /// and ``while r.seek_back_to(x)`` makes progress.
    ///
    /// :param object needle: The bit sequence to seek back to. This can be anything promotable to ``Tibs``.
    /// :param bool byte_aligned: If ``True``, only match on byte boundaries.
    /// :param object | None mask: If present, only the bits set in the mask need to match. Defaults to ``None``.
    /// :return: ``True`` if it was found, otherwise ``False`` with the position unchanged.
    /// :raises ValueError: if ``needle`` is empty.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> r = Reader(Tibs('0x00ff00ff'), 32)
    ///     >>> r.seek_back_to('0xff')
    ///     True
    ///     >>> r.pos
    ///     24
    ///
    #[pyo3(signature = (needle, /, byte_aligned = false, mask = None), text_signature = "($self, needle, /, byte_aligned=False, mask=None)")]
    pub fn seek_back_to(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        needle: Tibs,
        byte_aligned: bool,
        mask: Option<Tibs>,
    ) -> PyResult<bool> {
        Self::with_reader(slf, |r| {
            // Clamped rather than passed through, so that a `pos` left beyond a
            // truncated `Mutibs` searches what is there instead of raising.
            let end = r.position().min(r.source_len(py)?) as isize;
            let params = SearchParams {
                start: None,
                end: Some(end),
                byte_aligned,
                mask,
            };
            match r.search(py, needle, params, true)? {
                Some(found) => {
                    r.set_position(found);
                    Ok(true)
                }
                None => Ok(false),
            }
        })
    }

    /// Return the length of the source in bits.
    pub fn __len__(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<usize> {
        Self::with_reader(slf, |r| r.source_len(py))
    }

    pub fn __repr__(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<String> {
        Self::with_reader(slf, |r| {
            let source = match &r.source {
                ReaderSource::Immutable(tibs) => tibs.get().__repr__(),
                ReaderSource::Mutable(mutibs) => mutibs.try_borrow(py)?.repr_string(),
            };
            Ok(format!("Reader({source}, {})", r.position()))
        })
    }
}

impl Reader {
    /// Turn "not found" into the error the reading methods raise.
    ///
    /// The seeks report a miss as `False` because looking for something that
    /// is not there is a normal outcome, but `read_to` and `read_past` are
    /// named for the `Tibs` they return and have nothing to return here.
    fn require_found(&self, found: Option<usize>) -> PyResult<usize> {
        found.ok_or_else(|| {
            ReadError::new_err(format!(
                "Cannot read: the bits to find were not found at or after position {}. Use seek_to or seek_past if not finding them is expected.",
                self.position()
            ))
        })
    }
}

/// Check a bit count for `read_bits` and `peek_bits`.
fn validate_read_length(n: i64) -> PyResult<usize> {
    if n < 0 {
        return Err(PyValueError::new_err(format!(
            "Cannot read a negative number of bits ({n})."
        )));
    }
    Ok(n as usize)
}

///     The context manager returned by :meth:`Reader.bookmark`.
///
///     Entering gives back the ``Reader`` it came from, and leaving restores
///     the position that ``Reader`` had when the bookmark was made.
///
#[pyclass(module = "tibs")]
pub struct Bookmark {
    reader: Py<Reader>,
    pos: usize,
}

#[pymethods]
impl Bookmark {
    fn __enter__(&self, py: Python<'_>) -> Py<Reader> {
        self.reader.clone_ref(py)
    }

    /// Restore the saved position and let any exception propagate.
    #[pyo3(signature = (exc_type, exc_value, traceback))]
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: &Bound<'_, PyAny>,
        exc_value: &Bound<'_, PyAny>,
        traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let _ = (exc_type, exc_value, traceback);
        // The reader's own section, so that restoring the bookmark queues
        // behind any read in flight rather than being refused by it.
        Reader::with_reader(self.reader.bind(py), |reader| {
            reader.set_position(self.pos);
            Ok(())
        })?;
        Ok(false)
    }
}
