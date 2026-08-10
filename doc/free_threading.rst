.. currentmodule:: tibs

.. _free-threading:

Free-threaded Python
--------------------

CPython 3.13 added a build of the interpreter without the global interpreter
lock, and 3.14 made it an officially supported one. In these *free-threaded*
builds — ``python3.14t``, and the ``cp314t`` wheels — Python threads run at the
same time on different cores instead of taking turns, so a threaded program is
limited by its own locking rather than by the interpreter.

Tibs supports that build. It declares that it does not need the GIL, so
importing tibs doesn't silently switch the GIL back on for the whole process,
and pre-built wheels are published for it.

Nothing here asks you to write different code for the two builds: the guarantees
below hold on both.


The two rules
^^^^^^^^^^^^^

Everything in this appendix follows from two sentences.

**One call is atomic.** Any single method call on a shared object either
happens completely or not at all, and never sees another thread's
half-finished work. Two threads calling ``m.append(True)`` at the same moment
both get their bit; a thread calling ``m.count(1)`` while another is resizing
``m`` sees the container either before or after the resize, never during it.

**A sequence of calls is not.** ::

    if len(m) > 0:
        m.pop()          # another thread may have emptied it in between

That is a race, and it is a race for exactly the same reason it is for a
``list``. The two calls are your transaction, not the library's, and only you
know where it begins and ends. The GIL never made check-then-act atomic
either — this is not something free-threading takes away.

Guard a multi-call sequence with your own lock:

.. code-block:: python

    import threading

    lock = threading.Lock()

    with lock:
        if len(m) > 0:
            m.pop()

Watch for read-modify-write hidden in an expression. ``m.u = m.u + 1`` is two
calls and needs a lock; ``m |= mask`` and ``m[0:8] = b'\xff'`` are each one
call and don't.


What can be shared
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Type
     - Sharing between threads
   * - :class:`Tibs`, :class:`Dtype`, :class:`View`
     - Free. These are immutable, so no lock is taken at all and reads on the
       same object run genuinely in parallel.
   * - :class:`Mutibs`
     - Safe. Calls on the same object are serialised against each other; calls
       on different objects are not.
   * - :class:`MutableView`
     - Safe. The view itself is immutable — it only ever writes to its source —
       so its calls serialise on that source.
   * - :class:`Reader`
     - Safe. A reader has a cursor, so calls on the same reader serialise
       against each other even when reading a :class:`Tibs`.

An immutable object being free to share is the useful half of that table. A
:class:`Tibs` slice shares storage with the original rather than copying it,
so splitting one large container into per-thread pieces is cheap:

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor
    from tibs import Tibs

    data = Tibs.from_random(1_000_000_000)
    chunk = len(data) // 8
    pieces = [data[i:i + chunk] for i in range(0, len(data), chunk)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        total = sum(pool.map(lambda p: p.count(1), pieces))

On a free-threaded build those eight counts really do proceed at once. On a
GIL-enabled build they don't: tibs holds the GIL for the duration of a call,
so threads there give you concurrency of structure but no speed-up. Splitting
work across threads for performance is a reason to use ``3.14t``, and only
``3.14t``.

How much it repays depends on what the work is bound by. Counting bits is
limited by how fast memory can be read, so it is already quick and dividing it
up gains under 3x on a four-core machine. Decoding typed values spends its time
building Python objects instead, which is the part the free-threaded
interpreter really does in parallel, and gains rather more. See
:doc:`example_parallel_decode` for both measured side by side, and for when the
standard library is the better answer anyway.


Sharing a Mutibs
^^^^^^^^^^^^^^^^

Concurrent calls on one :class:`Mutibs` queue rather than fail. Every method
runs inside CPython's per-object critical section, so a second thread waits for
the first and then proceeds:

.. code-block:: pycon

    >>> import threading
    >>> m = Mutibs()
    >>> def work():
    ...     for _ in range(1000):
    ...         m.append(True)
    ...
    >>> threads = [threading.Thread(target=work) for _ in range(8)]
    >>> for t in threads: t.start()
    >>> for t in threads: t.join()
    >>> len(m)
    8000
    >>> m.all()
    True

Every one of the 8000 appends landed, and none of them wrote a bit it wasn't
given. The same holds for the rest of the class: methods that resize
(``extend``, ``insert``, ``pop``, ``del``), methods that rewrite in place
(``write_u``, ``write_bytes``, the ``.hex`` and ``.u`` setters), methods that
reorder (``reverse``, ``rotate_left``, ``byte_swap``) and the whole read
surface.

Methods that touch two containers, such as ``a == b``, ``a & b`` and
``a.extend(b)``, lock both together. A thread writing to the *operand* of a
comparison is as safe as one writing to the receiver.

Serialising is correct, but it is not parallelism. Eight threads hammering one
:class:`Mutibs` will be slower than one thread doing the same work, because
they now pay for the lock as well. Give each thread its own container where
you can, and share a mutable one only where the sharing is the point.


Objects that alias a Mutibs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :class:`MutableView` and a :class:`Reader` over a :class:`Mutibs` keep the
source alive and read through it, so another thread can move the bits under
them. This is safe, and worth knowing the shape of: such a call either sees a
coherent state or raises. It never reads stale, freed or half-written bits.

.. code-block:: python

    view = m.field(0, 7)     # bits 0-7 of m

    # In another thread:
    del m[-64:]              # m may now be shorter than the view expects

    view.u                   # a value, or IndexError/ValueError - never garbage

So handle :exc:`IndexError` and :exc:`ValueError` around a view or a reader
whose source another thread is resizing, in the same way you would handle
``list`` indices that another thread is shortening. If the field must still be
there, the resize and the read belong in the same critical region of *your*
code, under your own lock.

A :class:`Reader` over a :class:`Tibs` has no such problem: the source can't
change. Only the cursor is shared, so several threads reading one such reader
each get a real value from the source, in an order nobody promised. If you
want independent cursors, give each thread its own :class:`Reader` — they are
cheap, and the underlying :class:`Tibs` is shared without copying.


The small print
^^^^^^^^^^^^^^^

There is one way a call on a contended :class:`Mutibs` can still be refused,
with ``RuntimeError: Already borrowed``.

A search over a multi-megabit container checks for :kbd:`Ctrl-C` periodically
rather than ignoring it until the scan finishes. Running Python suspends the
critical section, and a thread entering it at that moment finds the object busy.
Nothing is corrupted — a refused call has not changed anything — and it costs
that one call. If you have code that must never see one, either keep the searches
off the shared object or hold your own lock across them.

Two more things that are contracts rather than bugs:

* Per-call atomicity does not make a *pair* of calls atomic, even when the
  pair is a round trip. ``m.invert()`` twice returns to where it started, but
  the all-zeros state in between is a real state another thread may observe.
* :class:`Mutibs` is deliberately not iterable, so there is no iterator to
  invalidate. Use ``to_tibs()`` for a snapshot to iterate over, which is also
  the right thing to hand to other threads.


On a GIL-enabled build
^^^^^^^^^^^^^^^^^^^^^^

None of the locking above costs anything on a normal build, where the GIL was
already providing it. The free-threaded build is shipped as its own wheel,
because abi3 doesn't yet extend to ``3.14t``.


How this is checked
^^^^^^^^^^^^^^^^^^^

``tests/test_concurrency.py`` in the repository is a stress test for all of the
above, and is the place to look if you want the details rather than the summary.
It runs several threads against one shared object and checks invariants that only
hold if the guarantees do. It runs on GIL-enabled builds too, but only really
bites on ``3.14t``::

    python3.14t -m pytest tests/test_concurrency.py
