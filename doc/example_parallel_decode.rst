.. currentmodule:: tibs


Using threads
-------------

A :class:`Tibs` is immutable, so any number of threads can read one without
locking, and slicing it shares its storage rather than copying it. This
makes splitting a large container across threads almost free:
each thread takes a piece and nothing has to be coordinated.

This example averages 12 million 12-bit samples — the resolution a great many ADCs
actually produce — dividing the work between one, two, four and eight threads
and printing how much faster each division is than doing it on a single thread.

.. literalinclude:: ../examples/parallel_decode.py
   :language: python

Whether any of it gets faster is up to the interpreter rather than the library.
On a free-threaded build the pieces are decoded at the same time, on my Quad-Core Intel Core i5 I get ::

    free-threaded build, GIL off
    1 threads:  564.5 ms   1.00x
    2 threads:  284.8 ms   1.98x
    4 threads:  151.3 ms   3.73x
    8 threads:  140.2 ms   4.03x

On a GIL-enabled build the same code should take about the same time for all the thread numbers.

See :doc:`free_threading` for what can be shared between threads, and for what
a :class:`Mutibs` does differently.
