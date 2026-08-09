.. currentmodule:: tibs

Mutibs
------


The Mutibs class (pronounced 'mew-tibs') is a mutable container for binary data.

It's mostly a superset of the Tibs class, and adds methods that can change the contained data in place.

It can be less efficient to use a Mutibs if an immutable Tibs would work equally well, but the differences are
likely to be small in most use cases.

Methods on Tibs which produce an iterator over the data, such as :meth:`Tibs.find_all_iter`
and :meth:`Tibs.chunks_iter`, as well as iterating over the instance directly, are
not available for Mutibs. This is because the underlying data could change while the iterator is active.
To use these methods call :meth:`Mutibs.to_tibs` first to create an immutable copy, or
:meth:`Mutibs.take_tibs` to move the data if you no longer need the mutable object.

The Python protocol methods include most of those from ``Tibs``. Additions are:

* ``[]`` can also assign to bits or slices.
* ``+=`` concatenates in place, e.g. ``m += '0b1'``.
* ``*=`` repeats in place, e.g. ``m *= 6``.
* ``<<=`` and ``>>=`` shift in place.
* ``&=``, ``|=`` and ``^=`` perform in-place bitwise operations.

.. autoclass:: tibs.Mutibs
   :members:
   :member-order: groupwise
   :undoc-members:
