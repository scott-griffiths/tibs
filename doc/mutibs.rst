.. currentmodule:: tibs

Mutibs
------


The Mutibs class (pronounced 'mew-tibs') is a mutable container for binary data.

It's mostly a superset of the Tibs class, and adds methods that can change the contained data in-place.

Unlike Tibs, a Mutibs instance is not hashable, and so can't be used as a dictionary key or in a set.
It can also be less efficient to use a Mutibs if an immutable Tibs would work equally well, but the differences are
likely to be small in most use-cases.

Methods on Tibs which produce an iterator over the data, such as :meth:`Tibs.find_all_iter`
and :meth:`Tibs.chunks_iter`, as well as iterating over the instance directly, are
not available for Mutibs. This is because the underlying data could change while the iterator is active.
To use these methods call :meth:`Mutibs.to_tibs` first to create an immutable copy.

The special methods include those from Tibs, plus the following which mutate the instance:

* ``[]`` : Setting slices, e.g. ``m[4:16] = [1, 0]``
* ``+=``: Concatenation in-place, e.g. ``m += '0b1'``
* ``*=``: Concatenation of self in-place, e.g. ``m *= 6``
* ``<<=``: Left bit shift in-place, e.g. ``m <<= 3``
* ``>>=``: Right bit shift in-place, e.g. ``m >>= 3``
* ``&=``: Bit-wise AND in-place
* ``|=``: Bit-wise OR in-place
* ``^=``: Bit-wise XOR in-place

.. autoclass:: tibs.Mutibs
   :members:
   :member-order: groupwise
   :undoc-members:
