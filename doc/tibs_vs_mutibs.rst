.. currentmodule:: tibs

Tibs vs Mutibs
--------------

* Tibs instances cannot change after they are created. This lets you use them as keys in dictionaries,
  they can be hashed and used in sets.
* Methods that return iterators over the data are available for Tibs, but not Mutibs. This is because for
  a Mutibs the data could change while the iterator is live. To use these methods on a Mutibs first convert
  to a Tibs.


Mutating and copy methods
^^^^^^^^^^^^^^^^^^^^^^^^^

Tibs has no mutating methods, but it can return a new Tibs. Mutibs also has mutating methods with similar names.

* :meth:`Tibs.reversed` c.f. :meth:`Mutibs.reverse`
* :meth:`Tibs.byte_swapped` c.f.  :meth:`Mutibs.byte_swap`

etc.