.. currentmodule:: tibs

Tibs vs Mutibs
--------------

* Tibs instances cannot change after they are created. This lets you use them as keys in dictionaries,
they can be hashed and used in sets.
* Methods that return iterators over the data are available for Tibs, but not Mutibs. This is because for
a Mutibs the data could change while the iterator is live. To use these methods on a Mutibs first convert
to a Tibs.
* Mutibs

Mutating and copy methods
^^^^^^^^^^^^^^^^^^^^^^^^^

Tibs has no mutating methods, but it can return a new Tibs. Mutibs has mutating methods with similar names.

* reversed -> reverse
* byte_swapped -> byte_swap
etc.