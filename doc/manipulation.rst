.. currentmodule:: tibs

.. note::

    This part of the documentation is under construction.
    For now, see the API docs for the most complete information.

Manipulations
-------------

Mutating and copy methods
^^^^^^^^^^^^^^^^^^^^^^^^^

``Mutibs`` has many mutating methods, which change the value in-place and return ``None``.
Many of these methods have siblings which do the same task but don't modify the instance and
instead return a new copy. These 'copy' methods are also available on the immutable ``Tibs``.

.. csv-table::
   :header: "Mutibs mutating methods", "Tibs/Mutibs copy equivalent"

   ":meth:`~Mutibs.byte_swap`", ":meth:`~Mutibs.byte_swapped`"
   ":meth:`~Mutibs.insert`", ":meth:`~Mutibs.inserted`"
   ":meth:`~Mutibs.invert`", ":meth:`~Mutibs.inverted`"
   ":meth:`~Mutibs.replace`", ":meth:`~Mutibs.replaced`"
   ":meth:`~Mutibs.reverse`", ":meth:`~Mutibs.reversed`"
   ":meth:`~Mutibs.rotate_left`", ":meth:`~Mutibs.rotated_left`"
   ":meth:`~Mutibs.rotate_right`", ":meth:`~Mutibs.rotated_right`"
   ":meth:`~Mutibs.set`", ":meth:`~Mutibs.set_at`"
   ":meth:`~Mutibs.unset`", ":meth:`~Mutibs.unset_at`"


The linguistic oddities here are ``set_at()`` and ``unset_at()``, as the past-participle of 'set' is
also 'set', so the naming pattern failed (English is annoying sometimes).

Other Mutibs methods
^^^^^^^^^^^^^^^^^^^^

Not all mutating methods have a copy equivalent - things like ``clear()`` don't make sense for a
``Tibs``, and you can use the ``+`` operator to do non-mutating extensions.


* append
* clear
* extend
* extend_left
* pop
* reserve / capacity
