.. currentmodule:: tibs

View and MutableView
--------------------

Views wrap bit data with interpretation settings such as byte order and bit
order. They do not change the meaning of normal ``Tibs`` or ``Mutibs`` indexing;
they only affect conversions such as integer, float, bytes and labelled field
access.

``Tibs`` view helpers return :class:`View`, an immutable interpretation wrapper.
``Mutibs`` view helpers return :class:`MutableView`, a live mutable wrapper that
can read and write interpreted values back into the original ``Mutibs`` while
preserving the selected length.

Calling ``View(m)`` directly with a ``Mutibs`` still creates an immutable
snapshot. Use ``m.view()``, ``m.le``, ``m.be``, ``m.lsb0`` or ``m.msb0`` when
you want live mutable behaviour.

See :doc:`views` in the manual for more details.

View
^^^^

.. autoclass:: tibs.View
   :members:
   :member-order: groupwise
   :undoc-members:

MutableView
^^^^^^^^^^^

.. autoclass:: tibs.MutableView
   :members:
   :member-order: groupwise
   :undoc-members:
