.. currentmodule:: tibs

View
----


The View class wraps a Tibs with different interpretation settings.

It records how operations such as integer conversion, byte conversion and
field extraction should interpret those bits.

A view created from a Mutibs stores a Tibs snapshot. Later
changes to the original Mutibs are not reflected in the view.

.. autoclass:: tibs.View
   :members:
   :member-order: groupwise
   :undoc-members: