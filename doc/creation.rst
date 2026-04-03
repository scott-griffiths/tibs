.. currentmodule:: tibs

Creation, Views and Interpretations
-----------------------------------

Creation
^^^^^^^^

Data views
^^^^^^^^^^

to_bin() - always available.

to_oct(), to_hex(), to_bytes() - need to be appropriate length.

Read-only properties are equivalent - bin, oct, hex, bytes.

Can always reconstruct the Tibs from a view - from_bin(), from_oct(), from_hex(), from_bytes().

Data interpretations
^^^^^^^^^^^^^^^^^^^^

Unlike the data views the interpretations have many-to-one relationships in both directions.