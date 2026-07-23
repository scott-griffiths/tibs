.. currentmodule:: tibs

Examples
--------

Some examples using the Tibs library. The code for these can be found in the
``examples/`` directory of the repo. They are grouped by the view (see
:doc:`manual`) each one leans on most, though several use more than one.

The bits as a sequence
^^^^^^^^^^^^^^^^^^^^^^

Searching a byte stream and slicing fields out of it.

.. toctree::
    :maxdepth: 1

    example_log_scan
    example_instruction_scan
    example_patch_config

Typed fields and views
^^^^^^^^^^^^^^^^^^^^^^

Reading and writing numeric fields, with byte order and bit labels handled by a view.

.. toctree::
    :maxdepth: 1

    example_construct
    example_sensor_samples
    example_little_endian_registers
    example_ebpf_instruction
    example_scattered_field

Sets of bits
^^^^^^^^^^^^

Treating the container as a set of bit positions for algebra and comparison.

.. toctree::
    :maxdepth: 1

    example_sieve
    example_fingerprints
