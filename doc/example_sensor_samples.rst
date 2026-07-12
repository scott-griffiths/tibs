.. currentmodule:: tibs


Decoding packed sensor samples
------------------------------

Many data acquisition formats pack fixed-width readings without padding each
sample to a full byte. Here each ADC reading is 12 bits, so four samples fit in
six bytes.

.. literalinclude:: ../examples/sensor_samples.py
   :language: python

For custom integer widths, ``from_values`` and ``to_values`` keep the packing
logic focused on the data width instead of on manual slicing.
