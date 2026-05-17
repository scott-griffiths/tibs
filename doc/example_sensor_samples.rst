.. currentmodule:: tibs


Decoding packed sensor samples
------------------------------

Many data acquisition formats pack fixed-width readings without padding each
sample to a full byte. Here each ADC reading is 12 bits, so four samples fit in
six bytes.

.. literalinclude:: ../examples/sensor_samples.py
   :language: python

The example trims the byte stream to the number of complete samples before
chunking it. That is useful when a real transport pads the final byte or record.
