.. currentmodule:: tibs

.. _exotic-floats:

Exotic floating-point formats
-----------------------------

Python floats are normally 64 bits long. The familiar IEEE formats also have
32-bit and 16-bit versions, but lower-precision floating-point formats have
become increasingly useful, largely because of machine-learning algorithms
and hardware.

The layout column in the tables below shows how the available bits are split
into sign, exponent and stored mantissa bits. The implicit leading significand
bit is not included in the mantissa count. More exponent bits provide greater
range; more mantissa bits provide greater precision.

Names such as E4M3 do not identify a unique encoding by themselves: OCP and
P3109 formats with similar layouts have different biases and special values.
Tibs therefore prefixes the OCP names and uses the P3109 ``binary8p3`` and
``binary8p4`` terminology instead of naming either family with E/M alone.


Draft IEEE P3109 eight-bit formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The P3109 formats are part of an ongoing IEEE standardisation project. Tibs
supports two of the formats in the public draft: K8P3SE, which has greater
range, and K8P4SE, which has greater precision. Each has only 256 possible bit
patterns.

.. list-table:: P3109 dtype reference
   :header-rows: 1
   :widths: 24 8 20 22 28

   * - Dtype
     - Bits
     - Layout
     - Finite non-zero range
     - Special values
   * - ``binary8p3``
     - 8
     - S/E/M 1/5/2, bias 16
     - ±2⁻¹⁷ to ±49,152
     - One zero, one NaN, ±infinity
   * - ``binary8p4``
     - 8
     - S/E/M 1/4/3, bias 8
     - ±2⁻¹⁰ to ±224
     - One zero, one NaN, ±infinity

Unlike most floating-point encodings, each format has only one zero and one
NaN bit pattern. Values outside the finite range round to positive or negative
infinity.

You can examine every possible decoded value with a single expression::

    >>> values = Tibs(bytes(range(256))).to_values("binary8p4")
    >>> len(values)
    256

When converting from a Python float, unrepresented values are rounded to the
nearest representable value, with ties going to the value whose trailing
significand bit is even.

.. note::
    P3109 remains an active project rather than a published standard. Tibs
    implements the public draft available for this release and may update
    these provisional formats as the standard develops.


OCP microscaling element formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Open Compute Project defines a family of Microscaling Formats (MX). These
normally combine an external scale with a block of low-precision elements.
Tibs supports the individual element encodings, including the E8M0 scale
encoding, but does not yet associate them into scaled blocks.

.. list-table:: OCP dtype reference
   :header-rows: 1
   :widths: 27 7 19 21 28

   * - Dtype
     - Bits
     - Layout
     - Finite non-zero range
     - Special values
   * - ``ocp_e5m2_saturate`` / ``ocp_e5m2_overflow``
     - 8
     - S/E/M 1/5/2, bias 15
     - ±2⁻¹⁶ to ±57,344
     - Signed zero, ±infinity and six NaN codes
   * - ``ocp_e4m3_saturate`` / ``ocp_e4m3_overflow``
     - 8
     - S/E/M 1/4/3, bias 7
     - ±2⁻⁹ to ±448
     - Signed zero and two NaN codes
   * - ``ocp_e3m2``
     - 6
     - S/E/M 1/3/2, bias 3
     - ±2⁻⁴ to ±28
     - Signed zero; no NaN or infinity
   * - ``ocp_e2m3``
     - 6
     - S/E/M 1/2/3, bias 1
     - ±2⁻³ to ±7.5
     - Signed zero; no NaN or infinity
   * - ``ocp_e2m1``
     - 4
     - S/E/M 1/2/1, bias 1
     - ±2⁻¹ to ±6
     - Signed zero; no NaN or infinity
   * - ``ocp_e8m0``
     - 8
     - Unsigned exponent, bias 127
     - 2⁻¹²⁷ to 2¹²⁷
     - One NaN; no zero or infinity
   * - ``ocp_int8``
     - 8
     - Signed integer × 2⁻⁶
     - -2 to 127/64
     - No NaN or infinity

The E4M3 and E5M2 names have an explicit packing policy. The ``saturate`` and
``overflow`` variants decode identical bits, but differ when a value reaches
the end of the finite range::

    >>> Tibs.from_value("ocp_e4m3_saturate", 1000.0).hex
    '7e'
    >>> Tibs.from_value("ocp_e4m3_overflow", 1000.0).hex
    'ff'

The formats behave as follows when packing values outside their finite range:

* ``ocp_e5m2_saturate`` clamps to ±57,344. Infinities are also clamped,
  despite the format having infinity encodings.
* ``ocp_e5m2_overflow`` converts terminal overflow to positive or negative
  infinity.
* ``ocp_e4m3_saturate`` clamps to ±448.
* ``ocp_e4m3_overflow`` converts terminal overflow to NaN because E4M3 has no
  infinity encoding.
* ``ocp_e3m2``, ``ocp_e2m3`` and ``ocp_e2m1`` clamp to their largest finite
  value with the appropriate sign.
* ``ocp_int8`` clamps to its asymmetric range. Tibs includes the optional most
  negative OCP value: ``0x80`` decodes as ``-2.0``.

The E8M0 format is unsigned and is designed to encode scaling factors. It uses
one byte for the powers of two from 2⁻¹²⁷ to 2¹²⁷, plus a NaN value. Tibs does
not round values when packing it: ``ocp_e8m0`` accepts only a positive,
in-range, exact power of two or NaN. Zero, negative values, infinity and values
between powers of two raise ``ValueError``::

    >>> Tibs.from_values("ocp_e8m0", [0.5, 1.0, 2.0]).hex
    '7e7f80'

The INT8 format is a signed two's-complement integer with an implicit factor of
2⁻⁶. Despite its name, it therefore decodes to a Python ``float``.

These are raw scalar elements only. Tibs does not store a shared scale, choose
one automatically, associate elements into fixed-size blocks, or implement
scaled block arithmetic. The OCP specification reserves names such as MXFP4
for the combination of a scale and a block of E2M1 elements; the Tibs name
``ocp_e2m1`` avoids implying that the block behaviour is implemented.


Conversion
^^^^^^^^^^

Ordinary conversions from Python floats use round-to-nearest, ties-to-even, and
round once: directly from the Python ``float`` value to the target format.

OCP E4M3 and E5M2 accept Python NaNs and write the canonical code ``0xff``.
Decoding accepts every NaN code defined by the format. The
smaller E2M1, E2M3 and E3M2 formats, and ``ocp_int8``, reject NaN because they
have no NaN representation. They saturate infinities.


.. _exotic-float-standards:

Specification versions
^^^^^^^^^^^^^^^^^^^^^^

The OCP definitions are frozen to `OCP Microscaling Formats v1.0`_ (September
2023) and `OCP OFP8 revision 1.0`_ (including its December 2023 correction).

The P3109 K8P3SE and K8P4SE support follows the public draft available when
each Tibs version is released. The `IEEE P3109 project`_ remains active, so
these provisional formats may be updated as the standard develops.

.. _OCP Microscaling Formats v1.0: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
.. _OCP OFP8 revision 1.0: https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
.. _IEEE P3109 project: https://standards.ieee.org/ieee/3109/11165/
