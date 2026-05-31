# Release Notes

### May 31st 2026: version 0.9.

Backwardly incompatible changes

* `Mutibs.replace()` now returns the number of replacements made instead of
  `None`.

Added

* Added the `Dtype` class and `DtypeKind` enum for describing typed binary
  values. Dtypes can be created from compact strings such as `Dtype("u8")`,
  `Dtype("i16")`, `Dtype("f32_le")`, `Dtype("hex8")` and `Dtype("bytes16")`,
  or explicitly with `Dtype.from_params(...)`.
* Added `Tibs.from_value`, `Tibs.from_values`, `Tibs.to_value`,
  `Tibs.to_values_iter` and `Tibs.to_values`. These accept either a `Dtype`
  instance or a dtype string.
* Added matching value conversion methods to `Mutibs`, except for
  `to_values_iter`.
* Added `start` and `end` parameters to `Tibs.byte_swapped`,
  `Mutibs.byte_swap` and `Mutibs.byte_swapped`.
* Added `start` and `end` parameters to the `to_u`, `to_i`, `to_f`, `to_bin`,
  `to_oct`, `to_hex` and `to_bytes` interpretation methods.

Improvements

* Added periodic Python signal checks to several longer-running operations so
  they can be interrupted more reliably.


### May 25th 2026: version 0.8.

Backwardly incompatible changes

* Renamed the constructor keyword from `endianness` to `byte_order` for
  `from_u`, `from_i` and `from_f`. The `Endianness` enum is unchanged.

Added

* Added the `MutableView` class. Views from `Mutibs` are now live mutable
  views, so interpreted writes through `m.le`, `m.lsb0`, selected fields, or
  explicit mutable views update the original `Mutibs`.
* Added fixed-width write methods and settable interpretation properties:
  `write_u`, `write_i`, `write_f`, `write_bin`, `write_oct`, `write_hex` and
  `write_bytes`, plus settable `.u`, `.i`, `.f`, `.bin`, `.oct`, `.hex` and
  `.bytes` properties where mutation is supported.
* Added labelled field helpers. `Tibs.field(a, b)` and `Mutibs.field(a, b)`
  use default MSB0 labels, while `View.field(a, b)` and
  `MutableView.field(a, b)` use the view's current bit order. Field endpoints
  are inclusive and can be given in either order.
* Added `View.from_indices` and `MutableView.from_indices` for low-level views
  over explicit source-bit positions.
* Added read-only `byte_order` and `bit_order` properties to `View` and
  `MutableView`.

Fixes

* Fixed LSB0 field extraction so it reverses labels within bytes without also
  reversing byte order.
* Improved error handling around field labels and byte-order construction,
  including reporting negative field labels as `ValueError`.


### May 4th 2026: version 0.7.

* Added replaced, inserted, inverted, rotated_right, rotated_left, set_at,
  unset_at methods.
* Added encode and decode methods to Tibs and Mutibs. These store/compress the full
  instance as a bytes object.
* Renamed:
  Tibs.chunks -> Tibs.chunks_iter
  Tibs.find_all -> Tibs.find_all_iter
  Tibs.rfind_all -> Tibs.rfind_all_iter
* Added new Tibs/Mutibs chunks/find_all methods that return a list.
* Added Tibs.rchunks_iter method.
* New View class that replaces both the endianness and the lsb0 mode.
* New Tibs.view() method and be / le / msb0 / lsb0 properties to create views.
* Various performance improvements.
* Added 'Tibs' cat mascot to the documentation. Thanks to Ada Griffiths.

### April 3rd 2026: version 0.6.0.

* Added LSB0 mode.
* Added byte endianness for integer and float interpretations.
* Added Tibs.rfind_all method.
* Mutibs.set method split into .set and .unset methods.
* New byte_swapped() method.
* New reversed() method.
* Mutibs methods no longer mutate and return self.
* .bin, .oct, .hex and .bytes readonly properties added.
* ror renamed to rotate_right, rol renamed to rotate_left
* Bug fixes and documentation improvements.

### March 2025: version 0.5.7

Just a single bug fix.

* Issue #1: Fix for panic when trying to construct 128 bit ints. Thanks to @mgorny and @FineFindus. 

### March 2025: version 0.5.6

First version used as a dependency for bitstring. This increases the visibility of tibs quite a lot even though
it isn't yet turned on by default in bitstring.

I shall be better at making release notes from now on!

### November 2025 - February 2026; versions 0.1.0 - 0.5.0.

From the first release with the `Tibs` and `Mutibs` classes to the first beta
with a completed initial API.

### November 2025; version 0.0.1.

#### Project start

The original version is a cut-down and rebranded version of bitformat.

Its main job is to reserve the name on PyPI.
