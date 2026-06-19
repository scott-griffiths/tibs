# Release Notes

### Unreleased: version 0.10.

Backwardly incompatible changes

* Removed `__hash__` method from `Tibs`. The way that we allow `Tibs` to compare
  equal to other types  (e.g. `Tibs('0xf') == '0b1111'`) means that the hash
  shouldn't have been available. The new recommendation is to use the `encode` method
  to convert to `bytes` objects to use as keys.

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

Backwardly incompatible changes

* Renamed:
  `Tibs.chunks` -> `Tibs.chunks_iter`
  `Tibs.find_all` -> `Tibs.find_all_iter`
  `Tibs.rfind_all` -> `Tibs.rfind_all_iter`
* New `View` class that replaces both the endianness and the lsb0 mode.

Added

* Added `replaced`, `inserted`, `inverted`, `rotated_right`, `rotated_left`, `set_at`,
  `unset_at` methods.
* Added `encode` and `decode` methods to `Tibs` and `Mutibs`. These store/compress the full
  instance as a bytes object.
* Added new `Tibs`/`Mutibs` `chunks`/`find_all` methods that return a list.
* Added `Tibs.rchunks_iter` method.
* New `Tibs.view()` method and `be` / `le` / `msb0` / `lsb0` properties to create views.
* Various performance improvements.
* Added 'Tibs' cat mascot to the documentation. Thanks to Ada Griffiths.

### April 3rd 2026: version 0.6.0.

Backwardly incompatible changes

* `Mutibs` methods no longer mutate and return self.
* `ror` renamed to `rotate_right`, `rol` renamed to `rotate_left`.

Added

* Added LSB0 mode.
* Added byte endianness for integer and float interpretations.
* Added `Tibs.rfind_all` method.
* `Mutibs.set` method split into `.set` and `.unset` methods.
* New `byte_swapped()` method.
* New `reversed()` method.
* `.bin`, `.oct`, `.hex` and `.bytes` readonly properties added.
* Bug fixes and documentation improvements.


### March 2025: version 0.5.7

Just a single bug fix.

* Issue #1: Fix for panic when trying to construct 128 bit ints. Thanks to @mgorny and @FineFindus. 


### March 2025: version 0.5.6

First version used as a dependency for bitstring. This increases the visibility of tibs quite a lot even though
it isn't yet turned on by default in bitstring.


### February 2026: version 0.5.0.

The first beta release. This rounded out the initial API, added `Mutibs.pop`,
made `count()` work with multi-bit patterns, updated to PyO3 0.28, and expanded
the documentation and generated Python test coverage.


### January 2026: version 0.4.

Refined the mutable sequence API: `append` now adds a single bit, the old
append-style operation was renamed to `extend`, and `prepend` became
`extend_left`. This version also improved chunk iteration performance and
continued moving constructor and conversion helpers into shared Rust code.


### December 2025: version 0.3.

Reworked the core representation and sharing model. `Mutibs` stopped wrapping a
`Tibs` internally, `Tibs` moved toward shared immutable storage and cheaper
slicing, byte constructors gained offset and length support, and raw-data access
and typing stubs were added.


### December 2025: version 0.2.

Focused on documentation, examples, and internal cleanup. String promotion and
constructor helpers were simplified, `replace` was made safer around
self-assignment, and skipped tests were either fixed or removed.


### November 2025: version 0.1.

The first releases using the `Tibs` and `Mutibs` names. These added the core
binary, octal and hexadecimal conversions, containment, replacement, mutable
operations including `Mutibs.rfind`, numeric conversions, arbitrary-size integer
construction, and early CI and Read the Docs support.


### November 2025; version 0.0.1.

Project start!

The original version is a cut-down and rebranded version of bitformat.

Its main job is to reserve the name on PyPI.
