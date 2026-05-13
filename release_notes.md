# Release Notes

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
