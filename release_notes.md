# Release Notes

### Unreleased: version 1.2

Backwardly incompatible changes

* Minimum Python version now 3.11 instead of 3.10. Python 3.10 reaches its end
  of life in October 2026, and dropping it lets a single abi3 wheel cover every
  supported version — the buffer protocol added below is only part of Python's
  stable ABI from 3.11, so supporting 3.10 meant building and testing a separate
  version-pinned wheel for it.

Added

* Eight methods for comparing two containers without building an intermediate
  object: `count_and`, `count_or`, `count_xor`, `count_andnot`, `intersects`,
  `is_disjoint`, `is_subset_of` and `is_superset_of`. The counts are equivalent
  to `(a & b).count(1)` and so on, and `count_xor` is the Hamming distance.
  Both containers must be the same length, as for the `&`, `|` and `^`
  operators.

  ```python
  >>> a, b = Tibs('0b1100'), Tibs('0b1010')
  >>> a.count_xor(b)      # the Hamming distance
  2
  >>> Tibs('0b1010').is_subset_of('0b1011')
  True
  >>> Tibs('0b1011').is_superset_of('0b1010')
  True
  >>> Tibs('0b1100').is_disjoint('0b0011')
  True
  ```

  Skipping the intermediate object makes the counts around 3x faster for short
  containers and much more than that for long ones. The four predicates stop as
  soon as they know the answer, so they can return almost immediately where
  building `a & b` would have to do all the work.

* The `value` parameter of `count` is now optional, and counts the set bits when
  it's not given, so `t.count()` is the same as `t.count(1)`. This matches the
  new `count_and` and friends, which are all counts of set bits.

  ```python
  >>> Tibs('0xef').count()
  7
  ```

* `count` now takes a `byte_aligned` parameter, so a pattern can be counted on
  byte boundaries only, matching `find` and `find_all`.

  ```python
  >>> Tibs('0x1f2e3f').count('0x0f', mask='0x0f', byte_aligned=True)
  2
  ```

* Added `extract` and `deposit` for reading and writing bit fields whose bits are
  scattered through a container by a mask, rather than being contiguous like the
  ones `field` handles. `extract` reads the masked bits, packed together;
  `deposit` (in place, on `Mutibs`) and `deposited` (returning a new container)
  write them back, leaving the unmasked bits untouched. These are the bit-level
  equivalents of the x86 PEXT and PDEP instructions.

  ```python
  >>> Tibs('0b11010110').extract('0b10110000')
  Tibs('0b101')
  >>> Tibs('0b11010110').deposited('0b111', '0b10110000').bin
  '11110110'
  ```

* Searches can now take a `mask`, so that patterns can contain don't-care bits.
  The mask must be the same length as the bits being searched for, and only the
  bits set in it need to match, which makes it easy to pick out a field from a
  fixed-width encoding. It's available on `find`, `rfind`, `find_all`,
  `find_all_iter`, `rfind_all_iter`, `count`, `replace` and `replaced`, and
  combines with `start`, `end`, `count` and `byte_aligned` as you'd expect. For
  `replace` the mask affects only the matching — the whole of each match is
  still replaced.

  ```python
  >>> Tibs('0x1f2e3f').find_all('0x0f', mask='0x0f', byte_aligned=True)
  [0, 16]
  ```

* The `u` and `i` interpretations no longer have a 128 bit limit. Any positive
  length now works, for `from_u` / `from_i`, the `u` and `i` properties and
  `to_u` / `to_i`, the `uN` / `iN` dtypes, views, and the `u` and `i` format
  codes. The zero-length case is still an error, as there's nothing to interpret.

  ```python
  >>> Tibs.from_u(2 ** 200 - 1, 200).u == 2 ** 200 - 1
  True
  ```
* Added `__format__` to `Tibs`, `Mutibs`, `View` and `MutableView`, so they can be
  used directly in f-strings and with `str.format()`. The type codes are `b`, `o`,
  `x` and `X` for the bit representations (equivalent to the `bin`, `oct` and `hex`
  properties, so leading zeros are kept), plus `u` and `i` for the unsigned and
  signed integer interpretations. The `#` flag adds a `0x` / `0X` / `0b` / `0o`
  prefix, and `_` groups the digits — with the group size settable through the
  otherwise unused precision field, so `f"{t:_.8b}"` groups binary by byte. Fill,
  alignment and width work as they do elsewhere in Python.

  ```python
  >>> f"{Tibs('0xac804f4b'):#_.2x}"
  '0xac_80_4f_4b'
  ```

  Three things differ from integer formatting, because a `Tibs` is a sequence rather
  than a number and its length is part of its value. Groups are counted from bit zero,
  so it's the last group that comes up short rather than the first. Padding is never
  itself grouped. And the fill character must not be one that could be mistaken for
  the data, so zero padding is rejected for `b`, `o`, `x` and `X` — it would silently
  change the apparent length of the value, where for an integer it changes nothing.
  Pad with `<`, `>` or `^` and a non-digit fill instead, or use `u` or `i`.

* `Tibs` now exports the buffer protocol, so it can be passed straight to
  anything that takes a bytes-like object — `memoryview(t)`, `array.array`,
  `numpy.frombuffer`, a socket or file `write`, and so on — without copying its
  data first.

  ```python
  >>> bytes(memoryview(Tibs('0xff00')))
  b'\xff\x00'
  ```

  The buffer is read-only and covers whole bytes, so for a length that isn't a
  multiple of 8 the last byte includes the padding bits, which are not masked to
  zero. Exporting needs the underlying storage to start on a byte boundary; when
  it doesn't (after slicing at a bit offset, say) a `BufferError` is raised, and
  `to_bytes` or `to_padded_bytes` will give you an owned copy instead.

  This is deliberately not offered on `Mutibs`. Its storage moves as the
  container is edited, so a borrowed view of it could not be kept valid.

* Added a `tibs.__version__` string, alongside the `tibs.__author__` that was
  already there.

* Free-threaded Python (3.14t) is now a supported configuration, with wheels
  built for it and the test suite run against them. Tibs declares that it does
  not need the global interpreter lock, so importing it doesn't switch the GIL
  back on, and work on separate containers runs on separate threads in parallel
  rather than taking turns.

  A `Tibs` is immutable and can be shared freely between threads. A `Mutibs`
  cannot: if two threads use the same one at the same time, one of them gets a
  `RuntimeError` telling it the object is already borrowed, so give each thread
  its own container or guard the shared one with a lock. Nothing here changes
  on a normal build of Python, where the GIL serializes the calls anyway.

Changed

* Inverting an empty `Tibs` or `Mutibs` with `~` now returns an empty container
  instead of raising a `ValueError`. This matches what `Tibs.inverted` and
  `Mutibs.invert` already did when there were no bits to invert.

Performance improvements

* The `bin`, `oct` and `hex` properties and the `to_bin`, `to_oct` and `to_hex`
  methods now build their string from whole bytes at a time instead of looking
  at each bit or digit in turn.
* Binary, octal and hex strings are parsed several digits at a time straight
  into bytes, which speeds up construction from a string. This made the cache of
  recently parsed strings redundant, so it has been removed, and with it the
  `hex`, `lru` and `once_cell` dependencies.
* `from_values` and `Dtype.pack_values` pack whole bytes at a time for numeric
  dtypes whose length is a multiple of 8 bits.
* `Mutibs.reverse` now takes the same byte-level fast path that `reversed`
  already used, rather than reversing a bit at a time.

### July 18th 2026: version 1.1

Added

* Added `Tibs.to_bools()` and `Mutibs.to_bools()` for converting to a list of
  bools. This is much faster than iterating with `list(...)`, and accepts the
  usual optional `start` and `end` parameters.

Fixed

* Fixed byte-aligned slice assignment and slice deletion writing to the wrong
  bits on a `Mutibs` created by slicing another `Mutibs` (the underlying
  storage of such objects does not always start on a byte boundary).
* Fixed a crash in `from_joined` when the same empty container was repeated
  in a list.
* Ranges passed to `set`, `unset`, `set_at` and `unset_at` now behave exactly
  like the list of their contents. Empty ranges (such as `range(1, 0)` or
  `range(0, 2, -1)`) are no-ops instead of crashing, and negative range values
  index from the end just as they do when passed in a list.

Performance improvements

* `find`, `rfind`, `find_all` and `count` with patterns of up to 64 bits now
  scan a byte at a time instead of a bit at a time.
* `find`, `rfind` and the `find_all` iterators with patterns longer than 64
  bits now use a fast 64-bit prefix scan with verification, falling back to
  the previous algorithm only for highly repetitive data.
* Setting and clearing bits by index with `set` and `unset` writes directly to
  the underlying bytes.


### July 12th 2026: version 1.0.0

This is the first stable release. The documented public API is now intended
to remain compatible across future 1.x releases.

Backwardly incompatible changes

* Tightened automatic promotion to `Tibs` and `Mutibs`. Constructors and other
  bit-sequence arguments now accept only unambiguous inputs: existing
  `Tibs`/`Mutibs` objects, strings, `bytes`/`bytearray`/`memoryview`, and strict
  list or tuple bit patterns containing only `True`, `False`, `0` or `1`.
  Arbitrary iterables, file-like objects such as `io.BytesIO`, `array.array`,
  and numeric lists such as `[1, 2, 3]` no longer silently convert through
  truthiness. Use `from_bools(...)` for truthy iterables, `from_bytes(...)` for
  explicit byte data, and `from_values(...)` for fixed-width numeric values.


### July 5th 2026: version 0.12.0

This is effectively the first beta before a 1.0 release. I don't plan to change
the API or add new features before the 1.0 release.

Backwardly incompatible changes

* Minimum Python version now 3.10 instead of 3.8
* Renamed the `Endianness` enum to `ByteOrder` as it's more consistent.

Added

* Added `Tibs.split_at()` and `Mutibs.split_at()` for partitioning a bit
  sequence at one or more bit positions.

Fixed

* Fix LSB0 view value ordering.


### June 27th 2026: version 0.11.0.

Backwardly incompatible changes

* Restored `Tibs.__hash__`, reversing the change made in version 0.10.0.
  `Tibs` is now hashable again, while `Mutibs` remains unhashable.
* Instead, equality no longer promotes strings, bytes or iterables to bit containers.
  `Tibs` and `Mutibs` compare equal to each other when their bit sequences
  match, but expressions such as `Tibs('0xf') == '0b1111'` now return `False`.
  Use `Tibs('0b1111')`, `Mutibs('0b1111')`, or representation properties such
  as `.bin` and `.hex` when comparing against literal representations.

Added

* Added `Tibs.to_padded_bytes()` and `Mutibs.to_padded_bytes()`, which convert
  to `bytes` after appending 0 to 7 zero bits on the right to reach a byte
  boundary.
* Added `Dtype.pack`, `Dtype.pack_values`, `Dtype.unpack`,
  `Dtype.unpack_values` and `Dtype.unpack_values_iter`.
* Added `DtypeKind.Bool` and `DtypeKind.Bits`. `Dtype("bool")` is a fixed
  one-bit dtype that packs `True`, `False`, `0` and `1`, and unpacks to Python
  `bool`. `Dtype("bitsN")` packs fixed-length bit sequences and unpacks them as
  immutable `Tibs` values.

Fixes

* `Dtype` instances now compare and hash by kind, length and byte order instead
  of by object identity.
* Byte order is now rejected for every non-numeric dtype kind.


### June 20th 2026: version 0.10.0.

Backwardly incompatible changes

* Removed `__hash__` method from `Tibs`. Because `Tibs` can compare equal to
  other types (for example `Tibs('0xf') == '0b1111'`), the hash should not have
  been available. The new recommendation is to use the `encode` method to
  convert to `bytes` objects to use as keys.

Fixes

* Fixed LSB0 field extraction and assignment ordering for combined bit-order and
  byte-order views. LSB0 labels now identify the physical bits while extracted
  fields are returned in field-value order instead of being bit-reversed.
* Fixed dtype length validation for `bin`, `oct` and `hex` values passed to
  `from_value` and `from_values`.


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
