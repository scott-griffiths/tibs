# Release Notes

### August 1st 2026: version 2.0 rc1

Lots of new features added. A few caused some small backwardly incompatible
changes, so as the user base is still small I just accepted the better API
and upped the major version number.

Backwardly incompatible changes

* `Dtype` is now a base class and parsing factory rather than a concrete class,
  because a dtype can now describe a structured, multi-field value as well as
  a single one. `Dtype("u8")` returns a `DtypeSingle` instance instead of an
  object whose exact type is `Dtype`, so `repr(Dtype("u8"))` is now
  `DtypeSingle('u8')` rather than `Dtype('u8')`, and `Dtype.from_params` has
  moved to `DtypeSingle.from_params`. This is why the release is a new major
  version rather than 1.2 — see `DtypeArray` and `DtypeTuple` below. Code that
  constructs dtypes with `from_params`, checks their exact type, or relies on
  the old `repr` will need updating; dtype strings for single values such as
  `Dtype("u32")` are unchanged.

* The `View` constructor and `View.from_indices` no longer accept a `Mutibs`.
  Use `Mutibs.view()` for a live`MutableView`, or `Mutibs.to_tibs()` for an immutable copy to view.

* Minimum Python version now 3.11 instead of 3.10. Python 3.10 reaches its end
  of life in October 2026, and dropping it lets a single abi3 wheel cover every
  supported version — the buffer protocol added below is only part of Python's
  stable ABI from 3.11, so supporting 3.10 meant building and testing a separate
  version-pinned wheel for it.

Added

* Added `DtypeArray` and `DtypeTuple`, alongside the existing scalar dtype
  (now `DtypeSingle`), so a single `Dtype` can describe a structured,
  multi-field value: `Dtype("[u8; 4]")` for four repeated `u8` fields, and
  `Dtype("(u8, u16_le, bool)")` for a fixed sequence of differently-typed
  fields. Both nest to any depth, so an array can contain tuples and vice
  versa. All three classes support `pack`, `unpack`, `pack_values`,
  `unpack_values` and `unpack_values_iter`, and a `DtypeTuple` can express the
  same explicit-width, no-padding layouts as the standard-size forms of
  `struct` — `"(i16_le, i32_le, i32_le)"` matches `"<hll"`.

  ```python
  >>> Dtype("(u8, [bool; 3])").pack((5, [True, False, True]))
  Tibs('0b00000101101')
  ```

* Added the `bf16` dtype for bfloat16, the non-IEEE 16-bit float that keeps the
  8-bit exponent of an `f32` and truncates the mantissa to 7 bits, so its
  encoding is exactly the top half of the `f32` one. It is the format machine
  learning and DSP data is usually stored in, and it is not interchangeable
  with `f16`: the same sixteen bits mean different numbers in each. `bf16` is
  the only accepted length — there is no `bf8` or `bf32` — and it takes `_le`
  and `_be` like the other numeric dtypes. It has a new `DtypeKind.BFloat`
  kind, because `(DtypeKind.Float, 16)` already means IEEE `binary16` and a
  length alone cannot distinguish the two. For the same reason `from_f`,
  `to_f` and the `f` property are unchanged and stay IEEE only; bfloat16 is
  reached through the dtype.

  ```python
  >>> Tibs.from_value("bf16", 1.0).hex
  '3f80'
  >>> Tibs("0x3f80").to_value("f16")
  1.875
  ```

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

* Added `extracted` and `deposit` for reading and writing bit fields whose bits
  are scattered through a container by a mask, rather than being contiguous like
  the ones `field` handles. `extracted` reads the masked bits, packed together;
  `deposit` (in place, on `Mutibs`) and `deposited` (returning a new container)
  write them back, leaving the unmasked bits untouched. These are the bit-level
  equivalents of the x86 PEXT and PDEP instructions.

  ```python
  >>> Tibs('0b11010110').extracted('0b10110000')
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
* Added `View.to_value`, `MutableView.to_value` and `MutableView.write_value`,
  so any dtype is reachable through a view rather than only the nine
  interpretations that have their own view property. The view's byte order and
  bit order are applied first and the dtype then decodes the value the view
  denotes, so `view.to_value(dtype, start, end)` is always
  `view.to_tibs().to_value(dtype, start, end)`. `write_value` is the write
  direction, and like the other view write methods it can't change the length of
  its source.

  Byte order is stated in one place only. A dtype that names a byte order is
  refused by an `le`, `be` or `lsb0` view, at any nesting depth, because the
  view's layout is applied first and the suffix would be a second byte order
  rather than the only one — `t.le.to_value("u16_le")` would otherwise swap
  twice and land back on the big-endian reading. Put the byte order on the view
  or on the dtype, not both. The plain `view()` claims no layout and still
  passes any dtype through. Byte order stays on `Dtype` because a view applies
  one byte order to the whole view, so per-field records like
  `"(u8, u16_le, bool)"` and runs like `Tibs.from_values("u16_le", ...)` have no
  view equivalent; read those from the source `Tibs` or `Mutibs`.

  ```python
  >>> m = Mutibs.from_bytes(bytes.fromhex("07 01 00 00 44 33 22 11"))
  >>> m.lsb0.le.field(63, 32).to_value("u32")
  287454020
  >>> m.lsb0.le.field(63, 32).write_value("u32", 0xdeadbeef)
  >>> m.hex
  '07010000efbeadde'
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

* `Tibs` and `Mutibs` can now be pickled, so they can be sent through
  `multiprocessing`, stored in a cache, or deep copied with `copy.deepcopy`,
  which previously raised a `TypeError`. The pickled state is the `Codec.Raw`
  encoding, so any bit length is restored exactly and pickling costs about what
  copying costs — pickle the result of `encode()` yourself if you want the
  compression that `Codec.Auto` can give.

  ```python
  >>> pickle.loads(pickle.dumps(Tibs('0b110101')))
  Tibs('0b110101')
  ```

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

* Compound dtype packing and unpacking now build a cached flat record layout
  and take a bytewise fast path, bringing `DtypeTuple` operations equivalent to
  a `struct` call (such as `"(i16_be, i16_be, i32_be)"` against `">hhl"`) to
  around 1.5-1.8x the time of the handwritten `struct` code, down from roughly
  10x slower when compound dtypes were first implemented.
* A number of other operations — string/hex/oct/bin conversion, `from_values`,
  `Mutibs.reverse` and others — moved from bit-at-a-time to byte-at-a-time or
  whole-buffer implementations. Across the local comparison suite (see
  `tests/performance_comparison.py`), tibs is now a geometric-mean 3.2x faster
  than bitarray (1.4x at the median) and 1.9x faster than equivalent standard
  library code (1.5x at the median).

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
