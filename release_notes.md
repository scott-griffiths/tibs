# Release Notes

### August 12th 2026: version 2.0.1

Lots of new features added. A few caused some small backwardly incompatible
changes, so as the user base is still small I just accepted the better API
and upped the major version number.

Backwardly incompatible changes

* `Dtype` is now a base class and parsing factory rather than a concrete class.
  `Dtype("u8")` returns a `DtypeSingle`; see `DtypeArray` and `DtypeTuple`
  below. Dtype strings are unchanged, but code that uses `from_params`, checks a
  dtype's exact type, or relies on its `repr` will need updating.

* The `View` constructor and `View.from_indices` no longer accept a `Mutibs`.
  Use `Mutibs.view()` for a live `MutableView`, or `Mutibs.to_tibs()` for an
  immutable copy to view.

* The minimum Python version is now 3.11 instead of 3.10.

* Renamed `set_at` and `unset_at` to `with_set` and `with_unset` on both `Tibs`
  and `Mutibs`.

* Renamed `Mutibs.as_tibs` to `Mutibs.take_tibs` and `Mutibs.as_raw_data` to
  `Mutibs.take_raw_data`, as both empty the object they are called on.

* Removed `to_raw_data` from `Tibs` and `Mutibs`. Use `to_padded_bytes()` with
  `len()` to round-trip the same value, or `encode(Codec.Raw)` to persist it.

* `Mutibs.capacity` is now a property rather than a method.

* `Mutibs.__iter__` is now `None`, so `isinstance(m, collections.abc.Iterable)`
  is `False` where it used to be `True`.

* Views now compare by the bits they present rather than by their whole source,
  so a `View` and a `MutableView` over the same bits and layout are now equal.

* The buffer protocol now needs a length that is a multiple of 8, and raises
  `BufferError` otherwise. It used to export a partial final byte, whose bits
  past the logical length are not padding but whatever the storage was sliced
  out of, so `Tibs('0xffff')[0:4]` and `Tibs('0b1111')` presented different
  bytes despite being equal. Use `to_padded_bytes()` for those.

* The `offset` and `length` parameters of `Tibs.from_bytes` and
  `Mutibs.from_bytes` are now `bit_offset` and `bit_length`. They always
  counted bits, which the old names left to the reader to guess on a method
  that takes bytes.

* Assigning to a single index now takes only `True`, `False`, `0` or `1`, and
  raises `TypeError` for anything else, rather than testing the value for
  truthiness. This is the rule `append` and the implicit bit patterns already
  followed. Use `bool(x)` at the call site to keep the old behaviour.

Added

* Added the `Reader` class, which wraps a `Tibs` or `Mutibs` with a bit position
  for reading fields in sequence.

  ```python
  >>> r = Reader(Tibs('0x47ff10'))
  >>> r.read_value('u8')
  71
  >>> r.read_value('(bool, u7)')
  (True, 127)
  >>> r.remaining
  8
  ```

* Added `DtypeArray` and `DtypeTuple` alongside the scalar dtype (now
  `DtypeSingle`), so that a single `Dtype` can describe a structured,
  multi-field value: `Dtype("[u8; 4]")` for repeated fields and
  `Dtype("(u8, u16_le, bool)")` for differently-typed ones. Both nest to any
  depth.

  ```python
  >>> Dtype("(u8, [bool; 3])").pack((5, [True, False, True]))
  Tibs('0b00000101101')
  ```

* Added the `bf16` dtype for bfloat16, the non-IEEE 16-bit float that keeps the
  8-bit exponent of an `f32` and truncates the mantissa to 7 bits.

  ```python
  >>> Tibs.from_value("bf16", 1.0).hex
  '3f80'
  >>> Tibs("0x3f80").to_value("f16")
  1.875
  ```

* Added eleven fixed-width floating point dtypes: the draft P3109 `binary8p3`
  and `binary8p4` formats; OCP `ocp_e4m3` and `ocp_e5m2`, each with separate
  `_saturate` and `_overflow` packing policies; `ocp_e3m2`, `ocp_e2m3`,
  `ocp_e2m1`, `ocp_e8m0` and `ocp_int8`. Packing rounds directly from Python
  binary64, to nearest with ties to even. The P3109 formats are provisional and
  may be updated as the draft standard develops. These are raw scalar encodings
  only: shared scales and MX block behaviour are not yet included.

  ```python
  >>> Tibs.from_values("ocp_e2m1", [0.5, 1.0, 3.0, 6.0]).hex
  '1257'
  >>> Tibs.from_value("ocp_e4m3_overflow", 1000.0).hex
  'ff'
  ```

* A `DtypeKind` whose length is intrinsic — the eleven narrow formats above,
  plus `bool` and `bf16` — is now accepted anywhere a dtype is, and its length
  may be omitted from `DtypeSingle.from_params`.

  ```python
  >>> Tibs("0x1257").to_values(DtypeKind.OcpE2M1)
  [0.5, 1.0, 3.0, 6.0]
  >>> DtypeSingle.from_params(DtypeKind.OcpE2M1)
  DtypeSingle('ocp_e2m1')
  ```

* Added eight methods for comparing two containers of the same length without
  building an intermediate object: `count_and`, `count_or`, `count_xor`,
  `count_andnot`, `intersects`, `is_disjoint`, `is_subset_of` and
  `is_superset_of`. The counts are equivalent to `(a & b).count(1)` and so on,
  and `count_xor` is the Hamming distance. Skipping the intermediate makes them
  faster, increasingly so as the containers get longer, and the four predicates
  stop as soon as they know the answer.

  ```python
  >>> a, b = Tibs('0b1100'), Tibs('0b1010')
  >>> a.count_xor(b)      # the Hamming distance
  2
  >>> Tibs('0b1010').is_subset_of('0b1011')
  True
  >>> Tibs('0b1100').is_disjoint('0b0011')
  True
  ```

* The `value` parameter of `count` is now optional, and counts the set bits when
  it's not given, so `t.count()` is the same as `t.count(1)`.

* `count` now takes a `byte_aligned` parameter, so that a pattern can be counted
  on byte boundaries only, matching `find` and `find_all`.

  ```python
  >>> Tibs('0x1f2e3f').count('0x0f', mask='0x0f', byte_aligned=True)
  2
  ```

* Added `extracted`, `deposit` and `deposited` for reading and writing bit
  fields whose bits are scattered through a container by a mask, rather than
  being contiguous like the ones `field` handles. These are the bit-level
  equivalents of the x86 PEXT and PDEP instructions.

  ```python
  >>> Tibs('0b11010110').extracted('0b10110000')
  Tibs('0b101')
  >>> Tibs('0b11010110').deposited('0b111', '0b10110000').bin
  '11110110'
  ```

* Searches can now take a `mask`, so that patterns can contain don't-care bits:
  only the bits set in the mask need to match. It's available on `find`,
  `rfind`, `find_all`, `find_all_iter`, `rfind_all_iter`, `count`, `replace` and
  `replaced`, and combines with `start`, `end`, `count` and `byte_aligned` as
  you'd expect. For `replace` the mask affects only the matching, and the whole
  of each match is still replaced.

  ```python
  >>> Tibs('0x1f2e3f').find_all('0x0f', mask='0x0f', byte_aligned=True)
  [0, 16]
  ```

* The `u` and `i` interpretations no longer have a 128 bit limit. Any positive
  length now works, for `from_u` / `from_i`, the `u` and `i` properties and
  `to_u` / `to_i`, the `uN` / `iN` dtypes, views, and the `u` and `i` format
  codes.

  ```python
  >>> Tibs.from_u(2 ** 200 - 1, 200).u == 2 ** 200 - 1
  True
  ```

* Added `View.to_value`, `MutableView.to_value` and `MutableView.write_value`,
  so that any dtype is reachable through a view rather than only the nine
  interpretations that have their own view property. The view's byte order and
  bit order are applied first, so `view.to_value(dtype, start, end)` is always
  `view.to_tibs().to_value(dtype, start, end)`.

  ```python
  >>> m = Mutibs.from_bytes(bytes.fromhex("07 01 00 00 44 33 22 11"))
  >>> m.lsb0.le.field(63, 32).to_value("u32")
  287454020
  >>> m.lsb0.le.field(63, 32).write_value("u32", 0xdeadbeef)
  >>> m.hex
  '07010000efbeadde'
  ```

  Byte order is stated in one place only, so an `le`, `be` or `lsb0` view
  refuses a dtype that names a byte order, at any nesting depth. The plain
  `view()` claims no layout and still passes any dtype through.

* Added `__format__` to `Tibs`, `Mutibs`, `View` and `MutableView`, so that they
  can be used directly in f-strings and with `str.format()`. The type codes are
  `b`, `o`, `x` and `X` for the bit representations (equivalent to the `bin`,
  `oct` and `hex` properties, so leading zeros are kept), `u` and `i` for the
  unsigned and signed integer interpretations, and `e`, `f` and `g` with their
  uppercase forms for the float one, which needs a length of 16, 32 or 64. The
  `#` flag adds a prefix, and `_` groups the digits, with the group size taken
  from the otherwise unused precision field.

  ```python
  >>> f"{Tibs('0xac804f4b'):#_.2x}"
  '0xac_80_4f_4b'
  >>> pi = Tibs.from_f(3.14159, 32)
  >>> f"{pi:f}", f"{pi:.2f}", f"{pi:.3e}", f"{pi:g}"
  ('3.141590', '3.14', '3.142e+00', '3.14159')
  ```

  Fill, alignment and width work as they do elsewhere in Python, except that
  zero padding is rejected for `b`, `o`, `x` and `X`, as it would silently
  change the apparent length of the value. Groups are counted from bit zero, so
  it's the last group that comes up short rather than the first, and padding is
  never itself grouped.

* `Tibs` now exports the buffer protocol, so that it can be passed straight to
  anything that takes a bytes-like object — `memoryview(t)`, `array.array`,
  `numpy.frombuffer`, a socket or file `write`, and so on — without copying its
  data first.

  ```python
  >>> bytes(memoryview(Tibs('0xff00')))
  b'\xff\x00'
  ```

  The buffer is read-only and covers whole bytes, so a final partial byte
  includes its padding bits, unmasked. Exporting needs the underlying storage to
  start on a byte boundary; when it doesn't, a `BufferError` is raised and
  `to_bytes` or `to_padded_bytes` will give you an owned copy instead. This is
  deliberately not offered on `Mutibs`, whose storage moves as it is edited.

* `Tibs` and `Mutibs` can now be pickled, so that they can be sent through
  `multiprocessing`, stored in a cache, or deep copied with `copy.deepcopy`,
  which previously raised a `TypeError`. The pickled state is the `Codec.Raw`
  encoding, so any bit length is restored exactly and pickling costs about what
  copying costs; pickle the result of `encode()` yourself if you want the
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
  rather than taking turns. A `Tibs` is immutable and can be shared freely, with
  no lock taken at all. A `Mutibs` can be shared too: every method runs inside
  CPython's per-object critical section, which makes a single call atomic but
  not a sequence of calls, exactly as for a `list`. There's a new appendix in
  the docs covering this in full.

Changed

* Inverting an empty `Tibs` or `Mutibs` with `~` now returns an empty container
  instead of raising a `ValueError`, matching `Tibs.inverted` and
  `Mutibs.invert`.

Performance improvements

* Compound dtype packing and unpacking are much faster. A `DtypeTuple` operation
  equivalent to a `struct` call (`"(i16_be, i16_be, i32_be)"` against `">hhl"`)
  is now in the same ballpark as the handwritten `struct` code, where it was an
  order of magnitude slower when compound dtypes were first implemented.
* Parsed dtype specs are now cached, so that passing a string where a dtype is
  expected no longer re-parses it on every call. This mattered most for short
  operations, where parsing could dominate; passing a prebuilt `Dtype` is
  quicker still. Bulk calls were never parse-bound and are unchanged.
* A number of other operations — string/hex/oct/bin conversion, `from_values`,
  `Mutibs.reverse` and others — moved from bit-at-a-time to byte-at-a-time or
  whole-buffer implementations. See `tests/performance_comparison.py` for how
  tibs measures up against bitarray and against equivalent standard library
  code.


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
