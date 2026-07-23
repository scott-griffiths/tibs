<p>
  <img src="https://raw.githubusercontent.com/scott-griffiths/tibs/main/doc/_static/tibs_cat.png" alt="Tibs cat" height="130" align="left" />
  <a href="https://github.com/scott-griffiths/tibs">
    <img src="https://raw.githubusercontent.com/scott-griffiths/tibs/main/doc/tibs.png" alt="tibs" height="110" />
  </a><br />
  A sleek Python library for binary data
</p>

<br clear="left" />


[![PyPI - Version](https://img.shields.io/pypi/v/tibs?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/tibs/)
[![CI badge](https://github.com/scott-griffiths/tibs/actions/workflows/new_ci.yaml/badge.svg)](https://github.com/scott-griffiths/tibs/actions/workflows/new_ci.yaml)
[![Docs](https://img.shields.io/readthedocs/tibs?logo=readthedocs&logoColor=white)](https://tibs.readthedocs.io/en/latest/)
![PyPI - License](https://img.shields.io/pypi/l/tibs)
&nbsp; &nbsp;
[![Pepy Total Downloads](https://img.shields.io/pepy/dt/tibs?logo=python&logoColor=white&labelColor=blue&color=blue)](https://www.pepy.tech/projects/tibs)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/tibs?label=%40&logoColor=white&labelColor=blue&color=blue)](https://pypistats.org/packages/tibs)


----

``tibs`` is a Rust-backed Python library for binary data that does not assume
everything fits neatly into bytes. Use it for packets, registers, instruction
formats, bitsets, compressed data and streams where fields can have many different
interpretations and be any number of bits long.

It is used to power the popular [bitstring](https://github.com/scott-griffiths/bitstring)
library, which is by the same author.

## Install

```bash
pip install tibs
```

Tibs works with Python 3.10 and later. There are pre-built wheels for most
common platforms; if there are issues then please let me know.

The full documentation is available on [Read the Docs](https://tibs.readthedocs.io/en/latest/).


## Why use it?

Tibs looks at binary data three ways, and the same object does all three.

**A sequence of bits.** Like `bytes`, but the unit is the bit and the length can be
anything. Slice, concatenate, search and replace at bit granularity — or pin
searches to byte boundaries for stream parsing. Immutable `Tibs` gives stable
values and cheap slices; `Mutibs` gives in-place edits, just as `bytes` pairs with
`bytearray`.

> `find` · `rfind` · `find_all` · `replace` · `count` · `starts_with` · `split_at` · `in`

**Typed fields.** Read and write integers, floats and strings of any bit length,
without hand-rolling shifts and masks. Views handle little-endian ordering and
LSB0 field labels so you don't reshuffle data yourself.

> `from_u` · `to_f` · `Dtype` · `pack` / `unpack` · `.le` · `.lsb0` · `field()` · f-string formatting

**A set of bits.** Bitwise algebra, cardinalities and set predicates, with no
intermediate object built along the way. `Mutibs` doubles as a large mutable
bitset.

> `&` `|` `^` `~` · `count_and` · `count_xor` · `intersects` · `is_subset_of` · `set` / `unset` · `all` / `any`

And it's fast — often significantly faster than similar libraries, with a Rust
core and a performance regression suite in CI.


## Quick taste

The same three views, in code.

**A sequence of bits.** `Tibs` is immutable, like `bytes` — good for parsing,
slicing, hashing and holding stable values. `Mutibs` is its mutable counterpart,
for patching in place.

```pycon
>>> from tibs import Tibs
>>> # Four flag bits, a 12-bit integer, then two payload bytes.
>>> packet = Tibs.from_joined(["0b1010", Tibs.from_u(3200, 12), b"OK"])
>>> packet
Tibs('0xac804f4b')

>>> flags, size, payload = packet.split_at([4, 16])
>>> flags.bin, size.u, payload.bytes
('1010', 3200, b'OK')

>>> patched = packet.to_mutibs()
>>> patched[4:16] = Tibs.from_u(2047, 12)
>>> patched[-8:] = b"!"
>>> patched
Mutibs('0xa7ff4f21')
>>> packet  # the original is untouched
Tibs('0xac804f4b')

```

**Typed fields.** Read fields as integers, floats or bytes, letting a view take
care of byte order and bit numbering — the sort of job that gets awkward quickly
with plain bytes and masks.

```pycon
>>> # Linux eBPF: little-endian instruction, LSB0 field labels.
>>> instruction = Tibs.from_bytes(bytes.fromhex("07 01 00 00 44 33 22 11")).lsb0.le
>>> dst_reg = instruction.field(11, 8).u
>>> f"r{dst_reg} += {instruction.field(63, 32):#x}"
'r1 += 0x11223344'

```

**A set of bits.** Compare containers as sets — cardinalities and predicates that
never build an intermediate object.

```pycon
>>> # Are all the required capability bits granted?
>>> required, granted = Tibs('0b0010_1000'), Tibs('0b1010_1100')
>>> required.is_subset_of(granted)
True

>>> # Hamming distance between two 256-bit fingerprints.
>>> a, b = Tibs.from_random(256, seed=b'doc-a'), Tibs.from_random(256, seed=b'doc-b')
>>> a.count_xor(b)
138

```

These aren't separate modes — the same object does all three. This log scanner
*searches* a byte stream, slices out each header, and reads its fields as
integers, in one pass:

```pycon
>>> sync = Tibs("0xaa55")
>>> log = bytes.fromhex("00 ff aa 55 11 02 c0 01 7f aa 55 22 01 40 aa")
>>> bits = Tibs(log)
>>> records = []
>>> for start in bits.find_all_iter(sync, byte_aligned=True):
...     payload_len = bits[start + 24:start + 32].u
...     payload = bits[start + 32:start + 32 + payload_len * 8]
...     if len(payload) == payload_len * 8:
...         records.append((bits[start + 16:start + 24].u, payload.bytes))
>>> records
[(17, b'\xc0\x01'), (34, b'@')]

```

The full documentation covers construction from ints, floats, bytes and strings;
endianness; searching and replacing; rotations; bit indexing; serialization; and
worked examples.

## Performance

Tibs is written in Rust with PyO3. The repository contains a dedicated
[performance regression suite](tests/performance_regression.py) and
[CI workflow](.github/workflows/performance.yaml) that compare benchmark
medians against the base commit.

The regression benchmarks cover operations such as:

- bit-pattern search, reverse search and byte-aligned search
- logical operations on large aligned and unaligned slices
- counting, inversion, shifting, reversing and concatenation
- bulk mutation, slice replacement, deletion, append and pop
- bytes conversion, random generation and bool-list construction
- typed packing and unpacking such as `u16` value streams
- a sieve workload using `Mutibs` as a large mutable bitset

For local comparisons, [`tests/performance_comparison_new.py`](tests/performance_comparison_new.py)
checks common operations against `bitarray`. With
`bitarray` installed, run:

```bash
python tests/performance_comparison_new.py --bytes 250000 --values 20000 --repeats 5
```

Benchmarks are machine-dependent, but the mean speedup over bitarray will typically be > 2x.

## Examples

The runnable examples in [`examples/`](examples/) are small, but they are meant
to look like real binary-data tasks:

| Example | Shows |
| --- | --- |
| [`log_scan.py`](examples/log_scan.py) | Find byte-aligned sync markers and pull records from a stream. |
| [`little_endian_registers.py`](examples/little_endian_registers.py) | Decode and rebuild little-endian register dumps with `u16_le`. |
| [`ebpf_instruction.py`](examples/ebpf_instruction.py) | Decode LSB0, little-endian instruction fields. |
| [`instruction_scan.py`](examples/instruction_scan.py) | Search for an opcode with `mask`, ignoring the register fields. |
| [`scattered_field.py`](examples/scattered_field.py) | Read and write a register field split around status bits with `extract`/`deposit`. |
| [`fingerprints.py`](examples/fingerprints.py) | Compare items as sets of bits with `count_and`, `count_xor` and `is_subset_of`. |
| [`sensor_samples.py`](examples/sensor_samples.py) | Pack and unpack 12-bit ADC samples. |
| [`patch_config.py`](examples/patch_config.py) | Patch compact config fields in place with `Mutibs`. |
| [`construct.py`](examples/construct.py) | Build and unpack a structured MPEG-style header. |
| [`sieve.py`](examples/sieve.py) | Use a large mutable bitset for a prime sieve. |


## Project status

Tibs has reached the 1.0 stable API milestone. Documented public behavior will
remain compatible across future 1.x releases. It is used to power the `bitstring` 
library and gets several million downloads per month.

There are over 1000 unit tests, including Hypothesis tests and performance
benchmarks.


For the full API reference, see the
[documentation](https://tibs.readthedocs.io/en/latest/).


## Credits

The `tibs` library was created by Scott Griffiths and is released under the MIT License.

The Tibs cat artwork was created by Ada Griffiths and is not covered by the software license. All rights reserved.

<p>
  <img src="https://raw.githubusercontent.com/scott-griffiths/tibs/main/doc/_static/tibs_white_sleeping.png" alt="Tibs cat" height="110" align="left" />
</p>
