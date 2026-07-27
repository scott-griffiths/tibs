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

``tibs`` is a Python library for binary data that does not assume
everything fits neatly into bytes. It's 100% written in Rust and has excellent performance.

Use it for packets, registers, instruction
formats, bitsets, compressed data and streams where fields can have many different
interpretations and be any number of bits long.

It is used to power the popular [bitstring](https://github.com/scott-griffiths/bitstring)
library, which is by the same author.

## Install

```bash
pip install tibs
```

Tibs works with Python 3.11 and later. There are pre-built wheels for most
common platforms; if there are issues then please let me know.

The full documentation is available on [Read the Docs](https://tibs.readthedocs.io/en/latest/).


## Overview

The tibs library provides two main classes: `Tibs`, which is an immutable sequence of bits
(similar to how `bytes` works in Python as a sequence of bytes) and `Mutibs`, which is a mutable version (similar to `bytearray` in Python).

A `Tibs` can be thought of as just a sequence of bits. It provides an interface very similar
to `bytes` and other Python containers - you can slice it, concatenate, search it etc. in a
familiar way, with `Mutibs` adding on mutating methods.

> `find` · `rfind` · `find_all` · `replace` · `count` · `starts_with` · `split_at` · `chunks` · `+` · `in`

This 'container of bits' mental model might be all that you need, but the library also gives
you two broad views of the binary data.

1. **As typed fields.** Pull integers, floats, strings, hex or binary of any
bit length straight out of the bits, without hand-rolling shifts and masks. Little-endian ordering and LSB0 field labels are handled elegantly so you don't reshuffle data
yourself, and `extract` / `deposit` reach fields that are scattered across a word.

> `from_u` · `to_f` · `bin` / `hex` · `Dtype` · `pack` / `unpack` · `.le` · `.lsb0` · `field()` · `extract` / `deposit` · f-string formatting

2. **As a set of bits.** Bitwise algebra, cardinalities and set predicates, with
no intermediate object built along the way. `Mutibs` can be used as a large mutable bitset.

> `&` `|` `^` `~` · `count_and` · `count_xor` · `intersects` · `is_subset_of` · `set` / `unset` · `all` / `any`

These aren't separate modes or separate types — it's one object, and the lenses are
just different questions you ask of the same bits.

And it's fast — usually significantly faster than similar libraries, 100% written in Rust
and with a large emphasis on performance.


## A Taster

Some real code to illustrate.

**As a sequence of bits.** `Tibs` works like `bytes`, except that the unit is the bit instead of the byte. `Mutibs` is its mutable counterpart, for patching in place.

```pycon
>>> from tibs import Tibs
>>> # A 5-bit header, a message, then 3 bits of padding: nothing is byte aligned.
>>> frame = Tibs('0b10110') + b'the cat rarely blinked' + [0, 0, 0]
>>> bytes(frame).find(b'cat')      # as bytes, the message has been scrambled
-1
>>> pos = frame.find(b'cat')       # but the tibs still knows where it is
>>> pos, frame[pos:pos + 24].bytes
(37, b'cat')

>>> patched = frame.to_mutibs()
>>> patched[pos:pos + 24] = b'squirrel'
>>> patched[5:-3].bytes
b'the squirrel rarely blinked'
>>> len(frame), len(patched)       # 40 bits longer, spliced in at bit 37
(184, 224)

```

**As typed fields.** Read and write integers, floats and strings of any bit length,
with a view taking care of byte order and bit numbering — the sort of job that gets
awkward quickly with plain bytes and masks.

```pycon
>>> # What's inside a float? A sign bit, an 8-bit exponent and a 23-bit fraction.
>>> x = Tibs.from_f(-118.625, 32)
>>> f"{x:_.8b}"                    # grouped into bytes to make it readable
'11000010_11101101_01000000_00000000'
>>> sign, exponent, fraction = x.split_at([1, 9])
>>> (-1) ** sign.u * 2 ** (exponent.u - 127) * (1 + fraction.u / 2 ** 23)
-118.625

>>> Tibs(b'\x00\x40\xed\xc2').le.f     # the same value, from a little-endian file
-118.625
>>> Tibs.from_u(x.u + 1, 32).f         # the adjacent float32, one bit away
-118.62500762939453

```

**As a set of bits.** Bitwise algebra and cardinalities over millions of bits, without
building an intermediate object just to count it.

```pycon
>>> from math import isqrt
>>> from tibs import Mutibs
>>> # A sieve of Eratosthenes over ten million numbers, one bit each.
>>> limit = 10_000_000
>>> sieve = Mutibs.from_ones(limit)
>>> sieve.unset([0, 1])
>>> for p in range(2, isqrt(limit) + 1):
...     if sieve[p]:
...         sieve.unset(range(p * p, limit, p))
...
>>> sieve.count(1)                     # primes below ten million
664579

>>> # Counting twin, cousin and sexy primes: pairs 2, 4 and 6 apart:
>>> [sieve.count_and(sieve >> d) for d in (2, 4, 6)]
[58980, 58622, 117207]

```


The full documentation covers construction from ints, floats, bytes and strings;
endianness; searching and replacing; rotations; bit indexing; serialization; views; dtypes and
much more.

## Performance

Tibs is written in Rust with PyO3. The repository contains a dedicated
[performance regression suite](tests/performance_regression.py) and
[CI workflow](.github/workflows/performance.yaml) that compare benchmark
medians against the base commit.

For local comparisons, [`tests/performance_comparison.py`](tests/performance_comparison.py)
checks common operations against the `bitarray` library and the standard Python library. With
`bitarray` installed, run:

```bash
python tests/performance_comparison.py
```

Benchmarks are machine-dependent, but tibs is often almost unreasonably fast.

## Examples

The runnable examples in [`examples/`](examples/) are small, but they are meant
to look like real binary-data tasks. They're grouped by the view each one leans
on most, though several use more than one.

**The bits as a sequence** — searching a stream and slicing fields out of it.

| Example | Shows |
| --- | --- |
| [`log_scan.py`](examples/log_scan.py) | Find byte-aligned sync markers and pull records from a stream. |
| [`instruction_scan.py`](examples/instruction_scan.py) | Search for an opcode with `mask`, ignoring the register fields. |
| [`patch_config.py`](examples/patch_config.py) | Patch compact config fields in place with `Mutibs`. |

**Typed fields and views** — numeric fields, with byte order and bit labels handled by a view.

| Example | Shows |
| --- | --- |
| [`construct.py`](examples/construct.py) | Build and unpack a structured MPEG-style header. |
| [`sensor_samples.py`](examples/sensor_samples.py) | Pack and unpack 12-bit ADC samples. |
| [`little_endian_registers.py`](examples/little_endian_registers.py) | Decode and rebuild little-endian register dumps with `u16_le`. |
| [`ebpf_instruction.py`](examples/ebpf_instruction.py) | Decode LSB0, little-endian instruction fields. |
| [`scattered_field.py`](examples/scattered_field.py) | Read and write a register field split around status bits with `extract`/`deposit`. |

**Sets of bits** — bitwise algebra and comparison.

| Example | Shows |
| --- | --- |
| [`sieve.py`](examples/sieve.py) | Use a large mutable bitset for a prime sieve. |
| [`fingerprints.py`](examples/fingerprints.py) | Compare items as sets of bits with `count_and`, `count_xor` and `is_subset_of`. |


## Project status

Tibs has passed the 1.0 stable API milestone. Documented public behavior will
remain compatible across future 1.x releases. It is already used to power the `bitstring` 
library and gets several million downloads per month.

There are thousands of unit tests, including Hypothesis tests and performance
benchmarks.


For the full API reference, see the
[documentation](https://tibs.readthedocs.io/en/latest/).


## Credits

The `tibs` library was created by Scott Griffiths and is released under the MIT License.

The Tibs cat artwork was created by Ada Griffiths and is not covered by the software license. All rights reserved.

<p>
  <img src="https://raw.githubusercontent.com/scott-griffiths/tibs/main/doc/_static/tibs_white_sleeping.png" alt="Tibs cat" height="110" align="left" />
</p>
