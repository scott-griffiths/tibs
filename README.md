<p>
  <img src="https://raw.githubusercontent.com/scott-griffiths/tibs/main/doc/tibs_white.png" alt="Tibs cat" height="130" align="left" />
  <a href="https://github.com/scott-griffiths/tibs">
    <img src="https://raw.githubusercontent.com/scott-griffiths/tibs/main/doc/tibs.png" alt="tibs" height="110" />
  </a><br />
  A sleek Python library for binary data
</p>

<br clear="left" />


[![PyPI - Version](https://img.shields.io/pypi/v/tibs?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/tibs/)
[![CI badge](https://github.com/scott-griffiths/tibs/actions/workflows/new_ci.yaml/badge.svg)](https://github.com/scott-griffiths/tibs/actions/workflows/new_ci.yaml)
[![Docs](https://img.shields.io/readthedocs/mutibs?logo=readthedocs&logoColor=white)](https://mutibs.readthedocs.io/en/latest/)
![PyPI - License](https://img.shields.io/pypi/l/tibs)
&nbsp; &nbsp;
[![Pepy Total Downlods](https://img.shields.io/pepy/dt/tibs?logo=python&logoColor=white&labelColor=blue&color=blue)](https://www.pepy.tech/projects/tibs)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/tibs?label=%40&logoColor=white&labelColor=blue&color=blue)](https://pypistats.org/packages/tibs)


----

``tibs`` is a simple but powerful Python library for creating, interpreting and manipulating binary data.
It is 100% written in Rust to give it excellent performance, and is from the same author as
the [bitstring](https://github.com/scott-griffiths/bitstring) library.

# Documentation

The full documentation is available on [Read the Docs](https://mutibs.readthedocs.io/en/latest/).

## Getting started

To install use

```
pip install tibs
```

There are pre-built wheels for most configurations - if there are issues then please let me know.
Tibs works with Python 3.8 and later.


## A taster

One way to get to know the library is to start a Python interactive session and try a
small binary record. `Tibs` is immutable, like `bytes`; `Mutibs` is the mutable version
for in-place editing.

```pycon
>>> from tibs import Tibs, Mutibs

>>> # Four flag bits, a 12-bit integer field, then two payload bytes.
>>> packet = Tibs.from_joined(["0b1010", Tibs.from_u(3200, 12), b"OK"])
>>> packet
Tibs('0xac804f4b')
>>> len(packet)
32

>>> # Slice at bit positions, then interpret each slice however you need.
>>> packet[:4].bin
'1010'
>>> packet[4:16].to_u()
3200
>>> packet[16:].bytes
b'OK'

>>> # Search and test using strings, bytes, booleans or other Tibs values.
>>> packet.find("0x4f", byte_aligned=True)
16
>>> packet.find_all("0b10")
[0, 2, 5, 8, 17, 23, 25, 28]

>>> # Convert to Mutibs when you want to patch the data in-place.
>>> patched = packet.to_mutibs()
>>> patched[4:16] = Tibs.from_u(2047, 12)
>>> patched[-8:] = b"!"
>>> patched
Mutibs('0xa7ff4f21')
>>> patched[4:16].to_u(), patched[-16:].bytes
(2047, b'O!')

>>> # The same operations are designed to scale to large bit sequences.
>>> Tibs.from_random(1_000_000, seed=b"readme").count(1)
500480
```

This only scratches the surface: the docs cover construction from ints, floats, bytes
and strings; endianness; searching and replacing; rotations; bit indexing; and more
worked examples.

For more information see the full [documentation](https://mutibs.readthedocs.io/en/latest/).

# Credits

The `tibs` library was created by Scott Griffiths and is released under the MIT License.

The Tibs cat artwork was created by Ada Griffiths and is not covered by the software license. All rights reserved.
