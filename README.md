[![tibs](https://raw.githubusercontent.com/scott-griffiths/tibs/main/doc/tibs.png)](https://github.com/scott-griffiths/tibs)

A sleek Python library for your binary data

[![PyPI - Version](https://img.shields.io/pypi/v/tibs?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/tibs/)
[![CI badge](https://github.com/scott-griffiths/tibs/actions/workflows/.github/workflows/test.yml/badge.svg)](https://github.com/scott-griffiths/tibs/actions/workflows/test.yml)
![PyPI - License](https://img.shields.io/pypi/l/tibs)
[![Docs](https://img.shields.io/readthedocs/mutibs?logo=readthedocs&logoColor=white)](https://mutibs.readthedocs.io/en/latest/)

----

> [!NOTE]
> This library is currently pre-alpha. This documentation is part reality and part planning.

## Documentation

The API docs are available [here](https://mutibs.readthedocs.io/en/latest/).

## Why is it called tibs?

Well it's 'bits' backwards (more or less) and the name was available!

## The basics

```python
from tibs import Tibs, Mutibs
```

The `Tibs` class is an immutable container of binary data. You can create a `Tibs` from binary or hex strings,
byte data, format strings etc. A number of creation methods are provided, all of which start with `from_`:

```python
>> > a = Tibs.from_string('0b110')
>> > b = Tibs.from_zeros(16)
>> > c = Tibs.from_bytes(b'some_bytes')
>> > d = Tibs.from_joined([a, b, c])
>> > e = Tibs.from_random(1000)
>> > f = Tibs.from_joined([a, b, c, d, e])
>> > g = Tibs.from_bools([1, 0, 0])
```

The `__init__` method redirects to `from_string`, `from_bytes`, or `from_bools` as appropriate,
so above you could also say `a = Tibs('0b110')`, `c = Tibs(b'some_bytes')` or `g = Tibs([1, 0 0])` which is often more
convenient.

The `Mutibs` class (pronounced 'mew-tibs') is a mutable version of `Tibs`.