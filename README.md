[![tibs](https://raw.githubusercontent.com/scott-griffiths/tibs/main/doc/tibs.png)](https://github.com/scott-griffiths/tibs)

A streamlined Python bit manipulation library written in Rust.

[![PyPI - Version](https://img.shields.io/pypi/v/tibs?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/tibs/)
[![CI badge](https://github.com/scott-griffiths/tibs/actions/workflows/.github/workflows/test.yml/badge.svg)](https://github.com/scott-griffiths/tibs/actions/workflows/test.yml)
![PyPI - License](https://img.shields.io/pypi/l/tibs)

[//]: # ([![Docs]&#40;https://img.shields.io/readthedocs/tibs?logo=readthedocs&logoColor=white&#41;]&#40;https://tibs.readthedocs.io/en/latest/&#41;)

----

> [!NOTE]
> This library is currently pre-alpha. This documentation is part reality and part planning.

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
```

The `__init__` method redirects to `from_string`, so above you could also say `a = Tibs('0b110')`, which is often more
convenient. The string can be formatted to do more than just binary, for example:

```python
>> > g = Tibs('u14=78')  # A 14-bit unsigned int with value 78
>> > h = Tibs('f16=-0.25')  # A 16-bit IEEE float
>> > j = Tibs('0o777')  # 9 bits from octal
```

The `Mutibs` class (pronounced 'mew-tibs') is a mutable version of `Tibs`.