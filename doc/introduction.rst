.. currentmodule:: bitformat
.. _introduction:

Getting Started
===============

Working with binary data in Python can be straight-forward. The built-in `bytes <https://docs.python.org/3/library/stdtypes.html#bytes>`_ and `bytearray <https://docs.python.org/3/library/stdtypes.html#bytearray>`_ types can be used to store binary data, and the `struct <https://docs.python.org/3/library/struct.html#module-struct>`_  and `array <https://docs.python.org/3/library/array.html#module-array>`_ modules can be used to pack and unpack binary data into Python objects.
If all you need to do is simple manipulation of whole byte binary data then these tools are usually sufficient.

Sometimes you might need to deal with things that are not a whole number of bytes long, or use binary formats beyond those that can be represented by the ``struct`` module.
There are a few third-party packages that can help with this, for example `bitstring <https://pypi.org/project/bitstring/>`_ and `bitarray <https://pypi.org/project/bitarray/>`_ are well established libraries for working with arbitrary length binary data. For dealing with binary formats the `construct <https://pypi.org/project/construct/>`_ library is well regarded.
These libraries all have their strengths and weaknesses, and I am personally well aware of the good and bad points of `bitstring` in particular as I have been writing and maintaining it since 2006.

The `bitformat` library has the lofty ambition to be as expressive as `bitstring`, as performant as `bitarray`, and as powerful as `construct`.
I don't know if it will succeed, but the beta versions are already very usable. I'm happy for feedback from any early adopters, but if you prefer to use a more mature library then any of the above are good choices.

This introduction will start with a brief tour of the main features of `bitformat`, followed by a more in depth look at the main classes.
For more exhaustive documentation see the :ref:`API reference<api>`.

Installation and download
-------------------------

To install the module, use pip::

    pip install bitformat

There are pre-built wheels for Windows, Linux and MacOS, and more will be added later. It is compatible with Python 3.11 and later.

To download the library, as well as for defect reports, enhancement requests and Git repository browsing go to `the project's home on GitHub. <https://github.com/scott-griffiths/bitformat/>`_


.. toctree::
    :maxdepth: 1
    :hidden:

    tldr
    bitstring_comparison
