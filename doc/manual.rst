.. currentmodule:: tibs


.. raw:: html

   <div style="display: flex; justify-content: left; margin: 0 0 1rem 0;">
     <div style="display: flex; align-items: flex-end; gap: 1rem;">
       <img src="_static/tibs_cat.png" alt="Tibs" style="width: 95px; height: auto;"/>
       <div style="display: flex; flex-direction: column; align-items: center;">
         <img src="_static/tibs.png" alt="tibs" style="width: 240px; height: auto;"/>
         <div>A sleek Python library for binary data</div>
       </div>
     </div>
   </div>

|github| |license| |pepy-downloads| |monthly-downloads|


.. |github| image:: https://img.shields.io/static/v1?label=&message=GitHub&logo=github&logoColor=white&labelColor=blue&color=blue
   :target: https://github.com/scott-griffiths/tibs
   :alt: GitHub

.. |license| image:: https://img.shields.io/pypi/l/tibs?labelColor=blue&color=blue
   :alt: PyPI - License

.. |pepy-downloads| image:: https://img.shields.io/pepy/dt/tibs?logo=python&logoColor=white&labelColor=blue&color=blue
   :target: https://www.pepy.tech/projects/tibs
   :alt: Pepy Total Downloads

.. |monthly-downloads| image:: https://img.shields.io/pypi/dm/tibs?label=%40&logoColor=white&labelColor=blue&color=blue
   :target: https://pypistats.org/packages/tibs
   :alt: PyPI - Downloads

User Manual
-----------

``tibs`` is a Python library for creating, manipulating and interpreting binary data.
It's designed to be light-weight and easy to use, and is written in Rust for efficiency.

The two most important classes available in the tibs module are:

* :doc:`tibs` — An immutable sequence of bits ('tibs' is 'bits' rearranged).
* :doc:`mutibs` — A mutable sequence of bits ('mutibs' is a mutable tibs).

These classes efficiently hold arbitrary length binary data; they can be constructed
from bytes, ints, floats, binary and hex strings, random data, and bools. They can then be
sliced, spliced, searched, replaced, rotated, reversed, byte-swapped, set, unset,
appended, extended, indexed, counted, concatenated, chunked, and joined back together,
before being interpreted as bytes, ints, floats, binary and hex strings, and bools.


Getting started
^^^^^^^^^^^^^^^

To install use ::

    pip install tibs


There are pre-built wheels for most configurations - if there are issues then please let me know.
Tibs works with Python 3.10 and later.


The rest of this manual will talk about all the major features of the library, but will not
be exhaustive - see the API docs for every method and parameter.



.. toctree::
    :maxdepth: 1
    :hidden:

    creation
    inspection
    manipulation
    views
    tibs_vs_mutibs
    credits
