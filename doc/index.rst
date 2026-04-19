.. currentmodule:: tibs

.. toctree::
    :maxdepth: 2
    :hidden:

    self
    manual
    examples
    api


.. raw:: html

   <div style="display: flex; justify-content: left; margin: 0 0 1rem 0;">
     <div style="display: flex; align-items: flex-end; gap: 1rem;">
       <img src="_static/tibs_white.png" alt="Tibs" class="only-light" style="width: 95px; height: auto;"/>
       <img src="_static/tibs_white_sleeping.png" alt="Tibs" class="only-dark pst-js-only" style="width: 130px; height: auto;"/>
       <div style="display: flex; flex-direction: column; align-items: center;">
         <img src="_static/tibs.png" alt="tibs" style="width: 240px; height: auto;"/>
         <div>A sleek Python library for binary data</div>
       </div>
     </div>
   </div>

Overview
--------

**tibs** is a simple but powerful Python library for creating, interpreting and manipulating binary data.
It is written in Rust, and from the same author as the `bitstring <https://github.com/scott-griffiths/bitstring>`_ library.

Getting started
^^^^^^^^^^^^^^^

To install use ::

    pip install tibs


There are pre-built wheels for most configurations - if there are issues then please let me know.
Tibs works with Python 3.8 and later.

Here's a very brief taster of what tibs looks like in an interactive Python session.
This is all explained in the documentation and API docs that follow. ::


    >>> from tibs import Tibs, Mutibs
    >>> t = Tibs('0x0f3')
    >>> len(t)
    12
    >>> t = [1, 1, 0, 1] + t
    >>> t.to_bin()
    '1101000011110011'
    >>> t.to_u()
    53491
    >>> t.to_f()
    -39.59375
    >>> list(t.find_all([1, 0]))
    [1, 3, 11]
    >>> t.replaced([1, 1, 0], [0, 0])
    Tibs('0b00100001100011')
    >>> Tibs.from_random(1_000_000_000).count(1)
    499997660



----


Credits
^^^^^^^

``tibs`` was created by Scott Griffiths and is released under the MIT License.

The Tibs cat artwork was created by Ada Griffiths and is not covered by the software license.
All rights reserved.
