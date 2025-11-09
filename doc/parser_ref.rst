.. currentmodule:: bitformat
.. _parser_ref:

Parser Reference
-----------------

Data types, fields and formats can all be constructed from strings; in fact this is how the default constructors for these types work - taking a string and parsing it to create the right instance.

The :class:`Dtype` and :class:`FieldType` base classes also return the correct type based on the parsed string, for example ``Dtype('[u4; 10]')`` will give a :class:`DtypeArray` instance, and ``FieldType('a: f32')`` will return a :class:`Field`.

All of these types can be created more explicitly using their ``from_params`` methods, but this quickly becomes unwieldly, especially for things like formats. They are still very useful if you need to create data types and fields programmatically though.

The source of truth for the string parsing is contained in the `Lark grammar file <https://github.com/scott-griffiths/bitformat/blob/main/bitformat/bitformat_grammar.lark>`_. A less exact, but easier to read summary is given in the table below. Optional elements are indicated with angle brackets ``<like_this>``.

.. list-table::
    :widths: 25 75
    :header-rows: 1

    * - Class
      - String Syntax
    * - :class:`Expression`
      - ``{`` ``<code_str>`` ``}`` or ``an_integer``
    * -
      - e.g. ``'{x + 1}'``, ``'{size < 0}'``, ``'6'``
    * - :class:`DtypeKind`
      - ``u`` | ``i`` | ``f`` | ``bin`` | ``oct`` | ``hex`` | ``bytes`` | ``bool`` | ``bits`` | ``pad``
    * -
      - e.g. ``'bin'``, ``'u'``
    * - :class:`DtypeSingle` (Dtype)
      - ``DtypeKind`` ``<size_expression>`` ``<_modifier>``
    * -
      - e.g. ``'u8'``, ``'bool'``, ``'f16_le'``, ``'bytes{n + 1}'``
    * - :class:`DtypeArray` (Dtype)
      - ``[`` ``DtypeSingle`` ``;`` ``<items_expression>`` ``]``
    * -
      - e.g. ``'[i4; 15]'``, ``'[bytes;]'``, ``'[f32; {v + 1}]'``
    * - :class:`DtypeTuple` (Dtype)
      - ``(`` ``Dtype1`` ``,`` ``...`` ``)``
    * -
      - e.g. ``'(bool, u7)'``, ``'([u8; 5], (bool, bool))'``
    * - :class:`Field` (FieldType)
      - ``<name_str:>`` ``Dtype``
    * -
      - e.g. ``'flag: bool'``, ``'u16'``
    * - :class:`Format` (FieldType)
      - ``<name_str:>`` ``(`` ``FieldType1`` ``,`` ``...`` ``)``
    * -
      - e.g. ``'(u8, bytes)'``, ``'h: (x: u8, y: [bytes; {x}])'``
    * - :class:`If` / Else (FieldType)
      - ``if`` ``Expression`` ``:`` ``FieldType1`` ``<else: FieldType2>``
    * -
      - e.g. ``'if {x > 0}: [u8; 4]'``
    * - :class:`Repeat` (FieldType)
      - ``repeat`` ``Expression`` ``:`` ``FieldType``
    * -
      - e.g. ``'repeat {x + 1}: u8'``, ``repeat 10: (bool, f16)``
    * - :class:`Pass` (FieldType)
      - ``pass``


