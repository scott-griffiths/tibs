from __future__ import annotations

import struct
from typing import Literal
from bitformat._common import DtypeKind
from ._dtypes import DtypeDefinition, AllowedSizes
from ._bits import Bits
from .rust import bits_from_any


# The definitions for each of the data types.
# Some of these definitions use a method to convert a bit length to the number of characters needed to print it.

# ----- Integer types -----

def to_u(bs: Bits, start: int, length: int) -> int:
    """Return data as an unsigned int from a slice of the bits."""
    assert start >= 0
    assert length >= 0
    if length == 0:
        raise ValueError("Cannot interpret empty Bits as an integer.")
    if length <= 64:
        return bs._to_u64(start, length)
    else:
        # Longer stores are unlikely in practice - this method is slower.
        bs = bs._getslice(start, length)
        return int.from_bytes(bs._to_int_byte_data(False), byteorder="big", signed=False)

def from_u(u: int, length: int) -> Bits:
    if length == 0:
        raise ValueError("A non-zero length must be specified with a 'u' initialiser.")
    u = int(u)
    if u < 0:
        raise ValueError(f"Unsigned integers cannot be initialised with the negative number {u}.")
    if u >= (1 << length):
        raise ValueError(f"{u} is too large an unsigned integer for a bit length of {length}. "
                         f"The allowed range is[0, {(1 << length) - 1}].")
    if length <= 64:
        return Bits._from_u64(u, length)
    else:
        b = u.to_bytes((length + 7) // 8, byteorder="big", signed=False)
        offset = 8 - (length % 8)
        if offset == 8:
            return Bits.from_bytes(b)
        else:
            return Bits._from_bytes_with_offset(b, offset=offset)

def u_bits2chars(bit_length: int) -> int:
    # How many characters is largest possible int of this length?
    return len(str((1 << bit_length) - 1))


u_defn = DtypeDefinition(DtypeKind.UINT,
                         "a two's complement unsigned int",
                         "unsigned int",
                         AllowedSizes(1, None),
                         from_u,
                         to_u,
                         int,
                         False,
                         u_bits2chars,
                         endianness_variants=True)


def to_i(bs: Bits, start: int, length: int) -> int:
    """Return data as a signed int from a slice of the bits."""
    if length == 0:
        raise ValueError("Cannot interpret empty Bits as an integer.")
    if length <= 64:
        return bs._to_i64(start, length)
    else:
        # Longer store are unlikely in practice - this method is slower.
        bs = bs._getslice(start, length)
        return int.from_bytes(bs._to_int_byte_data(True), byteorder="big", signed=True)

def from_i(i: int, length: int) -> Bits:
    if length == 0:
        raise ValueError("A non-zero length must be specified with an 'i' initialiser.")
    i = int(i)
    if i >= (1 << (length - 1)) or i < -(1 << (length - 1)):
        raise ValueError(f"{i} is too large a signed integer for a bit length of {length}. "
                         f"The allowed range is[{-(1 << (length - 1))}, {(1 << (length - 1)) - 1}")
    if length < 64:
        # Faster method for shorter lengths.
        return Bits._from_i64(i, length)
    else:
        b = i.to_bytes((length + 7) // 8, byteorder="big", signed=True)
        offset = 8 - (length % 8)
        if offset == 8:
            return Bits.from_bytes(b)
        else:
            return Bits._from_bytes_with_offset(b, offset=offset)

def i_bits2chars(bit_length: int) -> int:
    # How many characters is largest negative int of this length? (To include minus sign).
    return len(str((-1 << (bit_length - 1))))


i_defn = DtypeDefinition(DtypeKind.INT,
                         "a two's complement signed int",
                         "signed int",
                         AllowedSizes(1, None),
                         from_i,
                         to_i,
                         int,
                         True,
                         i_bits2chars,
                         endianness_variants=True)


# ----- Literal types -----

def to_bin(bs: Bits, start: int, length: int) -> str:
    """Return interpretation as a binary string."""
    return bs._slice_to_bin(start, length)

def from_bin(binstring: str, length: None = None) -> Bits:
    """Create from the value given in binstring."""
    return Bits._from_bin(binstring)

def to_oct(bs: Bits, start: int, length: int) -> str:
    """Return interpretation as an octal string."""
    return bs._slice_to_oct(start, length)

def from_oct(octstring: str, length: None = None) -> Bits:
    """Create from the value given in octstring."""
    return Bits._from_oct(octstring)

def to_hex(bs: Bits, start: int, length: int) -> str:
    """Return interpretation as a hexadecimal string."""
    return bs._slice_to_hex(start, length)

def from_hex(hexstring: str, length: None = None) -> Bits:
    """Create from the value given in hexstring."""
    return Bits._from_hex(hexstring)

def to_bytes(bs: Bits, start: int, length: int) -> bytes:
    """Return interpretation as bytes."""
    return bs._slice_to_bytes(start, length)

def from_bytes(data: bytearray | bytes | list, length: None = None) -> Bits:
    """Create from a bytes or bytearray object."""
    return Bits.from_bytes(bytes(data))


bin_defn = DtypeDefinition(DtypeKind.BIN,
                           "a binary string",
                           "binary string",
                           AllowedSizes(0, None),
                           from_bin,
                           to_bin,
                           str,
                           False,
                           bits_per_character=1)
oct_defn = DtypeDefinition(DtypeKind.OCT,
                           "an octal string",
                           "octal string",
                           AllowedSizes(0, None),
                           from_oct,
                           to_oct,
                           str,
                           False,
                           bits_per_character=3)
hex_defn = DtypeDefinition(DtypeKind.HEX,
                           "a hexadecimal string",
                           "hex string",
                           AllowedSizes(0, None),
                           from_hex,
                           to_hex,
                           str,
                           False,
                           bits_per_character=4)
bytes_defn = DtypeDefinition(DtypeKind.BYTES,
                             "a bytes object",
                             "bytes",
                             AllowedSizes(0, None),
                             from_bytes,
                             to_bytes,
                             bytes,
                             False,
                             bits_per_character=8)


# ----- Float types -----

def to_f(bs: Bits, start: int, length: int) -> float:
    """Interpret as a big-endian float."""
    fmt = {16: ">e", 32: ">f", 64: ">d"}[length]
    return struct.unpack(fmt, to_bytes(bs, start, length))[0]

def from_f(f: float | str, length: int | None) -> Bits:
    if length is None:
        raise ValueError("No length can be inferred for the float initialiser.")
    f = float(f)
    fmt = {16: ">e", 32: ">f", 64: ">d"}[length]
    try:
        b = struct.pack(fmt, f)
    except OverflowError:
        # If float64 doesn't fit it automatically goes to 'inf'. This reproduces that behaviour for other types.
        b = struct.pack(fmt, float("inf") if f > 0 else float("-inf"))
    return Bits.from_bytes(b)

def f_bits2chars(bit_length: Literal[16, 32, 64]) -> int:
    # These bit lengths were found by looking at lots of possible values
    if bit_length in [16, 32]:
        return 23  # Empirical value
    else:
        return 24  # Empirical value


f_defn = DtypeDefinition(DtypeKind.FLOAT,
                         "an IEEE floating point number",
                         "float",
                         AllowedSizes(sizes=(16, 32, 64)),
                         from_f,
                         to_f,
                         float,
                         True,
                         f_bits2chars,
                         endianness_variants=True)


# ----- Other known length types -----

def to_bits(bs: Bits, start: int, length: int) -> Bits:
    """Just return as a Bits."""
    return bs._getslice(start, length)

def from_bits(bs: BitsType, length: None = None) -> Bits:
    return bits_from_any(bs)

def bits_bits2chars(bit_length: int) -> int:
    # For bits type we can see how long it needs to be printed by trying any value
    temp = Bits.from_zeros(bit_length)
    return len(str(temp))


bits_defn = DtypeDefinition(DtypeKind.BITS,
                            "a Bits object",
                            "Bits",
                            AllowedSizes(0, None),
                            from_bits,
                            to_bits,
                            Bits,
                            False,
                            bits_bits2chars)


def to_bool(bs: Bits, start: int, _length: int) -> bool:
    """Interpret as a bool"""
    assert _length == 1
    return bs[start]

def from_bool(value: bool, length: None = None) -> Bits:
    return Bits.from_bools([value])

def bool_bits2chars(_: Literal[1]) -> int:
    # Bools are printed as 1 or 0, not True or False, so are one character each
    return 1


bool_defn = DtypeDefinition(DtypeKind.BOOL,
                            "a bool (True or False)",
                            "bool",
                            AllowedSizes(sizes=(1,)),
                            from_bool,
                            to_bool,
                            bool,
                            False,
                            bool_bits2chars)


# ----- Special case pad type -----

def to_pad(_bs: Bits, _start: int, _length: int) -> None:
    return None

def from_pad(value: None, length: int) -> None:
    raise ValueError("It's not possible to set a 'pad' value.")

pad_defn = DtypeDefinition(DtypeKind.PAD,
                           "a skipped section of padding",
                           "padding",
                           AllowedSizes(0, None),
                           from_pad,
                           to_pad,
                           None,
                           False,
                           None)

# ----------


dtype_definitions = [
    u_defn,
    i_defn,

    bin_defn,
    oct_defn,
    hex_defn,
    bytes_defn,

    f_defn,

    bits_defn,
    bool_defn,

    pad_defn,
]

