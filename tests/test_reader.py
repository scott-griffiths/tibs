import pytest

from tibs import DtypeArray, Mutibs, Reader, Tibs


# --- construction and state ------------------------------------------------


def test_reader_keeps_the_source_object_rather_than_a_copy():
    t = Tibs("0x1234")
    m = Mutibs("0x1234")

    assert Reader(t).source is t
    assert Reader(m).source is m


def test_reader_promotes_other_sources_once():
    r = Reader("0xabcd")

    assert isinstance(r.source, Tibs)
    assert r.source == Tibs("0xabcd")
    assert r.read_value("hex4") == "a"


def test_reader_starting_position():
    t = Tibs("0x0123")

    assert Reader(t).pos == 0
    assert Reader(t, 8).pos == 8
    assert Reader(t, 16).pos == 16
    assert Reader(t, 8).read_value("u8") == 0x23


@pytest.mark.parametrize("pos", [-1, 17, 1000])
def test_reader_rejects_an_out_of_range_starting_position(pos):
    with pytest.raises(ValueError):
        _ = Reader(Tibs("0x0123"), pos)


def test_reader_len_is_the_source_length():
    assert len(Reader(Tibs("0x0123"))) == 16
    assert len(Reader(Tibs())) == 0


def test_pos_setter_validates_against_the_source_length():
    r = Reader(Tibs("0xff"))

    r.pos = 8
    assert r.pos == 8
    r.pos = 0
    assert r.pos == 0

    with pytest.raises(ValueError):
        r.pos = 9
    with pytest.raises(ValueError):
        r.pos = -1
    assert r.pos == 0


def test_byte_pos_round_trips_and_needs_alignment():
    r = Reader(Tibs("0x0123"))

    r.byte_pos = 1
    assert r.pos == 8
    assert r.byte_pos == 1

    r.pos = 9
    with pytest.raises(ValueError):
        _ = r.byte_pos

    with pytest.raises(ValueError):
        r.byte_pos = 3
    assert r.pos == 9


def test_remaining_and_at_end():
    r = Reader(Tibs("0x0123"))

    assert (r.remaining, r.at_end) == (16, False)
    r.read_bits(15)
    assert (r.remaining, r.at_end) == (1, False)
    r.read_bits(1)
    assert (r.remaining, r.at_end) == (0, True)


def test_empty_source_starts_at_the_end():
    r = Reader(Tibs())

    assert (len(r), r.pos, r.remaining, r.at_end) == (0, 0, 0, True)


def test_repr():
    assert repr(Reader(Tibs("0x0123"), 4)) == "Reader(Tibs('0x0123'), 4)"
    assert repr(Reader(Mutibs("0x0123"))) == "Reader(Mutibs('0x0123'), 0)"


# --- reading ---------------------------------------------------------------


def test_read_value_advances_by_the_dtype_length():
    r = Reader(Tibs("0x47ff10"))

    assert r.read_value("u8") == 71
    assert r.pos == 8
    assert r.read_value("(bool, u7)") == (True, 127)
    assert r.pos == 16
    assert r.read_value("bits8") == Tibs("0x10")
    assert r.at_end


def test_read_value_matches_to_value_over_the_same_window():
    t = Tibs.from_random(200, seed=b"reader")
    r = Reader(t)
    for _ in range(25):
        start = r.pos
        assert r.read_value("u8") == t.to_value("u8", start, start + 8)


def test_read_value_containers_match_to_value():
    t = Tibs("0x010203")

    assert Reader(t).read_value("[u8; 3]") == t.to_value("[u8; 3]")
    assert Reader(t).read_value("(u8, u16)") == t.to_value("(u8, u16)")


def test_read_values_with_a_count():
    r = Reader(Tibs("0x0102030405"))

    assert r.read_values("u8", 2) == [1, 2]
    assert r.pos == 16
    assert r.read_values("u8", 0) == []
    assert r.pos == 16


def test_read_values_without_a_count_reads_the_whole_items_that_fit():
    r = Reader(Tibs("0x0102030405"))
    r.read_values("u8", 2)

    assert r.read_values("u8") == [3, 4, 5]
    assert r.at_end


def test_read_values_leaves_a_partial_item_under_the_cursor():
    r = Reader(Tibs("0b1111111111"))

    assert r.read_values("u4") == [15, 15]
    assert (r.pos, r.remaining, r.at_end) == (8, 2, False)


def test_read_values_with_no_whole_item_left_reads_nothing():
    r = Reader(Tibs.from_zeros(3))

    assert r.read_values("u8") == []
    assert r.pos == 0


def test_read_values_matches_to_values():
    t = Tibs("0x0102030405060708")

    assert Reader(t).read_values("u16") == t.to_values("u16")
    assert Reader(t).read_values("u16", 2) == t.to_values("u16", 0, 32)


def test_read_bits():
    r = Reader(Tibs("0xf00f"))

    assert r.read_bits(4) == Tibs("0xf")
    assert r.read_bits(0) == Tibs()
    assert r.pos == 4
    assert r.read_bits(12) == Tibs("0x00f")
    assert r.at_end


def test_read_bits_from_a_mutibs_gives_a_tibs():
    r = Reader(Mutibs("0xf00f"))
    bits = r.read_bits(8)

    assert isinstance(bits, Tibs)
    assert bits == Tibs("0xf0")


def test_read_to_stops_at_the_match():
    r = Reader(Tibs("0x0000ff12"))

    assert r.read_to("0xff") == Tibs("0x0000")
    assert r.pos == 16
    assert r.read_value("u8") == 0xFF


def test_read_to_a_match_already_under_the_cursor_reads_nothing():
    r = Reader(Tibs("0xff00"))

    assert r.read_to("0xff") == Tibs()
    assert r.pos == 0


def test_read_past_includes_the_match():
    r = Reader(Tibs("0x0000ff12"))

    assert r.read_past("0xff") == Tibs("0x0000ff")
    assert r.pos == 24


def test_read_to_and_read_past_honour_byte_aligned_and_mask():
    t = Tibs("0x3a5f")
    masked = t.find("0x0f", mask="0x0f")
    aligned = t.find("0x0f", mask="0x0f", byte_aligned=True)
    assert masked != aligned  # Otherwise the flags are not being tested.

    assert Reader(t).read_to("0x0f", mask="0x0f") == t[:masked]
    assert Reader(t).read_past("0x0f", mask="0x0f") == t[: masked + 8]
    assert Reader(t).read_to("0x0f", byte_aligned=True, mask="0x0f") == t[:aligned]


def test_read_to_raises_when_the_needle_is_missing():
    r = Reader(Tibs("0x0000"), 4)

    with pytest.raises(ValueError):
        r.read_to("0xff")
    with pytest.raises(ValueError):
        r.read_past("0xff")
    assert r.pos == 4


# --- looking ahead ---------------------------------------------------------


def test_peek_value_does_not_move():
    r = Reader(Tibs("0x47ff"), 8)

    assert r.peek_value("u8") == 255
    assert r.peek_value("u8") == 255
    assert r.pos == 8


def test_peek_bits_does_not_move():
    r = Reader(Tibs("0xf00f"))

    assert r.peek_bits(4) == Tibs("0xf")
    assert r.peek_bits(0) == Tibs()
    assert r.pos == 0


def test_bookmark_restores_the_position():
    r = Reader(Tibs("0x010203"), 8)

    with r.bookmark() as inner:
        assert inner is r
        assert r.read_values("u8") == [2, 3]
        assert r.at_end
    assert r.pos == 8


def test_bookmark_restores_after_an_exception():
    r = Reader(Tibs("0x010203"), 8)

    with pytest.raises(RuntimeError):
        with r.bookmark():
            r.read_bits(8)
            raise RuntimeError("boom")
    assert r.pos == 8


def test_bookmark_does_not_suppress_exceptions():
    r = Reader(Tibs("0x010203"))

    with pytest.raises(ValueError):
        with r.bookmark():
            r.read_bits(100)


def test_bookmarks_nest():
    r = Reader(Tibs("0x01020304"))

    with r.bookmark():
        r.read_bits(8)
        with r.bookmark():
            r.read_bits(8)
            assert r.pos == 16
        assert r.pos == 8
    assert r.pos == 0


# --- moving ----------------------------------------------------------------


def test_align_skips_to_the_next_boundary():
    r = Reader(Tibs("0xffff"), 3)

    assert r.align() == 5
    assert r.pos == 8
    assert r.align() == 0
    assert r.pos == 8


def test_align_generalises_beyond_bytes():
    r = Reader(Tibs.from_zeros(64), 9)

    assert r.align(16) == 7
    assert r.pos == 16
    assert r.align(32) == 16
    assert r.pos == 32
    assert r.align(1) == 0


def test_align_can_land_exactly_on_the_end():
    r = Reader(Tibs("0xff"), 5)

    assert r.align() == 3
    assert r.at_end


def test_align_past_the_end_raises():
    r = Reader(Tibs("0b111"), 1)

    with pytest.raises(ValueError):
        r.align()
    assert r.pos == 1


@pytest.mark.parametrize("boundary", [0, -8])
def test_align_rejects_a_non_positive_boundary(boundary):
    r = Reader(Tibs("0xff"), 1)

    with pytest.raises(ValueError):
        r.align(boundary)
    assert r.pos == 1


def test_seek_to_leaves_the_cursor_on_the_match():
    r = Reader(Tibs("0x0000ff12"))

    assert r.seek_to("0xff") is True
    assert r.pos == 16
    # The needle is under the cursor, so this finds it where it is.
    assert r.seek_to("0xff") is True
    assert r.pos == 16


def test_seek_past_leaves_the_cursor_after_the_match():
    r = Reader(Tibs("0x0000ff12"))

    assert r.seek_past("0xff") is True
    assert r.pos == 24


def test_seek_returns_false_without_moving_when_not_found():
    r = Reader(Tibs("0x0000"), 4)

    assert r.seek_to("0xff") is False
    assert r.seek_past("0xff") is False
    assert r.seek_back_to("0xff") is False
    assert r.pos == 4


def test_a_match_at_bit_zero_is_not_falsy():
    r = Reader(Tibs("0xff00"))

    assert r.seek_to("0xff") is True
    assert r.pos == 0


def test_seek_past_drives_a_scanning_loop():
    t = Tibs("0x00ff00ff00")
    r = Reader(t)

    starts = []
    while r.seek_past("0xff"):
        starts.append(r.pos - 8)
    assert starts == t.find_all("0xff")


def test_seek_back_to_moves_strictly_backwards():
    t = Tibs("0x00ff00ff")
    r = Reader(t, len(t))

    starts = []
    while r.seek_back_to("0xff"):
        starts.append(r.pos)
    assert starts == sorted(t.find_all("0xff"), reverse=True)


def test_seek_back_to_ignores_a_match_starting_at_the_cursor():
    t = Tibs("0x00ffff")
    r = Reader(t, 8)

    # The match at bit 8 does not end by bit 8, so there is nothing behind.
    assert r.seek_back_to("0xff") is False
    assert r.pos == 8


def test_seeks_honour_byte_aligned_and_mask():
    t = Tibs("0x3a5f")
    masked = t.find("0x0f", mask="0x0f")
    aligned = t.find("0x0f", mask="0x0f", byte_aligned=True)
    assert masked != aligned  # Otherwise the flags are not being tested.
    r = Reader(t)

    assert r.seek_to("0x0f", byte_aligned=True, mask="0x0f") is True
    assert r.pos == aligned

    r.pos = 0
    assert r.seek_to("0x0f", mask="0x0f") is True
    assert r.pos == masked

    r.pos = len(t)
    assert r.seek_back_to("0x0f", mask="0x0f") is True
    assert r.pos == t.rfind("0x0f", mask="0x0f")


@pytest.mark.parametrize(
    "call",
    [
        lambda r: r.seek_to(""),
        lambda r: r.seek_past(""),
        lambda r: r.seek_back_to(""),
        lambda r: r.read_to(""),
        lambda r: r.read_past(""),
    ],
)
def test_an_empty_needle_is_an_error(call):
    r = Reader(Tibs("0xff"), 4)

    with pytest.raises(ValueError):
        call(r)
    assert r.pos == 4


# --- the "pos only moves on success" invariant -----------------------------

# Each entry is a call that must fail on a reader over 8 bits, whatever the
# starting position. Rule: a method that raises leaves the cursor alone.
FAILING_CALLS = [
    ("read_value", lambda r: r.read_value("u64")),
    ("read_value bad dtype", lambda r: r.read_value("nonsense")),
    ("read_values count", lambda r: r.read_values("u8", 9)),
    ("read_values negative count", lambda r: r.read_values("u8", -1)),
    ("read_values bad dtype", lambda r: r.read_values("nonsense")),
    ("read_bits", lambda r: r.read_bits(9)),
    ("read_bits negative", lambda r: r.read_bits(-1)),
    ("read_to missing", lambda r: r.read_to("0b1010101010")),
    ("read_past missing", lambda r: r.read_past("0b1010101010")),
    ("read_to empty", lambda r: r.read_to("")),
    ("peek_value", lambda r: r.peek_value("u64")),
    ("peek_bits", lambda r: r.peek_bits(9)),
    ("align", lambda r: r.align(64) if r.pos else r.read_bits(9)),
    ("align zero", lambda r: r.align(0)),
    ("seek_to empty", lambda r: r.seek_to("")),
    ("seek_past empty", lambda r: r.seek_past("")),
    ("seek_back_to empty", lambda r: r.seek_back_to("")),
    ("pos setter", lambda r: setattr(r, "pos", 9)),
    ("byte_pos setter", lambda r: setattr(r, "byte_pos", 2)),
]


@pytest.mark.parametrize("label, call", FAILING_CALLS, ids=[c[0] for c in FAILING_CALLS])
@pytest.mark.parametrize("start", [0, 1, 7, 8])
def test_a_failed_call_never_moves_the_cursor(label, call, start):
    r = Reader(Tibs("0b10110001"), start)

    with pytest.raises(Exception):
        call(r)
    assert r.pos == start


# Calls that must fail on a bad dtype or needle *type* rather than a length.
@pytest.mark.parametrize(
    "call",
    [
        lambda r: r.read_value(4),
        lambda r: r.read_values(4),
        lambda r: r.peek_value(4),
        lambda r: r.read_to(object()),
        lambda r: r.seek_to(object()),
    ],
)
def test_a_bad_argument_type_never_moves_the_cursor(call):
    r = Reader(Tibs("0xff"), 4)

    with pytest.raises(TypeError):
        call(r)
    assert r.pos == 4


# --- Mutibs sources --------------------------------------------------------


def test_a_mutibs_source_is_read_live():
    m = Mutibs("0x01")
    r = Reader(m)

    assert r.read_value("u8") == 1
    assert r.at_end

    m += "0x0203"
    assert len(r) == 24
    assert r.remaining == 16
    assert not r.at_end
    assert r.read_values("u8") == [2, 3]


def test_appending_then_setting_pos_is_the_way_to_read_ahead():
    m = Mutibs("0x01")
    r = Reader(m)

    with pytest.raises(ValueError):
        r.pos = 16
    m += "0x0203"
    r.pos = 16
    assert r.read_value("u8") == 3


def test_a_truncated_mutibs_source_leaves_the_cursor_past_the_end():
    m = Mutibs("0x010203")
    r = Reader(m, 24)

    del m[8:]
    assert len(r) == 8
    assert r.pos == 24
    assert r.remaining == 0
    assert r.at_end

    with pytest.raises(ValueError):
        r.read_bits(1)
    assert r.seek_to("0x01") is False
    assert r.seek_back_to("0x01") is True
    assert r.pos == 0


def test_reads_from_a_mutibs_match_reads_from_the_equivalent_tibs():
    m = Mutibs.from_random(1000, seed=b"mutibs")
    t = m.to_tibs()
    mutable, immutable = Reader(m, 3), Reader(t, 3)

    for _ in range(50):
        assert mutable.read_value("u13") == immutable.read_value("u13")
        assert mutable.read_bits(5) == immutable.read_bits(5)
        assert mutable.pos == immutable.pos


# --- equivalences with the underlying API ----------------------------------


def test_reader_leaves_windowed_queries_to_the_source():
    t = Tibs("0x0102030405")
    r = Reader(t, 16)

    assert r.source.to_value("u8", 0, 8) == 1
    assert r.pos == 16


def test_read_value_of_an_array_dtype_matches_read_values_in_content():
    t = Tibs("0x01020304")

    assert Reader(t).read_value(DtypeArray.from_params("u8", 4)) == (1, 2, 3, 4)
    assert Reader(t).read_values("u8", 4) == [1, 2, 3, 4]
    assert Reader(t).read_value("[u8; 4]") == tuple(Reader(t).read_values("u8", 4))


def test_reading_the_whole_source_in_steps_reconstructs_it():
    t = Tibs.from_random(997, seed=b"steps")
    r = Reader(t)

    parts = []
    while r.remaining >= 7:
        parts.append(r.read_bits(7))
    parts.append(r.read_bits(r.remaining))
    assert Tibs.from_joined(parts) == t
    assert r.at_end
