from tibs import Dtype, Mutibs, Reader, Tibs


# Reading a bit-packed telemetry frame, one record at a time.
#
# The records are neither byte aligned nor fixed width: a 3-bit tag says what
# comes next, and how long it is depends on the tag. Nothing here tracks where a
# record starts except the Reader, which is left wherever the previous one
# ended.

SYNC = Tibs("0xa55a")
END, SAMPLE, LABEL = 0, 1, 2

SAMPLE_RECORD = Dtype("(u3, i10)")   # tag, temperature in tenths of a degree
LABEL_HEADER = Dtype("(u3, u5)")     # tag, length of the name in bytes

# Build a frame: some noise from a mid-stream start, the sync word, then the
# records and a terminating tag. A name is byte aligned in this format, so the
# header before it is followed by padding.
frame = Mutibs("0x00ff")
frame += SYNC
frame += SAMPLE_RECORD.pack((SAMPLE, 133))
frame += LABEL_HEADER.pack((LABEL, 4))
frame += Tibs.from_zeros(3) + b"pump"
frame += SAMPLE_RECORD.pack((SAMPLE, -48))
frame += Tibs.from_u(END, length=3)

reader = Reader(frame)

# Nothing before the sync word is a record, and the sync word only counts when
# it starts on a byte boundary.
assert reader.seek_past(SYNC, byte_aligned=True)
assert reader.byte_pos == 4

# The tag is peeked to decide what the record is, then read again as part of the
# record itself.
records = []
while reader.peek_value("u3") != END:
    if reader.peek_value("u3") == SAMPLE:
        _, tenths = reader.read_value(SAMPLE_RECORD)
        records.append(("sample", tenths / 10))
    else:
        _, length = reader.read_value(LABEL_HEADER)
        reader.align()               # skip to the byte boundary the name starts on
        records.append(("label", reader.read_bits(length * 8).bytes))

assert records == [
    ("sample", 13.3),
    ("label", b"pump"),
    ("sample", -4.8),
]

# Reading 13-bit records leaves the cursor part way through a byte, which is
# where a hand-threaded position would have started going wrong.
assert reader.pos == 101 and reader.pos % 8 == 5

reader.read_value("u3")              # the END tag
assert reader.at_end and reader.remaining == 0
