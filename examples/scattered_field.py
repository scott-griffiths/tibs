from tibs import Mutibs, Tibs


# Register layouts often split a field around fixed bits that sit in the middle
# of it. Here a 16-bit ADC result register holds a 10-bit sample interrupted by a
# 2-bit status field:
#
#   bit (MSB0)  0 1 2 3 4 5   6 7    8 9 10 11    12 13 14 15
#   field       sample[9:4]   status  sample[3:0]  reserved
#
# The sample's bits are contiguous within each run and in order overall, just not
# adjacent. `extract` gathers them with a mask; `deposit` writes them back,
# leaving the status and reserved bits untouched.

SAMPLE = Tibs("0b1111_1100_1111_0000")   # the ten bits that make up the sample
STATUS = Tibs("0b0000_0011_0000_0000")   # the two status bits in the middle

reg = Mutibs.from_bytes(bytes.fromhex("B3 A4"))
assert reg.bin == "1011001110100100"

# Read the scattered sample as a single value.
sample = reg.extract(SAMPLE)
assert sample.bin == "1011001010"
assert sample.u == 714

# The status field is a separate, ordinary contiguous read.
assert reg.extract(STATUS).bin == "11"

# Bump the sample and write it back. deposit only touches the masked positions,
# so status and reserved bits are preserved.
reg.deposit(Tibs.from_u(sample.u + 1, 10), SAMPLE)
assert reg.extract(SAMPLE).u == 715
assert reg.extract(STATUS).bin == "11"          # status untouched
assert reg.bin == "1011001110110100"

# The immutable Tibs.deposited returns a new value instead of mutating.
frozen = Tibs.from_bytes(bytes.fromhex("B3 A4"))
updated = frozen.deposited(Tibs.from_u(0, 10), SAMPLE)
assert updated.extract(SAMPLE).u == 0
assert frozen.extract(SAMPLE).u == 714          # original unchanged

# extract and deposit are inverses: writing back what you read is a no-op.
assert frozen.deposited(frozen.extract(SAMPLE), SAMPLE) == frozen
