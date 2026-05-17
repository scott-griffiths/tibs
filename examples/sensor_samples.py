from tibs import Tibs


BITS_PER_SAMPLE = 12
ADC_MAX = (1 << BITS_PER_SAMPLE) - 1
REFERENCE_VOLTS = 3.3


def pack_samples(samples):
    return Tibs.from_joined(
        Tibs.from_u(sample, BITS_PER_SAMPLE)
        for sample in samples
    )


def unpack_samples(payload):
    bits = Tibs.from_bytes(payload)
    complete_samples = len(bits) // BITS_PER_SAMPLE
    return [
        sample.u
        for sample in bits.chunks(BITS_PER_SAMPLE, complete_samples)
    ]


def sample_to_volts(sample):
    return sample * REFERENCE_VOLTS / ADC_MAX


samples = [0, 103, 2048, 4095]

packed = pack_samples(samples)
assert packed.hex == "000067800fff"
assert packed.bytes.hex() == "000067800fff"

round_tripped = unpack_samples(packed.bytes)
assert round_tripped == samples

voltages = [round(sample_to_volts(sample), 3) for sample in round_tripped]
assert voltages == [0.0, 0.083, 1.65, 3.3]
