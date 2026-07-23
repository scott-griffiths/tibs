from tibs import Mutibs, Tibs


# Comparing items reduced to fixed-length binary fingerprints, where bit i is
# set if the item has feature i. Chemical similarity searching, near-duplicate
# detection and Bloom filters all have this shape: the bits stop being a
# sequence to be read and become a set to be compared.

FEATURES = 256


def fingerprint(present):
    """Build a fingerprint from the feature numbers an item has."""
    bits = Mutibs.from_zeros(FEATURES)
    bits.set(present)
    return Tibs(bits)


reference = fingerprint([3, 17, 42, 88, 91, 150, 199, 201, 250])
candidate = fingerprint([3, 17, 42, 88, 120, 150, 199, 240])
unrelated = fingerprint([5, 60, 61, 62, 100])

# How many features each item has. With no argument, count() counts set bits.
assert reference.count() == 9
assert candidate.count() == 8

# Shared and combined features, neither of which builds an intermediate object.
shared = reference.count_and(candidate)
combined = reference.count_or(candidate)
assert (shared, combined) == (6, 11)

# The Tanimoto (Jaccard) coefficient is the standard similarity score here.
assert round(shared / combined, 3) == 0.545
assert round(reference.count_and(unrelated) / reference.count_or(unrelated), 3) == 0.0

# The Hamming distance is the number of features they disagree about, and is
# exactly the size of the symmetric difference.
assert reference.count_xor(candidate) == 5
assert reference.count_xor(candidate) == combined - shared

# Features the reference has that the candidate lacks, in a single pass.
assert reference.count_andnot(candidate) == 3
assert candidate.count_andnot(reference) == 2

# Screening questions answer without scanning the whole fingerprint: both of
# these stop as soon as the outcome is known.
assert reference.intersects(candidate) is True
assert reference.intersects(unrelated) is False

# A required-features screen, the usual first pass before scoring anything.
required = fingerprint([3, 42, 150])
assert required.is_subset_of(reference) is True
assert required.is_subset_of(candidate) is True
assert required.is_subset_of(unrelated) is False

# Only the candidates that survive the screen are worth scoring.
library = {"candidate": candidate, "unrelated": unrelated}
scored = {
    name: round(reference.count_and(fp) / reference.count_or(fp), 3)
    for name, fp in library.items()
    if required.is_subset_of(fp)
}
assert scored == {"candidate": 0.545}
