from tibs import Tibs


samples = [0, 103, 2048, 4095]

packed = Tibs.from_values("u12", samples)
assert packed.hex == "000067800fff"

round_tripped = Tibs.from_bytes(packed.bytes).to_values("u12")
assert round_tripped == samples

voltages = [round(sample * 3.3 / 4095, 3) for sample in round_tripped]
assert voltages == [0.0, 0.083, 1.65, 3.3]
