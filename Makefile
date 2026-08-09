.PHONY: test build build-fast coverage

test:
	.venv/bin/python -m pytest --benchmark-skip

# Fully optimised, as shipped. Use this before benchmarking or releasing.
build:
	.venv/bin/maturin develop --release

# Roughly twice as quick to link, for the rebuild-and-run loop. Correctness
# only: never take timings from a build made this way. See the `fastdev`
# profile in Cargo.toml.
build-fast:
	.venv/bin/maturin develop --profile fastdev

coverage:
	./coverage.sh
