.PHONY: test build build-fast stubs coverage

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

# Copy the docstrings of the built module into tibs.pyi. Run after changing a
# doc comment in src/; tests/test_stub_docstrings.py fails until you do.
stubs:
	.venv/bin/python scripts/stub_docstrings.py

coverage:
	./coverage.sh
