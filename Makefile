.PHONY: test build

test:
	.venv/bin/python -m pytest --benchmark-skip

build:
	.venv/bin/maturin develop --release
