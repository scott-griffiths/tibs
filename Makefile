.PHONY: test build coverage

test:
	.venv/bin/python -m pytest --benchmark-skip

build:
	.venv/bin/maturin develop --release

coverage:
	./coverage.sh
