# Agent Guidance for Tibs

This file gives project-specific guidance for automated coding assistants and contributors working on Tibs.

## Project Snapshot

- Tibs is a Python library for binary data, implemented in Rust with PyO3 and built with maturin.
- The public Python package exposes immutable `Tibs`, mutable `Mutibs`, `View`, and enum types from `src/lib.rs`.
- The package targets Python 3.8+ through `pyo3/abi3-py38`; continue supporting Python 3.8 unless explicitly instructed otherwise.
- The project is in beta. API changes are allowed when they improve the design, but still update tests, docs, examples, and type stubs to match.
- Rust uses edition 2024. Prefer local patterns in `src/` over introducing new abstractions.

## Repository Layout

- `src/`: Rust implementation and PyO3 bindings.
- `tibs.pyi`: public typing surface. Keep this in sync with exposed Python API changes.
- `tests/`: Python tests, including Hypothesis tests and benchmark tests.
- `doc/`: Sphinx source documentation.
- `examples/`: runnable examples that correspond to documentation examples.
- Generated or local-output directories/files such as `target/`, `dist/`, `doc/_build/`, `html/`, logs, benchmark outputs, and temporary scripts should not be edited unless the user asks.

## Development Workflow

- For Rust-only validation, start with `cargo check`.
- To rebuild the Python extension locally, use `make build` or `.venv/bin/maturin develop --release`.
- Running the complete test suite with benchmarks skipped is the usual recommendation: use `make test` or `.venv/bin/python -m pytest --benchmark-skip`.
- For focused Python checks, run targeted pytest commands against the relevant test file or test name.
- If changing docs, prefer updating the source `.rst` files and examples, not generated HTML.
- If changing public behavior, update Rust implementation, `tibs.pyi`, tests, docs, and examples as appropriate.

## Coding Expectations

- Preserve the existing API style: constructors and converters use names like `from_hex`, `to_hex`, properties like `.hex`, and Pythonic indexing/slicing behavior.
- Keep `Tibs` immutable and `Mutibs` mutable; do not blur ownership or mutation behavior between them.
- Error behavior is part of the Python API. Use clear Python exceptions through `PyResult` and existing PyO3 exception patterns.
- Be conservative with dependencies. Prefer the current dependency set unless a new dependency materially simplifies a real problem.
- Performance is crucial at both large and small scales. Any change that could affect performance should be checked thoroughly, with focused benchmarks or comparisons when relevant.
- Avoid obviously quadratic algorithms in core bit operations and search paths unless the input size is provably tiny or the tradeoff is explicitly accepted.
- Keep `tibs.pyi` updated with every change that affects the public Python API or typing surface.
- Match the documentation style of the section being edited. Method/reference docs should include full parameter information and usually examples. User manual pages should be less exhaustive, more conversational, and focused on explaining workflows.
- Keep examples concise and executable-looking; prefer examples that show actual binary data workflows.
- Do not treat generated docs, wheels, benchmark logs, or local scratch files as source of truth.

## Git And Local State

- The worktree may contain user edits and generated artifacts. Do not revert or clean unrelated changes.
- Before broad edits, inspect the current diff for files you plan to touch.
- Keep changes tightly scoped to the request.
- Avoid touching generated artifacts such as `dist/`, `target/`, `doc/_build/`, `html/`, benchmark result files, and logs unless explicitly requested.
