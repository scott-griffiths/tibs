#!/usr/bin/env python
"""The docstrings in ``tibs.pyi`` are generated from the doc comments in ``src/``.

A doc comment changed without regenerating the stub fails here. To fix it, build
and run ``make stubs`` (or ``python scripts/stub_docstrings.py``).
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "stub_docstrings", ROOT / "scripts" / "stub_docstrings.py"
)
stub_docstrings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stub_docstrings)


def test_stub_docstrings_match_runtime():
    current = stub_docstrings.STUB.read_text()
    assert stub_docstrings.generate(current) == current, (
        "tibs.pyi docstrings differ from the built module's: run `make stubs`"
    )
