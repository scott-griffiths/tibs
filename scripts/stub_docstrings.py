#!/usr/bin/env python
"""Write the docstrings of the built ``tibs`` module into ``tibs.pyi``.

The doc comments in ``src/`` are the only place a docstring is written by hand.
PyO3 turns them into ``__doc__`` at runtime, which is what ``help()`` and the
Sphinx reference read, but editors, type checkers and coding agents often read
the stub file instead. This script copies each runtime docstring into the
matching class, method or property in the stub, so the two cannot drift.

Build first (``make build-fast`` is enough), then run::

    python scripts/stub_docstrings.py           # rewrite tibs.pyi
    python scripts/stub_docstrings.py --check   # exit 1 if it is out of date

``tests/test_stub_docstrings.py`` runs the check, so a doc comment changed
without regenerating the stub fails the test suite.

Only docstrings are touched. Signatures, comments and layout in the stub are
left exactly as written. Methods that CPython implements as slots, such as
``__len__``, carry a generic docstring rather than one from ``src/``, so they
are left without one.
"""
import argparse
import ast
import inspect
import sys
import types
from pathlib import Path

import tibs

STUB = Path(__file__).resolve().parent.parent / "tibs.pyi"

SLOT_TYPES = (types.WrapperDescriptorType, types.MethodWrapperType)


def runtime_doc(obj):
    """The cleaned docstring of a runtime object, or None if it has no docstring of its own."""
    if obj is None or isinstance(obj, SLOT_TYPES):
        return None
    doc = getattr(obj, "__doc__", None)
    if not doc or not doc.strip():
        return None
    return inspect.cleandoc(doc)


def docstring_literal(doc, indent):
    """A triple-quoted literal for ``doc``, with continuation lines indented."""
    raw = "\\" in doc
    if '"""' in doc or doc.endswith(('"', "\\")):
        raw = False
        doc = doc.replace("\\", "\\\\").replace('"""', '\\"""')
        if doc.endswith('"'):
            doc = doc[:-1] + '\\"'
    lines = doc.split("\n")
    prefix = 'r"""' if raw else '"""'
    if len(lines) == 1:
        return f'{prefix}{lines[0]}"""'
    body = "\n".join(indent + line if line else "" for line in lines[1:])
    return f'{prefix}{lines[0]}\n{body}\n{indent}"""'


def is_docstring(node):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def is_ellipsis(node):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and node.value.value is Ellipsis)


def is_setter(node):
    # ``@x.setter`` and ``@x.deleter`` share the getter's docstring.
    return any(isinstance(d, ast.Attribute) for d in node.decorator_list)


class Rewriter:
    def __init__(self, source):
        self.source = source
        self.line_starts = [0]
        for line in source.splitlines(keepends=True):
            self.line_starts.append(self.line_starts[-1] + len(line))
        self.edits = []  # (start, end, replacement) as character offsets

    def offset(self, lineno, col):
        # ast columns are UTF-8 byte offsets; the stub is ASCII, so they are also characters.
        return self.line_starts[lineno - 1] + col

    def set_docstring(self, node, doc, indent):
        """Give ``node`` (a module, class or function) the docstring ``doc``, or none."""
        body = node.body
        old = body[0] if body and is_docstring(body[0]) else None
        rest = body[1:] if old else body
        if old is not None and doc is not None and ast.get_docstring(node, clean=True) == doc:
            return
        if old is None and doc is None:
            return
        if not rest or (len(rest) == 1 and is_ellipsis(rest[0])):
            # The body is only a docstring and/or ``...``: rewrite everything after the colon.
            start = self.offset(body[0].lineno, body[0].col_offset)
            while self.source[start - 1] != ":":
                start -= 1
            end = self.offset(body[-1].end_lineno, body[-1].end_col_offset)
            text = f"\n{indent}{docstring_literal(doc, indent)}" if doc else " ..."
            self.edits.append((start, end, text))
        elif old is not None:
            # Replace or remove whole lines, along with the blank line after a removed one.
            start = self.line_starts[old.lineno - 1]
            end = self.line_starts[old.end_lineno]
            if doc is None:
                if self.source[end:end + 1] == "\n":
                    end += 1
                self.edits.append((start, end, ""))
            else:
                self.edits.append((start, end, f"{indent}{docstring_literal(doc, indent)}\n"))
        else:
            start = self.line_starts[rest[0].lineno - 1]
            self.edits.append((start, start, f"{indent}{docstring_literal(doc, indent)}\n\n"))

    def result(self):
        text = self.source
        for start, end, replacement in sorted(self.edits, reverse=True):
            text = text[:start] + replacement + text[end:]
        return text


def generate(source):
    """Return ``source`` with every docstring set from the runtime module."""
    tree = ast.parse(source)
    rewriter = Rewriter(source)
    rewriter.set_docstring(tree, runtime_doc(tibs), "")
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        cls = getattr(tibs, node.name)
        rewriter.set_docstring(node, runtime_doc(cls), "    ")
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and not is_setter(member):
                doc = runtime_doc(getattr(cls, member.name, None))
                rewriter.set_docstring(member, doc, "        ")
    return rewriter.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="report whether tibs.pyi is up to date instead of rewriting it")
    args = parser.parse_args()
    current = STUB.read_text()
    updated = generate(current)
    if args.check:
        if updated != current:
            print("tibs.pyi docstrings are out of date: run python scripts/stub_docstrings.py")
            return 1
        print("tibs.pyi docstrings are up to date")
        return 0
    if updated != current:
        STUB.write_text(updated)
        print("Updated tibs.pyi")
    else:
        print("tibs.pyi already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
