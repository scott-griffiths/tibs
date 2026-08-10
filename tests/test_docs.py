#!/usr/bin/env python
"""Runs the ``>>>`` examples in ``doc/*.rst`` and in the public docstrings.

Neither is checked anywhere else: ``conf.py`` doesn't load
``sphinx.ext.doctest``, and pytest collects neither ``.rst`` nor an extension
module's docstrings by default, so both can go stale silently.

The documentation is written for a reader, not for this harness, so where the
two disagree the harness gives way. Two kinds of output vary for reasons that
have nothing to do with whether the example is correct, and both are handled
here rather than by rewording the manual:

* A ``set`` repr has no defined order, so it is compared as a set.
* Output derived from an unseeded ``from_random`` cannot be predicted at all.
  Those examples are still executed, but their output is not compared.

The manual writes ``Tibs`` rather than ``tibs.Tibs``, as though the reader had
done ``from tibs import *``, so the public names are injected as globals rather
than each file carrying an import line.
"""
import doctest
import os
import re
import sys
from pathlib import Path

import pytest

import tibs

DOC_DIR = Path(__file__).resolve().parent.parent / "doc"

# One block in byte_format.rst encodes ten billion zero bits to show off the
# Rice codec. That is the point of the example, but it allocates over a
# gigabyte and takes ~30 seconds, which is more than the rest of the test suite
# put together. Run it with TIBS_DOC_SLOW=1, and in particular before a release.
SLOW_FILES = {"byte_format.rst"}

# bitvec addresses a bit with a usize and spends 3 of those bits on the
# position within an element, so a container tops out at 2**(pointer - 3) - 1
# bits: 2**61 - 1 on a 64-bit build, but only 536,870,911 (about 64 MB) on a
# 32-bit one, which the x86 wheels are. Two examples are deliberately larger
# than that, because the point of both is what tibs does with enormous sparse
# containers. Shrinking them to suit the smallest platform we ship would lose
# the example, so they are skipped where they cannot fit.
POINTER_BITS = sys.maxsize.bit_length() + 1
BIT_CAPACITY = 2 ** (POINTER_BITS - 3) - 1
LARGE_ALLOCATION_FILES = {
    "byte_format.rst": 10_000_000_000,
    "serialization.rst": 1_000_000_000,
}

# Examples whose output is genuinely unpredictable, keyed by the source line.
# These are run but not checked. Keep this list tiny: an entry here is an
# example nothing verifies.
UNCHECKABLE = {
    "bitset.rst": (
        # Counts of a fresh unseeded Tibs.from_random(100_000_000).
        "t.count()",
        "t.count(0)",
        "t.count([1, 0, 1])",
    ),
}

DOC_FILES = sorted(p.name for p in DOC_DIR.glob("*.rst"))
assert DOC_FILES, "no documentation found to test"

SET_REPR = re.compile(r"^\{.*\}$", re.DOTALL)


class DocOutputChecker(doctest.OutputChecker):
    """Compares a ``set`` repr as a set, and everything else as usual."""

    def check_output(self, want, got, optionflags):
        if super().check_output(want, got, optionflags):
            return True
        want, got = want.strip(), got.strip()
        if SET_REPR.match(want) and SET_REPR.match(got):
            return self._elements(want) == self._elements(got)
        return False

    @staticmethod
    def _elements(text):
        # Good enough for the reprs the manual actually shows. Anything with a
        # comma nested inside an element would need a real parser, and would
        # simply fall through to a normal failure.
        return frozenset(part.strip() for part in text[1:-1].split(","))


def _globals():
    return {name: getattr(tibs, name) for name in tibs.__all__ if not name.startswith("_")}


def _read_doc(filename):
    """The text of one documentation file.

    The encoding is stated rather than left to the locale, which is what
    ``read_text`` would otherwise use. Sphinx reads these as UTF-8 and
    ``exotic_floats.rst`` writes exponents as superscripts, so on a Windows
    runner - where the locale encoding is cp1252 and CI checks the files out
    with CRLF endings - the default decoded ``e2 81 bb`` as far as its middle
    byte and stopped: ``'charmap' codec can't decode byte 0x81``. Nothing was
    wrong with the file or the library; only this read.
    """
    return (DOC_DIR / filename).read_text(encoding="utf-8")


def _run(filename):
    """Runs one file, returning the number of failures and examples."""
    unchecked = UNCHECKABLE.get(filename, ())
    text = _read_doc(filename)
    test = doctest.DocTestParser().get_doctest(
        text, _globals(), filename, str(DOC_DIR / filename), 0
    )
    seen = set()
    for example in test.examples:
        if example.source.strip() in unchecked:
            seen.add(example.source.strip())
            # Still run it - that it returns at all is worth checking - but
            # accept whatever comes back. ELLIPSIS is on, so '...' matches any
            # output; doctest.SKIP would not execute the example at all.
            example.want = "...\n"
    assert seen == set(unchecked), (
        f"UNCHECKABLE names examples that are no longer in doc/{filename}: "
        f"{sorted(set(unchecked) - seen)}"
    )

    runner = doctest.DocTestRunner(
        checker=DocOutputChecker(),
        optionflags=doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL,
        verbose=False,
    )
    runner.run(test)
    return runner.failures, runner.tries


@pytest.mark.parametrize("filename", DOC_FILES)
def test_doc_examples(filename):
    if filename in SLOW_FILES and not os.environ.get("TIBS_DOC_SLOW"):
        pytest.skip(f"{filename} allocates over a gigabyte; set TIBS_DOC_SLOW=1 to run it")

    needed = LARGE_ALLOCATION_FILES.get(filename)
    if needed is not None and needed > BIT_CAPACITY:
        pytest.skip(
            f"{filename} needs {needed:,} bits, more than the {BIT_CAPACITY:,} a "
            f"{POINTER_BITS}-bit build can hold"
        )

    failures, tries = _run(filename)
    assert failures == 0, (
        f"{failures} of {tries} examples in doc/{filename} failed "
        f"(the mismatches are printed above)"
    )


# The docstrings are the API reference, so their examples are documentation too.
DOC_CLASSES = sorted(
    name for name in tibs.__all__ if isinstance(getattr(tibs, name), type)
)
assert DOC_CLASSES, "no classes found to test"


@pytest.mark.parametrize("classname", DOC_CLASSES)
def test_docstring_examples(classname):
    runner = doctest.DocTestRunner(
        checker=DocOutputChecker(),
        optionflags=doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL,
        verbose=False,
    )
    finder = doctest.DocTestFinder(exclude_empty=True)
    for test in finder.find(getattr(tibs, classname), classname, globs=_globals()):
        if test.examples:
            runner.run(test)
    assert runner.failures == 0, (
        f"{runner.failures} of {runner.tries} examples in {classname} docstrings "
        f"failed (the mismatches are printed above)"
    )


def test_exception_lists_do_not_go_stale():
    # A new .rst file is picked up automatically; this guards the reverse, that
    # the lists above don't keep naming files that have been renamed or deleted.
    # UNCHECKABLE's per-example staleness is checked in _run.
    assert SLOW_FILES <= set(DOC_FILES), sorted(SLOW_FILES - set(DOC_FILES))
    assert set(UNCHECKABLE) <= set(DOC_FILES), sorted(set(UNCHECKABLE) - set(DOC_FILES))
    assert set(LARGE_ALLOCATION_FILES) <= set(DOC_FILES), sorted(
        set(LARGE_ALLOCATION_FILES) - set(DOC_FILES)
    )


def test_large_allocation_sizes_are_accurate():
    # The recorded size must be at least the largest literal in the file, or a
    # 32-bit runner would try to run an example that cannot fit.
    literal = re.compile(r"[\d_]{7,}")
    for filename, recorded in LARGE_ALLOCATION_FILES.items():
        text = _read_doc(filename)
        largest = 0
        for example in doctest.DocTestParser().get_examples(text):
            for match in literal.finditer(example.source):
                try:
                    largest = max(largest, int(match.group().replace("_", "")))
                except ValueError:
                    pass
        assert recorded >= largest, (
            f"doc/{filename} now uses {largest:,} bits, more than the "
            f"{recorded:,} recorded in LARGE_ALLOCATION_FILES"
        )
