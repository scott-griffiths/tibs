#!/bin/bash
# Rust coverage for Tibs, driven by the Python test suite.
#
# Builds the extension module with LLVM source-based instrumentation, runs
# pytest against it, and reports which Rust lines and functions the Python
# tests actually reach. Output goes under target/, which is already ignored.
# The normal release extension is rebuilt into .venv on the way out.
#
# Usage: ./coverage.sh [output-dir]
#
# Requires:
#   cargo install cargo-llvm-cov rustfilt
#   rustup component add llvm-tools
set -e

cd "$(dirname "$0")"
OUT=${1:-target/coverage}

for tool in cargo-llvm-cov rustfilt; do
    if ! command -v $tool &> /dev/null; then
        echo "$tool not found. Install with: cargo install cargo-llvm-cov rustfilt" >&2
        exit 1
    fi
done

# Whatever happens below, put the usual release build back in .venv so a failed
# or interrupted run does not leave an instrumented debug extension installed.
restore() {
    echo "==> restoring the release build"
    unset RUSTFLAGS LLVM_PROFILE_FILE CARGO_TARGET_DIR \
          CARGO_LLVM_COV CARGO_LLVM_COV_SHOW_ENV CARGO_LLVM_COV_TARGET_DIR
    # Instrumented build scripts run during the report step with no profile
    # path set, and drop default_*.profraw in the working directory.
    rm -f default_*.profraw
    .venv/bin/maturin develop --release > /dev/null
}
trap restore EXIT

mkdir -p "$OUT"

# A separate target dir keeps the normal build cache intact, and must be set
# before show-env so the profraw files land alongside it.
#
# The build is deliberately unoptimised: with --release, small functions are
# inlined and their out-of-line copies read zero, which looks like dead code.
# The suite still runs in about 15 seconds at opt-level 0.
export CARGO_TARGET_DIR="$PWD/$OUT/target"
eval "$(cargo llvm-cov show-env --export-prefix)"
cargo llvm-cov clean --workspace

echo "==> building instrumented extension"
.venv/bin/maturin develop > /dev/null

echo "==> running tests"
.venv/bin/python -m pytest --benchmark-skip -q | tail -2

echo "==> reports"
cargo llvm-cov report
cargo llvm-cov report --html --output-dir "$OUT/html" > /dev/null
cargo llvm-cov report --json --output-path "$OUT/cov.json" > /dev/null

.venv/bin/python - "$OUT/cov.json" "$PWD/src/" <<'PYEOF' | rustfilt
import collections
import json
import sys

data = json.load(open(sys.argv[1]))["data"][0]["functions"]
# Dependencies have a src/ directory too, so match on the full project path.
root = sys.argv[2]

# The same function shows up as several records, so key by name and keep the
# highest count: a function is only uncovered if every record reads zero.
functions = collections.defaultdict(lambda: [0, None, 0])
for f in data:
    source = f["filenames"][0] if f["filenames"] else ""
    if not source.startswith(root):
        continue  # dependency, not our code
    entry = functions[f["name"]]
    entry[0] = max(entry[0], f["count"])
    entry[1] = "src/" + source[len(root):]
    entry[2] = f["regions"][0][0] if f["regions"] else 0

# PyO3 generates a ___pymethod_* trampoline per method, all attributed to the
# #[pymethods] line and all reading zero. They are noise; the method bodies
# sitting next to them are reported separately and truthfully.
uncovered = sorted((v[1], v[2], name) for name, v in functions.items()
                   if v[0] == 0 and "__pymethod" not in name)

print(f"\nnever executed: {len(uncovered)} of {len(functions)} functions\n")
for source, line, name in uncovered:
    print(f"  {source}:{line}  {name}")
PYEOF

echo
echo "HTML report: $OUT/html/index.html"
