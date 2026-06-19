#!/usr/bin/env python3
"""Compare two pytest-benchmark JSON files and write a Markdown report."""

import argparse
import json
import math
import os
from pathlib import Path


STATUS_REGRESSION = "REGRESSION"
STATUS_IMPROVEMENT = "IMPROVEMENT"
STATUS_UNCHANGED = "UNCHANGED"


def _load_benchmarks(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    benchmarks = {}
    for benchmark in data.get("benchmarks", []):
        name = benchmark.get("name") or benchmark["fullname"]
        stats = benchmark.get("stats", {})
        median = stats.get("median")
        if median is None:
            median = stats["mean"]
        benchmarks[name] = {
            "name": name,
            "median": float(median),
            "rounds": int(stats.get("rounds", 0)),
        }
    return benchmarks


def _format_seconds(seconds):
    if seconds < 0.000001:
        return f"{seconds * 1_000_000_000:.2f} ns"
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} us"
    if seconds < 1:
        return f"{seconds * 1_000:.2f} ms"
    return f"{seconds:.2f} s"


def _display_name(fullname):
    return fullname.rsplit("::", 1)[-1].replace("test_", "")


def _compare(base, head, threshold):
    rows = []
    common_names = sorted(set(base) & set(head))
    for name in common_names:
        base_median = base[name]["median"]
        head_median = head[name]["median"]
        if base_median <= 0:
            continue

        change = (head_median - base_median) / base_median
        if change >= threshold:
            status = STATUS_REGRESSION
        elif change <= -threshold:
            status = STATUS_IMPROVEMENT
        else:
            status = STATUS_UNCHANGED

        rows.append(
            {
                "status": status,
                "name": name,
                "base_median": base_median,
                "head_median": head_median,
                "change": change,
            }
        )
    return rows


def _geomean_change(rows):
    ratios = [
        row["head_median"] / row["base_median"]
        for row in rows
        if row["base_median"] > 0 and row["head_median"] > 0
    ]
    if not ratios:
        return None
    return math.prod(ratios) ** (1 / len(ratios)) - 1


def _write_report(path, rows, missing_base, missing_head, threshold, base_label, head_label):
    significant = [row for row in rows if row["status"] != STATUS_UNCHANGED]
    regressions = [row for row in significant if row["status"] == STATUS_REGRESSION]
    improvements = [row for row in significant if row["status"] == STATUS_IMPROVEMENT]
    geomean_change = _geomean_change(rows)

    lines = [
        "<!-- tibs-benchmark-report -->",
        "## Tibs Performance Benchmarks",
        "",
        f"Compared `{head_label}` with `{base_label}` using pytest-benchmark median times.",
        f"Significance threshold: `{threshold:.0%}`. Lower times are better.",
        "",
    ]

    if geomean_change is not None:
        lines.append(f"Overall geometric mean change: `{geomean_change:+.1%}`.")
        lines.append("")

    if not significant:
        lines.append("No significant benchmark changes were detected.")
        lines.append("")
    else:
        lines.extend(
            [
                f"Significant regressions: `{len(regressions)}`.",
                f"Significant improvements: `{len(improvements)}`.",
                "",
                "| Status | Benchmark | Base median | PR median | Change |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        significant.sort(
            key=lambda row: (
                row["status"] != STATUS_REGRESSION,
                -abs(row["change"]),
                row["name"],
            )
        )
        for row in significant:
            lines.append(
                "| {status} | `{name}` | {base} | {head} | `{change:+.1%}` |".format(
                    status=row["status"].title(),
                    name=_display_name(row["name"]),
                    base=_format_seconds(row["base_median"]),
                    head=_format_seconds(row["head_median"]),
                    change=row["change"],
                )
            )
        lines.append("")

    if missing_base or missing_head:
        lines.append("Benchmark set changes:")
        for name in sorted(missing_base):
            lines.append(f"- Added in PR: `{_display_name(name)}`")
        for name in sorted(missing_head):
            lines.append(f"- Missing in PR: `{_display_name(name)}`")
        lines.append("")

    lines.append("Raw pytest-benchmark JSON files are uploaded as workflow artifacts.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bool(significant)


def _write_github_output(has_significant_changes):
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as output:
        output.write(
            f"has_significant_changes={str(has_significant_changes).lower()}\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base pytest-benchmark JSON")
    parser.add_argument("--head", required=True, help="Head pytest-benchmark JSON")
    parser.add_argument("--report", required=True, help="Markdown report path")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--head-label", default="PR")
    args = parser.parse_args()

    base = _load_benchmarks(args.base)
    head = _load_benchmarks(args.head)
    rows = _compare(base, head, args.threshold)
    missing_base = set(head) - set(base)
    missing_head = set(base) - set(head)
    has_significant_changes = _write_report(
        args.report,
        rows,
        missing_base,
        missing_head,
        args.threshold,
        args.base_label,
        args.head_label,
    )
    _write_github_output(has_significant_changes)


if __name__ == "__main__":
    main()
