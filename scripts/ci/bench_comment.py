#!/usr/bin/env python3
"""Render the bench-crates PR comparison comment from two Criterion baselines.

Reads every ``<criterion-dir>/**/{base_run,pr_run}/estimates.json`` pair saved
by ``cargo bench -- --save-baseline {base_run,pr_run}`` and prints a GitHub
Markdown comment to stdout: per benchmark, the faster run is bolded, and when
the relative difference exceeds ``NOISE_THRESHOLD`` the Δ cell is colored
(green = PR faster, red = PR slower) via GitHub's client-side math rendering
(``$\\color{..}..$``), the only inline text coloring GitHub Markdown supports.

Usage: bench_comment.py <base-sha> <head-sha> [criterion-dir]

The SHAs (merge target, PR head) are display-only. The first output line is the
HTML marker the workflow's comment step uses to upsert the sticky PR comment.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

MARKER = "<!-- bench-crates -->"

# Same-VM back-to-back legs still drift a few percent (frequency scaling,
# noisy neighbors), more on the rayon-parallel jolt-poly benches.
NOISE_THRESHOLD = 0.10

# (scale in ns, suffix)
UNITS = [(1.0, "ns"), (1e3, "µs"), (1e6, "ms"), (1e9, "s")]

TABLE_HEADER = (
    "| Benchmark | `base_run` (merge target) | `pr_run` (this PR) | Δ |\n"
    "|---|---|---|---|"
)


def load_baseline(criterion_dir: Path, name: str) -> dict[str, tuple[float, float]]:
    """Map benchmark full_id -> (mean, std_dev), both in nanoseconds."""
    out: dict[str, tuple[float, float]] = {}
    for est_path in criterion_dir.rglob("estimates.json"):
        if est_path.parent.name != name:
            continue
        bench = json.loads((est_path.parent / "benchmark.json").read_text())
        est = json.loads(est_path.read_text())
        out[bench["full_id"]] = (
            est["mean"]["point_estimate"],
            est["std_dev"]["point_estimate"],
        )
    return out


def time_cell(mean_ns: float, sd_ns: float, bold: bool) -> str:
    scale, suffix = UNITS[0]
    for candidate in UNITS:
        if mean_ns >= candidate[0]:
            scale, suffix = candidate
    text = f"{mean_ns / scale:.1f}±{sd_ns / scale:.2f}{suffix}"
    return f"**{text}**" if bold else text


def delta_cell(pct: float, color: str | None) -> str:
    # `+ 0.0` turns the -0.0 that rounding a tiny negative leaves into +0.0.
    text = f"{round(pct, 1) + 0.0:+.1f}"
    # Markdown un-escapes backslash-punctuation inside $...$ before math
    # rendering, and a bare % starts a TeX comment — so % needs a double
    # backslash to reach MathJax as \%.
    if color is not None:
        return f"${{\\color{{{color}}}\\mathbf{{{text}\\\\%}}}}$"
    return f"{text}%"


@dataclass
class Row:
    markdown: str
    pct: float
    significant: bool


def render_row(name: str, base: tuple[float, float], pr: tuple[float, float]) -> Row:
    base_mean, base_sd = base
    pr_mean, pr_sd = pr
    pct = (pr_mean - base_mean) / base_mean * 100.0
    significant = abs(pct) > NOISE_THRESHOLD * 100.0
    pr_faster = pr_mean < base_mean
    color = ("green" if pr_faster else "red") if significant else None
    cells = (
        f"`{name.replace('|', chr(92) + '|')}`",
        time_cell(base_mean, base_sd, not pr_faster),
        time_cell(pr_mean, pr_sd, pr_faster),
        delta_cell(pct, color),
    )
    return Row(f"| {' | '.join(cells)} |", pct, significant)


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    base_sha, head_sha = sys.argv[1], sys.argv[2]
    criterion_dir = Path(sys.argv[3]) if len(sys.argv) == 4 else Path("target/criterion")

    base = load_baseline(criterion_dir, "base_run")
    pr = load_baseline(criterion_dir, "pr_run")
    common = sorted(set(base) & set(pr))
    if not common:
        print(f"no benchmark pairs found under {criterion_dir}", file=sys.stderr)
        return 1

    rows = [render_row(name, base[name], pr[name]) for name in common]
    significant = sorted((r for r in rows if r.significant), key=lambda r: -abs(r.pct))

    lines = [
        MARKER,
        "## Benchmark comparison (crates)",
        "",
        f"Same-runner A/B: `base_run` = {base_sha[:9]} (merge target) vs "
        f"`pr_run` = {head_sha[:9]} (PR head merged onto it). "
        f"Bold = faster run; Δ colored when the difference exceeds the "
        f"±{NOISE_THRESHOLD:.0%} noise threshold.",
        "",
    ]
    if significant:
        lines += [
            f"**{len(significant)} of {len(rows)} benchmarks differ by more than "
            f"±{NOISE_THRESHOLD:.0%}:**",
            "",
            TABLE_HEADER,
            *(r.markdown for r in significant),
        ]
    else:
        lines.append(f"No differences above the ±{NOISE_THRESHOLD:.0%} noise threshold.")
    lines += [
        "",
        "<details>",
        f"<summary>All {len(rows)} benchmarks</summary>",
        "",
        TABLE_HEADER,
        *(r.markdown for r in rows),
        "",
        "</details>",
    ]

    only_one = sorted(set(base) ^ set(pr))
    if only_one:
        shown = ", ".join(f"`{n}`" for n in only_one[:10])
        suffix = ", …" if len(only_one) > 10 else ""
        lines += ["", f"_{len(only_one)} benchmark(s) present in only one run: {shown}{suffix}_"]

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
