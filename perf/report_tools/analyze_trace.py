#!/usr/bin/env python3
"""Aggregate a Chrome/Perfetto trace into per-span stats.

Input: path to trace-chrome JSON (array of {ph, name, ts, tid, pid, ...}).
Output: JSON with per-span {count, total_us, self_us, parent_counts}.

Self time = total time - time spent in child spans (on the same tid).
"""

import json
import sys
from collections import defaultdict, Counter


def aggregate(trace_path):
    with open(trace_path) as f:
        events = json.load(f)

    # Per-tid stack of (name, ts, child_time_us)
    stacks = defaultdict(list)

    # name -> {count, total_us, self_us}
    stats = defaultdict(lambda: {"count": 0, "total_us": 0.0, "self_us": 0.0})

    # parent name -> Counter(child name -> count) for structural navigation
    parent_children = defaultdict(Counter)

    for e in events:
        ph = e.get("ph")
        if ph == "B":
            tid = e.get("tid", 0)
            name = e.get("name", "?")
            ts = e.get("ts", 0.0)
            stacks[tid].append([name, ts, 0.0])
        elif ph == "E":
            tid = e.get("tid", 0)
            ts = e.get("ts", 0.0)
            if not stacks[tid]:
                continue
            name, start_ts, child_us = stacks[tid].pop()
            dur = ts - start_ts
            self_us = dur - child_us
            stats[name]["count"] += 1
            stats[name]["total_us"] += dur
            stats[name]["self_us"] += self_us
            # Credit parent's child_time
            if stacks[tid]:
                stacks[tid][-1][2] += dur
                parent = stacks[tid][-1][0]
                parent_children[parent][name] += 1

    return stats, parent_children


def to_ms(us):
    return us / 1000.0


def summarize(stats, parent_children, top_n=30):
    rows = []
    for name, s in stats.items():
        rows.append(
            {
                "name": name,
                "count": s["count"],
                "total_ms": to_ms(s["total_us"]),
                "self_ms": to_ms(s["self_us"]),
                "mean_total_us": s["total_us"] / s["count"] if s["count"] else 0.0,
            }
        )
    rows.sort(key=lambda r: r["self_ms"], reverse=True)
    return rows


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_trace.py <trace.json> [out.json]", file=sys.stderr)
        sys.exit(1)
    stats, parent_children = aggregate(sys.argv[1])
    rows = summarize(stats, parent_children)

    # Top total self time
    total_self_ms = sum(r["self_ms"] for r in rows)

    out = {
        "total_self_ms": total_self_ms,
        "distinct_spans": len(rows),
        "spans": rows,
        "parent_children": {p: dict(cs) for p, cs in parent_children.items()},
    }

    if len(sys.argv) >= 3:
        with open(sys.argv[2], "w") as f:
            json.dump(out, f, indent=2)
    else:
        json.dump(out, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
