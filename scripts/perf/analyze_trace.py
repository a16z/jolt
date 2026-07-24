#!/usr/bin/env python3
"""Rank spans in a tracing-chrome perfetto trace by inclusive/exclusive time."""
import json, sys, collections

path = sys.argv[1]
events = json.load(open(path))
if isinstance(events, dict):
    events = events["traceEvents"]

# tracing-chrome emits B/E pairs per (pid, tid)
stacks = collections.defaultdict(list)
incl = collections.Counter()   # name -> inclusive us
excl = collections.Counter()   # name -> exclusive us
count = collections.Counter()
child_time = collections.defaultdict(float)
thread_busy = collections.Counter()
t_min, t_max = float("inf"), 0.0

for ev in events:
    ph = ev.get("ph")
    if ph not in ("B", "E"):
        continue
    key = (ev.get("pid"), ev.get("tid"))
    ts = ev["ts"]
    t_min = min(t_min, ts); t_max = max(t_max, ts)
    if ph == "B":
        stacks[key].append([ev.get("name", "?"), ts, 0.0])  # name, start, child accum
    else:
        if not stacks[key]:
            continue
        name, start, child = stacks[key].pop()
        dur = ts - start
        incl[name] += dur
        excl[name] += dur - child
        count[name] += 1
        if stacks[key]:
            stacks[key][-1][2] += dur
        else:
            thread_busy[key] += dur

wall = (t_max - t_min) / 1e6
print(f"wall span: {wall:.1f}s   threads seen: {len(thread_busy) or len(stacks)}")
print(f"\n{'INCLUSIVE s':>12} {'EXCL s':>10} {'count':>8}  name")
for name, us in incl.most_common(40):
    print(f"{us/1e6:12.2f} {excl[name]/1e6:10.2f} {count[name]:8d}  {name[:110]}")
