#!/usr/bin/env python3
"""Busy-thread occupancy over time for chosen span names (leaf busy = deepest span)."""
import json, sys, collections
path, target = sys.argv[1], sys.argv[2]
events = json.load(open(path))
if isinstance(events, dict): events = events["traceEvents"]
# find target window (first span whose name contains target)
win = None
stacks = collections.defaultdict(list)
intervals = []  # (start,end) of ANY leaf span on a thread
for ev in sorted([e for e in events if e.get("ph") in "BE"], key=lambda e: e["ts"]):
    key=(ev.get("pid"),ev.get("tid")); ts=ev["ts"]
    if ev["ph"]=="B":
        stacks[key].append((ev.get("name","?"),ts))
    else:
        if not stacks[key]: continue
        name,start = stacks[key].pop()
        if target in name and win is None: win=(start,ts)
        if not stacks[key]:  # top-level busy interval on this thread
            intervals.append((start,ts))
if win is None: sys.exit(f"no span containing {target!r}")
w0,w1=win; dur=(w1-w0)/1e6
print(f"window {target}: {dur:.1f}s")
BUCKETS=40
hist=[0.0]*BUCKETS
for s,e in intervals:
    s,e=max(s,w0),min(e,w1)
    if e<=s: continue
    b0=int((s-w0)/(w1-w0)*BUCKETS); b1=int((e-w0)/(w1-w0)*BUCKETS)
    for b in range(b0,min(b1+1,BUCKETS)):
        bs=w0+b*(w1-w0)/BUCKETS; be=bs+(w1-w0)/BUCKETS
        hist[b]+=max(0,min(e,be)-max(s,bs))
step=dur/BUCKETS
for i,h in enumerate(hist):
    busy=h/1e6/step
    print(f"{i*step:6.1f}s {'#'*int(busy)} {busy:.1f}")
