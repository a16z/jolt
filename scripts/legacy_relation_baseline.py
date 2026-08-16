#!/usr/bin/env python3
"""Legacy per-relation baselines from a legacy chrome trace.

Legacy emits B/E begin/end pairs (not X duration events), so this does a stack
walk and attributes each span by its PARENT. Recursive re-entry is ignored so a
span is counted once (self-recursion would otherwise double count).

Buckets match crates/jolt-prover/src/vertical.rs's drive_rounds: that harness
times prove_round(bind, round) as ONE call, and our kernels bind the PREVIOUS
round's challenge inside it. So harness round r == legacy ingest_challenge[r-1]
+ compute_message[r], which means the bind that crosses a phase boundary is
charged to the LATER phase. Reproduced below.

Two relation models:

1. ROUND-SPLIT (the default): one span owner whose `initialize` /
   `compute_message` / `ingest_challenge` / `cache_openings` map onto
   prepare/cycle/address/claims. Relations with a `phase1_num_rounds = log_T`
   split run log_T cycle rounds then log_K address rounds.

2. EXPLICIT BUCKETS: a list of (bucket, [span names]) for relations that do not
   fit model 1 — several span owners, or buckets that are not the four method
   names. Each bucket sums the INCLUSIVE time of the named spans, so a spec must
   never name both a span and one of its descendants; `check_nesting` enforces
   that, because summing an owner's spans blindly is how the same work gets
   counted twice (see legacy_span_ranking.py's owner totals).

Usage: legacy_relation_baseline.py <trace.json> [--log-t N] [relation ...]
       Relations default to every one with a known span prefix.
"""
import json
import sys
from collections import defaultdict

# relation -> (span prefix, cycle-rounds-first?) for the round-split model.
RELATIONS = {
    "instruction-ra-virtualization": ("InstructionRaSumcheckProver", True),
    "ram-ra-virtualization": ("RamRaVirtualSumcheckProver", True),
    "booleanity-cycle": ("BooleanityCycleSumcheckProver", True),
    "bytecode-read-raf-cycle": ("BytecodeReadRafCycleSumcheckProver", True),
    "ram-read-write": ("RamReadWriteCheckingProver", True),
    "registers-read-write": ("RegistersReadWriteCheckingProver", True),
    "ram-val-check": ("RamValEvaluationProver", True),
    "registers-val-evaluation": ("RegistersValEvaluationProver", True),
}

# relation -> [(bucket, [span names])] for the explicit-bucket model. Bucket
# names are the vertical harness's columns; `address` carries the remainder
# member's prepare for spartan-outer, which has no address phase.
BUCKETED = {
    "spartan-outer": [
        ("prepare", ["OuterUniSkipInstanceProver::initialize"]),
        ("handoff", ["OuterUniSkipInstanceProver::compute_poly"]),
        ("address", ["OuterSharedState::new", "OuterLinearStage::initialize"]),
        (
            "cycle",
            [
                "OuterLinearStage::next_window",
                "OuterLinearStage::ingest_challenge",
                "OuterLinearStage::compute_message",
            ],
        ),
        ("claims", ["OuterLinearStage::cache_openings"]),
    ],
}


def parse(path):
    events = json.load(open(path))
    if isinstance(events, dict):
        events = events.get("traceEvents", events)
    stacks = defaultdict(list)
    inclusive = defaultdict(float)
    instances = defaultdict(list)
    ancestors = defaultdict(set)
    for event in events:
        phase = event.get("ph")
        if phase not in ("B", "E"):
            continue
        key = (event.get("pid"), event.get("tid"))
        if phase == "B":
            stacks[key].append([event.get("name"), event.get("ts")])
            continue
        if not stacks[key]:
            continue
        name, start = stacks[key].pop()
        duration = event.get("ts") - start
        if any(frame[0] == name for frame in stacks[key]):
            continue
        inclusive[name] += duration
        instances[name].append((start, duration))
        ancestors[name].update(frame[0] for frame in stacks[key])
    return inclusive, instances, ancestors


def check_nesting(name, spec, ancestors):
    """Reject a spec that names both a span and one of its ancestors."""
    named = {span for _, spans in spec for span in spans}
    for span in sorted(named):
        overlap = sorted(named & ancestors.get(span, set()))
        if overlap:
            print(
                f"{name}: SPEC ERROR — {span} is nested inside {', '.join(overlap)}, "
                "so its time is counted twice"
            )
            return False
    return True


def report_buckets(name, spec, inclusive, instances, ancestors):
    if not check_nesting(name, spec, ancestors):
        return
    total = 0.0
    lines = []
    for bucket, spans in spec:
        value = sum(inclusive.get(span, 0.0) for span in spans) / 1e3
        missing = [span for span in spans if span not in inclusive]
        counts = sum(len(instances.get(span, [])) for span in spans)
        note = f"   (MISSING: {', '.join(missing)})" if missing else f"   n={counts}"
        lines.append(f"    {bucket:8s} {value:9.1f} ms{note}")
        total += value
    if not any(inclusive.get(span) for _, spans in spec for span in spans):
        print(f"{name}: no spans found")
        return
    print(f"{name}  (explicit buckets)")
    for line in lines:
        print(line)
    print(f"    {'TOTAL':8s} {total:9.1f} ms")


def report(name, prefix, inclusive, instances, log_t):
    messages = sorted(instances.get(f"{prefix}::compute_message", []))
    binds = sorted(instances.get(f"{prefix}::ingest_challenge", []))
    if not messages:
        present = sorted(span for span in instances if span.startswith(f"{prefix}::"))
        if present:
            print(
                f"{name}: compute_message is NOT instrumented, so the round buckets are "
                f"unavailable — present spans: {', '.join(present)}. Recover by subtraction "
                "on the enclosing stage span or by temporary instrumentation; see the "
                "LEGACY PER-RELATION BASELINES section of CLAUDE-NOTES.md."
            )
        else:
            print(f"{name}: no spans found (prefix {prefix}::)")
        return
    total_us = lambda xs: sum(duration for _, duration in xs) / 1e3
    rounds = len(messages)
    if log_t is None or log_t >= rounds:
        cycle = total_us(messages) + total_us(binds)
        address = 0.0
        split = f"{rounds} rounds, unsplit"
    else:
        cycle = total_us(messages[:log_t]) + total_us(binds[: log_t - 1])
        address = total_us(messages[log_t:]) + total_us(binds[log_t - 1 :])
        split = f"{log_t} cycle + {rounds - log_t} address"
    prepare = inclusive.get(f"{prefix}::initialize", 0.0) / 1e3
    claims = inclusive.get(f"{prefix}::cache_openings", 0.0) / 1e3

    print(f"{name}  ({split})")
    print(f"    prepare  {prepare:9.1f} ms")
    print(f"    cycle    {cycle:9.1f} ms")
    print(f"    address  {address:9.1f} ms")
    print(f"    claims   {claims:9.1f} ms" + ("" if claims else "   (span not instrumented)"))
    print(f"    TOTAL    {prepare + cycle + address + claims:9.1f} ms")


def main():
    args = [a for a in sys.argv[1:]]
    path = args.pop(0)
    log_t = None
    if "--log-t" in args:
        index = args.index("--log-t")
        log_t = int(args[index + 1])
        del args[index : index + 2]
    known = list(RELATIONS) + list(BUCKETED)
    wanted = args or known
    inclusive, instances, ancestors = parse(path)
    for name in wanted:
        if name in BUCKETED:
            report_buckets(name, BUCKETED[name], inclusive, instances, ancestors)
        elif name in RELATIONS:
            prefix, _ = RELATIONS[name]
            report(name, prefix, inclusive, instances, log_t)
        else:
            print(f"unknown relation {name}; known: {', '.join(known)}")


if __name__ == "__main__":
    main()
