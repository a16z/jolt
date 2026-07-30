# Akita `2^28` rank, time, and RSS loop contract

Created: 2026-07-30 EDT

## Question and baseline

The first question is whether root rank `n_a = 7` for the D64, K256,
41-variable packed polynomial is genuinely required by the current protocol
and SIS security profile, or is an artifact of incomplete planner search or a
suboptimal fixed policy. The second phase admits one change at a time from the
largest remaining `2^28` trace or analytical-memory opportunity.

Accepted parent revisions:

- Jolt: `df0b7c5cc`
- Akita dependency: `67c3c88d`

Frozen target baseline:

- trace: `benchmark-runs/perfetto_traces/akita_28_relation.json`
- log:
  `benchmark-runs/akita-superlinear-2e28-2026-07-30/logs/akita_28_relation.log`
- commitment: 116.502160 s
- commitment accumulation: 113.542885 s
- evaluation proof: 22.586504 s
- whole prover: 219.681956 s
- maximum RSS: 85,299,347,456 bytes (79.441 GiB)
- proof verified, zero process swaps, unchanged system swapout counter

The three adjacent full-size commitment observations are 117.99, 115.48, and
116.50 seconds. Whole-prover time is noisier than the localized spans. A
candidate is screened on its directly affected span and is not credited with
unrelated whole-prover movement.

## Metrics and guards

For time candidates, the primary metric is the directly affected inclusive
Perfetto span, minimized. Promotion to `2^28` requires at least a 3% localized
screening gain, proof verification, no increase in analytical peak live bytes,
and no reproducible RSS or unaffected-span regression.

For memory candidates, the primary metric is analytically live bytes per cycle,
minimized. Promotion requires at least 1 GiB (4 B/cycle at `2^28`) removed from
the current peak or a newly tighter universal-trace ceiling, with no added full
pass over `T`, no extra asymptotic work, and no reproducible localized prover
regression. Allocator-only RSS movement without an ownership delta is not
credited.

All target runs must remain below 90 GiB, report zero process swaps, and leave
the system swapout counter unchanged. Any proof or transcript change must pass
the corresponding prover/verifier tests before benchmarking.

## Evaluator

The immutable end-to-end workload is the ignored forced-K256
`sha2_chain_akita_perf` test:

```text
PERF_LOG_T=28 PERF_TRACE=1 /usr/bin/time -l \
  cargo nextest run --release -p jolt-prover-legacy --features akita \
  -E 'test(sha2_chain_akita_perf)' --run-ignored all --no-capture --cargo-quiet
```

Small screens use the same test and binary shape at `PERF_LOG_T=25` or
`PERF_LOG_T=26`. The test source, trace parser, existing controls, and workload
are frozen. Raw stdout, RSS samples, and traces are retained per trial.

## Rank-audit falsifiers

The claim “rank 7 is required under the current D64 protocol” is falsified by
any fully valid 41-variable schedule with root rank at most 6 under the exact
production SIS table and challenge policy. The audit must cover:

1. every legal root block split;
2. alternative fixed root gadget bases, not only the shipped basis 3;
3. payload-slack selection versus raw root feasibility;
4. the root-response norm and SIS table cell that excludes rank 6;
5. any exact packed-support fact that could tighten that norm without changing
   soundness.

A forced unchecked rank is not evidence. A changed challenge family, ring
dimension, norm theorem, or response range is a protocol candidate and must be
analyzed separately.

## Search and budget

The search is a greedy, one-parent hill climb with one hypothesis per trial.
Planner/security analysis comes first. Runtime candidates are ordered by
expected seconds or bytes removed per implementation hour. Ties prefer the
smaller diff and narrower invariant surface.

- wall-clock budget: 10,800 seconds
- candidate budget: 8
- reserve: the last 2 trials are held for revalidation
- compute: local Apple M4 Max only
- external services and monetary spend: none
- stop: budget exhausted, two consecutive low-signal regions, a guard failure
  that invalidates the evaluator, or insufficient time for clean rollback

Every attempted candidate appends one ledger event. A discarded candidate is
removed before the next starts. Each accepted implementation is committed
separately.
