# Akita Metal T28 memory lifetime

## Decision

Akita will not retain complete packed-opening indices across Jolt's PIOP when
their exact size would exceed a caller-supplied retention budget. The commit
will retain the immutable packed one-hot lanes and enough public geometry to
build each index at its opening consumer. Jolt's proof protocol, transcript,
commitment, PIOP schedule, and verifier remain unchanged.

The first implementation moves index construction without changing either
index layout or kernel. Chunked construction is reserved for a measured
opening-phase memory failure; it is not part of the initial patch.

## Boundary and invariants

The device still builds and consumes each index exactly once. Deferral changes
only buffer lifetime, so its compulsory index traffic and computation are the
same as the eager route. Index construction must be charged to the complete
evaluation-proof cost even if it is scheduled outside the opening span.

The packed source is immutable after commitment. Backend choice and retention
policy are prover-local and are not transcript-bound. A qualified Metal route
must return an error rather than silently select a CPU implementation.

## T28 residency model

For `T = 2^28`, 30 live K256 columns, D512, `P = 2^18`, and 512 blocks per
column, the commit currently retains both one-shot indices:

| Buffer | Exact bytes | GiB |
|---|---:|---:|
| Fold records and counts | 32,715,571,200 | 30.469 |
| Coefficient-packing records and offsets | 18,182,307,840 | 16.934 |
| Combined indices | 50,897,879,040 | 47.402 |

The failed stage-1 run reached 126.75 GiB of Metal allocations against a
107.52 GiB recommended working set. Removing the two indices from that phase
predicts 79.35 GiB, leaving 28.17 GiB of headroom. This is a phase-residency
calculation, not a prediction of process RSS or opening latency.

The 7.5 GiB packed source remains live because stage 8 consumes it. PIOP's
resident rows, shift state, and outer workspace keep their existing ownership
and reuse schedule.

## Mechanism

Akita exposes an exact retained-byte estimate and accepts an opening
acceleration retention budget. Its automatic policy eagerly retains both
indices only when the estimate fits that budget; otherwise it stores two
zero-sized, one-shot markers with the packed source. The opening hint already
carries the configured backend, and the packed source plus opening plan contain
all geometry needed to rebuild each index. A consumer removes its marker,
builds and uses that index, waits for the consuming command, and releases it
before another index is allocated.

Jolt supplies a 32 GiB retained-index budget. The exact combined estimate fits
through T27 (23.70 GiB) and exceeds the budget at T28 (47.40 GiB), so only the
max-scale path defers construction. The existing proof-session drop remains the
boundary between PIOP residency and deferred opening allocation. A later
device-wide reservation ledger may replace the caller budget, but is not
required to test this decision.

## Falsification and acceptance

The patch is rejected if a verified T28 BTreeMap proof does not complete with
Metal commit, PIOP, and opening routes; if proof bytes, transcript, claimed
evaluation, or verifier output differ from CPU; or if any qualified operation
uses an unplanned CPU route.

The memory gate is at most 96 GiB of Metal allocations, at most 90 GiB peak
RSS, and no swap growth. Index construction is reported separately and added
to the evaluation-proof cost. T25 and T27 complete-proof time may not regress
by more than 3%. The first T28 performance gate is no slower than the prior
70.62-second verified combined-Metal proof.

The exact device-only stage-8 residency peak remains unmeasured. If later
device telemetry exceeds the memory gate, construction will be partitioned by
position with a 2 GiB temporary-buffer budget.

## Measured result

On 2026-08-23, the full modular BTreeMap proof at `T = 2^28` and a 150,000,000
cycle target completed and verified with the production Metal backend:

```text
cargo run --release -p jolt-prover --example modular_benchmark \
  --features prover-fixtures,metal -- \
  --name btreemap --scale 28 --target-trace-size 150000000 --backend metal
```

The prover took 56.52 seconds and reached 80.08 GiB peak RSS. Commit telemetry
reported zero retained opening-index bytes. Opening telemetry reported the
exact 50,897,879,040 deferred bytes, 2.9999 seconds of index construction, and
one indexed fold call. The proof verifier accepted. `/usr/bin/time -l` reported
zero process swaps. Host-wide swap use moved from 3.69 MiB to 6.44 MiB during
the run, so the stricter host-global no-growth sentinel remains inconclusive.

This is 20.0% faster than the prior 70.62-second combined-Metal run and below
the 90 GiB RSS guard by 9.92 GiB. The same-day optimized CPU control was 166.55
seconds, making the full-prover ratio 2.95x; this change fixes max-scale
residency but does not by itself establish a 5x end-to-end proof ratio. Opening's
62.21 GiB `allocation_bytes` counter is cumulative traffic, not a simultaneous
device-residency peak.
