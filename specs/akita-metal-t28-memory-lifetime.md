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
indices only when the estimate fits that budget; otherwise it retains no
index. The opening hint already carries the configured backend, and the packed
source plus opening plan contain all geometry needed to rebuild the index.
Each packed opening consumer builds its index, uses it, waits for the consuming
command, and releases it before another index is allocated.

Jolt supplies a zero retained-index budget for the composed max-scale path.
The existing proof-session drop remains the boundary between PIOP residency
and deferred opening allocation. A later device-wide reservation ledger may
replace the caller budget, but is not required to test this decision.

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

Unverified before implementation: whether Metal releases both private index
buffers immediately after their final command, and the exact stage-8 peak
after moving construction. If either fails the memory gate, construction will
be partitioned by position with a 2 GiB temporary-buffer budget.
