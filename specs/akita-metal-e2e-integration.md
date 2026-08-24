# Akita Metal end-to-end integration

The protocol-facing delta ledger for the combined commit and evaluation-proof
work is [akita-metal-protocol-changes.md](akita-metal-protocol-changes.md). This
document records the commit-backend checkpoint; statements below that the
protocol is unchanged mean that CPU and Metal use the same fixed protocol and
schedule, not that later Metal-motivated protocol revisions are absent.

## Status

The combined Metal prover passes the commit-phase acceptance gate on the
standard Jolt workloads through `T = 2^28`. One `JoltAkitaBackend::metal()`
selects both the existing Metal PIOP kernels and Akita's fp128 packed one-hot
commit backend. The commit backend itself does not change verifier equations or
proof messages.

Jolt pins the Akita fork revision
`30e99fed9a2885bcc66d8f20693b81fe6d4374e4` from branch
`perf/metal-commit-jolt-production-final`.

## Architecture

Preprocessing derives the canonical Akita schedule from the public
`OneHotTrace` shape. For qualified K256 shapes, it also prepares the Metal
setup, prewarms the D512 rank-one root matrix, and allocates the aligned packed
stream buffer. These costs remain outside proving and are reported separately.

During stage 0:

1. Jolt derives the same canonical prefix-packing plan used by the verifier.
2. A Jolt row generator writes selected-row bytes and committed-zero masks
   directly into the Akita-owned stream buffer. There is no full-size adapter
   copy.
3. Akita overlaps row production with the D512 root commit. The Metal backend
   processes the qualified root work; the bounded CPU portion handles work
   retained by the hybrid schedule. Outer commitments, compression, and proof
   construction remain on the host.
4. The packed owner is retained in the opening hint, so stage 8 reuses the
   committed source. Rebuildable transformed setup residency is released after
   the commitment.
5. At the largest qualified shape, the backend asks Jolt to prioritize stream
   generation. PIOP witness preparation starts when the final populated row is
   generated and then overlaps the remaining device commit. Smaller shapes keep
   the original full overlap.

Stages 1 through 8 continue through the existing `jolt-kernels` Metal backend.
Their cycle-major layout and protocol configuration do not change.

The division of responsibility is:

| Layer | Responsibility |
|---|---|
| Akita | fp128 arithmetic, canonical schedule, D512 Metal kernel, hybrid split, prepared setup, stream storage, commitment and hint |
| `jolt-akita` | PCS adapter, shape admission, fail-closed routing, committed-zero semantics, stream API, resource scheduling hint |
| `jolt-prover` | canonical trace plan, row-fill closure, stage-0 orchestration, PIOP/commit overlap |
| verifier | existing canonical schedule replay and proof verification |

## Invariants

- Backend choice is prover-local and is not transcript-bound. CPU and Metal use
  the same canonical Akita schedule and verifier statement.
- Qualified shapes are exactly K256 with packed dimensions 38 through 41.
  `metal_required` fails closed for qualified work; an unqualified shape uses
  the CPU route by design.
- Hybrid CPU work is part of the selected Metal algorithm, is timed inside the
  commit span, and is not a fallback.
- Selected row zero remains distinct from an omitted coefficient through the
  per-row committed-zero mask.
- The source stays cycle-major. No transpose or PIOP layout change is required.
- The measured commit span includes row generation, synchronization, complete
  Akita commit, readback, CPU hybrid work, and result assembly.

## Accepted performance

All measurements use the same machine, release build, fp128 field, canonical
schedule, tracing format, and workload input. Every listed proof verified. The
CPU implementation and evaluator are unchanged between the frozen CPU anchors
and the final Akita revision; intervening Akita commits touch only Metal code
and its benchmark.

| Workload | Trace | Columns | CPU commit | Metal commit | Speedup |
|---|---:|---:|---:|---:|---:|
| Fibonacci | `2^25` | 29 | 13.60 s | 1.98 s | 6.87x |
| SHA-2 chain | `2^25` | 29 | 14.90 s | 2.22 s | 6.71x |
| SHA-3 chain | `2^25` | 29 | 16.30 s | 2.64 s | 6.17x |
| BTreeMap | `2^25` | 30 | 8.54 s | 1.46 s | 5.85x |
| BTreeMap, 150M target cycles | `2^28` | 30 | 74.70 s | 13.30 s | 5.62x |

The `T = 2^28` workload contained 253,779,321 populated rows in a 268,435,456
row domain. Its commit metrics were:

- 2,023,057,407 hot entries;
- 235 Metal blocks and 8 concurrent CPU blocks;
- 12.908 s GPU time and 8.268 s CPU time, with the CPU portion overlapped;
- 12.966 s backend time and a 13.30 s complete stage-0 commit span;
- zero-copy stream input and a prepared-matrix cache hit;
- 82.44 GiB peak RSS, below the 90 GiB project guard.

The final full-prover CPU/Metal comparison was 180.62/70.62 seconds at
`T = 2^28`, or 2.56x end to end. The four `T = 2^25` end-to-end ratios ranged
from 2.06x to 2.58x.

The faster commitment did not reduce the max-scale wall clock relative to an
older 70.78-second combined-Metal run. The later PIOP spans varied from 32.5 to
35.3 seconds even though their GPU-active intervals were stable. For example,
stage 1's summed command-wall-minus-GPU-active interval grew from 0.96 to 1.65
seconds, and the stage-2 product output took 1.62 versus 1.88 seconds with the
same approximately 80 ms GPU-active interval. The observed difference occurs
after stage 0; current telemetry cannot distinguish command scheduling,
readback, and memory residency within those PIOP spans. This does not change
the complete-commit ratio.

## Validation

The retained credibility gate is:

1. Akita's 18 focused Metal parity/routing tests and all 44 `jolt-akita` tests,
   plus all-target clippy and formatting.
2. Jolt's K16 and forced-K256 CPU end-to-end proofs, including the committed
   program path.
3. One verified Metal proof for each standard `T = 2^25` workload.
4. One verified `T = 2^28` BTreeMap proof with the 150M-cycle target.
5. A second max-scale run after production cleanup: 13.30-second commit,
   70.62-second prover, verified proof.

Search-only kernel variants, environment overrides, raw logs, and the autonomous
research goal are not production artifacts.

## Upstream slices

Suggested review order after the Jolt Akita substrate lands:

1. Akita: generic fp128 D512 Metal root backend, exact parity tests, prepared
   setup/cache lifecycle, streaming packed-one-hot owner, and one focused
   benchmark.
2. Jolt Akita adapter: canonical adaptive schedules, committed-zero support,
   streaming PCS seam, shape admission, and the resource scheduling hint.
3. Jolt prover: compose the PIOP and commit backends, stream stage-0 rows, and
   apply the max-shape producer-priority policy.
4. Benchmark/documentation: the small reproducible workload gate and accepted
   evidence table, kept separate from protocol review.

Protocol-facing changes should remain isolated from kernel-heavy changes so the
schedule, transcript, zero-lane semantics, and verifier invariants can be
reviewed independently.
