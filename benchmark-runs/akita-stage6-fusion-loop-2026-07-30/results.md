# Akita Stage 6 fusion results

Date: 2026-07-30 EDT

## Outcome

One engineering optimization landed as `8f27504a5`: shared RA polynomials now
materialize Round 3 in coordinated source-index blocks. On the exact
`T = 2^28`, K256, D128 workload:

| Metric | Accepted parent | Blocked Round 3 | Change |
|---|---:|---:|---:|
| `SharedRaRound3::bind` | 2.348 s | 1.836 s | -21.83% |
| Stage 6b | 21.581 s | 20.644 s | -4.34% |
| Prover | 193.699 s | 189.876 s | -1.97% |
| Peak RSS | 81.080 GiB | 80.781 GiB | -0.299 GiB |
| Swaps | 0 | 0 | unchanged |

The proof verified. The source-derived peak is unchanged: the candidate
allocates the same `T/8` output polynomials at the same round. Its only new
state is an `O(number of polynomials)` pointer table.

Not all of the 3.823-second whole-prover movement is attributable to this
change. Unmodified commitment and evaluation-proof spans improved by 2.53
seconds in the pair. The defensible direct gain is the 0.513-second
materialization reduction; the surrounding Stage-6b aggregate improved by
0.937 seconds.

At `2^26`, two candidate runs measured 0.458 and 0.441 seconds for the same
three materializations. Three retained controls span 0.503–0.510 seconds.
This repeated smaller target established the direction before promotion.

## Why blocking helps

The old schedule assigned whole polynomials to Rayon tasks. Each polynomial
streamed the entire 54-byte `RaIndices` source to emit one field column.
Concurrent scans can share cache lines when their schedules happen to align,
but alignment is not guaranteed as tasks steal work.

The accepted schedule assigns bounded output-index blocks to tasks. For one
block it keeps the eight required source rows per output in a roughly 1.7 MiB
window, then emits the corresponding contiguous slice for every polynomial.
It advances only after all readers have consumed that source window. Output
bytes and arithmetic are unchanged; the improvement comes from predictable
source reuse and less cache churn.

## Rejected candidates

### Arithmetic zero collapse

A diagnostic over `2^20` cycles found plausible first-three-round product
headroom:

| Source width | Realistic product-count reduction |
|---:|---:|
| 2 cycles | 27.49% |
| 4 cycles | 17.72% |
| 8 cycles | 11.10% |

The implementation skipped exact zero pairs and combined lane-zero terms
algebraically. Although sound and proof-equivalent, the added field equality
checks and branches made Booleanity messages slower:

| Run | Messages | Challenge ingestion | Combined |
|---|---:|---:|---:|
| Control | 96.401 ms | 57.628 ms | 154.029 ms |
| Candidate A | 100.146 ms | 57.830 ms | 157.976 ms |
| Candidate B | 98.270 ms | 59.945 ms | 158.215 ms |

This rejects per-element field classification, not all source-aware
specialization. A future attempt needs byte-level masks or a layout that
exposes zero runs without extra field comparisons.

### Base-plus-increment pushforward fusion

Combining the nine dense increment columns with the base RA split-eq kernel
was slower at both tested sizes. At `2^26`, the fused function took 2.204
seconds. The retained base-only controls take 1.028–1.132 seconds, and the
separate dense increment scatter is only a small addition.

The split-eq kernel's touched sets and delayed reductions are valuable for the
sparse/optional base RA rows. Dense always-present columns prefer the existing
direct scatter. The predicted large-trace crossover did not occur, so this
candidate was restored.

## Validation

- shared-RA equivalence tests in both binding orders;
- Akita muldiv tests;
- standard host muldiv tests;
- host+ZK muldiv tests;
- warning-denying all-target clippy with `host`, `host,zk`, and Akita;
- formatting and diff checks;
- exact `2^28` proof verification with zero swaps.

## Traces

- Control: `benchmark-runs/perfetto_traces/akita_stage6_fusion_control_22.json`
- Rejected zero collapse:
  `akita_stage6_zero_collapse_22a.json`,
  `akita_stage6_zero_collapse_22b.json`
- Rejected pushforward fusion:
  `akita_stage6_pushforward_fusion_22a.json`,
  `akita_stage6_pushforward_fusion_26.json`
- Accepted blocked materialization:
  `akita_stage6_round3_blocked_26.json`,
  `akita_stage6_round3_blocked_26b.json`,
  `akita_stage6_round3_blocked_28.json`
