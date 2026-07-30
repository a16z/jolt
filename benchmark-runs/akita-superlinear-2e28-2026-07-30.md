# Akita `2^28` super-linear scaling experiments

Date: 2026-07-30 EDT

## Fixed evaluator

Control traces:

- `benchmark-runs/perfetto_traces/mem-stage5-reuse-2e26.json`
- `benchmark-runs/perfetto_traces/akita_28.json`

All target runs use the ignored `sha2_chain_akita_perf` harness with K256 and
retain a Perfetto trace. Correctness requires the proof to verify. Capacity
requires maximum RSS below 90 GiB and no increase in the system swapout
counter.

The performance metrics are inclusive Perfetto spans:

- commit: `AkitaCommitmentScheme::commit`,
  `TracePackedOneHot::commit_inner`, and
  `trace_onehot_commit_accumulate`;
- evaluation: `AkitaCommitmentScheme::batched_prove`,
  `TracePackedOneHot::decompose_fold`, root `ring_switch_build_w`, root
  `compute_multi_group_relation_quotient`, and root `digit_range_prove`;
- PIOP guard: the sum of `prove_stage1` through `prove_stage7_lattice`.

The evaluator and control traces are read-only. Each experiment changes one
implementation surface, screens with unit/e2e tests, and gets at most one
full `2^28` target run before a keep/discard decision.

## Experiment A: dense root rotations

Question: does the 64 MiB dense-rotation budget, rather than intrinsic
opening work, cause the `2^28` packed root decomposition cliff?

Minimal change: raise the bounded auto-selection budget just enough to admit
the production `D=64`, K256, `2^28` table:

```text
29 columns * 1024 blocks/column * 64 rotations * 128 bytes
  = 243,269,632 bytes
```

Expected:

- `dense_rotations=true`;
- root `TracePackedOneHot::decompose_fold` at most 3 seconds, versus
  13.15 seconds;
- total evaluation proof at most 26 seconds, versus 35.38 seconds;
- less than 0.5 GiB additional maximum RSS.

Falsifying outcome: less than 5 seconds total evaluation improvement, maximum
RSS at or above 90 GiB, any swapout, or a correctness failure.

## Experiment B: commit scaling

Question: after accounting for the root schedule change, which commit kernel
causes the remaining efficiency loss?

The schedule changes from:

```text
T=2^26: P=2^21, B=4,096,  n_a=6
T=2^28: P=2^20, B=32,768, n_a=7
```

The first discriminator is a schedule/work ledger, followed by the smallest
available schedule or task-geometry ablation. A candidate must improve
`trace_onehot_commit_accumulate` by at least 5% at `2^28` without regressing
commit RSS or proof correctness. Maximum budget: three small screens and one
full target run for the selected candidate.

### Schedule discriminator

The K256 planner was evaluated for packed arity 41 under root-rank payload
slack from 0 through 100,000 permille. Zero slack selects rank 8 with
`P=2^22`, 8,192 blocks, and a 102,504-byte estimated proof. The shipped
10-permille policy selects rank 7 with `P=2^20`, 32,768 blocks, and a
102,924-byte proof. Every larger slack value, including 100x the minimum
payload, still selects rank 7. Rank 6 is not a feasible candidate under the
current security profile, so forcing the `2^26` rank would bypass rather than
tune the planner. This schedule direction is rejected without a prover run.

### Candidate B1: share A loads across trace blocks

The rank-tiled K256 kernel currently assigns one trace block per task. At
`2^28`, 1,024 trace blocks each scan the same seven-rank, `P=2^20` A prefix:
about 7 GiB per block and 7 TiB in aggregate. A block cohort keeps independent
destination accumulators but loads and widens each A ring once for several
blocks at the same local position. The trace rows remain block-local and the
same negacyclic shift is added to every `(block, column, rank)` output.

The first screen uses the production K256 `2^25` workload, where the same
kernel is already active and the run is cheap enough for a controlled
baseline/candidate pair. It advances only if commitment accumulation improves
by at least 5%, proof verification passes, and RSS does not regress
materially. A passing screen receives one exact `2^28` run.

The cohort-four screen is rejected:

| `2^25` metric | Control | Cohort 4 | Change |
|---|---:|---:|---:|
| Commit accumulation | 10.071707 s | 10.945904 s | +8.68% |
| Whole commitment | 10.513429 s | 11.390572 s | +8.34% |
| Prover | 26.98 s | 27.91 s | +0.93 s |
| Maximum RSS | 20.298 GiB | 20.247 GiB | -0.051 GiB |

Both proofs verified and neither process swapped. The larger live
wide-accumulator and decoded-row working set costs more than the shared A
conversion saves. The retained negative trace is
`benchmark-runs/perfetto_traces/commit_cohort4_25.json`. Cohort two is the
last adjacent point: it retains 64 tasks and halves rather than quarters the
A conversion count. A miss closes block cohorts without a `2^28` run.

Cohort two is also rejected. It reduced the decoded-row and accumulator
working-set expansion relative to cohort four, but accumulation was still
10.619157 seconds (+5.44%) and whole commitment was 11.056315 seconds
(+5.16%). The proof verified, maximum RSS was 20.237 GiB, and the process did
not swap. The retained trace is
`benchmark-runs/perfetto_traces/commit_cohort2_25.json`.

Block cohorts are closed. The experiment confirms that the current
destination-major, one-block working set is more valuable than sharing A
conversion across independent blocks. Neither candidate earns a `2^28` run,
and all cohort code and tests were removed.

### Candidate B2: K256 row-batch cache geometry

The remaining commitment residue is small after accounting for schedule work:

```text
modeled work growth = 4 * (7 / 6) = 4.667x
observed accumulator growth = 113.543 / 22.813 = 4.977x
residual efficiency loss = 6.6%
```

The K256 kernel decodes 8,192 trace rows at a time, then scans each A rank over
that tile. A temporary runtime selector tested 4,096 and 16,384 rows against
two bracketing 8,192-row controls using one identical release binary:

| `2^25` row batch | Commit accumulation | Whole commitment | Prover |
|---:|---:|---:|---:|
| 8,192 control A | 10.353906 s | 10.791567 s | 26.859351 s |
| 4,096 | 11.588938 s | 12.051534 s | 28.698452 s |
| 16,384 | 11.982224 s | 12.454716 s | 29.434973 s |
| 8,192 control B | 11.698776 s | 12.166044 s | 28.738795 s |

The machine slowed monotonically across the sweep. Linear interpolation
between the two controls puts the contemporaneous defaults at 10.802 and
11.250 seconds: the candidates regress by 7.3% and 6.5%, respectively.
Neither clears the promotion gate under any reasonable drift treatment.
All four proofs verified. The selector was removed and the 8,192-row constant
is unchanged. Retained traces:

- `benchmark-runs/perfetto_traces/k256_batch8192_a_25.json`
- `benchmark-runs/perfetto_traces/k256_batch4096_25.json`
- `benchmark-runs/perfetto_traces/k256_batch16384_25.json`
- `benchmark-runs/perfetto_traces/k256_batch8192_b_25.json`

The commitment path is therefore closed for this experiment. Its apparent
5.0x scaling is mostly the exact 4.667x schedule-work increase; sharing A
loads and both adjacent cache geometries regress. A larger redesign would
need to change the schedule or accumulator ownership, not another local tile
constant.

## Experiment C: root relation-weight compilation

The accepted dense-rotation trace exposed a second root-opening cliff:
`relation_weight_compilation` grew from 0.663 seconds at `2^26` to 2.974
seconds at `2^28`. All seven recursive calls remained small; the excess was
entirely in the root call.

The root schedule explains the discontinuity:

| Root quantity | `2^26` | `2^28` | Growth |
|---|---:|---:|---:|
| Carried witness coefficients | 615,803,776 | 1,056,997,632 | 1.716x |
| B-setup input width | 1,056,768 | 9,863,168 | 9.333x |
| D-setup input width | 176,128 | 1,409,024 | 8.000x |

The full setup matrix is deliberately released after commitment to keep RSS
down. Relation compilation therefore regenerates only the touched B/D/A
matrix slices from the public seed. The A-column evaluation was already
parallel, but the 9.33x B scan and 8x D scan ran serially inside event
emission.

Two stacking Akita changes address that exact work:

1. Evaluate independent B and D setup columns in parallel, retain their scalar
   results only through E/T event emission, then drop them before the Z/R
   sections.
2. When constraint and setup contributions target the same physical interval
   with the same alpha exponent, add their scalars and emit one classified
   event. The uniform D64 root coalesces the paired E, T, and Z events.

The first `2^25` screen reduced relation compilation from 0.438 to 0.251
seconds. Coalescing reduced it again to 0.200 seconds; the whole evaluation
proof fell from 6.247 to 5.858 seconds. Both screens verified. Their traces
are `benchmark-runs/perfetto_traces/relation_parallel_25.json` and
`benchmark-runs/perfetto_traces/relation_fused_25.json`.

The single full-size candidate then produced:

| `2^28` metric | Dense control | Candidate | Change |
|---|---:|---:|---:|
| Relation event builder | 2.803088 s | 0.641417 s | -77.1% |
| Relation-weight compilation | 2.973814 s | 0.729790 s | -75.5% |
| Evaluation proof | 24.388854 s | 22.586504 s | -7.4% |
| Commitment | 115.484602 s | 116.502160 s | +0.9% |
| Whole prover | 216.013547 s | 219.681956 s | +1.7% |
| Maximum RSS | 85.124 GiB | 79.441 GiB | -5.683 GiB observed |

Only the localized opening gain is attributed to the change. The candidate's
PIOP stages and commitment were slower in this run, explaining the whole-
prover variance. The RSS difference is likewise not credited as a structural
5.7 GiB saving because the high-water point precedes opening; structurally,
the candidate adds less than 0.5 GiB of temporary setup scalars and removes
millions of later relation events.

The proof verified, `/usr/bin/time -l` reported zero process swaps, and the
system swapout counter remained fixed at 8,061,285. Maximum RSS was
85,299,347,456 bytes (79.441 GiB). The retained trace is
`benchmark-runs/perfetto_traces/akita_28_relation.json`; its log is
`benchmark-runs/akita-superlinear-2e28-2026-07-30/logs/akita_28_relation.log`.

Against the `2^26` control, evaluation now scales by 2.098x for a 4x increase
in `T`, close to the expected square-root regime. The relation compiler itself
is 0.730 seconds at `2^28` versus 0.663 seconds at `2^26`, so the identified
super-linear opening discontinuity is gone.

Landed upstream on the Akita performance branch as:

- `5f906ec3 perf(prover): parallelize relation setup columns`
- `67c3c88d perf(prover): coalesce aligned relation events`

## Append-only results

| Run | Change | Commit | Root decompose | Evaluation | Prove | Max RSS | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| control-26 | `720e1a7d1`, K256 | 23.29 s | 0.57 s, dense | 10.76 s | 54.13 s | 33.729 GiB | control |
| control-28 | `720e1a7d1`, K256 | 121.82 s | 13.15 s, sparse | 35.38 s | 236.72 s | 80.655 GiB | control |
| dense-28-observed | `6b1499427` + 256 MiB rotation budget | 117.99 s | 2.10 s, dense | 25.14 s | 224.46 s | not captured | inconclusive |
| dense-28-recheck | identical binary and candidate | 115.48 s | 2.09 s, dense | 24.39 s | 216.01 s | 85.124 GiB | accept |
| relation-28 | dense + Akita `67c3c88d` | 116.50 s | 2.10 s, dense | 22.59 s | 219.68 s | 79.441 GiB | accept |

The observed candidate passed and verified. The root accumulator switched from
`dense_rotations=false` to `true`; its duration fell from 13.10 to 2.04
seconds, while the enclosing decomposition fell by 11.05 seconds and the
evaluation proof fell by 10.25 seconds. The unchanged commit accumulator was
114.58 seconds in the control and 115.05 seconds in the candidate, so the
opening gain is not an unrelated commit fluctuation.

This run is not yet promotable. Sandboxed `/usr/bin/time -l` failed after the
test passed because it could not read `kern.clockrate`, so it did not emit
maximum RSS. During the run, the system swapout counter increased by 504
16-KiB pages (7.875 MiB), from 8,060,781 to 8,061,285. The trace and prover
result remain valid, but the run fails the predeclared capacity guard. The
measurement-repair recheck uses the same binary and evaluator with
`/usr/bin/time` allowed to read system counters; it is not a new candidate.

The repaired measurement passed and verified with zero process swaps. The
system swapout counter remained fixed at 8,061,285. Maximum RSS was
91,401,519,104 bytes, or 85.124 GiB, leaving 4.876 GiB below the 90 GiB
working limit. Its 89.43 decimal-GB internal high-water marker occurred in
Stage 4, before the root rotation table is allocated. The candidate's only
structural allocation delta is the exactly 232 MiB table in the later opening
phase; the larger difference from the old control's maximum is therefore
allocator/compressor run variance, not a live-set consequence of this change.

Across the two candidate traces, root decomposition averaged 2.092 seconds
versus 13.150 seconds in the control (-84.1%). Evaluation averaged 24.763
seconds versus 35.384 seconds (-10.621 seconds, -30.0%). The full prover
times were 224.46 and 216.01 seconds, but only the localized evaluation gain
is attributed to this candidate because the commit and PIOP spans also varied
between runs.

Verdict: accept for the local K256 `2^28` path. The mechanism repeated, the
gain clears the five-second threshold by more than 2x, proof verification and
unit/e2e guards pass, and the repaired capacity run stays below 90 GiB without
swap. Larger schedules still fall back to sparse rotations when the table
would exceed the bounded 256 MiB budget.

## Final scaling verdict

The two material discontinuities are removed:

- dense root rotations eliminate the 13.15-second decomposition fallback;
- parallel, coalesced relation compilation eliminates the 2.97-second
  setup-weight cliff.

The accepted `2^28` evaluation proof is 22.59 seconds, down from the original
35.38 seconds and 2.10x the `2^26` control for 4x `T`. Commitment remains
schedule-dominated: its 4.98x observed growth is close to the exact 4.67x work
increase. Three bounded attempts to remove the residual bandwidth loss
(cohorts four/two and adjacent row-batch sizes) all regressed and were
removed. There is no remaining measured cliff large enough to justify
bypassing the security planner or accepting a local performance regression.
