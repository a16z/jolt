# Akita `2^28`: root-rank audit, D128 schedule, and bounded optimization loop

Date: 2026-07-30  
Machine: Apple M4 Max  
Workload: forced-K256 `sha2_chain_akita_perf`, `T = 2^28`

## Result

The D64 root rank increase from 6 at `2^26` to 7 at `2^28` is not a
planner-search bug with a practical rank-6 schedule hiding behind a different
split. The only secure D64 rank-6 point found by the exhaustive audit requires
`P = 2^32`, only 8 live blocks, and a 24 TiB setup. The shipped D64 rank-7
schedule is the practical Pareto point under the current challenge family,
norm bound, and SIS table.

The practical route measured here is to change the large-trace protocol to
D128. At `2^28`, D128 uses rank 3 and keeps `n_a D = 384`, equal to the
`2^26` D64 schedule. The final run completed and verified with:

| Metric | Final D128 |
|---|---:|
| Prover | 199.25 s |
| Commitment | 87.93 s |
| Root accumulation | 87.61 s |
| Evaluation proof | 33.32 s |
| Maximum RSS | 89,992,118,272 bytes |
| Maximum RSS | 83.812 GiB |
| Gross RSS | 335.25 B/cycle |
| Process swaps | 0 |

Against the retained D64 control, commitment fell by 34.06 s (27.9%) and the
whole prover fell by 26.65 s (11.8%). The D128 opening is 10.36 s slower,
mainly because its basis-64 digit range proof is more expensive. A direct
basis-3 experiment found that paying another commitment rank to avoid that
range proof is a net loss on this workload.

The accepted source changes are:

- `d893d86ee`: dispatch the `2^28`, K256 packed trace through a
  transcript-bound D128 schedule and D128-specific root kernel;
- `69b30cde4`: use compact root challenge rotations for D128 opening.

The final retained trace is
`benchmark-runs/perfetto_traces/akita_28_final.json`.

These are repeated measurements on the fixed optimization workload, with an
adjacent control and a final accepted-state repeat. They establish the result
for this machine and trace shape; they are not a held-out cross-machine or
cross-workload evaluation.

## Why D64 rank grows

Let `N` be the root input in ring elements, split as

```text
N = P * B
```

where `P` is the number of positions per block and `B` is the number of live
blocks. The root commitment applies an SIS A matrix of rank `n_a` to every
block. The minimum secure `n_a` is not a function of `T` alone: it is selected
from the SIS table using the A width and the rounded collision-coefficient
bound. That bound depends on the gadget basis, fold-response digit count,
challenge norm, and number of blocks folded together.

The relevant schedules are:

| Schedule | `D` | log basis | `P` | `B` | fold digits | A bound | `n_a` | `n_a D` | setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D64, `2^26` | 64 | 3 | `2^21` | `2^12` | 4 | 1,048,575 | 6 | 384 | 12 GiB |
| D64, `2^28` | 64 | 3 | `2^20` | `2^15` | 5 | 8,388,607 | 7 | 448 | 18.8125 GiB |
| D128, `2^28` | 128 | 6 | `2^21` | `2^13` | 2 | 524,287 | 3 | 384 | 12 GiB |

Moving from `2^26` to `2^28` multiplies the logical trace by four. Under D64,
the selected split also halves `P`, so `B` grows eightfold. The larger folded
response needs another digit and enters the next A-bound bucket. At the
resulting width and bound, the production SIS table requires rank 7.

This also explains the apparent superlinear commitment growth. The hot
one-hot root kernel does work proportional to

```text
T * n_a * D
```

before blocking and cache effects. D64 therefore predicts a
`4 * (7 / 6) = 4.667x` increase from `2^26` to `2^28`; the measured root
accumulation grew from 21.76 to 119.06 s, or 5.47x. The remaining 17.3% over
the operation-count prediction is consistent with the larger setup,
eightfold block count, extra output digit, and worse cache/memory behavior.

D128 restores `n_a D = 384`. Its `2^28` root accumulation is 87.61 s, only
4.026x the `2^26` D64 result: 0.65% above exact linear scaling. In other
words, the accepted change removes the schedule cliff. Further commitment
work is now a constant-factor kernel problem, not a `2^28` scaling anomaly.

## Rank-6 audit

The audit enumerated every legal D64 root split for log bases 2 through 6,
then queried the production quantum-128 SIS table using the exact
coefficient-bound bucket for each point. It also compared payload-first and
rank-first planner selection.

For the current basis-3 A-bound cell, rank 6 supports width at most 43,756.
The practical root needs width 1,048,576, a factor of 24 larger. Changing the
fixed challenge shell cannot close that gap: the production `±1/±2`
challenge already minimizes `L2^2` among the checked 128-bit-entropy shells,
and alternative shells move the bound only by a few percent.

The one secure rank-6 schedule is:

```text
log basis = 6
P         = 2^32
B         = 8
fold digits = 1
A bound   = 16,383
setup     = 24 TiB
```

This point is cryptographically valid but physically unusable. It exposes a
real limitation in the scalar planner objective: rank-first selection does
not itself encode a setup-memory envelope and can prefer extreme payload
slack. It does not imply that the production rank-7 point can be improved to
rank 6 within the machine budget.

The exhaustive rows are retained in:

- `logs/rank-audit.log`;
- `logs/ring-dimension-audit.log`.

## D128 trade and the basis-3 falsifier

D128 changes the root geometry in three useful ways:

- root rank drops from 7 to 3;
- setup falls from 18.8125 to 12 GiB;
- root NTT traffic falls from 25,249,710,080 to 2,684,354,560 bytes.

The lower rank and ring count dominate commitment. The cost is a high-basis
opening schedule. In the final D128 trace, the 33.32 s opening contains:

| Opening span | Time |
|---|---:|
| `digit_range_prove` | 15.10 s |
| `digit_range_product_substage` | 11.49 s |
| `RingRelationProver::new` | 9.28 s |
| root `TracePackedOneHot::decompose_fold` | 7.98 s |
| `digit_range_fold_lanes` | 6.05 s |

Inclusive spans overlap, so these rows must not be summed.

The tested alternative pinned D128 to log basis 3. That schedule uses rank
4, `P = 2^19`, `B = 2^15`, and a 10.75 GiB setup. It made the opening 21.5%
faster but commitment 34.5% slower:

| D128 schedule | Commit | Opening | Prover |
|---|---:|---:|---:|
| basis 6, rank 3 (accepted) | 87.93 s | 33.32 s | 199.25 s |
| basis 3, rank 4 (rejected) | 118.22 s | 26.17 s | 222.30 s |

Thus the current rank-first D128 choice is the faster of the measured
schedules for this commit-dominated workload even though it is not the
cheapest opening schedule.

## Compact opening rotations

The D128 root decompose-fold needs many negacyclic rotations of sparse
challenges. Three representations were measured:

| Representation | Root decompose | Opening | Maximum RSS |
|---|---:|---:|---:|
| sparse control | 10.67 s | 36.06 s | 91,499,266,048 B |
| dense `i16` table | 4.09 s | 29.70 s | 91,762,704,384 B |
| compact base `i8` rows | 7.87 s | 33.15 s | 90,849,771,520 B |

The dense table is 243,269,632 bytes and caused a matching adjacent RSS
increase, so it violated the no-memory-trade guard. The accepted compact
representation stores one unrotated `[i8; 128]` row per prepared challenge
(950,272 bytes total) and performs each signed wrap rotation into the
worker-local accumulator. It recovered 2.80 s (26.2%) from root
decompose-fold relative to the sparse path without introducing a
trace-scaled table.

## Updated analytical memory model

For `T = 2^28`, one byte per cycle is 0.25 GiB. The D128 root changes two
long-lived terms:

```text
setup: 18.8125 -> 12 GiB
commitment hint:
  B * n_a * d_o * D
  = 8192 * 3 * 22 * 128
  = 69,206,016 bytes
  = 0.064453 GiB
```

The common state retained through Stages 1–5 is therefore:

```text
setup                         12.000000 GiB
trace: 64T                    16.000000 GiB
packed lanes + RAM-valid: 30T  7.500000 GiB
commitment hint                0.064453 GiB
-----------------------------------------
common                        35.564453 GiB
```

Applying the stage-local formulas derived in
`benchmark-runs/akita-memory-2e28-2026-07-29/analytical-memory-model.md`
gives:

| Phase | D128 analytical peak | B/cycle |
|---|---:|---:|
| Commit, including 2.5 GiB root NTT slot | about 38.11 GiB | about 152.45 |
| Stage 1 | 43.564 GiB | 174.258 |
| Stage 2, representative `mu = 0.027` | 47.996 GiB | 191.986 |
| Stage 3 transition | 47.814 GiB | 191.258 |
| Stage 4, representative `rho = 1.23` | 67.244 GiB | 268.978 |
| Stage 5 transition | **67.846 GiB** | **271.383** |
| Stage 6a | 53.096 GiB | 212.383 |
| Stage 6b steady | 60.564 GiB | 242.258 |
| Stage 6b transition | 63.846 GiB | 255.383 |
| Stage 7 | 35.064 GiB | 140.258 |
| Reconstruction | 19.314 GiB | 77.258 |
| Evaluation root preparation, conservative | about 29.33 GiB | about 117.32 |

For the representative SHA-2 trace density, the Stage-5 transition remains
the source-owned structural ceiling. It falls from 75.182 GiB under D64 to
67.846 GiB under D128, a reduction of 7.336 GiB. The final process RSS was
83.812 GiB, leaving 6.188 GiB below the 90 GiB working target and reporting
zero swaps.

This is not a universal trace ceiling. Stage 4 is
`47.564 + 16 rho` GiB under D128, where `rho` is the number of sparse register
entries per cycle. It exceeds 90 GiB at `rho > 2.652` and 95 GiB at
`rho > 2.965`; the representative workload has `rho` near 1.23.

The 15.966 GiB gap between the 67.846 GiB ownership model and maximum RSS does
not by itself establish another live `T`-scaled protocol object. The model
does not charge allocator-retained pages, all per-round scratch, thread
stacks, or macOS memory-residency/compression effects. It is a ceiling for the
identified long-lived owners, while `ru_maxrss` is a process high-water mark;
they measure different things. Reducing that gap requires
allocation/lifetime instrumentation at the high-water window, not another
protocol rewrite based only on the RSS number.

The evaluation estimate includes the 12 GiB setup, the 7.25 GiB packed lane
source while the root is prepared, about 4 GiB of D128 position weights, the
6 GiB root decompose/acceptance state, and small folded outputs. Its later
ring-switch window instead holds a 1.25 GiB negacyclic NTT slot and a
629,162,496-byte root output after the packed rows are released. Neither
window approaches the Stage-5 ceiling.

## Rejected engineering experiment

The K256 rank-tiled commit loop called `fill_row` once per trace cycle, copied
29 bytes into a temporary row, then copied those bytes into its existing 8K
row tile. A candidate used the source's bulk `fill_rows` API to fill the tile
directly. This removed the tiny virtual calls and one redundant 7.25 GiB lane
copy at `2^28`, with no memory cost.

The adjacent `2^25` screen rejected it:

```text
bulk-fill candidate  10.87 s root accumulation
row-fill control     10.71 s root accumulation
```

The 1.5% regression/noise-level result shows that wide-ring
shift-accumulation, not row dispatch or lane copying, dominates this kernel.
The source change was reverted.

## What remains

The root commitment now scales almost exactly linearly from `2^26`; its
87.61 s span is still the largest constant-factor target. The failed
bulk-fill experiment narrows the next useful work to the wide D128
shift/accumulator kernel itself: accumulator representation, coefficient
layout, and memory traffic should be microbenchmarked before another prover
edit.

For opening time, the highest localized target is the basis-64 digit range
path, especially `digit_range_product_substage` and
`digit_range_fold_lanes`. The basis-3 falsifier rules out changing the whole
schedule; an improvement must make the high-basis inner loops cheaper without
raising rank or retaining another trace-sized table.

For memory, the next evidence-gathering step is to attribute the roughly
16 GiB RSS/model residual around the high-water phase. A memory change should
remove a measured live owner or allocator-resident region and must not add a
pass over `T` or serialize phase destruction. Compacting the root centered
coefficients is not a global-peak optimization by itself because the
evaluation proof is well below the Stage-5 ceiling.

## Retained artifacts

| Artifact | Purpose |
|---|---|
| `benchmark-runs/perfetto_traces/akita_28_final.json` | accepted final D128 run |
| `benchmark-runs/perfetto_traces/akita_28_d64_control.json` | adjacent D64 control |
| `benchmark-runs/perfetto_traces/akita_28_d128_basis3.json` | rejected rank-4/basis-3 schedule |
| `benchmark-runs/perfetto_traces/akita_28_d128_dense_rotations.json` | dense-rotation time/RSS trade |
| `benchmark-runs/perfetto_traces/akita_28_d128_sparse_control.json` | sparse-rotation control |
| `benchmark-runs/perfetto_traces/akita_28_d128_compact_rotations.json` | accepted compact-rotation run |
| `benchmark-runs/perfetto_traces/akita_25_bulk_fill.json` | rejected bulk-fill screen |
| `benchmark-runs/perfetto_traces/akita_25_row_fill_control.json` | adjacent row-fill control |
| `events.jsonl` | append-only candidate ledger |
| `contract.md` | frozen evaluator and acceptance guards |

The raw `/usr/bin/time` and nextest logs, trace analyses, rank enumeration,
and run contract are in this directory.
