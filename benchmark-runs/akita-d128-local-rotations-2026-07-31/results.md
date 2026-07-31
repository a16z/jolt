# D128 task-local rotation results

Date: 2026-07-31 EDT

Machine: Apple M4 Max

Implementation: Jolt `18d3fef96`

## Outcome

Accepted. Expanding one trace block of rotations inside each position task
reduces D128 root decomposition by 59.3% and the complete Akita opening by
15.5% across two `T = 2^28` candidate runs. Complete prover time averages
159.51 seconds, 4.10 seconds below the 163.60-second parent.

This is an opening schedule change only. Commitments, evaluations, proof
messages, transcript inputs, and verifier work are unchanged.

## Mechanism

A compact challenge row contains 128 signed bytes. A negacyclic rotation by
`s` maps coefficient `i` to `i + s`; coefficients crossing degree 128 change
sign because `X^128 = -1`. The parent applies that index and sign calculation
for every contribution.

The candidate expands all 128 rotations for each of the 29 columns in the
current trace block. The expanded `[i16; 128]` rows are then reused across the
4,096 positions owned by the task. Contributions are processed eight rows at
a time so each destination coefficient is loaded and stored less often.

The table is local to a Rayon task:

| Rotation representation | Live bytes |
|---|---:|
| Compact prepared challenges, persistent | 950,272 |
| Fully expanded table, trace-wide | 243,269,632 |
| Candidate table, per active task | 950,272 |
| Candidate tables at 16 workers | 15,204,352 |

The trace-wide dense table is exactly 232 MiB. The candidate recovers its
reuse while bounding added live scratch to 14.5 MiB on this machine.

## Focused signal

The exact production-shape probe used D128, K256, 29 columns, 256 trace
blocks, 4,096 positions per task, and the production compact challenge
density. All candidate outputs matched the compact implementation.

```text
compact rotations:     about 158.5 ms
task-local expansion:   about 95.8 ms
ratio:                       0.604
```

This 39.6% reduction cleared the focused gate before the full prover was run.

## Full `T = 2^28` result

| Metric | Parent | Candidate 1 | Candidate 2 |
|---|---:|---:|---:|
| `trace_onehot_decompose_accumulate` | 7.182 s | 2.833 s | 2.857 s |
| `TracePackedOneHot::decompose_fold` | 7.324 s | 2.970 s | 2.991 s |
| `TracePackedOneHot::evaluate_and_fold` | 1.184 s | 1.181 s | 1.162 s |
| Complete Akita opening | 26.369 s | 22.162 s | 22.403 s |
| Complete prover | 163.60 s | 158.77 s | 160.24 s |
| Maximum RSS | 78.708 GiB | 80.141 GiB | 74.937 GiB |
| Prover-sampled memory maximum | 81.65 GB | 81.57 GB | 77.00 GB |
| Process swaps | 0 | 0 | 0 |

Candidate averages relative to the parent:

| Metric | Candidate average | Change |
|---|---:|---:|
| Decompose accumulation | 2.845 s | -60.4% |
| Complete decompose-fold | 2.981 s | -59.3% |
| Complete opening | 22.283 s | -4.086 s (-15.5%) |
| Complete prover | 159.51 s | -4.10 s (-2.5%) |

The first candidate command included a release rebuild and produced the
higher `/usr/bin/time` peak, while its in-prover sampled maximum remained
slightly below the parent. The no-rebuild repeat reported 74.937 GiB. The
two traces and the 14.5 MiB analytical bound show no structural RSS
regression; neither run swapped.

The unchanged `evaluate_and_fold` span also localizes the gain to the intended
decomposition pass. The `RingRelationProver::new` inclusive span falls from
8.62 to about 4.26 seconds because it contains that pass.

## Rejected sub-screens

Three narrower representation changes did not pass the focused gate:

- compressing the destination accumulator from i32 to i16 was flat or slower
  in repeated measurements and would add a downstream widening step;
- materializing rotated i8 rows for every visited ring took about 79 ms
  versus 47 ms for the compact control;
- retaining eight temporary i8 rotation tables took about 55 ms versus
  48 ms for the control.

The useful unit of reuse is therefore one trace block across a complete
position task. Per-ring materialization is too fine, while the trace-wide
table retains 232 MiB unnecessarily.

## Scope

The optimized branch requires D128, K256, and compact prepared rotations.
With the current policy it starts at `T = 2^28`. D64, K16, and sparse or
already-dense schedules execute the prior implementation.

## Validation

- exact-shape differential probe;
- all 49 enabled `jolt-akita` tests;
- D128/K256 blockwise opening equivalence;
- natural, forced-K256, and committed-program Akita muldiv proofs;
- standard and ZK Dory muldiv suites;
- two exact `T = 2^28` proofs with successful verification;
- zero process swaps in both full runs;
- scoped and workspace warning-denying Clippy in standard and ZK modes;
- `cargo fmt --check` and `git diff --check`.

## Traces

| Trace | Purpose | SHA-256 |
|---|---|---|
| `benchmark-runs/perfetto_traces/akita_28_deferred_carries.json` | accepted parent | `5c2da1657128cf77c7b826f1ee9036c109eb565d2d34e59a7489a635f6047f38` |
| `benchmark-runs/perfetto_traces/akita_28_local_rotations_run1.json` | candidate with rebuild | `f09aefc2b91b5d7e025a9739cce0cb16f45ab0ccaac77e6b5d5943291c12e542` |
| `benchmark-runs/perfetto_traces/akita_28_local_rotations.json` | clean no-rebuild repeat | `5f07bec2fffe82ab261d96621d1041f309329f22337cd343fb9278d331174574` |

