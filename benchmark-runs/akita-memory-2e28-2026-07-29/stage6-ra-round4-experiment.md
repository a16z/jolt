# Fourth delayed RA round experiment

Date: 2026-07-30 EDT

## Question

Can Akita reduce the Stage-6b materialization spike by keeping the shared and
per-polynomial RA sources indexed for four rounds, then allocating Fp128
coefficients at `T / 16` instead of `T / 8`?

The experiment was prover-local and gated on 16-byte fields. BN254 retained
the existing three-round path. K256, the proof protocol, batching, transcript,
workload, and benchmark harness were unchanged.

## Expected trade

The control trace jumps by 5.37 GB when the RA sources materialize at `T / 8`
in the `2^26` Stage 6b. Moving the output allocation to `T / 16` should roughly
halve that jump, but it adds a sumcheck round in which every coefficient is
computed through eight indexed table reads rather than one contiguous Fp128
read.

The `2^22` screen was the promotion gate. A clear Stage-6b regression would
reject the candidate before a costly `2^26` run.

## Correctness

Focused tests compared both the independent and shared delayed
representations against fully materialized Akita polynomials in low-to-high
and high-to-low binding order. The existing BN254 RA tests also passed, as did
the forced-K256 Akita end-to-end `muldiv` proof.

The independent tests caught and prevented an initial table-ordering mistake
in the high-to-low path before benchmarking.

## `2^22` result

| Variant | Prove | Stage 6b | Round-0 memory | Stage-6b plateau |
|---|---:|---:|---:|---:|
| Three-round control | 5.77 s | 324.133 ms | 5.15 GB | 5.52 GB |
| Four-round candidate | 5.74 s | 401.518 ms | 5.20 GB | 5.30 GB |

The candidate lowers the reported Stage-6b plateau by 0.22 GB, but Stage 6b
regresses by 77.385 ms, or 23.9%. The whole-proof headline happens to improve
by 30 ms because unrelated work moves in the opposite direction; it does not
overrule the direct phase regression.

## Outcome

Rejected at the small-trace gate. The extra indexed message round is much too
expensive for the measured memory reduction, so no `2^26` run was performed.
All candidate source and tests were reverted; no implementation commit was
created.

The result does not rule out reducing the materialization overlap by changing
allocation ownership or scheduling. It rules out paying for another generic
eight-lookup RA round through the current coefficient-access interface.

## Retained trace

- `benchmark-runs/perfetto_traces/mem-ra-round4-2e22.json`
