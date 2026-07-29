# Akita large-trace follow-up contract

Date: 2026-07-29 EDT

## Objective

Reduce the K256 Akita prover time at exactly `2^26` rows while retaining one
committed prefix-packed polynomial. Each accepted optimization gets its own
commit.

The fixed comparison point is the final two-run mean:

- Akita: 63.809687 s
- Dory: 111.167123 s
- Akita target: 55.710000 s

## Experiment S3: reuse cached RA indices in stage 7

Question: does replacing stage 7's second trace decode with the retained
`RaIndices` reduce the whole prover by a measurable amount?

Editable surface:

- `zkvm/packed.rs`
- `zkvm/claim_reductions/hamming_weight.rs`
- imports and directly affected tests

Frozen evaluator:

- `sha2_chain_akita_perf`
- physical K256, virtual v32
- `PERF_LOG_T=22` screen, followed by `PERF_LOG_T=26`
- traced release build

Metrics:

- `prove_packed`
- `prove_stage7_lattice`
- `shared_ra_polys::compute_all_G`
- `shared_ra_polys::compute_all_G_from_ra_indices`
- proof verification and peak RSS

Expected outcome: stage 7 falls by about 1.0--1.2 s at `2^26`.

Falsifying outcome: the `2^22` target span does not improve by at least 5%,
the projected `2^26` saving is below 0.8 s, or a full run fails to beat the
0.48 s historical noise floor.

Run budget: one baseline screen, one candidate screen, one full candidate,
then one adjacent baseline/candidate pair only if the first full run clears
the gate.

### Evaluator amendment 1

The first `2^22` baseline selected K16 because the production switchover is
`log_T = 25`. That run is excluded. The screen size is therefore `2^25`, the
smallest size that naturally selects K256; all other evaluator settings and
the full-size gate remain unchanged.

## Experiment D128: K256 commitment-only falsifier

Question: does a D128 root with the same K256 packed polynomial reduce
commitment enough to justify the protocol/config port?

Minimal surface:

- a Jolt D128/K256 planner config
- a D128 specialization of the existing rank-tiled trace commitment kernel
- benchmark-only entry points needed to run the exact target geometry

The verifier, transcript, proof serialization, and stages 1--8 remain out of
scope until the commitment gate passes.

Metrics:

- resolved root `n_a`
- `TracePackedOneHot::commit_inner`
- `trace_onehot_commit_accumulate`
- setup time and peak RSS

Expected outcome: D128 resolves to a smaller A rank and removes enough repeated
A-ring loads to save 5--8 s at `2^26`.

Falsifying outcome: the exact D128/K256 schedule does not reduce `n_a`, the
commitment kernel improves by less than 25% or 5 s, or memory exceeds the
current run by more than 10 GiB.

Run budget: one geometry probe, at most three kernel variants, one full-size
D64 control, and one full-size run for each viable D128 variant. A full
protocol port is authorized only after a candidate clears the commitment
gate.

## Known confounds

- The two retained final traces were produced from the accepted campaign tree,
  not the current documentation/port commits. Every new comparison records its
  exact revision and tree.
- The Akita D128 `nv=36` result changes different one-hot geometry and is
  supporting evidence, not a baseline for Jolt K256.
- Direct reduced commit accumulation previously lost to wide NEON
  accumulation. It is an ablation only if D128 changes the accumulator's cache
  residency.
- A traced run is valid only if the proof verifies and the trace contains
  exactly one primary prover span.
