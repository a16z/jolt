# Stage 6 Bytecode, Booleanity, RA Virtualization, Increment, and Advice Cycle Frontier Spec

## Scope

Stage 6 proves the largest regular batched sumcheck frontier in the Jolt VM
pipeline. It starts after Stage 1 through Stage 5 have produced mode-matched
typed verifier outputs and the transcript is positioned at the Stage 6
boundary. It ends after the verifier-owned Stage 6 proof field, clear claims or
committed ZK shape, advice cycle-phase state, and downstream Stage 6 output are
assembled.

The stage boundary is:

1. Consume Stage 1, Stage 2, Stage 3, Stage 4, and Stage 5 outputs plus
   normalized Jolt VM witness views.
2. Derive all Stage 6 Fiat-Shamir challenge vectors in the exact verifier
   order.
3. Prove the Stage 6 batched sumcheck over:
   - bytecode read-RAF;
   - RA booleanity;
   - RAM hamming booleanity;
   - RAM RA virtualization;
   - instruction RA virtualization;
   - increment claim reduction;
   - field-register increment claim reduction under `field-inline`;
   - trusted advice cycle phase, if present;
   - untrusted advice cycle phase, if present.
4. Evaluate every Stage 6 output opening at verifier-derived points.
5. Assemble the verifier-owned Stage 6 proof fields and claims.
6. Return typed Stage 6 public output for Stage 7 and Stage 8 without
   recomputing verifier reductions.

Stage 6 produces the verifier component for:

- `JoltStageProofs::stage6_sumcheck_proof`;
- clear-mode `ClearProofClaims::stage6`;
- ZK-mode committed Stage 6 sumcheck proof and output-claim commitments;
- clear-mode `Stage6ClearOutput` data needed by Stage 7 and Stage 8;
- ZK-mode `Stage6ZkOutput` data needed by Stage 7, Stage 8, and BlindFold.

Stage 6 does not prove hamming-weight claim reduction, advice address phase, or
the final PCS opening proof. It must preserve all public challenge vectors and
opening points because Stage 7 uses them for hamming-weight and advice
address-phase reductions, and Stage 8 may use Stage 6 increment/advice final
openings.

Every Stage 6 implementation path must explicitly support:

- advice cycle phase for trusted and untrusted advice when the corresponding
  layouts require it;
- BlindFold/ZK mode, including committed regular-batch proof data and committed
  output-claim rows;
- field-inline increment claim reduction behind the `field-inline` feature.

## Monitored Workflow

Stage work proceeds in one reviewable Stage 6 slice:

1. Confirm current inventory for `jolt-prover`, `jolt-verifier`,
   `jolt-backends`, `jolt-witness`, `jolt-program`, and
   `jolt-prover-harness`.
2. Tighten backend evidence for bytecode read-RAF, booleanity, increments, and
   advice before accepting prover orchestration.
3. Define the canonical Stage 6 input/output/prover-state API.
4. Refactor toward one public `prove` entrypoint, ordered first in `prove.rs`.
5. Implement transparent and advice paths through native Stage 6 verifier
   replay.
6. Add field-inline increment support to the same canonical path.
7. Add ZK committed-boundary proof assembly and verifier replay.
8. Run focused tests, real fixture replay, and kernel evidence gates.
9. Append final correctness and performance parity justification to this spec.
10. Stop for review before moving to Stage 7.

Every fact in the implementation should have a clear owner:

- `jolt-verifier` owns proof fields, verifier outputs, transcript checks,
  clear claims, committed proof shapes, output-claim counts, and output order;
- `jolt-claims` owns relation IDs, dimensions, formula semantics, opening IDs,
  challenge IDs, and public IDs;
- `jolt-program` owns canonical bytecode/preprocessing shape;
- `jolt-witness` owns trace-backed witness views, bytecode views, RA views,
  increment views, and advice views;
- `jolt-lookup-tables` owns lookup-table kind counts used by prior Stage 5
  output sizing, but Stage 6 should prefer Stage 5 verifier output lengths or
  `jolt-claims` helpers over local table enumeration;
- `jolt-riscv` owns instruction rows and circuit flags, but direct use in Stage
  6 prover orchestration should be moved behind `jolt-claims`,
  `jolt-program`, or witness adapters wherever possible;
- `jolt-backends` owns heavy compute and slot-keyed kernel results;
- `jolt-prover-harness` owns migration-only fixtures, verifier replay, and
  performance evidence.

The public prover path should remain linear enough to audit against
`jolt-verifier/src/stages/stage6/verify.rs`.

### Modular Crate Usage

Stage 6 should follow the verifier ownership split.

- Use `jolt-claims::protocols::jolt::formulas::bytecode` for bytecode
  read-RAF metadata, dimensions, input/output openings, and opening points.
- Use `jolt-claims::protocols::jolt::formulas::booleanity` for RA booleanity
  metadata and output openings.
- Use `jolt-claims::protocols::jolt::formulas::ram` for RAM hamming
  booleanity and RAM RA virtualization metadata.
- Use `jolt-claims::protocols::jolt::formulas::instruction` for instruction
  RA virtualization metadata.
- Use `jolt-claims::protocols::jolt::formulas::claim_reductions::increments`
  for increment claim-reduction metadata and opening order.
- Use `jolt-claims::protocols::jolt::formulas::claim_reductions::advice` plus
  `AdviceClaimReductionLayout` for advice cycle phase metadata.
- Under `field-inline`, use field-inline increment claim helpers rather than
  duplicating field-register formulas in `jolt-prover`.
- Use `jolt_opening_oracle_ref` to convert semantic opening IDs into witness
  oracles.
- Use `jolt-program` bytecode/preprocessing views instead of carrying raw
  `jolt_riscv::JoltInstructionRow` through production prover APIs when a
  modular facade exists.

Direct imports of `jolt_riscv::{CircuitFlags, CIRCUIT_FLAGS}` in Stage 6
orchestration are a cleanup target. The number and order of circuit flag claims
should be obtained from `jolt-claims` helpers or verifier-owned outputs, so the
prover does not fork flag ordering.

## Current Inventory

### jolt-verifier

Relevant verifier code lives in `crates/jolt-verifier/src/stages/stage6/`.

- `verify.rs` derives, in order:
  - `bytecode_gamma_powers = transcript.challenge_scalar_powers(8)`;
  - Stage 1 gamma powers;
  - Stage 2 gamma powers;
  - Stage 3 gamma powers;
  - Stage 4 gamma powers;
  - Stage 5 gamma powers;
  - Booleanity reference address/cycle from Stage 5 instruction read-RAF
    output, with transcript padding when the address is shorter than committed
    chunk bits;
  - nonzero `booleanity_gamma`;
  - instruction RA virtualization gamma powers;
  - `inc_gamma`;
  - `field_inc_gamma` under `field-inline`.
- Clear mode verifies a compressed Boolean batched sumcheck in this statement
  order:
  1. bytecode read-RAF;
  2. RA booleanity;
  3. RAM hamming booleanity;
  4. RAM RA virtualization;
  5. instruction RA virtualization;
  6. increment claim reduction;
  7. field-register increment claim reduction under `field-inline`;
  8. trusted advice cycle phase, if present;
  9. untrusted advice cycle phase, if present.
- Clear input claims are reconstructed from Stage 1 through Stage 5 verifier
  outputs and challenge vectors. The prover should not recompute earlier stage
  relation semantics independently.
- Stage 6 output claims include:
  - bytecode read-RAF bytecode RA openings;
  - booleanity instruction/bytecode/RAM RA openings;
  - RAM hamming-weight opening;
  - RAM RA virtualization committed RAM RA openings;
  - instruction RA virtualization committed instruction RA openings;
  - increment claim-reduction `ram_inc` and `rd_inc`;
  - field increment `field_rd_inc` under `field-inline`;
  - trusted/untrusted advice cycle-phase final openings when present.
- `outputs.rs` owns `Stage6PublicOutput`, `Stage6ClearOutput`,
  `Stage6ZkOutput`, `Stage6Output`, `VerifiedStage6Batch`, and the typed
  public/verified structs for every relation.
- `stages/zk/blindfold/stage6.rs` lowers Stage 6 into BlindFold using the same
  statement order, challenge vectors, optional advice rows, and field-inline
  extension.

Stage 6 prover code must import verifier-owned proof, claim, and output structs
directly rather than duplicating those shapes locally.

### jolt-prover

Current implementation lives in `crates/jolt-prover/src/stages/stage6/`.

- `input.rs` defines `Stage6ProverConfig`. It currently exposes bytecode
  context using `jolt_riscv::JoltInstructionRow`; this should move to a
  modular `jolt-program`/witness bytecode facade when available.
- `prove.rs` exposes helper entrypoints rather than one canonical public Stage
  6 `prove` entrypoint:
  - `derive_stage6_regular_batch_prefix`;
  - `evaluate_stage6_output_openings`;
  - `prove_stage6_transparent_sumchecks`;
  - advice cycle-phase helpers;
  - opening-claim append helpers.
- `prove.rs` imports `jolt_riscv::{CircuitFlags, CIRCUIT_FLAGS}` directly for
  Stage 1 flag gamma/count handling. This should be replaced by a shared
  protocol helper or verifier-owned count.
- The transparent sumcheck helper is compiled only under
  `not(feature = "field-inline")`.
- Field-inline prefix fields exist, but field increment claims currently use
  placeholder zero values in some paths. This is a correctness blocker for
  field-inline acceptance.
- There is no ZK committed Stage 6 proof assembly path.
- There is no canonical `Stage6ProverInput` bundle that consumes all prior
  stage outputs, witness provider, protocol config, and mode.
- There is no canonical `Stage6ProverOutput` carrying verifier-owned proof
  fields, clear/ZK output, advice phase state, opening state, and BlindFold
  private material.
- `output.rs` uses `HashMap` for slot collection. Replace it with `BTreeMap`
  for deterministic diagnostics.

Current request slots are:

- bytecode RA openings start at `0`;
- booleanity instruction RA openings start at `1024`;
- booleanity bytecode RA openings start at `2048`;
- booleanity RAM RA openings start at `3072`;
- RAM hamming booleanity opening is `4096`;
- RAM RA virtualization openings start at `4100`;
- instruction RA virtualization openings start at `5120`;
- increment `ram_inc` and `rd_inc` openings are `6144` and `6145`.

Advice cycle-phase openings are currently evaluated outside the shared
`Stage6OutputOpeningEvaluationRequest`; the accepted design may keep a
separate typed advice request, but it must be slot-keyed and verifier-order
auditable.

### jolt-backends

The current Stage 6 prover helpers use generic witness materialization and
evaluation bridges plus hand-written relation loops. That is useful during
bring-up but insufficient for replacement.

Required CPU work:

- preserve bytecode read-RAF gamma precomputation and simultaneous value
  computation;
- preserve booleanity phase splitting and cached opening behavior;
- preserve increment claim-reduction specialized paths;
- preserve advice two-phase state across Stage 6 and Stage 7;
- preserve RA delayed materialization/pushforward and one-hot evaluation;
- avoid dense fallback materialization where core uses structured relation
  kernels.

### jolt-prover-harness

The manifest currently contains:

- `stage6_regular_batch_inputs`;
- `stage6_output_openings`;
- `stage6_regular_batch_sumcheck`.

The corresponding tests live in
`crates/jolt-prover-harness/tests/frontier_stage6_batch.rs`.

Current manifest optimization IDs are generic sumcheck/opening IDs:

- `OPT-SC-007`;
- `OPT-EQ-004`;
- `OPT-OPEN-008` for output-opening checks.

Before production replacement, Stage 6 evidence should also account for:

- `OPT-REL-004`, `OPT-REL-005` for bytecode read-RAF;
- `OPT-REL-011` for increment claim reduction;
- `OPT-REL-013` for advice two-phase state;
- `OPT-REL-014` for booleanity phase splitting;
- `OPT-REL-015` for verifier-evaluable claim caching;
- relevant `OPT-RA-*`, `OPT-SC-*`, `OPT-EQ-*`, `OPT-FLD-*`, and
  `OPT-MEM-*` rows touched by the concrete CPU kernel.

## Target Prover Shape

Add a canonical public entrypoint near the top of `prove.rs`:

```rust
pub fn prove_stage6<F, W, B, T, C>(
    input: Stage6ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage6ProverOutput<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>;
```

`Stage6ProverInput` should make these dependencies explicit:

- `Stage6ProverConfig`;
- mode-matched Stage 1, Stage 2, Stage 3, Stage 4, and Stage 5 outputs;
- normalized witness provider;
- bytecode/preprocessing view from `jolt-program` or witness facade;
- trusted/untrusted advice layout state;
- selected proof mode.

`Stage6ProverOutput` should carry:

- verifier-owned `stage6_sumcheck_proof` payload;
- clear `Stage6Claims` or ZK committed output-claim rows;
- verifier-owned `Stage6Output`;
- advice cycle-phase state needed by Stage 7 and Stage 8;
- prover-local opening/blinding data needed by Stage 8 and BlindFold.

### Transparent Flow

Transparent Stage 6 should:

1. Validate prior stage dependencies using verifier-equivalent shape checks.
2. Derive every challenge vector in verifier order.
3. Build input claims using `jolt-claims` formulas and prior verifier outputs.
4. Build a backend request for all active Stage 6 instances in verifier
   statement order.
5. Append input claims and squeeze batching coefficients in that order.
6. Run optimized CPU kernels for the batched sumcheck.
7. Derive relation-specific opening points from the shared reduction point.
8. Evaluate output openings through slot-keyed witness/backend requests.
9. Evaluate advice cycle-phase openings when present.
10. Reconstruct expected output claims with verifier-equivalent formulas.
11. Append Stage 6 opening claims in verifier transcript order.
12. Assemble `Stage6Claims`, `Stage6ClearOutput`, and the proof payload.

### ZK Flow

ZK Stage 6 should:

1. Use the same statement order and input-claim formulas as transparent mode.
2. Produce committed round-polynomial proof data for every active relation.
3. Produce committed output-claim rows in the exact verifier order and count.
4. Store private coefficients, blindings, and advice rows needed by BlindFold.
5. Return `Stage6ZkOutput` public data matching verifier-owned output types.

### Field-Inline Flow

Field-inline Stage 6 must add field-register increment claim reduction after
base increment claim reduction and before optional advice instances.

It must:

- derive `field_inc_gamma` in verifier order;
- compute a real field increment input claim;
- evaluate `field_rd_inc` at the verifier-derived point;
- append the field-inline output claim in verifier order;
- include the field-inline output row in ZK committed output-claim counts.

Placeholder zero claims must be removed before field-inline acceptance.

### Advice Flow

Advice cycle phase is optional per advice kind.

- If an advice commitment is absent, the corresponding instance must not be
  included and no output opening should be appended.
- If an advice layout has no cycle phase, Stage 6 should not create a cycle
  phase instance; Stage 8 may use the direct final advice opening path instead.
- If a cycle phase exists, Stage 6 must produce the opening point and
  cycle-phase variables that Stage 7 address phase and Stage 8 final opening
  expect.

The prover must preserve trusted and untrusted ordering exactly:

1. trusted advice cycle phase, if present;
2. untrusted advice cycle phase, if present.

## Backend and Performance Requirements

Stage 6 replacement readiness requires these backend ledger entries, or more
specific successor entries, to be `ParityCertified` with evidence:

- `cpu_stage6_regular_batch_input_claims`;
- `cpu_stage6_regular_batch_sumcheck`;
- `cpu_materialized_opening_evaluations` for the output-opening replay
  subfrontier;
- `cpu_relation_stage_kernels`;
- `cpu_ra_one_hot_pushforward_kernels` or its certified narrower entries;
- relevant polynomial, EQ, univariate, field, and memory ledger entries.

Bytecode, delayed-RA, booleanity, increment, and read-RAF paths that use core
prefix/suffix decompositions must keep those algorithms in the backend
production kernels. Dense/materialized routes are reference oracles only and
must not be used to claim Stage 6 performance parity.

Performance evidence must include:

- focused Stage 6 input-claim and sumcheck microbenchmarks;
- bytecode read-RAF and booleanity relation evidence;
- advice-enabled fixture evidence;
- analytical peak-memory accounting for RA views, bytecode views, advice state,
  and relation caches;
- canonical fixture comparison against `jolt-core`;
- the default 15% timing and peak-memory parity threshold.

Correctness-only Stage 6 code is not replacement-ready.

## Required Tests and Gates

Before accepting Stage 6:

- harness static checks pass for manifest, optimization IDs, backend ledger,
  source drift, and workspace boundaries;
- `stage6_regular_batch_inputs`, `stage6_output_openings`, and
  `stage6_regular_batch_sumcheck` replay real fixtures;
- transparent `MuldivSmall` and `AdviceConsumer` verifier replay pass;
- advice fixtures cover trusted and untrusted cycle-phase paths;
- field-inline fixture replay passes after field-inline implementation lands;
- ZK fixture replay passes after committed Stage 6 implementation lands;
- output-opening order and transcript append order are covered by tamper tests;
- backend microbench evidence files are recorded in the ledger;
- `validate_frontier_replacement_ready` passes for Stage 6 with
  `ParityCertified` backend status;
- `validate_global_cpu_backend_inventory_coverage` passes.

Useful local commands:

```bash
cargo nextest run -p jolt-prover-harness frontier_stage6_batch --cargo-quiet
cargo nextest run -p jolt-prover-harness optimization_inventory --cargo-quiet
cargo nextest run -p jolt-prover-harness frontier_gates --cargo-quiet
```

Do not use broad prover E2E as the first debugging loop for Stage 6 backend
algorithm questions. Prove the isolated CPU kernels first, then wire the stage.

## Acceptance Checklist

- [ ] One canonical Stage 6 prover entrypoint exists.
- [ ] Stage 6 imports verifier-owned proof/output/claim structs directly.
- [ ] Stage 6 uses `jolt-claims` helpers for relation semantics and opening
      order.
- [ ] Bytecode context flows through `jolt-program` or witness facades.
- [ ] Direct `jolt-riscv` flag ordering is removed from orchestration.
- [ ] Transparent Stage 6 verifier replay passes for base and advice fixtures.
- [ ] Field-inline Stage 6 has real field increment outputs.
- [ ] ZK Stage 6 produces committed output rows and BlindFold material.
- [ ] Advice cycle-phase state is reusable by Stage 7 and Stage 8.
- [ ] Slot collection is deterministic.
- [ ] Backend kernel evidence is `ParityCertified`.
- [ ] Final correctness and performance justification is appended below.

## Final Justification Log

### Slice 0 — Deterministic-collection normalization + fully-scoped canonical `prove` plan (2026-05)

Landed: `crates/jolt-prover/src/stages/stage6/output.rs` `HashMap`→`BTreeMap`
(deterministic slot diagnostics), clippy-clean in all feature modes; existing
transparent `stage6_regular_batch_sumcheck_verifier_replay` re-verified green.

Canonical `prove` is fully scoped and ready to implement (transparent path,
cfg `not(field-inline)`), following the validated Stages 4/5 pattern. Mechanical,
no new crypto — every assembly source is identified:

- Input bundle `Stage6ProverInput` = config + `CheckedInputs` + Stage 1–5 clear
  outputs + witness (the helper `prove_stage6_transparent_sumchecks` and
  `derive_stage6_regular_batch_prefix` both take Stage 1–5).
- `Stage6ProverOutput` = `stage6_sumcheck_proof` + `Stage6Claims` (the
  `output_openings` type alias) + assembled `Stage6ClearOutput`.
- `Stage6PublicOutput` maps field-for-field from `Stage6RegularBatchPrefixOutput`
  (bytecode/stage1–5 gammas, booleanity reference points/gamma,
  instruction-RA gamma powers, inc gamma) plus `challenges`/`batching_coefficients`.
- `VerifiedStage6Batch` (14 sub-fields) maps from `Stage6RegularBatchProofOutput`:
  bytecode_read_raf / booleanity / ram_hamming_booleanity / ram_ra_virtualization
  / instruction_ra_virtualization / inc_claim_reduction, each from the matching
  `*_sumcheck_point` / `*_opening_point` / `*_ra_opening_points` fields, with
  per-instance `input_claim` from `prefix.input_claims` and `expected_output_claim`
  from a surfaced `Stage6ExpectedOutputs`.
- Advice cycle phase: `Stage6AdviceCyclePhaseProofOutput { sumcheck_point,
  opening_point, cycle_phase_variables }` →
  `VerifiedAdviceCyclePhaseSumcheck { kind, input_claim (from
  prefix.input_claims.*_advice_cycle_phase), sumcheck_point, opening_point,
  cycle_phase_variables, expected_output_claim (surfaced) }`, zipped over the
  trusted/untrusted `Option`s.
- Then surface `Stage6ExpectedOutputs` into the proof output (like Stages 3/4/5),
  reroute the harness `stage6` replay through `prove`, and validate against the
  native verifier (`stage6_regular_batch_sumcheck_verifier_replay`, ~250s).

Remaining before replacement-ready: backend-route the batched sumcheck through the
`jolt-backends` regular-batch kernel + perf-cert; field-inline Stage 6 prover path;
ZK committed-boundary.

### Slice 1 — Canonical transparent `prove` entrypoint implemented + validated (2026-05)

Status: transparent canonical API landed and verifier-validated. Backend-routing,
field-inline/ZK prover paths, and perf-cert remain. Not yet replacement-ready.

Code touched:

- `crates/jolt-prover/src/stages/stage6/output.rs`: added
  `Stage6RegularBatchExpectedOutputs`, `Stage6ProverOutput`
  (verifier-owned `stage6_sumcheck_proof` + `Stage6Claims` + assembled
  `Stage6ClearOutput`), and surfaced expected outputs on the proof output.
- `crates/jolt-prover/src/stages/stage6/input.rs`: added `Stage6ProverInput`
  (config ref + `CheckedInputs` + Stage 1–5 clear outputs + witness).
- `crates/jolt-prover/src/stages/stage6/prove.rs`: added the canonical public
  `prove` (cfg `not(field-inline)`), ordered first, mirroring
  `jolt-verifier/src/stages/stage6/verify.rs`. Assembles the full 14-field
  `VerifiedStage6Batch` — bytecode read-RAF / booleanity / RAM-Hamming booleanity
  / RAM & instruction RA-virtualization / increment claim-reduction, each with its
  derived points and per-instance expected outputs — plus the trusted/untrusted
  advice-cycle-phase `VerifiedAdviceCyclePhaseSumcheck` mapping.
- `crates/jolt-prover-harness/src/core_fixture.rs`: routed the Stage 6
  regular-batch verifier replay through the canonical `prove`; the compact
  comparison checks output openings, proof round-equality vs core, gammas, and the
  batch point/final claim.

Correctness evidence (all green):

- `cargo nextest run -p jolt-prover-harness stage6_regular_batch_sumcheck_verifier_replay stage6_regular_batch_input_checkpoint stage6_output_opening_checkpoint --features core-fixtures`
  — 4/4 pass, including both `..._verifier_replay_verifies_against_core_fixtures`
  (≈253s) and `..._against_core_advice_fixture` (≈175s): the canonical `prove`
  output is accepted by the native verifier and its proof matches core round-for-round.
- Compiles under transparent, `field-inline`, `zk`, `zk,field-inline`.

Replacement-ready gate status: the `stage6_output_openings` frontier is
replacement-ready (`validate_frontier_replacement_ready` passes against the
`ParityCertified` `cpu_materialized_opening_evaluations` kernel; test
`stage6_output_opening_frontier_is_replacement_ready_with_certified_kernel_evidence`).
The `stage6_regular_batch_inputs`/`stage6_regular_batch_sumcheck` frontiers remain
blocked on their `Required` kernel rows.

### Slice 2 — Stage 6 regular-batch input kernel evidence certified (2026-05)

Status: `stage6_regular_batch_inputs` is now replacement-ready. The full
`stage6_regular_batch_sumcheck` frontier remains pending backend routing and
performance certification.

Code touched:

- `crates/jolt-prover-harness/src/core_fixture.rs`: added
  `Stage6RegularBatchInputKernelBenchmarkFixture` plus a harness reference
  derivation for the Stage 6 input boundary. The reference independently mirrors
  verifier formulas for bytecode read-RAF input batching, RAM/instruction
  RA-virtualization inputs, increment claim-reduction input, and optional advice
  cycle-phase input, then compares against the native Stage 6 verifier output.
- `crates/jolt-prover-harness/benches/frontier_perf.rs`: added the canonical
  `cpu_stage6_regular_batch_input_claims` evidence writer and analytical memory
  budget.
- `crates/jolt-prover-harness/src/optimization.rs`: promoted
  `cpu_stage6_regular_batch_input_claims` to `ParityCertified` and recorded the
  canonical evidence file.
- `crates/jolt-prover-harness/tests/frontier_stage6_batch.rs`: added
  `stage6_regular_batch_input_frontier_is_replacement_ready_with_certified_kernel_evidence`.

Correctness evidence:

- `cargo check -p jolt-prover-harness --features core-fixtures --benches --tests -q`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'test(stage6_regular_batch_input_frontier_is_replacement_ready_with_certified_kernel_evidence) | test(stage6_output_opening_frontier_is_replacement_ready_with_certified_kernel_evidence) | test(stage6_regular_batch_input_checkpoint_matches_core_fixtures) | test(stage6_regular_batch_sumcheck_verifier_replay_verifies_against_core_fixtures) | test(stage6_regular_batch_sumcheck_verifier_replay_verifies_against_core_advice_fixture)'`
  passed 5/5. The two full replay tests remain slow (`~169s` advice,
  `~250s` transparent), which is expected until the full Stage 6 sumcheck moves
  to backend-owned optimized kernels.

Performance evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage6_regular_batch_input_claims cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  wrote
  `target/frontier-metrics/kernel-evidence/cpu_stage6_regular_batch_input_claims/frontier_perf_stage6_regular_batch_inputs.json`
  with status `Pass`.
- Reported ratios: time `0.9718099392177055`, peak RSS `1.0`.
- `backend_kernel_ledger_accounts_for_registered_frontier_cpu_optimizations`,
  `backend_kernel_ledger_covers_every_cpu_backend_inventory_id`,
  `prover_ready_frontiers_require_ported_or_certified_cpu_kernels`, and
  `registered_parity_certified_kernel_evidence_files_are_valid` pass.

Remaining before full Stage 6 replacement-ready: backend-route and perf-certify
`cpu_stage6_regular_batch_sumcheck`; implement field-inline Stage 6 increment
claims; implement ZK committed-boundary assembly/replay.

### Slice 3 — Stage 6 regular-batch backend routing through split-eq/RA kernels (2026-05)

Status: transparent/advice correctness is green for the backend-routed Stage 6
regular-batch sumcheck. This is not yet final replacement-ready certification:
`cpu_stage6_regular_batch_sumcheck` still needs canonical timing and memory
evidence before it can move from `Required` to `ParityCertified`.

Code touched:

- `crates/jolt-backends/src/cpu/read_write_matrix/stage6.rs`: replaced the
  dense Booleanity checkpoint with the core-shaped split-eq algorithm:
  `GruenSplitEqPolynomial` over address/cycle variables, an expanding
  low-to-high address table, and `SharedRaPolynomials` for the cycle phase.
- `crates/jolt-backends/src/cpu/read_write_matrix/stage6.rs`: replaced dense
  cycle materialization for bytecode read-RAF, RAM RA virtualization, and
  instruction RA virtualization with delayed `RaPolynomial` state keyed by
  deterministic per-cycle chunk indices.
- `crates/jolt-prover/src/stages/stage6/prove.rs`: removed the temporary dense
  bytecode debug replay path and passes Booleanity reference points to the
  backend in the same internal order as core.

Correctness evidence:

- `cargo check -p jolt-backends -q`
- `cargo check -p jolt-prover -q`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'test(stage6_regular_batch_sumcheck_verifier_replay_verifies_against_core_fixtures)'`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'test(stage6_regular_batch_sumcheck_verifier_replay_verifies_against_core_advice_fixture)'`

Algorithmic parity note: this slice removes the correctness-only dense
cycle-polynomial fallback for the RA-heavy Stage 6 relations. The accepted
shape is now backend-owned split-eq / delayed-RA orchestration, which is the
minimum plausible route to core timing and memory parity. The remaining work is
to add the canonical Stage 6 sumcheck evidence writer, run the benchmark, record
the evidence path in `cpu_stage6_regular_batch_sumcheck`, and only then claim
performance parity.

### Slice 4 — Stage 6 regular-batch sumcheck parity certification (2026-05-29)

Status: transparent/advice Stage 6 regular-batch sumcheck correctness remains
green, and the CPU backend kernel now has canonical timing and memory evidence
inside the 15% replacement gate.

Code touched:

- `crates/jolt-prover-harness/src/core_fixture.rs`: added a native core
  Stage 6 sumcheck benchmark fixture. The fixture advances the modular
  verifier through Stage 5, seeds the core reference opening accumulator with
  the exact pre-Stage-6 openings needed by `jolt-core`, filters out Stage 6
  output openings so the reference prover appends them itself, and compares the
  modular proof, challenges, batching coefficients, and final claim against the
  imported core proof.
- `crates/jolt-prover-harness/benches/frontier_perf.rs`: added the canonical
  `cpu_stage6_regular_batch_sumcheck` evidence writer and analytical memory
  accounting.
- `crates/jolt-backends/src/cpu/read_write_matrix/stage6.rs`: completed the
  prefix/suffix RA virtualization kernels for RAM RA virtualization and
  instruction RA virtualization by switching their cycle phase to
  `GruenSplitEqPolynomial`, low-to-high delayed `RaPolynomial` binding, and
  fixed-degree linear-product kernels.
- `crates/jolt-poly/src/split_eq.rs`: added generic split-eq
  `gruen_poly_from_evals` support needed by the fixed-degree RA product
  kernels.
- `crates/jolt-prover/src/stages/stage6/prove.rs`: passes RA virtualization
  cycle challenges in core low-to-high order to the backend.
- `crates/jolt-prover-harness/src/optimization.rs`: records the Stage 6
  regular-batch sumcheck evidence path and promotes
  `cpu_stage6_regular_batch_sumcheck` to `ParityCertified`.

Correctness evidence:

- `cargo fmt -q`
- `cargo check -p jolt-backends -q`
- `cargo check -p jolt-prover -q`
- `cargo check -p jolt-prover-harness --features core-fixtures --benches --tests -q`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'test(stage6_regular_batch_sumcheck_verifier_replay_verifies_against_core_fixtures) | test(stage6_regular_batch_sumcheck_verifier_replay_verifies_against_core_advice_fixture)'`

Performance evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage6_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  wrote
  `target/frontier-metrics/kernel-evidence/cpu_stage6_regular_batch_sumcheck/frontier_perf_stage6_regular_batch_sumcheck.json`
  with status `Warn`, which is accepted by the canonical parity gate.
- Reported ratios over 3 samples: time `1.073947798622785`, peak RSS
  `0.329065068913752`.
- Absolute means: core `26.15602766666667ms`, modular
  `28.090208333333333ms`; peak RSS core `23645208` bytes, modular `7780812`
  bytes.

Justification: the original Stage 6 regular-batch sumcheck benchmark failed at
`1.2240905447384158x` core time despite strong memory behavior, confirming the
missing prefix/suffix kernel suspicion. After porting the full split-eq
delayed-RA product shape for the RA virtualization relations, the same
canonical evidence path passes within the required 15% timing gate and remains
substantially below core peak memory. Transparent and advice replay both match
native `jolt-verifier` at the Stage 6 frontier.

### Slice 5 — Field-inline Stage 6 real frontier replay and bytecode projection (2026-05-30)

Status: field-inline Stage 6 correctness is green on the real SDK guest
frontier through native `jolt-verifier` replay. This does not promote
`cpu_field_inline_stage6_registers_inc_claim_reduction` to `ParityCertified`;
that kernel remains `Ported` until a canonical timing and memory evidence writer
exists and passes the 15% gate.

Code touched:

- `crates/jolt-claims/src/protocols/field_inline/formulas/bytecode.rs`: added
  `base_jolt_bytecode_row`, the canonical projection from field-inline bytecode
  rows into ordinary Jolt bytecode rows. It strips ordinary x-register operands
  from pure field ops, keeps the ordinary `rs1` read for `field.load_from_x`,
  and keeps the ordinary `rd` write for `field.store_to_x`. The helper matches
  by stable Jolt instruction tags so verifier-only field-inline builds do not
  need to enable `jolt-riscv/field-inline`.
- `crates/jolt-prover/src/stages/stage6/prove.rs`: uses that base projection
  for bytecode read-RAF row values and expected public values, then adds
  field-inline bytecode public values exactly like
  `jolt-verifier/src/stages/stage6/verify.rs`.
- `crates/jolt-backends/src/cpu/read_write_matrix/stage6.rs`: restored the
  bytecode read-RAF input check to an allocation-free scalar parallel sum after
  removing temporary diagnostic component vectors.

Correctness evidence:

- `cargo check -p jolt-verifier --features field-inline -q`
- `cargo check -p jolt-prover -q`
- `cargo check -p jolt-prover --features field-inline -q`
- `cargo check -p jolt-prover --features zk,field-inline -q`
- `cargo nextest run -p jolt-backends --features field-inline field_registers_inc_claim_reduction_matches_dense_reference --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline field_inline_eq_poly_guest_replays_modular_frontier_with_jolt_verifier --cargo-quiet`

Algorithmic parity note: the high-volume Stage 6 path must remain the
prefix/suffix, split-eq, delayed-RA backend path recorded in Slice 4. Dense or
materialized replays are acceptable only as reference oracles. The field-inline
increment claim-reduction addition is a small field-register increment kernel;
it is not a replacement for the full bytecode/RA prefix-suffix path and must get
its own benchmark evidence before field-inline Stage 6 performance parity is
claimed.

### Slice 6 — Regular increment claim-reduction prefix/suffix kernel (2026-05-30)

Status: the regular Stage 6 increment claim-reduction backend state now uses a
two-phase prefix/suffix algorithm instead of retaining full dense
`RamInc`/`RdInc` plus four dense equality tables. The `cpu_stage6_regular_batch_sumcheck`
certification evidence was regenerated after this kernel change and remains
inside the required timing and memory gate.

Code touched:

- `crates/jolt-backends/src/cpu/read_write_matrix/stage6.rs`: replaced the
  regular `IncClaimReductionState` dense implementation with a prefix phase
  over sqrt-sized P/Q buffers, followed by a suffix phase over prefix-bound
  increment and equality buffers. Added a dense-reference unit test for the
  transition and final output claims.
- `crates/jolt-backends/src/cpu/sumcheck/mod.rs`: propagated the new fallible
  increment round evaluation result.
- `crates/jolt-prover/src/stages/stage6/prove.rs`: fixed the field-inline
  Stage 6 increment optimization id to the ledger-owned
  `cpu_field_inline_stage6_registers_inc_claim_reduction` row.

Correctness evidence:

- `cargo nextest run -p jolt-backends inc_claim_reduction_prefix_suffix_matches_dense_reference --cargo-quiet`
- `cargo check -p jolt-prover -q`
- `cargo check -p jolt-prover --features field-inline -q`
- `cargo check -p jolt-prover --features zk,field-inline -q`
- `cargo nextest run -p jolt-prover-harness stage6 --features core-fixtures --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline --cargo-quiet -E 'binary(field_inline_sdk_guest)'`

Performance evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage6_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  regenerated
  `target/frontier-metrics/kernel-evidence/cpu_stage6_regular_batch_sumcheck/frontier_perf_stage6_regular_batch_sumcheck.json`
  with status `Warn`, which is accepted by the canonical parity gate.
- Reported ratios over 3 samples: time `1.053832181501639`, peak RSS
  `0.3053667990883594`.
- Absolute means: core `26.783861ms`, modular `28.22569466666667ms`; peak RSS
  core `25480216` bytes, modular `7780812` bytes.

Justification: the regular Stage 6 frontier now avoids the remaining dense
increment claim-reduction kernel in the high-volume certified path. Native
verifier replay still accepts the modular Stage 6 proof on real core fixtures
and the refreshed benchmark remains within the 15% timing gate while using less
peak memory than core. Field-inline Stage 6 correctness remains covered by the
SDK guest replay, but field-inline Stage 6 performance parity is still blocked
on a dedicated evidence writer for
`cpu_field_inline_stage6_registers_inc_claim_reduction`.

### Slice 7 — Field-inline increment prefix/suffix evidence (2026-05-30)

Status: the field-inline Stage 6 increment claim-reduction kernel now uses the
same prefix/suffix pattern as the regular increment reduction and has canonical
timing/memory evidence for the real field-inline SDK trace surface.

Code touched:

- `crates/jolt-backends/src/cpu/read_write_matrix/field_registers.rs`: replaced
  the field-register increment dense state with a prefix phase over sqrt-sized
  `P/Q` buffers and a suffix phase over prefix-bound `FieldRdInc` plus the
  verifier coefficient table. The state stores only the reversed `FieldRdInc`
  column, not full field-register rows, so peak memory is below the dense
  reference.
- `crates/jolt-prover-harness/benches/frontier_perf.rs`: added the canonical
  evidence writer for
  `cpu_field_inline_stage6_registers_inc_claim_reduction`. The writer traces the
  real `field-inline-eq-poly-guest`, extracts real field-register rows, checks
  the prefix/suffix kernel against the previous dense reference, and writes the
  ledger evidence.
- `crates/jolt-prover-harness/src/optimization.rs`: promoted
  `cpu_field_inline_stage6_registers_inc_claim_reduction` to
  `ParityCertified` with the new evidence file.

Correctness evidence:

- `cargo check -p jolt-backends --features field-inline -q`
- `cargo check -p jolt-prover-harness --features core-fixtures,field-inline --benches -q`
- `cargo nextest run -p jolt-backends --features field-inline field_registers_inc_claim_reduction_matches_dense_reference --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline field_inline_eq_poly_guest_replays_modular_frontier_with_jolt_verifier --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline --cargo-quiet -E 'binary(optimization_inventory)'`

Performance evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_field_inline_stage6_registers_inc_claim_reduction cargo bench -p jolt-prover-harness --features core-fixtures,field-inline --bench frontier_perf --quiet`
  wrote
  `target/frontier-metrics/kernel-evidence/cpu_field_inline_stage6_registers_inc_claim_reduction/frontier_perf_stage6_field_inline_registers_inc_claim_reduction.json`
  with status `Pass`.
- Reported ratios over 3 samples: time `0.7200247454568889`, peak RSS
  `0.6666666666666666`.

Justification: field-inline Stage 6 increment reduction is no longer a
dense-only or correctness-only path. The accepted backend shape is the
prefix/suffix algorithm, the real SDK frontier still replays through native
`jolt-verifier`, and the ledger now has canonical evidence under the 15%
timing and memory gate. Stage 4 and Stage 5 field-inline kernel evidence is
tracked in their stage specs; this Stage 6 increment row is no longer a
field-inline blocker. Slice 9 covers the Stage 6 committed boundary; full
BlindFold assembly remains top-level work.

### Slice 8 — Bytecode read-RAF split-eq pushforward evidence (2026-05-30)

Status: the Stage 6 bytecode read-RAF address-phase pushforward now mirrors the
optimized core algorithm instead of materializing full per-stage cycle equality
tables. This is the Stage 6 bytecode analogue of the broader prefix/suffix
requirement: instruction read-RAF uses `PrefixSuffixDecomposition`, while core
bytecode read-RAF uses a two-table split-eq construction over cycle variables
with touched-PC accumulation into bytecode-address tables.

Code touched:

- `crates/jolt-backends/src/cpu/read_write_matrix/stage6.rs`: replaced the
  dense `Eq(r_cycle)` pushforward in `BytecodeReadRafState::new` with
  `bytecode_pc_pushforwards`, which splits each cycle equality point into high
  and low tables, accumulates low-table contributions by touched PC, then scales
  by the high-table contribution. Field-inline extra bytecode stages use the
  same helper through `bytecode_pc_pushforward`.
- `crates/jolt-backends/src/cpu/read_write_matrix/stage6.rs`: added
  `bytecode_pc_pushforward_split_eq_matches_dense_reference` to check the
  split-eq/touched-PC result against the previous dense oracle.
- `crates/jolt-prover-harness/src/optimization.rs`: records
  `OPT-REL-004` and `OPT-REL-005` on
  `cpu_stage6_regular_batch_sumcheck`, because the certified Stage 6 kernel now
  owns bytecode read-RAF gamma precomputation and simultaneous Val/pushforward
  behavior.
- `crates/jolt-prover-harness/benches/frontier_perf.rs`: updated the canonical
  Stage 6 evidence writer so regenerated evidence includes those optimization
  IDs.

Correctness evidence:

- `cargo nextest run -p jolt-backends bytecode_pc_pushforward_split_eq_matches_dense_reference --cargo-quiet`
- `cargo check -p jolt-prover --features zk,field-inline -q`

Performance evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage6_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  wrote
  `target/frontier-metrics/kernel-evidence/cpu_stage6_regular_batch_sumcheck/frontier_perf_stage6_regular_batch_sumcheck.json`
  with status `Warn`, which is accepted by the canonical parity gate.
- Reported ratios over 3 samples: time `1.115014540475876`, peak allocation
  `0.9722659530046627`.

Justification: the Stage 6 regular-batch kernel no longer hides the bytecode
read-RAF address pushforward behind dense full-cycle equality tables. It now
preserves core's split-eq/touched-PC algorithm for the bytecode relation, keeps
the RA virtualization and increment prefix/suffix kernels from earlier slices,
and the regenerated ledger evidence remains inside the required 15% timing and
memory gates with the bytecode optimization IDs included. Full
`registered_parity_certified_kernel_evidence_files_are_valid` was not rerun
after target cleanup because the cleanup removed unrelated certified-kernel
evidence artifacts; the Stage 6 evidence writer itself validates this kernel's
canonical gate before writing.

### Slice 9 — ZK committed-boundary replay through Stage 6 (2026-05-30)

Status: Stage 6 now has a native committed-boundary prover path that shares the
same backend sumcheck loop as the clear path. The clear and committed paths use
one proof-sink abstraction: transparent mode appends compressed round
polynomials and opening claims, while ZK mode commits round polynomials and
commits verifier-order output claim rows.

Code quality notes:

- `Stage6CommittedBoundaryOutput` carries the verifier-visible committed
  proof/public/output-claim rows plus a hidden `Stage6ClearOutput` assembled from
  the committed run's transcript-derived points. This is prover state for the
  next frontier only; it is not spliced into the verifier proof shell.
- Clear and committed Stage 6 output assembly share
  `stage6_claims_and_verifier_output`, so the next-stage dependency object cannot
  drift between transparent and ZK frontiers.
- The real field-inline SDK frontier test now feeds Stage 7 from the committed
  Stage 6 boundary rather than a transparent Stage 6 replay.

Correctness evidence:

- `cargo check -p jolt-prover --features zk,field-inline -q`
- `cargo check -p jolt-prover-harness --features core-fixtures,field-inline,zk -q`
- `cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest_accepts_zk_committed_stage3_to_stage8_boundaries --features core-fixtures,field-inline,zk --cargo-quiet`

Justification: this proves native `jolt-verifier` acceptance at the Stage 6
committed boundary on real field-inline SDK trace data. The verifier replay runs
through `verify_until_stage1`, then stages 1-6 in ZK mode, and requires
`Stage6Output::Zk` with public output, committed output-claim shape, and
transcript state matching the modular prover. The heavy compute path remains the
same certified Stage 6 backend path from Slice 8, so the committed-boundary work
does not introduce a dense or fixture-only alternate prover.
