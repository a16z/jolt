# Stage 3 Shift, Instruction Input, and Register Reduction Frontier Spec

## Scope

Stage 3 proves the regular batched sumcheck over the Jolt VM transition claims
that bridge Spartan/product outputs into later register read-write checking. It
starts after Stage 2 has produced `Stage2Output` and the transcript is
positioned at the Stage 3 boundary. It ends after the verifier-owned Stage 3
proof field, clear claims or committed ZK shape, and downstream Stage 3 output
are assembled.

The stage boundary is:

1. Consume Stage 1 and Stage 2 outputs plus normalized Jolt VM witness views.
2. Derive Stage 3 Fiat-Shamir challenges in the exact `jolt-verifier` order.
3. Prove the regular Stage 3 batched sumcheck over:
   - Spartan shift;
   - instruction input virtualization;
   - register claim reduction.
4. Evaluate every Stage 3 output opening at the verifier-derived points.
5. Assemble the verifier-owned Stage 3 proof fields and claims.
6. Return typed Stage 3 public output for Stage 4 and later stages without
   recomputing verifier reductions.

Stage 3 produces the verifier component for:

- `JoltStageProofs::stage3_sumcheck_proof`;
- clear-mode `ClearProofClaims::stage3`;
- ZK-mode committed Stage 3 sumcheck proof and output-claim commitments that
  `jolt-verifier` lowers into BlindFold;
- clear-mode `Stage3ClearOutput` data needed by Stage 4;
- ZK-mode `Stage3ZkOutput` data needed by later BlindFold construction.

Stage 3 does not prove register read-write checking, RAM value checks, later
lookup reductions, or the final PCS opening proof. It must, however, preserve
the Stage 3 output aliases and transcript order because Stage 4 and BlindFold
depend on those exact openings.

Stage 3 should land as one complete stage slice. Subfrontier checkpoints are
useful for review and benchmarking, but Stage 3 is not accepted as
replacement-ready until transparent, advice, BlindFold/ZK committed-boundary,
field-inline compile/replay, and supported combined feature paths have their
documented correctness and performance evidence.

Every Stage 3 implementation path must explicitly support:

- advice, by consuming advice-influenced Stage 1 and Stage 2 verifier outputs
  without duplicating claim semantics;
- BlindFold/ZK mode, including committed regular-batch proof data, committed
  output-claim rows, and verifier acceptance at the Stage 3 committed boundary;
- field-inline builds, even though Stage 3 has no field-inline-specific
  relation. The stage must compile and replay under `field-inline` and
  `zk + field-inline` whenever those feature combinations are supported.

## Monitored Workflow

Stage work proceeds in one reviewable Stage 3 slice:

1. Confirm current inventory for `jolt-prover`, `jolt-verifier`,
   `jolt-backends`, `jolt-witness`, and `jolt-prover-harness`.
2. Tighten backend evidence before accepting prover orchestration.
3. Define the canonical Stage 3 input/output/prover-state API.
4. Refactor toward one public `prove` entrypoint, ordered first in `prove.rs`.
5. Implement transparent and advice paths through native Stage 3 verifier
   replay.
6. Confirm field-inline builds use the same canonical path.
7. Add ZK committed-boundary proof assembly and verifier replay.
8. Run focused tests, real fixture replay, and kernel evidence gates.
9. Append the final correctness and performance parity justification to this
   spec.
10. Stop for review before moving to Stage 4.

Every fact in the implementation should have a clear owner:

- `jolt-verifier` owns proof fields, verifier outputs, transcript checks,
  clear claims, committed proof shapes, and output aliases;
- `jolt-claims` owns relation IDs, dimensions, formula semantics, opening IDs,
  challenge IDs, and public IDs;
- `jolt-witness` owns trace-backed witness views and primitive row views;
- `jolt-backends` owns heavy compute and slot-keyed kernel results;
- `jolt-riscv` owns instruction and flag enums, but Stage 3 prover
  orchestration should normally reach those through `jolt-claims` formula
  helpers instead of spelling flag lists locally;
- `jolt-lookup-tables` owns lookup table semantics, but Stage 3 should not need
  it directly because instruction lookup/table reductions are Stage 2 and Stage
  5 work;
- `jolt-prover-harness` owns migration-only fixtures, verifier replay, and
  performance evidence.

The public prover path should remain linear enough to audit against
`jolt-verifier/src/stages/stage3/verify.rs`.

### Modular Crate Usage

Stage 3 should follow the same ownership split as `jolt-verifier`.

- Use `jolt-claims::protocols::jolt::formulas::spartan` for Spartan shift
  relation metadata, input openings, output openings, challenges, and public
  IDs.
- Use `jolt-claims::protocols::jolt::formulas::instruction` for instruction
  input virtualization metadata and output-opening order.
- Use `jolt-claims::protocols::jolt::formulas::claim_reductions::registers`
  for register claim-reduction metadata and output-opening order.
- Use `JoltOpeningId` as the semantic request key where practical, then convert
  to witness oracles with
  `jolt_witness::protocols::jolt_vm::jolt_opening_oracle_ref`.
- Use `JoltVirtualPolynomial` only as the witness-oracle identifier after the
  semantic opening ID has been chosen.
- Avoid direct `jolt_riscv::{CircuitFlags, InstructionFlags}` imports in
  production Stage 3 orchestration unless a local helper is explicitly bridging
  from a `jolt-claims` opening ID to a witness view and no shared helper exists.
- Do not import `jolt-lookup-tables` in Stage 3 production code. If lookup-table
  identity or table flags appear necessary here, that is probably a sign that
  Stage 2 or Stage 5 semantics are leaking across the stage boundary.

This keeps `jolt-riscv` as the instruction/flag definition crate while keeping
the Stage 3 prover keyed by the same semantic formula helpers as the verifier.

## Current Inventory

### jolt-verifier

Relevant verifier code lives in `crates/jolt-verifier/src/stages/stage3/`.

- `verify.rs` derives, in order:
  - `shift_gamma = transcript.challenge_scalar()`;
  - `instruction_gamma = transcript.challenge_scalar()`;
  - `registers_gamma = transcript.challenge_scalar()`.
- Clear mode verifies a compressed Boolean batched sumcheck in this statement
  order:
  1. Spartan shift;
  2. instruction input virtualization;
  3. register claim reduction.
- Clear input claims are computed from Stage 1 and Stage 2 outputs:
  - shift consumes `next_unexpanded_pc`, `next_pc`, `next_is_virtual`,
    `next_is_first_in_sequence`, and Stage 2 `next_is_noop`;
  - instruction input consumes the Stage 2 product left/right instruction
    inputs and verifies they match the instruction-claim reduction aliases;
  - register claim reduction consumes Stage 1 `rd_write_value`, `rs1_value`,
    and `rs2_value`.
- Clear output reconstruction uses:
  - `EqPlusOnePolynomial` from Stage 2 product tau and product remainder
    points for Spartan shift;
  - `eq(instruction_opening_point, product_remainder_opening_point)` for
    instruction input;
  - `eq(registers_opening_point, product_tau_low)` for register claim
    reduction.
- The verifier appends Stage 3 opening claims in this transcript order:
  1. shift `unexpanded_pc`;
  2. shift `pc`;
  3. shift `is_virtual`;
  4. shift `is_first_in_sequence`;
  5. shift `is_noop`;
  6. instruction input `left_operand_is_rs1`;
  7. instruction input `rs1_value`;
  8. instruction input `left_operand_is_pc`;
  9. instruction input `right_operand_is_rs2`;
  10. instruction input `rs2_value`;
  11. instruction input `right_operand_is_imm`;
  12. instruction input `imm`;
  13. register claim reduction `rd_write_value`.
- `Stage3Claims` still carries the aliased output values
  `instruction_input.unexpanded_pc`, `registers_claim_reduction.rs1_value`,
  and `registers_claim_reduction.rs2_value`, but those values are not appended
  separately. They alias shift `unexpanded_pc`, instruction `rs1_value`, and
  instruction `rs2_value`.
- ZK mode verifies committed consistency for `stage3_sumcheck_proof` and
  requires `13` committed output claims.
- `outputs.rs` owns `Stage3PublicOutput`, `Stage3ClearOutput`,
  `Stage3ZkOutput`, `Stage3Output`, `VerifiedStage3Batch`, and
  `VerifiedStage3Sumcheck`.
- `stages/zk/blindfold/stage3.rs` lowers Stage 3 into BlindFold using the same
  statement order, the same 13 output IDs, and explicit aliases for the three
  duplicated semantic openings.

Stage 3 prover code must import verifier-owned proof, claim, and output structs
directly rather than duplicating those shapes locally.

### jolt-prover

Current implementation lives in `crates/jolt-prover/src/stages/stage3/`.

- `input.rs` defines only `Stage3ProverConfig { log_t }`.
- `prove.rs` exposes helper entrypoints rather than one canonical public
  Stage 3 `prove` entrypoint:
  - `derive_stage3_regular_batch_prefix`;
  - `evaluate_stage3_output_openings`;
  - `prove_stage3_transparent_sumchecks`;
  - `append_stage3_opening_claims`.
- The transparent sumcheck helper materializes output-opening witness views,
  builds dense local polynomial contexts, computes round polynomials, verifies
  the final claim locally, and emits a clear compressed sumcheck proof.
- There is no ZK committed Stage 3 proof assembly path.
- There is no canonical `Stage3ProverInput` bundle that consumes Stage 1/2
  outputs, witness providers, protocol config, and mode.
- There is no canonical `Stage3ProverOutput` carrying verifier-owned proof
  fields, clear/ZK output, and prover-local opening or BlindFold state.
- `output.rs` uses `HashMap` for slot collection. Replace it with `BTreeMap`
  so Stage 3 matches the deterministic Stage 0/1 pattern and produces stable
  duplicate/missing/extra slot diagnostics.
- Stage 3 request slots are:
  - shift openings start at `0`;
  - instruction-input openings start at `16`;
  - register claim-reduction openings start at `32`.
- The request slot order for instruction-input openings is right-side fields
  first, then left-side fields. The transcript append order is verifier-owned
  and different. Keep both orders explicit.
- `request.rs` currently imports `jolt_riscv::{CircuitFlags, InstructionFlags}`
  to build `JoltVirtualPolynomial` lists directly. The target cleanup is to
  derive those lists from the `jolt-claims` Stage 3 formula opening helpers and
  use `jolt_opening_oracle_ref` for witness lookup, matching the verifier's
  modular ownership.

Current code is a useful correctness frontier, but it is not yet a
replacement-ready prover stage because the public API is helper-based, the
regular-batch sumcheck is dense-materialized in prover code, and ZK committed
proof output is missing.

### jolt-backends

Stage 3 heavy compute belongs in `jolt-backends`.

Current relevant APIs and kernels:

- `SumcheckBackend::materialize_sumcheck_views` for current dense transparent
  helper materialization;
- `SumcheckBackend::evaluate_sumcheck_views` for output-opening evaluations;
- `cpu_materialized_opening_evaluations`, a parity-certified opening kernel
  used by Stage 3 output-opening checkpoints;
- generic sumcheck primitives, which can remain test/reference scaffolding but
  must not be accepted as the replacement path if core has specialized Stage 3
  algorithms.

Known ledger status:

- `cpu_stage3_regular_batch_input_claims`: `ParityCertified`; canonical
  evidence file
  `target/frontier-metrics/kernel-evidence/cpu_stage3_regular_batch_input_claims/frontier_perf_stage3_regular_batch_inputs.json`.
- `cpu_stage3_regular_batch_sumcheck`: `ParityCertified`; canonical evidence
  file
  `target/frontier-metrics/kernel-evidence/cpu_stage3_regular_batch_sumcheck/frontier_perf_stage3_regular_batch_sumcheck.json`.
- `cpu_materialized_opening_evaluations`: `ParityCertified`; evidence file
  `target/frontier-metrics/kernel-evidence/cpu_materialized_opening_evaluations/cpu_openings_rlc_materialized_fallback.json`.

The optimization inventory identifies Stage 3 specialized shift and
instruction-input polynomial behavior under `OPT-SP-006`. The accepted Stage 3
regular-batch frontier names `OPT-SP-006` in both the manifest and backend
ledger, and the production shift statement uses the full core-style eq+1
prefix/suffix kernel. Do not hide a specialized core algorithm behind only
generic `OPT-SC-*` coverage.

Stage 3 is a backend-first frontier. If existing primitives miss timing or
memory parity for transparent, advice, BlindFold/ZK, field-inline, or supported
combined feature paths, add or port the specialized backend kernel before
accepting the prover slice.

### jolt-prover-harness

Existing Stage 3 harness coverage is useful but not complete acceptance.

Current manifest frontiers:

- `stage3_regular_batch_inputs`;
- `stage3_output_openings`;
- `stage3_regular_batch_sumcheck`.

Current transparent checkpoint and replay tests:

- `stage3_regular_batch_input_checkpoint_matches_core_fixtures`;
- `stage3_output_opening_checkpoint_matches_core_fixtures`;
- `stage3_regular_batch_sumcheck_verifier_replay_verifies_against_core_fixtures`;
- manifest gate tests for Stage 3 input, output opening, and sumcheck
  frontiers.

Current limitations:

- coverage is transparent only;
- fixtures are `MuldivSmall` and `AdviceConsumer`;
- field-inline Stage 3 replay is not separately documented;
- ZK committed-boundary replay through `stage3::verify` is missing;
- backend replacement-readiness for Stage 3 input claims and sumcheck is not
  certified.

Required new/expanded harness coverage:

- full clear native verifier replay that replaces `stage3_sumcheck_proof` and
  clear Stage 3 claims in a real proof shell;
- advice replay for the full Stage 3 clear component;
- field-inline replay showing Stage 3 uses the same canonical path and does not
  perturb field-inline proof shape;
- ZK committed-boundary replay that runs `verify_until_stage1`, `stage1::verify`,
  `stage2::verify`, and then `stage3::verify`, requiring `Stage3Output::Zk`;
- `zk + field-inline` committed-boundary replay if the workspace supports the
  combination;
- replacement-readiness gates for Stage 3 regular-batch input and sumcheck
  kernels once evidence writers exist.

## Target Prover Shape

### Public API

Stage 3 should expose one canonical public entrypoint first in `prove.rs`:

```rust
pub fn prove<F, W, B, T, C>(
    input: Stage3ProverInput<'_, W, ...>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage3ProverOutput<F, C>, ProverError>
```

The exact type parameters should follow the existing Stage 1/2 style and the
compiled feature surface. The input bundle should include:

- `Stage3ProverConfig` or a unified config containing `log_t`;
- Stage 1 and Stage 2 outputs, clear or ZK according to mode;
- a Jolt VM witness provider;
- verifier protocol config and proof mode;
- any field-inline provider only if a later unified API requires it. Stage 3
  should not introduce field-inline-local semantics.

The input and request builders should be keyed by verifier/protocol semantic
IDs (`JoltOpeningId`, `JoltChallengeId`, `JoltPublicId`, and
`JoltRelationId`) before lowering to witness `OracleRef`s. Do not encode Stage
3's instruction flag list by hand in the prover when the corresponding
`jolt-claims` formula helper already defines it.

`Stage3ProverOutput` should contain:

- `stage3_sumcheck_proof`;
- clear `Stage3Claims` when transparent;
- committed output-claim data and committed witness retention handles when ZK;
- typed `Stage3Output`-equivalent data for Stage 4;
- prover-local opening metadata needed by final opening proof assembly.

Do not make Stage 4 reconstruct Stage 3 public output from raw proof fields when
the Stage 3 prover already computed it.

### Clear Prover Order

The clear public `prove` implementation should read like the verifier:

1. Validate clear Stage 1 and Stage 2 dependencies.
2. Derive `shift_gamma`, `instruction_gamma`, and `registers_gamma`.
3. Verify Stage 2 product and instruction-reduction dependencies agree on the
   left/right instruction inputs.
4. Build Stage 3 input claims in verifier statement order:
   shift, instruction input, register claim reduction.
5. Prove the regular-batch compressed Boolean sumcheck in verifier statement
   order.
6. Derive shift, instruction-input, and register claim-reduction opening points
   from the batched verifier reduction.
7. Evaluate Stage 3 output openings through backend requests.
8. Recompute expected output claims using `jolt-claims` formulas and verifier
   public/challenge IDs.
9. Verify the batched final claim locally before emitting the proof component.
10. Append Stage 3 opening claims in verifier transcript order, respecting the
    three output aliases.
11. Return `Stage3ClearOutput`-equivalent data for Stage 4.

### ZK Prover Order

The ZK public `prove` implementation should mirror the verifier's committed
boundary:

1. Validate ZK Stage 1 and Stage 2 dependencies.
2. Derive the same public challenges as clear mode.
3. Produce committed Stage 3 batch proof data in `stage3_sumcheck_proof`.
4. Produce exactly `13` committed output-claim rows with the verifier-owned
   alias layout.
5. Retain the committed round witnesses and hidden output-claim witnesses for
   BlindFold.
6. Return public consistency data required by downstream ZK stages and
   BlindFold.

The Stage 3 spec does not require full BlindFold verification yet. The required
ZK acceptance scope is native verifier committed-boundary acceptance through
`stage3::verify` returning `Stage3Output::Zk`. Full BlindFold proof
verification remains Stage 8/full JoltProof work.

## Feature Requirements

### Advice

Stage 3 is advice-correct when:

- full clear Stage 3 native verifier replay passes for `muldiv` and advice
  fixtures;
- advice-influenced Stage 1 and Stage 2 openings are consumed through
  verifier-owned outputs, not duplicated local claim structs;
- input claims, output claims, transcript challenges, batching coefficients,
  opening points, and final claims match core fixture checkpoints;
- transcript state after Stage 3 matches native verifier replay.

Stage 3 itself does not add advice commitments, but it must preserve advice
semantics through prior-stage dependencies.

### BlindFold/ZK

Stage 3 is ZK-correct at the current frontier when:

- `stage3_sumcheck_proof` is a committed proof, not a clear proof with a ZK
  flag;
- committed proof statements are ordered as shift, instruction input, register
  claim reduction;
- committed output-claim rows match the verifier's `13` output IDs;
- BlindFold aliases are retained for instruction `unexpanded_pc` and register
  `rs1_value`/`rs2_value`;
- native verifier committed-boundary replay reaches `Stage3Output::Zk`;
- any Stage 3 ZK-specific heavy compute or commitment work not covered by the
  shared BlindFold kernel has performance evidence before acceptance.

### Field-Inline

Stage 3 has no field-inline-specific relation. Under the `field-inline`
feature, the canonical Stage 3 path must:

- compile without a separate public prover fork;
- consume the same Stage 1 and Stage 2 verifier outputs used by the verifier;
- leave field-inline proof shape untouched;
- participate in field-inline native verifier replay through Stage 3 when real
  field-inline fixtures are available;
- support `zk + field-inline` committed-boundary replay if the workspace
  supports that combination.

Field-inline must compile out cleanly when the feature is disabled.

## Backend and Kernel Requirements

Stage 3 must keep the backend-first rule:

1. Port or validate the CPU kernel in isolation.
2. Add focused backend/harness microbenchmarks with analytical memory
   accounting.
3. Record the kernel in the backend ledger.
4. Wire `jolt-prover` through backend requests.
5. Accept the frontier only after verifier correctness and performance parity
   both pass.

Required kernel surfaces:

- regular-batch input-claim construction:
  `cpu_stage3_regular_batch_input_claims` or a better specialized kernel;
- regular-batch compressed sumcheck:
  `cpu_stage3_regular_batch_sumcheck` or a better specialized kernel;
- output-opening evaluations:
  `cpu_materialized_opening_evaluations`, unless Stage 3 needs a more
  specialized opening kernel to meet parity;
- specialized Spartan shift and instruction-input virtual polynomial handling
  corresponding to `OPT-SP-006`;
- ZK committed round/output-claim work if not covered by the shared BlindFold
  kernel.

Missing parity on any required path is a backend task first. Do not accept a
slower prover-side workaround for transparent, advice, BlindFold/ZK,
field-inline, or `zk + field-inline`.

If a Stage 3 relation or virtual-polynomial path uses a core prefix/suffix
decomposition, the modular backend must preserve that full algorithm in the
accepted kernel. Dense/materialized versions may remain as reference oracles for
correctness tests, but they cannot be used as production prover paths or
replacement parity evidence.

## Performance Gates

Stage 3 replacement readiness requires the default frontier parity gate:

- timing ratio within 15%;
- peak-memory ratio within 15%;
- `KernelPortStatus::ParityCertified` for every required kernel;
- `validate_frontier_replacement_ready` passes with loaded evidence;
- `validate_global_cpu_backend_inventory_coverage` remains passing.

Current evidence state:

- materialized opening evaluations are parity-certified through
  `cpu_materialized_opening_evaluations`;
- Stage 3 regular-batch input claims are parity-certified through
  `cpu_stage3_regular_batch_input_claims`;
- Stage 3 regular-batch sumcheck proof generation is parity-certified through
  `cpu_stage3_regular_batch_sumcheck`, including `OPT-SP-006` shift
  prefix/suffix coverage;
- field-inline and ZK Stage 3 paths do not yet have accepted performance
  evidence.

Suggested evidence commands:

```bash
JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage3_regular_batch_input_claims cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage3_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
JOLT_WRITE_KERNEL_EVIDENCE=cpu_materialized_opening_evaluations cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
```

Use the actual evidence writer names when implemented. Do not mark Stage 3
replacement-ready while the input and sumcheck ledger rows remain `Required`.

## Correctness Gates

Focused prover tests should cover:

```bash
cargo nextest run -p jolt-prover stage3 --cargo-quiet
cargo nextest run -p jolt-prover stage3 --features zk --cargo-quiet
cargo nextest run -p jolt-prover stage3 --features field-inline --cargo-quiet
cargo nextest run -p jolt-prover stage3 --features zk,field-inline --cargo-quiet
```

Existing checkpoint and subfrontier tests should continue to pass:

```bash
cargo nextest run -p jolt-prover-harness stage3_regular_batch_input_checkpoint_matches_core_fixtures --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage3_output_opening_checkpoint_matches_core_fixtures --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage3_regular_batch_sumcheck_verifier_replay_verifies_against_core_fixtures --features core-fixtures --cargo-quiet
```

Required new/expanded verifier replay tests:

```bash
cargo nextest run -p jolt-prover-harness stage3_full_verifier_replay --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage3_zk_committed_boundary --features core-fixtures,zk --cargo-quiet
cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest_stage3 --features core-fixtures,field-inline --cargo-quiet
cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest_stage3_zk --features core-fixtures,zk,field-inline --cargo-quiet
```

Use actual test names when implemented. The important acceptance condition is
that native `jolt-verifier` accepts the assembled Stage 3 component, not merely
that local helper checkpoints match shapes.

Static rails:

```bash
cargo fmt -q
cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings
cargo clippy -p jolt-prover-harness --features core-fixtures,zk,field-inline --tests -q -- -D warnings
```

## Implementation Slice Plan

1. Normalize Stage 3 collections and current helpers.
   Replace `HashMap` with `BTreeMap`, keep duplicate/missing/extra slot errors
   targeted, and preserve existing checkpoint tests.

2. Define the canonical API.
   Add `Stage3ProverInput` and `Stage3ProverOutput` so the public `prove`
   entrypoint can consume Stage 1/2 outputs and return verifier-owned Stage 3
   proof/claim components plus downstream state.

3. Backend evidence first.
   Add or port Stage 3 input-claim and regular-batch sumcheck kernels before
   declaring the prover path replacement-ready. Update manifest optimization
   IDs if `OPT-SP-006` is part of the Stage 3 frontier.

4. Transparent/advice integration.
   Move existing helper logic behind the canonical `prove` flow and keep the
   verifier replay fixture accepted for `MuldivSmall` and `AdviceConsumer`.

5. Output openings.
   Evaluate shift, instruction-input, and register claim-reduction openings
   through deterministic backend requests. Assemble `Stage3Claims` directly
   from verifier-owned types and preserve transcript alias order.

6. Field-inline.
   Confirm the canonical path compiles and replays under `field-inline` without
   adding field-inline-local Stage 3 semantics.

7. ZK committed boundary.
   Emit committed Stage 3 proof fields with exactly 13 committed output claims.
   Retain committed witnesses for BlindFold and add native verifier
   committed-boundary replay through `stage3::verify`.

8. Full replay and parity gates.
   Run performance evidence and ledger gates for every Stage 3 frontier.

9. Justification log.
   Append exact code touched, correctness commands, performance evidence,
   remaining limitations, and confidence statement to this spec before marking
   Stage 3 complete.

## Acceptance Checklist

Stage 3 is accepted only when all of these are true:

- public `prove` is the canonical entrypoint and appears first in `prove.rs`;
- production Stage 3 imports verifier proof/claim/output structs directly from
  `jolt-verifier`;
- semantic ordering, dimensions, formulas, opening IDs, challenge IDs, public
  IDs, and aliases come from `jolt-claims` and `jolt-verifier`;
- witness data flows through `jolt-witness` views or primitive row providers;
- heavy compute flows through certified `jolt-backends` kernels;
- deterministic `BTreeMap`/`BTreeSet` collection is used for slot-keyed and
  variable-keyed data;
- regular-batch proof generation and all output openings are implemented in
  verifier transcript order;
- transparent and advice native verifier replay pass with real core fixtures;
- field-inline native verifier replay passes or is concretely documented as
  unsupported for Stage 3 fixtures;
- ZK committed-boundary replay runs through Stage 3 and returns
  `Stage3Output::Zk`;
- `zk + field-inline` is tested or concretely documented as unsupported;
- required performance evidence is loaded and parity-certified for every Stage
  3 backend kernel surface;
- this spec contains the final implementation-slice justification.

## Implementation Slice Justification Log

### Slice 1 — Canonical API, backend-routed sumcheck, transparent/advice correctness (2026-05)

Status: correctness landed and verifier-validated; perf-parity certification and
ZK committed-boundary remain (tracked below). Not yet marked replacement-ready.

Code touched:

- `crates/jolt-prover/src/stages/stage3/output.rs`: replaced `HashMap` slot
  collection with `BTreeMap` (deterministic duplicate/missing/extra slot
  diagnostics, matching Stage 0/1/2); added `Stage3ProverOutput` (verifier-owned
  `stage3_sumcheck_proof` + `Stage3Claims` + assembled `Stage3ClearOutput`) and
  `Stage3RegularBatchExpectedOutputs`.
- `crates/jolt-prover/src/stages/stage3/input.rs`: added the canonical
  `Stage3ProverInput` bundle (config + `CheckedInputs` + Stage 1/2 clear outputs
  + witness). Stage 3 has no field-inline relation, so the bundle is shared
  across feature modes.
- `crates/jolt-prover/src/stages/stage3/prove.rs`: added the canonical public
  `prove` entrypoint, ordered first. It mirrors
  `jolt-verifier/src/stages/stage3/verify.rs` in prover order: derive
  `shift_gamma`/`instruction_gamma`/`registers_gamma`, build the three Stage 3
  statements as backend regular-batch instances, drive the batched Boolean
  sumcheck through the backend kernel, evaluate output openings at the
  verifier-derived point, recompute expected output claims via the verifier
  formulas, check the final claim, append opening claims in verifier transcript
  order, and assemble `Stage3ClearOutput` for Stage 4. Added a
  `frontier-harness`-gated `prove_stage3_regular_batch_sumcheck_for_frontier`
  for isolated-kernel benchmarking.
- `crates/jolt-backends/src/sumcheck/request.rs` and
  `crates/jolt-backends/src/cpu/sumcheck/kernels/regular_batch.rs`: generalized
  `SumcheckRegularBatchInstance` from a single product of linear factors to a
  sum of product terms (`SumcheckRegularBatchProduct`). Stage 3's shift and
  instruction-input statements are sums of products, so each maps onto one
  instance carrying multiple product terms. The generalization is
  backward-compatible: a single-product instance is a one-element sum, leaving
  Stage 2's certified path byte-identical (Stage 2 replay re-verified green).
- `crates/jolt-prover-harness/src/core_fixture.rs`: routed the Stage 3
  regular-batch verifier replay through the canonical `prove` so the real
  `jolt-verifier` accepts the backend-routed proof.

Correctness evidence (all green):

- `cargo nextest run -p jolt-prover-harness stage3 --features core-fixtures` —
  6/6 pass, including
  `stage3_regular_batch_sumcheck_verifier_replay_verifies_against_core_fixtures`
  for `MuldivSmall` and `AdviceConsumer`: the backend-routed `prove` output is
  spliced into a real core proof shell and accepted by the native verifier.
- `cargo clippy -p jolt-prover --features zk,field-inline --all-targets -- -D warnings`
  passes; Stage 3 compiles under transparent, `zk`, `field-inline`, and
  `zk,field-inline`. Stage 3 adds no field-inline-local semantics; the single
  canonical path is shared.

Backend-first note: Stage 3 heavy compute now flows through the
`jolt-backends` `SumcheckRegularBatchState` kernel (`evaluate_sumcheck_regular_batch_round`
/ `bind_sumcheck_regular_batch_state`), the same machinery Stage 2 is certified
against, rather than a host-resident batch context.

Remaining before replacement-ready (not yet done):

- Perf-parity certification of `cpu_stage3_regular_batch_input_claims` and
  `cpu_stage3_regular_batch_sumcheck`: add the core-vs-modular benchmark
  fixtures (jolt-core Stage 3 provers on the reference side, the backend-routed
  `prove_stage3_regular_batch_sumcheck_for_frontier` on the modular side),
  emit canonical `KernelBenchmarkEvidence`, and flip the two ledger rows to
  `ParityCertified`. Open question to settle during certification: whether the
  generic sum-of-products kernel meets the 15% gate against core's specialized
  shift/instruction-input handling (`OPT-SP-006`), or whether that specialized
  algorithm must be ported.
- ZK committed-boundary: produce a committed `stage3_sumcheck_proof` with 13
  committed output-claim rows and native verifier acceptance returning
  `Stage3Output::Zk`. The modular prover does not yet implement committed
  (Pedersen) sumcheck round production for any stage, so this is shared
  infrastructure work.

Replacement-ready gate status: the `stage3_output_openings` and
`stage3_regular_batch_inputs` frontiers are replacement-ready —
`validate_frontier_replacement_ready` passes for both (tests
`stage3_output_opening_frontier_is_replacement_ready_with_certified_kernel_evidence`
and `stage3_regular_batch_input_frontier_is_replacement_ready_with_certified_kernel_evidence`).
The output-opening frontier uses the `ParityCertified`
`cpu_materialized_opening_evaluations` kernel. The input-claims frontier now has
its own `ParityCertified` `cpu_stage3_regular_batch_input_claims` kernel: the
`frontier_perf/stage3_regular_batch_inputs` microbench writes canonical
`KernelBenchmarkEvidence` (latest post-clean time ratio ≈0.9996, memory ratio
1.0, status `Pass`)
comparing the modular `derive_stage3_regular_batch_prefix` against an independent
reference recompute.

The full `stage3_regular_batch_sumcheck` frontier is also `ParityCertified`.
The core-vs-modular benchmark fixture runs the native `jolt-core` Stage 3
shift, instruction-input, and register claim-reduction sumcheck provers against
the modular backend-routed `prove_stage3_regular_batch_sumcheck_for_frontier`,
with the Stage 2 instruction-input aliases seeded at the verifier/product
remainder point and only prior-stage openings loaded into the core accumulator.
Current evidence
`cpu_stage3_regular_batch_sumcheck/frontier_perf_stage3_regular_batch_sumcheck.json`
reports status `Pass`, latest post-clean time ratio ≈0.7271, and memory ratio
≈0.6169. The
frontier manifest and ledger include `OPT-SP-006`, and the transparent Stage 3
regular-batch sumcheck now has timing and memory parity evidence against the
core specialized shift prefix/suffix path.

ZK verifier-boundary replay is now covered through Stage 3:

- `zk_stage3_committed_boundary_is_native_verifier_accepted` runs
  `verify_until_stage1`, `stage1::verify`, `stage2::verify`, and
  `stage3::verify` on real core ZK fixtures for `ZkMuldivSmall` and
  `ZkAdviceConsumer`, requiring `Stage3Output::Zk`.
- The ZK frontier-gates binary now serializes all real ZK core-fixture work
  behind a process lock and runs Stage 1/2/3 committed-boundary checks on
  64 MiB worker stacks. This avoids default test-thread stack overflows and
  makes `cargo nextest run -p jolt-prover-harness --features core-fixtures,zk
  --cargo-quiet -E 'binary(frontier_gates)'` pass deterministically.
- `field_inline_eq_poly_guest_accepts_zk_committed_stage3_to_stage5_boundaries` advances the
  real SDK `field-inline-eq-poly-guest` fixture through Stage 3 with
  `zk + field-inline` verifier flags. The mock VC packs the 13 hidden Stage 3
  output claims into one output-claim commitment at the fixture's capacity, which
  is the shape accepted by `verify_output_claim_commitments`.

Remaining before full Stage 3 acceptance at this slice: production modular ZK
committed proof assembly from the Stage 3 prover itself, including committed
round/output-claim witness retention for BlindFold. Slice 3 below supersedes
that gap for the Stage 3 committed-boundary proof object; full BlindFold
verification remains later-stage work.

### Slice 2 — Shift prefix/suffix kernel and `OPT-SP-006` certification (2026-05-30)

Status: transparent Stage 3 regular-batch proving now uses a production
prefix/suffix shift kernel and has stronger parity evidence. This supersedes the
generic dense/materialized shift path as replacement evidence; the dense path is
kept only as a private reference/audit path.

Code touched:

- `crates/jolt-witness/src/protocols/jolt_vm/mod.rs`: added
  `JoltVmStage3ShiftRows`, a trace-backed row view exposing only the primitive
  shift data needed by the backend kernel.
- `crates/jolt-backends/src/sumcheck/request.rs`,
  `crates/jolt-backends/src/traits.rs`, and
  `crates/jolt-backends/src/cpu/sumcheck/kernels/stage3_shift.rs`: added the
  Stage 3 shift request, backend trait, and CPU two-phase eq+1 prefix/suffix
  state. The kernel mirrors `jolt-core`'s `spartan/shift.rs`: first half
  prefix/suffix product buffers, second half suffix aggregation after binding
  the prefix challenges.
- `crates/jolt-prover/src/stages/stage3/prove.rs`: changed the canonical Stage
  3 sumcheck driver to use the specialized shift backend state plus the shared
  regular-batch backend for instruction-input and register reduction. The public
  orchestration remains verifier-ordered: shift, instruction-input, registers.
- `crates/jolt-prover-harness/src/manifest.rs`,
  `crates/jolt-prover-harness/src/optimization.rs`, and
  `crates/jolt-prover-harness/benches/frontier_perf.rs`: made `OPT-SP-006`
  explicit for `stage3_regular_batch_sumcheck`, pointed the ledger at
  `jolt_backends::cpu::sumcheck::kernels::stage3_shift`, and regenerated
  canonical evidence.

Correctness evidence:

- `cargo nextest run -p jolt-backends --features cpu stage3_shift_prefix_suffix_matches_dense_reference --cargo-quiet`
  passes. The test compares the backend kernel against a dense product-form
  reference with the same degree-2 sumcheck shape as core, not the wrong
  degree-1 MLE of the pointwise product.
- `cargo nextest run -p jolt-prover-harness --features core-fixtures stage3_regular_batch_sumcheck_verifier_replay_verifies_against_core_fixtures --cargo-quiet`
  passes, so native `jolt-verifier` accepts the Stage 3 component produced by
  the hybrid specialized/generic backend path.
- `cargo check -p jolt-prover --features zk,field-inline -q` and
  `cargo check -p jolt-prover-harness --features core-fixtures,zk,field-inline -q --tests`
  pass, preserving the required ZK, field-inline, and combined feature compile
  surfaces for this clear-prover slice.

Performance evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage3_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  rewrote
  `target/frontier-metrics/kernel-evidence/cpu_stage3_regular_batch_sumcheck/frontier_perf_stage3_regular_batch_sumcheck.json`.
- The evidence is `Pass` with 3 samples, optimization IDs
  `OPT-SC-007`, `OPT-EQ-004`, and `OPT-SP-006`, core time
  `2.6775416666666665 ms`, modular time `2.1652916666666666 ms`, time ratio
  ≈0.8087, core peak allocation `647704`, modular peak allocation `399564`,
  and memory ratio ≈0.6169.
- `cargo nextest run -p jolt-prover-harness --features core-fixtures stage3_regular_batch_sumcheck_frontier_is_replacement_ready_with_certified_kernel_evidence --cargo-quiet`
  passes.
- `cargo nextest run -p jolt-prover-harness --features core-fixtures backend_kernel_ledger_covers_every_cpu_backend_inventory_id registered_parity_certified_kernel_evidence_files_are_valid --cargo-quiet`
  passes.

Confidence statement:

The transparent/advice Stage 3 regular-batch frontier now has high confidence on
the required correctness and performance axes: it uses verifier-owned proof
shapes, real witness rows, the specialized backend prefix/suffix shift kernel,
deterministic frontier accounting, native verifier replay, and certified
core-parity evidence. ZK committed proof production remains the known later
BlindFold/frontier work; this slice preserves ZK and field-inline compile and
verifier-boundary surfaces without claiming full BlindFold proof verification.

### Slice 3 — Real Stage 3 committed-boundary proof production (2026-05-30)

Status: Stage 3 now has a modular committed-boundary prover path that produces
the verifier-owned `stage3_sumcheck_proof` as a committed sumcheck proof, not a
placeholder shape. The path uses the same optimized Stage 3 prefix/suffix shift
state and instruction/register regular-batch backend state as the clear prover,
commits each actual batched round polynomial, commits the 13 hidden Stage 3
output claims in verifier order, and retains committed round/output-claim witness
data for the later BlindFold assembly slice.

Code touched:

- `crates/jolt-prover/src/committed.rs`: added a reusable committed sumcheck
  builder. It owns round commitments, output-claim row commitments, randomized
  blindings, and prover-side witness retention. The builder is generic over
  `VectorCommitment`, so later stages can reuse it with the real Pedersen VC
  rather than stage-local ad hoc proof construction.
- `crates/jolt-prover/src/stages/stage3/prove.rs`: added
  `prove_committed_boundary`. It mirrors `stage3::verify`'s ZK transcript order:
  sample shift/instruction/register gammas, sample three batching coefficients
  without appending clear input claims, commit each real batched round polynomial,
  derive the same round challenges, evaluate the real output openings, and append
  the committed output-claim rows.
- `crates/jolt-prover/src/stages/stage3/output.rs`: added
  `Stage3CommittedBoundaryOutput`, carrying the verifier proof component,
  `Stage3PublicOutput`, the 13 hidden output-claim values, and retained committed
  witness state.
- `crates/jolt-prover-harness/tests/field_inline_sdk_guest.rs`: replaced the
  Stage 3 ZK boundary placeholder in the real SDK field-inline guest replay with
  the modular committed Stage 3 proof.

Correctness evidence:

- `cargo check -p jolt-prover -q` passes.
- `cargo check -p jolt-prover --features zk,field-inline -q` passes.
- `cargo check -p jolt-prover-harness --features core-fixtures,field-inline,zk -q`
  passes.
- `cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest_accepts_zk_committed_stage3_to_stage5_boundaries --features core-fixtures,field-inline,zk --cargo-quiet`
  passes. This test runs the real `field-inline-eq-poly-guest` trace, derives
  real modular Stage 3 committed round data from the witness/backend kernels,
  splices it into a ZK proof shell, and requires native `jolt-verifier` to
  accept `Stage3Output::Zk`. It also checks that the verifier public output
  matches the modular prover public output, that the 13 hidden Stage 3 output
  claims are packed into one committed output-claim row at the fixture VC
  capacity, and that the prover/verifier transcript states match after Stage 3.

Performance evidence:

- No new arithmetic kernel was introduced in this slice. The committed-boundary
  path reuses the already-certified Stage 3 full prefix/suffix shift kernel and
  regular-batch backend kernels from Slice 2. The only new work is vector
  commitment construction for actual round/output-claim rows, which is required
  only on the ZK committed boundary and is not a substitute for the Stage 3
  arithmetic parity evidence.

Scope note:

This proves native verifier acceptance at the Stage 3 committed boundary with
real modular Stage 3 data on the real SDK field-inline guest path. It is not a
full BlindFold proof verification: Stages 4-8 still need their committed-boundary
prover paths, and Stage 8 still owns the final BlindFold proof assembly and
verification handoff. The Stage 3 ZK test still uses committed Stage 1/2 boundary
proof-shell data to reach the Stage 3 verifier frontier; the Stage 3 component
itself is now real modular prover output.
