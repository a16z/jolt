# Stage 5 Instruction Read-RAF, RAM RA Reduction, and Register Value Frontier Spec

## Scope

Stage 5 proves the regular batched sumcheck over instruction read-RAF, RAM RA
claim reduction, and register value evaluation. It starts after Stage 2 and
Stage 4 have produced typed verifier outputs and the transcript is positioned
at the Stage 5 boundary. It ends after the verifier-owned Stage 5 proof field,
clear claims or committed ZK shape, and downstream Stage 5 output are
assembled.

The stage boundary is:

1. Consume Stage 2 and Stage 4 outputs plus normalized Jolt VM witness views.
2. Derive Stage 5 Fiat-Shamir challenges in the exact `jolt-verifier` order.
3. Prove the Stage 5 batched sumcheck over:
   - instruction read-RAF;
   - RAM RA claim reduction;
   - register value evaluation;
   - field-register value evaluation under `field-inline`.
4. Evaluate every Stage 5 output opening at verifier-derived points.
5. Assemble the verifier-owned Stage 5 proof fields and claims.
6. Return typed Stage 5 public output for Stage 6 without recomputing verifier
   reductions.

Stage 5 produces the verifier component for:

- `JoltStageProofs::stage5_sumcheck_proof`;
- clear-mode `ClearProofClaims::stage5`;
- ZK-mode committed Stage 5 sumcheck proof and output-claim commitments;
- clear-mode `Stage5ClearOutput` data needed by Stage 6;
- ZK-mode `Stage5ZkOutput` data needed by Stage 6 and BlindFold.

Stage 5 does not prove bytecode read-RAF, booleanity, increment reductions,
hamming-weight reductions, advice address phase, or final PCS openings. It does
own the only instruction lookup-table read-RAF frontier after the earlier
instruction claim reduction, so lookup/table semantics must be routed through
the modular lookup and witness crates rather than being forked in
`jolt-prover`.

Every Stage 5 implementation path must explicitly support:

- advice, by consuming advice-influenced Stage 2 and Stage 4 outputs without
  duplicating their claim semantics;
- BlindFold/ZK mode, including committed regular-batch proof data and committed
  output-claim rows;
- field-inline, including real field-register value-evaluation claims instead
  of placeholder zero outputs.

## Monitored Workflow

Stage work proceeds in one reviewable Stage 5 slice:

1. Confirm current inventory for `jolt-prover`, `jolt-verifier`,
   `jolt-backends`, `jolt-witness`, `jolt-lookup-tables`, and
   `jolt-prover-harness`.
2. Tighten backend evidence for instruction read-RAF and register value
   evaluation before accepting prover orchestration.
3. Define the canonical Stage 5 input/output/prover-state API.
4. Refactor toward one public `prove` entrypoint, ordered first in `prove.rs`.
5. Implement transparent and advice paths through native Stage 5 verifier
   replay.
6. Add field-inline register value-evaluation support to the same canonical
   path.
7. Add ZK committed-boundary proof assembly and verifier replay.
8. Run focused tests, real fixture replay, and kernel evidence gates.
9. Append final correctness and performance parity justification to this spec.
10. Stop for review before moving to Stage 6.

Every fact in the implementation should have a clear owner:

- `jolt-verifier` owns proof fields, verifier outputs, transcript checks,
  clear claims, committed proof shapes, output-claim counts, and output order;
- `jolt-claims` owns relation IDs, dimensions, formula semantics, opening IDs,
  challenge IDs, and public IDs;
- `jolt-witness` owns trace-backed witness views and Stage 5 instruction
  read-RAF witness contexts;
- `jolt-lookup-tables` owns lookup-table kind semantics and table-count
  constants used by instruction read-RAF;
- `jolt-riscv` owns instruction encodings and flags, but Stage 5 prover
  orchestration should not reach into it directly unless a witness adapter is
  explicitly bridging instruction rows into a canonical lookup-table API;
- `jolt-backends` owns heavy compute and slot-keyed kernel results;
- `jolt-prover-harness` owns migration-only fixtures, verifier replay, and
  performance evidence.

The public prover path should remain linear enough to audit against
`jolt-verifier/src/stages/stage5/verify.rs`.

### Modular Crate Usage

Stage 5 should follow the verifier ownership split.

- Use `jolt-claims::protocols::jolt::formulas::instruction` for
  instruction read-RAF metadata, dimensions, input/output openings,
  challenge/public IDs, and opening-point derivation.
- Use `jolt-claims::protocols::jolt::formulas::ram` for RAM RA
  claim-reduction metadata and opening IDs.
- Use `jolt-claims::protocols::jolt::formulas::registers` for register value
  evaluation metadata and opening IDs.
- Under `field-inline`, use
  `jolt-claims::protocols::field_inline::formulas::registers` for field
  register value-evaluation metadata.
- Use `jolt_witness::protocols::jolt_vm::jolt_opening_oracle_ref` to convert
  semantic `JoltOpeningId` values into witness oracles.
- Use `JoltVmStage5InstructionReadRafWitness` or its replacement witness
  facade for instruction read-RAF table/RA context construction.
- Let CPU backend lookup fast paths consume canonical lookup semantics from
  `jolt-lookup-tables` or shared adapters. If a bulk lookup API is missing, add
  it there instead of duplicating lookup formulas in `jolt-prover`.

Direct `jolt-riscv` imports in Stage 5 production orchestration should be
treated as a cleanup target. Direct `jolt-lookup-tables` use is acceptable only
for canonical lookup-table semantics and counts; it should not replace
`jolt-claims` formula helpers for proof ordering.

## Current Inventory

### jolt-verifier

Relevant verifier code lives in `crates/jolt-verifier/src/stages/stage5/`.

- `verify.rs` derives, in order:
  - `instruction_gamma = transcript.challenge_scalar()`;
  - `ram_gamma = transcript.challenge_scalar()`.
- Clear mode verifies a compressed Boolean batched sumcheck in this statement
  order:
  1. instruction read-RAF;
  2. RAM RA claim reduction;
  3. register value evaluation;
  4. field-register value evaluation under `field-inline`.
- Clear input claims are computed from:
  - Stage 2 product remainder and instruction claim-reduction lookup outputs
    plus left/right operands for instruction read-RAF;
  - Stage 2 RAM RAF evaluation, Stage 2 RAM read-write RA, and Stage 4 RAM
    value-check RA for RAM RA claim reduction;
  - Stage 4 register read-write `registers_val` for register value evaluation;
  - Stage 4 field-register read-write value claim under `field-inline`.
- The verifier reconstructs instruction read-RAF opening points from the
  instruction sumcheck point and splits the instruction address portion across
  virtual RA polynomials.
- The verifier requires RAM RA claim-reduction dependencies to use the same
  RAM address point.
- The verifier appends Stage 5 opening claims in this transcript order:
  1. instruction read-RAF lookup-table flags, in table-kind order;
  2. instruction read-RAF virtual `instruction_ra` openings;
  3. instruction read-RAF `instruction_raf_flag`;
  4. RAM RA claim-reduction `ram_ra`;
  5. register value-evaluation `rd_inc`;
  6. register value-evaluation `rd_wa`;
  7. field-register value-evaluation `field_rd_inc` under `field-inline`;
  8. field-register value-evaluation `field_rd_wa` under `field-inline`.
- ZK mode verifies committed consistency for `stage5_sumcheck_proof` and
  requires output-claim count:
  - `lookup_table_flags.len() + instruction_ra.len() + 1 + 1 + 2` without
    field-inline;
  - that count plus `2` with field-inline.
- `outputs.rs` owns `Stage5PublicOutput`, `Stage5ClearOutput`,
  `Stage5ZkOutput`, `Stage5Output`, `VerifiedStage5Batch`,
  `VerifiedInstructionReadRafSumcheck`, and `VerifiedStage5Sumcheck`.
- `stages/zk/blindfold/stage5.rs` lowers Stage 5 into BlindFold using the same
  statement order and output-claim order.

Stage 5 prover code must import verifier-owned proof, claim, and output structs
directly rather than duplicating those shapes locally.

### jolt-prover

Current implementation lives in `crates/jolt-prover/src/stages/stage5/`.

- `input.rs` defines `Stage5ProverConfig`.
- `prove.rs` exposes helper entrypoints rather than one canonical public Stage
  5 `prove` entrypoint:
  - `derive_stage5_regular_batch_prefix`;
  - `evaluate_stage5_output_openings`;
  - `prove_stage5_transparent_sumchecks`;
  - `append_stage5_opening_claims`.
- `request.rs` already follows the desired direction by using `jolt-claims`
  formula helpers and `jolt_opening_oracle_ref` for output-opening evaluation
  requests.
- `prove_stage5_transparent_sumchecks` is compiled only under
  `not(feature = "field-inline")`.
- Field-inline claim fields exist in some structs, but
  `output.rs` currently fills `field_rd_inc` and `field_rd_wa` with `F::zero()`
  placeholders. This is a correctness blocker for field-inline acceptance.
- There is no ZK committed Stage 5 proof assembly path.
- There is no canonical `Stage5ProverInput` bundle that consumes Stage 2/4
  outputs, witness provider, protocol config, and mode.
- There is no canonical `Stage5ProverOutput` carrying verifier-owned proof
  fields, clear/ZK output, prover-local opening state, and committed
  BlindFold rows.
- `output.rs` uses `HashMap` for slot collection. Replace it with `BTreeMap`
  so diagnostics are deterministic.

Current request slots are:

- lookup-table flag openings start at `0`;
- instruction RA openings start at `1024`;
- instruction RAF flag opening is `2048`;
- RAM RA claim-reduction opening is `2050`;
- register value-evaluation openings start at `2064`.

The slot ranges should remain stable until the stage is promoted, then move
behind a typed backend request if the backend API grows a more structured
slot-keying scheme.

### jolt-backends

The current Stage 5 prover helpers use the generic `SumcheckBackend`
materialization/evaluation bridge. That is useful for fixture replay, but it is
not sufficient replacement evidence for the optimized CPU path.

Required CPU work:

- preserve instruction read-RAF prefix/suffix decomposition and cache behavior;
- preserve register value-evaluation compact relation behavior;
- preserve RA delayed materialization and one-hot evaluation behavior;
- avoid dense fallback materialization where core uses a structured relation;
- produce focused timing and analytical memory evidence before accepting the
  `jolt-prover` Stage 5 frontier.

### jolt-prover-harness

The manifest currently contains:

- `stage5_regular_batch_inputs`;
- `stage5_output_openings`;
- `stage5_regular_batch_sumcheck`.

The corresponding tests live in
`crates/jolt-prover-harness/tests/frontier_stage5_batch.rs`.

Current manifest optimization IDs are generic sumcheck/opening IDs:

- `OPT-SC-007`;
- `OPT-EQ-004`;
- `OPT-OPEN-008` for output-opening checks.

Before production replacement, Stage 5 evidence should also account for the
relation-specific inventory rows it actually touches:

- `OPT-REL-001`, `OPT-REL-002`, `OPT-REL-003` for instruction read-RAF
  prefix/suffix lookup decomposition and caching;
- `OPT-REL-010` for register value-evaluation compact relation;
- `OPT-RA-*` rows for RA delayed materialization/pushforward used by the
  instruction RA openings;
- relevant `OPT-SC-*`, `OPT-EQ-*`, `OPT-FLD-*`, and `OPT-MEM-*` rows touched
  by the concrete CPU kernel.

If those IDs are covered only through broad ledger rows, the Stage 5 acceptance
PR must explain that coverage and name the evidence files.

## Target Prover Shape

Add a canonical public entrypoint near the top of `prove.rs`:

```rust
pub fn prove_stage5<F, W, B, T, C>(
    input: Stage5ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage5ProverOutput<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>;
```

The exact generic bounds may change, but the API should make these dependencies
explicit:

- `Stage5ProverConfig`;
- `Stage2Output` and `Stage4Output`, mode-matched to the requested proof mode;
- normalized Jolt VM witness provider;
- selected backend;
- transcript at the Stage 5 boundary;
- feature-mode flags determined at compile time, not runtime strings.

`Stage5ProverOutput` should carry:

- verifier-owned `stage5_sumcheck_proof` payload;
- clear `Stage5Claims` or ZK committed output-claim rows;
- verifier-owned `Stage5Output`;
- backend evidence/debug metadata only when enabled for harness builds;
- prover-local opening/blinding data needed by Stage 8 and BlindFold.

### Transparent Flow

Transparent Stage 5 should:

1. Validate Stage 2 and Stage 4 dependencies using verifier-equivalent shape
   checks.
2. Derive `instruction_gamma` and `ram_gamma`.
3. Compute input claims using `jolt-claims` formula helpers and prior verifier
   outputs.
4. Build a backend request for the Stage 5 batched sumcheck in verifier
   statement order.
5. Append input claims and squeeze batching coefficients in verifier order.
6. Run the optimized CPU backend kernel for the batched sumcheck.
7. Derive verifier output points from the reduction point.
8. Evaluate output openings through slot-keyed witness/backend requests.
9. Reconstruct expected output claims with the same formulas as the verifier.
10. Append Stage 5 opening claims in verifier transcript order.
11. Assemble `Stage5Claims`, `Stage5ClearOutput`, and the proof payload.

The prover may keep the existing helper functions during migration, but the
accepted stage should have one canonical orchestration path that the harness and
later stages call.

### ZK Flow

ZK Stage 5 should:

1. Use the same input-claim formulas and statement order as transparent mode.
2. Call committed/BlindFold-aware sumcheck proving instead of clear compressed
   proving.
3. Produce committed output-claim rows in the exact verifier order and count.
4. Store private round coefficients, blindings, and output-row material needed
   by BlindFold.
5. Return `Stage5ZkOutput` public data matching
   `jolt-verifier/src/stages/stage5/outputs.rs`.

Do not special-case ZK by replaying fixture commitments. The stage is accepted
only when the prover constructs committed Stage 5 data from the same backend
requests and witness views as the transparent path.

### Field-Inline Flow

Field-inline Stage 5 must add a fourth batched instance:

1. derive field-register value-evaluation input claim from Stage 4 field-inline
   output;
2. include the field-register statement in batching order after base register
   value evaluation;
3. evaluate `field_rd_inc` and `field_rd_wa` at verifier-derived points;
4. append field-inline opening claims after the base register value-evaluation
   claims;
5. include both field-inline output claims in committed ZK output-claim counts.

The current `F::zero()` placeholder claims must be removed before field-inline
acceptance.

## Backend and Performance Requirements

Stage 5 replacement readiness requires these backend ledger entries, or more
specific successor entries, to be `ParityCertified` with evidence:

- `cpu_stage5_regular_batch_input_claims`;
- `cpu_stage5_regular_batch_sumcheck`;
- `cpu_materialized_opening_evaluations` for the output-opening replay
  subfrontier;
- `cpu_relation_stage_kernels` for relation-specific optimized paths;
- RA, polynomial, EQ, univariate, field, and memory ledger entries used by the
  concrete CPU implementation.

Instruction read-RAF performance parity requires the full core prefix/suffix
lookup decomposition and cache behavior in the backend kernel. Dense
materialization or generic relation scans are acceptable only as correctness
oracles; they cannot certify Stage 5 replacement readiness.

Performance evidence must include:

- focused Stage 5 input-claim and sumcheck microbenchmarks;
- analytical peak-memory accounting for materialized vectors, caches, and
  witness views;
- canonical fixture comparison against `jolt-core`;
- the default 15% timing and peak-memory parity threshold unless the global
  frontier spec is updated.

Correctness-only Stage 5 code is not replacement-ready.

## Required Tests and Gates

Before accepting Stage 5:

- harness static checks pass for manifest, optimization IDs, backend ledger,
  source drift, and workspace boundaries;
- `stage5_regular_batch_inputs`, `stage5_output_openings`, and
  `stage5_regular_batch_sumcheck` replay real fixtures;
- transparent `MuldivSmall` and `AdviceConsumer` verifier replay pass;
- field-inline fixture replay passes after field-inline implementation lands;
- ZK fixture replay passes after committed Stage 5 implementation lands;
- output-opening order and transcript append order are covered by tamper tests;
- backend microbench evidence files are recorded in the ledger;
- `validate_frontier_replacement_ready` passes for Stage 5 with
  `ParityCertified` backend status;
- `validate_global_cpu_backend_inventory_coverage` passes.

Useful local commands:

```bash
cargo nextest run -p jolt-prover-harness frontier_stage5_batch --cargo-quiet
cargo nextest run -p jolt-prover-harness optimization_inventory --cargo-quiet
cargo nextest run -p jolt-prover-harness frontier_gates --cargo-quiet
```

Do not use broad prover E2E as the first debugging loop for Stage 5 backend
algorithm questions. Prove the isolated CPU kernel first, then wire the stage.

## Acceptance Checklist

- [ ] One canonical Stage 5 prover entrypoint exists.
- [ ] Stage 5 imports verifier-owned proof/output/claim structs directly.
- [ ] Stage 5 uses `jolt-claims` formula helpers for relation semantics and
      opening order.
- [ ] Instruction lookup semantics come from `jolt-lookup-tables` or shared
      adapters, not prover-local formulas.
- [ ] Direct `jolt-riscv` use is limited to witness adapter boundaries.
- [ ] Transparent Stage 5 verifier replay passes for base and advice fixtures.
- [ ] Field-inline Stage 5 has real field-register value outputs.
- [ ] ZK Stage 5 produces committed output rows and BlindFold material.
- [ ] Slot collection is deterministic.
- [ ] Backend kernel evidence is `ParityCertified`.
- [ ] Final correctness and performance justification is appended below.

## Final Justification Log

### Slice 1 — Canonical transparent `prove` entrypoint (2026-05)

Status: transparent canonical API landed and verifier-validated. Backend-routing,
field-inline/ZK prover paths, and perf-cert remain. Not yet replacement-ready.

Code touched:

- `crates/jolt-prover/src/stages/stage5/output.rs`: `HashMap`→`BTreeMap`; added
  `Stage5RegularBatchExpectedOutputs` and `Stage5ProverOutput` (verifier-owned
  `stage5_sumcheck_proof` + `Stage5Claims` + assembled `Stage5ClearOutput`).
- `crates/jolt-prover/src/stages/stage5/input.rs`: added the self-contained
  `Stage5ProverInput` (config + `CheckedInputs` + Stage 2/4 clear outputs +
  witness).
- `crates/jolt-prover/src/stages/stage5/prove.rs`: added the canonical public
  `prove` (cfg `not(field-inline)`), ordered first, mirroring
  `jolt-verifier/src/stages/stage5/verify.rs`: derive `instruction_gamma`/
  `ram_gamma`, prove the instruction read-RAF + RAM-RA reduction + register
  value-evaluation batched sumcheck (via the verified host helper), and assemble
  `Stage5ClearOutput` — including the `VerifiedInstructionReadRafSumcheck` with its
  `r_address`/`r_cycle`/full/flag/per-RA opening points. Surfaced per-instance
  expected outputs.
- `crates/jolt-prover-harness/src/core_fixture.rs`: routed the Stage 5
  regular-batch verifier replay through the canonical `prove`.

Correctness evidence (all green):

- `cargo nextest run -p jolt-prover-harness stage5_regular_batch_sumcheck_verifier_replay stage5_regular_batch_input_checkpoint stage5_output_opening_checkpoint --features core-fixtures`
  — 3/3 pass. The canonical `prove` output is accepted by the native verifier and
  the assembled `Stage5ClearOutput` matches core.
- Compiles under transparent, `field-inline`, `zk`, `zk,field-inline` (canonical
  path is `cfg(not(field-inline))`, matching the transparent-only Stage 5 prover).

Remaining before replacement-ready: backend-route the full batched sumcheck
through the `jolt-backends` regular-batch kernel + perf-cert
`cpu_stage5_regular_batch_sumcheck`; field-inline Stage 5 prover path
(field-registers value-evaluation); ZK committed-boundary.

Replacement-ready gate status: the `stage5_output_openings` and
`stage5_regular_batch_inputs` frontiers are replacement-ready.

- `stage5_output_openings`: `validate_frontier_replacement_ready` passes against
  the `ParityCertified` `cpu_materialized_opening_evaluations` kernel (test
  `stage5_output_opening_frontier_is_replacement_ready_with_certified_kernel_evidence`).
- `stage5_regular_batch_inputs`: now `ParityCertified`. The input-claim derivation
  (`derive_stage5_regular_batch_prefix`: `instruction_gamma`/`ram_gamma` draws then
  the instruction read-RAF, RAM-RA-reduction, and register value-evaluation input
  claims from Stage 2/4 clear outputs) was benchmarked against the verifier-mirroring
  reference prefix via `Stage5RegularBatchInputKernelBenchmarkFixture`. The loader
  asserts `run_reference_prefix() == run_modular_prefix() == expected` and that the
  reference input claims/gammas equal the values the native verifier computes
  (`stage5_regular_batch_input_claims`, `Stage5PublicOutput.{instruction_gamma,ram_gamma}`),
  so the timing comparison is over verifier-validated equal outputs. Evidence
  `cpu_stage5_regular_batch_input_claims/frontier_perf_stage5_regular_batch_inputs.json`
  reports latest post-clean status `Warn`, time ratio ≈1.1441, and memory ratio
  1.0 — within the 15% fail threshold. The overhead is the modular path's
  `validate_stage5_dependencies`
  guard, which the bare reference skips, on a sub-microsecond derivation (absolute
  delta is nanoseconds). Gate test
  `stage5_regular_batch_input_frontier_is_replacement_ready_with_certified_kernel_evidence`
  passes.

This earlier input-frontier note is resolved by the backend instruction
prefix/suffix slice below, which certifies the full Stage 5 regular-batch
sumcheck frontier.

## Implementation Update: Backend RAM/Register Stage 5 Slice

This slice added the canonical Stage 5 full-sumcheck benchmark writer for
`cpu_stage5_regular_batch_sumcheck` and moved the RAM RA claim-reduction plus
register value-evaluation instances onto Stage 5-specific `jolt-backends` CPU
state APIs. The modular prover now avoids the previous dense RAM/register
materialization path and obtains RAM/register output openings from backend state.

Correctness evidence:

- `cargo check -p jolt-backends --features cpu -q`
- `cargo check -p jolt-prover --features frontier-harness -q`
- `cargo check -p jolt-prover-harness --features core-fixtures --benches --tests -q`
- The Stage 5 evidence writer compares the modular proof, challenges, batching
  coefficients, and final output claim against the core Stage 5 regular-batch
  fixture before running the parity gate.

Perf evidence:

- Baseline dense route failed at time ratio `34.85045297529117`, peak RSS ratio
  `74.98159417052878`.
- After backend RAM/register state routing, the same canonical writer failed at
  time ratio `29.123126295536572`, peak RSS ratio `0.2204191848043276`.
- After cached instruction address-prefix weights, it failed at time ratio
  `25.16031057832418`, peak RSS ratio `0.2160395903763744`.
- After reusing the instruction hybrid-address buffer, it failed at time ratio
  `23.747016856261652`, peak RSS ratio `0.22467044322126284`.

Replacement status: still **not replacement-ready**. Memory parity is now strong,
but timing parity still requires porting the core instruction read-RAF
prefix/suffix CPU kernel into `jolt-backends` and routing the Stage 5 instruction
instance through it. Do not mark `cpu_stage5_regular_batch_sumcheck`
`ParityCertified` until the canonical evidence writer passes the 15% timing and
memory gate.

## Implementation Update: Backend Instruction Prefix/Suffix Stage 5 Slice

This slice moved instruction read-RAF heavy compute behind
`Stage5ValueEvaluationSumcheckBackend` alongside the RAM/register Stage 5
states. `jolt-witness` now exposes deterministic per-cycle instruction read-RAF
rows, and `jolt-backends::cpu` owns the full prefix/suffix algorithm:

- lookup-table suffix aggregation by table and phase;
- lookup prefix-polynomial checkpointing through `jolt-lookup-tables`;
- RAF operand/identity prefix/suffix state;
- expanding-table virtual instruction-RA materialization;
- final cycle-round product sumcheck and instruction output claims.

`jolt-prover` Stage 5 now remains linear: derive verifier prefix challenges,
construct backend requests for instruction/RAM/register instances, run the
batched rounds in verifier order, ask backend state for output claims, and
assemble verifier-owned Stage 5 objects.

Correctness evidence:

- `cargo fmt -q`
- `cargo check -p jolt-backends --features cpu -q`
- `cargo check -p jolt-witness -q`
- `cargo check -p jolt-prover --features frontier-harness -q`
- `cargo check -p jolt-prover-harness --features core-fixtures --benches --tests -q`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'test(stage5_regular_batch_input_checkpoint_matches_core_fixtures) | test(stage5_regular_batch_sumcheck_verifier_replay_verifies_against_core_fixtures) | test(stage5_output_opening_checkpoint_matches_core_fixtures)'`
- The canonical Stage 5 evidence writer asserts modular proof, challenges,
  batching coefficients, and output claim match the core fixture before writing
  evidence.

Perf evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage5_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  wrote
  `target/frontier-metrics/kernel-evidence/cpu_stage5_regular_batch_sumcheck/frontier_perf_stage5_regular_batch_sumcheck.json`
  with latest post-clean status `Warn`.
- Reported ratios: time `1.0588723660722457`, peak RSS `0.9137198976872593`.
- `stage5_regular_batch_sumcheck_frontier_is_replacement_ready_with_certified_kernel_evidence`
  passes against this evidence file.
- The optimization ledger and frontier manifest now account for
  `OPT-SC-007`, `OPT-EQ-004`, `OPT-REL-001`, `OPT-REL-002`, `OPT-REL-003`,
  and `OPT-REL-010` on the Stage 5 regular-batch sumcheck frontier.
- `backend_kernel_ledger_accounts_for_registered_frontier_cpu_optimizations`,
  `backend_kernel_ledger_covers_every_cpu_backend_inventory_id`,
  `prover_ready_frontiers_require_ported_or_certified_cpu_kernels`, and
  `registered_parity_certified_kernel_evidence_files_are_valid` pass.

Replacement status: transparent Stage 5 regular-batch CPU parity is now
performance-certified for the canonical `sha2-chain-2^16` fixture. Remaining
Stage 5 promotion work is feature-surface completion: field-inline
field-register value evaluation and ZK committed-boundary assembly/replay.

## Implementation Update: Field-inline field-register value-evaluation slice

This slice wires the field-inline Stage 5 clear prover frontier through a
backend-owned field-register value-evaluation state, so the field-register
component follows the same verifier-mirroring orchestration as the regular
register value-evaluation instance.

Code touched:

- `crates/jolt-backends/src/sumcheck/request.rs`,
  `crates/jolt-backends/src/sumcheck/result.rs`, and
  `crates/jolt-backends/src/traits.rs`: added the field-register
  value-evaluation request, output, and `Stage5ValueEvaluationSumcheckBackend`
  trait surface.
- `crates/jolt-backends/src/cpu/read_write_matrix/field_registers.rs`: added
  `FieldRegistersValEvaluationState`, computing the Stage 5
  `lt_cycle * field_rd_inc * field_rd_wa` relation from sparse traced
  field-register rows and the fixed Stage 4 field-register address/cycle point.
- `crates/jolt-prover/src/stages/stage5/input.rs` and
  `crates/jolt-prover/src/stages/stage5/prove.rs`: enabled the field-inline
  clear `prove` entrypoint, batching instruction read-RAF, RAM RA reduction,
  register value-evaluation, and field-register value-evaluation in verifier
  order.
- `crates/jolt-prover-harness/tests/field_inline_sdk_guest.rs`: extended the
  real `field-inline-eq-poly-guest` replay through Stage 5 and required native
  `jolt-verifier` acceptance.

Correctness evidence:

- `cargo check -p jolt-backends --features field-inline -q`
- `cargo check -p jolt-prover --features field-inline -q`
- `cargo check -p jolt-prover --features zk,field-inline -q`
- `cargo check -p jolt-prover-harness --features core-fixtures,zk,field-inline -q`
- `cargo clippy -p jolt-prover --features field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover-harness --features core-fixtures,field-inline -q --all-targets -- -D warnings`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline field_inline_eq_poly_guest_replays_modular_frontier_with_jolt_verifier --cargo-quiet`
  passes, proving Stage 0 through Stage 5 on real field-inline SDK trace data
  and replaying the proof components through native `jolt-verifier`.

Performance evidence:

- `cpu_field_inline_stage5_registers_val_evaluation` is registered as `Ported`
  with `OPT-FLD-003` and `OPT-REL-010` coverage.
- No canonical field-inline Stage 5 benchmark evidence has been written yet, so
  the field-inline Stage 5 frontier must not be claimed `ParityCertified` until
  `frontier_perf/stage5_field_inline_registers_val_evaluation` or an equivalent
  combined field-inline frontier benchmark passes the 15% timing and memory
  gate.

Remaining before full Stage 5 acceptance:

- Add canonical field-inline Stage 5 benchmark evidence and promote the
  field-register value-evaluation row only after it passes.
- Implement the Stage 5 committed/ZK prover assembly and native verifier
  boundary replay for `zk` and `zk + field-inline`.

## Implementation Update: Field-inline value-evaluation prefix/suffix evidence (2026-05-30)

This slice replaces the dense field-inline Stage 5 `LT` table with
`jolt_poly::LtPolynomial` split binding and adds canonical real-guest evidence
for the field-register value-evaluation kernel. The dense `wa` factor is kept
because the attempted RA-style `wa` path was slower on the real frontier trace
(`1.1799732822637807` timing ratio) despite good memory; the accepted kernel
uses split/prefix-suffix `LT` where it materially reduces memory without adding
that overhead.

Code touched:

- `crates/jolt-poly/src/lt.rs`: ported LowToHigh split-`LT` binding and
  sumcheck pair access, with tests against dense low-to-high binding.
- `crates/jolt-backends/src/cpu/read_write_matrix/field_registers.rs`: changed
  `FieldRegistersValEvaluationState` to use split `LtPolynomial` while keeping
  the degree-3 `inc * wa * lt` round polynomial identical to the dense
  reference.
- `crates/jolt-prover-harness/benches/frontier_perf.rs`: added the canonical
  `cpu_field_inline_stage5_registers_val_evaluation` evidence writer over real
  `field-inline-eq-poly-guest` SDK trace data.
- `crates/jolt-prover-harness/src/optimization.rs`: promoted
  `cpu_field_inline_stage5_registers_val_evaluation` to `ParityCertified` with
  canonical evidence.

Correctness evidence:

- `cargo nextest run -p jolt-poly lt --cargo-quiet`
- `cargo nextest run -p jolt-backends --features field-inline field_registers_val_evaluation_matches_dense_reference --cargo-quiet`
- The evidence writer asserts equality between the dense reference run and the
  modular split-`LT` run before measuring performance.

Performance evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_field_inline_stage5_registers_val_evaluation cargo bench -p jolt-prover-harness --features core-fixtures,field-inline --bench frontier_perf --quiet`
  wrote
  `target/frontier-metrics/kernel-evidence/cpu_field_inline_stage5_registers_val_evaluation/frontier_perf_stage5_field_inline_registers_val_evaluation.json`
  with status `Pass`.
- Reported ratios: time `0.8805542283803156`, peak RSS
  `0.7315756035578145`.

Replacement status: field-inline Stage 5 value-evaluation CPU parity is now
performance-certified for the real field-inline SDK guest frontier. Remaining
Stage 5 work after Slice 3 below is full BlindFold assembly at Stage 8; the
native committed boundary is now covered for the real `zk + field-inline`
frontier.

## Implementation Update: Real committed-boundary proof production (2026-05-30)

This slice adds the Stage 5 modular committed-boundary prover path. It uses the
same certified instruction read-RAF, RAM RA reduction, register value-evaluation,
and field-register value-evaluation backend states as the clear prover, but
follows the verifier ZK transcript schedule: no clear input-claim absorption,
committed round polynomials, and committed output-claim rows in the Stage 5
verifier/spec order.

Code touched:

- `crates/jolt-prover/src/stages/stage5/prove.rs`: added
  `prove_committed_boundary` for standard and `field-inline` feature shapes.
  Each path samples instruction/RAM gammas, batches the Stage 5 statements in
  verifier order, commits the actual batched round polynomial each round,
  recomputes the expected final claim from real backend output openings, and
  retains the committed witness data for later BlindFold assembly.
- `crates/jolt-prover/src/stages/stage5/output.rs`: added
  `Stage5CommittedBoundaryOutput`, carrying the verifier-owned
  `stage5_sumcheck_proof`, `Stage5PublicOutput`, hidden output-claim values, and
  retained committed witness state.
- `crates/jolt-prover-harness/tests/field_inline_sdk_guest.rs`: extended the
  real SDK ZK replay to Stage 5 with modular committed Stage 3, Stage 4, and
  Stage 5 proof components.

Correctness evidence:

- `cargo check -p jolt-prover --features zk,field-inline -q`
- `cargo check -p jolt-prover-harness --features core-fixtures,field-inline,zk -q`
- `cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover-harness --features core-fixtures,field-inline,zk -q --all-targets -- -D warnings`
- `cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest_accepts_zk_committed_stage3_to_stage5_boundaries --features core-fixtures,field-inline,zk --cargo-quiet`
  passes. The test runs the real `field-inline-eq-poly-guest` trace, produces
  real modular committed Stage 3/4/5 proofs, and requires native
  `jolt-verifier` to return `Stage5Output::Zk`. It checks that Stage 5 public
  verifier output matches the modular prover output, the verifier output-claim
  count matches the prover's hidden output-claim vector, one output-claim row is
  used at fixture VC capacity, and prover/verifier transcript states match after
  Stage 5.

Performance evidence:

- No new arithmetic kernel was introduced in this slice. The Stage 5 committed
  path reuses the already-certified full prefix/suffix instruction read-RAF
  kernel, RAM/register value kernels, and field-inline split-`LT`
  value-evaluation kernel. The added work is vector-commitment construction and
  witness retention for ZK committed boundary/BlindFold.

Scope note:

This proves native verifier acceptance at the Stage 5 committed boundary with
real modular Stage 5 data on the real SDK `zk + field-inline` path. It does not
yet prove full BlindFold correctness: Stages 6-8 still need committed-boundary
or final-opening/BlindFold assembly, and Stage 8 remains responsible for full
ZK `JoltProof` verification.
