# Stage 7 Hamming-Weight and Advice Address-Phase Frontier Spec

## Scope

Stage 7 proves the regular batched sumcheck over hamming-weight claim reduction
and optional advice address-phase reductions. It starts after Stage 6 has
produced typed verifier outputs and the transcript is positioned at the Stage 7
boundary. In clear mode it also consumes Stage 4 RAM value-check initial
evaluation state for advice final-scaling data. It ends after the verifier-owned
Stage 7 proof field, clear claims or committed ZK shape, and downstream Stage 7
output are assembled.

The stage boundary is:

1. Consume Stage 6 output, plus Stage 4 output in clear mode when advice
   address phase is present.
2. Derive Stage 7 Fiat-Shamir challenges in the exact verifier order.
3. Prove the Stage 7 batched sumcheck over:
   - hamming-weight claim reduction;
   - trusted advice address phase, if present and required by the advice
     layout;
   - untrusted advice address phase, if present and required by the advice
     layout.
4. Evaluate every Stage 7 output opening at verifier-derived points.
5. Assemble the verifier-owned Stage 7 proof fields and claims.
6. Return typed Stage 7 public output for Stage 8 without recomputing verifier
   reductions.

Stage 7 produces the verifier component for:

- `JoltStageProofs::stage7_sumcheck_proof`;
- clear-mode `ClearProofClaims::stage7`;
- ZK-mode committed Stage 7 sumcheck proof and output-claim commitments;
- clear-mode `Stage7ClearOutput` data needed by Stage 8;
- ZK-mode `Stage7ZkOutput` data needed by Stage 8 and BlindFold.

Stage 7 does not prove bytecode/instruction read-RAF, booleanity, increment
reductions, advice cycle phase, or final PCS openings. It does bridge the RA
families from Stage 6 into the final Stage 8 opening point, so its output order
and opening-point derivation are Stage 8-critical.

Every Stage 7 implementation path must explicitly support:

- advice address phase for trusted and untrusted advice when the layout has an
  address phase;
- BlindFold/ZK mode, including committed regular-batch proof data and committed
  output-claim rows;
- field-inline builds, even though Stage 7 has no field-inline-specific
  relation today.

## Monitored Workflow

Stage work proceeds in one reviewable Stage 7 slice:

1. Add `jolt-prover/src/stages/stage7/` production modules.
2. Add Stage 7 harness manifest rows and focused tests.
3. Tighten backend evidence for hamming-weight reduction and advice
   address-phase reductions before accepting prover orchestration.
4. Define the canonical Stage 7 input/output/prover-state API.
5. Implement transparent and advice paths through native Stage 7 verifier
   replay.
6. Confirm field-inline builds use the same canonical path.
7. Add ZK committed-boundary proof assembly and verifier replay.
8. Run focused tests, real fixture replay, and kernel evidence gates.
9. Append final correctness and performance parity justification to this spec.
10. Stop for review before moving to Stage 8.

Every fact in the implementation should have a clear owner:

- `jolt-verifier` owns proof fields, verifier outputs, transcript checks,
  clear claims, committed proof shapes, output-claim counts, and output order;
- `jolt-claims` owns relation IDs, dimensions, formula semantics, opening IDs,
  challenge IDs, public IDs, hamming dimensions, and advice layouts;
- `jolt-witness` owns trace-backed RA views and advice views;
- `jolt-program` owns public I/O/advice memory layout values that feed advice
  layout construction;
- `jolt-backends` owns heavy compute and slot-keyed kernel results;
- `jolt-riscv` should not be needed directly in Stage 7 production
  orchestration;
- `jolt-lookup-tables` should not be needed directly except for shared
  dimension constants already exposed through formula dimensions;
- `jolt-prover-harness` owns migration-only fixtures, verifier replay, and
  performance evidence.

The public prover path should remain linear enough to audit against
`jolt-verifier/src/stages/stage7/verify.rs`.

### Modular Crate Usage

Stage 7 should follow the verifier ownership split.

- Use `jolt-claims::protocols::jolt::formulas::claim_reductions::hamming_weight`
  for hamming-weight metadata, input/output openings, dimensions, challenges,
  and public IDs.
- Use `jolt-claims::protocols::jolt::formulas::claim_reductions::advice` plus
  `AdviceClaimReductionLayout` for advice address-phase metadata.
- Use `JoltFormulaDimensions` and `JoltRaPolynomialLayout` to derive RA family
  counts instead of manually counting instruction/bytecode/RAM RA polynomials.
- Use `JoltOpeningId` as semantic request keys, then convert to witness oracles
  with namespace-owned helpers.
- Use Stage 6 verifier-owned output points for booleanity and virtualization
  references. Do not recompute Stage 6 relation points from witness data.
- Use Stage 4 verifier-owned RAM value-check advice contribution state in clear
  mode for advice final scaling.

This keeps Stage 7 independent of instruction encodings and lookup-table
semantics while preserving the verifier's hamming/advice formulas.

## Current Inventory

### jolt-verifier

Relevant verifier code lives in `crates/jolt-verifier/src/stages/stage7/`.

- `verify.rs` constructs `JoltFormulaDimensions` from trace length,
  one-hot config, bytecode size, and RAM K, then builds
  `HammingWeightClaimReductionDimensions`.
- Trusted and untrusted advice layouts are constructed with
  `AdviceClaimReductionLayout::balanced`; an advice address-phase statement is
  included only when the corresponding layout has an address phase.
- `verify.rs` derives `hamming_gamma = transcript.challenge_scalar()`.
- Clear mode verifies a compressed Boolean batched sumcheck in this statement
  order:
  1. hamming-weight claim reduction;
  2. trusted advice address phase, if present;
  3. untrusted advice address phase, if present.
- Clear hamming input claims are computed from Stage 6:
  - RAM hamming booleanity output;
  - Stage 6 booleanity RA output claims;
  - Stage 6 RA virtualization output claims.
- Clear hamming output reconstruction uses:
  - Stage 6 booleanity address point;
  - Stage 6 RA virtualization address chunks;
  - `hamming_gamma`;
  - Stage 7 output opening claims for instruction, bytecode, and RAM RA
    families.
- Clear advice address-phase input claims consume Stage 6 advice cycle-phase
  output claims.
- Clear advice address-phase output reconstruction uses Stage 4 RAM
  value-check advice contributions and the layout's final-output scale.
- The verifier appends Stage 7 opening claims in this transcript order:
  1. hamming-weight instruction RA outputs;
  2. hamming-weight bytecode RA outputs;
  3. hamming-weight RAM RA outputs;
  4. trusted advice address-phase final opening, if present;
  5. untrusted advice address-phase final opening, if present.
- ZK mode verifies committed consistency and requires output-claim count:
  - hamming instruction RA count;
  - hamming bytecode RA count;
  - hamming RAM RA count;
  - one row for trusted advice address phase if present;
  - one row for untrusted advice address phase if present.
- `outputs.rs` owns `Stage7PublicOutput`, `Stage7ClearOutput`,
  `Stage7ZkOutput`, `Stage7Output`, `VerifiedStage7Batch`,
  `VerifiedHammingWeightClaimReductionSumcheck`, and typed advice
  address-phase outputs.
- `stages/zk/blindfold/stage7.rs` lowers Stage 7 into BlindFold using the same
  statement order, optional advice rows, and output-claim order.

Stage 7 prover code must import verifier-owned proof, claim, and output structs
directly rather than duplicating those shapes locally.

### jolt-prover

There is currently no `crates/jolt-prover/src/stages/stage7/` module.

The Stage 7 implementation must add:

- `mod.rs`, `input.rs`, `output.rs`, `request.rs`, and `prove.rs`;
- a canonical public Stage 7 `prove` entrypoint;
- clear and ZK output assembly into verifier-owned types;
- request builders for hamming output openings and advice address-phase
  openings;
- transcript append helpers that match verifier order;
- tests or harness adapters that replay Stage 7 through `jolt-verifier`.

Do not copy large blocks of verifier logic into prover-local formula code.
Where output formulas are needed by both sides, move helpers to `jolt-claims`
or use existing formula expressions.

### jolt-backends

There are no Stage 7-specific frontend kernel ledger entries in the narrow
manifest today. Broad relation-stage ledger rows cover some required work, but
Stage 7 acceptance should add or identify focused backend entries for:

- Stage 7 hamming-weight input claims;
- Stage 7 hamming-weight sumcheck rounds;
- advice address-phase input claims and sumcheck rounds;
- Stage 7 output-opening evaluations.

Required CPU work:

- preserve hamming-weight reduction over RA families;
- preserve verifier-evaluable claim caching where repeated EQ evaluations are
  used;
- preserve advice two-phase state reuse from Stage 6 to Stage 7;
- preserve booleanity phase split state from Stage 6;
- avoid dense RA-family materialization where core uses structured RA views.

### jolt-prover-harness

The manifest currently has no Stage 7 frontier rows.

Add at least:

- `stage7_regular_batch_inputs`;
- `stage7_output_openings`;
- `stage7_regular_batch_sumcheck`.

Add focused tests, likely in
`crates/jolt-prover-harness/tests/frontier_stage7_batch.rs`.

Initial fixture coverage should include:

- `MuldivSmall` for base RA-family hamming reduction;
- `AdviceConsumer` for trusted/untrusted advice address-phase coverage;
- `ZkMuldivSmall` and `ZkAdviceConsumer` once committed Stage 7 lands;
- field-inline fixture replay for compile/output-shape coverage.

Required optimization IDs should include:

- `OPT-REL-012` for hamming-weight reduction over RA families;
- `OPT-REL-013` for advice claim-reduction two-phase state;
- `OPT-REL-014` for booleanity phase splitting;
- `OPT-REL-015` for verifier-evaluable claim caching;
- `OPT-OPEN-008` for output-opening replay until streaming Stage 8 takes over;
- relevant `OPT-RA-*`, `OPT-SC-*`, `OPT-EQ-*`, `OPT-FLD-*`, and
  `OPT-MEM-*` rows touched by the concrete CPU kernel.

## Target Prover Shape

Add a canonical public entrypoint in `prove.rs`:

```rust
pub fn prove_stage7<F, W, B, T, C>(
    input: Stage7ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage7ProverOutput<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>;
```

`Stage7ProverInput` should make these dependencies explicit:

- checked trace shape and public I/O/advice layout;
- preprocessing bytecode size or normalized dimensions object;
- one-hot config and trace polynomial order;
- mode-matched Stage 4 output when clear advice address phase is needed;
- mode-matched Stage 6 output;
- normalized witness provider;
- selected proof mode.

`Stage7ProverOutput` should carry:

- verifier-owned `stage7_sumcheck_proof` payload;
- clear `Stage7Claims` or ZK committed output-claim rows;
- verifier-owned `Stage7Output`;
- final RA opening points and advice final opening points needed by Stage 8;
- prover-local BlindFold private material in ZK mode.

### Transparent Flow

Transparent Stage 7 should:

1. Build formula dimensions and advice layouts exactly as the verifier does.
2. Decide active statements in verifier order.
3. Derive `hamming_gamma`.
4. Compute input claims from Stage 6 and optional advice state.
5. Append input claims and squeeze batching coefficients in active statement
   order.
6. Run optimized CPU kernels for the batched sumcheck.
7. Derive the hamming opening point from the hamming sumcheck point and Stage 6
   booleanity cycle point.
8. Evaluate hamming RA-family output openings at that point.
9. Derive and evaluate advice address-phase output openings when present.
10. Reconstruct expected output claims with verifier-equivalent formulas.
11. Append Stage 7 opening claims in verifier transcript order.
12. Assemble `Stage7Claims`, `Stage7ClearOutput`, and the proof payload.

### ZK Flow

ZK Stage 7 should:

1. Use the same active statement order and public challenge derivation as clear
   mode.
2. Prove committed consistency for every active Stage 7 sumcheck instance.
3. Commit output-claim rows in the exact verifier order.
4. Store private coefficients and blindings needed by BlindFold.
5. Return `Stage7ZkOutput` public data matching verifier-owned output types.

ZK Stage 7 does not consume Stage 4 verifier output directly. It should use the
same public data that `jolt-verifier` uses in ZK mode and leave hidden claim
binding to BlindFold.

### Advice Flow

Advice address phase is active per advice kind only when:

- the corresponding advice commitment exists; and
- `AdviceClaimReductionLayout::dimensions().has_address_phase()` is true.

If address phase is inactive, Stage 7 must not append a final advice opening
for that kind. Stage 8 then takes the final advice opening point from Stage 6
cycle phase where applicable.

Trusted advice must precede untrusted advice in statement order, output-claim
order, and transcript append order.

## Backend and Performance Requirements

Stage 7 replacement readiness requires new or identified backend ledger entries
to be `ParityCertified` with evidence:

- Stage 7 hamming-weight input-claim kernel;
- Stage 7 hamming-weight sumcheck kernel;
- Stage 7 advice address-phase kernel;
- Stage 7 output-opening evaluation kernel;
- broad relation, RA, polynomial, EQ, univariate, field, and memory kernels
  touched by the concrete implementation.

Hamming-weight and advice address-phase paths must preserve any core
prefix/suffix or structured RA decomposition used by the optimized prover. Dense
RA-family materialization is only a reference oracle; it is not acceptable
production code or replacement parity evidence.

Performance evidence must include:

- focused Stage 7 input-claim and sumcheck microbenchmarks;
- hamming-weight RA-family benchmarks;
- advice address-phase benchmarks;
- analytical peak-memory accounting for RA family views, cached EQ values, and
  advice state;
- canonical fixture comparison against `jolt-core`;
- the default 15% timing and peak-memory parity threshold.

Correctness-only Stage 7 code is not replacement-ready.

## Required Tests and Gates

Before accepting Stage 7:

- harness manifest rows exist for Stage 7 frontiers;
- harness static checks pass for manifest, optimization IDs, backend ledger,
  source drift, and workspace boundaries;
- `stage7_regular_batch_inputs`, `stage7_output_openings`, and
  `stage7_regular_batch_sumcheck` replay real fixtures;
- transparent `MuldivSmall` and `AdviceConsumer` verifier replay pass;
- advice fixtures cover active and inactive address-phase layouts when
  practical;
- field-inline fixture replay passes;
- ZK fixture replay passes after committed Stage 7 implementation lands;
- output-opening order and transcript append order are covered by tamper tests;
- backend microbench evidence files are recorded in the ledger;
- `validate_frontier_replacement_ready` passes for Stage 7 with
  `ParityCertified` backend status;
- `validate_global_cpu_backend_inventory_coverage` passes.

Expected local commands after Stage 7 tests exist:

```bash
cargo nextest run -p jolt-prover-harness frontier_stage7_batch --cargo-quiet
cargo nextest run -p jolt-prover-harness optimization_inventory --cargo-quiet
cargo nextest run -p jolt-prover-harness frontier_gates --cargo-quiet
```

## Acceptance Checklist

- [x] `crates/jolt-prover/src/stages/stage7/` exists.
- [x] One canonical Stage 7 prover entrypoint exists.
- [x] Stage 7 imports verifier-owned proof/output/claim structs directly.
- [x] Stage 7 uses `jolt-claims` hamming/advice helpers for relation semantics
      and opening order.
- [x] Stage 7 has no direct `jolt-riscv` dependency.
- [x] Transparent Stage 7 verifier replay passes for base and advice fixtures.
- [x] Field-inline Stage 7 compiles and replays.
- [ ] ZK Stage 7 produces committed output rows and BlindFold material.
- [x] Advice address-phase active/inactive cases are covered.
- [x] Backend kernel evidence is `ParityCertified` for transparent/advice
      Stage 7 regular-batch inputs and sumcheck.
- [ ] Final correctness and performance justification is appended below.

## Final Justification Log

### In progress — input-claim derivation landed and verifier-validated

Stage 7 is being built backend-first and slice-by-slice. The first landed slice is
the canonical Fiat-Shamir prefix / input-claim derivation.

Implemented:

- `crates/jolt-prover/src/stages/stage7/{mod,input,output,prove}.rs`.
  - `Stage7ProverConfig` (log_t, `HammingWeightClaimReductionDimensions`, trusted/
    untrusted `AdviceClaimReductionLayout`), `Stage7ProverInput`, and the prefix
    output types `Stage7RegularBatchInputClaims` / `Stage7RegularBatchPrefixOutput`,
    all using the verifier-owned dimension/layout/claim semantics from
    `jolt-claims` (no prover-local formula duplication).
  - `derive_stage7_regular_batch_prefix(config, stage6, transcript)` mirrors
    `jolt-verifier/src/stages/stage7/verify.rs` in prover order: draw
    `hamming_gamma = transcript.challenge_scalar()`, then evaluate the
    hamming-weight claim-reduction input claim from Stage 6 RA-family openings
    (`ram_hamming_booleanity.ram_hamming_weight`, `booleanity.{instruction,bytecode,ram}_ra`,
    and the RA-virtualization claims) via the shared `hamming_weight::claim_reduction`
    expression, plus the optional trusted/untrusted advice address-phase input
    claims from the Stage 6 advice cycle-phase output claims (only when the advice
    layout has an address phase). Compiles under transparent and `field-inline`.

Correctness evidence (green):

- `cargo nextest run -p jolt-prover-harness stage7_regular_batch_input_checkpoint_matches_core_fixtures --features core-fixtures`
  — passes for `MuldivSmall` and `AdviceConsumer`. The harness builds a real core
  proof, runs the native verifier through Stage 6, derives the Stage 7 prefix on a
  transcript clone, then runs the native Stage 7 verifier and asserts
  `modular.input_claims == {hamming_weight_claim_reduction, trusted/untrusted
  advice address-phase}` reconstructed from `Stage7ClearOutput.batch` and
  `modular.hamming_gamma == Stage7ClearOutput.public.hamming_gamma`. So the derived
  input claims and the Stage 7 Fiat-Shamir challenge match the verifier exactly.

### Transparent hamming-weight sumcheck landed and verifier-validated (MuldivSmall)

The canonical `prove` entrypoint now proves the full Stage 7 hamming-weight
compressed-boolean batched sumcheck and is accepted end-to-end by the native
verifier for the hamming-weight-only case.

Implemented (`crates/jolt-prover/src/stages/stage7/prove.rs`, `output.rs`):

- Backend kernel `SumcheckBackend::materialize_sumcheck_ra_pushforward`
  (`jolt-backends`) computes the RA-family pushforward
  `G_i(k) = Σ_j eq(r_cycle, j)·ra_i(k, j)` by streaming per-cycle one-hot chunk
  indices through the generic `WitnessProvider::committed_stream` path and
  reducing via `ra::pushforward_indices`. Validated against a dense reference by
  `cpu_ra_pushforward_kernel_matches_dense_reference_via_committed_stream`.
- `prove<F, W, B, T, C>(Stage7ProverInput, backend, transcript)` (ordered first):
  derives `hamming_gamma` + the hamming input claim, builds `G_i` (backend
  pushforward, `r_cycle` = Stage 6 booleanity cycle point), `eq_bool` =
  `EqPolynomial::evals(stage6.batch.booleanity.r_address)`, and per-poly
  `eq_virt_i` from the Stage 6 RA-virtualization opening points truncated to
  `log_k_chunk`. The hamming round message
  `Σ_i G_i·(γ^{3i} + γ^{3i+1}·eq_bool + γ^{3i+2}·eq_virt_i)` is encoded as ONE
  `SumcheckRegularBatchInstance::new_products` (factors `[G_i,
  (γ^{3i} + γ^{3i+1}·eq_bool + γ^{3i+2}·eq_virt_i)]`). Tables are bit-reversed
  over `log_k_chunk` so the kernel's `HighToLow` binding reproduces the core
  `LowToHigh` proof. The reduced RA output openings are evaluated at the
  verifier-derived hamming opening point (`reverse(hamming_point) ++ r_cycle`)
  via `WitnessProvider::try_evaluate_oracle_view`; `Stage7Claims`,
  `Stage7ClearOutput`, and the `stage7_sumcheck_proof` are assembled and the
  opening claims appended in verifier transcript order.

Correctness evidence (green):

- `cargo nextest run -p jolt-prover-harness stage7_regular_batch_verifier_replay_verifies_against_core_fixtures --features core-fixtures`
  — passes for `MuldivSmall`. The harness builds a real core proof, runs the
  native verifier through Stage 6, runs the modular Stage 7 `prove` on a
  `CpuBackend`, splices `stage7_sumcheck_proof` + `Stage7Claims` into the proof
  shell, and the full native verifier (`fixture.verify()`) accepts it. The
  spliced claims also equal the native verifier's `Stage7ClearOutput.output_claims`.
- Compiles under transparent and `field-inline` (`prove` is `cfg(not(field-inline))`,
  matching the transparent-only Stage 7 prover; field-inline compiles out cleanly).

### Input-claims kernel certified (`stage7_regular_batch_inputs`)

The `stage7_regular_batch_inputs` frontier is registered in the harness manifest
and is now `ParityCertified`.

- `cpu_stage7_regular_batch_input_claims` ledger row (`OPT-SC-007`, `OPT-EQ-004`)
  is `ParityCertified` with evidence
  `cpu_stage7_regular_batch_input_claims/frontier_perf_stage7_regular_batch_inputs.json`:
  status `Pass`, time ratio ≈1.008, memory ratio 1.0 over 51 samples. The
  benchmark (`Stage7RegularBatchInputKernelBenchmarkFixture`) times the modular
  `derive_stage7_regular_batch_prefix` against a verifier-mirroring reference; the
  loader asserts `run_reference_prefix() == run_modular_prefix()`, and the
  separate `stage7_regular_batch_input_checkpoint_matches_core_fixtures` test
  anchors the derivation to the native verifier's
  `Stage7ClearOutput.batch.hamming_weight_claim_reduction.input_claim`.
- Perf-regression fix: the first cut computed the hamming input claim via the
  symbolic `jolt-claims` formula expression (`claim_reduction::input.try_evaluate`),
  which the perf gate flagged at a 178× time / 138× memory ratio (the symbolic
  expression is rebuilt and walked for every RA poly). `derive` now computes the
  input claim inline — `Σ_i γ^{3i}·hw_i + γ^{3i+1}·booleanity_i + γ^{3i+2}·virtualization_i`
  — matching the verifier value while restoring parity. The symbolic expression is
  verifier-side ceremony the prover does not need.
- Gate: `stage7_regular_batch_input_frontier_is_replacement_ready_with_certified_kernel_evidence`
  passes; `backend_kernel_ledger_covers_every_cpu_backend_inventory_id` still passes.

Remaining before Stage 7 is replacement-ready (tracked):

- ZK committed-boundary Stage 7.
- Field-inline replay for the Stage 7 frontier.

### Regular-batch hamming/advice sumcheck kernel certified (`stage7_regular_batch_sumcheck`)

The transparent Stage 7 regular-batch sumcheck is now certified with
protocol-shaped CPU kernels for both the hamming-weight reduction and active
advice address-phase reductions, instead of the initial generic regular-batch
encoding.

Implemented:

- Added `SumcheckStage7HammingStateRequest` / `SumcheckStage7HammingState` and
  `crates/jolt-backends/src/cpu/sumcheck/kernels/stage7_hamming.rs`.
- The CPU kernel materializes the RA-family prefix/suffix pushforward, shared
  `eq_bool`, per-RA `eq_virt_i`, and evaluates each round as the fused core
  expression
  `Σ_i G_i·(γ^{3i} + γ^{3i+1}·eq_bool + γ^{3i+2}·eq_virt_i)` with
  `LowToHigh` binding, matching core's `HammingWeightClaimReductionProver`.
- `ra::pushforward_indices` now uses deferred small-scalar accumulation before
  reducing touched slots, preserving core's prefix/suffix accumulation shape.
- The prover no longer evaluates Stage 7 RA output openings by rewalking the
  witness oracle. It reads the final bound `G_i(ρ)` claims directly from the
  hamming backend state, matching core's `cache_openings` behavior and avoiding
  a second RA-family traversal.
- The Stage 7 hamming state uses bounded witness-stream chunks (`1024`) so peak
  memory is not dominated by full-trace per-polynomial one-hot buffers.
- Added `SumcheckStage7AdviceAddressStateRequest` /
  `SumcheckStage7AdviceAddressState` and
  `crates/jolt-backends/src/cpu/sumcheck/kernels/stage7_advice.rs`.
- The advice kernel mirrors core's `AdviceClaimReductionProver` address phase:
  it permutes advice words and the Stage 4 EQ table by `(address, cycle)`,
  replays Stage 6 cycle-phase challenges into the carried state, keeps advice as
  compact `u64` coefficients until the first active address bind, and uses the
  same dummy-round scaling as core.
- The Stage 7 prover now runs one mixed batched sumcheck for hamming plus trusted
  and untrusted advice address phases, including differing round counts and
  verifier-order final opening claims.

Correctness evidence (green):

- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'test(stage7_regular_batch_sumcheck_matches_core_fixtures)'`
  passes. The modular Stage 7 proof, challenges, batching coefficients, and
  final claim match the native verifier/core fixture for both `MuldivSmall` and
  `AdviceConsumer`; the latter exercises active trusted and untrusted advice
  address phases.
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'binary(frontier_stage7_batch)'`
  passes all 5 Stage 7 frontier tests, including verifier replay and the new
  replacement-ready sumcheck gate.

Performance evidence (green under the 15% frontier gate):

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage7_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  wrote
  `target/frontier-metrics/kernel-evidence/cpu_stage7_regular_batch_sumcheck/frontier_perf_stage7_regular_batch_sumcheck.json`.
- Evidence status: `Warn`; time ratio `1.0760070486065136`; peak-memory ratio
  `1.0342628631822848`; samples `3`, measured on the advice-inclusive
  `AdviceConsumer` fixture.
- The failed intermediate evidence is intentionally not accepted: the generic
  product-polynomial path was `3.31x` slower, the fused path plus witness
  re-evaluation was still `3.07x` slower, and the fused/state-derived path with
  bounded chunks was hamming-only. The accepted replacement-ready shape adds the
  core-shaped advice address state, exact verifier replay against `AdviceConsumer`,
  release-only invariant checks matching core, and a lightweight measured trace
  fixture that avoids cloning unused final memory.

Ledger and manifest gates (green):

- `stage7_regular_batch_sumcheck` is registered in the manifest.
- `cpu_stage7_regular_batch_sumcheck` is `ParityCertified` in the backend
  kernel ledger with `OPT-SC-007`, `OPT-EQ-004`, `OPT-RA-003`, `OPT-RA-007`,
  `OPT-RA-008`, and `OPT-REL-013`.
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'binary(frontier_manifest) or binary(optimization_inventory) or binary(frontier_gates)'`
  passes all 26 static ledger/manifest/frontier gate tests.

### Field-inline clear frontier replay through Stage 7

The clear Stage 7 prover path now compiles and runs under `field-inline` instead
of being compiled out. This is one canonical path: field-inline builds use the
same `Stage7ProverInput`, `Stage7ProverOutput`, request shape, hamming backend
state, advice backend state, and verifier-owned `Stage7Claims` /
`Stage7ClearOutput` assembly as ordinary Jolt. There is intentionally no
field-inline-specific Stage 7 relation today; Stage 7 consumes the ordinary RA
families already sanitized and checked by earlier stages, including the
field-inline-adjusted Stage 6 bytecode/read-RAF boundary.

Implemented:

- Removed the `cfg(not(feature = "field-inline"))` gates from
  `crates/jolt-prover/src/stages/stage7/{input,output,request,prove}.rs`.
- Extended the real SDK field-inline guest frontier test to prove Stage 7 after
  Stage 6, splice `stage7_sumcheck_proof` and `ClearProofClaims::stage7` into the
  proof shell, replay `jolt-verifier` through Stage 7, and assert exact
  `Stage7ClearOutput` plus transcript-state equality.

Correctness evidence (green):

- `cargo check -p jolt-prover -q`
- `cargo check -p jolt-prover --features field-inline -q`
- `cargo check -p jolt-prover --features zk,field-inline -q`
- `cargo check -p jolt-prover-harness --features core-fixtures,field-inline --tests -q`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline field_inline_eq_poly_guest_replays_modular_frontier_with_jolt_verifier --cargo-quiet`

The replay uses real traced `field-inline-eq-poly-guest` data, not synthetics.
The verifier accepts the modular proof shell through Stage 7 with native
field-inline bytecode metadata enabled, so the current clear frontier is:

- transparent/base: covered by core fixture replay and native verifier replay;
- advice: covered by `AdviceConsumer` Stage 7 fixture replay with active
  trusted/untrusted advice address phases;
- field-inline clear: covered by real SDK guest replay through native
  `jolt-verifier` Stage 7.

Performance justification:

- No new Stage 7 field-inline kernel is required at this stage because Stage 7
  has no field-inline-specific statement. The field-inline path invokes the same
  Stage 7 hamming/advice backend kernels that are already
  `ParityCertified` for the clear Stage 7 frontier:
  `cpu_stage7_regular_batch_input_claims` and
  `cpu_stage7_regular_batch_sumcheck`.
- Dense/materialized RA-family paths remain reference-only. Replacement
  readiness for Stage 7 continues to depend on the full prefix/suffix,
  delayed-RA hamming/advice kernels, not toy dense or fixture-only replay.

### ZK committed-boundary replay through Stage 7

Stage 7 now has a committed-boundary prover path that shares the transparent
hamming/advice backend loop through the same proof-sink pattern used by Stage 6.
The committed path skips clear input-claim transcript appends, commits each
round polynomial with `CommittedSumcheckBuilder`, commits output claims in the
verifier's final-opening order, and returns hidden `Stage7ClearOutput` prover
state for Stage 8.

Implemented:

- `Stage7CommittedBoundaryOutput` carries `stage7_sumcheck_proof`, public output,
  committed output-claim values, hidden next-stage `Stage7ClearOutput`, and
  committed witness material for later BlindFold assembly.
- `prove_stage7_regular_batch_sumcheck_with_sink` is the one canonical hamming
  plus advice address-phase loop for clear and committed paths.
- The real field-inline SDK frontier test now proves committed Stage 7 after
  committed Stage 6, splices the proof into the ZK shell, runs native
  `jolt-verifier` through `stage7::verify`, and checks `Stage7Output::Zk`.

Correctness evidence:

- `cargo check -p jolt-prover --features zk,field-inline -q`
- `cargo check -p jolt-prover-harness --features core-fixtures,field-inline,zk -q`
- `cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest_accepts_zk_committed_stage3_to_stage8_boundaries --features core-fixtures,field-inline,zk --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'binary(frontier_stage7_batch)'`

Performance evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage7_regular_batch_input_claims cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  wrote status `Pass`, time ratio `0.9999230769230768`, memory ratio `1.0`.
- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage7_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  wrote status `Pass`, time ratio `0.9935130418406102`, memory ratio
  `0.9578253451217621`.

Justification: Stage 7 is now verifier-accepted for transparent/base, advice,
field-inline clear, and ZK + field-inline committed frontiers. The ZK path does
not introduce a second relation implementation; it reuses the certified
prefix/suffix/delayed-RA hamming and advice address kernels and changes only the
proof transcript sink. Full BlindFold proof verification still belongs to the
Stage 8/top-level BlindFold slice, but committed Stage 7 rows and public outputs
are now available for that assembly.
