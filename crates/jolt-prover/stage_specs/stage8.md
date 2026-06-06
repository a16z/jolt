# Stage 8 Final Opening Frontier Spec

## Scope

Stage 8 constructs the final batched PCS opening proof for committed Jolt VM
polynomials. It starts after Stage 6 and Stage 7 have produced mode-matched
typed verifier outputs and the transcript is positioned at the Stage 8
boundary. It ends after `joint_opening_proof` is produced, the final opening
inputs are bound to the transcript, and clear or ZK Stage 8 output data is
available for the verifier and BlindFold.

The stage boundary is:

1. Consume prover commitments, retained PCS opening hints, Stage 6 output,
   Stage 7 output, preprocessing PCS setup, and normalized witness/advice
   views.
2. Derive the common final opening point from Stage 7 hamming-weight output.
3. Derive the PCS-specific opening point using the trace polynomial order.
4. Build the final opening ID/order list exactly as `jolt-verifier` expects.
5. Compute scaling factors for dense increment embeddings and advice
   embeddings.
6. Append clear opening claims and squeeze RLC powers in transparent mode, or
   squeeze the same RLC powers without revealing hidden evaluations in ZK mode.
7. Build the joint commitment and joint claim or hidden evaluation commitment.
8. Prove the final PCS opening using retained hints and optimized CPU/PCS
   kernels.
9. Bind final opening inputs to the transcript.
10. Return typed Stage 8 output for verifier replay and BlindFold.

Stage 8 produces:

- `JoltProof::joint_opening_proof`;
- clear-mode `Stage8ClearOutput`;
- ZK-mode `Stage8ZkOutput`;
- final opening IDs and constraint coefficients used by BlindFold;
- final transcript binding for the proof.

Stage 8 does not prove any relation sumcheck. It is the final opening and RLC
frontier, so performance depends on retained commitment hints, streaming RLC
construction, and avoiding full witness rematerialization.

Every Stage 8 implementation path must explicitly support:

- advice final openings for trusted and untrusted advice commitments;
- BlindFold/ZK mode, including hidden evaluation commitment extraction and
  final opening constraint coefficients;
- field-inline final opening of field-register `rd_inc`;
- streaming CPU opening/RLC kernels, not only materialized fallback.

## Monitored Workflow

Stage work proceeds in one reviewable Stage 8 slice:

1. Add `jolt-prover/src/stages/stage8/` production modules.
2. Add Stage 8 harness manifest rows and focused tests.
3. Port and certify streaming Stage 8 RLC/opening kernels before accepting
   prover orchestration.
4. Define the canonical Stage 8 input/output/prover-state API.
5. Implement transparent final opening proof through native Stage 8 verifier
   replay.
6. Add advice and field-inline final opening support.
7. Add ZK final opening proof assembly and BlindFold handoff data.
8. Run focused tests, real fixture replay, and kernel evidence gates.
9. Append final correctness and performance parity justification to this spec.
10. Stop for review before treating the modular prover as end-to-end complete.

Every fact in the implementation should have a clear owner:

- `jolt-verifier` owns final opening order, opening IDs, constraint
  coefficients, transcript binding, and output structs;
- `jolt-claims` owns committed polynomial IDs, opening IDs, advice final
  opening IDs, advice embedding scale formulas, and field-inline opening IDs;
- `jolt-openings`/PCS crates own PCS proof generation, verification, retained
  hints, and ZK opening APIs;
- `jolt-witness` owns streaming witness/advice views for RLC construction;
- `jolt-backends` owns CPU RLC/opening kernels and memory policy;
- `jolt-riscv` and `jolt-lookup-tables` should not be needed directly in Stage
  8 production orchestration;
- `jolt-prover-harness` owns migration-only fixtures, verifier replay, and
  performance evidence.

The public prover path should remain linear enough to audit against
`jolt-verifier/src/stages/stage8/verify.rs`.

### Modular Crate Usage

Stage 8 should follow the verifier ownership split.

- Use `jolt-claims::protocols::jolt::JoltCommittedPolynomial` and
  `JoltOpeningId` for final opening IDs.
- Use `jolt-claims::protocols::jolt::formulas::committed_openings` for advice
  commitment embedding scale.
- Use `jolt-claims::protocols::jolt::formulas::claim_reductions::advice` for
  final advice opening IDs.
- Under `field-inline`, use field-inline opening helpers for the field
  `rd_inc` final opening.
- Use verifier-owned Stage 6 and Stage 7 outputs to choose opening points and
  claims. Do not rederive relation points from witness data.
- Use retained PCS opening hints from Stage 0 commitments. Recomputing hints
  from dense polynomials is not acceptable for replacement.
- Use `jolt-backends::cpu::openings` and PCS traits for streaming RLC/opening
  computation.

Stage 8 should not need direct instruction, lookup-table, or RISC-V semantics.
If those appear in Stage 8 code, a prior stage or dimension helper is leaking
across the boundary.

## Current Inventory

### jolt-verifier

Relevant verifier code lives in `crates/jolt-verifier/src/stages/stage8/`.

- `verify.rs` reconstructs formula dimensions and RA layout from trace length,
  one-hot config, bytecode size, and RAM K.
- The common final opening point is the Stage 7 hamming-weight opening point.
- The dense increment embedding scale is
  `EqPolynomial::zero_selector(r_address_stage7)`, where
  `r_address_stage7` is the first `committed_chunk_bits` variables of the
  final opening point.
- The PCS opening point is produced by
  `trace_polynomial_order.commitment_opening_point(...)` and then converted to
  `Point::high_to_low`.
- The common opening point is also stored as `Point::high_to_low`.
- The final PCS batch order intentionally differs from proof payload order.
  Verifier final opening order is:
  1. `RamInc` for increment claim reduction, scaled by dense embedding;
  2. `RdInc` for increment claim reduction, scaled by dense embedding;
  3. field-inline field `RdInc`, scaled by dense embedding;
  4. committed instruction RA openings for hamming-weight claim reduction;
  5. committed bytecode RA openings for hamming-weight claim reduction;
  6. committed RAM RA openings for hamming-weight claim reduction;
  7. trusted advice final opening, if present;
  8. untrusted advice final opening, if present.
- Advice final opening points come from:
  - Stage 7 advice address phase when the advice layout has an address phase;
  - Stage 6 advice cycle phase when the layout has no address phase.
- Advice scaling uses `advice_commitment_embedding_scale`.
- Clear mode appends `LabelWithCount(b"rlc_claims", opening_claims.len())`,
  appends the scaled opening claim values, squeezes RLC powers, computes the
  joint claim, combines commitments, verifies the PCS proof, and binds opening
  inputs with `PCS::bind_opening_inputs`.
- ZK mode combines commitments with RLC powers, verifies with `PCS::verify_zk`,
  obtains a hiding evaluation commitment, and binds opening inputs with
  `PCS::bind_zk_opening_inputs`.
- `outputs.rs` owns `Stage8OpeningId`, `Stage8ClearOutput`,
  `Stage8ZkOutput`, and `Stage8Output`.

Stage 8 prover code must import verifier-owned opening ID/output structs
directly rather than duplicating those shapes locally.

### jolt-prover

There is currently no `crates/jolt-prover/src/stages/stage8/` module.

The Stage 8 implementation must add:

- `mod.rs`, `input.rs`, `output.rs`, `request.rs`, and `prove.rs`;
- a canonical public Stage 8 `prove` entrypoint;
- final opening order construction shared with verifier-owned IDs;
- clear and ZK PCS proof generation;
- retained opening-hint plumbing from Stage 0 commitment output;
- streaming RLC backend requests;
- advice and field-inline final opening support;
- verifier replay/harness adapters.

Stage 8 is not accepted if it only materializes dense polynomials and calls a
generic PCS proof path. The materialized fallback is useful for tests, but
production replacement requires the optimized streaming RLC path.

### jolt-backends

The backend ledger already names a broad required Stage 8 entry:

- `cpu_opening_stage8_kernels`;

and related certified or required entries:

- `cpu_materialized_opening_evaluations`;
- `cpu_polynomial_representation_kernels`;
- `cpu_rlc_polynomial_vector_matrix_product`;
- commitment/hint rows from Stage 0, including retained opening hints.

Required CPU work:

- preserve opening accumulator behavior;
- preserve pending output-claim row order for ZK;
- preserve streaming Stage 8 RLC construction;
- preserve advice Lagrange/embedding scaling;
- preserve folded one-hot table optimization for RLC VMP;
- preserve retained PCS opening hints from commitments;
- avoid full trace/advice rematerialization in the accepted path.

### jolt-prover-harness

The harness currently has `frontier_openings.rs` and opening-related
subfrontiers for earlier stage output openings, but no explicit Stage 8
production frontier rows in the manifest.

Add at least:

- `stage8_final_opening_order`;
- `stage8_streaming_rlc`;
- `stage8_joint_opening_proof`;
- `stage8_zk_final_opening` once ZK Stage 8 lands.

Initial fixture coverage should include:

- `MuldivSmall` for base final opening order and PCS proof;
- `AdviceConsumer` for trusted/untrusted advice final openings;
- `FieldInlineSmall` for field-inline final `rd_inc`;
- `ZkMuldivSmall` and `ZkAdviceConsumer` for hidden evaluation commitments and
  BlindFold handoff.

Required optimization IDs should include:

- `OPT-OPEN-001` for opening accumulator behavior;
- `OPT-OPEN-002` for pending ZK output rows;
- `OPT-OPEN-003` for streaming Stage 8 RLC;
- `OPT-OPEN-004` for dense-polynomial embedding factor;
- `OPT-OPEN-005` for advice embedding factor;
- `OPT-OPEN-006` for RLC streaming VMP;
- `OPT-OPEN-007` for folded one-hot tables;
- `OPT-OPEN-009` for joint-claim computation before PCS proof;
- `OPT-OPEN-010` for ZK evaluation commitment extraction;
- `OPT-POLY-013` for RLC polynomial representation;
- `OPT-COM-006` for retained opening hints;
- relevant `OPT-MEM-*`, `OPT-FLD-*`, and PCS rows touched by the concrete
  implementation.

## Target Prover Shape

Add a canonical public entrypoint in `prove.rs`:

```rust
pub fn prove_stage8<F, PCS, VC, W, B, T, ZkProof>(
    input: Stage8ProverInput<'_, F, PCS, VC, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage8ProverOutput<F, PCS::Output, VC::Output>, ProverError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + AdditivelyHomomorphic,
    W: WitnessProvider<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>;
```

The exact bounds may differ for clear and ZK PCS APIs, but the input should
make these dependencies explicit:

- checked trace shape, public I/O, and proof mode;
- preprocessing PCS setup;
- prover commitments and retained opening hints;
- trusted and untrusted advice commitments plus retained hints;
- one-hot config and trace polynomial order;
- Stage 6 and Stage 7 outputs;
- normalized witness/advice provider;
- selected backend and PCS configuration.

`Stage8ProverOutput` should carry:

- `joint_opening_proof`;
- verifier-owned `Stage8Output`;
- final opening IDs and constraint coefficients;
- joint claim and joint commitment in clear mode;
- hiding evaluation commitment in ZK mode;
- BlindFold binding data in ZK mode.

### Transparent Flow

Transparent Stage 8 should:

1. Reconstruct formula dimensions and validate commitment layout.
2. Take the final opening point from Stage 7 hamming-weight output.
3. Compute dense embedding scale and PCS/common opening points.
4. Resolve advice final opening points from Stage 7 or Stage 6 according to
   advice layout.
5. Build final opening IDs, commitments, opening claims, and scaling factors in
   verifier final-opening order.
6. Append `rlc_claims` count and scaled opening-claim values to the transcript.
7. Squeeze RLC powers.
8. Compute the joint claim once from scaled claims and RLC powers.
9. Combine commitments.
10. Dispatch a streaming RLC/opening backend request that uses retained opening
    hints and witness/advice streams.
11. Produce `joint_opening_proof`.
12. Bind opening inputs with `PCS::bind_opening_inputs`.
13. Assemble `Stage8ClearOutput`.

The proof payload order from Stage 0 commitments must not be confused with
Stage 8 final-opening order.

### ZK Flow

ZK Stage 8 should:

1. Build the same final opening ID order and scaling factors.
2. Squeeze RLC powers in the verifier's ZK order without appending hidden
   evaluation values.
3. Combine commitments into the joint commitment.
4. Produce the ZK PCS opening proof with the hidden evaluation value.
5. Extract or return the hiding evaluation commitment expected by
   `PCS::verify_zk`.
6. Bind opening inputs with `PCS::bind_zk_opening_inputs`.
7. Assemble `Stage8ZkOutput` for BlindFold.

The ZK path must preserve output-claim row ordering from prior committed
sumchecks. A row-order-only test should fail if any Stage 5/6/7 committed row
is swapped before Stage 8/BlindFold.

### Advice Flow

For each advice kind:

- If the advice commitment is absent, do not include a final advice opening.
- If the advice layout has an address phase, use the Stage 7 advice
  address-phase opening point and claim.
- If the advice layout has no address phase, use the Stage 6 advice cycle-phase
  opening point and claim.
- Scale the opening by `advice_commitment_embedding_scale`.

Trusted advice must precede untrusted advice in final opening order.

### Field-Inline Flow

Under `field-inline`, Stage 8 must insert field-register `rd_inc` immediately
after base `RamInc` and `RdInc`, scaled by the same dense embedding factor.

The field-inline opening ID should come from field-inline claim helpers and be
wrapped as `Stage8OpeningId::FieldInline`.

## Backend and Performance Requirements

Stage 8 replacement readiness requires these backend ledger entries, or more
specific successor entries, to be `ParityCertified` with evidence:

- `cpu_opening_stage8_kernels`;
- `cpu_polynomial_representation_kernels` or certified narrower RLC entries;
- `cpu_rlc_polynomial_vector_matrix_product`;
- retained opening-hint commitment rows from Stage 0;
- relevant PCS, field, memory, and one-hot RLC rows.

Performance evidence must include:

- focused streaming Stage 8 RLC benchmark;
- CPU RLC VMP microbenchmark;
- final PCS opening proof benchmark with retained hints;
- advice final opening benchmark;
- field-inline final opening benchmark after field-inline support lands;
- analytical peak-memory accounting showing the accepted path does not
  materialize the full witness unless the fixture explicitly selects the
  fallback;
- canonical fixture comparison against `jolt-core`;
- the default 15% timing and peak-memory parity threshold.

`cpu_materialized_opening_evaluations` is useful as a correctness reference.
It is not sufficient replacement evidence for Stage 8 when core uses streaming
RLC.

## Required Tests and Gates

Before accepting Stage 8:

- harness manifest rows exist for Stage 8 final opening frontiers;
- harness static checks pass for manifest, optimization IDs, backend ledger,
  source drift, and workspace boundaries;
- final opening order is tested independently of proof payload order;
- transparent `MuldivSmall` and `AdviceConsumer` verifier replay pass;
- advice final openings cover Stage 7 address phase and Stage 6 cycle phase
  selection when practical;
- field-inline final opening replay passes after field-inline implementation
  lands;
- ZK final opening replay passes after ZK Stage 8 implementation lands;
- tamper tests cover dense embedding scale, advice embedding scale, opening
  order, and RLC claim transcript binding;
- backend microbench evidence files are recorded in the ledger;
- `validate_frontier_replacement_ready` passes for Stage 8 with
  `ParityCertified` backend status;
- `validate_global_cpu_backend_inventory_coverage` passes.

Expected local commands after Stage 8 tests exist:

```bash
cargo nextest run -p jolt-prover-harness frontier_openings --cargo-quiet
cargo nextest run -p jolt-prover-harness optimization_inventory --cargo-quiet
cargo nextest run -p jolt-prover-harness frontier_gates --cargo-quiet
```

## Acceptance Checklist

- [ ] `crates/jolt-prover/src/stages/stage8/` exists.
- [ ] One canonical Stage 8 prover entrypoint exists.
- [ ] Stage 8 imports verifier-owned opening ID/output structs directly.
- [ ] Final opening order matches `jolt-verifier` exactly.
- [ ] Dense and advice embedding scales are computed from `jolt-claims`
      helpers or verifier-equivalent shared helpers.
- [ ] Retained opening hints from Stage 0 flow into Stage 8.
- [ ] Streaming RLC is the accepted production path.
- [ ] Transparent Stage 8 verifier replay passes for base and advice fixtures.
- [ ] Field-inline final opening passes.
- [ ] ZK Stage 8 returns hidden evaluation commitment and BlindFold data.
- [ ] Backend kernel evidence is `ParityCertified`.
- [ ] Final correctness and performance justification is appended below.

## Final Justification Log

### In progress — deterministic final-opening structure landed and verifier-validated

Stage 8 is being built slice-by-slice. The first landed slice is the
deterministic final-opening structure (everything the verifier computes at the
value level, up to but excluding the PCS proof generation).

Implemented (`crates/jolt-prover/src/stages/stage8/{mod,input,output,prove}.rs`):

- `Stage8ProverConfig` (log_t, committed_chunk_bits, `JoltRaPolynomialLayout`,
  `TracePolynomialOrder`, trusted/untrusted advice layouts) and
  `Stage8ProverInput` (Stage 6 + Stage 7 clear outputs), using verifier-owned
  dimension/order types from `jolt-claims`/`jolt-verifier`.
- `derive_stage8_opening_structure` mirrors
  `jolt-verifier/src/stages/stage8/verify.rs`: takes the common opening point
  from Stage 7 hamming-weight output, computes the dense increment embedding
  scale (`EqPolynomial::zero_selector`) and the PCS opening point
  (`TracePolynomialOrder::commitment_opening_point` → `Point::high_to_low`),
  builds the final opening IDs and scaled opening-claim values in the verifier's
  batch order (`RamInc`, `RdInc`, instruction RA, bytecode RA, RAM RA) from the
  Stage 6 increment claim-reduction openings and the Stage 7 reduced RA openings,
  appends `rlc_claims` + the scaled values to the transcript, squeezes the RLC
  powers, and computes the joint claim and constraint coefficients.

Correctness evidence (green):

- `cargo nextest run -p jolt-prover-harness stage8_opening_structure_matches_core_fixtures --features core-fixtures`
  — passes for `MuldivSmall`. The harness runs the native verifier through Stage 7,
  runs `derive_stage8_opening_structure` on a transcript clone, runs the native
  Stage 8 verifier on the original transcript, and asserts the modular structure
  equals `Stage8ClearOutput`: identical `opening_ids`, scaled opening values
  (`opening_claims[i].evaluation.value`), `constraint_coefficients`,
  `opening_point`, `pcs_opening_point`, and `joint_claim`. The same fixture's full
  `fixture.verify()` still accepts the real proof.
- Compiles under transparent and `field-inline` (the prover slice is
  `cfg(not(field-inline))`; field-inline compiles out cleanly).

### Joint-polynomial constituents built and layout-validated (all families)

Every committed constituent of the joint opening polynomial is now built
prover-side, and each one's committed layout is validated by evaluation against
the verifier-owned reduced claims — retiring the byte-exact layout risk.

- RA family: `build_stage8_ra_constituents` / `evaluate_stage8_ra_constituents`
  build each committed RA polynomial as a `OneHotPolynomial` with
  `OneHotIndexOrder::ColumnMajor` over `2^committed_chunk_bits` columns ×
  `2^log_t` rows (matching Stage 0's commitment layout), sourcing per-cycle chunk
  indices from `WitnessProvider::committed_stream`.
- Dense increments: `build_stage8_dense_constituent` /
  `evaluate_stage8_dense_constituents` build `RamInc`/`RdInc` as the `2^log_t`
  increment values zero-padded into the top block of the full
  `2^(committed_chunk_bits+log_t)` space (the dense embedding).
- `cargo nextest run -p jolt-prover-harness stage8_ra_constituents_evaluate_to_reduced_claims --features core-fixtures`
  — passes for `MuldivSmall`: each RA constituent evaluated at the PCS opening
  point equals the Stage 7 hamming-weight reduced claim (scale 1), and each dense
  increment constituent evaluates to `inc_claim · dense_embedding_scale` (the
  Stage 6 reduced increment scaled by the dense embedding). `fixture.verify()`
  still accepts the real proof.

### Joint RLC polynomial materialized and validated

- `materialize_stage8_joint_polynomial` / `evaluate_stage8_joint_polynomial`
  (`stage8/prove.rs`): materialize `Σ γ^i · constituent_i` into one dense
  `2^(committed_chunk_bits+log_t)` evaluation vector matching the Dory commitment
  matrix (dense increments top-block `flat = cycle`, RA polynomials ColumnMajor
  `flat = col·2^log_t + row`), weighted by the raw `gamma_powers`.
  `derive_stage8_structure_and_gamma` exposes the `gamma_powers`.
- `cargo nextest run -p jolt-prover-harness stage8_ra_constituents_evaluate_to_reduced_claims --features core-fixtures`
  now also asserts `joint_polynomial.evaluate(pcs_opening_point) == joint_claim`
  (MuldivSmall) — the exact input to the Dory opening opens to the
  transcript-derived joint claim. `fixture.verify()` still accepts the real proof.

### Materialized joint opening proof generated and verifier-accepted

- `prove_stage8` now derives the verifier-owned Stage 8 opening structure,
  combines the verifier-order commitments and retained Stage 0 Dory opening
  hints with the raw RLC powers, materializes the joint RLC polynomial, calls
  `DoryScheme::open`, binds the final opening inputs, and returns the
  `joint_opening_proof` plus joint commitment.
- `load_stage8_joint_opening_replay_fixture` runs modular Stage 0 commitments
  to obtain commitments and retained hints, advances the native verifier through
  Stage 7, calls `prove_stage8`, splices the generated `joint_opening_proof`
  into the proof shell, and requires native `fixture.verify()` acceptance.
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'binary(frontier_stage8_batch)'`
  passes all 3 Stage 8 transparent/base tests:
  `stage8_opening_structure_matches_core_fixtures`,
  `stage8_ra_constituents_evaluate_to_reduced_claims`, and
  `stage8_joint_opening_proof_verifies_against_core_fixtures`.

### Streaming RLC source and advice final openings landed

- `prove_stage8` now calls `PCS::open_poly` on `Stage8JointRlcSource` instead of
  materializing the full joint evaluation vector for the proof path. Dory's
  `open_poly` adapter calls the source `fold_rows` method, so final opening
  work uses the streaming matrix-vector product hook.
- `crates/jolt-backends/src/cpu/poly/mod.rs` owns the Stage 8 streaming RLC VMP
  kernel. It mirrors core's Dory-left factorization: split the Dory left vector
  into address factors and cycle-row factors, pre-fold instruction/bytecode/RAM
  RA coefficient tables by address chunk, and stream compact per-cycle Jolt VM
  rows over dense increments plus one-hot RA families. This is the
  replacement-grade Stage 8 opening kernel, not a materialized or shape-only
  fallback.
- Advice final openings are included in the deterministic Stage 8 opening
  structure. Trusted/untrusted advice use the verifier-equivalent Stage 7
  address-phase opening when an address phase exists, otherwise the Stage 6
  cycle-phase opening, with the `jolt-claims` advice embedding scale.
- `load_stage8_joint_opening_replay_fixture` now uses modular Stage 0
  commitments/hints for base and advice fixtures, appends trusted/untrusted
  advice commitments in verifier order when present, and splices the modular
  Stage 8 `joint_opening_proof` into a native proof shell.

Correctness evidence:

- `cargo check -p jolt-backends -q`
- `cargo check -p jolt-prover -q`
- `cargo check -p jolt-prover-harness --features core-fixtures --benches --tests -q`
- `cargo clippy -p jolt-prover -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover --features zk -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover --features field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings`
- `cargo nextest run -p jolt-backends --cargo-quiet`
- `cargo nextest run -p jolt-witness --cargo-quiet`
- `cargo nextest run -p jolt-backends cpu_poly_stage8_streaming_rlc_vmp_matches_dense_reference --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'binary(frontier_stage8_batch)'`
  passes all 3 Stage 8 tests. Each test loops over `MuldivSmall` and
  `AdviceConsumer`, so the opening structure, constituent evaluations, and
  modular joint opening proof are native-verifier accepted for base and advice
  fixtures.
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'binary(frontier_manifest) or binary(optimization_inventory) or binary(frontier_gates)'`
  passes, including the replacement-ready evidence and ledger gates.
- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_opening_stage8_kernels cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  wrote
  `target/frontier-metrics/kernel-evidence/cpu_opening_stage8_kernels/frontier_perf_stage8_streaming_rlc.json`
  with latest post-clean status `Pass`, time ratio `0.9351386094873696`, and
  peak-memory ratio `0.812442734102987`.

The Stage 8 transparent/advice path should remain on the full Dory-left
factorized streaming kernel. A dense materialized RLC vector or generic
row-scan fallback is only acceptable as a reference test oracle, not as the
replacement prover path or parity evidence.

Relation-stage prefix/suffix kernels remain a separate hard performance gate.
Stage 5 instruction read-RAF owns the full lookup-table prefix/suffix
decomposition path, and Stage 6 owns the split-eq, delayed-RA, bytecode,
booleanity, and increment kernels. Stage 8 must consume their reduced opening
claims and retained commitment hints; it should not be used to mask a missing
relation prefix/suffix kernel.

Remaining before Stage 8 is replacement-ready (tracked):

- ZK final opening (`PCS::verify_zk` + BlindFold handoff);
- field-inline/address-major performance certification for the embedded
  retained-commitment path. The correctness replay now uses actual Stage 0
  retained `RamInc`, `RdInc`, field `FieldRdInc`, and RA commitments/hints, but
  the accepted replacement path still needs canonical timing and peak-memory
  evidence for every required feature combination;
- full frontier replacement gates once field-inline performance evidence and ZK
  final-opening paths are wired. The transparent/advice Stage 8 streaming-RLC
  opening kernel already has canonical parity evidence loaded through
  `cpu_opening_stage8_kernels`.

### Field-inline final-opening audit

The initial real field-inline SDK replay exposed an address-major layout bug in
the modular Stage 8 assembly: dense `RamInc`/`RdInc`, field `FieldRdInc`, and RA
constituents were embedded as cycle-major even though the verifier-derived PCS
point was address-major. The fix makes Stage 8 constituent construction and the
backend streaming RLC VMP order-aware:

- `TracePolynomialOrder::CycleMajor` keeps dense/RA flat indices as
  `address * T + cycle` and uses the Dory-left factorized streaming kernel;
- `TracePolynomialOrder::AddressMajor` embeds dense field and VM increments at
  `cycle * K + 0`, embeds RA as `cycle * K + address`, and uses the exact
  streaming VMP path rather than the cycle-major factorization;
- field-inline final opening inserts `FieldRdInc` immediately after `RamInc`
  and `RdInc`, using the verifier-owned opening ID and the Stage 6 field
  increment claim scaled by the dense embedding factor.

Evidence added in this slice:

- `cargo check -p jolt-backends -q`
- `cargo check -p jolt-prover --features field-inline -q`
- `cargo check -p jolt-prover-harness --features core-fixtures,field-inline --tests -q`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline field_inline_eq_poly_guest_replays_modular_frontier_with_jolt_verifier --cargo-quiet`

The replay uses real `field-inline-eq-poly-guest` SDK trace data and
verifier-equivalent Stage 8 PCS constituents.

Follow-up slice:

- `CommitmentStageConfig` now carries `TracePolynomialOrder`, and
  `CommitmentRequestItem` threads that order into the CPU commitment backend.
  Through-Stage-8 one-hot commitments select `ColumnMajor` for cycle-major and
  `RowMajor` for address-major, so field-inline Stage 0 RA commitments are now
  reused directly by the Stage 8 replay instead of being reconstructed locally.
- Stage 0 final-opening trace embedding now also applies to dense and compact
  trace polynomials. For address-major field-inline replays, the CPU commitment
  backend embeds `RamInc`, `RdInc`, and field `FieldRdInc` into the full
  `trace_rows * address_columns` PCS domain at `cycle * K`, so Stage 8 consumes
  the actual retained Stage 0 commitments and opening hints rather than
  reconstructing verifier-equivalent local constituents.
- The real field-inline frontier harness now splices the modular Stage 8
  `joint_opening_proof` into a native proof shell and calls
  `jolt_verifier::stages::stage8::verify` directly. The harness uses a
  Stage8-only mock vector commitment whose output type matches the mock PCS
  hiding commitment required by the verifier's Stage 8 generic bound; Stages 1-7
  remain replayed with the normal mock round commitment.
- Additional evidence:
  - `cargo check -p jolt-backends -q`
  - `cargo check -p jolt-prover -q`
  - `cargo check -p jolt-prover --features zk,field-inline -q`
  - `cargo check -p jolt-prover-harness --features core-fixtures --tests -q`
  - `cargo check -p jolt-prover-harness --features core-fixtures,field-inline --tests -q`
  - `cargo nextest run -p jolt-prover stage0 --cargo-quiet`
  - `cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline field_inline_eq_poly_guest_replays_modular_frontier_with_jolt_verifier --cargo-quiet`

This proves that the field-inline Stage 8 final-opening component now matches
the native verifier frontier through `stage8::verify` while using actual
retained Stage 0 `RamInc`, `RdInc`, field `FieldRdInc`, and RA commitments in
address-major order. It is still not a full field-inline `JoltProof`
replacement certification until the address-major embedded commitment/opening
path has canonical timing and peak-memory parity evidence and the ZK final
opening path is implemented.

### ZK final opening and Stage 3-8 committed replay

Stage 8 now has a ZK final-opening prover entrypoint that mirrors the native
verifier's ZK transcript order. Unlike clear Stage 8, the ZK derivation does not
append clear RLC opening claims before squeezing RLC powers; it computes the
hidden joint claim prover-side, opens the streaming joint RLC polynomial with
`PCS::open_zk_poly`, binds the hiding evaluation commitment, and returns the
BlindFold handoff material (`hiding_evaluation_commitment` and blind).

Implemented:

- `ZkOpeningScheme::open_zk_poly` with a dense default and a Dory streaming
  override, so Stage 8 ZK can use the existing streaming RLC source rather than
  forcing materialization for Dory.
- `prove_stage8_zk`, returning verifier-equivalent opening IDs, constraint
  coefficients, joint commitment, PCS proof, hidden evaluation commitment, and
  hidden evaluation blind.
- `derive_stage8_zk_structure_and_gamma`, which shares final-opening ordering
  with clear Stage 8 but follows the ZK verifier transcript boundary.
- The real field-inline SDK frontier harness now proves committed stages 3-7,
  proves the Stage 8 ZK final opening, runs native `jolt-verifier` through
  `stage8::verify`, and checks `Stage8Output::Zk`.

Correctness evidence:

- `cargo check -p jolt-openings -q`
- `cargo check -p jolt-dory -q`
- `cargo check -p jolt-prover -q`
- `cargo check -p jolt-prover --features zk,field-inline -q`
- `cargo check -p jolt-prover-harness --features core-fixtures,field-inline,zk -q`
- `cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest_accepts_zk_committed_stage3_to_stage8_boundaries --features core-fixtures,field-inline,zk --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'binary(frontier_stage8_batch)'`

Performance evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_opening_stage8_kernels cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  wrote
  `target/frontier-metrics/kernel-evidence/cpu_opening_stage8_kernels/frontier_perf_stage8_streaming_rlc.json`
  with status `Pass`, time ratio `0.9351386094873696`, and peak-memory ratio
  `0.812442734102987`.

Justification: Stage 8 is now verifier-accepted for transparent/advice,
field-inline clear, and ZK + field-inline final-opening frontiers. The Dory ZK
opening path has a streaming `open_zk_poly` hook, so the accepted path continues
to use the Stage 8 streaming RLC source and retained Stage 0 hints instead of a
dense replacement. This completes native verifier acceptance through Stage 8 at
the committed-boundary/frontier level. Full replacement certification still
requires the top-level BlindFold proof assembly over the committed witnesses.

### Post-clean global evidence restoration

After the target cleanup, the global backend-kernel ledger evidence was restored
and revalidated. The restoration kept the backend-first rule: failing evidence
was fixed in kernels or benchmarks before re-writing canonical JSON.

Fixes made while restoring evidence:

- `cpu_one_hot_vector_matrix_product`: removed duplicate validation from the
  allocation wrapper and used the already-validated one-hot shape inside the
  hot column-major loop, matching the core commitment layout.
- `cpu_compressed_unipoly`: moved the backend wrapper onto the same trusted
  hot-path compression contract as core and forced the tiny wrapper calls
  inline.
- `cpu_linear_product_d4`: corrected the public D4 kernel to compute only the
  verifier grid `[1, 2, 3, infinity]` instead of routing through the internal
  `[1, 2, 3, 4, infinity]` helper and discarding the extra point.
- `cpu_advice_commitment_contexts`: measured core with retained advice hints
  and serialized pure BlindFold-retained commitment batches to match core's
  peak-memory profile.
- Stage 0 one-hot commitment tests now assert the requested trace order:
  cycle-major RA commitments use `OneHotIndexOrder::ColumnMajor`.

Canonical evidence and gates:

- All registered `ParityCertified` backend kernel evidence files exist again;
  `target/frontier-metrics/kernel-evidence` contains 53 JSON files.
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'binary(optimization_inventory)'`
  passes all 12 tests, including canonical path validation, evidence validity,
  inventory coverage, and replacement-readiness checks.
- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'binary(frontier_stage3_batch) or binary(frontier_stage4_batch) or binary(frontier_stage5_batch) or binary(frontier_stage6_batch) or binary(frontier_stage7_batch) or binary(frontier_stage8_batch) or binary(frontier_openings)'`
  passes 45/45 tests, including verifier replay and Stage 8 joint opening
  verification against core fixtures.
- `cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline --cargo-quiet -E 'binary(frontier_stage0_commitments) or binary(frontier_stage4_batch) or binary(frontier_stage5_batch) or binary(frontier_stage6_batch) or binary(field_inline_sdk_guest)'`
  passes 19/19 tests on field-inline surfaces.
- `cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline,zk --cargo-quiet -E 'binary(field_inline_sdk_guest) or binary(frontier_stage0_commitments)'`
  passes 13/13 tests, including real SDK `zk + field-inline` committed-boundary
  replay through Stage 8.
- `cargo clippy --all --features host -q --all-targets -- -D warnings` and
  `cargo clippy --all --features host,zk -q --all-targets -- -D warnings`
  both pass.
- `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host` and
  `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk`
  both pass.

Justification: the Stage 3-8 committed-boundary/frontier prover is now
correctness-accepted by native verifier replay for transparent/advice,
field-inline, ZK, and `zk + field-inline` surfaces covered by the harness, and
every registered parity-certified CPU kernel has canonical timing and
peak-memory evidence within the accepted gate. The remaining known non-frontier
gap is full top-level BlindFold proof assembly over committed witnesses.

### Post-clean manifest rail tightening

The harness manifest now tracks Stage 8 as a first-class frontier instead of
leaving it as implicit coverage under `frontier_stage8_batch` tests:

- `stage8_final_opening` covers transparent/advice fixtures
  (`MuldivSmall`, `AdviceConsumer`);
- `stage8_zk_final_opening` covers ZK final-opening fixtures
  (`ZkMuldivSmall`, `ZkAdviceConsumer`);
- `stage8_field_inline_final_opening` covers the field-inline final-opening
  fixture (`FieldInlineSmall`).

The same pass fixed adjacent manifest drift that would otherwise weaken
replacement gates:

- Stage 6 regular-batch metadata now includes `OPT-REL-004` and `OPT-REL-005`,
  matching the bytecode read-RAF evidence owned by
  `cpu_stage6_regular_batch_sumcheck`;
- Stage 7 regular-batch sumcheck fixtures now include `AdviceConsumer`, matching
  the committed acceptance path and benchmark evidence;
- Stage 7 regular-batch sumcheck metadata now includes `OPT-REL-013`, matching
  the advice address-phase work owned by `cpu_stage7_regular_batch_sumcheck`.

Validation:

- `cargo nextest run -p jolt-prover-harness --features core-fixtures --cargo-quiet -E 'binary(frontier_manifest) or binary(optimization_inventory)'`
  passes 16/16.

This does not change protocol code. It makes the manifest reflect the Stage 8
frontier and the registered Stage 6/7 kernel evidence, so later top-level proof
work cannot accidentally claim replacement readiness while bypassing those
gates.
