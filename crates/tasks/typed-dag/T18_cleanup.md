# T18: Cleanup Old Pipeline

**Status**: `[x]` Done (merged into T06 — old pipeline deleted during cleanup)
**Depends on**: T17 (E2E muldiv passes)
**Blocks**: Nothing
**Crate**: `jolt-zkvm`
**Estimated scope**: Small (deletions)

## Objective

Delete all old pipeline code that has been replaced by the typed DAG.

## Deletions

### Files to delete

- `jolt-zkvm/src/pipeline.rs` — old `prove_stages()` loop
- `jolt-zkvm/src/stage.rs` — `ProverStage` trait, `CompositeStage`
- `jolt-zkvm/src/stages/s2_product_virtual.rs`
- `jolt-zkvm/src/stages/s3_shift.rs`
- `jolt-zkvm/src/stages/s3_instruction_input.rs`
- `jolt-zkvm/src/stages/s3_claim_reductions.rs`
- `jolt-zkvm/src/stages/s4_ram_rw.rs`
- `jolt-zkvm/src/stages/s4_rw_checking.rs`
- `jolt-zkvm/src/stages/s5_ram_checking.rs`
- `jolt-zkvm/src/stages/s5_registers_val_eval.rs`
- `jolt-zkvm/src/stages/s6_booleanity.rs`
- `jolt-zkvm/src/stages/s6_ra_booleanity.rs`
- `jolt-zkvm/src/stages/s7_hamming_reduction.rs`

### Code to delete from prover.rs

- `CommittedTables` struct (replaced by `PolynomialTables`)
- `build_prover_stages()` function
- `build_verifier_descriptors()` function
- `EagerVerifierSource` struct
- `prove_pipeline()` function
- Old `prove()` that delegates to `prove_pipeline()`

### Tests to update/delete

- `tests/synthetic_pipeline.rs` — replace with typed DAG tests
- `tests/full_pipeline.rs` — replace with typed DAG tests
- `tests/soundness.rs` — update to use new `prove()`/`verify()`

### Evaluators to review

- `evaluators/catalog.rs` — may be partially replaced by T04 bridge
- `evaluators/kernel.rs` — keep (still used)
- `evaluators/segmented.rs` — review if still needed

## Notes

- Only delete after T17 confirms the new pipeline is correct.
- Keep `KernelEvaluator` and `ComputeBackend` infrastructure — these are
  used by the new stage functions.
- Keep any test utilities that are reusable.

## Acceptance Criteria

- [ ] All old pipeline code deleted
- [ ] No dead code warnings
- [ ] All tests pass (new pipeline)
- [ ] `cargo clippy -p jolt-zkvm` passes
- [ ] `cargo nextest run -p jolt-zkvm` passes
