# T08: S2 Stage Function (5-Instance Batch)

**Status**: `[ ]` Not started
**Depends on**: T04 (IR→Kernel Bridge), T05 (Stage Output Types), T06 (Input Claim Formulas)
**Blocks**: T15 (Prove Orchestrator)
**Crate**: `jolt-zkvm`
**Estimated scope**: Large (~400 lines)

## Objective

Implement `prove_stage2()` — the 5-instance batched sumcheck corresponding
to jolt-core's Stage 2.

## Instances (batched, share challenges)

| # | Instance | Rounds | Degree | jolt-core location |
|---|----------|--------|--------|--------------------|
| 1 | RamReadWriteChecking | `log_k + log_T` | 3 | `zkvm/ram/read_write_checking.rs` |
| 2 | ProductVirtualRemainder | `log_T` | varies | `zkvm/spartan/product.rs` |
| 3 | InstructionLookupsClaimReduction | `log_T` | 2 | `zkvm/claim_reductions/instruction_lookups.rs` |
| 4 | RamRafEvaluation | `log_k` | 2 | `zkvm/ram/raf_evaluation.rs` |
| 5 | OutputCheck | `log_k` | 2 | `zkvm/ram/output_check.rs` |

Instance 2 uses uni-skip for its first round.

## Function Signature

```rust
fn prove_stage2<F, T, B>(
    s1: &SpartanOutput<F>,
    tables: &PolynomialTables<F>,
    config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> Stage2Output<F>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: ComputeBackend,
```

## Implementation Steps

1. **Squeeze challenges** from transcript (must match jolt-core order):
   - γ for RamRW batching
   - Random eq point for RamRW
   - Challenges for InstructionLookupsCR
   - Challenges for RamRaf
   - Challenges for OutputCheck
   - PV uni-skip challenges

2. **Compute input claims** using T06 functions:
   - `ram_rw_input_claim(&s1.evals, gamma)`
   - `instruction_lookups_cr_input_claim(&s1.evals, gamma_il)`
   - `ram_raf_input_claim(...)` (from initial RAM state)
   - `output_check_input_claim(...)` (from final RAM state)
   - `pv_remainder_input_claim(...)` (from uni-skip)

3. **Build witnesses** for each instance:
   - Upload relevant polynomial tables to backend
   - Build `KernelDescriptor` from IR (T04 bridge)
   - Compile kernel, wrap in `KernelEvaluator`

4. **Build `SumcheckClaim`** for each instance (num_vars, degree, claimed_sum)

5. **Run `BatchedSumcheckProver::prove_with_handler`** on all 5 instances

6. **Extract typed output**:
   - Each sub-instance's evaluation point from shared challenges
   - Polynomial evaluations at those points
   - Populate `Stage2Output<F>` fields

## Key Complexity

- **Different `num_vars`**: RamRW has `log_k + log_T`, PV/InstrCR have `log_T`,
  Raf/Output have `log_k`. The `BatchedSumcheckProver` handles padding, but
  offset extraction must be correct.
- **Uni-skip for PV**: The PV instance's first round uses an analytic polynomial.
  This is set via `KernelEvaluator::first_round_polynomial()`.
- **RamRW binding order**: Uses specific phase decomposition (address → cycle).

## Reference

- jolt-core: `jolt-core/src/zkvm/prover.rs::prove_stage2()` (lines 880-990)
- RamRW: `jolt-core/src/zkvm/ram/read_write_checking.rs`
- PV: `jolt-core/src/zkvm/spartan/product.rs`
- InstrLookupsCR: `jolt-core/src/zkvm/claim_reductions/instruction_lookups.rs`

## Acceptance Criteria

- [ ] 5 instances batched correctly
- [ ] Challenge squeeze order matches jolt-core
- [ ] Input claims match jolt-core formulas
- [ ] Uni-skip for PV handled
- [ ] Unit test: synthetic data → verify output evals match brute force
- [ ] `cargo clippy -p jolt-zkvm` passes
