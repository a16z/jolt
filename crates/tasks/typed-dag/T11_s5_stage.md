# T11: S5 Stage Function (3-Instance Batch)

**Status**: `[ ]` Not started
**Depends on**: T04, T05, T06
**Blocks**: T15
**Crate**: `jolt-zkvm`
**Estimated scope**: Medium (~250 lines)

## Objective

Implement `prove_stage5()` — InstructionReadRaf + RamRaClaimReduction + RegistersValEvaluation.

## Instances

| # | Instance | Rounds | Degree | jolt-core |
|---|----------|--------|--------|-----------|
| 1 | InstructionReadRaf | `log_k` | 2 | `zkvm/instruction_lookups/read_raf_checking.rs` |
| 2 | RamRaClaimReduction | `log_k` | 2 | `zkvm/claim_reductions/ram_ra.rs` |
| 3 | RegistersValEvaluation | varies | 3 | `zkvm/registers/val_evaluation.rs` |

## Function Signature

```rust
fn prove_stage5<F, T, B>(
    s2: &Stage2Output<F>,
    s4: &Stage4Output<F>,
    tables: &PolynomialTables<F>,
    config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> Stage5Output<F>
```

## Key Complexity

- **InstructionReadRaf**: Proves instruction lookups are correctly read via
  read-after-flag checking. Input claim reads from S2's
  InstructionLookupsClaimReduction output.
- **RegistersValEvaluation**: Proves register values equal sum of increments
  weighted by LT polynomial. Uses split representation for √N memory.
- **Committed outputs**: `rd_inc_at_s5` for IncCR, `instruction_ra_at_s5`
  for HammingWeightCR.

## Reference

- jolt-core: `jolt-core/src/zkvm/prover.rs::prove_stage5()` (lines 1140-1207)
- InstructionReadRaf: `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`
- RamRaCR: `jolt-core/src/zkvm/claim_reductions/ram_ra.rs`
- RegistersValEval: `jolt-core/src/zkvm/registers/val_evaluation.rs`

## Acceptance Criteria

- [ ] 3 instances batched correctly
- [ ] InstructionReadRaf input claim from s2 outputs
- [ ] RegistersValEval uses LT polynomial
- [ ] Committed evals extracted for downstream stages
- [ ] Unit test with synthetic data
- [ ] `cargo clippy -p jolt-zkvm` passes
