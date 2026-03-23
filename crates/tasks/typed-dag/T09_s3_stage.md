# T09: S3 Stage Function (3-Instance Batch)

**Status**: `[ ]` Not started
**Depends on**: T04, T05, T06
**Blocks**: T15
**Crate**: `jolt-zkvm`
**Estimated scope**: Medium (~300 lines)

## Objective

Implement `prove_stage3()` — Shift + InstructionInput + RegistersClaimReduction.

## Instances

| # | Instance | Rounds | Degree | jolt-core |
|---|----------|--------|--------|-----------|
| 1 | Shift | `log_T` | 2 | `zkvm/spartan/shift.rs` |
| 2 | InstructionInput | `log_T` | 3 | `zkvm/spartan/instruction_input.rs` |
| 3 | RegistersClaimReduction | `log_T` | 2 | `zkvm/claim_reductions/registers.rs` |

## Function Signature

```rust
fn prove_stage3<F, T, B>(
    s1: &SpartanOutput<F>,
    s2: &Stage2Output<F>,
    tables: &PolynomialTables<F>,
    config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> Stage3Output<F>
```

## Key Complexity

- **Shift uses EqPlusOne**: Instead of standard eq polynomial, uses
  `EqPlusOnePolynomial`. The `KernelEvaluator` or a specialized witness
  impl handles this via prefix-suffix decomposition.
- **Shift reads from S1 AND S2**: Input claim combines virtual evals
  from Spartan outer (S1) and the PV product claim (S2).
- **r_product deferred**: The shift product claim uses `r_product` from
  S2's PV output point — this is now trivially available as `s2.pv_point`.

## Reference

- jolt-core: `jolt-core/src/zkvm/prover.rs::prove_stage3()` (lines 993-1061)
- Shift: `jolt-core/src/zkvm/spartan/shift.rs`
- InstrInput: `jolt-core/src/zkvm/spartan/instruction_input.rs`
- RegistersCR: `jolt-core/src/zkvm/claim_reductions/registers.rs`

## Acceptance Criteria

- [ ] 3 instances batched correctly
- [ ] Shift uses EqPlusOne (not standard eq)
- [ ] Shift r_product sourced from s2.pv_point
- [ ] Challenge squeeze order matches jolt-core
- [ ] Unit test with synthetic data
- [ ] `cargo clippy -p jolt-zkvm` passes
