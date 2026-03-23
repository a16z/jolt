# T13: S7 Stage Function (HammingWeightClaimReduction)

**Status**: `[ ]` Not started
**Depends on**: T04, T05, T06
**Blocks**: T15
**Crate**: `jolt-zkvm`
**Estimated scope**: Medium (~200 lines)

## Objective

Implement `prove_stage7()` — HammingWeightClaimReduction. This is the
critical stage that produces the **unified opening point**.

## Instances

| # | Instance | Rounds | Degree | jolt-core |
|---|----------|--------|--------|-----------|
| 1 | HammingWeightClaimReduction | `log_k_chunk` | 2 | `zkvm/claim_reductions/hamming_weight.rs` |

Only `log_k_chunk` rounds (address variables only). The cycle variables
were already bound by IncCR in S6.

## Function Signature

```rust
fn prove_stage7<F, T, B>(
    s5: &Stage5Output<F>,
    s6: &Stage6Output<F>,
    tables: &PolynomialTables<F>,
    config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> Stage7Output<F>
```

## Key Complexity

### Unified Point Construction

The output point is:
```
unified_point = [r_addr_s7 || r_cycle_s6]
```
where:
- `r_addr_s7` = challenges from this stage's sumcheck (length `log_k_chunk`)
- `r_cycle_s6` = `s6.r_cycle_s6` (from IncCR in Stage 6, length `log_T`)

This concatenation creates the single point at which ALL committed polynomials
are opened for the PCS proof.

### Input Claim

The input claim batches three types of evaluations per RA polynomial family
(instruction, bytecode, RAM) with γ-powers:
```
Σ_i [γ^{3i} · hw_claim_i + γ^{3i+1} · bool_claim_i + γ^{3i+2} · virt_claim_i]
```

Where:
- `hw_claim_i` = Hamming weight evaluations from S6
- `bool_claim_i` = Booleanity RA evals from S6
- `virt_claim_i` = RA virtual evals from S5/S6

### Output

All RA polynomial evaluations at the unified point:
- `instruction_ra[0..d]` at `unified_point`
- `bytecode_ra[0..d]` at `unified_point`
- `ram_ra[0..d]` at `unified_point`

These go directly to the PCS opening in T14.

## Reference

- jolt-core: `jolt-core/src/zkvm/prover.rs::prove_stage7()` (lines 1832-1897)
- HammingWeightCR: `jolt-core/src/zkvm/claim_reductions/hamming_weight.rs`
- Point construction: `normalize_opening_point()` in hamming_weight.rs

## Acceptance Criteria

- [ ] Single instance, `log_k_chunk` rounds
- [ ] Unified point correctly constructed as `[r_addr || r_cycle_s6]`
- [ ] Input claim batches all 3 eval types per RA family
- [ ] All RA polynomial evals at unified point in output
- [ ] Unit test: verify unified_point.len() == log_k_chunk + log_T
- [ ] `cargo clippy -p jolt-zkvm` passes
