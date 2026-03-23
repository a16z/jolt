# T12: S6 Stage Function (6-Instance Batch)

**Status**: `[ ]` Not started
**Depends on**: T04, T05, T06
**Blocks**: T15
**Crate**: `jolt-zkvm`
**Estimated scope**: Large (~500 lines) — most complex stage

## Objective

Implement `prove_stage6()` — the largest batched stage with 6 instances.

## Instances

| # | Instance | Rounds | Degree | jolt-core |
|---|----------|--------|--------|-----------|
| 1 | BytecodeReadRaf | `log_k` | 2 | `zkvm/bytecode/read_raf_checking.rs` |
| 2 | Booleanity | `log_T + log_k` | 3 | `subprotocols/booleanity.rs` |
| 3 | HammingBooleanity | `log_k` | 3 | claim_reductions |
| 4 | RamRaVirtual | `log_T + log_k` | varies | `zkvm/ram/` |
| 5 | InstructionRaVirtual | `log_T + log_k` | varies | `zkvm/instruction_lookups/ra_virtual.rs` |
| 6 | IncClaimReduction | `log_T` | 2 | `zkvm/claim_reductions/increments.rs` |

## Function Signature

```rust
fn prove_stage6<F, T, B>(
    s2: &Stage2Output<F>,
    s4: &Stage4Output<F>,
    s5: &Stage5Output<F>,
    tables: &PolynomialTables<F>,
    config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> Stage6Output<F>
```

## Key Complexity

This is the most complex stage for several reasons:

### IncClaimReduction (Instance 6)
- **Reads from 3 prior stages**: S2, S4, S5
- **4 committed polynomial evals as input**:
  - `s2.ram_inc_at_s2.eval` (RamInc from RamRW)
  - `s4.ram_inc_at_s4.eval` (RamInc from RamValCheck)
  - `s4.rd_inc_at_s4.eval` (RdInc from RegistersRW)
  - `s5.rd_inc_at_s5.eval` (RdInc from RegistersValEval)
- **Output**: `r_cycle_s6` point + reduced `ram_inc`/`rd_inc` evals
- **CRITICAL**: `r_cycle_s6` feeds into S7 to construct unified point

### Booleanity (Instance 2)
- Proves one-hot encoding of all RA polynomials
- Operates over `(address || cycle)` domain → `log_T + log_k_chunk` rounds
- Zero-check: claimed sum is 0

### RA Virtual (Instances 4, 5)
- Prove RA polynomial correctness via Twist/Shout lookup argument
- Use `SharedRaPolynomials` pattern for memory efficiency
- Toom-Cook mode for eq factoring

### Different `num_vars`
- 6 instances with 3 different `num_vars` values
- BatchedSumcheckProver handles padding, but offset extraction critical

## Reference

- jolt-core: `jolt-core/src/zkvm/prover.rs::prove_stage6()` (lines 1211-1380)
- Booleanity: `jolt-core/src/subprotocols/booleanity.rs`
- RA Virtual: `jolt-core/src/zkvm/instruction_lookups/ra_virtual.rs`
- IncCR: `jolt-core/src/zkvm/claim_reductions/increments.rs`

## Acceptance Criteria

- [ ] 6 instances batched correctly
- [ ] IncCR reads 4 committed evals from S2/S4/S5
- [ ] IncCR output `r_cycle_s6` extracted correctly
- [ ] Booleanity is a zero-check
- [ ] RA virtual uses appropriate witness (Toom-Cook or equivalent)
- [ ] BytecodeReadRaf matches jolt-core
- [ ] Unit test with synthetic data
- [ ] `cargo clippy -p jolt-zkvm` passes
