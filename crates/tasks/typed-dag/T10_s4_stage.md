# T10: S4 Stage Function (2-Instance Batch)

**Status**: `[ ]` Not started
**Depends on**: T04, T05, T06
**Blocks**: T15
**Crate**: `jolt-zkvm`
**Estimated scope**: Medium (~250 lines)

## Objective

Implement `prove_stage4()` — RegistersReadWriteChecking + RamValCheck.

## Instances

| # | Instance | Rounds | Degree | jolt-core |
|---|----------|--------|--------|-----------|
| 1 | RegistersReadWriteChecking | `log_k + log_T` | 3 | `zkvm/registers/read_write_checking.rs` |
| 2 | RamValCheck | `log_T` | 3 | `zkvm/ram/val_check.rs` |

## Function Signature

```rust
fn prove_stage4<F, T, B>(
    s2: &Stage2Output<F>,
    s3: &Stage3Output<F>,
    tables: &PolynomialTables<F>,
    config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> Stage4Output<F>
```

## Key Complexity

- **RegistersRW has chunked address binding**: Like RamRW in S2, this
  operates over `(register_address || cycle)` domain.
- **RamValCheck reads from S2**: Uses `ram_val_s2` and `ram_val_final_s2`
  from the Stage 2 output for its input claim.
- **RamValCheck combines initial RAM state**: The input claim includes
  `Val_init(r_address)` evaluated from the initial RAM state polynomial.
- **Committed outputs**: Both `ram_inc_at_s4` and `rd_inc_at_s4` are
  produced here and consumed by IncCR in S6.

## Reference

- jolt-core: `jolt-core/src/zkvm/prover.rs::prove_stage4()` (lines 1064-1136)
- RegistersRW: `jolt-core/src/zkvm/registers/read_write_checking.rs`
- RamValCheck: `jolt-core/src/zkvm/ram/val_check.rs`

## Acceptance Criteria

- [ ] 2 instances batched correctly
- [ ] RegistersRW uses chunked address domain
- [ ] RamValCheck input claim reads from s2 outputs
- [ ] Committed evals (ram_inc, rd_inc) extracted correctly
- [ ] Unit test with synthetic data
- [ ] `cargo clippy -p jolt-zkvm` passes
