# T14: PCS Opening Collection

**Status**: `[x]` Done
**Depends on**: T05 (Stage Output Types)
**Blocks**: T15 (Prove Orchestrator)
**Crate**: `jolt-zkvm`
**Estimated scope**: Medium (~200 lines)

## Objective

Implement `prove_opening()` — the final step that collects all committed
polynomial claims, normalizes them to the unified point, and produces a
single batch Dory opening proof.

## Function Signature

```rust
fn prove_opening<PCS, F, T>(
    s6: &Stage6Output<F>,
    s7: &Stage7Output<F>,
    tables: &PolynomialTables<F>,
    commitments: &PcsCommitments<PCS>,
    hints: &PcsHints<PCS>,
    transcript: &mut T,
) -> PCS::Proof
where
    PCS: CommitmentScheme<Field = F> + AdditivelyHomomorphic,
    F: Field,
    T: Transcript<Challenge = F>,
```

## Implementation Steps

### 1. Extract unified point from S7

```rust
let unified = &s7.unified_point;
let r_addr = &unified[..log_k_chunk];
```

### 2. Collect dense polynomial claims (Lagrange-normalized)

Dense polys (RamInc, RdInc) are at `r_cycle_s6` (length `log_T`), but the
unified point is `(r_addr || r_cycle)` (length `log_k + log_T`). The dense
polys are zero-padded in the Dory matrix, so:

```
eval_at_unified = eval_at_r_cycle × ∏(1 - r_addr_i)
```

### 3. Collect RA polynomial claims (already at unified point)

```
instruction_ra[i] @ unified_point → from s7
bytecode_ra[i]    @ unified_point → from s7
ram_ra[i]         @ unified_point → from s7
```

### 4. Build ProverClaim list

Each claim needs:
- Full evaluation table (from `tables`)
- Point (unified for all)
- Eval (from S6/S7 outputs, Lagrange-normalized)

### 5. RLC reduction

```rust
let (reduced, _) = RlcReduction::reduce_prover(claims, transcript);
```

All claims are at the same point → RLC produces a single claim.

### 6. PCS open

```rust
PCS::open(&reduced[0].polynomial, &reduced[0].point, reduced[0].eval, ...)
```

## Key Details

- **Lagrange factor**: `eq_zero_selector(r_addr) = ∏(1 - r_addr_i)`
- **Transcript binding**: In non-ZK mode, claim evals are appended to
  transcript before sampling RLC gamma. In ZK mode, they're secret.
- **Hint handling**: PCS hints from the commit phase are combined via
  `PCS::combine_hints()` during RLC reduction.

## Reference

- jolt-core: `jolt-core/src/zkvm/prover.rs::prove_stage8()` (lines 1903-2048)
- `EqPolynomial::zero_selector()` in jolt-poly
- `RlcReduction` in jolt-openings

## Acceptance Criteria

- [x] All committed polynomial claims collected (ram_inc, rd_inc, instruction_ra, bytecode_ra, ram_ra)
- [x] Dense poly claims Lagrange-normalized correctly (zero_selector × eval)
- [x] RA claims at unified point from S7
- [x] RLC reduction produces single claim (debug_assert)
- [x] PCS open succeeds
- [ ] `cargo clippy -p jolt-zkvm` passes (blocked by pre-existing use_lt / Debug errors in other files)
