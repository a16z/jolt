# T01: Claim Types

**Status**: `[x]` Done
**Depends on**: Nothing
**Blocks**: T05 (Stage Output Types)
**Crate**: `jolt-openings`
**Estimated scope**: Small (~50 lines)

## Objective

Add lightweight claim types for inter-stage routing in the typed DAG pipeline.

## Deliverables

### 1. `VirtualEval<F>` in `jolt-openings/src/claims.rs`

```rust
/// Evaluation of a virtual polynomial — used only for inter-stage routing.
/// Virtual polys are derived from R1CS witness columns and never committed
/// to PCS. This is a zero-cost newtype for type clarity.
#[derive(Clone, Copy, Debug)]
pub struct VirtualEval<F>(pub F);
```

### 2. `CommittedEval<F>` in `jolt-openings/src/claims.rs`

```rust
/// Evaluation of a committed polynomial at a specific point.
/// Carries enough data for downstream claim reductions.
/// Does NOT carry the full evaluation table — that stays in PolynomialTables.
#[derive(Clone, Debug)]
pub struct CommittedEval<F: Field> {
    pub point: Vec<F>,
    pub eval: F,
}
```

### 3. Re-export from `jolt-openings/src/lib.rs`

Both types should be public in the crate root.

## Notes

- `VirtualEval` is `Copy` — it's just a scalar wrapper.
- `CommittedEval` is NOT `Copy` — it owns a `Vec<F>` for the point.
- The existing `ProverClaim<F>` carries a full `Vec<F>` evaluation table.
  `CommittedEval` is intentionally lighter — tables stay in `PolynomialTables`
  and are only attached for PCS opening in T14.
- No changes to existing `ProverClaim` or `VerifierClaim`.

## Acceptance Criteria

- [ ] Types compile with `cargo clippy -p jolt-openings`
- [ ] Re-exported from crate root
- [ ] No changes to existing types
