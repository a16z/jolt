# T17: E2E muldiv Test

**Status**: `[ ]` Not started
**Depends on**: T15 (Prove Orchestrator), T16 (Verifier DAG)
**Blocks**: T18 (Cleanup)
**Crate**: `jolt-zkvm`
**Estimated scope**: Medium (~100 lines test code, but debugging may be substantial)

## Objective

Run the `muldiv` guest program end-to-end through the new typed DAG pipeline
and verify the proof. This is the canonical correctness check that validates
the entire rewrite.

## Deliverables

### 1. E2E test

```rust
#[test]
fn test_muldiv_e2e() {
    // 1. Compile muldiv guest ELF
    // 2. Trace execution
    // 3. Generate witness → PolynomialTables
    // 4. prove() → JoltProof
    // 5. verify() → Ok(())
}
```

### 2. Run in both modes

```bash
# Standard mode
cargo nextest run -p jolt-zkvm muldiv --cargo-quiet --features host

# ZK mode (deferred — after BlindFold integration)
# cargo nextest run -p jolt-zkvm muldiv --cargo-quiet --features host,zk
```

## What This Tests

- All 7 sumcheck stages produce valid proofs
- Fiat-Shamir transcript consistency between prover and verifier
- Input claim formulas are correct (any mismatch → transcript divergence)
- Claim reduction chain works (IncCR → HammingWeightCR → unified point)
- PCS opening proof verifies at the unified point
- Lagrange normalization for dense polys is correct

## Debugging Strategy

If the test fails:
1. **Transcript mismatch**: Add transcript checkpoints after each stage.
   Compare prover vs verifier challenge values at each stage boundary.
2. **Wrong input_claim**: Print the input_claim value for each instance.
   Compare against jolt-core's values for the same trace.
3. **Wrong evaluations**: Compare polynomial evaluations at challenge points
   against brute-force evaluation.
4. **PCS failure**: Verify that the unified point is constructed correctly.
   Check Lagrange factors.

## Reference

- jolt-core muldiv test: `cargo nextest run -p jolt-core muldiv --features host`
- Guest program: somewhere in examples/guests/

## Acceptance Criteria

- [ ] muldiv prove + verify succeeds
- [ ] Proof size comparable to jolt-core (same structure)
- [ ] No panics, no assertion failures
- [ ] `cargo nextest run -p jolt-zkvm muldiv --cargo-quiet --features host`
