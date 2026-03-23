# T16: Verifier DAG

**Status**: `[ ]` Not started
**Depends on**: T06 (Input Claim Formulas), T15 (Prove Orchestrator)
**Blocks**: T17 (E2E muldiv)
**Crate**: `jolt-verifier`
**Estimated scope**: Large (~400 lines)

## Objective

Rewrite jolt-verifier to use the same typed DAG structure as the prover.
Same stage output types, same input_claim formulas, but verifying instead
of proving.

## Deliverables

### 1. `verify()` function (same DAG shape)

```rust
pub fn verify<PCS>(
    proof: &JoltProof<Fr>,
    config: &ProverConfig,
    transcript: &mut impl Transcript<Challenge = Fr>,
) -> Result<(), VerifyError>
where
    PCS: CommitmentScheme<Field = Fr>,
{
    let s1_v = verify_spartan(&proof.s1, config, transcript)?;
    let s2_v = verify_stage2(&s1_v, &proof.s2, config, transcript)?;
    let s3_v = verify_stage3(&s1_v, &s2_v, &proof.s3, config, transcript)?;
    let s4_v = verify_stage4(&s2_v, &s3_v, &proof.s4, config, transcript)?;
    let s5_v = verify_stage5(&s2_v, &s4_v, &proof.s5, config, transcript)?;
    let s6_v = verify_stage6(&s2_v, &s4_v, &s5_v, &proof.s6, config, transcript)?;
    let s7_v = verify_stage7(&s5_v, &s6_v, &proof.s7, config, transcript)?;
    verify_opening::<PCS>(&s6_v, &s7_v, &proof, transcript)?;
    Ok(())
}
```

### 2. `verify_stageN()` functions

Each verifier stage function:
1. Squeezes same challenges from transcript
2. Computes input_claim using shared formulas (T06) + prior stage outputs
3. Verifies sumcheck round polynomials
4. Reads evaluations from the proof's stage output
5. Returns the same typed output struct

The key difference from prover: evaluations come from the proof, not from
evaluating polynomial tables.

### 3. Opening verification

```rust
fn verify_opening<PCS>(
    s6: &Stage6Output<Fr>,
    s7: &Stage7Output<Fr>,
    proof: &JoltProof<Fr>,
    transcript: &mut impl Transcript,
) -> Result<(), VerifyError>
```

Reconstructs the RLC claim from proof data and verifies the PCS proof.

### 4. Delete old verifier

Replace:
- `StageDescriptor` (partially — may keep for BlindFold)
- `DescriptorSource` trait
- `EagerVerifierSource`
- Config-driven `verify()` with stage loop

## Design Notes

- **Same output types**: Prover and verifier use the SAME `Stage2Output<F>`,
  etc. The prover fills them by evaluating polynomials. The verifier fills
  them from proof data. The types don't care.
- **Shared formulas**: Input claim functions from T06 are called identically
  by both prover and verifier. This guarantees Fiat-Shamir consistency.
- **IR downstream**: The claim definitions from jolt-ir drive the verifier's
  output formula check (`eq(point) × g(openings, challenges) == final_eval`).

## Acceptance Criteria

- [ ] `verify()` compiles with same DAG shape as `prove()`
- [ ] Uses shared input_claim formulas from T06
- [ ] Verifies each stage's sumcheck proof
- [ ] Verifies PCS opening proof
- [ ] `cargo clippy -p jolt-verifier` passes
