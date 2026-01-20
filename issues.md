# ZK Implementation Issues

Issues identified during review of BlindFold/ZK implementation against zk.md specification.

---

## Critical

### 1. Pedersen commitments from ZK sumcheck never verified against BlindFold witness

**Location:**
- `jolt-core/src/subprotocols/sumcheck.rs:318-324` (commitments stored)
- `jolt-core/src/zkvm/verifier.rs` (no verification of round_commitments)

**Description:** ZK sumcheck creates Pedersen commitments to `compressed_poly.coeffs_except_linear_term` per round. These commitments derive Fiat-Shamir challenges. However, BlindFold creates a SEPARATE commitment `W_bar` to the entire flattened witness vector. The individual round commitments from `ZkSumcheckProof.round_commitments` are **never verified** to match the coefficients used in `BlindFoldWitness`.

**Attack:** A malicious prover can:
1. Commit to polynomial P1 to derive challenges in ZK sumcheck
2. Use different polynomial P2 in BlindFold witness
3. BlindFold R1CS verifies P2 is internally consistent (sum check + Horner chaining)
4. But challenges were derived from P1, not P2

**Evidence:**
- `verify_blindfold` (verifier.rs:525-682) never references `stage*_sumcheck_proof.round_commitments`
- Design doc zk.md says commitments should be part of instance, but implementation uses single W_bar

**Fix:** Either:
1. Add constraints to R1CS that verify round commitments open to witness coefficients, OR
2. Restructure to use per-round commitment verification as in zk.md Part 1

---

### 2. ~~Final output claim not enforced in R1CS~~ (PARTIALLY FIXED)

**Location:** `jolt-core/src/subprotocols/blindfold/r1cs.rs:196-313`

**Status:** R1CS constraint infrastructure added. Data flow from prover needs completion.

**What was done:**
- Added `FinalOutputConfig` to `StageConfig` for configuring final output constraints
- Added `FinalOutputVariables` to track batching coefficients (public) and evaluations (witness)
- Added `add_final_output_constraint()` method that encodes `final_claim = Σⱼ αⱼ · yⱼ`
- Added `FinalOutputWitness` and updated `BlindFoldWitness` to support final output data
- Extended `ZkStageData` with `batching_coefficients` and `expected_evaluations` fields
- Added tests for single/multiple evaluation constraints

**Remaining work (TODO(#issue2) in sumcheck.rs):**
- Extend `SumcheckInstanceProver` trait to return expected evaluations after `cache_openings()`
- Populate `expected_evaluations` field in `ZkStageData` with actual values
- Wire up prover to use `with_final_output()` on stage configs and pass final output witness data

---

### 3. ZK sumcheck degree bounds not enforced

**Location:** `jolt-core/src/subprotocols/sumcheck.rs:601-629`

**Description:** `verify_transcript_only` checks `round_commitments.len() == num_rounds` but completely ignores polynomial degree bounds. The `degree_bound` parameter is not even passed to this function. The `poly_degrees` stored in proof are never verified.

**Attack:** Prover can claim higher polynomial degrees via `poly_degrees`, gaining extra freedom in BlindFold coefficient choices.

**Evidence:**
```rust
pub fn verify_transcript_only(
    &self,
    num_rounds: usize,  // no degree_bound parameter
    transcript: &mut ProofTranscript,
) -> Result<Vec<F::Challenge>, ProofVerifyError>
```

**Fix:** Add degree bound verification:
```rust
for (commitment, &degree) in self.round_commitments.iter().zip(&self.poly_degrees) {
    if degree > degree_bound {
        return Err(ProofVerifyError::InvalidInputLength(degree_bound, degree));
    }
    // ... rest of verification
}
```

---

## High

### 4. ~~verify_standard transcript ordering mismatch~~ (FIXED)

**Location:** `jolt-core/src/subprotocols/sumcheck.rs:448-459`

**Status:** Fixed - reordered to append input claims before deriving batching coefficients.

---

## Medium

### 5. ~~Missing `finalize()` call in `prove_zk`~~ (FIXED)

**Location:** `jolt-core/src/subprotocols/sumcheck.rs:326`

**Status:** Fixed - added finalize() loop matching standard prove function.

---

### 6. BlindFold initial claims weak binding

**Location:** `jolt-core/src/zkvm/verifier.rs:648-666`

**Description:** Initial claims are cross-checked between `proof.blindfold_initial_claims` and `proof.blindfold_proof.real_instance.x` - both prover-supplied values.

**Mitigation:** If initial claims are wrong, R1CS constraints fail because:
- Challenges verified to match main transcript
- First round's sum check `2*c0 + c1 + ... = initial_claim` would fail

**Status:** Indirectly enforced via R1CS, but explicit verifier-derived cross-check would strengthen formal security argument.

---

## Low

### 7. ~~Unused `jolt_stage_configs` function~~ (FIXED)

**Location:** `jolt-core/src/subprotocols/blindfold/mod.rs:95-104`

**Status:** Fixed - removed unused placeholder function.

---

## Summary

| Issue | Severity | Status |
|-------|----------|--------|
| Round commitments not verified | Critical | Fixed |
| Final output claim not in R1CS | Critical | Partially Fixed (R1CS infrastructure done, data flow pending) |
| Degree bounds not enforced | Critical | Real vulnerability |
| verify_standard transcript order | High | Fixed |
| Missing finalize() in prove_zk | Medium | Fixed |
| Initial claims weak binding | Medium | Mitigated by R1CS |
| Unused jolt_stage_configs | Low | Fixed |
