# ZK Implementation Issues

Issues identified during review of BlindFold/ZK implementation against zk.md specification.

---

## Critical

### 1. ~~Pedersen commitments from ZK sumcheck never verified against BlindFold witness~~ (FIXED)

**Status:** Fixed - `round_commitments` are now part of `RelaxedR1CSInstance` and verified in BlindFold verification (`protocol.rs:299-317`). Each round commitment is checked against the folded witness coefficients and blindings.

---

### 2. ~~Final output claim not enforced in R1CS~~ (FIXED)

**Status:** Fixed - Complete infrastructure for output constraints:
- `OutputClaimConstraint` collected from each sumcheck instance
- Batching via `batch_output_constraints()` in verifier
- Stage configs populated with `final_output` constraints
- BlindFold R1CS encodes `final_claim = Σⱼ αⱼ · yⱼ` constraints
- Challenge values and opening values properly wired through

---

### 3. ~~ZK sumcheck degree bounds not enforced~~ (FIXED)

**Status:** Fixed - `verify_transcript_only` now takes `degree_bound` parameter and enforces it:
```rust
for &degree in &self.poly_degrees {
    if degree > degree_bound {
        return Err(ProofVerifyError::InvalidInputLength(degree_bound, degree));
    }
}
```

---

## High

### 4. ~~verify_standard transcript ordering mismatch~~ (FIXED)

**Status:** Fixed - reordered to append input claims before deriving batching coefficients.

---

## Medium

### 5. ~~Missing `finalize()` call in `prove_zk`~~ (FIXED)

**Status:** Fixed - added finalize() loop matching standard prove function.

---

### 6. BlindFold initial claims weak binding

**Location:** `jolt-core/src/zkvm/verifier.rs`

**Description:** Initial claims are cross-checked between prover-supplied values.

**Status:** Mitigated by R1CS - if initial claims are wrong, first round's sum check constraint `2*c0 + c1 + ... = initial_claim` fails. Challenges verified to match main transcript.

---

## Low

### 7. ~~Unused `jolt_stage_configs` function~~ (FIXED)

**Status:** Fixed - removed unused placeholder function.

---

## Summary

| Issue | Severity | Status |
|-------|----------|--------|
| Round commitments not verified | Critical | **Fixed** |
| Final output claim not in R1CS | Critical | **Fixed** |
| Degree bounds not enforced | Critical | **Fixed** |
| verify_standard transcript order | High | **Fixed** |
| Missing finalize() in prove_zk | Medium | **Fixed** |
| Initial claims weak binding | Medium | Mitigated by R1CS |
| Unused jolt_stage_configs | Low | **Fixed** |

---

## ZK Status

**All critical issues resolved.** The ZK implementation now includes:

1. **Hiding sumcheck**: Round polynomials committed via Pedersen commitments
2. **ZK-Dory**: Evaluation proofs with `y_com`/`y_blinding` for hiding claimed values
3. **BlindFold R1CS**: Encodes all verifier checks in O(log n) constraints
4. **Output constraints**: Final claims bound to polynomial openings via batched constraints
5. **Evaluation commitment verification**: `y_com` verified against BlindFold witness values
6. **Degree bound enforcement**: Polynomial degrees verified against bounds

The `muldiv_e2e_dory` test passes end-to-end with ZK mode enabled.
