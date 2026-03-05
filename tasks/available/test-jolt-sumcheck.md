# test-jolt-sumcheck: Comprehensive tests for jolt-sumcheck

**Scope:** crates/jolt-sumcheck/

**Depends:** impl-jolt-sumcheck

**Verifier:** ./verifiers/scoped.sh /workdir jolt-sumcheck

**Context:**

Write comprehensive tests for the `jolt-sumcheck` crate. This is one of the most critical testing tasks — the RFC (finding 8) specifically calls out the lack of sumcheck testing as a major problem. This task must deliver systematic, thorough sumcheck tests.

**Do not modify source logic — test-only changes.**

### Test categories

#### 1. Completeness tests (honest prover always passes)

Implement a simple `SumcheckInstanceProver` for a concrete polynomial (e.g., product of two multilinear polynomials) and verify:

- Single-instance: `prove → verify` succeeds
- Claimed sum matches actual sum over the hypercube
- Challenge vector has correct length (== num_vars)
- Final evaluation matches polynomial evaluation at challenge vector

Test with:
- 1-variable polynomial
- 2-variable polynomial
- 5-variable polynomial
- 10-variable polynomial

#### 2. Soundness tests (cheating prover fails)

Create a deliberately incorrect `SumcheckInstanceProver` that:
- Returns wrong round polynomial for one round → verify fails
- Returns correct round polynomials but wrong claimed sum → verify fails
- Returns round polynomial with degree exceeding bound → verify fails

Also test:
- Modify one coefficient of one round polynomial in a valid proof → verify fails
- Swap two round polynomials → verify fails

#### 3. Property-based tests (proptest)

**Completeness property:**
For random multilinear polynomial `f` with `n` vars:
1. Compute `claimed_sum = sum_{x in {0,1}^n} f(x)`
2. Create a `SumcheckInstanceProver` for `f`
3. `prove(claim, witness, transcript) → proof`
4. `verify(claim, proof, transcript) → Ok`

Run with `n` in 1..8, random field elements.

**Soundness property:**
For random `f` with `n` vars:
1. Compute `wrong_sum = claimed_sum + 1`
2. `prove(wrong_claim, witness, transcript) → proof`
3. `verify(wrong_claim, proof, transcript) → Err` (with high probability)

**Transcript consistency:**
Prover and verifier use the same transcript state. Verify that running prove then verify on independent transcript copies produces the same challenge sequence.

#### 4. Batched sumcheck tests

- Two instances with same `num_vars`: batched prove → verify succeeds
- Two instances with different `num_vars` (e.g., 3 and 5): power-of-2 scaling works
- Three instances: all claims verified
- One correct + one incorrect instance → batch verify fails

#### 5. Streaming sumcheck tests

If `StreamingSumcheckProver` is implemented:
- Streaming prover produces identical proof to non-streaming prover for the same polynomial
- Works correctly with chunk sizes that don't evenly divide the evaluation count

#### 6. Edge cases

- 0-variable polynomial (sum over {0,1}^0 = single evaluation)
- Polynomial that is identically zero
- Polynomial with degree-0 round polynomials (all round polys are constants)
- Very large `num_vars` (16+) — verify it doesn't OOM or timeout

#### 7. Fuzz targets

**`fuzz_verifier_soundness`:** Generate random `SumcheckProof` (random round polynomials), random `SumcheckClaim`, feed to verifier. Should almost always reject (soundness). Must not panic.

#### 8. Concrete test vectors (regression)

Hardcode a known polynomial, its true sum, and expected round polynomials. Verify the prover produces exactly these round polynomials and the verifier accepts.

Example: $f(x_1, x_2) = x_1 \cdot x_2$ over $\{0,1\}^2$.
- True sum = $f(0,0) + f(0,1) + f(1,0) + f(1,1) = 0 + 0 + 0 + 1 = 1$
- Round 1: $s_1(X) = \sum_{x_2} f(X, x_2) = X \cdot 0 + X \cdot 1 = X$, so $s_1 = [0, 1]$ (coefficients)
- Verify $s_1(0) + s_1(1) = 0 + 1 = 1$ ✓

### Current Progress

The `jolt-sumcheck` crate already has **23 test functions** (21 unit + 2 integration):

| Category | Status | Existing tests |
|----------|--------|----------------|
| Completeness | Done | `plain_sum_prove_verify`, `eq_product_prove_verify`, `single_variable`, `deterministic_proofs` |
| Soundness | Done | `wrong_claimed_sum_fails`, `wrong_round_count_is_rejected`, `degree_bound_exceeded_is_rejected`, `tampered_round_coefficient_rejected`, `transcript_label_mismatch_fails` |
| Batched | Done | `batched_prove_verify`, `batched_single_claim_matches_unbatched`, `batched_three_claims`, `batched_wrong_claim_fails`, `batched_different_num_vars`, `batched_mixed_degree_and_num_vars`, `batched_challenge_slicing` |
| Streaming | Done | `streaming_prover_produces_correct_rounds`, `streaming_prover_multi_chunk` |
| Edge cases | Partial | `zero_claimed_sum`, `degree_3_triple_product` |
| Multi-backend | Done | `keccak_transcript_prove_verify`, `blake2b_and_keccak_both_verify` (integration) |
| Property-based (proptest) | Not started | |
| Fuzz targets | Not started | |
| Concrete test vectors | Not started | |

**Remaining work:**
- Property-based tests with `proptest`
- Fuzz target (`fuzz_verifier_soundness`)
- Hardcoded concrete test vectors (regression)

**Note:** The code samples above reference `SumcheckInstanceProver` — the actual trait is `SumcheckWitness`. `SumcheckClaim` fields and `SumcheckProver::prove` signature may also differ. Update samples if using this task as a reference.

**Acceptance:**

- Completeness tested for 1,2,5,10-variable polynomials
- Soundness tested with 5+ different cheating strategies
- At least 8 proptest properties
- Batched sumcheck tested with 2,3 instances and mismatched variable counts
- Streaming sumcheck basic tests (if implemented)
- Concrete test vectors for known polynomials
- Fuzz targets compile and run without crashes
- All tests pass
- No modifications to non-test source code
