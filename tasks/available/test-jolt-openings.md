# test-jolt-openings: Comprehensive tests for jolt-openings

**Scope:** crates/jolt-openings/

**Depends:** impl-jolt-openings

**Verifier:** ./verifiers/scoped.sh /workdir jolt-openings

**Context:**

Write comprehensive tests for the `jolt-openings` crate. The implementation task includes basic inline tests. This task adds property-based tests for the accumulator and reduction logic, edge cases, and fuzz targets.

**Do not modify source logic — test-only changes.**

Use the `MockCommitmentScheme` (included in jolt-openings as a test-only module) for all tests that don't need a real PCS.

### Test categories

#### 1. Trait contract tests

Verify that `MockCommitmentScheme` satisfies the trait contracts:
- `commit → prove → verify` succeeds for a random polynomial and point
- `verify` fails when evaluation is wrong
- `verify` fails when point is wrong
- `verify` fails when commitment doesn't match

#### 2. Accumulator property tests (proptest)

**ProverOpeningAccumulator:**
- Accumulate `k` random (poly, point, eval) triples → `reduce_and_prove` produces valid proofs that the verifier accepts
- Accumulate polys at the *same* point → they get batched into a single proof
- Accumulate polys at *different* points → separate proofs per point
- Empty accumulator → `reduce_and_prove` returns empty vec

**VerifierOpeningAccumulator:**
- Mirror of prover tests: same accumulation, `reduce_and_verify` succeeds
- Tamper with one eval → `reduce_and_verify` fails

#### 3. RLC reduction tests

- Two polynomials at the same point: combined poly = `p1 + rho * p2`, verify opening of combined poly
- Three polynomials: `p1 + rho * p2 + rho^2 * p3`
- Single polynomial: no combining needed, verify it passes through

#### 4. HomomorphicCommitmentScheme tests

- `combine_commitments([C], [1])` == `C`
- `combine_commitments([C1, C2], [a, b])` == `a*C1 + b*C2`
- `batch_prove → batch_verify` round-trip for multiple polys at different points

#### 5. Edge cases

- Polynomial of size 1 (0 variables)
- Evaluation at the zero vector
- All-zero polynomial
- Very large number of accumulated openings (100+)

#### 6. Fuzz targets

**`fuzz_verifier`:** Generate arbitrary bytes, deserialize as a proof, feed to verifier. Must not panic (should return Err).

**Acceptance:**

- Accumulator tested with 1, 2, 10, 100 accumulated openings
- RLC reduction correctness verified via proptest
- Tampered proofs/evals reliably rejected
- Edge cases covered
- Fuzz targets compile and run without crashes
- All tests pass
- No modifications to non-test source code
