# test-jolt-dory: Comprehensive tests for jolt-dory

**Scope:** crates/jolt-dory/

**Depends:** impl-jolt-dory

**Verifier:** ./verifiers/scoped.sh /workdir jolt-dory

**Context:**

Write comprehensive tests for the `jolt-dory` crate. This tests the Dory commitment scheme wrapper against the `CommitmentScheme`, `HomomorphicCommitmentScheme`, and `StreamingCommitmentScheme` trait contracts.

**Do not modify source logic — test-only changes.**

### Test categories

#### 1. Basic round-trip (CommitmentScheme)

- Commit to a random polynomial, prove opening at a random point, verify → succeeds
- Commit to polynomial, prove with wrong evaluation → verify fails
- Commit to polynomial, prove with wrong point → verify fails
- Commit to polynomial A, try to verify with commitment to polynomial B → fails

Test with polynomial sizes: 2^4, 2^8, 2^12.

#### 2. Homomorphic operations (HomomorphicCommitmentScheme)

- `combine_commitments` with trivial scalars [1, 0] returns first commitment
- `combine_commitments` correctness: `commit(a*p1 + b*p2) == combine([C1, C2], [a, b])`
- `batch_prove → batch_verify` for 2 polynomials at same point
- `batch_prove → batch_verify` for 3 polynomials at different points
- Batch verify with one tampered evaluation → fails

#### 3. Streaming (StreamingCommitmentScheme)

- Streaming commit matches non-streaming commit for same polynomial
- Streaming in one chunk == non-streaming
- Streaming in many small chunks == non-streaming
- Streaming with varying chunk sizes produces same commitment

#### 4. Instance-local parameters

- Two `DoryScheme` instances with different `DoryParams` produce different setups
- Operations on one instance don't affect the other (no global state leakage)
- Verify with mismatched setup params → fails

#### 5. Serde round-trip

- `DoryCommitment`: serialize → deserialize → compare
- `DoryProof`: serialize → deserialize → verify still works
- `DoryVerifierSetup`: serialize → deserialize → verify still works
- `DoryParams`: serialize → deserialize → compare

#### 6. Property-based tests (proptest)

- For random polynomial and random point: `commit → prove → verify` succeeds
- For random polynomial, random point, and random (wrong) eval: `commit → prove → verify` fails

#### 7. Edge cases

- Polynomial with 1 evaluation (0 variables)
- Polynomial with all-zero evaluations
- Polynomial with all-one evaluations
- Maximum supported polynomial size (depends on DoryParams)

**Acceptance:**

- CommitmentScheme contract fully tested (correct and incorrect cases)
- HomomorphicCommitmentScheme batch operations tested
- StreamingCommitmentScheme consistency with non-streaming verified
- Instance isolation verified (no global state)
- Serde round-trip for all public wrapper types
- Property-based tests for commit-prove-verify
- Edge cases covered
- All tests pass
- No modifications to non-test source code
