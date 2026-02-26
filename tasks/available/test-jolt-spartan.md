# test-jolt-spartan: Comprehensive tests for jolt-spartan

**Scope:** crates/jolt-spartan/

**Depends:** impl-jolt-spartan

**Verifier:** ./verifiers/scoped.sh /workdir jolt-spartan

**Context:**

Write comprehensive tests for the `jolt-spartan` crate. Test the R1CS abstraction, Spartan prover/verifier, and the univariate skip optimization.

**Do not modify source logic — test-only changes.**

Use `MockCommitmentScheme` from `jolt-openings` for tests that don't need Dory.

### Test categories

#### 1. R1CS construction and satisfaction

Build small R1CS instances by hand and verify:

**Trivial R1CS:** $x \cdot x = x$ (boolean constraint, satisfied by x=0 or x=1)
- Verify `multiply_witness` produces correct Az, Bz, Cz
- Verify `Az ∘ Bz == Cz` for satisfying witnesses
- Verify `Az ∘ Bz != Cz` for non-satisfying witnesses

**Arithmetic R1CS:** $x \cdot y = z$ (multiplication gate)
- Satisfied by (3, 4, 12)
- Not satisfied by (3, 4, 13)

**Multi-constraint R1CS:** $x^2 + y^2 = z$ (two multiplication gates + addition)

#### 2. Spartan prove → verify round-trip

- Trivial R1CS with MockPCS: prove → verify succeeds
- Arithmetic R1CS with MockPCS: prove → verify succeeds
- Multi-constraint R1CS: prove → verify succeeds
- Non-satisfying witness → prove still runs but verify fails (or prove returns error)

#### 3. Soundness

- Valid proof but modify one field in `SpartanProof` → verify fails
- Proof from one R1CS instance, verify against different R1CS → fails
- Proof with wrong SpartanKey → fails

#### 4. UniformR1CS

If `UniformR1CS` has special structure-exploiting behavior:
- Construct a uniform R1CS and verify it produces the same results as the general R1CS trait for the same constraint system

#### 5. Univariate skip

- Prove with `FirstRoundStrategy::Standard` and `FirstRoundStrategy::UnivariateSkip` → both produce valid proofs
- Verify both proofs → both succeed
- (Bonus) Verify that UniSkip proof is the same as Standard proof (deterministic transcript means same challenges)

#### 6. Property-based tests (proptest)

- Generate random satisfiable R1CS instances (random A, B matrices, compute C = A∘B): prove → verify succeeds
- Generate unsatisfiable witness for a valid R1CS: verify fails

#### 7. Integration with jolt-dory

- Same tests as #2 but with `DoryScheme` instead of MockPCS
- Verifies the full stack: Spartan + sumcheck + Dory opening proofs

#### 8. SpartanKey consistency

- `SpartanKey::from_r1cs` is deterministic (same R1CS → same key)
- Key works across multiple prove/verify cycles

**Acceptance:**

- Hand-built R1CS instances tested (boolean, multiplication, multi-constraint)
- Prove → verify round-trip with MockPCS and DoryScheme
- Soundness: modified proofs rejected, wrong keys rejected
- Univariate skip produces valid proofs
- Property-based tests for random R1CS instances
- SpartanKey determinism verified
- All tests pass
- No modifications to non-test source code
