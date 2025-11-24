# Groth16 Transpilation Experiment

**Branch:** `groth16-experiment`
**Started:** 2025-11-24
**Goal:** Explore transpilation of Jolt's Stage 1 verifier to Groth16-compatible circuits

## Background

This experiment investigates the feasibility of transpiling Jolt's verification logic into Groth16 circuits for EVM verification. Based on the analysis in:
- `docs/Groth16/Groth16_Conversion_Scope.md` - Overall conversion strategy
- `docs/Groth16/Partial_Transpilation_Exploration.md` - Partial transpilation approach
- `docs/Groth16/zkLean_Infrastructure_Reuse.md` - Using zkLean for extraction

## Approach

### Phase 1: Minimal Verifier (Current)

**Objective:** Create an isolated Stage 1-only verifier to use as transpilation target

**Why Stage 1 only?**
- Stage 1 = Spartan outer sumcheck (R1CS constraint verification)
- Self-contained: ~30 constraints per cycle, no dependencies on other stages
- No Dory PCS: Avoids expensive pairing operations for initial experiment
- Smallest unit that still represents meaningful verification work

**What we're extracting:**
1. **Univariate-skip first round** (`verify_stage1_uni_skip`)
   - Lagrange polynomial interpolation
   - First round polynomial verification
   - Challenge sampling

2. **Remaining sumcheck rounds** (`OuterRemainingSumcheckVerifier`)
   - Streaming first cycle-bit round
   - Linear-time remaining rounds
   - Final R1CS evaluation check

**What we're NOT including:**
- Dory PCS opening verification (Stage 5)
- RAM verification (Twist memory checking)
- Register verification (Twist memory checking)
- Instruction lookups (Shout)
- Bytecode lookups (Shout)
- Stages 2-7

### Phase 2: Transpilation Approaches (Future)

We will experiment with multiple transpilation strategies:

**Option A: Direct zkLean → Gnark**
- Extend zkLean extractor with Gnark backend
- Generate Gnark API calls directly from MLE AST
- Pros: Reuses existing infrastructure
- Cons: May not handle all primitives efficiently

**Option B: Intermediate Representation**
- Define IR for field operations + sumcheck
- Multiple backends (Gnark, Circom, etc.)
- Pros: Flexible, future-proof
- Cons: More engineering effort

**Option C: Manual Translation**
- Hand-write Gnark circuit for Stage 1
- Use zkLean for validation/testing
- Pros: Full control, can optimize
- Cons: Maintenance burden, error-prone

## Implementation Log

### 2025-11-24: Repository Setup

**Changes:**
1. Updated `.gitignore` to exclude personal documentation
   - `Theory/`, `docs/`, markdown files
   - Temporary build artifacts

2. Pulled latest changes from main (`dbc011e2` → `4c0b621c`)
   - **Major change:** DAG structure removed, replaced with stage-based verifier
   - New files: `jolt-core/src/zkvm/prover.rs`, `jolt-core/src/zkvm/verifier.rs`
   - zkLean extractor significantly updated

3. Created `groth16-experiment` branch

**Architecture Analysis:**

Current verifier structure ([verifier.rs:187-211](jolt-core/src/zkvm/verifier.rs#L187-L211)):
```rust
fn verify_stage1(&mut self) -> Result<(), anyhow::Error> {
    // 1. Univariate-skip first round
    let spartan_outer_uni_skip_state = verify_stage1_uni_skip(
        &self.proof.stage1_uni_skip_first_round_proof,
        &self.spartan_key,
        &mut self.transcript,
    )?;

    // 2. Remaining sumcheck rounds
    let spartan_outer_remaining = OuterRemainingSumcheckVerifier::new(
        n_cycle_vars,
        &spartan_outer_uni_skip_state,
        self.spartan_key,
    );

    let _r_stage1 = BatchedSumcheck::verify(
        &self.proof.stage1_sumcheck_proof,
        vec![&spartan_outer_remaining],
        &mut self.opening_accumulator,
        &mut self.transcript,
    )?;

    Ok(())
}
```

**Key components:**
- `verify_stage1_uni_skip()` - [spartan/mod.rs](jolt-core/src/zkvm/spartan/mod.rs)
- `OuterRemainingSumcheckVerifier` - [spartan/outer.rs](jolt-core/src/zkvm/spartan/outer.rs)
- `BatchedSumcheck::verify()` - [subprotocols/sumcheck.rs](jolt-core/src/subprotocols/sumcheck.rs)

### 2025-11-24: Creating Isolated Stage 1 Verifier

**Objective:** Create `jolt-core/src/zkvm/stage1_only_verifier.rs` as standalone module

**Design decisions:**
1. **Separate module** (not editing existing verifier)
   - Keeps experiment isolated
   - Original verifier remains untouched
   - Easy to compare/test both implementations

2. **Minimal dependencies**
   - Only R1CS constraints and Spartan outer sumcheck
   - No commitment scheme dependencies
   - Self-contained proof structure

3. **Compatible interfaces**
   - Reuses existing types where possible
   - Can use same preprocessing
   - Easy to swap in tests

**Implementation complete:**
- [x] Defined `Stage1OnlyProof` struct
  - Contains only uni-skip + sumcheck proofs
  - Includes `from_full_proof()` helper for testing
- [x] Defined `Stage1OnlyPreprocessing` struct
  - Minimal: just contains `UniformSpartanKey`
  - Derived from trace length
- [x] Implemented `Stage1OnlyVerifier`
  - Two-step verification: uni-skip + remaining sumcheck
  - Compatible with existing `BatchedSumcheck::verify()`
  - Uses `OuterRemainingSumcheckVerifier` from full implementation
- [x] Module compiles successfully
- [x] Basic unit tests added (preprocessing creation, invalid trace length rejection)

**Next steps:**
- [ ] Add integration test using full prover
- [ ] Run test on simple example (fibonacci)
- [ ] Document verification algorithm in detail
- [ ] Prepare for transpilation experiments

## Mathematical Foundation

### Spartan Outer Sumcheck (Stage 1)

**Goal:** Verify R1CS constraint satisfaction over execution trace

**R1CS form:**
```
Az(x) ∘ Bz(x) - Cz(x) = 0  for all x ∈ {0,1}^n
```

In Jolt, constraints are conditional equalities:
```
if condition(x) { left(x) - right(x) = 0 }
```
So: `a(x) = condition`, `b(x) = left - right`, `c(x) = 0`

**Sumcheck statement:**
```
∑_{x ∈ {0,1}^n} eq(τ, x) · [Az(x) · Bz(x) - Cz(x)] = 0
```

**Univariate-skip first round:**
- Reduces first variable by evaluating Lagrange polynomial
- Computes `t1(Y)` over extended domain
- Multiplies by `L(τ_high, Y)` to get `s1(Y)`
- Verifier samples `r0`

**Remaining rounds:**
- Streaming first cycle-bit round (cubic from endpoints)
- Linear-time rounds for subsequent bits
- Final check: `eq(τ, r) · [Az(r) · Bz(r) - Cz(r)]`

**Complexity:**
- ~30 R1CS constraints per cycle
- Uniform constraints (same set every cycle)
- `n = log(trace_length)` sumcheck rounds

## Files Modified/Created

### New Files
- `GROTH16_EXPERIMENT.md` - This documentation
- `jolt-core/src/zkvm/stage1_only_verifier.rs` - Isolated Stage 1 verifier
  - `Stage1OnlyProof<F, ProofTranscript>` - Minimal proof structure
  - `Stage1OnlyPreprocessing<F>` - Minimal preprocessing (Spartan key only)
  - `Stage1OnlyVerifier<F, ProofTranscript>` - Verifier implementation
  - ~250 lines with documentation and tests

### Modified Files
- `.gitignore` - Added personal documentation patterns
- `jolt-core/src/zkvm/mod.rs` - Added `pub mod stage1_only_verifier;`

## References

- [Spartan Paper](https://eprint.iacr.org/2019/550.pdf) - Original Spartan SNARK
- [Jolt Paper](https://eprint.iacr.org/2023/1217.pdf) - Jolt zkVM architecture
- zkLean extractor: [zklean-extractor/](zklean-extractor/)
- Current implementation: [jolt-core/src/zkvm/spartan/outer.rs](jolt-core/src/zkvm/spartan/outer.rs)
