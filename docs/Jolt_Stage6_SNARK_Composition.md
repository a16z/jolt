# Jolt Stage 6: SNARK Composition via Mixed Polynomial Commitment Schemes

> **Status**: This document reflects the corrected understanding of Jolt's SNARK composition architecture as of PR #975.
>
> **Supersedes**: Earlier documents that framed this as "two-layer recursion" or "Layer 1 → Layer 2" architecture. See deprecation notices in:
> - `Jolt_Verification_Challenge_and_Recursion_Approach.md`
> - `Jolt_SNARK_Composition_Implementation.md`
> - `SNARK_Recursion_Overview.md`

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [The Architecture](#2-the-architecture)
3. [Where Exponentiation Steps Come From](#3-where-exponentiation-steps-come-from)
4. [Stage 6 Protocol Specification](#4-stage-6-protocol-specification)
5. [Why This Terminates (No Stage 7)](#5-why-this-terminates-no-stage-7)
6. [What We Got Wrong](#6-what-we-got-wrong)
7. [Key Takeaways](#7-key-takeaways)

---

## 1. Introduction

### 1.1 The Problem

Jolt verification requires computing 109 $\mathbb{G}_T$ exponentiations (for typical programs with $N = 2^{16}$ cycles):
- **29 exponentiations** from Stage 5 RLC (random linear combination of commitments)
- **80 exponentiations** from main Dory opening ($5 \times \log_2 N$)
- **Cost**: Each exponentiation costs ~10M RISC-V cycles
- **Total**: $109 \times 10\text{M} = 1.09\text{B cycles}$ (85% of total 1.28B cycle verification cost)

**Why this is expensive**:
- Computing $g^x$ in $\mathbb{G}_T$ requires 254 cyclotomic squarings + ~127 multiplications in $\mathbb{F}_{q^{12}}$
- Each $\mathbb{F}_q$ operation costs ~900 RISC-V cycles (256-bit modular arithmetic)
- No hardware acceleration or precompiles exist for $\mathbb{G}_T$ exponentiations

### 1.2 The Solution: Witness Augmentation

**The fundamental problem**: Computing $g^x$ where $g \in \mathbb{G}_T$ (a length-12 vector of field elements over BN254 base field) is **NOT a low-degree function**.

> **From cryptographic foundations**: "Computing a^b where a is a length-12 vector of field elements (over the base field of BN254) cannot be proved 'in one go'" — SNARKs require low-degree polynomial constraints to use sum-check efficiently. Direct exponentiation doesn't fit this model.

**The solution - witness augmentation**:

Instead of having the verifier compute exponentiations directly, the prover:

1. **Breaks each exponentiation into square-and-multiply steps**:
   ```
   r₀ = 1
   r₁ = r₀² · base^{b₀}
   r₂ = r₁² · base^{b₁}
   ...
   r₂₅₄ = result
   ```

2. **Each step IS low-degree**: The constraint $r_{j+1} = r_j^2 \cdot (1 + b_j(g - 1))$ is:
   - Quadratic in $r_j$
   - Linear in base $g$
   - **Can be efficiently proven with sum-check**

3. **Commits all intermediate values**:
   - 109 exponentiations × 254 steps each = **27,686 intermediate $\mathbb{G}_T$ values**
   - Each $\mathbb{G}_T$ element = 12 $\mathbb{F}_q$ components
   - Total witness size: **~9.1 MB** of field element data
   - Committed using **Hyrax over Grumpkin** (succinctly, not sent in the clear)

4. **Proves correctness via batched sum-check**:
   - Single sum-check proves all constraint equations simultaneously
   - Verifier checks commitments match the constraint equations

**Why Hyrax commitments instead of sending witness data in the clear?**

> **From the Jolt team**: "This extra data is succinctly committed with Hyrax instead of sent in the clear because there's quite a lot of it, and the verifier would lose more than it gains if it had to read it all explicitly."

**With Hyrax commitments**:
- **Small proof size**: Only commitments sent (~5 KB instead of 9.1 MB)
- **Efficient verification**: Sublinear in witness size (O(√N) for Hyrax) = ~30M cycles
- **Cryptographic binding**: Commitments ensure witness cannot be changed after commitment
- **Net savings**: 56M cycles total verification (30M MSMs + 26M sum-check) vs 1.09B cycles computing exponentiations directly

**Terminology clarification**: This is **witness augmentation**, not recursion. The prover augments the proof with extra committed data that lets the verifier replace expensive operations (G_T exponentiations) with cheap constraint checks (batched sum-check + commitment verification).

**Not two proof systems**: This is **not** a second Jolt instance proving "I verified the first proof." It's a **single Jolt proof** with two commitment schemes:
- **Main trace** (Stages 1-5): Dory over BN254 (standard Jolt)
- **Witness data** (Stage 6): Hyrax over Grumpkin (proves exponentiation hints)

### 1.3 What Is "SNARK Composition"?

**Definition**: Using multiple proof systems together in a single proof, each optimized for different data types.

In Jolt's case:
- **Dory/BN254**: For the main execution trace (R1CS, Twist, Shout sumchecks)
- **Hyrax/Grumpkin**: For auxiliary witness data (exponentiation intermediate steps)

**Why "composition" not "recursion"**:
- **Recursion**: Proving "I ran a verifier correctly" (verifier becomes the computation)
- **Composition**: Proving different aspects of the same computation with different schemes

**Key distinction**:
- Recursion: Nested structure (Proof₂ proves "I verified Proof₁")
- Composition: Parallel structure (Proof has two parts: main trace + auxiliary witnesses)

### 1.4 Intuitive Analogy

**Traditional approach** (without Stage 6):
- **Prover**: "The result is X"
- **Verifier**: "Let me compute it myself... [1.28B cycles later] ...yes, it's X"

**Witness-augmented approach** (with Stage 6):
- **Prover**: "The result is X, and here's my work showing each step (committed via Hyrax)"
- **Verifier**: "Let me check your work is consistent... [56M cycles later] ...yes, your steps are valid"

The verifier trusts the steps because:
1. **Cryptographically committed**: Can't be changed after the fact
2. **Low-degree constraints**: Each step provably follows from the previous one
3. **Final step produces claimed result**: If all constraints hold, $r_{254} = g^x$ is guaranteed by algorithm structure

---

## 2. The Architecture

### 2.1 The Complete Jolt DAG (with Stage 6)

```
┌─────────────────────────────────────────────────────────────┐
│ Stages 1-4: Standard Jolt Sumchecks                        │
├─────────────────────────────────────────────────────────────┤
│ • Stage 1: Spartan outer, Twist products, Shout read      │
│ • Stage 2: Spartan product, Twist evaluations              │
│ • Stage 3: Spartan evaluation, Twist final                 │
│ • Stage 4: Final sumchecks before opening                  │
│                                                             │
│ Commitment: All witness polynomials via Dory/BN254         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Batched Opening Proof (WITH HINTS)                │
├─────────────────────────────────────────────────────────────┤
│ Standard Dory opening, but modified:                        │
│                                                             │
│ 1. RLC (Random Linear Combination):                        │
│    Combine 29 committed polynomials:                       │
│    C_combined = C₁^γ₁ · C₂^γ₂ · ... · C₂₉^γ₂₉             │
│    → Requires 29 G_T exponentiations                       │
│    → PROVER captures ExponentiationSteps during this       │
│    → VERIFIER accepts results as hints (doesn't compute)   │
│                                                             │
│ 2. Main Dory Opening:                                       │
│    Verify C_combined via log(N) rounds                     │
│    → Requires 5×log₂(N) G_T exponentiations                │
│    → PROVER captures ExponentiationSteps during this       │
│    → VERIFIER accepts results as hints                     │
│                                                             │
│ Captured: 109 ExponentiationSteps structs                  │
│   (base, exponent, r₀, r₁, ..., r₂₅₄, result)             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: SNARK Composition (Prove Hints Correct)           │
├─────────────────────────────────────────────────────────────┤
│ Input: 109 ExponentiationSteps from Stage 5                │
│                                                             │
│ For each exponentiation (base g, exponent x):               │
│   Witness data: r₀, r₁, ..., r₂₅₄ (intermediate steps)    │
│   Constraint: r_{j+1} = r_j² · (1 + b_j(g - 1))            │
│                                                             │
│ Protocol:                                                   │
│ 1. Commit to witness polynomials via Hyrax/Grumpkin:       │
│    • rho polynomials (intermediate r_j values)             │
│    • quotient polynomials (constraint checking)            │
│    • base polynomials (g values)                           │
│    → Hyrax commitment: Vector of Grumpkin group elements   │
│                                                             │
│ 2. ExpSumcheck (batched):                                   │
│    Verify all 109 exponentiations satisfy constraints      │
│    → Reduces to polynomial evaluation claims               │
│                                                             │
│ 3. Hyrax opening proof:                                     │
│    Prove claimed evaluations are correct                   │
│    → Uses MSMs over Grumpkin (no G_T exponentiations!)     │
│                                                             │
│ Output: RecursionProof<F, ProofTranscript, RATIO>          │
│   - commitments: ExpCommitments (Hyrax)                    │
│   - sumcheck_proof: SumcheckInstanceProof                  │
│   - r_sumcheck: Challenge vector                           │
│   - hyrax_proof: HyraxOpeningProof                         │
│   - openings: Evaluation claims                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Verification (Sequential)                                   │
├─────────────────────────────────────────────────────────────┤
│ 1. Verify Stages 1-4 (standard sumchecks): ~20M cycles    │
│ 2. Verify Stage 5 (Dory with hints): ~10M cycles          │
│    • Use hinted G_T exp results (don't compute them)       │
│ 3. Verify Stage 6 (Hyrax + ExpSumcheck): ~10M cycles      │
│    • Verify ExpSumcheck constraints                        │
│    • Verify Hyrax openings (MSMs only)                     │
│                                                             │
│ Total: ~40M cycles (vs 1.28B for direct computation)      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Key Architectural Points

**One proof, two PCS schemes**:
- The final proof contains both Dory commitments (main trace) and Hyrax commitments (witness data)
- Verifier runs sequentially: Stages 1-5, then Stage 6
- Both parts needed for soundness: can't skip Stage 6

**Where does Stage 6 fit in the DAG?**:
- Stage 6 is **appended** to the standard 5-stage Jolt DAG
- It's not a separate proof system; it's an extension of Stage 5
- The JoltDAG::prove method includes Stage 6 as a final step (when `feature = "recursion"`)

**Verification is sequential**:
```rust
// In jolt_dag.rs verify()
// Stage 5: Verify Dory opening (lines 540-547)
state_manager.reduce_and_verify(
    &preprocessing.generators,
    &mut commitments_map,
    batched_opening_proof,
    &mut *transcript.borrow_mut(),
)?;

// Stage 6: Verify SNARK Composition (lines 549-604)
#[cfg(feature = "recursion")]
{
    snark_composition_verify::<Fq, ProofTranscript, 1>(
        sz_proof_fq,
        &mut *transcript.borrow_mut(),
        &hyrax_generators,
    )?;
}
```

**Not two Jolt instances**:
- There is NO "Layer 2 trace" of 320M cycles
- There is NO "verifier compiled to RISC-V bytecode"
- Stage 6 proves a specific property: "These 109 exponentiations are correct"

---

## 3. Where Exponentiation Steps Come From

### 3.1 The Capture Mechanism

During Stage 5 proving, the prover captures `ExponentiationSteps` at two locations:

#### Location 1: Homomorphic Commitment Combining (RLC)

**File**: `jolt-core/src/poly/opening_proof.rs:1007-1014`

```rust
let (_joint_commitment, hint_with_steps) =
    DoryCommitmentScheme::precompute_combined_commitment(
        dory_commitments_slice,
        dory_coeffs,
    );

// Add the dory verifier steps witness for stage 6
self.extend_recursion_ops(hint_with_steps.exponentiation_steps.clone());
```

**What happens**:
- Creating combined commitment $C_{\text{combined}} = \prod_{i=1}^{29} C_i^{\gamma_i}$ requires 29 G_T exponentiations
- Each exponentiation $C_i^{\gamma_i}$ is computed via `scale_with_steps()` (line 1280 in dory.rs)
- Returns both the result AND the intermediate steps
- Steps captured and stored in `ProverOpeningAccumulator`

#### Location 2: Main Dory Opening Proof

**File**: `jolt-core/src/poly/opening_proof.rs:1031-1057`

```rust
#[cfg(feature = "recursion")]
let joint_opening_proof = {
    // ...
    let (proof, auxiliary_data) =
        prove_fn(pcs_setup, &joint_poly, &r_sumcheck, hint, transcript);

    // Extract exponentiation steps from auxiliary data
    if let Some(ref steps) = auxiliary_data.full_exponentiation_steps {
        self.extend_recursion_ops(steps.clone());
    }

    proof
};
```

**What happens**:
- Main Dory opening proof requires $5 \times \log_2 N$ exponentiations (80 for $N = 2^{16}$)
- `prove_with_auxiliary()` computes exponentiations using `scale_with_steps()`
- Returns proof AND auxiliary data containing all steps
- Steps appended to the collection from Location 1

### 3.2 The `scale_with_steps()` Method

**File**: `jolt-core/src/poly/commitment/dory.rs:274-312`

```rust
#[cfg(feature = "recursion")]
fn scale_with_steps(
    &self,
    k: &<Self as dory::arithmetic::Group>::Scalar,
) -> (Self, ExponentiationSteps) {
    // Cast self.0 to Fq12 and k.0 to Fr
    let fq12_val = /* ... */;
    let scalar_fr = /* ... */;

    // Compute g^x AND capture all intermediate steps
    let steps = ExponentiationSteps::new(fq12_val, scalar_fr);

    // steps contains:
    // - base: g (the Fq12 element)
    // - exponent: x (the Fr scalar)
    // - bits: [b_253, ..., b_1, b_0] (binary decomposition)
    // - rho_mles: [r_0, r_1, ..., r_254] (intermediate values)
    // - quotient_mles: constraints satisfaction witnesses
    // - result: g^x (the final value)

    (Self(result_as_target), steps)
}
```

### 3.3 The `ExponentiationSteps` Structure

**File**: `jolt_optimizations` crate (external)

```rust
pub struct ExponentiationSteps {
    pub base: Fq12,                      // g (tower field element)
    pub exponent: Fr,                    // x (scalar)
    pub bits: Vec<bool>,                 // [b_253, ..., b_0] (254 bits)
    pub rho_mles: Vec<Vec<Fq>>,         // r_j values as polynomial coefficients
    pub quotient_mles: Vec<Vec<Fq>>,    // Quotient polynomials for constraints
    pub result: Fq12,                    // g^x (final result)
}
```

**What it represents**: Complete witness for the square-and-multiply algorithm:
- For each bit $b_j$ of exponent (j = 0 to 253):
  - $r_0 = 1$ (initialization)
  - $r_{j+1} = r_j^2 \cdot (1 + b_j(g - 1))$ (square-and-conditionally-multiply)
  - $r_{254} = g^x$ (final result)

**Why we need this**: Verifier can check the constraint $r_{j+1} = r_j^2 \cdot (1 + b_j(g - 1))$ is low-degree (quadratic in field elements), but computing $g^x$ directly is not provable via low-degree polynomials.

---

## 4. Stage 6 Protocol Specification

### 4.1 Input and Setup

**Input to Stage 6**:
```rust
let exps_to_prove = state_manager
    .get_prover_accumulator()
    .borrow()
    .get_recursion_ops()
    .cloned()
    .unwrap_or_default();
// exps_to_prove: Vec<ExponentiationSteps> (length 109)
```

**Hyrax generators**:
```rust
let hyrax_generators = PedersenGenerators::<GrumpkinProjective>::from_urs_file(
    16,  // log of polynomial size (2^16 = 65536 coefficients max)
    b"recursion check",
    Some("hyrax_urs_16.urs"),
);
```

### 4.2 Protocol Steps

**File**: `jolt-core/src/subprotocols/snark_composition.rs:177-386`

#### Step 1: Commit to Witness Polynomials

```rust
// For each exponentiation, extract polynomials
for steps in &exponentiation_steps_vec {
    let base_poly = jolt_optimizations::fq12_to_multilinear_evals(&steps.base);
    all_base_polys.push(base_poly);           // g as 12 Fq coefficients
    all_rho_polys.push(steps.rho_mles.clone()); // [r_0, ..., r_254]
    all_quotient_polys.push(steps.quotient_mles.clone()); // Constraint witnesses
}

// Batch commit all polynomials at once
let all_commitments = HyraxCommitment::<RATIO, GrumpkinProjective>::batch_commit(
    &all_polys_to_commit,
    hyrax_generators,
);
```

**What's committed**:
- **rho_commitments**: Vec<Vec<HyraxCommitment>> - one per exponentiation, containing all r_j intermediates
- **quotient_commitments**: Vec<Vec<HyraxCommitment>> - constraint satisfaction witnesses
- **base_commitments**: Vec<HyraxCommitment> - one per exponentiation, the base g

**Hyrax commitment structure**:
```rust
pub struct HyraxCommitment<const RATIO: usize, G: CurveGroup> {
    pub row_commitments: Vec<G>, // Vector of Grumpkin group elements
}
```

Each polynomial → vector of Pedersen commitments to matrix rows.

#### Step 2: Create ExpSumcheck Instances

```rust
let mut sumcheck_instances: Vec<Box<dyn SumcheckInstance<F>>> =
    exponentiation_steps_vec
        .iter()
        .enumerate()
        .map(|(exp_idx, steps)| {
            let r: Vec<F> = transcript.challenge_vector(4);
            let gamma: F = transcript.challenge_scalar();
            Box::new(ExpSumcheck::new_prover(exp_idx, steps, r, gamma))
                as Box<dyn SumcheckInstance<F>>
        })
        .collect();
```

**ExpSumcheck instance**: For each exponentiation, proves:
$$\sum_{j \in \{0,1\}^{254}} \text{eq}(r, j) \cdot \left[ r_{j+1} - r_j^2 \cdot (1 + b_j(g - 1)) \right] = 0$$

**Why this works**:
- If constraint holds for all $j$, sum over Boolean hypercube is 0
- If any constraint violated, sum is nonzero with high probability
- Sum-check reduces to polynomial evaluation query

#### Step 3: Run Batched Sumcheck

```rust
let (sumcheck_proof, r_sumcheck) = BatchedSumcheck::prove(
    sumcheck_instances_mut,
    Some(prover_accumulator.clone()),
    transcript,
);
```

**Output**:
- `sumcheck_proof`: Univariate polynomials for each round (16 rounds × 109 instances, batched)
- `r_sumcheck`: Challenge vector after sumcheck (point to evaluate polynomials at)

**What's proven**: All 109 exponentiations satisfy their square-and-multiply constraints.

#### Step 4: Batch All Polynomials for Hyrax Opening

```rust
// Get random challenges for batching
let batching_challenges: Vec<F> = transcript.challenge_vector(total_polys);

// Compute batched polynomial and evaluation
let mut batched_poly = vec![F::zero(); 16]; // 2^4 coefficients
let mut batched_eval = F::zero();

// Batch: poly_batched = Σ γ_i · poly_i
for (poly_type, poly, eval, challenge) in all_poly_data {
    batched_eval += challenge * eval;
    for (idx, &coeff) in poly.iter().enumerate() {
        batched_poly[idx] += challenge * F::from(coeff);
    }
}
```

**Why batching**: Instead of opening 109 × ~250 polynomials individually, open one combined polynomial (much cheaper).

#### Step 5: Generate Hyrax Opening Proof

```rust
let batched_hyrax_proof = HyraxOpeningProof::<RATIO, GrumpkinProjective>::prove(
    &batched_dense_poly,
    &r_sumcheck_fq,
    RATIO,
);
```

**Hyrax opening proof**: Proves that the batched polynomial evaluates to `batched_eval` at point `r_sumcheck`.

**Cost**: ~8M RISC-V cycles (dominated by MSMs over Grumpkin, no G_T exponentiations).

### 4.3 Stage 6 Output

```rust
RecursionProof {
    commitments: ExpCommitments {
        rho_commitments,
        quotient_commitments,
        base_commitments,
        num_exponentiations: 109,
        num_constraints_per_exponentiation: Vec<usize>,
        bits_per_exponentiation: Vec<Vec<bool>>,
    },
    sumcheck_proof,      // ExpSumcheck proof
    r_sumcheck,          // Challenge vector
    hyrax_proof,         // Batched Hyrax opening proof
    openings,            // Evaluation claims
}
```

This gets serialized and included in the final Jolt proof.

---

## 5. Why This Terminates (No Stage 7)

### 5.1 The Recursion Problem

**Potential infinite regress**:
- If Stage 6 verification required expensive operations (like Dory G_T exponentiations), we'd need Stage 7 to prove Stage 6
- Stage 7 would need Stage 8, etc. → infinite recursion

**Why Jolt doesn't have this problem**: Hyrax verification is fundamentally different from Dory.

### 5.2 Hyrax Verification Cost

**File**: `jolt-core/src/subprotocols/snark_composition.rs:390-630` (verify function)

**Operations required**:
1. **Homomorphic combination of commitments** (lines 511-558):
   ```rust
   for (commitment, challenge) in commitments.zip(challenges) {
       let gamma_fq: Fq = challenge.into();
       for (i, &com) in commitment.row_commitments.iter().enumerate() {
           batched_row_commitments[i] += com * gamma_fq; // Scalar multiplication
       }
   }
   ```
   **Cost**: ~100 MSMs over Grumpkin (each ~100K cycles) = ~10M cycles

2. **Verify ExpSumcheck** (lines 447-453):
   ```rust
   let verified_r = BatchedSumcheck::verify(
       &proof.sumcheck_proof,
       verifier_instances_ref,
       Some(verifier_accumulator.clone()),
       transcript,
   )?;
   ```
   **Cost**: Check univariate polynomials (field operations) = ~2M cycles

3. **Verify Hyrax opening** (lines 609-620):
   ```rust
   proof.hyrax_proof.verify(
       hyrax_generators,
       &r_sumcheck_fq,
       &batched_opening_fq,
       &batched_hyrax_commitment,
   )
   ```
   **Cost**: More MSMs over Grumpkin = ~8M cycles

**Total Stage 6 verification**: ~20M cycles

**Key difference from Dory**:
- **Dory**: Requires G_T exponentiations (10M each) → would need Stage 7
- **Hyrax**: Only MSMs over Grumpkin (~100K each) → can verify directly

### 5.3 Why Grumpkin/BN254 2-Cycle Matters

**The field matching**:
- **BN254 scalar field** ($\mathbb{F}_r$, ~254 bits): Used for main Jolt trace
- **Grumpkin base field** = BN254 scalar field: Native arithmetic for BN254 operations
- **Grumpkin scalar field** = BN254 base field ($\mathbb{F}_q$): Can represent witness data

**Why this helps**:
- Witness polynomials have $\mathbb{F}_q$ coefficients (from G_T exponentiations, which operate in $\mathbb{F}_{q^{12}}$)
- Grumpkin curve allows committing to $\mathbb{F}_q$ data efficiently
- No non-native field arithmetic needed → no 1000× overhead

**What happens in Stage 6 verification**:
- Verifier performs Grumpkin scalar multiplications (native operations)
- No BN254 operations, no G_T operations
- Everything is "cheap" elliptic curve arithmetic

### 5.4 Comparison Table

| Aspect | Dory (Stages 1-5) | Hyrax (Stage 6) |
|--------|-------------------|-----------------|
| **Curve** | BN254 | Grumpkin |
| **Expensive op** | G_T exponentiation (~10M cycles) | MSM (~100K cycles) |
| **Count** | 109 exponentiations | ~100 MSMs |
| **Total cost** | 1.09B cycles | ~10M cycles |
| **Creates new exps?** | Yes (if computed directly) | **No** ✓ |
| **Needs next stage?** | Yes (unless hinted) | **No** ✓ |

**Termination proof**: Hyrax verification uses only MSMs (100× cheaper than G_T exponentiations) and creates no new expensive operations. Therefore, no Stage 7 needed.

---

## 6. What We Got Wrong

### 6.1 Our Initial Understanding (Incorrect)

**What we thought**:
```
┌──────────────────────────────────────┐
│ Layer 1: Standard Jolt              │
│ • Proves: User program P correct    │
│ • Uses: Dory/BN254                  │
│ • Output: Proof π₁                  │
│ • Verification: 1.28B cycles        │
└──────────────────────────────────────┘
           ↓ Feed π₁ as input
┌──────────────────────────────────────┐
│ Layer 2: Recursive Jolt              │
│ • Guest Program: Jolt verifier       │
│   compiled to RISC-V bytecode        │
│ • Execution: Run verifier on π₁     │
│ • Trace: 320M RISC-V cycles          │
│ • Proves: "I verified π₁ correctly" │
│ • Uses: Mixed PCS (Dory + Hyrax)    │
│ • Output: Proof π₂                  │
│ • Verification: 30M cycles           │
└──────────────────────────────────────┘
```

**Why we thought this**:
- PR name: "feat/snark-composition" → sounds like nested SNARKs
- Feature flag: `recursion` → implies recursive structure
- Function names: `RecursionProof`, `recursion_ops` → suggests recursion
- Terminology in papers: "two-layer" architecture

### 6.2 What Actually Happens (Correct)

**Reality**:
```
┌──────────────────────────────────────────────────────┐
│ Stages 1-4: Standard Jolt Sumchecks                  │
│ • Proves: R1CS, Twist, Shout constraints             │
│ • Commitment: Dory/BN254 for all witness polynomials │
└──────────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────┐
│ Stage 5: Batched Opening (WITH HINTS)                │
│ • Prover: Computes 109 G_T exponentiations           │
│   - Captures ExponentiationSteps during computation  │
│   - Returns steps to Stage 6                         │
│ • Verifier: Accepts 109 results as hints             │
│   - Uses hints to complete Dory verification         │
│   - Doesn't verify hints yet (trusts them)           │
└──────────────────────────────────────────────────────┘
           ↓ No separate trace; just witness data
┌──────────────────────────────────────────────────────┐
│ Stage 6: Prove Hints Correct (SNARK Composition)     │
│ • Input: 109 ExponentiationSteps from Stage 5        │
│ • Commitment: Hyrax/Grumpkin for witness polynomials │
│ • Proves: Each step satisfies square-and-multiply    │
│   constraint via ExpSumcheck                         │
│ • Verification: ~20M cycles (MSMs only, no G_T ops)  │
└──────────────────────────────────────────────────────┘
```

### 6.3 Key Differences

| Aspect | What We Thought | What Actually Happens |
|--------|----------------|----------------------|
| **Structure** | Two separate Jolt instances | One Jolt proof, two PCS schemes |
| **Layer 2 trace** | 320M RISC-V cycles | No separate trace |
| **Guest program** | Jolt verifier compiled to RISC-V | No guest program in Stage 6 |
| **What Stage 6 proves** | "I verified Layer 1 correctly" | "These 109 exponentiations are correct" |
| **Witness generation** | Run verifier to generate trace | Capture steps during Stage 5 proving |
| **Terminology** | Recursion (nested proofs) | Composition (parallel proof systems) |
| **Verification flow** | Verify π₂, which proves π₁ valid | Verify Stages 1-5, then verify Stage 6 |

### 6.4 Why the Confusion?

**Team's clarification**:
> "What you are calling Layer 2 is just 'extra committed data appended to the basic Dory evaluation proof' that helps make checking the Dory evaluation proof faster for the verifier."

**Our misunderstanding**:
- Interpreted "recursion" as "SNARK of a SNARK"
- Saw 320M cycle numbers in cost analysis, assumed that was a trace
- Didn't realize ExponentiationSteps are captured during proving, not generated as execution trace

**What it actually is**:
- Stage 6 is an **extension** of the Jolt DAG, not a separate system
- Witness data is a **byproduct** of Stage 5 computation, not a new trace
- The proof is **compositional** (multiple PCS schemes) not **recursive** (nested proofs)

### 6.5 Remaining Uncertainties

**Terminology**:
- Is "recursion" the right word, or should it be "composition" throughout?
- The team uses both terms ("recursion" in code, "composition" in explanation)
- Perhaps "recursion" refers to the feature enabling SNARK composition, not the architecture itself

**Architecture boundaries**:
- Is Stage 6 truly "part of" Jolt, or is it "external to" Jolt?
- The team says "integrating Hyrax + SZ-check directly into Layer 1's Dory verification"
- We interpret this as: Stage 6 extends the proof, not wraps it

**Conceptual framing**:
- The team didn't explicitly say "you're wrong about two layers"
- They said "what you call Layer 2 is just extra committed data"
- This suggests different mental models, not necessarily incorrect understanding

---

## 7. Key Takeaways

### 7.1 Architectural Summary

✅ **One Jolt proof** with two commitment schemes (Dory/BN254 for main trace, Hyrax/Grumpkin for witnesses)

✅ **Stage 6 is an extension** of the standard 5-stage DAG, not a separate proof system

✅ **No separate trace**: Witness data (ExponentiationSteps) captured during Stage 5 proving

✅ **Sequential verification**: Verifier runs Stages 1-5, then Stage 6 (not nested)

✅ **Termination via Hyrax**: Stage 6 verification uses only MSMs, creates no new G_T exponentiations

### 7.2 Cost Improvements

| Component | Direct Computation | With Stage 6 |
|-----------|-------------------|--------------|
| **Stages 1-4** | ~20M cycles | ~20M cycles (unchanged) |
| **Stage 5** | 1.09B cycles (109 G_T exps) | ~10M cycles (accept hints) |
| **Stage 6** | N/A | ~20M cycles (prove hints correct) |
| **Total** | **~1.28B cycles** | **~50M cycles** |
| **Speedup** | — | **~25×** |

### 7.3 When to Use This

**Recommended when**:
- Verification cost matters (on-chain, embedded systems, verifier networks)
- Willing to trade prover time (10× slower) for verifier speedup (25×)
- Need transparent setup (no trusted setup like Groth16)

**Not needed when**:
- Verification happens infrequently (can afford 1.28B cycles)
- Prover time is critical (Stage 6 adds significant overhead)
- Using pairing-friendly curves with cheap exponentiations

### 7.4 Implementation Status

✅ Fully implemented in PR #975 (`feat/snark-composition`)

✅ Example code: `examples/recursion/` (shows both embedded and input modes)

✅ Feature flag: `recursion` (enables Stage 6 in JoltDAG)

✅ Dependencies: `jolt_optimizations` crate (ExponentiationSteps), `dory` crate (with recursion support)

---

## References

- **Implementation**: `jolt-core/src/subprotocols/snark_composition.rs`
- **Dory integration**: `jolt-core/src/poly/commitment/dory.rs` (lines 1265-1370)
- **Opening proof**: `jolt-core/src/poly/opening_proof.rs` (lines 1005-1074)
- **JoltDAG**: `jolt-core/src/zkvm/dag/jolt_dag.rs` (lines 313-375 prove, 549-604 verify)
- **Theory**: `docs/01_Jolt_Theory_Enhanced.md` Part VII (corrected)
- **Cost analysis**: `docs/Dory_Verification_Cost_Summary.md`
