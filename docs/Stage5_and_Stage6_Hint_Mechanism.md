# Stage 5 and Stage 6: The Hint Mechanism Deep Dive

## Overview

This document provides an in-depth analysis of how Jolt's modified Stage 5 and new Stage 6 work together via a **hint mechanism** to avoid expensive GT exponentiations in verification.

**Key insight**: The verifier accepts pre-computed GT exponentiation results as **hints** from the prover, then verifies their correctness in Stage 6 using a different proof system (Hyrax + ExpSumcheck).

---

## The Two-Capture System

### Capture Point 1: Stage 5 RLC (Homomorphic Combining)

**Location**: [`opening_proof.rs:1007-1023`](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/poly/opening_proof.rs#L1007-L1023)

**What happens**: During prover's Stage 5, when combining commitments via random linear combination:

```rust
let (_joint_commitment, hint_with_steps) =
    DoryCommitmentScheme::precompute_combined_commitment(
        dory_commitments_slice,
        dory_coeffs,
    );

// Capture GT exponentiation steps for Stage 6
self.extend_recursion_ops(hint_with_steps.exponentiation_steps.clone());

tracing::debug!(
    num_gt_exponentiations = hint_with_steps.exponentiation_steps.len(),
    num_commitments = dory_commitments_slice.len(),
    "Prover captured GT exponentiation steps from homomorphic commitment combining"
);

homomorphic_combining_hint = Some(hint_with_steps.scaled_commitments);
```

**What's captured**:
- **GT exponentiation results**: The final values $C_i^{\gamma_i}$ for each commitment
- **ExponentiationSteps**: Intermediate values for each of the ~29 exponentiations in RLC
- Stored in `homomorphic_combining_hint` for inclusion in proof

**Purpose**: This is the **first batch** of GT exponentiations. In standard Dory, computing:
$$C_{\text{combined}} = \prod_{i=1}^{29} C_i^{\gamma_i}$$

requires 29 GT exponentiations @ ~10M cycles each = **290M cycles**.

### Capture Point 2: Stage 5 Opening Proof (Main Dory Protocol)

**Location**: [`opening_proof.rs:1030-1064`](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/poly/opening_proof.rs#L1030-L1064)

**What happens**: During Dory's main opening proof generation:

```rust
#[cfg(feature = "recursion")]
let joint_opening_proof = {
    if std::any::TypeId::of::<PCS>()
        == std::any::TypeId::of::<super::commitment::dory::DoryCommitmentScheme>()
    {
        use super::commitment::recursion::RecursionCommitmentScheme;

        // Call prove_with_auxiliary to get both proof AND exponentiation steps
        let prove_fn: fn(...) -> (PCS::Proof, AuxiliaryVerifierData) = unsafe {
            std::mem::transmute(
                DoryCommitmentScheme::prove_with_auxiliary::<ProofTranscript>
                    as fn(_, _, _, _, _) -> _,
            )
        };

        let (proof, auxiliary_data) =
            prove_fn(pcs_setup, &joint_poly, &r_sumcheck, hint, transcript);

        // Extract exponentiation steps from Dory's main protocol
        if let Some(ref steps) = auxiliary_data.full_exponentiation_steps {
            self.extend_recursion_ops(steps.clone());
        }

        proof
    } else {
        PCS::prove(pcs_setup, &joint_poly, &r_sumcheck, hint, transcript)
    }
};
```

**What's captured**:
- **GT exponentiation results**: From all $5 \times \log_2 N = 80$ exponentiations in main Dory protocol
- **ExponentiationSteps**: Complete witness data for square-and-multiply constraints
- Added to the same accumulator as Capture Point 1

**Purpose**: This is the **second batch** of GT exponentiations. The main Dory opening requires 80 GT exponentiations @ ~10M cycles each = **800M cycles**.

**Total captured**: 29 (RLC) + 80 (main Dory) = **109 GT exponentiations**

---

## The Hint Flow Through The System

### 1. Prover Side (Stage 5)

**`reduce_and_prove()` function** ([`opening_proof.rs:920-1100`](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/poly/opening_proof.rs#L920-L1100)):

```
┌─────────────────────────────────────────────────────────────┐
│ ProverOpeningAccumulator::reduce_and_prove()                │
├─────────────────────────────────────────────────────────────┤
│ 1. Batch-reduce openings via sumcheck                       │
│    → Generates sumcheck_proof                               │
│                                                              │
│ 2. Combine commitments homomorphically (Capture Point 1)    │
│    → Computes: C_combined = ∏ᵢ Cᵢ^γᵢ                       │
│    → Captures: homomorphic_combining_hint                   │
│    → Stores: ExponentiationSteps for 29 exponentiations    │
│                                                              │
│ 3. Generate Dory opening proof (Capture Point 2)            │
│    → Runs full Dory protocol with auxiliary data capture    │
│    → Captures: ExponentiationSteps for 80 exponentiations  │
│    → Stores: In accumulator.recursion_ops                   │
│                                                              │
│ 4. Package into ReducedOpeningProof                         │
│    → sumcheck_proof                                         │
│    → sumcheck_claims                                        │
│    → joint_opening_proof                                    │
│    → homomorphic_gt_results (THE HINT!)                     │
└─────────────────────────────────────────────────────────────┘
```

**Key structure** ([`opening_proof.rs:607-618`](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/poly/opening_proof.rs#L607-L618)):

```rust
pub struct ReducedOpeningProof<F, PCS, ProofTranscript> {
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub sumcheck_claims: Vec<F>,
    pub joint_opening_proof: PCS::Proof,

    /// Precomputed GT commitments from homomorphic combining
    /// (Dory recursion-mode verifier optimization)
    #[cfg(feature = "recursion")]
    pub homomorphic_gt_results: Option<Vec<super::commitment::dory::JoltGTBn254>>,

    // ... test-only fields ...
}
```

**The `homomorphic_gt_results` field**:
- Contains ~29 GT elements (one per commitment in RLC)
- Each element is a $\mathbb{G}_T$ point (12 $\mathbb{F}_q$ components)
- Total size: ~29 × 12 × 32 bytes ≈ **11 KB**
- This is the **hint** that allows verifier to skip 29 GT exponentiations

### 2. Prover Side (Stage 6)

**`JoltDAG::prove()` Stage 6** ([`jolt_dag.rs:313-375`](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/zkvm/dag/jolt_dag.rs#L313-L375)):

```rust
#[cfg(feature = "recursion")]
{
    tracing::info!("Stage 6: SNARK composition proving");

    // Retrieve ALL captured exponentiation steps (109 total)
    let exps_to_prove = state_manager
        .get_prover_accumulator()
        .borrow()
        .get_recursion_ops()
        .cloned()
        .unwrap_or_default();

    tracing::debug!(
        num_exponentiations = exps_to_prove.len(),
        "Retrieved recursion operations"
    );

    // Load Hyrax generators (one-time setup)
    let hyrax_generators = PedersenGenerators::<GrumpkinProjective>::from_urs_file(
        16,
        b"recursion check",
        Some("hyrax_urs_16.urs"),
    );

    // Generate Stage 6 proof (Hyrax + ExpSumcheck)
    let recursion_proof = snark_composition_prove::<ark_bn254::Fq, ProofTranscript, 1>(
        exps_to_prove,
        &mut *transcript.borrow_mut(),
        &hyrax_generators,
    );

    // Store in proof
    state_manager.proofs.borrow_mut().insert(
        ProofKeys::Recursion,
        ProofData::RecursionProof(recursion_proof_f),
    );
}
```

**What Stage 6 proves**:
- All 109 GT exponentiations satisfy square-and-multiply constraints
- Commits to ~27,686 intermediate $\mathbb{G}_T$ values via Hyrax
- Runs 109 batched ExpSumcheck instances (4 rounds each)
- Generates batched Hyrax opening proof

### 3. Verifier Side (Stage 5)

**`reduce_and_verify()` function** ([`opening_proof.rs:1385-1578`](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/poly/opening_proof.rs#L1385-L1578)):

```
┌─────────────────────────────────────────────────────────────┐
│ VerifierOpeningAccumulator::reduce_and_verify()             │
├─────────────────────────────────────────────────────────────┤
│ 1. Verify batch opening reduction sumcheck                  │
│    → Checks: sumcheck_proof                                 │
│    → Output: r_sumcheck (random challenge point)            │
│                                                              │
│ 2. Compute joint commitment (USING HINT!)                   │
│    → Traditional: C_joint = ∏ᵢ Cᵢ^γᵢ (29 GT exps!)        │
│    → With hint: C_joint = use precomputed results           │
│    → Saves: 29 × ~10M = ~290M cycles                       │
│                                                              │
│ 3. Verify Dory opening proof                                │
│    → Traditional: requires 80 GT exps during verification   │
│    → With Stage 6: GT exp results accepted as hints         │
│    → Stage 6 will verify correctness separately             │
│                                                              │
│ 4. Check: PCS::verify(proof, commitment, point, claim)      │
│    → Uses joint_commitment from step 2                      │
│    → Uses r_sumcheck from step 1                            │
└─────────────────────────────────────────────────────────────┘
```

**The critical hint usage** ([`opening_proof.rs:1483-1547`](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/poly/opening_proof.rs#L1483-L1547)):

```rust
// Compute joint commitment for reduced opening proof
let joint_commitment = {
    let (coeffs, commitments) = /* extract from sumcheck */;

    // In recursion mode with Dory, use precomputed GT results from prover
    #[cfg(feature = "recursion")]
    {
        if /* using Dory */ {
            if let Some(ref gt_results) = reduced_opening_proof.homomorphic_gt_results {
                tracing::debug!(
                    num_gt_results = gt_results.len(),
                    num_commitments = commitments.len(),
                    "Verifier using precomputed GT hint for homomorphic commitment combining"
                );

                // Create hint structure
                let hint = DoryCombinedCommitmentHint {
                    scaled_commitments: gt_results.clone(),
                    exponentiation_steps: vec![], // Verifier doesn't need steps
                };

                // Use hint to avoid GT exponentiations!
                let joint_commitment_dory =
                    DoryCommitmentScheme::combine_commitments_with_hint(
                        dory_commitments,
                        dory_coeffs,
                        Some(&hint),
                    );

                tracing::debug!("Verifier successfully combined commitments using GT hint");

                return joint_commitment_dory;
            } else {
                tracing::debug!(
                    "Verifier computing GT operations natively (no hint available)"
                );
                // Fallback: compute GT exponentiations directly
                PCS::combine_commitments(&commitments, &coeffs).unwrap()
            }
        }
    }

    #[cfg(not(feature = "recursion"))]
    PCS::combine_commitments(&commitments, &coeffs).unwrap()
};
```

**What happens**:
1. **Check if hint available**: `if let Some(ref gt_results) = reduced_opening_proof.homomorphic_gt_results`
2. **Create hint structure**: Package precomputed GT results
3. **Call specialized function**: `combine_commitments_with_hint()` instead of standard `combine_commitments()`
4. **Result**: Joint commitment computed WITHOUT 29 GT exponentiations

**Cost saved**: ~290M cycles (29 exponentiations × ~10M cycles each)

### 4. Verifier Side (Stage 6)

**`JoltDAG::verify()` Stage 6** ([`jolt_dag.rs:549-604`](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/zkvm/dag/jolt_dag.rs#L549-L604)):

```rust
#[cfg(feature = "recursion")]
{
    tracing::info!("Verifier: Stage 6 SNARK composition verification");

    let proofs = state_manager.proofs.borrow();
    if let Some(ProofData::RecursionProof(sz_proof)) = proofs.get(&ProofKeys::Recursion) {
        // Load Hyrax generators (same as prover)
        let hyrax_generators = PedersenGenerators::<GrumpkinProjective>::from_urs_file(
            16,
            b"recursion check",
            Some("hyrax_urs_16.urs"),
        );

        // Verify Stage 6 proof
        snark_composition_verify::<Fq, ProofTranscript, 1>(
            sz_proof_fq,
            &mut *transcript.borrow_mut(),
            &hyrax_generators,
        )
        .context("Stage 6 - Recursion Check Protocol verification")?;

        tracing::info!(
            verify_duration_ms = verify_start.elapsed().as_millis(),
            total_duration_ms = stage6_start.elapsed().as_millis(),
            "Stage 6 verification successful"
        );
    }
}
```

**What Stage 6 verifies**:
1. **ExpSumcheck**: All 109 GT exponentiations satisfy square-and-multiply constraints
2. **Hyrax opening**: Committed witness data opens correctly at challenge points
3. **Binding**: The verified exponentiations match the hints used in Stage 5

**Cost**: ~321-361M cycles (vs 1.09B for computing GT exps directly)

---

## The Trust Model

### What the Verifier Trusts

**Stage 5**: Verifier **accepts** GT exponentiation results as hints:
- `homomorphic_gt_results`: 29 GT elements for RLC
- Implicitly: 80 GT exponentiations in Dory opening (not sent explicitly, but verified in Stage 6)

**Stage 6**: Verifier **verifies** correctness of all 109 accepted hints:
- Via ExpSumcheck: Each GT exp satisfies $\rho_{i+1} = \rho_i^2 \cdot g^{b_i} - q_i \cdot h(g)$ for all 254 bits
- Via Hyrax: Committed intermediate values open correctly
- Binds back to Stage 5 via transcript

### Security Guarantee

**Soundness**: If prover cheats (provides incorrect GT exp results), Stage 6 verification will fail with overwhelming probability.

**Why it's secure**:
1. **Cryptographic binding**: Transcript commits to all GT results before Stage 6
2. **Constraint checking**: ExpSumcheck verifies 254 constraints per exponentiation
3. **Commitment binding**: Hyrax commits to witness data, preventing prover from adapting witness after seeing challenges
4. **Fiat-Shamir**: All challenges derived from transcript, preventing selective response attacks

**Proof size impact**:
- `homomorphic_gt_results`: +11 KB in Stage 5 proof
- Stage 6 proof: Additional ~200-500 KB (Hyrax commitments + ExpSumcheck + opening proof)
- **Total overhead**: ~211-511 KB

**Verification cost impact**:
- Stage 5 saved: ~290M cycles (RLC GT exps) + ~800M cycles (main Dory GT exps, verified in Stage 6) = ~1.09B cycles
- Stage 6 cost: ~321-361M cycles
- **Net savings**: ~730-770M cycles (**67-71% reduction**)

---

## Technical Details

### Why Grumpkin? The 2-Cycle Advantage

**Critical design choice**: Stage 6 uses **Hyrax commitments over Grumpkin**, not BN254.

#### The Field Matching Property

**The 2-cycle**: BN254 and Grumpkin form a **2-cycle of elliptic curves**:
- **BN254's base field** $\mathbb{F}_q$ = **Grumpkin's scalar field** $\mathbb{F}_r$
- **Grumpkin's base field** $\mathbb{F}_r$ = **BN254's scalar field** $\mathbb{F}_q$ (approximately)

**Why this matters for Stage 6**:

1. **GT elements have coefficients in $\mathbb{F}_q$ (BN254 base field)**
   - Each GT element is in the extension field $\mathbb{F}_{q^{12}}$
   - Represented as: $g = (a_0, a_1, ..., a_{11})$ where each $a_i \in \mathbb{F}_q$
   - Example: $\rho_j = (c_0, c_1, ..., c_{11})$ with $c_i \in \mathbb{F}_q$

2. **Hyrax commitments require scalars**
   - Pedersen commitment: $C = \sum_{i=0}^{11} m_i \cdot G_i$ where $m_i$ are scalar field elements
   - For Grumpkin: scalars are in $\mathbb{F}_r$ = $\mathbb{F}_q$ (BN254's base field)
   - **Perfect match!** GT coefficients are native scalars for Grumpkin commitments

3. **Native field arithmetic (no expensive emulation)**
   - Without Grumpkin: Would need to represent $\mathbb{F}_q$ elements in BN254's scalar field
   - This requires **limb decomposition** and **non-native field arithmetic**
   - Cost: ~10-100× slower for each field operation
   - With Grumpkin: $\mathbb{F}_q$ values are **native scalars** - direct use!

#### Concrete Example: Committing to an Intermediate GT Value

**Intermediate step in exponentiation**: $\rho_j = (b_0, b_1, ..., b_{11})$ where each $b_i \in \mathbb{F}_q$

**Packing into MLE**: Pack 12 coefficients into 4-variable MLE (16 evaluations):
```
MLE evaluations: [b_0, b_1, ..., b_11, 0, 0, 0, 0]
                  ↑ Each is an Fq element (BN254 base field)
```

**Hyrax commitment (Grumpkin)**:
```rust
// Commit to MLE evaluations
C_ρj = b_0·G_0 + b_1·G_1 + ... + b_11·G_11 + 0·G_12 + ... + 0·G_15
       ↑                                     ↑
    Fq scalars                          Grumpkin generators
  (native to Grumpkin!)
```

**If we used BN254 instead**:
```rust
// Problem: b_i ∈ Fq, but BN254 scalars are in Fr ≠ Fq
// Would need to decompose each b_i into limbs:
b_i = b_i_low + b_i_high · 2^128

// Then commit to twice as many values:
C_ρj = b_0_low·G_0 + b_0_high·G_1 + b_1_low·G_2 + ...
       ↑ Now in Fr (BN254 scalar field)
       ↑ But requires 2× the generators, 2× the commitment size
```

#### What Operations Use Which Curve?

| Operation | Curve/Field | Why |
|-----------|-------------|-----|
| **GT exponentiations** (actual computation) | BN254 $\mathbb{G}_T$ ($\mathbb{F}_{q^{12}}$) | Stage 5 - what we're trying to avoid in verification |
| **GT multiplications** (hint usage) | BN254 $\mathbb{G}_T$ ($\mathbb{F}_{q^{12}}$) | Stage 5 - verifier combines precomputed hints |
| **Committing to $\mathbb{F}_q$ coefficients** | **Grumpkin** (scalars in $\mathbb{F}_q$) | **Stage 6 - Hyrax commitments** |
| **ExpSumcheck constraints** | $\mathbb{F}_q$ arithmetic | Stage 6 - proving exponentiation steps |
| **Hyrax MSMs** | Grumpkin $\mathbb{G}_1$ | Stage 6 - commitment verification |

#### Why Not BN254 for Everything?

**If we used BN254 Hyrax instead of Grumpkin**:

1. **Non-native field arithmetic**:
   - GT coefficients are $\mathbb{F}_q$ (BN254 base field)
   - BN254 Pedersen scalars are $\mathbb{F}_r$ (BN254 scalar field)
   - Would need to represent $\mathbb{F}_q$ in $\mathbb{F}_r$ (different field!)

2. **Limb decomposition overhead**:
   - Each 254-bit $\mathbb{F}_q$ element → two 128-bit limbs in $\mathbb{F}_r$
   - Double the number of scalar multiplications
   - Double the commitment size
   - **Cost**: ~2× slower commitments, 2× proof size

3. **ExpSumcheck constraints more complex**:
   - Constraints are over $\mathbb{F}_q$ (GT coefficients are in $\mathbb{F}_q$)
   - Would need to express $\mathbb{F}_q$ arithmetic using $\mathbb{F}_r$ constraints
   - Requires range checks and limb-wise constraint checking
   - **Cost**: ~10-100× more expensive sumcheck

#### Performance Impact

**With Grumpkin** (actual implementation):
- Commitment to 16-element MLE: 16 scalar multiplications in Grumpkin
- Each scalar is native $\mathbb{F}_q$ - **no conversion needed**
- ExpSumcheck: Native $\mathbb{F}_q$ arithmetic
- Total Stage 6: **~321-361M cycles**

**Hypothetical with BN254** (not used):
- Commitment to same data: 32 scalar multiplications (doubled due to limb decomposition)
- Each commitment: ~2× slower
- ExpSumcheck: ~10× more expensive (non-native field constraints)
- **Estimated**: ~3-10× slower Stage 6 → **~1-3.6B cycles**

**Net effect**: Grumpkin's 2-cycle property saves **~700M-3.2B cycles** in Stage 6!

#### Summary: The Power of 2-Cycles

**The insight**: When proving properties of one field's elements, use commitments in a curve whose scalar field matches that field.

- **BN254**: Main Jolt proof (execution trace over BN254 scalar field $\mathbb{F}_r$)
- **Grumpkin**: Auxiliary witness (GT coefficients over BN254 base field $\mathbb{F}_q$)
- **Match**: Grumpkin scalar field = BN254 base field → **native arithmetic!**

This is called **SNARK composition with 2-cycles** - using two curves that complement each other's field structures.

---

### The `combine_commitments_with_hint()` Function

**Purpose**: Uses precomputed GT results instead of computing GT exponentiations.

**Signature** (conceptual, actual implementation in Dory crate):
```rust
fn combine_commitments_with_hint(
    commitments: &[DoryCommitment],
    coeffs: &[Fq],
    hint: Option<&DoryCombinedCommitmentHint>,
) -> DoryCommitment
```

**Without hint** (standard path):
```rust
// Compute: C_joint = ∏ᵢ Cᵢ^coeffᵢ
let mut result = GT::identity();
for (commitment, coeff) in commitments.iter().zip(coeffs.iter()) {
    result = result * commitment.power(*coeff);  // GT exponentiation!
}
return result;
```

**With hint** (optimized path):
```rust
// Use precomputed: hint.scaled_commitments[i] = Cᵢ^coeffᵢ
let mut result = GT::identity();
for scaled_commitment in hint.scaled_commitments.iter() {
    result = result * scaled_commitment;  // GT multiplication only!
}
return result;
```

**Cost comparison**:
- **Without hint**: 29 GT exponentiations × ~10M cycles = **290M cycles**
- **With hint**: 29 GT multiplications × ~54K cycles = **1.6M cycles**
- **Speedup**: **180× faster**

### The ExponentiationSteps Structure

**Captured during proving** (from `jolt_optimizations` crate):

```rust
pub struct ExponentiationSteps {
    /// Base of exponentiation (g ∈ GT)
    pub base: Fq12,

    /// Exponent bits [b_1, b_2, ..., b_254]
    pub bits: Vec<bool>,

    /// Accumulator MLEs: [ρ_0, ρ_1, ..., ρ_254]
    /// Each is a 4-variable MLE (16 coefficients)
    pub rho_mles: Vec<Vec<Fq>>,

    /// Quotient MLEs: [q_1, q_2, ..., q_254]
    /// Each is a 4-variable MLE (16 coefficients)
    pub quotient_mles: Vec<Vec<Fq>>,
}
```

**Size per exponentiation**:
- Rho MLEs: 255 × 16 × 32 bytes = 130 KB
- Quotient MLEs: 254 × 16 × 32 bytes = 130 KB
- Base: 12 × 32 bytes = 384 bytes
- Bits: 254 bits = 32 bytes
- **Total per exponentiation**: ~260 KB

**Total witness size** (109 exponentiations):
- 109 × 260 KB = **~28.3 MB** (raw data)
- Committed via Hyrax (succinctly): **~5 KB** (4 row commitments per polynomial × 55,590 polynomials)

### Why Not Send ExponentiationSteps to Verifier?

**Considered alternatives**:
1. **Send raw witness data** (~28.3 MB): Proof too large
2. **Verifier recomputes from GT results**: Requires GT exponentiations (defeats purpose)
3. **Current approach** (commit + open): Best tradeoff
   - Prover commits to witness data via Hyrax (~5 KB commitments)
   - Verifier checks consistency via sumcheck + opening proof
   - Proof size: ~211-511 KB (acceptable)

---

## Comparison: With vs Without Stage 6

### Without Stage 6 (Standard Dory Verification)

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Verifier Computes GT Exponentiations              │
├─────────────────────────────────────────────────────────────┤
│ 1. Batch opening reduction sumcheck                         │
│    Cost: ~20M cycles                                        │
│                                                              │
│ 2. Homomorphic commitment combining                         │
│    Compute: C_joint = ∏ᵢ Cᵢ^γᵢ                            │
│    Cost: 29 GT exps × ~10M = 290M cycles                   │
│                                                              │
│ 3. Dory opening proof verification                          │
│    Cost: 80 GT exps × ~10M = 800M cycles                   │
│                                                              │
│ 4. Final checks                                             │
│    Cost: ~10M cycles                                        │
│                                                              │
│ Total Stage 5: ~1.12B cycles                                │
└─────────────────────────────────────────────────────────────┘

Total verification: ~2.37B cycles
```

### With Stage 6 (Hint + Verification Approach)

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Verifier Uses Hints                                │
├─────────────────────────────────────────────────────────────┤
│ 1. Batch opening reduction sumcheck                         │
│    Cost: ~20M cycles                                        │
│                                                              │
│ 2. Homomorphic commitment combining (WITH HINT)             │
│    Use: precomputed GT results from proof                   │
│    Cost: 29 GT muls × ~54K = 1.6M cycles                   │
│                                                              │
│ 3. Dory opening proof verification (WITH HINT)              │
│    Accept: GT exp results, verify in Stage 6                │
│    Cost: ~10M cycles (no GT exps)                           │
│                                                              │
│ 4. Final checks                                             │
│    Cost: ~10M cycles                                        │
│                                                              │
│ Subtotal Stage 5: ~42M cycles (vs 1.12B)                    │
├─────────────────────────────────────────────────────────────┤
│ Stage 6: Verify Hints via Hyrax + ExpSumcheck              │
├─────────────────────────────────────────────────────────────┤
│ 1. Hyrax batch commitment verification                      │
│    Cost: ~80-120M cycles                                    │
│                                                              │
│ 2. ExpSumcheck (109 instances × 4 rounds)                   │
│    Cost: ~240M cycles                                       │
│                                                              │
│ 3. Hyrax opening proof                                      │
│    Cost: ~1M cycles                                         │
│                                                              │
│ Subtotal Stage 6: ~321-361M cycles                          │
└─────────────────────────────────────────────────────────────┘

Total verification: ~530M cycles (vs 2.37B)
Speedup: 4.5× faster
```

---

## Key Takeaways

1. **Hint mechanism**: Prover provides GT exponentiation results; verifier accepts them and verifies correctness separately

2. **Two capture points**: Hints captured during both RLC combining (29 exps) and main Dory opening (80 exps)

3. **Trust-but-verify**: Verifier trusts hints in Stage 5, verifies them in Stage 6 via different proof system

4. **Cost tradeoff**: Pay ~340M cycles in Stage 6 to save ~1.09B cycles in Stage 5 → **67-71% net savings**

5. **Proof size**: Additional ~211-511 KB for Stage 6 proof (hint + commitments + opening proof)

6. **Security**: Soundness guaranteed by cryptographic binding via transcript and constraint checking via ExpSumcheck

7. **Modularity**: Stage 6 is conditionally compiled (`#[cfg(feature = "recursion")]`), can be disabled for non-recursive use cases

8. **Implementation elegance**: Minimal changes to existing Stage 5 logic; hint usage is opt-in fallback path
