# Stage 5 and Stage 6: The Hyrax Approach to Verification Efficiency

## Summary: Verification Cost Reduction

PR #975 reduces Jolt verification cost from **~1.2B cycles** to **~400M cycles** through SNARK composition (Stage 6).

### Baseline Verification (No Stage 6)

| Component | Cost (cycles) | % of Total |
| --- | --- | --- |
| **Stage 5 GT exponentiations** (109 @ 10M each) | **1,090M** | **91%** |
| Stage 5 pairings (5 @ 20M each) | 100M | 8% |
| Stages 1-4 sumchecks | 20M | 2% |
| G1/G2 operations | 50M | 4% |
| Miscellaneous | 20M | 2% |
| **Total** | **~1,200M** | **100%** |

### Modified Verification (With Stage 6)

| Component | Cost (cycles) | % of Total |
| --- | --- | --- |
| **Stage 6: Hyrax approach** | **~340M** | **85%** |
| Stage 5 pairings (5 @ 20M each) | 100M | 25% |
| Stages 1-4 sumchecks | 20M | 5% |
| G1/G2 operations | 50M | 12% |
| Stage 5 RLC (with hints) | 2M | <1% |
| Miscellaneous | 20M | 5% |
| **Total** | **~400M** | **100%** |

**Overall speedup**: ~1.2B / ~400M = **~3× faster**

**Key insight**: Trade 1.1B cycles of GT exponentiations for 340M cycles of Grumpkin MSMs + sumcheck.

---

## Overview

PR #975 addresses the GT exponentiation bottleneck (109 exponentiations, ~1.1B cycles) through a two-stage approach: Stage 5 modifications that leverage hints from the prover, and a new Stage 6 that verifies the correctness of these hints using a separate commitment scheme. This reduces verification cost from ~1.2B cycles to ~400M cycles, achieving a 2.8× overall speedup.

## The Solution Architecture: SNARK Composition

The modified verification protocol employs a **proof-carrying-data pattern** called SNARK composition:

**The key insight**: Instead of having the verifier compute expensive GT exponentiations, the prover provides the *answers* along with a *proof that those answers are correct*. The verifier trades expensive computation for cheaper proof verification.

**Traditional verifier** (Stage 5 only):
```
1. Receive GT commitments
2. Compute 109 GT exponentiations (expensive: ~1.1B cycles)
3. Perform equality checks using computed results
```

**SNARK composition verifier** (Stage 5 + Stage 6):
```
Stage 5:
1. Receive GT commitments
2. Receive precomputed exponentiation results as hints (~11 KB for 29 RLC results)
3. Perform equality checks using hints (cheap: just GT multiplications)

Stage 6:
4. Verify a separate proof that those hints are correct
   - Different curve: Grumpkin (for field matching with BN254's base field Fq)
   - Different PCS: Hyrax (MSM-based, not pairing-based)
   - Sumcheck over square-and-multiply constraints
```

**Why this is called "SNARK composition"**: Stage 6 is essentially a custom SNARK that encapsulates and verifies the Dory verifier's GT operations. We're composing two SNARKs:
- **Main SNARK** (Stages 1-5): Jolt's execution proof using Dory/BN254
- **Auxiliary SNARK** (Stage 6): Proof of correct GT exponentiations using Hyrax/Grumpkin

**The three-step pattern**:
1. **Prover augmentation**: Capture all intermediate square-and-multiply values during Dory operations
2. **Verifier acceptance**: Accept precomputed results as hints in Stage 5 (no computation)
3. **Deferred verification**: Verify correctness of hints in Stage 6 using a different proof system

This transforms expensive native computation into a proof verification problem, achieving a 2.8× overall speedup.

## Stage 5: Modified Opening Proof with Hints

Stage 5 contains Jolt's batched opening proof for all committed polynomials from Stages 1-4. The modification in PR #975 introduces hint-based verification at two specific points in this stage.

### Traditional Stage 5 Flow

In the baseline implementation, Stage 5 verification proceeds as follows:

1. **Random Linear Combination (RLC)**: Batch multiple polynomial commitments into a single joint commitment using random challenges
   - Compute: $C_{\text{joint}} = \sum_{i=1}^{n} \gamma_i \cdot C_i$ where $C_i \in \mathbb{G}_T$
   - Operations: 29 $\mathbb{G}_T$ exponentiations ($\gamma_i \cdot C_i$) + 28 $\mathbb{G}_T$ additions (accumulation)
   - Cost: ~290M cycles (exponentiations dominate; additions ~1M cycles total)

2. **Main Dory Opening**: Verify that the joint polynomial evaluates correctly at the challenge point
   - Multiple pairing operations and consistency checks
   - Cost: 80 $\mathbb{G}_T$ exponentiations (~800M cycles)

3. **Pairing Checks**: Five BN254 pairings to finalize Dory verification
   - After all folding rounds complete, verify the base case (size-1 claim) using the `Scalar-Product` protocol
   - Verification equation: $e(E_1, E_2) \stackrel{?}{=} R + c \cdot Q + c^2 \cdot C + \text{[blinding terms]}$
   - This expands to 5 pairing operations: 4 in `apply_fold_scalars` + 1 in `verify_final_pairing`
   - Cost: ~100M cycles (5 pairings × 20M cycles each)

Total traditional Stage 5 cost: ~1.19 billion cycles

### Modified Stage 5 Flow

The modified implementation captures exponentiation results at two critical points and uses them as hints:

#### Capture Point 1: RLC Combining (lines 1007-1023 in opening_proof.rs)

During the random linear combination phase, the prover computes intermediate $\mathbb{G}_T$ results when combining commitments homomorphically:

```rust
let (_joint_commitment, hint_with_steps) =
    DoryCommitmentScheme::precompute_combined_commitment(
        dory_commitments_slice,
        dory_coeffs,
    );
self.extend_recursion_ops(hint_with_steps.exponentiation_steps.clone());
homomorphic_combining_hint = Some(hint_with_steps.scaled_commitments);
```

The `scaled_commitments` field contains 29 $\mathbb{G}_T$ values representing intermediate exponentiation results. These are stored for later inclusion in the proof.

#### Capture Point 2: Main Dory Opening (lines 1030-1064 in opening_proof.rs)

During the main Dory opening proof generation, the prover captures 80 additional exponentiation steps:

```rust
let (proof, auxiliary_data) =
    prove_fn(pcs_setup, &joint_poly, &r_sumcheck, hint, transcript);
if let Some(ref steps) = auxiliary_data.full_exponentiation_steps {
    self.extend_recursion_ops(steps.clone());
}
```

The `auxiliary_data.full_exponentiation_steps` contains the complete sequence of intermediate $\mathbb{G}_T$ values from all 80 exponentiations in the main Dory protocol.

#### Hint Usage in Verification (lines 1494-1527 in opening_proof.rs)

**The "helping the verifier" pattern**: The prover provides the verifier with pre-computed exponentiation results as *hints*, allowing the verifier to skip expensive computations in Stage 5. However, these hints must be proven correct—that's what Stage 6 does.

When the verifier processes the opening proof, the 29 RLC results are transmitted in the proof as `homomorphic_gt_results`:

```rust
if let Some(ref gt_results) = reduced_opening_proof.homomorphic_gt_results {
    // gt_results = [C₁^γ₁, C₂^γ₂, ..., C₂₉^γ₂₉]  (29 GT elements, already exponentiated)
    let hint = DoryCombinedCommitmentHint {
        scaled_commitments: gt_results.clone(),
        exponentiation_steps: vec![],  // Verifier doesn't need witness
    };
    let joint_commitment = DoryCommitmentScheme::combine_commitments_with_hint(
        dory_commitments, dory_coeffs, Some(&hint)
    );
}
```

**What the verifier receives**: 29 GT elements (each is 12 Fq coefficients = 384 bytes) totaling ~11 KB. These are the *results* of exponentiations, not the intermediate steps.

**Traditional approach** (without hints):
$$C_{\text{joint}} = \prod_{i=1}^{29} C_i^{\gamma_i} \quad \text{(29 GT exponentiations @ 10M cycles each = 290M)}$$

**With hints** (Stage 5 modified):
The verifier receives pre-computed $[C_1^{\gamma_1}, C_2^{\gamma_2}, \ldots, C_{29}^{\gamma_{29}}]$ and simply multiplies them:
$$C_{\text{joint}} = C_1^{\gamma_1} \cdot C_2^{\gamma_2} \cdot \ldots \cdot C_{29}^{\gamma_{29}} \quad \text{(29 GT multiplications)}$$

**Cost breakdown**:
- **29 $\mathbb{G}_T$ multiplications**: Each is 12 $\mathbb{F}_q$ element multiplications (~54K cycles per GT mul)
- **Total**: $29 \times 54K \approx 1.6M$ cycles
- **Rounded**: ~2M cycles vs ~290M cycles (~180× speedup for this component)

This saves approximately 288 million cycles in the RLC phase alone.

**What about the other 80 exponentiations?** The remaining 80 exponentiations from the main Dory opening are NOT transmitted as hints to Stage 5. Instead, the verifier simply doesn't compute them at all during Stage 5—their verification is deferred entirely to Stage 6. The full witness for all 109 exponentiations (including intermediate square-and-multiply steps) is captured by the prover and will be committed to in Stage 6 using Hyrax.

### Modified Stage 5 Cost Breakdown

| Operation | Traditional Cost | Modified Cost | Savings |
|-----------|-----------------|---------------|---------|
| RLC GT exponentiations (29) | ~290M cycles | ~2M cycles (hint acceptance) | ~288M cycles |
| Main Dory GT exponentiations (80) | ~800M cycles | 0 cycles (deferred to Stage 6) | ~800M cycles |
| Pairing operations (5) | ~100M cycles | ~100M cycles | 0 cycles |
| Other operations | ~20M cycles | ~20M cycles | 0 cycles |
| **Total Stage 5** | **~1.21B cycles** | **~122M cycles** | **~1.09B cycles** |

The critical insight is that Stage 5 savings come from two distinct mechanisms: direct hint usage for RLC (180× speedup on that component) and complete deferral of main Dory exponentiations to a different verification approach in Stage 6.

## Stage 6: Verification of Hints

Stage 6 introduces an entirely new verification component that proves the 109 accepted $\mathbb{G}_T$ exponentiation results are correct. This stage employs three key techniques:

1. **Witness augmentation**: Prover commits to all intermediate values from the square-and-multiply algorithm
2. **Alternative curve**: Use Grumpkin (BN254-Grumpkin 2-cycle) for efficient native arithmetic
3. **Alternative PCS**: Use Hyrax (MSM-based) instead of Dory (pairing-based) to avoid circular dependency

### Mathematical Foundation: Square-and-Multiply Constraints

To understand how Stage 6 verifies the exponentiations, we first need to establish what mathematical constraints define a correct exponentiation.

**The Square-and-Multiply Algorithm**: To compute $g^x$ where $x$ is a 254-bit scalar with binary representation $x = \sum_{i=0}^{253} b_i \cdot 2^i$ (where $b_i \in \{0, 1\}$), we use:

1. **Initialize**: $\rho_0 = 1$ (identity in $\mathbb{G}_T$)
2. **For each bit $i$ from 0 to 253**:
   - Square the accumulator: $\rho_i^2$
   - Conditionally multiply by base if bit is 1: $\rho_i^2 \cdot g^{b_i}$
   - Store result: $\rho_{i+1}$
3. **Result**: $\rho_{254} = g^x$

**The Core Constraint**: At each step $i$, the witness must satisfy:

$$\rho_{i+1} = \rho_i^2 \cdot g^{b_i}$$

Where:
- $\rho_i, \rho_{i+1}, g \in \mathbb{G}_T$ (elements of the BN254 pairing target group)
- $b_i \in \{0, 1\}$ is the $i$-th bit of the exponent

**Why Quotient Polynomials?** Elements in $\mathbb{G}_T$ are represented as polynomials with 12 coefficients over a base field (i.e., degree at most 11: $a_0 + a_1 t + \cdots + a_{11} t^{11}$). When multiplying two such polynomials, the result can have up to 23 terms (degree 22), which must be reduced back to 12 coefficients.

**Polynomial reduction** works like modular arithmetic: just as $17 \bmod 5 = 2$ (where $17 = 3 \cdot 5 + 2$), a high-degree polynomial is divided by a fixed polynomial $h(t)$ to get a remainder with 12 coefficients:

$$\text{(degree-22 polynomial)} = q_i \cdot h(t) + \rho_{i+1}$$

Rearranging: $\rho_i^2 \cdot g^{b_i} = \rho_{i+1} + q_i \cdot h(t)$, where:
- $\rho_{i+1}$ is the **remainder** (12 coefficients, the canonical form we keep)
- $q_i$ is the **quotient** (captures the high-degree terms we removed)
- $h(t) = t^{12} - 18t^6 + 82$ is the irreducible polynomial defining BN254's $\mathbb{F}_{q^{12}}$ extension field

The quotient polynomials are part of the witness because the verifier needs to check this division was performed correctly. This adds 254 additional polynomials per exponentiation to the witness size.

**Why This Ensures Correctness**: If all 254 constraints hold and $\rho_0 = 1$, then by induction:

$$\rho_{254} = \rho_0 \cdot g^{b_0} \cdot g^{2 b_1} \cdot g^{4 b_2} \cdots g^{2^{253} b_{253}} = g^{\sum_{i=0}^{253} b_i \cdot 2^i} = g^x$$

Thus the constraints uniquely determine the exponentiation result.

**Simplified Constraint Form**: The core constraint can be rewritten as:

$$r_{j+1} = r_j^2 \cdot (1 + b_j(g-1))$$

This expands to: when $b_j = 0$, we get $r_{j+1} = r_j^2$; when $b_j = 1$, we get $r_{j+1} = r_j^2 \cdot g$, matching the square-and-multiply algorithm structure.

**Error Polynomial Definition**: For each step $j \in \{0, 1, \ldots, 253\}$, define the error term:

$$e_j := r_{j+1} - r_j^2 \cdot (1 + b_j(g-1))$$

**Correctness Criterion**: If $e_j = 0$ for all $j \in \{0, \ldots, 253\}$, then $r_{254} = g^x$ (the exponentiation is correct).

**Verification Strategy**: Instead of computing the exponentiations directly, Stage 6 uses sumcheck to verify that all error terms $e_j$ are zero. This requires the prover to commit to witness data containing the intermediate values and quotients.

### The Witness: ExponentiationSteps

Based on the constraints above, for each of the 109 exponentiations $g^x$, the prover generates a witness containing:

- **Rho values**: 255 intermediate accumulator values ($\rho_0, \rho_1, \ldots, \rho_{254}$), where each $\rho_i \in \mathbb{G}_T$
- **Quotient values**: 254 quotient values ($q_1, q_2, \ldots, q_{254}$), where each $q_i \in \mathbb{G}_T$
- **Base value**: The base element $g \in \mathbb{G}_T$

**Why MLEs?** Each value above is a $\mathbb{G}_T$ element, which is **not a single field element** but rather an $\mathbb{F}_{q^{12}}$ element - essentially a degree-11 polynomial over $\mathbb{F}_q$ with 12 coefficients: $a_0 + a_1 t + \cdots + a_{11} t^{11}$.

**Note on representation**: The Fq12 element starts as a **univariate polynomial** (degree-11) because that's how field extensions are constructed algebraically - as quotient rings $\mathbb{F}_q[t] / p(t)$. However, for the proof system, we **reinterpret** the 12 coefficients as evaluations of a **multilinear polynomial**. This shift is driven by:
1. **Sumcheck compatibility**: The SZ-Check protocol requires multilinear polynomials (sumcheck operates over Boolean hypercube variables)
2. **Hyrax structure**: Hyrax commitments work with MLEs and exploit tensor product structure for efficient verification
3. **Transparent PCS**: MLEs enable transparent polynomial commitment schemes (no trusted setup)

The univariate representation is the **data format** (what Fq12 elements are), while the MLE representation is the **proof encoding** (how we commit to and verify them).

To commit to these using Hyrax (which works with multilinear polynomials), each $\mathbb{G}_T$ element's 12 coefficients are packed into a 4-variable MLE:

**How the packing works**: A 4-variable multilinear polynomial $f(x_0, x_1, x_2, x_3)$ is uniquely determined by its $2^4 = 16$ evaluations on the Boolean hypercube $\{0,1\}^4$. To encode an Fq12 element:
1. Take its 12 coefficients: $[a_0, a_1, \ldots, a_{11}]$
2. Pad to 16 values: $[a_0, a_1, \ldots, a_{11}, 0, 0, 0, 0]$
3. Construct the unique MLE using Lagrange basis:

$$f(x_0, x_1, x_2, x_3) = \sum_{b_0, b_1, b_2, b_3 \in \{0,1\}} a_{8b_0 + 4b_1 + 2b_2 + b_3} \cdot \chi_{(b_0,b_1,b_2,b_3)}(x_0, x_1, x_2, x_3)$$

where the **Lagrange basis polynomial** is:

$$\chi_{(b_0,b_1,b_2,b_3)}(x_0, x_1, x_2, x_3) = \prod_{i=0}^{3} \left( b_i \cdot x_i + (1-b_i) \cdot (1-x_i) \right)$$

**Example**: For the evaluation at $(0,0,0,1)$ (which stores $a_1$):
$$\chi_{(0,0,0,1)}(x_0, x_1, x_2, x_3) = (1-x_0)(1-x_1)(1-x_2) \cdot x_3$$

This polynomial is 1 when $(x_0, x_1, x_2, x_3) = (0,0,0,1)$ and 0 at all other Boolean points.

**Witness structure**:
- Each $\mathbb{G}_T$ element → one 4-variable MLE (16 evaluations on Boolean hypercube)
- We get **one MLE per $\mathbb{G}_T$ value** (not one MLE across all 255 steps)
- Per exponentiation: $255 + 254 + 1 = 510$ $\mathbb{G}_T$ values = 510 MLEs
- Total across 109 exponentiations: $510 \times 109 = 55,590$ $\mathbb{G}_T$ values = **55,590 MLEs committed**
- Total witness size: ~260 KB per exponentiation, ~28 MB total

### Component A: Hyrax Batch Commitment Verification

**What's happening here**: The prover sends 55,590 Hyrax commitments (one per MLE in the witness) to the verifier. The verifier must check that these commitments are well-formed and can be batched for efficient verification in later components. This component does NOT open the commitments—it only prepares them for the batched opening check in Component C.

#### The Hyrax Commitment Scheme

Hyrax is a polynomial commitment scheme based on Pedersen commitments and multi-scalar multiplications, designed for small polynomials. For a 4-variable multilinear polynomial (16 coefficients), Hyrax:

1. Arranges coefficients into a $4 \times 4$ matrix $M$
2. Uses shared random generators $\{G_0, G_1, G_2, G_3\}$ for columns (same for all polynomials)
3. Commits to each row: $C_i = \sum_{j=0}^3 M_{i,j} \cdot G_j$ (4 Pedersen commitments)

The commitment to a single polynomial consists of 4 Grumpkin elliptic curve points (one per row).

**What the prover sends**: All 55,590 × 4 = 222,360 Grumpkin points representing the row commitments. This is the commitment data (~7.1 MB compressed).

**What the verifier does NOT do**: The verifier does NOT recompute these commitments. The commitments are binding—the prover cannot change the witness after sending them.

#### Homomorphic Combination for Batched Verification

**Why batch?** Instead of verifying 55,590 polynomial commitments separately (which would require 55,590 opening checks), the verifier batches them using random linear combination. This reduces 55,590 separate checks to a single batched check in Component C.

**The batching process**: Given random challenges $\{\gamma_i\}$ from the Fiat-Shamir transcript, the verifier computes a linear combination of all commitments:

$$C_{\text{batched}} = \sum_{i=1}^{55,590} \gamma_i \cdot C_i$$

Since each Hyrax commitment $C_i$ consists of 4 row commitments (one per matrix row), the batching is done **per row**. The result $C_{\text{batched}}$ is an **array of 4 Grumpkin points**:

$$C_{\text{batched}}[j] = \sum_{i=1}^{55,590} \gamma_i \cdot C_i[j] \quad \text{for } j \in \{0, 1, 2, 3\}$$

**What this achieves**: By Schwartz-Zippel, if even ONE of the 55,590 committed polynomials is incorrect, the random linear combination will fail the opening check in Component C with overwhelming probability. This transforms 55,590 separate checks into one batched check.

**The computation**: This requires **4 separate multi-scalar multiplications (MSMs)** over Grumpkin:
- Each MSM: 55,590 bases (row commitments from all polynomials) × 55,590 scalars (random challenges)
- Total operations: 4 MSMs × 55,590 = 222,360 scalar multiplications

#### Cost Analysis

The MSM dominates the cost of this component. Using Pippenger's algorithm for efficient batched scalar multiplication:

- **Bases**: 222,360 Grumpkin curve points (4 MSMs × 55,590 bases each)
- **Scalars**: 222,360 field elements (random challenges)

**Cost estimation methodology** (consistent with baseline costs in Verifier costs.md):

1. **Single scalar multiplication** (Grumpkin): ~500K cycles
   - Uses double-and-add: 254 doublings + ~127 additions (50% Hamming weight)
   - Base field operations over 254-bit scalars

2. **Naive approach** (no batching):
   - 222,360 operations × 500K cycles = **~111B cycles**

3. **Pippenger optimization**:
   - Complexity: O(n / log(n)) group operations vs naive O(n)
   - Reduction factor: ~log(n) / c where c is window size (typically 8-16)
   - For n = 222,360 (log₂ ≈ 18), reduction: ~18 / 10 ≈ 1.8× vs O(n) naive
   - However, additional optimizations (bucket aggregation, precomputation) achieve ~1000× total reduction
   - 222,360-base MSM: **~80-120M cycles** (~0.1% of naive 111B cycles)

The logarithmic scaling of Pippenger's algorithm means this 222K-base MSM costs far less than naive scalar multiplication (which would require billions of cycles). The estimate assumes optimized Pippenger implementation with precomputation tables for fixed generators.

### Component B: ExpSumcheck Batched Verification

The second component verifies that all accepted exponentiation results satisfy the square-and-multiply constraints described above.

#### Batched Sumcheck Protocol

As explained in the Mathematical Foundation section above, each exponentiation requires verifying 254 step-by-step constraints. The error polynomial for step $j$ is:

$$e_j := r_{j+1} - r_j^2 \cdot (1 + b_j(g-1))$$

The implementation uses two levels of batching to verify all 109 exponentiations efficiently:

**First level: Within each exponentiation** (254 steps per exponentiation):

For a single exponentiation $k$, the verifier must check that all 254 error terms are zero. However, checking $e_{k,0} = 0, e_{k,1} = 0, \ldots, e_{k,253} = 0$ separately would be inefficient. Instead, we use **random linear combination**: the verifier samples a random challenge $\gamma$ from the transcript and checks:

$$\sum_{j=0}^{253} \gamma^j \cdot e_{k,j} = 0$$

**Why this is secure**: By Schwartz-Zippel lemma, if even ONE error term $e_{k,j} \neq 0$, then the random linear combination will be non-zero with overwhelming probability (except with probability $\leq 254/|\mathbb{F}| \approx 2^{-246}$). A malicious prover cannot make incorrect values "cancel out" because they don't know $\gamma$ until after committing to the witness.

**Key insight**: Each $\mathbb{G}_T$ element is represented as a 4-variable MLE (16 evaluations). The sumcheck protocol operates over these MLE variables, not over the 254 steps directly.

**Why 4 rounds, not 8?** Standard sumcheck over 254 elements would require $\log_2(254) \approx 8$ rounds. However, we're NOT summing over the 254 steps directly. Instead:

- The 254 error terms $e_0, e_1, \ldots, e_{253}$ are **algebraically batched** using random linear combination: $\sum_{j=0}^{253} \gamma^j \cdot e_j$
- Each error term $e_j$ references GT elements stored as **4-variable MLEs** (one MLE per GT element)
- The sumcheck operates over the $2^4 = 16$ evaluations on the Boolean hypercube $\{0,1\}^4$, **NOT** over the 254 steps
- Result: **4 rounds** (one per MLE variable), not 8 rounds

**Wait, but we still need to check all 254 constraints, right?** Yes! Here's how it works:

Think of it like this: instead of iterating through 254 steps in the sumcheck protocol, we **compute all 254 error terms simultaneously at each point** $x \in \{0,1\}^4$ of the Boolean hypercube.

**Concrete example** (simplified to 3 steps instead of 254):

Suppose we have 3 constraints to check: $e_0 = 0, e_1 = 0, e_2 = 0$, where:
- $e_0 = r_1 - r_0^2 \cdot (1 + b_0(g - 1))$
- $e_1 = r_2 - r_1^2 \cdot (1 + b_1(g - 1))$
- $e_2 = r_3 - r_2^2 \cdot (1 + b_2(g - 1))$

Each $r_i$ and $g$ are 4-variable MLEs: $r_i(x_0, x_1, x_2, x_3)$ for $x \in \{0,1\}^4$.

**Step 1: Random linear combination (before sumcheck)**

Verifier samples $\gamma$ from transcript and defines the batched error polynomial:
$$E(x_0, x_1, x_2, x_3) := \gamma^0 \cdot e_0(x_0, x_1, x_2, x_3) + \gamma^1 \cdot e_1(x_0, x_1, x_2, x_3) + \gamma^2 \cdot e_2(x_0, x_1, x_2, x_3)$$

**What this means at each Boolean point**: For example, at point $(0,0,0,1)$:
- Evaluate: $r_0(0,0,0,1), r_1(0,0,0,1), r_2(0,0,0,1), r_3(0,0,0,1), g(0,0,0,1)$
- Compute: $e_0(0,0,0,1) = r_1(0,0,0,1) - r_0(0,0,0,1)^2 \cdot (1 + b_0(g(0,0,0,1) - 1))$
- Similarly: $e_1(0,0,0,1)$ and $e_2(0,0,0,1)$
- Combine: $E(0,0,0,1) = \gamma^0 \cdot e_0(0,0,0,1) + \gamma^1 \cdot e_1(0,0,0,1) + \gamma^2 \cdot e_2(0,0,0,1)$

**Repeat for all 16 points**: $(0,0,0,0), (0,0,0,1), \ldots, (1,1,1,1)$

**Step 2: Sumcheck claim**

Prove that:
$$\sum_{x \in \{0,1\}^4} E(x_0, x_1, x_2, x_3) = 0$$

This is a sum over **16 points** (the Boolean hypercube), not over 3 constraints. The 3 constraints are already combined into $E$.

**Step 3: Sumcheck rounds (4 rounds, one per variable)**

**Round 1** (binding $x_0$):
- **Prover** computes univariate: $g_1(X_0) = \sum_{x_1, x_2, x_3 \in \{0,1\}} E(X_0, x_1, x_2, x_3)$
  - At $X_0 = 0$: sum over 8 points $(0, x_1, x_2, x_3)$
  - At $X_0 = 1$: sum over 8 points $(1, x_1, x_2, x_3)$
  - This is a degree-2 univariate (3 coefficients)
- **Verifier** checks: $g_1(0) + g_1(1) \stackrel{?}{=} 0$ (the claimed sum)
- **Verifier** sends random challenge $r_0 \in \mathbb{F}$
- New claim: $\sum_{x_1, x_2, x_3 \in \{0,1\}} E(r_0, x_1, x_2, x_3) \stackrel{?}{=} g_1(r_0)$

**Round 2** (binding $x_1$):
- **Prover** computes: $g_2(X_1) = \sum_{x_2, x_3 \in \{0,1\}} E(r_0, X_1, x_2, x_3)$
  - At $X_1 = 0$: sum over 4 points $(r_0, 0, x_2, x_3)$
  - At $X_1 = 1$: sum over 4 points $(r_0, 1, x_2, x_3)$
- **Verifier** checks: $g_2(0) + g_2(1) \stackrel{?}{=} g_1(r_0)$
- **Verifier** sends random challenge $r_1 \in \mathbb{F}$
- New claim: $\sum_{x_2, x_3 \in \{0,1\}} E(r_0, r_1, x_2, x_3) \stackrel{?}{=} g_2(r_1)$

**Round 3** (binding $x_2$):
- **Prover** computes: $g_3(X_2) = \sum_{x_3 \in \{0,1\}} E(r_0, r_1, X_2, x_3)$
  - At $X_2 = 0$: sum over 2 points $(r_0, r_1, 0, x_3)$
  - At $X_2 = 1$: sum over 2 points $(r_0, r_1, 1, x_3)$
- **Verifier** checks: $g_3(0) + g_3(1) \stackrel{?}{=} g_2(r_1)$
- **Verifier** sends random challenge $r_2 \in \mathbb{F}$
- New claim: $\sum_{x_3 \in \{0,1\}} E(r_0, r_1, r_2, x_3) \stackrel{?}{=} g_3(r_2)$

**Round 4** (binding $x_3$):
- **Prover** computes: $g_4(X_3) = E(r_0, r_1, r_2, X_3)$
  - At $X_3 = 0$: evaluate $E$ at $(r_0, r_1, r_2, 0)$
  - At $X_3 = 1$: evaluate $E$ at $(r_0, r_1, r_2, 1)$
- **Verifier** checks: $g_4(0) + g_4(1) \stackrel{?}{=} g_3(r_2)$
- **Verifier** sends random challenge $r_3 \in \mathbb{F}$
- Final claim: $E(r_0, r_1, r_2, r_3) \stackrel{?}{=} g_4(r_3)$

**Step 4: Final check (Hyrax opening)**

To verify $E(r_0, r_1, r_2, r_3) = g_4(r_3)$, the verifier needs:
- $r_0(r_0, r_1, r_2, r_3), r_1(r_0, r_1, r_2, r_3), r_2(r_0, r_1, r_2, r_3), r_3(r_0, r_1, r_2, r_3), g(r_0, r_1, r_2, r_3)$
- Hyrax opening proves these evaluations are correct
- Verifier computes $e_0, e_1, e_2$ using the opened values
- Verifier checks: $\gamma^0 \cdot e_0 + \gamma^1 \cdot e_1 + \gamma^2 \cdot e_2 \stackrel{?}{=} g_4(r_3)$

**Bottom line**:
- **4 rounds** because we're summing over a 4-dimensional Boolean hypercube (2^4 = 16 points)
- At each of the 16 points, ALL 3 constraints are computed and combined (via $\gamma$ powers)
- The 3 constraints don't add rounds because they're batched algebraically BEFORE sumcheck starts
- Each round binds one variable, halving the remaining sum

**Why not 3 rounds for 3 constraints?** Because we're NOT iterating through constraints in sumcheck. We're iterating through the 16 evaluations of the MLE representation. The constraint count is irrelevant to the number of rounds.

**Second level: Across all exponentiations** (109 total exponentiations):

**What happens**: Each of the 109 exponentiations is a separate sumcheck instance (4 rounds each). Instead of running them sequentially (which would require 109 × 4 = 436 rounds of interaction), Jolt uses `BatchedSumcheck` to combine them.

**How BatchedSumcheck works**:

1. **Batching coefficient sampling**: Verifier samples 109 random challenges $\{\beta_1, \beta_2, \ldots, \beta_{109}\}$ from the transcript

2. **Combined claim**: Instead of proving 109 separate claims, prove one combined claim:
   $$\sum_{k=1}^{109} \beta_k \cdot \left(\sum_{x \in \{0,1\}^4} E_k(x)\right) = 0$$
   where $E_k$ is the batched error polynomial for exponentiation $k$ (from the first-level batching with $\gamma$)

3. **Shared rounds**: All 109 instances proceed through rounds together:
   - **Round $i$**: Each instance computes its univariate $g_{k,i}(X)$
   - **Prover sends**: $G_i(X) = \sum_{k=1}^{109} \beta_k \cdot g_{k,i}(X)$ (single combined polynomial)
   - **Verifier sends**: Single challenge $r_i$ (same for all 109 instances)
   - **All instances bind**: Each instance updates with the same $r_i$

4. **Result**: Only **4 rounds** of interaction (not 436), with each round sending one combined univariate polynomial

**Why this is secure**: By Schwartz-Zippel, if even ONE of the 109 exponentiations is incorrect, the random linear combination will fail with overwhelming probability (except with probability $\leq 109 \cdot \deg / |\mathbb{F}| \approx 2^{-246}$).

**What this is NOT**: This is NOT the same as Component A's batching. Component A batches 55,590 Hyrax **commitments** for efficient opening. This batches 109 **sumcheck instances** for efficient verification. They serve different purposes:
- **Component A batching**: Reduces 55,590 opening checks to 1 opening check (via random linear combination of commitments)
- **BatchedSumcheck**: Reduces 436 rounds of interaction to 4 rounds (via random linear combination of claims)

#### Cost Analysis

The ExpSumcheck verification cost breaks down per round:

**Per round operations** (4 rounds total):
1. Receive and decompress univariate polynomial (~5-10 field operations)
2. Evaluate polynomial at random challenge point (~10-20 field operations)
3. Compute expected output claim by evaluating witness polynomials at the challenge point (dominates: ~1,500-2,000 field operations amortized across 109 instances)

Total per round: ~3,600 field operations

**Total ExpSumcheck cost**:
- 4 rounds × ~3,600 field operations/round = ~14,400 field operations
- Field: Grumpkin scalar field (254-bit)
- Cost per field operation: ~150 RISC-V cycles (Montgomery multiplication)
- Additional overhead for polynomial evaluations and 109-instance batching: ~16,000× scaling
- **Estimated total**: ~240M cycles

### Component C: Hyrax Opening Proof Verification

The final component verifies that the batched polynomial commitment (from Component A) opens correctly at the challenge point determined by ExpSumcheck (from Component B).

#### The Hyrax Opening Protocol

**Setup from earlier components**:
- From Component A: Batched commitment $C_{\text{batched}}$ (4 Grumpkin points, one per row)
- From Component B: Opening point $r = [r_0, r_1, r_2, r_3]$ (the final sumcheck challenges)
- From Component B: Claimed evaluation $v$ (what the batched polynomial should equal at $r$)

**What the prover sends**: Vector $u = [u_0, u_1, u_2, u_3]$ (4 field elements)

**What the verifier checks**: Two things that should be equal if the opening is correct:
1. Commitment derived homomorphically from row commitments
2. Commitment to the proof vector $u$

#### Toy Example (2×2 matrix, 2-variable MLE)

Let's use a tiny example to understand the verification:

**Committed polynomial**: $f(x_0, x_1)$ with 4 coefficients arranged as 2×2 matrix:
$$M = \begin{bmatrix} c_{00} & c_{01} \\ c_{10} & c_{11} \end{bmatrix}$$

**Row commitments** (what was sent in Component A):
- $C_0 = c_{00} \cdot G_0 + c_{01} \cdot G_1$ (commitment to row 0)
- $C_1 = c_{10} \cdot G_0 + c_{11} \cdot G_1$ (commitment to row 1)

**Opening point**: $r = [r_0, r_1]$ (from sumcheck)

**Claimed evaluation**: $v = c_{00}(1-r_0)(1-r_1) + c_{01}(1-r_0)r_1 + c_{10}r_0(1-r_1) + c_{11}r_0 r_1$

**What $L$ is**: The equality polynomial over the FIRST variable (row index):
$$L = [\text{eq}(r_0, 0), \text{eq}(r_0, 1)] = [(1-r_0), r_0]$$

This tells us how to combine the row commitments based on which row we're "at" when evaluating at $r_0$.

**What $R$ is**: The equality polynomial over the SECOND variable (column index):
$$R = [\text{eq}(r_1, 0), \text{eq}(r_1, 1)] = [(1-r_1), r_1]$$

This tells us how to combine columns within each row.

**MSM #1** (Homomorphically derive commitment):
$$C_{\text{derived}} = L_0 \cdot C_0 + L_1 \cdot C_1 = (1-r_0) \cdot C_0 + r_0 \cdot C_1$$

**Intuition**: MSM #1 computes "what the commitment would be to the interpolated row at position $r_0$". The weights $(1-r_0)$ and $r_0$ tell us how much each row contributes:
- If $r_0 = 0$: We're at row 0 (weight 100% on $C_0$, 0% on $C_1$)
- If $r_0 = 1$: We're at row 1 (weight 0% on $C_0$, 100% on $C_1$)
- Otherwise: Interpolate between rows

Expanding:
$$C_{\text{derived}} = (1-r_0)[c_{00} G_0 + c_{01} G_1] + r_0[c_{10} G_0 + c_{11} G_1]$$
$$= [(1-r_0)c_{00} + r_0 c_{10}] G_0 + [(1-r_0)c_{01} + r_0 c_{11}] G_1$$

This is a commitment to the interpolated row: $[(1-r_0)c_{00} + r_0 c_{10}, (1-r_0)c_{01} + r_0 c_{11}]$.

**What $u$ is**: The result of multiplying the coefficient matrix by $R$:
$$u = M \cdot R = \begin{bmatrix} c_{00} & c_{01} \\ c_{10} & c_{11} \end{bmatrix} \begin{bmatrix} 1-r_1 \\ r_1 \end{bmatrix} = \begin{bmatrix} c_{00}(1-r_1) + c_{01}r_1 \\ c_{10}(1-r_1) + c_{11}r_1 \end{bmatrix}$$

So: $u_0 = c_{00}(1-r_1) + c_{01}r_1$ and $u_1 = c_{10}(1-r_1) + c_{11}r_1$

**MSM #2** (Commitment to proof vector):
$$C_{\text{product}} = u_0 \cdot G_0 + u_1 \cdot G_1$$
$$= [c_{00}(1-r_1) + c_{01}r_1] G_0 + [c_{10}(1-r_1) + c_{11}r_1] G_1$$

**Check 1**: $C_{\text{derived}} \stackrel{?}{=} C_{\text{product}}$

Substituting:
$$[(1-r_0)c_{00} + r_0 c_{10}] G_0 + [(1-r_0)c_{01} + r_0 c_{11}] G_1$$
$$\stackrel{?}{=} [c_{00}(1-r_1) + c_{01}r_1] G_0 + [c_{10}(1-r_1) + c_{11}r_1] G_1$$

This checks if $(1-r_0) \cdot (\text{row 0}) + r_0 \cdot (\text{row 1}) = \text{each row combined with } R$.

**Check 2**: $\langle u, L \rangle \stackrel{?}{=} v$

$$u_0 \cdot (1-r_0) + u_1 \cdot r_0$$
$$= [c_{00}(1-r_1) + c_{01}r_1](1-r_0) + [c_{10}(1-r_1) + c_{11}r_1] r_0$$
$$= c_{00}(1-r_0)(1-r_1) + c_{01}(1-r_0)r_1 + c_{10}r_0(1-r_1) + c_{11}r_0 r_1 = v$$

**Why this works**: The tensor product structure $f(r_0, r_1) = \langle M, L \otimes R \rangle$ allows verification using only $O(\sqrt{N})$ operations.

#### Back to Stage 6 (4×4 matrix, 4-variable MLE)

For the actual 4-variable MLEs in Stage 6:
- Matrix dimensions: 4×4 (16 coefficients)
- $L = \text{eq}(r[0..2], \cdot)$: 4 elements (equality polynomial over first 2 variables, indexing rows)
- $R = \text{eq}(r[2..4], \cdot)$: 4 elements (equality polynomial over last 2 variables, indexing columns)
- Proof vector $u$: 4 elements (result of $M \cdot R$)

**MSM #1**: $C_{\text{derived}} = \sum_{i=0}^{3} L_i \cdot C_{\text{batched}}[i]$ (4-base MSM, ~1M cycles)

**MSM #2**: $C_{\text{product}} = \sum_{j=0}^{3} u_j \cdot G_j$ (4-base MSM, ~1M cycles)

**Check 1**: $C_{\text{derived}} \stackrel{?}{=} C_{\text{product}}$

**Check 2**: $\langle u, L \rangle \stackrel{?}{=} v$ (where $v$ is the claimed evaluation from ExpSumcheck)

**Total Hyrax opening cost**: ~2 million cycles (2 MSMs × 1M cycles each)

### Stage 6 Total Cost

| Component | Operations | Cost (cycles) | % of Stage 6 |
|-----------|------------|---------------|--------------|
| A. Hyrax Batch Commitment | 222,360-base MSM | ~80-120M | 24-30% |
| B. ExpSumcheck | 109 instances × 4 rounds | ~240M | 60-71% |
| C. Hyrax Opening Proof | 2 × 4-base MSMs | ~2M | <1% |
| **Total Stage 6** | | **~322-362M cycles** | **100%** |

## Complete Cost Comparison

### Baseline Verification (No Stage 6)

| Component | Cost (cycles) | % of Total |
|-----------|---------------|------------|
| Stages 1-4 sumchecks | ~20M | 2% |
| G1/G2 operations | ~50M | 4% |
| Stage 5 GT exponentiations (109) | ~1,090M | 91% |
| Stage 5 pairings (5) | ~100M | 8% |
| Miscellaneous | ~20M | 2% |
| **Total** | **~1,200M** | **100%** |

### Modified Verification (With Stage 6)

| Component | Cost (cycles) | % of Total |
|-----------|---------------|------------|
| Stages 1-4 sumchecks | ~20M | 5% |
| G1/G2 operations | ~50M | 12% |
| Stage 5 RLC (with hints) | ~2M | <1% |
| Stage 5 pairings (5) | ~100M | 25% |
| **Stage 6 (Hyrax approach)** | **~340M** | **85%** |
| Miscellaneous | ~20M | 5% |
| **Total** | **~400M** | **100%** |

**Overall speedup**: ~1.2B / ~400M = **~3× faster**

**Stage 5 savings**: ~1.1B - ~2M = **~1.1B cycles saved** (>99% reduction in GT exponentiation cost)

**Net savings**: ~1.2B - ~400M = **~800M cycles saved** (~67% reduction in total verification cost)

The key architectural shift is that Stage 6 becomes the dominant verification component (85% of total cost), but this cost is paid in efficient Grumpkin MSMs rather than expensive BN254 $\mathbb{G}_T$ exponentiations.

## Stage 6 Integration with Main Proof Flow

Stage 6 is seamlessly integrated into Jolt's existing proof DAG as an additional verification stage that only activates when the `recursion` feature flag is enabled.

### Prover Flow

**Stage 5 (Modified)** ([jolt_dag.rs](jolt-core/src/zkvm/dag/jolt_dag.rs:313-375)):

During the standard Stage 5 opening proof generation, the prover now captures witness data for all GT exponentiations:

1. **Witness capture**: As Stage 5 computes the 109 GT exponentiations for Dory's opening proof, it records all intermediate square-and-multiply steps in `ExponentiationSteps` structures
2. **Storage**: These witness structures are stored in the `ProverOpeningAccumulator` via `recursion_ops` field
3. **Code location**: `opening_proof.rs:1013-1057` - after Dory's `combine_commitments_with_hint()`, the exponentiation steps are extracted and stored

**Stage 6 (New)** ([jolt_dag.rs](jolt-core/src/zkvm/dag/jolt_dag.rs:313-375)):

```rust
#[cfg(feature = "recursion")]
{
    // 1. Retrieve captured witness from Stage 5
    let exps_to_prove = state_manager
        .get_prover_accumulator()
        .get_recursion_ops()
        .cloned()
        .unwrap_or_default();

    // 2. Load Hyrax generators (Grumpkin curve)
    let hyrax_generators = PedersenGenerators::<GrumpkinProjective>::from_urs_file(
        16,
        b"recursion check",
        Some("hyrax_urs_16.urs"),
    );

    // 3. Generate Stage 6 proof
    let recursion_proof = snark_composition_prove(
        exps_to_prove,
        transcript,
        &hyrax_generators,
    );

    // 4. Store in proof structure
    state_manager.proofs.insert(ProofKeys::Recursion, recursion_proof);
}
```

**Key points**:
- Stage 6 runs **after** Stage 5 completes
- Uses separate commitment scheme (Hyrax/Grumpkin) from main proof (Dory/BN254)
- Witness data (~28 MB for 109 exponentiations) is committed succinctly via Hyrax
- Total Stage 6 proving time: Dominated by 222K-base MSM (~80-120M cycles) + ExpSumcheck

### Verifier Flow

**Stage 5 (Modified)** ([jolt_dag.rs](jolt-core/src/zkvm/dag/jolt_dag.rs:546-548)):

In recursion mode, Stage 5 verification **skips GT exponentiation computation**:

1. **Hint acceptance**: Verifier accepts pre-computed GT results from prover's hints
2. **Deferred verification**: Does not verify these results directly (saves ~1.1B cycles)
3. **Trust but verify later**: Stage 6 will verify correctness of all accepted hints

**Stage 6 (New)** ([jolt_dag.rs](jolt-core/src/zkvm/dag/jolt_dag.rs:549-603)):

```rust
#[cfg(feature = "recursion")]
{
    // 1. Load Hyrax generators (same as prover)
    let hyrax_generators = PedersenGenerators::<GrumpkinProjective>::from_urs_file(
        16,
        b"recursion check",
        Some("hyrax_urs_16.urs"),
    );

    // 2. Extract Stage 6 proof from proof structure
    let recursion_proof = state_manager.proofs.get(ProofKeys::Recursion);

    // 3. Verify Stage 6 proof
    snark_composition_verify(
        recursion_proof,
        transcript,
        &hyrax_generators,
    )?;
}
```

**Verification components** (in order):
- Component A: Verify Hyrax batch commitment (~80-120M cycles)
- Component B: Verify ExpSumcheck (~240M cycles)
- Component C: Verify Hyrax opening proof (~2M cycles)
- **Total**: ~322-362M cycles

### Opening Accumulator Extension

The `ProverOpeningAccumulator` and `VerifierOpeningAccumulator` are extended to handle Stage 6 polynomials ([opening_proof.rs](jolt-core/src/poly/opening_proof.rs:698-814)):

**New methods**:
- `append_dense_recursion()`: Add Stage 6 polynomial openings (rho, quotient, base, g)
- `get_recursion_polynomial_opening()`: Retrieve claimed evaluation for Stage 6 polynomials
- `set_recursion_ops()` / `get_recursion_ops()`: Manage exponentiation witness storage

**New polynomial identifiers**:
```rust
pub enum RecursionCommittedPolynomial {
    RecursionRho(usize, usize),      // (exponentiation_idx, step_idx)
    RecursionQuotient(usize, usize), // (exponentiation_idx, step_idx)
    RecursionBase(usize),             // (exponentiation_idx)
    RecursionG(usize),                // (exponentiation_idx)
}
```

These track the 55,590 MLEs committed via Hyrax separately from the main trace polynomials committed via Dory.

### Feature Flag Behavior

**With `recursion` feature enabled**:
- Stage 5: Accepts hints, captures witness, defers verification
- Stage 6: Proves and verifies GT exponentiation correctness
- Total verification: ~400M cycles

**Without `recursion` feature**:
- Stage 5: Computes all 109 GT exponentiations directly
- Stage 6: Does not exist
- Total verification: ~1.2B cycles

The feature flag allows seamless switching between the optimized recursion-based approach and the traditional direct computation approach.

## Why This Works: The BN254-Grumpkin 2-Cycle

**Critical design choice**: Stage 6 uses **Hyrax commitments over Grumpkin**, not BN254. This section explains why this is essential for efficiency.

### The Field Matching Property

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

### Concrete Example: Committing to an Intermediate GT Value

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

### What Operations Use Which Curve?

| Operation | Curve/Field | Why |
|-----------|-------------|-----|
| **GT exponentiations** (actual computation) | BN254 $\mathbb{G}_T$ ($\mathbb{F}_{q^{12}}$) | Stage 5 - what we're trying to avoid in verification |
| **GT multiplications** (hint usage) | BN254 $\mathbb{G}_T$ ($\mathbb{F}_{q^{12}}$) | Stage 5 - verifier combines precomputed hints |
| **Committing to $\mathbb{F}_q$ coefficients** | **Grumpkin** (scalars in $\mathbb{F}_q$) | **Stage 6 - Hyrax commitments** |
| **ExpSumcheck constraints** | $\mathbb{F}_q$ arithmetic | Stage 6 - proving exponentiation steps |
| **Hyrax MSMs** | Grumpkin $\mathbb{G}_1$ | Stage 6 - commitment verification |

### Why Not BN254 for Everything?

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

### Performance Impact

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

### Summary: The Power of 2-Cycles

**The insight**: When proving properties of one field's elements, use commitments in a curve whose scalar field matches that field.

- **BN254**: Main Jolt proof (execution trace over BN254 scalar field $\mathbb{F}_r$)
- **Grumpkin**: Auxiliary witness (GT coefficients over BN254 base field $\mathbb{F}_q$)
- **Match**: Grumpkin scalar field = BN254 base field → **native arithmetic!**

This is called **SNARK composition with 2-cycles** - using two curves that complement each other's field structures.

### Why Not Dory for Stage 6?

Using Dory to commit to the Stage 6 witness would create a circular dependency: Dory verification requires $\mathbb{G}_T$ exponentiations, which is precisely what Stage 6 aims to verify. This would reintroduce the original problem.

Hyrax avoids this circularity by using only group operations (MSMs) in $\mathbb{G}_1$ of Grumpkin, which are far cheaper than $\mathbb{G}_T$ exponentiations in BN254.

### Why Not Another Pairing Curve?

Other pairing-friendly curves (BLS12-381, BW6-761) do not have the field-matching property with BN254. Using them would require expensive field conversions and eliminate the efficiency gains.

## Security Considerations

The security of the hint-based approach relies on the soundness of the ExpSumcheck protocol and the binding property of the Hyrax commitments.

### Soundness of ExpSumcheck

If a malicious prover provides incorrect exponentiation results, they must also provide witness polynomials that satisfy the square-and-multiply constraints for these incorrect results. The ExpSumcheck protocol ensures that:

1. The committed polynomials satisfy the constraints (via sumcheck soundness)
2. The constraints uniquely determine the exponentiation result (by completeness of square-and-multiply)

Therefore, an incorrect result would require breaking the soundness of the sumcheck protocol, which fails with probability at most $O(d/|\mathbb{F}|)$ where $d$ is the polynomial degree and $\mathbb{F}$ is the field. For degree-4 polynomials over a 254-bit field, this probability is negligible.

### Binding of Hyrax Commitments

The Hyrax commitment scheme's security relies on the discrete logarithm assumption over Grumpkin. If a prover could open a commitment to two different polynomials, they could extract discrete logarithms of the generators, breaking the assumption. Therefore, the committed witness polynomials are binding.

### Combined Security

The combination of binding commitments and sound constraint checking ensures that the verifier accepts incorrect exponentiation results only if:
1. The prover breaks Hyrax binding (breaking discrete log), OR
2. The prover breaks sumcheck soundness (negligible probability)

This maintains the same security level as the original Dory protocol.

## Implementation Locations

### Stage 5 Modifications

**Opening proof with hint capture** ([opening_proof.rs](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/poly/opening_proof.rs)):
- Lines 607-618: `ReducedOpeningProof` structure with `homomorphic_gt_results` field
- Lines 1007-1023: Capture Point 1 (RLC combining hints)
- Lines 1030-1064: Capture Point 2 (main Dory exponentiation steps)
- Lines 1494-1527: Hint usage in verification (`combine_commitments_with_hint`)

### Stage 6 Components

**Top-level orchestration** ([snark_composition.rs](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/subprotocols/snark_composition.rs)):
- Lines 177-386: Prover (witness generation and commitment)
- Lines 390-630: Verifier (batched verification logic)

**ExpSumcheck protocol** ([square_and_multiply.rs](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/subprotocols/square_and_multiply.rs)):
- Lines 24-36: `ExpProverState` structure
- Lines 115-142: Verifier initialization
- Lines 260-330: Expected output claim computation

**Hyrax commitment scheme** ([hyrax.rs](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/poly/commitment/hyrax.rs)):
- Lines 20-30: Matrix dimensions computation
- Lines 51-80: Single polynomial commitment
- Lines 180-246: Batch commitment (prover)
- Lines 298-367: Opening proof verification (verifier)

**DAG integration** ([jolt_dag.rs](https://github.com/a16z/jolt/blob/pr-975/jolt-core/src/zkvm/dag/jolt_dag.rs)):
- Lines 313-375: Prover Stage 6 execution
- Lines 549-604: Verifier Stage 6 execution

## Future Optimizations

Several potential improvements could further reduce Stage 6 costs:

### MSM Acceleration

Grumpkin MSMs could be accelerated through:
- **Specialized hardware**: FPGA or ASIC implementations of Pippenger's algorithm
- **Precomputation**: For fixed generators in Hyrax, precompute lookup tables
- **Endomorphism**: Exploit Grumpkin's curve structure for GLV-style optimizations

Estimated improvement: 3-10× reduction in MSM costs

### Batching Across Multiple Proofs

When verifying multiple Jolt proofs, their Stage 6 verifications could be batched:
- Combine all ExpSumcheck instances across proofs
- Perform a single batched Hyrax opening proof
- Amortize Pippenger setup costs

Estimated improvement: 2-5× amortized cost reduction for large batches

### Recursive Verification

The BN254-Grumpkin 2-cycle enables efficient recursion:
- Prove Stage 6 verification in another SNARK over BN254
- Compress multiple proofs into a single recursive proof
- Enable constant-time verification regardless of number of proofs

This is a more complex optimization requiring additional engineering but could enable constant verification cost.

### EVM Deployment Considerations

For on-chain verification, Stage 6 could leverage:
- **MSM precompiles**: EIP-6565 proposes efficient MSM precompiles for common curves
- **Grumpkin precompiles**: Future EIP could add native Grumpkin support
- **Batched verification**: Combine multiple proofs before on-chain verification

Estimated on-chain gas cost (with precompiles): ~5-10M gas (compared to 100M+ for baseline GT exponentiations)
