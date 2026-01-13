# Jagged Assist

## Overview

The **Jagged Assist** is an optimization technique from the "Jagged Polynomial Commitments" paper (Hemo et al., May 2025) that reduces verifier costs when dealing with a large number of columns in a jagged PCS.

## Background: The Jagged PCS

### The Problem

In zkVMs, computation traces are represented as multiple tables (one per CPU instruction), where each table has a different height. Committing to each column separately creates significant verification overhead:

- Verifier work scales linearly with the number of columns (2^k)
- In hash-based systems like FRI, this becomes the dominant cost
- Creates a trade-off: adding columns reduces prover costs but increases recursion costs

### The Jagged Solution

The jagged PCS allows committing to the entire trace as a **single dense polynomial** while enabling the verifier to emulate access to individual column polynomials.

**Key Definitions:**

- **Jagged function**: `p : {0,1}^n × {0,1}^k → F` where `p(x,y) = 0` for `x ≥ h_y` (column y has height h_y)
- **Dense representation**: `q : {0,1}^m → F` where `q(i) = p(row_t(i), col_t(i))`
- **Cumulative heights**: `t_y = h_1 + h_2 + ... + h_y` (monotonically non-decreasing)
- **Total trace area**: `M = Σ h_y = 2^m`

## Why the Jagged Assist is Needed

The basic jagged verifier must evaluate:

```
Σ_{y ∈ {0,1}^k} eq(z_c, y) · ĝ(z_r, i, t_{y-1}, t_y)
```

This requires **2^k evaluations of ĝ**. When k is large (many columns), this cost becomes prohibitive.

The function `g : {0,1}^{n+3m} → {0,1}` is defined as:
```
g(a, b, c, d) = 1  iff  b < d  AND  b = a + c
```

While g is computable by a width-4 read-once branching program (making single evaluations O(m)), doing 2^k of them is still expensive.

## The Jagged Assist Protocol

The Jagged Assist delegates the computation of multiple MLE evaluations to the prover, reducing the verifier's work to a **single evaluation of ĝ**.

### Core Idea

Instead of the verifier computing 2^k evaluations, the prover proves:
```
∀i ∈ [K]: ĥ(x_i) = y_i
```

The verifier reduces this to checking a single claim.

### The Sum-check Expression

**Step 1: Random Linear Combination**

Given K = 2^k claimed evaluations `(x_1, y_1), ..., (x_K, y_K)`, the verifier samples random `r_1, ..., r_k ∈ F` and reduces to:

```
Σ_{i ∈ [K]} r_i · h(x_i) = Σ_{i ∈ [K]} r_i · y_i
```

**Step 2: Rewrite Using MLE Definition**

```
Σ_{i ∈ [K]} r_i · Σ_{b ∈ {0,1}^m} eq(b, x_i) · h(b)  =  Σ_{b ∈ {0,1}^m} h(b) · Σ_{i ∈ [K]} r_i · eq(b, x_i)
```

**Step 3: Sum-check on Product**

Run sumcheck on:
```
Σ_{b ∈ {0,1}^m} P(b)

where P(b) = h(b) · Σ_{i ∈ [K]} r_i · eq(b, x_i)
```

**Step 4: Final Verification**

At the end, the verifier checks:
```
P(ρ) = h(ρ) · Σ_{i ∈ [K]} r_i · eq(ρ, x_i)
```

This requires only **one evaluation of h** plus O(k·m) arithmetic for the eq computations.

### Prover's Round-j Message

In round j, the prover computes for each λ ∈ Λ (a set of 3 distinct field elements):

```
Σ_{b ∈ {0,1}^{m-j}} P(ρ, λ, b) = Σ_{i ∈ [K]} r_i · eq((ρ,λ), x_i[1,...,j]) · h(ρ, λ, x_i[j+1,...,m])
```

## Complexity Analysis

### Basic Jagged (without assist)
- **Prover**: O(2^m) field ops, at most 5·2^m + 2^n + 2^k multiplications
- **Verifier**: O(m · 2^k) arithmetic circuit

### With Jagged Assist (Theorem 1.5)
- **Prover**:
  - Evaluates h on O(m · K) points
  - O(m · K) additional arithmetic operations
  - For width-w ROBP: O(m · w² · (K + w)) operations
- **Verifier**:
  - Single evaluation of ĥ
  - O(m · K) additional arithmetic operations
- **Soundness error**: 2m / |F|

### Key Insight for ROBPs

When h is computable by a width-w read-once branching program, the prover can use **Lemma 4.6** to generate all required evaluations efficiently:

```
For inputs x_1, ..., x_k and streaming access to ρ:
After reading bit (j-1) of ρ, output:
  { h(ρ[1,...,j-1], λ, x_i[j+1,...,m]) }_{λ ∈ Λ}

Total cost: O(n · w² · (k · 2^b + |Λ| · w))
```

For our function g with width w=4 and b=4 (reading 4 bits at a time: a, b, c, d):
- The assist adds only O(m² · 2^k) work to the prover
- This is negligible compared to the main sumcheck cost

## Fancy Jagged (Section 6)

For tables (groups of columns with same height) with power-of-2 widths:
- Verifier work proportional to number of **tables**, not columns
- Uses a width-6 branching program (slightly larger than basic width-4)
- Cost factor: (6/4)² = 2.25× larger per evaluation

---

## Application to Jolt Stage 3 (Recursion)

### Current Stage 3 Structure

Jolt's recursion pipeline has three stages:
- **Stage 1**: Constraint sumchecks (GT Exp, GT Mul, G1 Scalar Mul)
- **Stage 2**: Virtualization sumcheck → produces claim `M(r_s, r_x) = v_sparse`
- **Stage 3**: Jagged transform sumcheck → produces claim `q(r_dense) = v_dense`

The Stage 3 verifier must compute the jagged indicator:

```
f̂_jagged(r_s, r_x, r_dense) = Σ_{y ∈ {0,1}^{num_s_vars}} eq(r_s, y) · ĝ(r_x, r_dense, t_{y-1}, t_y)
```

Where:
- `K = 2^{num_s_vars}` can be **16,000+ rows** (15 poly types × num_constraints)
- Each iteration requires a branching program evaluation: O(64 × num_bits)
- **Current cost: ~21M+ field operations**

### The Jagged Assist as "Stage 3b"

The Jagged Assist adds a **new sumcheck phase** that batch-proves all K evaluations of ĝ, reducing the verifier to a single ĝ evaluation.

### Does It Need a PCS?

**No.** The Jagged Assist is a **pure sumcheck protocol**. Here's why:

| Component | Needs PCS? | Reason |
|-----------|------------|--------|
| Dense polynomial `q` | **Yes** | Committed via Dory (existing) |
| Jagged indicator `f̂_jagged` | No | Computed by verifier from public data |
| Branching program `ĝ` | No | Public function, verifier can evaluate directly |
| **Jagged Assist** | **No** | Pure sumcheck, soundness from protocol |

The function `ĝ` is publicly defined:
```
g(a, b, c, d) = 1  iff  b < d  AND  b = a + c
```

The verifier can evaluate it at any point using `JaggedBranchingProgram::eval_multilinear` in O(num_bits × w²) time. No commitment required.

### Protocol Flow (Stage 3 + Jagged Assist)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Jagged Transform Sumcheck                                         │
│                                                                             │
│  Input:  M(r_s, r_x) = v_sparse  (from Stage 2)                            │
│  Prove:  v_sparse = Σ_i q(i) · f_jagged(r_s, r_x, i)                       │
│  Output: q(r_dense) = v_dense   (goes to PCS)                              │
│          + need to verify f̂_jagged(r_s, r_x, r_dense)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3b: Jagged Assist (Batch MLE Verification)                          │
│                                                                             │
│  Problem: Verifier needs K = 2^{num_s_vars} evaluations of ĝ               │
│                                                                             │
│  Current (without assist):                                                  │
│    for y in 0..K:                                                          │
│        f̂_jagged += eq(r_s, y) · ĝ(r_x, r_dense, t_{y-1}, t_y)             │
│    Cost: K × O(num_bits × w²) ≈ 21M field ops                              │
│                                                                             │
│  With Jagged Assist:                                                        │
│    1. Prover claims: v_y = ĝ(r_x, r_dense, t_{y-1}, t_y) for each y        │
│    2. Run batch verification sumcheck (see below)                          │
│    3. Verifier does ONE ĝ evaluation at random point                       │
│    Cost: O(m × K) + O(num_bits × w²) ≈ 330K field ops                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PCS Opening (Dory)                                                         │
│                                                                             │
│  Verify: q(r_dense) = v_dense                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Jagged Assist Sumcheck Flow

**Setup:**
- K claimed evaluations: `(x_y, v_y)` where `x_y = (r_x, r_dense, t_{y-1}, t_y)`
- Claim: `ĝ(x_y) = v_y` for all `y ∈ [K]`

**Step 1: Verifier samples randomness**
```
Sample r_1, ..., r_K ∈ F  (or use powers: r_y = r^y for efficiency)
```

**Step 2: Reduce to single sum**
```
Original:  ∀y: ĝ(x_y) = v_y

Batched:   Σ_y r_y · ĝ(x_y) = Σ_y r_y · v_y
                   ↑                    ↑
              prover proves      verifier computes directly (O(K) ops)
```

**Step 3: Rewrite LHS using MLE definition**
```
Σ_y r_y · ĝ(x_y) = Σ_y r_y · Σ_{b ∈ {0,1}^m} eq(b, x_y) · g(b)
                 = Σ_{b ∈ {0,1}^m} g(b) · Σ_y r_y · eq(b, x_y)
                 = Σ_{b ∈ {0,1}^m} P(b)

where P(b) = g(b) · Σ_y r_y · eq(b, x_y)
```

**Step 4: Run sumcheck on P**
```
Sumcheck proves: Σ_{b ∈ {0,1}^m} P(b) = claimed_sum

After m rounds, verifier has random point ρ ∈ F^m
```

**Step 5: Verifier checks final claim**
```
Verifier computes:
  1. ĝ(ρ)                      ← ONE branching program evaluation, O(m × w²)
  2. Σ_y r_y · eq(ρ, x_y)      ← O(K × m) field operations

Accepts iff: P(ρ) = ĝ(ρ) · Σ_y r_y · eq(ρ, x_y)
```

### Why This Works (Soundness)

The sumcheck protocol guarantees that if the prover cheats on any `v_y`:
1. The batched sum `Σ_y r_y · v_y` will (w.h.p.) not equal `Σ_y r_y · ĝ(x_y)`
2. The sumcheck will produce a random point ρ where the claimed polynomial disagrees
3. The final check `P(ρ) = ĝ(ρ) · Σ_y r_y · eq(ρ, x_y)` will fail

**Soundness error:** `2m / |F|` (negligible for large fields)

### Integration with Existing Code

The Jagged Assist integrates with `stage3/jagged.rs`:

```rust
// Current: JaggedSumcheckVerifier::expected_output_claim
// Loops over K rows, each doing a branching program eval

// With Jagged Assist:
// 1. Prover sends claimed v_y values (or derives from sumcheck)
// 2. New sumcheck phase (JaggedAssistSumcheck)
// 3. Verifier does ONE branching_program.eval_multilinear() call
```

**Key files to modify:**
- `stage3/jagged.rs` - Add assist sumcheck after main sumcheck
- `stage3/mod.rs` - Export new types
- `recursion_prover.rs` - Prover generates assist proof
- `recursion_verifier.rs` - Verifier runs assist verification

### Cost Comparison

| Metric | Without Assist | With Assist | Improvement |
|--------|----------------|-------------|-------------|
| Verifier field ops | K × 1,300 ≈ 21M | 330K | **~64×** |
| Verifier circuit size | O(K × m × w²) | O(m × K) + O(m × w²) | **~64×** |
| Prover extra work | 0 | O(m × w² × K) ≈ 5M | Negligible |
| PCS commitments | 1 (dense q) | 1 (same) | No change |
| Rounds | m | 2m | 2× more rounds |

### Prover Efficiency (Lemma 4.6)

The prover can generate sumcheck messages efficiently because `g` is a width-4 ROBP:

```
Standard approach: Evaluate ĝ at O(m × K) points naively
                   Cost: O(m × K × 2^m) - INFEASIBLE

ROBP approach:     Use streaming algorithm from Lemma 4.6
                   Precompute backward matrices A_j^(y) for each y
                   Stream through sumcheck challenges
                   Cost: O(m × w² × K) - EFFICIENT
```

This is why the Jagged Assist is practical - the structured nature of `g` (being an ROBP) makes the prover's work tractable.

---

## Implementation Summary

### Chosen Approach: Explicit Claims with Random Batching (Theorem 1.5)

The implementation follows **Theorem 1.5 / Lemma 5.1** from the paper exactly:

**What the prover sends:**
- K field elements: `v_0, v_1, ..., v_{K-1}` where `v_y = ĝ(r_x, r_dense, t_{y-1}, t_y)`
- The sumcheck proof for the batch verification

**Why random batching is essential for soundness:**

The verifier samples random coefficients `r_1, ..., r_K` **after** seeing the prover's claimed values. This is critical because:

1. The batch verification reduces K claims to one: `Σ_y r_y · ĝ(x_y) = Σ_y r_y · v_y`
2. If the prover cheats on even one `v_y`, the LHS and RHS differ by a non-zero polynomial in the `r_y` variables
3. By Schwartz-Zippel, random `r_y` values will detect this with high probability

**Alternative considered and rejected:**

We considered deriving the input claim implicitly from Stage 3's output (avoiding sending the `v_y` values), but this would use fixed coefficients `eq(r_s, y)` instead of random ones. This breaks soundness because a cheating prover could solve the single linear constraint `Σ_y eq(r_s, y) · v_y = target` with many wrong `v_y` values.

### Complete Protocol Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  After Stage 3 completes:                                                   │
│  - Verifier has: r_s, r_x, r_dense                                         │
│  - Verifier needs: f̂_jagged(r_s, r_x, r_dense) = Σ_y eq(r_s,y) · ĝ(x_y)   │
│  - Where x_y = (r_x, r_dense, t_{y-1}, t_y) for each y ∈ [K]               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: Prover sends claimed evaluations                                   │
│                                                                             │
│  Prover computes and sends: v_0, v_1, ..., v_{K-1}                         │
│  where v_y = ĝ(r_x, r_dense, t_{y-1}, t_y)                                 │
│                                                                             │
│  Proof size overhead: K field elements                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: Verifier samples batching randomness                               │
│                                                                             │
│  After seeing v_y, verifier samples r ∈ F (via Fiat-Shamir)                │
│  Sets batching coefficients: r_y = r^y  (powers for efficiency)            │
│                                                                             │
│  Both parties compute: claimed_sum = Σ_y r_y · v_y                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: Batch verification sumcheck                                        │
│                                                                             │
│  Prove: Σ_{b ∈ {0,1}^m} P(b) = claimed_sum                                 │
│  where P(b) = g(b) · Σ_y r_y · eq(b, x_y)                                  │
│                                                                             │
│  Sumcheck runs for m rounds, producing random point ρ ∈ F^m                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4: Final verification                                                 │
│                                                                             │
│  Verifier computes:                                                         │
│    1. ĝ(ρ) via branching program        ← O(m × w²) = O(m × 16) ops       │
│    2. eq_sum = Σ_y r_y · eq(ρ, x_y)     ← O(K × m) ops                    │
│                                                                             │
│  Accepts iff: final_claim = ĝ(ρ) · eq_sum                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 5: Compute f̂_jagged for Stage 3                                      │
│                                                                             │
│  Using the (now verified) v_y values:                                       │
│  f̂_jagged = Σ_y eq(r_s, y) · v_y                                          │
│                                                                             │
│  This is O(K) field operations                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proof Components

| Component | Size | Notes |
|-----------|------|-------|
| Claimed evaluations | K field elements | Sent by prover |
| Sumcheck proof | m univariate polynomials | Degree 2 (product of two linears) |
| Total overhead | K + 3m field elements | Much smaller than 21M ops saved |

### Rust Implementation Sketch

```rust
// In recursion_prover.rs
pub struct JaggedAssistProof<F: JoltField> {
    /// Claimed evaluations v_y = ĝ(r_x, r_dense, t_{y-1}, t_y)
    pub claimed_evaluations: Vec<F>,
    /// Sumcheck proof for batch verification
    pub sumcheck_proof: SumcheckInstanceProof<F>,
}

// In recursion_verifier.rs
impl JaggedAssistVerifier {
    pub fn verify(&self, proof: &JaggedAssistProof, ...) -> Result<F, ProofVerifyError> {
        // 1. Append claimed_evaluations to transcript
        for v in &proof.claimed_evaluations {
            transcript.append_scalar(v);
        }

        // 2. Sample batching randomness
        let r: F = transcript.challenge_scalar();
        let r_powers: Vec<F> = compute_powers(r, K);

        // 3. Compute expected sum
        let claimed_sum: F = r_powers.iter()
            .zip(&proof.claimed_evaluations)
            .map(|(r_y, v_y)| *r_y * v_y)
            .sum();

        // 4. Verify sumcheck
        let (final_claim, rho) = proof.sumcheck_proof.verify(claimed_sum, m, 2, transcript)?;

        // 5. Final verification: one branching program eval
        let g_at_rho = branching_program.eval_multilinear(&rho);
        let eq_sum = compute_batched_eq(&r_powers, &rho, &evaluation_points);

        if final_claim != g_at_rho * eq_sum {
            return Err(ProofVerifyError::InvalidProof);
        }

        // 6. Compute f_jagged using verified v_y values
        let f_jagged: F = eq_r_s.iter()
            .zip(&proof.claimed_evaluations)
            .map(|(eq_y, v_y)| *eq_y * v_y)
            .sum();

        Ok(f_jagged)
    }
}
```

### Key Implementation Notes

1. **Transcript ordering**: The claimed evaluations MUST be appended to the transcript BEFORE sampling the batching randomness. This ensures the prover commits to values before learning the random coefficients.

2. **Power-of-r batching**: Using `r_y = r^y` instead of K independent random values reduces transcript overhead while maintaining soundness (as long as K < |F|).

3. **Integration with BatchedSumcheck**: The Jagged Assist sumcheck can use the existing `BatchedSumcheck` infrastructure. The `SumcheckInstanceProver` trait works directly - `cache_openings` is a no-op since there's no committed polynomial.

4. **Evaluation points structure**: Each `x_y = (r_x, r_dense, t_{y-1}, t_y)` where the cumulative heights `t_y` are public constants derived from the jagged layout.

---

## Prover Implementation: `compute_message` via ROBP Streaming (Lemma 4.6)

### The Sumcheck Being Proved

The Jagged Assist sumcheck proves:

```
Σ_{b ∈ {0,1}^{4n}} P(b) = claimed_sum

where P(b) = g(b) · Q(b)
      g(a,b,c,d) = 1 iff b < d AND b = a + c
      Q(b) = Σ_y r^y · eq(b, x_y)
```

Here:
- `b = (a, b, c, d)` is a point in `{0,1}^{4n}` where `n = num_bits`
- `g` is the branching program function (boolean on hypercube, MLE is ĝ)
- `x_y = (r_x, r_dense, t_{y-1}, t_y)` for each row `y`
- `r^y` are the batching coefficients

### Sumcheck Round Structure

In round `j`, the prover computes univariate polynomial `h_j(X)`:

```
h_j(X) = Σ_{b' ∈ {0,1}^{4n-j-1}} P(ρ[0..j-1], X, b')
       = Σ_{b'} ĝ(ρ[0..j-1], X, b') · Q(ρ[0..j-1], X, b')
```

For degree-2 polynomial, we evaluate at `λ ∈ {0, 1, 2}`:
```
h_j(λ) = Σ_{b' ∈ {0,1}^{4n-j-1}} ĝ(ρ[0..j-1], λ, b') · Q(ρ[0..j-1], λ, b')
```

### Why ROBP Streaming is Essential

**The challenge:** Direct computation of `h_j(λ)` requires summing over `2^{4n-j-1}` terms.
For n=16 bits, this is `2^63` terms per evaluation - completely infeasible.

**The solution (Lemma 4.6):** Because `g` is computed by a width-4 read-once branching
program (ROBP), we can evaluate ĝ at partial points using the **forward-backward
decomposition**:

```
ĝ(prefix, λ, suffix) = Σ_{s,s'} forward[s] · T_λ[s,s'] · backward[s']
```

Where:
- `forward[s]` = probability of reaching state `s` after processing `prefix`
- `T_λ[s,s']` = MLE of transition from state `s` to `s'` on input `λ`
- `backward[s']` = probability of reaching accept state from `s'` given `suffix`

### ROBP Review: The Function g

The function `g(a, b, c, d) = 1 iff b < d AND b = a + c` is computed by a width-4
ROBP that reads one bit from each of (a, b, c, d) at each layer:

**States (2 bits = 4 states):**
- Bit 0: `carry` from addition `a + c`
- Bit 1: `b_less_than_d_so_far` (comparison status)

**Initial state:** `(carry=0, comparison=0)`
**Accept state:** `(carry=0, comparison=1)` - no carry and b < d

**Transition function:** At layer `i`, read bits `(a_i, b_i, c_i, d_i)`:
```
sum = a_i + c_i + carry
expected_b = sum mod 2
new_carry = sum >= 2

if b_i != expected_b:
    FAIL (reject)
else:
    if b_i != d_i:
        new_comparison = (d_i == 1)  // d_i > b_i means b < d at this bit
    else:
        new_comparison = comparison  // equal bits, keep previous
```

### Forward-Backward Decomposition

**Key insight from the paper:** For a width-w ROBP processing n bits:

```
ĥ(z₁, ..., zₙ) = Σ_{s ∈ [w]} forward_n[s] · accept[s]

where forward_j[s] = Pr[reach state s after processing z₁, ..., z_j]
```

More specifically, if we split the input at position j:
```
ĥ(z₁, ..., zₙ) = Σ_{s, s'} forward_j[s] · T_j(z_j)[s, s'] · backward_{j+1}[s']
```

Where:
- `forward_j[s]` depends only on `(z₁, ..., z_{j-1})`
- `T_j(z_j)[s, s']` is the MLE of the transition matrix at layer j
- `backward_{j+1}[s']` depends only on `(z_{j+1}, ..., zₙ)`

### Computing the Transition Matrix MLE

The transition matrix at layer `j` depends on the 4 input bits `(a_j, b_j, c_j, d_j)`.

For boolean inputs, `T[s, s']` is 1 if the transition from state `s` to state `s'`
is valid for those input bits, and 0 otherwise.

The MLE extends this to field elements:

```rust
/// Compute T(z)[s, s'] for field element z = (za, zb, zc, zd)
fn transition_mle<F: JoltField>(
    za: F, zb: F, zc: F, zd: F,
    from_state: MemoryState,
) -> [F; 4] {
    // Sum over all 16 possible bit combinations
    let mut result = [F::zero(); 4];

    for bits in 0..16 {
        let a_bit = (bits & 1) != 0;
        let b_bit = ((bits >> 1) & 1) != 0;
        let c_bit = ((bits >> 2) & 1) != 0;
        let d_bit = ((bits >> 3) & 1) != 0;

        // Check if this transition is valid
        if let StateOrFail::State(to_state) = transition_function(
            BitState { a_bit, b_bit, c_bit, d_bit },
            from_state
        ) {
            // eq((a_bit, b_bit, c_bit, d_bit), (za, zb, zc, zd))
            let eq_val = eq_bit(a_bit, za)
                       * eq_bit(b_bit, zb)
                       * eq_bit(c_bit, zc)
                       * eq_bit(d_bit, zd);

            result[to_state.to_index()] += eq_val;
        }
    }
    result
}

fn eq_bit<F: JoltField>(bit: bool, z: F) -> F {
    if bit { z } else { F::one() - z }
}
```

### Backward Precomputation

For each evaluation point `x_y = (r_x, r_dense, t_{y-1}, t_y)`, we precompute
the backward vectors. Note that:
- Coordinates `c` (t_{y-1}) and `d` (t_y) are **known constants** (integers)
- Coordinates `a` (r_x) and `b` (r_dense) are **field elements** from challenges

The backward computation depends on which variable we're in:

```rust
/// Precompute backward vectors for all layers
fn precompute_backward<F: JoltField>(
    r_x: &[F],       // a coordinates (field elements)
    r_dense: &[F],   // b coordinates (field elements)
    t_prev: usize,   // c coordinates (integer -> bits)
    t_curr: usize,   // d coordinates (integer -> bits)
    num_bits: usize,
) -> Vec<[F; 4]> {
    let num_layers = 4 * num_bits;
    let mut backward = vec![[F::zero(); 4]; num_layers + 1];

    // Initialize: accept state gets 1, others get 0
    backward[num_layers][MemoryState::success().to_index()] = F::one();

    // Work backwards
    for layer in (0..num_layers).rev() {
        // Determine which coordinate this layer corresponds to
        let (coord_type, bit_idx) = get_coordinate_info(layer, num_bits);

        // Get the z value for this coordinate
        let z = match coord_type {
            CoordType::A => r_x.get(bit_idx).cloned().unwrap_or(F::zero()),
            CoordType::B => r_dense.get(bit_idx).cloned().unwrap_or(F::zero()),
            CoordType::C => bit_to_field::<F>((t_prev >> bit_idx) & 1),
            CoordType::D => bit_to_field::<F>((t_curr >> bit_idx) & 1),
        };

        // backward[layer][s] = Σ_{s'} T(layer, z)[s, s'] · backward[layer+1][s']
        for s in 0..4 {
            let transition = transition_mle_single(z, MemoryState::from_index(s), coord_type);
            backward[layer][s] = F::zero();
            for s_prime in 0..4 {
                backward[layer][s] += transition[s_prime] * backward[layer + 1][s_prime];
            }
        }
    }

    backward
}
```

### Variable Ordering in the Sumcheck

The sumcheck runs over 4n variables. We need to define the ordering:

**Option A: Interleaved** - `(a₀, b₀, c₀, d₀, a₁, b₁, c₁, d₁, ...)`
- Matches natural ROBP layer structure
- Each layer processes one (a,b,c,d) bit tuple

**Option B: Concatenated** - `(a₀, a₁, ..., a_{n-1}, b₀, ..., d_{n-1})`
- Simpler coordinate indexing
- Need to handle ROBP layers differently

**Recommended: Concatenated** (matches current branching_program.rs)

With concatenated ordering:
- Variables 0..n-1 are `a` coordinates → come from `r_x`
- Variables n..2n-1 are `b` coordinates → come from `r_dense`
- Variables 2n..3n-1 are `c` coordinates → come from `t_prev` bits
- Variables 3n..4n-1 are `d` coordinates → come from `t_curr` bits

### Forward State Updates

During the sumcheck, as we bind variables to challenges, we update the forward state:

```rust
fn update_forward_for_challenge<F: JoltField>(
    forward: &mut [F; 4],
    challenge: F,
    layer: usize,
    num_bits: usize,
    coord_type: CoordType,
) {
    let mut new_forward = [F::zero(); 4];

    for s in 0..4 {
        if forward[s].is_zero() { continue; }

        // Compute transition MLE for this state
        let transition = transition_mle_single(challenge, MemoryState::from_index(s), coord_type);

        for s_prime in 0..4 {
            new_forward[s_prime] += forward[s] * transition[s_prime];
        }
    }

    *forward = new_forward;
}
```

### Complete compute_message Algorithm

```rust
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for JaggedAssistProver<F, T> {
    fn compute_message(&mut self, round: usize, _previous_claim: F) -> UniPoly<F> {
        let num_bits = self.params.num_bits;
        let (coord_type, bit_idx) = get_coordinate_info(round, num_bits);

        // Evaluate at λ ∈ {0, 1, 2}
        let mut evals = [F::zero(); 3];

        for lambda_idx in 0..3 {
            let lambda = F::from_u64(lambda_idx as u64);

            // Sum over all K evaluation points
            for y in 0..self.params.num_rows {
                // Get the z-coordinate value for this evaluation point
                let z_y = self.get_z_coordinate(y, round);

                // Compute ĝ contribution using forward-backward decomposition
                // ĝ(bound_prefix, λ, suffix_from_x_y) = forward · T(λ) · backward
                let g_contrib = self.compute_g_partial(y, lambda, coord_type);

                // Compute Q contribution: r^y · eq(bound_prefix, x_y_prefix) · eq(λ, z_y)
                let eq_lambda = eq_field(lambda, z_y);
                let q_contrib = self.r_powers[y] * self.eq_cache[y] * eq_lambda;

                // For boolean suffix variables, multiply by eq_suffix (precomputed in backward)
                // This is already factored into backward computation

                evals[lambda_idx] += g_contrib * q_contrib;
            }
        }

        // Construct degree-2 polynomial from evaluations
        // h(0) = evals[0], h(1) = evals[1], h(2) = evals[2]
        UniPoly::from_evals(&evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        let r: F = r_j.into();
        let num_bits = self.params.num_bits;
        let (coord_type, _bit_idx) = get_coordinate_info(round, num_bits);

        // Update forward states for all evaluation points
        for y in 0..self.params.num_rows {
            update_forward_for_challenge(
                &mut self.forward_states[y],
                r,
                round,
                num_bits,
                coord_type,
            );
        }

        // Update eq_cache: eq(ρ[0..j], x_y[0..j]) *= eq(r_j, x_y[j])
        for y in 0..self.params.num_rows {
            let z_y = self.get_z_coordinate(y, round);
            self.eq_cache[y] *= eq_field(r, z_y);
        }

        self.round += 1;
    }
}
```

### Handling the ROBP Layer vs Sumcheck Variable Mismatch

**Critical issue:** The ROBP processes bits in order `(a₀,b₀,c₀,d₀), (a₁,b₁,c₁,d₁), ...`
but sumcheck variables are ordered `a₀,a₁,...,b₀,b₁,...,c₀,c₁,...,d₀,d₁,...`

**Solution: Deferred ROBP transitions**

Instead of maintaining forward/backward per ROBP layer, we track which bits have
been bound and compute the partial MLE differently:

```rust
/// Compute ĝ(bound_vars, λ, suffix) where:
/// - bound_vars are the sumcheck challenges so far
/// - λ is the current variable being evaluated
/// - suffix comes from the evaluation point x_y
fn compute_g_partial_with_deferred_transitions<F: JoltField>(
    bound_challenges: &[F],  // ρ[0..round]
    lambda: F,               // current variable
    eval_point: &JaggedAssistEvalPoint<F>,
    num_bits: usize,
    round: usize,
) -> F {
    // Build the full 4n-dimensional point by combining:
    // - bound_challenges for variables [0..round]
    // - lambda for variable [round]
    // - x_y values for variables [round+1..4n)

    // Then evaluate ĝ using the branching program's MLE evaluator
    // This leverages the existing eval_multilinear which does the
    // forward-backward computation internally

    let mut a_coords = vec![F::zero(); num_bits];
    let mut b_coords = vec![F::zero(); num_bits];
    let mut c_coords = vec![F::zero(); num_bits];
    let mut d_coords = vec![F::zero(); num_bits];

    // Fill in bound challenges + lambda + suffix from x_y
    for var in 0..4*num_bits {
        let (coord_type, bit_idx) = get_coordinate_info(var, num_bits);
        let value = if var < round {
            bound_challenges[var]
        } else if var == round {
            lambda
        } else {
            // Get from x_y
            match coord_type {
                CoordType::A => eval_point.r_x.get(bit_idx).cloned().unwrap_or(F::zero()),
                CoordType::B => eval_point.r_dense.get(bit_idx).cloned().unwrap_or(F::zero()),
                CoordType::C => bit_to_field((eval_point.t_prev >> bit_idx) & 1),
                CoordType::D => bit_to_field((eval_point.t_curr >> bit_idx) & 1),
            }
        };

        match coord_type {
            CoordType::A => a_coords[bit_idx] = value,
            CoordType::B => b_coords[bit_idx] = value,
            CoordType::C => c_coords[bit_idx] = value,
            CoordType::D => d_coords[bit_idx] = value,
        }
    }

    let branching_program = JaggedBranchingProgram::new(num_bits);
    branching_program.eval_multilinear(
        &Point::from(a_coords),
        &Point::from(b_coords),
        &Point::from(c_coords),
        &Point::from(d_coords),
    )
}
```

### Optimization: Caching ROBP Evaluations

The above approach calls `eval_multilinear` for each (y, λ) pair in each round.
That's `K × 3 × 4n` branching program evaluations.

**Optimization 1: Cache per-row ROBP state**

For each row y, precompute the backward vectors once and reuse across rounds.

**Optimization 2: Incremental forward updates**

Instead of rebuilding the full point each round, maintain a running forward state
that's updated as variables are bound.

**Optimization 3: Batch the 3 lambda evaluations**

Since λ ∈ {0, 1, 2}, and the ROBP is width-4, we can compute all three evaluations
with one pass through the transition logic.

### Cost Analysis

With the ROBP streaming approach:

| Operation | Per Round | Total |
|-----------|-----------|-------|
| Forward state updates | O(K × w²) = O(K × 16) | O(4n × K × 16) |
| Backward precomputation | - | O(4n × K × w²) = O(4n × K × 16) |
| G partial evaluations | O(K × 3 × w²) | O(4n × K × 48) |
| Eq cache updates | O(K) | O(4n × K) |

**Total: O(4n × K × w²) = O(64n × K)**

For K = 16,000 rows and n = 16 bits:
- **ROBP approach: ~16M field operations** ✓
- Naive approach: ~2^63 operations ✗

### Implementation Tasks

1. **Add coordinate info helper:**
   ```rust
   enum CoordType { A, B, C, D }
   fn get_coordinate_info(var_idx: usize, num_bits: usize) -> (CoordType, usize);
   ```

2. **Add transition MLE computation to branching_program.rs:**
   ```rust
   fn transition_mle_single(z: F, from_state: MemoryState, coord_type: CoordType) -> [F; 4];
   ```

3. **Add backward precomputation:**
   ```rust
   fn precompute_backward(...) -> Vec<[F; 4]>;
   ```

4. **Update JaggedAssistProver fields:**
   ```rust
   struct JaggedAssistProver<F, T> {
       // Existing fields...
       forward_states: Vec<[F; 4]>,        // Per-row forward state
       backward_states: Vec<Vec<[F; 4]>>,  // Per-row backward vectors
       bound_challenges: Vec<F>,           // Challenges bound so far
   }
   ```

5. **Implement efficient compute_message using forward-backward decomposition**

6. **Implement ingest_challenge to update forward states and eq_cache**

### Changes Needed for Integration

#### 1. Add to RecursionProof

```rust
// In recursion_prover.rs
pub struct RecursionProof<F, T, PCS> {
    pub stage1_proof: SumcheckInstanceProof<F, T>,
    pub stage2_proof: SumcheckInstanceProof<F, T>,
    pub stage3_proof: SumcheckInstanceProof<F, T>,
    pub jagged_assist_proof: JaggedAssistProof<F, T>,  // NEW
    pub opening_proof: PCS::Proof,
    // ...
}
```

#### 2. Add prove_jagged_assist to RecursionProver

```rust
// In recursion_prover.rs
impl RecursionProver {
    fn prove_jagged_assist<T: Transcript>(
        &self,
        transcript: &mut T,
        accumulator: &mut ProverOpeningAccumulator<F>,
        r_stage1: &[F::Challenge],  // r_x
        r_stage3: &[F::Challenge],  // r_dense
        row_cumulative_sizes: &[usize],
    ) -> Result<JaggedAssistProof<F, T>, Error> {
        let r_x: Vec<F> = r_stage1.iter().map(|c| (*c).into()).collect();
        let r_dense: Vec<F> = r_stage3.iter().map(|c| (*c).into()).collect();
        let num_bits = std::cmp::max(r_x.len(), r_dense.len());

        // Create prover (computes claimed_evaluations internally)
        let mut assist_prover = JaggedAssistProver::new(
            r_x,
            r_dense,
            row_cumulative_sizes,
            num_bits,
            transcript,
        );

        let claimed_evaluations = assist_prover.get_claimed_evaluations().to_vec();

        // Run sumcheck
        let (sumcheck_proof, _) = BatchedSumcheck::prove(
            vec![&mut assist_prover],
            accumulator,
            transcript,
        );

        Ok(JaggedAssistProof {
            claimed_evaluations,
            sumcheck_proof,
        })
    }
}
```

#### 3. Add verify_jagged_assist to RecursionVerifier

```rust
// In recursion_verifier.rs
impl RecursionVerifier {
    fn verify_jagged_assist<T: Transcript>(
        &self,
        proof: &JaggedAssistProof<F, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        r_stage1: &[F::Challenge],
        r_stage3: &[F::Challenge],
        row_cumulative_sizes: &[usize],
    ) -> Result<F, Error> {
        let r_x: Vec<F> = r_stage1.iter().map(|c| (*c).into()).collect();
        let r_dense: Vec<F> = r_stage3.iter().map(|c| (*c).into()).collect();
        let num_bits = std::cmp::max(r_x.len(), r_dense.len());

        // Create verifier
        let verifier = JaggedAssistVerifier::new(
            proof.claimed_evaluations.clone(),
            r_x.clone(),
            r_dense,
            row_cumulative_sizes,
            num_bits,
            transcript,
        );

        // Verify sumcheck
        let _r_assist = BatchedSumcheck::verify(
            &proof.sumcheck_proof,
            vec![&verifier],
            accumulator,
            transcript,
        )?;

        // Compute f_jagged using verified claimed_evaluations
        // eq_r_s[y] = eq(r_s, y) for all y
        let eq_r_s = self.compute_eq_r_s(&r_stage2);
        let f_jagged = verifier.compute_f_jagged(&eq_r_s);

        Ok(f_jagged)
    }
}
```

#### 4. Wire into Stage 3 Verifier

The Stage 3 verifier's `expected_output_claim` currently does the expensive K-iteration loop. With Jagged Assist:

```rust
// In jagged.rs - JaggedSumcheckVerifier
fn expected_output_claim(...) -> F {
    // OLD: expensive loop over K rows
    // for row_idx in 0..num_rows {
    //     f_jagged += eq(r_s, y) * branching_program.eval_multilinear(...);
    // }

    // NEW: f_jagged comes from Jagged Assist verification
    // The verifier receives it from verify_jagged_assist()
    dense_claim * self.f_jagged_from_assist
}
```

### File Changes Summary

| File | Change |
|------|--------|
| `stage3/jagged_assist.rs` | Implement `compute_message` (naive first, ROBP later) |
| `recursion_prover.rs` | Add `JaggedAssistProof` to `RecursionProof`, add `prove_jagged_assist` |
| `recursion_verifier.rs` | Add `verify_jagged_assist`, wire to Stage 3 |
| `stage3/jagged.rs` | Modify `expected_output_claim` to use pre-verified `f_jagged` |
| `stage3/mod.rs` | Already updated ✓ |

---

## References

- Theorem 1.4: Basic Jagged protocol
- Theorem 1.5: Jagged Assist for batch MLE proofs
- Lemma 5.1: Batch proving protocol details
- Lemma 4.6: Efficient ROBP evaluation for streaming inputs
- Section 6: Fancy Jagged for table-based structures
