# Making Jolt Zero-Knowledge via BlindFold

This document describes the approach to making Jolt zero-knowledge using the technique from the Vega paper ("Vega: Low-Latency Zero-Knowledge Proofs over Existing Credentials").

## Overview

The approach consists of three steps:

1. **Make all sumcheck prover messages hiding Pedersen commitments**
2. **Make Dory PCS zero-knowledge**
3. **Introduce a succinct Spartan R1CS that encodes sumcheck verification, and use BlindFold to hide the witness**

Steps 1 and 2 are straightforward. Step 3 is the core contribution and is detailed below.

## Core Insight

Instead of applying BlindFold to the original Jolt instance (which would be expensive), we:

1. Run the **non-ZK Jolt prover** on the original instance-witness pair (but with hiding commitments)
2. Write a **small R1CS circuit** that encodes only the **verifier's checks** from the Jolt sumchecks
3. Apply BlindFold to this small circuit

The verifier's checks in sumcheck-based protocols are only **O(log n)** operations, so this circuit is exponentially smaller than the original computation.

---

## Part 1: Succinct Verifier R1CS

### Making the Circuit Succinct: O(log n) Instead of O(n)

#### Problem: Witness Polynomials are O(n)-sized

In Jolt's sumchecks, the prover's first message includes commitments to large polynomials (witness, memory, etc.). Naively encoding "the committed value equals this polynomial" in R1CS would cost O(n) constraints.

#### Solution: Move Large Polynomial Evaluations Outside the Circuit

1. Prover sends commitment `C_w̃` to witness polynomial (as before)
2. Verifier's challenges `c₁, c₂, ...` are derived via Fiat-Shamir (depending on `C_w̃`)
3. At the end, the circuit only needs `y = w̃(r)` at a random point `r`
4. Instead of encoding `w̃` in the circuit:
   - Prover sends commitment `C_y = Com(y, ρ)` to the claimed evaluation
   - Circuit takes `y` as a **WITNESS** (not computed in-circuit)
   - Use the **ZK evaluation argument of Dory PCS** to prove: "C_y is a commitment to the evaluation of C_w̃ at point r"

This moves the O(n) work **outside** the R1CS circuit and into the PCS.

### What the Verifier Never Sees

The verifier should **never see the claimed evaluations directly**. Instead:

1. The prover sends a **blinded commitment** to the evaluation
2. The verifier uses the **ZK-Dory evaluation proof** to verify that this commitment contains the correct evaluation

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  PROVER has:                                                        │
│    • Committed polynomial C_w̃ (e.g., commitment to witness poly)   │
│    • Evaluation point r (public, from Fiat-Shamir)                  │
│    • Actual evaluation y = w̃(r) (SECRET)                           │
│                                                                     │
│  PROVER sends:                                                      │
│    • C_y = Com(y, ρ)  ← Blinded commitment to the evaluation        │
│                                                                     │
│  VERIFIER receives:                                                 │
│    • C_y (commitment only, NOT y itself)                            │
│                                                                     │
│  ZK-Dory proves:                                                    │
│    "C_y is a commitment to the evaluation of C_w̃ at point r"       │
│                                                                     │
│  Verifier learns: NOTHING about y, only that the relation holds     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Circuit Structure

The R1CS circuit takes the **evaluation values as witness** (not public inputs):

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Split-Committed R1CS Instance                   │
├─────────────────────────────────────────────────────────────────────┤
│ COMMITMENTS (all prover messages are commitments):                  │
│   W̄₁, W̄₂, ..., W̄ₖ         ← Commitments to round poly coefficients│
│   C_y₁, C_y₂, ..., C_yₘ     ← Commitments to claimed evaluations   │
│                                                                     │
│ PUBLIC INPUTS:                                                      │
│   c₁, c₂, ..., cₖ           ← Fiat-Shamir challenges only          │
│   r₁, r₂, ..., rₘ           ← Evaluation points only               │
│                                                                     │
│ NO claimed evaluations in public inputs!                            │
├─────────────────────────────────────────────────────────────────────┤
│                     Split-Committed R1CS Witness                    │
├─────────────────────────────────────────────────────────────────────┤
│ Round polynomial coefficients + randomness:                         │
│   W₁, rW₁, W₂, rW₂, ..., Wₖ, rWₖ                                   │
│                                                                     │
│ Claimed evaluations + randomness (HIDDEN from verifier):            │
│   y₁, ρ₁, y₂, ρ₂, ..., yₘ, ρₘ                                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        R1CS Constraints                             │
├─────────────────────────────────────────────────────────────────────┤
│ Sumcheck round checks (using witness coefficients):                 │
│   • gᵢ(0) + gᵢ(1) = claimed_sumᵢ₋₁                                 │
│   • claimed_sumᵢ = gᵢ(cᵢ)                                          │
│                                                                     │
│ Final checks (using witness evaluations yⱼ):                        │
│   • Algebraic identity that combines the yⱼ values                  │
│   • e.g., Σⱼ αʲ · yⱼ = expected_value (from sumcheck)              │
│                                                                     │
│ Commitment consistency (for evaluation commitments):                │
│   • C_yⱼ = Com(yⱼ, ρⱼ)  ← Verified via split-committed R1CS        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Commitment Verification

The commitment verification is **not done via R1CS constraints** — it's part of the **satisfaction definition** of split-committed R1CS itself.

An instance `(Ē, u, W̄₁, ..., W̄ₗ, x)` is satisfied by witness `(E, rE, W₁, rW₁, ..., Wₗ, rWₗ)` if:

```
1. Ē = Com(E, rE)                          ← Commitment checks
2. W̄ᵢ = Com(Wᵢ, rWᵢ)  for all i ∈ [ℓ]     ← Commitment checks
3. |Wᵢ| = s for all i ∈ [ℓ]
4. (A·Z) ∘ (B·Z) = u·(C·Z) + E              ← R1CS constraint check
```

The **instance** contains the commitments (which ARE the sumcheck prover messages), and the **witness** contains the openings (plaintext coefficients + randomness).

At the end of BlindFold, the verifier receives the **folded witness** and checks satisfaction against the **folded instance**, which includes verifying all commitment openings.

---

## Part 2: Split-Committed Relaxed R1CS

### Why Relaxed R1CS

Standard R1CS: `(A·Z) ∘ (B·Z) = C·Z`

This doesn't fold nicely because when you add two satisfying assignments, you get cross-terms. So we use **relaxed R1CS**:

```
(A·Z) ∘ (B·Z) = u·(C·Z) + E
```

Where:
- `u ∈ F` is a scalar (u=1 for non-relaxed)
- `E ∈ Fᵐ` is an error vector (E=0 for non-relaxed)

### Structure

**Instance** (public):
```
u = (Ē, u, W̄₁, ..., W̄ₗ, x)

where:
  Ē     = commitment to error vector E
  u     = scalar
  W̄ᵢ   = commitment to witness segment Wᵢ
  x     = public inputs
```

**Witness** (secret):
```
w = (E, rE, W₁, rW₁, ..., Wₗ, rWₗ)

where:
  E     = error vector
  rE    = randomness for Ē
  Wᵢ    = witness segment i
  rWᵢ   = randomness for W̄ᵢ
```

**Satisfaction**: `(u, w)` is satisfying if:
```
1. Ē = Com(E, rE)
2. W̄ᵢ = Com(Wᵢ, rWᵢ)  for all i
3. (A·Z) ∘ (B·Z) = u·(C·Z) + E

where Z = (W₁, ..., Wₗ, u, x)
```

---

## Part 3: Nova Folding Scheme

Given two instance-witness pairs `(u₁, w₁)` and `(u₂, w₂)`, we fold them into a single pair `(u', w')`.

### Step 1: Compute the Cross-Term T

```rust
fn compute_cross_term(
    A: &SparseMatrix, B: &SparseMatrix, C: &SparseMatrix,
    u1: &Instance, w1: &Witness,
    u2: &Instance, w2: &Witness,
) -> Vec<F> {
    // Form Z vectors
    let Z1 = concat(&w1.W, &[u1.u], &u1.x);  // (W₁, u₁, x₁)
    let Z2 = concat(&w2.W, &[u2.u], &u2.x);  // (W₂, u₂, x₂)

    // Compute matrix-vector products
    let AZ1 = A.mul_vector(&Z1);
    let BZ1 = B.mul_vector(&Z1);
    let CZ1 = C.mul_vector(&Z1);
    let AZ2 = A.mul_vector(&Z2);
    let BZ2 = B.mul_vector(&Z2);
    let CZ2 = C.mul_vector(&Z2);

    // Cross-term: T = (AZ₁ ∘ BZ₂) + (AZ₂ ∘ BZ₁) - u₁(CZ₂) - u₂(CZ₁)
    let mut T = vec![F::zero(); AZ1.len()];
    for i in 0..T.len() {
        T[i] = AZ1[i] * BZ2[i]
             + AZ2[i] * BZ1[i]
             - u1.u * CZ2[i]
             - u2.u * CZ1[i];
    }
    T
}
```

### Step 2: Prover Commits to Cross-Term

```rust
let rT = F::random();
let T_bar = pedersen_commit(&T, rT);
transcript.append(T_bar);
```

### Step 3: Verifier Sends Challenge

```rust
let r = transcript.challenge();  // Fiat-Shamir
```

### Step 4: Fold Instances (Both Parties Compute)

```rust
fn fold_instances(u1: &Instance, u2: &Instance, T_bar: &Commitment, r: F) -> Instance {
    Instance {
        // Ē' = Ē₁ + r·T̄ + r²·Ē₂
        E_bar: u1.E_bar + T_bar.scale(r) + u2.E_bar.scale(r * r),

        // u' = u₁ + r·u₂
        u: u1.u + r * u2.u,

        // W̄ᵢ' = W̄ᵢ₁ + r·W̄ᵢ₂  for each segment
        W_bar: u1.W_bar.iter().zip(&u2.W_bar)
            .map(|(w1, w2)| *w1 + w2.scale(r))
            .collect(),

        // x' = x₁ + r·x₂
        x: u1.x.iter().zip(&u2.x)
            .map(|(x1, x2)| *x1 + r * *x2)
            .collect(),
    }
}
```

### Step 5: Fold Witnesses (Prover Only)

```rust
fn fold_witnesses(w1: &Witness, w2: &Witness, T: &[F], rT: F, r: F) -> Witness {
    Witness {
        // E' = E₁ + r·T + r²·E₂
        E: w1.E.iter().zip(&T).zip(&w2.E)
            .map(|((e1, t), e2)| *e1 + r * *t + r * r * *e2)
            .collect(),

        // rE' = rE₁ + r·rT + r²·rE₂
        rE: w1.rE + r * rT + r * r * w2.rE,

        // Wᵢ' = Wᵢ₁ + r·Wᵢ₂  for each segment
        W: w1.W.iter().zip(&w2.W)
            .map(|(seg1, seg2)| {
                seg1.iter().zip(seg2)
                    .map(|(a, b)| *a + r * *b)
                    .collect()
            })
            .collect(),

        // rWᵢ' = rWᵢ₁ + r·rWᵢ₂
        rW: w1.rW.iter().zip(&w2.rW)
            .map(|(r1, r2)| *r1 + r * *r2)
            .collect(),
    }
}
```

### Why Folding Preserves Satisfiability

If both `(u₁, w₁)` and `(u₂, w₂)` are satisfying, then `(u', w')` is also satisfying:

```
Let Z' = Z₁ + r·Z₂

(A·Z') ∘ (B·Z')
= (A·Z₁ + r·A·Z₂) ∘ (B·Z₁ + r·B·Z₂)
= (A·Z₁)∘(B·Z₁) + r·[(A·Z₁)∘(B·Z₂) + (A·Z₂)∘(B·Z₁)] + r²·(A·Z₂)∘(B·Z₂)
= [u₁·(C·Z₁) + E₁] + r·T + r²·[u₂·(C·Z₂) + E₂]
= (u₁ + r·u₂)·(C·Z₁ + r·C·Z₂) + (E₁ + r·T + r²·E₂)
= u'·(C·Z') + E'  ✓
```

---

## Part 4: The Complete BlindFold Protocol

```
┌─────────────────────────────────────────────────────────────────────┐
│                      BlindFold Protocol                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SETUP: Prover has (u, w) for the real statement                    │
│         - u = (Ē, 1, W̄₁,...,W̄ₗ, x)     ← u=1, E=0 for real        │
│         - w = (0, 0, W₁,rW₁,...,Wₗ,rWₗ)                            │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 1: Prover samples random satisfying pair (u_rand, w_rand)     │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│    // Sample random witness segments                                │
│    for i in 1..=ℓ:                                                  │
│        W_rand[i] = random_vector(segment_size)                      │
│        rW_rand[i] = random_field_element()                          │
│        W̄_rand[i] = Com(W_rand[i], rW_rand[i])                      │
│                                                                     │
│    // Sample random scalar and public inputs                        │
│    u_rand = random_field_element()                                  │
│    x_rand = random_vector(public_input_size)                        │
│                                                                     │
│    // Compute error vector to make it satisfy                       │
│    Z_rand = (W_rand[1],...,W_rand[ℓ], u_rand, x_rand)              │
│    E_rand = (A·Z_rand) ∘ (B·Z_rand) - u_rand·(C·Z_rand)            │
│    rE_rand = random_field_element()                                 │
│    Ē_rand = Com(E_rand, rE_rand)                                    │
│                                                                     │
│    u_rand = (Ē_rand, u_rand, W̄_rand[1],...,W̄_rand[ℓ], x_rand)    │
│    w_rand = (E_rand, rE_rand, W_rand[1],..., rW_rand[ℓ])           │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 2: Prover sends random instance to verifier                   │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│    Prover → Verifier: u_rand = (Ē_rand, u_rand, W̄_rand, x_rand)   │
│                                                                     │
│    Note: This reveals NOTHING about the real witness                │
│          (it's completely random)                                   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 3: Compute cross-term and commit                              │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│    T = cross_term(u, w, u_rand, w_rand)                             │
│    rT = random_field_element()                                      │
│    T̄ = Com(T, rT)                                                  │
│                                                                     │
│    Prover → Verifier: T̄                                            │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 4: Verifier sends challenge                                   │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│    r = Hash(transcript || T̄)   // Fiat-Shamir                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 5: Both parties fold instances                                │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│    u_folded = fold_instances(u, u_rand, T̄, r)                      │
│                                                                     │
│    // Verifier computes this from public data                       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 6: Prover folds witnesses                                     │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│    w_folded = fold_witnesses(w, w_rand, T, rT, r)                   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 7: Prover sends folded witness                                │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│    Prover → Verifier: w_folded                                      │
│                                                                     │
│    This is: (E', rE', W₁', rW₁', ..., Wₗ', rWₗ')                   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 8: Verifier checks satisfaction                               │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│    // Check commitment openings                                     │
│    assert Ē_folded == Com(E', rE')                                  │
│    for i in 1..=ℓ:                                                  │
│        assert W̄ᵢ_folded == Com(Wᵢ', rWᵢ')                         │
│                                                                     │
│    // Check R1CS                                                    │
│    Z' = (W₁', ..., Wₗ', u_folded, x_folded)                        │
│    assert (A·Z') ∘ (B·Z') == u_folded·(C·Z') + E'                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Is Zero-Knowledge

The key insight: **w_folded = w + r·w_rand**

Since `w_rand` is uniformly random and unknown to the verifier (they only see commitments), the folded witness `w_folded` is uniformly random from the verifier's perspective, regardless of what `w` is.

```
Verifier sees:
  - u_rand (random instance - reveals nothing)
  - T̄ (hiding commitment - reveals nothing)
  - w_folded = w + r·w_rand (masked by random w_rand)

Verifier does NOT see:
  - w (the real witness)
  - w_rand (the random witness)
  - T (the cross-term)
```

---

## Part 5: Complete ZK-Jolt Protocol

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ZK-Jolt Protocol                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ STEP 1: Prover runs sumchecks with hiding commitments               │
│ ─────────────────────────────────────────────────────               │
│   For each sumcheck round i:                                        │
│     • Compute round polynomial gᵢ(X)                                │
│     • Send W̄ᵢ = Com(coefficients of gᵢ, rᵢ)                        │
│     • Receive challenge cᵢ = Hash(transcript)                       │
│                                                                     │
│ STEP 2: Prover commits to all claimed evaluations                   │
│ ─────────────────────────────────────────────────────               │
│   For each polynomial P that needs evaluation at point r:           │
│     • Compute y = P(r)                                              │
│     • Send C_y = Com(y, ρ)                                          │
│     • Keep (y, ρ) secret                                            │
│                                                                     │
│ STEP 3: Prover constructs Split-Committed R1CS                      │
│ ─────────────────────────────────────────────────────               │
│   Instance (public):                                                │
│     • All commitments: W̄₁,...,W̄ₖ, C_y₁,...,C_yₘ                   │
│     • All challenges: c₁,...,cₖ                                     │
│     • Evaluation points: r₁,...,rₘ                                  │
│                                                                     │
│   Witness (secret):                                                 │
│     • Round poly coefficients + randomness                          │
│     • Evaluation values + randomness: (y₁,ρ₁),...,(yₘ,ρₘ)          │
│                                                                     │
│ STEP 4: Prover runs BlindFold                                       │
│ ─────────────────────────────────────────────────────               │
│     • Sample random relaxed instance-witness pair                   │
│     • Fold with the real instance                                   │
│     • Send folded witness to verifier                               │
│                                                                     │
│ STEP 5: Prover sends ZK-Dory evaluation proofs                      │
│ ─────────────────────────────────────────────────────               │
│   For each evaluation C_yⱼ at point rⱼ on polynomial Pⱼ:           │
│     • Prove: "C_yⱼ commits to Pⱼ(rⱼ)"                              │
│     • This is a ZK proof — verifier learns nothing about yⱼ        │
│                                                                     │
│ VERIFIER checks:                                                    │
│ ─────────────────────────────────────────────────────               │
│   1. Verify folded witness satisfies folded instance                │
│      (includes commitment opening checks)                           │
│                                                                     │
│   2. Verify all ZK-Dory evaluation proofs                           │
│      (links C_yⱼ to committed polynomials)                          │
│                                                                     │
│   Verifier sees: Only commitments and proofs                        │
│   Verifier learns: Nothing about actual values                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Efficiency Analysis

### Circuit Size

The succinct verifier R1CS only encodes:

1. **Sumcheck algebraic checks**: `O(d · log n)` constraints
   - Round polynomial consistency: `gᵢ(0) + gᵢ(1) = claimed_sum`
   - Next claim computation: `claimed_sumᵢ = gᵢ(cᵢ)`

2. **Final identity using committed evaluations**: `O(1)` constraints
   - Algebraic relation between evaluation values

3. **Commitment openings**: Handled by split-committed R1CS framework
   - Not explicit constraints
   - Verified at BlindFold finalization

### Costs

| Component | Prover | Verifier | Communication |
|-----------|--------|----------|---------------|
| Sumcheck (with hiding commits) | O(n) | - | O(d · log n) commitments |
| BlindFold | O(d · log n) | O(d · log n) | O(d · log n) field elements |
| ZK-Dory evaluation proofs | O(n) per poly | O(log n) per poly | O(log n) per poly |
| Total | O(n) | O(d · log n + m · log n) | O((d + m) · log n) |

Where:
- `n` = computation size
- `d` = degree of round polynomials
- `m` = number of polynomial evaluations

### What the Verifier Sees vs. Learns

| What | Who Sees It |
|------|-------------|
| Round polynomial coefficients | Nobody (committed, witness in R1CS) |
| Claimed evaluations yⱼ | Nobody (committed, witness in R1CS) |
| Commitments W̄ᵢ, C_yⱼ | Verifier (in instance) |
| Challenges cᵢ | Verifier (public, Fiat-Shamir) |
| Evaluation points rⱼ | Verifier (public) |
| Folded witness | Verifier (masked by randomness) |

The verifier learns **nothing** about the actual computation — only that the committed values satisfy the required algebraic relations.

---

## Part 7: Jolt-Specific Considerations

### Multi-Stage Sumcheck Architecture

Unlike Spartan2 which has 2 sumcheck phases (outer + inner), Jolt consists of **6 sumcheck stages** with multiple batched instances per stage:

| Stage | Description | Batched Instances |
|-------|-------------|-------------------|
| 1 | Spartan Outer (with univariate skip) | 1 |
| 2 | Product Virtualization + RAM Checking | 4 |
| 3 | Instruction Constraints (Shift + Encoding) | 2 |
| 4 | Register Constraints + RAM Value | 4 |
| 5 | Value + Lookup Table Evaluation | 4 |
| 6 | One-Hot Encoding + Hamming Properties | 7 |

**Total: 22 batched sumcheck instances across 6 stages**

Additionally, Stage 7 is the polynomial opening proof (handled by ZK-Dory, outside the verifier R1CS).

### Verifier R1CS Structure for Jolt

The verifier R1CS must encode:

1. **Per-Stage Batching Verification**
   ```
   For each stage s:
     - Batching coefficients α₁, ..., αₖ derived from transcript
     - Combined claim: claim_s = Σᵢ αᵢ · claimᵢ (scaled by 2^(max_rounds - roundsᵢ))
     - Round consistency: Σᵢ αᵢ · gᵢ(0) + Σᵢ αᵢ · gᵢ(1) = combined_claimed_sum
   ```

2. **Inter-Stage Challenge Threading**
   - Stage 1 challenges → Stage 2 inputs
   - Each stage's final point feeds into next stage's evaluation claims
   - All stages feed polynomial opening claims into the accumulator

3. **Univariate Skip Verification** (Stages 1-2)
   - First round uses univariate skip optimization
   - Circuit verifies the uni-skip proof separately from remaining rounds

### Circuit Size Analysis

```
Per stage:  O(num_instances × num_rounds × degree) constraints
Total:      O(22 × log(n) × d) constraints

Where:
  n = number of RISC-V cycles
  d = maximum polynomial degree per round (~3 for cubic)
  log(n) ≈ 20 for typical programs
```

This is still **O(log n)** but with a constant factor ~10× larger than Spartan2's 2-sumcheck structure.

### Witness Structure

The split-committed R1CS witness contains:

```
For each stage s ∈ {1..6}:
  For each round r:
    - Round polynomial coefficients + randomness
  For each batched instance i:
    - Final evaluation claim + randomness

Total witness segments: O(6 × log(n)) coefficient vectors + O(22) evaluation claims
```

### Commitment Batching

To reduce verifier overhead, round polynomial commitments within a stage can be batched:

1. Prover commits to each round's coefficients: W̄ᵣ = Com(coeffsᵣ, ρᵣ)
2. At stage end, verifier receives batched commitment verification
3. BlindFold folding operates on the full witness across all stages

---

## References

- [Vega: Low-Latency Zero-Knowledge Proofs over Existing Credentials](https://eprint.iacr.org/2024/XXX) - Kaviani, Setty
- [Nova: Recursive Zero-Knowledge Arguments from Folding Schemes](https://eprint.iacr.org/2021/370) - Kothapalli, Setty, Tzialla
- [HyperNova: Recursive arguments for customizable constraint systems](https://eprint.iacr.org/2023/573) - Kothapalli, Setty
- [Spartan: Efficient and general-purpose zkSNARKs without trusted setup](https://eprint.iacr.org/2019/550) - Setty
