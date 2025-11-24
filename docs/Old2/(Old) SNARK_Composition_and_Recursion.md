---
title: "Jolt SNARK Composition: Recursive Verification for On-Chain Deployment"
abstract: "Technical deep-dive into Jolt's SNARK composition system using BN254↔Grumpkin curve cycles to reduce verification costs from 1.5B to ~30M RISC-V cycles, enabling viable on-chain deployment."
author: "Jolt Team"
date: "2025-01-XX"
status: "In Development (PR #975)"
---

# SNARK Composition and Recursion in Jolt

> **📘 Context Documents**:
> - [01_Jolt_Theory.md](01_Jolt_Theory.md) - Mathematical foundations
> - [02_Jolt_Complete_Guide.md](02_Jolt_Complete_Guide.md) - Standard Jolt architecture
> - [03_Verifier_Mathematics_and_Code.md](03_Verifier_Mathematics_and_Code.md) - Verifier implementation
>
> **🔬 Current Status**: Active development in [PR #975](https://github.com/a16z/jolt/pull/975)

---

## Table of Contents

1. [The Verification Cost Problem](#the-verification-cost-problem)
2. [Solution Overview: SNARK Composition](#solution-overview-snark-composition)
3. [The BN254↔Grumpkin Curve Cycle](#the-bn254grumpkin-curve-cycle)
4. [Architecture: Two-Layer Proof System](#architecture-two-layer-proof-system)
5. [Component Deep Dives](#component-deep-dives)
6. [Performance Analysis](#performance-analysis)
7. [Implementation Guide](#implementation-guide)

---

## Introduction: What This Document Covers

This document explains Jolt's **SNARK composition** implementation (PR #975), which enables recursive proof verification.

**Goal**: Reduce Jolt verification from ~1.5B cycles to on-chain viable costs (target: ~30M cycles) through proof recursion.

**Required for**: EVM verifiers and other on-chain deployment scenarios.

**Key technical components**:
1. **BN254↔Grumpkin curve cycle** - Enables efficient native field arithmetic across layers
2. **Hyrax polynomial commitments** - Replaces Dory in recursive layer to avoid pairing operations
3. **ExpSumcheck protocol** - Proves correctness of $\mathbb{G}_T$ exponentiation witnesses
4. **Two-layer architecture** - Layer 1 (standard Jolt) + Layer 2 (recursive verification)

**Target audience**: Developers working on Jolt's recursion implementation, EVM verifier integration, or understanding the proof compression stack.

---

## The Verification Cost Problem

### Standard Jolt Verification Bottleneck

**Total verification cost**: $\sim 1.5$ billion RISC-V cycles

**Breakdown** (for reference case: $\log N = 10$, i.e., 40 exponentiations):
```
Standard Jolt Verifier: 1.5B RV64 cycles
├─ $\mathbb{G}_T$ Exponentiations: ~1.2B cycles (80%) ← THE BOTTLENECK
│  └─ 40 exponentiations = 4 × log N (with batching + Pippenger)
│     ├─ D₁ folding: 10 exps (1 × log N)
│     ├─ D₂ folding: 10 exps (1 × log N)
│     └─ C update: 20 exps (2 × log N, Pippenger amortized)
├─ Pairings (4-5 total): ~200M cycles (13%)
├─ Sumcheck verifications: ~100M cycles (7%)
└─ Misc operations: ~50M cycles
```

**Note**: Typical Jolt programs ($\log N = 16$) use 64 exponentiations → ~2.0B cycles total

### Understanding the ~40 $\mathbb{G}_T$ Exponentiations

**The Naive Count (Without Batching)**:

Dory-Reduce protocol runs $\log N$ rounds (typically $\log N \approx 10-23$ for Jolt proofs).

**Why three commitments?** Dory proves an **inner product** $\langle \vec{v}_1, \vec{v}_2 \rangle = z$ between two committed vectors. Each round updates:
1. $C$ (inner product commitment: proves the result $z$ is correct)
2. $D_1$ (first vector commitment: proves $\vec{v}_1$ is correct)
3. $D_2$ (second vector commitment: proves $\vec{v}_2$ is correct)

Each Dory-Reduce round proceeds as follows:

**Prover sends 6 new commitments** (cross-terms from folding):
- $C_+, C_-$: Cross-terms for inner product
- $D_{1L}, D_{1R}$: Left/right halves of first vector
- $D_{2L}, D_{2R}$: Left/right halves of second vector

**Verifier updates the 3 main commitments** using formulas (from [Theory/Dory.md:297](Theory/Dory.md)):

**1. Inner product commitment update:**
$$
C_{\text{new}} = C_{\text{old}} + \chi + \beta D_{2,\text{old}} + \beta^{-1}D_{1,\text{old}} + \alpha C_+ + \alpha^{-1}C_-
$$

**2. First vector commitment update:**
$$
D_{1,\text{new}} = D_{1L} + \alpha D_{1R}
$$

**3. Second vector commitment update:**
$$
D_{2,\text{new}} = D_{2L} + \alpha^{-1} D_{2R}
$$

**Notation clarification**:
- $(C_{\text{old}}, D_{1,\text{old}}, D_{2,\text{old}})$: Current commitments being verified
- $(C_+, C_-, D_{1L}, D_{1R}, D_{2L}, D_{2R})$: **New** proof elements sent by prover this round
- $(C_{\text{new}}, D_{1,\text{new}}, D_{2,\text{new}})$: Updated commitments for next round
- $\chi = \langle \Gamma_1, \Gamma_2 \rangle$: Pre-computed public value (accounts for randomization)
- $\alpha, \beta$: Random challenges from Fiat-Shamir

**Exponentiation count per round:**
- $C_{\text{new}}$ update: **5 exponentiations** ($\chi^1$, $D_{2,\text{old}}^\beta$, $D_{1,\text{old}}^{\beta^{-1}}$, $C_+^\alpha$, $C_-^{\alpha^{-1}}$)
- $D_{1,\text{new}}$ update: **1 exponentiation** ($D_{1R}^\alpha$ only; $D_{1L}^1$ is free)
- $D_{2,\text{new}}$ update: **1 exponentiation** ($D_{2R}^{\alpha^{-1}}$ only; $D_{2L}^1$ is free)

**Per round cost (naively)**: $5 + 1 + 1 = \mathbf{7}$ $\mathbb{G}_T$ **exponentiations per round**

**For $\log N = 10$ rounds**: Naively $7 \times 10 = 70$ exponentiations (some sources estimate higher counts when including additional implementation overhead like blinding factor updates)

**The Batching Optimization**:

Key insight from [Theory/Dory.md:311-312](Theory/Dory.md):
> "The verifier **does not need to compute the new claim $(C_{\text{new}}, D_{1,\text{new}}, D_{2,\text{new}})$ at each step**. Instead, the verifier can simply accumulate all prover messages and challenges across all $\log n$ rounds."

**How it works**:

1. **Without batching** (naive approach):
   - Start with $(C_0, D_{1,0}, D_{2,0})$ (initial commitments)
   - After round 1: Compute $(C_1, D_{1,1}, D_{2,1})$ from $(C_0, D_{1,0}, D_{2,0})$ (7 exps)
   - After round 2: Compute $(C_2, D_{1,2}, D_{2,2})$ from $(C_1, D_{1,1}, D_{2,1})$ (7 exps)
   - Repeat for all $\log N$ rounds
   - Total: $7 \times \log N = 70-161$ exponentiations (for $\log N = 10-23$)

2. **With batching** (optimized):
   - Accumulate **all** challenges: $\alpha_1, \ldots, \alpha_{\log N}$ and $\beta_1, \ldots, \beta_{\log N}$
   - Accumulate **all** proof messages: $(C_+^{(i)}, C_-^{(i)}, D_{1L}^{(i)}, D_{1R}^{(i)}, D_{2L}^{(i)}, D_{2R}^{(i)})$ for $i = 1, \ldots, \log N$
   - Compute **once at the end** using batched multi-exponentiation:

   $$
   \begin{aligned}
   C_{\text{final}} &= C_0 \cdot \prod_{i=1}^{\log N} \left(\chi_i^{\beta_i} \cdot D_{2,i-1}^{\beta_i} \cdot D_{1,i-1}^{\beta_i^{-1}} \cdot (C_+^{(i)})^{\alpha_i} \cdot (C_-^{(i)})^{\alpha_i^{-1}}\right) \\
   D_{1,\text{final}} &= \prod_{i=1}^{\log N} \left(D_{1L}^{(i)} \cdot (D_{1R}^{(i)})^{\alpha_i}\right) \\
   D_{2,\text{final}} &= \prod_{i=1}^{\log N} \left(D_{2L}^{(i)} \cdot (D_{2R}^{(i)})^{\alpha_i^{-1}}\right)
   \end{aligned}
   $$

   - Uses **Pippenger's batched multi-exponentiation algorithm**
   - Key optimization: All bases collected first, then single batched computation
   - Exploits shared base values and pre-computation across rounds

**Pippenger's Algorithm Explained**:

Pippenger's algorithm is the state-of-the-art method for computing multi-exponentiations efficiently. Given bases $g_1, \ldots, g_n \in \mathbb{G}_T$ and exponents $e_1, \ldots, e_n \in \mathbb{F}_r$, it computes:
$$\prod_{i=1}^n g_i^{e_i}$$

**Why naive approach is $O(n \cdot 254)$**:

Computing a single exponentiation $g^e$ where $e$ is a 254-bit number uses **square-and-multiply**:

**Algorithm**: Compute $g^e$ where $e = (b_{253}, b_{252}, \ldots, b_1, b_0)$ in binary

$$
\begin{aligned}
&\text{result} = 1 \\
&\textbf{for } i \text{ from } 253 \text{ down to } 0: \\
&\quad \text{result} = \text{result}^2 \qquad \qquad \text{// 1 squaring per bit (254 total)} \\
&\quad \textbf{if } b_i = 1: \\
&\quad \quad \text{result} = \text{result} \cdot g \quad \text{// 1 multiplication per set bit (}\sim\text{127 on average)}
\end{aligned}
$$

**Tiny example**: Compute $g^{13}$ where $13 = 1101_2$ (4-bit exponent)

**Why this works (binary reconstruction)**:

The exponent $13 = 1101_2$ in expanded form:
$$13 = 1 \cdot 2^3 + 1 \cdot 2^2 + 0 \cdot 2^1 + 1 \cdot 2^0 = 8 + 4 + 1$$

Therefore: $g^{13} = g^8 \cdot g^4 \cdot g^1$

The algorithm builds this by:
- Processing bits from **most significant to least significant** (left to right)
- **Squaring** doubles the current exponent: $g^x \to g^{2x}$
- **Multiplying by $g$** adds 1 to exponent: $g^x \to g^{x+1}$

At each step: $\text{new exponent} = 2 \times (\text{previous exponent}) + b_i$

**Step-by-step trace**:

$$
\begin{aligned}
&e = 13 = 1101_2 \quad \text{(bits: } b_3=1, b_2=1, b_1=0, b_0=1\text{)} \\[0.5em]
&\text{Start: } \text{result} = 1 \quad \text{(exponent = 0)} \\[1em]
&\textbf{Iteration } i=3 \text{ (bit } b_3=1\text{, most significant):} \\
&\quad \text{result} = 1^2 = 1 \qquad \qquad \qquad \quad \text{// square (exponent: } 0 \to 0\text{)} \\
&\quad \text{result} = 1 \cdot g = g^1 \qquad \qquad \quad \text{// multiply: bit=1 (exponent: } 0 \to 1\text{)} \\
&\quad \text{// Exponent so far: } 1 \text{ (binary: ``1'')} \\[1em]
&\textbf{Iteration } i=2 \text{ (bit } b_2=1\text{):} \\
&\quad \text{result} = (g^1)^2 = g^2 \qquad \qquad \quad \text{// square (exponent: } 1 \to 2\text{)} \\
&\quad \text{result} = g^2 \cdot g = g^3 \qquad \qquad \text{// multiply: bit=1 (exponent: } 2 \to 3\text{)} \\
&\quad \text{// Exponent so far: } 3 = 1 \cdot 2 + 1 \text{ (binary: ``11'')} \\[1em]
&\textbf{Iteration } i=1 \text{ (bit } b_1=0\text{):} \\
&\quad \text{result} = (g^3)^2 = g^6 \qquad \qquad \quad \text{// square (exponent: } 3 \to 6\text{)} \\
&\quad \text{(skip multiply, bit=0)} \qquad \quad \, \text{// bit=0, exponent stays 6} \\
&\quad \text{// Exponent so far: } 6 = 3 \cdot 2 + 0 \text{ (binary: ``110'')} \\[1em]
&\textbf{Iteration } i=0 \text{ (bit } b_0=1\text{, least significant):} \\
&\quad \text{result} = (g^6)^2 = g^{12} \qquad \qquad \text{// square (exponent: } 6 \to 12\text{)} \\
&\quad \text{result} = g^{12} \cdot g = g^{13} \qquad \quad \text{// multiply: bit=1 (exponent: } 12 \to 13\text{)} \\
&\quad \text{// Final exponent: } 13 = 6 \cdot 2 + 1 \text{ (binary: ``1101'')} \\[1em]
&\text{Final: } g^{13} \quad \checkmark
\end{aligned}
$$

**Mathematical verification**:
- Start: exponent = 0
- After $b_3=1$: $0 \times 2 + 1 = 1$
- After $b_2=1$: $1 \times 2 + 1 = 3$
- After $b_1=0$: $3 \times 2 + 0 = 6$
- After $b_0=1$: $6 \times 2 + 1 = 13$ ✓

This is **Horner's method** applied to binary representation:
$$13 = ((1 \cdot 2 + 1) \cdot 2 + 0) \cdot 2 + 1$$

**Cost**: 4 squarings + 3 multiplications = **7 operations** (one squaring per bit, one multiplication per set bit)

**For 254-bit exponents**: 254 squarings + ~127 multiplications ≈ **381 group operations**

**For $n$ independent exponentiations**: $n \times 381$ operations = $O(n \cdot 254)$

---

**Pippenger's approach**: $O(n / \log n)$ group operations

Key idea: **Share work across multiple exponentiations** instead of computing each independently.

**Bucket Method**: Process all exponentiations bit-by-bit together
- For each bit position $j$: Group bases whose exponent has bit $j = 1$
- Combine groups efficiently using doubling

**Window Technique**: Process $w$ bits at a time (instead of 1 bit)
- Window size $w \approx \log_2(n)$ is optimal
- Reduces number of iterations from 254 to $254/w \approx 254/\log_2(n)$

**Toy Example**: Compute $g_1^{13} \cdot g_2^{11} \cdot g_3^{7} \cdot g_4^{5}$

Exponents in binary (4 bits for simplicity):
- $e_1 = 13 = 1101_2$
- $e_2 = 11 = 1011_2$
- $e_3 = 7 = 0111_2$
- $e_4 = 5 = 0101_2$

**Naive approach** (4 separate exponentiations):
- $g_1^{13}$: 4 squarings + 3 multiplications = 7 ops
- $g_2^{11}$: 4 squarings + 3 multiplications = 7 ops
- $g_3^7$: 4 squarings + 2 multiplications = 6 ops
- $g_4^5$: 4 squarings + 2 multiplications = 6 ops
- **Total**: 26 operations

**Pippenger (window size $w=2$)**:

Process 2 bits at a time, creating 4 buckets per window:

**Window 1** (bits 3-2, most significant):
- Exponent bits: $e_1[3:2]=11$, $e_2[3:2]=10$, $e_3[3:2]=01$, $e_4[3:2]=01$
- Bucket[3]: $g_1$ (exponent has bits = 11)
- Bucket[2]: $g_2$ (exponent has bits = 10)
- Bucket[1]: $g_3 \cdot g_4$ (both have bits = 01)
- Combine: $((bucket[3]^3 \cdot bucket[2]^2 \cdot bucket[1]^1) = g_1^3 \cdot g_2^2 \cdot g_3 \cdot g_4$
- Cost: 3 bucket multiplications + 3 exponentiations = **6 ops**

**Window 2** (bits 1-0, least significant):
- Exponent bits: $e_1[1:0]=01$, $e_2[1:0]=11$, $e_3[1:0]=11$, $e_4[1:0]=01$
- Bucket[3]: $g_2 \cdot g_3$ (both have bits = 11)
- Bucket[1]: $g_1 \cdot g_4$ (both have bits = 01)
- Combine: $bucket[3]^3 \cdot bucket[1]^1 = g_2^3 \cdot g_3^3 \cdot g_1 \cdot g_4$
- Cost: 3 bucket operations + 2 exponentiations = **5 ops**

**Final**: $(result_1)^{2^2} \cdot result_2 = (g_1^3 \cdot g_2^2 \cdot g_3 \cdot g_4)^4 \cdot g_2^3 \cdot g_3^3 \cdot g_1 \cdot g_4$
- Cost: 2 squarings + 1 multiplication = **3 ops**

**Total Pippenger**: $6 + 5 + 3 = 14$ operations vs naive 26 operations
- **Speedup**: ~1.86× for just 4 exponentiations!

**For $n = 40$ exponentiations** (254-bit exponents):
- Naive: $40 \times 381 = 15,240$ operations
- Pippenger (window $w=6$): $\approx 254/6 \times 40 = 1,693$ operations
- **Speedup**: ~9× faster!

The speedup grows as $n$ increases because more exponentiations share more work.

---

**Additional preprocessing optimization**:

The $\chi$ terms can be **precomputed during setup**:
- $\chi_i = \langle \Gamma_{1,i}, \Gamma_{2,i} \rangle$ where $\Gamma_1, \Gamma_2$ are public randomness generators
- These are **deterministic** values based on the Dory SRS (Structured Reference String)
- Computed once during preprocessing and stored
- During verification: **lookup instead of exponentiation** (essentially free!)
- Saves $\log N$ exponentiations from the total count

**Detailed breakdown of batched exponentiation count**:

#### Naive Approach: Sequential Round-by-Round Updates

In the **naive (unbatched) approach**, the verifier updates all three commitments sequentially after each round:

$$
\begin{aligned}
\text{Round 1:} \quad & D_{1,1} = D_{1L}^{(1)} \cdot (D_{1R}^{(1)})^{\alpha_1} && \text{(1 exponentiation: } D_{1R}\text{)} \\
& D_{2,1} = D_{2L}^{(1)} \cdot (D_{2R}^{(1)})^{\alpha_1^{-1}} && \text{(1 exponentiation: } D_{2R}\text{)} \\
& C_1 = C_0 \cdot \chi_1 \cdot D_{2,0}^{\beta_1} \cdot D_{1,0}^{\beta_1^{-1}} \cdot (C_+^{(1)})^{\alpha_1} \cdot (C_-^{(1)})^{\alpha_1^{-1}} && \text{(5 exponentiations)} \\[0.5em]
\text{Round 2:} \quad & D_{1,2} = D_{1L}^{(2)} \cdot (D_{1R}^{(2)})^{\alpha_2} && \text{(1 exponentiation: } D_{1R}\text{)} \\
& D_{2,2} = D_{2L}^{(2)} \cdot (D_{2R}^{(2)})^{\alpha_2^{-1}} && \text{(1 exponentiation: } D_{2R}\text{)} \\
& C_2 = C_1 \cdot \chi_2 \cdot D_{2,1}^{\beta_2} \cdot D_{1,1}^{\beta_2^{-1}} \cdot (C_+^{(2)})^{\alpha_2} \cdot (C_-^{(2)})^{\alpha_2^{-1}} && \text{(5 exponentiations)} \\
& \vdots
\end{aligned}
$$

**Per round cost**: $1 + 1 + 5 = 7$ exponentiations
**Total for $\log N = 10$ rounds**: $7 \times 10 = 70$ exponentiations

**Problem**: Creates a **dependency chain** - each $C_i$ depends on the newly computed $D_{1,i}$ and $D_{2,i}$ from that round.

#### Batching Optimization: Compute All Rounds at Once

**Key insight**: The $D_1$ and $D_2$ updates are **independent of $C$**! We can compute all $D_1$ and $D_2$ values first, then use them for $C$.

**What the prover sends per round**:
- $D_{1L}^{(i)}, D_{1R}^{(i)}$: Left/right splits for first vector
- $D_{2L}^{(i)}, D_{2R}^{(i)}$: Left/right splits for second vector
- $C_+^{(i)}, C_-^{(i)}$: Cross-terms for inner product

**Total**: 6 $\mathbb{G}_T$ elements per round

**Batched formulas** (computing all rounds simultaneously):

$$
\begin{aligned}
\text{Step 1:} \quad & D_{1,\text{final}} = \prod_{i=1}^{\log N} \left(D_{1L}^{(i)} \cdot (D_{1R}^{(i)})^{\alpha_i}\right) \\[0.5em]
\text{Step 2:} \quad & D_{2,\text{final}} = \prod_{i=1}^{\log N} \left(D_{2L}^{(i)} \cdot (D_{2R}^{(i)})^{\alpha_i^{-1}}\right) \\[0.5em]
\text{Step 3:} \quad & C_{\text{final}} = C_0 \cdot \prod_{i=1}^{\log N} \left(\chi_i^{\beta_i} \cdot D_{2,i}^{\beta_i} \cdot D_{1,i}^{\beta_i^{-1}} \cdot (C_+^{(i)})^{\alpha_i} \cdot (C_-^{(i)})^{\alpha_i^{-1}}\right)
\end{aligned}
$$

Where $D_{1,i}$ and $D_{2,i}$ are the intermediate values computed in steps 1 and 2.

**Exponentiation count** (for $\log N = 10$ rounds):

**Step 1: Computing $D_{1,\text{final}}$**
$$D_{1,\text{final}} = \prod_{i=1}^{10} \left(D_{1L}^{(i)} \cdot (D_{1R}^{(i)})^{\alpha_i}\right)$$

Bases per round:
- $D_{1L}^{(i)}$ has exponent 1, so no exponentiation operation needed (just group multiplication)
- $D_{1R}^{(i)}$ must be raised to $\alpha_i$ (requires 1 exponentiation)

**Count**: $1$ exponentiation per round × $10$ rounds = **10 exponentiations**

**Step 2: Computing $D_{2,\text{final}}$**
$$D_{2,\text{final}} = \prod_{i=1}^{10} \left(D_{2L}^{(i)} \cdot (D_{2R}^{(i)})^{\alpha_i^{-1}}\right)$$

Bases per round:
- $D_{2L}^{(i)}$ has exponent 1, so no exponentiation operation needed (just group multiplication)
- $D_{2R}^{(i)}$ must be raised to $\alpha_i^{-1}$ (requires 1 exponentiation)

**Count**: $1$ exponentiation per round × $10$ rounds = **10 exponentiations**

**Step 3: Computing $C_{\text{final}}$** (single multi-exponentiation with Pippenger)
$$C_{\text{final}} = C_0 \cdot \prod_{i=1}^{10} \left(\chi_i^{\beta_i} \cdot D_{2,i}^{\beta_i} \cdot D_{1,i}^{\beta_i^{-1}} \cdot (C_+^{(i)})^{\alpha_i} \cdot (C_-^{(i)})^{\alpha_i^{-1}}\right)$$

This is a **multi-exponentiation** of 50 bases (5 bases per round × 10 rounds):

Bases per round:
- $\chi_i$: **precomputed** (free lookup from Dory SRS) ✓
- $D_{2,i}$: **base value** computed in Step 2, but must exponentiate to $\beta_i$
- $D_{1,i}$: **base value** computed in Step 1, but must exponentiate to $\beta_i^{-1}$
- $C_+^{(i)}$: must exponentiate to $\alpha_i$
- $C_-^{(i)}$: must exponentiate to $\alpha_i^{-1}$

**Naive cost**: $4$ bases need exponentiation per round × $10$ rounds = $40$ individual exponentiations

**Pippenger cost**: Multi-exp of 40 bases ≈ **20 equivalent exponentiations** (~2× speedup)

**Count**: $2$ equivalent exponentiations per round × $10$ rounds = **20 exponentiations** (amortized)

#### Final Tally

**Total exponentiations with batching**: $10 + 10 + 20 = \mathbf{40}$ (for $\log N = 10$)

**General formula**: For $\log N$ rounds, batching + Pippenger requires $4 \times \log N$ exponentiations

Examples:
- $\log N = 8$ rounds: $32$ exponentiations (very small programs)
- $\log N = 10$ rounds: $40$ exponentiations (small programs, used as reference example)
- $\log N = 12$ rounds: $48$ exponentiations (Fibonacci example: trace ≈ 2048)
- $\log N = 16$ rounds: $64$ exponentiations (most examples: max_trace_length = 65536)
- $\log N = 24$ rounds: $96$ exponentiations (maximum: recursion)

**Typical Jolt programs**: $\log N \approx 12$-$16$ → **~48-64 $\mathbb{G}_T$ exponentiations**

**Improvement** (for $\log N = 10$): $70 \to 40$ exponentiations = **43% reduction** from naive approach

**With Pippenger multi-exponentiation**:
- Naive sequential computation: 40 exponentiations at cost of ~40 × 254 = ~10,000 group operations
- Pippenger batched computation: ~10,000 / 5 = ~2,000 group operations
- **Effective cost**: Equivalent to ~8 naive exponentiations

**Result**: Batching (56% fewer exponentiations) + Pippenger (5× faster computation) = **~10× total speedup**

### Why $\mathbb{G}_T$ Exponentiations Are Expensive

**$\mathbb{G}_T = \mathbb{F}_q^{12}$**: The target group of BN254 pairings

**What is a 12th-degree extension field?**

A **field extension** creates a larger field from a smaller one. The **degree** is the dimension as a vector space.

**Analogy**: Complex numbers $\mathbb{C}$ are a degree-2 extension of reals $\mathbb{R}$:
- Every complex number: $z = a + bi$ (2 real coefficients)
- Satisfies: $i^2 = -1$ (polynomial equation)

**For BN254**: $\mathbb{F}_q^{12}$ is a degree-12 extension of base field $\mathbb{F}_q$:
- Built via tower: $\mathbb{F}_q \to \mathbb{F}_q^2 \to \mathbb{F}_q^6 \to \mathbb{F}_q^{12}$
- Every element: 12 coefficients from $\mathbb{F}_q$
- Representation: $(a_0, a_1, \ldots, a_{11})$ where each $a_i \in \mathbb{F}_q$

**Complexity per exponentiation**:
- Each $\mathbb{G}_T$ element: 3072 bits (12 × 256-bit $\mathbb{F}_q$ elements)
- Exponentiation via square-and-multiply: $\sim 254$ $\mathbb{F}_q^{12}$ multiplications
- Each $\mathbb{F}_q^{12}$ multiplication: $\sim 144$ base field $\mathbb{F}_q$ multiplications
- **No EVM precompile** (unlike pairings which have precompiles at 0x08)
- **Circuit cost**: 50k-100k R1CS constraints per exponentiation

#### Karatsuba's Algorithm: Efficient Extension Field Multiplication

Before breaking down the multiplication cost, we need to understand **Karatsuba's algorithm** - the key technique for efficient multiplication in extension fields.

**The Problem**: Multiply two degree-1 polynomials
$$(a_0 + a_1 x)(b_0 + b_1 x) = a_0 b_0 + (a_0 b_1 + a_1 b_0)x + a_1 b_1 x^2$$

**Naive cost**: 4 multiplications ($a_0 b_0$, $a_0 b_1$, $a_1 b_0$, $a_1 b_1$)

**Karatsuba's insight (1960)**: Can compute with only **3 multiplications**!

**Key observation**: The middle coefficient can be derived:
$$(a_0 b_1 + a_1 b_0) = (a_0 + a_1)(b_0 + b_1) - a_0 b_0 - a_1 b_1$$

**Algorithm**:
1. $p_0 = a_0 b_0$ (1 multiplication)
2. $p_2 = a_1 b_1$ (1 multiplication)
3. $p_1 = (a_0 + a_1)(b_0 + b_1) - p_0 - p_2$ (1 multiplication, 3 additions)

**Trade-off**: Fewer multiplications (expensive) at cost of more additions (cheap)

**Concrete example in $\mathbb{F}_q^2$**:

Multiply $A = 7 + 3u$ and $B = 5 + 2u$ (where $u^2 = -1$):

**Naive** (4 muls):
$$
\begin{aligned}
(7 + 3u)(5 + 2u) &= 7 \cdot 5 + 7 \cdot 2u + 3u \cdot 5 + 3u \cdot 2u \\
&= 35 + 14u + 15u + 6u^2 = 35 + 29u - 6 = 29 + 29u
\end{aligned}
$$

**Karatsuba** (3 muls):
$$
\begin{aligned}
p_0 &= 7 \cdot 5 = 35 \\
p_2 &= 3 \cdot 2 = 6 \\
p_1 &= (7+3)(5+2) - 35 - 6 = 70 - 41 = 29 \\
\text{Result} &= 35 + 29u + 6(-1) = 29 + 29u
\end{aligned}
$$

**Savings**: 25% fewer multiplications!

#### How Karatsuba Scales to Higher Degrees

**General complexity** for multiplying degree-$n$ polynomials:
- **Naive (schoolbook)**: $n^2$ multiplications
- **Karatsuba**: $O(n^{\log_2 3}) \approx O(n^{1.585})$ multiplications

**Where does $n^{\log_2 3}$ come from?**

Karatsuba recursively splits the problem:
- Degree-$n$ multiplication → 3 subproblems of size $n/2$
- Recurrence: $T(n) = 3T(n/2) + O(n)$
- By Master Theorem: $T(n) = O(n^{\log_2 3})$

**Concrete costs** for various extension degrees:

| Extension | Naive (schoolbook) | Karatsuba | Savings |
|-----------|-------------------|-----------|---------|
| $\mathbb{F}_q^2$ | $2^2 = 4$ muls | $3$ muls | 25% |
| $\mathbb{F}_q^4$ | $4^2 = 16$ muls | $3^2 = 9$ muls | 44% |
| $\mathbb{F}_q^8$ | $8^2 = 64$ muls | $3^3 = 27$ muls | 58% |
| $\mathbb{F}_q^{16}$ | $16^2 = 256$ muls | $3^4 = 81$ muls | 68% |

**Why $3^k$ for degree $2^k$?**
- Degree $2^k$ means $k$ levels of recursive splitting
- Each level multiplies cost by 3 (Karatsuba's 3 subproblems)
- Total: $3^k$ base field multiplications

#### Applying to BN254's Tower Construction

BN254 uses: $\mathbb{F}_q \to \mathbb{F}_q^2 \to \mathbb{F}_q^6 \to \mathbb{F}_q^{12}$

This tower has **non-power-of-2 degrees** (6 and 12), so the analysis is more nuanced:

**Level 1: $\mathbb{F}_q^2$ multiplication**
- Degree 2: Direct Karatsuba applies
- **Cost**: $3$ $\mathbb{F}_q$ muls

**Level 2: $\mathbb{F}_q^6$ multiplication**
- Represent as $\mathbb{F}_q^2[v] / (v^3 - (u+1))$ (three $\mathbb{F}_q^2$ coefficients)
- Degree 3 polynomial multiplication: $(a_0 + a_1 v + a_2 v^2)(b_0 + b_1 v + b_2 v^2)$
- **Naive**: $3^2 = 9$ $\mathbb{F}_q^2$ muls
- **Karatsuba for degree 3**: Can reduce to 6 $\mathbb{F}_q^2$ muls using generalized Karatsuba
  - Split: $(a_0 + a_1 v + a_2 v^2) = a_0 + v(a_1 + a_2 v)$
  - Apply 2-way Karatsuba twice with cross-products
- **Cost**: $6 \times 3 = 18$ $\mathbb{F}_q$ muls

**Level 3: $\mathbb{F}_q^{12}$ multiplication**
- Represent as $\mathbb{F}_q^6[w] / (w^2 - v)$ (two $\mathbb{F}_q^6$ coefficients)
- Degree 2: Direct Karatsuba applies
- **Cost**: $3$ $\mathbb{F}_q^6$ muls = $3 \times 18 = 54$ $\mathbb{F}_q$ muls

**Summary for BN254 tower**:
- **Pure Karatsuba at all levels**: 54 base field muls
- **Karatsuba at low levels, schoolbook at $\mathbb{F}_q^{12}$**:
  - Represent $\mathbb{F}_q^{12}$ as 6 coefficients of $\mathbb{F}_q^2$
  - Schoolbook: $6 \times 6 = 36$ $\mathbb{F}_q^2$ muls
  - Each $\mathbb{F}_q^2$ mul: 3 $\mathbb{F}_q$ muls
  - **Total**: $36 \times 3 = 108$ $\mathbb{F}_q$ muls
- **Pure schoolbook (no Karatsuba)**: $12 \times 12 = 144$ $\mathbb{F}_q$ muls

**The 144 number**: Most implementations use either pure schoolbook or hybrid approaches (108-130 muls) rather than full Karatsuba tower (54 muls) due to trade-offs:
- **Code complexity**: Harder to implement and audit
- **Memory overhead**: Temporary storage for intermediate sums
- **Addition cost**: Not free in circuits (many constraints)
- **Cache effects**: Less sequential memory access

#### Why Different Implementation Costs (54 vs 108 vs 144)

As calculated in the Karatsuba section above:
- **Pure Karatsuba tower**: 54 base field multiplications (optimal)
- **Hybrid (Karatsuba at low levels only)**: 108 base field multiplications
- **Pure schoolbook**: 144 base field multiplications

**Most crypto libraries use 108-144** rather than the optimal 54 because:

1. **Practical implementation**: Karatsuba at $\mathbb{F}_q^2$ level (simple), schoolbook at higher levels
2. **Code simplicity**: Easier to audit and maintain
3. **Circuit costs**: In zero-knowledge proofs, additions aren't free - Karatsuba's many additions become expensive constraints
4. **Conservative estimates**: The 144 number includes overhead from reduction modulo the tower polynomial

**Jolt's actual implementation** (via arkworks-algebra):

Jolt uses the arkworks library which implements **full Karatsuba tower**:
- Fp2 multiplication: Karatsuba (3 base muls)
- Fp6 multiplication: Karatsuba for degree-3 (6 Fp2 muls = 18 base muls)
- Fp12 multiplication: Karatsuba over Fp6 (3 Fp6 muls = 54 base muls)

**Result**: Jolt uses **~54 base field multiplications** per $\mathbb{F}_q^{12}$ multiplication (the optimal Karatsuba tower).

**For cost estimates in this document**: We use **~54-144** depending on context:
- **54**: Jolt's actual implementation cost
- **144**: Conservative upper bound used in some literature (includes overhead and alternative implementations)

**Concrete example for $\log N = 10$** (40 exponentiations, using Jolt's actual 54 base muls per Fp12 mul):
$$
40 \text{ exponentiations} \times 254 \text{ muls} \times 54 \text{ base ops} \approx 550K \text{ base field ops} \approx 450M \text{ RISC-V cycles}
$$

**With conservative estimate** (using 144 from literature):
$$
40 \text{ exponentiations} \times 254 \text{ muls} \times 144 \text{ base ops} \approx 1.5M \text{ base field ops} \approx 1.2B \text{ RISC-V cycles}
$$

**For typical Jolt programs** ($\log N = 16$, i.e., 64 exponentiations, using conservative 144):
$$
64 \text{ exponentiations} \times 254 \text{ muls} \times 144 \text{ base ops} \approx 2.3M \text{ base field ops} \approx 2.0B \text{ RISC-V cycles}
$$

### On-Chain Viability Threshold

**Target**: Reduce to **≤30M RISC-V cycles** for economically viable on-chain verification

**Why 30M?**
- At ~30M cycles: Gas cost becomes competitive with existing zkVMs
- Enables Jolt deployment on Ethereum L1/L2s
- Makes recursive proof composition practical

**Challenge**: Need **50× reduction** (1.5B → 30M cycles)

---

## Solution Overview: SNARK Composition

### The Core Idea: Help the Verifier

Instead of making the verifier compute expensive operations:
1. **Prover provides** the results of expensive computations ($\mathbb{G}_T$ exponentiations)
2. **Verifier checks** these results are correct (much cheaper!)
3. **Recursive proof** proves the provided results are accurate

**Analogy**: Like showing your work on a math test
- Without composition: Teacher re-does entire calculation (slow)
- With composition: Teacher checks your intermediate steps (fast)

### What Gets "Helped"

**Expensive operations that get offloaded**:
1. **$\mathbb{G}_T$ exponentiations** (~40 total): Prover computes, provides results + proof
2. **Some pairing operations**: Can be precomputed/batched
3. **Homomorphic combinations**: Expensive $\mathbb{G}_T$ operations

**What verifier still does directly**:
1. **Equality checks**: Verify algebraic relationships hold
2. **Sumcheck verification**: Check univariate polynomials (cheap)
3. **Hyrax opening verification**: 2 multi-scalar multiplications (MSMs)

### The Recursion Trick

**Layer 1 (Main proof)**: Prove original RISC-V program execution
- Uses full Dory PCS (powerful but expensive to verify)
- Verification: ~1.5B cycles (too expensive for on-chain)

**Layer 2 (Recursion proof)**: Prove Layer 1 verification was done correctly
- RISC-V guest program that runs Layer 1 verifier
- Uses Hyrax PCS (lightweight, optimized for small polynomials)
- Verification: ~330M cycles (78% reduction!)

**Layer 3 (Optional, future)**: Further recursion layers
- Each layer reduces verification cost by ~√N
- Can stack until reaching on-chain viability threshold

---

## The BN254 $\leftrightarrow$ Grumpkin Curve Cycle

### Why We Need Two Curves

**First, what are base field vs scalar field?**

Every elliptic curve has **two associated fields**:

1. **Base field** $\mathbb{F}_q$ - "where the curve lives"
   - Point coordinates $(x, y)$ are in $\mathbb{F}_q$
   - Curve equation operations (point addition formulas) use $\mathbb{F}_q$ arithmetic
   - Example: BN254 has $q \approx 2^{256}$

2. **Scalar field** $\mathbb{F}_r$ - "group order" (how many points exist)
   - Scalars $k \in \mathbb{F}_r$ multiply points: $[k]P = P + P + \cdots + P$ ($k$ times)
   - After $r$ additions, you return to identity: $[r]P = \mathcal{O}$
   - Example: BN254 has $r \approx 2^{254}$
   - **Note**: For prime-order curves (like BN254/Grumpkin), $r$ is a **fixed property** of the curve - every non-identity point has order $r$, so any point can serve as a generator. The scalar field is determined by the curve, not by generator choice.

**In a SNARK circuit**:
- All constraints and polynomials are defined over **one field** (typically the scalar field $\mathbb{F}_r$)
- This is the "native" field where arithmetic is cheap (1 constraint per multiplication)

**The field arithmetic problem**:
- **Dory verification operates on $\mathbb{F}_q$ (base field) values**
  - Including $\mathbb{F}_q^{12}$ extension field elements (from $\mathbb{G}_T$ exponentiations discussed earlier)
  - Extension fields are built on top of base field: $\mathbb{F}_q^{12} = \mathbb{F}_q[x]/p(x)$
  - **Why $\mathbb{F}_q$?** Pairings force this choice:
    - BN254 pairings map to $\mathbb{G}_T \subseteq \mathbb{F}_q^{12}$ (target group in 12th extension of base field)
    - Embedding degree = 12 is fixed by curve construction
    - All pairing outputs and operations are $\mathbb{F}_q^{12}$ elements
    - **Cannot be changed** - this is inherent to how pairing-friendly curves work

- **Standard Jolt proves computations over $\mathbb{F}_r$ (scalar field)**
  - **Why $\mathbb{F}_r$?** Polynomial commitments force this choice:
    - Commitments use BN254 group $\mathbb{G}_1$ which has order $r$
    - Polynomial coefficients become scalars multiplying group elements: $C = [f(\tau)]G$
    - R1CS witness variables are scalars in $\mathbb{F}_r$
    - Could use different curve (like Grumpkin with scalar field $\mathbb{F}_q$), but then face the same mismatch in reverse

- **Problem**: Can't efficiently prove base field operations in scalar field circuit!
  - **Non-native field arithmetic**: Each $\mathbb{F}_q$ operation requires ~256 $\mathbb{F}_r$ operations
    - "~256" comes from bit length: $\mathbb{F}_q$ has 256-bit elements, $\mathbb{F}_r$ has 254-bit elements
    - Must simulate 256-bit arithmetic using multiple smaller operations
    - Each bit position needs separate handling (carries, borrows, etc.)
  - Need range checks, carry propagation, overflow handling
  - Example: One $\mathbb{F}_q$ multiplication might cost 1000+ $\mathbb{F}_r$ constraints
  - Those 54 Karatsuba $\mathbb{F}_q$ multiplications? Would become ~54,000 constraints!

#### Concrete Example: Why Non-Native Arithmetic Is Expensive

**Scenario**: Prove that $a \cdot b = c$ where $a, b, c \in \mathbb{F}_q$ (BN254 base field, 256-bit prime)

**In native $\mathbb{F}_q$ circuit** (if we had one):

**Constraint**: $a \cdot b = c$
**Cost**: 1 multiplication constraint

**What "native" means**: The SNARK circuit is defined over $\mathbb{F}_q$, so:
- Variables $a, b, c$ are single field elements (atomic circuit variables)
- The multiplication $a \cdot b \bmod q$ is a **primitive operation** in the constraint system
- **One R1CS constraint**: $a \cdot b = c$ (single multiplication gate)
- Prover computes the actual 256-bit multiplication (with carries, modular reduction, etc.) **outside the circuit**
  - Yes, this takes real CPU work (hundreds of cycles)
  - But it's done **once** by the prover, not encoded in constraints
- Circuit only verifies the algebraic constraint: $a \cdot b \equiv c \pmod{q}$
- **Key insight**: Native field = can express operation as **single R1CS constraint**
  - Prover work ≠ circuit complexity
  - Native: expensive prover work, cheap circuit (1 constraint)
  - Non-native: expensive prover work AND expensive circuit (1000+ constraints)

**In non-native $\mathbb{F}_r$ circuit** (BN254 scalar field):

Since $q \approx 2^{256}$ doesn't fit in $\mathbb{F}_r$ elements, we must:

1. **Represent each $\mathbb{F}_q$ element as limbs** (chunks):
   - Split $a \in \mathbb{F}_q$ into limbs: $a = a_0 + a_1 \cdot 2^{64} + a_2 \cdot 2^{128} + a_3 \cdot 2^{192}$
   - Each limb $a_i$ is a 64-bit value (fits in $\mathbb{F}_r$)
   - Same for $b$ and $c$: total 12 variables

2. **Compute multiplication with carries**:
   - Multiply limbs: $a_i \cdot b_j$ for all $i, j$ (16 multiplications)
   - Compute intermediate products and carries
   - Accumulate into result limbs $c_0, c_1, c_2, c_3$

3. **Range-check all limbs** (ensure each is actually 64-bit):
   - Need to prove $0 \leq a_i < 2^{64}$ for each limb
   - Typically requires ~10-20 constraints per range check
   - Total: ~12 range checks × 15 constraints = 180 constraints

4. **Check modular reduction**:
   - Actual result might be $> q$, need reduction modulo $q$
   - Requires division with remainder: $(a \cdot b) = k \cdot q + c$
   - Prove $c < q$ (the remainder is valid)
   - Additional ~50 constraints

**Total cost**: ~1000 constraints for one $\mathbb{F}_q$ multiplication!

**Multiplication in Dory opening proof**:
- Need ~54 $\mathbb{F}_q$ multiplications (Karatsuba tower for $\mathbb{F}_q^{12}$)
- In non-native circuit: $54 \times 1000 = 54{,}000$ constraints
- In native circuit (via 2-cycle): $54 \times 1 = 54$ constraints
- **Improvement: 1000× reduction in constraints**

**Solution**: Use a **2-cycle of curves**

### The 2-Cycle Construction

```
        ┌─────────────────────────────────────┐
        │         BN254 Curve                 │
        │  Base field:   𝔽_q (256 bits)       │
        │  Scalar field: 𝔽_r (254 bits)       │
        │  Use: Main Jolt proofs (Layer 1)    │
        └──────────┬────────────────┬──────────┘
                   │                │
     𝔽_q ops ←─────┘                └────→ 𝔽_r ops
   proven in 𝔽_r                      proven in 𝔽_q
                   │                │
        ┌──────────▼────────────────▼──────────┐
        │       Grumpkin Curve                 │
        │  Base field:   𝔽_r (254 bits)        │
        │  Scalar field: 𝔽_q (256 bits)        │
        │  Use: Recursion proofs (Layer 2)     │
        └──────────────────────────────────────┘
```

#### How Can Scalar Field Be Larger Than Base Field?

**Wait, Grumpkin has scalar field (256 bits) > base field (254 bits)?**

Yes! This might seem paradoxical, but it's mathematically sound:

**Common misconception**: "Base field determines coordinate space, so group size $\leq$ base field size"

**Reality**: Group order can exceed base field size!

**The mathematics**:

1. **Base field** $\mathbb{F}_r$ (254 bits):
   - Determines possible coordinate values: $x, y \in \mathbb{F}_r$
   - **Coordinate space size**: $r^2 \approx 2^{508}$ possible pairs

2. **Scalar field** $\mathbb{F}_q$ (256 bits):
   - Determines number of points actually on the curve
   - **Group order**: $q \approx 2^{256}$ points

3. **No contradiction**:
   - $q \approx 2^{256}$ (points on curve)
   - $r^2 \approx 2^{508}$ (possible coordinates)
   - Since $2^{256} \ll 2^{508}$, there's plenty of room! ✓

**Hasse's theorem** constrains the relationship:
$$|q - (r+1)| \leq 2\sqrt{r}$$

**Important clarification**: The "256 vs 254 bit" description is a simplification!
- BN254 and Grumpkin are constructed so their field sizes are **nearly equal** (both ~254-256 bits)
- The "256 vs 254" refers to approximate bit lengths - actual values are specific primes chosen to satisfy both Hasse's theorem and the cycle property
- **Key property that makes the 2-cycle work**:
  - BN254's base field = Grumpkin's scalar field (exact equality: $\mathbb{F}_q^{\text{BN254}} = \mathbb{F}_q^{\text{Grumpkin}}$)
  - BN254's scalar field = Grumpkin's base field (exact equality: $\mathbb{F}_r^{\text{BN254}} = \mathbb{F}_r^{\text{Grumpkin}}$)
- This **reciprocal relationship** (one curve's base = other's scalar) is rare and carefully engineered
- Allows each curve to efficiently prove operations in the other curve's native field

**Intuition with small example**:

Curve $y^2 = x^3 + 3$ over $\mathbb{F}_7$ (base field has 7 elements):
- Base field size: 7
- Coordinate possibilities: $7 \times 7 = 49$ pairs
- **Actual curve points**: 12 points (including $\mathcal{O}$)
- Scalar field: $\mathbb{F}_{12}$ (group order = 12)
- **Result**: Scalar field (12) > base field (7), but $12 \ll 49$ ✓

**Why no "repetition"**:
- We're not indexing coordinates by scalars
- Scalars multiply points: $[k]P$ (group operation)
- Coordinates are field elements: $(x, y)$ (algebraic values)
- These are different mathematical objects!

**Key properties**:
1. **BN254 base field = Grumpkin scalar field**: $\mathbb{F}_q^{\text{BN254}} = \mathbb{F}_q^{\text{Grumpkin}}$
2. **BN254 scalar field = Grumpkin base field**: $\mathbb{F}_r^{\text{BN254}} = \mathbb{F}_r^{\text{Grumpkin}}$
3. Can prove BN254 operations in Grumpkin circuit (and vice versa)
4. **Native field arithmetic**: Operations in one curve are cheap in the other

### Why This Helps

**Dory verification involves**:
- $\mathbb{G}_T$ exponentiations ($\mathbb{F}_q^{12}$ operations)
- These are **base field** $\mathbb{F}_q$ operations
- Need to commit to $\mathbb{F}_q$ values and prove constraints over them

**Using Grumpkin for recursion**:
- Grumpkin scalar field = $\mathbb{F}_q$ (same as BN254 base field)
- Can represent $\mathbb{F}_q$ values naturally as scalars in Grumpkin circuit
- Grumpkin group operations are over base field $\mathbb{F}_r$
- **Native field arithmetic**: $\mathbb{F}_q$ operations proven efficiently!

**How $\mathbb{G}_T = \mathbb{F}_q^{12}$ elements are handled**:

A common confusion: "$\mathbb{F}_q^{12}$ is 12× larger than $\mathbb{F}_q$, how can we represent it?"

**Answer**: We don't represent $\mathbb{G}_T$ elements as single scalars! Instead:

1. **Store as 12 separate $\mathbb{F}_q$ coefficients**:
   $$g \in \mathbb{F}_q^{12} \to (a_0, a_1, \ldots, a_{11}) \text{ where each } a_i \in \mathbb{F}_q$$
   Each coefficient is a **scalar** in Grumpkin (since Grumpkin's scalar field = $\mathbb{F}_q$)

2. **Commit to polynomials over $\mathbb{F}_q$**:

   **What's the accumulator?** Recall from the exponentiation batching section:
   - Computing $g^e$ (where $e = b_{253} b_{252} \ldots b_1 b_0$ in binary) uses square-and-multiply
   - **Accumulator** $\rho_i$ tracks the intermediate result after processing bit $i$:
     - $\rho_0 = 1$ (initial state)
     - $\rho_{i+1} = \rho_i^2 \cdot g^{b_i}$ (square current value, multiply by $g$ if bit is 1)
     - $\rho_{254} = g^e$ (final result after all 254 bits)
   - Each $\rho_i \in \mathbb{G}_T = \mathbb{F}_q^{12}$ (extension field element)

   **Representing accumulators in Grumpkin**:
   - Each accumulator $\rho_i \in \mathbb{F}_q^{12}$ has 12 coefficients: $\rho_i = (c_0, c_1, \ldots, c_{11})$
   - Treat these 12 coefficients as a **polynomial** over $\mathbb{F}_q$
   - Hyrax commits to these coefficients using Grumpkin group elements
   - **Pedersen commitment**: $V = c_0 \cdot G_0 + \cdots + c_{11} \cdot G_{11}$ where $c_i \in \mathbb{F}_q$ (scalars), $G_i \in$ Grumpkin (group)
     - **Reminder**: Pedersen commitments are additively homomorphic: $\text{Com}(a) + \text{Com}(b) = \text{Com}(a+b)$
     - Binding: Cannot find two different messages with same commitment (computationally hard under discrete log assumption)
     - Hiding: Commitment reveals nothing about the message (information-theoretically secure with randomness)
     - Why it works here: Scalars $c_i \in \mathbb{F}_q$ = Grumpkin's scalar field, so scalar multiplication is native!

3. **Prove $\mathbb{F}_q^{12}$ arithmetic via constraints**:

   **High-level**: Prove the square-and-multiply chain is correct
   - **One constraint per bit**: $\rho_i^2 \cdot g^{b_i} = \rho_{i+1}$ for $i = 0, 1, \ldots, 253$
   - These are **R1CS constraints** in the Grumpkin-based recursive SNARK

   **Zooming in: What does one constraint look like?**

   Each accumulator $\rho_i \in \mathbb{F}_q^{12}$ has 12 coefficients: $\rho_i = (a_0, a_1, \ldots, a_{11})$

   **Step**: Compute $\rho_{i+1} = \rho_i^2 \cdot g^{b_i}$

   a. **Squaring in $\mathbb{F}_q^{12}$**: $\rho_i^2 = \rho_i \cdot \rho_i$
      - Uses Karatsuba multiplication over the tower: $\mathbb{F}_q^{12} = \mathbb{F}_q^{2 \times 6}$
      - Breaks down into ~54 $\mathbb{F}_q$ base field multiplications (as discussed in Karatsuba section)
      - Each $\mathbb{F}_q$ multiplication = **1 R1CS constraint** (native in Grumpkin!)
      - So: ~54 R1CS constraints for the squaring

   b. **Conditional multiply by $g$**: $(\rho_i^2) \cdot g^{b_i}$
      - If $b_i = 1$: multiply by $g$ (another ~54 $\mathbb{F}_q$ multiplications)
      - If $b_i = 0$: multiply by 1 (no-op)
      - Constraint: $g^{b_i} = b_i \cdot g + (1 - b_i) \cdot 1$ (ensures $b_i \in \{0, 1\}$)
      - Another ~54 R1CS constraints for conditional multiplication
      - Plus ~1 constraint to check $b_i \cdot (b_i - 1) = 0$ (boolean check)

   c. **Store result**: $\rho_{i+1} = (\rho_i^2) \cdot g^{b_i}$
      - Result has 12 coefficients: $\rho_{i+1} = (c_0, c_1, \ldots, c_{11})$
      - Each coefficient is a combination of $\rho_i$'s coefficients
      - Constrained via the multiplication result

   **Per-bit cost**: ~100-150 R1CS constraints (approximately)
   - Squaring: ~54 constraints
   - Conditional multiplication: ~54 constraints
   - Boolean check + bookkeeping: ~10 constraints

   **Total for exponentiation**: $254 \text{ bits} \times 100\text{-}150 \text{ constraints/bit} \approx 25{,}000\text{-}40{,}000$ R1CS constraints

   **Key insight**: Each individual $\mathbb{F}_q$ multiplication is **native** (1 constraint) because:
   - Grumpkin circuit is over $\mathbb{F}_q$
   - So the ~54 Karatsuba multiplications per $\mathbb{F}_q^{12}$ operation = 54 constraints
   - Compare to non-native: would be $54 \times 1000 = 54{,}000$ constraints per $\mathbb{F}_q^{12}$ operation!
   - For full exponentiation: would be $254 \times 54{,}000 \approx 13.7\text{M constraints}$ (completely impractical)

**Result**: $\mathbb{F}_q^{12}$ operations decomposed into many $\mathbb{F}_q$ operations, all proven efficiently in Grumpkin circuit.

**Hyrax verification (in Grumpkin)**:
- Requires 2 multi-scalar multiplications (MSMs) over Grumpkin group
- MSMs are in base field $\mathbb{F}_r$ (Grumpkin group operations)
- Since we're verifying the Grumpkin proof in BN254 circuit, $\mathbb{F}_r$ operations are **native** in BN254!
- Much cheaper than proving arbitrary $\mathbb{F}_q^{12}$ tower field operations

---

## Architecture: Two-Layer Proof System

### Layer 1: Standard Jolt Proof

**Purpose**: Prove execution of original RISC-V program

```
┌───────────────────────────────────────────────────┐
│ Layer 1: Main Computation (BN254)                │
├───────────────────────────────────────────────────┤
│ Input: RISC-V program + input data               │
│                                                   │
│ Trace Generation:                                 │
│   → Execute program in RISC-V emulator           │
│   → Generate execution trace (T cycles)          │
│                                                   │
│ Proof Generation:                                 │
│   → Stage 1-4: Batched sumchecks                 │
│   → Stage 5: Dory opening proof                  │
│   → PCS: Dory (full power)                       │
│   → Curve: BN254                                  │
│                                                   │
│ Output: JoltProof (π₁)                           │
│   → Proof size: ~10 KB                           │
│   → Verification cost: ~1.5B cycles (unhelped)   │
└───────────────────────────────────────────────────┘
```

**This is unchanged** - standard Jolt as described in other docs.

### Layer 2: Recursion Proof (SNARK Composition)

**Purpose**: Prove Layer 1 verification was performed correctly

```
┌───────────────────────────────────────────────────┐
│ Layer 2: Recursion (Grumpkin)                    │
├───────────────────────────────────────────────────┤
│ Input: Layer 1 proof (π₁) + verifier params      │
│                                                   │
│ Guest Program (RISC-V):                          │
│   fn verify_jolt_proof(π₁, vk₁) -> bool {       │
│       // Run Layer 1 verifier                    │
│       // Track all 𝔾_T exponentiations           │
│       // Return verification result              │
│   }                                               │
│                                                   │
│ Proof Generation:                                 │
│   → Execute verifier in RISC-V emulator          │
│   → Extract exponentiation witnesses             │
│   → Commit via Hyrax (Grumpkin)                  │
│   → Run ExpSumcheck (square-and-multiply)        │
│   → Batch all openings                           │
│   → Generate Hyrax opening proof                 │
│                                                   │
│ Output: RecursionProof (π₂)                      │
│   → Proof size: ~15 KB                           │
│   → Verification cost: ~330M cycles              │
└───────────────────────────────────────────────────┘
```

### Modified Layer 1 Verifier (Used in Layer 2)

**Key changes to verifier when running as guest**:

```rust
// Standard verifier (1.5B cycles)
fn verify_jolt(proof: JoltProof) -> bool {
    // Stage 1-4: Sumcheck verification (cheap)
    verify_sumchecks(proof.sumchecks)?;

    // Stage 5: Dory verification (EXPENSIVE)
    let openings = extract_opening_claims(...);

    // Compute 4×log N $\mathbb{G}_T$ exponentiations (40 for log N=10, 1.2B cycles!)
    let exp_results = compute_gt_exponentiations(...);

    dory_verify(openings, exp_results)
}

// Modified verifier with hints (330M cycles)
fn verify_jolt_with_hints(
    proof: JoltProof,
    hints: RecursionHints  // Provided by prover!
) -> bool {
    // Stage 1-4: Sumcheck verification (unchanged)
    verify_sumchecks(proof.sumchecks)?;

    // Stage 5: Dory verification (MODIFIED)
    let openings = extract_opening_claims(...);

    // Use provided exponentiation results (CHECK, don't compute)
    let exp_results = hints.gt_exponentiations;

    // Verify exponentiations are correct (proven by RecursionProof)
    verify_exponentiation_constraints(exp_results, hints.exp_proof)?;

    // Simplified Dory verification (just equality checks)
    dory_verify_with_precomputed(openings, exp_results)
}
```

**What changed**:
1. ❌ **Don't compute** $\mathbb{G}_T$ exponentiations
2. ✅ **Accept** precomputed results from prover
3. ✅ **Check** algebraic constraints hold (cheap!)
4. ✅ **Verify** RecursionProof proves exponentiation correctness

---

## Component Deep Dives

### 1. Hyrax Polynomial Commitment Scheme

**File**: `jolt-core/src/poly/commitment/hyrax.rs`

**Paper**: [Wahby et al. 2018 - "Doubly-efficient zkSNARKs without trusted setup"](https://eprint.iacr.org/2017/1132)

#### Why Hyrax Instead of Dory for Recursion?

| Property | Dory | Hyrax | Recursion Choice |
|----------|------|-------|------------------|
| **Proof size** | O(log N) | O(√N) | Hyrax (small N) |
| **Verifier time** | O(log N) | O(√N) | Hyrax (small N) |
| **Prover time** | O(N log N) | O(N) | **Hyrax** ✓ |
| **Setup** | Transparent | Transparent | Both OK |
| **Batch efficiency** | Excellent | Good | Hyrax for 4-var |
| **Circuit cost** | High ($\mathbb{G}_T$ exps) | **Low (2 MSMs)** | **Hyrax** ✓ |

**Key insight**: For recursion, N is small (typically 16 coefficients = 4 variables)

**What is N here?**
- In PCS literature, **N = degree bound** (maximum degree the scheme supports)
- For multilinear polynomials over $\ell$ variables: $N = 2^\ell$ (number of evaluations in Boolean hypercube)
- Equivalently: N = number of coefficients in the multilinear polynomial
- Example: 4-variable multilinear polynomial → $N = 2^4 = 16$ coefficients

**In complexity expressions**:
- **Prover time O(N)**: Linear in number of polynomial coefficients (must read/process all coefficients)
- **Verifier time O(√N) for Hyrax**: Due to matrix structure with dimensions $\sqrt{N} \times \sqrt{N}$
- **Proof size O(√N) for Hyrax**: Verifier must check row/column vectors of length $\sqrt{N}$

**In Jolt's recursive layer - what actually uses Hyrax?**

**Important clarification**: The recursive layer (Layer 2) uses Hyrax for a **specific, narrow purpose** - NOT for committing to the full execution trace!

**Layer 2 architecture**:

1. **Main trace (330M cycles of verifier execution)**:
   - Still uses **standard Jolt machinery**: R1CS, lookups, memory checking
   - **Problem**: If we use Dory for polynomial openings, we get the $\mathbb{G}_T$ exponentiation bottleneck again (infinite recursion!)
   - **Solution**: Replace Dory's polynomial commitment with **Hyrax** for Layer 2

2. **Why Hyrax works here**:
   - Hyrax can commit to **polynomials of any size** (including large trace polynomials)
   - The key advantage: **Verification uses only 2 MSMs** (no $\mathbb{G}_T$ exponentiations!)
   - MSMs are over Grumpkin group → when verified in Layer 3 (or on-chain), these are $\mathbb{F}_r$ operations
   - $\mathbb{F}_r$ operations are **native in BN254** → cheap to verify!

3. **The polynomials committed via Hyrax**:
   - **NOT necessarily small!** Can be large trace polynomials (millions of coefficients)
   - Hyrax's O(√N) verification cost is acceptable for the **final layer** that needs on-chain verification
   - The trade-off: Slightly worse asymptotics (√N vs log N) but **much simpler verification** (MSMs vs pairings + $\mathbb{G}_T$ exps)

4. **Additional use: ExpSumcheck witness**:
   - Also commits to exponentiation witnesses (accumulators $\rho_i$) for ExpSumcheck protocol
   - These **are** genuinely small polynomials (16-256 coefficients)
   - But the main benefit of Hyrax is avoiding $\mathbb{G}_T$ exponentiations, not polynomial size

**Why Hyrax in recursion**:
- **Primary reason**: Avoids $\mathbb{G}_T$ exponentiations → breaks the recursion bottleneck
- **Secondary benefit**: For small polynomials, Hyrax's O(√N) is comparable to Dory's O(log N)
- **Key property**: Verification translates to simple operations (MSMs) that are native in the "outer" curve

**Different commitment strategies for different layers**:
- Large N (Layer 1): Use Dory (logarithmic scaling)
- Small N (Layer 2): Use Hyrax (simpler, cheaper to verify in circuit)

**Why Hyrax wins for small N**:
- Dory's O(log N) advantage negligible when log N ≈ 4
- Hyrax's simpler structure = cheaper to prove in circuit
- **Dory's recursion cost**: ~40 $\mathbb{G}_T$ exponentiations → each costs ~25K-40K R1CS constraints in recursive layer (as discussed above)
- **Hyrax's recursion cost**: 2 multi-scalar multiplications (MSMs) over Grumpkin → much cheaper in BN254 circuit (native $\mathbb{F}_r$ arithmetic)
- Massive savings: avoid expensive $\mathbb{F}_q^{12}$ arithmetic in favor of simple scalar multiplications
- No trusted setup, based on discrete log assumption

#### Hyrax: The Big Picture

**The core heuristic**: Instead of committing to a polynomial directly, reshape its coefficients into a **matrix** and commit to each **row** separately using Pedersen commitments.

**Why Hyrax avoids sumcheck** (key insight from Thaler's survey):
- In Bulletproofs/IPA setting, sumcheck is actually **very slow**
- Each field operation in sumcheck → scalar multiplication in the commitment scheme
- Scalar multiplications are **~1000× slower** than field operations (each requires ~400 group additions)
- Hyrax exploits **multiplicative structure** in polynomial evaluation: $p(r) = \vec{b}^T M \vec{a}$
- Reduces evaluation to vector-matrix-vector product instead of sumcheck

**Why this helps for recursion**:
- **Commitment size**: $O(\sqrt{N})$ group elements (commit to rows, not all coefficients)
- **Verification**: Simple operations (multi-scalar multiplications) instead of pairings
- **No sumcheck overhead**: Evaluation proof is just sending vector $M \vec{a}$ (no cryptographic operations!)
- **Recursion-friendly**: MSMs over Grumpkin are native $\mathbb{F}_r$ operations in BN254 circuit

**The key trade-offs**:
- **Dory**:
  - ✓ O(log N) proof size
  - ✓ Logarithmic verifier time
  - ✗ Verification uses pairings + $\mathbb{G}_T$ exponentiations (expensive in recursion!)
- **Hyrax**:
  - ✗ O(√N) proof size (larger)
  - ✗ O(√N) verifier time (not logarithmic)
  - ✓ Verification uses only 2 MSMs (much simpler, recursion-friendly!)

**Dory = Hyrax commitment compressed via pairings**:
- Dory commits to the √N Hyrax commitments using AFGHO pairing-based compression
- Gets best of both: small commitment (1 group element) + fast evaluation (Hyrax-style)
- But verification still requires pairings → expensive in recursion

#### The Matrix Representation (Core Innovation)

**Setup**: We have a multilinear polynomial with $N = 2^\ell$ coefficients

**Hyrax's trick**: Arrange coefficients into a matrix with dimensions:
- Rows: $2^{\ell/\iota}$
- Columns: $2^{\ell(1-1/\iota)}$

For $\iota = 2$ (most common): $\sqrt{N} \times \sqrt{N}$ square matrix

**Column-major ordering**: $T[\text{row}, \text{col}] = w[\text{row} + \text{num\_rows} \times \text{col}]$

#### Small Example: 4-Variable Polynomial ($\iota = 2$)

**Polynomial**: $f(x_0, x_1, x_2, x_3)$ with 16 coefficients

**Coefficients (witness vector)**:
$$w = [c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_{10}, c_{11}, c_{12}, c_{13}, c_{14}, c_{15}]$$

**Reshape into 4×4 matrix** (column-major):
$$
T = \begin{bmatrix}
c_0 & c_4 & c_8 & c_{12} \\
c_1 & c_5 & c_9 & c_{13} \\
c_2 & c_6 & c_{10} & c_{14} \\
c_3 & c_7 & c_{11} & c_{15}
\end{bmatrix}
$$

**Reading order**: Column by column
- Column 0: $c_0, c_1, c_2, c_3$
- Column 1: $c_4, c_5, c_6, c_7$
- Column 2: $c_8, c_9, c_{10}, c_{11}$
- Column 3: $c_{12}, c_{13}, c_{14}, c_{15}$

**Why this ordering?** Maps to variable splitting:
- **Row index** (2 bits): Corresponds to variables $x_0, x_1$ (lower bits)
- **Column index** (2 bits): Corresponds to variables $x_2, x_3$ (upper bits)

**Larger Example: 9-Variable Polynomial** ($\iota = 3$)

- Coefficients: $N = 2^9 = 512$
- Matrix: $8 \times 64$ (rows × columns)
- Communication: 8 row commitments = $512^{1/3}$ group elements
- Verifier work: Process 64 columns = $512^{2/3}$ operations

#### Pedersen Commitment to Rows

**Setup**: Verifier publishes public generators (Grumpkin curve points):
- Column generators: $G_0, G_1, \ldots, G_{n_{\text{cols}}-1}$
- Blinding generator: $H$

**Commitment formula for row $k$**:
$$V_k = H^{r_k} \cdot \prod_{j=0}^{n_{\text{cols}}-1} G_j^{T[k,j]}$$

where:
- $T[k,j]$ = coefficient in row $k$, column $j$
- $r_k$ = random blinding factor (keeps commitment hiding)
- $V_k$ = commitment to row $k$ (a single Grumpkin group element)

**Properties**:
- **Binding**: Cannot open to different values (under discrete log assumption)
- **Hiding**: Reveals nothing about coefficients (perfectly hiding due to $r_k$)
- **Homomorphic**: Supports linear combinations in the exponent

**Concrete Example: 4×4 Matrix**

Recall our matrix:
$$
T = \begin{bmatrix}
c_0 & c_4 & c_8 & c_{12} \\
c_1 & c_5 & c_9 & c_{13} \\
c_2 & c_6 & c_{10} & c_{14} \\
c_3 & c_7 & c_{11} & c_{15}
\end{bmatrix}
$$

**Row commitments** (one per row):
$$
\begin{aligned}
V_0 &= H^{r_0} \cdot G_0^{c_0} \cdot G_1^{c_4} \cdot G_2^{c_8} \cdot G_3^{c_{12}} \\
V_1 &= H^{r_1} \cdot G_0^{c_1} \cdot G_1^{c_5} \cdot G_2^{c_9} \cdot G_3^{c_{13}} \\
V_2 &= H^{r_2} \cdot G_0^{c_2} \cdot G_1^{c_6} \cdot G_2^{c_{10}} \cdot G_3^{c_{14}} \\
V_3 &= H^{r_3} \cdot G_0^{c_3} \cdot G_1^{c_7} \cdot G_2^{c_{11}} \cdot G_3^{c_{15}}
\end{aligned}
$$

**What prover sends**: $C = [V_0, V_1, V_2, V_3]$ (4 Grumpkin group elements)

**Communication cost**: $\sqrt{N}$ group elements (4 for N=16)

**Key insight**: Instead of sending all 16 coefficients, send only 4 commitments!

#### Concrete Walkthrough: Proving an Evaluation

Let's walk through a complete example with our 4×4 matrix.

**Polynomial**: $f(x_0, x_1, x_2, x_3)$ with 16 coefficients arranged as:
$$
T = \begin{bmatrix}
c_0 & c_4 & c_8 & c_{12} \\
c_1 & c_5 & c_9 & c_{13} \\
c_2 & c_6 & c_{10} & c_{14} \\
c_3 & c_7 & c_{11} & c_{15}
\end{bmatrix}
$$

**Goal**: Prove that $f(0.5, 0.3, 0.7, 0.2) = v$ (some specific value)

**Step 1: Prover commits to rows**

Sends: $[V_0, V_1, V_2, V_3]$ (4 group elements)

**Step 2: Verifier sends challenge point**

Verifier: "Prove evaluation at $r = (0.5, 0.3, 0.7, 0.2)$"

**Step 3: Hyrax's trick - split the evaluation**

The evaluation $f(0.5, 0.3, 0.7, 0.2)$ can be rewritten as:
$$f(r) = \sum_{i=0}^{15} c_i \cdot \chi_i(r)$$

where $\chi_i(r)$ is the multilinear Lagrange basis (equals 1 at binary representation of $i$, 0 elsewhere).

**Hyrax splits this**:
- **Row selector** $L$: Uses variables $x_0, x_1$ (lower 2 bits)
  - $L = [\check{\chi}_0, \check{\chi}_1, \check{\chi}_2, \check{\chi}_3]$ where $\check{\chi}_i$ uses $(0.5, 0.3)$
  - Example: $\check{\chi}_0 = (1-0.5) \cdot (1-0.3) = 0.35$
  - Example: $\check{\chi}_1 = 0.5 \cdot (1-0.3) = 0.35$

- **Column selector** $R$: Uses variables $x_2, x_3$ (upper 2 bits)
  - $R = [\hat{\chi}_0, \hat{\chi}_4, \hat{\chi}_8, \hat{\chi}_{12}]$ where $\hat{\chi}_j$ uses $(0.7, 0.2)$
  - Example: $\hat{\chi}_0 = (1-0.7) \cdot (1-0.2) = 0.24$
  - Example: $\hat{\chi}_4 = 0.7 \cdot (1-0.2) = 0.56$

**Key observation**: $f(r) = L \cdot T \cdot R^T$ (matrix-vector multiplication!)

**Step 4: Verifier compresses the commitment**

Verifier computes (locally):
$$T' = V_0^{L_0} \cdot V_1^{L_1} \cdot V_2^{L_2} \cdot V_3^{L_3}$$

This is a **multi-scalar multiplication** (MSM) with 4 scalars.

$T'$ now commits to the vector $(L \cdot T)$ = one row weighted by $L$.

**Step 5: Inner product argument**

Prover and verifier run **Bulletproofs protocol** to prove:
$$\langle (L \cdot T), R \rangle = v$$

This costs $O(\log 4) = 2$ rounds of interaction.

**Total communication**:
- Commitment phase: 4 group elements (the row commitments)
- Evaluation phase: ~4 group elements (Bulletproofs rounds)
- **Total**: ~8 group elements

**Compare to naive**:
- Send all 16 coefficients directly: 16 field elements
- Hyrax: 8 group elements (smaller for large N!)

**Why this helps for recursion**:
- **No pairings**: Just MSMs (multi-scalar multiplications)
- **No $\mathbb{G}_T$ exponentiations**: Everything stays in Grumpkin
- When verified in BN254 circuit: MSMs use native $\mathbb{F}_r$ arithmetic
- Much cheaper than Dory's $\mathbb{G}_T$ operations!

#### The Evaluation Protocol (Proving $f(r) = v$)

**Goal**: Prove that polynomial $f$ (committed via matrix $T$) evaluates to $v$ at point $r = (r_1, \ldots, r_\ell)$

**Key mathematical decomposition** (Hyrax's main trick):

Define $\chi$-polynomials that decompose the Lagrange basis:
$$\chi_b(r) = \prod_{i=1}^{\ell} \chi_{b_i}(r_i) \text{ where } \chi_0(r_i) = 1-r_i, \; \chi_1(r_i) = r_i$$

Split into **lower** and **upper** bits:
$$
\begin{aligned}
\check{\chi}_b &= \prod_{k=1}^{\ell/\iota} \chi_{b_k}(r_k) && \text{(lower } \ell/\iota \text{ bits)} \\
\hat{\chi}_b &= \prod_{k=\ell/\iota+1}^{\ell} \chi_{b_k}(r_k) && \text{(upper } \ell - \ell/\iota \text{ bits)}
\end{aligned}
$$

Define vectors:
$$
\begin{aligned}
L &= (\check{\chi}_0, \check{\chi}_1, \ldots, \check{\chi}_{2^{\ell/\iota}-1}) && \text{(row selector)} \\
R &= (\hat{\chi}_0, \hat{\chi}_{2^{\ell/\iota}}, \ldots, \hat{\chi}_{2^{\ell/\iota} \cdot (2^{\ell-\ell/\iota}-1)}) && \text{(column selector)}
\end{aligned}
$$

**Critical property**:
$$L_i \cdot R_j = \check{\chi}_i \cdot \hat{\chi}_{2^{\ell/\iota} \cdot j} = \chi_{i + 2^{\ell/\iota} \cdot j}$$

Therefore:
$$L \cdot T \cdot R^T = \sum_{i,j} T[i,j] \cdot L_i \cdot R_j = \sum_k w_k \cdot \chi_k = f(r_1, \ldots, r_\ell)$$

**Protocol steps**:

**Step 1: Compression** (Verifier computes)
$$T' = \prod_{k=0}^{2^{\ell/\iota}-1} V_k^{\check{\chi}_k} = \text{Com}(L \cdot T)$$

This is a **multi-scalar multiplication (MSM)** of length $2^{\ell/\iota}$.

**Step 2: Prover sends claim**
Prover sends commitment $\omega$ claiming $\omega = \text{Com}(f(r_1, \ldots, r_\ell))$

**Step 3: Inner product argument**
Execute **Bulletproofs inner product protocol** to prove:
$$\langle L \cdot T, R \rangle = f(r_1, \ldots, r_\ell)$$

This proves the dot product of vector committed in $T'$ with public vector $R$ equals the value in $\omega$.

#### The Bulletproofs Inner Product Argument (Adapted)

**Purpose**: Prove $y = \langle \bar{x}, \bar{a} \rangle$ given $\xi = \text{Com}(\bar{x})$, $\tau = \text{Com}(y)$, and public vector $\bar{a}$

**Key technique**: Recursive halving via **bullet-reduce**

**Each reduction round**:
1. Split vectors in half: $\bar{x} = (\bar{x}_1, \bar{x}_2)$, $\bar{a} = (\bar{a}_1, \bar{a}_2)$
2. Prover sends crossover commitments:
   $$
   \begin{aligned}
   \Upsilon_{-1} &= \text{Com}(\bar{x}_1) \text{ with evaluation } \langle \bar{x}_1, \bar{a}_2 \rangle \\
   \Upsilon_1 &= \text{Com}(\bar{x}_2) \text{ with evaluation } \langle \bar{x}_2, \bar{a}_1 \rangle
   \end{aligned}
   $$
3. Verifier sends challenge $c$
4. Both compute folded values:
   $$
   \begin{aligned}
   \Upsilon' &= \Upsilon^{c^2_{-1}} \cdot \Upsilon \cdot \Upsilon_1^{c^{-2}} \\
   \bar{a}' &= c^{-1} \cdot \bar{a}_1 + c \cdot \bar{a}_2 \\
   \bar{x}' &= c \cdot \bar{x}_1 + c^{-1} \cdot \bar{x}_2
   \end{aligned}
   $$

**After $\log n$ rounds**: Reduced to scalar problem with final algebraic check

**Cost per round**: 2 group elements sent (the crossover commitments)

**Total Bulletproofs cost**: $2 \log(2^{\ell - \ell/\iota}) = 2(\ell - \ell/\iota)$ group elements

#### Complete Cost Analysis

**Communication** (Lemma 5 from paper):
$$
\begin{aligned}
\text{Commitment phase:} & \quad 2^{\ell/\iota} = |w|^{1/\iota} \text{ group elements} \\
\text{Evaluation phase:} & \quad 4 + 2(\ell - \ell/\iota) = O(\log |w|) \text{ group elements} \\
\text{Total:} & \quad O(|w|^{1/\iota})
\end{aligned}
$$

**Verifier computation** (dominated by two MSMs):
1. Computing $T'$: MSM of length $2^{\ell/\iota}$
2. Bulletproofs verification: MSM of length $2^{\ell - \ell/\iota}$

**Total verifier cost**:
$$O(2^{\ell/\iota} + 2^{\ell(1-1/\iota)}) = O(|w|^{1/\iota} + |w|^{(ι-1)/\iota})$$

For $\iota \geq 2$, dominated by $|w|^{(\iota-1)/\iota}$.

**Prover computation**: $O(|w|)$ with small constants
- Initial commitments: $2^{\ell/\iota}$ MSMs of length $2^{\ell - \ell/\iota}$ each
- Bulletproofs rounds: $O(\log |w|)$ rounds, constant work per round

#### The Parameter $\iota$ Trade-off

**Matrix dimensions**:
- Rows: $2^{\ell/\iota}$ → Communication cost
- Columns: $2^{\ell(1-1/\iota)}$ → Verifier computation cost

**Trade-off table** (for $|w| = 1024 = 2^{10}$):

| $\iota$ | Communication | Verifier Time | Concrete (|w|=1024) |
|---------|---------------|---------------|---------------------|
| 2       | $\|w\|^{1/2}$ | $\|w\|^{1/2}$ | 32 vs 32            |
| 3       | $\|w\|^{1/3}$ | $\|w\|^{2/3}$ | 10 vs 102           |
| 4       | $\|w\|^{1/4}$ | $\|w\|^{3/4}$ | 6 vs 181            |

**Optimal choice**: $\iota = 2$ minimizes total verifier work

**Proof** (by AM-GM inequality):
$$2^{\ell/\iota} + 2^{\ell(1-1/\iota)} \geq 2\sqrt{2^{\ell/\iota} \cdot 2^{\ell(1-1/\iota)}} = 2\sqrt{2^\ell} = 2\sqrt{|w|}$$

Equality when $2^{\ell/\iota} = 2^{\ell(1-1/\iota)}$, which gives $\iota = 2$.

**Practical considerations**:
- **$\iota = 2$**: Balanced (Jolt's choice for recursion)
- **$\iota = 3$**: Smaller proofs, higher verifier cost (useful when bandwidth expensive)
- **$\iota > 3$**: Rarely used (diminishing returns)

#### Why Hyrax for Small Polynomials

**Jolt's recursion context**: Proving square-and-multiply constraints
- Polynomials: 4 variables = 16 coefficients
- Need to commit to ~40 such polynomials (accumulator steps)

**Hyrax advantages for this use case**:

1. **Small constant factors**:
   - For 16 coefficients: Only 4 row commitments ($4 \times 4$ matrix)
   - Total proof: 4 group elements + ~8 Bulletproofs elements = 12 group elements

2. **Simple MSM operations**:
   - Verifier does 2 MSMs of length 4 each
   - MSMs over Grumpkin (single field, no extension tower)
   - Much cheaper to prove in BN254 circuit than Dory's $\mathbb{G}_T$ operations

3. **No pairings**:
   - Dory requires pairings (expensive in circuits)
   - Hyrax only needs group operations (cheap in circuits)

4. **Transparent setup**:
   - Generators can be hash-to-curve (deterministic)
   - No trusted setup ceremony required

**Concrete comparison** (for 16-coefficient polynomial):
- **Hyrax**: 12 group elements, 2 MSMs of length 4
- **Dory**: ~4 group elements, but requires pairings and $\mathbb{G}_T$ operations
- **In circuit**: Hyrax ~10× cheaper to verify than Dory

#### Optimized Batch Commit for 4-Variable Polynomials

```rust
// Specialized for recursion's common case: 16-coefficient polynomials
fn batch_commit_4var(
    batch: &[&[Fq]],  // Each polynomial: 16 coefficients
    generators: &PedersenGenerators<Grumpkin>,
) -> Vec<HyraxCommitment> {
    const ROW_SIZE: usize = 4;
    const NUM_ROWS: usize = 4;

    // Flatten all rows from all polynomials
    let all_rows: Vec<_> = batch
        .iter()
        .flat_map(|poly| poly.chunks(ROW_SIZE))
        .collect();

    // Parallel commit all rows
    let row_commitments: Vec<Grumpkin> = all_rows
        .par_iter()
        .map(|row| pedersen_commit(row, generators))
        .collect();

    // Reorganize into per-polynomial commitments
    reorganize(row_commitments, batch.len())
}
```

**Optimization**: Batch *all* row commitments across *all* polynomials, then reorganize
- Better parallelism (more independent work)
- Reduces overhead of parallel dispatch
- Critical for recursion performance

---

### 2. Square-and-Multiply Sumcheck

**File**: `jolt-core/src/subprotocols/square_and_multiply.rs`

#### The Problem: Proving G_T Exponentiations

**Standard approach** (what we're avoiding):
```rust
// Expensive! ~30M cycles per exponentiation
fn gt_exponentiation(base: Fq12, exp: Fr) -> Fq12 {
    let mut result = Fq12::one();
    let bits = exp.to_bits();  // 254 bits

    for bit in bits {
        result = result.square();          // Fq12 multiplication
        if bit == 1 {
            result = result * base;        // Fq12 multiplication
        }
    }
    result
}
```

**Problem**: Each Fq^12 operation = 144 base field multiplications = expensive in circuit!

#### Square-and-Multiply as Constraints

**Key insight**: Instead of proving *computation* of exponentiation, prove the *intermediate steps* satisfy square-and-multiply constraints!

**The constraint system**:

Given: Base $a \in \mathbb{F}_q^{12}$, exponent $e \in \mathbb{F}_r$ with bits $b_1, b_2, \ldots, b_t$

Prove: Result $\rho_t = a^e$ via accumulator sequence

$$
\begin{aligned}
\rho_0 &= 1 && \text{(initial accumulator)} \\
\rho_1 &= \rho_0^2 \cdot a^{b_1} && \text{(after bit 1)} \\
\rho_2 &= \rho_1^2 \cdot a^{b_2} && \text{(after bit 2)} \\
&\vdots \\
\rho_t &= \rho_{t-1}^2 \cdot a^{b_t} && \text{(final result = } a^e \text{)}
\end{aligned}
$$

**Constraint per bit $i$**:
$$
\rho_i^2 \cdot a^{b_i} - \rho_{i+1} = 0
$$

Expanded form:
$$
\rho_i \cdot \rho_i \cdot (b_i \cdot a + (1 - b_i) \cdot 1) - \rho_{i+1} = 0
$$

**Why this works**:
- If $b_i = 1$: constraint becomes $\rho_i^2 \cdot a = \rho_{i+1}$ ✓
- If $b_i = 0$: constraint becomes $\rho_i^2 \cdot 1 = \rho_{i+1}$ ✓
- Enforces correct square-and-multiply logic!

#### Representing as Polynomial Problem

**Polynomials** (all 4-variable MLEs over 16-element domain):

1. **ρ polynomials**: $\rho_0(x), \rho_1(x), \ldots, \rho_t(x)$
   - Each $\rho_i(x)$ is MLE of accumulator values at step $i$
   - 16 coefficients ($\mathbb{F}_q^{12}$ element = 12 $\mathbb{F}_q$ coefficients, plus structure)

2. **Quotient polynomials**: $q_1(x), q_2(x), \ldots, q_t(x)$
   - Ensure constraint satisfaction over entire domain
   - Handle potential violations at non-Boolean points

3. **Base polynomial**: $a(x)$
   - MLE of base $\mathbb{F}_q^{12}$ element

4. **G polynomial**: $g(x)$
   - Tower field polynomial: $g(X) = X^{12} - 18X^6 + 82$
   - Defines $\mathbb{F}_q^{12}$ arithmetic structure

**The sumcheck claim**:

For each bit $i$ from 1 to $t$, verify:
$$
\sum_{x \in \{0,1\}^4} \text{eq}(r, x) \cdot \left[\rho_i(x)^2 \cdot \text{base}(x)^{b_i} - \rho_{i+1}(x) + q_i(x) \cdot g(x)\right] = 0
$$

Where:
- $\text{eq}(r, x)$: Equality polynomial (for random challenge $r$)
- $q_i(x) \cdot g(x)$: Quotient term ensures constraint holds everywhere
- Sum = 0 proves constraint satisfied at all Boolean points

#### ExpSumcheck Implementation

```rust
pub struct ExpSumcheck<F: JoltField> {
    exponentiation_index: usize,
    num_constraints: usize,      // Number of bits in exponent
    gamma_powers: Vec<F>,        // For batching constraints
    r: Vec<F>,                   // Random challenge point
    bits: Vec<bool>,             // Exponent bits (public)
    prover_state: ExpProverState<F>,
}

struct ExpProverState<F: JoltField> {
    rho_polys: Vec<MultilinearPolynomial<F>>,     // Accumulators
    quotient_polys: Vec<MultilinearPolynomial<F>>, // Quotients
    base_poly: MultilinearPolynomial<F>,           // Base
    g_poly: MultilinearPolynomial<F>,              // Tower field
    eq_poly: MultilinearPolynomial<F>,             // eq(r, x)
    bits: Vec<bool>,                               // Exponent bits
}
```

**Sumcheck rounds** (4 rounds for 4-variable polynomials):
```rust
fn compute_prover_message(&mut self, round: usize) -> Vec<F> {
    // Degree-4 univariate polynomial per round
    const DEGREE: usize = 4;

    // Parallel evaluation over half-hypercube
    (0..poly_len/2).into_par_iter().map(|i| {
        let mut evals = [F::zero(); DEGREE];

        for eval_point in 0..DEGREE {
            // Evaluate constraint at this point
            for constraint_idx in 0..num_constraints {
                let rho_prev = rho_polys[idx].eval_at(i, eval_point);
                let rho_curr = rho_polys[idx+1].eval_at(i, eval_point);
                let quotient = quotient_polys[idx].eval_at(i, eval_point);
                let base = base_poly.eval_at(i, eval_point);
                let g = g_poly.eval_at(i, eval_point);

                // Constraint: ρᵢ² · base^{bᵢ} - ρᵢ₊₁ + qᵢ · g = 0
                let base_power = if bits[idx] { base } else { F::one() };
                let constraint = rho_prev * rho_prev * base_power
                               - rho_curr
                               + quotient * g;

                // Batch with gamma^idx
                evals[eval_point] += gamma_powers[idx] * constraint;
            }

            // Multiply by eq polynomial
            evals[eval_point] *= eq_poly.eval_at(i, eval_point);
        }

        evals
    }).reduce(|| [F::zero(); DEGREE], |mut a, b| {
        for i in 0..DEGREE { a[i] += b[i]; }
        a
    })
}
```

**After sumcheck**: Have opening claims for all polynomials at random point r
- These get batched and proven via Hyrax

---

### 3. SNARK Composition Orchestration

**File**: `jolt-core/src/subprotocols/snark_composition.rs`

#### Overall Flow

```rust
pub fn snark_composition_prove<F, ProofTranscript, const RATIO: usize>(
    exponentiation_steps_vec: Vec<ExponentiationSteps>,
    transcript: &mut ProofTranscript,
    hyrax_generators: &PedersenGenerators<Grumpkin>,
) -> RecursionProof<F, ProofTranscript, RATIO>
where
    F: JoltField + From<Fq> + Into<Fq>,
    ProofTranscript: Transcript,
{
    // Step 1: Commit to all polynomials
    let (rho_commits, quotient_commits, base_commits) =
        commit_exponentiation_polynomials(
            exponentiation_steps_vec,
            hyrax_generators
        );

    // Step 2: Append commitments to transcript
    append_commitments_to_transcript(
        transcript,
        &rho_commits,
        &quotient_commits,
        &base_commits
    );

    // Step 3: Get random challenges for batching
    let r: Vec<F> = transcript.challenge_vec(4);  // 4 vars
    let gamma: F = transcript.challenge_scalar();

    // Step 4: Run ExpSumcheck for each exponentiation
    let sumcheck_instances: Vec<ExpSumcheck<F>> =
        exponentiation_steps_vec
            .iter()
            .enumerate()
            .map(|(idx, steps)| {
                ExpSumcheck::new_prover(idx, steps, r.clone(), gamma)
            })
            .collect();

    // Batch all sumchecks together
    let batched_sumcheck = BatchedSumcheck::new(sumcheck_instances);
    let sumcheck_proof = batched_sumcheck.prove(transcript);

    // Step 5: Extract opening claims from sumcheck
    let opening_claims = extract_all_opening_claims(
        &exponentiation_steps_vec,
        &sumcheck_proof.final_point
    );

    // Step 6: Batch all openings with random challenges
    let (batched_poly, batched_eval, batched_commitment) =
        batch_polynomials_and_commitments(
            opening_claims,
            &rho_commits,
            &quotient_commits,
            &base_commits,
            transcript
        );

    // Step 7: Generate Hyrax opening proof
    let hyrax_proof = HyraxOpeningProof::prove(
        hyrax_generators,
        &batched_poly,
        &sumcheck_proof.final_point,
        batched_eval,
        transcript
    );

    RecursionProof {
        commitments: ExpCommitments {
            rho_commitments: rho_commits,
            quotient_commitments: quotient_commits,
            base_commitments: base_commits,
            num_exponentiations: exponentiation_steps_vec.len(),
            bits_per_exponentiation: extract_bits(&exponentiation_steps_vec),
        },
        sumcheck_proof,
        r_sumcheck: sumcheck_proof.final_point,
        hyrax_proof,
        openings: opening_claims,
    }
}
```

#### Commitment Phase

```rust
fn commit_exponentiation_polynomials(
    steps_vec: Vec<ExponentiationSteps>,
    gens: &PedersenGenerators<Grumpkin>,
) -> (Vec<Vec<HyraxCommitment>>, Vec<Vec<HyraxCommitment>>, Vec<HyraxCommitment>) {

    // Collect all polynomials that need committing
    let all_rho_polys: Vec<&[Fq]> = steps_vec
        .iter()
        .flat_map(|steps| &steps.rho_mles)
        .collect();

    let all_quotient_polys: Vec<&[Fq]> = steps_vec
        .iter()
        .flat_map(|steps| &steps.quotient_mles)
        .collect();

    let all_base_polys: Vec<&[Fq]> = steps_vec
        .iter()
        .map(|steps| &steps.base[..])
        .collect();

    // Batch commit (uses optimized 4-var path)
    let rho_commits = HyraxCommitment::batch_commit(&all_rho_polys, gens);
    let quotient_commits = HyraxCommitment::batch_commit(&all_quotient_polys, gens);
    let base_commits = HyraxCommitment::batch_commit(&all_base_polys, gens);

    // Reorganize into per-exponentiation structure
    reorganize_commitments(rho_commits, quotient_commits, base_commits, &steps_vec)
}
```

#### Batching Phase

**Challenge generation**:
```rust
// Generate random challenge for each polynomial
let mut challenges = Vec::new();
for exp_idx in 0..num_exponentiations {
    for rho_idx in 0..rho_counts[exp_idx] {
        challenges.push(transcript.challenge_scalar::<F>());
    }
    for q_idx in 0..quotient_counts[exp_idx] {
        challenges.push(transcript.challenge_scalar::<F>());
    }
    challenges.push(transcript.challenge_scalar::<F>());  // base
}
```

**Polynomial batching**:
```rust
// Initialize batched polynomial (16 coefficients for 4-var)
let mut batched_poly = vec![F::zero(); 16];
let mut batched_eval = F::zero();

// Combine all polynomials with challenges
for (poly_type, poly_coeffs, eval, challenge) in all_opening_claims {
    batched_eval += challenge * eval;
    for (idx, &coeff) in poly_coeffs.iter().enumerate() {
        batched_poly[idx] += challenge * F::from(coeff);
    }
}
```

**Commitment batching**:
```rust
// Homomorphically combine commitments
let mut batched_commitment = vec![Grumpkin::zero(); NUM_ROWS];

for (commitment, challenge) in all_commitments.zip(challenges) {
    let gamma_fq: Fq = challenge.into();
    for (i, &row_commit) in commitment.row_commitments.iter().enumerate() {
        batched_commitment[i] += row_commit * gamma_fq;
    }
}
```

**Result**: Single polynomial + commitment to prove via Hyrax

---

### 4. Recursion Trait Extensions

**File**: `jolt-core/src/poly/commitment/recursion.rs`

```rust
pub trait RecursionCommitmentScheme: CommitmentScheme {
    /// Precomputed data for efficient combined commitment verification
    type CombinedCommitmentHint: Sync + Send + Clone + Debug + Default;

    /// Auxiliary data computed by prover to help verifier
    type AuxiliaryVerifierData: Default + Debug + Sync + Send + Clone;

    /// Precomputes combined commitment and hint for recursion mode
    fn precompute_combined_commitment<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> (Self::Commitment, Self::CombinedCommitmentHint);

    /// Homomorphically combines commitments using precomputed hint
    fn combine_commitments_with_hint<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
        hint: Option<&Self::CombinedCommitmentHint>,
    ) -> Self::Commitment;

    /// Generates proof with auxiliary data for recursion
    fn prove_with_auxiliary<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        hint: Self::OpeningProofHint,
        transcript: &mut ProofTranscript,
    ) -> (Self::Proof, Self::AuxiliaryVerifierData);
}
```

**Purpose**: Separates recursion-specific functionality from core PCS trait
- Not all schemes need recursion support
- Auxiliary data = information prover provides to help verifier
- Hints = precomputed values for efficient batching

---

## Performance Analysis

### Current Results (PR #975)

**Without SNARK composition** (standard Jolt):
```
Total verification: 1,500,000,000 cycles
├─ $\mathbb{G}_T$ exponentiations: 1,200,000,000 (80%)
├─ Pairings: 100,000,000 (7%)
├─ Scalar muls: 50,000,000 (3%)
└─ Misc: 150,000,000 (10%)
```

**With SNARK composition** (current PR):
```
Total verification: 330,000,000 cycles (78% reduction!)
├─ Sumchecks: 150,000,000 (45%)
├─ Hyrax verification: 80,000,000 (24%)
│  └─ 2 MSMs over Grumpkin
├─ Equality checks: 50,000,000 (15%)
└─ Misc: 50,000,000 (15%)
```

### Breakdown of Improvements

**$\mathbb{G}_T$ exponentiations**: 1.2B → ~0 cycles
- ✅ Offloaded to prover
- ✅ Verified via ExpSumcheck (~50M cycles)
- ✅ Net savings: ~1.15B cycles

**Pairings**: 100M → 0 cycles
- ✅ Can be precomputed/provided by prover
- ✅ Checked via algebraic constraints

**New costs**:
- ❌ ExpSumcheck verification: ~50M cycles
- ❌ Hyrax opening verification: ~80M cycles (2 MSMs)
- ❌ Additional sumchecks: ~100M cycles

**Net result**: 1.5B → 330M = **78% reduction**

### Path to On-Chain Viability

**Current status**: 330M cycles (still too expensive)

**Remaining optimizations**:

1. **Further recursion layers** (in progress):
   - Layer 3: Prove Layer 2 verification
   - Each layer: ~√N reduction
   - Layer 3 estimate: ~18M cycles ✓

2. **Hyrax MSM optimization** (planned):
   - Current: General MSM algorithm
   - Optimized: Pippenger with precomputation
   - Estimated savings: 40% → ~50M cycles

3. **Sumcheck batching improvements** (planned):
   - More aggressive batching across components
   - Estimated savings: 20% → ~30M cycles

4. **Native field arithmetic leverage** (in progress):
   - Grumpkin ops proven in BN254 (Fr is native)
   - Better circuit compilation
   - Estimated savings: 30% → ~50M cycles

**Projected final cost**: ~10-15M cycles ✓ (viable!)

### Memory Usage

**Recursion overhead**:
```
Additional memory vs. standard Jolt:
├─ Exponentiation witnesses: ~5 MB
│  └─ 40 exps × (254 bits × 12 Fq elements)
├─ Hyrax commitments: ~2 MB
│  └─ 4 row commitments × 40 exps × 2 groups
├─ Sumcheck state: ~3 MB
└─ Total overhead: ~10 MB (acceptable)
```

---

## Implementation Guide

### Using SNARK Composition in Your Application

#### 1. Enable Recursion Feature

**In `Cargo.toml`**:
```toml
[dependencies]
jolt-core = { version = "*", features = ["recursion"] }
```

**Feature gates**:
- Grumpkin curve support
- Hyrax PCS implementation
- Square-and-multiply sumcheck
- Recursion-specific witness generation

#### 2. Generate Recursion Proof

```rust
use jolt_core::{
    poly::commitment::hyrax::HyraxGenerators,
    subprotocols::snark_composition::snark_composition_prove,
};
use jolt_optimizations::ExponentiationSteps;

// Step 1: Run Layer 1 verification and extract exponentiation witnesses
fn extract_exponentiation_witnesses(
    proof: &JoltProof,
    preprocessing: &VerifierPreprocessing,
) -> Vec<ExponentiationSteps> {
    // Run verifier and track all $\mathbb{G}_T$ exponentiations
    let mut exp_tracker = ExponentiationTracker::new();

    verify_jolt_with_tracking(
        proof,
        preprocessing,
        &mut exp_tracker
    );

    exp_tracker.into_exponentiation_steps()
}

// Step 2: Generate recursion proof
fn generate_recursion_proof(
    layer1_proof: JoltProof,
    layer1_preprocessing: &VerifierPreprocessing,
) -> RecursionProof {
    // Extract witness data
    let exp_steps = extract_exponentiation_witnesses(
        &layer1_proof,
        layer1_preprocessing
    );

    // Setup Hyrax generators (one-time, can be cached)
    const MATRIX_RATIO: usize = 1;
    let hyrax_gens = HyraxGenerators::<MATRIX_RATIO, Grumpkin>::new(4); // 4 vars

    // Generate proof
    let mut transcript = Transcript::new(b"Jolt Recursion v1");
    snark_composition_prove(
        exp_steps,
        &mut transcript,
        &hyrax_gens.gens
    )
}
```

#### 3. Verify Recursion Proof

```rust
fn verify_with_recursion(
    layer1_proof: &JoltProof,
    recursion_proof: &RecursionProof,
    preprocessing: &VerifierPreprocessing,
) -> Result<bool, ProofVerifyError> {
    // Step 1: Verify recursion proof (cheap!)
    let recursion_valid = verify_recursion_proof(
        recursion_proof,
        &preprocessing.hyrax_generators
    )?;

    if !recursion_valid {
        return Ok(false);
    }

    // Step 2: Verify Layer 1 with provided hints
    let hints = RecursionHints {
        gt_exponentiations: recursion_proof.extract_exponentiation_results(),
        exp_proof: recursion_proof.clone(),
    };

    verify_jolt_with_hints(
        layer1_proof,
        preprocessing,
        hints
    )
}
```

#### 4. End-to-End Example

```rust
// Complete recursion flow
fn main() {
    // Setup (once per program)
    let program = compile_guest_program();
    let layer1_preprocessing = preprocess_prover(&program);
    let layer2_preprocessing = setup_recursion_preprocessing(&layer1_preprocessing);

    // Proving
    let input = b"my input data";

    // Layer 1: Prove program execution
    let (output, layer1_proof) = prove_program_execution(
        &program,
        input,
        &layer1_preprocessing
    );

    println!("Layer 1 proof size: {} KB", layer1_proof.size() / 1024);

    // Layer 2: Prove verification
    let layer2_proof = generate_recursion_proof(
        layer1_proof.clone(),
        &layer1_preprocessing
    );

    println!("Layer 2 proof size: {} KB", layer2_proof.size() / 1024);

    // Verification (on-chain or off-chain)
    let verified = verify_with_recursion(
        &layer1_proof,
        &layer2_proof,
        &layer2_preprocessing
    );

    println!("Verification result: {}", verified);
    println!("Verification cost: ~330M cycles");
}
```

### Configuration Options

```rust
// Tune for your use case
pub struct RecursionConfig {
    /// Matrix aspect ratio for Hyrax (1 = square matrix)
    pub hyrax_matrix_ratio: usize,

    /// Number of recursion layers (1 = single recursion)
    pub num_recursion_layers: usize,

    /// Enable parallel commitment computation
    pub parallel_commits: bool,

    /// Batch size for polynomial commitments
    pub commit_batch_size: usize,
}

impl Default for RecursionConfig {
    fn default() -> Self {
        Self {
            hyrax_matrix_ratio: 1,        // Square matrices (optimal for 4-var)
            num_recursion_layers: 1,      // Single recursion (PR #975)
            parallel_commits: true,       // Use rayon parallelism
            commit_batch_size: 128,       // Batch 128 polys at once
        }
    }
}
```

---

## Appendix: Technical Details

### A. Fq^12 Tower Field Structure

**BN254 $\mathbb{G}_T = \mathbb{F}_q^12**: 12th-degree extension field

**Construction**:
```
Fq → Fq² → Fq⁶ → Fq¹²

Fq²  = Fq[u]   / (u² + 1)
Fq⁶  = Fq²[v]  / (v³ - (u + 1))
Fq¹² = Fq⁶[w]  / (w² - v)
```

**Representation**: Element of Fq^12
```rust
struct Fq12 {
    c0: Fq6,  // Real part
    c1: Fq6,  // Imaginary part
}

struct Fq6 {
    c0: Fq2,
    c1: Fq2,
    c2: Fq2,
}

struct Fq2 {
    c0: Fq,
    c1: Fq,
}
```

**Multiplication cost**: Fq^12 × Fq^12
- Uses Karatsuba algorithm
- ~144 base field multiplications
- ~100 base field additions
- **Total**: ~250 base field operations

### B. Curve Cycle Security Considerations

**BN254 security**: ~100 bits (weakened from 128 due to advancements)
- Pairing-based assumptions (SXDH)
- Discrete log in $\mathbb{G}_1$, $\mathbb{G}_2$, $\mathbb{G}_T$

**Grumpkin security**: ~120 bits
- Discrete log in base group
- No pairing structure (simpler)

**2-cycle property**:
- ✅ Both curves ~100-120 bit security
- ✅ Well-studied in literature (Halo, Nova, etc.)
- ⚠️ Quantum: Both vulnerable (non-post-quantum)

**Best practices**:
- Use constant-time operations (avoid timing attacks)
- Validate all group elements (check subgroup membership)
- Fresh randomness per proof (avoid nonce reuse)

### C. Future Optimizations

**Short-term (months)**:
1. Pippenger MSM for Hyrax verification
2. Better sumcheck batching
3. Optimized Fq arithmetic in circuit

**Medium-term (6-12 months)**:
1. Layer 3+ recursion (further √N reductions)
2. Prover parallelization improvements
3. Custom SNARK compiler optimizations

**Long-term (research)**:
1. Folding schemes (Nova-style) integration
2. Proof aggregation across multiple Jolt proofs
3. Incremental verification (IVC)

---

## References

- **PR #975**: https://github.com/a16z/jolt/pull/975
- **Hyrax Paper**: "Doubly-efficient zkSNARKs without trusted setup" (Wahby et al., 2018)
- **BN254 Curve**: "Pairing-Friendly Elliptic Curves of Prime Order" (Barreto-Naehrig, 2005)
- **Grumpkin**: https://hackmd.io/@aztec-network/ByzgNxBfd (Aztec Protocol)
- **Curve Cycles**: "Recursive Proof Composition without a Trusted Setup" (Bowe et al., 2019)

---

**Document Status**: Draft (matches PR #975 implementation)
**Last Updated**: 2025-01-XX
**Next Review**: Upon PR merge
