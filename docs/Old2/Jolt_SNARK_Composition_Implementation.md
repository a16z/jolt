# Jolt SNARK Composition: Implementation Strategy

> **⚠️ DEPRECATED**: This document uses outdated "two-layer recursion" framing.
>
> **See instead**: [Jolt_Stage6_SNARK_Composition.md](Jolt_Stage6_SNARK_Composition.md) for corrected architecture.
>
> **Key correction**: There is NO "Layer 2 guest program" that is "the Jolt verifier compiled to RISC-V bytecode." Stage 6 does not generate a 330M cycle trace by running the verifier. Instead, Stage 6 commits to witness data (exponentiation intermediate steps) that are **captured as a byproduct** of Stage 5 Dory proving.
>
> **What this document gets wrong**:
> - Section 1.1: "The verifier IS the guest program" - No, there's no guest program in Stage 6
> - Section 1.2: "Trace Length: ~330M RISC-V cycles" - There is no trace, just witness data
> - Section 1.4: "Guest Program" column showing "Jolt Verifier" - Incorrect framing
>
> **What remains useful**: Mathematical details of ExpSumcheck and SZ-Check protocol remain accurate.

> **Prerequisites**:
> - [Dory_Verification_Cost_Summary.md](Dory_Verification_Cost_Summary.md) - The scalability problem
> - [SNARK_Composition_and_Recursion.md](SNARK_Composition_and_Recursion.md) - Detailed technical deep-dive

---

## TL;DR

**Problem**: Jolt verification costs ~1.30B RISC-V cycles (70× too expensive for on-chain deployment)

**Solution**: Two-layer SNARK composition
- **Layer 1**: Standard Jolt proof (BN254, Dory PCS)
- **Layer 2**: Proves "I verified Layer 1 correctly" (Mixed PCS: Dory/BN254 + Hyrax/Grumpkin)
- **Result**: Final verifier costs ~30M cycles (achieves 70× speedup)

**Key enabler**: BN254↔Grumpkin curve cycle makes non-native field arithmetic efficient

---

## Part 1: The Two-Layer Architecture

### 1.1 Why SNARK of a SNARK?

**Core insight**: The Jolt verifier is just another computation—we can prove its execution like any other program.

**The recursion strategy**:
```
┌─────────────────────────────────────────────────────────────┐
│ Original Problem:                                            │
│ "Prove I executed this RISC-V program correctly"            │
│                                                              │
│ Standard Jolt Verifier: 1.30B cycles                        │
│   ├─ 90% cost: 93 𝔾_T exponentiations (1.09B cycles)       │
│   │   └─ 64 from main Dory + 29 from stage 5 RLC          │
│   ├─ 6% cost: 40 sumchecks (130M cycles)                   │
│   └─ 4% cost: Pairings + misc (60M cycles)                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    Too expensive for EVM
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Recursive Solution:                                          │
│ "Prove I verified the Jolt proof correctly"                 │
│                                                              │
│ Step 1: Run verifier as RISC-V program (330M cycles)       │
│   • Don't compute 𝔾_T exponentiations (too expensive!)     │
│   • Accept them as "hints" from prover                      │
│   • Check algebraic constraints instead                     │
│                                                              │
│ Step 2: Prove verifier execution was correct               │
│   • Generate Jolt proof of the verifier program             │
│   • Use Grumpkin curve (enables native field arithmetic)    │
│   • Use Hyrax PCS (avoids 𝔾_T exponentiations)             │
│                                                              │
│ Step 3: Verify the recursion proof on-chain                │
│   • Final verifier: ~30M cycles                             │
│   • Uses Grumpkin MSMs (EVM-friendly operations)            │
└─────────────────────────────────────────────────────────────┘
```

**Why is this valid?**

The recursion proof establishes a chain of trust:
1. **Layer 1 proof π₁**: "Program P with input x produced output y"
2. **Layer 2 proof π₂**: "I ran the Layer 1 verifier correctly, it accepted π₁"
3. **On-chain verifier**: "π₂ is valid" → therefore π₁ is valid → therefore P(x) = y

**Security property**: If the Layer 2 verifier accepts π₂, and π₂ correctly encodes verification of π₁, then π₁ must be a valid proof (by soundness of both layers).

**CRITICAL CLARIFICATION - The Verifier IS the Guest Program**

This is the key insight: **We compile the Layer 1 verifier itself into RISC-V bytecode and treat it as a guest program.**

**What does this mean concretely?**

1. **Layer 1**: You have some program P (e.g., SHA256) that you want to prove
   - Compile P to RISC-V → bytecode₁
   - Run tracer on P → execution trace₁
   - Generate Jolt proof π₁ (proves P executed correctly)

2. **Layer 2**: You want to prove "I verified π₁ correctly"
   - **Compile the Jolt verifier to RISC-V** → bytecode₂ (the verifier becomes a guest!)
   - Run tracer on the verifier with input π₁ → execution trace₂
   - Generate Jolt proof π₂ (proves the verifier executed correctly)

**So what is the "guest program" in Layer 2?**

The guest program IS the verifier code itself. The Jolt verifier contains:
- Sumcheck verification loops (checking polynomial claims)
- Field arithmetic operations (BN254 operations become RISC-V instructions)
- Dory verification logic (the expensive 𝔾_T exponentiations)
- Hash function calls (for Fiat-Shamir)
- All the control flow (if statements, loops, function calls)

All of this gets compiled to RISC-V instructions, generating bytecode just like any other program.

**Why this works as a SNARK:**
- Jolt proves correct execution of ANY RISC-V program
- The verifier is just another RISC-V program
- π₂ proves: "I executed the verifier bytecode correctly on input π₁, and the verifier output 'accept'"
- If the verifier accepted, and we proved we ran the verifier correctly, then π₁ must be valid

**The key modifications from standard Jolt:**
Layer 2 is **modified Jolt**, not standard Jolt:
1. **Curve changed**: BN254 → Grumpkin (enables native field arithmetic for Layer 1's operations)
2. **PCS mixed**: Most polynomials use Dory over Grumpkin, but exponentiation witnesses use Hyrax
3. **Hint handling**: The 93 𝔾_T exponentiations (64 from main Dory + 29 from RLC) are accepted as "hints" (untrusted advice) rather than computed
4. **ExpSumcheck added**: New protocol (SZ-Check) to prove the exponentiation hints are correct

The IOP structure (Spartan + Twist + Shout + Sumcheck) remains identical—only the curve and PCS components change.

### 1.2 Architecture Diagram

```
┌───────────────────────────────────────────────────────────────┐
│ LAYER 1: Standard Jolt (BN254)                               │
├───────────────────────────────────────────────────────────────┤
│ Input: RISC-V program P, input x                             │
│ Execution: Tracer generates T-cycle execution trace          │
│ Proving:                                                      │
│   • Stages 1-4: Batched sumchecks over trace                │
│   • Stage 5: Dory opening proof (4 × log T exponentiations) │
│ Output: Proof π₁ (~10 KB)                                    │
│                                                               │
│ Verification Cost: 1.30B cycles (unhelped)                   │
│   └─ Bottleneck: 93 𝔾_T exponentiations @ 10M each          │
│      (64 main Dory + 29 stage 5 RLC)                        │
└───────────────────────────────────────────────────────────────┘
                              ↓ Feed into Layer 2
┌───────────────────────────────────────────────────────────────┐
│ LAYER 2: Recursive Jolt (Grumpkin)                           │
├───────────────────────────────────────────────────────────────┤
│ Input: Layer 1 proof π₁, verification key vk₁                │
│                                                               │
│ Guest Program (RISC-V):                                       │
│   fn verify_with_hints(π₁, vk₁, hints) -> bool {            │
│     // Verify sumchecks (cheap: ~90M cycles)                 │
│     verify_sumchecks(π₁)?;                                   │
│                                                               │
│     // Use precomputed 𝔾_T exponentiations from hints       │
│     let exp_results = hints.gt_exponentiations;              │
│                                                               │
│     // Check algebraic constraints (instead of computing)    │
│     verify_exponentiation_constraints(exp_results)?;         │
│                                                               │
│     // Simplified Dory verification                          │
│     dory_verify_with_precomputed(π₁, exp_results)           │
│   }                                                           │
│                                                               │
│ Trace Length: ~330M RISC-V cycles (verifier execution)      │
│   • 90M: Sumcheck verification (unchanged)                   │
│   • 200M: ExpSumcheck (prove exponentiations correct)        │
│   • 40M: Simplified Dory + misc                              │
│                                                               │
│ Proof Generation (Mixed PCS Strategy):                        │
│   • IOP: Spartan R1CS + Twist + Shout (same as Layer 1)     │
│   • PCS for main trace: Dory over BN254 (same as Layer 1!)  │
│   • PCS for exp witnesses: Hyrax over Grumpkin (special)     │
│   • Stage 1-4: Batched sumchecks (prove trace correct)       │
│   • SZ-Check: Prove exponentiation hints correct (Hyrax)     │
│   • Stage 5: Mixed opening proof (Dory + Hyrax)              │
│                                                               │
│ Output: Proof π₂ (~15 KB)                                    │
│   └─ Standard Jolt proof structure, different curve/PCS      │
│                                                               │
│ Verification Cost: ~30M cycles                               │
│   └─ Verify π₂ using Jolt verifier over Grumpkin            │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────┐
│ ON-CHAIN VERIFIER (EVM)                                       │
├───────────────────────────────────────────────────────────────┤
│ Verifies π₂ (the recursion proof)                            │
│ Cost: ~30M gas (economically viable!)                        │
│                                                               │
│ Uses: Grumpkin curve operations                              │
│   • Scalar multiplications (efficient in EVM)                │
│   • No 𝔾_T exponentiations (would be prohibitive)           │
└───────────────────────────────────────────────────────────────┘
```

### 1.3 Cost Comparison

| Layer | Curve | PCS | What It Proves | Trace Length | Verification Cost |
|-------|-------|-----|----------------|--------------|-------------------|
| **Layer 1** | BN254 | Dory | RISC-V program execution | T (varies) | 1.30B cycles (if direct) |
| **Layer 2** | BN254 + Grumpkin | Dory/BN254 (trace) + Hyrax/Grumpkin (witnesses) | Layer 1 verifier executed correctly | ~330M | **30M cycles** |
| **Speedup** | | | | | **70× improvement** |

**Key trade-off**: Proving time increases (now 2 layers), but verification becomes practical for on-chain deployment.

### 1.4 Side-by-Side Comparison

| Component | Layer 1 (Standard Jolt) | Layer 2 (Modified Jolt) |
|-----------|---------|---------|
| **Guest Program** | User's SHA256 | **Jolt Verifier** |
| **Bytecode** | SHA256's RISC-V instructions | **Verifier's RISC-V instructions** |
| **Execution Trace** | SHA256 running on data | **Verifier running on π₁** |
| **IOP Structure** | Spartan + Twist + Shout + Sumcheck | **Identical** |
| **Curve** | BN254 | **BN254 (trace) + Grumpkin (witnesses)** |
| **PCS** | Dory over BN254 | **Dory/BN254 (main trace) + Hyrax/Grumpkin (SZ-check witnesses)** |

**What changes**:
1. Guest program: User's program → Verifier (both compiled to RISC-V)
2. Curves: BN254 only → BN254 (main) + Grumpkin (witnesses)
3. PCS: Pure Dory → Mixed strategy (Dory for trace, Hyrax for SZ-check)
4. Additional protocol: SZ-Check added to prove exponentiation hints correct

**What stays the same**:
- The IOP (Spartan + Twist + Shout + Sumcheck) - this is what makes it a SNARK
- The proof structure (stages 1-5, ~40 sumchecks)

---

## Part 2: The BN254↔Grumpkin Curve Cycle

### 2.1 Why Grumpkin?

As explained in [Dory_Verification_Cost_Summary.md](Dory_Verification_Cost_Summary.md), Layer 1's 𝔾_T exponentiations operate in $\mathbb{F}_q$ (BN254's base field), which would be non-native if we proved Layer 1 verification in a BN254 circuit.

**The curve cycle property:**

```
BN254:                        Grumpkin:
  Base field:   𝔽_q            Base field:   𝔽_r
  Scalar field: 𝔽_r            Scalar field: 𝔽_q
       ↓                              ↓
       Swap the fields!
```

**Key advantage**: Layer 1's $\mathbb{F}_q$ operations (the expensive 𝔾_T exponentiations) become **native** operations in Grumpkin's circuit, since Grumpkin's scalar field is $\mathbb{F}_q$.

Without this cycle, each $\mathbb{F}_q$ multiplication would cost ~1000 constraints (non-native field arithmetic). With Grumpkin, it costs 1 constraint (native).

---

**Heuristic: Base Field vs. Scalar Field**

**Base field** ($\mathbb{F}_p$): Where curve point operations happen
- Point coordinates: $(x, y) \in \mathbb{F}_p \times \mathbb{F}_p$
- Point addition formula: $\lambda = \frac{y_2 - y_1}{x_2 - x_1}$ (all operations in $\mathbb{F}_p$)
- **Operations are cheap when done in the base field**

**Scalar field** ($\mathbb{F}_r$): The field of exponents/scalars
- Scalar multiplication $[k]P$: exponent $k \in \mathbb{F}_r$
- Group order: $|\mathbb{G}| = r$ (how many points on the curve)

**In SNARK circuits:**
- Circuit's native field is the **scalar field** (where witness values and constraints live)
- To verify curve operations from another curve, you need that curve's **base field** operations
- **The curve cycle**: BN254 base field = Grumpkin scalar field = $\mathbb{F}_q$
  - Layer 1 Dory data (in BN254 base field $\mathbb{F}_q$) becomes native in Layer 2 circuit (Grumpkin scalar field $\mathbb{F}_q$)

**Note on exponentiation:**
Exponentiation is just repeated group operation: $g^x = \underbrace{g \cdot g \cdot \ldots \cdot g}_{x \text{ times}}$
- The exponent $x$ is a scalar (from scalar field)
- But the **computation** happens in the field where group elements live:
  - For $\mathbb{G}_T$: elements are in $\mathbb{F}_q^{12}$ → operations use $\mathbb{F}_q^{12}$ arithmetic
  - For elliptic curves: points are in $\mathbb{F}_p \times \mathbb{F}_p$ → operations use $\mathbb{F}_p$ arithmetic

**Why this matters**: When proving curve operations in a circuit, the cost depends on whether the curve's base field matches the circuit's scalar field (native) or not (non-native, ~1000× more expensive).

---

---

## Part 3: The Mixed PCS Strategy: Dory + Hyrax

### 3.1 Understanding the Architecture

**The key architectural insight**: Layer 2 uses **two different polynomial commitment schemes** for different types of polynomials.

**Why this is necessary**:
- Dory over BN254 requires expensive $\mathbb{G}_T$ exponentiations during verification
- If Layer 2 used only Dory, verifying Layer 2 would create the same problem recursively (infinite regress)
- Solution: Use Dory for large polynomials (efficient) + Hyrax for special witness polynomials (terminates recursion)

**Important constraint**: Grumpkin has **no pairing structure**, so Dory cannot be instantiated over Grumpkin. Dory requires pairings ($e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$), which only exist on pairing-friendly curves like BN254.

### 3.2 The Mixed PCS Strategy

Layer 2 uses **two PCS schemes** based on what is being committed:

| Polynomial Type | Size | Coefficients Field | PCS Used | Why |
|----------------|------|----------|----------|-----|
| **Main execution trace** | 330M evals | BN254::Fr | **Dory over BN254** | Large polynomial needs O(log N) proof size, same as Layer 1 |
| **Exponentiation witnesses** | ~1024 evals (padded from 768) | BN254::Fq (64 $\mathbb{G}_T$ elements = 768 Fq coefficients) | **Hyrax over Grumpkin** | Tiny polynomial, Hyrax's O(√N) acceptable, **terminates recursion** |

**Critical insight**: The expensive part of Dory isn't the commitments or opening proofs for the main trace - it's the **exponentiations during Dory verification itself**. Layer 2:
1. Still uses Dory over BN254 for main trace (efficient for large polynomials)
2. Avoids computing Dory's exponentiations by accepting them as "hints"
3. Proves the hints correct using Hyrax over Grumpkin + SZ-Check (separate protocol)

**Why Hyrax terminates recursion**:
- Hyrax verification = **2 multi-scalar multiplications (MSMs)** on Grumpkin
- MSMs are EVM-friendly (can be computed on-chain directly or use precompiles)
- No new exponentiations created → no Layer 3 needed
- The recursion terminates at Layer 2

### 3.3 Hyrax: Matrix-Based Polynomial Commitments

**What is Hyrax?**

Hyrax (Wahby et al., 2018) is a polynomial commitment scheme based on discrete logarithms (no pairings required). It's designed for multilinear polynomials and has special properties that make it ideal for terminating recursion.

**Core idea**: Treat a polynomial's coefficient vector as a **matrix**, commit to each row separately using Pedersen commitments.

**Mathematical structure**:

For a multilinear polynomial $\tilde{f}: \{0,1\}^v \to \mathbb{F}$ with $N = 2^v$ coefficients $[f_0, f_1, \ldots, f_{N-1}]$:

1. **Matrix embedding**: Arrange coefficients into $\sqrt{N} \times \sqrt{N}$ matrix:
   $$M = \begin{bmatrix}
   f_0 & f_1 & \cdots & f_{\sqrt{N}-1} \\
   f_{\sqrt{N}} & f_{\sqrt{N}+1} & \cdots & f_{2\sqrt{N}-1} \\
   \vdots & \vdots & \ddots & \vdots \\
   f_{N-\sqrt{N}} & \cdots & \cdots & f_{N-1}
   \end{bmatrix}$$

2. **Pedersen commitment per row**: For generators $G_1, \ldots, G_{\sqrt{N}}, H$:
   $$C_i = \sum_{j=0}^{\sqrt{N}-1} M_{ij} \cdot G_j + r_i \cdot H$$

   Where $r_i$ is a random blinding factor for row $i$.

3. **Commitment**: The Hyrax commitment is the vector $[C_1, C_2, \ldots, C_{\sqrt{N}}]$ (all $\sqrt{N}$ row commitments)

**Proof size**: $O(\sqrt{N})$ group elements + $O(\sqrt{N})$ field elements

**Verification**: Requires **2 multi-scalar multiplications (MSMs)** of size $\sqrt{N}$ each

**Why this terminates recursion**:
- MSMs are "simple enough" to verify directly on-chain (or in a cheaper verifier)
- Unlike Dory, Hyrax verification doesn't create new expensive operations that need proving
- The verification cost is acceptable: $O(\sqrt{N})$ group operations for small $N$

### 3.4 Why Grumpkin for Hyrax Commitments?

**The field matching property**:
- Hyrax commits to values in BN254::Fq (the exponentiation witnesses are $\mathbb{G}_T$ elements = 12 Fq coefficients)
- Grumpkin's scalar field = BN254::Fq
- Therefore: Committing to Fq values using Grumpkin = **native field arithmetic**

**Concrete example of the field matching**:

Recall that Hyrax commitments are **Pedersen commitments** (scalar products with curve points):

$$C_i = \sum_{j=0}^{\sqrt{N}-1} M_{ij} \cdot [G_j] + r_i \cdot [H]$$

Where:
- $M_{ij}$ are the polynomial coefficients we want to commit to (the **scalars**)
- $[G_j], [H]$ are elliptic curve generator points (the **curve points**)
- $[\cdot]$ denotes scalar multiplication: $[s]P = \underbrace{P + P + \cdots + P}_{s \text{ times}}$

**The field matching constraint**: For scalar multiplication $[s]P$ to be efficient, the scalar $s$ must be in the **scalar field** of the curve where $P$ lives.

**In our case**:
- Witness values: $w \in \mathbb{G}_T \cong \mathbb{F}_q^{12}$ (12 field elements from BN254's **base field**)
- Need to commit: $C = \sum_{i} w_i \cdot [G_i]$ where $w_i \in \mathbb{F}_q$
- **Problem if using BN254**: BN254's scalar field is $\mathbb{F}_r$, but our coefficients are in $\mathbb{F}_q$
  - Would need to represent $\mathbb{F}_q$ elements as $\mathbb{F}_r$ elements (expensive conversion!)
  - Or do non-native field arithmetic (1000× more constraints)
- **Solution with Grumpkin**: Grumpkin's scalar field = $\mathbb{F}_q$ (exactly what we need!)
  - $w_i \in \mathbb{F}_q$ = Grumpkin scalar field → **native scalar multiplications**
  - Each $[w_i]G_i$ is a standard elliptic curve operation, no field conversions needed

**Why this matters**:
- **Native scalar multiplication** (Grumpkin): ~100K cycles per operation
- **Non-native field arithmetic** (BN254): ~100M cycles per operation (1000× slower!)
- For 768 coefficients: 768 × 100K = 77M cycles (Grumpkin) vs 768 × 100M = 77B cycles (BN254)

**The curve cycle property makes this possible**:
- BN254 base field ($\mathbb{F}_q$) = Grumpkin scalar field → Grumpkin can commit to BN254 base field elements natively
- Grumpkin base field ($\mathbb{F}_r$) = BN254 scalar field → BN254 can commit to Grumpkin base field elements natively
- This bidirectional efficiency is why curve cycles are essential for recursion

### 3.5 Concrete Example: The Two Commitment Tracks

Let's trace through a concrete example with a tiny program (log N = 11, so 44 Dory exponentiations in Layer 1):

#### Layer 1 (Standard Jolt over BN254)

$$
\begin{array}{c}
\text{User's Program (e.g., simple computation)} \rightarrow \text{Execution Trace (2048 cycles)} \\
\downarrow \\
\text{Jolt Prover commits to polynomials:} \\
\begin{cases}
\tilde{r}(i, j): \text{Register polynomial (2048 evaluations)} \\
\tilde{m}(i, k): \text{Memory polynomial (2048 evaluations)} \\
\tilde{inst}(i): \text{Instruction polynomial (2048 evaluations)} \\
\tilde{bc}(j): \text{Bytecode polynomial (256 instructions)} \\
\ldots \text{(many more)}
\end{cases} \\
\downarrow \\
\text{All committed using: Dory over BN254} \\
\downarrow \\
\text{Proof } \pi_1 \text{ includes:} \\
\begin{cases}
\text{Dory commitments} \in \mathbb{G}_T \text{ (pairing group elements)} \\
\text{Sumcheck messages} \\
\text{Opening proof (requires 44 } \mathbb{G}_T \text{ exponentiations to verify)}
\end{cases}
\end{array}
$$

#### Layer 2 (Modified Jolt over Grumpkin) - The Two Tracks

**Track 1: Normal execution trace (most of the verifier)**

$$
\begin{array}{c}
\text{Layer 1 Verifier as Guest Program} \rightarrow \text{Execution Trace (330M cycles)} \\
\downarrow \\
\text{Why so many cycles? The verifier is complex:} \\
\begin{cases}
\text{40 sumcheck loops (many iterations)} \\
\text{Field arithmetic (BN254 field ops)} \\
\text{Hashing (Fiat-Shamir transcript)} \\
\text{Dory verification logic} \\
\text{Total: 330M RISC-V instructions}
\end{cases} \\
\downarrow \\
\text{Example trace cycles:} \\
\begin{aligned}
&\text{Cycle 1000: LOAD } \pi_1 \text{ from memory} \\
&\text{Cycle 5000: MUL computing challenge}^2 \\
&\text{Cycle 10000: ADD in sumcheck loop} \\
&\text{Cycle 20000: LOAD exponentiation hint}
\end{aligned} \\
\downarrow \\
\text{Jolt Layer 2 commits to polynomials:} \\
\begin{cases}
\tilde{r}_{\text{L2}}(i, j): \text{330M register evaluations} \\
\tilde{m}_{\text{L2}}(i, k): \text{330M memory evaluations} \\
\tilde{inst}_{\text{L2}}(i): \text{330M instruction evaluations} \\
\tilde{bc}_{\text{L2}}(j): \text{50K bytecode instructions}
\end{cases} \\
\downarrow \\
\textbf{Committed using: Dory over BN254 (same as Layer 1!)} \\
\downarrow \\
\text{At verification time (Layer 2 verifier):} \\
\begin{cases}
\text{Commitments opened using Dory over BN254} \\
\text{Opening cost: } O(\log 330M) \approx 28 \text{ scalar mults} \\
\text{Each scalar mult: } \sim 100K \text{ cycles} \\
\textbf{Total: } \sim 3M \text{ cycles (cheap!)}
\end{cases}
\end{array}
$$

**Track 2: The 93 exponentiation witnesses (special handling)**

> **Note on Exponentiation Count**: The 109 exponentiations are 64 (main Dory: 4 × log₂ N for N = 2^16) + 29 (Stage 5 RLC). While Jolt commits to ~50 total witness polynomials, only **29 distinct commitment terms** are combined in Stage 5's random linear combination due to batching and virtual polynomials.
>
> **Why Addition, Not Multiplication?** The order of operations matters:
> 1. **First**: Homomorphic combination (29 exps) → Creates single $C_{\text{combined}}$
> 2. **Then**: Dory opening (64 exps) → Verifies the ONE combined commitment
>
> The 64 intermediate values (cross-term commitments) are for verifying the single combined commitment, not for each of the 29 individual commitments. Without batching: $29 \times 64 = 1{,}856$ exponentiations!

$$
\begin{array}{c}
\text{Layer 1's 93 Dory exponentiations} \rightarrow \text{Results provided as "hints"} \\
(64 \text{ main} + 29 \text{ RLC}) \\
\downarrow \\
\text{These 93 values are:} \\
\begin{cases}
\text{NOT part of the normal execution trace} \\
\text{Treated as "untrusted advice" to the verifier} \\
\text{Must be proven correct separately}
\end{cases} \\
\downarrow \\
64 \text{ } \mathbb{G}_T \text{ elements} \times 12 \text{ } \mathbb{F}_q \text{ coefficients} = 768 \text{ field elements (padded to } 2^{10}) \\
\downarrow \\
\textbf{Committed using: Hyrax over Grumpkin (separate polynomial!)} \\
\downarrow \\
\text{Proven correct using: SZ-Check} \\
\begin{cases}
\text{For each of 109 exponentiations: } w_i = g_i^{x_i} \\
\text{(64 main + 29 RLC)} \\
\text{Prove via sumcheck over square-and-multiply chain} \\
\text{Links Hyrax-committed witnesses to the trace}
\end{cases} \\
\downarrow \\
\text{At verification time (Layer 2 verifier):} \\
\begin{cases}
\text{Hyrax commitment opened (2 MSMs, no } \mathbb{G}_T \text{ exponentiations!)} \\
\text{SZ-Check verified (proves witnesses correct)}
\end{cases}
\end{array}
$$

#### Why This Works - The Security Argument

**Q: Why is it secure to have witnesses committed separately with Hyrax?**

**A:** The ExpSumcheck protocol **cryptographically binds** the Hyrax-committed witnesses to the Dory-committed trace:

1. **Layer 2's execution trace** (Dory-committed) includes:
   - Loading witness values from memory
   - Using those values in verification formulas
   - The trace polynomial captures: "At cycle 20000, register x10 = witness[5]"

2. **ExpSumcheck** proves:
   - The Hyrax-committed polynomial encodes values w₁, ..., w₄₄
   - Each wᵢ satisfies wᵢ = gⁱˣⁱ (proven via square-and-multiply chain)
   - These are opened at specific points during sumcheck verification

3. **The binding happens via sumcheck challenges**:
   - Layer 2's sumcheck asks: "What is witness[5] at evaluation point r?"
   - This point r is randomly chosen by the verifier (Fiat-Shamir)
   - Prover must provide consistent answer across:
     - The Dory-committed trace (shows register x10 = witness[5])
     - The Hyrax-committed witness polynomial (shows witness[5] = some value)
     - The ExpSumcheck proof (proves that value is correct exponentiation)
   - If any inconsistency exists, soundness error is negligible

#### Why Two PCS Schemes?

| Polynomial | Size | PCS Used | Why? |
|-----------|------|----------|------|
| **Execution trace** | 330M evaluations | **Dory over BN254** | Large polynomial → logarithmic proof size needed, **same PCS as Layer 1** |
| **Exponentiation witnesses** | ~512 evaluations | **Hyrax over Grumpkin** | Tiny polynomial → Hyrax's √N overhead acceptable, and **Hyrax verification terminates recursion** |

**The key point**: Layer 2 still uses Dory over BN254 for the main trace (same as Layer 1). The difference is:
- Layer 2 **avoids computing** Dory's 109 exponentiations (64 main + 29 RLC, accepts as hints)
- The hints are committed separately with Hyrax over Grumpkin
- Hyrax verification = 2 MSMs (cheap, no 𝔾_T exponentiations)
- This breaks the infinite regress

**Toy Example - What Goes in Each Commitment:**

$$
\begin{array}{l|l}
\textbf{Layer 2 Dory Commitment (large)} & \textbf{Layer 2 Hyrax Commitment (small)} \\
\hline
\text{Polynomial: } \tilde{r}_{\text{L2}}[c, j] & \text{Polynomial: } \tilde{w}[i] \\
\text{Size: } 330M \times 64 \text{ registers (huge)} & \text{Size: } 1024 = 2^{10} \text{ field elements} \\
& (64 \text{ exponentiations} \times 12 \text{ coefficients, padded}) \\
\hline
\text{Example values:} & \text{Example values:} \\
\tilde{r}_{\text{L2}}[20000, 10] = 42 & \tilde{w}[60] = 789012345 \\
\text{(register x10 at cycle 20000)} & \text{(witness } w_5 = g_5^{x_5}, \text{ coefficient 0)} \\
\tilde{r}_{\text{L2}}[20001, 10] = 123 & \tilde{w}[61] = 234567890 \\
\text{(register x10 at cycle 20001)} & \text{(witness } w_5 = g_5^{x_5}, \text{ coefficient 1)} \\
\tilde{r}_{\text{L2}}[50000, 5] = 789012345 & \tilde{w}[72] = 111222333 \\
\text{(LOAD witness hint into register)} & \text{(witness } w_6 = g_6^{x_6}, \text{ coefficient 0)} \\
\end{array}
$$

**The critical connection - How Hyrax witnesses bind to Dory trace:**

The binding happens through **polynomial evaluation consistency** enforced by sumcheck:

1. **Layer 2's execution trace** (Dory-committed) contains a specific cycle that loads a witness:
   - Cycle 50000: `LOAD address_of_witness[5] → register x5`
   - The trace polynomial $\tilde{r}_{\text{L2}}$ encodes: "At cycle 50000, register 5 was assigned value 789012345"

2. **The Hyrax-committed witness polynomial** $\tilde{w}$ encodes:
   - $\tilde{w}[60] = 789012345$ (first coefficient of $w_5 = g_5^{x_5}$)
   - This is one of 12 $\mathbb{F}_q$ coefficients representing the $\mathbb{G}_T$ element $w_5$

3. **SZ-Check proves**: $\tilde{w}[60\ldots71]$ (the 12 coefficients) actually represent $g_5^{x_5}$
   - Via square-and-multiply chain verification
   - Proves the exponentiation was computed correctly

4. **Sumcheck binds them together** at verification time:
   - Verifier picks random point $r = (r_0, r_1, \ldots, r_{18})$ (18 = log₂(330M))
   - Queries Dory: "What is $\tilde{r}_{\text{L2}}(r)$?" → Prover responds with value $v_1$
   - Queries Hyrax: "What is $\tilde{w}(r')$?" → Prover responds with value $v_2$
   - **Consistency check**: The verifier's sumcheck protocol ensures that if cycle 50000 loads witness[5], then the values must match at the random evaluation point
   - If prover cheated (wrong witness or wrong trace), the polynomials won't agree at random $r$ except with probability $\leq 1/|\mathbb{F}|$ (negligible)

**Why this is secure**:
- The prover commits to **both** polynomials before seeing the random challenge $r$
- Schwartz-Zippel lemma: Two different polynomials can only agree at a random point with negligible probability
- Even if prover provides wrong witness value, they can't make it consistent with the trace polynomial at a random point
- The sumcheck protocol cryptographically enforces the relationship between trace loads and witness values

**Intuition**: It's like two witnesses giving testimony separately - if they're telling the truth, their stories match. If one lies, their stories won't align when cross-examined with random questions they couldn't have prepared for.

**Summary of exponentiation handling**: Layer 2 accepts 109 G_T exponentiations (64 main Dory + 29 RLC) as hints, commits them via Hyrax over Grumpkin, and proves correctness via sumcheck over the square-and-multiply algorithm (detailed formulas and cost analysis in [Dory_Verification_Cost_Summary.md](Dory_Verification_Cost_Summary.md)).

### 3.6 Hyrax Opening Protocol and Why Recursion Terminates

**The key question**: Layer 2 uses Dory for main trace. Why doesn't verifying Layer 2 create the same problem (requiring Layer 3)?

**Answer**: Layer 2's expensive operations (Dory G_T exponentiations) are **not computed**, they're **accepted as hints and proven with SZ-check**. The SZ-check verification uses Hyrax openings, which have cheap verification.

For a typical trace size $N = 2^{16}$ (64K cycles):
- Number of Dory rounds: $\log_2 N = 16$
- Exponentiations per round: 4 (after optimizations)
- **Total exponentiations**: $4 \times 16 = 64$

$$w_i = g_i^{x_i} \quad \text{for } i = 1, \ldots, 64$$

Where each $g_i, w_i \in \mathbb{G}_T \cong \mathbb{F}_q^{12}$.

**Why 4 exponentiations per round?** (from Dory protocol):
1. $D_1$ folding: 1 exponentiation ($D_{1R}^{\alpha}$)
2. $D_2$ folding: 1 exponentiation ($D_{2R}^{\alpha^{-1}}$)
3. $C$ update: 2 exponentiations (via Pippenger optimization, amortized from 4)
   - Originally 4: $D_2^{\beta}, D_1^{\beta^{-1}}, C_+^{\alpha}, C_-^{\alpha^{-1}}$
   - Pippenger multi-exponentiation: 4 bases → amortized cost of 2 exponentiations

**Note**: The number scales with trace size:
- Small trace ($\log_2 N = 11$, 2K cycles): $4 \times 11 = 44$ exponentiations
- Medium trace ($\log_2 N = 16$, 64K cycles): $4 \times 16 = 64$ exponentiations
- Large trace ($\log_2 N = 20$, 1M cycles): $4 \times 20 = 80$ exponentiations

For this document, **we use 64 as the canonical example** (trace size 64K, common for many programs).

#### Additional Exponentiations: Homomorphic Combination Costs

**Critical detail**: The 64 exponentiations above are from the main Dory opening protocol (the iterative halving algorithm). But there's another source of G_T exponentiations that must also be offloaded:

**Homomorphic combination of commitments** (in stage 5, batched opening proof):

When verifying multiple polynomial openings together, Dory combines commitments via **random linear combination (RLC)**:

$$C_{\text{combined}} = \prod_{i=1}^{29} C_i^{\gamma_i}$$

Where:
- Each $C_i$ is a Dory commitment (a $\mathbb{G}_T$ element)
- $\gamma_i$ are random challenges from the verifier
- Computing $C_i^{\gamma_i}$ is a $\mathbb{G}_T$ exponentiation (cyclotomic exponentiation)

**How many polynomials are combined?** For typical programs: **29 polynomials**

Breakdown (from `jolt-core/src/zkvm/witness.rs`):
- **8 fixed polynomials**: `LeftInstructionInput`, `RightInstructionInput`, `WriteLookupOutputToRD`, `WritePCtoRD`, `ShouldBranch`, `ShouldJump`, `RdInc`, `RamInc`
- **16 instruction lookup polynomials**: `InstructionRa(0)` through `InstructionRa(15)` (decomposition parameter D=16)
- **3 RAM polynomials**: `RamRa(0)` through `RamRa(2)` (for moderate programs with d=3)
- **2 bytecode polynomials**: `BytecodeRa(0)` through `BytecodeRa(1)` (for typical programs with d=2)

**Total: 8 + 16 + 3 + 2 = 29 G_T exponentiations** for the homomorphic combination.

**Cost of homomorphic combination**:
- 29 exponentiations × 10M cycles each ≈ **580M cycles**

**Complete exponentiation count**:
- **Main Dory protocol**: 64 exponentiations = 1.28B cycles
- **Homomorphic combination**: 29 exponentiations = 580M cycles
- **Total**: 109 exponentiations = **1.09B cycles**

**Why homomorphic combination also needs SZ-Check**:
- These 29 exponentiations happen during stage 5 (batched opening proof)
- Each one is a full cyclotomic exponentiation in $\mathbb{G}_T$
- Like the main Dory exponentiations, these can be offloaded to SZ-Check
- The prover provides the combined commitment as a hint, then proves correctness via sumcheck

**Implementation location**: `jolt-core/src/poly/commitment/dory.rs:1255-1268` (the `combine_commitments` function)

**Why this is expensive** (detailed in [Dory_Verification_Cost_Summary.md](Dory_Verification_Cost_Summary.md)):
- Each exponentiation: ~381 $\mathbb{F}_q^{12}$ multiplications (square-and-multiply with 254-bit exponent)
- Each $\mathbb{F}_q^{12}$ multiplication: ~54 $\mathbb{F}_q$ operations (Karatsuba for degree-12 polynomials)
- Each $\mathbb{F}_q$ operation: ~1000 RISC-V cycles (256-bit modular arithmetic)
- **Total per exponentiation**: ~381 × 54 × 1000 ≈ **10M cycles**
- **For 109 exponentiations**: 109 × 10M = **1.09B cycles**

**The SZ-Check solution**: Instead of computing the exponentiations:
1. **Accept results as hints**: Prover provides all 93 results (64 from main protocol + 29 from combination) as untrusted advice
2. **Commit using Hyrax**: Commit to these results using Hyrax over Grumpkin
3. **Prove correctness via algorithm structure**: Use sumcheck to prove each $w_i$ equals $g_i^{x_i}$ by verifying the square-and-multiply intermediate steps

**SZ** stands for **Schwartz-Zippel**, referring to the lemma that ensures soundness of the sumcheck-based proof.

### 3.7 The Mathematics of SZ-Check

**Square-and-multiply algorithm**: To compute $g^x$ where $x = \sum_{j=0}^{253} b_j 2^j$ (binary representation):

$$\begin{aligned}
r_0 &= 1 \\
r_{j+1} &= \begin{cases}
r_j^2 & \text{if } b_j = 0 \\
r_j^2 \cdot g & \text{if } b_j = 1
\end{cases}
\end{aligned}$$

After 254 steps: $r_{254} = g^x$

**Why is this valid?** Quick example with $g = 3$ and $x = 11 = 1011_2$:

| Step | $b_j$ | $r_j$ value | Exponent so far |
|------|-------|-------------|-----------------|
| 0    | -     | $r_0 = 1$   | $3^0 = 1$ |
| 1    | $b_3=1$ | $r_1 = 1^2 \cdot 3 = 3$ | $3^1$ |
| 2    | $b_2=0$ | $r_2 = 3^2 = 9$ | $3^2$ |
| 3    | $b_1=1$ | $r_3 = 9^2 \cdot 3 = 243$ | $3^5$ |
| 4    | $b_0=1$ | $r_4 = 243^2 \cdot 3 = 177147$ | $3^{11}$ ✓ |

**Pattern**: Reading bits MSB to LSB, each step doubles the exponent (via squaring) and optionally adds 1 (via multiplying by $g$). So: $1 \to 1 \to 2 \to 5 \to 11$ in binary: $0_2 \to 1_2 \to 10_2 \to 101_2 \to 1011_2$.

**Key insight**: Instead of computing this, we can **prove the algorithm ran correctly** by checking each step:

For each step $j$, the constraint is:
$$r_{j+1} = r_j^2 \cdot (1 + b_j(g-1))$$

This simplifies to:
$$e_j := r_{j+1} - r_j^2 - b_j \cdot r_j^2 \cdot (g - 1) = 0$$

**If all 254 constraints hold**, then $r_{254}$ must equal $g^x$ (the algorithm structure forces correctness).

**The SZ-Check protocol**:

1. **Prover's witness**: For each exponentiation $w_i = g_i^{x_i}$:
   - Compute all 254 intermediate values: $r_{i,0}, r_{i,1}, \ldots, r_{i,254}$
   - Each $r_{i,j} \in \mathbb{G}_T \cong \mathbb{F}_q^{12}$ (12 base field coefficients)
   - Create multilinear polynomial $\tilde{r}_i(j)$ interpolating these 254 values over log₂(256) = 8 variables

2. **Hyrax commitment structure**:
   - **64 separate Hyrax commitments**, one per exponentiation's intermediate chain
   - Each commitment is to an 8-variable polynomial (256 coefficients after padding)
   - But since each $r_{i,j}$ has 12 $\mathbb{F}_q$ coefficients, each MLE $\tilde{r}_i$ actually encodes 256 × 12 = 3,072 field elements
   - **Hyrax verification per commitment**: 2 MSMs of size √3072 ≈ 55 group operations each
   - **Total Hyrax verification**: 64 commitments × 2 MSMs × 55 ops ≈ 7,040 Grumpkin scalar multiplications

3. **Sumcheck proof**: Prove that all square-and-multiply constraints hold:
   $$\sum_{i=1}^{64} \gamma^i \cdot \sum_{j=0}^{253} e_{i,j} = 0$$

   Where:
   - $e_{i,j} = r_{i,j+1} - r_{i,j}^2 \cdot (1 + b_j(g_i - 1))$ is the error for exponentiation $i$, step $j$
   - $\gamma$ is a random challenge batching all 64 exponentiations into one sumcheck
   - **Batching strategy**: Random linear combination reduces 64 separate sumchecks to 1

**Cost analysis**:
- **Prover**: Must compute 64 × 254 intermediate values (same as computing exponentiations directly!)
  - But this happens off-chain, we only care about verifier cost
- **Verifier**: Sumcheck with ~18 rounds (log₂ 200K)
  - Each round: degree-3 polynomial evaluation (~10 field operations)
  - Total: ~180 field operations = **~180K cycles** (vs 1.28B to compute directly)
  - **Speedup**: 7000× faster!

**Security**: By Schwartz-Zippel lemma, if the prover cheats (provides wrong $w_i$), they cannot construct consistent intermediate values except with probability $\leq \frac{254 \cdot 64 \cdot 3}{|\mathbb{F}|} \approx 2^{-240}$ (negligible).

### 3.7.1 Concrete Small Example: Computing $g^{13}$ Step-by-Step

Let's walk through proving $w = g^{13}$ where $g \in \mathbb{G}_T$ is some element from Dory verification.

**Setup**:
- Base: $g = 5$ (using simple number for clarity; in reality $g \in \mathbb{G}_T \cong \mathbb{F}_q^{12}$)
- Exponent: $x = 13 = 1101_2$ (binary representation)
- Claim: $w = 5^{13} = 1220703125$

**Step 1: Square-and-multiply algorithm**

Process bits from MSB to LSB: $b_3 b_2 b_1 b_0 = 1101_2$

$$\begin{array}{c|c|c|c|l}
\text{Step } j & b_j & \text{Operation} & r_j & \text{Check} \\
\hline
0 & - & \text{Initialize} & 1 & - \\
1 & b_3=1 & r_1 = r_0^2 \cdot g = 1^2 \cdot 5 = 5 & 5 & r_1 = 5 \\
2 & b_2=1 & r_2 = r_1^2 \cdot g = 5^2 \cdot 5 = 125 & 125 & r_2 = 125 = 5^3 \\
3 & b_1=0 & r_3 = r_2^2 = 125^2 = 15625 & 15625 & r_3 = 15625 = 5^6 \\
4 & b_0=1 & r_4 = r_3^2 \cdot g = 15625^2 \cdot 5 = 1220703125 & 1220703125 & r_4 = 5^{13}
\end{array}$$

**Verification**: $5^{13} = 1220703125$ ✓

**Step 2: Prover creates witness polynomial**

The prover commits to $\tilde{r}[j]$ using Hyrax:

$$\tilde{r}[j] = \begin{cases}
1 & \text{if } j = 0 \\
5 & \text{if } j = 1 \\
125 & \text{if } j = 2 \\
15625 & \text{if } j = 3 \\
1220703125 & \text{if } j = 4 \\
0 & \text{if } j \geq 5 \text{ (padding)}
\end{cases}$$

Represented as multilinear polynomial over $\log_2 8 = 3$ variables (padded to power of 2).

**Step 3: SZ-Check constraint verification**

For each step $j$, verify: $r_{j+1} = r_j^2 \cdot (1 + b_j(g-1))$

$$\begin{aligned}
e_0 &= r_1 - r_0^2 \cdot (1 + b_3 \cdot (5-1)) \\
&= 5 - 1^2 \cdot (1 + 1 \cdot 4) \\
&= 5 - 1 \cdot 5 = 0 \quad \checkmark \\
\\
e_1 &= r_2 - r_1^2 \cdot (1 + b_2 \cdot (5-1)) \\
&= 125 - 5^2 \cdot (1 + 1 \cdot 4) \\
&= 125 - 25 \cdot 5 = 0 \quad \checkmark \\
\\
e_2 &= r_3 - r_2^2 \cdot (1 + b_1 \cdot (5-1)) \\
&= 15625 - 125^2 \cdot (1 + 0 \cdot 4) \\
&= 15625 - 15625 \cdot 1 = 0 \quad \checkmark \\
\\
e_3 &= r_4 - r_3^2 \cdot (1 + b_0 \cdot (5-1)) \\
&= 1220703125 - 15625^2 \cdot (1 + 1 \cdot 4) \\
&= 1220703125 - 244140625 \cdot 5 \\
&= 1220703125 - 1220703125 = 0 \quad \checkmark
\end{aligned}$$

**All error terms are zero!** ✓

**Step 4: Sumcheck protocol**

Claim to prove: $\sum_{j=0}^{7} e_j = 0$ (over 8 values since we padded to 3 variables)

The verifier runs sumcheck over the error polynomial $e(x_0, x_1, x_2)$:

**Round 0**: Prover sends $s_0(X) = \sum_{x_1, x_2 \in \{0,1\}} e(X, x_1, x_2)$
- This is a univariate polynomial (degree ≤ 3) claiming the partial sum over the first variable
- Verifier checks: $s_0(0) + s_0(1) \stackrel{?}{=} 0$ (matches claimed sum)
- Verifier sends random challenge $r_0 \in \mathbb{F}$

**Round 1**: Prover sends $s_1(X) = \sum_{x_2 \in \{0,1\}} e(r_0, X, x_2)$
- Claims the partial sum at the fixed point $r_0$ over the second variable
- Verifier checks: $s_1(0) + s_1(1) \stackrel{?}{=} s_0(r_0)$ (consistency with previous round)
- Verifier sends random challenge $r_1 \in \mathbb{F}$

**Round 2**: Prover sends $s_2(X) = e(r_0, r_1, X)$
- Claims the evaluation along the last variable
- Verifier checks: $s_2(0) + s_2(1) \stackrel{?}{=} s_1(r_1)$ (consistency)
- Verifier sends random challenge $r_2 \in \mathbb{F}$

**Final check**: Verifier queries $\tilde{r}$ at $(r_0, r_1, r_2)$ via Hyrax opening and computes $e(r_0, r_1, r_2)$ directly, checking it equals $s_2(r_2)$

**Security**: If prover tries to cheat with $w' = 5^{14} = 6103515625$ (wrong):
- Chain must end at $r_4 = 6103515625$
- But following algorithm with $x = 1101_2$ forces $r_4 = 5^{13} = 1220703125$
- To claim $r_4 = 5^{14}$, prover must violate a constraint:
  $$e_3 = 6103515625 - 15625^2 \cdot 5 = 6103515625 - 1220703125 \neq 0$$
- Sumcheck fails! ✗

**Step 5: Scaling to real exponentiations**

In practice:
- Base $g \in \mathbb{G}_T \cong \mathbb{F}_q^{12}$ (12 field elements, each 256 bits)
- Exponent $x$ is 254 bits → 254 steps
- Each $r_j \in \mathbb{G}_T$ → each intermediate value is 12 field elements
- Hyrax commits to 254 × 12 = 3048 field elements per exponentiation
- For 64 exponentiations: 64 × 3048 = 195,072 field elements
- Padded to $2^{18} = 262,144$ for efficient sumcheck

**Cost comparison**:
- **Computing directly**: 64 × 254 steps × 54 $\mathbb{F}_q$ ops = 878K field operations
- **SZ-Check**: ~18 sumcheck rounds × 10 field ops = ~180 field operations
- **Speedup**: 4880× faster!

### 3.8 Hyrax Opening Protocol and Why Recursion Terminates

**The key question**: Layer 2 uses Dory for main trace. Why doesn't verifying Layer 2 create the same problem (requiring Layer 3)?

**Answer**: Layer 2's expensive operations (Dory G_T exponentiations) are **not computed**, they're **accepted as hints and proven with SZ-check**. The SZ-check verification uses Hyrax openings, which have cheap verification. Let's see how this works with a concrete example.

#### 3.8.1 Hyrax Opening - Concrete Toy Example

Continuing from the $5^{13}$ example in section 3.7.1, let's see how the Hyrax opening works after the sumcheck protocol.

After sumcheck binds all variables to random challenges, the verifier needs to check the polynomial evaluation. Since we're padding to a 4×4 matrix (16 coefficients, 4 variables), we have **4 challenges**. Let's use concrete numbers:

**Suppose sumcheck gave challenges**: $r_0 = 0.3$, $r_1 = 0.7$, $r_2 = 0.4$, $r_3 = 0.9$

**Prover claims**: $\tilde{r}(0.3, 0.7, 0.4, 0.9) = v$ (some value)

**Note**: Our original polynomial had only 8 coefficients (3 variables), but we padded to 16 (4 variables) for the square matrix. The extra 8 coefficients are zeros.

**Hyrax opening proof structure**:

**1. Matrix embedding**: The polynomial coefficients $[1, 5, 125, 15625, 1220703125, 0, 0, 0]$ are arranged in a $\sqrt{8} \times \sqrt{8}$ matrix (pad to 4×4 for simplicity):

$$M = \begin{pmatrix}
1 & 5 & 125 & 15625 \\
1220703125 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix}$$

**2. Row commitments** (Pedersen commitments on Grumpkin):
   - $C_0 = 1 \cdot [G_0] + 5 \cdot [G_1] + 125 \cdot [G_2] + 15625 \cdot [G_3] + r_0' \cdot [H]$
   - $C_1 = 1220703125 \cdot [G_0] + 0 \cdot [G_1] + 0 \cdot [G_2] + 0 \cdot [G_3] + r_1' \cdot [H]$
   - $C_2 = 0 \cdot [G_0] + 0 \cdot [G_1] + 0 \cdot [G_2] + 0 \cdot [G_3] + r_2' \cdot [H]$
   - $C_3 = 0 \cdot [G_0] + 0 \cdot [G_1] + 0 \cdot [G_2] + 0 \cdot [G_3] + r_3' \cdot [H]$

   Where $r_i'$ are random blinding factors, and $[G_j], [H]$ are random Grumpkin generator points.

**3. Hyrax opening protocol** (to prove $\tilde{r}(0.3, 0.7, 0.4, 0.9) = v$):

   The verifier needs to check that the claimed evaluation is consistent with the commitments. The key insight is that we can split the 4-variable polynomial evaluation into row and column parts using the matrix structure.

   **Challenge decomposition - From 3 variables to 4**:

   **Step 1: The original polynomial structure**
   - **Original**: $\tilde{r}(x_0, x_1, x_2)$ has 8 coefficients over $\{0,1\}^3$ (from our SZ-Check example)
   - **Natural 2×4 matrix view**:
     - Variable $x_0 \in \{0,1\}$ → row selector (2 rows)
     - Variables $(x_1, x_2) \in \{0,1\}^2$ → column selector (4 columns)
   - **Coefficient mapping**: Index $(b_0, b_1, b_2) \in \{0,1\}^3$ → matrix position $(b_0, 2b_1 + b_2)$
     - Row 0: $(0,0,0) \to M[0,0]$, $(0,0,1) \to M[0,1]$, $(0,1,0) \to M[0,2]$, $(0,1,1) \to M[0,3]$
     - Row 1: $(1,0,0) \to M[1,0]$, $(1,0,1) \to M[1,1]$, $(1,1,0) \to M[1,2]$, $(1,1,1) \to M[1,3]$

   **Step 2: Padding for Hyrax**
   - **Problem**: Hyrax needs $\sqrt{N} \times \sqrt{N}$ square matrices, but $\sqrt{8} \approx 2.83$ (not an integer)
   - **Solution**: Pad to 4×4 (16 coefficients) by adding a dummy variable $x_3$
   - **Padded polynomial**: $\tilde{r}_{\text{padded}}(x_0, x_1, x_2, x_3)$ over $\{0,1\}^4$
     - Original 8 coefficients: $(x_0, x_1, x_2, x_3)$ where $x_3 = 0$
     - Padded 8 coefficients: $(x_0, x_1, x_2, x_3)$ where $x_3 = 1$ → all set to 0
   - **Result**: 4×4 matrix with last 2 rows being zeros

   **Step 3: How challenges map** (for 4×4 matrix with 4 variables):
   - $(r_0, r_1)$ corresponds to variables $(x_0, x_1)$ → determines **row weights** $\lambda_i$ for $i \in \{0,1,2,3\}$
   - $(r_2, r_3)$ corresponds to variables $(x_2, x_3)$ → determines **column weights** $\mu_j$ for $j \in \{0,1,2,3\}$
   - **Note**: $x_3$ is the dummy padding variable, so its challenge $r_3$ effectively interpolates over the padded dimension

   **Important - Row selection with powers of 2**:

   - **Fundamental constraint**: In Hyrax, the number of rows must be a power of 2 (since rows are selected by bits)
     - 1 bit → 2 rows (like $x_0 \in \{0,1\}$)
     - 2 bits → 4 rows (like $(x_0, x_1) \in \{0,1\}^2$)
     - 3 bits → 8 rows, etc.
     - **You cannot have exactly 3 rows!**

   - **Our 3-variable example problem**: We have 3 variables total but want a square matrix
     - If we use 1 variable for rows: 2 rows (need 3 more variables for 8 columns - doesn't work)
     - If we use 2 variables for rows: 4 rows (need only 1 variable for 2 columns - gives 2×8, not square)
     - **Solution**: Pad from 3 variables to 4 variables (8 → 16 coefficients), then split as 2+2 for 4×4

   - **How challenges work in 4×4**: With 4 variables $(x_0, x_1, x_2, x_3)$ split as 2+2:
     - Row selector: $(x_0, x_1) \in \{0,1\}^2$ → at challenge $(r_0, r_1)$, multilinear extension gives **4 row weights**
     - Column selector: $(x_2, x_3) \in \{0,1\}^2$ → at challenge $(r_2, r_3)$, gives **4 column weights**

   - **For our original 8 coefficients**: We pad 8 → 16, so the last 2 rows are zeros. Thus $\lambda_2, \lambda_3$ multiply zero rows.

   So evaluating $\tilde{r}(r_0, r_1, r_2, r_3)$ becomes:
   1. Use $(r_0, r_1)$ to take a weighted combination of rows → yields a "virtual row" of 4 values
   2. Use $(r_2, r_3)$ to take a weighted combination of those 4 values → yields final scalar

   **Step 3a**: Prover computes **column polynomial evaluations at the row challenge**

   For each column $j$, compute the weighted sum of that column's values using row weights:

   $$\text{col}_j = \sum_{i=0}^{3} \lambda_i \cdot M_{i,j}$$

   Where $\lambda_i$ are weights derived from $(r_0, r_1)$ via the **multilinear Lagrange basis** over $\{0,1\}^2$.

   **Understanding the construction**: For any challenges $(a, b)$ and index $i = (b_0, b_1)$ in binary:
   $$\text{weight}_i = \chi_{b_0}(a) \cdot \chi_{b_1}(b)$$

   Where $\chi_b(r) = b \cdot r + (1-b)(1-r)$ is the univariate Lagrange basis:
   - $\chi_0(r) = 1-r$ (weight for bit = 0)
   - $\chi_1(r) = r$ (weight for bit = 1)

   **Row weights** for $(r_0, r_1) = (0.3, 0.7)$:
   - $\lambda_0 = \chi_0(r_0) \cdot \chi_0(r_1) = (1-r_0)(1-r_1) = (1-0.3)(1-0.7) = 0.21$ (row $00_2 = 0$)
   - $\lambda_1 = \chi_0(r_0) \cdot \chi_1(r_1) = (1-r_0) \cdot r_1 = (1-0.3) \times 0.7 = 0.49$ (row $01_2 = 1$)
   - $\lambda_2 = \chi_1(r_0) \cdot \chi_0(r_1) = r_0 \cdot (1-r_1) = 0.3 \times (1-0.7) = 0.09$ (row $10_2 = 2$)
   - $\lambda_3 = \chi_1(r_0) \cdot \chi_1(r_1) = r_0 \cdot r_1 = 0.3 \times 0.7 = 0.21$ (row $11_2 = 3$)

   **Note**: These sum to 1 as expected. Since rows 2 and 3 are padding (all zeros), $\lambda_2$ and $\lambda_3$ don't affect the final result.

   **Concrete calculation** (using actual matrix values and $\lambda$ weights):

   Using $\lambda_0 = 0.21$, $\lambda_1 = 0.49$, $\lambda_2 = 0.09$, $\lambda_3 = 0.21$:

   - $\text{col}_0 = 0.21 \cdot 1 + 0.49 \cdot 1220703125 + 0.09 \cdot 0 + 0.21 \cdot 0 = 598{,}144{,}531.46$
   - $\text{col}_1 = 0.21 \cdot 5 + 0.49 \cdot 0 + 0.09 \cdot 0 + 0.21 \cdot 0 = 1.05$
   - $\text{col}_2 = 0.21 \cdot 125 + 0.49 \cdot 0 + 0.09 \cdot 0 + 0.21 \cdot 0 = 26.25$
   - $\text{col}_3 = 0.21 \cdot 15625 + 0.49 \cdot 0 + 0.09 \cdot 0 + 0.21 \cdot 0 = 3{,}281.25$

   These 4 column values represent the polynomial evaluated at row challenge $(r_0, r_1) = (0.3, 0.7)$ for each column.

   **CRITICAL: Understanding Hyrax's Two-Phase Protocol**

   Hyrax uses a **matrix commitment scheme** with two distinct phases:

   **Phase 1: Commitment (happens during preprocessing or earlier in proof)**

   The prover commits to the polynomial by committing to each **row** of the matrix separately using Pedersen commitments:

   - **Row 0**: $C_0 = [1]G_0 + [5]G_1 + [125]G_2 + [15{,}625]G_3 + [r_0']H$
     - This commits to the 4 values in row 0: $(1, 5, 125, 15{,}625)$
     - $r_0'$ is a random blinding factor (keeps commitment hiding)

   - **Row 1**: $C_1 = [1{,}220{,}703{,}125]G_0 + [0]G_1 + [0]G_2 + [0]G_3 + [r_1']H$
     - Commits to row 1: $(1{,}220{,}703{,}125, 0, 0, 0)$

   - **Row 2**: $C_2 = [0]G_0 + [0]G_1 + [0]G_2 + [0]G_3 + [r_2']H$
     - Commits to row 2: $(0, 0, 0, 0)$ (padding)

   - **Row 3**: $C_3 = [0]G_0 + [0]G_1 + [0]G_2 + [0]G_3 + [r_3']H$
     - Commits to row 3: $(0, 0, 0, 0)$ (padding)

   Prover **sends** $C_0, C_1, C_2, C_3$ to verifier (4 elliptic curve points on Grumpkin).

   Verifier **stores** these commitments - they cannot be changed later (binding property).

   **Phase 2: Opening (happens now, after sumcheck gives challenges)**

   The verifier wants to know: What is $\tilde{r}(r_0, r_1, r_2, r_3) = \tilde{r}(0.3, 0.7, 0.4, 0.9)$?

   **What the prover computes**: Using the matrix structure, evaluating the polynomial happens in two steps:
   1. "Collapse" rows using $(r_0, r_1)$ → get a 1D column vector
   2. "Collapse" columns using $(r_2, r_3)$ → get a single scalar

   **Step 1 - How prover computes column values**: For each column $j \in \{0,1,2,3\}$, compute weighted sum across all rows:

   $$\text{col}_j = \sum_{i=0}^{3} \lambda_i \cdot M_{i,j}$$

   Where $\lambda_i$ are row weights from $(r_0, r_1) = (0.3, 0.7)$: $\lambda_0 = 0.21, \lambda_1 = 0.49, \lambda_2 = 0.09, \lambda_3 = 0.21$

   **Concrete calculations**:
   - $\text{col}_0 = 0.21 \cdot 1 + 0.49 \cdot 1{,}220{,}703{,}125 + 0.09 \cdot 0 + 0.21 \cdot 0 = 598{,}144{,}531.46$
   - $\text{col}_1 = 0.21 \cdot 5 + 0.49 \cdot 0 + 0.09 \cdot 0 + 0.21 \cdot 0 = 1.05$
   - $\text{col}_2 = 0.21 \cdot 125 + 0.49 \cdot 0 + 0.09 \cdot 0 + 0.21 \cdot 0 = 26.25$
   - $\text{col}_3 = 0.21 \cdot 15{,}625 + 0.49 \cdot 0 + 0.09 \cdot 0 + 0.21 \cdot 0 = 3{,}281.25$

   **What these values represent mathematically**: Each $\text{col}_j$ is the polynomial evaluated at row challenge $(r_0, r_1)$ and column Boolean index $j$:
   - $\text{col}_0 = \tilde{r}(0.3, 0.7, 0, 0)$ — column index $(0,0)_2$
   - $\text{col}_1 = \tilde{r}(0.3, 0.7, 0, 1)$ — column index $(0,1)_2$
   - $\text{col}_2 = \tilde{r}(0.3, 0.7, 1, 0)$ — column index $(1,0)_2$
   - $\text{col}_3 = \tilde{r}(0.3, 0.7, 1, 1)$ — column index $(1,1)_2$

   **What prover sends to verifier** (in the clear, not committed):
   1. **The 4 column values**: $\text{col}_0 = 598{,}144{,}531.46$, $\text{col}_1 = 1.05$, $\text{col}_2 = 26.25$, $\text{col}_3 = 3{,}281.25$
   2. **The combined blinding factor**: $r' = 0.21 r_0' + 0.49 r_1' + 0.09 r_2' + 0.21 r_3'$

   **What verifier already has**:
   - Row commitments $C_0, C_1, C_2, C_3$ (from Phase 1)
   - Generators $G_0, G_1, G_2, G_3, H$ (public parameters from Dory SRS)
   - Challenges $(r_0, r_1, r_2, r_3)$ (just received from sumcheck)
   - Row weights $\lambda_0 = 0.21, \lambda_1 = 0.49, \lambda_2 = 0.09, \lambda_3 = 0.21$ (computed from $(r_0, r_1)$)

   **Why this is secure**:
   - The column values are **not commitments** - they're revealed in the clear as field elements
   - But the prover can't lie because MSM 1 checks consistency:
     - Left side: Weighted combination of **committed rows** (uses $C_0, C_1, C_2, C_3$ that can't be changed)
     - Right side: What you'd get if the **claimed column values** are correct
     - If prover lies, these elliptic curve points won't match!

   **Step 3b**: Verifier performs two checks using MSMs:

   **MSM 1 (Row commitment check)**: Verify that claimed column values are consistent with row commitments.

   The verifier computes:
   $$\text{Left} = \sum_{i=0}^{3} \lambda_i \cdot [C_i]$$

   $$\text{Right} = \sum_{j=0}^{3} \text{col}_j \cdot [G_j] + \text{blinding} \cdot [H]$$

   Check: $\text{Left} \stackrel{?}{=} \text{Right}$

   **Why this works**: The left side is a linear combination of row commitments weighted by $\lambda_i$ (from $(r_0, r_1)$). The right side reconstructs what that combination should be using the claimed column values. If they match, the column values are consistent with the original commitments.

   **In our example** with $\lambda_0 = 0.21$, $\lambda_1 = 0.49$, $\lambda_2 = 0.09$, $\lambda_3 = 0.21$:

   $$\text{Left} = 0.21 \cdot [C_0] + 0.49 \cdot [C_1] + 0.09 \cdot [C_2] + 0.21 \cdot [C_3]$$

   Expanding $C_0$ and $C_1$ (the non-zero rows):
   $$= 0.21 \cdot (1[G_0] + 5[G_1] + 125[G_2] + 15625[G_3] + r_0'[H])$$
   $$+ 0.49 \cdot (1220703125[G_0] + 0[G_1] + 0[G_2] + 0[G_3] + r_1'[H]) + \ldots$$

   $$= (0.21 \cdot 1 + 0.49 \cdot 1220703125)[G_0] + (0.21 \cdot 5)[G_1] + (0.21 \cdot 125)[G_2]$$
   $$+ (0.21 \cdot 15625)[G_3] + (0.21 r_0' + 0.49 r_1' + \ldots)[H]$$

   $$= 598{,}144{,}531.46[G_0] + 1.05[G_1] + 26.25[G_2] + 3{,}281.25[G_3] + \text{blinding}[H]$$

   **Now let's compute the right side**: The prover claims the polynomial evaluated at row point $(r_0, r_1)$ gives a row vector with 4 column values. Let's call these claimed values $(\text{col}_0, \text{col}_1, \text{col}_2, \text{col}_3)$.

   The right side computes:
   $$\text{Right} = \text{col}_0 \cdot [G_0] + \text{col}_1 \cdot [G_1] + \text{col}_2 \cdot [G_2] + \text{col}_3 \cdot [G_3] + r'[H]$$

   For the check to pass, the prover must claim:
   - $\text{col}_0 = 598{,}144{,}531.46$
   - $\text{col}_1 = 1.05$
   - $\text{col}_2 = 26.25$
   - $\text{col}_3 = 3{,}281.25$
   - $r' = 0.21 r_0' + 0.49 r_1' + 0.09 r_2' + 0.21 r_3'$ (correct blinding factor combination)

   Substituting into right side:
   $$\text{Right} = 598{,}144{,}531.46[G_0] + 1.05[G_1] + 26.25[G_2] + 3{,}281.25[G_3] + (0.21 r_0' + 0.49 r_1' + \ldots)[H]$$

   This exactly matches the left side! ✓

   **Why should Left = Right? The Mathematical Reason**

   Both sides are computing the same thing via different paths - this is the genius of Hyrax's matrix commitment.

   **Left side expansion** (using linearity of Pedersen commitments):
   $$\text{Left} = \sum_{i=0}^{3} \lambda_i \cdot C_i = \sum_{i=0}^{3} \lambda_i \cdot \left(\sum_{j=0}^{3} M_{i,j} \cdot [G_j] + r_i'[H]\right)$$

   Rearranging the double sum (switching order of summation):
   $$\text{Left} = \sum_{j=0}^{3} \underbrace{\left(\sum_{i=0}^{3} \lambda_i \cdot M_{i,j}\right)}_{\text{This is } \text{col}_j!} \cdot [G_j] + \underbrace{\left(\sum_{i=0}^{3} \lambda_i \cdot r_i'\right)}_{\text{This is } r'!}[H]$$

   The inner sum $\sum_{i=0}^{3} \lambda_i \cdot M_{i,j}$ is exactly how we defined $\text{col}_j$ - the weighted sum of column $j$ across all rows!

   Therefore:
   $$\text{Left} = \sum_{j=0}^{3} \text{col}_j \cdot [G_j] + r'[H] = \text{Right}$$

   **When are they equal?**
   - Prover sends **correct** column values: $\text{col}_j = \sum_{i=0}^{3} \lambda_i \cdot M_{i,j}$
   - Prover sends **correct** blinding: $r' = \sum_{i=0}^{3} \lambda_i \cdot r_i'$

   **Security**:
   - If prover lies, the elliptic curve equation won't balance
   - Finding alternative values that work requires solving discrete log (computationally infeasible)
   - The commitments $C_i$ are binding - prover can't change the matrix after committing

   **MSM 2 (Column evaluation check)**: Verify the final polynomial evaluation using the column polynomial and remaining challenges $(r_2, r_3)$.

   **What just happened in MSM 1**: MSM 1 gave us 4 column values $(\text{col}_0, \text{col}_1, \text{col}_2, \text{col}_3)$ that represent the "collapsed row" at the evaluation point $(r_0, r_1)$. Think of this as: "if we fix the row coordinates to $(r_0, r_1)$, the polynomial becomes a 1D function over columns."

   **What MSM 2 does**: Now we need to collapse the columns using $(r_2, r_3)$ to get the final scalar value $v = \tilde{r}(r_0, r_1, r_2, r_3)$.

   **Visual: Two-Step Dimension Reduction**

   **Step 0: 4×4 Matrix** (polynomial coefficients with row weights)

   $$M = \begin{pmatrix}
   1 & 5 & 125 & 15{,}625 \\
   1{,}220{,}703{,}125 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0
   \end{pmatrix} \quad
   \begin{array}{l}
   \lambda_0 = 0.21 \\
   \lambda_1 = 0.49 \\
   \lambda_2 = 0.09 \\
   \lambda_3 = 0.21
   \end{array}$$

   $$\downarrow \text{ MSM 1: Collapse rows using } (r_0, r_1) = (0.3, 0.7) \text{, compute } \text{col}_j = \sum_{i=0}^{3} \lambda_i \cdot M_{i,j}$$

   **Step 1: 1×4 Column Vector** (after collapsing rows with column weights)

   $$\begin{pmatrix} 598{,}144{,}531.46 & 1.05 & 26.25 & 3{,}281.25 \end{pmatrix}$$

   $$\begin{array}{cccc}
   \uparrow & \uparrow & \uparrow & \uparrow \\
   \mu_0 = 0.06 & \mu_1 = 0.54 & \mu_2 = 0.04 & \mu_3 = 0.36
   \end{array}$$

   $$\downarrow \text{ MSM 2: Collapse columns using } (r_2, r_3) = (0.4, 0.9) \text{, compute } v = \sum_{j=0}^{3} \mu_j \cdot \text{col}_j$$

   **Step 2: Single Scalar** (final polynomial evaluation)

   $$v = 35{,}889{,}854.76 = \tilde{r}(0.3, 0.7, 0.4, 0.9)$$

   **Key insight**:
   - **2D → 1D (MSM 1)**: Use $(r_0, r_1)$ to collapse 4 rows → get 4-element column vector
   - **1D → 0D (MSM 2)**: Use $(r_2, r_3)$ to collapse 4 columns → get single scalar
   - This is how multilinear extension evaluation works: bind variables progressively

   **Computing column weights** from $(r_2, r_3) = (0.4, 0.9)$ using the **same Lagrange basis construction as $\lambda$**:

   For column index $j = (b_2, b_3)$ in binary:
   $$\mu_j = \chi_{b_2}(r_2) \cdot \chi_{b_3}(r_3)$$

   **Column weights** (identical formula to row weights, just different challenges):
   - $\mu_0 = \chi_0(r_2) \cdot \chi_0(r_3) = (1-r_2)(1-r_3) = (1-0.4)(1-0.9) = 0.06$ (column $00_2 = 0$)
   - $\mu_1 = \chi_0(r_2) \cdot \chi_1(r_3) = (1-r_2) \cdot r_3 = (1-0.4) \times 0.9 = 0.54$ (column $01_2 = 1$)
   - $\mu_2 = \chi_1(r_2) \cdot \chi_0(r_3) = r_2 \cdot (1-r_3) = 0.4 \times (1-0.9) = 0.04$ (column $10_2 = 2$)
   - $\mu_3 = \chi_1(r_2) \cdot \chi_1(r_3) = r_2 \cdot r_3 = 0.4 \times 0.9 = 0.36$ (column $11_2 = 3$)

   **Key point**: $\mu$ values use the exact same Lagrange basis formula as $\lambda$ values - this is how multilinear extensions work!

   **In our example** with column values from MSM 1:
   - $\text{col}_0 = 598{,}144{,}531.46$
   - $\text{col}_1 = 1.05$
   - $\text{col}_2 = 26.25$
   - $\text{col}_3 = 3{,}281.25$

   Computing the final evaluation:
   $$v = \mu_0 \cdot \text{col}_0 + \mu_1 \cdot \text{col}_1 + \mu_2 \cdot \text{col}_2 + \mu_3 \cdot \text{col}_3$$
   $$v = 0.06 \cdot 598{,}144{,}531.46 + 0.54 \cdot 1.05 + 0.04 \cdot 26.25 + 0.36 \cdot 3{,}281.25$$
   $$= 35{,}888{,}671.89 + 0.567 + 1.05 + 1{,}181.25 = 35{,}889{,}854.76$$

   **Verification**: If the prover claimed at the start that $\tilde{r}(0.3, 0.7, 0.4, 0.9) = 35{,}889{,}854.76$, and our calculation gives the same value, then the opening is valid! ✓

   **Why does $v = \sum_{j=0}^{3} \mu_j \cdot \text{col}_j$ equal $\tilde{r}(0.3, 0.7, 0.4, 0.9)$?**

   Recall from MSM 1 that $\text{col}_j = \sum_{i=0}^{3} \lambda_i \cdot M_{i,j}$. Substituting:

   $$v = \sum_{j=0}^{3} \mu_j \cdot \text{col}_j = \sum_{j=0}^{3} \mu_j \cdot \left(\sum_{i=0}^{3} \lambda_i \cdot M_{i,j}\right)$$

   Rearranging (switching summation order):
   $$v = \sum_{i=0}^{3} \sum_{j=0}^{3} \lambda_i \cdot \mu_j \cdot M_{i,j}$$

   **This is the multilinear extension formula!** The matrix index $(i, j)$ encodes a 4-bit boolean index $(b_0, b_1, b_2, b_3)$ where:
   - Row $i$ encodes bits $(b_0, b_1)$
   - Column $j$ encodes bits $(b_2, b_3)$

   The weights are Lagrange basis products:
   - $\lambda_i = \chi_{b_0}(r_0) \cdot \chi_{b_1}(r_1)$ for row $i = (b_0, b_1)$
   - $\mu_j = \chi_{b_2}(r_2) \cdot \chi_{b_3}(r_3)$ for column $j = (b_2, b_3)$

   Where $\chi_b(r) = b \cdot r + (1-b)(1-r)$ is the Lagrange basis function.

   Therefore:
   $$\lambda_i \cdot \mu_j = \prod_{k=0}^{3} \chi_{b_k}(r_k)$$

   And since $M_{i,j} = r(b_0, b_1, b_2, b_3)$:
   $$v = \sum_{(b_0,b_1,b_2,b_3) \in \{0,1\}^4} r(b_0, b_1, b_2, b_3) \cdot \prod_{k=0}^{3} \chi_{b_k}(r_k) = \tilde{r}(r_0, r_1, r_2, r_3)$$

   **Conclusion**: Hyrax's two-step evaluation is mathematically equivalent to the multilinear extension formula, just organized as matrix operations for efficiency!

   **What we verified**: The prover's claimed column values (from MSM 1) correctly collapse to the claimed polynomial evaluation (MSM 2), and those column values are consistent with the original committed rows (MSM 1 left-right check).

   **Security**: If the prover cheats with wrong polynomial values, they must either:
   1. Break the binding of Pedersen commitments (computationally hard), or
   2. Find column values that satisfy both MSM checks with wrong data (probability $\leq 1/|\mathbb{F}|$ by Schwartz-Zippel)

**4. Verification cost**:
   - MSM 1: 4 scalar multiplications on Grumpkin (size of matrix row)
   - MSM 2: 4 scalar multiplications on Grumpkin (size of matrix column)
   - Total: **8 Grumpkin scalar multiplications** (very cheap in RISC-V!)
   - Compare to: Computing $5^{13}$ directly = ~20M RISC-V cycles

**Key insight**: The matrix structure lets Hyrax verify with only $2\sqrt{N}$ operations instead of $N$.

**In the real case** (from section 3.7.1):
- Matrix size: $\sqrt{3048} \times \sqrt{3048} \approx 55 \times 55$ (for one exponentiation's intermediate chain)
- Verification: 2 MSMs of 55 operations each = 110 Grumpkin scalar multiplications
- For 64 exponentiations: 64 × 110 = 7,040 scalar multiplications
- At ~100K cycles each: 7,040 × 100K ≈ **700M cycles** for all Hyrax openings

This is still much better than 1.28B cycles to compute exponentiations directly!

**5. Critical Security Question: How Do We Trust the Dory Commitments?**

You might ask: "What prevents the prover from committing to fake exponentiation results with Dory, then making matching Hyrax commitments to consistent-but-wrong intermediate chains?"

**The answer**: The Dory commitments are **not verified in Layer 2** - they're **verified in Layer 1**, and Layer 2 treats them as **public input data**.

**The complete verification flow**:

1. **Layer 1 proof generation**:
   - Guest program executes (e.g., SHA3)
   - Jolt prover creates π₁
   - π₁ includes Dory commitments to trace polynomials
   - π₁ includes 64 G_T exponentiation results (as **output values**, not commitments)

2. **Layer 1 verification** (the expensive 1.5B cycle operation):
   - Verifier checks all Dory commitments via opening proofs
   - Verifier **computes the 64 G_T exponentiations** (this is the expensive part!)
   - Verifier confirms these match claimed output values
   - If verification passes: **Dory commitments are correct, exponentiations are correct**

3. **Layer 2 proof generation** (recursion):
   - Guest = Layer 1 verifier code
   - **Input**: π₁ + the 64 exponentiation values (as public inputs, not recomputed)
   - Guest code **reads** exponentiation values from I/O memory (doesn't compute them)
   - Creates Hyrax commitments to exponentiation witnesses
   - SZ-Check proves: "IF the input exponentiation values are correct, THEN I verified Layer 1 correctly"

4. **Layer 2 verification**:
   - Verifies Layer 2's trace (that it executed the verifier code correctly)
   - **Checks public I/O**: The exponentiation values Layer 2 read match Layer 1's claimed values
   - Hyrax/SZ-Check proves the intermediate chains are consistent with those input values

**The binding happens through cryptographic commitment to π₁**:

This is the critical part: Layer 2 doesn't just "check" some abstract proof - it must be **cryptographically bound** to the specific Layer 1 proof π₁.

**Two deployment modes** (from `examples/recursion/src/main.rs`):

1. **Embedded mode** (stronger binding, but inflexible):
   - π₁ is **embedded directly into Layer 2's bytecode** during compilation
   - Layer 2's bytecode commitment (via Dory) cryptographically binds to π₁
   - Prover cannot substitute a different proof without changing the bytecode
   - The verifier knows: "This Layer 2 proof is specifically about *this exact* Layer 1 proof π₁"
   - **Major drawback**: If π₁ changes at all (even different inputs to the original program), you must:
     - Recompile Layer 2 bytecode (with new embedded π₁)
     - **Recompute preprocessing** (expensive! ~minutes for typical programs)
     - Generate new Layer 2 proof
   - **Use case**: When you want to prove "I verified *this specific computation*" (e.g., a particular blockchain state transition)

2. **Input mode** (more flexible, practical for most use cases):
   - π₁ is provided as **input** to Layer 2 (not embedded)
   - Layer 2's input commitment (via Dory) binds to π₁
   - π₁ becomes part of Layer 2's public I/O
   - The verifier receives both π₁ and π₂, and verifies: "Layer 2 claims to have verified this specific π₁"
   - **Advantage**: **Preprocessing is reusable!** Different π₁ values can use the same preprocessing
   - Layer 2's bytecode is generic: "I'm a verifier that can verify any Jolt proof"
   - Only constraint: π₁ must fit within `max_input_size` (configured in preprocessing)
   - **Use case**: General-purpose proof aggregation (verify multiple different proofs with same Layer 2 program)

   **Trust assumption in input mode**:

   This is critical to understand: **The Layer 2 verifier does NOT independently verify π₁**. Instead:

   - Layer 2 proves: "I correctly executed the verification algorithm, and it accepted the proof I was given"
   - Layer 2 does NOT prove: "The proof I verified was for the correct computation"
   - **The verifier must trust that π₁ is the correct proof for the intended computation**

   **What this means in practice**:

   **Scenario 1: Honest setup (typical use case)**
   - User runs Layer 1 prover: generates π₁ for their computation
   - User runs Layer 2 prover: takes π₁ as input, generates π₂
   - User sends both π₁ and π₂ to verifier
   - Verifier checks: "Does π₂ prove that Layer 2 verified π₁?" → Yes
   - **But**: Verifier must also check π₁ independently OR trust the source

   **Scenario 2: What if attacker substitutes π₁?**
   - Honest user generates π₁ (proof of correct computation)
   - Attacker intercepts and replaces with π₁' (proof of different/wrong computation)
   - Attacker generates π₂' proving "I verified π₁'"
   - Verifier receives π₁' and π₂'
   - π₂' verification **succeeds** (Layer 2 did correctly verify π₁')
   - **Attack succeeds** unless verifier independently checks π₁'

   **The security model**:

   Input mode provides **proof aggregation/compression**, not end-to-end soundness:
   - **What it proves**: "Layer 2 correctly executed verification of the provided proof"
   - **What it doesn't prove**: "The provided proof is for the correct statement"

   **Two ways to use input mode securely**:

   1. **Verifier also checks π₁** (defeats the purpose of recursion):
      - Verifier receives π₁ and π₂
      - Verifier checks π₁ directly (expensive, 1.5B cycles)
      - Verifier checks π₂ (cheap, 330M cycles)
      - Both must pass
      - **Why bother?** This doesn't save verification cost!

   2. **Trust the source of π₁** (practical approach):
      - π₁ comes from a trusted prover/source
      - Or: π₁'s statement is included in π₂'s public I/O and checked separately
      - **Example**: Blockchain rollup where Layer 1 statement is "state transition A→B"
         - π₂ includes (A, B) in public I/O
         - Verifier checks: "π₂ claims to verify a proof of transition A→B"
         - This binds π₂ to the correct statement

   **Better approach: Bind statement to π₂**:

   The key is to include Layer 1's **statement** (inputs/outputs) in Layer 2's **public I/O**:

   ```rust
   // Layer 2 guest code
   #[jolt::provable]
   fn verify_with_statement(
       proof_bytes: &[u8],
       claimed_input: &[u8],
       claimed_output: &[u8]
   ) -> bool {
       let proof = deserialize(proof_bytes);
       // Check proof claims correct input/output
       assert_eq!(proof.public_input, claimed_input);
       assert_eq!(proof.public_output, claimed_output);
       // Verify proof
       JoltRV64IMAC::verify(proof, ...).is_ok()
   }
   ```

   Now:
   - Layer 2's I/O commitment includes (claimed_input, claimed_output)
   - Verifier receives these as part of π₂
   - Verifier knows: "π₂ proves verification of a proof claiming input X → output Y"
   - Prover can't substitute different π₁ without changing the statement

**Why the prover can't present a fake proof to Layer 2**:

In **embedded mode**:
- Layer 2's bytecode includes the serialized π₁
- Bytecode is committed during preprocessing: $C_{\text{bytecode}} = \text{Commit}(\text{bytecode containing } \pi_1)$
- This commitment is public and given to the verifier
- If prover tries to use different π₁' in execution, the bytecode trace won't match $C_{\text{bytecode}}$
- The bytecode sumcheck (one of Jolt's 5 components) catches the mismatch

In **input mode**:
- π₁ is part of Layer 2's public inputs
- Layer 2's I/O commitment includes π₁: $C_{\text{I/O}} = \text{Commit}(\ldots, \pi_1, \ldots)$
- Verifier receives both π₁ and π₂
- Verifier checks: "Does π₂ prove that Layer 2 received and verified this specific π₁?"
- If prover tries to use π₁' inside Layer 2 but claim π₁ externally, I/O check fails

**Example attack that fails (embedded mode)**:
- Prover creates invalid proof π₁' for wrong computation
- Embeds π₁' in Layer 2 bytecode, generates π₂
- Verifier receives π₂ with bytecode commitment $C_{\text{bytecode}}$
- Verifier extracts π₁' from the bytecode (or verifier already has embedded π₁')
- **Verifier checks π₁' directly** and rejects (invalid Layer 1 proof)
- OR: Verifier only trusts Layer 2 proofs with specific bytecode containing valid π₁

**Example attack that fails (input mode)**:
- Prover creates valid π₁ for correct computation
- Creates π₁' (invalid) for wrong computation
- Tries to run Layer 2 verifier on π₁' but claim π₁ externally
- Layer 2 reads π₁' from input memory during execution
- Layer 2's I/O commitment binds to what was actually read (π₁')
- Verifier checks: "Does π₂'s I/O commitment match claimed input π₁?"
- Mismatch detected, verification fails

**The crucial point**:
- Layer 2 doesn't just "check that a proof verified"
- Layer 2 proves: "I correctly executed the verifier algorithm on **this specific proof data** π₁"
- The proof data π₁ is either embedded (bytecode commitment) or input (I/O commitment)
- Both commitments are Dory commitments verified by the Layer 2 verifier
- Prover cannot substitute different proof data without breaking commitment binding

**Why Layer 2 doesn't need to verify Dory commitments**:
- Layer 2 proves: "I executed the verification algorithm correctly on the given inputs"
- The verification algorithm itself checks the Dory commitments
- If the verification algorithm accepts, the Dory commitments were valid
- Layer 2 just proves the algorithm executed correctly - it doesn't re-verify the cryptography

**The soundness guarantee**:
- If Layer 1 proof is invalid → Layer 1 verifier rejects (computes exponentiations, catches mismatch)
- If Layer 2 proof is invalid → Layer 2 verifier rejects (catches incorrect execution of verifier)
- If both proofs verify → Layer 1 computation was correct (transitive soundness)

This is the same principle as **IVC (Incrementally Verifiable Computation)**: each layer proves correct execution of the previous verifier, with public I/O binding the layers together.

#### 3.8.2 Why Recursion Terminates

Now we can answer the key question: why doesn't Layer 2's verification create the same problem, requiring a Layer 3?

**The answer**: SZ-check verification uses Hyrax openings, which only require:

1. **Sumcheck rounds**: ~log(√N) rounds for N coefficients
   - For 3,048 coefficients per exponentiation: log₂(55) ≈ 6 rounds
   - Very cheap: ~60 field operations per opening

2. **Two MSMs on Grumpkin** (from section 3.9):
   - Each MSM: √N = 55 scalar multiplications
   - Total per opening: 110 Grumpkin scalar muls
   - For 64 exponentiations: 7,040 scalar muls ≈ **700M cycles**

**Why this terminates**:
- **No new G_T exponentiations** are created - only Grumpkin scalar multiplications
- Grumpkin operations are much cheaper than G_T operations (~100K vs ~10M cycles)
- Layer 2 verification: 330M (trace) + 700M (Hyrax openings) ≈ **1B cycles total**
- This is acceptable (vs 1.5B for Layer 1, and much better than computing exponentiations directly)
- Crucially: **Layer 2's verifier doesn't create new expensive exponentiations to recurse on**
- **No Layer 3 needed** - the recursion stops here

### 3.9 Summary: The Complete Architecture

**Layer 2's commitment strategy**:

1. **Main execution trace** (330M cycles of verifier execution):
   - Committed using **Dory over BN254** (same as Layer 1)
   - Includes all register/memory operations
   - Shows hints being loaded and used
   - But does NOT compute the exponentiations

2. **Exponentiation witnesses** (64 $\mathbb{G}_T$ elements):
   - Committed separately using **Hyrax over Grumpkin**
   - Represents the 64 exponentiation results from Layer 1's Dory verification
   - Small polynomial (~512 Fq coefficients)
   - Hyrax's $O(\sqrt{N})$ overhead acceptable for tiny polynomial

3. **SZ-Check intermediate chains** (64 × 254 intermediate values):
   - Also committed using **Hyrax over Grumpkin**
   - Proves each witness satisfies $w_i = g_i^{x_i}$ via square-and-multiply constraints
   - Binds witnesses to trace via sumcheck challenges

**Why this works**:
- Layer 2 avoids computing expensive exponentiations (would cost 1.28B cycles)
- SZ-check proves hints correct via algorithm structure (costs 180K cycles)
- Hyrax verification uses only Grumpkin MSMs (100M cycles on-chain, acceptable)
- No new exponentiations created → recursion terminates

**The key insight**: Different polynomials use different PCS based on their properties and role:
- **Dory**: Efficient for large polynomials, but expensive verification (exponentiations)
- **Hyrax**: Less efficient for large polynomials ($O(\sqrt{N})$), but cheap verification (MSMs)
- Use Dory where it's efficient, Hyrax where it terminates recursion

---

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│    Dory     │         │   Hyrax     │         │  SZ-Check   │
│ (The Trace) │         │ (Witnesses) │         │ (Algorithm) │
└─────────────┘         └─────────────┘         └─────────────┘
      │                       │                       │
      │                       │                       │
      ├───────────────────────┼───────────────────────┤
      │                       │                       │
      │  "I used value v"     │  "I stored value v"   │  "v = g^x (proven)"
      │  (at frame 50000)     │  (at position 5)      │  (with proof)
      │                       │                       │
      └───────────────────────┴───────────────────────┘
                              │
                      All three must agree
                      at random point r
```

**Security**: The prover must satisfy all three simultaneously at a random evaluation point. They cannot:
- Lie to Dory (trace would be inconsistent)
- Lie to Hyrax (witness would mismatch trace)
- Lie to SZ-Check (algorithm constraints would fail)

#### The Data Flow

$$
\begin{array}{l|l|l}
\text{Data} & \text{PCS Used} & \text{Why} \\
\hline
\text{Layer 2 execution trace} & \text{Dory/BN254} & \text{Large (330M), Dory is } O(\log N) \\
\text{(registers, memory, PC, etc.)} & & \text{Shows witnesses being used} \\
\hline
\text{64 exponentiation results} & \text{Hyrax/Grumpkin} & \text{Tiny (~512 elements), Hyrax } O(\sqrt{N}) \text{ OK} \\
w_1, \ldots, w_{64} \in \mathbb{G}_T & & \text{Contains witness values} \\
\hline
\text{Intermediate chains (254 steps × 64)} & \text{Hyrax/Grumpkin} & \text{Proves witnesses correct} \\
\text{(for SZ-Check)} & & \text{via algorithm structure} \\
\end{array}
$$

**Summary**:
- **Dory (BN254)**: "Here's what the verifier did" (the execution trace - 330M cycles)
- **Hyrax (Grumpkin)**: "Here are the expensive values" (the 64 exponentiation results + intermediate chains)
- **SZ-Check**: "Here's why the answers are correct" (sumcheck proof over square-and-multiply algorithm)
- **Binding**: Random evaluation points force consistency across all three components

---

## Part 4: Practical Implementation Details from `examples/recursion`

Beyond the cryptographic theory, the actual implementation in [examples/recursion/](../examples/recursion/) reveals important practical considerations:

### 4.1 Two Deployment Modes

The recursion example supports **two modes** for passing Layer 1 proofs to Layer 2:

**Mode 1: Input Mode (Dynamic)**
- Layer 1 proofs passed as runtime input to Layer 2 guest
- Serialization flow:
  ```
  verifier_preprocessing → proofs[] → devices[] → serialize to bytes → pass as input
  ```
- More flexible but requires larger `max_input_size`
- Example: `max_input_size = 200000` bytes for fibonacci

**Mode 2: Embedded Mode (Static)**
- Layer 1 proofs baked into Layer 2 guest binary at compile time
- Generates `embedded_bytes.rs` containing proof data
- Smaller runtime input (empty or minimal)
- Requires larger `memory_size` and `max_trace_length` (67M cycles for fibonacci vs 5M)
- Trade-off: Compile-time cost vs runtime flexibility

**Key insight**: Embedded mode moves the proof data from runtime input into the binary itself, which affects memory layout but enables simpler deployment (see section 3.8.1 for complete trust model and security implications).

**Batch verification**: Layer 2 can verify multiple Layer 1 proofs in a single recursion proof, amortizing preprocessing costs.

### 4.2 Trace-Only Mode

The implementation includes a **trace-only mode** for development:

```rust
RunConfig::Trace => {
    // Skip proof generation, just trace execution
    let (_, _, io_device) = recursion.trace(&input_bytes, &[], &[]);
    let rv = postcard::from_bytes::<u32>(&io_device.outputs).unwrap_or(0);
}
```

**Why this matters**:
- Trace generation: ~seconds
- Proof generation: ~minutes to hours
- **Use trace mode** to:
  - Debug guest program logic
  - Measure cycle counts
  - Verify inputs/outputs
  - Test memory configurations
  - Iterate rapidly on implementation

**Trace-to-file mode** (`--disk` flag): Streams trace to disk instead of RAM (reduces memory footprint for large traces).

### 4.3 Memory Layout Insights

From the recursion example, key observations about memory:

**Input region**:
- Must be 8-byte aligned (`(input_bytes.len() + 7) & !7`)
- Checked at runtime: `assert!(input_bytes.len() < memory_config.max_input_size)`
- Location: Below DRAM start (0x80000000)

**Program size**:
- Dynamically determined after preprocessing:
  ```rust
  recursion.memory_config.program_size = Some(
      recursion_verifier_preprocessing.shared.memory_layout.program_size
  );
  ```
- Not known until bytecode is compiled and committed

**Output region**:
- Preallocated based on `max_output_size`:
  ```rust
  let mut output_bytes = vec![0; max_output_size as usize];
  ```
- Contains serialized result (e.g., `u32` success/fail indicator)

### 4.4 Preprocessing Reuse

**Critical optimization**: Verifier preprocessing can be **cached and reused**:

```rust
// Generate once
let guest_verifier_preprocessing =
    JoltVerifierPreprocessing::from(&guest_prover_preprocessing);

// Reuse for all verifications
for input in inputs {
    let is_valid = verify(&verifier_preprocessing, proof, device, ...);
}
```

**What's in preprocessing?**
- Bytecode commitment (fixed for a given program)
- Memory layout configuration
- SRS generators (for Dory/Hyrax)
- Circuit-specific data

**Cost savings**: Preprocessing takes ~seconds to minutes. Reusing it across verifications is essential for practical deployment.

### 4.5 Field and PCS Type Aliases

From `jolt-sdk/src/host_utils.rs`:

```rust
pub use jolt_core::ark_bn254::Fr as F;
pub use jolt_core::poly::commitment::dory::DoryCommitmentScheme as PCS;
```

**Key point**: The recursion example uses **F = BN254::Fr** and **PCS = Dory** for both Layer 1 and Layer 2 main traces.

**Why this matters**:
- Confirms Layer 2 uses Dory over BN254 (not "Dory over Grumpkin")
- Field operations in Layer 2 guest are native BN254::Fr
- Hyrax over Grumpkin is an **additional** PCS, not a replacement

### 4.6 Actual Cycle Counts (From Implementation)

From the memory configuration and trace lengths:

| Program | Mode | Max Trace Length | Actual Use Case |
|---------|------|------------------|----------------|
| **fibonacci** | Input | 5M cycles | Quick verification of simple Layer 1 proof |
| **fibonacci** | Embedded | 67M cycles | Full recursion including deserialization overhead |
| **muldiv** | Input | 3M cycles | Minimal arithmetic proof |
| **muldiv** | Embedded | 800K cycles | Small proof, less overhead |

**Insight**: The 330M cycle estimate from theory documents includes:
- Sumcheck verification: ~130M
- SZ-check: ~150M
- Deserialization: ~40M
- Misc (hashing, field ops): ~10M

The actual implementation matches this breakdown closely.

### 4.7 Deployment Workflow

**Complete end-to-end flow**:

```bash
# Step 1: Generate Layer 1 proofs
cargo run --release -- generate --example fibonacci --workdir ./output

# Step 2: Verify Layer 1 proofs and create Layer 2 proof (input mode)
cargo run --release -- verify --example fibonacci --workdir ./output

# Step 3: Create embedded version for deployment
cargo run --release -- verify --example fibonacci --workdir ./output --embed

# Step 4: The generated embedded_bytes.rs is compiled into Layer 2 guest
# Step 5: Layer 2 guest proves verification of Layer 1 proof(s)
# Step 6: On-chain verifier checks Layer 2 proof (~30M gas)
```

**Production pipeline**:
1. Off-chain: Generate Layer 1 proof (user's computation)
2. Off-chain: Generate Layer 2 proof (recursive verification)
3. On-chain: Submit Layer 2 proof
4. On-chain: EVM verifies Layer 2 proof (cheap!)
5. On-chain: If Layer 2 valid → Layer 1 valid → computation correct

### 4.8 Key Implementation Files

From PR #975 and current codebase:

| File | Purpose |
|------|---------|
| `examples/recursion/src/main.rs` | CLI tool for generating/verifying recursive proofs |
| `examples/recursion/guest/src/lib.rs` | Layer 2 guest program (verifier as RISC-V) |
| `examples/recursion/guest/src/embedded_bytes.rs` | Generated file with baked-in proof data |
| `examples/recursion/guest/src/provable_macro.rs` | Generated config for memory/trace limits |
| `jolt-sdk/src/host_utils.rs` | Type aliases (F, PCS) for recursion |

**Development tip**: Start with `fibonacci` example in input mode, then scale to embedded mode for production.

---

## Part 5: Implementation Deep-Dive

### 5.1 The Modified Layer 1 Verifier

**Standard verifier (1.30B cycles):**

```rust
fn verify_jolt(proof: JoltProof, vk: VerifyingKey) -> bool {
    // Stage 1-4: Sumcheck verification (~130M cycles)
    for stage in 1..=4 {
        verify_batched_sumchecks(proof.sumchecks[stage], ...)?;
    }

    // Stage 5: Dory verification (~1.92B cycles - THE BOTTLENECK)
    let openings = extract_opening_claims(proof);

    // Compute 64 𝔾_T exponentiations (30M cycles each!)
    // For log N=16: D₁ folding (16), D₂ folding (16), C update (32)
    let mut D1 = proof.initial_D1;
    let mut D2 = proof.initial_D2;
    let mut C = proof.initial_C;

    for round in 0..16 {
        // These are 𝔾_T exponentiations - extremely expensive!
        D1 = D1 * proof.D1L[round] * pow_gt(proof.D1R[round], challenges.alpha[round]);
        D2 = D2 * proof.D2L[round] * pow_gt(proof.D2R[round], challenges.alpha_inv[round]);
        C = C * chi[round] * pow_gt(D2, challenges.beta[round])
                          * pow_gt(D1, challenges.beta_inv[round])
                          * pow_gt(proof.C_plus[round], challenges.alpha[round])
                          * pow_gt(proof.C_minus[round], challenges.alpha_inv[round]);
    }

    // Check final values match expected
    check_final_commitment(C, D1, D2, openings)
}
```

**Cost breakdown:**
- 64 `pow_gt` calls @ 30M cycles each = 1.92B cycles
- Everything else: ~130M cycles
- **Total: 1.28B cycles**

---

**Modified verifier with hints (~330M cycles):**

```rust
fn verify_jolt_with_hints(
    proof: JoltProof,
    vk: VerifyingKey,
    hints: RecursionHints  // Provided by Layer 2 prover
) -> bool {
    // Stage 1-4: Sumcheck verification (UNCHANGED - ~130M cycles)
    for stage in 1..=4 {
        verify_batched_sumchecks(proof.sumchecks[stage], ...)?;
    }

    // Stage 5: Modified Dory verification (~200M cycles)
    let openings = extract_opening_claims(proof);

    // DON'T compute exponentiations - accept from hints!
    let exp_witnesses = hints.gt_exponentiations; // 64 precomputed values

    // Extract components (NO 𝔾_T exponentiations computed!)
    let D1_results = exp_witnesses[0..16];   // D₁ folding results
    let D2_results = exp_witnesses[16..32];  // D₂ folding results
    let C_components = exp_witnesses[32..64]; // C update components

    // Check algebraic constraints (field operations - cheap!)
    // These will be proven correct by ExpSumcheck in Layer 2
    let mut D1 = proof.initial_D1;
    let mut D2 = proof.initial_D2;
    let mut C = proof.initial_C;

    for round in 0..16 {
        // Use precomputed values, just check structure
        D1 = D1 * proof.D1L[round] * D1_results[round]; // No pow_gt!
        D2 = D2 * proof.D2L[round] * D2_results[round]; // No pow_gt!
        C = C * chi[round] * C_components[round * 2] * C_components[round * 2 + 1];
    }

    // Final check
    check_final_commitment(C, D1, D2, openings)
}
```

**Cost breakdown:**
- No 𝔾_T exponentiations computed: 0 cycles
- Sumchecks: ~130M cycles (unchanged)
- Constraint checking: ~50M cycles (field ops)
- ExpSumcheck verification (proves witnesses correct): ~150M cycles
- **Total: ~330M cycles**

**Speedup: 1.28B → 330M = 6.2× faster!**

### 5.2 The Exponentiation Witnesses

**What exactly are these witnesses?**

For each Dory reduction round $i$ (typically 16 rounds for log N=16):

**D₁ folding witness** (1 per round):
$$w_{D_1}^{(i)} = (D_{1R}^{(i)})^{\alpha_i}$$

This is a 𝔾_T element (12 $\mathbb{F}_q$ coefficients) representing:
- $D_{1R}^{(i)}$: Right half of first vector commitment (from proof)
- $\alpha_i$: Verifier challenge (254-bit scalar)
- Result: The exponentiation $g^{\alpha_i}$ where $g = D_{1R}^{(i)}$

**D₂ folding witness** (1 per round):
$$w_{D_2}^{(i)} = (D_{2R}^{(i)})^{\alpha_i^{-1}}$$

**C update witnesses** (5 per round, but 2 after optimization):
$$w_{C,1}^{(i)} = (D_2^{(i)})^{\beta_i}, \quad w_{C,2}^{(i)} = (D_1^{(i)})^{\beta_i^{-1}}$$
$$w_{C,3}^{(i)} = (C_+^{(i)})^{\alpha_i}, \quad w_{C,4}^{(i)} = (C_-^{(i)})^{\alpha_i^{-1}}$$

($\chi_i$ is precomputed, so doesn't need witness)

**Pippenger optimization**: The C update witnesses can be batched using multi-exponentiation, reducing to 2 equivalent witnesses per round (amortized).

**Total witnesses:**
- D₁ folding: 16 witnesses
- D₂ folding: 16 witnesses
- C update (amortized): 32 equivalent witnesses
- **Total: 64 𝔾_T elements**

**Representation in Hyrax commitment:**
- Each 𝔾_T element → 12 $\mathbb{F}_q$ coefficients
- 64 elements × 12 coefficients = 768 field elements
- Pad to next power of 2: 1024 = $2^{10}$
- **Hyrax commits to 10-variable multilinear polynomial**

### 5.3 Layer 2 Cost Breakdown

**Full Layer 2 verification cost:**

```
Layer 2 Verifier (Grumpkin): ~330M cycles
├─ Modified Layer 1 verifier: ~180M cycles
│  ├─ Sumcheck verification (Stages 1-4): ~130M
│  ├─ Simplified Dory (no exponentiations): ~40M
│  └─ Misc (transcript, field ops): ~10M
│
├─ ExpSumcheck verification: ~150M cycles
│  ├─ 64 exponentiation proofs @ ~2.3M each
│  └─ Batching optimizations reduce overhead
│
└─ Hyrax opening proof: ~30M cycles
   └─ 2 Grumpkin MSMs (~15M each)
```

**Comparison table:**

| Component | Standard Verifier | Modified Verifier | Savings |
|-----------|-------------------|-------------------|---------|
| **Sumchecks (Stages 1-4)** | 130M | 130M | 0 (unchanged) |
| **Dory exponentiations** | 1.92B | 0 | **1.92B** |
| **Simplified Dory** | - | 40M | -40M (new cost) |
| **ExpSumcheck** | - | 150M | -150M (new cost) |
| **Hyrax opening** | - | 30M | -30M (new cost) |
| **Total** | **1.28B** | **350M** | **1.7B savings (83% reduction)** |

### 5.4 Final On-Chain Deployment

**What the on-chain verifier does:**

```solidity
contract JoltVerifier {
    // Verifies Layer 2 proof (Grumpkin)
    function verify(
        bytes calldata proof,        // π₂ from Layer 2
        bytes calldata publicInputs  // Original program inputs
    ) public view returns (bool) {
        // Parse proof components
        (Commitment C, Commitment D, ...) = parseProof(proof);

        // Verify 2 Grumpkin MSMs (multi-scalar multiplications)
        // These are the Hyrax opening proof components
        bool check1 = verifyMSM(C, proof.msmProof1);
        bool check2 = verifyMSM(D, proof.msmProof2);

        // EVM precompiles can accelerate these operations
        // bn256Add (0x06), bn256ScalarMul (0x07) for similar curves
        // Grumpkin equivalents would be added

        return check1 && check2 && finalCheck(...);
    }
}
```

**Cost estimate:**

| Operation | Gas Cost | RISC-V Cycles Equivalent |
|-----------|----------|--------------------------|
| Parse proof | ~50K gas | ~50K cycles |
| Grumpkin MSM 1 | ~12M gas | ~12M cycles |
| Grumpkin MSM 2 | ~12M gas | ~12M cycles |
| Final checks | ~6M gas | ~6M cycles |
| **Total** | **~30M gas** | **~30M cycles** |

**Achieves target: ≤30M cycles for economical on-chain deployment!**

### 5.5 The Complete Trust Chain

**Why this works - step by step:**

```
1. Layer 1 Prover:
   ↓ Generates proof π₁: "Program P with input x produced output y"

2. Layer 2 Prover:
   ↓ Runs modified Layer 1 verifier as RISC-V program
   ↓ Provides exponentiation witnesses as hints
   ↓ Generates ExpSumcheck proofs that witnesses are correct
   ↓ Generates Jolt proof π₂: "I verified π₁ correctly"

3. On-Chain Verifier:
   ↓ Verifies π₂ (the recursion proof)
   ↓ If π₂ is valid, then Layer 2 verifier ran correctly
   ↓ Layer 2 verifier checked: sumchecks + exponentiation constraints
   ↓ ExpSumcheck proved: exponentiation witnesses are correct
   ↓ Therefore: π₁ is a valid Jolt proof
   ↓ Therefore: P(x) = y
```

**Security property:**

For a malicious prover to break this:
- They'd need to forge Layer 2 proof π₂, OR
- Layer 2 verification would need to be buggy, OR
- ExpSumcheck would need to accept incorrect witnesses

Each has negligible probability under standard cryptographic assumptions.

**Trade-off accepted:**

- **Prover time**: Increases significantly (must prove 2 layers)
  - Layer 1: Standard Jolt proving (~20 seconds for typical program)
  - Layer 2: Prove 330M-cycle verifier execution (~40 seconds)
  - **Total: ~60 seconds (vs ~20 seconds for Layer 1 only)**

- **Verifier time**: Decreases dramatically (our goal!)
  - Direct verification: 1.30B cycles (~2-3 minutes)
  - Recursive verification: **30M cycles (~2 seconds)**
  - **Result: 70× speedup enables on-chain deployment**

---

## Summary

### Key Insights

1. **Why recursion works**: The verifier is just another computation—we can prove it like any program

2. **Why curve cycle is essential**: Avoids 1000× blowup in circuit size from non-native field arithmetic

3. **Why Hyrax for Layer 2**: Avoids creating more 𝔾_T exponentiations that would need Layer 3

4. **Why ExpSumcheck is clever**: Proves $g^x = w$ without computing it, using the square-and-multiply structure

5. **Why this achieves the goal**: 70× verification speedup (1.30B → 30M) enables economical on-chain deployment

### Implementation Checklist

- [ ] Implement modified Layer 1 verifier accepting hints
- [ ] Implement exponentiation witness extraction
- [ ] Implement Hyrax PCS over Grumpkin
- [ ] Implement ExpSumcheck protocol
- [ ] Integrate Layer 2 trace generation
- [ ] Implement on-chain Grumpkin verifier (Solidity)
- [ ] Optimize batching and amortization
- [ ] End-to-end testing (Layer 1 → Layer 2 → On-chain)

### References

- **Detailed mathematics**: [SNARK_Composition_and_Recursion.md](SNARK_Composition_and_Recursion.md)
- **The problem**: [Dory_Verification_Cost_Summary.md](Dory_Verification_Cost_Summary.md)
- **Hyrax paper**: [Wahby et al. 2018](https://eprint.iacr.org/2017/1132)
- **Curve cycles**: [Chiesa et al. 2014](https://eprint.iacr.org/2014/595)

---

**Status**: Implementation planned for future Jolt release. See [GitHub PR #975](https://github.com/a16z/jolt/pull/975) for current progress.
