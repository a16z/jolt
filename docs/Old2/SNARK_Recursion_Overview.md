# Jolt SNARK Recursion: Accelerating Verification via Proof Compression

> **⚠️ DEPRECATED**: This document uses outdated "two-layer recursion" and "Layer 1/Layer 2" framing.
>
> **See instead**: [Jolt_Stage6_SNARK_Composition.md](Jolt_Stage6_SNARK_Composition.md) for corrected architecture.
>
> **Key correction**: Stage 6 is not a "Recursive Jolt" layer that proves "I verified Layer 1." It's a Stage 6 extension that commits to exponentiation witness data via Hyrax and proves correctness via ExpSumcheck. There is no "Guest: Jolt verifier compiled to RISC-V bytecode" or "Total trace: ~330M RISC-V cycles."
>
> **What this document gets wrong**:
> - Part 2: "Layer 2: Recursive Jolt" diagram with "Guest: Jolt verifier"
> - Uses "Layer 1" and "Layer 2" terminology throughout
> - Describes a 330M cycle trace that doesn't exist
>
> **What remains useful**: Cost breakdown of G_T exponentiations and SZ-Check protocol details.

## TL;DR

**Problem**: Jolt verification costs ~1.30B cycles due to $\mathbb{G}_T$ exponentiations (see [Dory_Verification_Cost_Summary.md](Dory_Verification_Cost_Summary.md)). This is prohibitively expensive for untrusted verifiers in many practical scenarios.

**Solution**: Two-layer recursion where Layer 2 proves "I correctly verified Layer 1" by accepting exponentiations as hints and proving correctness via SZ-Check.

**Result**: 1.30B → ~30M cycles (70× reduction), enabling **efficient verification for untrusted parties**. An untrusted verifier can run the Layer 2 verifier and, if the proof is valid, accept the result while consuming 70× fewer cycles than direct verification.

**Why this matters**: Enables verification in resource-constrained environments (on-chain EVM, embedded systems, verifier networks) where 1.30B cycles would be infeasible. Verification remains trustless while becoming practical.

**Note**: Costs shown are for typical programs with $\log_2 N = 16$ ($N = 65{,}536$ cycles). See Part 1 for scaling across different trace sizes.

---

## Part 1: The Verification Bottleneck

### Cost for Typical Program ($\log_2 N = 16$)

From [Dory_Verification_Cost_Summary.md](Dory_Verification_Cost_Summary.md), Jolt verification costs:

| Component | Cost (cycles) | % of Total |
|-----------|---------------|------------|
| **Dory $\mathbb{G}_T$ exponentiations** (93 @ 10M each) | **1.09B** | **90%** |
| Sumcheck verification (~40 instances) | 130M | 6% |
| Other operations (pairings, field ops) | 70M | 4% |
| **Total** | **~1.30B** | **100%** |

**The bottleneck**: $\mathbb{G}_T$ exponentiations (in execution order)
- **29** from stage 5 RLC (random linear combination - **fixed per program**, independent of $N$)
  - Creates combined commitment: $C_{\text{combined}} = \prod_{i=1}^{29} C_i^{\gamma_i}$
- **80** from main Dory opening: $5 \times \log_2 N$ (for $N = 2^{16}$: $5 \times 16 = 80$)
  - Verifies the single combined commitment from RLC
- Each exponentiation: ~10M RISC-V cycles (with cyclotomic squaring optimization)
  - **Algorithm**: Binary exponentiation with Granger-Scott cyclotomic squaring
  - 254 cyclotomic squarings: 254 × 18 $\mathbb{F}_q$ muls × 900 cycles = 4.1M cycles
  - 127 general multiplications: 127 × 54 $\mathbb{F}_q$ muls × 900 cycles = 6.2M cycles
  - Total: ~10.3M cycles per exponentiation
  - See [Dory_Verification_Cost_Summary.md](Dory_Verification_Cost_Summary.md) Section 1.2 for detailed breakdown
- **Why BN254 creates recursion challenges**:
  - BN254 widely adopted (EVM precompiles, tooling ecosystem) → standard choice for verification
  - But BN254 SNARK circuits operate in $\mathbb{F}_r$ (scalar field), while $\mathbb{G}_T$ exponentiations compute in $\mathbb{F}_q$ (base field)
  - **Non-native field arithmetic**: Each $\mathbb{F}_q$ operation requires ~1000 $\mathbb{F}_r$ constraints
  - Proving one exponentiation naively: $11{,}430$ $\mathbb{F}_q$ ops $\times$ 1000 = ~11M constraints (prohibitive!)
  - Creates a **double bottleneck**: expensive to compute (10M cycles) AND expensive to prove (11M constraints)
  - **Additional constraint**: No hardware/precompile acceleration exists for $\mathbb{G}_T$ exponentiations (unlike pairings or EC scalar multiplication) → cannot be optimized away in resource-constrained environments

**The verification cost problem**: $109 \times 10\text{M} = 1.09\text{B cycles}$ for exponentiations alone (plus ~220M for sumchecks and other operations = ~1.30B total)

**Why this matters for untrusted verifiers**:
1. **Computation cost**: 1.30B cycles is prohibitively expensive for:
   - **On-chain smart contracts**: Would cost >2B gas (exceeds EVM block limit by 60×)
   - **Verifier networks**: High cost per verification → limits throughput and economics
   - **Embedded systems**: May exceed available compute budget entirely
   - **Off-chain services**: 2-5 seconds per verification too slow for interactive applications

2. **Integration cost**: Non-native field arithmetic makes wrapping in other SNARKs infeasible
   - To prove verification in another circuit: $93 \times 20\text{M constraints} = 1.09\text{B constraints}$
   - Creates memory, time, and SRS size bottlenecks

3. **Deployment barriers**: Must maintain BN254 compatibility (widely deployed, has precompiles) while achieving practical verification costs

### How Costs Scale with Trace Size

The number of exponentiations (and thus total cost) **scales logarithmically** with trace size:

| Trace Size | $\log_2 N$ | Stage 5 RLC | Main Dory (Opening) | **Total Exps** | **Total Cost** |
|------------|--------|-------------|---------------------|----------------|----------------|
| 2,048 | 11 | 29 | $4 \times 11 = 44$ | **73** | **1.46B cycles** |
| 8,192 | 13 | 29 | $4 \times 13 = 52$ | **81** | **1.62B cycles** |
| **65,536** | **16** | **29** | **$5 \times 16 = 80$** | **93** | **1.09B cycles** |
| 262,144 | 18 | 29 | $4 \times 18 = 72$ | **101** | **2.02B cycles** |
| 1,048,576 | 20 | 29 | $5 \times 20 = 100$ | **109** | **1.30B cycles** |

**Key observations** (columns ordered by execution):
- Stage 5 RLC exponentiations are **fixed at 29** (independent of trace size) - happens **first**
- Main Dory opening exponentiations scale as $4 \times \log_2 N$ (linear in log N) - happens **second**
- For small traces ($\log_2 N \leq 11$), RLC dominates (29 out of 73 = 40%)
- For large traces ($\log_2 N \geq 20$), main Dory dominates (80 out of 109 = 73%)
- **Total cost scales logarithmically**: $10\times$ larger trace → only +16 exponentiations (~300M cycles)

**Per-exponentiation cost** is constant at ~10M cycles regardless of trace size (fixed 254-bit exponent).

**Throughout this document**: We use $\log_2 N = 16$ (65K cycles) as the representative example, as it covers typical Jolt programs.

---

## Part 2: The Recursion Strategy

### High-Level Architecture

```
┌────────────────────────────────────────────────────────┐
│ LAYER 1: Standard Jolt (BN254)                        │
├────────────────────────────────────────────────────────┤
│ Input: User program P, input x                        │
│ Output: Proof $\pi_1$ (~10 KB)                        │
│ Verification Cost: 1.30B cycles (if computed)         │
│   └─ 109 $\mathbb{G}_T$ exponentiations @ 10M each     │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│ LAYER 2: Recursive Jolt (BN254 + Grumpkin)           │
├────────────────────────────────────────────────────────┤
│ Guest: Jolt verifier compiled to RISC-V bytecode     │
│                                                        │
│ Modified verification algorithm:                      │
│   1. Verify sumchecks: ~90M cycles (unchanged)       │
│   2. Accept 109 $\mathbb{G}_T$ exponentiations as hints │
│   3. Use hints in Dory verification: ~90M cycles     │
│   4. Prove hints correct via SZ-Check: ~150M cycles  │
│                                                        │
│ Total trace: ~330M RISC-V cycles                     │
│                                                        │
│ Mixed PCS:                                             │
│   • Main trace → Dory over BN254                     │
│   • Exponentiation witnesses → Hyrax over Grumpkin   │
│                                                        │
│ Output: Proof $\pi_2$ (~15 KB)                        │
│ Final Verification Cost: ~30M cycles ✓                │
│   (70× compression vs. Layer 1's 1.30B)              │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│ DEPLOYMENT OPTIONS:                                    │
├────────────────────────────────────────────────────────┤
│ • Off-chain: Fast verification (30M cycles)          │
│ • On-chain: Feasible EVM deployment (~30M gas)       │
│ • Embedded: Resource-constrained environments         │
│ • Verifier networks: Efficient distributed validation │
└────────────────────────────────────────────────────────┘
```

### Why This Works: Transitive Soundness

**Layer 2 proves**: "I executed the verifier algorithm correctly on input $\pi_1$"

If Layer 2's trace is proven correct:
1. The trace shows the verifier algorithm executed faithfully
2. The verifier algorithm checks all cryptographic constraints of $\pi_1$
3. The trace shows the verifier output "accept"
4. Therefore: $\pi_1$ must be valid → user computation was correct

**Security binding**: Layer 2 commits to either:
- **Bytecode** (embedded mode): $\pi_1$ is part of the committed program
- **Public I/O** (input mode): $\pi_1$ is part of committed inputs

Prover cannot substitute a different $\pi_1$ without breaking the commitment.

---

## Part 3: The Mixed PCS Strategy

### Why We Need Two Different PCS

**Problem**: If Layer 2 used only Dory, we'd recreate the bottleneck:
- Verifying Layer 2's Dory commitments would require computing G_T exponentiations again
- This would necessitate Layer 3, then Layer 4, ... (infinite recursion)

**Solution**: Use different PCS for different data based on properties

| Data | Size | PCS | Verification Cost | Why |
|------|------|-----|-------------------|-----|
| **Layer 2 trace** | 330M cycles | Dory over BN254 | Exponentiations offloaded | Efficient for large polynomials |
| **Exponentiation witnesses** | ~3K coefficients | Hyrax over Grumpkin | 2 MSMs × 55 ops | Small polynomial, O(√N) acceptable |
| **SZ-Check chains** | 93 × 254 steps | Hyrax over Grumpkin | Batched with witnesses | Proves algorithm structure |

### Why Dory Over BN254 for Main Trace?

Layer 2's main execution trace (330M cycles) uses the **same PCS as Layer 1**:
- Reuses all Jolt infrastructure (Spartan R1CS + Twist + Shout)
- Efficient O(log N) verification for large polynomials
- The exponentiations from Layer 2's Dory verification are **also offloaded** via hints
- Enables recursive composition: Layer 2 can verify Layer 1 using the same machinery

### Why Hyrax Over Grumpkin for Witnesses?

**Context**: Layer 2 accepts 93 G_T exponentiation results as "hints" (each hint = 12 F_q coefficients). These hints must be committed to a polynomial and proven correct via SZ-Check. The question: which PCS to use for these witness polynomials?

**The decision chain** (in logical order):

1. **Why not Dory for everything?** → Would recreate the bottleneck
   - **Problem**: If we use Dory for witness commitments, verifying Layer 2 would require computing G_T exponentiations again
   - This creates infinite regress: Layer 2 verification → Layer 3, Layer 3 verification → Layer 4, etc.
   - **Need**: A PCS whose verification is "simple enough" to terminate recursion

2. **Why Hyrax for witnesses?** → Verification is simple (MSMs) and witness polynomial is small
   - **Hyrax verification**: 2 multi-scalar multiplications (MSMs) of size √N on an elliptic curve
   - MSMs are "elementary operations" (no pairings, no exponentiations in G_T, no new hard problems)
   - **What we're committing to**: Exponentiation witness polynomial (N=1024, padded from 109 witnesses × 12 F_q coefficients each)
   - **Decision**: Use Hyrax for small witness polynomial (terminates recursion with simple MSM verification)

3. **Why Grumpkin specifically?** → Field matching for the witness data
   - **What we're committing to**: Witness polynomial with coefficients in $\mathbb{F}_q$ (BN254's base field)
     - Each G_T element = 12 F_q coefficients
     - 109 witnesses × 12 coefficients ≈ 1K F_q values (padded to 2^10)
   - **Hyrax commitment formula**: $C = \sum_i w_i \cdot [G_i]$ where $w_i$ are polynomial coefficients and $[G_i]$ are curve generators
   - **Curve choice matters**: For efficient scalar multiplication $w_i \cdot [G_i]$, the scalar $w_i$ must be in the curve's **scalar field**
   - **Option A - BN254 curve**: Scalar field = $\mathbb{F}_r$ ≠ $\mathbb{F}_q$ → would need to represent F_q elements in F_r (non-native, ~1000× slower)
   - **Option B - Grumpkin curve**: Scalar field = $\mathbb{F}_q$ (exactly matches our witness data!) → native arithmetic
   - **Decision**: Use Grumpkin for Hyrax commitments (field matching)

**Hyrax structure** (matrix-based commitment):
- **Commitment**: Arrange N coefficients into √N × √N matrix, commit each row via Pedersen
- **Opening**: Requires 2 MSMs of size √N each (row and column proofs)
- **Cost**: For N=1024, each MSM has 32 operations → ~64 Grumpkin scalar multiplications total
- **Per scalar multiplication**: ~100K RISC-V cycles → ~6.4M cycles total per witness opening
- **For 109 witnesses**: ~8M cycles (vs 1.09B if we computed G_T exponentiations directly)

**The curve cycle enables this**:

$$
\begin{array}{rcl}
\text{BN254 (Layer 1):} & & \text{Grumpkin (Layer 2 witnesses):} \\
\text{Base field: } \mathbb{F}_q & \longrightarrow & \text{Scalar field: } \mathbb{F}_q \\
\text{Scalar field: } \mathbb{F}_r & \longleftarrow & \text{Base field: } \mathbb{F}_r
\end{array}
$$

Layer 1's $\mathbb{G}_T$ elements (in $\mathbb{F}_q$) become native scalars in Grumpkin's commitments

**Why recursion terminates**:
- Hyrax verification creates **no new G_T exponentiations**
- Only Grumpkin scalar multiplications (~100K cycles each)
- Much cheaper than G_T exponentiations (~10M cycles each, 200× difference)
- No Layer 3 needed → recursion stops here

---

## Part 4: The SZ-Check Protocol

### The Core Problem

How do we prove w = g^x without computing the exponentiation (which costs 20M cycles)?

### The Solution: Verify Algorithm Structure

**Insight**: Instead of computing g^x, prove the square-and-multiply algorithm was executed correctly.

**Square-and-multiply algorithm**: For $g^x$ where $x = \sum_j b_j 2^j$ (binary):

$$
\begin{align}
r_0 &= 1 \\
\text{For } j &= 0 \text{ to } 253: \\
&\quad \text{If } b_j = 0: \quad r_{j+1} = r_j^2 \quad \text{(square only)} \\
&\quad \text{If } b_j = 1: \quad r_{j+1} = r_j^2 \cdot g \quad \text{(square and multiply)} \\
\\
\text{Unified constraint:} \quad &r_{j+1} = r_j^2 \cdot (1 + b_j(g - 1))
\end{align}
$$

After 254 steps: $r_{254} = g^x$ (guaranteed by algorithm structure)

### The Protocol

**Prover's witness**:
1. Compute all 254 intermediate values: $r_0, r_1, \ldots, r_{254}$
2. Encode as multilinear polynomial $\tilde{r}(j)$ over $\log_2(256) = 8$ variables
3. Commit to $\tilde{r}$ using Hyrax over Grumpkin
4. Repeat for all 109 exponentiations

**Verifier's check via sumcheck**:

Define error polynomial for step $j$:
$$e(j) = r_{j+1} - r_j^2 \cdot (1 + b_j(g - 1))$$

Prove via sumcheck: $\sum_{j=0}^{253} e(j) = 0$

If all error terms are zero → each step follows the algorithm → $r_{254}$ must equal $g^x$

**Batching**: Use random linear combination to batch all 109 exponentiations into a single sumcheck:
$$\sum_{i=1}^{93} \gamma^i \cdot \sum_{j=0}^{253} e_{i,j} = 0$$

Where $\gamma$ is a random challenge from the verifier.

### Cost Analysis

**Per exponentiation**:
- Prover: Compute 254 intermediate values (same cost as computing exponentiation)
- Verifier: Sumcheck with 8 rounds ($\log_2 256$)
  - Each round: degree-3 polynomial evaluation (~10 field ops)
  - Total: ~80 field operations ≈ ~80K cycles per exponentiation

**For 109 exponentiations**:
- Naive: 93 separate sumchecks = 93 × 80K ≈ 7.5M cycles
- With batching: Single sumcheck over batched claim ≈ **~150M cycles**
  - Additional cost from evaluating 93 polynomials at random point
  - Hyrax openings for 93 commitments: 93 × 2 MSMs × 55 ops ≈ 10K scalar muls ≈ 100M cycles
  - Sumcheck protocol itself: ~50M cycles

**Comparison**:
- Computing directly: 109 × 10M = 1.09B cycles
- SZ-Check: 150M cycles
- **Savings: 92% reduction**

### Security

**Schwartz-Zippel lemma**: If prover provides incorrect $w \neq g^x$, they must construct a polynomial $\tilde{e}(j)$ that:
1. Sums to zero over the Boolean hypercube
2. Evaluates to match the verifier's checks
3. But encodes an incorrect computation chain

The probability of constructing such a polynomial is negligible ($< 2^{-240}$).

---

## Part 5: Cost Breakdown

### Layer 1 (Standard Jolt)

| Component | Cost |
|-----------|------|
| Prover time | ~2 seconds |
| Prover cycles | ~1M RISC-V cycles |
| Verifier (direct) | **1.30B cycles** |

**Bottleneck**: 85% of verification time spent on 109 G_T exponentiations.

### Layer 2 (Recursive Jolt)

| Component | Cost | Notes |
|-----------|------|-------|
| Prover time | ~20 seconds | Proving 330M cycle trace |
| Prover cycles | ~500M RISC-V cycles | Includes SZ-Check witness computation |
| **Verifier** | **30M cycles** | **70× improvement!** |

**Layer 2 verification breakdown**:

| Operation | Cost | Notes |
|-----------|------|-------|
| Sumcheck verification | ~5M cycles | Layer 2's ~40 sumcheck instances |
| Dory verification (offloaded) | ~15M cycles | Layer 2's Dory exponentiations also offloaded recursively |
| Hyrax openings | ~8M cycles | 109 witnesses, 2 MSMs each |
| Misc (field ops, transcript) | ~2M cycles | Hashing, polynomial evaluations |
| **Total** | **~30M cycles** | 70× faster than Layer 1's 1.30B cycles |

---

## Part 6: Deployment Modes

### Embedded Mode (Production)

**How it works**:
- Layer 1 proof $\pi_1$ serialized and embedded into Layer 2's bytecode at compile time
- Generates `embedded_bytes.rs` containing proof data
- Layer 2 guest reads proof from memory (part of program image)

**Advantages**:
- **Strong cryptographic binding**: $\pi_1$ is part of bytecode commitment (Dory commitment)
- Prover cannot substitute different $\pi_1$ without changing bytecode
- Simpler deployment (no runtime input handling)

**Disadvantages**:
- Must recompute Layer 2 preprocessing (~minutes) if $\pi_1$ changes
- Larger memory footprint (proof embedded in binary)
- Inflexible (tied to specific $\pi_1$)

**Use case**: Production deployment where $\pi_1$ is fixed (e.g., specific program verification)

### Input Mode (Development)

**How it works**:
- Layer 1 proof $\pi_1$ passed as runtime input to Layer 2 guest
- Deserialized from input memory during execution
- Layer 2 preprocessing is independent of $\pi_1$

**Advantages**:
- **Reusable preprocessing**: Same Layer 2 verifier can verify any $\pi_1$
- Flexible (can verify different proofs without recompiling)
- Smaller binary size

**Disadvantages**:
- **Weaker binding**: $\pi_1$ committed via I/O commitment, not bytecode
- Must explicitly bind claimed statement to $\pi_2$'s public outputs
- Slightly larger trace (deserialization overhead)

**Use case**: Development, testing, batch verification scenarios

### Security Best Practices

**Input mode risk**: Prover could generate valid $\pi_2$ for $\pi_1$ (correct proof) but claim it verified $\pi_1'$ (different computation).

**Mitigation**: Bind claimed statement to $\pi_2$'s public I/O:
```rust
#[jolt::provable]
fn verify_with_statement(
    proof_bytes: &[u8],
    claimed_input: &[u8],
    claimed_output: &[u8]
) -> bool {
    let proof = deserialize(proof_bytes);

    // Bind statement to this proof!
    assert_eq!(proof.public_input, claimed_input);
    assert_eq!(proof.public_output, claimed_output);

    JoltRV64IMAC::verify(proof, ...).is_ok()
}
```

Now $\pi_2$'s public I/O includes the claimed statement, preventing statement substitution.

---

## Part 7: Key Formulas

### Exponentiation Count (Depends on Trace Size)

**In execution order**:

**Step 1: Stage 5 RLC** (random linear combination - **fixed per program**):
$$C_{\text{combined}} = \prod_{i=1}^{29} C_i^{\gamma_i}$$

Where 29 is the number of commitment terms combined in Stage 5 RLC (independent of N):
- 8 fixed polynomials (LeftInstructionInput, WriteLookupOutputToRD, etc.)
- 16 instruction lookup polynomials (InstructionRa(0) through InstructionRa(15))
- 3 RAM polynomials (RamRa(0) through RamRa(2) for moderate programs)
- 2 bytecode polynomials (BytecodeRa(0) through BytecodeRa(1))

> **Note**: Jolt commits to ~50 total witness polynomials during proof generation, but only **29 distinct commitment terms** are combined in Stage 5's random linear combination. The difference is due to batching (same evaluation points) and virtual polynomials (not committed).

Cost: **29 exponentiations** (fixed)

**Step 2: Main Dory opening** (scales with trace size):
$$\text{Main Exponentiations} = 4 \times \log_2 N$$

Examples:
- Small program (N = 2^11 = 2K cycles): 4 × 11 = **44 exponentiations**
- Typical program (N = 2^16 = 65K cycles): 4 × 16 = **64 exponentiations**
- Large program (N = 2^20 = 1M cycles): 4 × 20 = **80 exponentiations**

This verifies the single combined commitment from Step 1.

**Total exponentiation count**:
$$\text{Total} = 29 + (4 \times \log_2 N)$$

For N = 2^16: **29 + 64 = 109 exponentiations**

> **Why Addition, Not Multiplication?**
>
> The formula is $(4 \times \log_2 N) + 29$, not $(4 \times \log_2 N) \times 29$, because of the **order of operations**:
>
> 1. **First**: Homomorphic combination (29 exponentiations) → Creates $C_{\text{combined}} = \prod_{i=1}^{29} C_i^{\gamma_i}$
> 2. **Then**: Dory opening ($4 \times \log_2 N$ exponentiations) → Verifies the single combined commitment
>
> The Dory protocol's intermediate values are for the ONE combined commitment, not for each of the 29 individual commitments. Without this batching optimization, we'd need $29 \times (4 \times \log_2 N) = 1{,}856$ exponentiations (20× worse!).

### Verification Cost Formula

**Total cost** (as function of trace size):
$$\text{Cost}(N) = [(4 \times \log_2 N) + 29] \times 20\text{M cycles} + 200\text{M cycles}$$

Where:
- $(4 \times \log_2 N) + 29$ = number of G_T exponentiations
- 20M cycles = cost per exponentiation (constant)
- 200M cycles = sumchecks + other operations (roughly constant)

**For typical N = 2^16**:
$$\text{Cost} = [64 + 29] \times 20\text{M} + 200\text{M} = 1.09\text{B} + 0.22\text{B} = 2.08\text{B cycles}$$

### Verification Speedup (After Recursion)

$$\text{Speedup} = \frac{\text{Cost}(N)}{30\text{M cycles}}$$

For N = 2^16: $\frac{2.08\text{B}}{30\text{M}} = 70\times$

**Note**: Speedup varies slightly with trace size due to logarithmic scaling.

### Per-Exponentiation Cost (Constant)

**Computing $g^x$ in $\mathbb{G}_T$** (independent of trace size $N$):

$$
\begin{align}
\text{Square-and-multiply:} \quad & \text{254-bit exponent (fixed by BN254 scalar field size)} \\
& \rightarrow 254 \text{ squarings} + 127 \text{ multiplications (average, 50\% Hamming weight)} \\
& \rightarrow \sim 381 \text{ } \mathbb{F}_q^{12} \text{ multiplications (AVERAGE CASE)} \\
\\
& \text{Note: Worst case (all bits = 1): } 254 + 254 = 508 \text{ } \mathbb{F}_q^{12} \text{ muls} \\
& \quad \quad \text{Best case (only MSB = 1): } 254 + 1 = 255 \text{ } \mathbb{F}_q^{12} \text{ muls} \\
\\
\mathbb{F}_q^{12} \text{ multiplication (Karatsuba):} \quad & \rightarrow \sim 54 \text{ } \mathbb{F}_q \text{ base field operations} \\
\\
\mathbb{F}_q \text{ operation (256-bit modular arithmetic):} \quad & \rightarrow \sim 1000 \text{ RISC-V cycles} \\
\\
\text{Total:} \quad & 381 \times 54 \times 1000 \approx 20\text{M cycles (average case)}
\end{align}
$$

**Why constant?** Exponent size is determined by BN254 scalar field (254 bits), not by trace size.

---

## Part 8: Comparison with Other Approaches

| Approach | Verification | Setup | Proof Size | Prover Time |
|----------|-------------|-------|------------|-------------|
| **Jolt (direct)** | 1.30B cycles | Transparent | 10 KB | ~2s |
| **Jolt + Recursion** | **30M cycles** | **Transparent** | **15 KB** | **~22s** |
| Groth16 wrapper | <1M cycles | Trusted (circuit-specific) | 192 bytes | ~60s |
| PLONK wrapper | ~2M cycles | Trusted (universal) | ~1 KB | ~40s |

**Trade-offs**:

**Jolt recursion advantages**:
- Fully transparent (no trusted setup ceremony)
- Relatively simple (reuses Jolt infrastructure)
- 70× verification speedup (1.30B → 30M cycles)
- Enables deployment in resource-constrained environments

**Jolt recursion disadvantages**:
- Larger proof than Groth16 (~15 KB vs 192 bytes)
- Slower verification than Groth16 (30M vs <1M cycles)
- Slower proving than direct Jolt (~22s vs ~2s)

**When to use what**:
- **Development**: Direct Jolt verification (fast iteration)
- **Testing**: Jolt recursion with input mode (flexible)
- **Production (general)**: Jolt recursion with embedded mode (secure + transparent)
- **Production (minimal verification cost)**: Groth16 if trusted setup acceptable

---

## Part 9: Why This Matters

### Enabling Efficient Verification for Untrusted Parties

**The achievement**: 70× reduction in verification cost (1.30B → 30M cycles)

**Core value proposition**: Any untrusted party can now efficiently verify Jolt proofs by running the Layer 2 verifier. If the Layer 2 proof is valid, they accept the result—trusting the proof's correctness, not the original prover. Verification remains trustless while becoming 70× more efficient.

**Practical implications for untrusted verifiers**:

1. **Off-chain verification**: Fast enough for interactive applications
   - 30M cycles ≈ milliseconds on modern hardware (vs. 2-5 seconds for direct verification)
   - Enables real-time proof validation in user-facing applications

2. **Verifier networks**: Economical for distributed validation at scale
   - Lower cost per verification → higher throughput
   - More validators can afford to participate
   - Economics of large-scale verification become viable

3. **On-chain deployment**: Within reach for many blockchain VMs
   - EVM: ~30M gas is feasible (though may need Groth16 wrap for optimal cost)
   - Other VMs: Different constraints, but 30M cycles more universally achievable
   - Enables trustless on-chain verification where 1.30B cycles was impossible

4. **Embedded systems**: Feasible for resource-constrained devices
   - Mobile, IoT, edge computing scenarios
   - Power and compute budget constraints satisfied
   - Verification on devices that couldn't handle 1.30B cycles

**Comparison with alternatives**:
- **Jolt direct**: 1.30B cycles → Too expensive for untrusted verifiers in most scenarios
- **Jolt recursion**: 30M cycles → Practical for untrusted parties (transparent, 70× faster)
- **Groth16 wrapper**: <1M cycles → Fastest for untrusted verifiers (requires trusted setup)

### Path to Production

**Development workflow**:
1. Write Jolt guest program
2. Test with direct verification off-chain (fast prover: ~2s)
3. Benchmark and optimize guest code
4. Generate Layer 1 proofs

**Deployment workflow**:
1. Switch to recursion with input mode (flexible testing)
2. Batch multiple Layer 1 proofs if needed
3. Test Layer 2 verification performance with untrusted verifiers
4. For production: Use embedded mode (stronger security)
5. Deploy Layer 2 verifier to target environment (EVM, verifier network, off-chain service, etc.)
6. Untrusted parties verify $\pi_2$ proofs at 30M cycle cost (70× faster than Layer 1)

---

## Summary

**The Innovation**: Accept expensive operations as hints, prove correctness via algorithm structure verification.

**Core motivation**: Enable **efficient verification for untrusted parties**. Any untrusted verifier can run the Layer 2 verifier and, if the proof is valid, accept the result while consuming 70× fewer cycles than direct verification. Verification remains trustless while becoming practical.

**Three Key Insights**:
1. **Hints instead of computation**: Don't compute $\mathbb{G}_T$ exponentiations (1.09B cycles saved)
2. **Mixed PCS**: Use Hyrax for small witnesses to terminate recursion (no Layer 3 needed)
3. **SZ-Check**: Verify square-and-multiply algorithm structure (150M cycles vs 1.09B direct)

**Result**: Jolt verification reduced from 1.30B → 30M cycles (70× improvement), making verification practical for untrusted parties in diverse deployment scenarios (on-chain, embedded, verifier networks) while maintaining full transparency.

**Status**: Fully implemented in `examples/recursion/` (PR #975)

---

## Open Questions

### Why Not Use Hyrax + SZ-Check Directly in Layer 1?

**The question**: The recursion strategy (Hyrax for witness commitments + SZ-Check for proving exponentiations) could theoretically be applied **directly to Layer 1** to avoid computing G_T exponentiations there. Why introduce Layer 2 at all?

**Hypothetical Layer 1 (modified)**:
```
Layer 1 (with hints):
  - Main trace: Dory over BN254
  - Dory exponentiations: Accept as hints (not computed)
  - Exponentiation witnesses: Commit via Hyrax over Grumpkin
  - Prove witnesses correct: SZ-Check
  → Direct verification cost: sumchecks + Hyrax openings + SZ-Check ≈ 30M cycles?
```

**Why this might not work**:
- **Verifier complexity**: Layer 1 verifier would need to understand both BN254 (main trace) and Grumpkin (witnesses) commitments natively
- **Integration challenges**: Mixing Dory verification logic (expects exponentiations) with SZ-Check witness verification in the same protocol
- **Implementation simplicity**: Layer 2 approach treats the entire verifier as a "black box" guest program, avoiding protocol-level modifications

**Why the current approach works**:
- Layer 2 guest runs the **standard** Layer 1 verifier (compiled to RISC-V)
- Jolt proves RISC-V execution correctness (no protocol modifications needed)
- Clean separation: Layer 1 is unchanged, Layer 2 adds recursion as an "external wrapper"

**Question for developers**: Is there a fundamental reason why Hyrax+SZ-Check cannot be integrated directly into Layer 1's Dory verification, or is it primarily an implementation/modularity choice?

### Embedded vs Input Mode: Trust Assumptions and Use Cases

**The question**: Jolt's recursion implementation supports two modes for handling the Layer 1 proof ($\pi_1$) in Layer 2. What are the intended use cases, trust assumptions, and best practices for each?

#### Mode 1: Embedded Mode

**How it works**:
- Layer 1 proof $\pi_1$ is **baked directly into Layer 2's bytecode** during preprocessing
- Cryptographically strong binding: Layer 2's bytecode commitment includes $\pi_1$
- Verifier preprocessing commits to the bytecode containing $\pi_1$

**Trust model**:
- **No additional trust assumptions**: The statement for $\pi_1$ is fixed and committed to during preprocessing
- Verifier knows exactly which $\pi_1$ is being verified (it's part of the bytecode commitment)
- Cannot substitute a different $\pi_1$ without recompilation and new preprocessing

**Trade-offs**:
- ✅ Strongest security guarantees (cryptographic binding via bytecode commitment)
- ✅ No ambiguity about which statement is being proven
- ❌ Inflexible: Each different $\pi_1$ requires full recompilation and preprocessing
- ❌ Preprocessing must be recomputed for every new Layer 1 proof

**Use case**: When $\pi_1$ is for a **fixed statement** that rarely changes (e.g., proving a specific computation or verifying a standard program)

#### Mode 2: Input Mode

**How it works**:
- Layer 1 proof $\pi_1$ provided as **runtime input** to Layer 2 guest program
- Layer 2 proves: "I correctly verified *the input I received*"
- Same Layer 2 bytecode/preprocessing can verify different $\pi_1$ proofs

**Critical trust nuance**:
- ⚠️ **Layer 2 does NOT prove the input is for the correct statement**
- $\pi_2$ only proves "verification computation was correct for the given input"
- Without additional measures, a malicious prover could submit $\pi_1$ for a different (easier) statement

**Two approaches to secure input mode**:

1. **Approach A: Include statement in public I/O** (recommended):
   - Layer 2 guest extracts and outputs the statement from $\pi_1$
   - Statement becomes part of Layer 2's public outputs
   - Verifier checks: "Does the statement in $\pi_2$.public_outputs match what I expect?"
   - ✅ Cryptographically sound: Statement binding enforced by $\pi_2$'s commitment to public I/O

2. **Approach B: Trust the proof source**:
   - Accept $\pi_1$ from a trusted source (e.g., your own Layer 1 prover)
   - Assume the source provides $\pi_1$ for the correct statement
   - ⚠️ Trust-based, not cryptographic: Verifier must trust the Layer 1 prover

**Trade-offs**:
- ✅ Flexible: Same preprocessing verifies any $\pi_1$ (assuming compatible parameters)
- ✅ Efficient: No recompilation/preprocessing for different proofs
- ⚠️ Requires care: Must ensure statement binding (via Approach A) or accept trust assumption (Approach B)
- ❌ More complex: Verifier must validate statement binding or trust model

**Use case**: When verifying **many different Layer 1 proofs** (e.g., aggregating multiple proofs, proof-of-work verification, or general-purpose verification service)

#### Open Questions for Developers

1. **Best practices**: What is the recommended approach for production systems? Should input mode always use Approach A (statement in public I/O)?

2. **Embedded mode rationale**: Given the inflexibility, when is embedded mode genuinely preferred over input mode with statement binding?

3. **Hybrid approaches**: Can preprocessing be partially cached (e.g., cache everything except bytecode commitment) to reduce embedded mode's recompilation cost?

4. **Statement binding verification**: What is the exact mechanism for extracting and verifying the statement in input mode? (Implementation details in `examples/recursion/`?)

### Groth16 Final Step: Constraint Count Limits

**The question**: The ultimate goal for on-chain deployment is often to wrap Layer 2's proof in a Groth16 proof (constant ~200 byte size, ~280K gas verification on EVM). What is the practical constraint count limit for Groth16 proving, and does Layer 2's 30M cycle verification fit within it?

**Current Layer 2 cost estimate**:
- **Verification cycles**: ~30M RISC-V cycles
- **R1CS constraints** (estimated): $30\text{M cycles} \times 30 \text{ constraints/cycle} \approx 900\text{M constraints}$
- **Assumption**: Each Jolt cycle requires ~30 R1CS constraints (similar to Layer 1)

**Groth16 proving challenges**:
1. **Memory bottleneck**: Groth16 proving requires large FFTs over constraint domain
   - FFT size: Smallest power of 2 ≥ constraint count
   - For 900M constraints → need $2^{30}$ FFT (1 billion elements)
   - Memory requirement: Potentially 100s of GB for witness polynomials + intermediate values

2. **Proving time**: Groth16 proving time scales roughly as $O(N \log N)$ for $N$ constraints
   - Multi-exponentiation of size $N$ (dominant cost)
   - Polynomial commitments and FFTs

3. **SRS size**: Structured reference string must support constraint count
   - Powers of tau ceremony must generate sufficient elements
   - Largest known ceremonies: ?

**Open questions for developers**:
1. What is the **largest Groth16 circuit proven in practice**? (Constraint count, hardware, time)
2. Does **900M constraints** fit within practical Groth16 limits, or is further optimization needed?
3. Are there **alternative final-step SNARKs** better suited for large constraint counts? (e.g., Plonky2, Halo2, Nova/Sangria-based folding schemes)
4. Can **constraint count be reduced** via:
   - Custom R1CS gadgets for hot paths?
   - Sparse constraint exploitation?
   - Additional layers of recursion?
5. What is the **actual constraint count** for Layer 2 verification? (The 30 constraints/cycle is an estimate - actual count may differ)

**Related consideration**: If Groth16 is infeasible at this scale, alternatives include:
- **Plonky2**: Recursive SNARK optimized for large circuits (but not EVM-friendly)
- **Nova/Sangria**: Incremental verifiable computation (IVC) with small recursive overhead
- **Additional recursion layer**: Compress 30M cycles further before Groth16 step


---

## References

- **Detailed implementation**: [Jolt_SNARK_Composition_Implementation.md](Jolt_SNARK_Composition_Implementation.md)
- **Cost analysis**: [Dory_Verification_Cost_Summary.md](Dory_Verification_Cost_Summary.md)
- **Jolt theory**: [01_Jolt_Theory_Enhanced.md](01_Jolt_Theory_Enhanced.md)
- **Verifier code**: [03_Verifier_Mathematics_and_Code.md](03_Verifier_Mathematics_and_Code.md)
- **Implementation**: `examples/recursion/` in codebase
