# Overview

This document provides an analytical cost breakdown of Jolt's verification process, using a typical program with $\log_2 N = 16$ (65,536 cycles) as the baseline. The goal is to understand where verification cycles go and identify optimization opportunities.

**Important**: These are analytical estimates based on operation counts and per-operation cycle costs. Actual measurements may differ due to implementation details, caching, and architectural constraints. Use this as a model for understanding where costs come from, not as exact cycle predictions.

# The Bottom Line

For our test case, verification costs about **~1.5 billion RISC-V cycles** (empirical measurement). Here's where they go:

| Component | Cost (cycles) | % of Total |
| --- | --- | --- |
| Dory $\mathbb{G}_T$ exponentiations (109 @ ~11M each) | 1.2B | **80%** |
| Pairings (5 @ 20M each) | 100M | 6.7% |
| G1/G2 scalar-point muls | 50M | 3.3% |
| Simple $\mathbb{G}_T$ ops + misc | 150M | 10% |
| **Total** | **~1.50B** | **~100%** |

The story is simple: **$\mathbb{G}_T$ exponentiations dominate everything**. They account for 80% of the total cost, with pairings and elliptic curve ops making up most of the rest. This is why PR #975 focuses on bypassing exponentiations—cutting down 109 exponentiations at ~11M cycles each is where the real wins are.

**Source**: Empirical data from Jolt dev team: "The whole verifier with no 'help' is about 1.5B RV64 cycles. ~1.2B of that is just G_T exps."

# Cost breakdown

## A Few Notes on Cost Estimates

**Field operation costs**: Throughout this doc, the analysis uses ~150 cycles per field operation as a rough average. This comes from benchmarking BN254 Montgomery multiplication (~100 cycles) plus overhead for memory access and control flow (~50 cycles). Your mileage may vary—realistic range is 100-200 cycles depending on implementation details. For sumcheck verification, this uncertainty means the difference between ~14M and ~26M cycles, which is negligible compared to the multi-billion cycle exponentiations.

**Why $\log_2 N = 16$?**: This analysis uses the Fibonacci example as the baseline. It has max_trace_length = 65,536 cycles ($2^{16}$). Remember, sumcheck rounds scale with $\log_2(\text{max\_trace\_length})$, not the actual execution length. So even if Fibonacci only uses 1,040 cycles in practice, verification pays for the padded size.

## Breaking Down $\mathbb{G}_T$ Exponentiations

**Analytical model**: $(5 \times \log_2 N) + 29$ exponentiations based on code inspection (10 GT ops per round, $\nu = \log_2 N / 2$ rounds).

For $\log_2 N = 16$:
- **Analytical prediction**: $5 \times 16 + 29 = 109$ exponentiations

**Stage 5 RLC** (Random Linear Combination):
$$C_{\text{combined}} = \prod_{i=1}^{29} C_i^{\gamma_i}$$

This batching step costs us **29 exponentiations** upfront (independent of trace size).

**Main Dory Protocol** (Recursive Folding):
The main protocol contributes $5 \times \log_2 N$ exponentiations:

- For $\log_2 N = 16$: $5 \times 16 = 80$ exponentiations
- **Subtotal: 80 exponentiations**

**Grand Total**: $29 + 80 = 109$ exponentiations

Each exponentiation costs roughly **~11M cycles** (empirically measured), giving us:

$$109 \text{ exps} \times 11\text{M cycles/exp} \approx \mathbf{1,200M \text{ cycles}}$$

This is why the hint-based approach in PR #975 is such a big deal—bypassing these exponentiations cuts the dominant cost.

### Where Do the 29 Polynomials Come From?

The RLC phase combines commitments to 29 different polynomials from Jolt's execution trace. Looking at `jolt-core/src/zkvm/witness.rs` (lines 47-63, 123-148), the `CommittedPolynomial` enum defines exactly which polynomials get committed. These break down into three groups:

**First: 2 fixed increment polynomials** (`RdInc`, `RamInc`) that track register and RAM value changes per cycle. These are the only non-address polynomials in the committed set - everything else is a one-hot address polynomial for Twist/Shout memory checking.

**Second: 16 instruction lookup polynomials** (`InstructionRa(0)` through `InstructionRa(15)`), arising from Shout's D=16 decomposition parameter. Each 64-bit instruction operand is split into 16 4-bit chunks, with each chunk proven via a separate lookup into a 16-entry table.

**Third: Memory system polynomials** consisting of `ram_d` RAM polynomials (`RamRa(0)` through `RamRa(ram_d-1)`) plus `bytecode_d` bytecode polynomials (`BytecodeRa(0)` through `BytecodeRa(bytecode_d-1)`). The chunking parameter `d` is computed dynamically as `ceil(log₂(K) / 8)` where K is the memory/bytecode size, ensuring each chunk fits within 2^8=256 entries. For this test case, ram_d + bytecode_d = 11 to reach the total of 29.

**Important note**: Polynomials like `LeftInstructionInput`, `RightInstructionInput`, `Product`, `WriteLookupOutputToRD`, `WritePCtoRD`, `ShouldBranch`, and `ShouldJump` are **virtual polynomials** (see `VirtualPolynomial` enum at witness.rs:571-616), used in R1CS constraints and sumchecks but never committed via polynomial commitments. They're proven through sumcheck protocol interactions, not via Dory commitments.

**Key observation**: This 29-exponentiation cost is **fixed**—it doesn't scale with trace length. Whether you're verifying 1,000 cycles or 1,000,000 cycles, you still pay 29 exponentiations upfront. It's determined by the witness structure, not execution length.

### Main Dory Protocol: The Recursive Folding

After batching the 29 commitments, the verifier checks the combined commitment using Dory's recursive folding protocol. This runs for $\nu = \frac{\log_2 N}{2} = 8$ rounds (for $\log_2 N = 16$).

8 rounds × 10 exponentiations per round = 80 total exponentiations. Looking at the implementation in `dory/src/core/interactive_protocol.rs`, each round requires **10 GT exponentiations**:

Each round, the verifier updates folded commitments using the prover's messages and challenges $\alpha_i, \beta_i$:

1. **`dory_reduce_verify_update_c()`** (lines 411-451): **4 $\mathbb{G}_T$ scalar multiplications**
   - Computes: $\beta_i \cdot D_2$, $\beta_i^{-1} \cdot D_1$, $\alpha_i \cdot C_+$, $\alpha_i^{-1} \cdot C_-$
   - Where: $D_1$, $D_2$ from previous round; $C_+$, $C_-$ sent by prover in round $i$

2. **`dory_reduce_verify_update_ds()`** (lines 457-511): **6 $\mathbb{G}_T$ scalar multiplications**
   - For $D_1$: $\alpha_i \cdot D_{1L}$, $(\alpha_i\beta_i) \cdot \Delta_{1L}$, $\beta_i \cdot \Delta_{1R}$
   - For $D_2$: $\alpha_i^{-1} \cdot D_{2L}$, $(\alpha_i^{-1}\beta_i^{-1}) \cdot \Delta_{2L}$, $\beta_i^{-1} \cdot \Delta_{2R}$
   - Where: $D_{1L}$, $D_{1R}$, $D_{2L}$, $D_{2R}$ sent by prover; $\Delta$ values from SRS

**Per-round cost**: 10 $\mathbb{G}_T$ scalar multiplications. Each takes ~11M cycles using square-and-multiply over 254-bit scalars in the extension field $\mathbb{F}_{q^{12}}$. Cryptography papers often write these operations as exponentiations ($G^\alpha$) in multiplicative notation, while the implementation uses additive notation ($\alpha \cdot G$)—they're equivalent operations.

**Where does $5 \times \log_2 N$ come from?** Dory's two-dimensional matrix structure means only $\nu = \log_2(\sqrt{N}) = \log_2 N / 2$ rounds of folding are needed. With 10 exponentiations per round, the total is $10 \times (\log_2 N / 2) = 5 \times \log_2 N$ exponentiations. The factor-of-2 savings comes from working with a square matrix rather than a linear vector—folding both dimensions simultaneously.

**Open question - Could Pippenger reduce this further?**

The current implementation computes those 10 GT exponentiations per round independently, which is where the $5 \times \log_2 N$ coefficient comes from. Pippenger's multi-exponentiation algorithm can batch multiple exponentiations ($g_1^{a_1}, g_2^{a_2}, \ldots, g_k^{a_k}$) and compute them together more efficiently—typically giving about a 2× speedup when there's structure to exploit. This raises an interesting question: could the GT exponentiation operations be accumulated across Dory's folding rounds and batched all at once with Pippenger? If feasible, this might push the coefficient down from $5 \times \log_2 N$ to something closer to $4 \times \log_2 N$.

The challenge is that each Dory round uses random challenges from the previous round, creating a dependency chain. Would those exponentiations need to be computed immediately to proceed to the next round, or is there a way to defer them and batch everything at the end? Also, Pippenger's advantage comes from exploiting shared structure in bases or exponents—it's unclear whether Dory's GT exponentiations have enough commonality to make batching worthwhile, or if they're too heterogeneous. Worth exploring whether the protocol structure permits such batching or if the interactive nature fundamentally requires round-by-round computation.

### Why Does Each $\mathbb{G}_T$ Exponentiation Cost ~11M Cycles?

This is the heart of the problem. Let's break down why computing $g^x$ in $\mathbb{G}_T$ is so expensive.

**First, what is $\mathbb{G}_T$?** It's the target group of BN254's pairing: $\mathbb{G}_T = \mathbb{F}_{q^{12}}$, a 12th-degree extension field over a 254-bit prime base field $\mathbb{F}_q$. So every element is essentially twelve 254-bit integers.

**The Algorithm**: Jolt computes $g^x$ using square-and-multiply with a special optimization. Since $\mathbb{G}_T$ elements live in the cyclotomic subgroup, the **Granger-Scott trick** enables faster squaring. The code is at `jolt-core/src/poly/commitment/dory.rs:254` calling arkworks' `cyclotomic_exp()`.

**Step 1: High-Level Algorithm (Square-and-Multiply)**

The standard approach processes the 254-bit exponent bit-by-bit from most significant to least significant. Every iteration squares the current result (254 total squarings), and whenever the algorithm encounters a 1-bit in the exponent, it multiplies by the base $g$ (which happens roughly 127 times for random exponents with 50% Hamming weight). This gives a total operation count of 254 cyclotomic squarings plus approximately 127 general $\mathbb{F}_{q^{12}}$ multiplications. The Granger-Scott optimization is crucial here—it makes cyclotomic squaring about **3× faster** than general multiplication, which matters enormously since squaring happens in every single iteration.

**Step 2: Extension Field Arithmetic**

Both squaring and multiplication in $\mathbb{F}_{q^{12}}$ are built from lower-level field operations. Jolt uses Karatsuba at each level of the extension tower to reduce operation counts:

**Cyclotomic squaring** (Granger-Scott optimization):

Jolt uses Karatsuba multiplication at each level of the extension tower to reduce operation counts. Starting from the tower structure $\mathbb{F}_q \to \mathbb{F}_{q^2} \to \mathbb{F}_{q^6} \to \mathbb{F}_{q^{12}}$:

- Normally, squaring in $\mathbb{F}_{q^{12}} = \mathbb{F}_{q^6}[w]/(w^2 - \gamma)$ requires 3 $\mathbb{F}_{q^6}$ multiplications ($\gamma$ is a constant making the polynomial irreducible)
  - Elements are $a + bw$ where $a, b \in \mathbb{F}_{q^6}$
  - Squaring gives $(a + bw)^2 = (a^2 + \gamma b^2) + (2ab)w$, which is 2 squarings + 1 multiplication = 3 $\mathbb{F}_{q^6}$ muls
- Each $\mathbb{F}_{q^6} = \mathbb{F}_{q^2}[v]/(v^3 - \xi)$ multiplication costs 6 $\mathbb{F}_{q^2}$ muls (Karatsuba reduces naive 9 down to 6; $\xi$ is another irreducible constant)
- Total naive cost: 3 × 6 = **18** $\mathbb{F}_{q^2}$ multiplications
- But Granger-Scott exploits cyclotomic subgroup structure (elements satisfy $x^{(q^6-1)} = 1$, allowing compressed representation) → only **6** $\mathbb{F}_{q^2}$ multiplications (3× improvement)
- Each $\mathbb{F}_{q^2}$ multiplication uses Karatsuba (naive 4 → 3) to get down to base field $\mathbb{F}_q$ multiplications
- **Per cyclotomic squaring**: 6 × 3 = **18 base field multiplications**

**General $\mathbb{F}_{q^{12}}$ multiplication**: With Karatsuba applied recursively at each tower level, general multiplication costs **54 base field multiplications** (vs 144 naive).

See implementation: `ark-ff-0.3.0/src/fields/models/fp12_2over3over2.rs:137-209`

**Step 3: Base Field Operations in RISC-V**

Each 254-bit $\mathbb{F}_q$ multiplication costs **~900 cycles** (Montgomery multiplication).

**Putting it all together**:
- 254 cyclotomic squarings × 18 base muls × 900 cycles = **4.1M cycles** (37%)
- 127 general muls × 54 base muls × 900 cycles = **6.2M cycles** (56%)
- **Subtotal for field operations: ~10.3M cycles**

Beyond the raw field arithmetic, we also incur several sources of overhead. Memory operations for loading and storing $\mathbb{F}_{q^{12}}$ elements add up quickly—each element is 12 × 32 bytes = 384 bytes, and they're constantly moved between registers and memory. Control flow overhead from loop management and conditional branches in the square-and-multiply algorithm contributes additional cycles, as do final reductions and normalization steps needed to keep results in canonical form.

**Total per exponentiation**: ~10.3M cycles for field operations + overhead ≈ **10.5-11M cycles per exponentiation**

**The takeaway**: Even with all the optimizations (Granger-Scott, Karatsuba, Montgomery), Jolt is still doing **~11,430 base field multiplications** per exponentiation plus overhead.

---

## Sumcheck Verification Costs

Compared to the exponentiations, sumcheck verification is relatively cheap—included in the **~150M cycles** for "simple GT ops + misc" overhead. The sumchecks themselves are a small fraction of this. Let's break down the sumcheck structure since it represents the core zkVM logic.

| Stage | Components | Instances | Rounds | Field Ops | RISC-V Cycles |
| --- | --- | --- | --- | --- | --- |
| **Stage 1** | Spartan outer | 1 | 16 | ~1,760 | ~260K |
| **Stage 2** | Spartan + Registers + RAM + Lookups | 6 | ~96 | ~14,000 | ~2.1M |
| **Stage 3** | Spartan + Registers + RAM + Lookups | 8 | ~128 | ~16,480 | ~2.5M |
| **Stage 4** | RAM + Bytecode + Lookups | 7 | ~112 | ~12,640 | ~1.9M |
| **Total** |  | **22** | **~352** | **~44,880** | **~6.7M** |

Each sumcheck instance runs for $\log_2 N = 16$ rounds (matching our trace size). The variation in per-round cost comes from the complexity of the polynomial being verified—Spartan's matrix-vector products are more expensive than simple Hamming weight checks.

### What Happens in Each Sumcheck Round?

The per-round cost depends on four operations (from `jolt-core/src/subprotocols/sumcheck.rs` and `jolt-core/src/poly/unipoly.rs`). First, the prover sends a compressed polynomial by omitting the $c_1$ coefficient to save proof size, sending only $[c_0, c_2, c_3, \ldots, c_d]$. This compression trick works because the verifier knows the claimed sum. Second, the verifier decompresses and evaluates the polynomial $s(X) = c_0 + c_1 X + c_2 X^2 + \cdots + c_d X^d$ by recovering the missing coefficient via $c_1 = \text{hint} - 2c_0 - c_2 - \cdots - c_d$ (exploiting the fact that $s(0) + s(1)$ equals the claimed sum) and then evaluating $s(r)$ at the verifier's random challenge $r$. The cost here scales with polynomial degree: degree-1 (Hamming weight sumchecks) takes around 4 field operations, degree-2 (Spartan, registers) takes about 7 ops, and degree-3 (product layer, booleanity) needs roughly 11 ops.

Third, the Fiat-Shamir transcript is updated by hashing the polynomial and sampling the next challenge using Blake2b-256 (Jolt's default transcript hash). Each Blake2b compression function call processes a 128-byte block and costs roughly 1,000-2,000 RISC-V cycles, though this is still negligible compared to field operations. Finally, the verifier computes the next claim (`expected_output_claim`), and this varies wildly by sumcheck type: simple Hamming weight checks need 10-50 field operations, moderate operations like register/RAM checking require 50-150 field ops, and complex cases like Spartan's product layer with matrix-vector products can cost 100-400 field ops per round.

### Sumcheck Stage Organization

**Empirical total**: Approximately **~7M cycles** across 22 sumcheck instances, 352 total rounds (each instance runs for log₂(N) = 16 rounds). This represents less than 0.5% of the 1.5B cycle total verification cost.

The breakdown below lists the sumcheck instances in each stage with rough per-round operation estimates. **Note: Per-round operation counts are analytical estimates based on polynomial degree and algorithm structure, not empirically measured.**

**Stage 1: Spartan Outer Sumcheck** (1 instance, 16 rounds)

Verifies R1CS constraints $Az \circ Bz - Cz = 0$ across all cycles. Degree-2 polynomial with heavy matrix-vector products in `expected_output_claim`.

- Implementation: `jolt-core/src/zkvm/spartan/mod.rs::stage1_instances()`
- Estimated per-round cost: ~7 (decompression) + ~100 (matrix ops) ≈ ~110 field ops
- Estimated total: 16 × 110 ≈ **~1,760 field ops → ~260K cycles**

**Stage 2: Spartan + Memory + Lookups** (multiple instances, 16 rounds each)

Components implementing `stage2_instances()` in spartan/, registers/, ram/, instruction_lookups/:

1. **Spartan InnerSumcheck**: Batched Az/Bz/Cz evaluation
2. **Registers ReadWriteChecking**: Twist fingerprint matching for register read/writes
3. **RAM RafEvaluation**: Evaluates read/write address fingerprint polynomial
4. **RAM ReadWriteChecking**: Time-ordered memory checking
5. **RAM OutputSumcheck**: Verifies program outputs match claimed values
6. **Instruction Lookups BooleanitySumcheck**: Proves lookup indices are well-formed

**Stage 3: Memory + Lookups Evaluation** (multiple instances, 16 rounds each)

Components implementing `stage3_instances()`:

1. **Spartan PCSumcheck**: PC (program counter) advancement check
2. **Spartan ProductVirtualizationSumcheck**: Virtualizes intermediate products for R1CS
3. **Registers ValEvaluationSumcheck**: Address-ordered register value checking
4. **RAM ValEvaluationSumcheck**: Value accumulation proof for RAM
5. **RAM ValFinalSumcheck**: Final val polynomial evaluation
6. **RAM HammingBooleanitySumcheck**: Address well-formedness checks
7. **Instruction Lookups ReadRafSumcheck**: Shout read-checking for instruction lookups
8. **Instruction Lookups HammingWeightSumcheck**: Lookup decomposition check

**Stage 4: RAM + Bytecode + Lookups** (multiple instances, 16 rounds each)

Final sumcheck batch implementing `stage4_instances()`:

1. **RAM HammingWeightSumcheck**: Address chunking Hamming weight
2. **RAM BooleanitySumcheck**: Address chunks are Boolean
3. **RAM RaSumcheck**: Read address virtualization
4. **Bytecode ReadRafSumcheck**: Offline memory checking for bytecode
5. **Bytecode BooleanitySumcheck**: Bytecode address well-formedness
6. **Bytecode HammingWeightSumcheck**: Bytecode decomposition check
7. **Instruction Lookups RaSumcheck**: Final lookup virtualization

**Note on costs**: Per-sumcheck costs vary widely. Simple degree-1 Hamming weight checks use ~10-50 field ops/round, while complex degree-2/3 sumchecks with matrix-vector products can use 100-200+ ops/round. Without detailed profiling of each sumcheck's `expected_output_claim` computation, precise per-stage totals would be speculative. The empirically measured ~7M total across all 22 instances is the reliable figure.

## Other Dory Operations

Beyond the 1.2B cycles for GT scalar multiplications, the remaining **~300M cycles** (derived from 1.5B total - 1.2B GT ops) come from other cryptographic operations and overhead. The breakdown below is based on operation counts verified from code and estimated per-operation costs:

| Operation Type | Count (verified) | Cost per Op (estimated) | Total Cost | % of Total |
| --- | --- | --- | --- | --- |
| **Pairings** ($e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$) | 5 | ~20M cycles | **~100M cycles** | ~6.7% |
| **G1/G2 Scalar-Point Multiplications** | 52 | ~500K-1M cycles | **~40-50M cycles** | ~3% |
| **Simple $\mathbb{G}_T$ Operations + Misc** | many | varies | **~150M cycles** | ~10% |
| **Total (non-GT-scalar-mul)** | - | - | **~300M cycles** | **~20%** |

**Note**: Operation counts (5 pairings, 52 scalar muls) are verified from Dory source code. Per-operation cycle costs are analytical estimates based on algorithm complexity (not empirically measured). The "Simple GT ops + misc" is a residual category to account for the empirically measured 1.5B total.

### A. Pairings (5 pairings @ ~20M cycles each ≈ 100M cycles)

BN254 bilinear pairings $e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$ compute the final verification checks after Dory's folding completes. The count is **fixed at 5**—independent of trace size N.

**Operation count (verified from code)**: The external `dory` crate performs four pairings in `apply_fold_scalars` (called once after all folding rounds complete at `inner_product.rs:84`), then one final pairing in `verify_final_pairing` (at `inner_product.rs:86`) that performs the actual verification check. There's also an additional pairing at `evaluate.rs:293` for VMV sigma protocol verification, though this typically isn't counted in the baseline figure.

**Cost per pairing (analytical estimate)**: Each pairing has two phases. The **Miller loop** processes the 254-bit scalar (BN254's embedding degree), computing line functions and accumulating in $\mathbb{F}_{q^{12}}$, estimated at ~12M cycles. Then **final exponentiation** maps the Miller loop output to $\mathbb{G}_T$ via $(q^{12} - 1)/r$ exponentiation using cyclotomic subgroup structure, estimated at ~8M cycles. **Total: ~20M cycles per pairing** (analytical estimate based on algorithm complexity, not empirically measured for Jolt verifier).

**What they verify**: After $\log_2 N$ folding rounds, Dory reduces to a base case verified via the **Scalar-Product protocol**. The verification equation checks:

$$e(E_1P_1, E_2P_2) \stackrel{?}{=} R \cdot Q^c \cdot C'^{c^2} \cdot [\text{blinding factor terms}]$$

Where:
- $E_1, E_2$: Opened scalar values from prover's randomized witness
- $R$: Commitment to mask product $e(d_1, d_2)$
- $Q$: Commitment to cross-term $e(d_1, v_2) + e(v_1, d_2)$
- $C'$: Folded inner-pairing product commitment
- $c$: Verifier's random challenge

The left side (1 pairing) uses opened values; the right side combines commitments from earlier rounds. This equation expands to 5 total pairings in the implementation.

For the full mathematical treatment, see [Theory/Dory.md](../Theory/Dory.md) lines 236-269 and [01_Jolt_Theory_Enhanced.md](01_Jolt_Theory_Enhanced.md) lines 6372-6425.

**EVM deployment note**: Pairings have precompile 0x08 (bn256Pairing) that costs only ~113K gas for a 2-pair check. So pairings aren't a bottleneck for on-chain verification—only in RISC-V recursion where there are no precompiles.

### B. G1/G2 Scalar-Point Multiplications (52 ops @ ~500K-1M cycles each ≈ 40-50M cycles)

Beyond pairings, Dory verification requires elliptic curve scalar multiplications on BN254's base curve (G1) and twisted curve (G2) throughout the protocol—preparing pairing inputs, computing linear combinations of commitments with challenges, and updating protocol state. These operations come in two flavors: single scalar multiplications computing $[k]P$ for scalar $k$ and point $P$, and multi-scalar multiplications (MSMs) computing $\sum_{i} [k_i]P_i$ efficiently using Pippenger's algorithm.

**Operation count (verified from code)**: Counting from `dory/src/core/interactive_protocol.rs`:
- **48 scalar multiplications during folding**: Each of ν = 8 Dory folding rounds performs 6 scalar muls (3 for E₁ updates at lines 531-533, 3 for E₂ updates at lines 537-539)
- **4 additional scalar muls**: The `apply_fold_scalars` phase adds 2 (lines 601, 612), and the final pairing check adds 2 more (lines 637-638)
- **Total**: **52 G1/G2 scalar multiplications**

**Cost per operation (analytical estimate)**: A single scalar multiplication uses double-and-add (analogous to square-and-multiply), performing 254 point doublings plus roughly 127 point additions for the average case (50% Hamming weight). Cost depends on the curve: ~500K cycles for G1 (256-bit base field) vs ~1M cycles for G2 (512-bit extension field). When multiple scalar multiplications can be batched as MSMs, Pippenger's algorithm amortizes the cost to roughly 50% of the naive approach by exploiting common scalar bits.

**Estimated total**: At ~750K cycles average (accounting for mix of G1/G2 operations and potential Pippenger batching), the 52 operations give approximately **40M cycles**. Adding overhead for point additions and protocol bookkeeping brings the estimate to **~40-50M cycles** (analytical estimate, not empirically measured).

These operations are unavoidable in Dory's structure—before the final pairing checks, the verifier must compute linear combinations of commitment points using the verifier's random challenges to properly update the protocol state.

### C. Simple $\mathbb{G}_T$ Operations + Misc (~150M cycles, residual estimate)

The remaining **~150M cycles** (derived as 1.5B total - 1.2B GT scalar muls - 100M pairings - 50M G1/G2 ops) come from the "everything else" category—operations that are cheap compared to GT scalar multiplications.

**Simple $\mathbb{G}_T$ operations** (NOT full scalar multiplications): These occur throughout the Dory protocol in `interactive_protocol.rs` for accumulating products, combining pairing results, and building verification equations. Per-operation costs are analytically derived:
- **$\mathbb{F}_{q^{12}}$ multiplication**: ~49K cycles (54 base field muls × ~900 cycles each, from earlier analysis)
- **$\mathbb{F}_{q^{12}}$ addition**: ~1.2K cycles (12 base field additions × ~100 cycles each)
- These simple GT operations are ~200-10,000× cheaper than GT scalar multiplications (~11M cycles)
- Estimated total: **~20M cycles** across entire verification (rough estimate based on typical operation counts)

**Transcript management**: Blake2b-256 for Fiat-Shamir hashing throughout the protocol:
- Each sumcheck round: 1 append (polynomial coefficients) + 1 challenge (random scalar)
- 352 total sumcheck rounds → 704 transcript operations
- Additional: Batching coefficients (start of each stage), Dory protocol updates (8 folding rounds + final checks), commitment appends
- Estimated total: **~800-1,000 transcript operations** at 1,000-2,000 cycles each → **~1-2M cycles**

**Miscellaneous overhead** (~128M cycles, residual): The remaining cycles come from operations difficult to measure precisely without detailed profiling:
- Memory operations (loading/storing field elements, commitments)
- Control flow overhead (loops, conditionals, function calls)
- Simple scalar arithmetic permeating verification logic
- Field operations in sumchecks (~7M as calculated earlier)

**Note**: This entire category is essentially a residual—what's left after accounting for the dominant operations (GT scalar muls, pairings, G1/G2 ops). The 150M total is derived from the empirical 1.5B measurement, not independently measured.

---

# Optimization via SNARK Composition (PR #975)

Now that the cycle distribution is clear, here's the optimization approach in PR #975. The idea is simple but powerful: instead of making the verifier compute those 109 expensive exponentiations, have the prover provide the *answers* along with a *proof that those answers are correct*.

**The trade-off**: Exchange 1,200M cycles of $\mathbb{G}_T$ exponentiations for ~340M cycles of Grumpkin MSMs + sumcheck verification.

**Overall result**: ~1.5B cycles → ~640M cycles (**~2.3× speedup**)

**Note**: These Stage 6 cost estimates (340M cycles) come from PR #975's experimental implementation and analysis in `Hyrax_costs.md`. PR #975 is experimental and will not be merged as-is, but demonstrates the hint-based optimization approach that will be extended to other expensive operations in future work.

## The Architecture: Hints + Deferred Verification

This optimization pattern is sometimes called **SNARK composition** or **proof-carrying data**. Here's how it works:

**Traditional approach (baseline)**: The verifier receives GT commitments, computes 109 GT exponentiations (1,200M cycles), and performs equality checks.

**Optimized approach (Stage 5 + Stage 6)**: In Stage 5, the verifier receives GT commitments along with precomputed exponentiation results as hints (~11 KB). It performs equality checks using these hints—just GT multiplications costing ~2M cycles instead of expensive exponentiations. Then in Stage 6, the verifier checks a separate proof that those hints are correct (~340M cycles). This auxiliary proof uses a different curve (Grumpkin, for field matching with BN254's base field Fq) and a different PCS (Hyrax, which uses MSM-based commitments that are much cheaper than Dory's GT exponentiations), running sumcheck over the square-and-multiply constraints.

This is called **SNARK composition** because Stage 6 is essentially a separate SNARK proving that the Dory verifier's GT operations were computed correctly. There are two proofs: the main Jolt execution proof (Stages 1-5, using Dory/BN254) and an auxiliary proof for the GT exponentiations (Stage 6, using Hyrax/Grumpkin).

The pattern works in three steps: First, the prover captures all intermediate square-and-multiply values while computing the Dory exponentiations. Second, instead of computing these exponentiations itself, the verifier accepts precomputed results as hints in Stage 5. Third, the verifier checks a separate proof in Stage 6 that those hints are correct.

## Modified Cost Breakdown

Let's see how the costs change with this optimization:

### Baseline Verification (No Optimization)

| Component | Cost (cycles) | % of Total |
|-----------|---------------|------------|
| Stage 5 GT exponentiations (109) | 1,200M | **80%** |
| Pairings (5) | 100M | 6.7% |
| G1/G2 operations | 50M | 3.3% |
| Simple GT ops + misc (incl. sumchecks) | 150M | 10% |
| **Total** | **~1,500M** | **100%** |

###  Optimized Verification (With Stage 6)

| Component | Cost (cycles) | % of Total |
|-----------|---------------|------------|
| **Stage 6 (Hyrax approach)** | **~340M** | **53%** |
| Pairings (5) | 100M | 16% |
| G1/G2 operations | 50M | 8% |
| Stage 5 RLC (with hints) | 2M | <1% |
| Simple GT ops + misc (incl. sumchecks) | 150M | 23% |
| **Total** | **~642M** | **100%** |

**Key observations**:
- **Stage 5 savings**: 1,200M → 2M cycles (~1,198M saved) by accepting hints instead of computing exponentiations
- **Stage 6 cost**: ~340M cycles to verify those hints were correct
- **Net savings**: ~860M cycles (~57% reduction in total verification cost)
- **New bottleneck**: Stage 6 becomes the dominant component (53% of cost), but it's paid in efficient Grumpkin MSMs rather than expensive BN254 $\mathbb{G}_T$ exponentiations

## Stage 5: Modified Dory Verification with Hints

In baseline Jolt verification, Stage 5 is where the Dory polynomial commitment scheme verifies all the polynomial openings from Stages 1-4. This involves two expensive phases: first, a random linear combination (RLC) that batches 29 polynomial commitments using random challenges ($C_{\text{joint}} = \sum_{i=1}^{29} \gamma_i \cdot C_i$), which requires 29 GT exponentiations; second, the main Dory folding protocol with 8 rounds, each requiring 10 GT exponentiations, for a total of 80 more. Combined, Stage 5 performs **109 GT exponentiations** costing approximately 1,200M cycles.

PR #975 modifies Stage 5 to accept hints from the prover. Instead of computing these 109 exponentiations, the verifier receives the precomputed results directly (about 11 KB of additional proof data). The verifier then uses these hint values in the Dory equality checks, which now only require simple GT multiplications and additions instead of full exponentiations. This reduces Stage 5's GT exponentiation cost from 1,200M cycles down to approximately **2M cycles** for the GT arithmetic using the hints—a savings of roughly 1,198M cycles.

The catch, of course, is that accepting hints without verification would be unsound. This is where Stage 6 comes in: it provides a separate proof that all 109 hint values were computed correctly.

## Stage 6 Breakdown

Stage 6 verifies the 109 exponentiation hints using three components (note: actual implementation may verify more for padding). Each exponentiation $g^x$ requires proving that 254 square-and-multiply steps were done correctly.

**Total witness size**: For each exponentiation, the prover commits to:
- 255 intermediate accumulator values ($\rho_0, \rho_1, \ldots, \rho_{254}$)
- 254 quotient values (from polynomial reduction in $\mathbb{F}_{q^{12}}$)
- 1 base element $g$

Each $\mathbb{G}_T$ element has 12 coefficients over $\mathbb{F}_q$, packed into a 4-variable MLE (16 evaluations on Boolean hypercube). This gives:
- **Per exponentiation**: 510 $\mathbb{G}_T$ values = 510 MLEs
- **For 109 exponentiations**: ~55,590 MLEs (actual implementation may include padding)
- **Witness size**: ~28 MB for 109 exps (~260 KB per exponentiation)

### Component A: Hyrax Batch Commitment (~90-140M cycles)

**What it does**: The verifier receives ~55,590 individual Hyrax commitments from the prover (one per MLE in the witness) and homomorphically combines them into a single batched commitment for efficient verification.

Each Hyrax commitment consists of 4 Grumpkin curve points representing row commitments in Hyrax's 4×4 matrix structure (used for committing to 4-variable MLEs). To batch all 55,590 commitments, the verifier must combine each row position separately using random linear combination:

$$C_{\text{batched}}[j] = \sum_{i=1}^{55,590} \gamma_i \cdot C_i[j] \quad \text{for } j \in \{0, 1, 2, 3\}$$

This requires **4 multi-scalar multiplications (MSMs)** over Grumpkin—one for each row position:
- Each MSM combines 55,590 points: $C_{\text{batched}}[j]$ for a single row
- Total: 4 MSMs × 55,590 operations each = 222,360 scalar multiplications total

Note: This is different from standard Hyrax verification (which uses 2 MSMs for opening). These 4 MSMs are for batching the commitments themselves before verification.

**Cost breakdown**:
- Naive approach: 222,360 ops × 500K cycles = ~111B cycles
- With Pippenger optimization: ~90-140M cycles (~1000× reduction from naive)

The logarithmic scaling of Pippenger's algorithm ($O(n/\log n)$ vs $O(n)$ naive) is what makes this tractable. For $n \approx 222{,}360$ ($\log_2 n \approx 18$), bucket aggregation and precomputation provide massive savings.

### Component B: ExpSumcheck (~240M cycles)

**What it does**: Verifies that all accepted exponentiation results satisfy the square-and-multiply constraints (109 exponentiations for $\log_2 N = 16$).

For each step $j$ of exponentiation $k$, the constraint is:
$$e_j := r_{j+1} - r_j^2 \cdot (1 + b_j(g-1)) = 0$$

Where $r_j$ are intermediate accumulator values and $b_j$ are bits of the exponent.

**Two levels of batching**:

1. **Within each exponentiation** (254 steps): Batch all 254 error terms using random $\gamma$:
   $$\sum_{j=0}^{253} \gamma^j \cdot e_j = 0$$

2. **Across 109 exponentiations**: Batch all 109 instances using random $\{\beta_k\}$:
   $$\sum_{k=1}^{109} \beta_k \cdot \left(\sum_{x \in \{0,1\}^4} E_k(x)\right) = 0$$

**Why 4 rounds, not 254?** Each $\mathbb{G}_T$ element is stored as a 4-variable MLE. Sumcheck operates over the $2^4 = 16$ evaluations on the Boolean hypercube, NOT over the 254 steps directly. The 254 constraints are algebraically batched (via $\gamma$ powers) BEFORE sumcheck starts.

**Per-round cost**: ~3,600 field operations (dominated by evaluating witness polynomials at challenge points)

**Total cost**:
- 4 rounds × ~3,600 field ops/round = ~14,400 field ops
- Additional overhead for 109-instance batching: ~16,000× scaling
- **Estimated total**: ~240M cycles

### Component C: Hyrax Opening Proof (~2M cycles)

**What it does**: Verifies that the batched polynomial commitment (from Component A) opens correctly at the challenge point determined by ExpSumcheck (from Component B).

Uses Hyrax's tensor product structure for efficient verification with two small MSMs, each involving only 4 scalar multiplications:
- **MSM #1**: $C_{\text{derived}} = \sum_{i=0}^{3} L_i \cdot C_{\text{batched}}[i]$ (combining 4 batched row commitments)
- **MSM #2**: $C_{\text{product}} = \sum_{j=0}^{3} u_j \cdot G_j$ (combining 4 generator points)
- **Check**: $C_{\text{derived}} \stackrel{?}{=} C_{\text{product}}$ and $\langle u, L \rangle \stackrel{?}{=} v$

Each MSM computes 4 scalar multiplications + 3 point additions. At ~500K cycles per scalar multiplication on Grumpkin, each MSM costs roughly 4 × 500K = 2M cycles, but these are tiny MSMs where Pippenger provides minimal benefit.

**Total cost**: 2 MSMs with 4 bases each ≈ **~2M cycles** (may be closer to 4-5M without significant Pippenger speedup for such small sizes)

### Stage 6 Total

| Component | Operations | Cost (cycles) | % of Stage 6 |
|-----------|------------|---------------|--------------|
| A. Hyrax Batch Commitment | 222,360-base MSM | 90-140M | 26-41% |
| B. ExpSumcheck | 109 instances × 4 rounds | ~240M | 59-71% |
| C. Hyrax Opening Proof | 2 × 4-base MSMs | ~2M | <1% |
| **Total Stage 6** | | **~340M cycles** | **100%** |

## Why Grumpkin? The BN254-Grumpkin 2-Cycle

The choice of Grumpkin for Stage 6 comes down to **field matching**. BN254 and Grumpkin form what's called a 2-cycle of elliptic curves, meaning their base and scalar fields swap: BN254's base field $\mathbb{F}_q$ is exactly Grumpkin's scalar field, and Grumpkin's base field is approximately BN254's scalar field (the orders are very close but not identical).

This matters because of what Stage 6 needs to commit to. Each $\mathbb{G}_T$ element in the Dory witness is an element of $\mathbb{F}_{q^{12}}$, represented as 12 coefficients where each coefficient lives in BN254's base field $\mathbb{F}_q$. When we commit to these coefficients using Hyrax on Grumpkin, we're computing Pedersen commitments like $C = \sum_{i=0}^{11} m_i \cdot G_i$ where the $m_i$ values need to be scalars for the Grumpkin curve. Since Grumpkin's scalar field is exactly $\mathbb{F}_q$, the GT coefficients are already in the right format—they're native scalars that can be used directly in the commitment without any conversion.

Without Grumpkin, we'd have a serious problem. If we tried to use BN254 for Stage 6, we'd need to represent $\mathbb{F}_q$ elements (the GT coefficients) as scalars in BN254's scalar field. Since these fields don't match, we'd need limb decomposition—breaking each coefficient into multiple smaller pieces and handling them separately. This roughly doubles the number of scalar multiplications per commitment and makes the sumcheck arithmetic 10× more expensive because every field operation requires non-native field constraints. The result would be somewhere between 1-3.6B cycles for Stage 6 instead of the 340M cycles achieved with Grumpkin. The 2-cycle property saves roughly 700M-3.2B cycles in Stage 6 alone.

### Why Hyrax for Stage 6?

Hyrax is chosen for Stage 6 because it uses MSM-based commitments in $\mathbb{G}_1$ of Grumpkin, which are far cheaper than Dory's $\mathbb{G}_T$ exponentiations in BN254. Using Dory for Stage 6 would defeat the purpose—we'd be back to verifying expensive GT exponentiations. Hyrax commitments cost roughly 90-140M cycles (via Pippenger MSMs over Grumpkin), while equivalent Dory commitments would cost significantly more due to GT exponentiations. The choice of Hyrax is about cost efficiency, not avoiding circularity—it's simply a cheaper PCS for this auxiliary proof.

## Summary

This analysis identifies where Jolt verification cycles go. The baseline verifier costs roughly 1.5B RISC-V cycles, with GT exponentiations accounting for 80% (1.2B cycles) of the total. Pairings add another 100M cycles, G1/G2 scalar multiplications contribute 50M cycles, and the remaining 150M comes from simple GT operations, sumchecks, and miscellaneous overhead.

PR #975 demonstrates an optimization approach that reduces verification to approximately 640M cycles—a 2.3× speedup. The core idea is replacing expensive GT exponentiations with a hint-based mechanism: the prover provides precomputed exponentiation results along with a separate proof (Stage 6) that these results are correct. Stage 6 uses Grumpkin and Hyrax instead of BN254 and Dory, trading 1.2B cycles of GT exponentiations for 340M cycles of MSM-based verification plus sumcheck. This experimental work won't be merged as-is but establishes the pattern for future optimizations.

## Future Work

The hint-based approach from PR #975 will be extended to other expensive verifier operations beyond GT exponentiations. Similar optimization patterns can be applied to GT multiplications, pairings, and other costly cryptographic operations—the prover provides hints for expensive computations, and a separate auxiliary proof verifies correctness using cheaper primitives. This transforms expensive native computation into proof verification problems.

Additionally, Jolt plans to integrate lattice-based techniques into the polynomial commitment scheme. This will provide post-quantum security while potentially reducing both proving and verification costs. The specific approach is still being developed.

The emerging pattern is clear: identify expensive operations, provide hints from the prover, and verify those hints with cheaper proof systems. Sumcheck continues to be the universal verification primitive enabling these transformations.

## Open Questions and Uncertainties

Several aspects of this analysis involve estimates or open questions that warrant further investigation:

**Potential Pippenger optimization for Dory GT exponentiations**: The current implementation performs 10 GT exponentiations per Dory folding round independently, yielding the $5 \times \log_2 N$ coefficient. Could these operations be accumulated across rounds and batched using Pippenger's multi-exponentiation algorithm? If the protocol structure permits deferring exponentiations until after all challenges are generated, this might reduce the coefficient from $5 \times \log_2 N$ to approximately $4 \times \log_2 N$. The challenge is understanding whether Dory's round dependencies require immediate computation or allow batching.

**Cycle costs per field operation**: Throughout this analysis, we use ~900 cycles for BN254 base field multiplication, ~150 cycles for average field operations in sumcheck, and ~500K cycles for Grumpkin scalar multiplication. These are reasonable estimates based on Montgomery multiplication benchmarks and operation complexity, but actual costs depend heavily on implementation details, compiler optimizations, and micro-architectural factors. More precise measurements would strengthen the analysis.

**Stage 6 component costs**: The ExpSumcheck estimate of ~240M cycles is based on rough per-round field operation counts multiplied by scaling factors for 109-instance batching. The Hyrax batch commitment estimate of 90-140M cycles assumes Pippenger provides significant speedup for 55,590-base MSMs. Both could benefit from more detailed profiling or implementation-based measurements.

**Miscellaneous overhead breakdown**: The ~30M cycles attributed to memory operations, control flow, and arithmetic overhead is largely estimated to make the totals match empirical measurements. While the transcript operation count (800-1,000 Blake2b hashes at 1,000-2,000 cycles each) is analytically grounded, the remaining breakdown is less precise.

**G1/G2 operation costs with Pippenger**: The 52 G1/G2 scalar multiplications in Dory verification are estimated at ~750K cycles average, assuming some Pippenger batching benefit. The actual batching opportunities depend on which operations can be deferred and combined, which requires deeper protocol analysis.
