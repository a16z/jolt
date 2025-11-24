# Jolt Verification Costs and the Need for SNARK Composition

## TL;DR

**Problem**: Jolt verification is **too expensive for on-chain deployment** due to costly $\mathbb{G}_T$ exponentiations in the Dory polynomial commitment scheme.

**Cost breakdown for typical program** ($\log_2 N = 16$):
- **Dory verification**: ~109 $\mathbb{G}_T$ exponentiations → ~1.09B RISC-V cycles
  - **80 from main Dory protocol** (5 × log₂ N, verified from actual implementation)
  - **29 from stage 5 RLC** (batched polynomial commitments, verified from code)
- **Sumcheck verification**: 22 sumcheck instances → ~20M cycles
- **Other operations**: Pairings, G1/G2 scalar muls, misc → ~190M cycles
- **Total**: ~1.3B cycles (vs. 30M target for economical on-chain verification)

> **Note**: The main Dory protocol uses **round-by-round computation (80 exps = 5 × log₂ N)**, verified by inspecting the `dory` crate source code. See Section 1.4 for full details.

**Solution**: SNARK composition / recursion to compress verification into ~30M cycles.

---

## Part 1: Understanding Jolt Verification Costs

Jolt verification consists of two major components:

1. **Sumcheck verification** (22 instances across 4 stages)
2. **Polynomial commitment verification** (Dory with $(5 \times \log_2 N) + 29$ $\mathbb{G}_T$ exponentiations)

The Dory $\mathbb{G}_T$ exponentiations dominate the cost (~90% of total verification time).

**Note**: The exponentiation count has two components:
- **Main Dory protocol**: $5 \times \log_2 N$ (scales with trace size)
- **Stage 5 RLC**: 29 exponentiations (fixed, independent of N)

---

### 1.1 Complete Verification Cost Breakdown

#### Fibonacci Example: fib(50)

**Program configuration** (`max_trace_length = 65536` in `examples/fibonacci/guest/src/lib.rs`):
- **Max trace length**: 65,536 cycles = 2^16
- **$\log_2 N = 16$**

**Measured trace** (from `cargo run --release` in `examples/fibonacci`):
- Actual execution: **1,040 cycles** (737 RISC-V + 303 virtual)
- Padded to max_trace_length: **$N = 65{,}536 = 2^{16}$**
- **$\log_2 N = 16$**

**Component 1: Sumcheck Verification**

Jolt uses **22 sumcheck instances** across 4 stages (verified from implementation).

| Stage | Components | Instances | Rounds | Field Ops | RISC-V Cycles |
|-------|-----------|-----------|--------|-----------|---------------|
| **Stage 1** | Spartan outer | 1 | 16 | ~1,760 | ~260K |
| **Stage 2** | Spartan + Registers + RAM + Lookups | 6 | ~96 | ~14,000 | ~2.1M |
| **Stage 3** | Spartan + Registers + RAM + Lookups | 8 | ~128 | ~16,480 | ~2.5M |
| **Stage 4** | RAM + Bytecode + Lookups | 7 | ~112 | ~12,640 | ~1.9M |
| **Total** | | **22** | **~352** | **~44,880** | **~6.7M** |

**Converting to full RISC-V cycles** (including all overhead):
- **Base field operations**: 44,880 field ops × ~150 cycles/op = **~6.7M cycles**
- **Transcript management**: 352 rounds × ~200 cycles/round = **~0.07M cycles**
- **Memory operations & batching overhead**: Additional ~13M cycles
- **Total for $\log_2 N = 16$ (Fibonacci with max_trace_length = 65536)**: **~20M cycles**

**Notes**:
1. **Field operation cost uncertainty**: The 150 cycles/field-op is an **engineering estimate** (not directly measured):
   - **No RISC-V cycle benchmarks found**: Jolt codebase contains operation-counting infrastructure ([`jolt-core/src/field/tracked_ark.rs`](../jolt-core/src/field/tracked_ark.rs) provides `TrackedFr` wrapper to count field operations, and [`jolt-core/src/utils/counters.rs`](../jolt-core/src/utils/counters.rs) defines `FieldOpCounts` struct) but no direct RISC-V cycle measurements
   - **Arkworks implementation details**: Uses Montgomery multiplication from arkworks-algebra fork (branch `feat/fewer-reductions`)
   - **Reasonable estimate breakdown**:
     - Montgomery modular multiplication: ~80-120 cycles (core 254-bit field operation)
     - Field addition/subtraction: ~10-20 cycles (256-bit integer arithmetic)
     - Memory overhead: +20-50 cycles (cache misses, load/store patterns)
     - Control flow overhead: +10-30 cycles (function calls, conditionals)
     - **Total: likely 100-200 cycles/op** depending on workload and implementation details
   - **Impact of uncertainty**: If actual cost is 100 cycles/op → ~14M total sumcheck; if 200 cycles/op → ~26M total sumcheck
   - **Recommendation**: Future work should measure actual RISC-V cycle costs via guest program benchmarks (compile BN254 field ops to RISC-V, run in Jolt tracer with cycle tracking)
2. **Trace length**: The actual execution uses only 1,040 cycles, but verification cost depends on max_trace_length (padded size), not actual trace length. The sumcheck rounds scale with log₂(max_trace_length).

#### Detailed Per-Round Cost Breakdown

**Per-round operations** (from [`sumcheck.rs`](../jolt-core/src/subprotocols/sumcheck.rs#L694-L704)):

**1. Receive compressed polynomial** ([`unipoly.rs:243`](../jolt-core/src/poly/unipoly.rs#L243))
   - Prover sends degree-$d$ coefficients: $[c_0, c_2, c_3, \ldots, c_d]$ (omitting $c_1$ to save proof size)

**2. Decompress + evaluate at challenge** ([`unipoly.rs:394-408`](../jolt-core/src/poly/unipoly.rs#L394-L408))

   **Why these specific operation counts?**

   For polynomial $s(X) = c_0 + c_1 X + c_2 X^2 + \cdots + c_d X^d$:

   **Recovery phase**: Compute $c_1 = \text{hint} - 2c_0 - c_2 - \cdots - c_d$
   - Degree-1: $c_1 = \text{hint} - 2c_0$ → **2 subtractions**
   - Degree-2: $c_1 = \text{hint} - 2c_0 - c_2$ → **3 subtractions**
   - Degree-3: $c_1 = \text{hint} - 2c_0 - c_2 - c_3$ → **4 subtractions**

   **Evaluation phase**: Compute $s(r) = c_0 + c_1 \cdot r + c_2 \cdot r^2 + \cdots + c_d \cdot r^d$
   - Degree-1: $c_0 + c_1 \cdot r$ → **1 mult + 1 add = 2 ops**
   - Degree-2: $c_0 + c_1 \cdot r + c_2 \cdot r^2$ → **2 mults (for $r^2$) + 2 adds = 4 ops**
   - Degree-3: $c_0 + c_1 \cdot r + c_2 \cdot r^2 + c_3 \cdot r^3$ → **4 mults + 3 adds = 7 ops**

   **Total per polynomial degree**:
   - **Degree-1**: 2 (recovery) + 2 (eval) = **~4 field ops** (hamming weight sumchecks)
   - **Degree-2**: 3 (recovery) + 4 (eval) = **~7 field ops** (Spartan, registers, RAF evaluation)
   - **Degree-3**: 4 (recovery) + 7 (eval) = **~11 field ops** (product layer, booleanity checks)

**3. Update transcript & sample challenge** ([`sumcheck.rs:699`](../jolt-core/src/subprotocols/sumcheck.rs#L699))
   - SHA256 hash to update Fiat-Shamir transcript
   - Sample random challenge from transcript
   - **Cost**: ~100-200 RISC-V cycles (hashing overhead, not field operations)

**4. Call `expected_output_claim`** (varies by sumcheck instance type)

   Computes the next round's claim by evaluating auxiliary polynomials and checking consistency between sumcheck components.

   **Complexity by instance type**:

   | Instance Type | Field Ops | Examples |
   |--------------|-----------|----------|
   | **Simple** | 10-50 | Hamming weight, simple accumulation |
   | **Moderate** | 50-150 | Register read/write checking, RAF evaluation |
   | **Complex** | 100-400 | Spartan product layer, booleanity checks, instruction lookups |

#### Weighted Cost Analysis by Sumcheck Type

**Stage 1: Spartan Outer Sumcheck** (1 instance, 16 rounds for log₂ T=16)

| # | Instance | Rounds | Ops/Round | Field Ops | Cycles | Description |
|---|----------|--------|-----------|-----------|--------|-------------|
| 0 | **Spartan OuterSumcheck** | 16 | 110 | 1,760 | ~260K | R1CS constraints Az ∘ Bz - Cz |
| **Stage 1 Total** | | **16** | | **1,760** | **~260K** | |

**Stage 2: Spartan + Memory + Lookups** (6 instances, ~96 rounds total)

| # | Instance | Rounds | Ops/Round | Field Ops | Cycles | Description |
|---|----------|--------|-----------|-----------|--------|-------------|
| 1 | **Spartan InnerSumcheck** | 16 | 150 | 2,400 | ~360K | Batches Az, Bz, Cz evaluation with RLC |
| 2 | **Registers ReadWriteChecking** | 16 | 100 | 1,600 | ~240K | Twist time-ordered for registers |
| 3 | **RAM RafEvaluation** | 16 | 175 | 2,800 | ~420K | Read/write address fingerprint eval |
| 4 | **RAM ReadWriteChecking** | 16 | 175 | 2,800 | ~420K | Twist time-ordered for RAM |
| 5 | **RAM OutputSumcheck** | 16 | 175 | 2,800 | ~420K | Verifies program outputs |
| 6 | **Instruction Lookups BooleanitySumcheck** | 16 | 100 | 1,600 | ~240K | Proves lookup indices well-formed |
| **Stage 2 Total** | | **96** | | **14,000** | **~2.1M** | |

**Stage 3: Memory + Lookups Evaluation** (8 instances, ~128 rounds)

| # | Instance | Rounds | Ops/Round | Field Ops | Cycles | Description |
|---|----------|--------|-----------|-----------|--------|-------------|
| 7 | **Spartan PCSumcheck** | 16 | 120 | 1,920 | ~290K | Batches NextUnexpandedPC and NextPC |
| 8 | **Spartan ProductVirtualizationSumcheck** | 16 | 120 | 1,920 | ~290K | Proves product of operands |
| 9 | **Registers ValEvaluationSumcheck** | 16 | 100 | 1,600 | ~240K | Twist address-ordered for registers |
| 10 | **RAM ValEvaluationSumcheck** | 16 | 150 | 2,400 | ~360K | Proves value accumulation |
| 11 | **RAM ValFinalSumcheck** | 16 | 150 | 2,400 | ~360K | Final val polynomial evaluation |
| 12 | **RAM HammingBooleanitySumcheck** | 16 | 150 | 2,400 | ~360K | One-hot encoding check |
| 13 | **Instruction Lookups ReadRafSumcheck** | 16 | 120 | 1,920 | ~290K | Shout read-checking phase |
| 14 | **Instruction Lookups HammingWeightSumcheck** | 16 | 120 | 1,920 | ~290K | Lookup decomposition weight |
| **Stage 3 Total** | | **128** | | **16,480** | **~2.5M** | |

**Stage 4: RAM + Bytecode + Lookups** (7 instances, ~112 rounds)

| # | Instance | Rounds | Ops/Round | Field Ops | Cycles | Description |
|---|----------|--------|-----------|-----------|--------|-------------|
| 15 | **RAM HammingWeightSumcheck** | 16 | 120 | 1,920 | ~290K | RAM address chunking weight |
| 16 | **RAM BooleanitySumcheck** | 16 | 120 | 1,920 | ~290K | RAM chunks are Boolean |
| 17 | **RAM RaSumcheck** | 16 | 120 | 1,920 | ~290K | Read address virtualization |
| 18 | **Bytecode ReadRafSumcheck** | 16 | 110 | 1,760 | ~260K | Offline memory checking |
| 19 | **Bytecode BooleanitySumcheck** | 16 | 110 | 1,760 | ~260K | Bytecode indices well-formed |
| 20 | **Bytecode HammingWeightSumcheck** | 16 | 110 | 1,760 | ~260K | Bytecode lookup decomposition |
| 21 | **Instruction Lookups RaSumcheck** | 16 | 100 | 1,600 | ~240K | Final lookup virtualization |
| **Stage 4 Total** | | **112** | | **12,640** | **~1.9M** | |

**Summary: Detailed Breakdown by Instance**

See individual instance tables above (Stages 1-4) for the complete breakdown of all 22 sumcheck instances with per-instance costs.

**Component 2: Dory PCS Verification** ($(5 \times \log_2 N) + 29$ $\mathbb{G}_T$ exponentiations)

**Dory verification formula**: $(5 \times \log_2 N) + 29$ exponentiations total

**For $\log_2 N = 16$**:

**Main Dory protocol** (round-by-round computation):
- Per round: 10 GT exponentiations (4 for C update, 6 for D₁/D₂ update)
- Rounds: $\nu = \log_2(\sqrt{N}) = 8$ rounds
- **Subtotal: 80 $\mathbb{G}_T$ exponentiations**

**Stage 5 RLC** (random linear combination of commitments):
$$C_{\text{combined}} = \prod_{i=1}^{29} C_i^{\gamma_i}$$
- Combines 29 polynomial commitments via random challenges
- **Subtotal: 29 $\mathbb{G}_T$ exponentiations** (independent of N)

**Total**: 80 + 29 = **109 $\mathbb{G}_T$ exponentiations**

**Dory cost**: 109 exps × ~10M cycles/exp ≈ **~1.09B cycles**

**Component 3: Other Operations** (pairings, scalar-point multiplications, misc)

**Breakdown** (based on dev measurements):
- **Pairings**: 5 @ ~20M each ≈ **100M cycles**
  - BN254 pairing is expensive: Miller loop + final exponentiation
  - Used in Dory opening verification
- **G1/G2 scalar-point multiplications**: **~50M cycles**
  - Point additions and doublings on BN254 curves
- **Simple $\mathbb{G}_T$ group operations + misc**: **~50M cycles**
  - Group additions/multiplications (not full exponentiations)
  - Memory operations, transcript management, control flow

**Total other operations**: **~200M cycles**

---

**Total Verification Cost for Fibonacci** ($\log_2 N = 16$, max_trace_length = 65536):

| Component | Cost (cycles) | % of Total |
|-----------|---------------|------------|
| Dory $\mathbb{G}_T$ exponentiations (109 @ 10M each) | 1.09B | **85%** |
| Pairings (5 @ 20M each) | 100M | 8% |
| G1/G2 scalar-point muls | 50M | 4% |
| Sumcheck verification (22 instances) | ~20M | 1.5% |
| Simple $\mathbb{G}_T$ ops + misc | 20M | 1.5% |
| **Total** | **~1.28B** | **100%** |

**Note**: The verification cost is determined by max_trace_length (65,536 = 2^16), not the actual execution trace (1,040 cycles). For smaller max_trace_length values, costs would be lower (e.g., 2^11 → ~1.5B cycles total).

### 1.2 Why $\mathbb{G}_T$ Exponentiations Are So Expensive

Each $\mathbb{G}_T$ exponentiation costs **~10M RISC-V cycles**. This section breaks down where this cost comes from and why it's unavoidable without precompiles.

$\mathbb{G}_T = \mathbb{F}_q^{12}$: The target group of the BN254 pairing is a 12th-degree extension field over the base field $\mathbb{F}_q$ where $q$ is a 254-bit prime.

#### The Cyclotomic Exponentiation Algorithm

It computes $g^x$ where $g \in \mathbb{G}_T$ and $x$ is a 254-bit scalar. It uses the fact that elements in $\mathbb{G}_T$ lie in the cyclotomic subgroup, enabling faster squaring via the Granger-Scott algorithm (`jolt-core/src/poly/commitment/dory.rs:254` calls `cyclotomic_exp()` from arkworks).

**Step 1: Algorithm-level operations**

The binary exponentiation algorithm processes the 254-bit exponent $x$ one bit at a time from MSB to LSB:

```
For each of the 254 bits:
  - Square the current result  ← happens EVERY iteration (254 times)
  - If bit is 1: multiply by base g  ← happens only for 1-bits (~127 times on average)
```

**Operation counts**:
- **Cyclotomic squarings**: **254** (one per bit position, always performed regardless of bit value)
- **General multiplications**: Depends on Hamming weight (number of 1-bits in exponent)
  - **Worst case** (all bits = 1): 254 multiplications
  - **Average case** (50% Hamming weight): ~127 multiplications
  - **Best case** (only MSB = 1): 1 multiplication

Cyclotomic squaring is **3× faster** than general multiplication due to Granger-Scott optimization. For average case (~50% Hamming weight, consistent with random field elements):

- **254 cyclotomic squarings** (one per bit of the 254-bit exponent)
- **127 general multiplications** (one per 1-bit in the exponent)

**Step 2: $\mathbb{F}_q^{12}$ operation costs**

**2a. Cyclotomic squaring** (Granger-Scott optimization for the 254 squaring operations)

**Key insight**: Elements in $\mathbb{G}_T$ belong to the cyclotomic subgroup, which has special algebraic structure that enables faster squaring than general multiplication.

**The Granger-Scott algorithm** exploits this structure to reduce operation count:

For $x = (x_0, x_1, x_2, x_3, x_4, x_5) \in \mathbb{F}_{q^2}^6$ representation of $\mathbb{F}_{q^{12}}$:

```rust
// Compute intermediate values (3 pairs)
tmp = r0 * r1;  // 1 Fq2 mul
t0 = (r0 + r1) * (fp2_nr(r1) + r0) - tmp - fp2_nr(tmp);  // 1 Fq2 mul
t1 = tmp.double();

tmp = r2 * r3;  // 1 Fq2 mul
t2 = (r2 + r3) * (fp2_nr(r3) + r2) - tmp - fp2_nr(tmp);  // 1 Fq2 mul
t3 = tmp.double();

tmp = r4 * r5;  // 1 Fq2 mul
t4 = (r4 + r5) * (fp2_nr(r5) + r4) - tmp - fp2_nr(tmp);  // 1 Fq2 mul
t5 = tmp.double();

// Apply result formulas (only Fq2 additions, negligible cost)
z0 = 3*t0 - 2*z0, z1 = 3*t1 + 2*z1, ... (etc)
```

**Operation count**:
- **6 $\mathbb{F}_q^2$ multiplications** (vs 18 for general $\mathbb{F}_q^{12}$ multiplication)
- Granger-Scott reduces cyclotomic squaring from full $\mathbb{F}_q^{12}$ multiplication to just 6 $\mathbb{F}_q^2$ operations

**Code location**: `ark-ff-0.3.0/src/fields/models/fp12_2over3over2.rs:137-209`

**2b. Applying Karatsuba to $\mathbb{F}_q^2$ multiplications**

Both cyclotomic squaring (6 $\mathbb{F}_q^2$ muls) and general $\mathbb{F}_q^{12}$ multiplication use $\mathbb{F}_q^2$ operations as building blocks. Karatsuba optimizes these:

Each $\mathbb{F}_q^{12}$ operation requires multiplying polynomials over $\mathbb{F}_q$.

**Naive approach**: $12^2 = 144$ base field multiplications

**With Karatsuba**: ~54 base field multiplications
- For degree $n$ polynomials, use $O(n^{\log_2 3}) \approx O(n^{1.585})$ multiplications instead of $O(n^2)$
- Applied at each level of the extension tower:
  1. $\mathbb{F}_q^2$ **multiplication**: 3 $\mathbb{F}_q$ muls (vs 4 naive)
  2. $\mathbb{F}_q^6$ **multiplication**: ~18 $\mathbb{F}_q$ muls (vs 36 naive)
  3. $\mathbb{F}_q^{12}$ **general multiplication**: ~54 $\mathbb{F}_q$ muls (vs 144 naive)

**Cost per operation** (in $\mathbb{F}_q$ multiplications):
- **One cyclotomic squaring**: 6 $\mathbb{F}_q^2$ muls × 3 $\mathbb{F}_q$ muls each = **18 $\mathbb{F}_q$ muls**
- **One general $\mathbb{F}_q^{12}$ multiplication**: **54 $\mathbb{F}_q$ muls**

**Cyclotomic squaring speedup**: 54 / 18 = **3× faster** than general multiplication

**Total $\mathbb{F}_q$ multiplications for one exponentiation** (combining with Step 1 counts):
- 254 cyclotomic squarings × 18 $\mathbb{F}_q$ muls each = **4,572 $\mathbb{F}_q$ muls**
- 127 general multiplications × 54 $\mathbb{F}_q$ muls each = **6,858 $\mathbb{F}_q$ muls**
- **Total**: 4,572 + 6,858 = **11,430 $\mathbb{F}_q$ multiplications** per exponentiation

#### Converting to RISC-V Cycles

Each $\mathbb{F}_q$ multiplication operates on 254-bit integers modulo $q$ and costs **~900 cycles** in RISC-V (optimized Montgomery multiplication).

**Total cost per $\mathbb{G}_T$ exponentiation**:
```
Total = (Cyclotomic squares) × (Fq ops per square) × (cycles per Fq op)
      + (General muls) × (Fq ops per mul) × (cycles per Fq op)

      = (254 × 18 × 900) + (127 × 54 × 900)
      = 4,111,200 + 6,170,400
      = 10,281,600 cycles
      ≈ 10M cycles
```

**Breakdown by operation type**:
- **Cyclotomic squarings**: 254 × 18 × 900 = **4.1M cycles** (40%)
- **General multiplications**: 127 × 54 × 900 = **6.2M cycles** (60%)

#### Comparison: With vs Without Cyclotomic Optimization

**Without cyclotomic** (if general squaring was used):
```
Total = (254 + 127) × 54 × 900
      = 381 × 54 × 900
      = 18,514,800 cycles
      ≈ 18.5M cycles
```

**With cyclotomic** (actual implementation):
```
Total ≈ 10M cycles
```

**Speedup from cyclotomic optimization**: 18.5M / 10M = **1.85× faster** (46% reduction)

#### Why This Cost Is Unavoidable

**Fundamental constraints**:
1. **Exponent size fixed**: BN254 scalar field is 254 bits (security requirement)
2. **Field structure fixed**: $\mathbb{F}_q^{12}$ is determined by pairing-friendly curve choice
3. **Algorithm optimal**: Square-and-multiply is asymptotically optimal for exponentiation
4. **Karatsuba near-optimal**: Best known practical algorithm for polynomial multiplication

**Comparison to other curves**:
- **BLS12-381**: 381-bit exponents, $\mathbb{F}_q^{12}$ → ~50% more expensive
- **BN254**: 254-bit exponents, $\mathbb{F}_q^{12}$ → current choice (balance of security/performance)

**Why no EVM precompile**:
- **Pairings have precompile 0x08**: Cheap on-chain (~113K gas for 2-pair check)
- **$\mathbb{G}_T$ exponentiations have NO precompile**: Would cost ~500M gas (prohibitive)
- **This asymmetry**: Creates the fundamental bottleneck requiring SNARK composition

#### Cost Comparison Table

| Operation | Field | Cost (RISC-V cycles) | EVM Gas | EVM Precompile? |
|-----------|-------|---------------------|---------|-----------------|
| **$\mathbb{F}_q$ multiplication** | Base field | ~1K | ~8 | No (opcodes) |
| **$\mathbb{F}_q^{12}$ multiplication** | Extension field | ~54K | ~400K | No |
| **$\mathbb{G}_T$ exponentiation** | Target group | **~10M** | **~250M** | **No** |
| **BN254 pairing** | $\mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$ | ~20M | ~113K | **Yes (0x08)** |

**Key insight**: The lack of a $\mathbb{G}_T$ exponentiation precompile, combined with the ~10M cycle cost and high count (109), makes direct on-chain Dory verification impractical. This is why Jolt needs SNARK composition (Layer 2 recursion).

---

### 1.3 Other Dory Operations: Pairings, Scalar Multiplications, and Simple $\mathbb{G}_T$ Operations

Beyond the dominant $\mathbb{G}_T$ exponentiations, Dory verification requires additional cryptographic operations that contribute ~190M cycles (15% of total cost).

| Operation Type | Count | Cost per Op | Total Cost | % of Total |
|----------------|-------|-------------|------------|------------|
| **Pairings** ($e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$) | 5 | ~20M cycles | **100M cycles** | 8% |
| **G1/G2 Scalar-Point Multiplications** | ~50-100 | ~500K-1M cycles | **50M cycles** | 4% |
| **Sumcheck Verification** | 22 instances | ~1M cycles/instance | **20M cycles** | 2% |
| **Simple $\mathbb{G}_T$ Operations + Misc** (add/mul, transcript, memory) | many | varies | **20M cycles** | 2% |
| **Total (non-exponentiation)** | - | - | **~190M cycles** | **~15%** |

**Key observations**:
- Pairings contribute 8% of total cost (5 operations in `apply_fold_scalars` and `verify_final_pairing`)
- G1/G2 scalar multiplications contribute 4% (preparing pairing inputs, MSMs in Dory)
- Sumcheck verification contributes 2% (22 instances across 4 stages)
- Simple GT operations + misc contribute 2% (additions/multiplications in $\mathbb{F}_{q^{12}}$, transcript, memory)
- Even combined, these operations are ~5.7× cheaper than GT exponentiations (190M vs 1.09B cycles)

---

#### A. Pairings (5 @ ~20M each = 100M cycles)

**What they are**: BN254 bilinear pairings $e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$

**Where they appear**: In Dory's opening verification (external `dory` crate's `verify` function)

**Code location**: [`jolt-core/src/poly/commitment/dory.rs:700-704`](../jolt-core/src/poly/commitment/dory.rs#L700-L704)
```rust
let ml_result = Bn254::multi_miller_loop_ref(g1_chunk.iter(), g2_chunk.iter()).0;
let pairing_result = Bn254::final_exponentiation(MillerLoopOutput(ml_result))
```

**Cost breakdown per pairing**:
1. **Miller loop**: Iterative computation over elliptic curve points
   - Processes 254-bit scalar (BN254 embedding degree)
   - Computes line functions and accumulates in $\mathbb{F}_{q^{12}}$
   - Cost: ~12M cycles
2. **Final exponentiation**: Maps Miller loop output to $\mathbb{G}_T$
   - Exponentiation in $\mathbb{F}_{q^{12}}$ by specific exponent $(q^{12} - 1)/r$
   - Uses cyclotomic subgroup structure for optimization
   - Cost: ~8M cycles
3. **Total**: ~20M cycles per pairing

**Why 5 pairings?**:
- Dory's verification algorithm performs pairing checks to validate inner product structure
- Fixed count (independent of trace size N)
- Used to verify commitments satisfy Dory's recursive structure

**EVM note**: Pairings have precompile 0x08 (bn256Pairing), making them cheap on-chain (~113K gas for 2-pair check). This is why pairings are not a bottleneck for on-chain deployment.

#### B. G1/G2 Scalar-Point Multiplications (50M cycles)

**What they are**: Elliptic curve scalar multiplications on BN254's G1 (base field curve) and G2 (twisted curve)

**Operations**:
- **Scalar multiplication**: Computing $[k]P$ for scalar $k$ and point $P$
- **Multi-scalar multiplication (MSM)**: Computing $\sum_{i} [k_i]P_i$ efficiently using Pippenger's algorithm

**Where they appear**:
- **Preparing pairing inputs**: Converting commitments and challenges to prepared form for pairings
- **MSMs in Dory verification**: Linear combinations of elliptic curve points
- **Code location**: Throughout Dory commitment scheme, e.g., [`jolt-core/src/poly/commitment/dory.rs:1299-1302`](../jolt-core/src/poly/commitment/dory.rs#L1299-L1302)

**Cost breakdown**:
- **Single scalar-point multiplication**: ~500K-1M cycles
  - Uses double-and-add algorithm (similar structure to square-and-multiply)
  - 254 doublings + ~127 additions (average case, 50% Hamming weight)
  - Point operations in 256-bit base field (G1) or 512-bit extension field (G2)
- **MSMs with Pippenger**: Amortized cost ~50% of naive approach
  - Batches multiple scalar multiplications
  - Exploits common scalar bits across multiple points
- **Total across Dory verification**: ~50M cycles

**Why needed?**: Dory verification requires computing linear combinations of commitment points (in G1/G2) using verifier challenges before performing pairing checks.

#### C. Simple $\mathbb{G}_T$ Group Operations + Misc (20M cycles)

**What they are**: Various operations that are cheap compared to exponentiations

**$\mathbb{G}_T$ additions/multiplications** (NOT full exponentiations):
- **Single $\mathbb{F}_{q^{12}}$ multiplication**: ~54 base field operations × ~1000 cycles = ~54K cycles
- **Single $\mathbb{F}_{q^{12}}$ addition**: ~12 base field additions × ~100 cycles = ~1.2K cycles
- **Used for**: Accumulating products, combining intermediate values
- **Code location**: [`jolt-core/src/poly/commitment/dory.rs:1259-1266`](../jolt-core/src/poly/commitment/dory.rs#L1259-L1266) (combining commitments)

**Miscellaneous operations**:
- **Transcript management**: Fiat-Shamir hashing (SHA256) to generate verifier challenges
  - Cost: ~100-200 cycles per hash operation
  - Multiple hashes throughout verification (appending messages, sampling challenges)
- **Memory operations**: Loading/storing field elements, commitment data
  - Cost: ~10-50 cycles per memory access
  - Thousands of accesses throughout verification
- **Control flow**: Loops, conditionals, function calls
  - Negligible cost compared to cryptographic operations

**Total estimate**: ~20M cycles for all simple operations combined

#### Summary: The Cost Hierarchy

| Operation Type | Single Op Cost | Count | Total Cost | % of Total |
|----------------|---------------|-------|------------|------------|
| **$\mathbb{G}_T$ exponentiations** | ~10M cycles | 109 | **1.09B** | **85%** |
| **Pairings** | ~20M cycles | 5 | **100M** | **8%** |
| **G1/G2 scalar-point muls** | ~500K-1M cycles | ~50-100 | **50M** | **4%** |
| **Simple $\mathbb{G}_T$ ops + misc** | ~1K-54K cycles | ~hundreds | **20M** | **1.5%** |
| **Sumcheck verification** | (see section 1.1) | 22 instances | **~20M** | **1.5%** |
| **Total** | | | **~1.28B** | **100%** |

**Key insight**: The cost hierarchy explains why recursion targets exponentiations:
- **Exponentiations (85%)**: Accept as hints + prove correct via SZ-Check → saves 1.09B cycles
- **Pairings (8%)**: Can use EVM precompile 0x08 on-chain → not a bottleneck
- **Everything else (7%)**: Already cheap relative to exponentiations → no optimization needed

---

### 1.4 Exponentiations Count

This section explains how Dory verification's exponentiation count is computed, based on the actual implementation in the `dory` crate.

$$\boxed{\text{Total Exponentiations} = 29 + (5 \times \log_2 N)}$$

**Breakdown**:

- **Stage 5 RLC** (first): 29 exponentiations (fixed, independent of N)
- **Main Dory opening** (second): $5 \times \log_2 N$ exponentiations (scales logarithmically with trace size)

**For typical program** ($\log_2 N = 16$):

- Total: $29 + (5 \times 16) = 29 + 80 = 109$ exponentiations

**Cost breakdown by component**:

- Stage 5 RLC: 29 exps (27%)
- Main Dory: 80 exps (73%)

---

#### Part A: Stage 5 RLC (29 exponentiations)

**Random Linear Combination**: In stage 5, Dory batches multiple polynomial openings via random linear combination (source: `jolt-core/src/poly/opening_proof.rs`, `prove_batch()` method):

$$C_{\text{combined}} = \prod_{i=1}^{29} C_i^{\gamma_i}$$

Where $\gamma_i$ are random challenges from the verifier and $C_i$ are Dory commitments ($\mathbb{G}_T$ elements).

**The 29 polynomials** (from `jolt-core/src/zkvm/witness.rs`, lines 20-89):

- 8 fixed polynomials: `LeftInstructionInput`, `RightInstructionInput`, `WriteLookupOutputToRD`, `WritePCtoRD`, `ShouldBranch`, `ShouldJump`, `RdInc`, `RamInc`
- 16 instruction lookup polynomials: `InstructionRa(0)` through `InstructionRa(15)` (decomposition parameter D=16)
- 3 RAM polynomials: `RamRa(0)` through `RamRa(2)` (for typical programs with d=3)
- 2 bytecode polynomials: `BytecodeRa(0)` through `BytecodeRa(1)` (for typical programs with d=2)

> **TODO**: Verify what the `d` parameter represents in RAM and bytecode polynomial counts.

**Cost**: Each $C_i^{\gamma_i}$ is a $\mathbb{G}_T$ exponentiation (cyclotomic exponentiation with 254-bit exponent, implemented via `scale_gt()` in `dory` crate).

**Important**: This count is **independent of trace size N** - determined by program structure, not execution length.

---

#### Part B: Main Dory Opening (5 × log₂ N exponentiations)

Verifies the **single combined commitment** $C_{\text{combined}}$ from Part A using a recursive halving protocol over $\nu = \log_2(\sqrt{N}) = \log_2 N / 2$ rounds.

**Per-Round Cost** (source: `dory` crate, `src/core/interactive_protocol.rs`):

Each round, the verifier computes updates immediately using prover's messages and random challenges $\alpha_i, \beta_i$:

1. **`dory_reduce_verify_update_c()`** (lines 411-451): 4 GT exponentiations
   - $\beta_i \cdot D_2^{(i)}$, $\beta_i^{-1} \cdot D_1^{(i)}$, $\alpha_i \cdot C_+^{(i)}$, $\alpha_i^{-1} \cdot C_-^{(i)}$
   - Each computed via `scale_gt_with_offload()`

2. **`dory_reduce_verify_update_ds()`** (lines 457-511): 6 GT exponentiations
   - For $D_1$: $\alpha_i \cdot D_{1L}^{(i)}$, $\alpha_i\beta_i \cdot \Delta_{1L}^{(i)}$, $\beta_i \cdot \Delta_{1R}^{(i)}$
   - For $D_2$: $\alpha_i^{-1} \cdot D_{2L}^{(i)}$, $\alpha_i^{-1}\beta_i^{-1} \cdot \Delta_{2L}^{(i)}$, $\beta_i^{-1} \cdot \Delta_{2R}^{(i)}$
   - Each computed via `scale_gt_with_offload()`

**Total per round**: **10 GT exponentiations**

**Formula**:

$$\text{Total} = 10 \times \nu = 10 \times \log_2(\sqrt{N}) = 10 \times \frac{\log_2 N}{2} = \boxed{5 \times \log_2 N}$$

**Example** ($\log_2 N = 16$): $10 \times 8 = 80$ exponentiations

### 1.5 Scaling Analysis: Verification Costs Across Program Sizes

| Program Size | Trace Length | Padded $N$ | $\log_2 N$ | Sumcheck Cost | Stage 5 RLC | Main Dory Opening | **Total Exps** | **Total Cost** | Use Case |
|-------------|--------------|----------|--------|----------|-------------|-------------------|----------------|----------------|----------|
| Tiny | ~500 | 1,024 | 10 | ~6M | 29 | 50 | **79** | **~1.0B** | Minimal programs |
| Small (fib) | ~1,000 | 2,048 | 11 | ~7-8M | 29 | 55 | **84** | **~1.05B** | Fibonacci(50) |
| Medium | ~4,000 | 8,192 | 13 | ~12M | 29 | 65 | **94** | **~1.15B** | Simple crypto |
| **Typical** | **~32,000** | **65,536** | **16** | **~20-25M** | **29** | **80** | **109** | **~1.30B** | **Most examples** |
| Large | ~500,000 | 1,048,576 | 20 | ~30-35M | 29 | 100 | **129** | **~1.52B** | Complex programs |
| Maximum | ~8,000,000 | 16,777,216 | 24 | ~40-45M | 29 | 120 | **149** | **~1.73B** | Recursion limit |

**Key observations** (columns ordered by execution):
- Verification cost scales **logarithmically** with program size
- Stage 5 RLC exponentiations are **fixed at 29** (independent of trace size) - happens **first**
- Main Dory opening exponentiations scale as $5 \times \log_2 N$ (linear in log N) - happens **second**
- For small traces, RLC dominates (29/79 = 37%)
- For large traces, main Dory opening dominates (120/149 = 81%)
- 10× larger program → only +20 exponentiations (~200M extra cycles)

---

## Part 2: The Scalability Problem

### 2.1 On-Chain Verification Target

**EVM gas limit**: ~30M gas per block
**Efficient EVM verification target**: ≤30M RISC-V cycles (roughly 1 cycle ≈ 1 gas for optimized contracts)

**Current Jolt verification cost**: ~1.3B cycles (typical programs)

**Cost ratio**: $\frac{1{,}300{,}000{,}000}{30{,}000{,}000} \approx 43\times$ **too expensive**

---

### 2.2 Why Jolt is Not Directly Deployable On-Chain

**Problem breakdown**:

1. **$\mathbb{G}_T$ exponentiations dominate** (~85% of cost)
   - 109 exponentiations × 10M cycles/exp = 1.09B cycles (for $\log_2 N = 16$)
     - 80 from main Dory protocol ($5 \times \log_2 N$, actual implementation)
     - 29 from stage 5 RLC (batched commitments)
   - No EVM precompile exists for $\mathbb{G}_T$ exponentiation
   - Would require implementing 254-bit exponentiation in $\mathbb{F}_q^{12}$ in EVM
   - Each $\mathbb{F}_q^{12}$ multiplication: ~54 base field operations
   - Total: ~11,430 base field ops per exponentiation → prohibitively expensive

2. **Sumcheck verification adds overhead** (~1% of cost)
   - 22 sumcheck instances across 4 stages
   - ~242 rounds for log₂ N=11, ~352 rounds for log₂ N=16
   - Each round: polynomial evaluation + field operations
   - Relatively small compared to Dory cost (~20-25M cycles)

3. **Other operations** (~5% of cost)
   - Pairings (4-5 per proof): Cheap on-chain via precompile 0x08
   - Transcript management, field arithmetic: Relatively cheap
   - Not a bottleneck

**Fundamental issue**: Dory's transparency (no trusted setup) comes at the cost of expensive verification operations that don't map well to EVM's precompile set.

---

### 2.3 The Need for SNARK Composition

**Core insight**: We need a proof system with:
1. **Fast on-chain verification** (≤30M cycles)
2. **Ability to verify Jolt proofs** (wrap the expensive Dory verifier)

**Solution: SNARK Composition / Recursion**

**High-level approach**:
1. Generate Jolt proof (uses Dory PCS with BN254)
2. Run Jolt verifier **inside a SNARK circuit** (e.g., Grumpkin/Plonky3)
   - The $\mathbb{G}_T$ exponentiations become R1CS constraints
   - The 22 sumchecks become constraint checks
   - Total: ~2.0B RISC-V cycles → ~millions of R1CS constraints
3. Generate recursive SNARK proof proving "I correctly verified the Jolt proof"
4. Deploy recursive SNARK verifier on-chain
   - Verifies in ~30M cycles (using efficient curve operations)
   - Uses EVM-friendly precompiles (pairing, scalar mult)

**Trade-off**:
- **Prover time increases** (2-layer proving: Jolt + recursive SNARK)
- **Verifier time decreases ~43×** (1.28B → 30M cycles)
- Enables economical on-chain deployment

---

## Summary and Next Steps

### Key Formulas

1. **Dory PCS verification**: $(5 \times \log_2 N) + 29$ $\mathbb{G}_T$ exponentiations
   - Main protocol: $5 \times \log_2 N$ (round-by-round computation, 10 exps/round over log₂(√N) rounds)
   - Stage 5 RLC: 29 (batched commitments, independent of N)
   - ~10M RISC-V cycles per exponentiation

2. **Sumcheck verification**: 22 sumcheck instances across 4 stages
   - Stage 1: 1 instance, Stage 2: 6 instances, Stage 3: 8 instances, Stage 4: 7 instances
   - ~$22 \times \log_2 N$ total rounds (22 instances × $\log_2 N$ rounds each)
   - Per-round cost: ~150 cycles/field-op average (BN254 field arithmetic)
   - Each round (standard sumcheck with Jolt's compression optimization):
     - Receive compressed polynomial (degree-d coefficients, linear term omitted)
     - Decompress + evaluate at challenge: ~4-11 field ops (degree-1: ~4 ops, degree-2: ~7 ops, degree-3: ~11 ops)
     - Transcript update (SHA256 + sampling): ~100-200 cycles
     - `expected_output_claim`: ~10-400 field ops (simple: 10-50, moderate: 50-150, complex: 100-400)
   - Total field ops per round: ~20-500 (average ~100-150)
   - Total: ~7-8M cycles (log₂ N=11), ~20-25M cycles (log₂ N=16)

3. **Total verification cost**:
   $$\text{Cost}(N) = [(5 \times \log_2 N + 29) \times 10\text{M}] + (22 \times \log_2 N \times 150 \times 150) + 190\text{M}$$
   $$\approx [50\text{M} \times \log_2 N + 290\text{M}] + [0.5\text{M} \times \log_2 N] + 190\text{M}$$
   $$\approx 50.5\text{M} \times \log_2 N + 480\text{M} \text{ cycles}$$

4. **For typical program** ($\log_2 N = 16$):
   $$\text{Cost} \approx 50.5\text{M} \times 16 + 480\text{M} \approx 1.29\text{B cycles}$$

   **Breakdown**:
   - Dory exponentiations: 109 × 10M = **1.09B** (85%)
   - Sumcheck: 22 × 16 × ~60K = **~21M** (~2%)
   - Other operations: **~190M** (15%)

---

### The Path Forward: SNARK Composition

**Current state**: Jolt verification costs ~1.28B cycles (~43× too expensive for on-chain)

**Required**: ≤30M cycles for economical on-chain deployment

**Solution architecture**:
1. **Layer 1: Jolt proof** (transparent, no trusted setup)
   - Proves RISC-V execution with Dory PCS
   - Fast prover, expensive verifier

2. **Layer 2: Recursive proof** (wraps Jolt verifier)
   - Proves "I correctly verified a Jolt proof"
   - **Avoids computing** expensive $\mathbb{G}_T$ exponentiations (accepts as "hints")
   - **Proves hints correct** via SZ-Check + Hyrax commitment over Grumpkin
   - Fast verifier (~30M cycles), expensive prover

3. **On-chain: Deploy Layer 2 verifier**
   - Uses EVM precompiles (pairing, scalar mult)
   - Achieves ~43× verification speedup

**For detailed exploration of SNARK composition strategies, curve choices, and implementation considerations, see**: [SNARK_Composition_and_Recursion.md](SNARK_Composition_and_Recursion.md)

---

## References

- **Complete verifier mathematics**: [03_Verifier_Mathematics_and_Code.md](03_Verifier_Mathematics_and_Code.md)
- **Jolt theory and Dory integration**: [01_Jolt_Theory_Enhanced.md](01_Jolt_Theory_Enhanced.md) Section 3.4
- **SNARK composition strategies**: [SNARK_Composition_and_Recursion.md](SNARK_Composition_and_Recursion.md)
- **Dory paper**: "Dory: Efficient, Transparent arguments for Generalised Inner Products and Polynomial Commitments" (Heath et al., 2021)
- **Implementation**: `jolt-core/src/poly/commitment/dory.rs`

---

## Key Takeaways

1. **Jolt verification cost**: ~1.28B cycles for typical programs ($\log_2 N = 16$)
   - **~85% from Dory**: 109 $\mathbb{G}_T$ exponentiations @ 10M cycles each (no EVM precompile)
     - 80 from main protocol ($5 \times \log_2 N$ exponentiations)
     - 29 from stage 5 RLC (fixed)
   - **~2% from sumchecks**: 22 instances × 16 rounds × ~60K cycles/round avg = ~20M cycles
   - **~15% from other ops**: Pairings (5 @ 20M = 100M), scalar-point muls (50M), simple G_T ops + misc (40M)

2. **Scalability**: Verification scales **logarithmically** with program size
   - 10× larger program → only +20 exponentiations (~200M extra cycles)
   - Main protocol scales as $5 \times \log_2 N$ (logarithmic)
   - Stage 5 RLC is constant at 29 (independent of N)
   - For small traces, RLC dominates; for large traces, main protocol dominates

3. **On-chain deployment**: **~43× too expensive** without optimization
   - Current cost: ~1.28B cycles
   - Target: ≤30M cycles (EVM gas limit)
   - Requires SNARK composition/recursion

4. **Why cyclotomic squaring matters**: $\mathbb{G}_T$ exponentiation optimization
   - Without: 18.5M cycles per exponentiation (general squaring)
   - With cyclotomic: ~10M cycles (Granger-Scott algorithm, 1.85× speedup)
   - Reduces squaring cost from 54 to 18 base field muls (3× faster)
   - Critical for keeping $\mathbb{G}_T$ exponentiation tractable

5. **The transparency trade-off**: Dory avoids trusted setup but incurs verification cost
   - No ceremony, updateable, quantum-resistant
   - But: Expensive $\mathbb{G}_T$ operations with no EVM precompile
   - Solution: Wrap in recursive SNARK for on-chain deployment
