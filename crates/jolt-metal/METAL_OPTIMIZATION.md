# Metal Kernel Optimization Plan

Baseline measurements on M1 Pro (14 GPU cores, 200 GB/s bandwidth).

## Sumcheck Hot Loop

Each sumcheck round over `n` elements:
1. **Reduce**: For each pair `(lo[i], hi[i])`, evaluate composition `P(t)` at `D` grid points → `D` field elements (accumulated across all pairs)
2. **Bind**: Halve each buffer via interpolation: `buf[i] = lo + r·(hi - lo)`

Repeats for `log₂(n)` rounds. Round 1 dominates (buffer halves each round).

## Current Roofline Analysis

### Cost model per `fr_mul` (CIOS Montgomery)

8 outer rounds × (8 `mul` + 8 `mulhi` + ~12 add/compare) ≈ **~224 ALU ops**.
At 14 cores × 128 ALUs × 1.3 GHz = 2.33 TIOPS peak.
Theoretical: **~96ns per fr_mul per thread** (single-threaded) or **~0.1ns throughput** (fully parallel).

### Per-pair costs

| Kernel | Reads/pair | Compute/pair | Balance |
|--------|-----------|-------------|---------|
| ProductSum D=4 P=1 | 288 B (4×2×32 + weight) | ~7 fr_mul + 8 fr_add | Balanced |
| ProductSum D=8 P=1 | 544 B (8×2×32 + weight) | ~22 fr_mul + 24 fr_add | Compute-bound |
| EqProduct | 128 B + weight | 3 fr_mul + 3 fr_add | Bandwidth-bound |
| HammingBooleanity | 128 B + weight | 5 fr_mul + 7 fr_add | Balanced |

### Current performance (post-OPT-7/OPT-10)

| Op | Size | Metal | CPU | Speedup |
|----|------|-------|-----|---------|
| reduce D=4 | 2^14 | 1.09ms | 569μs | 0.52x (CPU) |
| reduce D=4 | 2^20 | 17.8ms | 28.7ms | **1.61x** |
| reduce D=4 unw | 2^20 | 15.5ms | 27.1ms | **1.75x** |
| reduce D=8 | 2^14 | 2.02ms | 1.82ms | 0.90x (CPU) |
| reduce D=8 | 2^20 | 64.2ms | 89.3ms | **1.39x** |
| reduce D=8 unw | 2^20 | 52.7ms | 91.8ms | **1.74x** |
| sumcheck D=4 L2H 2^20 | 90ms | 93ms | 0.97x |
| sumcheck D=4 H2L 2^20 | 60ms | 88ms | **1.47x** |
| sumcheck D=8 L2H 2^20 | 168ms | 240ms | **1.43x** |
| sumcheck D=8 H2L 2^20 | 161ms | 243ms | **1.51x** |

D=8 improved dramatically from Toom-Cook (OPT-7): 30 fr_mul vs 58 naive.
D=4 uses naive weight-folded approach (Toom-Cook regressed due to lost weight folding).
H2L in-place binding avoids buffer allocation — massive win for D=4 sumcheck (1.47× vs 0.97×).

## Register Pressure Problem

BN254 Fr = 8 × u32 registers per element.

**Current D=8 kernel register usage:**
- `lo[8]`: 64 regs
- `hi[8]`: 64 regs
- `diff[8]`: 64 regs
- `cur[8]`: 64 regs
- `evals[8]`: 64 regs
- `wide_acc[8]`: 8 × 18 = 144 regs
- CIOS temporaries: ~30 regs
- Total: **~500 registers per thread**

M1 GPU register file: 32 KB = 8192 × u32 per EU.
At 500 regs/thread → **~16 threads/EU** → **0.5 simdgroups/EU**.
Need 4+ simdgroups for good latency hiding → **running at ~12% occupancy**.

D=4 uses ~280 registers → ~29 threads → ~0.9 simdgroups. Still low, but 2x better.

## Optimizations (priority order)

### OPT-1: Simdgroup WideAcc reduction (before acc_reduce) ✅

**Impact: 10-15%** | Effort: Medium | **Status: IMPLEMENTED**

Simd_shuffle the 18-limb WideAcc *before* `acc_reduce`. Only lane 0 of each simdgroup calls the expensive `acc_reduce` (8 CIOS rounds). Saves 248 × ~3500 ops = ~865K ops per threadgroup.

**Measured results:**

| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| reduce D=4 2^14 | 985μs | 947μs | -4% |
| reduce D=4 2^20 | 18.6ms | 18.1ms | -3% |
| reduce D=8 2^14 | 3.7ms | 2.37ms | **-36%** |
| reduce D=8 2^20 | 86.8ms | 85.8ms | -1% |
| sumcheck D=8 2^20 | ~221ms | 210ms | -5% |

Big win at small sizes (acc_reduce dominates). Marginal at large sizes (grid-stride loop dominates).

### OPT-2: Streaming pair loading (D≥8 occupancy fix) ❌

**Impact: Negative (−6%)** | Effort: Medium | **Status: ABANDONED**

**Hypothesis**: Loading `lo[D]` + `hi[D]` + `diff[D]` + `cur[D]` all at once requires ~256 registers (D=8), killing occupancy. Split-pass streaming (processing 2 grid points per pass, re-reading inputs 4×) would drop register pressure from ~500 to ~130, improving occupancy ~4×.

**Implementation**: `generate_split_pass_reduce_kernel` in compiler.rs — complete, correct, tested. Enabled at `SPLIT_PASS_THRESHOLD >= 8`.

**Result**: −6% regression (90.8ms vs 85.8ms at D=8 2^20). The 4× bandwidth re-read cost outweighs the occupancy improvement. D=8 is more compute-bound than bandwidth-bound — the extra memory traffic is not free.

**Kept**: The split-pass codegen infrastructure remains in compiler.rs (threshold set to 1024, effectively disabled). The barrier correctness fix discovered during this work was applied to both single-pass and split-pass tree reductions.

**Bug found & fixed**: `threadgroup_barrier` inside `if (lid < num_simdgroups)` is undefined behavior in Metal (not all threads reach the barrier). This caused incorrect results when n_pairs > 32 (2+ simdgroups active). Fix: remove the outer `if` guard, let all 256 threads reach every barrier. Applied to both single-pass and split-pass kernels.

### OPT-3: Deferred fr_reduce in eval body ✅

**Impact: -4% D=8 weighted, -7% D=4 unweighted** | Effort: Low | **Status: IMPLEMENTED**

`fr_mul` calls `fr_reduce` after every multiply (conditional subtraction, ~40 ops). Intermediate products are immediately reduced only to be multiplied again.

**Key insight**: CIOS with unreduced inputs (< 2r) still produces output < 2r, because BN254's 4r²/R < 2r. So `fr_mul_unreduced` chains are safe. Only reduce before `fr_add` (which assumes [0, r) inputs).

**Implementation**: Added `fr_mul_unreduced` to `bn254_fr.metal` (same CIOS, skips final `fr_reduce`). Refactored `fr_mul` to call `fr_mul_unreduced` + `fr_reduce`. In `emit_product_sum`, product chains use `fr_mul_unreduced`; explicit `fr_reduce` before `fr_add(sum, prod)`. Saves `(D-2)` reduces per product group per grid point.

**Measured results (post-OPT-1 → post-OPT-3):**

| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| reduce D=8 2^20 weighted | 85.8ms | 82.6ms | **-3.7%** |
| reduce D=8 2^20 unweighted | 75.8ms | 76.2ms | ~0% |
| reduce D=4 2^20 unweighted | 15.9ms | 14.8ms | **-6.9%** |

Modest improvement. The unweighted D=8 kernel is more bandwidth-bound than compute-bound, so removing compute doesn't help there. The WideAcc accumulator (acc_fmadd/acc_add_fr) also accepts unreduced values safely — no reduce needed before accumulation.

### OPT-4: Multi-round command batching

**Impact: 20-30% at medium sizes** | Effort: Medium

Each `pairwise_reduce` + `interpolate_pairs_batch` creates 2 separate command buffers (encode, commit, wait, encode, commit, wait). Dispatch overhead is ~20-80μs per command buffer.

**Fix**: Add a `reduce_and_bind_round` method that encodes both reduce and bind into a single command buffer. For a full sumcheck (log₂(n) rounds), batch all rounds:

```
[reduce_1 | bind_1 | reduce_2 | bind_2 | ... | reduce_k | bind_k]
```

One commit + wait for the entire sumcheck. Saves `2 × log₂(n) - 1` command buffer round-trips.

For n=2^20, that's ~39 command buffers → 1. At 50μs each = **~2ms saved**.

### OPT-5: Karatsuba acc_fmadd ✅

**Impact: -26% at small sizes (2^14), ~0% at 2^20** | Effort: High | **Status: IMPLEMENTED**

Replaced schoolbook 8×8 (64 mul+mulhi) with Karatsuba on 4-limb halves: 3 × (4×4) = 48 mul+mulhi (25% reduction). The cross-term P1 = (aL+aH)*(bL+bH) - aL*bL - aH*bH is always non-negative (equals aH*bL + aL*bH).

**Implementation details:**
- `schoolbook_4x4` / `schoolbook_4x4_wide` helpers in wide_accumulator.metal
- `acc_add_limbs` helper for accumulating N-limb arrays at arbitrary WideAcc offsets
- Carry bits from 4-limb sums (aL+aH, bL+bH) handled branchlessly via bitmask
- Operation sequence minimizes peak register pressure (~17 temp regs vs ~26 for schoolbook)
- P0 and P2 computed, subtracted from Pm, accumulated, and freed sequentially

**Measured results:**

| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| reduce D=4 2^14 weighted | 998μs | 735μs | **-26%** |
| reduce D=4 2^20 weighted | 18.5ms | 19.1ms | ~0% |
| reduce D=8 2^14 weighted | 2.35ms | 2.30ms | -2% |
| reduce D=8 2^20 weighted | 82.6ms | 83.2ms | ~0% |

Big win at small sizes where acc_fmadd dominates. At 2^20, the eval body's fr_mul chain dominates accumulation cost, so fewer acc_fmadd muls don't help. Later sumcheck rounds (small buffers) benefit most.

### OPT-6: fr_sqr specialization

**Impact: ~30% faster for squarings** | Effort: Medium

CIOS squaring exploits `a[i]*a[j] = a[j]*a[i]` symmetry, halving cross-terms. Used in Toom-Cook grid evaluation when `t² * x` patterns appear. Not currently exploited in the eval body.

### OPT-7: Toom-Cook kernel codegen (D=8) ✅

**Impact: -20% weighted, -29% unweighted at 2^20** | Effort: High | **Status: IMPLEMENTED (D=8 only)**

Balanced binary splitting for ProductSum evaluation: O(D log D) multiplies instead of O(D²).

**Algorithm (D=8, P=1):**
1. Split 8 inputs into two groups of 4
2. Each group: two sub-pairs evaluated at {1,2,∞} via `emit_eval_linear_prod_2` (3 fr_mul each)
3. Extrapolate both sub-pairs to {3,4} via `emit_ex2` (adds only)
4. Point-wise multiply → 5 fr_mul per half-product (11 total per half)
5. Extrapolate half-products to {5,6,7} via `emit_ex4_2` + `emit_ex4` (adds only)
6. Final point-wise multiply at 8 points → 8 fr_mul
7. **Total: 30 fr_mul** (vs 56 naive fr_mul_unreduced)

**D=4 not used:** Toom-Cook D=4 (10 fr_mul) regressed because losing weight folding adds 4 `acc_fmadd` (192 widening muls) that outweigh the 4 fewer `fr_mul`. Naive weight-folded D=4 (14 fr_mul, `acc_add_fr`) is faster.

**Measured results (D=8 2^20):**

| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| weighted | 80 ms | 64 ms | **-20%** |
| unweighted | 74 ms | 53 ms | **-29%** |
| sumcheck_round 4r | 202 ms | 167 ms | **-17%** |

### OPT-10: Pre-allocated reduce buffers ✅

**Impact: eliminates per-dispatch allocation** | Effort: Low | **Status: IMPLEMENTED**

Pre-allocate `reduce_partials` (256 KB) and `reduce_params` (16 B) on `MetalBackend` construction. `dispatch_reduce` and `reduce` write params via `contents()` pointer instead of creating new buffers. Eliminates 2 `MTLBuffer` allocations per dispatch.

### OPT-11: H2L in-place binding for sumcheck rounds ✅

**Impact: 4-33% sumcheck round improvement** | Effort: Low | **Status: IMPLEMENTED (benchmark + backend)**

The L2H interpolation path allocates a new output buffer per polynomial per round (can't write in-place due to GPU race condition). The H2L path writes in-place (`buf[i] = lo + r*(hi-lo)` where `lo=buf[i]`, `hi=buf[i+half]`) — zero allocation.

**Measured sumcheck round improvement (2^20, 4 rounds):**

| Benchmark | L2H Metal | H2L Metal | Improvement |
|-----------|-----------|-----------|-------------|
| D=4 | 90 ms | 60 ms | **33%** |
| D=8 | 168 ms | 161 ms | **4%** |

The D=4 improvement is massive because reduce is fast and bind allocation dominates. D=8 improve is modest since reduce dominates.

**Prover integration**: Switching from L2H to H2L requires setting `BindingOrder::HighToLow` in the stage pipeline and `reverse_challenges: false` in `StageDescriptor`. The mathematical result is identical — only the memory layout convention changes.

Also optimized H2L batch path to reuse pre-allocated `reduce_params` buffer (zero per-dispatch allocation for uniform-length batches).

## Test Compilation Speedup: `CompileMode::FastCompile`

**Problem**: D=8 kernel generates ~5000 lines of MSL across 5 variants. Apple's LLVM
spends ~3 minutes inlining 56 `fr_mul` calls (each expanding to ~224 ALU ops) into the
eval body. Tests were taking 183+ seconds due to shader compilation, not GPU execution.

**Solution**: `#define FR_NOINLINE` marks `fr_mul`, `fr_add`, `fr_sub`, `acc_fmadd`,
`acc_reduce` etc. with `__attribute__((noinline))`. LLVM compiles each function once as
a call target instead of inlining everywhere.

| Test | Before | After | Speedup |
|------|--------|-------|---------|
| `pairwise_reduce_product_sum_d8_large` | 183s | 6.7s | **27x** |
| Full suite (47 tests) | ~210s+ | 89s | **2.4x+** |

Usage: `MetalBackend::new_fast_compile()` enables noinline mode. Tests use this.
Benchmarks and production use `MetalBackend::new()` (full inlining).

## Validation Strategy

Quick iteration benchmarks (see `benches/metal_vs_cpu.rs`):
- `sample_size(10)` everywhere, 2 sizes only (2^14, 2^20)
- Run specific bench: `cargo bench -p jolt-metal --bench metal_vs_cpu -- <pattern>`
- E.g., `cargo bench -p jolt-metal --bench metal_vs_cpu -- pairwise_reduce`

After each optimization, validate:
1. **Correctness**: `cargo nextest run -p jolt-metal --cargo-quiet` (unit tests)
2. **Performance**: `cargo bench -p jolt-metal --bench metal_vs_cpu -- pairwise_reduce`
3. **E2E**: `cargo nextest run -p jolt-zkvm --test hybrid_backend --cargo-quiet`

## Implementation Order

1. ✅ OPT-1 (simdgroup WideAcc) — standalone shader change, low risk
2. ❌ OPT-2 (streaming for D≥8) — bandwidth cost outweighed occupancy gain
3. ✅ OPT-3 (deferred fr_reduce) — small shader change, compounds with everything
4. ✅ OPT-5 (Karatsuba acc_fmadd) — 25% fewer muls, big win at small sizes
5. ✅ OPT-7 (Toom-Cook D=8) — 30 vs 56 muls, 20-29% improvement
6. ✅ OPT-10 (pre-allocated buffers) — eliminates per-dispatch allocation overhead
7. ✅ OPT-11 (H2L in-place binding) — 4-33% sumcheck round improvement
8. OPT-4 (command batching) — needs trait/API change, remaining low-hanging fruit
9. OPT-6 (fr_sqr) — deferred, not used in ProductSum hot path
