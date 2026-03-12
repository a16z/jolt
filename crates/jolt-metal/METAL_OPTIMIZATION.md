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

### Baseline performance

| Op | Size | Metal | CPU | Speedup |
|----|------|-------|-----|---------|
| reduce D=4 | 2^20 | 18.6ms | 76.8ms | **4.1x** |
| reduce D=4 | 2^22 | 66.9ms | 394ms | **5.9x** |
| reduce D=8 | 2^20 | 86.8ms | 120.9ms | **1.4x** |
| reduce D=8 | 2^22 | 325ms | 617ms | **1.9x** |
| interpolate | 2^22 | 18.0ms | 14.7ms | **0.8x** (CPU wins) |
| sumcheck round D=4 | 2^22 | 112ms | 637ms | **5.7x** |

D=8 is suspiciously slow (1.9x vs 5.9x for D=4). Root cause: **register pressure**.

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

### OPT-1: Simdgroup WideAcc reduction (before acc_reduce)

**Impact: 10-15%** | Effort: Medium

Currently each of 256 threads calls `acc_reduce` (8 CIOS rounds ≈ 3500 ops), then simd_shuffles the 8-limb Fr result. 248 of those `acc_reduce` calls are wasted.

**Fix**: Simd_shuffle the 18-limb WideAcc *before* `acc_reduce`. Only lane 0 of each simdgroup calls `acc_reduce`.

Saves: 248 × ~3500 ops = ~868K ops per threadgroup.
Costs: 32 threads × 18 limbs × 5 rounds = 2880 shuffle ops.
**Net: ~865K ops saved per threadgroup.**

### OPT-2: Streaming pair loading (D≥8 occupancy fix)

**Impact: 2-4x for D=8** | Effort: Medium

**Root cause**: Loading `lo[D]` + `hi[D]` + `diff[D]` + `cur[D]` all at once requires ~256 registers (D=8). This kills occupancy.

**Fix**: Load one pair at a time, compute incrementally:

```metal
// For each grid point t:
Fr prod = fr_one();
for (int k = 0; k < 8; k++) {
    Fr lo_k = input_k[2*i];
    Fr hi_k = input_k[2*i+1];
    Fr val = fr_add(lo_k, fr_mul(t_scaled, fr_sub(hi_k, lo_k)));
    prod = fr_mul(prod, val);
}
```

Register usage drops from ~500 to ~100 (just prod + lo_k + hi_k + val + acc + CIOS temps).
Expected occupancy: ~80 threads/EU → 2.5 simdgroups → **4x improvement for D=8**.

**Tradeoff**: Loses incremental interpolation — pays `D` extra fr_mul per grid point (for the `t * diff` term). But 4x occupancy improvement should dominate.

**Also**: For D=4, streaming doesn't help much (already near occupancy limit due to WideAcc). Keep the current unrolled body for D=4 and only stream for D≥8.

### OPT-3: Deferred fr_reduce in eval body

**Impact: 5-10%** | Effort: Low

`fr_mul` calls `fr_reduce` after every multiply (conditional subtraction, ~32 ops). Intermediate products are immediately reduced only to be multiplied again.

BN254 Fr is 254 bits. An unreduced CIOS output is at most 256 bits. Two unreduced values multiplied: product ≤ 512 bits → fits in CIOS's 9-limb accumulator.

**Fix**: Add `fr_mul_unreduced` that skips the final `fr_reduce`. Use it for intermediate products. Only reduce before writing to WideAcc or before the final fr_add.

Saves ~`(D-2)` × 32 ops per grid point per pair. For D=8: ~192 ops/pair.

### OPT-4: Multi-round command batching

**Impact: 20-30% at medium sizes** | Effort: Medium

Each `pairwise_reduce` + `interpolate_pairs_batch` creates 2 separate command buffers (encode, commit, wait, encode, commit, wait). Dispatch overhead is ~20-80μs per command buffer.

**Fix**: Add a `reduce_and_bind_round` method that encodes both reduce and bind into a single command buffer. For a full sumcheck (log₂(n) rounds), batch all rounds:

```
[reduce_1 | bind_1 | reduce_2 | bind_2 | ... | reduce_k | bind_k]
```

One commit + wait for the entire sumcheck. Saves `2 × log₂(n) - 1` command buffer round-trips.

For n=2^20, that's ~39 command buffers → 1. At 50μs each = **~2ms saved**.

### OPT-5: Karatsuba acc_fmadd

**Impact: ~15% on compute-bound kernels** | Effort: High

`acc_fmadd` does schoolbook 8×8 = 64 mul+mulhi. Karatsuba on 4-limb halves: 3 × (4×4) = 48 muls (25% reduction).

Complicated by signed intermediates in the accumulator. Defer to later.

### OPT-6: fr_sqr specialization

**Impact: ~30% faster for squarings** | Effort: Medium

CIOS squaring exploits `a[i]*a[j] = a[j]*a[i]` symmetry, halving cross-terms. Used in Toom-Cook grid evaluation when `t² * x` patterns appear. Not currently exploited in the eval body.

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

1. OPT-1 (simdgroup WideAcc) — standalone shader change, low risk
2. OPT-3 (deferred fr_reduce) — small shader change, compounds with everything
3. OPT-2 (streaming for D≥8) — compiler.rs refactor, big occupancy win
4. OPT-4 (command batching) — needs trait/API change, deferred
5. OPT-5/6 — deferred, diminishing returns
