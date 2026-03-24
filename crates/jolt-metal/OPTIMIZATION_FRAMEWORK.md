# Sumcheck Hybrid Backend Optimization Framework

Performance optimization loop for the Metal/CPU hybrid backend targeting BN254 sumcheck
operations on Apple Silicon. Goal: **5x e2e speedup** — see [`OPTIMIZATION_GOAL.md`](OPTIMIZATION_GOAL.md).

## 1. Roofline Model

### Hardware Parameters

All Apple Silicon GPUs share: SIMD width = 32, max threadgroup = 1024, threadgroup memory = 32 KB.

| Chip | GPU Cores | Mem BW (GB/s) | FP32 TFLOPS | Est. Montgomery TFLOPS* |
|------|-----------|---------------|-------------|-------------------------|
| M1 | 8 | 68 | 2.6 | 1.3-1.6 |
| M1 Pro | 16 | 200 | 5.2 | 2.6-3.1 |
| M1 Max | 32 | 400 | 10.4 | 5.2-6.2 |
| M2 Pro | 19 | 200 | 6.8 | 3.4-4.1 |
| M3 Max | 40 | 400 | 16.4 | 8.2-9.8 |
| M4 | 10 | 120 | 4.6 | 2.3-2.8 |
| M4 Pro | 20 | 273 | 9.2 | 4.6-5.5 |
| M4 Max | 40 | 546 | 18.4 | 9.2-11.0 |

\* Montgomery multiplication via CIOS requires `mulhi(u32,u32)` which Apple GPU emulates
as ~2 32-bit ops. Effective throughput ≈ 50-60% of FP32 TFLOPS for Montgomery-heavy code.

### Why the Naive Roofline is Misleading for BN254

Standard roofline analysis computes arithmetic intensity as `ops/byte` and compares against
the `peak_compute / peak_bandwidth` crossover. By that metric, our kernels look bandwidth-bound
(AI ≈ 2-4 MACs/byte vs crossover ≈ 13 on M1 Pro).

**But BN254 Montgomery multiplication is the real bottleneck.** A single `fr_mul` requires
~64 chained u32 multiply-accumulate operations with deep data dependencies (each CIOS round
depends on the previous). This means:
- Effective per-thread `fr_mul` latency: ~40-80 ns (30-60 GPU cycles)
- GPU throughput for `fr_mul`: ~5-10 Gmul/s (NOT the 28 Gmul/s from peak INT32 TFLOPS)
- CPU throughput for BN254 `fr_mul`: ~25-40 ns/mul/core → 200-320 Mmul/s on 8 cores

**The GPU is not fundamentally faster per-mul than the CPU** for BN254. Both are u32-based
ALUs. The GPU wins by providing additional parallel compute lanes, roughly doubling total
system Montgomery throughput. This explains the consistent ~2x hybrid speedup.

### Register Pressure (Binding Constraint)

The EU register file is ~32-48 KB per execution unit. Occupancy = concurrent threads / max threads.

| Field | Limbs | Regs/Fr | Regs/WideAcc | D=4 regs/thread | D=8 regs/thread | D=4 occupancy | D=8 occupancy |
|-------|-------|---------|--------------|-----------------|-----------------|---------------|---------------|
| BN254 (256-bit) | 8 | 8×u32 | 18×u32 | ~200 | ~400 | ~2.5 simdgroups/EU | ~0.6 simdgroups/EU |
| Goldilocks (64-bit) | 2 | 2×u32 | 6×u32 | ~50 | ~100 | 20 simdgroups/EU | 10 simdgroups/EU |

Optimal latency hiding requires 8+ simdgroups/EU. BN254 D=8 can barely fit 2 — the EU stalls
on memory with too few threads to swap to. This compounds the CIOS dependency chain problem.

### Corrected Performance Ceiling

The practical ceiling for hybrid speedup over CPU-only is:

```
speedup_max = (CPU_throughput + GPU_throughput) / CPU_throughput
```

For BN254 on M1 Pro (8 perf CPU cores + 16 GPU cores):
- CPU: ~300 Mmul/s (8 cores × ~40 Mmul/s/core)
- GPU: ~300-600 Mmul/s (limited by occupancy, not raw TFLOPS)
- **Max hybrid speedup: 2-3x** (matching observed 1.6-2.4x)

This ceiling cannot be broken by tuning the current 1-thread-per-pair kernel. To exceed 3x:
1. **Cooperative threading** (O0a): Distribute fr_mul across 8 threads per limb, making GPU
   4-8x faster per-mul than CPU via parallelized CIOS. This is the primary path to 5x.
2. Reduce CPU-side sequential overhead between rounds (O3: pipeline overlap)
3. Eliminate dispatch overhead (O1: batch command buffers)
4. Use a smaller field (Goldilocks: 10-50x GPU speedup — out of scope for this goal)

## 2. Per-Kernel Analytical Model

### 2.1 pairwise_reduce (ProductSum)

The dominant hot path (~80% of prover time). One dispatch per sumcheck round.

**Operation per pair (2 elements → 1 partial sum across D+1 eval dimensions):**

| D | Variant | fr_mul/pair | fr_add/pair | Reads (bytes) | Writes (bytes) | AI (MACs/byte) |
|---|---------|-------------|-------------|---------------|----------------|-----------------|
| 4 | Toom-Cook weighted | ~15 | ~16 | 4×32 + 32 = 160 | 0 (threadgroup) | ~19 |
| 4 | Toom-Cook unweighted | ~15 + 4×weight | ~16 | 4×32 = 128 | 0 | ~27 |
| 8 | Toom-Cook D8 deferred, weighted | ~30 | ~32 | 8×32 + 32 = 288 | 0 | ~21 |
| 8 | Toom-Cook D8 deferred, unweighted | ~30 + 8×weight | ~32 | 8×32 = 256 | 0 | ~30 |
| 2 (eq_product) | Direct eval | ~4 | ~4 | 2×32 + 32 = 96 | 0 | ~8 |
| 2 (hamming) | Custom expr, D=3 | ~6 | ~6 | 2×32 + 32 = 96 | 0 | ~12 |

**Observation**: All ProductSum kernels are compute-bound (AI > 13). EqProduct/Hamming are
near the crossover — they could benefit from memory-level optimizations on high-BW chips.

**Theoretical minimum time** (compute-bound, 100% occupancy):

```
T_min = (n/2) × fr_muls_per_pair × cycles_per_fr_mul / (num_EUs × threads_per_EU × clock_GHz)
```

For M1 Pro (16 EUs, ~1.3 GHz clock, ~80 threads/EU at D=4 occupancy):

| D | n=2^20 | fr_muls | T_compute_min | T_observed_Metal | Efficiency |
|---|--------|---------|---------------|------------------|------------|
| 4 | 524K pairs | 15 | 7.9M muls / (16×80×1.3G) ≈ 4.7 ms | ~4.1 ms* | **87%** |
| 8 | 524K pairs | 30 | 15.7M muls / (16×20×1.3G) ≈ 37.8 ms | — | — |

\* From README benchmark at 256K pairs (interpolating for 2^20).

The D=4 kernel is operating at ~87% of theoretical peak — very efficient. D=8 has 4× worse
occupancy so the practical limit is much worse.

### 2.2 interpolate_pairs (bind)

Per pair: 1 fr_sub + 1 fr_mul + 1 fr_add. Reads 2×32 = 64 bytes, writes 32 bytes.

```
AI = (~3 field ops × ~16 u32 ops each) / 96 bytes ≈ 0.5 MACs/byte
```

**Memory-bandwidth-bound.** Theoretical minimum:

```
T_min = n × 96 bytes / bandwidth
```

| Chip | n=2^20 | T_bw_min | T_observed | BW utilization |
|------|--------|----------|------------|----------------|
| M1 Pro | 1M | 100M bytes / 200 GB/s = 0.5 ms | ~0.8 ms | **63%** |

Interpolation should saturate memory bandwidth. Current 63% utilization suggests room for
improvement in memory access patterns (H2L split-half has poor coalescing vs L2H interleaved).

### 2.3 fused_interpolate_reduce (H2L only)

Combines interpolation + reduce into one pass. Saves one full read of all D input buffers.

**Savings per round**: D × N × 32 bytes of bandwidth (one fewer buffer scan).

For D=8 at N=2^20: saves 8 × 1M × 32 = 256 MB of reads per round.

**Expected speedup** over separate interpolate+reduce:
- Compute-bound kernels (D≥4): Marginal — compute dominates, not memory
- Near-crossover kernels (D=2): Significant — fusing saves the dominant cost

### 2.4 sum / dot_product (reductions)

| Op | AI (MACs/byte) | Bound | T_min at M1 Pro |
|----|----------------|-------|-----------------|
| sum | ~1 (acc_add_fr only) | Memory BW | n×32 / 200 GB/s |
| dot_product | ~16 (acc_fmadd) | Compute | n×64 / 200 GB/s (if BW) or n×muls/peak (if compute) |

Sum at 1M elements: T_bw_min = 32M / 200G = 0.16 ms. Observed: 0.5 ms → **32% BW utilization**.
The gap is dispatch overhead + reduction tree overhead.

### 2.5 product_table (eq polynomial construction)

Per round k (2^k threads): 1 fr_mul + 1 fr_sub per thread. Reads 32 bytes, writes 64 bytes.

```
AI = ~16 u32 ops / 96 bytes ≈ 0.17 MACs/byte → memory-bandwidth-bound
```

Rounds < 1024 elements stay on CPU (dispatch overhead dominates). GPU rounds are batched into
one command buffer with implicit barriers between rounds.

## 3. Observed vs Theoretical Comparison (M1 Pro Baseline)

### Critical Finding: Compute-Bound Despite Low Arithmetic Intensity

All scenarios have arithmetic intensity 1.4-3.7 MACs/byte — below the roofline crossover
of ~13. Naively, they should be bandwidth-bound. But observed utilization of memory
bandwidth is only 3-5%, far too low for a bandwidth-bound workload.

**The resolution**: BN254 Montgomery multiplication has such deep dependency chains (~32
serial MADs minimum per CIOS) that the GPU cannot issue enough in-flight operations to
saturate memory bandwidth, even though there are "enough" bytes to read. The workload is
**latency-bound on Montgomery multiply**, not bandwidth-bound or throughput-bound.

This means the correct comparison is against Montgomery multiply throughput, not memory bandwidth.

### Isolated Kernel Benchmarks

Benchmark: `cargo bench -p jolt-metal --bench metal_vs_cpu -q`

| Kernel | Size | Metal Observed | Theoretical Min | Efficiency | Bottleneck |
|--------|------|----------------|-----------------|------------|------------|
| pairwise_reduce D4 | 2^18 | 4.1 ms | ~2.4 ms (compute) | 59% | Register pressure, occupancy |
| pairwise_reduce D4 | 2^14 | 0.91 ms | ~0.15 ms (compute) | 16% | Dispatch latency (~200 µs) |
| pairwise_reduce D8 | 2^18 | 15.2 ms | ~9.4 ms (compute) | 62% | Register pressure (0.6 sg/EU) |
| pairwise_reduce D8 | 2^14 | 1.4 ms | ~0.6 ms (compute) | 43% | Dispatch latency |
| sum | 2^20 | 0.5 ms | 0.16 ms (BW) | 32% | Reduction tree + dispatch |
| dot_product | 2^20 | 2.4 ms | ~1.6 ms (compute) | 67% | CIOS chains |
| interpolate | 2^20 | ~0.8 ms | 0.5 ms (BW) | 63% | H2L coalescing |
| product_table 20v | total | 2.66 ms | ~1.5 ms (BW) | 56% | Mixed CPU/GPU rounds |

### End-to-End Sumcheck (20 rounds, 2^20 → 2^0)

Benchmark: `cargo bench -p jolt-metal --bench sumcheck_e2e -q`

| Scenario | Buffers | Muls/pair | CPU (ms) | Hybrid (ms) | Speedup | CPU compute eff.* |
|----------|---------|-----------|----------|-------------|---------|-------------------|
| Toom D4, P=1 | 5 | ~19 | 59 | 30 | 1.98x | ~83% |
| Toom D4, P=3 | 13 | ~47 | 110 | 53 | 2.08x | ~73% |
| Toom D8, P=1 | 9 | ~47 | 114 | 63 | 1.81x | ~80% |
| EqProduct (D=2) | 2 | ~4 | 36 | 22 | 1.64x | ~62% |
| Hamming (D=3) | 2 | ~8 | 42 | 25 | 1.68x | ~84% |

\* CPU compute efficiency = theoretical_cpu_min / observed_cpu. Theoretical computed at
~300 Mmul/s (8 perf cores × ~40 Mmul/s/core for BN254 Montgomery).

### Overhead Breakdown (D4 P1 example: 30 ms hybrid)

| Component | Estimated time | % of total |
|-----------|---------------|------------|
| GPU kernel execution (8 Metal rounds) | ~10-15 ms | 35-50% |
| GPU dispatch overhead (8× encode+commit+wait) | ~2-4 ms | 7-13% |
| CPU rounds (12 rounds after switchover) | ~5-8 ms | 17-27% |
| Round polynomial reconstruction + Lagrange | ~2-3 ms | 7-10% |
| Transcript hashing (Blake2b × 20 rounds) | ~1-2 ms | 3-7% |
| Product table / eq polynomial | ~2-3 ms | 7-10% |

**Key insight**: GPU kernel execution is only ~40% of total time. The remaining 60% is
overhead that cannot be reduced by kernel optimization alone. Dispatch latency and
CPU-side sequential work are the primary targets for further improvement.

### Per-Round GPU Efficiency Decay

The GPU is only efficient for the first few rounds. Later Metal rounds have diminishing returns:

| Round | Buffer size | GPU kernel time | Dispatch overhead | GPU efficiency |
|-------|------------|-----------------|-------------------|----------------|
| 0 | 2^20 | ~1.5 ms | ~100 µs | 94% useful |
| 1 | 2^19 | ~0.75 ms | ~100 µs | 88% useful |
| 2 | 2^18 | ~0.38 ms | ~100 µs | 79% useful |
| 3 | 2^17 | ~0.19 ms | ~100 µs | 66% useful |
| 4 | 2^16 | ~0.10 ms | ~100 µs | 50% useful |
| 5-7 | 2^15-2^13 | <0.05 ms | ~100 µs | <33% useful |

Rounds 5-7 are nearly pure dispatch overhead. An optimal switchover would hand off
to CPU after round 4 (buffer < 2^16), not round 8 (buffer < 2^12).

## 4. Optimization Opportunities (Ranked by Expected Impact)

Since GPU kernel execution is only ~40% of hybrid time, dispatch and CPU-side overhead
dominate. Optimizations target four categories:

1. **Thread strategy**: Current 1-thread-per-pair with serial CIOS is the root cause of
   low GPU utilization. Cooperative threading can break through the 2-3x ceiling.
2. **Dispatch overhead**: ~15% of time, reducible by batching and threshold tuning
3. **CPU-side sequential work**: ~45% of time, reducible by pipelining and overlap
4. **GPU kernel throughput**: ~40% of time, limited by Montgomery mul latency under
   current thread model

### Tier 0: Architectural (Ceiling-Breaking)

These change the fundamental thread model. The current 1-thread-per-pair strategy
assigns ~960 serial dependent u32 MADs per pair (D=8 Toom-Cook) to a single thread.
The CIOS dependency chain (~32 serial MADs per fr_mul) cannot be parallelized within
one thread, and low occupancy (0.6 simdgroups/EU for D=8) means almost no latency
hiding from thread-swapping. This is why GPU throughput roughly matches CPU — both
are serializing the same dependency chain.

Cooperative threading distributes one fr_mul across multiple threads, converting the
serial CIOS chain into parallel cross-thread communication. This is the only path to
making GPU fundamentally faster per-mul than CPU.

#### O0a: Thread-per-limb cooperative Montgomery multiplication
**Current**: One thread computes all 8 limbs of a CIOS round sequentially. Each CIOS
round has a carry chain: limb K depends on limb K-1's carry. 8 rounds × 8 limbs × 2 ops
= ~128 serial u32 multiply-adds per fr_mul, with limited ILP.
**Proposal**: Assign 8 threads to one field element (one thread per u32 limb). Within a
32-thread simdgroup, this processes 4 field element pairs simultaneously. Each CIOS round
becomes a parallel step across 8 threads with `simd_shuffle` for carry propagation.

**Thread mapping** (within one 32-thread simdgroup):
```
Threads  0-7:  pair 0, limbs 0-7
Threads  8-15: pair 1, limbs 0-7
Threads 16-23: pair 2, limbs 0-7
Threads 24-31: pair 3, limbs 0-7
```

**CIOS round K (all 8 threads in parallel)**:
```metal
// Thread `t` owns limb `t % 8` of the accumulator
uint my_limb = thread_index_in_simdgroup % 8;

// Step 1: All 8 threads multiply their limb of A by B[k] (parallel)
uint prod_lo = A_limbs[my_limb] * B_k;
uint prod_hi = mulhi(A_limbs[my_limb], B_k);

// Step 2: Add to accumulator with carry propagation via simd_shuffle
// Carry from limb i goes to limb i+1
uint carry = simd_shuffle_up(overflow, 1);  // get carry from previous limb
T_limbs[my_limb] = add_with_carry(T_limbs[my_limb], prod_lo, carry);

// Step 3: Montgomery reduction step (q = T[0] * n_inv, parallel multiply q * N[my_limb])
uint q = simd_broadcast(T_zero_times_ninv, 0);  // broadcast from thread 0
uint qn_lo = q * N_limbs[my_limb];
// ... carry propagation via simd_shuffle
```

**Why this helps**:
- Each CIOS round is 1-2 cycles across 8 parallel threads instead of 8 serial steps
- fr_mul drops from ~40-80 ns to ~5-10 ns (8x latency reduction)
- 4 pairs processed per simdgroup → same work, 4x fewer simdgroups needed
- Register pressure per thread drops dramatically (1 limb instead of 8)
- Occupancy increases from ~0.6 to ~10+ simdgroups/EU

**Expected gain**: 4-8x on GPU kernel throughput → potential 3-5x e2e improvement
**Risk**: High complexity. Carry propagation via simd_shuffle adds ~2 cycles per round
(vs 0 for sequential carry). Total CIOS latency is ~16 cycles (8 rounds × 2 shuffle steps)
vs ~64 cycles sequential — 4x improvement, not 8x. Cross-pair communication needed for
Toom-Cook eval body (shuffle between thread groups within simdgroup).
**Benchmark**: First build a standalone `fr_mul` microbenchmark with cooperative threads,
then extend to full pairwise_reduce.
**Prerequisite**: O0c (profiling) to validate that kernel time is actually the bottleneck.

#### O0b: SIMD-cooperative Toom-Cook evaluation
**Current**: One thread evaluates the entire Toom-Cook body (30 fr_muls for D=8). Each
fr_mul is serial → the whole eval body is a long sequential chain.
**Proposal**: With thread-per-limb (O0a), the Toom-Cook eval body runs identically but
each fr_mul is now cooperative. Additionally, independent fr_muls within the Toom-Cook
DAG can execute concurrently across the 4 pair-groups within a simdgroup:

```
Toom-Cook D=8 has this dependency structure:
  Half A: eval_linear_prod_2(p0,p1) → ex2 → a1..a_inf  (11 muls, partly parallel)
  Half B: eval_linear_prod_2(p2,p3) → ex2 → b1..b_inf  (11 muls, partly parallel)
  Final:  a_k * b_k for k=1..8                          (8 muls, fully parallel)
```

The 8 final pointwise multiplications are independent — with cooperative threads, all 8
can run in 8 cycles total (1 mul latency) instead of 8 × ~40 ns sequential.

**Expected gain**: Combined with O0a, reduces per-pair Toom-Cook time from ~1200 ns to
~150-300 ns (4-8x). This directly translates to GPU kernel speedup.
**Risk**: Same as O0a. The eval body needs careful rewrite to use cooperative primitives.
**Prerequisite**: O0a working first.

#### O0c: Instrument and profile (prerequisite for all above)
**Before any kernel rearchitecture**, instrument the hot path with `tracing` spans to
get actual time measurements. The `jolt-profiling` crate provides Chrome/Perfetto output.

**Gaps to fill** (~30 lines of code):
1. Add `tracing` dep to `jolt-compute` Cargo.toml
2. Add `#[instrument]` spans to `MetalBackend::dispatch_reduce`, `dispatch_fused_reduce`,
   `dispatch_elementwise_reduce` (captures GPU dispatch + wait time)
3. Add per-round spans inside `SumcheckProver::prove()` loop:
   `round_polynomial`, `transcript_absorb`, `transcript_squeeze`, `bind`
4. Add a tracing subscriber setup in `sumcheck_e2e.rs` gated on an env var
   (e.g., `JOLT_TRACE=1 cargo bench ...`)

**Output**: Perfetto JSON trace showing exact time distribution per round, per operation.
This validates or refutes the 40/15/45 overhead split and determines whether O0a/O0b
or O1-O3 should be pursued first.
**Risk**: None. Pure instrumentation, no behavioral changes.
**Expected time**: 1-2 hours.

### Tier 1: High Impact (>10% e2e improvement)

#### O1: Batch command buffer encoding
**Current**: Each `pairwise_reduce` and `interpolate_pairs_inplace` call creates a new
command buffer, encodes, commits, and waits for completion. 8 Metal rounds = ~16 dispatches.
**Proposal**: Encode all GPU-side work for a single sumcheck round (reduce + interpolate)
into one command buffer with a memory barrier between them. For the fused H2L path, this
is already a single dispatch — extend to the non-fused path too.
**Expected gain**: 2-3 ms (halve dispatch count, ~100 µs saved per eliminated CB roundtrip)
**Benchmark**: `sumcheck_e2e` — compare before/after on all scenarios
**Risk**: Low — Metal supports explicit barriers between dispatches in same CB.
The sumcheck challenge scalar must be known before interpolation, which is only available
after reduce completes. So reduce+interpolate cannot share a CB unless fused. But
multiple buffer interpolations within one round CAN be batched.

#### O2: Raise hybrid switchover threshold
**Current**: 2^12 = 4096 elements. Last 8 Metal rounds have <33% GPU efficiency.
**Evidence**: Per-round efficiency analysis shows rounds 5-7 are nearly pure dispatch
overhead. Moving switchover to 2^16 would save ~3 inefficient GPU rounds.
**Experiment**: Run `sumcheck_e2e` with thresholds {2^10, 2^12, 2^14, 2^16, 2^18}.
The optimal threshold depends on D:
- D=4 (high occupancy): switchover may be lower (GPU stays efficient longer)
- D=8 (low occupancy): switchover should be higher (GPU overhead dominates sooner)
Consider per-kernel-shape threshold instead of a single global value.
**Expected gain**: 2-5 ms
**Benchmark**: `threshold` bench for crossover curves, then `sumcheck_e2e` with modified const

#### O3: Pipeline CPU/GPU overlap
**Current**: Each round is fully sequential: GPU reduce → CPU read partials → CPU reconstruct
round polynomial → CPU transcript → CPU squeeze challenge → GPU interpolate → next round.
**Proposal**: While GPU executes round N's interpolation (which only needs the challenge
scalar, already computed), CPU can begin processing round N's partial sums and round
polynomial reconstruction in parallel. The overlap window is the interpolation dispatch time.
**Implementation**: Use `addCompletedHandler` instead of `waitUntilCompleted` for the reduce
command buffer. Start CPU polynomial reconstruction as soon as reduce partials are available
(before interpolation completes). Enqueue interpolation into a separate command buffer.
**Expected gain**: 3-5 ms (overlap CPU reconstruction with GPU interpolation)
**Risk**: Medium — requires careful synchronization and non-trivial refactoring of the
`SumcheckProver::prove()` loop. The protocol is inherently sequential per round, but
WITHIN a round there's overlap opportunity.
**Benchmark**: `sumcheck_e2e` all scenarios

### Tier 2: Medium Impact (3-10% e2e improvement)

#### O4: D=8 register pressure reduction — 2-pass accumulation
**Current**: D=8 kernel uses Fr accumulators (8 limbs each × 9 evals = 72 limbs). Deferred-read
Toom-Cook helps but peak register usage is still ~400 u32, giving 0.6 simdgroups/EU.
**Proposal**: For D=8, use 2-pass accumulation: compute evals 0-4 in pass 1, evals 5-8 in
pass 2, merge results. Each pass uses ~250 registers → ~1.5 simdgroups/EU (2.5x current).
**Expected gain**: ~25% on pairwise_reduce D8 kernel time. At ~40% of total hybrid time,
this is ~10% e2e for D8 scenarios. ~6% e2e for D4 (occupancy is already better).
**Benchmark**: `pairwise_reduce` D8 at 2^20 (isolated), then `sumcheck_e2e -- toom_D8`
**Risk**: Medium — 2× bandwidth re-read trades memory for compute. Since workload is
Montgomery-latency-bound (not bandwidth-bound), the extra reads may be free. But
2 dispatches instead of 1 adds ~100 µs overhead. Must benchmark both.
**Note**: Superseded by O0a if cooperative threading works — O0a solves register pressure
by reducing per-thread state to 1 limb instead of 8.

#### O5: L2H coalescing for non-fused interpolation
**Current**: H2L interpolation reads `buf[i]` and `buf[i+half]` — the second read is strided
by N/2 × 32 bytes, causing poor GPU cache utilization. Observed: 63% BW utilization.
**Proposal**: For non-fused interpolation paths (standard grid kernels like EqProduct/Hamming),
use L2H layout. L2H reads `buf[2i]` and `buf[2i+1]` — perfectly coalesced. This helps
EqProduct and Hamming where fused kernel doesn't apply (unweighted).
**Expected gain**: ~37% improvement on interpolation time. Interpolation is ~15% of round
time for low-D kernels → ~5% e2e for EqProduct/Hamming.
**Benchmark**: `metal_vs_cpu -- interpolate` at large sizes

#### O6: Toom-Cook specialization for D=2/D=3
**Current**: EqProduct (D=2) and Hamming (D=3) use the generic `Custom` expression path,
which evaluates the expression DAG per grid point with per-variable SSA emission.
**Proposal**: Implement direct eval specializations. D=2 needs exactly 2 muls per pair
(at points {0, 2}). D=3 needs 4 muls with incremental differences. The Custom path
adds overhead from expression tree walking and challenge buffer reads.
**Expected gain**: ~15% on eq_product/hamming GPU kernel time → ~6% e2e on those scenarios
**Benchmark**: `sumcheck_e2e` eq_product and hamming

### Tier 3: Low Impact / Speculative (<3% e2e or needs research)

#### O7: Per-kernel-shape switchover threshold
**Current**: Single global `HYBRID_THRESHOLD = 2^12`.
**Proposal**: Different kernels have different GPU efficiency curves. D=4 (high occupancy)
stays efficient to smaller sizes than D=8 (low occupancy). Store per-shape threshold
in `MetalKernel` and use it during switchover decisions.
**Benchmark**: `threshold` bench per D value

#### O8: Per-generation threadgroup size tuning
**Current**: 128 threads for all reduce kernels on all hardware.
**Proposal**: M3/M4 dynamic caching may benefit from larger threadgroups (256) since
register allocation is per-SIMD-group rather than per-dispatch. Profile on M3/M4 hardware.
**Benchmark**: `pairwise_reduce` D4 and D8 with group sizes {64, 128, 256, 512}

#### O9: BN254 CIOS micro-optimization (single-thread)
**Current**: Standard CIOS with 8 rounds of multiply-accumulate.
**Proposal**: Explore (a) Karatsuba splitting for the 8×8 case, (b) using Metal's `mad24`
for 24-bit multiply with carry chains, (c) software pipelining of CIOS rounds.
**Expected gain**: ~5-10% on GPU kernel time → ~2-4% e2e
**Risk**: High — compiler may already optimize these patterns; hand-tuning MSL
is fragile across GPU generations.
**Note**: Largely superseded by O0a — cooperative threading changes the multiplication
algorithm fundamentally, making single-thread micro-optimizations irrelevant.

#### O10: Product table: pure GPU for all rounds
**Current**: Rounds with <1024 elements fall back to CPU.
**Proposal**: Keep all product_table rounds on GPU when batched into one command buffer.
**Expected gain**: ~0.5-1 ms (eliminate CPU↔GPU sync for small rounds)
**Risk**: Low but small impact. Product table is <10% of total time.

## 5. Optimization Loop Methodology

### 5.1 Experiment Protocol

Each optimization attempt follows this cycle:

```
1. HYPOTHESIS
   - What bottleneck are you targeting?
   - What is the expected speedup and why?

2. BASELINE
   - Run the relevant benchmark(s) 3 times, record median
   - Record hardware: `system_profiler SPHardwareDataType | grep Chip`
   - Record commit hash: `git rev-parse --short HEAD`

3. IMPLEMENT
   - Branch from current: `git checkout -b opt/<name>`
   - Make the change
   - Verify correctness: `cargo nextest run -p jolt-metal --cargo-quiet`
   - Verify no regressions: `cargo clippy -p jolt-metal --message-format=short -q -- -D warnings`

4. BENCHMARK
   - Run the same benchmark(s) 3 times on same hardware
   - Record median, min, max
   - Compute speedup vs baseline

5. ANALYZE
   - Did it match the hypothesis?
   - If not, why? (Roofline model wrong? Hidden bottleneck?)
   - Does it regress any other scenario?

6. DECIDE
   - If speedup > 3%: merge, update results table below
   - If speedup 1-3%: keep if zero-cost, otherwise discard
   - If regression: discard, document why in the log below

7. CLEANUP
   - Delete branch if discarded
   - Update the results table and experiment log
```

### 5.2 Benchmark Commands

```bash
# Isolated kernel benchmarks (fast iteration, ~2 min)
cargo bench -p jolt-metal --bench metal_vs_cpu -q

# Specific kernel only
cargo bench -p jolt-metal --bench metal_vs_cpu -q -- pairwise_reduce/D4
cargo bench -p jolt-metal --bench metal_vs_cpu -q -- pairwise_reduce/D8

# End-to-end sumcheck (production-representative, ~5 min)
cargo bench -p jolt-metal --bench sumcheck_e2e -q

# Specific e2e scenario
cargo bench -p jolt-metal --bench sumcheck_e2e -q -- toom_D4_P1
cargo bench -p jolt-metal --bench sumcheck_e2e -q -- toom_D8_P1

# Threshold sweep (find optimal hybrid switchover point)
cargo bench -p jolt-metal --bench threshold -q

# H2L vs L2H round comparison
cargo bench -p jolt-metal --bench metal_vs_cpu -q -- sumcheck_round

# Field128 throughput (validate occupancy theory with small field)
cargo bench -p jolt-metal --bench field128_throughput -q
```

### 5.3 Which Benchmark to Use When

| Optimizing... | Primary benchmark | Secondary | Why |
|---------------|-------------------|-----------|-----|
| Reduce kernel throughput | `metal_vs_cpu -- pairwise_reduce` | `sumcheck_e2e` | Isolated = no noise from other ops |
| Interpolation throughput | `metal_vs_cpu -- interpolate` | `sumcheck_round_h2l` | Same |
| Dispatch latency | `sumcheck_round` (4 rounds) | `sumcheck_e2e` (20 rounds) | Round loop amplifies per-dispatch cost |
| Hybrid switchover | `threshold` | `sumcheck_e2e` with modified threshold | threshold gives crossover curve |
| Fused kernel | `sumcheck_e2e -- toom` | `metal_vs_cpu -- sumcheck_round_h2l` | Fused only fires in H2L weighted path |
| Register pressure | `metal_vs_cpu -- pairwise_reduce/D8` | `field128_throughput` | D8 = worst case; field128 validates occupancy model |
| End-to-end latency | `sumcheck_e2e` (all scenarios) | — | The number that matters |

### 5.4 Interpreting Results

**Speedup over CPU baseline** is the primary metric for the hybrid backend. We track:

- `hybrid_time / cpu_time` — the ratio that matters for production
- `metal_kernel_time / total_hybrid_time` — what fraction of time is actual GPU work
- `metal_dispatch_overhead` — measured at small sizes where compute is negligible

**Three regimes:**

1. **Kernel-limited** (GPU kernel > 60% of time): Kernel optimizations help. Targets:
   register pressure, occupancy, Montgomery throughput.
2. **Dispatch-limited** (overhead > 30% of time): Batching and threshold tuning help.
   Targets: CB encoding, switchover point, fused kernels.
3. **Protocol-limited** (CPU sequential > 40% of time): No kernel optimization will help.
   Targets: transcript, reconstruction, pipelining CPU/GPU overlap.

**Current state**: We are in the protocol-limited regime for most scenarios. GPU kernel
execution is ~40% of time, CPU sequential work is ~45%, dispatch overhead is ~15%.

**Diminishing returns ceiling**: For BN254 on Apple Silicon, the hybrid speedup ceiling
is ~2-3x (GPU provides roughly equal Montgomery throughput to CPU). Observed 1.6-2.1x
means we're at 55-70% of the practical ceiling. Reaching 2.5x requires reducing
CPU-side overhead AND optimizing dispatch — kernel optimization alone won't get there.

**When to pivot to field optimization**: If all Tier 1 and Tier 2 optimizations are
exhausted and speedup is still <2.5x, the architectural conclusion is that BN254 is
too expensive for meaningful GPU acceleration. The right move is a smaller field
(Goldilocks: 4× less register pressure → 10× better occupancy → 10-50× GPU speedup).

## 6. Results Tracking

All baselines, experiment results, and decisions are recorded in
[`OPTIMIZATION_LOG.md`](OPTIMIZATION_LOG.md). That file is the single source of truth
for what was tried, what worked, and what didn't.

## 7. Current Dispatch Constants

For reference, these are the tuning knobs in the codebase:

| Constant | Value | File | Purpose |
|----------|-------|------|---------|
| `reduce_group_size` | 128 | `metal_device_config.rs` | Threads per threadgroup for reduce |
| `elementwise_group_size` | 256 | `metal_device_config.rs` | Threads per threadgroup for sum/dot |
| `max_reduce_groups` | 256 | `metal_device_config.rs` | Max threadgroups per reduce dispatch |
| `simd_size` | 32 | `metal_device_config.rs` | SIMD width (fixed on Apple GPU) |
| `split_pass_threshold` | 1024 | `metal_device_config.rs` | D threshold for multi-pass (disabled) |
| `HYBRID_THRESHOLD` | 2^12 | `sumcheck_e2e.rs` | Hybrid Metal→CPU switchover size |
| `PAR_THRESHOLD` | 1024 | `device.rs` | Product table CPU→GPU switchover |
| `CompileMode` | Performance | `compiler.rs` | Full LLVM inlining vs fast compile |

## 8. Architecture-Specific Notes

### M3/M4 Dynamic Caching

M3+ GPUs dynamically allocate registers per SIMD-group rather than statically partitioning
at dispatch time. This means kernels with variable register pressure across threadgroups
can achieve higher average occupancy. For our workloads (every thread does the same thing),
the benefit is modest but measurable.

**Action item**: When M3/M4 hardware is available, run the full benchmark suite and record
M3-specific baselines. The occupancy improvement may shift optimal `reduce_group_size` and
`split_pass_threshold`.

### M4 Max Memory Bandwidth (546 GB/s)

M4 Max has ~2.7× the bandwidth of M1 Pro. For bandwidth-bound kernels (interpolation, sum,
product_table), this shifts the roofline significantly. Kernels that are compute-bound on
M1 Pro may become mixed on M4 Max.

**Action item**: Re-run threshold sweep on M4 Max to find the new crossover points.
