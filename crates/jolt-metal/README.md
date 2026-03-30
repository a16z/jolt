# jolt-metal

Apple Metal GPU compute backend for the Jolt zkVM.

Implements `jolt_compute::ComputeBackend` using Metal compute shaders.
Targets Apple Silicon (M1+) unified memory architecture.

## Architecture

```
CompositionFormula
    ↓ compiler.rs (AOT)
MSL source string
    ↓ MTLDevice.makeLibrary(source:)
MTLComputePipelineState
    ↓ wrapped as
MetalKernel<F>
    ↓ dispatched by
MetalBackend (ComputeBackend impl)
    ↓ buffers
MetalBuffer<T> (MTLBuffer + typed len)
```

## Dependency chain

```
jolt-field ← jolt-compute ← jolt-metal
                               ↑
                            jolt-compiler (for CompositionFormula → MSL compilation)
```

No dependency on jolt-sumcheck, jolt-zkvm, or any protocol crate.
The sumcheck protocol runs on the host CPU. Only polynomial buffer
operations (reduce, interpolate, product table) execute on GPU.

## Apple GPU Compute Model

### Thread hierarchy

```
Thread              single lane, own registers
  └─ Simdgroup      32 threads, lock-step SIMD (like a CUDA warp)
       └─ Threadgroup   1–1024 threads, shares 32 KB threadgroup memory
            └─ Grid         all threadgroups in a dispatch, across all EUs
```

Simdgroups execute in lock-step — no sync needed, supports `simd_shuffle_down`
for intra-simdgroup reductions. Threadgroups sync via `threadgroup_barrier()`.
No cross-threadgroup sync within a dispatch (requires a new dispatch).

### Memory hierarchy

```
Registers        ~32 KB/EU       0 cycles        per-thread, split across active threads
Threadgroup mem  32 KB/EU        ~2-4 cycles     explicit, programmer-controlled
L2 / tex cache   ~4-8 MB        ~50-100 cycles  shared across all EUs
SLC              ~16-32 MB      ~100-200 cycles  system-level cache
DRAM (unified)   16 GB          ~300-500 cycles  ~200 GB/s (M1 Pro)
```

### Occupancy and register pressure

The EU register file (32 KB) is shared across all active threads. More
registers per thread → fewer concurrent threads → less latency hiding.

| Field | Regs/element | D=4 kernel | D=8 kernel | Threads/EU | Simdgroups/EU |
|-------|-------------|------------|------------|------------|---------------|
| BN254 (256-bit, 8×32) | 8 | ~200 | ~400 | ~80 / ~20 | ~2.5 / ~0.6 |
| Goldilocks (64-bit, 2×32) | 2 | ~50 | ~100 | ~640 / ~320 | 20 / 10 |

For good latency hiding you want 8+ active simdgroups per EU. BN254 D=8
can barely fit 2, which is why pairwise_reduce runs at ~4% memory bandwidth
utilization — the EU stalls waiting for memory with too few threads to swap to.

### Current threadgroup size: 256

We use 256 threads per threadgroup everywhere. This is a reasonable default
for reduction kernels (8 simdgroups, power-of-2 for tree reduction, 8 KB
shared memory for Fr values). For heavy pairwise_reduce kernels the EU can't
actually run 256 threads simultaneously due to register pressure — the
hardware reduces occupancy silently. Tuning threadgroup size per kernel
(e.g., 64 for D=8) is a potential optimization but won't help much since
the per-thread register count is the binding constraint, not the threadgroup
partitioning.

### M-series GPU specs

| Chip | EUs | Max concurrent threads | DRAM BW |
|------|-----|----------------------|---------|
| M1 | 8 | ~4K-16K | 68 GB/s |
| M1 Pro | 16 | ~8K-32K | 200 GB/s |
| M1 Max | 32 | ~16K-64K | 400 GB/s |
| M3 Max | 40 | ~20K-80K | 400 GB/s |

## Benchmark Results (M1 Pro, 16 GPU EUs, ~200 GB/s)

Measured with `cargo bench -p jolt-metal --bench metal_vs_cpu`.

### pairwise_reduce (hot path — ~80% of prover time)

| D | Size | Metal | CPU | Speedup |
|---|------|-------|-----|---------|
| 4 | 16K | 0.91 ms | 0.62 ms | 0.68x |
| 4 | 256K | 4.1 ms | 20.7 ms | **5.0x** |
| 4 | 16M | 234 ms | 568 ms | **2.4x** |
| 8 | 16K | 1.4 ms | 2.0 ms | 1.4x |
| 8 | 256K | 15.2 ms | 25.0 ms | **1.6x** |
| 8 | 16M | 916 ms | 2016 ms | **2.2x** |

Metal wins at ≥256K elements. Small sizes are dominated by dispatch latency.

### Reductions (sum, dot product)

| Op | Size | Metal | CPU | Speedup |
|----|------|-------|-----|---------|
| sum | 4K | 240 µs | 65 µs | 0.27x |
| sum | 64K | 412 µs | 220 µs | 0.53x |
| sum | 1M | 498 µs | 2354 µs | **4.7x** |
| dot_product | 4K | 242 µs | 67 µs | 0.28x |
| dot_product | 64K | 451 µs | 224 µs | 0.50x |
| dot_product | 1M | 2394 µs | 2448 µs | **1.0x** |

Sum uses `acc_add_fr` (zero multiplies) for deferred accumulation, giving
large speedups. Dot product uses `acc_fmadd` (CIOS-bound), roughly matching CPU.

### Product table (eq polynomial construction)

| n_vars | Metal | CPU | Speedup |
|--------|-------|-----|---------|
| 12 | 261 µs | 205 µs | 0.79x |
| 16 | 639 µs | 995 µs | **1.6x** |
| 20 | 2660 µs | 7177 µs | **2.7x** |

GPU rounds batched into a single command buffer. Small rounds (< 1024 elements)
computed on CPU to avoid dispatch overhead.

### Element-wise arithmetic (raw field ops)

| Op | Size | Metal | CPU | Speedup |
|----|------|-------|-----|---------|
| fr_mul | 1M | 17.3 ms | 16.0 ms | 0.93x |
| fr_sqr | 1M | 11.4 ms | 14.2 ms | **1.2x** |
| fr_add | 1M | 14.7 ms | 5.5 ms | 0.37x |
| fr_fmadd | 16K thr | 16.4 ms | 98.3 ms | **6.0x** |
| fr_fmadd | 256K thr | 154 ms | 1553 ms | **10.1x** |

BN254 Fr multiply (CIOS) requires ~128 `ulong` multiplications per element,
each emulated as ~4 32-bit ops on Apple GPU. This makes pure fr_mul roughly
CPU-parity. The fmadd kernel wins big because it amortizes Montgomery reduction
via the wide accumulator (256 multiply-adds → 1 reduction).

### Throughput analysis

BN254 is a worst-case field for GPU acceleration:
- **fr_mul**: ~512 ALU cycles/element → ~60M ops/s across 64 GPU lanes
- **Peak theoretical**: ~160M fr_mul/s (register pressure limits to ~50% occupancy)
- **Observed pairwise_reduce D4 @ 16M**: 34 Melem/s × ~6 fr_muls/elem ≈ 200M fr_mul equiv/s
- **Memory bandwidth utilization**: ~4% (purely compute-bound)

GPU-friendly fields (Goldilocks, 64-bit) would see 10–50x GPU speedup. BN254's
256-bit multiply chains have limited ILP, so the real Metal wins come from fused
operations (WideAcc), bandwidth-bound ops (interpolation), and kernel parallelism.

## Non-Goals (for now)

- **MSM on GPU**: Multi-scalar multiplication is in jolt-dory, not
  jolt-compute. Dory MSM acceleration would be a separate crate or
  a jolt-dory feature.
- **WebGPU**: Separate crate (jolt-webgpu). Different shader language
  (WGSL), different buffer model, different reduction strategy.
- **CUDA**: Separate crate (jolt-cuda). PTX codegen, different memory
  model, different thread hierarchy.
- **Older Apple devices**: Focused on Apple Silicon (M1+). Intel Macs
  with discrete AMD GPUs are not targeted.
