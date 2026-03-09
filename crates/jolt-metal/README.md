# jolt-metal

Apple Metal GPU compute backend for the Jolt zkVM.

Implements `jolt_compute::ComputeBackend` using Metal compute shaders.
Targets Apple Silicon (M1+) unified memory architecture.

## Architecture

```
jolt-ir::KernelDescriptor
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
                            jolt-ir (for KernelDescriptor → MSL compilation)
```

No dependency on jolt-sumcheck, jolt-zkvm, or any protocol crate.
The sumcheck protocol runs on the host CPU. Only polynomial buffer
operations (reduce, interpolate, product table) execute on GPU.

## Implementation Phases

### Phase 1: BN254 Field Arithmetic in MSL

The foundation everything else builds on. BN254 Fr is a 254-bit prime
field. Metal has no native 256-bit integer, so we implement Montgomery
arithmetic over 4x64-bit limbs.

**Deliverables:**
- `src/shaders/bn254_fr.metal` — MSL header with Fr type and operations
  - 4-limb representation: `struct Fr { uint64_t limbs[4]; }`
  - Montgomery multiplication (`mont_mul`)
  - Montgomery reduction (`mont_reduce`)
  - Addition with carry chain (`fr_add`)
  - Subtraction with borrow chain (`fr_sub`)
  - Negation (`fr_neg`)
  - `fr_from_u64` — scalar-to-field promotion (for compact buffer binding)
  - Constants: `MODULUS`, `INV` (Montgomery parameter), `R`, `R2`
- `src/shaders/wide_accumulator.metal` — Delayed reduction accumulator
  - 6-limb (or 8-limb) wide integer for deferred fmadd
  - `acc_fmadd(acc, a, b)` — multiply-add without reduction
  - `acc_reduce(acc) -> Fr` — single Barrett/Montgomery reduction
  - `acc_merge(dst, src)` — merge two accumulators (for threadgroup reduce)
- `src/field/mod.rs` — Rust-side constants and test harness
- **Parity test:** Generate 10k random Fr pairs, compute `a*b` on both
  CPU (jolt-field) and Metal, verify exact match

**Hardware notes:**
- M1/M2 (Metal 2.4): `uint64_t` supported but 64-bit multiply emulated
  via 32-bit ops (~4x slower than native). Still faster than scalar CPU
  at sufficient parallelism.
- M3+ (Metal 3.1): Native 64-bit integer multiply. Best target.
- Fallback: 8x32-bit limb representation for older devices. Can be added
  later as a compile-time variant.

**Key risk:** Montgomery mul performance. If too slow on M1/M2, consider
Karatsuba or radix-2^29 representation to reduce multiply count.

**Estimated effort:** ~1 week

---

### Phase 2: Buffer Management and Element-wise Operations

Wire up `MetalBuffer<T>` to real Metal allocations and implement the
simple element-wise operations.

**Deliverables:**
- `src/buffer.rs` — Complete `MetalBuffer<T>` implementation
  - `upload`: `device.newBufferWithBytes(data, .storageModeShared)`
  - `download`: `buffer.contents()` pointer cast + `slice::from_raw_parts` + copy
  - `alloc`: `device.newBufferWithLength(len * size_of::<T>(), .storageModeShared)`
  - `len` / `is_empty`: trivial accessors
  - Shared memory mode (Apple Silicon UMA — no explicit CPU↔GPU copy needed,
    just synchronization)
- `src/shaders/elementwise.metal` — Simple compute shaders
  - `kernel_sum` — parallel reduction to scalar
  - `kernel_dot_product` — parallel fused multiply-reduce
  - `kernel_scale` — `buf[i] *= scalar`
  - `kernel_add` — `out[i] = a[i] + b[i]`
  - `kernel_sub` — `out[i] = a[i] - b[i]`
  - `kernel_accumulate` — `buf[i] += scalar * other[i]`
  - All use `Fr` type from Phase 1 and delayed reduction from wide accumulator
- `src/device.rs` — Dispatch helpers
  - `dispatch_1d(pipeline, buf_len)` — choose threadgroup size, encode, commit
  - Command buffer lifecycle: create, encode, commit, waitUntilCompleted
  - Reusable command buffer pool (avoid per-dispatch allocation)
- **ComputeBackend methods completed:** `upload`, `download`, `alloc`, `len`,
  `is_empty`, `sum`, `dot_product`, `scale`, `add`, `sub`, `accumulate`
- **Parity tests:** Every operation tested against CpuBackend on random inputs

**Key design decision:** Use `.storageModeShared` (not `.storageModePrivate`).
On Apple Silicon UMA, shared buffers are directly accessible from both CPU
and GPU with zero-copy. Private buffers would require explicit blit commands.
For our workload (frequent small readbacks of ~128 bytes per round), shared
mode avoids blit overhead.

**Estimated effort:** ~3 days

---

### Phase 3: Interpolation (Sumcheck Bind Step)

The bind step runs every sumcheck round. Halves all polynomial buffers.
This is bandwidth-bound and embarrassingly parallel — ideal GPU workload.

**Deliverables:**
- `src/shaders/interpolate.metal` — Interpolation compute shaders
  - `kernel_interpolate_pairs_low_to_high`:
    `out[i] = buf[2i] + scalar * (buf[2i+1] - buf[2i])`
  - `kernel_interpolate_pairs_high_to_low`:
    `buf[i] = buf[i] + scalar * (buf[i + half] - buf[i])`
  - `kernel_interpolate_pairs_compact_u8`:
    Same formula but promotes `uint8_t` → `Fr` via `fr_from_u64`
  - Threadgroup size: 256 (one thread per output element)
- `src/device.rs` — Implement `interpolate_pairs`, `interpolate_pairs_inplace`
  - Allocating variant: create output buffer, dispatch, return new buffer
  - In-place variant: dispatch over first half, then truncate length
  - Batch variants: single command buffer encoding all buffers, one commit
- **ComputeBackend methods completed:** `interpolate_pairs`,
  `interpolate_pairs_inplace`, `interpolate_pairs_batch`,
  `interpolate_pairs_batch_inplace`
- **Parity tests:** Random polynomials, bind on both backends, compare

**Performance note:** This is bandwidth-bound (~64 bytes read, ~32 bytes
written per element). On M3 Max at 400 GB/s, a 2^20 buffer (32 MB)
completes in ~0.16 ms. The CPU (Rayon, 12 cores) does the same in ~1 ms.
Expected speedup: 3-6x for large buffers.

**Estimated effort:** ~2 days

---

### Phase 4: Product Table (Eq Polynomial Construction)

Builds the eq weight table on-device, avoiding a 2^n element transfer.

**Deliverables:**
- `src/shaders/product_table.metal` — Iterative doubling shader
  - Start with `[Fr::one()]`
  - Per round `i` (of `n` total): dispatch over `2^i` elements
    - `buf[j] *= (1 - r_i)` for `j < 2^i`
    - `buf[j + 2^i] = buf[j] * r_i` (copy then scale)
  - Requires `n` sequential dispatches (each doubling the active range)
  - Each dispatch is embarrassingly parallel within its active range
- `src/device.rs` — Implement `product_table`
  - Upload `point` array (n field elements, tiny)
  - Run `n` dispatches, doubling active threads each time
  - Return buffer (stays on device)
- **ComputeBackend method completed:** `product_table`
- **Parity test:** Compare against `CpuBackend::product_table` for random points

**Performance note:** The table has 2^n elements. For n=20 (typical),
this is 32 MB. Building on-device avoids uploading 32 MB over the bus.
The iterative doubling requires 20 dispatches but each is trivially
parallel. Total GPU time: ~2 ms vs ~5 ms CPU.

**Estimated effort:** ~1 day

---

### Phase 5: ProductSum Kernel Compilation and pairwise_reduce

The core hot path — ~80% of prover time. Compiles Toom-Cook evaluation
kernels from `KernelDescriptor::ProductSum` to MSL, then dispatches
`pairwise_reduce` as a two-level GPU reduction.

**Deliverables:**
- `src/shaders/toom_cook.metal` — Toom-Cook evaluation in MSL
  - `eval_prod_4(pairs) -> Fr[4]` — D=4 specialization
  - `eval_prod_8(pairs) -> Fr[8]` — D=8 specialization
  - `eval_prod_16(pairs) -> Fr[16]` — D=16 specialization
  - `eval_prod_generic(pairs, D) -> Fr[D]` — fallback
  - Each uses the same balanced binary factoring + extrapolation as
    `jolt-cpu-kernels::toom_cook`, translated to MSL
- `src/compiler.rs` — `compile_product_sum(device, D, num_products) -> MetalKernel<F>`
  - Generates MSL source with hardcoded D and num_products
  - Compiles via `device.makeLibrary(source:)`
  - Extracts `newComputePipelineState(function:)`
  - Caches compiled pipelines by (D, num_products) key
- `src/shaders/reduce.metal` — Two-level reduction shader
  - **Level 1 (per-thread):** At position `i`, read D pairs from all
    input buffers, call `eval_prod_D`, multiply each eval by `weights[i]`,
    accumulate into threadgroup shared memory via wide accumulator
  - **Threadgroup barrier** + tree reduction within shared memory
  - **Level 2 (optional):** If >256 threadgroups, second dispatch reduces
    partial sums. Otherwise, CPU reduces the small array after readback.
- `src/reduction.rs` — Dispatch orchestration
  - Choose threadgroup size (256 default, adjust for occupancy)
  - Choose grid size: `ceil(half / threadgroup_size)` threadgroups
  - Encode input buffers, weight buffer, output buffer as arguments
  - Handle `BindingOrder::LowToHigh` vs `HighToLow` via shader variant
    or runtime uniform
- `src/device.rs` — Implement `pairwise_reduce`, `pairwise_reduce_fixed`
- **Parity tests:** Run 10k random inputs through CpuBackend and
  MetalBackend, verify exact match for D=4,8,16

**Performance analysis:**
- Arithmetic intensity for D=4: ~10 field muls per 256 bytes = 0.04 ops/byte
- This is heavily bandwidth-bound. GPU wins via raw bandwidth:
  - M3 Max: 400 GB/s → 2^20 positions in ~0.16 ms
  - CPU (12 cores, ~50 GB/s): same in ~1.3 ms
- Expected speedup: **3-8x** depending on chip and D value
- Higher D values (16, 32) shift toward compute-bound → more GPU benefit

**Key risk:** Threadgroup shared memory pressure. D=16 with wide
accumulators needs `16 * 48 bytes = 768 bytes` per thread. At 256
threads/group that's 192 KB — exceeds the 32 KB limit. Solution: reduce
threads/group or use register spilling. D=4 is fine (192 bytes/thread).

**Estimated effort:** ~1 week

---

### Phase 6: Custom Kernel Compilation (MSL Codegen from Expr)

The "5th IR visitor" — generates MSL source from `jolt-ir::Expr` trees.
Handles the ~20% of prover time that isn't ProductSum.

**Deliverables:**
- `src/compiler.rs` — `compile_custom(device, expr, challenges, num_inputs, degree) -> MetalKernel<F>`
  - Walk the `Expr` DAG in topological order
  - Emit MSL variable declarations: `Fr v0 = ...; Fr v1 = ...;`
  - Map `Var::Opening(i)` → `fr_add(lo[i], fr_mul(t, fr_sub(hi[i], lo[i])))`
  - Map `Var::Challenge(i)` → `constant Fr c_i` (baked at compile time)
  - Map `Constant(k)` → `fr_from_u64(k)` or precomputed constant
  - Map `Add/Sub/Mul/Neg` → `fr_add/fr_sub/fr_mul/fr_neg` calls
  - Outer loop: `for t in 0..degree+1` (standard grid)
  - Compile the generated MSL → `MTLComputePipelineState`
- `src/reduction.rs` — Same two-level reduction, parameterized by kernel
- `src/device.rs` — Wire into `pairwise_reduce` (already dispatches
  through `CompiledKernel`, just needs Custom pipelines)
- **Parity tests:** For each of the ~16 static claim definitions, compile
  both CPU and Metal kernels, verify on random data

**Design notes:**
- No stack machine on GPU. The Expr DAG is flattened to SSA-style MSL.
  Each `ExprId` maps to a local variable. Children always have lower IDs
  than parents (topological order guaranteed by `ExprArena`), so a single
  forward pass emits valid MSL.
- Challenge values are `constant` buffer parameters, set at dispatch time.
  Changing challenges doesn't recompile the pipeline — only the argument
  buffer changes.
- Common subexpressions are naturally shared (same `ExprId` referenced
  multiple times). MSL compiler handles the rest.

**Estimated effort:** ~3 days

---

### Phase 7: Tensor Reduce (Split-Eq)

Implements the split-eq factored reduction for RA virtual sumchecks.
This is the second most important hot path after ProductSum.

**Deliverables:**
- `src/shaders/tensor_reduce.metal` — 2D dispatch shader
  - Threadgroup dimension maps to `outer_weights` (eq_outer table)
  - Thread dimension maps to `inner_weights` (eq_inner table)
  - Inner weight table loaded into threadgroup shared memory
  - Per-thread: position `(x_out, x_in)` → pair index `x_out * |inner| + x_in`
  - Evaluate kernel, multiply by `outer[x_out] * shared_inner[x_in]`
  - Threadgroup reduction first across inner dimension, then across outer
- `src/device.rs` — Implement `tensor_pairwise_reduce`,
  `tensor_pairwise_reduce_fixed`
- **Parity test:** Verify `tensor_pairwise_reduce` matches
  `CpuBackend::tensor_pairwise_reduce` on random split-eq inputs

**Performance note:** The 2D structure improves GPU utilization vs flat
`pairwise_reduce`. Inner weights in shared memory save bandwidth
(loaded once per threadgroup, used by all threads). For typical
`outer=2^10, inner=2^10`: 1024 threadgroups of 1024 threads — excellent
occupancy on Apple Silicon.

**Estimated effort:** ~2 days

---

### Phase 8: Multi-Kernel Dispatch and Pipeline Optimization

Optimize the end-to-end sumcheck loop: fuse dispatches, reduce
synchronization, and pipeline command buffers.

**Deliverables:**
- `src/device.rs` — Implement `pairwise_reduce_multi`
  - Single command buffer encoding all kernels over the same input data
  - One readback of all results (fused output buffer)
  - Avoids per-kernel command buffer creation and synchronization
- `src/device.rs` — Command buffer pipelining
  - Double-buffering: while GPU executes round N's `pairwise_reduce`,
    CPU processes round N-1's results (Fiat-Shamir, transcript)
  - `interpolate_pairs` for round N+1 dispatched as soon as challenge
    is known, overlapping with CPU work
  - Reduce GPU idle time between rounds
- `src/device.rs` — Implement `interpolate_pairs_batch` and
  `interpolate_pairs_batch_inplace` with fused dispatch
  - All buffers encoded in a single command buffer
- **Benchmark:** Full sumcheck loop (reduce + interpolate + repeat)
  comparing MetalBackend vs CpuBackend on realistic polynomial sizes

**Performance target:** For 2^20 sumcheck with D=4:
- CPU baseline: ~500 ms (Rayon, 12 P-cores on M3 Max)
- Metal target: ~100 ms (5x speedup from bandwidth + parallelism)
- Metal + pipeline target: ~80 ms (overlap hides dispatch latency)

**Estimated effort:** ~3 days

---

### Phase 9: Compact Buffer Promotion and Edge Cases

Handle compact polynomial types (u8, u16, bool) and edge cases.

**Deliverables:**
- `src/shaders/interpolate_compact.metal` — Promotion shaders
  - `kernel_interpolate_pairs_u8`: reads `uint8_t`, promotes to `Fr`,
    interpolates, writes `Fr` output
  - `kernel_interpolate_pairs_u16`: same for `uint16_t`
  - `kernel_interpolate_pairs_bool`: `Fr(0)` or `Fr(1)` promotion
  - After first bind, buffer is `Fr` type — subsequent rounds use the
    standard Fr interpolation shader
- `src/device.rs` — Handle `interpolate_pairs<T, F>` where `T != F`
  - Dispatch compact shader for first bind
  - Return `MetalBuffer<F>` (type promotion at the buffer level)
- Edge cases:
  - Buffer length < threadgroup size: single threadgroup dispatch
  - Buffer length = 1: skip dispatch, return element directly
  - Empty buffer: return empty buffer
  - Very large buffers (>2^24): multi-pass reduction to avoid
    exceeding Metal's max threadgroup count
- **Parity tests:** Compact u8 and bool buffers through full bind chain

**Estimated effort:** ~2 days

---

### Phase 10: Integration Testing and Benchmarks

End-to-end validation against CpuBackend and performance benchmarks.

**Deliverables:**
- `tests/parity.rs` — Comprehensive parity test suite
  - Every `ComputeBackend` method tested with random data
  - Multiple field element counts: 2^10, 2^15, 2^20, 2^24
  - Both BindingOrder variants
  - All D values: 4, 8, 16, 32
  - Compact buffer types: u8, u16, bool
  - Edge cases: empty, single element, non-power-of-two
- `benches/metal_vs_cpu.rs` — Criterion benchmarks
  - `pairwise_reduce` for D=4,8,16 at sizes 2^14 through 2^22
  - `interpolate_pairs` at same sizes
  - `product_table` for n=10 through n=22
  - `tensor_pairwise_reduce` with balanced split at 2^20
  - `sum` and `dot_product` at various sizes
  - Full simulated sumcheck round (reduce + interpolate + repeat)
- `tests/sumcheck_loop.rs` — Full sumcheck simulation
  - Build random polynomial, run 20 rounds of reduce+bind
  - Verify final evaluation matches on both backends

**Estimated effort:** ~2 days

---

## Phase Summary

| Phase | What | Methods Completed | Effort |
|-------|------|-------------------|--------|
| 1 | BN254 Fr in MSL + wide accumulator | (foundation) | ~1 week |
| 2 | Buffer management + element-wise ops | upload/download/alloc/len/sum/dot/scale/add/sub/accumulate | ~3 days |
| 3 | Interpolation | interpolate_pairs (all variants) | ~2 days |
| 4 | Product table | product_table | ~1 day |
| 5 | ProductSum kernels + pairwise_reduce | pairwise_reduce, pairwise_reduce_fixed, compile (ProductSum) | ~1 week |
| 6 | Custom kernel MSL codegen | compile (Custom) | ~3 days |
| 7 | Tensor reduce (split-eq) | tensor_pairwise_reduce (all variants) | ~2 days |
| 8 | Multi-kernel + pipeline | pairwise_reduce_multi, command buffer pipelining | ~3 days |
| 9 | Compact buffers + edge cases | interpolate_pairs<u8/bool, F> | ~2 days |
| 10 | Integration tests + benchmarks | (validation) | ~2 days |

**Total estimated effort: ~5 weeks**

Each phase produces a testable increment. Parity tests against
CpuBackend are the primary correctness gate — every Metal result must
match CPU exactly (bit-for-bit, not approximate).

## Performance Expectations

Conservative estimates for M3 Max (400 GB/s memory bandwidth, 12 P-cores):

| Operation | CPU (Rayon) | Metal | Speedup |
|-----------|-------------|-------|---------|
| pairwise_reduce D=4, 2^20 | ~3 ms | ~0.5 ms | 6x |
| pairwise_reduce D=16, 2^20 | ~8 ms | ~1 ms | 8x |
| interpolate_pairs 2^20 | ~1 ms | ~0.2 ms | 5x |
| product_table n=20 | ~5 ms | ~2 ms | 2.5x |
| Full sumcheck round 2^20 | ~12 ms | ~2 ms | 6x |

Actual numbers depend on Metal compiler quality for BN254 arithmetic.
Phase 1 benchmarks will validate or update these estimates.

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
