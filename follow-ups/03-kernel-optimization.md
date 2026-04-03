# Kernel Optimization

## What

Optimize the compute kernels across all backends (CPU, Metal, and future backends) for the operations that dominate proving time: sumcheck evaluation, polynomial binding, MSM, and NTT.

## Why

The prover is schedule-driven — it walks a sequence of `Op`s and dispatches compute to the backend via compiled kernels. Kernel performance directly determines proving time. The `ComputeBackend` abstraction means kernel optimizations are backend-local and don't affect protocol correctness.

## Scope

**CPU (`jolt-cpu`):**
- Sumcheck inner loop: `evaluate` and `bind` on `CpuKernel`. Currently uses Toom-Cook for degree-3+ compositions. Profile whether Karatsuba or schoolbook is faster for the actual degree distributions in Jolt (most compositions are degree 2–3).
- Domain iteration (uniskip): Lagrange interpolation + extended evaluation. The `evaluate_domain_kernel` function has a parallel reduction that could benefit from better chunking.
- Polynomial binding: `bind_high` / `bind_low` on dense and compact polynomials. These are memory-bound — focus on cache-friendly access patterns.
- MSM: Currently in `jolt-crypto`. Pippenger with optimal bucket sizing for the commitment sizes in Jolt.

**Metal (`jolt-metal`):**
- GPU kernel launch overhead vs. compute time tradeoff. Small sumcheck rounds may be faster on CPU. The `jolt-hybrid` combinator exists for this — tune the crossover threshold.
- Memory transfer minimization: keep polynomial data on GPU across rounds, only transfer round polynomial coefficients back.
- MSL codegen from `KernelSpec` — the compiler already generates Metal shaders. Optimize the generated code (shared memory usage, threadgroup sizing).

**General:**
- Profile the full pipeline on real programs (sha2, sha3, fibonacci) to identify which kernels dominate.
- The profiling infrastructure (`jolt-profiling`) with Chrome trace + Perfetto support already exists. Use it to measure before/after.

## How It Fits

Each backend implements `ComputeBackend::compile(spec) → CompiledKernel` and `ComputeBackend::execute(kernel, inputs, output)`. Optimizations are entirely within these implementations. The runtime (`jolt-zkvm/src/runtime.rs`) and protocol structure are unaffected.

## Dependencies

None — this is a parallel track. Can start immediately on the CPU backend.

## Unblocks

Nothing directly, but proving time improvements affect all downstream work (ZK, wrapping, etc.) since they make iteration faster.

## Open Questions

- What's the target hardware profile? Optimizing for server (many cores, large L3) vs. laptop (fewer cores, smaller cache) vs. mobile (ARM, small memory) leads to different choices.
- Should we have backend-specific `CompileParams` that let the compiler make backend-aware decisions (e.g., batch sizes, parallelism hints)?
