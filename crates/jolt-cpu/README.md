# jolt-cpu

CPU compute backend and kernel compiler for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate compiles `jolt-compiler` composition formulas into `jolt-compute` CPU kernels. It bridges the symbolic formula representation (field-agnostic) and the concrete CPU backend (field-specific closures).

### Compilation Strategies

- **`ProductSum` D=4,8,16,32** — Hand-optimized closures with fully unrolled product evaluation. These cover ~80% of prover time (instruction RA sumchecks and claim reductions).
- **`ProductSum` generic** — Loop-based fallback for other D values.
- **`EqProduct`** — Hand-coded kernel for `eq * g` (degree 2, 2 inputs).
- **`HammingBooleanity`** — Hand-coded kernel for `eq * h * (h-1)` (degree 3, 2 inputs).
- **`Custom`** — The `Expr` is walked once at compile time to produce a stack-machine closure that evaluates the expression at each grid point.

## Public API

- **`compile<F>(desc) -> CpuKernel<F>`** — Compiles a `KernelDescriptor` into a CPU kernel. Dispatches to the appropriate strategy based on the descriptor's shape and degree.
- **`compile_with_challenges<F>(desc, challenges) -> CpuKernel<F>`** — Like `compile`, but bakes Fiat-Shamir-derived challenge values into `Custom` expression kernels.
- **`pub mod toom_cook`** — Re-exported Toom-Cook evaluation functions (`eval_prod_4`, `eval_prod_8`, etc.) used by both this crate and `jolt-core`.

## Dependency Position

```
jolt-field    ─┐
jolt-compiler ─┼─► jolt-cpu
jolt-compute  ─┘
```

Used by `jolt-sumcheck` and `jolt-zkvm`.

## Benchmarks

BN254 Fr on Apple Silicon (M-series). Run with `cargo bench -p jolt-cpu`.

### `ProductSum` kernel — single pair evaluation

| D (degree) | P=1 | P=4 |
|------------|-----|-----|
| 4 | 1.9 Melem/s (530 ns) | 3.1 Melem/s (5.2 µs) |
| 8 | 3.7 Melem/s (2.1 µs) | 3.8 Melem/s (8.4 µs) |
| 16 | 2.0 Melem/s (8.0 µs) | 2.0 Melem/s (32 µs) |

### `Custom` kernel — IR-compiled single pair evaluation

| Expression | Throughput | Time |
|------------|------------|------|
| `x² - x` (booleanity, 1 input) | 3.5 Melem/s | 286 ns |
| `x₀·x₁·x₂·x₃` (product, 4 inputs) | 5.5 Melem/s | 730 ns |

### Direct Rayon vs ComputeBackend — abstraction overhead

Compares the direct Rayon `par_iter().fold().reduce()` pattern (as used in the
witness hot path) against `CpuBackend::pairwise_reduce` with a compiled kernel.
Both use `FieldAccumulator` delayed reduction and the same Toom-Cook kernels.

| D | Size | Direct Rayon | Backend | Overhead |
|---|------|--------------|---------|----------|
| 4 | 2^16 | 14.3 Mpair/s | 13.1 Mpair/s | ~8% |
| 4 | 2^18 | 16.7 Mpair/s | 15.0 Mpair/s | ~10% |
| 4 | 2^20 | 20.3 Mpair/s | 19.1 Mpair/s | ~6% |
| 8 | 2^16 | 5.0 Mpair/s | 4.9 Mpair/s | ~2% |
| 8 | 2^18 | 5.8 Mpair/s | 5.5 Mpair/s | ~5% |
| 8 | 2^20 | 6.4 Mpair/s | 5.7 Mpair/s | ~11% |
| 16 | 2^16 | 1.5 Mpair/s | 1.5 Mpair/s | ~2% |
| 16 | 2^18 | 1.6 Mpair/s | 1.5 Mpair/s | ~2% |
| 16 | 2^20 | 1.6 Mpair/s | 1.5 Mpair/s | ~2% |

**Conclusion:** The backend abstraction adds 2–11% overhead on CPU. The
`CpuKernel::evaluate` signature uses `&mut [F]` output slices (no per-pair
heap allocation), and `lo`/`hi` scratch buffers are hoisted into the fold
initializer. The remaining gap at small D is from `Vec`-based scratch buffers
vs stack-allocated arrays in the direct path. At D≥16 the kernel computation
dominates and overhead is within noise (~2%).

**Remaining path to zero overhead:** A const-generic
`pairwise_reduce<const D: usize>` variant could use fixed-size `[F; D]` scratch
arrays for `lo`/`hi`, eliminating the last heap allocation in the hot loop.

## Feature Flags

This crate has no feature flags.

## License

MIT
