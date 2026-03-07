# jolt-cpu-kernels

CPU kernel compiler for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate compiles `jolt-ir` kernel descriptors into `jolt-compute` CPU kernels. It bridges the symbolic IR (field-agnostic) and the concrete CPU backend (field-specific closures).

### Compilation Strategies

- **`ProductSum` D=4,8,16** — Hand-optimized closures with fully unrolled product evaluation. These cover ~80% of prover time (instruction RA sumchecks and claim reductions).
- **`ProductSum` generic** — Loop-based fallback for other D values.
- **`Custom`** — The `Expr` is walked once at compile time to produce a stack-machine closure that evaluates the expression at each grid point.

## Public API

- **`compile<F>(desc) -> CpuKernel<F>`** — Compiles a `KernelDescriptor` into a CPU kernel. Dispatches to the appropriate strategy based on the descriptor's shape and degree.

## Dependency Position

```
jolt-field ─┐
jolt-ir    ─┼─► jolt-cpu-kernels
jolt-compute ─┘
```

Used by `jolt-sumcheck` and `jolt-zkvm`.

## Benchmarks

BN254 Fr on Apple Silicon (M-series). Run with `cargo bench -p jolt-cpu-kernels`.

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

## Feature Flags

This crate has no feature flags.

## License

MIT
