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

## Feature Flags

This crate has no feature flags.

## License

MIT
