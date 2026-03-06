# jolt-field

Field abstractions for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines the core `Field` trait and associated types used throughout the Jolt proving system. It provides a backend-agnostic interface over prime-order scalar fields, currently implemented for the BN254 scalar field (`Fr`).

The crate also provides optimized arithmetic primitives — Barrett and Montgomery reduction, fused multiply-add accumulators, and specialized challenge types for Fiat-Shamir batching — all tuned for the BN254 field.

## Public API

### Core Traits

- **`Field`** — Prime field element abstraction. All arithmetic is modular over the field's prime order. Elements are `Copy`, thread-safe, and serializable. Provides conversions from integer types, random sampling, square, inverse, and bit-width queries.

- **`UnreducedOps`** — Exposes unreduced (wider-than-field) multiplication results. In Montgomery form, a product of two 4-limb elements produces up to 8 limbs before reduction. This trait gives access to those raw limbs so that `FMAdd` accumulators can defer reduction across many multiply-add steps.

- **`ReductionOps`** — Converts wide limb representations back to field elements via Montgomery REDC or Barrett reduction.

- **`Challenge<F>`** — A Fiat-Shamir challenge value that can be combined with field elements via `Add`, `Sub`, `Mul`.

- **`WithChallenge`** — Associates a `Field` with its default `Challenge` type, selected at compile time via the `challenge-254-bit` feature flag.

- **`OptimizedMul<Rhs, Output>`** — Multiplication with fast-path short-circuits for zero and one. Used in sumcheck hot loops where many evaluations multiply by 0 or 1.

### Accumulation Traits

- **`FMAdd<Left, Right>`** — Fused multiply-add without intermediate reduction. Accumulates products into a wide integer, deferring the expensive modular reduction until the end.

- **`BarrettReduce<F>`** / **`MontgomeryReduce<F>`** — Finalize a wide accumulator back to a field element.

### Types

- **`Fr`** — BN254 scalar field element. A `#[repr(transparent)]` newtype over `ark_bn254::Fr` with native serde support. This is the concrete field used throughout Jolt.

- **`MontU128Challenge<F>`** — Default challenge type. Restricts challenges to 125 bits for cheaper multiplication via a sparse Montgomery path.

- **`Mont254BitChallenge<F>`** — Full 254-bit challenge wrapping a field element directly. Simpler but slower. Enabled with the `challenge-254-bit` feature.

- **`DefaultChallenge<F>`** — Type alias that resolves to `MontU128Challenge` or `Mont254BitChallenge` based on feature flags.

### Signed Integer Types

The `signed` module provides fixed-width signed big integers (`S64`, `S128`, `S192`, `S256`, `S96`, `S160`, `S224`) with truncating arithmetic, used internally for accumulator bookkeeping.

## Dependency Position

`jolt-field` is a **leaf crate** with no internal Jolt dependencies. All other `jolt-*` crates depend on it.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `challenge-254-bit` | No | Use full 254-bit challenges instead of 125-bit |
| `allocative` | No | Enable memory profiling via the `allocative` crate |

## License

MIT
