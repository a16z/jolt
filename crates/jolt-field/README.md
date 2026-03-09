# jolt-field

Field abstractions for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines the core `Field` trait and associated types used throughout the Jolt proving system. It provides a backend-agnostic interface over prime-order scalar fields, currently implemented for the BN254 scalar field (`Fr`).

The crate also provides optimized arithmetic primitives -- wide accumulators, fused multiply-add, and specialized challenge types for Fiat-Shamir batching -- all tuned for the BN254 field.

## Public API

### Core Traits

- **`Field`** -- Prime field element abstraction. Elements are `Copy`, thread-safe, serializable. Provides conversions from integer types, random sampling, square, inverse, and bit-width queries.
- **`Challenge<F>`** -- A Fiat-Shamir challenge value combinable with field elements via `Add`, `Sub`, `Mul`.
- **`WithChallenge`** -- Associates a `Field` with its default `Challenge` type.
- **`OptimizedMul<Rhs, Output>`** -- Multiplication with fast-path short-circuits for zero and one.

### Accumulation

- **`FieldAccumulator`** -- Trait for accumulators that defer modular reduction across multiply-add steps.
- **`NaiveAccumulator`** -- Simple accumulator using standard field arithmetic (no deferred reduction).
- **`WideAccumulator`** -- BN254-specific accumulator using unreduced wide limbs for deferred reduction.
- **`Limbs<N>`** -- Fixed-size limb array type used by wide arithmetic internals.

### Types

- **`Fr`** -- BN254 scalar field element (`#[repr(transparent)]` newtype over `ark_bn254::Fr`).
- **`MontU128Challenge<F>`** -- Default challenge type (125-bit for cheaper multiplication).
- **`Mont254BitChallenge<F>`** -- Full 254-bit challenge wrapping a field element directly.
- **`DefaultChallenge<F>`** -- Type alias resolving to `MontU128Challenge` or `Mont254BitChallenge` based on feature flags.

### Signed Integer Types

The `signed` module provides fixed-width signed big integers (`S64`, `S128`, `S192`, `S256`, etc.) with truncating arithmetic.

## Dependency Position

`jolt-field` is a **leaf crate** with no internal Jolt dependencies. All other `jolt-*` crates depend on it.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `bn254` | **Yes** | Enable BN254 scalar field implementation |
| `dory-pcs` | No | Enable Dory PCS interop (implies `bn254`) |
| `challenge-254-bit` | No | Use full 254-bit challenges instead of 125-bit |
| `allocative` | No | Enable memory profiling via the `allocative` crate |

## License

MIT
