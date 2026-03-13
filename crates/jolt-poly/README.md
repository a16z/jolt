# jolt-poly

Polynomial types and operations for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate provides multilinear and univariate polynomial representations used throughout Jolt's sumcheck-based proving system. Polynomials are represented by their evaluations over the Boolean hypercube, with specialized compact representations for small-scalar coefficients.

## Public API

### Core Traits

- **`MultilinearEvaluation<F>`** -- Point evaluation interface for multilinear polynomials. Methods: `num_vars()`, `len()`, `evaluate(point)`.
- **`MultilinearBinding<F>`** -- In-place variable binding for sumcheck. Method: `bind(scalar)`.
- **`UnivariatePolynomial<F>`** -- Shared interface for univariate types. Method: `degree()`.

### Polynomial Types

- **`Polynomial<F: Field>`** -- Full evaluation table stored as `Vec<F>`. Supports in-place variable binding, evaluation, arithmetic operators.
- **`Polynomial<T>` (compact mode)** -- When `T` is a small primitive (`bool`, `u8`, etc.), stores evaluations natively and promotes to field on demand. Up to 32x memory reduction.
- **`UnivariatePoly<F>`** -- Coefficient-form univariate polynomial with Lagrange interpolation and compression.
- **`CompressedPoly<F>`** -- Compressed univariate with linear term omitted (saves one field element per sumcheck round).
- **`EqPolynomial<F>`** -- Equality polynomial `eq(x, r)`. Materializes all `2^n` evaluations via bottom-up doubling.
- **`EqPlusOnePolynomial<F>`** -- Successor polynomial `eq+1(x, y)` evaluating to 1 when `y = x + 1`. Used in Spartan shift sumcheck.
- **`EqPlusOnePrefixSuffix<F>`** -- Prefix-suffix decomposition of `eq+1` for sqrt-sized sumcheck buffers.
- **`IdentityPolynomial`** -- Maps hypercube points to their integer index.
- **`LtPolynomial<F>`** -- Less-than polynomial `LT(x, r)` with split optimization for sqrt-sized sumcheck buffers. Used in register/RAM value evaluation.

### Streaming and Sparse Access

- **`MultilinearPoly<F>`** -- Core trait for multilinear polynomial access: evaluation, row iteration, fold, sparsity hints.
- **`RlcSource`** -- Lazy random linear combination of multiple `MultilinearPoly` sources.
- **`OneHotPolynomial`** -- Sparse polynomial where each row has at most one nonzero entry (value 1). Enables O(T) PCS commit via generator lookup.

### Binding and Evaluation

- **`BindingOrder`** -- Controls the order in which variables are bound during sumcheck (MSB-first vs LSB-first).

### Utility Modules

- **`lagrange`** -- Lagrange interpolation, symmetric power sums, polynomial multiplication, and Newton-form interpolation over integer domains.
- **`math`** -- Bit-manipulation utilities on `usize` via the `Math` trait (`pow2`, `log_2`).
- **`thread`** -- Threading utilities: `drop_in_background_thread` (rayon) and `unsafe_allocate_zero_vec` (zero-initialized allocation).

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `parallel` | **Yes** | Enable rayon parallelism for eq table construction and polynomial operations |

## License

MIT
