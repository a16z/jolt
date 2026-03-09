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
- **`IdentityPolynomial`** -- Maps hypercube points to their integer index.

### Binding and Evaluation

- **`BindingOrder`** -- Controls the order in which variables are bound during sumcheck (MSB-first vs LSB-first).
- **`EvaluationSource`** -- Abstraction over polynomial evaluation data sources.
- **`RlcSource`** -- Source for random linear combination evaluation.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `parallel` | **Yes** | Enable rayon parallelism for eq table construction and polynomial operations |

## License

MIT
