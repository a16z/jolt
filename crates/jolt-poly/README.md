# jolt-poly

Polynomial types and operations for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate provides multilinear and univariate polynomial representations used throughout Jolt's sumcheck-based proving system. Polynomials are represented by their evaluations over the Boolean hypercube, with specialized compact representations for small-scalar coefficients.

## Public API

### Core Traits

- **`MultilinearPolynomial<F>`** — A multilinear polynomial in `n` variables, represented by its `2^n` evaluations over the Boolean hypercube `{0,1}^n`. Provides `evaluate`, `bind` (fix a variable), and `evaluations` (materialized table).

- **`UnivariatePolynomial<F>`** — Shared interface for univariate polynomial types (`UnivariatePoly`, `CompressedPoly`). Provides `degree()`. Evaluation and coefficient access are inherent methods because the compressed form requires an external hint.

### Polynomial Types

- **`Polynomial<F>`** — Full evaluation table stored as `Vec<F>`. Supports in-place variable binding, random generation, and serde. This is the workhorse type for sumcheck.

- **`CompactPolynomial<T, F>`** — Stores evaluations as small scalars (`T: SmallScalar` — `bool`, `u8`, `u16`, `u32`, `u64`, `i64`, `i128`, `u128`) and promotes to field elements on demand. Reduces memory by up to 32x compared to `Polynomial`.

- **`UnivariatePoly<F>`** — Coefficient-form univariate polynomial. Supports evaluation, Lagrange interpolation, degree queries, and compression via `compress()`. Used for sumcheck round polynomials. Implements `UnivariatePolynomial`.

- **`CompressedPoly<F>`** — Compressed univariate polynomial with the linear term omitted. Stores `[c0, c2, c3, ...]` to save one field element per sumcheck round in proof serialization. Supports `evaluate_with_hint(hint, point)` and `decompress(hint)` to recover the full polynomial. Implements `UnivariatePolynomial`.

- **`EqPolynomial<F>`** — The equality polynomial `eq(x, r) = prod_i (r_i * x_i + (1 - r_i)(1 - x_i))`. Materializes all `2^n` evaluations via a bottom-up doubling construction. Fundamental to sumcheck evaluation.

- **`IdentityPolynomial`** — Evaluates to the integer index on the Boolean hypercube: `I(b) = sum_i b_i * 2^(n-1-i)`. Used for indexed lookups.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `parallel` | **Yes** | Enable rayon parallelism for eq table construction and polynomial operations |

## License

MIT
