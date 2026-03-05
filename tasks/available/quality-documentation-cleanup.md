# quality-cleanup-refactor: Documentation and Code Quality

**Scope:** all new crates

**Depends:** all implementation and testing tasks

**Verifier:** ./verifiers/scoped.sh /workdir

**Context:**

Perform final quality improvements across all new crates. This includes documentation, code formatting, linting, and creating comprehensive README files for each crate.

### Tasks

#### 1. Documentation Pass

For each crate, ensure:

**Module-level documentation:**
```rust
//! # jolt-poly
//!
//! This crate provides polynomial types and operations for multilinear,
//! univariate, and specialized polynomials used in the Jolt zkVM.
//!
//! ## Overview
//!
//! The crate is designed to be backend-agnostic and reusable outside
//! of Jolt. It provides efficient implementations optimized for the
//! sumcheck protocol and polynomial commitment schemes.
//!
//! ## Main Types
//!
//! - [`DensePolynomial`]: Full field-element coefficient storage
//! - [`CompactPolynomial`]: Memory-efficient small scalar storage
//! - [`EqPolynomial`]: Equality polynomial implementation
//! - [`UnivariatePoly`]: Univariate polynomials in coefficient form
//!
//! ## Examples
//!
//! ```rust
//! use jolt_poly::{DensePolynomial, MultilinearPolynomial};
//! use ark_bn254::Fr;
//!
//! let poly = DensePolynomial::<Fr>::random(4, &mut rng);
//! let point = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
//! let eval = poly.evaluate(&point);
//! ```
```

**Item-level documentation with LaTeX math:**
```rust
/// Multilinear polynomial evaluation via multilinear extension.
///
/// For a polynomial $f$ with evaluations over the Boolean hypercube $\{0,1\}^n$,
/// the multilinear extension at point $r \in \mathbb{F}^n$ is:
///
/// $$\tilde{f}(r) = \sum_{x \in \{0,1\}^n} f(x) \cdot \widetilde{eq}(x, r)$$
///
/// where $\widetilde{eq}$ is the equality polynomial.
///
/// # Arguments
///
/// * `point` - The evaluation point $r \in \mathbb{F}^n$
///
/// # Returns
///
/// The evaluation $\tilde{f}(r)$
///
/// # Panics
///
/// Panics if `point.len() != self.num_vars()`
fn evaluate(&self, point: &[F]) -> F {
    // ...
}
```

#### 2. README Files

Create comprehensive README.md for each crate:

**Template for crates/jolt-poly/README.md:**
```markdown
# jolt-poly

Polynomial types and operations for the Jolt zkVM.

## Overview

This crate provides efficient polynomial implementations optimized for use in
sumcheck protocols and polynomial commitment schemes. It is designed to be
backend-agnostic and can be used as a standalone dependency.

## Features

- **Dense polynomials**: Full coefficient storage for general use
- **Compact polynomials**: Memory-efficient storage for small scalars
- **Specialized polynomials**: Equality, identity, and Lagrange polynomials
- **Univariate polynomials**: Support for interpolation and evaluation

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
jolt-poly = { git = "https://github.com/a16z/jolt" }
```

### Basic Example

```rust
use jolt_poly::{DensePolynomial, MultilinearPolynomial};
use ark_bn254::Fr as TestField;

// Create a random polynomial with 4 variables
let poly = DensePolynomial::<TestField>::random(4, &mut rand::thread_rng());

// Evaluate at a point
let point = vec![TestField::from(1), TestField::from(2),
                 TestField::from(3), TestField::from(4)];
let result = poly.evaluate(&point);

// Bind the first variable
let bound_poly = poly.bind(TestField::from(5));
assert_eq!(bound_poly.num_vars(), 3);
```

### Memory-Efficient Polynomials

For polynomials with small integer evaluations:

```rust
use jolt_poly::CompactPolynomial;

// Store as u8 values, evaluate as field elements
let values: Vec<u8> = vec![0, 1, 1, 0, 1, 0, 0, 1];
let compact = CompactPolynomial::<u8, TestField>::new(values);

// Same API as DensePolynomial
let eval = compact.evaluate(&point);
```

## Performance

The crate is optimized for:

- Fast binding operations (critical for sumcheck)
- Memory efficiency via compact representations
- Cache-friendly evaluation order
- Parallel operations via `rayon` (with `parallel` feature)

## Documentation

For detailed API documentation, run:

```bash
cargo doc --open -p jolt-poly
```

For the mathematical background, see the [Jolt Book](https://jolt.a16zcrypto.com/).

## License

This project is licensed under the MIT License.
```

#### 3. Clippy Fixes

Run clippy with strict settings and fix all warnings:

```bash
#!/bin/bash
# Run on each crate
for crate in jolt-poly jolt-openings jolt-sumcheck jolt-spartan \
             jolt-instructions jolt-dory jolt-zkvm; do
    echo "Running clippy on $crate..."
    cargo clippy -p $crate --all-features -- \
        -D warnings \
        -D clippy::all \
        -D clippy::pedantic \
        -W clippy::nursery \
        -A clippy::module_name_repetitions \
        -A clippy::must_use_candidate \
        -A clippy::missing_errors_doc
done
```

Common fixes needed:

```rust
// Before: Unnecessary clone
let poly_clone = poly.clone();

// After: Take reference
let poly_ref = &poly;

// Before: Inefficient string creation
format!("Error: {}", msg)

// After: Use Display impl
msg.to_string()

// Before: Manual implementation of default
impl Foo {
    fn new() -> Self {
        Self { field: 0 }
    }
}

// After: Derive Default
#[derive(Default)]
struct Foo {
    field: u32,
}
```

#### 4. Format All Code

Ensure consistent formatting:

```bash
# Format all crates
cargo fmt --all

# Check formatting in CI
cargo fmt --all -- --check
```

#### 5. API Consistency

Ensure consistent API patterns across crates:

**Constructor naming:**
```rust
// Consistent pattern:
impl Foo {
    pub fn new(params: Params) -> Self { ... }
    pub fn with_capacity(cap: usize) -> Self { ... }
    pub fn from_vec(vec: Vec<T>) -> Self { ... }
}
```

**Error handling:**
```rust
// Each crate has its own error type
#[derive(Debug, thiserror::Error)]
pub enum JoltPolyError {
    #[error("polynomial size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },
    // ...
}

// Consistent Result type alias
pub type Result<T> = std::result::Result<T, JoltPolyError>;
```

**Trait bounds:**
```rust
// Consistent ordering: sized types, then traits, then lifetimes
fn foo<T, F>(x: T) -> F
where
    T: Clone + Send + Sync + 'static,
    F: Field,
{
    // ...
}
```

#### 6. Remove Dead Code

Identify and remove unused code:

```bash
# Find unused dependencies
cargo machete

# Find dead code
cargo +nightly rustc -- -D dead_code
```

#### 7. Improve Test Organization

Ensure tests are well-organized:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    mod dense_polynomial {
        use super::*;

        #[test]
        fn test_new() { ... }

        #[test]
        fn test_evaluate() { ... }
    }

    mod compact_polynomial {
        use super::*;

        #[test]
        fn test_conversion() { ... }
    }
}
```

#### 8. Add Benchmarks

Create benchmark files for performance-critical operations:

**crates/jolt-poly/benches/polynomial.rs:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jolt_poly::{DensePolynomial, MultilinearPolynomial};

fn benchmark_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_evaluate");

    for num_vars in [4, 8, 12, 16] {
        let poly = DensePolynomial::<TestField>::random(num_vars, &mut rng);
        let point: Vec<_> = (0..num_vars).map(|_| TestField::random(&mut rng)).collect();

        group.bench_function(format!("evaluate_{}_vars", num_vars), |b| {
            b.iter(|| {
                black_box(poly.evaluate(black_box(&point)))
            })
        });
    }

    group.finish();
}

fn benchmark_bind(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_bind");

    for num_vars in [4, 8, 12, 16] {
        let poly = DensePolynomial::<TestField>::random(num_vars, &mut rng);
        let scalar = TestField::random(&mut rng);

        group.bench_function(format!("bind_{}_vars", num_vars), |b| {
            b.iter(|| {
                black_box(poly.clone().bind(black_box(scalar)))
            })
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_evaluate, benchmark_bind);
criterion_main!(benches);
```

#### 9. CI Configuration

Add GitHub Actions workflow for the new crates:

**.github/workflows/refactored-crates.yml:**
```yaml
name: Refactored Crates CI

on:
  push:
    branches: [ main, refactor/* ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
      with:
        components: rustfmt, clippy

    - name: Format check
      run: cargo fmt --all -- --check

    - name: Clippy
      run: |
        cargo clippy --workspace --all-features -- -D warnings

    - name: Test
      run: cargo nextest run --workspace

    - name: Doc test
      run: cargo test --doc --workspace

  benchmarks:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable

    - name: Run benchmarks
      run: |
        for crate in jolt-poly jolt-sumcheck jolt-spartan; do
          cargo bench -p $crate --no-run
        done
```

#### 10. Crate Metadata

Update each crate's Cargo.toml with proper metadata:

```toml
[package]
name = "jolt-poly"
version = "0.1.0"
authors = ["a16z Crypto"]
edition = "2021"
description = "Polynomial types and operations for the Jolt zkVM"
repository = "https://github.com/a16z/jolt"
license = "MIT"
keywords = ["polynomial", "multilinear", "sumcheck", "zkvm"]
categories = ["cryptography", "mathematics"]

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

### Current Progress

| Subtask | Status | Notes |
|---------|--------|-------|
| README.md for each crate | Done | All 8 crates have READMEs |
| Clippy clean | Done | All crates pass `cargo clippy -D warnings` |
| Formatting | Done | All crates pass `cargo fmt --check` |
| Benchmarks | Mostly done | 5/8 crates have benchmarks (see `quality-benchmark-suite`) |
| Module-level docs (`//!`) | Partial | Some crates have them, not all |
| Item-level doc comments | Partial | Public API docs added to `jolt-openings` accumulators; other crates need review |
| LaTeX math in docs | Not started | |
| CI configuration | Partial | `rust.yml` updated; dedicated refactored-crates workflow not yet created |
| Crate metadata | Not started | `Cargo.toml` metadata (authors, description, keywords, categories) |
| Dead code removal | Not started | |

**Note:** The code samples in sections 1–2 reference outdated type names (`DensePolynomial`, `CompactPolynomial`, `SmallScalar`). The actual API uses `Polynomial<T>` with `DensePolynomial<F>` as a type alias and no `SmallScalar` trait. Update samples if using this task as a reference.

### Acceptance Criteria

- All crates have comprehensive module and item documentation
- LaTeX math notation used where appropriate
- README.md created for each crate
- All clippy warnings fixed
- Consistent code formatting
- Dead code removed
- Well-organized test structure
- Benchmarks added for key operations
- CI configuration updated
- Proper crate metadata