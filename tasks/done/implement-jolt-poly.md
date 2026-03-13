# impl-jolt-poly: Clean-room implementation of jolt-poly

**Scope:** crates/jolt-poly/

**Depends:** scaffold-workspace

**Verifier:** ./verifiers/scoped.sh /workdir jolt-poly

**Context:**

Implement the `jolt-poly` crate — generic polynomial types and operations for multilinear, univariate, and specialized polynomials. This crate is backend-agnostic and reusable outside Jolt.

**This is a clean-room rewrite.** Study `jolt-core/src/poly/` for algorithmic reference, but design the API and write the code from scratch. Do not copy code from jolt-core.

**Dependency:** `jolt-field` (the `Field` trait and its associated traits).

### Reference material

The old code lives in `jolt-core/src/poly/` (22 files, ~824 KB). Key files to study for algorithmic understanding:

- `jolt-core/src/poly/dense.rs` — dense polynomial with field-element evaluations
- `jolt-core/src/poly/compact.rs` — compressed storage for small scalars
- `jolt-core/src/poly/eq_poly.rs` — equality polynomial
- `jolt-core/src/poly/unipoly.rs` — univariate polynomials
- `jolt-core/src/poly/identity.rs` — identity polynomial
- `jolt-core/src/poly/lagrange.rs` — Lagrange basis
- `jolt-core/src/utils/small_scalar.rs` — SmallScalar trait

Also read the Jolt Book sections on multilinear extensions: https://jolt.a16zcrypto.com/

### Public API contract

```rust
/// A multilinear polynomial over `F` in `num_vars` variables,
/// represented by its evaluations over the Boolean hypercube {0,1}^n.
pub trait MultilinearPolynomial<F: Field>: Send + Sync {
    fn num_vars(&self) -> usize;
    fn len(&self) -> usize;
    fn evaluate(&self, point: &[F]) -> F;
    fn bind(&self, scalar: F) -> DensePolynomial<F>;
    fn evaluations(&self) -> Cow<[F]>;
}

/// Dense multilinear polynomial: stores all 2^n evaluations as Vec<F>.
pub struct DensePolynomial<F: Field> {
    evaluations: Vec<F>,
    num_vars: usize,
}

impl<F: Field> DensePolynomial<F> {
    pub fn new(evaluations: Vec<F>) -> Self;
    pub fn zeros(num_vars: usize) -> Self;
    pub fn random(num_vars: usize, rng: &mut impl RngCore) -> Self;
    pub fn bind_in_place(&mut self, scalar: F);
    pub fn evaluate_and_consume(self, point: &[F]) -> F;
    pub fn evaluations_mut(&mut self) -> &mut [F];
}

/// Compact multilinear polynomial: stores small scalars (u8, u16, etc.)
/// and converts to F on demand.
pub struct CompactPolynomial<S: SmallScalar, F: Field> { ... }

/// Trait for scalar types that can be stored compactly and promoted to F.
pub trait SmallScalar: Copy + Send + Sync + Into<u64> + 'static {
    fn from_field<F: Field>(f: F) -> Option<Self>;
}

/// Equality polynomial: eq(x, r) = prod_i (r_i * x_i + (1 - r_i)(1 - x_i)).
pub struct EqPolynomial<F: Field> { point: Vec<F> }

impl<F: Field> EqPolynomial<F> {
    pub fn new(point: Vec<F>) -> Self;
    pub fn evaluations(&self) -> Vec<F>;
    pub fn evaluate(&self, point: &[F]) -> F;
}

/// Univariate polynomial in coefficient form.
pub struct UnivariatePoly<F: Field> { coefficients: Vec<F> }

impl<F: Field> UnivariatePoly<F> {
    pub fn new(coefficients: Vec<F>) -> Self;
    pub fn degree(&self) -> usize;
    pub fn evaluate(&self, point: F) -> F;
    pub fn interpolate(points: &[(F, F)]) -> Self;
}

/// Identity polynomial.
pub struct IdentityPolynomial { num_vars: usize }

/// Lagrange basis polynomial.
pub struct LagrangePolynomial<F: Field> { ... }
```

### Implementation notes

- `DensePolynomial::evaluate` uses the standard multilinear extension formula: iterate over the boolean hypercube, weight each evaluation by the product of `(1 - r_i)` or `r_i` terms. Use rayon `par_iter` for the inner loop when the `parallel` feature is enabled.
- `bind_in_place` halves the evaluation array: `new[i] = old[2i] * (1 - scalar) + old[2i+1] * scalar`. This is the core hot path — must be vectorizable.
- `EqPolynomial::evaluations` builds the full 2^n table bottom-up: start with `[1]`, then for each variable multiply each entry by `r_i` or `(1 - r_i)`.
- `CompactPolynomial` stores `Vec<S>` where `S: SmallScalar` and promotes to `F` lazily in `evaluations()` and `evaluate()`.
- `UnivariatePoly::interpolate` uses Lagrange interpolation. `evaluate` uses Horner's method.
- All types implement `Serialize`/`Deserialize` via serde.

### File structure

```
jolt-poly/src/
├── lib.rs              # Re-exports, module declarations
├── traits.rs           # MultilinearPolynomial trait
├── dense.rs            # DensePolynomial
├── compact.rs          # CompactPolynomial + SmallScalar
├── eq.rs               # EqPolynomial
├── univariate.rs       # UnivariatePoly
├── identity.rs         # IdentityPolynomial
└── lagrange.rs         # LagrangePolynomial
```

### Documentation standard

- Rustdoc on every public item
- LaTeX math in doc comments (e.g., `$\widetilde{eq}(x, r) = \prod_{i=1}^{n}(r_i x_i + (1-r_i)(1-x_i))$`)
- Reference the Jolt Book where relevant

**Acceptance:**

- All types and traits from the API contract above are implemented
- `MultilinearPolynomial` trait implemented for `DensePolynomial` and `CompactPolynomial`
- `EqPolynomial` produces correct evaluations (test: sum over hypercube == 1)
- `UnivariatePoly::interpolate` round-trips through `evaluate`
- `bind` + sequential evaluation == direct evaluation (algebraic identity)
- All types are `Serialize + Deserialize`
- `parallel` feature enables rayon in hot paths
- No file exceeds 500 lines
- Rustdoc on all public items with LaTeX math
- `cargo clippy` clean
- Basic unit tests inline in each source file
