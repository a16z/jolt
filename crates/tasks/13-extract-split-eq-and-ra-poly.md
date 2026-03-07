# Task 13: Extract GruenSplitEqPolynomial, RaPolynomial, and mles_product_sum to jolt-poly

**Status:** Not started
**Dependencies:** None (these types depend only on jolt-field + existing jolt-poly types)
**Blocks:** S2 (lookup RA virtual sumcheck) in jolt-zkvm

## Objective

Extract three performance-critical polynomial types from `jolt-core/src/poly/` into the `jolt-poly` crate so that `jolt-zkvm` and `jolt-sumcheck` can implement S1/S2 without depending on jolt-core.

## Types to extract

### 1. `GruenSplitEqPolynomial<F>` (~860 lines)

**Source:** `jolt-core/src/poly/split_eq_poly.rs`

Sqrt-decomposed equality polynomial that factors `eq(w, x)` into prefix tables `E_in` and `E_out`, enabling streaming sumcheck computation over windows. Used by both the Spartan outer sumcheck (S1) and the RA virtual sumcheck (S2).

**Public API to preserve:**

| Method | Purpose |
|--------|---------|
| `new(w, binding_order)` | Constructor — parallel `EqPolynomial::evals_cached` for both halves |
| `new_with_scaling(w, binding_order, scaling_factor)` | Constructor with initial scalar absorbed |
| `bind(r)` | Absorb one challenge into `current_scalar`, pop prefix table |
| `E_in_current()` / `E_out_current()` | Active prefix table slices |
| `E_out_in_for_window(window_size)` | Factored eq tables for streaming window head |
| `E_active_for_window(window_size)` | Full eq table over active window bits |
| `gruen_poly_deg_2(q_0, previous_claim)` | Quadratic round polynomial via Gruen optimization |
| `gruen_poly_deg_3(q_constant, q_quadratic_coeff, s_0_plus_s_1)` | Cubic version |
| `gruen_poly_from_evals(q_evals, s_0_plus_s_1)` | Arbitrary-degree version |
| `par_fold_out_in(closures)` | Rayon-parallel fold: outer over `x_out`, inner over `x_in` |
| `par_fold_out_in_unreduced<const N>(per_g_values)` | Const-generic version with delayed Montgomery reduction |
| `merge()` | Materialize as dense polynomial (for testing) |

**Dependencies (all already in jolt-field or jolt-poly):**
- `JoltField` + `F::Challenge` + `F::UnreducedProductAccum` + `F::mul_to_product_accum` / `reduce_product_accum`
- `EqPolynomial` (jolt-poly) — `evals_cached`, `evals_cached_rev`, `evals_parallel`, `mle`
- `UnivariatePoly` (jolt-poly)
- `BindingOrder` enum — currently in `jolt-core/src/poly/multilinear_polynomial.rs`. **Must also be extracted** (simple enum: `LowToHigh`, `HighToLow`).
- `rayon` for parallel construction

**Does NOT depend on:** sumcheck, transcripts, PCS, or any zkvm types.

### 2. `RaPolynomial<I, F>` (~400 lines)

**Source:** `jolt-core/src/poly/ra_poly.rs`

Lazy one-hot polynomial that defers materialization through the first 3 sumcheck rounds via a state machine (`Round1` → `Round2` → `Round3` → `RoundN`). This avoids materializing `2^n` field elements for sparse RA polynomials until the table is 8x smaller.

**State machine:**

| State | After | Internal storage |
|-------|-------|------------------|
| `Round1` | Construction | `F: Vec<F>` (eq evals by address) + `Arc<Vec<Option<I>>>` (lookup indices) |
| `Round2` | 1st bind | Two prefix-scaled eq tables `F_0`, `F_1` |
| `Round3` | 2nd bind | Four prefix-scaled eq tables `F_00`..`F_11` |
| `RoundN` | 3rd bind | Fully materialized `MultilinearPolynomial<F>` |

**Trait impls to preserve:**
- `PolynomialBinding<F>` — `bind_parallel(&mut self, r, order)` drives state transitions
- `PolynomialEvaluation<F>` — `sumcheck_evals(index, degree, order) -> Vec<F>` dispatches via `get_bound_coeff`

**Dependencies:**
- `JoltField`, `ChallengeFieldOps`, `FieldChallengeOps`
- `MultilinearPolynomial<F>`, `PolynomialBinding<F>`, `PolynomialEvaluation<F>`, `BindingOrder` — these traits are in `jolt-core/src/poly/multilinear_polynomial.rs` and must also be extracted
- `EqPolynomial::mle` (jolt-poly)
- `rayon` for parallel chunk processing in Round3 bind
- `unsafe_allocate_zero_vec`, `drop_in_background_thread` — utility functions from `jolt-core/src/utils/thread.rs` (simple wrappers around unsafe alloc + `rayon::spawn`)

**Does NOT depend on:** sumcheck, transcripts, PCS.

### 3. `mles_product_sum` (~1400 lines)

**Source:** `jolt-core/src/subprotocols/mles_product_sum.rs`

Specialized product-sum kernels for the RA virtual sumcheck hot path. Evaluates:
```
q(X) = Σ_t ∏_{i=0}^{D-1} mle_{t,i}(X, ·)
```
where each `mle_{t,i}` is a `RaPolynomial<u8, F>`, at `D` grid points using Karatsuba-style stack-allocated expansions with delayed Montgomery reduction.

**Public functions:**

| Function | Signature |
|----------|-----------|
| `compute_mles_product_sum` | `(mles, claim, eq_poly) -> UnivariatePoly<F>` — top-level dispatcher |
| `compute_mles_product_sum_evals_sum_of_products_d4/d8/d16` | `(mles, n_products, eq_poly) -> Vec<F>` — hot-path SoP evaluators |
| `finish_mles_product_sum_from_evals` | `(sum_evals, claim, eq_poly) -> UnivariatePoly<F>` — recover q(0) + interpolate |
| `eval_linear_prod_assign` | `(pairs, evals)` — dispatcher for per-size product kernels |

**Dependencies:**
- `RaPolynomial<u8, F>` (extracted above)
- `GruenSplitEqPolynomial<F>` (extracted above) — `par_fold_out_in_unreduced` is the workhorse
- `UnivariatePoly<F>` (jolt-poly)
- `JoltField` + `BarrettReduce` + `FMAdd` (jolt-field)
- `SmallAccumS<F>` (from `jolt-core/src/utils/accumulation.rs`) — signed accumulator for deferred modular reduction. **Must also be extracted** (small type, ~50 lines).

## Migration approach

### Step 1: Extract supporting types to jolt-poly

These small types currently in jolt-core must move first:

1. `BindingOrder` enum (from `multilinear_polynomial.rs`) — trivial enum, no deps
2. `PolynomialBinding<F>` trait (from `multilinear_polynomial.rs`) — depends on `BindingOrder`
3. `PolynomialEvaluation<F>` trait (from `multilinear_polynomial.rs`)
4. `MultilinearPolynomial<F>` enum (from `multilinear_polynomial.rs`) — dispatches over scalar types. This is the big one; it wraps `DensePolynomial<F>`, `CompactPolynomial<T>`, `OneHotPolynomial`, `RlcPolynomial` etc.
5. `unsafe_allocate_zero_vec` + `drop_in_background_thread` — utility functions (~20 lines each)

### Step 2: Extract GruenSplitEqPolynomial to jolt-poly

Move `split_eq_poly.rs` → `jolt-poly/src/split_eq.rs`. Update imports from `crate::field` to `jolt_field`, from `super::*` to `crate::*`.

### Step 3: Extract RaPolynomial to jolt-poly

Move `ra_poly.rs` → `jolt-poly/src/ra.rs`. Depends on Step 1 types.

### Step 4: Extract mles_product_sum

This can go in either `jolt-poly` (if we view it as a polynomial evaluation primitive) or `jolt-sumcheck` (if we view it as sumcheck support). Recommended: **`jolt-poly`**, since it has no sumcheck protocol dependency — it just evaluates polynomial products.

Also extract `SmallAccumS<F>` (and related accumulation types if needed) to `jolt-field` or `jolt-poly`.

### Step 5: Update jolt-core

Replace `jolt-core/src/poly/split_eq_poly.rs`, `ra_poly.rs`, and `subprotocols/mles_product_sum.rs` with re-exports from `jolt-poly`.

## Verification

```bash
# jolt-poly compiles with new types
cargo clippy -p jolt-poly --message-format=short -q --all-targets -- -D warnings

# Existing jolt-poly tests still pass
cargo nextest run -p jolt-poly --cargo-quiet

# jolt-core tests still pass (using re-exports)
cargo nextest run -p jolt-core --cargo-quiet --features host

# Key correctness test
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
```

## Risk assessment

- `GruenSplitEqPolynomial` and `RaPolynomial` are **pure polynomial utilities** with clean dependency boundaries. Low risk.
- `mles_product_sum` uses `BarrettReduce` / `FMAdd` / `SmallAccumS` which are jolt-core field extension traits. If these aren't already on the `Field` trait in `jolt-field`, they need to be added or the accumulation types need to be extracted alongside.
- `MultilinearPolynomial<F>` enum extraction is the highest-risk item — it has many variants and is used pervasively in jolt-core. Consider whether to extract the full enum or just the traits + `RaPolynomial` variant.

## Reference files

| File | Lines | Destination |
|------|-------|-------------|
| `jolt-core/src/poly/split_eq_poly.rs` | 858 | `jolt-poly/src/split_eq.rs` |
| `jolt-core/src/poly/ra_poly.rs` | 404 | `jolt-poly/src/ra.rs` |
| `jolt-core/src/subprotocols/mles_product_sum.rs` | 1397 | `jolt-poly/src/mles_product_sum.rs` |
| `jolt-core/src/poly/multilinear_polynomial.rs` | ~300 (subset) | `jolt-poly/src/binding.rs` (traits + enum) |
| `jolt-core/src/utils/thread.rs` | ~40 (subset) | `jolt-poly/src/utils.rs` |
| `jolt-core/src/utils/accumulation.rs` | ~50 (subset) | `jolt-field/src/accumulation.rs` |
