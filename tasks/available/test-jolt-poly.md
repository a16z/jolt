# test-jolt-poly: Comprehensive tests for jolt-poly

**Scope:** crates/jolt-poly/

**Depends:** impl-jolt-poly

**Verifier:** ./verifiers/scoped.sh /workdir jolt-poly

**Context:**

Write comprehensive tests for the `jolt-poly` crate. The implementation task (`impl-jolt-poly`) includes basic inline unit tests. This task adds property-based tests, edge case coverage, and fuzz targets.

**Do not modify source logic — test-only changes.**

### Test categories

#### 1. Property-based tests (proptest)

Add `proptest` as a dev-dependency.

**DensePolynomial:**
- For random `f` with `n` vars and random point `r in F^n`: `f.evaluate(r) == bind_all_then_read(f, r)` (bind each variable sequentially, read the single remaining evaluation)
- For random `f` and `g`: `(f + g).evaluate(r) == f.evaluate(r) + g.evaluate(r)` (linearity)
- `f.evaluate(r)` where all components of `r` are 0 or 1 matches the corresponding evaluation in the table

**EqPolynomial:**
- `sum_{x in {0,1}^n} eq(x, r) == 1` for any random `r`
- `eq(x, x) == 1` for any binary `x`
- `eq(x, y) == 0` for distinct binary `x, y`

**UnivariatePoly:**
- `interpolate(points).evaluate(x_i) == y_i` for all interpolation points
- Degree of interpolation result == `len(points) - 1`
- `evaluate` via Horner matches naive coefficient summation

**CompactPolynomial:**
- For any `CompactPolynomial<u8, F>` and random point `r`: evaluation matches the equivalent `DensePolynomial` with promoted values
- `evaluations()` returns the same values as manual promotion

#### 2. Edge case tests

- Zero polynomial (all evaluations = 0): evaluate returns 0 at any point
- Constant polynomial (1 var, both evaluations equal): evaluate returns constant
- Single-variable polynomial: `evaluate([r]) == (1-r)*f[0] + r*f[1]`
- `bind_in_place` with 0: keeps even-indexed evaluations
- `bind_in_place` with 1: keeps odd-indexed evaluations
- `EqPolynomial` with 0 variables: single evaluation = 1
- `UnivariatePoly` with degree 0: constant function
- `UnivariatePoly::interpolate` with a single point

#### 3. Fuzz targets

Create `fuzz/` directory with `cargo-fuzz` targets:

**`fuzz_dense_eval`:** Generate arbitrary `(Vec<u8>, Vec<u8>)`, interpret as `(evaluations, point)`, construct `DensePolynomial`, call `evaluate`. Must not panic.

**`fuzz_univariate_interpolate`:** Generate arbitrary `Vec<(u64, u64)>`, promote to field, call `interpolate`. Must not panic (or return meaningful error).

#### 4. Regression / known-value tests

- Hardcoded 2-variable polynomial: $f(x_1, x_2) = 1 + 2x_1 + 3x_2 + 4x_1 x_2$ (evaluations: [1, 3, 4, 10]). Verify evaluation at specific points.
- Hardcoded 3-variable eq polynomial: verify all 8 evaluations match the product formula.

**Acceptance:**

- At least 10 proptest properties across the polynomial types
- Edge cases for all zero/constant/single-variable polynomials
- Fuzz targets compile and run for at least 60 seconds without crash
- All existing tests still pass
- No modifications to non-test source code
- Tests are well-organized with descriptive names
