# test-jolt-poly-integration: Integration tests for jolt-poly

**Scope:** crates/jolt-poly/tests/

**Depends:** impl-jolt-poly, test-jolt-poly

**Verifier:** ./verifiers/scoped.sh /workdir jolt-poly

**Context:**

Write integration tests for the `jolt-poly` crate that test the public API as external users would use it. These tests go in the `tests/` directory and verify the crate's functionality from an external perspective.

### Integration Test Files

Create the following test files in `crates/jolt-poly/tests/`:

#### 1. `polynomial_api.rs` - Test public API usage patterns

Test common usage patterns combining multiple API calls:
- Create polynomial via different constructors, verify they behave identically
- Test chaining operations: create → bind → evaluate
- Test conversions between polynomial types
- Verify all polynomial types work with generic functions accepting `&dyn MultilinearPolynomial<F>`

#### 2. `type_interop.rs` - Test conversions between polynomial types

Test interoperability between different polynomial representations:
- `DensePolynomial` ↔ `CompactPolynomial` conversions
- Verify evaluations remain consistent across conversions
- Test that `bind()` on different types produces equivalent results
- Test `EqPolynomial::evaluations()` produces correct `DensePolynomial`

#### 3. `math_properties.rs` - Verify mathematical invariants

Test mathematical properties that must hold:
- Schwartz-Zippel: two distinct polynomials disagree at random points with high probability
- Multilinear extension uniqueness: same evaluations → same polynomial
- Linear combinations: `(αf + βg)(x) = αf(x) + βg(x)`
- Identity polynomial properties: `I(i) = i` for all binary inputs

### Test Structure

Each test file should:
- Import the crate as an external dependency: `use jolt_poly::*;`
- Use the same field implementation as unit tests
- Focus on multi-step workflows and API combinations
- Test error cases and edge conditions from user perspective

### Specific Test Cases

**API Patterns:**
```rust
#[test]
fn test_polynomial_workflow() {
    let poly = DensePolynomial::<TestField>::random(4, &mut rng);
    let point = vec![/* random field elements */];

    // Test multiple evaluation methods produce same result
    let eval1 = poly.evaluate(&point);
    let eval2 = poly.clone().bind_all(point.clone());
    assert_eq!(eval1, eval2);
}
```

**Type Interop:**
```rust
#[test]
fn test_compact_to_dense_consistency() {
    let values: Vec<u8> = vec![/* test values */];
    let compact = CompactPolynomial::<u8, TestField>::new(values.clone());
    let dense = compact.to_dense();

    // Verify evaluations match
    for point in test_points {
        assert_eq!(compact.evaluate(&point), dense.evaluate(&point));
    }
}
```

**Math Properties:**
```rust
#[test]
fn test_polynomial_linearity() {
    let f = DensePolynomial::random(3, &mut rng);
    let g = DensePolynomial::random(3, &mut rng);
    let alpha = TestField::random(&mut rng);
    let beta = TestField::random(&mut rng);

    // Test linear combination property
    let combined = f.scale(alpha).add(&g.scale(beta));
    let point = random_point(3);

    let expected = f.evaluate(&point) * alpha + g.evaluate(&point) * beta;
    assert_eq!(combined.evaluate(&point), expected);
}
```

### Acceptance Criteria

- All three integration test files created and passing
- Tests use the crate as an external dependency
- At least 5 test cases per file
- Tests demonstrate real usage patterns
- Good test organization and documentation
- All tests pass with `cargo nextest run -p jolt-poly`
- No modifications to source code (tests/ directory only)