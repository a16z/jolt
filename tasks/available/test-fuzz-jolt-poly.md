# fuzz-jolt-poly: Fuzz testing for jolt-poly

**Scope:** crates/jolt-poly/fuzz/

**Depends:** impl-jolt-poly, test-jolt-poly, test-jolt-poly-integration

**Verifier:** ./verifiers/scoped.sh /workdir jolt-poly

**Context:**

Create fuzz tests for the `jolt-poly` crate to find edge cases and potential panics in polynomial operations. Use `cargo-fuzz` with libFuzzer.

### Setup

1. Initialize fuzzing in the crate:
```bash
cd crates/jolt-poly
cargo fuzz init
```

2. Add required dependencies to `fuzz/Cargo.toml`:
```toml
[dependencies]
libfuzzer-sys = "0.4"
jolt-poly = { path = ".." }
jolt-field = { path = "../../jolt-field" }
arbitrary = { version = "1", features = ["derive"] }
```

### Fuzz Targets

Create the following fuzz targets in `crates/jolt-poly/fuzz/fuzz_targets/`:

#### 1. `dense_new.rs` - Fuzz DensePolynomial construction

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_poly::DensePolynomial;
use jolt_field::ark_bn254::Fr as TestField;

fuzz_target!(|data: Vec<u8>| {
    // Try to construct polynomial from arbitrary data
    if data.is_empty() {
        return;
    }

    // Interpret bytes as field elements
    let evaluations: Vec<TestField> = data
        .chunks(32) // 32 bytes per field element
        .map(|chunk| {
            let mut bytes = [0u8; 32];
            bytes[..chunk.len()].copy_from_slice(chunk);
            TestField::from_random_bytes(&bytes).unwrap_or(TestField::zero())
        })
        .collect();

    // Must be power of 2
    let len = evaluations.len().next_power_of_two();
    let mut padded_evals = evaluations;
    padded_evals.resize(len, TestField::zero());

    // Should not panic
    let poly = DensePolynomial::new(padded_evals);

    // Basic invariant checks
    assert_eq!(poly.len(), len);
    assert_eq!(poly.num_vars(), len.trailing_zeros() as usize);
});
```

#### 2. `evaluate.rs` - Fuzz polynomial evaluation

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_poly::{DensePolynomial, MultilinearPolynomial};
use jolt_field::ark_bn254::Fr as TestField;
use arbitrary::{Arbitrary, Unstructured};

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    num_vars: u8, // Limit to reasonable size
    eval_data: Vec<u8>,
    point_data: Vec<u8>,
}

fuzz_target!(|input: FuzzInput| {
    // Limit polynomial size for fuzzing efficiency
    let num_vars = (input.num_vars % 10) as usize + 1;
    let poly_size = 1 << num_vars;

    // Create polynomial from fuzz data
    let evaluations: Vec<TestField> = input.eval_data
        .chunks(32)
        .take(poly_size)
        .map(|chunk| {
            let mut bytes = [0u8; 32];
            bytes[..chunk.len().min(32)].copy_from_slice(&chunk[..chunk.len().min(32)]);
            TestField::from_random_bytes(&bytes).unwrap_or(TestField::zero())
        })
        .collect();

    if evaluations.len() != poly_size {
        return; // Skip invalid inputs
    }

    let poly = DensePolynomial::new(evaluations.clone());

    // Create evaluation point from fuzz data
    let point: Vec<TestField> = input.point_data
        .chunks(32)
        .take(num_vars)
        .map(|chunk| {
            let mut bytes = [0u8; 32];
            bytes[..chunk.len().min(32)].copy_from_slice(&chunk[..chunk.len().min(32)]);
            TestField::from_random_bytes(&bytes).unwrap_or(TestField::zero())
        })
        .collect();

    if point.len() != num_vars {
        return; // Skip invalid inputs
    }

    // Should not panic
    let result = poly.evaluate(&point);

    // Verify evaluation at binary points matches table lookup
    let binary_point: Vec<TestField> = point
        .iter()
        .map(|&x| if x == TestField::zero() || x == TestField::one() { x } else { return })
        .collect();

    if binary_point.len() == num_vars {
        let index = binary_point
            .iter()
            .enumerate()
            .map(|(i, &b)| if b == TestField::one() { 1 << i } else { 0 })
            .sum::<usize>();

        assert_eq!(result, evaluations[index]);
    }
});
```

#### 3. `bind_operations.rs` - Fuzz binding operations

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_poly::{DensePolynomial, MultilinearPolynomial};
use jolt_field::ark_bn254::Fr as TestField;

#[derive(Debug, arbitrary::Arbitrary)]
struct BindFuzzInput {
    poly_data: Vec<u8>,
    bind_values: Vec<u8>,
}

fuzz_target!(|input: BindFuzzInput| {
    // Create small polynomial for fuzzing
    let num_vars = 4; // Fixed size for efficiency
    let poly_size = 1 << num_vars;

    let evaluations: Vec<TestField> = input.poly_data
        .chunks(32)
        .take(poly_size)
        .map(|chunk| {
            let mut bytes = [0u8; 32];
            bytes[..chunk.len().min(32)].copy_from_slice(&chunk[..chunk.len().min(32)]);
            TestField::from_random_bytes(&bytes).unwrap_or(TestField::zero())
        })
        .collect();

    if evaluations.len() != poly_size {
        return;
    }

    let mut poly = DensePolynomial::new(evaluations);
    let original_poly = poly.clone();

    // Bind variables one by one
    let bind_values: Vec<TestField> = input.bind_values
        .chunks(32)
        .take(num_vars)
        .map(|chunk| {
            let mut bytes = [0u8; 32];
            bytes[..chunk.len().min(32)].copy_from_slice(&chunk[..chunk.len().min(32)]);
            TestField::from_random_bytes(&bytes).unwrap_or(TestField::zero())
        })
        .collect();

    // Test bind operations
    for (i, &value) in bind_values.iter().enumerate() {
        if i < poly.num_vars() {
            poly = poly.bind(value);
            assert_eq!(poly.num_vars(), original_poly.num_vars() - i - 1);
            assert_eq!(poly.len(), original_poly.len() >> (i + 1));
        }
    }

    // Test evaluate vs sequential binding
    if bind_values.len() >= num_vars {
        let eval_result = original_poly.evaluate(&bind_values[..num_vars]);
        let bind_result = {
            let mut p = original_poly;
            for &v in &bind_values[..num_vars] {
                p = p.bind(v);
            }
            p.evaluations()[0]
        };
        assert_eq!(eval_result, bind_result);
    }
});
```

#### 4. `compact_polynomial.rs` - Fuzz compact polynomial operations

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_poly::{CompactPolynomial, MultilinearPolynomial, SmallScalar};
use jolt_field::ark_bn254::Fr as TestField;

fuzz_target!(|data: Vec<u8>| {
    if data.is_empty() {
        return;
    }

    // Create CompactPolynomial from u8 values
    let len = data.len().next_power_of_two();
    let mut scalars = data;
    scalars.resize(len, 0u8);

    let compact = CompactPolynomial::<u8, TestField>::new(scalars.clone());

    // Test basic operations
    assert_eq!(compact.len(), len);
    assert_eq!(compact.num_vars(), len.trailing_zeros() as usize);

    // Test evaluation at random point
    let num_vars = compact.num_vars();
    let point: Vec<TestField> = (0..num_vars)
        .map(|i| TestField::from(((i * 31) % 7) as u64))
        .collect();

    let result = compact.evaluate(&point);

    // Compare with dense polynomial evaluation
    let dense_evals: Vec<TestField> = scalars
        .iter()
        .map(|&s| TestField::from(s as u64))
        .collect();
    let dense = DensePolynomial::new(dense_evals);
    let dense_result = dense.evaluate(&point);

    assert_eq!(result, dense_result);
});
```

#### 5. `univariate_interpolation.rs` - Fuzz univariate polynomial interpolation

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use jolt_poly::UnivariatePoly;
use jolt_field::ark_bn254::Fr as TestField;

#[derive(Debug, arbitrary::Arbitrary)]
struct InterpolationInput {
    points: Vec<(u8, u8)>, // Small values for efficiency
}

fuzz_target!(|input: InterpolationInput| {
    if input.points.is_empty() || input.points.len() > 100 {
        return; // Skip invalid or too large inputs
    }

    // Convert to field elements
    let field_points: Vec<(TestField, TestField)> = input.points
        .iter()
        .map(|&(x, y)| (TestField::from(x as u64), TestField::from(y as u64)))
        .collect();

    // Check for duplicate x-coordinates (invalid for interpolation)
    let mut x_coords: Vec<_> = field_points.iter().map(|(x, _)| x).collect();
    x_coords.sort();
    x_coords.dedup();

    if x_coords.len() != field_points.len() {
        return; // Skip inputs with duplicate x-coordinates
    }

    // Should not panic
    let poly = UnivariatePoly::interpolate(&field_points);

    // Verify interpolation property
    for &(x, y) in &field_points {
        assert_eq!(poly.evaluate(x), y);
    }

    // Verify degree
    assert!(poly.degree() < field_points.len());
});
```

### Running the Fuzz Tests

Create a script `fuzz/run-all.sh`:

```bash
#!/bin/bash
set -e

echo "Running jolt-poly fuzz tests..."

# Run each target for 5 minutes
FUZZ_TIME=300

echo "Fuzzing dense_new..."
cargo +nightly fuzz run dense_new -- -max_total_time=$FUZZ_TIME

echo "Fuzzing evaluate..."
cargo +nightly fuzz run evaluate -- -max_total_time=$FUZZ_TIME

echo "Fuzzing bind_operations..."
cargo +nightly fuzz run bind_operations -- -max_total_time=$FUZZ_TIME

echo "Fuzzing compact_polynomial..."
cargo +nightly fuzz run compact_polynomial -- -max_total_time=$FUZZ_TIME

echo "Fuzzing univariate_interpolation..."
cargo +nightly fuzz run univariate_interpolation -- -max_total_time=$FUZZ_TIME

echo "All fuzz tests completed!"
```

### Coverage and Corpus Management

Add to `fuzz/.gitignore`:
```
target
corpus
artifacts
```

Create initial seed corpus for better coverage:
```bash
# Create corpus directories
mkdir -p fuzz/corpus/dense_new
mkdir -p fuzz/corpus/evaluate
# ... etc

# Add seed inputs
echo -n "seed_input_1" > fuzz/corpus/dense_new/seed1
```

### Current Progress

| Fuzz target | Status | Notes |
|------------|--------|-------|
| `dense_poly_ops.rs` | Done | Exists at `fuzz/fuzz_targets/dense_poly_ops.rs` (covers construction, evaluation, and bind operations) |
| `evaluate.rs` | Not started | May overlap with `dense_poly_ops.rs` |
| `bind_operations.rs` | Not started | May overlap with `dense_poly_ops.rs` |
| `compact_polynomial.rs` | Not started | |
| `univariate_interpolation.rs` | Not started | |
| `run-all.sh` script | Not started | |

**Note:** The existing `dense_poly_ops.rs` target may already cover some of the `dense_new`, `evaluate`, and `bind_operations` targets described above. Review its contents before creating redundant targets. The code samples reference `CompactPolynomial` and `SmallScalar` — these have been replaced by `Polynomial<T>` with `T: Into<F>`. Update samples if using this task as a reference.

### Acceptance Criteria

- All 5 fuzz targets created and compile successfully
- Each target runs for at least 5 minutes without crashes
- Fuzz targets test edge cases and boundary conditions
- Property-based invariants are checked
- Clear documentation in each fuzz target
- `run-all.sh` script created
- Corpus directories set up
- No modifications to source code (only fuzz/ directory)