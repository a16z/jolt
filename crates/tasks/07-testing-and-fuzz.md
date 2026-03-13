# Task 07: testing-and-fuzz

**Status:** In progress
**Phase:** 1b (after core IR + evaluate backend + R1CS backend)
**Dependencies:** Tasks 01 + 02 + 03

## Objective

Harden `jolt-ir` with integration tests, fuzz targets, benchmarks, and documentation. Ensure the crate is production-ready before downstream integration.

## Deliverables

### README.md
- Crate purpose and design philosophy
- Quick start example (builder → expr → evaluate → SoP)
- API overview with links to doc comments
- Design decisions (i128 constants, arena allocation, no Div/Inv)

### Integration tests (`tests/`)

#### `tests/expr_eval.rs`
- Construct expressions via `ExprBuilder`, evaluate with known values
- All operator combinations: `a + b`, `a * b`, `a - b`, `-a`, nested
- Integer literal ops: `2 * h`, `h + 1`, `h - 3`
- Constants: `i128` → field promotion correctness
- Large expressions (100+ nodes)

#### `tests/sop_consistency.rs`
- **Critical invariant:** `expr.evaluate(o, c) == expr.to_sum_of_products().evaluate(o, c)` for random inputs
- Distribution: `(a + b) * c` → two terms
- Nested distribution: `(a + b) * (c + d)` → four terms
- Subtraction: `(a - b) * c`
- Negation: `-(a * b)`

#### `tests/r1cs_roundtrip.rs`
- Emit R1CS from SoP, assign witness, verify A·B=C satisfaction
- Output variable value matches `sop.evaluate()`
- Representative claim expressions (booleanity-style, weighted sums)

#### `tests/edge_cases.rs`
- Zero-variable expression (just a constant)
- Single-variable expression
- Same variable squared (`h * h`)
- Expression with only challenges, no openings (and vice versa)
- CSE on expressions with repeated subtrees
- Negative constants, large i128 constants

### Fuzz targets (`fuzz/`)

#### `fuzz_targets/expr_eval.rs`
- Random expression tree structure + random constants + random var IDs
- Evaluate with random field values
- Assert `evaluate() == to_sum_of_products().evaluate()`

#### `fuzz_targets/sop_normalize.rs`
- Random expressions → normalize to SoP
- No panics, no out-of-bounds
- Term count within bounds

#### `fuzz_targets/cse.rs`
- Random expressions with repeated subtrees
- CSE → evaluate unchanged, arena size ≤ original

#### `fuzz_targets/r1cs_roundtrip.rs`
- Random expressions → SoP → emit R1CS → verify satisfaction
- No panics, correct output

### Benchmarks (`benches/`)
- `to_sum_of_products()` for expressions of varying sizes
- `evaluate()` vs `sop.evaluate()` comparison
- `emit_r1cs()` constraint generation
- `fold_constants()` and `eliminate_common_subexpressions()`
