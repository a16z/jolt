# Task 03: backend-r1cs

**Status:** Done
**Phase:** After task 02
**Dependencies:** `jolt-ir` core (task 01), evaluate backend (task 02 for testing)
**Blocks:** BlindFold integration in `jolt-zkvm` (ZK mode)

## Objective

Implement the **R1CS backend**: given a `SumOfProducts`, emit R1CS constraints (A, B, C row triples) that enforce the expression's value equals a designated output variable.

This replaces the `R1csConstraintVisitor` in `jolt-core/src/subprotocols/blindfold/r1cs.rs` and the hand-written `OutputClaimConstraint` → R1CS pipeline.

## Deliverables

### `src/backends/r1cs.rs`

The R1CS backend consumes `SumOfProducts` (not raw `Expr`) because SoP form maps directly to R1CS multiplication gates:
- Each `SopTerm` with 0 factors → linear constraint (no multiplication)
- Each `SopTerm` with 1 factor → `coeff * factor = output` (one mult gate)
- Each `SopTerm` with 2+ factors → chain of mult gates: `aux = f0 * f1`, `aux2 = aux * f2`, ...
- Final sum: `output = Σ term_results`

Types:

```rust
/// An R1CS variable index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct R1csVar(pub u32);

/// A linear combination: Σ cᵢ · xᵢ
pub struct LinearCombination<F> {
    pub terms: Vec<(F, R1csVar)>,
}

/// A single R1CS constraint: A · B = C (where A, B, C are linear combinations).
pub struct R1csConstraint<F> {
    pub a: LinearCombination<F>,
    pub b: LinearCombination<F>,
    pub c: LinearCombination<F>,
}

/// Result of R1CS emission from a SumOfProducts.
pub struct R1csEmission<F> {
    pub constraints: Vec<R1csConstraint<F>>,
    pub aux_vars: Vec<R1csVar>,
    pub output_var: R1csVar,
}
```

Visitor-style API matching the existing `SumOfProductsVisitor` pattern:

```rust
impl SumOfProducts {
    pub fn emit_r1cs<F: Field>(
        &self,
        opening_vars: &[R1csVar],
        challenge_values: &[F],
        next_var: &mut u32,
    ) -> R1csEmission<F>;

    pub fn estimate_aux_var_count(&self) -> usize;
}
```

## Relationship to BlindFold

The BlindFold R1CS builder (`jolt-core/src/subprotocols/blindfold/r1cs.rs`) currently:
1. Takes `OutputClaimConstraint` (hand-written per sumcheck)
2. Resolves `ValueSource::Challenge(idx)` to baked constants
3. Resolves `ValueSource::Opening(id)` to witness variables
4. Emits mult-chain constraints

With `jolt-ir`, step 1 becomes `claim_def.expr.to_sum_of_products()` and steps 2-4 become `sop.emit_r1cs(...)`.

## Testing

- Small expressions: `a * b` → 1 constraint, `a * b * c` → 2 constraints (chain)
- Linear: `a + b` → 0 mult constraints (just linear combination)
- Weighted sum: `α*a + β*b` → linear combination, 0 mult constraints
- **Round-trip test:** emit R1CS, assign concrete witness values, verify `A·B = C` holds for all constraints
- **Consistency test:** `sop.emit_r1cs()` witness assignment evaluates to same value as `sop.evaluate()`
- `estimate_aux_var_count()` matches actual aux vars allocated

## Reference

- `jolt-core/src/subprotocols/blindfold/r1cs.rs:1-100` — existing `R1csConstraintVisitor`
- `jolt-core/src/subprotocols/blindfold/output_constraint.rs:162-183` — existing `visit()` pattern
- `jolt-core/src/subprotocols/blindfold/output_constraint.rs:272-311` — `SumOfProductsVisitor` trait
