# Task 02: backend-evaluate

**Status:** Done
**Phase:** 1b (after core IR)
**Dependencies:** `jolt-ir` core (task 01)
**Blocks:** `jolt-zkvm` verifier (standard mode claim checking)

## Objective

Implement the **evaluate backend**: given an `Expr`, concrete opening values `&[F]`, and challenge values `&[F]`, compute the result as a field element `F`.

This is the most critical backend — it replaces every hand-written `input_claim()` / `output_claim()` method on `SumcheckInstanceParams`. It must be correct and fast.

## Deliverables

### `src/backends/evaluate.rs`

```rust
use jolt_field::Field;
use crate::{Expr, Var, ExprVisitor};

struct EvaluateVisitor<'a, F> {
    openings: &'a [F],
    challenges: &'a [F],
}

impl<F: Field> ExprVisitor for EvaluateVisitor<'_, F> {
    type Output = F;
    fn visit_constant(&mut self, val: i128) -> F { F::from_i128(val) }
    fn visit_var(&mut self, var: Var) -> F {
        match var {
            Var::Opening(id) => self.openings[id as usize],
            Var::Challenge(id) => self.challenges[id as usize],
        }
    }
    fn visit_neg(&mut self, inner: F) -> F { -inner }
    fn visit_add(&mut self, a: F, b: F) -> F { a + b }
    fn visit_sub(&mut self, a: F, b: F) -> F { a - b }
    fn visit_mul(&mut self, a: F, b: F) -> F { a * b }
}

impl Expr {
    pub fn evaluate<F: Field>(&self, openings: &[F], challenges: &[F]) -> F;
}
```

Also add `SumOfProducts::evaluate`:
```rust
impl SumOfProducts {
    pub fn evaluate<F: Field>(&self, openings: &[F], challenges: &[F]) -> F;
}
```

And `ClaimDefinition::evaluate`:
```rust
impl ClaimDefinition {
    pub fn evaluate<F: Field>(&self, openings: &[F], challenges: &[F]) -> F {
        self.expr.evaluate(openings, challenges)
    }
}
```

## Testing

- Evaluate known expressions with known values (booleanity: `γ*(h²-h)` with specific `h`, `γ`)
- Evaluate zero/one constants
- Evaluate with single opening, single challenge
- **Critical property test:** For random expressions, `expr.evaluate(o, c) == expr.to_sum_of_products().evaluate(o, c)` — this is the invariant that guarantees SoP normalization correctness.
- Panics on out-of-bounds Opening/Challenge ids

## Notes

- This task adds `jolt-field` as a real dependency (for `Field` trait)
- Keep the visitor zero-alloc: no intermediate Vecs, just scalar accumulation
- Consider whether `Expr::evaluate` should be `#[inline]` — it will be called once per sumcheck stage per verification, not in a hot loop
