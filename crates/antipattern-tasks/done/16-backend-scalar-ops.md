# Task 16: Route Scalar Operations Through Backend

## Status: TODO

## Anti-Pattern
Two pure-scalar CPU functions called from op handlers bypass the backend:

1. **`evaluate_claim()`** (runtime.rs ~25 lines): Evaluates a `ClaimFormula` — iterates over terms with coefficients and factors (Eval, Challenge, EqChallengePair). Called from `Op::AbsorbInputClaim`.

2. **`evaluate_mle()`** (runtime.rs ~10 lines): MLE evaluation via repeated halving. O(N) field ops where N = table size. Called from `Op::EvaluatePreprocessed`.

Both are pure functions with no state. Small individual cost but called many times.

## Fix
Add to `ComputeBackend`:

```rust
fn evaluate_mle<F: Field>(&self, evals: &[F], point: &[F]) -> F;
fn evaluate_claim<F: Field>(&self, formula: &ClaimFormula, evaluations: &HashMap<PolynomialId, F>, challenges: &[F]) -> F;
```

CPU backend: lift existing code. Runtime: `backend.evaluate_mle(...)` / `backend.evaluate_claim(...)`.

## Risk: Very Low
Pure functions, no state. Direct code lift.

## Dependencies: None
