# jolt-ir

Symbolic expression IR for sumcheck claim formulas in the Jolt zkVM.

## Purpose

Every sumcheck in Jolt proves that a polynomial identity holds — for example, booleanity checks that `gamma * (H^2 - H) = 0`. Previously, each formula was written 4 separate times in 4 different representations that had to stay in sync. `jolt-ir` replaces all four with one canonical definition.

Developers write each claim formula once as an `Expr`. All backends — evaluation, BlindFold R1CS, Lean4, circuit transpilation — derive from it mechanically.

## Quick start

```rust
use jolt_ir::{ExprBuilder, ClaimDefinition, OpeningBinding, ChallengeBinding, ChallengeSource};

// Define a booleanity check: gamma * (H^2 - H)
let b = ExprBuilder::new();
let h = b.opening(0);
let gamma = b.challenge(0);
let expr = b.build(gamma * (h * h - h));

// Evaluate with concrete field values
let result: F = expr.evaluate(&[h_val], &[gamma_val]);

// Normalize to sum-of-products for R1CS emission
let sop = expr.to_sum_of_products();
assert_eq!(sop.len(), 2); // gamma*H*H and -gamma*H

// Critical invariant: both evaluation paths agree
assert_eq!(
    expr.evaluate(&openings, &challenges),
    sop.evaluate(&openings, &challenges),
);
```

## Architecture

### Arena-allocated expression DAG

Expressions are stored as flat arrays of `ExprNode` variants, referenced by `ExprId(u32)` indices. No heap allocation per node, no `Box`, no pointers. The arena is instance-local (no global mutable state).

### Builder with operator overloading

`ExprBuilder` uses `RefCell` interior mutability so multiple `ExprHandle`s coexist. Handles implement `Add`, `Sub`, `Mul`, `Neg`, and support integer literals on the right (`h * 2`, `h + 1`) and left (`3i128 * h`).

### Visitor-based backends

The `ExprVisitor` trait provides bottom-up traversal. Each backend implements 6 methods (one per node type):

- **Evaluate** (`backends/evaluate.rs`): computes `F` values from concrete openings and challenges
- **R1CS** (`backends/r1cs.rs`): emits A·B=C constraints for BlindFold ZK mode
- **Circuit** (`backends/circuit.rs`): `CircuitEmitter` trait for transpilation to external circuit frameworks (gnark, bellman, plonky2)
- **Lean4** (`backends/lean.rs`): emits Lean4 syntax with CSE-aware `let` bindings

### Sum-of-products normalization

`to_sum_of_products()` mechanically distributes multiplication over addition, producing a flat `sum(coeff * product(factors))` form. This maps directly to R1CS multiplication gates and replaces hand-written `OutputClaimConstraint` / `ProductTerm` construction.

## Design decisions

### Constants are `i128`, not generic `F`

The IR is field-agnostic. Constants in claim formulas are always small structural integers (`0`, `1`, `-1`, register counts, chunk sizes). Actual field-sized values (gamma powers, eq evaluations, batching coefficients) enter as `Var::Challenge` variables, resolved to `F` at evaluation time via `ChallengeBinding`. Backends promote constants via `F::from_i128(val)`.

If a future formula needs a compile-time field constant larger than `i128`, model it as a `Var::Challenge` with `ChallengeSource::Derived`.

### No `Div` or `Inv` nodes

Claim expressions never require division. Keeping the node set minimal simplifies every backend.

### Instance-local arenas

Unlike the `MleAst` global arena (`static OnceLock<RwLock<Vec<Node>>>`), each `ExprBuilder` owns its arena privately. No global state, no synchronization, no cross-expression interference.

## Crate boundaries

`jolt-ir` is consumed by `jolt-spartan` (via `ir_r1cs` bridge) and `jolt-zkvm`. The sumcheck crate remains a generic protocol implementation with no `jolt-ir` dependency.

The IR does **not** own verifier orchestration (transcript sequencing, commitment schemes, opening proofs). That stays as Rust code in `jolt-zkvm`.
