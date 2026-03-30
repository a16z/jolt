# jolt-ir

Symbolic expression IR for sumcheck claim formulas in the Jolt zkVM.

## Purpose

Every sumcheck in Jolt proves that a polynomial identity holds -- for example, booleanity checks that `gamma * (H^2 - H) = 0`. Previously, each formula was written 4 separate times in 4 different representations that had to stay in sync. `jolt-ir` replaces all four with one canonical definition.

Developers write each claim formula once as an `Expr`. All backends -- evaluation, BlindFold R1CS, Lean4, circuit transpilation -- derive from it mechanically.

## Quick start

```rust
use jolt_ir::{ExprBuilder, ClaimDefinition, OpeningBinding, PolynomialId};

// Define a booleanity check: gamma * (H^2 - H)
let b = ExprBuilder::new();
let h = b.opening(0);
let gamma = b.challenge(0);
let expr = b.build(gamma * (h * h - h));

// Normalize to CompositionFormula for backend dispatch
let formula = expr.to_composition_formula();
assert_eq!(formula.degree(), 3); // gamma * H * H term has degree 3
```

## Two consumer paths

`jolt-ir` serves two fundamentally different kinds of consumer:

**Protocol consumers** (verifier, BlindFold R1CS, Lean4, circuit transpilation) consume `Expr` directly via `ExprVisitor`. They care about *what* the formula means -- symbolic structure for constraint emission, formal verification, or evaluation at a single point.

**Compute backends** (CPU, Metal, CUDA) consume `CompositionFormula`, the normalized sum-of-products form. The compiler dispatches on recognized patterns — `is_eq_product()`, `is_hamming_booleanity()`, `as_product_sum()` — to emit hand-optimized codegen. Unrecognized formulas fall through to the generic formula compiler that walks the terms directly.

```
Expr ──→ ExprVisitor(Evaluate)    → F values
     ──→ ExprVisitor(R1CS)        → constraints
     ──→ ExprVisitor(Circuit)     → circuit gates
     ──→ ExprVisitor(Lean)        → Lean4 terms

CompositionFormula ──→ jolt-cpu   → CpuKernel<F> (Rust closures)
                   ──→ jolt-metal → MetalKernel   (MSL source)
```

## Architecture

### Arena-allocated expression DAG

Expressions are stored as flat arrays of `ExprNode` variants, referenced by `ExprId(u32)` indices. No heap allocation per node, no `Box`, no pointers. The arena is instance-local.

### Builder with operator overloading

`ExprBuilder` uses `RefCell` interior mutability. Handles implement `Add`, `Sub`, `Mul`, `Neg`, and support integer literals.

### Visitor-based backends

The `ExprVisitor` trait provides bottom-up traversal. Each backend implements 6 methods (one per node type): Evaluate, R1CS, Circuit (`CircuitEmitter`), Lean4.

### Composition formula

`to_composition_formula()` distributes multiplication over addition, producing `CompositionFormula` — a normalized `Σᵢ coeffᵢ × ∏ⱼ factorᵢⱼ` form where each factor is `Factor::Input(u32)` or `Factor::Challenge(u32)`. Pattern detection methods (`as_product_sum()`, `is_eq_product()`, `is_hamming_booleanity()`, `is_linear_combination()`) enable compute backends to dispatch to specialized codegen.

### R1CS exports

- **`R1csEmission<F>`** -- Emitted R1CS constraints from sum-of-products normalization.
- **`R1csConstraint`** / **`LinearCombination`** / **`LcTerm`** / **`R1csVar`** -- R1CS constraint primitives.

### Expression core types

- **`Expr`** / **`ExprArena`** / **`ExprId`** / **`ExprNode`** / **`Var`** -- Arena-allocated expression DAG and node types.
- **`ExprBuilder`** / **`ExprHandle`** -- Builder with operator overloading for constructing expressions.
- **`ExprVisitor`** -- Trait for bottom-up traversal of expression DAGs.
- **`PolynomialId`** -- Unique identifier for polynomials in the protocol.

### Additional types

- **`CompositionFormula`** / **`ProductTerm`** / **`Factor`** -- Normalized sum-of-products representation for backend dispatch.
- **`ClaimDefinition`** -- Associates an expression with its opening and challenge bindings.
- **`CircuitEmitter`** -- Visitor-based backend for circuit gate emission.

### Public modules

- **`protocol`** -- Protocol graph: claim-level IR for SNARK structure, capturing the full claim dependency DAG.
- **`toom_cook`** -- Toom-Cook evaluation point constants and interpolation matrices.
- **`zkvm`** -- Concrete claim definitions and polynomial identifiers for the Jolt zkVM pipeline.

## Crate boundaries

`jolt-ir` is consumed by `jolt-spartan` (via `ir_r1cs` bridge) and `jolt-zkvm`. The sumcheck crate remains generic with no `jolt-ir` dependency. `jolt-ir` depends only on `jolt-field`.

## License

MIT
