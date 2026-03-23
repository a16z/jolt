# jolt-ir

Symbolic expression IR for sumcheck claim formulas in the Jolt zkVM.

## Purpose

Every sumcheck in Jolt proves that a polynomial identity holds -- for example, booleanity checks that `gamma * (H^2 - H) = 0`. Previously, each formula was written 4 separate times in 4 different representations that had to stay in sync. `jolt-ir` replaces all four with one canonical definition.

Developers write each claim formula once as an `Expr`. All backends -- evaluation, BlindFold R1CS, Lean4, circuit transpilation -- derive from it mechanically.

## Quick start

```rust
use jolt_ir::{ExprBuilder, ClaimDefinition, OpeningBinding, ChallengeBinding, ChallengeSource};

// Define a booleanity check: gamma * (H^2 - H)
let b = ExprBuilder::new();
let h = b.opening(0);
let gamma = b.challenge(0);
let expr = b.build(gamma * (h * h - h));

// Normalize to sum-of-products for R1CS emission
let sop = expr.to_sum_of_products();
assert_eq!(sop.len(), 2); // gamma*H*H and -gamma*H
```

## Two consumer paths

`jolt-ir` serves two fundamentally different kinds of consumer:

**Protocol consumers** (verifier, BlindFold R1CS, Lean4, circuit transpilation) consume `Expr` directly via `ExprVisitor`. They care about *what* the formula means -- symbolic structure for constraint emission, formal verification, or evaluation at a single point.

**Compute backends** (CPU, Metal, CUDA) consume `KernelDescriptor`, not `Expr`. A `KernelDescriptor` wraps the symbolic formula with operational metadata needed to evaluate it efficiently over million-element buffers: polynomial degree (grid size), tensor split (thread mapping), and recognized fast-path patterns. `Expr` only reaches compute backends inside `KernelShape::Custom`; the `ProductSum`, `EqProduct`, and `HammingBooleanity` shapes bypass it entirely with hand-optimized codegen.

```
Expr ──→ ExprVisitor(Evaluate)    → F values
     ──→ ExprVisitor(R1CS)        → constraints
     ──→ ExprVisitor(Circuit)     → circuit gates
     ──→ ExprVisitor(Lean)        → Lean4 terms

KernelDescriptor ──→ jolt-cpu   → CpuKernel<F> (Rust closures)
                 ──→ jolt-metal → MetalKernel   (MSL source)
```

## Architecture

### Arena-allocated expression DAG

Expressions are stored as flat arrays of `ExprNode` variants, referenced by `ExprId(u32)` indices. No heap allocation per node, no `Box`, no pointers. The arena is instance-local.

### Builder with operator overloading

`ExprBuilder` uses `RefCell` interior mutability. Handles implement `Add`, `Sub`, `Mul`, `Neg`, and support integer literals.

### Visitor-based backends

The `ExprVisitor` trait provides bottom-up traversal. Each backend implements 6 methods (one per node type): Evaluate, R1CS, Circuit (`CircuitEmitter`), Lean4.

### Sum-of-products normalization

`to_sum_of_products()` distributes multiplication over addition, producing `SumOfProducts` -- a flat `sum(coeff * product(factors))` form mapping directly to R1CS constraints.

### Kernel descriptors

- **`KernelDescriptor`** -- The codegen unit for compute backends. Carries shape, degree, and optional tensor split.
- **`KernelShape`** -- Dispatch discriminant: `ProductSum` (Toom-Cook grid), `EqProduct` (eq * g), `HammingBooleanity` (eq * h * (h-1)), `Custom` (arbitrary `Expr`). The first three are recognized patterns compiled into hand-optimized code; `Custom` is the escape hatch where the `Expr` is walked for codegen.
- **`TensorSplit`** -- Split-eq sqrt decomposition for GPU thread hierarchy.

### R1CS exports

- **`R1csEmission<F>`** -- Emitted R1CS constraints from sum-of-products normalization.
- **`R1csConstraint`** / **`LinearCombination`** / **`LcTerm`** / **`R1csVar`** -- R1CS constraint primitives.

### Additional types

- **`SumOfProducts`** / **`SopTerm`** / **`SopValue`** -- Normalized sum-of-products representation.
- **`ClaimDefinition`** -- Associates an expression with its opening and challenge bindings.

## Crate boundaries

`jolt-ir` is consumed by `jolt-spartan` (via `ir_r1cs` bridge) and `jolt-zkvm`. The sumcheck crate remains generic with no `jolt-ir` dependency. `jolt-ir` depends only on `jolt-field`.

## License

MIT
