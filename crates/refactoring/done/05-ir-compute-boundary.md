# Sin 5: Unclear Boundary Between IR Frontend and Compute Backend

## Current State

The jolt-ir crate has excellent frontend/backend separation for its EXISTING backends
(evaluate, R1CS, circuit, Lean, Z3). The `ExprVisitor` trait is the sole consumption
interface. No backend imports from the frontend. Clean.

But the relationship between jolt-ir and the COMPUTE layer (jolt-compute → jolt-cpu →
jolt-metal) has a blurred boundary:

```
jolt-ir frontend
    │
    ├── Expr, ClaimDefinition, SumOfProducts     (pure description)
    ├── KernelDescriptor, KernelShape             (contains optimization hints)
    ├── EqProduct, HammingBooleanity              (backend optimization decisions)
    └── TensorSplit, EvalGrid, EqHandling         (backend execution parameters)
    │
    ▼
jolt-compute trait
    │
    ├── compile_kernel(desc: &KernelDescriptor)   (consumes optimization hints)
    └── pairwise_reduce / tensor_pairwise_reduce  (3 methods for eq conventions)
    │
    ▼
jolt-cpu / jolt-metal
    │
    └── pattern-match on KernelShape to generate code
```

The problem: `KernelDescriptor` lives in jolt-ir (the "frontend") but contains backend
optimization decisions. `KernelShape::EqProduct` is not a formula — it's an instruction
to the backend to use a hand-coded kernel. This is the IR telling the backend HOW to
compute, not WHAT to compute.

## What Should Be Frontend vs Backend

### Frontend (jolt-ir) should own:

- **Symbolic expressions** (Expr, ExprBuilder) — WHAT to compute
- **Normalized forms** (SumOfProducts) — canonical representation
- **Claim definitions** (ClaimDefinition, OpeningBinding) — formula + metadata
- **Protocol graph** (ProtocolGraph, SumcheckVertex) — claim dependencies
- **Polynomial identity** (PolynomialId) — naming
- **Execution plans** — WHAT buffers to prepare, not HOW to prepare them

### Backend (jolt-compute + implementations) should own:

- **Kernel shape recognition** — detecting ProductSum/EqProduct/HammingBooleanity patterns
- **Eval grid selection** — Toom-Cook vs standard grid
- **Kernel compilation** — generating closures/MSL/CUDA
- **Eq factoring strategy** — flat vs tensor-split
- **Thread hierarchy mapping** — TensorSplit outer/inner for GPU

### The Boundary Should Be: CompositionFormula

The IR gives the backend a `CompositionFormula` (from Sin 0). The backend decides
everything else:

```
jolt-ir (frontend)                    jolt-compute (backend)
─────────────────                    ──────────────────────
CompositionFormula ──────────────►   Pattern recognition:
  terms: Vec<ProductTerm>                - Pure product? → Toom-Cook
  num_inputs: usize                      - a*b? → EqProduct kernel
  degree: usize                          - a*b*(b-1)? → Hamming kernel
                                         - General SoP → codegen/stack-VM

ExecutionPlan ───────────────────►   Buffer management:
  inputs: Vec<InputSetup>               - Upload/pre-combine per plan
  eq_strategy: EqStrategy               - Eq as input / factored / tensor
  binding_order: BindingOrder            - Pair layout
```

## What Moves Where

### OUT of jolt-ir:

| Type | Current location | New location | Reason |
|------|-----------------|--------------|--------|
| `KernelShape` | jolt-ir/kernel.rs | Deleted (replaced by CompositionFormula) | Optimization concern |
| `EqProduct` | jolt-ir/kernel.rs | jolt-cpu pattern matcher | Backend optimization |
| `HammingBooleanity` | jolt-ir/kernel.rs | jolt-cpu pattern matcher | Backend optimization |
| `EvalGrid` | jolt-ir/kernel.rs | jolt-compute (or per-backend) | Execution strategy |
| `EqHandling` | jolt-ir/kernel.rs | jolt-compute (EqInput enum) | Execution strategy |

### STAYS in jolt-ir:

| Type | Reason |
|------|--------|
| `CompositionFormula` | Describes WHAT (sum-of-products structure) |
| `TensorSplit` | Part of ExecutionPlan (WHAT shape of factoring, not HOW) |
| `BindingOrder` | Part of ExecutionPlan |
| `ExecutionPlan` | Self-describing prover setup |
| `EqStrategy` | Declarative (AsInput/Factored/TensorSplit), not imperative |

### NEW in jolt-compute:

```rust
/// Backend's view of a kernel to compile.
/// Produced by the backend's own pattern recognition, not by the IR.
pub struct CompilationTarget<F: Field> {
    /// The formula (from IR).
    pub formula: CompositionFormula,
    /// Challenge values to bake in.
    pub challenges: Vec<F>,
    /// Degree for grid selection.
    pub degree: usize,
}
```

The `ComputeBackend` trait becomes:

```rust
trait ComputeBackend {
    type CompiledKernel<F: Field>;

    /// Compile a composition formula into a backend-specific kernel.
    /// The backend chooses the best strategy (Toom-Cook, hand-coded, codegen).
    fn compile_kernel<F: Field>(
        &self,
        formula: &CompositionFormula,
        challenges: &[F],
        degree: usize,
    ) -> Self::CompiledKernel<F>;

    // ... reduce, interpolate, etc.
}
```

No `KernelDescriptor` parameter — the backend receives the formula and makes all
optimization decisions internally.

## The Toom-Cook Question

Currently `KernelDescriptor` tells the backend "this is a ProductSum, use Toom-Cook."
In the new design, the backend recognizes this pattern:

```rust
// In jolt-cpu
fn compile_kernel<F>(formula: &CompositionFormula, challenges: &[F], degree: usize) -> CpuKernel<F> {
    // Check: is every term a product of the same arity with disjoint inputs?
    if let Some(product_info) = formula.as_uniform_product() {
        // Yes → Toom-Cook grid, specialized eval_prod_D kernel
        return compile_toom_cook(product_info.arity, product_info.num_products);
    }
    // ... other patterns ...
}
```

Metal does the same check and generates Toom-Cook MSL. The pattern recognition logic
is duplicated across backends, but it's small (~20 lines) and each backend may want
different thresholds (e.g., Metal might only Toom-Cook for D≥4, CPU for D≥2).

## Summary

The IR should be a pure symbolic description layer. When `KernelShape::EqProduct` lives
in the IR, the IR is making an optimization decision that only the backend should make.
Moving pattern recognition to the backend gives each backend independence to optimize
differently, and makes the IR a clean, extensible description of computation.

## Decisions

- IR frontend: pure formula + polynomial metadata (including sparsity annotations)
- Compute backend: pattern-matches formula + metadata → specialized kernels
- Backend impl needs enough expressiveness to fully exploit structure from the IR
- No `KernelShape`, `EvalGrid`, `EqHandling` in jolt-ir — these are compute concerns
