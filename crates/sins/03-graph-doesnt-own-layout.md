# Sin 3: Protocol Graph Doesn't Own Buffer Layout

## The Problem

The `ProtocolGraph` (jolt-ir) defines *what* to prove: claim formulas, dependencies,
staging, challenge specs. But it does NOT define *how* to lay out buffers for proving.
These decisions are scattered:

| Decision | Where it's made | Should be |
|----------|----------------|-----------|
| Binding order | `binding_order_for()` inspects `VariableGroup::Address` | In the graph vertex |
| Weight table type | `build_witness()` matches on `sv.weighting` (Eq/EqPlusOne/Lt/Derived) | In the graph vertex |
| Eq factoring | `KernelEvaluator::with_toom_cook_eq` decides to factor eq | In the execution plan |
| Pre-combination | EqProduct arm of `build_witness()` pre-combines tables | In the execution plan |
| Domain padding | `table.resize(domain_size, F::zero())` in `build_witness()` | Part of buffer setup |

The graph carries some hints: `SumcheckVertex.weighting: PublicPolynomial` and
`Phase.variable_group: VariableGroup`. But these are hints that the prover must
interpret, not declarations that the prover executes.

## Why This Is a Sin

The graph is half-declarative. It says "this vertex weights by Eq" and "this phase uses
Address variables", but it doesn't say "use Toom-Cook factoring" or "use H2L binding" or
"pre-combine these polynomials." The prover infers these from the shape, which means:

1. The graph doesn't fully specify the proof strategy
2. `build_witness()` must contain inference logic (the 4-arm match)
3. Testing a new proof strategy requires changing prover code, not just graph construction
4. The verifier doesn't need this information (it only checks claims), so it's not wrong
   for the graph to omit it — but the PROVER graph should have it

## Proposed Solution: Execution Plan on the Vertex

### Extend SumcheckVertex

The `SumcheckVertex` in the protocol graph should carry (or lazily produce) an
`ExecutionPlan` (from Sin 2):

```rust
pub struct SumcheckVertex {
    // Existing: what to prove
    pub formula: FormulaRef,
    pub deps: Vec<ClaimId>,
    pub num_vars: SymbolicExpr,
    pub degree: usize,

    // New: how to prove it
    pub execution_plan: ExecutionPlan,
}
```

Or, if we want to keep the graph lightweight (for the verifier which doesn't need plans):

```rust
pub struct SumcheckVertex {
    pub formula: FormulaRef,
    pub deps: Vec<ClaimId>,
    pub num_vars: SymbolicExpr,
    pub degree: usize,
    pub weighting: PublicPolynomial,
    pub phases: Vec<Phase>,
}

impl SumcheckVertex {
    /// Compile an execution plan for this vertex.
    /// Only needed by the prover — verifier never calls this.
    pub fn to_execution_plan(&self) -> ExecutionPlan {
        self.formula.definition.to_execution_plan_with(
            self.weighting,
            self.phases.first().map(|p| p.variable_group),
        )
    }
}
```

This keeps the graph serializable and lightweight while giving the prover a single
method to get the full execution plan.

### What build_witness becomes

```rust
fn build_witness(...) -> Box<dyn SumcheckCompute<F>> {
    let plan = sv.to_execution_plan();
    execute_plan(plan, trace_polys, committed_store, backend, challenges)
}

fn execute_plan(...) -> Box<dyn SumcheckCompute<F>> {
    // Generic loop: upload buffers per plan.inputs, compile kernel, construct evaluator
    // No match on shape, no inference, no per-vertex special logic
}
```

### What Changes

The `weighting` and `phases` fields on `SumcheckVertex` are no longer ambient hints.
They're consumed by `to_execution_plan()` which produces a complete, self-describing
plan. The prover doesn't interpret hints — it executes plans.

The graph builder (`build_jolt_protocol()`) remains the single place where layout
decisions are made. But now those decisions are encoded as structured data, not
implicit conventions.

## Decisions — Absorbed by Analysis 11

The graph doesn't need a separate "execution plan" — the graph IS the execution plan.

- Each vertex carries: formula, algorithm, eq variant, input bindings
- The topological walk of the graph defines the execution order
- Buffer layout follows from vertex input declarations
- No `to_execution_plan()` method needed — the prover reads the vertex directly
- See [Analysis 11](11-unified-execution-model.md) for the unified model
