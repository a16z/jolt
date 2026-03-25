# Sin 2: compile_descriptor Returns Untyped Side-Channel Data

## The Problem

```rust
// In jolt-ir/src/claim.rs
pub fn compile_descriptor<F: Field>(&self, challenges: &[F]) -> (KernelDescriptor, Vec<F>)
```

The `Vec<F>` means three different things depending on which `KernelShape` was produced:

| Shape | What `Vec<F>` contains | How build_witness uses it |
|-------|----------------------|--------------------------|
| EqProduct | Per-opening pre-combination weights | CPU-side weighted sum of polynomial tables |
| HammingBooleanity | `[scale_factor]` | Multiply eq buffer by scale |
| Custom | Challenge values by slot index | Pass to `compile_kernel_with_challenges` |

The caller MUST pattern-match on the shape to interpret the `Vec<F>`. This is a type-level
lie — the same return type carries three different protocols.

## Why This Is a Sin

The whole point of `ClaimDefinition` is "single source of truth". The formula is defined
once and all backends derive from it. But `compile_descriptor` breaks this promise:

1. The IR produces `(shape, opaque_data)`
2. The orchestration layer pattern-matches on shape to interpret opaque_data
3. Each match arm does bespoke buffer preparation

So the "single source of truth" is really "single source of the formula, but the buffer
preparation protocol is duplicated between `compile_descriptor` (which produces
shape-specific data) and `build_witness` (which consumes it with shape-specific logic)."

## Concrete Consequences

In `build_witness()` (prover.rs:556-625):

```rust
match &desc.shape {
    KernelShape::EqProduct => {
        // materialized = per-opening weights
        let mut combined = vec![F::zero(); n];
        for (table, &weight) in poly_tables.iter().zip(materialized.iter()) {
            for (i, c) in combined.iter_mut().enumerate() {
                *c += weight * table[i];
            }
        }
        // ... upload combined as single buffer
    }
    KernelShape::HammingBooleanity => {
        // materialized = [scale_factor]
        let scale = materialized[0];
        let scaled_w: Vec<F> = w_table.iter().map(|&w| w * scale).collect();
        // ... upload scaled weights
    }
    KernelShape::Custom { .. } => {
        // materialized = challenge values (already baked into kernel)
        // ... upload raw buffers
    }
}
```

Three completely different buffer preparation strategies behind one `Vec<F>`.

## Proposed Solution: Self-Describing Execution Plan

### Replace compile_descriptor with to_execution_plan

```rust
/// Complete description of how to set up buffers and compile a kernel
/// for a sumcheck instance. Self-describing — no pattern matching needed.
pub struct ExecutionPlan {
    /// The composition formula (SoP, from Sin 0 fix).
    pub formula: CompositionFormula,
    /// How to prepare each input buffer.
    pub inputs: Vec<InputSetup>,
    /// How to handle the eq polynomial.
    pub eq_strategy: EqStrategy,
    /// Composition degree.
    pub degree: usize,
    /// Variable binding order.
    pub binding_order: BindingOrder,
}

/// How to prepare a single input buffer for the kernel.
pub enum InputSetup {
    /// Upload a single polynomial table directly.
    Single(PolynomialId),
    /// Pre-combine multiple polynomial tables with weights.
    /// `buffer[i] = Σ_j weights[j] * tables[j][i]`
    PreCombined {
        sources: Vec<PolynomialId>,
        weights: Vec<ChallengeRef>,
    },
    /// Scale a polynomial table by a challenge-derived factor.
    Scaled {
        source: PolynomialId,
        scale: ChallengeRef,
    },
}

/// Reference to a challenge value (resolved at plan execution time).
pub enum ChallengeRef {
    /// Direct challenge: `challenges[index]`
    Direct(u32),
    /// Product of challenges: `Π challenges[indices[i]]`
    Product(Vec<u32>),
    /// Constant structural value.
    Constant(i128),
}
```

### How build_witness becomes generic

```rust
fn build_witness(...) -> Box<dyn SumcheckCompute<F>> {
    let plan = formula.to_execution_plan(&challenge_labels);

    // Upload input buffers — no match needed
    let input_bufs: Vec<B::Buffer<F>> = plan.inputs.iter().map(|setup| {
        match setup {
            InputSetup::Single(poly_id) => {
                backend.upload(get_table(*poly_id))
            }
            InputSetup::PreCombined { sources, weights } => {
                let resolved_weights = resolve_challenges(weights, &challenges);
                let combined = pre_combine(sources, &resolved_weights, store, trace);
                backend.upload(&combined)
            }
            InputSetup::Scaled { source, scale } => {
                let s = resolve_challenge(scale, &challenges);
                let table = get_table(*source);
                let scaled: Vec<F> = table.iter().map(|&v| v * s).collect();
                backend.upload(&scaled)
            }
        }
    }).collect();

    let kernel = backend.compile_kernel(&plan.formula, &resolved_challenges);
    Box::new(KernelEvaluator::new(input_bufs, eq_input, kernel, plan, backend))
}
```

One code path. The `ExecutionPlan` fully describes what to do. The prover is a dumb
executor.

### Where the intelligence lives

`ClaimDefinition::to_execution_plan()` is the single place that decides:
- Which polynomials to pre-combine vs upload raw
- Which challenge values to bake vs resolve at runtime
- How to handle eq (AsInput vs Factored vs TensorSplit)

This is where the EqProduct "pre-combine" and HammingBooleanity "scale" decisions
are made — inside the IR, not inside the prover. The prover just executes the plan.

### What This Eliminates

- The untyped `Vec<F>` return from `compile_descriptor`
- All 4 match arms in `build_witness()`
- The conceptual split between "compile" and "prepare buffers"
- Any possibility of build_witness misinterpreting the data for a given shape

## Decisions — Simplified by Analysis 11

The `ExecutionPlan` concept from the original proposal is no longer needed. Under the
unified model, each vertex in the graph carries its `CompositionFormula` + algorithm enum
(`Dense`, `Sparse`, `UnivariateSkip`) + eq variant. The vertex IS the descriptor.

- `compile_descriptor` eliminated entirely — no replacement needed
- The vertex's formula is compiled to a kernel by the backend at setup time
- Buffer preparation is driven by the vertex's input declarations in the graph
- See [Analysis 11](11-unified-execution-model.md) for the unified model
