# Sin 1: Eq Is Special-Cased Everywhere, Inconsistently

## The Problem

The eq polynomial has 3 different calling conventions across the stack:

| Layer | Where eq lives | Calling convention |
|-------|---------------|-------------------|
| Standard-grid kernels | `inputs[0]` (a regular input buffer) | `pairwise_reduce(inputs=[eq, ...], weights=unit)` |
| Toom-Cook kernels | separate weight buffer (factored partial eq) | `pairwise_reduce(inputs=[polys...], weights=partial_eq)` or `tensor_pairwise_reduce(inputs, outer, inner)` |
| HammingBooleanity | pre-scaled into weight table by prover | `pairwise_reduce(inputs=[scaled_eq, h], ...)` |

The `ComputeBackend` trait itself has 3 methods that differ only in how eq/weights are
provided:
- `pairwise_reduce(inputs, weights, ...)` — flat weight buffer
- `pairwise_reduce_unweighted(inputs, ...)` — implicit unit weights
- `tensor_pairwise_reduce(inputs, outer, inner, ...)` — factored weight buffer

The prover's `build_witness()` manually decides which convention to use per kernel shape.
The `KernelEvaluator` has two interpolation modes (`StandardGrid` vs `ToomCook`) that
exist solely because eq is handled differently in each.

## Why This Is a Sin

Every new sumcheck composition requires deciding: is eq an input? A weight? Factored?
Scaled? This decision is currently spread across:

1. `ClaimDefinition::to_kernel_descriptor()` — strips eq out, shifts Opening indices
2. `build_witness()` — 4-arm match deciding where to put eq
3. `KernelEvaluator` — two modes for how to reconstruct round poly from raw evals
4. `ComputeBackend` trait — 3 methods for different eq conventions

Changing eq handling (e.g., switching a kernel from flat to tensor-split) requires
coordinated changes in all 4 places.

## What Eq Actually Is

Eq is a polynomial like any other. It multiplies every composition:

```
round_poly(X) = Σ_x eq(w, x) · f(inputs(x), X)
```

The "weight" in `pairwise_reduce` IS eq. There is no separate concept of "weights" —
it's eq, always. The distinction between "eq as input" and "eq as weight" is an
optimization choice about whether to include eq inside the kernel's evaluation grid or
factor it out for cheaper handling.

## Proposed Solution: Eq Handling Is a Backend Decision

### In the KernelDescriptor (after Sin 0 fix)

```rust
pub struct KernelDescriptor {
    pub formula: CompositionFormula,
    pub degree: usize,
    /// Eq polynomial management strategy.
    /// The backend uses this to decide whether eq is:
    /// - included as an input buffer (standard grid)
    /// - factored into a weight buffer (Toom-Cook optimization)
    /// - tensor-decomposed (GPU thread hierarchy)
    pub eq_strategy: EqStrategy,
    pub binding_order: BindingOrder,
}

pub enum EqStrategy {
    /// Eq is included as input buffer index 0.
    /// The kernel evaluates `eq(pair) * f(other_inputs(pair))`.
    /// Weight buffer is unit (all ones) or absent.
    AsInput,
    /// Eq is factored out: `eq(w, x) = eq_single(w_0, x_0) · eq(w[1..], x[1..])`.
    /// Partial eq `eq(w[1..], ·)` becomes the weight buffer.
    /// The evaluator multiplies by `eq_single(w_k, X)` during reconstruction.
    Factored,
    /// Eq is tensor-decomposed into outer × inner factors.
    /// Uses `tensor_pairwise_reduce(inputs, outer, inner, ...)`.
    TensorSplit(TensorSplit),
}
```

### In the ComputeBackend trait

Reduce the 3 methods to 1:

```rust
trait ComputeBackend {
    /// Composition-reduce over paired inputs.
    ///
    /// `eq` describes how the weighting polynomial is provided:
    /// - `EqInput::Flat(buf)` — standard flat weight buffer
    /// - `EqInput::Unit` — implicit all-ones (kernel includes eq as input)
    /// - `EqInput::Tensor { outer, inner }` — factored for GPU
    fn pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        eq: EqInput<'_, Self, F>,
        kernel: &Self::CompiledKernel<F>,
        num_evals: usize,
        order: BindingOrder,
    ) -> Vec<F>;
}

enum EqInput<'a, B: ComputeBackend, F: Field> {
    Flat(&'a B::Buffer<F>),
    Unit,
    Tensor {
        outer: &'a B::Buffer<F>,
        inner: &'a B::Buffer<F>,
    },
}
```

This removes `pairwise_reduce_unweighted` and `tensor_pairwise_reduce` as separate
trait methods. The backend dispatches internally based on the `EqInput` variant.

### In build_witness()

No more per-shape eq handling. The descriptor's `eq_strategy` tells the prover exactly
what to do:

```rust
fn build_witness(...) -> Box<dyn SumcheckCompute<F>> {
    let desc = formula.to_kernel_descriptor();
    let kernel = backend.compile_kernel(&desc, &challenges);
    let input_bufs = upload_polynomial_tables(&desc.formula, ...);

    let eq_input = match desc.eq_strategy {
        EqStrategy::AsInput => {
            // eq is already in input_bufs[0]
            EqSetup::Unit
        }
        EqStrategy::Factored => {
            let partial_eq = compute_partial_eq(&eq_point[1..]);
            EqSetup::Flat(backend.upload(&partial_eq))
        }
        EqStrategy::TensorSplit(ts) => {
            let (outer, inner) = compute_tensor_eq(&eq_point, ts);
            EqSetup::Tensor(backend.upload(&outer), backend.upload(&inner))
        }
    };

    Box::new(KernelEvaluator::new(input_bufs, eq_input, kernel, desc, backend))
}
```

One code path. No match on `KernelShape`. The eq strategy is declared, not inferred.

### In KernelEvaluator

One interpolation mode, parameterized by `EqStrategy`:

```rust
fn round_polynomial(&self) -> UnivariatePoly<F> {
    let raw_evals = self.backend.pairwise_reduce(
        &self.inputs, self.eq_input, &self.kernel, self.num_evals, self.order
    );
    self.reconstruct(raw_evals)
}
```

Reconstruction is the same regardless of eq strategy — the raw_evals are always
"evaluations of the full composition at grid points". Whether the grid is standard
or Toom-Cook is determined by the formula structure (which the backend pattern-matched
during compilation), not by eq handling.

### What This Eliminates

- `pairwise_reduce_unweighted` method
- `tensor_pairwise_reduce` method
- `tensor_pairwise_reduce_fixed` method
- `InterpolationMode` enum in KernelEvaluator
- `ToomCookState` struct
- Per-shape eq setup in `build_witness()`
- The conceptual split between "eq as input" and "eq as weight"

## Decisions

- Three eq variants: `TensorEq`, `Eq`, `None` — clean separation, no more 3 calling conventions
- This is the right granularity — tensor decomposition is a genuine algorithmic difference, not just an optimization
