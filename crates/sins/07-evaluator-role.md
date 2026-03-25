# Analysis 7: Rethinking the Evaluator's Role

## The Confusion

`KernelEvaluator` currently does two distinct jobs:

1. **Kernel dispatch**: calls `backend.pairwise_reduce()` and `backend.interpolate_pairs()`
2. **Reconstruction**: converts raw evaluations → `UnivariatePoly` (grid-specific math)

Job 1 belongs in the compute layer. Job 2 is protocol math (claim derivation, Toom-Cook
interpolation, eq factor multiplication). These are tangled in one struct via the
`InterpolationMode` enum, which is why the grid choice (a backend decision) leaks into
the evaluator (an orchestration-layer type).

## What the Evaluator Should Be

The evaluator should be a **thin adapter** between the protocol engine (jolt-sumcheck)
and the compute backend. It implements `SumcheckCompute` by:

1. Asking the backend for raw evaluations (a fixed-size `Vec<F>`)
2. Asking a **reconstructor** to convert raw evals → round polynomial
3. Asking the backend to bind (interpolate pairs)

The reconstructor is the piece that knows about grids:

```rust
/// Converts backend output to a round polynomial.
/// Knows about eval grids but NOT about backends or buffers.
trait Reconstructor<F: Field> {
    /// Given raw evaluations from the backend and the current running claim,
    /// produce the full round polynomial.
    fn reconstruct(&self, raw_evals: &[F], claim: F) -> UnivariatePoly<F>;

    /// Update internal state after a round (e.g., advance Toom-Cook eq scalars).
    fn advance(&mut self, challenge: F);
}
```

Two implementations:

```rust
/// Standard grid {0, 2, ..., degree}: derive P(1) = claim - P(0), interpolate.
struct StandardReconstructor { degree: usize }

/// Toom-Cook grid {1, ..., D-1, ∞}: recover h(0) from claim, interpolate,
/// multiply by outer_scalar · eq_single(w_k, X).
struct ToomCookReconstructor<F> { eq_w: Vec<F>, round: usize, outer_scalar: F, ... }
```

## Who Decides Which Reconstructor?

The backend, when it compiles the kernel. The backend already knows whether it chose
Toom-Cook or standard grid — it should return that information alongside the compiled
kernel:

```rust
trait ComputeBackend {
    type CompiledKernel<F: Field>;

    /// Compile a composition formula. Returns the kernel AND the grid metadata
    /// needed for reconstruction.
    fn compile_kernel<F: Field>(
        &self,
        formula: &CompositionFormula,
        challenges: &[F],
        degree: usize,
    ) -> (Self::CompiledKernel<F>, EvalGridInfo);
}

/// Metadata the backend returns so the evaluator can reconstruct correctly.
enum EvalGridInfo {
    /// Standard grid: num_evals evaluations on {0, 2, ..., degree}.
    Standard { num_evals: usize },
    /// Toom-Cook grid: D evaluations on {1, ..., D-1, ∞}.
    ToomCook { d: usize },
}
```

The evaluator constructs the appropriate `Reconstructor` from `EvalGridInfo`. No
`InterpolationMode` enum needed — the backend's choice flows through data, not
through evaluator construction paths.

## New Evaluator Structure

```rust
pub struct SumcheckEvaluator<F: Field, B: ComputeBackend> {
    inputs: Vec<B::Buffer<F>>,
    eq: EqInput<B, F>,
    kernel: B::CompiledKernel<F>,
    reconstructor: Box<dyn Reconstructor<F>>,
    backend: Arc<B>,
    binding_order: BindingOrder,
}

impl<F: Field, B: ComputeBackend> SumcheckCompute<F> for SumcheckEvaluator<F, B> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let raw = self.backend.pairwise_reduce(
            &self.input_refs(), self.eq.as_ref(), &self.kernel,
            self.reconstructor.num_evals(), self.binding_order,
        );
        self.reconstructor.reconstruct(&raw, self.claim)
    }

    fn bind(&mut self, challenge: F) {
        self.backend.interpolate_pairs_batch_inplace(&mut self.inputs, challenge, self.binding_order);
        self.eq.bind(&self.backend, challenge, self.binding_order);
        self.reconstructor.advance(challenge);
    }
}
```

## What This Clarifies

- **Backend** owns: kernel compilation, grid choice, raw evaluation, binding
- **Reconstructor** owns: claim derivation, interpolation, eq factor math
- **Evaluator** owns: nothing — it's glue between backend and reconstructor
- **Orchestrator** (prove_from_graph) owns: constructing the evaluator from an ExecutionPlan

No layer does another layer's job. The evaluator doesn't interpret shapes or modes —
it delegates everything.

## Decisions — DISSOLVED by Analysis 11

The evaluator as a separate architectural concept is eliminated. Both jobs (kernel dispatch
and reconstruction) move into the backend behind an associated type.

### The backend factory model

```rust
trait ComputeBackend {
    type CompiledKernel<F: Field>: Send + Sync;
    type SumcheckWitness<F: Field>: SumcheckCompute<F>;

    /// AOT: compile formula structure into a kernel.
    /// The compiled kernel encapsulates: algorithm, reconstruction strategy,
    /// eq handling, grid choice — everything the evaluator used to decide.
    /// Challenges are symbolic — runtime parameters, not baked in.
    fn compile_kernel<F: Field>(&self, formula: &CompositionFormula) -> Self::CompiledKernel<F>;

    /// Runtime: pair a pre-compiled kernel with data to get a ready-to-use
    /// SumcheckCompute impl. This replaces both build_witness() and KernelEvaluator.
    fn make_witness<F: Field>(
        &self,
        kernel: &Self::CompiledKernel<F>,
        inputs: Vec<Self::Buffer<F>>,
        challenges: &[F],
    ) -> Self::SumcheckWitness<F>;
}
```

### What goes where

- `KernelEvaluator` — deleted. Replaced by `B::SumcheckWitness<F>`.
- `InterpolationMode` — baked into `CompiledKernel` at AOT compile time.
- `ToomCookState` — internal to the backend's `SumcheckWitness` impl.
- `Reconstructor` trait — unnecessary as a separate abstraction. Reconstruction is internal
  to the backend's `SumcheckWitness::round_polynomial()`.
- `build_witness()` (160-line 4-arm match in jolt-zkvm) — replaced by `backend.make_witness()`.
- `compile_descriptor()` (in jolt-ir) — replaced by `backend.compile_kernel()`.

### Setup and proving flow

```
SETUP (jolt-zkvm walks jolt-ir graph, asks backend to compile):
  jolt-ir: ProtocolGraph with CompositionFormula on each sumcheck vertex
      ↓
  jolt-zkvm: for v in graph.sumcheck_vertices() {
      kernels[v.id] = backend.compile_kernel(&v.formula);
  }
      ↓
  jolt-compute backend impl (e.g. CpuBackend):
      pattern-matches formula → picks Toom-Cook / specialized / general SoP
      bakes reconstruction strategy into CompiledKernel

PROVING (jolt-zkvm graph executor, per sumcheck vertex):
  let inputs = source.resolve(vertex.input_bindings);
  let challenges = cache.get_runtime_challenges(vertex);
  let mut witness = backend.make_witness(&kernels[&vertex.id], inputs, &challenges);
  // jolt-sumcheck owns the round loop:
  SumcheckProver::prove(&claim, &mut witness, transcript);
```

### Dependency chain

```
jolt-ir (owns CompositionFormula)
    ↓ consumed by
jolt-compute (ComputeBackend trait: compile_kernel takes &CompositionFormula)
    ↓ implemented by
jolt-cpu-kernels / jolt-metal (pattern-match formula → kernel + reconstruction)
    ↓ orchestrated by
jolt-zkvm (walks graph at setup, calls compile_kernel per vertex, stores kernel map)
```

The backend is a consumer of jolt-ir formula types. jolt-compute depends on jolt-ir for
`CompositionFormula`. Concrete backends depend on jolt-compute + jolt-ir.

- See [Analysis 11](11-unified-execution-model.md) for the full unified model
