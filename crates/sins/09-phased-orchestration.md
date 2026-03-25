# Analysis 9: Orchestrator Owns Phase Sequencing, Backend Owns Kernels

## The Question

Multi-phase sumchecks (RAM read-write has 3 phases, claim reductions have 2) need to
switch algorithms within a single sumcheck. Who picks the kernel for each phase?

## Current State

- `PhasedEvaluator` (jolt-sumcheck): composes arbitrary `SumcheckCompute` impls with
  transition closures. Each phase is a completely independent evaluator.
- `SegmentedEvaluator` (jolt-zkvm): chains `KernelEvaluator` instances with per-segment
  transitions and re-parameterization hooks.

Both work but have different responsibilities mixed:
- Phase sequencing (how many rounds per phase, when to transition) — orchestration concern
- Kernel selection per phase — should be driven by the execution plan
- Buffer management across phases (what's carried forward) — backend concern

## Proposed: Phased Execution Plans

The `ExecutionPlan` (from Sin 2/3) extends to multi-phase:

```rust
pub struct ExecutionPlan {
    pub phases: Vec<PhasePlan>,
}

pub struct PhasePlan {
    /// Number of sumcheck rounds in this phase.
    pub num_rounds: SymbolicExpr,
    /// Algorithm for this phase.
    pub algorithm: Algorithm,
    /// Composition formula for this phase's kernel.
    pub formula: CompositionFormula,
    /// How to handle eq in this phase.
    pub eq_strategy: EqStrategy,
    /// Binding order for this phase.
    pub binding_order: BindingOrder,
    /// What to carry forward to the next phase.
    pub transition: Option<TransitionSpec>,
}

pub enum Algorithm {
    Dense,
    Sparse,
}

pub struct TransitionSpec {
    /// Which buffers from this phase carry into the next.
    pub carry_buffers: Vec<usize>,
    /// Additional polynomials to materialize at the transition point.
    pub materialize: Vec<PolynomialId>,
}
```

## How the Evaluator Executes This

```rust
impl<F: Field, B: ComputeBackend> SumcheckCompute<F> for PhasedEvaluator<F, B> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let phase = &self.plan.phases[self.current_phase];
        match phase.algorithm {
            Algorithm::Dense => {
                let raw = self.backend.pairwise_reduce(...);
                self.reconstructor.reconstruct(&raw, self.claim)
            }
            Algorithm::Sparse => {
                let raw = self.backend.sparse_reduce(...);
                self.reconstructor.reconstruct(&raw, self.claim)
            }
        }
    }

    fn bind(&mut self, challenge: F) {
        // ... bind current phase's buffers ...
        self.round_in_phase += 1;

        // Phase transition?
        if self.round_in_phase == self.current_phase_rounds() {
            self.transition_to_next_phase(challenge);
        }
    }
}
```

The evaluator is a generic executor of `PhasePlan` sequences. No phase-specific code.
The plan fully describes what happens in each phase.

## AOT Kernel Compilation

All kernels for all phases are compiled at setup time (before the sumcheck loop):

```rust
fn setup_phased_evaluator(plan: &ExecutionPlan, backend: &B) -> PhasedEvaluator<F, B> {
    let kernels: Vec<B::CompiledKernel<F>> = plan.phases.iter().map(|phase| {
        backend.compile_kernel(&phase.formula, &phase_challenges, phase.formula.degree)
    }).collect();
    // All kernels compiled. No compilation during sumcheck rounds.
    PhasedEvaluator { plan, kernels, ... }
}
```

## What This Gives Us

1. **Freedom to switch algorithms per phase**: Dense for some rounds, sparse for others,
   all declared in the plan.
2. **AOT compilation for all phases**: No JIT, no runtime kernel selection.
3. **Backend stays dumb**: It compiles kernels and runs them. The orchestrator sequences.
4. **The plan is the single source of truth**: Both prover and (if needed) verifier can
   inspect the plan to understand the proof strategy.
5. **New phases are data, not code**: Adding a new phase to a sumcheck means adding an
   entry to the plan, not writing a new evaluator class.

## Decisions — DISSOLVED by Analysis 11

Phased execution within a vertex is eliminated. Phases become separate vertices in the graph.
Multi-phase sumchecks (e.g., RAM read-write: sparse → sparse → dense) are expressed as
multiple vertices, not one vertex with a `Vec<PhasePlan>`. The graph IS the phase sequencer.

- No `PhasePlan`, no `TransitionSpec`, no phase tracking in the evaluator
- Buffer materialization between phases becomes edge computations between vertices
- AOT kernel compilation per vertex (one algorithm, one kernel) — no switching
- See [Analysis 11](11-unified-execution-model.md) for the unified model
