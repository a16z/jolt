# Segmented Evaluator: ML-Compiler-Style Kernel Composition

**Date:** 2026-03-12
**Branch:** `refactor/crates`
**Context:** Final infrastructure for InstructionReadRaf + BytecodeReadRaf stages

---

## Philosophy

Every round of every sumcheck should be a `pairwise_reduce` call with a compiled kernel.
No hand-written `SumcheckCompute` loops. Full backend genericity (CPU/GPU/WebGPU).

If `pairwise_reduce` alone isn't expressive enough, expand the compute API —
don't fall back to custom code. Think ML-compiler: express the computation in
the IR, compile it to a kernel, let the backend handle parallelism.

---

## Current Stack

```
Layer 0: ComputeBackend::pairwise_reduce(inputs, weights, kernel, num_evals, order) → [F; D]
         ↑ one kernel, reads pairs from all inputs, accumulates D weighted sums

Layer 1: KernelDescriptor { shape: KernelShape, degree, tensor_split }
         ↑ describes the pointwise function compiled into a kernel
         ↑ ProductSum, Custom(Expr), EqProduct, HammingBooleanity

Layer 2: KernelEvaluator implements SumcheckCompute
         ↑ round_polynomial() = pairwise_reduce() → interpolate
         ↑ bind()             = interpolate_pairs_batch()
         ↑ ONE kernel, ONE buffer set, ALL rounds identical
```

Every round: **one pairwise_reduce → one interpolation → one UnivariatePoly**.
The kernel evaluates the same formula at every pair position. Buffers halve
each round. This is the invariant.

---

## What Breaks for ReadRaf Stages

Both ReadRaf stages have TWO variable groups (address + cycle) with:

1. **Different formulas** — address-phase kernel ≠ cycle-phase kernel
2. **Different buffer sets** — address buffers consumed, cycle buffers materialized at transition
3. **Different interpolation modes** — address may use StandardGrid, cycle uses ToomCook
4. **Parametric challenges** — checkpoint values update every 2 rounds within address phases

But critically: **within each segment, every round IS a standard pairwise_reduce**.
The gap is in composing segments, not in the per-round primitive.

---

## Changes Required

### 1. `KernelEvaluator`: configurable `BindingOrder`

Currently hardcoded to `LowToHigh` in `reduce_raw()`. Address phases may
need `HighToLow`. Add a field, pass it through to `pairwise_reduce` and
`interpolate_pairs_batch_inplace`.

**File:** `crates/jolt-zkvm/src/evaluators/kernel.rs`

```rust
pub struct KernelEvaluator<F: Field, B: ComputeBackend> {
    // ...existing fields...
    binding_order: BindingOrder,  // NEW — defaults to LowToHigh
}
```

Changes:
- Add `binding_order` field, initialize in `new_with_mode`
- `reduce_raw()`: use `self.binding_order` instead of `BindingOrder::LowToHigh`
- `bind()`: use `interpolate_pairs_batch_inplace` with `self.binding_order`
  (currently uses `interpolate_pairs_batch` which is LowToHigh-only)
- Add builder method: `with_binding_order(mut self, order: BindingOrder) -> Self`
- All existing constructors default to `LowToHigh` (no behavior change)
- All existing tests pass unchanged

**Estimated:** ~20 lines changed

### 2. `KernelEvaluator`: kernel re-parameterization

For every-2-round checkpoint updates in InstructionReadRaf address phases.
The kernel formula stays the same, but challenge values change.

**File:** `crates/jolt-zkvm/src/evaluators/kernel.rs`

```rust
impl<F: Field, B: ComputeBackend> KernelEvaluator<F, B> {
    /// Replace the compiled kernel with one using updated challenge values.
    ///
    /// The descriptor shape must match the original. On CPU this is ~free
    /// (new closure capture). On GPU, use push constants / uniform buffers
    /// to avoid shader recompilation.
    pub fn update_kernel(&mut self, kernel: B::CompiledKernel<F>) {
        self.kernel = kernel;
    }
}
```

**Estimated:** ~5 lines

### 3. `SegmentedEvaluator`: kernel composition across phases

**File:** `crates/jolt-zkvm/src/evaluators/segmented.rs` (NEW)

Core type that chains `KernelEvaluator` instances with transition callbacks:

```rust
/// Composes multiple kernel evaluators across phase boundaries within a
/// single sumcheck instance.
///
/// Each segment is a standard `KernelEvaluator` — every round within a
/// segment is a `pairwise_reduce` call with a compiled kernel. At segment
/// boundaries, a transition callback materializes new buffers and produces
/// the next evaluator.
///
/// # Motivation
///
/// ReadRaf sumchecks have two variable groups (address + cycle) with
/// different kernels, buffer sets, and interpolation modes. Rather than
/// writing custom `SumcheckCompute` impls with hand-coded round loops,
/// `SegmentedEvaluator` composes existing `KernelEvaluator` instances
/// so that every round stays within the backend-generic kernel framework.
///
/// # Round hooks
///
/// Optional per-round hooks run after each `bind()` within a segment.
/// Used for intra-segment re-parameterization (e.g., recompiling the
/// kernel with updated checkpoint values every 2 rounds).
pub struct SegmentedEvaluator<F: Field, B: ComputeBackend> {
    /// Active segment's kernel evaluator.
    current: KernelEvaluator<F, B>,

    /// Remaining segments: (rounds_in_segment, transition_fn).
    /// VecDeque for O(1) pop_front.
    continuations: VecDeque<(usize, SegmentTransition<F, B>)>,

    /// Rounds completed within the current segment.
    round_in_segment: usize,

    /// Total rounds in the current segment.
    rounds_this_segment: usize,

    /// All challenges accumulated across all segments (for transitions).
    challenges: Vec<F>,

    /// Optional hook called after each bind() within the current segment.
    /// Receives (round_in_segment, challenge, &mut KernelEvaluator).
    round_hook: Option<RoundHook<F, B>>,

    /// Hooks for future segments, consumed at each transition.
    future_hooks: VecDeque<Option<RoundHook<F, B>>>,

    backend: Arc<B>,
}

/// Transition callback: receives all challenges so far + backend ref,
/// returns the next segment's KernelEvaluator.
pub type SegmentTransition<F, B> =
    Box<dyn FnOnce(Vec<F>, &Arc<B>) -> KernelEvaluator<F, B> + Send + Sync>;

/// Per-round hook: called after bind() with (round_in_segment, challenge, evaluator).
/// Can re-parameterize the kernel, update weights, etc.
pub type RoundHook<F, B> =
    Box<dyn FnMut(usize, F, &mut KernelEvaluator<F, B>) + Send + Sync>;
```

**`SumcheckCompute` implementation:**

```rust
impl<F: Field, B: ComputeBackend> SumcheckCompute<F> for SegmentedEvaluator<F, B> {
    fn set_claim(&mut self, claim: F) {
        self.current.set_claim(claim);
    }

    fn first_round_polynomial(&self) -> Option<UnivariatePoly<F>> {
        self.current.first_round_polynomial()
    }

    fn round_polynomial(&self) -> UnivariatePoly<F> {
        self.current.round_polynomial()
    }

    fn bind(&mut self, challenge: F) {
        self.current.bind(challenge);
        self.challenges.push(challenge);

        // Fire round hook if present
        if let Some(ref mut hook) = self.round_hook {
            hook(self.round_in_segment, challenge, &mut self.current);
        }

        self.round_in_segment += 1;

        // Check for segment transition
        if self.round_in_segment == self.rounds_this_segment {
            if let Some((next_rounds, transition)) = self.continuations.pop_front() {
                let challenges = self.challenges.clone();
                self.current = transition(challenges, &self.backend);
                self.round_in_segment = 0;
                self.rounds_this_segment = next_rounds;
                self.round_hook = self.future_hooks.pop_front().flatten();
            }
        }
    }
}
```

**Builder API:**

```rust
impl<F: Field, B: ComputeBackend> SegmentedEvaluator<F, B> {
    /// Start building from the first segment.
    pub fn new(
        first_segment: KernelEvaluator<F, B>,
        rounds: usize,
        backend: Arc<B>,
    ) -> Self { ... }

    /// Add a continuation segment with its transition callback.
    pub fn then(
        mut self,
        rounds: usize,
        transition: SegmentTransition<F, B>,
    ) -> Self { ... }

    /// Attach a round hook to the current (last added) segment.
    pub fn with_round_hook(mut self, hook: RoundHook<F, B>) -> Self { ... }
}
```

**Estimated:** ~120 lines (struct + impl + builder + SumcheckCompute)

---

## How Each ReadRaf Stage Maps

### BytecodeReadRaf — 2 segments, no hooks

```
Segment 0 (address, log_K rounds):
  KernelEvaluator(Custom, degree 2, LowToHigh, StandardGrid)
  kernel = γ⁰·opening(0)·opening(1) + γ¹·opening(2)·opening(3) + ...
           + γ⁵·opening(0)·opening(10) + γ⁶·opening(4)·opening(10)
  inputs = [F₀, Val₀, F₁, Val₁, F₂, Val₂, F₃, Val₃, F₄, Val₄, Int]
  weights = unit (no eq in address phase)

  → transition(challenges, backend):
    1. Evaluate bound Val polys at challenge point → 5 scalars
    2. Precompute combined_eq from 5 per-stage GruenSplit eq polys:
       combined_eq(j) = Σ_s γ^s · bound_val_s · eq_s(j) + raf terms
    3. Materialize RA polys from trace
    4. Return KernelEvaluator::with_toom_cook_eq(...)

Segment 1 (cycle, log_T rounds):
  KernelEvaluator(ProductSum(d,1), degree d+1, LowToHigh, ToomCook)
  inputs = [ra₀, ra₁, ..., ra_{d-1}]
  weights = combined_eq (Toom-Cook factored)
```

### InstructionReadRaf — 5 segments (4 address + 1 cycle), with hooks

```
Segments 0-3 (address phases, log_m rounds each):
  KernelEvaluator(Custom, degree 3, HighToLow, StandardGrid)
  kernel = Σ_t checkpoint_t · opening(weight_idx) · opening(suffix_t_idx)
           + γ · opening(P_left) · opening(Q_left)
           + γ² · opening(P_right) · opening(Q_right)
           + identity_coeff · opening(P_id) · opening(Q_id)
  inputs = [expanding_weight, suffix₀, ..., suffix_{N}, P_l, Q_l, P_r, Q_r, P_id, Q_id]
  weights = unit

  round_hook = every 2 rounds: recompile kernel with updated prefix checkpoints
    |round, _challenge, evaluator| {
        if round % 2 == 1 {
            let new_kernel = backend.compile_kernel_with_challenges(&desc, &new_checkpoints);
            evaluator.update_kernel(new_kernel);
        }
    }

  → transition(challenges, backend):
    1. Condense u_evals through expanding table
    2. Reinitialize suffix polys for next address chunk
    3. Reinitialize P/Q tables from prefix registry
    4. Update expanding table buffer
    5. Compile kernel with new checkpoint values
    6. Return new KernelEvaluator

Segment 4 (cycle, log_T rounds):
  KernelEvaluator(ProductSum(d+1,1), degree d+2, LowToHigh, ToomCook)
  inputs = [ra₀, ..., ra_{d-1}, combined_val]
  weights = eq(r_reduction, ·) (Toom-Cook factored)

  → transition(challenges, backend):
    1. Materialize ra_polys = products of expanding tables across phases
    2. Compute combined_val = Val(r_address) + γ · RafVal(r_address)
    3. Return KernelEvaluator::with_toom_cook_eq(...)
```

---

## Why No New Kernel Shapes

A `KernelShape` describes a **pointwise function** evaluated at each pair position.
The ReadRaf stages don't need new pointwise functions:

| Computation | Existing Shape | Notes |
|---|---|---|
| `Σ_t c_t · F_t · Val_t` (Bytecode address) | `Custom(Expr)` | sum of weighted products, degree 2 |
| `Σ_t c_t · weight · suffix_t + raf_terms` (Instruction address) | `Custom(Expr)` | sum of weighted products, degree 3 |
| `Π_i ra_i` (Bytecode cycle) | `ProductSum(d, 1)` | exact fit |
| `Π_i ra_i · combined_val` (Instruction cycle) | `ProductSum(d+1, 1)` | include combined_val as factor |

The gap was never in "what to compute per position" (Layer 1). It was in
"how to compose different per-position computations across rounds" (Layer 2-3).
`SegmentedEvaluator` fills that gap without touching Layers 0-1.

---

## Why No New ComputeBackend Primitives

`pairwise_reduce` is sufficient for every round:
- Address phases: one `pairwise_reduce` with a (possibly large) Custom kernel
- Cycle phases: one `pairwise_reduce` with ProductSum kernel

The per-table combine formulas, prefix checkpoints, and RAF decompositions are
all expressible in a single `Custom(Expr)`. The Expr may be large (30+ tables
× suffixes + 3 decomposition pairs) but:
- On CPU: compiles to a single inlined closure. No AST walk overhead after
  monomorphization if we specialize per-table.
- On GPU: one shader dispatch reading from ~50-80 input buffers. Modern GPUs
  handle this fine.

If profiling later shows the large Custom kernel is slow, we can add
`grouped_pairwise_reduce` (multiple kernels on different buffer subsets,
one pass over positions). But that's an optimization, not a correctness
requirement.

---

## Implementation Order

```
Step 1: KernelEvaluator — add binding_order field
        ~20 lines changed in kernel.rs
        Run existing tests (must all pass unchanged)

Step 2: KernelEvaluator — add update_kernel() method
        ~5 lines in kernel.rs

Step 3: SegmentedEvaluator — new file segmented.rs
        ~120 lines: struct, builder, SumcheckCompute impl
        Tests: 2-segment prove/verify (StandardGrid → ToomCook transition)
               3-segment prove/verify (multiple StandardGrid → ToomCook)
               round-hook test (re-parameterize kernel every 2 rounds)

Step 4: Wire into evaluators/mod.rs exports

Step 5: Update stage_parity_plan.md with new infrastructure
```

Each step is independently testable. Step 3 is the bulk of the work.

---

## Test Strategy

### Unit tests for SegmentedEvaluator

1. **Two-segment prove/verify (simple):**
   Synthetic data. Segment 0: `eq · g` (Custom, degree 2, 3 rounds).
   Transition: materialize new buffers from bound state.
   Segment 1: `eq · h` (Custom, degree 2, 3 rounds).
   Verify round-trip.

2. **Two-segment StandardGrid → ToomCook:**
   Segment 0: Custom formula (StandardGrid, degree 2, M rounds).
   Transition: compute product inputs + eq weights.
   Segment 1: ProductSum(D, 1) (ToomCook, degree D+1, N rounds).
   This directly exercises the BytecodeReadRaf pattern.

3. **Multi-segment with round hooks:**
   4 segments of 2 rounds each. Hook recompiles kernel with new challenge
   values after each round. Verify prove/verify round-trip matches brute-force.

4. **HighToLow segment → LowToHigh segment:**
   Tests the binding_order field across a transition. Segment 0 uses
   HighToLow, segment 1 uses LowToHigh. Exercises the InstructionReadRaf
   address→cycle pattern.

5. **Single-segment (degenerate):**
   SegmentedEvaluator with 0 continuations should behave identically to
   a plain KernelEvaluator.

### Integration with existing stages

After SegmentedEvaluator passes its unit tests, the ReadRaf stages
(InstructionReadRaf, BytecodeReadRaf) will be implemented as `ProverStage`
types whose `build()` returns a `SegmentedEvaluator`. Those are separate
work items tracked in stage_parity_plan.md.

---

## Relationship to PrefixSuffixEvaluator

`PrefixSuffixEvaluator` is a specialized two-phase evaluator where Phase 1
is hardcoded to `Σ P_i · Q_i` (degree 2, HighToLow). It could be
reimplemented as a `SegmentedEvaluator` where Segment 0 uses a
`Custom` kernel for the P·Q inner product. But:

- PrefixSuffixEvaluator provides useful helpers (`combined_suffix`, `prefix_evals`)
- It's already working and tested (5 tests, used by ShiftSumcheckStage)
- Refactoring it would be churn with no functional benefit

They coexist. PrefixSuffixEvaluator = specialized convenience wrapper.
SegmentedEvaluator = general-purpose composition mechanism.

---

## Future: `compile_kernel_template` / `parameterize` Split

Currently `compile_kernel_with_challenges` bakes challenge values at compile
time. For GPU backends, separating shape compilation (expensive) from challenge
parameterization (cheap push constants) avoids shader recompilation when
checkpoint values update:

```rust
fn compile_kernel_template(&self, desc: &KernelDescriptor) -> CompiledKernelTemplate<F>;
fn parameterize(&self, template: &CompiledKernelTemplate<F>, challenges: &[F]) -> CompiledKernel<F>;
```

On CPU, both are ~free (closure capture). This refinement is deferred until
GPU backends are integrated. The round-hook + `update_kernel` approach works
for all backends today.
