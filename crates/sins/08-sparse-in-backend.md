# Analysis 8: Sparse Algorithm Should Be a Backend Concern

## Current State

`SparseRwEvaluator` in jolt-zkvm is a bespoke `SumcheckCompute` impl that walks sorted
entry lists. It doesn't use `ComputeBackend` at all — it's pure CPU code in the
orchestration layer. This means:

1. The sparse algorithm is coupled to the orchestration layer (jolt-zkvm)
2. No GPU acceleration path exists for sparse reduce
3. The `ComputeBackend` trait has no concept of sparse operations
4. The orchestrator must know whether to construct a `KernelEvaluator` (dense) or
   `SparseRwEvaluator` (sparse) — algorithm selection leaks into `build_witness`

## What Sparse Actually Is

Sparse reduce is: iterate over a sorted list of (position, values) entries, evaluate a
formula per entry, accumulate into round polynomial coefficients. The formula per entry
is still a sum-of-products expression — the same `CompositionFormula` works.

The difference from dense is not the FORMULA but the DATA ACCESS PATTERN:
- Dense: all 2^n positions have values, stored contiguously
- Sparse: only k << 2^n positions have non-zero values, stored as sorted entries

## Proposed: Expand the Backend Trait

```rust
trait ComputeBackend {
    // ... existing dense methods ...

    /// Sparse entry buffer on device.
    type SparseBuffer<F: Field>: Send + Sync;

    /// Upload sparse entries (sorted by position).
    fn upload_sparse<F: Field>(
        &self,
        entries: &[(usize, Vec<F>)],  // (position, values_per_entry)
    ) -> Self::SparseBuffer<F>;

    /// Sparse weighted reduce: evaluate kernel at each entry position,
    /// weight by eq[position], accumulate into num_evals sums.
    ///
    /// Unlike dense pairwise_reduce, this iterates entries (not all positions).
    /// GPU backends can use atomic accumulation or segment-reduce.
    fn sparse_reduce<F: Field>(
        &self,
        entries: &Self::SparseBuffer<F>,
        eq: &Self::Buffer<F>,
        kernel: &Self::CompiledKernel<F>,
        num_evals: usize,
    ) -> Vec<F>;

    /// Bind sparse entries at a challenge value.
    /// Merges adjacent entry pairs, halving the position space.
    fn sparse_bind<F: Field>(
        &self,
        entries: &mut Self::SparseBuffer<F>,
        challenge: F,
        order: BindingOrder,
    );
}
```

## Why This Matters

With sparse in the backend:

1. **GPU acceleration**: Sparse reduce maps to segment-reduce on GPU (each entry is
   independent). Not as parallel as dense, but still useful for large entry counts.

2. **Algorithm selection moves to the plan**: The `ExecutionPlan` (from Sin 2/3) declares
   `Algorithm::Dense` or `Algorithm::Sparse`, and the evaluator dispatches accordingly.
   No bespoke `SumcheckCompute` impl needed.

3. **Same kernel for both**: The `CompositionFormula` is identical for dense and sparse
   versions of the same sumcheck. Only the data access pattern differs. The backend
   compiles one kernel, dispatches it two ways.

4. **Multi-phase composability**: RAM read-write phases 1-2 (sparse) → phase 3 (dense)
   can be expressed as a sequence of plans, all going through the backend.

## What Changes

| Before | After |
|--------|-------|
| `SparseRwEvaluator` in jolt-zkvm (bespoke impl) | Generic evaluator + `backend.sparse_reduce()` |
| Algorithm selection in `build_witness()` | Algorithm declared in `ExecutionPlan` |
| Sparse is CPU-only | Sparse can be GPU-accelerated |
| Dense and sparse have different code paths | Same formula, same kernel, different dispatch |

## CPU Backend Implementation

For CpuBackend, `SparseBuffer<F> = Vec<SparseEntry<F>>` where entries are sorted by
position. `sparse_reduce` walks entries and accumulates. `sparse_bind` merges adjacent
pairs. This is exactly what `SparseRwEvaluator` does today — just moved behind the trait.

## Transition for Multi-Phase Sumchecks

A RAM read-write sumcheck currently uses:
- Phase 1: `SparseRwEvaluator` (cycle variables, sparse)
- Phase 2: `SparseRwEvaluator` (address variables, sparse)
- Phase 3: `KernelEvaluator` (remaining variables, dense)

With sparse in the backend, the orchestrator (SegmentedEvaluator/PhasedEvaluator) just
switches between `backend.sparse_reduce()` and `backend.pairwise_reduce()` per phase.
The evaluator struct is the same — only the dispatch method changes.

## Decisions

- Sparse is a first-class backend primitive — both CPU and GPU need it
- Evaluator does NOT know sparse vs dense — it dispatches what the plan says
- Sparsity is a property of the polynomial, declared in the IR, consumed by the backend
