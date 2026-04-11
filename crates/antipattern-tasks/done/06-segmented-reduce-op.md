# Task 06: Segmented Reduce as Explicit Op

## Status: TODO

## Anti-Pattern
`runtime.rs` `segmented_reduce()` (~70 lines) is a runtime sub-interpreter that:
1. Iterates over outer eq positions
2. Skips zero-weight positions
3. Extracts inner columns from mixed-size buffers
4. Calls `backend.reduce()` per column
5. Accumulates with outer eq weights

This is iteration orchestration. The compiler knows the segmentation structure (inner/outer sizes, which inputs are inner-only) and could express this differently.

## Current Code (runtime.rs)
```rust
fn segmented_reduce(...) -> Vec<F> {
    for (a, &weight) in outer_eq.iter().enumerate() {
        if weight.is_zero() { continue; }
        let mut col_bufs = Vec::new();
        for (j, data) in input_data.iter().enumerate() {
            if seg.inner_only[j] {
                col_bufs.push(upload(data));
            } else {
                let start = a * inner_size;
                col_bufs.push(upload(&data[start..start + inner_size]));
            }
        }
        let evals = backend.reduce(compiled_kernel, &col_refs, challenges);
        accumulate with weight...
    }
}
```

## Design Options

### Option A: Keep as dedicated op, move logic to backend
Add `Op::InstanceSegmentedReduce` (from Task 04) with the runtime handler calling a backend method:
```rust
backend.segmented_reduce(compiled_kernel, &full_buffers, &inner_only, outer_eq, challenges)
```
This pushes the iteration into the backend where it can be GPU-optimized.

### Option B: Compiler unrolls outer loop
The compiler emits one `InstanceReduce` per non-zero outer position, with column extraction as explicit buffer ops. This is more granular but may be too many ops for large outer dimensions.

### Option C: New iteration pattern
Add `Iteration::Segmented { inner: Box<Iteration>, outer_eq_challenges: Vec<usize> }` that the backend natively supports. The kernel spec already has iteration patterns — segmented is just another one.

### Recommendation: Option A (immediate) → Option C (later)
Option A is the quickest path: the runtime handler is already written, we just move it behind a clean op boundary. Option C is the ideal long-term design but requires backend trait changes.

For now:
1. The new `Op::InstanceSegmentedReduce` from Task 04 handles the dispatch
2. The `segmented_reduce()` function stays but is called from the clean op handler
3. The three-way `if is_prefix_suffix / else if segmented / else standard` branch in BatchedSumcheckRound disappears because each case is a different op

## What Changes

### Compiler side
When emitting rounds for a segmented phase, emit `Op::InstanceSegmentedReduce` instead of `Op::InstanceReduce`.

### Runtime side
```rust
Op::InstanceSegmentedReduce { batch, instance, kernel, config } => {
    let outer_eq = state.segmented_outer_eqs.get(&(*batch, *instance)).unwrap();
    let evals = segmented_reduce(&device_buffers, outer_eq, config, ...);
    state.last_round_instance_evals[*instance] = evals;
}
```

Clean, explicit, no branching.

## Anticipating Stage 6
Stage 6 has `RamHammingBooleanity` which uses sparse iteration (address-only, cycles already bound). This is NOT the same as segmented reduce — sparse iteration is already an `Iteration::Sparse` pattern handled by the backend. But having clean segmented ops means we won't conflate the two.

## Test
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
```

## Risk: Low-Medium
The segmented_reduce function already works. We're just giving it a clean op wrapper.

## Dependencies: Task 04, Task 05
