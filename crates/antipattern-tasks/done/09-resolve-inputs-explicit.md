# Task 09: Make Buffer Resolution Explicit Ops

## Status: TODO

## Anti-Pattern
`runtime.rs` `resolve_inputs()` (~100 lines) is a runtime function that:
1. For each kernel input binding, checks if a buffer exists and is "fresh"
2. Based on `force_refresh` flag and binding type, decides whether to re-materialize
3. Dispatches to different materialization paths per `InputBinding` variant

The runtime makes caching decisions that the compiler could express as explicit materialization ops.

## Current Code Pattern
```rust
fn resolve_inputs(..., force_refresh: bool, ...) {
    for binding in &kdef.inputs {
        let skip = if let Some(buf) = device_buffers.get(&pi) {
            match binding {
                InputBinding::Provided { .. } => {
                    if force_refresh { data.len() == expected_size } else { true }
                }
                _ => !force_refresh,
            }
        } else { false };
        
        if skip { continue; }
        
        match binding {
            InputBinding::Provided { poly } => { provider.materialize(*poly) }
            InputBinding::EqTable { challenges, .. } => { backend.eq_table(...) }
            InputBinding::EqProject { source, challenges, .. } => { /* 30-line projection */ }
            InputBinding::BytecodeVal { .. } => { /* specialized */ }
            // ... 10+ variants
        }
    }
}
```

## Fix

### Option A: Explicit materialize ops per binding type
```rust
Op::MaterializeProvided { poly: PolynomialId }
Op::MaterializeEqTable { poly: PolynomialId, challenges: Vec<usize> }
Op::MaterializeEqProject { poly: PolynomialId, source: PolynomialId, challenges: Vec<usize>, ... }
```

This is too many ops and bloats the op vocabulary.

### Option B: Keep InstanceResolveInputs but make force_refresh compile-time
The `InstanceResolveInputs` op from Task 04 already specifies `force_refresh: bool`. We keep `resolve_inputs()` as a runtime helper but the decision of WHEN to call it is compiler-controlled.

The key insight: `resolve_inputs` is a **buffer cache manager**, not protocol logic. The protocol decision is "when do inputs need refreshing" — that's what the compiler controls. The materialization itself (how to build an eq table, how to project) is implementation detail that belongs in the runtime/backend.

### Recommendation: Option B (now) with path to Option A (later)

For now:
1. `Op::InstanceResolveInputs { batch, instance, kernel, force_refresh }` calls `resolve_inputs()` — clean boundary
2. The compiler decides force_refresh based on phase structure (true at activation, false mid-phase)
3. `resolve_inputs()` stays as-is but is only called from the explicit op handler

Later (post-stage 6):
- Push `EqProject` and other complex materializations into the backend trait
- Backend.materialize(binding, challenges) → buffer

## What Changes Now

### Compiler
Emits `InstanceResolveInputs` at the right points in the unrolled sequence (phase starts only).

### Runtime
`resolve_inputs()` is called from `Op::InstanceResolveInputs` handler instead of inline in the BatchedSumcheckRound loop.

### EqProject Direction Decision
Lines 1302-1359: The runtime decides projection direction based on eq table size vs inner/outer size. This is a genuine runtime decision because it depends on challenge values (eq table zeros). However, the SHAPES are known at compile time. Consider: the compiler could emit `ProjectInner` or `ProjectOuter` variants if it knows the dimension to project. The eq table sparsity is a runtime optimization, not a direction decision.

## Test
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
```

## Risk: Low
This is mostly about clean op boundaries. The actual materialization code doesn't change.

## Dependencies: Task 05 (InstanceResolveInputs op exists)
