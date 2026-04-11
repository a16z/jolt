# Task 13: Inline Phase Config Into Ops (Eliminate Runtime Phase Search)

## Status: TODO

## Anti-Pattern
Two op handlers search the `inst.phases` array at runtime to find the matching phase/segmented config:

1. **InstanceResolveInputs** (line ~720): `inst.phases.iter().find(|p| p.kernel == *kernel)` then conditionally checks `.segmented` to decide whether to build an outer eq table.

2. **InstanceSegmentedReduce** (line ~803): Two cascading `.find()` calls — first finding the phase by kernel, then extracting its `.segmented` config.

The compiler already knows which phase is active when it emits these ops.

## Fix
Add the needed config directly to the op variants:

```rust
Op::InstanceResolveInputs {
    batch, instance, kernel, force_refresh,
    segmented: Option<SegmentedConfig>,  // NEW: None if not segmented
}

Op::InstanceSegmentedReduce {
    batch, instance, kernel, round_within_phase,
    // segmented config already known — pass index or inline
    inner_num_vars: usize,
    outer_num_vars: usize,
    inner_only: Vec<bool>,
}
```

Or alternatively, split `InstanceResolveInputs` into two ops:
- `Op::InstanceResolveInputs { ... }` — just does input resolution
- `Op::BuildOuterEq { batch, instance, segmented: SegmentedConfig }` — explicit eq table construction

## Compiler Changes
The emit function already has the phase reference at hand — just embed the config.

## Benefits
- No phase structure inspection at runtime
- Each op is self-contained — doesn't need to search the batch definition

## Test
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
```

## Risk: Low
Pure data movement from runtime lookup to compile-time embedding.

## Dependencies: None
