# Task 14: Embed Instance Degree in BatchAccumulateInstance

## Status: TODO

## Anti-Pattern
`BatchAccumulateInstance` (line ~851) conditionally interpolates evaluations when the instance has fewer evals than `max_evals`. The runtime computes:

```rust
if full_evals.len() < max_evals {
    let poly = UnivariatePoly::interpolate(&points);
    for s in full_evals.len()..max_evals {
        full_evals.push(poly.evaluate(F::from_u64(s as u64)));
    }
}
```

This is a runtime conditional based on kernel degree. The compiler knows the degree at emit time.

## Fix
Add `num_evals: usize` to the op:

```rust
Op::BatchAccumulateInstance {
    batch: usize,
    instance: usize,
    max_evals: usize,
    num_evals: usize,  // NEW: this instance's native eval count
}
```

The handler unconditionally extrapolates from `num_evals` to `max_evals` (no-op when equal).

## Compiler Changes
Emit `num_evals` from `kernels[phase.kernel].spec.num_evals`.

## Test
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
```

## Risk: None
Pure data embedding.

## Dependencies: None
