# Task 04: Design Granular Op Vocabulary for Batched Sumcheck

## Status: TODO

## Anti-Pattern
`Op::BatchedSumcheckRound` is a ~300-line sub-interpreter. A single op triggers: per-instance iteration, inactive contribution, phase boundary detection, scalar captures, PrefixSuffix lifecycle, segmented reduce dispatch, degree extrapolation, and batch combination.

The compiler knows the exact per-instance schedule at compile time. We should unroll this into per-instance, per-round ops that the runtime can execute as a flat sequence.

## Design

### New Op Variants

```rust
/// Begin a new batched sumcheck round. Initializes the combined accumulator.
Op::BatchRoundBegin {
    batch: usize,
    round: usize,
    max_evals: usize,
    bind_challenge: Option<usize>,
}

/// Inactive instance: add coeff * (claim/2) to all eval slots, halve stored claim.
Op::BatchInactiveContribution {
    batch: usize,
    instance: usize,
}

/// Resolve kernel inputs for a phase start (instance activation or phase transition).
Op::InstanceResolveInputs {
    batch: usize,
    instance: usize,
    kernel: usize,
    force_refresh: bool,
}

/// Bind previous phase's kernel inputs at the given challenge.
Op::InstanceBindPreviousPhase {
    batch: usize,
    instance: usize,
    kernel: usize,       // previous phase's kernel
    challenge: usize,
}

/// Capture a scalar from a fully-bound buffer into a challenge slot.
Op::ScalarCapture {
    poly: PolynomialId,
    challenge: usize,
}

/// Standard dense reduce for one instance within a batched round.
Op::InstanceReduce {
    batch: usize,
    instance: usize,
    kernel: usize,
}

/// Segmented reduce for one instance (mixed-dimensional inputs).
Op::InstanceSegmentedReduce {
    batch: usize,
    instance: usize,
    kernel: usize,
    config: SegmentedConfig,
}

/// Extrapolate lower-degree instance evals to max_evals via interpolation,
/// then accumulate into the batch combined polynomial with batch coefficient.
Op::BatchAccumulateInstance {
    batch: usize,
    instance: usize,
    max_evals: usize,
}

/// Finalize batched round: interpolate combined evals to coefficients.
Op::BatchRoundFinalize {
    batch: usize,
}
```

### What Gets Removed
`Op::BatchedSumcheckRound` is deleted entirely. The compiler unrolls each round of each batch into a sequence of the above ops.

### Compiler Emit Pattern
For batch B with instances [I0, I1, I2] where I0 is inactive for 3 rounds:

```
// Round 0:
BatchRoundBegin { batch: B, round: 0, max_evals: 4, bind_challenge: None }
BatchInactiveContribution { batch: B, instance: 0 }         // I0 inactive
InstanceResolveInputs { batch: B, instance: 1, kernel: K1, force_refresh: true }
InstanceReduce { batch: B, instance: 1, kernel: K1 }
BatchAccumulateInstance { batch: B, instance: 1, max_evals: 4 }
InstanceResolveInputs { batch: B, instance: 2, kernel: K2, force_refresh: true }
InstanceReduce { batch: B, instance: 2, kernel: K2 }
BatchAccumulateInstance { batch: B, instance: 2, max_evals: 4 }
BatchRoundFinalize { batch: B }
AbsorbRoundPoly { ... }
Squeeze { ... }

// Round 1:
BatchRoundBegin { batch: B, round: 1, max_evals: 4, bind_challenge: Some(ch) }
BatchInactiveContribution { batch: B, instance: 0 }
InstanceReduce { batch: B, instance: 1, kernel: K1 }        // already resolved
BatchAccumulateInstance { batch: B, instance: 1, max_evals: 4 }
InstanceReduce { batch: B, instance: 2, kernel: K2 }
BatchAccumulateInstance { batch: B, instance: 2, max_evals: 4 }
BatchRoundFinalize { batch: B }
...

// Round 3 (I0 activates):
BatchRoundBegin { batch: B, round: 3, max_evals: 4, bind_challenge: Some(ch) }
InstanceResolveInputs { batch: B, instance: 0, kernel: K0, force_refresh: true }
InstanceReduce { batch: B, instance: 0, kernel: K0 }
BatchAccumulateInstance { batch: B, instance: 0, max_evals: 4 }
...
```

### Phase Transitions
When instance I hits a phase boundary, the compiler emits:
```
InstanceBindPreviousPhase { batch: B, instance: I, kernel: prev_K, challenge: ch }
ScalarCapture { poly: P, challenge: C }  // for each scalar_capture in phase
InstanceResolveInputs { batch: B, instance: I, kernel: new_K, force_refresh: false }
```

## What This Task Does NOT Do
- Does NOT implement the ops in the runtime (that's Task 05)
- Does NOT handle PrefixSuffix (that's Task 07)
- Does NOT handle SegmentedReduce internals (that's Task 06)
- Just defines the new Op variants in `module.rs` and the compiler emit logic

## Test
```bash
cargo clippy -p jolt-compiler --message-format=short -q --all-targets -- -D warnings
```
Must compile. No runtime changes yet, so no equivalence test needed.

## Risk: Medium
Designing the right vocabulary is the hardest part. Must anticipate stage 6 patterns:
- Multi-phase instances (RamRaVirtualization: cycle + address phases)
- Sparse vs dense iteration
- Cross-stage advice reduction
- Booleanity (Gruen decomposition)

## Dependencies: Tasks 01-03 (they simplify the runtime before the big refactor)
