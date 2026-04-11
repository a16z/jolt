# Task 07: PrefixSuffix Lifecycle as Explicit Ops

## Status: TODO

## Anti-Pattern
The runtime manages `PrefixSuffixState` as an opaque stateful object across rounds:
- **Created** at phase start when runtime detects `Iteration::PrefixSuffix` (line 387-397)
- **Stepped** each round via `ingest_challenge()` + `compute_address_round()` (lines 422-457)
- **Materialized** at phase boundary via `materialize_outputs()` (line 344-357)
- **Destroyed** after materialization (line 357)

The runtime must know the PrefixSuffix lifecycle: when to create, when it's a PS round vs normal, when to transition. This is all deterministic from the compiler's perspective.

## Current Runtime Branching
```rust
// Instance activation:
if is_prefix_suffix {
    let ps = PrefixSuffixState::new(...);
    state.prefix_suffix_states.insert((batch, inst), ps);
} else { ... }

// Each round:
if is_prefix_suffix {
    ps.ingest_challenge(scalar, round);
} else {
    bind_kernel_inputs(...);
}

// Reduce:
let evals = if is_prefix_suffix {
    let [eval_0, eval_2] = ps.compute_address_round();
    let eval_1 = previous_claim - eval_0;
    vec![eval_0, eval_1, eval_2]
} else if segmented { ... } else { ... }

// Phase transition:
if prev_is_ps {
    ps.ingest_challenge(last_challenge);
    let outputs = ps.materialize_outputs();
    // upload to device
    state.prefix_suffix_states.remove(&key);
}
```

## New Ops

```rust
/// Initialize PrefixSuffix state for an instance.
Op::PrefixSuffixInit {
    batch: usize,
    instance: usize,
}

/// PrefixSuffix bind: ingest a challenge into the PS state machine.
/// Replaces the `if is_prefix_suffix { ps.ingest_challenge(...) }` branch.
Op::PrefixSuffixBind {
    batch: usize,
    instance: usize,
    challenge: usize,
}

/// PrefixSuffix reduce: compute address round evals.
/// Replaces the `ps.compute_address_round()` + `eval_1 = claim - eval_0` logic.
Op::PrefixSuffixReduce {
    batch: usize,
    instance: usize,
}

/// PrefixSuffix materialize: generate output polynomials and upload to device.
/// Replaces the phase-transition materialization block.
Op::PrefixSuffixMaterialize {
    batch: usize,
    instance: usize,
}
```

## How Compiler Emits Them

For a PrefixSuffix phase (e.g., InstructionReadRaf address phase):

```
// Phase start:
PrefixSuffixInit { batch: B, instance: I }

// Round 0 (no bind):
PrefixSuffixReduce { batch: B, instance: I }
BatchAccumulateInstance { batch: B, instance: I, max_evals: 3 }

// Round 1:
PrefixSuffixBind { batch: B, instance: I, challenge: ch }
PrefixSuffixReduce { batch: B, instance: I }
BatchAccumulateInstance { batch: B, instance: I, max_evals: 3 }

// ... more rounds ...

// Phase transition (last round of PS phase):
PrefixSuffixBind { batch: B, instance: I, challenge: last_ch }
PrefixSuffixMaterialize { batch: B, instance: I }
// Now transition to cycle phase:
InstanceResolveInputs { batch: B, instance: I, kernel: cycle_K, ... }
```

## Runtime Handlers

Each handler is ~10 lines:

```rust
Op::PrefixSuffixInit { batch, instance } => {
    let trace = lookup_trace.as_ref().unwrap();
    let ps = PrefixSuffixState::new(&iteration_config, &state.challenges, trace);
    state.prefix_suffix_states.insert((*batch, *instance), ps);
}

Op::PrefixSuffixBind { batch, instance, challenge } => {
    let scalar = state.challenges[*challenge];
    let ps = state.prefix_suffix_states.get_mut(&(*batch, *instance)).unwrap();
    let round = ps.total_round();
    ps.ingest_challenge(scalar, round);
}

Op::PrefixSuffixReduce { batch, instance } => {
    let ps = state.prefix_suffix_states.get_mut(&(*batch, *instance)).unwrap();
    let [eval_0, eval_2] = ps.compute_address_round();
    let previous_claim = state.batch_instance_claims[*batch][*instance];
    let eval_1 = previous_claim - eval_0;
    state.last_round_instance_evals[*instance] = vec![eval_0, eval_1, eval_2];
}

Op::PrefixSuffixMaterialize { batch, instance } => {
    let ps = state.prefix_suffix_states.remove(&(*batch, *instance)).unwrap();
    let outputs = ps.materialize_outputs();
    for (poly_id, data) in outputs {
        device_buffers.insert(poly_id, DeviceBuffer::Field(backend.upload(&data)));
    }
}
```

## `eval_1 = previous_claim - eval_0` Note
This is still protocol logic in the runtime (the sumcheck relation). However, it's now contained in a single, well-named op handler rather than embedded in a 300-line conditional. We could push this further by having PrefixSuffix return all 3 evals, but that requires changing the PrefixSuffixState API. Acceptable as-is.

## Anticipating Stage 6
Stage 6 has `InstructionRaVirtualization` and `RamRaVirtualization` which both use two-phase structures. InstructionRa likely uses PrefixSuffix for the address phase. The explicit lifecycle ops make it trivial to compose with phase transitions.

## Test
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
```

## Risk: Medium
PrefixSuffix is the most stateful component. The ops are straightforward wrappers, but the lifecycle ordering must be exactly right. The compiler must emit them in the correct sequence.

## Dependencies: Task 04, Task 05
