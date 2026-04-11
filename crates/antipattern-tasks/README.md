# Anti-Pattern Tasks

Incremental refactoring of the jolt-zkvm runtime and compiler to eliminate protocol logic from the runtime, establishing clean patterns before stage 6.

## Principle
The compiler expresses the protocol. The runtime is a dumb executor: a flat `match op { ... }` loop with no conditionals, no iteration type inspection, no lifecycle management. Each round of each instance is an explicit op in the schedule.

## Testing Strategy
- **Baseline**: jolt-equivalence tests (especially `transcript_divergence`) verify byte-identical Fiat-Shamir transcripts between jolt-core (reference) and jolt-zkvm (new)
- **Incremental**: Each task maintains parity. If parity breaks, debug using jolt-equivalence's diagnostic tools (checkpoint comparison, per-op hash dumps)
- **Sandbox**: jolt-equivalence is a debugging sandbox — add targeted tests freely to fuzz specific polynomial constructions or claim formulas against the jolt-core reference

## Task Ordering (Strategic)

### Phase 1: Low-Risk Independent Fixes
These can be done in any order, each independently testable:

| Task | Anti-Pattern | Risk |
|------|-------------|------|
| [00-baseline](00-baseline.md) | Capture regression baseline | None |
| [01-absorb-round-poly-encoding](01-absorb-round-poly-encoding.md) | Runtime branches on iteration type for transcript encoding | Low |
| [02-evaluate-op-split](02-evaluate-op-split.md) | Runtime branches on buffer size in Evaluate | Low |
| [03-input-claim-prescaling](03-input-claim-prescaling.md) | Runtime computes 2^offset for inactive scaling | Low |
| [11-split-jolt-core-module](11-split-jolt-core-module.md) | Monolithic module builder | None |

### Phase 2: The Big Unroll
Must be done in order (04 → 05 → 08):

| Task | What | Risk |
|------|------|------|
| [04-unroll-vocabulary](04-unroll-vocabulary.md) | Design new granular Op variants | Medium |
| [05-unroll-runtime](05-unroll-runtime.md) | Implement runtime handlers + delete old BatchedSumcheckRound | Medium-High |
| [08-builder-unroll-emit](08-builder-unroll-emit.md) | Builder emits unrolled ops | Medium |

### Phase 3: Specialized Iteration Patterns
After the unroll, these clean up the remaining runtime intelligence:

| Task | What | Risk |
|------|------|------|
| [06-segmented-reduce-op](06-segmented-reduce-op.md) | Segmented reduce as explicit op | Low-Medium |
| [07-prefix-suffix-lifecycle](07-prefix-suffix-lifecycle.md) | PrefixSuffix lifecycle as explicit ops | Medium |
| [09-resolve-inputs-explicit](09-resolve-inputs-explicit.md) | Buffer resolution as explicit ops | Low |

### Phase 4: Cleanup
| Task | What | Risk |
|------|------|------|
| [10-cleanup-debug](10-cleanup-debug.md) | Remove domain-specific debug instrumentation | None |

## Phase 1-4 Complete
Tasks 00-11 are done (see `done/`). The runtime is now a flat op interpreter with granular per-instance, per-round ops. The `BatchedSumcheckRound` sub-interpreter is deleted.

## Phase 5-6 Complete

Tasks 13-21 are done (see `done/`). Task 12 (resolve_inputs to ops) is **deferred** — `resolve_inputs()` is now a clean protocol-unaware dispatcher after Phase 5-6 work eliminated all protocol logic from the runtime. Revisit if the schedule needs to be fully introspectable (e.g., for GPU command buffer recording).

**What was achieved:**
- All computation dispatches through `ComputeBackend` trait methods (eq tables, projections, materialization, PrefixSuffix state machine, scalar ops, polynomial arithmetic, segmented reduce, Lagrange projection)
- Protocol-specific data providers (`bytecode_raf.rs`, `derived.rs`, `preprocessed.rs`, `provider.rs`) moved from jolt-zkvm to jolt-witness
- jolt-zkvm is now protocol-unaware: `runtime.rs` is a generic flat `match op { ... }` executor, `prove.rs` orchestrates PCS generics
- Metal backend stubs delegate to CPU fallback via `jolt-cpu` for all backend methods
- BytecodeVal protocol logic encapsulated in `BytecodeData::materialize_val()` (jolt-witness)
- PrefixSuffix routed through opaque `ComputeBackend::PrefixSuffixState<F>` associated type
