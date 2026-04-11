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

## Completion
When all tasks are done, move their files to `done/`. The runtime should be a flat op interpreter with zero protocol knowledge, and we'll be well-positioned for stage 6.

## Stage 6 Readiness
Stage 6 contains: BytecodeReadRaf, Booleanity (Gruen), RamHammingBooleanity, RamRaVirtualization, InstructionRaVirtualization, IncClaimReduction, and optional AdviceClaimReduction. Key patterns:
- Multi-phase instances (cycle + address)
- Sparse vs dense iteration
- Cross-stage sumcheck (advice spans stages 6-7)
- Claim consolidation from multiple prior stages

The unrolled op vocabulary from Tasks 04-08 handles all of these naturally: each phase of each instance is explicit in the op stream.
