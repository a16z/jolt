# Task 12: Compile resolve_inputs Into Explicit Materialization Ops

## Status: DEFERRED

## Original Anti-Pattern
`resolve_inputs()` was a ~230-line hidden compiler inside the runtime, matching on 10+ `InputBinding` variants to decide how to construct each kernel input buffer.

## Current State (Post Phase 5-6)
After tasks 13-21, `resolve_inputs()` is now a clean protocol-unaware dispatcher:
- Each of the 10 `InputBinding` arms is a 3-5 line backend call (e.g., `backend.eq_table()`, `backend.eq_project()`, `backend.scale_from_host()`)
- BytecodeVal delegates to `BytecodeData::materialize_val()` — no protocol logic in the handler
- The function is generic over `B: ComputeBackend` and takes `&impl BufferProvider<F>`
- No protocol-specific types, no iteration-type inspection, no field arithmetic

## Remaining Value
Converting to per-binding-type `Op::Materialize*` variants would:
- Make the schedule fully explicit (each materialization visible as an op)
- Eliminate the `InputBinding` enum dispatch at runtime
- Remove the `force_refresh` heuristic (compiler tracks buffer staleness)

This is a "nice to have" that further flattens the schedule but does not remove protocol awareness (there is none left to remove).

## Decision
Deferred — not blocking any design goal. The current resolve_inputs is protocol-unaware infrastructure. Revisit if the schedule needs to be fully introspectable (e.g., for GPU command buffer recording).
