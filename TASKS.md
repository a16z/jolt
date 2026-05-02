# Tasks

## Philosophy

The compiler lowers ALL protocol knowledge into data carried by Ops.
The runtime is a flat dispatcher — every handler is mechanical, ≤ 30 LOC, zero interpretation of protocol-specific data structures.
If the runtime needs to "interpret" rules, the compiler hasn't lowered far enough.

## Status

The legacy `jolt-compiler` / `jolt-zkvm` / `jolt-verifier` stack and the old
`jolt-compute` / `jolt-cpu` / `jolt-metal` backend crates have been retired
from the active workspace. `jolt-host` is also retired in favor of the upstream
`jolt-trace` name. The V1–V9 list below is historical reference for the old
compiler path; new equivalence work should target the Bolt pipeline in
`crates/bolt`, `crates/jolt-kernels`, `crates/jolt-witness`,
`crates/jolt-trace`, and the modular primitive crates.

## Test Gate

```bash
# Bolt equivalence and generated-artifact checks
cargo check -p jolt-equivalence --tests --quiet
cargo nextest run -p jolt-equivalence --cargo-quiet
cargo nextest run -p bolt --cargo-quiet
cargo nextest run -p jolt-kernels --cargo-quiet
```

## Approach

Use dual-path validation: implement the new path alongside the old, assert
equality, then delete the old. Add jolt-equivalence tests for non-trivial
assertions. See CLAUDE.md "Task Loop Protocol" for the full workflow.

## Violations

Each violation is a place where the runtime interprets protocol-specific data instead of mechanically dispatching compiler-generated ops.

- [x] V1: `InstanceScatter` (220 LOC) → decomposed into 6 sub-ops: InitInstanceWeights (4), UpdateInstanceWeights (7), SuffixScatter (22), QBufferScatter (25), MaterializePBuffers (19), InitExpandingTable (4). All ≤ 30 LOC.
- [x] V2: `InstanceDecompReduce` (114 LOC) → split into ReadCheckingReduce (12 LOC) + RafReduce (28 LOC). Both ≤ 30 LOC.
- [x] V3: `checkpoint_eval.rs` checkpoint interpreter (CheckpointRule branch) eliminated — `eval_checkpoint_rule`/`apply_action` no longer in runtime path (kept test-only as parity reference). Prefix-MLE side (`eval_prefix_mle`, `compute_read_checking_from_rules`, `compute_combined_val_from_rules`) tracked separately as V9.
- [x] V4: `InstanceBindBuffers` → replaced with `Op::Bind` using `InstanceConfig::bindable_polys()`. Op variant and handler deleted.
- [x] V5: `InstanceMaterialize` (73 LOC) → split into MaterializeRA (20 LOC) + MaterializeCombinedVal (14 LOC). Both ≤ 30 LOC.
- [x] V6: `UpdateInstanceCheckpoints` deleted. Replaced by `Op::CheckpointEvalBatch` with compile-time-lowered `ScalarExpr` (monomial sum over `ValueSource::{Pow2, Challenge, OneMinusChallenge, Checkpoint}`). Handler is 15 LOC, atomic pre-batch snapshot semantics. Parity test: `checkpoint_eval_parity.rs` covers all 16 `CheckpointAction` variants.
- [ ] V9: Finish `ReadCheckingReduce` lowering to `Op::Reduce`. The original `checkpoint_eval.rs` interpreter has been dismantled: `eval_prefix_mle` is gone (replaced by the compile-time `prefix_mle_lowering.rs` + the data-driven `compute_read_checking_from_lowered` in `runtime/prefix_suffix.rs`), `compute_combined_val_from_rules` was inlined into `MaterializeCombinedVal`, and `CheckpointRule`-branch interpretation was removed. **Remaining**: the `ReadCheckingReduce` handler still routes through `compute_read_checking_from_lowered` (131 LOC, data-driven but protocol-shaped). Lowering to generic `Op::Reduce` needs `KernelSpec.formula` extended for combine-matrix + prefix×suffix shape (OPS.md Group B blocker). **Files**: `crates/jolt-zkvm/src/runtime/prefix_suffix.rs`, `crates/jolt-zkvm/src/runtime/handlers.rs:ReadCheckingReduce`.
    - [x] Phase-0: `ValueSource` extended with `IndexedPoly(PolynomialId)` + `SelectByIndex { index_poly, values }`; `eval_scalar_expr` takes `(index, buffers)`. `InstanceScalarUpdate` (formerly `CheckpointEvalBatch`) passes empty buffers.
    - [x] Phase-C: `compute_combined_val_from_rules` (47 LOC) deleted; inlined into `MaterializeCombinedVal` handler (<30 LOC). Still data-driven over `CombineEntry`, no new ops.
- [x] V7: `InputBinding::BytecodeVal` → decomposed into `BytecodeField(idx)` preprocessed polys + `Op::WeightedSum` + `Op::EqGather` on `BytecodeRegEq(idx)`. `bytecode_data` field removed from RuntimeState.
- [x] V8: `LookupTraceData` moved from `RuntimeState` into `BufferProvider` trait (`jolt-compute`). `ProverData` exposes it via `with_lookup_trace()` builder. Runtime handlers read via `provider.lookup_trace()`.

## Notes

Space for recording design decisions, dead ends, and insights during execution.

- Bolt stage implementation now follows `crates/bolt/JOLT_PROTOCOL_IMPLEMENTATION.md`:
  protocol MLIR first, then role/compute/CPU lowering, generated
  `jolt-prover`/`jolt-verifier` artifacts, arithmetic, real-data equivalence,
  tamper checks, and sub-20% stage perf before a stage is marked complete.

## Done

(completed items move here with one-line summary of what changed)
