# Tasks

## Philosophy

The compiler lowers ALL protocol knowledge into data carried by Ops.
The runtime is a flat dispatcher — every handler is mechanical, ≤ 30 LOC, zero interpretation of protocol-specific data structures.
If the runtime needs to "interpret" rules, the compiler hasn't lowered far enough.

## Test Gate

```bash
# Correctness: byte-identical Fiat-Shamir transcripts + proof acceptance
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
cargo nextest run -p jolt-equivalence zkvm_proof_accepted --cargo-quiet
# All equivalence tests (includes any dual-path validation tests you add)
cargo nextest run -p jolt-equivalence --cargo-quiet
# Lint
cargo clippy -p jolt-compiler -p jolt-compute -p jolt-cpu -p jolt-zkvm -p jolt-dory -p jolt-openings -p jolt-verifier --message-format=short -q --all-targets -- -D warnings
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
- [ ] V9: `checkpoint_eval.rs` remaining interpreters: `eval_prefix_mle` (597 LOC, 21 PrefixMleFormula variants), `compute_read_checking_from_rules` (85 LOC). Plan: lower to generic `Op::MaterializeFromExprs` + `Op::EvaluateScalarExpr` with extended ValueSource (`IndexedPoly`, `SelectByIndex`) + compile-time mask polynomials. **Files**: `checkpoint_eval.rs`, `handlers.rs:ReadCheckingReduce`.
    - [x] Phase-0: `ValueSource` extended with `IndexedPoly(PolynomialId)` + `SelectByIndex { index_poly, values }`; `eval_scalar_expr` takes `(index, buffers)`. CheckpointEvalBatch passes empty buffers.
    - [x] Phase-C: `compute_combined_val_from_rules` (47 LOC) deleted; inlined into `MaterializeCombinedVal` handler (<30 LOC). Still data-driven over `CombineEntry`, no new ops.
- [x] V7: `InputBinding::BytecodeVal` → decomposed into `BytecodeField(idx)` preprocessed polys + `Op::WeightedSum` + `Op::EqGather` on `BytecodeRegEq(idx)`. `bytecode_data` field removed from RuntimeState.
- [x] V8: `LookupTraceData` moved from `RuntimeState` into `BufferProvider` trait (`jolt-compute`). `ProverData` exposes it via `with_lookup_trace()` builder. Runtime handlers read via `provider.lookup_trace()`.

## Notes

Space for recording design decisions, dead ends, and insights during execution.

## Done

(completed items move here with one-line summary of what changed)
