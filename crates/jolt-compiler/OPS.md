# OPS.md — `Op` variant inventory and target primitive form

**Status:** living target document. The compiler's current `Op` enum is the
union of (a) primitives we're keeping, (b) legacy forms we're lowering,
and (c) protocol-named variants pending rename. This doc pins the target
state each variant should reach by the end of phase O5 of the streamlining
plan (`crates/jolt-bench/opt/05-streamlining.md`).

**Enforcement**: [`Op::is_primitive`](src/module.rs) returns `true` iff
a variant is in its canonical primitive form. Initially — today — it
returns `true` for every variant, so the post-emit `debug_assert!` in
`compile()` is a no-op. As each sub-phase of O4 and O5 lands, it flips
the affected variants to `false` and removes them from emission in the
same commit. The assertion then catches any emit site that slipped back
to the old form.

## Taxonomy

| Category | Meaning | `is_primitive()` target | Resolved by |
|---|---|---|---|
| **Primitive** | One logical step. No runtime conditional. Backend-agnostic. Generic name that doesn't leak protocol concepts. | `true` (final) | — |
| **Batch-scaffold** | Compiler-emitted structural marker around a batch-round window. Primitive in behavior; named after its structural role, not a protocol. | `true` (final) | — |
| **Redundant** | Primitive behavior but duplicates another variant's action under a different name. The redundancy is a consequence of runtime-side dedup that compile-time tracking can eliminate. | `false` after phase | O4.a / O4.b |
| **Conditional** | Runtime branch ("materialize only if …") that compile-time producer-analysis can replace with either a primitive op or nothing. | `false` after phase | O4.b |
| **Protocol-specific** | Variant name or behavior leaks a specific proof-protocol concept (instruction lookups, RAM, address decomposition). Lowers to a sequence of primitives, or renames in place if only the name leaks. | `false` after phase | O5 |

## Variant catalog

### Primitive — compute (14)

These ops describe backend-dispatchable compute. They cover the full
sumcheck inner loop and post-sumcheck evaluation. `is_primitive()` is
`true` and stays `true`.

| Variant | Description |
|---|---|
| `Reduce { specs: Vec<ReduceSpec> }` | Fused reduce over a set of specs. Covers flat / segmented / Gruen / domain / sparse axes. Unified reduce surface established in T2. |
| `Bind { polys, challenge, order }` | Bind polynomial buffers at a challenge with a given binding order. The canonical bind primitive. |
| `Materialize { binding }` | Build a device buffer from an `InputBinding`. |
| `WeightedSum { output, terms, identity_term, overall_scale }` | Host-side linear combination: `out[i] = Σ c_j · p_j[i] + id(i)`, optionally scaled. |
| `LagrangeProject { polys, challenge, domain_size, domain_start, stride, group_offsets, kernel_tau }` | Lagrange basis projection with optional kernel-tau scaling. Post-uniskip projection. |
| `DuplicateInterleave { polys }` | `buf'[2i] = buf'[2i+1] = buf[i]`. |
| `RegroupConstraints { polys, group_indices, old_stride, new_stride, num_cycles }` | Constraint-dim regroup for group-split uniskip. |
| `ExpandingTableUpdate { table, challenge, current_len }` | Double an eq-table-like buffer by consuming one challenge variable. |
| `InitExpandingTable { table, size }` | Zero-initialize an expanding-table slot. |
| `BuildSegmentedEq { batch, instance, outer_challenges, outer_num_vars }` | Build an eq-table from a challenge list (or all-ones vector when empty) and stash under `(batch, instance)` in the runtime's per-instance segmented state. Consumed by segmented reduces. |
| `ScaleEval { poly, factor_challenges }` | Multiply an evaluation by `∏(1 − ch[i])`. |
| `Evaluate { poly, mode }` | Extract an evaluation from a polynomial. |
| `EvaluatePreprocessed { source, at_challenges, store_as }` | Evaluate a preprocessed polynomial's MLE at a challenge-derived point. |
| `CaptureScalar { poly, challenge }` | Capture the single scalar from a fully-bound 1-element buffer into a challenge slot. |

### Primitive — PCS (7)

Opening proof orchestration. Generic in the commitment scheme.

| Variant | Description |
|---|---|
| `Commit { polys, tag, num_vars }` | Commit polynomials + absorb commitments. |
| `CommitStreaming { polys, tag, chunk_size, num_vars }` | Streaming commit (chunked). |
| `ReduceOpenings` | RLC-reduce accumulated opening claims. |
| `Open` | Generate PCS opening proofs. |
| `BindOpeningInputs { point_challenges }` | Post-proof transcript bind: `PCS::bind_opening_inputs(transcript, point, eval)`. |
| `CollectOpeningClaim { poly, at_stage }` | Accumulate an opening claim keyed by stage. |
| `CollectOpeningClaimAt { poly, point_challenges, committed_num_vars }` | Accumulate an opening claim with an explicit multi-stage point. |

### Primitive — orchestration (9)

Host-only bookkeeping: transcript, challenge derivation, stage lifecycle.

| Variant | Description |
|---|---|
| `Preamble` | Absorb public preamble into the transcript. |
| `BeginStage { index }` | Start a new verifier stage (flushes prior stage proof). |
| `AbsorbRoundPoly { num_coeffs, tag, encoding }` | Interpolate evals → monomial coeffs, absorb into transcript. |
| `RecordEvals { polys }` | Push polynomial evals into the current stage proof. |
| `AbsorbEvals { polys, tag }` | Absorb polynomial evals into the Fiat-Shamir transcript. |
| `AbsorbInputClaim { formula, tag, batch, instance, inactive_scale_bits }` | Evaluate a `ClaimFormula`, absorb its scalar into the transcript, and initialize the runtime's per-instance claim. |
| `AppendDomainSeparator { tag }` | Append a zero-payload transcript label. |
| `Squeeze { challenge }` | Squeeze a Fiat-Shamir challenge. |
| `ComputePower { target, base, exponent }` | Derive a challenge as `challenges[base]^exponent`. |

### Primitive — resource (3)

Buffer lifecycle.

| Variant | Description |
|---|---|
| `ReleaseDevice { poly }` | Release a device buffer. |
| `ReleaseHost { polys }` | Release host polynomial data. |
| `AliasEval { from, to }` | Alias one evaluation under another polynomial ID. |

### Batch-scaffold (4)

Structural markers around a batch-round window. The compiler emits these
to frame a batch-round's per-instance dispatch. `is_primitive()` is
`true` — the names describe their structural role, not a protocol.

| Variant | Description |
|---|---|
| `BatchRoundBegin { batch, round, max_evals, bind_challenge }` | Zero the combined accumulator; update per-instance claims from the previous round. |
| `BatchInactiveContribution { batch, instance }` | Add `coeff × claim/2` contribution for an inactive instance. |
| `BatchAccumulateInstance { batch, instance, max_evals, num_evals }` | Extrapolate per-instance evals + accumulate into combined. |
| `BatchRoundFinalize { batch }` | Store combined evals as `last_round_coeffs`. |

---

### Redundant — bind family (3)

Same primitive behavior as `Op::Bind`, with a runtime-side dedup set that
the compiler can eliminate. Target: collapse into `Op::Bind` with
pre-deduped `polys` at emission time. **Resolved in O4.a.**

| Variant | Target lowering |
|---|---|
| `InstanceBind { batch, instance, kernel, challenge }` | `Op::Bind { polys: dedup(kernel.inputs), challenge, order: kernel.binding_order }` — compiler tracks per-instance bound-poly sets. |
| `InstanceBindPreviousPhase { batch, instance, kernel, challenge }` | Same as `InstanceBind` with polys from the previous phase's kernel. |
| `BindCarryBuffers { polys, challenge, order }` | Direct `Op::Bind` — already primitive-shaped; distinction is compiler-side only (emitted when phase has carry bindings). Just rename the emission site. |

### Conditional — materialize family (2)

Runtime branch ("materialize only if buffer doesn't exist / has wrong size")
that compile-time producer-analysis can replace. **Resolved in O4.b.**

| Variant | Target lowering |
|---|---|
| `MaterializeUnlessFresh { binding, expected_size }` | Either `Op::Materialize { binding }` or nothing — compiler tracks which polys earlier ops have produced, emits straight `Op::Materialize` only where needed. |
| `MaterializeIfAbsent { binding }` | Same pattern: either `Op::Materialize` or nothing. |

### Protocol-specific — rename only (landed)

Primitive behavior, name-only protocol leak. **Landed via rename** — see
below.

| Former variant | Renamed to | Commit |
|---|---|---|
| `CheckpointEvalBatch { updates: Vec<(usize, CheckpointEvalAction)> }` | `InstanceScalarUpdate { updates: Vec<(usize, ScalarUpdateAction)> }` | S5.rename |
| `MaterializeSegmentedOuterEq { batch, instance, segmented: SegmentedConfig }` | `BuildSegmentedEq { batch, instance, outer_challenges: Vec<ChallengeIdx>, outer_num_vars: usize }` | S5.build_segmented_eq |

Also renamed: `CheckpointEvalAction` → `ScalarUpdateAction`; runtime
`state.instance_checkpoints` → `state.instance_scalars`;
`build_checkpoint_batch` helper → `build_scalar_update_batch`. Behavior
unchanged: evaluate compiled `ScalarExpr`s against a pre-batch snapshot
of per-instance scalar state, write back atomically. The `usize` index
into per-instance scalar slots was kept as-is (it's a checkpoint-slot
index, not an `InstanceIdx` — the original OPS.md proposal to retype it
was inaccurate; the `usize` is correct for the schema).

`BuildSegmentedEq` drops the unused `SegmentedConfig::inner_num_vars`
and `SegmentedConfig::inner_only` fields from the op — the build step
only reads `outer_eq_challenges` and `outer_num_vars`. Consumers of the
eq table (segmented reduces) still pull `SegmentedConfig` from
`PhaseDef::segmented` where the inner-side fields live. `build_outer_eq`
in `jolt-zkvm/src/runtime/helpers.rs` now takes those two values
directly instead of a `&SegmentedConfig`.

### Protocol-specific — lower to primitives (9)

Protocol-specific both in name and behavior. Each lowers to a sequence of
primitives. **Resolved in O5** (tentative targets below — refined when the
corresponding O5 sub-phase actually lands).

| Variant | Tentative target (pending O5.<n>) |
|---|---|
| `ReadCheckingReduce { kernel, round, r_x_challenge }` | `Op::Reduce { specs: [ReduceSpec { axes: Flat, kernel: composed_prefix_suffix, .. }] }`. The per-round prefix×suffix + combine-matrix evaluation becomes a single kernel formula; compiler lowers the combine matrix into `KernelSpec.formula` during emission. |
| `RafReduce { batch, instance, kernel }` | `Op::Reduce` with a product-of-sums formula kernel; raf/gamma weighting lowered into the composed formula. |
| `SuffixScatter { kernel, phase }` | `Op::WeightedSum` + `Op::Reduce` with sparse axes. If the pattern doesn't fit `WeightedSum`, introduce a generic `Op::Scatter { dst, source, index_poly }` primitive. |
| `QBufferScatter { kernel, phase }` | Same shape as `SuffixScatter` — either `Op::WeightedSum` + `Op::Reduce`, or the new generic `Op::Scatter`. |
| `MaterializeRA { kernel }` | `Op::WeightedSum` (RLC of per-chunk RA one-hot polys) + primitive `Op::Materialize` ops. Protocol-specific kernel-input construction (which chunks, which weights) moves into compiler lowering, not a runtime op. |
| `MaterializeCombinedVal { kernel }` | `Op::WeightedSum` (combined-val = Σ flag_table × table_val + raf × gamma) + `Op::Materialize`. |
| `MaterializePBuffers { kernel }` | `Op::InstanceScalarUpdate` (after O5 rename of `CheckpointEvalBatch`) + `Op::Materialize` for the P buffers. |
| `InitInstanceWeights { r_reduction, num_prefixes }` | Reroute to `Op::ExpandingTableUpdate` + `Op::InitExpandingTable`. The "instance-weights" naming is lookup-specific; the mechanism is already generic. |
| `UpdateInstanceWeights { expanding_table, chunk_bits, num_phases, phase }` | Reroute to `Op::ExpandingTableUpdate`. Same rationale as `InitInstanceWeights`. |

---

## Summary

| Category | Count | `is_primitive()` target |
|---|---|---|
| Primitive — compute | 14 | `true` |
| Primitive — PCS | 7 | `true` |
| Primitive — orchestration | 9 | `true` |
| Primitive — resource | 3 | `true` |
| Batch-scaffold | 4 | `true` |
| Redundant (landed O4.a) | 0 (was 3) | — |
| Conditional (deferred to O6/O7) | 2 | `true` (ratchet unchanged until pass ships) |
| Protocol-specific: rename (landed S5.rename, S5.build_segmented_eq) | 0 (was 2) | — |
| Protocol-specific: lower (→ O5) | 9 | `true` (ratchet unchanged until lowered) |
| **Current total** | **48** | |

Post-O5 target: 36 primitive + batch-scaffold variants (plus any new generic
primitives introduced during O5 lowering — e.g., possibly `Op::Scatter`;
`Op::BuildSegmentedEq` and `Op::InstanceScalarUpdate` already landed via
rename). Estimated final variant count: ~38.

## Invariant wiring

`impl Op { pub fn is_primitive(&self) -> bool }` — exhaustive match over
every variant. Today returns `true` everywhere (no-op). Flipped to `false`
per-variant in the O4/O5 sub-phase that removes the corresponding emission
site; each flip lands in the same commit that updates the compiler to stop
emitting the variant.

Post-emit assertion fires in:
- `crates/jolt-compiler/src/compiler/mod.rs::compile()` — the real compiler path.
- `crates/jolt-compiler/examples/jolt_core_module.rs::build_module()` — the
  hand-written reference module.

Both sites run:

```rust
debug_assert!(
    module.prover.ops.iter().all(Op::is_primitive),
    "compiler emitted non-primitive op: {:?}",
    module.prover.ops.iter().find(|op| !op.is_primitive()),
);
```

Release builds skip the assertion entirely.

## Cross-reference

- **`crates/jolt-bench/opt/05-streamlining.md`** — the overhaul plan this
  document supports. Phases O4 and O5 are the ones that flip variants
  in `is_primitive()`.
- **`crates/jolt-bench/opt/02-unified-reduce.md`** — Ticket 2's
  unified-reduce work (`Op::Reduce` + `ReduceSpec`) that already landed
  and set the primitive shape for the reduce family.
- **`crates/jolt-zkvm/src/runtime/handlers.rs`** — the other side of
  every op: handler arms that O4/O5 will collapse (4 bind handlers → 1,
  3 materialize handlers → 1, ~10 protocol-specific handlers → 0).
