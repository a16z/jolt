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

### Primitive — compute (16)

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
| `TraceGatherMultiply { dst, source_table, shift, mask }` | `dst[j] *= source_table[(lookup_keys[j] >> shift) & mask]` for every cycle. Generic trace-driven gather-multiply; consumed by the runtime via `BufferProvider::lookup_trace`. |
| `TraceGatherProduct { dst, source_tables, shifts, mask }` | `dst[j] = ∏_k source_tables[k][(lookup_keys[j] >> shifts[k]) & mask]` for every cycle. Generic trace-driven gather-product across multiple source tables. |
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

### Protocol-specific — lower to primitives (5)

Protocol-specific both in name and behavior. Each lowers to a sequence of
primitives. **Resolved in O5** (refined targets below — the original
OPS.md targets were optimistic; reading the actual handlers after
landing the first two S5 renames exposed three distinct blocker classes).

**Blocker groups:**

- **(A) State-in-host + trace-driven**: `state.instance_weights: Vec<F>`
  is host-allocated and accessed by trace cycle index. The handlers
  below either write to or read from this state while also consuming
  `provider.lookup_trace()`. No existing primitive combines both —
  either the state must relocate to a device buffer
  (`PolynomialId::InstanceWeights` or similar, cascading into every
  op that reads `instance_weights[j]`), or a new primitive family
  (`Op::TraceGather*` / `Op::TraceScatter*`) must land first.
- **(B) Kernel formula lowering**: The per-round prefix×suffix +
  `combine_entries` evaluation in `ReadCheckingReduce` / `RafReduce`
  would need to lower into a `KernelSpec.formula` that the existing
  `Op::Reduce` machinery can evaluate. Substantial compiler work —
  requires extending `KernelSpec` with the needed formula shape
  (sum-over-entries, gamma-weighted product) and verifying the
  `Op::Reduce` backend path produces the same result.
- **(C) WeightedSum shape mismatch**: `Op::WeightedSum` computes
  `Σ challenge^power × source[i] + identity_scale × i`. The handlers
  below produce either (1) trace-driven gather-products (not a linear
  combination) or (2) per-cycle conditional constant injection (not a
  linear combination). Either needs new primitive surface or
  preprocessed polys that encode pure functions of `i` (identity,
  bit-uninterleave, chunk-size constant).

| Variant | Blocker | Refined target |
|---|---|---|
| `MaterializeCombinedVal { kernel }` | A | NOT a `WeightedSum` — combines a pre-computed `table_values` array (built from `instance_scalars` × `combine_entries`) with a trace-driven gather-by-`table_kind_indices[j]` plus per-cycle conditional from `is_interleaved[j]`. Needs new primitive `Op::TraceGatherIndexed` and conditional-scalar injection. |
| `SuffixScatter { kernel, suffix_len }` | A | NOT a `WeightedSum` — trace-driven scatter into `num_tables × suffixes_per_table` output polys, weighted by `instance_weights[j]` and `suffix_ops[t].eval(key & suffix_mask)`. Needs new `Op::TraceScatter { outputs, index_source, weight_source, value_fn }`. Field set already slimmed in S5.scatter_field_slim — `phase` collapsed into `suffix_len` at emission time. |
| `QBufferScatter { kernel, suffix_len }` | A | Same primitive family as `SuffixScatter` — 6 Q-buffer outputs with bit-uninterleave and a conditional on `is_interleaved[j]`. Same new primitive applies with richer output set. Field set already slimmed in S5.scatter_field_slim (same as `SuffixScatter`). |
| `ReadCheckingReduce { kernel, round, r_x_challenge }` | B | `Op::Reduce { specs: [ReduceSpec { axes: Flat, kernel: composed_prefix_suffix, .. }] }`. Compiler lowers the `combine_entries` matrix + `prefix_lowered[round]` into `KernelSpec.formula` at compile time. Substantial `KernelSpec` extension. |
| `RafReduce { batch, instance, kernel }` | B | `Op::Reduce` with a product-of-sums formula over the Q/P buffers, gamma-weighted. Reads `state.read_checking_evals` + `state.batch_instance_claims[batch][instance]` as implicit inputs; generic `Op::Reduce` doesn't express this state flow yet. |

**Graduation order** (revised):

1. **S5.field_slim** (landed): slimmed `UpdateInstanceWeights`
   `{num_phases, phase}` → `suffix_len`. Reduces field-set protocol
   leakage even though the variant itself persists pending Group A
   resolution. One commit.
1a. **S5.scatter_field_slim** (landed): slimmed `SuffixScatter` and
   `QBufferScatter` `phase` → `suffix_len` (same pattern,
   `(num_phases − 1 − phase) × chunk_bits` precomputed at emission
   time). Variants persist pending Group A. One commit.
1b. **S5.init_weights_slim** (landed): dropped `num_prefixes` from
   `InitInstanceWeights` by pre-sizing `state.instance_scalars` at
   runtime init from `max(ic.num_prefixes)` across kernels. Handler
   resets via `.fill(None)` instead of re-allocating. Variant persists
   pending Group A. One commit.
2. **S5.materialize_p_buffers** (landed): Group C lowered.
   `Op::MaterializePBuffers` removed. Four new derived polys
   (`PBufferScale`, `PBufferHalfScale`, `PBufferUninterleaveLo`,
   `PBufferUninterleaveRo`, parameterized by `chunk_bits`) computed
   on-demand by `DerivedSource::compute()`. Emission replaced with
   3× `Op::WeightedSum`.
3. **Group A prerequisite** (landed): `state.instance_weights`
   relocated to `device_buffers[PolynomialId::InstanceWeights]`.
   Preparatory for lowering the four host-state + trace-driven ops.
4. **S5.init_instance_weights** (landed): `Op::InitInstanceWeights`
   removed. Emission replaced with `InitExpandingTable` +
   `|r_reduction| × ExpandingTableUpdate` on `InstanceWeights`, plus
   an `InstanceScalarUpdate { Clear, ..num_prefixes }` for the
   scalar reset. The eq-table construction matches
   `EqPolynomial::evals` verified by transcript parity.
5. **S5.update_instance_weights** (landed): `Op::UpdateInstanceWeights`
   removed. Replaced with new generic primitive
   `Op::TraceGatherMultiply { dst, source_table, shift, mask }`
   that does `dst[j] *= source_table[(lookup_keys[j] >> shift) & mask]`
   for every cycle `j`. The emission sites pass
   `dst = InstanceWeights`; the primitive is generic over any
   destination/source pair and carries no protocol-specific fields.
6. **S5.materialize_ra** (landed): `Op::MaterializeRA` removed.
   Replaced with new generic primitive `Op::TraceGatherProduct
   { dst, source_tables, shifts, mask }` that does
   `dst[j] = ∏_k source_tables[k][(lookup_keys[j] >> shifts[k]) & mask]`
   for every cycle `j`. The compiler emits `n_vra` of these ops per
   materialization, one per RA chunk, selecting the appropriate
   `ExpandingTable` slice and shift offsets at compile time.
3. **S5.instance_weights_device** (Group A, prerequisite): relocate
   `state.instance_weights` to a `PolynomialId::InstanceWeights`
   device buffer. Cascades through handlers — every
   `state.instance_weights[j]` read becomes a device-buffer read.
   Introduces new `Op::TraceGatherMultiply` and `Op::TraceScatter`
   primitives. Pre-requisite for ops 1–6 above. Multi-commit.
4. **S5.kernel_formula** (Group B): extend `KernelSpec.formula` to
   express `combine_entries` + gamma-weighted product shapes.
   Substantial compiler work; standalone. Pre-requisite for ops 7–8.
5. Then the individual ops lower to primitives one-by-one.

---

## Summary

| Category | Count | `is_primitive()` target |
|---|---|---|
| Primitive — compute | 16 | `true` |
| Primitive — PCS | 7 | `true` |
| Primitive — orchestration | 9 | `true` |
| Primitive — resource | 3 | `true` |
| Batch-scaffold | 4 | `true` |
| Redundant (landed O4.a) | 0 (was 3) | — |
| Conditional (deferred to O6/O7) | 2 | `true` (ratchet unchanged until pass ships) |
| Protocol-specific: rename (landed S5.rename, S5.build_segmented_eq) | 0 (was 2) | — |
| Protocol-specific: lowered (landed S5.materialize_p_buffers, S5.init_instance_weights, S5.update_instance_weights, S5.materialize_ra) | 0 (was 4) | — |
| Protocol-specific: lower (→ O5) | 5 | `true` (ratchet unchanged until lowered) |
| **Current total** | **46** | |

Post-O5 target: ~44 primitive + batch-scaffold variants. New primitives
landed during Group A partial lowering: `Op::TraceGatherMultiply`
(replaces `UpdateInstanceWeights`) and `Op::TraceGatherProduct`
(replaces `MaterializeRA`). Outstanding additions expected during
remaining lowering:
- `Op::TraceScatter` (Group A) — needs suffix-op / bit-uninterleave
  subroutines for `SuffixScatter` and `QBufferScatter`. Likely two
  separate primitives due to per-op protocol math.
- `Op::TraceGatherIndexed` (Group A) — for `MaterializeCombinedVal`.
  Also needs a scalar-compute side-channel for `table_values[t]`
  (probably via `InstanceScalarUpdate` with extended
  `state.instance_scalars` slots).
- Extended `Op::Reduce` / new `KernelSpec.formula` shapes
  (Group B) — for `ReadCheckingReduce` and `RafReduce`.

Landed-via-rename/lowering this session: `Op::BuildSegmentedEq`
(S5.build_segmented_eq), `Op::InstanceScalarUpdate` (S5.rename),
`Op::TraceGatherMultiply` (S5.update_instance_weights),
`Op::TraceGatherProduct` (S5.materialize_ra). `Op::MaterializePBuffers`
fully removed via 3× `WeightedSum` + 4 new derived polys
(S5.materialize_p_buffers).

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
