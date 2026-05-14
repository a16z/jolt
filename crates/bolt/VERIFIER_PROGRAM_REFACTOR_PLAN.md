# Verifier Program Refactor Plan (S2 — S6)

This is the implementation plan for the post-S1 verifier refactor. It is a
companion to `crates/bolt/GOAL.md`. Read GOAL.md first; this document assumes
the `Audit Tiers` framing introduced there, but the scope is broader than
follow-up LOC cleanup.

The target is a complete refactor of how the generated verifier is constructed,
compiled, and emitted. The current stage-shaped Rust is useful reference
material, but it is not the desired architecture. The desired architecture is a
typed verifier program: an ordered plan of verifier steps over commitments,
transcript events, sumcheck drivers, value-graph computations, opening batches,
and PCS checks.

S1 (the audit-tier split that landed in PR #1523) reframes the shared
verifier runtime as two explicitly-bounded tiers:

```text
Tier A (Bolt verifier runtime)        stages/common.rs           1,265 LOC
Tier B (audited Jolt verifier core)   stages/jolt_relations.rs     638 LOC
Tier C (generated stage data + verifier.rs)                      6,430 LOC
```

S2 — S6 below progressively replace hardcoded stage-shaped Rust with typed
verifier-program data by lifting today's hand-written Rust into MLIR
vocabulary. Each slice is intended to be one reviewable PR stacked on top of the
previous one.

## Guiding principle

> Most code in the verifier templates is an *interpreter* for plans, not
> protocol math. Lift the rest of the protocol math into MLIR vocabulary so
> the interpreter is the only thing humans need to audit.

Concrete corollary: **Tier A should not be per-protocol generated code at
all**. It is a small, generic interpreter that happens to be checked into
each protocol's verifier crate by historical accident. S2 corrects this.

The remaining slices (S3 — S6) close the gap between "generic interpreter"
and "Jolt-specific math evaluators" by extending the compute dialect with
the polynomial primitives, point reorderings, indexed-eval addressing, and
relation-expression vocabulary that Tier B currently fills with hand-written
Rust.

Numbered stages may remain as proof-layout slots and diagnostic scopes, but
they should not remain the verifier's core execution model. A generated
verifier should answer "what verifier program is executed?" rather than "how
many stage modules are hardcoded?"

## Design thesis: typed verifier IR, not better templates

The plan should not optimize for making today's Rust templates shorter. The
real target is a small typed verifier IR whose Rust rendering is nearly a
serialization detail:

1. **MLIR owns semantic facts.** If verifier execution depends on a fact, the
   fact should exist as a dialect op, typed attr, or typed plan field before the
   Rust emitter sees it.
2. **The Rust emitter projects typed plans.** It should not reconstruct protocol
   meaning by parsing symbols, matching string prefixes, or emitting bespoke
   helper bodies that duplicate protocol math.
3. **Symbols are typed handles.** Examples in this document use `&'static str`
   where that keeps the prose readable, but implementation APIs should prefer
   `TypedPlanSymbol<Tag>`-style references. Plain strings are diagnostics and
   serialization names, not execution contracts.
4. **The runtime interprets typed value domains.** Scalars, points, field
   vectors, eval families, relation outputs, and bytecode encodings are
   different verifier value kinds. Do not force point transforms or vectors into
   `FieldExprKind` just because field expressions exist today.
5. **Relations should be scheduled dataflow, not a second DSL.** Prefer one
   typed verifier value graph over separate mini-interpreters for field
   expressions, point expressions, eval families, and relation terms. A relation
   check should usually be "the expected sumcheck output is scalar value X"
   where X is produced by the same typed graph as other verifier values.
6. **Protocol vocabulary stays at the protocol boundary.** Generic runtime code
   may carry opaque IDs or be generic over protocol-specific enums, but it
   should not define Jolt relation variants such as `Stage6BytecodeReadRaf`.
7. **No fallback execution paths.** When a typed path lands, delete the string
   path in the same slice. Rollback by reverting the slice, not by keeping a
   compatibility parser in the verifier.
8. **Auditability beats clever compaction.** A generated table that is longer
   but explains the verifier contract is better than a short table whose
   meaning only appears in the emitter.

## Verifier program model

The long-term runtime boundary should look like an interpreter for typed
verifier steps, not a sequence of bespoke `verify_stageN` functions:

```rust
pub struct VerifierProgramPlan {
    pub params: VerifierParams,
    pub steps: &'static [VerifierStepPlan],
    pub values: &'static [ValuePlan],
    pub sumchecks: &'static [SumcheckDriverPlan],
    pub openings: &'static [OpeningClaimPlan],
    pub opening_batches: &'static [OpeningBatchPlan],
    pub pcs_checks: &'static [PcsCheckPlan],
}

pub enum VerifierStepPlan {
    ReceiveCommitments { batch: CommitmentBatchId },
    TranscriptAbsorb { event: TranscriptEventId },
    TranscriptSqueeze { output: ValueId },
    EvaluateValues { until: ValueId },
    VerifySumcheck { driver: SumcheckDriverId },
    EmitOpeningBatch { batch: OpeningBatchId },
    VerifyPcsOpening { check: PcsCheckId },
}
```

`SumcheckDriverPlan` is the central verifier concept. It should contain the
proof slot, round schedule, input-claim plan, batching plan, output-claim
emission, value/eval observation, expected-output scalar, and opening-claim
emission for one sumcheck driver. The runtime should not need to know whether a
driver came from "Stage 3" or "Stage 6" except for diagnostics and proof-record
layout.

Commitment receipt and PCS opening verification are first-class verifier steps,
not synthetic stages. Partial verification targets should eventually be named
checkpoints or proof slots, not `through_stage5` / `through_stage6` booleans.

## Non-regression contracts

This refactor is allowed to move code across boundaries, but it is not allowed
to make the verifier harder to audit, larger at the finish line, slower without
an explicit decision, or weaker semantically.

1. **Readability cannot regress.** Every slice must preserve or improve the
   ability to audit generated Rust without reading Bolt lowering code. The final
   stack must pass an independent readability review where reviewers see the
   current verifier surface and the new generated verifier surface, but not the
   compiler implementation.
2. **LOC cannot regress at completion.** The completed stack must not exceed the
   locked post-S1 generated-verifier baselines in `GOAL.md`, and the intended
   trajectory is downward. Temporary slice-local LOC growth is acceptable only
   when it moves hand-written protocol math into declarative typed plan data,
   is called out in the PR, and does not weaken the final LOC target.
3. **Performance cannot silently regress.** Existing core-vs-Bolt perf oracles
   are useful smoke gates, but their current thresholds are intentionally broad.
   Before interpreter-heavy slices land, capture a verifier-time baseline and
   gate repeated regressions. A repeatable verifier regression must either be
   fixed, justified as the price of a specific readability/security gain, or
   explicitly approved before landing.
4. **Semantic and tamper behavior cannot regress.** Core/Bolt accept-reject
   equivalence, transcript state agreement, opening equality behavior, and PCS
   verification must stay covered by positive and negative oracles.
5. **No compatibility shadows.** Do not keep old stage-local execution paths as
   fallbacks once typed verifier-program execution lands. Rollback by reverting
   the slice, not by leaving dual interpreters in the verifier.

## Scoreboard (target trajectory)

| Slice | Tier A | Tier B | Tier C | jolt-verifier total | Notes |
|-------|-------:|-------:|-------:|--------------------:|-------|
| post-S1 (today) | 1,265 | 638 | 6,430 | 7,905 | hard ceilings: A 1,400 / B 700 / C surface 6,100 |
| post-S2 | ~50 | 638 | 6,430 | ~6,640 | Tier A moves to `bolt-verifier-runtime` crate |
| post-S2.5 | ~150-250 | 638 | ~6,200 | ~6,900 | top-level verifier executes a typed verifier program |
| post-S3 | ~50 | ~500 | 6,430 | ~6,500 | poly, point reordering, gamma-power vector ops |
| post-S4 | ~50 | ~350 | 6,500 | ~6,500 | typed indexed-eval addressing |
| post-S5 | ~50 | ~290 | 6,700 | ~6,700 | relations as typed plans (Tier C grows) |
| post-S6 | ~50 | ~50 | 6,800 | ~6,800 | bytecode encoding as typed plans (optional) |

Scoreboard numbers are planning targets for individual slices. The
non-regression contracts above are stricter: the completed stack must not end
larger, less readable, slower without approval, or semantically weaker than the
post-S1 baseline.

---

## S2: Promote Tier A to `bolt-verifier-runtime` crate

**Goal.** Stop emitting Tier A as a per-protocol template. Instead, ship it
once as a real workspace crate that the generated `jolt-verifier` crate
declares as a Cargo dependency.

**Why first.** Largest leverage, smallest semantics change. Tier A is mostly
generic Bolt scaffolding, but it is not perfectly protocol-agnostic yet:
today it still owns Jolt-shaped IDs such as `RelationKind` and
`SourceStage::{Stage6, Stage7}` and several `Fr`-concrete helper entry points.
S2 should therefore be treated as both extraction and boundary correction, not
as proof that the runtime is already pure generic infrastructure. Once Tier A
is a real crate with an explicit public API, every subsequent slice operates
against a stable, versionable surface.

### Concrete plumbing

1. New crate `crates/bolt-verifier-runtime/` with `src/lib.rs` seeded from
   today's `crates/bolt/src/protocols/jolt/verifier_common.rs.template`, but
   reviewed as a public runtime API rather than copied verbatim. At minimum:
   cross-crate paths, the error-conversion macro, and protocol-ID ownership
   must be corrected during the move.
2. Workspace `Cargo.toml` registers the new crate.
3. `crates/bolt/src/protocols/jolt/artifacts.rs::verifier_runtime_modules`
   stops emitting the `common` module (the `ProtocolRuntimeModule` entry
   for `common` is removed). The `jolt_relations` entry stays.
4. Stage emitters (`crates/bolt/src/protocols/jolt/emit/rust/stage{1..8}.rs`)
   change `super::common::{...}` imports to
   `bolt_verifier_runtime::{...}`. The `super::jolt_relations::{...}`
   imports are unchanged.
5. The Bolt manifest emitter (the part of `artifacts.rs` that writes the
   generated `Cargo.toml` for `jolt-verifier`) adds a `bolt-verifier-runtime`
   dependency entry, mirroring how `jolt-field`, `jolt-poly`, etc. are
   already wired.
6. `crates/jolt-verifier/src/stages/common.rs` is deleted from the goldens.
   `crates/jolt-verifier/src/stages/mod.rs` drops `pub mod common;`. Do not
   leave a `jolt_verifier::stages::common` compatibility re-export; update every
   call site in the same slice.
7. `crates/jolt-equivalence/src/plan_adapters/generated_stage*.rs` and the
   oracle modules update their `use jolt_verifier::stages::common::*`
   imports to `use bolt_verifier_runtime::*`.
8. Decide and implement the protocol-ID boundary. Preferred: make runtime plan
   structs generic over a protocol relation enum, or carry an opaque
   `RelationId`/`TypedPlanSymbol<RelationTag>` that is defined by the protocol
   layer. Avoid defining Jolt relation variants in `bolt-verifier-runtime`.
   If this makes S2 too large, split it into S2a extraction and S2b
   protocol-ID cleanup before starting S3, and keep the temporary ownership
   debt gated.

### Test/gate updates

- `verifier_cleanup.rs` removes `bolt_runtime_loc` (it no longer counts
  toward the verifier crate). Add an equivalent gate over the new crate:
  `BOLT_VERIFIER_RUNTIME_LOC_CEILING ≈ 1,400`.
- `checked_in_generated_verifier_respects_boundary_hygiene` adds
  `bolt-verifier-runtime` to the *allowed* import list (it is the
  intentional runtime dependency).
- `commitment_ir.rs::assert_rust_source_compiles` no longer needs to stage
  `common.rs` into the test harness.

### Blockers and complications

- **Cross-crate blast radius.** `jolt-equivalence` has many import sites
  that name `jolt_verifier::stages::common::*`. The full-cutover rule
  (workspace policy) forbids leaving back-compat re-exports, so all sites
  must move to `bolt_verifier_runtime` in the same PR. Mechanical, but
  large.
- **Genericity debt.** `RelationKind` and `SourceStage` are the current
  strongest evidence that Tier A is not purely generic. Do not quietly move
  those names into a crate called `bolt-verifier-runtime` and declare the
  boundary clean. Either genericize them in S2 or document the exact remaining
  ownership debt with a follow-up gate.
- **Type aliases in stage files.** Generated stage files contain
  `pub type Stage6FieldExprPlan = FieldExprPlan;` and similar. After S2
  the right-hand side is `bolt_verifier_runtime::FieldExprPlan`. The
  per-stage Bolt emitter that produces these aliases must update its import
  path and the alias targets. This is one localized change per stage
  emitter.
- **`impl_runtime_plan_error_conversion!` macro.** Currently
  `pub(crate) use`. Either promote to `pub` in the new crate, or keep it
  per-stage by re-exporting it through `jolt-verifier`. Lean: `pub` in the
  new crate.
- **Generated `Cargo.toml` dependency injection.** The Bolt manifest
  emitter must learn that `bolt-verifier-runtime` is a workspace-relative
  path dependency for non-published builds, but a versioned
  registry dependency for published builds. Check how `jolt-field` etc.
  are currently wired and follow the same pattern.
- **`cargo check` chain.** Existing CI runs
  `cargo check -p bolt -p jolt-verifier -p jolt-prover -p jolt-equivalence`.
  Add `-p bolt-verifier-runtime`.

### Acceptance criteria

```text
cargo check -p bolt -p bolt-verifier-runtime -p jolt-verifier
            -p jolt-prover -p jolt-equivalence  --quiet
cargo nextest run -p bolt --test verifier_cleanup --no-capture
cargo nextest run -p bolt --test commitment_ir --cargo-quiet
cargo nextest run -p jolt-equivalence --test generated_role_crates --cargo-quiet
cargo nextest run -p jolt-equivalence --test bolt_commitment --no-capture
```

All green. The `verifier_cleanup` metrics output reports
`bolt_runtime_loc = 0` (Tier A no longer in the verifier crate) and a new
`bolt_verifier_runtime_loc` line.

### Rollback

S2 is reversible by re-introducing the `ProtocolRuntimeModule` entry for
`common` and reverting the import-path changes. The new crate can stay (it
is harmless) or be deleted. No data or proof-format changes are involved.

### Estimated wall-clock

One agent session, 60-90 minutes. The work is mechanical but touches many
files because of the cross-crate import changes.

---

## S2.5: Introduce the verifier-program executor

**Goal.** Make the top-level verifier execute a typed `VerifierProgramPlan`
instead of manually calling a fixed list of numbered stages. This slice changes
the architectural shape before the deeper algebraic cleanup starts.

This does not require fully deleting stage modules immediately. Stage modules
may still own proof-slot conversion and local plan constants at first. The
important cutover is that the top-level verifier sees a program of typed steps,
and `stageN` names become proof-slot labels rather than the verifier's control
flow.

### Concrete plumbing

1. Add shared runtime types:

   ```rust
   VerifierProgramPlan
   VerifierStepPlan
   ProofSlot
   VerifierCheckpoint
   ArtifactStore
   ```

   `ProofSlot` is the protocol/proof-layout identity. `VerifierStepPlan` is the
   execution identity. They should not be conflated.
2. Represent commitment receipt, transcript absorbs/squeezes, sumcheck driver
   verification, opening-batch emission, and PCS opening verification as program
   steps.
3. Replace the hardcoded stage call chain in `verifier.rs` with a loop over the
   generated program. Existing `verify_stageN_with_program` functions can be
   used as step executors during this slice, but there must be one execution
   path, not old/manual and new/program paths side by side.
4. Replace target APIs such as `through_stage5`, `through_stage6`, and
   `through_stage7` with named checkpoints or proof-slot targets. The old names
   can survive as user-facing constructors only if they are constants for the
   new checkpoint values, not separate branches.
5. Make stage artifacts addressable through `ArtifactStore` keyed by typed
   proof slots / claim IDs rather than Rust fields that assume a permanent
   number of stages.
6. Stage 8 becomes `VerifyPcsOpening { check }` plus supporting opening-batch
   plans. It should not be special because it happens to be the eighth module.

### Acceptance criteria

- `verifier.rs` no longer manually sequences every numbered stage as bespoke
  control flow.
- There is a checked-in generated `VERIFIER_PROGRAM` or equivalent top-level
  const plan that a reviewer can scan.
- Commitment handling, all current sumcheck stages, and PCS opening
  verification appear as typed verifier steps.
- Partial verification targets are typed checkpoints/proof slots.
- No duplicate old/new verifier execution path remains.
- Existing semantic, tamper, and import gates still pass.

### Blockers and complications

- **Artifact typing.** Today's `JoltVerificationArtifacts` has hardcoded fields
  for each stage. Moving to `ArtifactStore` risks hiding what each step
  produces. The generated program should keep named artifact IDs and readable
  comments/table rows so the dataflow remains auditable.
- **Error mapping.** Stage-specific errors are useful diagnostics. Preserve
  that quality by attaching proof-slot/checkpoint context to generic runtime
  errors instead of flattening everything into one opaque error.
- **Proof serialization.** This slice should not change proof format. It should
  adapt the existing proof fields into typed proof slots.

### Estimated wall-clock

Two to three agent sessions. This is mostly control-flow architecture and
artifact typing, not new protocol math.

---

## S3: Typed verifier value graph + polynomial primitives

**Goal.** Lift the small set of pure-dataflow primitives in Tier B into a
typed verifier value graph, backed by MLIR ops. Tier B's relation evaluators
stop calling `EqPolynomial::mle`, `bytecode_gamma_powers`, `field_powers`,
`reverse_slice`, `prefix_point`, `suffix_point`, and the `normalize_*_point`
helpers directly. The important design point is that this is not "add more
variants to `FieldExprKind`"; it is a typed graph over scalar, point, and
field-vector values.

### Dialect changes

Extend `crates/bolt/irdl/compute.mlir` (no new dialect; these are dataflow
ops, same family as the existing `point_*` and `field_*` ops). Add an explicit
vector carrier instead of overloading `@compute::@point<>`:

```mlir
irdl.type @field_vector
```

Then add typed scalar, point, and vector producers:

```mlir
compute.poly_mle              %p1, %p2 -> %scalar
compute.poly_eq_indexed       %point, index = N -> %scalar
compute.poly_identity_eval    %point -> %scalar
compute.poly_lt_eval          %x, %y -> %scalar
compute.poly_operand_eval     %point, side = "left" | "right" -> %scalar

compute.point_reverse         %p -> %p'
compute.point_split           %p, at = log_k -> %lo, %hi
compute.point_prefix          %p, length = N -> %p'
compute.point_suffix          %p, length = N -> %p'

compute.field_pow_vector      %base, count = N -> %vec : !field_vector
compute.field_vector_product  %vec -> %scalar
compute.field_vector_sum      %vec -> %scalar
```

`compute::poly_mle` is the workhorse. The other `poly_*` ops are convenient
specializations for relations that already exist (`identity`, `lt`,
`operand`, `eq_indexed`). `field_vector_*` ops exist because verifier math
uses ordered scalar tuples whose meaning is not a multilinear query point.

### Runtime additions

`bolt-verifier-runtime` (post-S2) gains typed value storage and interpreters:

```rust
pub enum VerifierValueKind {
    Scalar,
    Point,
    FieldVector,
}

pub enum ScalarExprKind {
    OpeningEval,
    Add,
    Sub,
    Mul,
    Neg,
    Pow(usize),
    LagrangeBasisEval { domain_start: i64, domain_size: usize, index: usize },
    PolyMle,
    PolyEqIndexed { index: usize },
    PolyIdentityEval,
    PolyLtEval,
    PolyOperandEval { side: OperandSide },
    FieldVectorProduct,
    FieldVectorSum,
}

pub enum PointExprKind {
    Reverse,
    Prefix { length: usize },
    Suffix { length: usize },
    SplitLo { at: usize },
    SplitHi { at: usize },
}

pub enum FieldVectorExprKind {
    Powers { count: usize },
    EvalFamily,
}
```

`ValueStore` should store each value in the correct domain. The scalar
interpreter may call `jolt-poly`'s `EqPolynomial::mle`, but point transforms
must be evaluated by the point interpreter and vector producers by the vector
interpreter. This keeps the type boundary honest and prevents the runtime from
becoming a bag of ad hoc helper functions.

### Tier B impact

After S3, `verifier_jolt_relations.rs.template` no longer defines:
`bytecode_gamma_powers`, `field_powers`, `reverse_slice`, `prefix_point`,
`suffix_point`, `indexed_boolean_eq`, `normalize_bytecode_read_raf_point`,
`normalize_instruction_read_raf_point`, `operand_polynomial_eval`,
`identity_polynomial_eval`, `lt_polynomial_eval`. Total: ~140 LOC removed.

The `expected_stage67_*` evaluators stay hand-written for now, but their
*bodies* shrink because the primitives they invoke are now provided by the
runtime.

### Blockers and complications

- **`jolt-poly` API stability.** `EqPolynomial::mle` must have a stable
  signature and bit-exact semantics. `jolt-poly` is on the Bolt side of the
  generic compiler, so this is fine, but worth a sanity check that we're
  not pinning to an evolving signature.
- **Value graph foundation.** If S3 starts feeling like a pile of unrelated
  enum variants, stop and implement the typed value graph first. A clean
  scalar/point/vector split is more important than landing every primitive in
  one PR.
- **`@compute::@field_vector<>` as MLIR type.** Do not re-use `@point<>` for
  gamma-power vectors. Points are multilinear query coordinates; gamma powers
  are ordered scalar tuples. Conflating them would make later relation-plan
  typing harder and hide meaning from reviewers.
- **Stage emitter updates.** The Stage 6/7 emitters (the largest in
  `crates/bolt/src/protocols/jolt/emit/rust/`) currently inline calls to
  these helpers when building `expected_stage67_*` bodies. They must learn
  to lower those call sites to `compute.poly_mle` etc. and let the
  generic interpreter run them.
- **Validator updates.** Each new `compute::*` op needs an entry in the
  Bolt validator so malformed plans (wrong arity, wrong operand types) are
  rejected at compile time.

### Acceptance criteria

`muldiv` e2e test passes in both `--features host` and `--features host,zk`
(this is the workspace's primary correctness check). Tier B drops to
~500 LOC. New metrics in `verifier_cleanup.rs`:
`compute_poly_op_call_sites`, `compute_point_op_call_sites` reported but
not gated yet.

### Rollback

Each new MLIR op is independently revertible. Worst case we keep the
`poly_mle` op (which is unambiguous) and revert the `normalize_*_point`
lowering if the dataflow ordering turns out to be subtle.

### Estimated wall-clock

Two agent sessions, ~120-180 minutes total. Most of the time is in the
emitter changes (Stage 6/7 bodies) and validator updates.

---

## S4: Typed indexed-eval addressing

**Goal.** Eliminate the last big string-dispatch site in Tier B:
`indexed_evals_by_prefix_any(evals, "stage6.booleanity.eval.InstructionRa_")`
and friends. Replace with a typed eval-family vocabulary.

### The problem today

Stage 6/7 evaluators do:

```rust
let booleanity_evals =
    indexed_evals_by_prefix_any(evals, "stage6.booleanity.eval.InstructionRa_")?;
```

This works because the prover emits evals named `..._0, ..._1, ..._2` and
the verifier wants them as `Vec<Fr>`. The verifier reconstructs the family
by string-prefix matching plus integer-suffix parsing. This is the *only*
remaining "execution-relevant string dispatch" that the current
`verifier_cleanup` gates do not yet enforce-to-zero.

### Dialect changes

Extend `compute::sumcheck_eval` with an `eval_family` attribute (or add a
sibling `compute::sumcheck_eval_family` op):

```mlir
compute.sumcheck_eval_family {
  sym_name = "stage6.booleanity.instruction_ra_family"
  produced_by = @stage6_booleanity_driver
  oracle_family = "InstructionRa"
  count = 3
} -> %eval_vec : !field_vector
```

The existing `compute::sumcheck_eval` ops with names like
`...InstructionRa_0`, `...InstructionRa_1`, ... are replaced by a single
family op when the consumer needs the block as a vector. The prover side
already emits these as a contiguous block; the verifier side gains a typed
`!field_vector` handle to the whole block.

### Runtime additions

Add `EvalFamilyPlan` to `bolt-verifier-runtime`:

```rust
pub struct EvalFamilyPlan {
    pub symbol: &'static str,
    pub source: &'static str,        // sumcheck driver symbol
    pub oracle_family: &'static str, // diagnostic
    pub count: usize,
}
```

`ValueStore` learns to materialize a `FieldVector` value for family symbols
when the parent sumcheck output is observed. Consumers ask for a typed vector,
not a set of scalars recovered by prefix matching.

### Tier B impact

Each `expected_stage67_*` that today calls `indexed_evals_by_prefix_any`
instead reads a typed `&[Fr]` from the store. Estimated ~60 LOC reduction
in Tier B.

### Blockers and complications

- **Proof format.** The proof's `evals` field is already
  `Vec<StageNamedEval<F>>` with explicit `name` strings (verified in
  `verifier_common.rs.template:393-397`). S4 changes how the *verifier
  consumes* named evals, not how they are serialized. No back-compat
  break. *This was the biggest risk; it is not real.*
- **Prover/verifier symmetry.** The prover-side emitter must annotate the
  same eval block as a family. If this is a separate emitter, both must be
  updated together.
- **`indexed_evals_by_prefix` and `indexed_evals_by_prefix_any` removal.**
  These two helpers in Tier A become unused after S4. They can be deleted
  from `bolt-verifier-runtime` to keep the API surface minimal.
- **Testing.** Add a `verifier_cleanup` gate
  `RELATION_INDEXED_EVAL_PREFIX_SITES_CEILING = 0` that fires if any
  generated stage source still calls `indexed_evals_by_prefix*`.

### Acceptance criteria

`muldiv` passes in both modes. Zero `indexed_evals_by_prefix*` call sites
in the generated verifier. Tier B drops to ~350 LOC.

### Rollback

Pure refactor. Roll back by reverting the eval-family slice. Do not keep
`indexed_evals_by_prefix_any` in the runtime as a verifier fallback once the
typed family path lands; that would leave two execution paths and weaken the
full-cutover guarantee.

### Estimated wall-clock

One agent session, 60-90 minutes.

---

## S5: Relations as typed plan data

**Goal.** Make the Stage 6/7 relation evaluators (`expected_stage67_*`)
declarative MLIR-derived plans rather than hand-written Rust functions.
After S5, the only thing left in Tier B is the bytecode-row encoding
(addressed by S6).

### The shape

After S3 + S4, every `expected_stage67_*` body is structurally:

```text
val = Σ_i γ^i · (Π_j eq_mle(P_ij, query_point) · eval_k_ij) + boundary_terms
```

That is a flat algebraic expression DAG over plan operands. The preferred
abstraction is not a separate relation-expression language; it is a relation
metadata record pointing at a scalar value produced by the typed verifier value
graph introduced in S3/S4:

```rust
pub struct RelationPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub kind: RelationId,
    pub query_point: &'static str,
    pub expected_output: &'static str, // scalar produced by value graph
}
```

The runtime gains a generic `evaluate_relation_plan(plan, store) -> Fr` that
only resolves the typed `expected_output` value and applies relation-level
diagnostics/error mapping. The algebra itself should already be visible as
scalar/point/vector plan rows. Introduce a separate `RelationTerm` enum only if
the value graph cannot express a relation without contortions, and document why
that second interpreter is worth the extra audit surface.

### Dialect changes

Add a top-level `compute::relation` op that ties a relation kind to a typed
expected-output scalar:

```mlir
compute.relation {
  sym_name = "stage6.booleanity"
  kind = "Stage6Booleanity"
  query_point = @stage6_booleanity_point
  expected_output = @stage6_booleanity_expected
}
```

Stage emitters lower this to a `RelationPlan` in the generated stage files.
The expression feeding `expected_output` should be ordinary typed value-graph
data, not hidden inside the emitter.

### Tier C impact (acknowledged growth)

The relation plans and their supporting scalar/point/vector value rows live in
the *generated* stage Rust, not in the runtime. Stage 6/7 will grow by an
estimated ~200 LOC of relation/value tables. This is the explicit trade: Tier B
shrinks by ~200 LOC and Tier C grows by ~200, but Tier C is declarative data
(audit-easy) while Tier B is hand-written Rust (audit-hard).

The `GENERATED_VERIFIER_TARGET_LOC` ceiling will need to bump from 6,100 to
~6,300. That is OK provided the structure of stage files becomes more
declarative.

### Blockers and complications

- **Expressiveness limit.** The typed value graph must express every algebraic
  shape that the Stage 6/7 relations currently use. Today's shapes are
  uniform; future relations may not be. If Bolt acquires a new protocol with
  relations that have control flow (e.g., conditional structure based on entry
  data), extend the value graph or add a carefully-scoped relation interpreter
  rather than escaping into generated helper functions.
- **Bytecode-read-RAF special case.** The bytecode-read-RAF evaluator has
  a `for index in 0..stage_value_evals.len()` loop with a per-index
  `int_contrib` mask. Model the loop as a value-graph reduction over a static
  row table, but keep the mask values in the bytecode-encoding plan because
  they are Jolt-specific (S6 territory). For S5, the relation consumes the
  resulting scalar or vector as an opaque typed value; the relation metadata
  itself stays generic.
- **Performance.** The verifier-side value graph walks a small DAG. We expect
  zero perf concern. Keep a smoke benchmark anyway since this code runs once
  per Stage 6/7 sumcheck instance.
- **Audit story.** Tier B drops to ~290 LOC, almost all of which is the
  bytecode encoder. We should explicitly document in `GOAL.md` that the
  relation algebra is *no longer* part of the audit surface; only the
  generic interpreter (in `bolt-verifier-runtime`) and the bytecode
  encoder are.

### Acceptance criteria

Same correctness gates plus: `expected_stage67_*` functions in Tier B
are deleted. New `compute::relation` ops and any value-graph ops they depend on
have validator coverage. `muldiv` passes in both modes.

### Rollback

This is the most invasive slice. If the relation interpreter design is
wrong, S5 should be revertible by restoring the deleted
`expected_stage67_*` functions and removing the relation-plan emitter
output. Recommend keeping the per-relation hand-written code in a
preserved-for-comparison file in the worklog while S5 is being shaken
down.

### Estimated wall-clock

Three to four agent sessions, ~6 hours of agent runtime in total. Highest
risk slice in this plan.

---

## S6: Bytecode encoding as typed plan data (optional)

**Goal.** Capture the Jolt-specific bytecode-row encoding rules as a
typed table rather than hand-written Rust. After S6, Tier B is ~50 LOC
and is essentially just `Stage67BytecodeEntry` (the data table itself,
not the math).

### The encoding today

`stage67_bytecode_entry_stage_values` decomposes a bytecode row into 5
gamma-weighted sums (one per Stage 1-5). The pattern for each stage:

```rust
let mut stage1 = entry.address() + entry.imm() * stage1_gamma_powers[1];
for (flag, gamma) in flags.iter().zip(stage1_gamma_powers.iter().skip(2)) {
    if *flag { stage1 += *gamma; }
}
```

This is a small DSL for "for each entry field, conditionally add a
gamma-weighted contribution." The fields and their conditions vary per
stage but the *shape* is uniform.

### Capture as data

```rust
pub struct BytecodeEncodingPlan {
    pub stage_terms: [&'static [BytecodeTerm]; 5],
}

pub enum BytecodeTerm {
    Address                                    { gamma_index: usize },
    Imm                                        { gamma_index: usize },
    CircuitFlag       { index: usize,            gamma_index: usize },
    EntryFlag         { which: BytecodeFlag,     gamma_index: usize },
    RegisterEq        { which: RegisterRole,     gamma_index: usize },
    LookupTableIndex  { offset: usize,           gamma_base: usize },
}

pub enum BytecodeFlag { IsInterleaved, IsBranch, IsNoop, LeftIsRs1, ... }
pub enum RegisterRole { Rd, Rs1, Rs2 }
```

The runtime gains `evaluate_bytecode_encoding(entries, plan, gamma_powers,
register_points) -> [Fr; 5]`. Tier B keeps only the
`Stage67BytecodeEntry` trait and the per-stage `Stage6BytecodeEntry` /
`Stage7BytecodeEntry` impls (~50 LOC).

### Dialect changes

Optional: a `bytecode.encoding_plan` op family in a new `bytecode.mlir`
sub-dialect, OR keep `BytecodeEncodingPlan` purely as runtime-side data
that the Bolt emitter populates from a small declarative source. Lean:
declarative source first, dialect later if a second protocol uses the
same shape.

### Blockers and complications

- **Protocol-specificity.** This encoding is genuinely Jolt-specific. If
  Bolt only ever has one protocol with this exact shape, S6 is overkill
  and we should leave Tier B at ~290 LOC of typed Rust.
- **Lookup-table indexing.** `stage5 += stage5_gamma_powers[2 + table];`
  is a *variable-index* contribution where the index depends on entry
  data. The plan vocabulary must support this (`LookupTableIndex` above
  handles it).
- **Conditional terms.** `if entry.is_interleaved() { ... }` becomes
  `BytecodeTerm::EntryFlag { which: IsInterleaved, ... }`. Mechanical.
- **Audit win is small.** The remaining 290 LOC of bytecode encoding is
  *already* easy to audit; it is a flat sequence of "if flag then add
  gamma^k" lines. The audit benefit of S6 is lower than S2-S5.

### Recommendation

**Skip S6 unless** Bolt acquires a second protocol that wants the same
encoding shape. The 290 LOC of typed bytecode-encoding Rust is acceptable
audit surface as long as it is well-organized and tested. Mark S6 as
"optional follow-up; revisit when there is a second user."

### Estimated wall-clock

If pursued: two agent sessions, ~3 hours. Otherwise zero.

---

## Cross-cutting concerns

### Coordination with Markos' equivalence track

Each slice that touches generated stage outputs (S2, S2.5, S3, S4, S5) will
require equivalence-side adapter updates in
`crates/jolt-equivalence/src/plan_adapters/generated_stage*.rs`. The
correct workflow is:

1. Land the slice (compiler change + regenerated stage Rust + equivalence
   adapter update) in one PR.
2. Stack on top of the most recent Markos tip
   (`origin/jolt-v2/equivalence` today).
3. Do not split the slice across two PRs (one for compiler, one for
   equivalence). The cross-crate type changes are too tightly coupled.

If Markos pushes new commits to `jolt-v2/equivalence` while a slice is in
flight, the right move is `git rebase origin/jolt-v2/equivalence` and
re-run the full check matrix, not `git merge`.

### Generic compiler vs Jolt-specific quarantine

GOAL.md's "Locked Genericity Decisions" require that generic Bolt
compiler infrastructure not contain Jolt-specific code, and that Jolt
emitters be progressively lifted into generic CPU emitters.

S2 strengthens this only if the extraction also corrects the protocol-ID
boundary. A crate named `bolt-verifier-runtime` should not permanently define
Jolt relation variants. Either make relation/stage identifiers generic or keep
the remaining Jolt vocabulary explicitly quarantined and gated until a follow-up
removes it.

S3 + S4 are pure compute-dialect extensions; no Jolt content. They should add a
typed verifier value graph (`Scalar`, `Point`, `FieldVector`, eval families)
rather than a larger bag of scalar-only field-expression variants.

S5 introduces `compute::relation` as generic relation metadata over the value
graph, not a Jolt-specific relation language. If a closed enum is useful for
readability, define it at the protocol/stage boundary and make the generic
runtime parameterized over it; otherwise use an opaque `RelationId` and keep
only diagnostic strings in the runtime.

S6 is explicitly Jolt-specific. If pursued, it goes under
`crates/bolt/src/protocols/jolt/`, not in `bolt-verifier-runtime`.

### Trust boundary and audit surface evolution

The audit surface contracts as we move through the slices:

```text
post-S1:  Tier A (1,265 LOC)  + Tier B (638 LOC)  = 1,903 LOC audited
post-S2:  bolt-runtime crate  + Tier B (638 LOC)  = ~1,900 LOC, but
                                                    Tier A is now
                                                    versioned and reviewed
                                                    once across protocols
post-S3:  bolt-runtime crate  + Tier B (~500 LOC) = ~1,750 LOC
post-S4:  bolt-runtime crate  + Tier B (~350 LOC) = ~1,600 LOC
post-S5:  bolt-runtime crate  + Tier B (~290 LOC) = ~1,540 LOC
post-S6:  bolt-runtime crate  + Tier B (~50 LOC)  = ~1,300 LOC
```

The shape of the audit also changes: post-S5, almost all of Tier B is the
bytecode encoder, which is *data-table-shaped* hand-written Rust. That is
qualitatively easier to review than algebraic relation evaluators.

### MLIR dialect growth

`compute.mlir` is currently 808 lines. After S3 + S4 + S5 it will probably
land around 1,000-1,100 lines with the new poly, point-reordering,
pow-vector, eval-family, and relation ops. Still small enough to be one
dialect; no need for a `poly` or `relation` sub-dialect at that point.
Revisit if `compute.mlir` exceeds ~1,500 lines.

### Performance

Verifier perf is not the center of this track, but it is a non-regression
contract. The existing ignored `jolt-equivalence/tests/bolt_perf.rs` gates give
us real-data core-vs-Bolt setup/prove/verify/proof-size/RSS measurements, but
their current thresholds are broad smoke limits. Treat them as necessary but
not sufficient for interpreter-heavy slices.

Before S2.5, S3, or S5 lands, record a local verifier-time baseline from the
SHA2-chain perf oracle and rerun it after the slice. If a verifier-time change
is noisy, rerun before calling it real. If a repeatable regression remains,
either fix it or explicitly document why the readability/security gain is worth
the cost. Do not loosen perf thresholds to make a refactor pass.

The expected trend is "no measurable verifier change," because the interpreter
dispatches are straightforward and relation/value-graph evaluation happens a
small fixed number of times per verification. Any accidental
quadratic-in-trace-length loop is a blocker.

### Compatibility with the `zk` feature

All five slices are `cfg`-independent. They affect plan generation and
verifier interpretation, both of which are identical between the standard
and `zk` modes. Each slice must run `muldiv` in both
`--features host` and `--features host,zk` per the workspace's primary
correctness check.

### Emitter discipline

The Rust emitter should become a pretty-printer for typed plans, not the place
where verifier semantics are invented. Use these checks during every slice:

- If emitted Rust contains a new helper function with protocol math, ask why
  that helper is not a typed value-graph op or relation plan row.
- If the emitter parses or assembles meaning from a symbol string, add a typed
  MLIR attr or plan field instead.
- If a runtime API accepts `&'static str` and branches on its contents, replace
  it with a typed enum, typed symbol, or protocol-supplied ID.
- If two interpreters can evaluate the same verifier fact, delete one in the
  same slice.
- If a generated table is too dense for blind review, improve the table shape
  before counting the LOC reduction as a win.

---

## Open questions

These deserve human (Markos / Quang) input before S2 starts:

1. **`bolt-verifier-runtime` location.** Standalone workspace crate under
   `crates/bolt-verifier-runtime/`, or sub-crate of `bolt`? Recommend
   standalone for versionability.
2. **`bolt-verifier-runtime` re-export through `bolt`.** Should `bolt`
   re-export `bolt_verifier_runtime` for convenience, or keep them
   separate? Recommend separate (stricter trust boundary).
3. **Per-stage type aliases.** Generated stage files contain
   `pub type Stage6FieldExprPlan = FieldExprPlan` and ~20 similar aliases
   per stage. Are these aliases load-bearing for downstream code (e.g.,
   `jolt-equivalence` adapters) or can they be deleted along with S2?
   Spot-checking suggests they exist purely for readability and can go.
4. **Protocol relation IDs.** Should S2 parameterize runtime plan structs over
   a protocol relation enum, or should it introduce an opaque `RelationId` /
   `TypedPlanSymbol<RelationTag>`? Lean: protocol enum for readability when a
   stage matches on relation kind, opaque ID for generic runtime storage.
   Either is better than putting Jolt variants in the generic runtime crate.
5. **Checkpoint naming.** What user-facing partial-verification checkpoints do
   we want after S2.5? Lean: keep familiar names such as `ThroughStage5` only as
   aliases for typed checkpoint constants while shifting internal execution to
   proof slots / verifier steps.
6. **Value graph granularity.** Should S3 implement the scalar/point/vector
   value graph before any new polynomial op, or can it land as part of the same
   PR? Lean: same PR if small, but stop and split if the value typing gets
   noisy.
7. **S6 threshold.** Is "second protocol with bytecode-row encoding"
   the right trigger to actually do S6, or is there a different
   readability / audit reason to do it preemptively?

---

## Sequencing decision tree

```text
Resolve S2 protocol-ID shape.                    [before coding]
  Choose generic relation enum parameterization or opaque RelationId.

Land S2 first.                                   [unconditional]
  Re-baseline ceilings.
  Delete old common path; no compatibility re-export.

Land S2.5 next.                                  [unconditional]
  Make verifier.rs execute a typed verifier program.
  Stages become proof-slot labels / diagnostic scopes, not control flow.

Land S3 next.                                    [unconditional]
  Establish scalar/point/field-vector value graph.
  Tier B should drop ~140 LOC.

Decide on S4 vs Markos' next slice.              [coordinate]
  S4 is independent of Markos' work; either order is fine.

Pause before S5.                                 [explicit human review]
  Decide whether the relation-as-data shape is right for Jolt's
  current and likely-future relations. If yes, land S5. If not,
  stop here; the verifier is in a defensible state with Tier B at
  ~350 LOC of typed Rust.

S6 only on demand.                               [optional]
  Defer until there is a concrete second user or audit pressure.
```

---

This plan is the agent-side complement to GOAL.md's `Concrete Gates` and
`Iteration Algorithm`. Update both files together when a slice lands so
the goal definition and the implementation plan stay in sync.
