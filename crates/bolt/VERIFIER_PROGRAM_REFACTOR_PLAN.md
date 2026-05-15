# Verifier Program Refactor Plan

This is the implementation plan for the verifier-program refactor. It is a
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

The current stack has already moved beyond the original S1 framing:

- `bolt-verifier-runtime` exists as a standalone crate.
- `JoltRelationKind` is owned by the Jolt verifier layer instead of the generic
  runtime.
- Generated stages use typed relation enums instead of relation `&'static str`
  dispatch.
- The top-level verifier executes a typed verifier-program plan.
- Structured output-claim and polynomial-eval plans have started replacing
  handwritten verifier math.

The remaining plan is therefore **pass-first, runtime-second**. The primary
goal is not to grow a larger verifier runtime. The primary goal is to make
verifier construction a typed planning pipeline whose final Rust emission is
nearly mechanical:

```text
protocol / compute IR
        |
        v
verifier planning passes
        |
        v
typed CPU / Rust-plan IR
        |
        v
dumb Rust emission
        |
        v
small runtime crates for boring reusable mechanics
```

## Guiding principle

> Verifier semantics should be resolved before Rust emission. Runtime crates are
> allowed only for boring, reusable execution mechanics.

Concrete corollary: **`bolt-verifier-runtime` is not the architecture.** It is a
useful crate for verifier-program execution, typed value interpretation,
transcript/sumcheck/opening plumbing, and error structure. It should not become
the bucket where all common generated verifier Rust goes. If a fact is semantic,
it belongs in MLIR/codegen planning or in an explicit protocol boundary, not in a
runtime helper that reconstructs meaning from strings.

Numbered stages may remain as proof-layout slots and diagnostic scopes, but
they should not remain the verifier's core execution model. A generated
verifier should answer "what verifier program is executed?" rather than "how
many stage modules are hardcoded?"

## Design thesis: typed verifier IR, not better templates

The plan should not optimize for making today's Rust templates shorter. The
real target is a typed verifier-planning pipeline whose Rust rendering is
nearly a serialization detail:

1. **MLIR owns semantic facts.** If verifier execution depends on a fact, the
   fact should exist as a dialect op, typed attr, or typed plan field before the
   Rust emitter sees it.
2. **Planning passes own semantic assembly.** The gap between `cpu` and emitted
   Rust should be filled by explicit verifier-planning passes, not by ad hoc
   emitter parsing.
3. **The Rust emitter projects typed plans.** It should not reconstruct protocol
   meaning by parsing symbols, matching string prefixes, or emitting bespoke
   helper bodies that duplicate protocol math.
4. **Symbols are typed handles.** Examples in this document use `&'static str`
   where that keeps the prose readable, but implementation APIs should prefer
   `TypedPlanSymbol<Tag>`-style references. Plain strings are diagnostics and
   serialization names, not execution contracts.
5. **The runtime interprets typed value domains.** Scalars, points, field
   vectors, eval families, relation outputs, and bytecode encodings are
   different verifier value kinds. Do not force point transforms or vectors into
   `FieldExprKind` just because field expressions exist today.
6. **Relations should be scheduled dataflow, not a second DSL.** Prefer one
   typed verifier value graph over separate mini-interpreters for field
   expressions, point expressions, eval families, and relation terms. A relation
   check should usually be "the expected sumcheck output is scalar value X"
   where X is produced by the same typed graph as other verifier values.
7. **Protocol vocabulary stays at the protocol boundary.** Generic runtime code
   may carry opaque IDs or be generic over protocol-specific enums, but it
   should not define Jolt relation variants such as `Stage6BytecodeReadRaf`.
8. **Polynomial utilities come from `jolt-poly`.** If verifier math needs
   equality, less-than, identity, Lagrange, indexed equality, or projected-bit
   polynomial helpers, first use or add the right primitive in `jolt-poly`.
   Do not reimplement those helpers in generated verifier Rust or runtime code.
9. **No fallback execution paths.** When a typed path lands, delete the string
   path in the same slice. Rollback by reverting the slice, not by keeping a
   compatibility parser in the verifier.
10. **Auditability beats clever compaction.** A generated table that is longer
   but explains the verifier contract is better than a short table whose
   meaning only appears in the emitter.

## Verifier program model

The long-term emitted boundary should look like typed verifier-program data,
not a sequence of bespoke `verify_stageN` functions:

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
emission for one sumcheck driver. Planning passes should assemble this shape
before Rust emission. The runtime should not need to rediscover whether a driver
came from "Stage 3" or "Stage 6" except for diagnostics and proof-record layout.

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
| current stack | ~0 in generated crate | <=700 | bounded by cleanup gates | bounded by cleanup gates | runtime extraction, typed relations, top-level verifier program, output-claim plans |
| next pass-first slice | narrow runtime | <=700 | stable or lower | stable or lower | classify runtime/codegen/poly ownership and add planning-pass seams |
| value graph | narrow runtime | <=638 | ~6,300-6,500 | ~6,500 | typed scalar/point/vector/eval-family plans |
| eval families | narrow runtime | ~350 | ~6,500 | ~6,500 | typed indexed-eval addressing |
| relation plans | narrow runtime | ~290 | ~6,700 | ~6,700 | relations as typed plans (Tier C may grow temporarily) |
| bytecode encoding | narrow runtime | ~50 | ~6,800 | ~6,800 | optional, only if audit pressure justifies it |

Scoreboard numbers are planning targets for individual slices. The
non-regression contracts above are stricter: the completed stack must not end
larger, less readable, slower without approval, or semantically weaker than the
post-S1 baseline.

---

## S2: Runtime extraction status and boundary correction

**Status.** The current stack has already performed the main extraction:
`bolt-verifier-runtime` is a standalone workspace crate, generated
`jolt-verifier` depends on it, and Tier A is no longer emitted as
`stages/common.rs`.

**Revised goal.** Keep that progress, but treat the crate as provisional and
narrow. The runtime is useful for boring verifier-program mechanics, but it
must not become the place where all shared generated verifier Rust accumulates.
The remaining S2 work is classification and boundary correction:

- keep generic verifier-program sequencing, value interpretation,
  transcript/sumcheck/opening plumbing, and structural error types in
  `bolt-verifier-runtime`;
- keep `JoltRelationKind`, bytecode contracts, RAM/register/lookup semantics,
  point normalizations tied to Jolt proof layout, and other protocol math in the
  Jolt layer or in typed MLIR/codegen plans;
- move reusable polynomial helpers to `jolt-poly` instead of implementing them
  in the runtime;
- add CPU-to-Rust planning seams so future semantics move into passes instead
  of runtime helpers or Rust emitter logic.

### Locked decisions already reflected by the current stack

- `bolt-verifier-runtime` is a standalone workspace crate at
  `crates/bolt-verifier-runtime/`. It is not a sub-crate of `bolt` and is not
  re-exported through `bolt`.
- Relation IDs use generic enum parameterization, not opaque string-backed
  symbols. Relations are used for verifier dispatch, so a closed protocol enum
  gives exhaustive-match pressure and keeps strings out of execution contracts.
- The generic runtime owns the relation trait and generic plan structs. The
  Jolt layer owns the closed relation enum:

  ```rust
  pub trait ProtocolRelation: Copy + Eq + core::fmt::Debug + 'static {}
  impl<T: Copy + Eq + core::fmt::Debug + 'static> ProtocolRelation for T {}

  pub struct SumcheckClaimPlan<R: ProtocolRelation> {
      pub relation: Option<R>,
      // ...
  }

  pub struct SumcheckDriverPlan<R: ProtocolRelation> {
      pub relation: Option<R>,
      // ...
  }

  pub struct SumcheckInstanceResultPlan<R: ProtocolRelation> {
      pub relation: R,
      // ...
  }
  ```

  `JoltRelationKind` lives in the generated Jolt verifier's
  `jolt_relations` module, and stage files alias it as
  `StageNRelationKind`. `TypedPlanSymbol<Tag>` remains useful for value,
  opening, claim, and diagnostic symbols, but not for relation dispatch.
- `SourceStage` is not part of the generic runtime. Stage 8 gets a local
  `Stage8SourceStage` enum until the later PCS-opening plan model removes the
  source-stage special case entirely.
- Per-stage type aliases are not load-bearing. Keep or delete them based on
  readability after the import cutover, but do not preserve them as API
  compatibility shims.

### Runtime boundary classification

Keep in `bolt-verifier-runtime` only when the code is protocol-neutral
execution machinery:

- top-level verifier-program sequencing;
- typed plan records for transcript events, sumcheck drivers, opening claims,
  opening batches, and PCS checks;
- `ValueStore`-style typed value interpretation once value domains are
  explicit;
- generic sumcheck verification loops;
- opening equality checks and transcript append mechanics;
- structural error reporting that can be parameterized by protocol IDs.

Do not add to `bolt-verifier-runtime` when the code depends on Jolt protocol
meaning:

- `JoltRelationKind` variants;
- bytecode row contracts and lookup-table semantics;
- Stage 2 RAM/product/instruction-lookup algebra;
- Stage 5 instruction read-RAF algebra;
- Stage 6/7 bytecode, hamming, RA, inc-reduction, and point-normalization
  semantics;
- string-prefix eval-family lookup as a permanent API.

If a helper is polynomial math rather than verifier mechanics, prefer
`jolt-poly`. Current candidates include less-than MLE evaluation, identity
polynomial evaluation, Lagrange basis evaluation, indexed Boolean equality, and
projected/operand polynomial evaluation.

### Concrete plumbing already done or retained

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
8. Move `RelationKind` out of Tier A and into the Jolt layer as
   `JoltRelationKind`. Genericize relation-bearing runtime structs over
   `R: ProtocolRelation`.
9. Move `SourceStage` out of Tier A. Stage 8 defines a local source-stage enum
   until S2.5/S4/S5 replace that source lookup with typed verifier-program /
   opening-batch data.

### Follow-up cleanup created by this revision

1. Rename the old "runtime extraction" milestone in future PR descriptions to
   "runtime boundary narrowing" or "typed planning pipeline".
2. Audit `bolt-verifier-runtime` for APIs that branch on strings or encode Jolt
   protocol facts. Each such API needs one of three dispositions: typed planning
   pass, Jolt verifier boundary, or `jolt-poly`.
3. Replace runtime-local polynomial helpers with `jolt-poly` calls before adding
   more runtime surface.
4. Treat duplicate plan container variants in the runtime as transitional
   emitter-shape leakage. Collapse them as the CPU-to-Rust plan layer becomes
   explicit.
5. Do not expand the runtime to compensate for missing MLIR/codegen passes.

### Test/gate updates

- Keep `verifier_cleanup.rs` focused on the generated verifier surface, but add
  or retain a separate runtime ceiling so `bolt-verifier-runtime` cannot grow
  quietly.
- Keep `bolt-verifier-runtime` as the only intentional verifier runtime
  dependency. The verifier remains forbidden from depending on `jolt-prover`,
  `jolt-kernels`, `jolt-core`, `jolt-equivalence`, and tracing/profiling crates.
- Add targeted checks for runtime APIs that branch on string contents. Those
  sites should ratchet down as S2.75/S3/S4 land.
- Add `jolt-poly` reuse checks when a helper is moved out of the runtime or
  Jolt verifier core.

### Remaining risks and complications

- **Runtime surface creep.** The extraction solved generated-code duplication,
  but the crate can still become too bespoke if it accepts every helper the
  emitter wants. Treat every new runtime API as a boundary decision.
- **String-keyed transitional APIs.** `ValueStore`, eval-prefix helpers,
  point-order strings, and proof/source-stage strings are useful during the
  transition, but they should be removed or typed as planning passes land.
- **Duplicate plan containers.** Multiple near-duplicate stage plan structs are
  signs that emitter shape leaked into the runtime. Collapse them once typed
  planning output is explicit.
- **Polynomial helper ownership.** Less-than, identity, Lagrange, indexed
  equality, and projected-bit polynomial helpers belong in `jolt-poly`, not in
  `bolt-verifier-runtime` or generated verifier Rust.
- **Equivalence adapters.** Any plan-shape cleanup still has cross-crate blast
  radius in `jolt-equivalence`; keep full-cutover discipline.

### Acceptance criteria

```text
cargo check -p bolt -p bolt-verifier-runtime -p jolt-verifier
            -p jolt-prover -p jolt-equivalence  --quiet
cargo nextest run -p bolt --test verifier_cleanup --no-capture
cargo nextest run -p bolt --test commitment_ir --cargo-quiet
cargo nextest run -p jolt-equivalence --test generated_role_crates --cargo-quiet
cargo nextest run -p jolt-equivalence --test bolt_commitment --no-capture
```

All green. The metrics output shows Tier A remains absent from the generated
verifier crate, runtime LOC is explicitly bounded, and runtime string-dispatch
sites are reported.

### Rollback

The extraction should not be rolled back unless the runtime boundary itself
turns out wrong. If the boundary audit overreaches, revert the specific API
deletion or helper movement rather than reintroducing generated `common.rs`.
No data or proof-format changes are involved.

### Estimated wall-clock

Extraction has already happened. The remaining boundary audit is one focused
agent slice, mostly classification, API deletion, `jolt-poly` reuse, and test
gate updates.

---

## S2.5: Verifier-program executor status and next tightening

**Status.** The current stack has introduced the typed top-level
`VerifierProgramPlan` and routes verifier execution through it. Keep this shape.

**Revised goal.** Tighten the source of truth. The top-level plan should be
assembled by verifier-planning passes instead of stringly artifact glue or
emitter-side stage scans. This slice remains architectural: it changes how the
verifier is scheduled, not proof serialization.

This does not require fully deleting stage modules immediately. Stage modules
may still own proof-slot conversion and local plan constants at first. The
important cutover is that the top-level verifier sees a program of typed steps,
and `stageN` names become proof-slot labels rather than the verifier's control
flow.

### Concrete plumbing

1. Add typed top-level verifier-program types:

   ```rust
   pub enum ProofSlot {
       Commitments,
       Stage1Outer,
       Stage2,
       Stage3,
       Stage4,
       Stage5,
       Stage6,
       Stage7,
       Evaluation,
   }

   pub enum VerifierStepPlan {
       ReceiveCommitments { slot: ProofSlot },
       VerifyStage { slot: ProofSlot },
       VerifyPcsOpening { slot: ProofSlot },
   }

   pub enum VerifierCheckpoint {
       AfterStage5,
       AfterStage6,
       AfterStage7,
       AfterEvaluation,
   }

   pub enum EvaluationPolicy {
       Skip,
       VerifyIfPresent,
       Required,
   }

   pub struct VerifierTarget {
       pub checkpoint: VerifierCheckpoint,
       pub evaluation: EvaluationPolicy,
   }
   ```

   `ProofSlot` is the protocol/proof-layout identity. `VerifierStepPlan` is the
   execution identity. They should not be conflated.
2. Represent commitment receipt, transcript absorbs/squeezes, sumcheck driver
   verification, opening-batch emission, and PCS opening verification as program
   steps.
3. Keep `JoltProof` serialization unchanged:
   `commitments`, `stage1_outer`, `stage2`, ..., `stage7`, `evaluation`.
   The verifier program is execution metadata, not proof data.
4. Replace the hardcoded stage call chain in `verifier.rs` with a loop over the
   generated program. Existing `verify_stageN_with_program` functions remain as
   slot executors during this slice, but there must be one execution path, not
   old/manual and new/program paths side by side.
5. Replace target APIs such as `through_stage5`, `through_stage6`, and
   `through_stage7` with named checkpoints or proof-slot targets. The old names
   can survive as user-facing constructors only if they are constants for the
   new checkpoint values, not separate branches.
6. Make stage artifacts addressable through an internal `ArtifactStore` keyed by
   typed proof slots. Public `JoltVerificationArtifacts` keeps its current field
   shape for this slice and is materialized from the store at the end so
   downstream callers are not forced to migrate yet.
7. Stage 8 becomes `VerifyPcsOpening { check }` plus supporting opening-batch
   plans. It should not be special because it happens to be the eighth module.
8. Keep stage-local error quality. Stage failures still map to
   `JoltVerifyError::StageN`; only structural verifier-program failures use a
   new top-level program error.

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
- Existing before/after verifier-time perf baselines remain attached to this
  architectural change; future tightening slices should compare against them.

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
artifact typing, not new protocol math. The current stack has already completed
the main shape; remaining work belongs in S2.75 planning-pass tightening.

---

## S2.75: Add CPU-to-Rust verifier planning passes

**Goal.** Insert an explicit planning layer between `cpu` IR and emitted Rust.
This is the main course correction from Markos' feedback: do not make the Rust
emitter or `bolt-verifier-runtime` responsible for discovering verifier
semantics.

The current `cpu` layer still carries protocol-plan material such as stages,
proof slots, relation symbols, opening sources, point order, output claims, and
PCS policies. That is acceptable as a transitional representation, but the
Rust emitter should not parse those fields into execution meaning. It should
consume a typed Rust-plan / verifier-plan representation produced by passes.

### Proposed passes

1. `resolve-cpu-program-steps`: materialize ordered program steps for
   transcript events, sumcheck drivers, opening-batch emission, and PCS checks.
2. `annotate-jolt-relations`: attach typed relation metadata before emission,
   including relation kind, proof slot, output-evaluator strategy, external
   verifier data requirements, and kernel ABI where relevant.
3. `plan-verifier-sumchecks`: lower verifier sumcheck ops plus batches and
   instance results into `SumcheckDriverPlan` rows.
4. `plan-field-and-output-claims`: canonicalize field expressions, structured
   polynomial evals, and output-claim equations into a typed scalar/point/value
   graph.
5. `plan-opening-flow`: resolve opening inputs, claims, equalities, batches,
   point/eval sources, arities, and ordering.
6. `plan-point-normalization`: replace `point_order` strings and stage-specific
   point transforms with typed point-transform plans.
7. `validate-rust-target-plan`: reject unsupported field/PCS/transcript
   combinations, unresolved symbols, role-specific forbidden ops, and
   relation/runtime-boundary violations before the emitter runs.

### Acceptance criteria

- New verifier semantics are added as planning-pass output, not as Rust emitter
  parsing or new runtime string dispatch.
- Relation strings are converted to typed relation IDs before Rust token
  emission.
- Output-claim and point-normalization planning can be inspected without
  reading emitted Rust templates.
- The Rust emitter's responsibility is mostly formatting typed const data and
  importing the right runtime/Jolt-boundary APIs.

---

## S3: Typed verifier value graph + polynomial primitives

**Goal.** Introduce the typed verifier value graph and use it for the first
relation-output checks. This is broader than Stage 6/7 helper cleanup: stages
2-7 all contain handwritten expected-output math. The first conversions should
be Stage 3 and Stage 4 because they exercise the value-graph path without RAM
external data, bytecode rows, lookup-table families, or univariate-skip proof
shape.

The important design point is that this is not "add more variants to
`FieldExprKind`"; it is a typed graph over scalar, point, field-vector, and
eval-family values.

### Current status

The scalar/point source registry has been moved out of
`verifier_output_claims` into `verifier_values`. This is only the foundation:
it gives scalar and point domains an explicit home. `VerifierStagePlan` now
carries enough scalar/point source data for Stage 3-7 output-claim validation
and verifier opening-flow validation to consume plan-derived value sources
instead of rebuilding point sources in each stage. Verifier field-flow
validation also consumes plan-derived scalar sources; the remaining CPU scalar
source builders are explicitly scoped to prover/role-neutral validation.
Field-vector values, eval-family values, and relation-output execution are not
yet lowered through a single typed graph.

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

`bolt-verifier-runtime` may retain typed value storage and interpreters, but
the planning passes should decide which values exist and how they are connected.
The runtime should evaluate typed plan rows, not infer semantics from symbol
names:

```rust
pub enum VerifierValueKind {
    Scalar,
    Point,
    FieldVector,
    EvalFamily,
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
interpreter should call `jolt-poly` for equality, less-than, identity, Lagrange,
and projected-bit polynomial helpers. Point transforms must be evaluated by the
point interpreter and vector producers by the vector interpreter. This keeps the
type boundary honest and prevents the runtime from becoming a bag of ad hoc
helper functions.

### Locked conversion order

1. Build the value-graph foundation: scalar, point, field-vector, and
   eval-family storage and plan rows.
2. Convert Stage 3 expected-output checks first:
   Spartan shift, instruction input, and registers claim reduction.
3. Convert Stage 4 next:
   registers read-write, RAM val check, `lt` evaluation, suffix points, and
   the Stage 4 register-read-write point normalization.
4. Convert the small Stage 5 relations next:
   RAM RA claim reduction and registers val evaluation.
5. Convert Stage 2 in two parts:
   RAM read-write / product remainder / instruction lookup first, then RAM RAF,
   RAM output, and univariate-skip-specific logic later.
6. Convert Stage 6/7 family relations after eval-family support:
   booleanity, hamming booleanity, RAM RA virtual, instruction RA virtual, inc
   reduction, and Stage 7 hamming-weight reduction.
7. Defer S5 instruction read-RAF and S6 bytecode read-RAF until lookup-table,
   bytecode-entry, and external-data domains are designed.

### Deliberate deferrals

Do not start S3 by designing generic bytecode-row algebra, RAM sparse
evaluators, lookup-table MLE plans, or univariate-skip verifier replacement.
Those are real verifier facts, but starting there would likely create a too
general mini-language before the scalar/point/vector graph has been validated
on simpler relations.

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
- **Stage emitter updates.** Stage 3/4 should be the first emitters to lower
  relation-output math to value-graph rows. Stage 6/7 remain later consumers
  once eval families and bytecode/external-data boundaries are clear.
- **Validator updates.** Each new `compute::*` op needs an entry in the
  Bolt validator so malformed plans (wrong arity, wrong operand types) are
  rejected at compile time.

### Acceptance criteria

`muldiv` e2e test passes in both `--features host` and `--features host,zk`
(this is the workspace's primary correctness check). Stage 3 and Stage 4
expected-output helper bodies are replaced by typed value-graph plan rows. New
metrics in `verifier_cleanup.rs`: `compute_poly_op_call_sites`,
`compute_point_op_call_sites`, `value_graph_relation_outputs`, and
`handwritten_expected_output_functions` reported. Gate only the Stage 3/4
cutover in this slice; do not claim Stage 6/7 relation math is solved yet.

### Rollback

Each new MLIR op is independently revertible. Worst case we keep the
`poly_mle` op (which is unambiguous) and revert the `normalize_*_point`
lowering if the dataflow ordering turns out to be subtle.

### Estimated wall-clock

Two agent sessions, ~120-180 minutes total. Most of the time is in the
planning/emitter changes for the first Stage 3/4 conversions and validator
updates.

---

## S4: Typed indexed-eval addressing

**Goal.** Keep eval-family reconstruction out of verifier runtime/Tier B and
finish moving indexed-eval family facts into typed verifier-plan data.

### Current status

The original Tier B problem has been removed on the current stack. Generated
Jolt relation code and `bolt-verifier-runtime` no longer call
`indexed_evals_by_prefix*`, and `verifier_cleanup` now gates both generated
relation code and the runtime against reintroducing indexed eval-prefix APIs.

Stage 5 instruction read-RAF and Stage 6 bytecode read-RAF now carry explicit
Bolt-side `IndexedEvalFamilyPlan` rows in the verifier-stage plan. The relevant
relation/output-claim planners consume those rows instead of re-deriving the
family from raw evals at the point of Rust token emission.

Stage 5/6 family membership now originates as first-class
`piop.sumcheck_eval_family`, `compute.sumcheck_eval_family`, and
`cpu.sumcheck_eval_family` rows. The CPU-to-verifier planning layer parses those
rows directly; it no longer infers family membership from symbol spelling,
oracle prefixes, or raw eval ordering.

This is still not the final architecture. The current family op is typed
planning metadata with explicit `evals` membership, not yet a first-class
field-vector value in the verifier value graph. S3 should decide how field
vectors/eval families flow as typed values; S4's job is to make the existing
Stage 5/6 family facts explicit and non-prefix-derived before Rust token
emission.

### Dialect changes

Add a sibling no-result eval-family op in the PIOP, compute, and CPU dialects:

```mlir
compute.sumcheck_eval_family {
  sym_name = "stage6.bytecode_read_raf.eval.BytecodeRa"
  source = "stage6.sumcheck"
  oracle_family = "BytecodeRa"
  count = 3
  evals = [@stage6.bytecode_read_raf.eval.BytecodeRa_0,
           @stage6.bytecode_read_raf.eval.BytecodeRa_1,
           @stage6.bytecode_read_raf.eval.BytecodeRa_2]
}
```

The existing scalar `sumcheck_eval` ops remain the source of the serialized
named evals. The family row is an explicit membership declaration consumed by
the verifier-plan layer. A later S3/S5 value-graph slice can promote this shape
to a typed field-vector value if that makes the Rust emission smaller and
clearer.

### Runtime additions

No runtime API is added for the metadata-row phase. `bolt-verifier-runtime`
continues to expose only the boring reusable mechanics, while the Jolt verifier
plan carries `IndexedEvalFamilyPlan` rows. Runtime `FieldVector`/`ValueStore`
support belongs in S3 only if the typed verifier value graph needs it.

### Tier B impact

Stage 6/7 relation evaluators should continue to receive typed eval-family
plans or typed field-vector values, never reconstruct families from prefixes.
Remaining reductions are expected on the Bolt emitter/planning side, not from
deleting more Tier B prefix code.

### Blockers and complications

- **Proof format.** The proof's `evals` field is already
  `Vec<StageNamedEval<F>>` with explicit `name` strings (verified in
  `verifier_common.rs.template:393-397`). S4 changes how the *verifier
  consumes* named evals, not how they are serialized. No back-compat
  break. *This was the biggest risk; it is not real.*
- **Prover/verifier symmetry.** The prover-side emitter must annotate the
  same eval block as a family. If this is a separate emitter, both must be
  updated together.
- **Typed source of truth.** `IndexedEvalFamilyPlan` is now parsed from
  CPU/MLIR eval-family rows for Stage 5/6. Remaining risk is scope, not
  ambiguity: Stage 7 and future relation plans need the same rule when they
  start consuming family-shaped data.
- **Testing.** Add a `verifier_cleanup` gate
  `RELATION_INDEXED_EVAL_PREFIX_SITES_CEILING = 0` that fires if any
  generated stage source still calls `indexed_evals_by_prefix*` or exposes
  eval-prefix fields. This gate is now present.

### Acceptance criteria

`muldiv` passes in both modes. Zero `indexed_evals_by_prefix*` call sites
in the generated verifier and runtime. Indexed eval-family membership is
represented as typed CPU/verifier-plan data before Rust token emission.

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

Each slice that touches generated stage outputs (S2 boundary cleanup, S2.75,
S3, S4, S5) will require equivalence-side adapter updates in
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

The current stack has corrected the most important protocol-ID boundary by
moving Jolt relation variants out of `bolt-verifier-runtime`. The next boundary
work is subtler: do not let the runtime or the Rust emitter become a second
home for Jolt semantics under generic names. Any branch on relation kind,
point-order spelling, eval-name prefix, source-stage string, or proof-slot
string must either become typed planning-pass output or stay visibly
quarantined in `crates/bolt/src/protocols/jolt/` / `jolt_relations.rs`.

S2.75, S3, and S4 are primarily compute/planning extensions. They should add a
typed verifier value graph (`Scalar`, `Point`, `FieldVector`, eval families)
rather than a larger bag of scalar-only field-expression variants or runtime
string dispatch.

S5 introduces `compute::relation` as generic relation metadata over the value
graph, not a Jolt-specific relation language. Closed relation enums live at the
protocol/stage boundary; the generic runtime is parameterized over the protocol
relation enum when it needs equality or dispatch. Opaque typed symbols are for
generic storage paths that do not branch on relation kind.

S6 is explicitly Jolt-specific. If pursued, it goes under
`crates/bolt/src/protocols/jolt/`, not in `bolt-verifier-runtime`.

### Trust boundary and audit surface evolution

The audit surface contracts as we move through the slices:

```text
post-S1:  Tier A (1,265 LOC)  + Tier B (638 LOC)  = 1,903 LOC audited
current:  narrow runtime    + Tier B (<=700 LOC) = runtime reviewed once,
                                                    Jolt math quarantined
post-S3:  narrow runtime    + Tier B (~500 LOC)  = value graph absorbs
                                                    simpler relation math
post-S4:  narrow runtime    + Tier B (~350 LOC)  = eval-family strings gone
post-S5:  narrow runtime    + Tier B (~290 LOC)  = relations become plans
post-S6:  narrow runtime    + Tier B (~50 LOC)   = optional bytecode data
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

Before S2.75, S3, or S5 lands, record a local verifier-time baseline from the
SHA2-chain perf oracle and rerun it after the slice. If a verifier-time change
is noisy, rerun before calling it real. If a repeatable regression remains,
either fix it or explicitly document why the readability/security gain is worth
the cost. Do not loosen perf thresholds to make a refactor pass. The existing
S2/S2.5 progress should keep its captured baselines as reference data.

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

- If the emitter needs to classify a verifier fact, add or extend a planning
  pass instead.
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

## Locked sequencing decisions

The investigation pass resolved the implementation choices that should guide the
next autonomous implementation slices:

1. Keep `bolt-verifier-runtime`, but define it narrowly. It is a verifier
   interpreter crate, not a generic codegen dumping ground.
2. Runtime relation-bearing plan structs remain generic over
   `R: ProtocolRelation`; Jolt relation variants live in `JoltRelationKind`.
3. Add CPU-to-Rust planning passes before adding more runtime helper surface.
4. Stage 8 source-stage handling is local to Stage 8 until the PCS/opening
   model is made fully typed.
5. Legacy names such as `ThroughStage5` may remain as user-facing constructors,
   but internally they resolve to `VerifierTarget { checkpoint, evaluation }`.
6. S2.75, S3, and S5 need before/after verifier-time baselines because they add
   or change interpreter/planning execution.
7. S3 starts with the value-graph foundation plus Stage 3/4 conversions. It
   does not start with bytecode, RAM sparse evaluators, lookup-table MLEs, or
   univariate-skip verifier replacement.
8. S6 remains optional and should be deferred unless there is a second protocol
   user or concrete audit pressure for typed bytecode-row encoding.

---

## Sequencing decision tree

```text
Current stack.                                   [already in progress]
  Runtime extraction, typed relation IDs, top-level verifier program,
  output-claim plans, cleanup gates.

Do S2 boundary audit next.                        [unconditional]
  Narrow runtime APIs.
  Move polynomial helpers to jolt-poly.
  Classify runtime vs Jolt verifier core vs planning-pass ownership.

Add S2.75 planning seams next.                    [unconditional]
  Make CPU-to-Rust verifier planning explicit.
  Prevent new semantics from landing in emitters or runtime string dispatch.

Land S3 after planning seams.                     [unconditional]
  Establish scalar/point/field-vector/eval-family value graph.
  Convert Stage 3 then Stage 4 relation-output checks first.
  Defer bytecode, RAM sparse, lookup-table, and univariate-skip special cases.

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
