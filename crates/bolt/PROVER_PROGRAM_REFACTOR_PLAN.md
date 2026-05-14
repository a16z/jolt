# Prover Program Refactor Plan

This is the prover-side companion to `crates/bolt/VERIFIER_PROGRAM_REFACTOR_PLAN.md`.
It asks which parts of the readable emitted-verifier work should also apply to
generated prover Rust, and where the prover needs a different shape.

The short answer is: the design philosophy transfers almost completely, but the
mechanics do not transfer one-for-one. The verifier refactor is now explicitly
pass-first and runtime-second: typed planning passes should resolve semantics
before Rust emission, and runtime crates should contain only boring reusable
mechanics. The prover should mirror that philosophy, but not by copying the
verifier runtime shape.

The prover already delegates much of its execution to `jolt-kernels`, so its
main readability problem is not hand-written helper logic in generated files.
It is large, expanded, stage-local, and partially stringly plan data. Markos'
assessment matches this: prover codegen is currently easier because it mostly
calls big kernels; verifier remains the priority because CPU-to-Rust verifier
emission still does too much semantic work.

The desired prover architecture is therefore:

```text
MLIR / Bolt IR owns protocol facts
        |
        v
prover planning passes
        |
        v
typed CPU / Rust-plan IR
        |
        v
thin generated Rust const data
        |
        v
efficient prover kernels and executors
```

The generated prover should remain an auditable declaration of what the prover
does, not a pile of reconstructed protocol logic and not an opaque call into a
black box.

## Current observations

The verifier refactor plan targets generated verifier stages that historically
embedded a lot of semantic checking logic. By contrast, prover stages such as
stage 5 already have the right broad shape: a static program plan plus a thin
entry point that calls a kernel executor.

The prover-side pain is concentrated elsewhere:

- Some generated prover stages are still very large. A recent scan put
  `crates/jolt-prover/src/stages/stage6.rs` at roughly 2,525 LOC and
  `crates/jolt-prover/src/stages/stage7.rs` at roughly 1,940 LOC.
- Much of the bulk appears to come from repeated opening inputs, evals, claim
  rows, batch rows, and per-stage plan tables.
- Prover plan types are stage-local and kernel-owned today, for example through
  `jolt_kernels::stageN::{...}`.
- Several fields still look more like serialized strings than typed execution
  contracts.
- The prover has an intentionally different import boundary than the verifier:
  the verifier should avoid depending on prover and kernel crates, while the
  prover is expected to call `jolt-kernels`.

This means the prover refactor should optimize for compact typed plans over
efficient kernels, not for replacing prover kernels with a generic verifier-like
interpreter.

This also means prover work should not compete with verifier-side CPU-to-Rust
planning. The best near-term prover contribution is to avoid divergence: share
typed plan vocabulary where it is genuinely common, move polynomial helpers to
`jolt-poly`, and keep prover generated Rust from growing new stringly execution
contracts while verifier planning is being cleaned up.

## Guiding principle

> The prover generated Rust should be readable protocol data over efficient
> kernels. Planning passes should own semantic assembly; Rust should only
> project typed prover plans and call the execution layer.

Concrete corollaries:

1. **MLIR owns semantic facts.** If prover execution depends on a relation,
   opening family, batching rule, transcript event, or claim dependency, that
   fact should exist as typed IR before the Rust emitter sees it.
2. **Planning passes own semantic assembly.** The gap between `cpu` and emitted
   Rust should be filled by explicit prover-planning passes when the current
   direct emission path starts carrying semantic decisions.
3. **Generated Rust is a projection.** The emitter should not reconstruct
   prover semantics by parsing symbol names, matching prefixes, or expanding
   regular families into hundreds of nearly identical rows when a typed family
   plan would say the same thing.
4. **Prover kernels stay hot-path aware.** Do not move prover relation logic
   into a generic interpreter merely to shrink emitted Rust. Any abstraction
   around prover execution must preserve the current performance model unless
   benchmark data justifies a change.
5. **Typed handles beat strings.** Strings are acceptable for diagnostics and
   serialization names. Execution contracts should use enums, typed symbols,
   typed family IDs, or structured plan fields.
6. **Polynomial utilities come from `jolt-poly`.** If prover kernels or emitted
   plans need equality, less-than, identity, Lagrange, indexed equality, or
   projected-bit polynomial helpers, use or add the right primitive in
   `jolt-poly` rather than duplicating it in `jolt-kernels` or generated Rust.
7. **Compactness must stay auditable.** A range or family plan is good when it
   exposes structure such as "all `InstructionRa` openings." It is bad if it
   hides protocol meaning behind clever emitter-only conventions.
8. **No compatibility shadows.** When replacing a string path or expanded table
   with a typed plan, perform the full cutover for that surface rather than
   keeping old and new forms in parallel.

## What transfers from the verifier refactor

### MLIR-owned semantics

This applies fully. The same reason verifier math should not be reconstructed
inside Rust templates also applies to prover execution metadata. The prover
should receive typed plan rows that are already semantically meaningful.

Examples:

- relation and kernel IDs
- transcript absorb and squeeze events
- sumcheck driver schedules
- opening inputs and opening claim groups
- eval families and indexed eval addressing
- batching structure
- proof-record slots

### Typed program plans

This applies strongly, but the plan should be prover-shaped. A verifier program
asks how to check proof data. A prover program asks how to generate proof data
from witness-backed executors and kernels.

The target should look more like:

```rust
pub struct ProverProgramPlan {
    pub params: ProverParams,
    pub steps: &'static [ProverStepPlan],
    pub transcript_events: &'static [TranscriptEventPlan],
    pub sumchecks: &'static [ProverSumcheckDriverPlan],
    pub opening_inputs: &'static [OpeningInputPlan],
    pub opening_families: &'static [OpeningFamilyPlan],
    pub opening_batches: &'static [OpeningBatchPlan],
    pub kernels: &'static [KernelInvocationPlan],
}

pub enum ProverStepPlan {
    ExecuteKernel { kernel: KernelInvocationId },
    AbsorbTranscript { event: TranscriptEventId },
    SqueezeTranscript { output: ChallengeId },
    ProveSumcheck { driver: ProverSumcheckDriverId },
    EmitOpeningClaims { batch: OpeningBatchId },
}
```

This is a sketch, not a locked API. The important part is that the generated
file declares a typed execution plan rather than emitting stage-shaped Rust
logic or enormous flat arrays when the protocol structure is regular.

### Runtime extraction

This applies only after the plan vocabulary is clearer, and not through
`bolt-verifier-runtime`.

The verifier plan has shifted away from "create a runtime first" and toward
"create typed planning passes first." The prover should follow the same rule.
A prover runtime crate is useful only if it owns boring execution mechanics
over `jolt-kernels`; it should not become a second place to encode prover
semantics.

The prover should either use a dedicated prover runtime crate or a shared plan
crate that contains only concepts genuinely common to prover and verifier. The
verifier runtime has verifier-specific concerns such as expected-output checks,
accept-reject behavior, and PCS verification boundaries. Those should not leak
into prover APIs.

Candidate crate boundaries:

- `bolt-plan-runtime`: shared typed plan primitives, IDs, value domains, family
  plans, transcript event plans, and opening/batch descriptors.
- `bolt-prover-runtime`: prover-specific execution plan interpretation over
  `jolt-kernels`.
- `bolt-verifier-runtime`: verifier-specific proof checking over typed verifier
  plans.

The concrete crate split should be decided only after identifying the repeated
plan structs and duplicated concepts in generated prover and verifier surfaces.
Do not create `bolt-prover-runtime` just to move code out of generated files.
First add metrics and planning seams that prove the shared surface is real.

### Readability and LOC gates

This applies directly, with prover-specific metrics.

The verifier cleanup tests should inspire a prover cleanup test that tracks:

- generated prover stage LOC
- largest static table sizes
- number of emitted stringly execution fields
- number of stage-local plan structs or duplicate aliases
- helper function count inside generated prover stages
- generated prover surface LOC versus kernel/runtime LOC

The goal is not to minimize total workspace LOC blindly. The goal is to move
semantic bulk into typed, reusable, audited plan/runtime concepts and reduce
generated-stage noise.

### Subjective readability review

This applies, but the review packet should include both the generated prover
stage and the relevant kernel ABI. For the prover, semantics are intentionally
split between plan data and execution kernels. A fair readability review should
ask whether a reviewer can understand what proof work is scheduled without
reading Bolt lowering code.

## What does not transfer directly

### The verifier import boundary

The verifier should not depend on `jolt-prover`, `jolt-kernels`, or `jolt-core`
implementation details. The prover has the opposite expectation: it is the
proof-producing side and should call optimized kernel executors.

So the prover contract should not say "no `jolt-kernels`." It should say:

- generated prover Rust may depend on prover runtime and kernel crates;
- generated prover Rust should not reconstruct kernel semantics locally;
- kernel ABI rows should be typed, grouped, and auditable.

### Verifier output-claim checks

Verifier-side output-claim plans check that proof data is consistent with
expected scalar values. The prover analogue is not to verify those claims, but
to declare produced claims, openings, batches, and transcript dependencies in a
typed way.

The shared abstraction is the vocabulary of claims, openings, eval families, and
relations. The execution meaning is side-specific.

### Generic interpretation of hot prover math

The verifier can afford more generic interpretation because its workload is
small relative to proving. The prover cannot assume that. Any move from
specialized kernels to generic interpreters requires explicit profiling and a
clear reason beyond generated Rust readability.

## Proposed slices

### P0: Mirror verifier cleanup only where it is already shared

Do this opportunistically while verifier remains the priority.

- Move duplicated polynomial helpers in verifier runtime, `jolt_relations`, and
  `jolt-kernels` toward `jolt-poly`.
- Keep prover generated Rust compatible with the typed plan vocabulary that the
  verifier planning passes introduce.
- Avoid adding a prover runtime crate until repeated plan/execution mechanics
  are visible and measured.

This is the safest prover-side work while Markos is changing higher dialects and
the verifier CPU-to-Rust planning path is still moving.

### P1: Add prover generated-surface cleanup metrics

Add a prover-side cleanup test parallel to the verifier cleanup test. It should
initially report and gate the current generated surface without changing
execution.

Suggested metrics:

- per-stage generated prover LOC
- stage 6 and stage 7 static table sizes
- count of stringly plan fields that affect execution
- count of emitted helper functions in generated stages
- count of duplicated stage-local plan structs

The purpose of P1 is visibility. It should make the prover readability problem
concrete before changing plan APIs.

### P2: Identify and type the largest repeated plan families

Inspect the largest generated tables in prover stages 6 and 7 and classify
which rows are regular families.

Likely candidates:

- `InstructionRa_*` opening groups
- lookup-table flag families
- repeated eval plans
- opening claim groups
- opening batch membership
- point slice and concat patterns

Replace regular expanded rows with explicit family or range plans only when the
family name, bounds, and derivation are visible in generated Rust.

### P3: Introduce typed prover plan IDs and enums

Remove string-backed execution contracts from prover plan rows where those
strings affect behavior. Prefer closed enums, typed IDs, and typed symbols.

This should cover fields such as:

- kernel kind and ABI
- step kind
- claim kind
- field expression kind
- relation or source kind
- opening family kind

When a typed path lands, delete the string path in the same slice.

This should be implemented through prover planning data where possible, not by
teaching the Rust emitter more string-to-token mappings.

### P4: Add prover CPU-to-Rust planning seams

After P1-P3 identify the repeated structures, add explicit prover planning
passes between CPU IR and emitted Rust. These passes will differ from the
verifier passes because prover execution is kernel-backed and witness-backed.

Likely passes:

- `plan-prover-kernel-invocations`: typed kernel IDs, ABI rows, executor data
  requirements, and output artifact contracts.
- `plan-prover-sumchecks`: typed driver/batch/claim/opening plans for proof
  production.
- `plan-prover-opening-flow`: typed opening inputs, produced claims, opening
  batches, and transcript dependencies.
- `plan-prover-families`: compact repeated opening/eval/claim rows into named
  families and ranges.

The Rust emitter should consume those plans directly rather than rediscovering
kernel/proof semantics from CPU op attrs.

### P5: Factor common plan vocabulary

After P2-P4 reveal the real common surface, factor shared plan primitives out of
stage-local modules.

Good shared candidates:

- typed symbol and ID wrappers
- transcript event plans
- opening input descriptors
- opening family descriptors
- opening batch descriptors
- field expression descriptors
- eval-family descriptors

Do not prematurely force prover and verifier to share side-specific execution
types. Shared vocabulary is good; shared accidental abstractions are not.

### P6: Move toward a top-level prover program

Once per-stage plans are compact and typed, consider a top-level
`JoltProverProgramPlan` that schedules the proof-producing flow across stages.

This should remain secondary to P2-P5. The prover has witness state, executor
state, and hot kernels that make top-level orchestration more complex than the
verifier case. A top-level plan is useful if it clarifies scheduling and proof
record layout; it is not useful if it merely wraps existing stage calls in a
new layer.

## Non-regression contracts

1. **Readability cannot regress.** Generated prover Rust should become easier
   to audit as proof work, not merely shorter.
2. **Performance cannot silently regress.** Prover kernels are hot-path code.
   Any interpreter-heavy or dispatch-heavy change needs benchmark coverage or
   explicit approval.
3. **Generated LOC should trend down.** Temporary LOC growth is acceptable only
   if it replaces opaque template logic with typed declarative plan data and has
   a clear path to reducing generated-stage noise.
4. **Kernel semantics must stay single-sourced.** Do not duplicate kernel math
   in generated Rust for readability.
5. **No compatibility shims.** Full cutover for each changed plan surface.
6. **Equivalence and proof serialization must remain stable unless explicitly
   scoped.** A readability refactor should not change proof semantics or record
   layout by accident.
7. **Verifier priority remains explicit.** Prover refactor slices should not
   block or complicate verifier CPU-to-Rust planning unless they remove shared
   duplication, especially in `jolt-poly`.

## Implementation readiness

The most implementation-ready path is:

1. Reuse or add the missing `jolt-poly` helpers that both verifier and prover
   need.
2. Add prover cleanup metrics.
3. Use those metrics to lock the current stage 6 and stage 7 generated-surface
   baseline.
4. Classify repeated plan rows into named families.
5. Replace the largest safe repeated families with typed family plans.
6. Type the remaining string-backed execution fields through planning data,
   not emitter-local token maps.

This path is valuable even if the final top-level prover program is deferred.
It attacks the current readability problem directly while preserving the
existing prover execution model.

## Open questions

- Should shared plan concepts live in a neutral `bolt-plan-runtime` crate, or
  should prover and verifier each own their own runtime crate until the common
  surface is clearer?
- Which prover facts should become planning-pass output versus remaining kernel
  ABI data?
- Which stage 6 and stage 7 expansions are purely regular families, and which
  encode irregular protocol decisions that should remain explicit rows?
- What prover benchmark should become the repeatable perf baseline for these
  slices?
- How much of the prover stage shape is proof-record layout that should remain
  numbered, versus historical scaffolding that can become a top-level program?

The working bias should be conservative: first make the generated prover's
existing structure visible and typed, then decide whether a broader top-level
program refactor buys enough clarity to justify its cost.
