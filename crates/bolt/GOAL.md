# Bolt Jolt Verifier Goal

Bolt's first full-field, non-zk Jolt implementation is semantically complete
enough to move the active goal from stage bring-up to verifier-pipeline
hardening. The next long-haul objective is to make the Bolt-generated Jolt
verifier compact, human-readable, auditable, and security-hardened while
preserving the existing full-`Fr` Jolt semantics.

## Objective

Refactor the Bolt-generated Jolt verifier pipeline so the generated verifier is
a typed verifier program: a small orchestration layer plus declarative plans for
commitments, transcript events, sumcheck drivers, value-graph computations,
opening batches, and PCS checks, backed by reusable verifier runtime modules.
The compiler should continue to own protocol facts through MLIR and typed plan
data; generated Rust should not rediscover Jolt semantics late through ad hoc
string matching, repeated stage-local helper code, or hardcoded assumptions that
the verifier must always have a fixed number of numbered stages.

Starting baseline:

```text
generated jolt-verifier: ~21.5k LOC
stage6 + stage7:        ~13.2k LOC
verifier.rs:              649 LOC
```

Current locked cleanup baseline (post-S1 audit-tier split):

```text
generated jolt-verifier total:     7,905 LOC
generated verifier surface:        6,002 LOC
tier A (Bolt verifier runtime):    1,265 LOC   stages/common.rs
tier B (Jolt verifier core):         638 LOC   stages/jolt_relations.rs
stage6 + stage7:                   1,660 LOC
verifier.rs:                         428 LOC
```

The shared runtime is now split along an explicit audit boundary. See
"Audit Tiers" below.

Target:

```text
generated verifier surface:        <= 4k-6k LOC
stretch generated surface:         <= 2k-3k LOC
verifier.rs orchestration:         <= 350-500 LOC
stage6 + stage7 generated surface: <= 2k-3k LOC
tier A (Bolt verifier runtime):    shrink toward <= 800 LOC as more helpers
                                   become typed plan data driven from MLIR
tier B (Jolt verifier core):       stable around current size; growth here is
                                   a protocol-math decision, not emitter creep
```

The goal is to reduce the human-facing generated verifier surface by roughly an
order of magnitude. Shared runtime code may exist, but it must be modular,
boring to audit, and driven by explicit MLIR-derived plan data.

Current S2.75-S5 implementation status:

- S3/S4/S5 have largely landed in the generated verifier shape: Stage 2-7 emit
  typed scalar/point/vector/eval-family refs and `RelationOutputPlan` rows with
  closed relation IDs, old eval-prefix reconstruction sites are gated at zero,
  and handwritten
  `expected_stage67_*` output helpers are gone from generated/runtime code.
- S2.75 remains the main incomplete architecture item, but its central
  CPU-to-Rust planning boundary is now explicit: `VerifierStagePlan` is built
  through named planning functions for program steps, transcript flow,
  sumchecks, value/relation outputs, opening flow, and Stage 5/6 relation-local
  inputs. Relation-output, verifier sumcheck-flow, and verifier opening-flow
  validation are now owned by `VerifierStagePlan`. Remaining S2.75 work is
  concentrated in target validation still owned by emitters.
- Performance evidence remains a live completion gate. The SHA2-chain perf
  oracles must be rerun for the interpreter-heavy slices before this goal is
  closed.

## Audit Tiers

The verifier code is partitioned into three explicit audit tiers. The `Concrete
Gates` section enforces a per-tier LOC ceiling for each tier independently so
growth is attributed to the right trust boundary.

- **Tier A (Bolt verifier runtime):** generic, protocol-agnostic helpers
  (plan structs, `ValueStore`, sumcheck driver loop, opening-equality
  interpreter, transcript helpers). Lives in
  `crates/jolt-verifier/src/stages/common.rs`, generated from
  `crates/bolt/src/protocols/jolt/verifier_common.rs.template`. Tier A should
  ratchet *down* over time as more helpers move into typed plan data driven
  from MLIR.

- **Tier B (audited Jolt verifier core):** hand-written Jolt-specific verifier
  math and relations (Stage 6/7 evaluators, `normalize_*_point`,
  `bytecode_gamma_powers`, the `Stage67Bytecode*` glue, polynomial-evaluation
  primitives). Lives in `crates/jolt-verifier/src/stages/jolt_relations.rs`,
  generated from
  `crates/bolt/src/protocols/jolt/verifier_jolt_relations.rs.template`. Tier
  B is the audit surface for Jolt-specific math; it is expected to stay
  roughly its current size and any growth here must be reviewed as a
  *protocol-math* decision, not emitter LOC.

- **Tier C (generated declarative stage data + orchestration):** typed plans,
  thin `verify_stageN` wrappers, and `verifier.rs`. This is everything else
  under `crates/jolt-verifier/src/`. Tier C should be aggressively shrunk
  toward the stretch target.

This split was introduced by the S1 audit-tier refactor. The pre-split
"shared verifier runtime" framing (a single ~1.8k-LOC `common.rs` mixing
generic Bolt scaffolding and Jolt-specific math) is retired.

The implementation plan for slices S2 — S6 (which progressively replace
stage-shaped Rust with a typed verifier-program model by lifting hand-written
Rust into MLIR vocabulary) lives in
`crates/bolt/VERIFIER_PROGRAM_REFACTOR_PLAN.md`.

## Design Philosophy

The end state is a typed verifier IR, not a cleverer Rust template system.
Generated verifier Rust should be mostly const plan data plus thin calls into
reviewed interpreters. The Rust emitter should project MLIR-derived facts; it
should not infer verifier semantics from string names, stage-local naming
conventions, or generated helper bodies.

Typed symbol handles should carry verifier references wherever possible. Plain
strings may remain as diagnostics and serialization names, but they should not
be the execution contract for claim, eval, point, relation, or opening lookup.

Verifier values should live in explicit domains:

```text
scalar field values
multilinear query points
ordered field vectors / eval families
opening claims and batches
relation expected-output values
protocol-specific encoding tables
```

When a cleanup slice moves logic out of generated Rust, the semantics must stay
visible in MLIR, typed attrs, or typed plan rows. Moving semantics into opaque
runtime code without a readable generated plan is not progress.

This verifier cleanup is coupled to the generic protocol cleanup in
`GENERIC_PROTOCOL_GOAL.md`: shrinking the generated verifier should move generic
mechanics into Bolt IR/typed plans and shared runtime, not into Jolt-specific
emitter special cases.

Numbered stages may remain as proof-layout slots and diagnostic scopes, but
they are not the final architecture. The final architecture is an explicit
verifier program whose steps describe commitment receipt, transcript events,
sumcheck verification, value evaluation, opening batching, and PCS verification.

## Locked Genericity Decisions

The next cleanup track should make Jolt a quarantined protocol package over
generic Bolt compiler infrastructure:

- Root `bolt::*` exports should be generic-only. Jolt APIs should be imported
  from `bolt::protocols::jolt::*`.
- Jolt-specific emitters are not the long-term target. Quarantine them first so
  leakage is explicit, then progressively lift stage emission into a generic
  `cpu -> Rust` backend driven by typed MLIR-derived plans.
- Replace the current Jolt evaluation-proof special case with either a generic
  protocol extension hook or generic PCS/evaluation IR. Start with the minimal
  extension hook if that keeps the cleanup mechanical.
- Add hygiene gates for generic compiler modules, initially targeting
  `crates/bolt/src/{schema.rs,pass.rs,emit/rust}`. Any temporary Jolt allowlist
  must be explicit and shrink over time.
- Namespace and file-layout refactors should preserve generated
  `jolt-prover`/`jolt-verifier` output byte-for-byte unless the change
  intentionally updates artifact structure.
- At the end of each goal-mode slice, report which quarantined Jolt emitters are
  still genuinely protocol-specific and which are ready to lift into generic
  typed-plan emission.

## Current State

The initial quarantine slice is complete:

```text
generic artifact assembly no longer owns Jolt artifact APIs
root bolt exports keep Jolt helpers under bolt::protocols::jolt
generic compiler source is guarded against accidental Jolt protocol strings
checked-in generated verifier is under the current LOC target
shared verifier plan/runtime scaffolding exists in stages/common.rs
```

The remaining work is no longer a stage bring-up task or a pure LOC cleanup.
This goal is now owned as a verifier-quality cutover: make the generated
verifier a Rust artifact that a reviewer can audit directly, without needing to
understand Bolt lowering internals.

## Completion Plan

Work proceeds in reviewable slices. Each slice must reduce a concrete readability
or soundness risk, regenerate checked-in artifacts, and preserve the semantic
oracles.

1. **Typed plan vocabulary**

   Replace execution-relevant strings with typed Rust plan data:

   ```text
   StageId
   ProofSlot
   ProgramStepKind
   TranscriptEventKind
   RelationKind
   FieldExprKind
   OracleId
   DomainId
   ClaimKind
   PointSource
   EvalSource
   OpeningEqualityMode
   ```

   Diagnostic names may remain as `symbol` fields, but verifier execution must
   not branch on loose strings when a closed enum or structured ID can express
   the same fact.

2. **Generic stage runtime**

   Collapse stage-local verifier loops into one reviewed runtime path:

   ```text
   verify transcript events
   evaluate field expression plans
   verify sumcheck drivers
   collect named evals and output claims
   construct opening claims
   check opening equalities
   convert proof records into runtime inputs
   ```

   Stage files should keep only stage-specific plan constants, typed relation
   adapters when truly required, and thin `verify_stageN` wrappers.

3. **Readable generated plan layout**

   Reformat emitted Rust so a human can scan it by protocol concept:

   ```text
   params
   transcript events
   opening inputs
   field expressions
   sumcheck drivers
   evals
   opening claims
   opening batches
   stage program
   ```

   Avoid generated one-line functions, `#[rustfmt::skip]` blocks for ordinary
   data, pipe-delimited operand strings, and stage-local macro tricks unless
   they make the artifact materially easier to read.

4. **Compact expression encoding**

   Replace verbose string formulas and operand encodings with typed compact
   expression rows:

   ```text
   FieldExprKind::Add { lhs, rhs }
   FieldExprKind::Mul { lhs, rhs }
   FieldExprKind::Pow { base, exponent }
   FieldExprKind::LagrangeBasisEval { domain, index, point }
   ```

   Repeated operands and symbols should be pooled or indexed when doing so makes
   the generated Rust clearer, not merely shorter.

5. **Typed relation dispatch**

   Replace relation-name matching with typed `RelationKind` dispatch and
   relation-specific plan payloads. Any remaining string dispatch must be
   explicitly allowlisted, documented as temporary, and protected by tests that
   fail when the allowlist grows.

6. **Stage 8 boundary cleanup**

   Keep Dory/evaluation verification reviewed and explicit, but give Stage 8 a
   typed input contract from earlier stages:

   ```text
   committed opening IDs
   virtual opening IDs
   ordered opening batches
   evaluation point sources
   PCS proof slots
   transcript labels
   ```

   Stage 8 does not have to become fully generic before this goal completes, but
   it must not rely on name-then-position fallback or hidden generated ordering.

7. **Hardening suite**

   Add or preserve negative oracles while the Rust shape changes. The generated
   verifier must reject malformed plans/proofs with typed errors, not panic or
   silently default missing values.

8. **Independent readability review**

   Treat readability as a first-class gate. Before declaring this goal complete,
   run a blind review where reviewers see only:

   ```text
   current handwritten/verifier-style Rust surface
   generated jolt-verifier Rust surface
   brief protocol-stage context
   no Bolt lowering or emitter implementation details
   ```

   A small council of reviewers, human or model, should be able to answer:

   ```text
   What does each stage verify?
   Where are transcript challenges absorbed or squeezed?
   Which openings are checked, and at what points?
   Where does sumcheck verification happen?
   What code is trusted runtime logic versus declarative generated data?
   How would a malformed proof be rejected?
   ```

   The generated verifier passes this gate only if the answers are findable from
   the emitted Rust with minimal compiler context.

## Non-Negotiables

- Preserve the current full-field non-zk Jolt protocol path:
  `Transcript<Challenge = Fr>`.
- Preserve or improve readability. A slice that reduces LOC by hiding verifier
  semantics in opaque runtime code, compiler-only conventions, or dense tables
  that blind reviewers cannot audit fails this goal.
- Do not regress final LOC. Temporary slice-local growth is acceptable only when
  it moves hand-written protocol math into declarative typed plan data and the
  completed stack remains no larger than the locked post-S1 baselines above.
- Do not silently regress performance. Existing SHA2-chain core-vs-Bolt perf
  oracles are required smoke gates; interpreter-heavy slices must also record a
  before/after verifier-time baseline and fix or explicitly justify any
  repeatable regression.
- `jolt-verifier` must not depend on `jolt-prover`, `jolt-kernels`,
  `jolt-core`, `jolt-equivalence`, `jolt-profiling`, or tracer internals.
- Bolt compiler boundaries remain:
  `protocol -> concrete -> party -> compute -> cpu -> Rust`.
- Verifier CPU IR must remain kernel-free. Prover kernels are temporary
  implementation details below the dialect boundary.
- Jolt semantics should be represented in protocol builders, dialect ops,
  validators, lowering passes, or typed verifier plans. The Rust emitter should
  not infer protocol meaning from loose strings when a typed enum, attr, op, or
  plan field can carry it.
- Do not keep old stage-local execution paths as verifier fallbacks once typed
  verifier-program execution lands. Rollback is a git operation, not an
  alternate runtime path.
- Generated verifier files should be mostly declarative:

```rust
pub const STAGE_PLAN: StagePlan = ...;

pub fn verify_stage(...) -> Result<StageArtifacts, VerifyError> {
    runtime::verify_stage(&STAGE_PLAN, ...)
}
```

## Target Architecture

The final verifier shape should read like this:

```text
crates/jolt-verifier
  src/lib.rs
  src/verifier.rs
    public API
    proof shape
    verifier-program execution
    proof-slot/checkpoint layout
    error mapping

  src/stages/
    commitment.rs
    stage1_outer.rs
    stage2.rs
    ...
    mostly declarative generated plans

  src/runtime/ or shared verifier crate
    generic stage verifier
    generic field expression evaluator
    generic opening-claim machinery
    generic sumcheck/eval proof conversion
    transcript helpers
    typed relation evaluators
```

Generated stage files should answer:

```text
What claims exist?
What expressions are evaluated?
What transcript events happen?
What openings are checked?
What relations are verified?
```

Runtime modules should answer:

```text
How is a field expression plan evaluated?
How is a stage plan verified?
How are opening/eval consistency checks performed?
How are proof records converted into runtime verifier inputs?
```

The checked-in generated Rust should expose a top-level `VERIFIER_PROGRAM` (or
equivalent) that a reviewer can scan to see the whole verifier flow. A reviewer
should not need to know how many stage files the emitter happened to produce in
order to understand verifier control flow.

## Main Refactor Tracks

1. **Verifier runtime extraction**

   Move duplicated stage-local machinery into one runtime:

   ```text
   field expression evaluation
   opening claim lookup and equality checks
   sumcheck driver verification
   transcript squeeze/absorb helpers
   stage proof conversion
   stage plan execution
   ```

2. **Shared verifier plan types**

   Replace stage-specific copies such as `Stage6FieldExprPlan` and
   `Stage7OpeningClaimPlan` with shared plan structs:

   ```text
   FieldExprPlan
   OpeningClaimPlan
   OpeningEqualityPlan
   SumcheckClaimPlan
   SumcheckDriverPlan
   SumcheckEvalPlan
   StagePlan
   RelationPlan
   ```

3. **Compact field expression encoding**

   Stage 6 and Stage 7 are bloated by per-expression constants and operand
   arrays. Replace those with compact tables or pooled operand slices.

4. **Typed relation dispatch**

   Replace stringly relation handling with typed plan data where practical:

   ```text
   RelationKind::RamReadWrite
   RelationKind::InstructionReadRaf
   RelationKind::BytecodeReadRaf
   RelationKind::Booleanity
   RelationKind::HammingBooleanity
   RelationKind::RegistersReadWrite
   ...
   ```

   Any remaining string dispatch must be explicitly allowlisted and covered by
   schema tests.

5. **Clean top-level verifier API**

   `verifier.rs` should be readable orchestration: proof shape, verifier
   inputs, verifier programs, stage ordering, evaluation proof handling, and
   clear error mapping. Repeated per-stage proof conversion should disappear.

## One-Time Hardening Work

Before large readability refactors, add a durable verifier hardening suite.
The suite should include positive equivalence and negative tamper oracles.

Verifier tamper cases:

```text
valid generated proof verifies
core and Bolt verifier accept/reject agree
tampered sumcheck coefficient rejects
tampered sumcheck point rejects
tampered named eval rejects
tampered commitment rejects
missing commitment rejects
missing stage proof rejects
reordered stage proof rejects
stage proof in the wrong slot rejects
wrong transcript state rejects
wrong evaluation proof rejects
missing evaluation setup rejects
missing evaluation proof rejects
extra/missing opening claims reject
opening claims in the wrong order reject
opening equality mismatch rejects
PCS proof mismatch rejects
```

MLIR/compiler hardening cases:

```text
unknown dialects rejected
prover-only ops rejected in verifier pipeline
verifier-only ops rejected in prover pipeline
unthreaded transcript ops rejected
hidden or reordered opening batch claims rejected
unsupported equality modes rejected
duplicate proof slots rejected
invalid point arity rejected
invalid round schedule rejected
invalid relation kind rejected
verifier CPU IR contains no kernel dispatch
generated verifier imports no forbidden crates
```

## Concrete Gates

Readability and LOC gates (enforced by `crates/bolt/tests/verifier_cleanup.rs`):

```text
total generated jolt-verifier LOC trends down
generated verifier surface (Tier C) <= 6.1k, target <= 6k, stretch <= 3k
tier A bolt verifier runtime <= 1.4k, ratcheting down
tier B audited Jolt verifier core <= 700 (growth requires protocol-math review)
verifier.rs <= 500 LOC, stretch <= 350
stage6 + stage7 generated LOC <= 3k-5k, stretch <= 2k-3k
no duplicate stage-local generic plan structs
no duplicate stage-local field-expression interpreter
no duplicate stage-local opening equality interpreter
no giant per-expression operand constants after compaction
stage files are mostly declarative plan data
generated functions are rustfmt-formatted and scan-friendly
execution-relevant plan fields are typed unless explicitly allowlisted
relation dispatch string sites trend to zero
field-expression formula strings trend to zero
```

LOC gates are guardrails, not the quality definition. A shorter verifier that
hides semantics in opaque runtime code or compiler-only conventions fails this
goal. The Tier A and Tier B ceilings are independently enforced so cleanup
work cannot accidentally grow the audited Jolt math while shrinking generic
scaffolding (or vice versa).

Security and boundary gates:

```text
jolt-verifier imports are allowlisted
no jolt-prover dependency from jolt-verifier
no jolt-kernels dependency from jolt-verifier
no jolt-core dependency from jolt-verifier
no prover role ops in verifier MLIR
no kernel attrs in verifier CPU IR
all transcript-producing ops thread transcript state
all opening batches preserve explicit ordered claims
all relation dispatch is typed or allowlisted
missing proof data returns typed verifier errors
duplicate or inconsistent opening claims reject explicitly
name-then-position eval fallback is absent from verifier execution
```

Performance gates:

```text
existing SHA2-chain core-vs-Bolt perf oracles remain runnable and green
before/after verifier-time baseline recorded for interpreter-heavy slices
repeatable verifier-time regressions are fixed or explicitly approved
perf thresholds are not loosened to land readability refactors
```

Current perf evidence from May 16, 2026 on `quang/bolt-stack`:

```text
2^16 SHA2-chain oracle, 1 sample: passed
  verify_ms ratio: 1.090x
  prove_ms ratio: 0.856x

2^20 SHA2-chain oracle, 1 sample: passed
  verify_ms ratio: 1.124x
  prove_ms ratio: 1.298x against 1.300x gate

2^20 SHA2-chain oracle, 3 samples: passed
  verify_ms ratio: 0.987x
  prove_ms mean ratio: 1.329x, 95% CI [1.284x, 1.373x]
```

The previous hard perf blocker has moved to a fragile-margin risk: current
oracles are green, but the 2^20 prover-time ratio remains close enough to the
threshold that future completion audits should rerun it.

Semantic gates:

```bash
cargo fmt --check
cargo check -p bolt -p jolt-verifier -p jolt-prover -p jolt-equivalence --quiet
cargo nextest run -p bolt --test verifier_cleanup --no-capture
cargo nextest run -p bolt --test commitment_ir --cargo-quiet
cargo nextest run -p jolt-equivalence --test generated_role_crates --cargo-quiet
cargo nextest run -p jolt-equivalence --test bolt_commitment --no-capture
```

Required semantic outcomes:

```text
core accepts Bolt proof
Bolt verifier accepts Bolt proof
core and Bolt transcript state histories match
core and Bolt observable proof artifacts match
core and Bolt reject equivalent tampered proofs
generated prover/verifier crates stay in sync with artifact rail
```

Subjective readability gate:

```text
blind reviewers can identify stage claims, transcript events, openings,
sumcheck drivers, relation checks, and error paths from emitted Rust alone
reviewers do not need to inspect Bolt lowering or emitter code to understand
the generated verifier's security boundary
reviewers prefer the new generated verifier shape over the current generated
shape for auditability, or remaining objections are converted into tracked
cleanup items
```

Perf remains a regression guard, not the center of this task. The existing
`sha2-chain` e2e/proving Perfetto traces are useful for confirming cleanup does
not accidentally move prover cost, but the main objective is verifier
readability, simplicity, and security.

## Iteration Algorithm

Each cleanup loop should follow the same rule:

```text
1. Measure current LOC, duplication, imports, and typed-vs-string dispatch.
2. Pick one duplication class or hygiene issue to eliminate.
3. Move generic logic into runtime only if semantics remain explicit in MLIR or
   typed plan data.
4. Regenerate checked-in verifier artifacts through the compiler rail.
5. Run hardening, equivalence, import, and schema gates.
6. Keep the change only if readability improves and no oracle weakens.
```

Use this scoring function when choosing work:

```text
score =
  LOC reduction
+ fewer duplicate structs/functions
+ fewer string dispatch sites
+ fewer generated helper bodies
+ stronger negative oracles
+ clearer verifier.rs
+ clearer stage files under blind review
+ more verifier facts represented as typed plans
- semantic opacity introduced into runtime
- compiler-context required to audit emitted Rust
```

## Definition Of Done

This long-haul cleanup is complete when:

```text
generated verifier surface is <= 4k-6k LOC
verifier.rs is <= 500 LOC
stage files are mostly declarative plans
generic verifier mechanics live once
Jolt relation semantics are typed and auditable
field expressions and program steps use typed plan data
remaining string symbols are diagnostic or explicitly allowlisted
MLIR verifier pathway has malformed-input rejection tests
tamper suite covers commitments, transcript, stages, openings, evals, and PCS proof
core/Bolt accept/reject equivalence is preserved
generated verifier import boundaries are enforced
independent blind readability review passes
```

The desired end state is not merely fewer lines. The verifier should be easy to
navigate, easy to audit, and hard for the compiler pipeline to accidentally
weaken.
