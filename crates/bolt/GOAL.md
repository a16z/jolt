# Bolt Jolt Verifier Goal

Bolt's first full-field, non-zk Jolt implementation is semantically complete
enough to move the active goal from stage bring-up to verifier-pipeline
hardening. The next long-haul objective is to make the Bolt-generated Jolt
verifier compact, human-readable, auditable, and security-hardened while
preserving the existing full-`Fr` Jolt semantics.

## Objective

Refactor the Bolt-generated Jolt verifier pipeline so the generated verifier is
a small orchestration layer plus declarative verifier plans, backed by reusable
verifier runtime modules. The compiler should continue to own protocol facts
through MLIR and typed plan data; generated Rust should not rediscover Jolt
semantics late through ad hoc string matching or repeated stage-local helper
code.

Starting baseline:

```text
generated jolt-verifier: ~21.5k LOC
stage6 + stage7:        ~13.2k LOC
verifier.rs:              649 LOC
```

Current locked cleanup baseline:

```text
generated jolt-verifier: 14,092 LOC
stage6 + stage7:         7,346 LOC
verifier.rs:               492 LOC
```

Target:

```text
generated verifier surface:        <= 4k-6k LOC
stretch generated surface:         <= 2k-3k LOC
verifier.rs orchestration:         <= 350-500 LOC
stage6 + stage7 generated surface: <= 2k-3k LOC
shared runtime/helpers:            allowed when generic, named, and reviewed
```

The goal is to reduce the human-facing generated verifier surface by roughly an
order of magnitude. Shared runtime code may exist, but it must be modular,
boring to audit, and driven by explicit MLIR-derived plan data.

This verifier cleanup is coupled to the generic protocol cleanup in
`GENERIC_PROTOCOL_GOAL.md`: shrinking the generated verifier should move generic
mechanics into Bolt IR/typed plans and shared runtime, not into Jolt-specific
emitter special cases.

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

## Immediate Goal-Mode Slice

First objective for another agent:

```text
Quarantine Jolt-specific artifact APIs out of generic Rust artifact assembly
while preserving generated output and all current gates.
```

Required steps:

1. Move `JoltProtocolStage`, `jolt_artifact_config`, `jolt_rust_artifact`,
   `assemble_jolt_*`, `write_jolt_generated_crates`, and
   `validate_jolt_rust_artifact_imports` out of generic
   `crates/bolt/src/emit/rust/artifacts.rs` into a Jolt-owned module such as
   `crates/bolt/src/protocols/jolt/artifacts.rs`.
2. Keep `ProtocolArtifactConfig`, `ProtocolStage`, `ProtocolRustArtifact`,
   `GeneratedCrate`, `assemble_generated_crates`, `write_generated_crates`, and
   `validate_rust_artifact_imports` in the generic artifact layer.
3. Stop re-exporting Jolt APIs from `crates/bolt/src/lib.rs`; update callers in
   Bolt tests, `jolt-equivalence`, and perf harnesses to import from
   `bolt::protocols::jolt`.
4. Add the first genericity hygiene test that rejects new Jolt protocol strings
   in generic compiler modules, using a small documented allowlist only for
   migration leftovers.
5. Run focused generation/import gates and confirm checked-in generated role
   crates are unchanged unless an intentional artifact-structure change is
   documented.

Acceptance criteria:

```text
generic artifact assembly has no Jolt stage enum or Jolt artifact config
root bolt exports are generic-only
Jolt artifact helpers are namespaced under protocols::jolt
generated jolt-prover/jolt-verifier are byte-for-byte unchanged, or changes are intentional
genericity hygiene gate exists
existing generated-artifact and verifier-boundary gates pass
```

## Non-Negotiables

- Preserve the current full-field non-zk Jolt protocol path:
  `Transcript<Challenge = Fr>`.
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
    stage ordering
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

Readability and LOC gates:

```text
total generated jolt-verifier LOC trends down
verifier.rs <= 500 LOC, stretch <= 350
stage6 + stage7 generated LOC <= 3k-5k, stretch <= 2k-3k
no duplicate stage-local generic plan structs
no duplicate stage-local field-expression interpreter
no duplicate stage-local opening equality interpreter
no giant per-expression operand constants after compaction
stage files are mostly declarative plan data
```

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
```

Semantic gates:

```bash
cargo fmt --check
cargo check -p bolt -p jolt-verifier -p jolt-prover -p jolt-equivalence --quiet
cargo test -p bolt --test verifier_cleanup -- --nocapture
cargo test -p bolt --test commitment_ir --quiet
cargo test -p jolt-equivalence --test generated_role_crates --quiet
cargo test -p jolt-equivalence --test bolt_commitment -- --nocapture
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
- semantic opacity introduced into runtime
```

## Definition Of Done

This long-haul cleanup is complete when:

```text
generated verifier surface is <= 4k-6k LOC
verifier.rs is <= 500 LOC
stage files are mostly declarative plans
generic verifier mechanics live once
Jolt relation semantics are typed and auditable
MLIR verifier pathway has malformed-input rejection tests
tamper suite covers commitments, transcript, stages, openings, evals, and PCS proof
core/Bolt accept/reject equivalence is preserved
generated verifier import boundaries are enforced
```

The desired end state is not merely fewer lines. The verifier should be easy to
navigate, easy to audit, and hard for the compiler pipeline to accidentally
weaken.
