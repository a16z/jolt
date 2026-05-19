# Spec: `jolt-prover` Model Crate

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-19 |
| Status | draft |
| PR | TBD |

## Context

Jolt is moving toward a modular prover/verifier split. `jolt-verifier` is the
handwritten verifier model crate; `jolt-prover` should be the corresponding
handwritten prover model crate.

Bolt is a separate track. It may eventually target the same contracts, but this
work must not depend on Bolt-generated code, Bolt IR, or generated role crates.
The manual prover exists to make the modular protocol executable, auditable,
and usable for recursion work before Bolt can emit the final shape.

The design priority is asymmetric:

```text
protocol code: small, explicit, auditable
compute code: allowed to be large, specialized, optimized, and replaceable
```

`jolt-prover` is therefore an orchestration crate. It should make proof
dataflow, transcript ordering, stage dependencies, proof assembly, opening
plans, and ZK/recursion boundaries clear. Heavy prover-side computation should
live behind kernel interfaces in dedicated compute crates.

## Goal

Build a standalone manual `jolt-prover` crate that:

- computes proofs accepted by the modular `jolt-verifier`;
- models stage-by-stage prover dataflow with typed inputs and outputs;
- keeps transcript-sensitive ordering visible in source;
- emits both public proof data and rich prover artifacts;
- delegates heavy witness, polynomial, and sumcheck-message computation to
  optimized kernel crates;
- uses existing modular protocol crates for formulas, sumcheck proof types,
  openings, PCS proving, and BlindFold;
- uses `jolt-core` only as a reference oracle in tests and compatibility
  harnesses;
- gives recursion, Dory-assist, BlindFold, and wrapper work a prover-side model
  before Bolt is ready.

## Non-Goals

- Do not use Bolt, generated `jolt-prover`, generated `jolt-verifier`,
  generated plans, or Bolt IR in this work.
- Do not make `jolt-prover` a port of Bolt-generated Rust.
- Do not put large optimized `compute_message` implementations directly in
  `jolt-prover` orchestration modules.
- Do not make `jolt-equivalence` reconstruct prover semantics. If later-stage
  opening inputs, witness slices, or proof fragments are required, expose them
  from `jolt-prover` or compute kernels.
- Do not replace `jolt-core` in one step. Core remains the reference oracle
  while the modular prover is brought up.
- Do not design a stable third-party prover API before the internal modular
  split is correct.
- Do not introduce a separate `jolt-proof` crate unless proof ownership becomes
  a real blocker.

## Design Principles

### Prover Is Orchestration

`jolt-prover` owns the proof pipeline:

```text
input resolution
  -> witness/oracle construction
  -> commitments
  -> stages 1..7
  -> stage 8 opening proof
  -> optional BlindFold/ZK proof
  -> optional Dory-assist boundary/proof
```

It should answer:

- what is computed next;
- which previous stage artifacts are consumed;
- which transcript messages are absorbed and squeezed;
- which claims and openings are produced;
- how the public proof is assembled;
- which rich artifacts are retained for later stages, tests, ZK, and recursion.

It should not answer:

- how to run a fast sparse RA bind;
- how to parallelize read-RAF prefix/suffix accumulation;
- how to materialize dense oracle data efficiently;
- how to execute a SIMD/GPU/rayon map-reduce kernel;
- how Dory MSMs or polynomial commitments are implemented.

### Kernels Own Heavy Computation

Heavy prover-side computation should be isolated in a kernel layer, initially
expected to be `jolt-kernels` or a similarly named `jolt-prover-kernels` crate.

Kernel code may be large, specialized, and representation-aware. It may own:

- sumcheck round-message computation;
- witness materialization;
- sparse/dense polynomial evaluation;
- prefix/suffix decomposition execution;
- RA sparse index processing;
- large rayon map-reduce loops;
- cache layouts and streaming bind state;
- optional unsafe preallocation when justified and tested.

Kernel code must not decide protocol truth. It must not own:

- which claims exist;
- which transcript labels are used;
- which stages are active;
- which formulas define input/output claims;
- which openings are public;
- which verifier equations define correctness.

Those facts belong to `jolt-claims`, `jolt-sumcheck`, `jolt-openings`,
`jolt-prover`, and `jolt-verifier`.

### Sumcheck Driver Is Generic

The current `jolt-core` prover has many relation-specific `compute_message`
methods, driven by a generic batched sumcheck loop. In the modular split, the
generic loop should live in `jolt-sumcheck`; the concrete message computation
should live behind relation/kernel implementations.

Desired layering:

```text
jolt-sumcheck
  generic prover driver:
    batching
    front-loaded padding
    round loop
    clear/committed round messages
    transcript absorbs/squeezes
    challenge binding
    final evaluation claims

jolt-kernels / prover kernels
  relation-specific message computation:
    RAM read/write
    register read/write
    Spartan product
    bytecode read-RAF
    booleanity
    claim reductions
    etc.

jolt-prover
  stage orchestration:
    instantiate relation provers
    call the generic driver
    record artifacts and opening dependencies
```

A representative trait boundary:

```rust
pub trait SumcheckRelationProver<F> {
    fn shape(&self) -> SumcheckShape;
    fn input_claim(&self, openings: &ProverOpeningInputs<F>) -> F;
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UnivariatePoly<F>;
    fn bind(&mut self, round: usize, challenge: F);
    fn finish(&mut self) {}
    fn opening_dependencies(&self, point: &[F]) -> Vec<OpeningDependency<F>>;
}
```

The exact trait shape should be driven by the first manual stage ports, but the
ownership should stay stable: `jolt-sumcheck` drives, kernels compute,
`jolt-prover` orchestrates.

### Flexible Kernel Granularity

The kernel boundary should support multiple granularities:

```text
coarse:
  prove all of stage 6

medium:
  prove one batched stage sumcheck

fine:
  compute one relation's round message
  evaluate one sparse polynomial at a point
  materialize one oracle column
```

The model crate should not force one granularity forever. It is acceptable for
early implementations to use coarse kernels if that unblocks correctness, then
replace them with finer kernels where the protocol model benefits.

Possible shape:

```rust
pub trait KernelSet<F> {
    type Stage1: Stage1Kernels<F>;
    type Stage2: Stage2Kernels<F>;
    type Stage3: Stage3Kernels<F>;
    type Stage4: Stage4Kernels<F>;
    type Stage5: Stage5Kernels<F>;
    type Stage6: Stage6Kernels<F>;
    type Stage7: Stage7Kernels<F>;
}
```

Each stage can then choose whether its kernel trait is coarse or decomposed
internally.

## Crate Responsibilities

```text
common
  public I/O types and shared VM constants

jolt-program
  reusable program, trace, bytecode, RAM preprocessing model

jolt-claims
  symbolic Jolt formulas, claim metadata, opening requirements

jolt-r1cs
  guest R1CS builders and common lowering helpers

jolt-sumcheck
  generic sumcheck proof types, verifier, and prover driver

jolt-openings
  evaluation claims, opening plans, RLC reduction

jolt-dory / jolt-hyperkzg
  PCS proving, opening, verification, and setup

jolt-blindfold
  generic BlindFold protocol and verifier/prover machinery

jolt-kernels or jolt-prover-kernels
  optimized prover-side compute kernels

jolt-prover
  manual prover orchestration and artifact flow

jolt-verifier
  manual verifier orchestration and proof model
```

## Target Layout

```text
crates/jolt-prover/src/
  lib.rs
  error.rs
  config.rs
  inputs.rs
  preprocessing.rs
  artifacts.rs
  prover.rs

  commitment/
    mod.rs
    inputs.rs
    outputs.rs
    prove.rs

  stages/
    mod.rs
    stage1/
      mod.rs
      inputs.rs
      outputs.rs
      prove.rs
    stage2/
      mod.rs
      inputs.rs
      outputs.rs
      prove.rs
    stage3/
    stage4/
    stage5/
    stage6/
    stage7/
    stage8/
      mod.rs
      opening_plan.rs
      prove.rs

  openings/
    mod.rs
    accumulator.rs
    plan.rs
    prove.rs

  zk/
    mod.rs
    blindfold.rs

  dory_assist/
    mod.rs
```

This layout is a target, not a requirement for the first PR. The important
properties are stage-local contracts, explicit artifact flow, and no hidden
catch-all prover object.

## Public API Shape

Use a top-level free function, mirroring the `jolt-verifier` model:

```rust
pub fn prove<F, PCS, VC, T, K>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    inputs: JoltProverInputs<'_, F>,
    kernels: &mut K,
    transcript: &mut T,
) -> Result<(jolt_verifier::JoltProof<PCS, VC>, JoltProverArtifacts<F, PCS, VC>), ProverError>
where
    F: jolt_field::Field,
    PCS: jolt_openings::CommitmentScheme<Field = F>,
    VC: jolt_crypto::VectorCommitment<Field = F>,
    T: jolt_transcript::Transcript<Challenge = F>,
    K: KernelSet<F>;
```

The exact bounds will evolve with existing modular trait names. The shape
matters more than the spelling:

- inputs are explicit;
- preprocessing is explicit;
- kernel set is explicit;
- transcript is threaded visibly;
- proof and rich artifacts are both returned.

### Proof Type Ownership

The current verifier spec keeps `JoltProof` in `jolt-verifier`. For the first
manual prover, `jolt-prover` may assemble that verifier-owned proof type through
a narrow conversion module.

Rules:

- `jolt-verifier` must not depend on `jolt-prover`;
- `jolt-prover` may depend on `jolt-verifier` proof types initially;
- all such coupling should be isolated in proof assembly/conversion code;
- introduce a shared proof crate only if this dependency becomes a real
  architectural blocker.

## Artifact Model

The prover should emit both the public proof and rich internal artifacts.

Public proof:

```text
what verifier consumes
```

Prover artifacts:

```text
what later prover stages, opening proof generation, BlindFold, Dory assist,
tests, and debugging consume
```

Representative shape:

```rust
pub struct JoltProverArtifacts<F, PCS, VC> {
    pub commitment: CommitmentArtifacts<F, PCS>,
    pub stage1: Stage1Artifacts<F>,
    pub stage2: Stage2Artifacts<F>,
    pub stage3: Stage3Artifacts<F>,
    pub stage4: Stage4Artifacts<F>,
    pub stage5: Stage5Artifacts<F>,
    pub stage6: Stage6Artifacts<F>,
    pub stage7: Stage7Artifacts<F>,
    pub stage8: Option<OpeningArtifacts<F, PCS>>,
    pub zk: Option<BlindFoldArtifacts<F, VC>>,
}
```

Artifacts should include named values required by:

- later stage input derivation;
- opening dependency construction;
- standard-mode opening proof assembly;
- ZK/BlindFold committed-sumcheck witness construction;
- Dory-assist boundary construction;
- core parity and modular prover/verifier tests.

Do not force tests to reconstruct hidden stage state from public proof data.

## Stage Contract

Each stage should be a free function with explicit dependencies:

```rust
pub struct Deps<'a, F> {
    pub previous: &'a PreviousStageArtifacts<F>,
}

pub fn deps<F>(previous: &PreviousStageArtifacts<F>) -> Deps<'_, F> {
    Deps { previous }
}

pub fn prove<F, T, K>(
    inputs: StageNInputs<'_, F>,
    deps: Deps<'_, F>,
    kernels: &mut K,
    transcript: &mut T,
) -> Result<StageNArtifacts<F>, ProverError>
where
    T: Transcript<Challenge = F>;
```

Stage code should make these visible:

- input claim construction;
- transcript absorbs/squeezes;
- sumcheck proof calls;
- batching order;
- output claim values;
- opening dependencies;
- data retained for ZK/BlindFold.

Stage code may call large kernels, but the stage should still reveal the
protocol structure.

## Sumcheck Proving Flow

A stage proving flow should look like:

```rust
let claims = stage_n::claims(deps, inputs)?;
let mut relation_provers = kernels.stage_n().relation_provers(inputs, deps, &claims)?;

let sumcheck = jolt_sumcheck::BatchedProver::prove(
    claims,
    &mut relation_provers,
    transcript,
)?;

let openings = stage_n::opening_dependencies(&sumcheck, deps, inputs)?;

Ok(StageNArtifacts {
    proof: sumcheck.proof,
    challenges: sumcheck.challenges,
    claims: sumcheck.claims,
    openings,
    kernel_outputs: sumcheck.kernel_outputs,
})
```

The exact API can differ, but the control flow should remain legible:

```text
claims -> relation provers -> batched sumcheck -> openings -> artifacts
```

## Opening Proof Flow

Stage 8 should be owned by `jolt-prover` as typed opening proof assembly:

```text
stage artifacts
  -> opening plan
  -> RLC opening claim reduction
  -> joint polynomial/hint construction
  -> PCS opening proof
  -> public proof attachment
```

The expensive materialization of a joint polynomial or opening hint may be
delegated to kernels or PCS helper APIs, but the opening plan and transcript
ordering should be explicit in `jolt-prover`.

## ZK And BlindFold

Initial bring-up may be standard/non-ZK only. The design must still preserve
the data required for ZK.

When ZK is enabled:

- stage sumchecks produce committed round messages;
- committed coefficients and blindings are retained for BlindFold;
- clear opening claims are not exposed in the public proof;
- `jolt-prover` constructs the Jolt-specific BlindFold protocol/witness from
  stage artifacts and formula metadata;
- `jolt-blindfold` proves the generic verifier-equation R1CS.

The same `jolt-claims` formula metadata should drive:

- standard prover claim computation;
- standard verifier checks;
- BlindFold R1CS claim constraints;
- wrapper R1CS claim constraints.

## Dory Assist And Wrapper

Dory assist is not part of the first `jolt-prover` milestone, but the artifact
model should leave a clear boundary:

```text
Jolt information-theoretic stages
  -> evaluation/opening boundary
  -> optional Dory-assist prover
  -> optional wrapper plan
```

`jolt-prover` should expose the stage-7/evaluation claims and opening boundary
that Dory assist needs. The Dory-assist prover may live in a module or separate
crate once the protocol is concrete.

## Compatibility Boundary

`jolt-core` remains the reference implementation while the modular prover is
ported.

Allowed:

- tests that generate real `jolt-core` proofs and compare modular artifacts;
- compatibility conversion under test-only or explicit compatibility features;
- fixture metadata documenting trace length, RAM domain, mode, advice, and
  expected core result.

Forbidden in production `jolt-prover`:

- direct calls into `jolt-core` prover internals;
- private core stage verifier/prover helpers;
- hidden `OpeningId` map reconstruction as the primary artifact model;
- core accumulator-shaped state as the modular prover API.

## Testing Strategy

### Kernel Tests

Every optimized kernel should have a slow reference or independently checkable
oracle where feasible:

```text
optimized kernel output == slow/reference kernel output
```

Coverage should include:

- sparse and dense paths;
- boundary trace lengths and padding;
- optional advice absent/present;
- random challenges;
- phase transitions;
- malformed input shape rejection.

### Prover/Verifier E2E

Each stage should unlock a modular prover/verifier test:

```text
manual jolt-prover stage artifacts
  -> assembled proof through current checkpoint
  -> manual jolt-verifier accepts through same checkpoint
```

Once full standard mode exists:

```text
manual jolt-prover proof
  -> manual jolt-verifier
  -> accepted
```

### Core Reference Parity

Core parity should compare public observable artifacts without moving semantics
into the test harness:

- commitments;
- transcript checkpoints;
- sumcheck proof shapes;
- named claims/evals;
- opening plans;
- final proof acceptance by core where compatible.

If a test needs stage-specific prover semantics to derive expected values, that
logic belongs in `jolt-prover`, `jolt-kernels`, `jolt-witness`, or an owning
modular protocol crate, not the harness.

### Soundness/Tampering

Soundness primarily belongs to `jolt-verifier`, but prover integration tests
should include representative negative checks around malformed artifacts:

- wrong stage dependency rejected before proof assembly;
- missing opening dependency rejected;
- invalid kernel output shape rejected;
- invalid transcript replay rejected by verifier;
- tampered generated proof rejected by verifier.

### No Bolt Gates

This spec does not require Bolt tests, generated artifact regeneration, or Bolt
equivalence gates. Bolt can later use these contracts as a target, but that is
not part of this work.

## Milestones

1. Scaffold `jolt-prover` with errors, inputs, preprocessing, artifacts, proof
   assembly, and no Bolt dependency.
2. Add prover-side sumcheck driver APIs to `jolt-sumcheck`, or define the
   minimum local trait needed for Stage 1 and migrate it once proven.
3. Add kernel trait boundaries and a first slow/reference kernel path.
4. Implement commitment phase with typed committed oracle artifacts.
5. Implement Stage 1 and verify with `jolt-verifier` Stage 1.
6. Implement Stage 2, including product uni-skip and the five-instance batch.
7. Port Stages 3 through 7 incrementally, preserving typed artifacts required
   by later stages.
8. Implement Stage 8 typed opening plan and PCS opening proof.
9. Activate full standard-mode prover/verifier E2E tests.
10. Add ZK/BlindFold proving from retained committed-sumcheck artifacts.
11. Add field-inline prover support once field-inline protocol facts exist in
    `jolt-claims` and guest R1CS rows exist in `jolt-r1cs`.
12. Add Dory-assist prover/boundary support once the Dory-assist protocol is
    concrete.

## Open Design Questions

### Kernel Crate Name

Should the optimized compute crate be named:

```text
jolt-kernels
jolt-prover-kernels
jolt-witness + relation-specific kernel modules
```

The name should make clear that the crate owns prover-side compute, not
protocol semantics.

### Witness Construction Ownership

Some witness construction is reusable and not stage-specific. Decide whether
to introduce a manual `jolt-witness` crate or keep witness helpers in
`jolt-kernels` until the boundary becomes clear.

### Proof Model Ownership

The verifier spec currently keeps `JoltProof` in `jolt-verifier`. Confirm
whether `jolt-prover -> jolt-verifier` proof-type dependency is acceptable for
the first manual prover, or whether a small shared proof crate is worth the
extra churn.

### Generic PCS Scope

The manual prover should be generic over PCS in principle, but Dory is the
first concrete compatibility target. Decide how generic the first milestone
must be before it becomes abstraction overhead.

### Field Inline Product Relation

When field inline is added, decide whether FMUL/FINV reuse existing product
virtualization or require a distinct field-product relation. This should be
settled in the field-inline protocol specs before prover kernels encode a fast
path.

## Review Checklist

- Is `jolt-prover` clearly an orchestration crate?
- Are heavy `compute_message` implementations outside protocol orchestration?
- Can protocol reviewers understand stage order, claims, transcript messages,
  and openings without reading optimized kernels?
- Can optimized kernels be swapped or tested against slow references?
- Does the design avoid Bolt dependencies completely?
- Does `jolt-equivalence` stay thin, with no hidden prover semantic
  reconstruction?
- Are artifacts rich enough for later stages, BlindFold, Dory assist, and
  wrapper work?
- Is the initial proof-type dependency on `jolt-verifier` acceptable and
  isolated?

## References

- [`jolt-verifier` model crate spec](./jolt-verifier-model-crate.md)
- [Field inline, Dory assist, and wrapper pipeline spec](./extended-jolt-field-inline-wrapper.md)
- Existing `jolt-core` prover implementation:
  `jolt-core/src/zkvm/prover.rs`
- Existing `jolt-core` sumcheck prover trait:
  `jolt-core/src/subprotocols/sumcheck_prover.rs`
