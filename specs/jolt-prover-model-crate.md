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

`jolt-verifier` is the protocol source of truth for the modular stack:
proof shape, stage ordering, verifier-visible claims, transcript semantics, and
public proof types are defined there. `jolt-prover` should natively target that
model. It coordinates compute primitives to produce the commitments, sumcheck
round polynomials, bindings, evaluation proofs, and auxiliary artifacts that
`jolt-verifier` checks.

`jolt-prover` is therefore an orchestration crate. It should make proof
dataflow, transcript ordering, stage dependencies, proof assembly, opening
plans, and ZK/recursion boundaries clear while importing verifier-owned proof
types and protocol surfaces where appropriate. Heavy prover-side computation
should live behind kernel interfaces in dedicated compute crates.

## Goal

Build a standalone manual `jolt-prover` crate that:

- computes proofs accepted by the modular `jolt-verifier`;
- treats `jolt-verifier` as the native protocol target and proof-shape owner;
- supports transparent and ZK/BlindFold proof selections as first-class modes
  in the initial crate design;
- supports trusted and untrusted advice as first-class prover inputs and
  committed/opened polynomials;
- models stage-by-stage prover dataflow with typed inputs and outputs;
- keeps transcript-sensitive ordering visible in source;
- emits both public proof data and rich prover artifacts;
- delegates heavy witness, polynomial, and sumcheck-message computation to
  optimized kernel crates;
- uses existing modular protocol crates for formulas, sumcheck proof types,
  openings, PCS proving, and BlindFold;
- separates protocol coordination from backend compute so future prover
  backends can vary in granularity without changing verifier semantics;
- uses `jolt-core` only as a reference oracle in tests and compatibility
  harnesses;
- gives recursion, Dory-assist, BlindFold, and wrapper work a prover-side model
  before Bolt is ready;
- preserves the stage-8 Dory/opening boundary as structured prover data so
  recursion and wrapper tooling do not need to recover it from opaque proof
  bytes.

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
- Do not introduce a separate `jolt-proof` crate for the initial model.
  `jolt-verifier` owns the proof model.
- Do not make `jolt-prover` own wrapper R1CS assembly or wrapper SNARK proving.
  It should export the prover-side witness inputs that `jolt-wrapper` consumes.

## Design Principles

### Verifier Is The Protocol Target

`jolt-prover` should be allowed, and expected, to import `jolt-verifier`.
Verifier-owned proof types and stage-visible protocol structures are the target
that the prover produces.

The dependency is intentionally one-way:

```text
jolt-prover -> jolt-verifier
jolt-verifier -/-> jolt-prover
```

This avoids a premature shared proof crate and keeps one canonical place for
the verifier-visible Jolt proof model. The prover may use `jolt-claims` and
other modular protocol crates where they help compute or name claims, but the
final public artifact is shaped for `jolt-verifier`.

The intended contract is:

```text
jolt-verifier describes what is true and what must be checked.
jolt-prover coordinates computation to produce data satisfying that model.
```

### Prover Is Orchestration

`jolt-prover` owns the proof pipeline:

```text
input resolution
  -> witness/oracle construction
  -> commitments
  -> stages 1..7
  -> stage 8 opening proof and visibility-specific binding
  -> BlindFold proof when the selected mode is ZK
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

- commitment input materialization;
- commitment batch computation;
- sumcheck round-message computation;
- challenge binding over witness state;
- evaluation proof helper computation;
- opening hint/joint polynomial materialization;
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
`jolt-prover`, and especially `jolt-verifier`.

This backend boundary is meant to be flexible. A backend may implement a whole
stage, a single relation's round-message routine, a single polynomial
evaluation primitive, or a PCS-specific opening helper. Changing that
granularity should not change public proof semantics.

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
  generic sumcheck proof types, verifier, committed proof types, R1CS lowering,
  and prover driver

jolt-openings
  evaluation claims, opening plans, RLC reduction

jolt-dory / jolt-hyperkzg
  PCS proving, opening, verification, and setup

jolt-hyrax
  Hyrax commitments/openings and reusable R1CS verifier helpers

jolt-blindfold
  generic BlindFold protocol and verifier/prover machinery

jolt-crypto
  vector commitments, Pedersen commitments, homomorphic commitment operations,
  setup derivation, and committed-row openings

jolt-kernels or jolt-prover-kernels
  optimized prover-side compute kernels

jolt-wrapper
  selected-verifier R1CS assembly and wrapper SNARK backends

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
  selection.rs

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

  advice/
    mod.rs
    inputs.rs
    commit.rs
    artifacts.rs
    openings.rs

  zk/
    mod.rs
    accumulator.rs
    blindfold.rs
    artifacts.rs

  dory_assist/
    mod.rs
    artifacts.rs
    public_inputs.rs
    witness_plan.rs
    witness.rs
    packing.rs
    prove.rs

  wrapper/
    mod.rs
    witness_inputs.rs
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
    selection: ProofSelection,
    kernels: &mut K,
    transcript: &mut T,
    rng: &mut impl CryptoRngCore,
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
- proof selection is explicit;
- kernel set is explicit;
- transcript is threaded visibly;
- RNG is explicit because ZK commitments, BlindFold, and hiding openings need
  auditable entropy plumbing;
- proof and rich artifacts are both returned.

`rng` should be accepted even when proving a transparent proof. Transparent
mode can ignore it, but the API should not need to be redesigned when a caller
switches to BlindFold.

Selection-specific bounds should be enforced by the implementation:

```text
transparent:
  PCS: CommitmentScheme

BlindFold:
  PCS: CommitmentScheme + ZkOpeningScheme<HidingCommitment = VC::Output>
  VC: VectorCommitment<Field = F>
  VC::Output: HomomorphicCommitment<F> + AppendToTranscript
```

`JoltProverPreprocessing<PCS, VC>` should carry or derive the vector-commitment
setup used for committed sumcheck rounds. When possible, use
`VC::Setup: DeriveSetup<PCS::ProverSetup>` so the Pedersen/VC setup is derived
from the same PCS prover setup material rather than configured independently.

### Proof Selection

ZK must be first class in the first version of `jolt-prover`. Do not design a
transparent-only prover and bolt BlindFold on later.

Representative selection:

```rust
pub enum ProofVisibility {
    Transparent,
    BlindFold,
}

pub struct ProofSelection {
    pub visibility: ProofVisibility,
    pub dory_assist: DoryAssistSelection,
    pub wrapper_handoff: WrapperHandoffSelection,
}
```

The exact selection type should align with `jolt-verifier::ProtocolSelection`.
The important rule is that stage orchestration sees the selected visibility at
the same time it proves each stage.

Transparent mode:

```text
sumcheck rounds reveal compressed coefficients
opening claims are public proof data
stage 8 absorbs clear joint claim
no BlindFold proof payload
```

BlindFold mode:

```text
sumcheck rounds reveal VectorCommitment commitments and degrees
coefficients and blindings are retained as private prover artifacts
opening claims are hidden behind committed output-claim rows
stage 8 absorbs the ZK evaluation commitment
BlindFold proves the verifier-equation R1CS
```

The modular prover should avoid a cargo-feature-shaped architecture. Build
flags can gate dependencies or tests if needed, but proof visibility should be
an explicit protocol selection in the model.

### Proof Type Ownership

`JoltProof` lives in `jolt-verifier`, and `jolt-prover` should assemble it
directly. This is not a temporary accident: the prover is natively targeting
the verifier model.

Rules:

- `jolt-verifier` must not depend on `jolt-prover`;
- `jolt-prover` depends on `jolt-verifier` proof and verifier-visible artifact
  types where appropriate;
- proof assembly should still be concentrated in clear modules so stage code
  remains about dataflow, not serialization details;
- introduce a shared proof crate only if the verifier-owned model becomes a
  real architectural blocker.

The point is to keep verifier semantics canonical. `jolt-prover` computes data
that satisfy that canonical shape; it should not define a parallel proof model.

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
    pub selection: ProofSelection,
    pub commitment: CommitmentArtifacts<F, PCS>,
    pub advice: AdviceArtifacts<F, PCS>,
    pub stage1: Stage1Artifacts<F>,
    pub stage2: Stage2Artifacts<F>,
    pub stage3: Stage3Artifacts<F>,
    pub stage4: Stage4Artifacts<F>,
    pub stage5: Stage5Artifacts<F>,
    pub stage6: Stage6Artifacts<F>,
    pub stage7: Stage7Artifacts<F>,
    pub stage8: Option<OpeningArtifacts<F, PCS>>,
    pub zk: Option<BlindFoldArtifacts<F, VC>>,
    pub dory_boundary: Option<DoryBoundaryArtifacts<F, PCS>>,
    pub dory_assist: Option<DoryAssistArtifacts<F, VC>>,
    pub wrapper: Option<WrapperWitnessArtifacts<F>>,
}
```

Artifacts should include named values required by:

- later stage input derivation;
- opening dependency construction;
- transparent-mode opening proof assembly;
- ZK/BlindFold committed-sumcheck witness construction;
- Dory-assist boundary construction;
- Dory verifier trace and witness-plan construction;
- wrapper witness input construction;
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
    selection: ProofVisibility,
    kernels: &mut K,
    transcript: &mut T,
    rng: &mut impl CryptoRngCore,
) -> Result<StageNArtifacts<F>, ProverError>
where
    T: Transcript<Challenge = F>;
```

Stage code should make these visible:

- input claim construction;
- transcript absorbs/squeezes;
- transparent or committed sumcheck proof calls;
- batching order;
- output claim values;
- opening dependencies;
- data retained for ZK/BlindFold when selected.

Stage code may call large kernels, but the stage should still reveal the
protocol structure.

## Sumcheck Proving Flow

A stage proving flow should look like:

```rust
let claims = stage_n::claims(deps, inputs)?;
let mut relation_provers = kernels.stage_n().relation_provers(inputs, deps, &claims)?;

let sumcheck = jolt_sumcheck::BatchedProver::prove(
    selection.visibility,
    claims,
    &mut relation_provers,
    transcript,
    rng,
)?;

let openings = stage_n::opening_dependencies(&sumcheck, deps, inputs)?;

Ok(StageNArtifacts {
    proof: sumcheck.proof,
    challenges: sumcheck.challenges,
    claims: sumcheck.claims,
    openings,
    zk: sumcheck.zk_artifacts,
    kernel_outputs: sumcheck.kernel_outputs,
})
```

The exact API can differ, but the control flow should remain legible:

```text
claims -> relation provers -> batched sumcheck -> openings -> artifacts
```

In transparent mode, `sumcheck.proof` contains clear round coefficients. In
BlindFold mode, `sumcheck.proof` contains commitments/degrees and
`sumcheck.zk_artifacts` carries coefficients, blindings, committed output
claims, input/output constraints, and challenge values for BlindFold witness
construction.

## Advice

Advice must be first class in the initial prover design.

Jolt has two advice classes:

```text
trusted advice:
  commitment is supplied from preprocessing / trusted setup material and
  appended to the transcript when present

untrusted advice:
  commitment is generated by the prover from public proof inputs and included
  in the proof when present
```

Both advice classes need the same typed artifact treatment:

```rust
pub struct AdviceArtifacts<F, PCS> {
    pub trusted: Option<AdvicePolynomialArtifacts<F, PCS>>,
    pub untrusted: Option<AdvicePolynomialArtifacts<F, PCS>>,
}

pub struct AdvicePolynomialArtifacts<F, PCS> {
    pub commitment: PCS::Commitment,
    pub polynomial: AdvicePolynomial<F>,
    pub opening_hint: PCS::OpeningProofHint,
    pub dimensions: AdviceDimensions,
}
```

The exact polynomial representation can be optimized later. The first design
must still preserve:

- advice bytes/words after padding to the declared max advice size;
- trusted and untrusted commitments;
- opening hints needed by stage 8;
- advice dimensions and whether an address phase is required;
- the opening points and claims produced by the RAM value check and advice
  reduction phases;
- the Lagrange selector factor used when embedding smaller advice polynomials
  into the main Dory batch opening;
- BlindFold input/output claim constraints for every advice sumcheck.

Advice affects several prover surfaces:

```text
commitment phase:
  materialize advice polynomial, commit, append transcript message

stage 4:
  RAM val-check formula includes advice/public decomposition

stage 6:
  advice claim-reduction cycle phase for trusted/untrusted when present

stage 7:
  advice address phase when dimensions require it

stage 8:
  advice openings join the ordinary batched opening through selector factors

BlindFold:
  advice claim formulas lower into input/output constraints, including public
  final-scale values
```

The prover should not infer advice presence from legacy proof shape. Advice
presence, dimensions, commitments, and opening dependencies should be typed
inputs/artifacts.

## Opening Proof Flow

Stage 8 should be owned by `jolt-prover` as typed opening proof assembly:

```text
stage artifacts
  -> opening plan
  -> RLC opening claim reduction
  -> joint polynomial/hint construction
  -> PCS opening proof
  -> public proof attachment
  -> typed Dory/opening boundary snapshot
```

The expensive materialization of a joint polynomial or opening hint may be
delegated to kernels or PCS helper APIs, but the opening plan and transcript
ordering should be explicit in `jolt-prover`.

For Dory, stage 8 should also retain the data needed to run the Dory verifier
as a prover-side computation:

- evaluation claims and commitments checked by Dory;
- the Dory proof payload in structured form, not just serialized bytes;
- Dory verifier setup inputs;
- transcript-derived Dory scalars and checkpoints;
- opening IDs and the claim-to-polynomial mapping used by the verifier;
- any opening-plan metadata needed to construct the assist public inputs.

## ZK And BlindFold

ZK/BlindFold is a first-class initial target. The crate may bring up stages
incrementally, but every stage port should implement both visibility modes or
explicitly fail under an unsupported selected mode. Do not merge a stage whose
artifact contract cannot support BlindFold.

When BlindFold is selected:

- stage sumchecks produce committed round messages;
- committed coefficients and blindings are retained for BlindFold;
- committed output-claim rows and their blindings are retained;
- input and output claim constraints are retained from the same formula metadata
  used by transparent proving;
- input/output constraint challenge values are retained in stage artifacts;
- clear opening claims are not exposed in the public proof;
- stage 8 records the hidden joint opening claim and evaluation blinding for
  BlindFold's extra constraint;
- `jolt-prover` constructs the Jolt-specific BlindFold protocol/witness from
  stage artifacts and formula metadata;
- `jolt-blindfold` proves the generic verifier-equation R1CS.

The same `jolt-claims` formula metadata should drive:

- standard prover claim computation;
- standard verifier checks;
- BlindFold R1CS claim constraints;
- wrapper R1CS claim constraints.

This means every relation prover needs two synchronized views:

```text
value computation:
  compute the concrete input/output claims and round messages

constraint metadata:
  describe the same input/output claims as formulas over openings, challenges,
  constants, and public values
```

Changing a claim formula without changing its BlindFold constraint metadata is
a correctness bug.

## Recursion Implications

The recursion spec and paper split ordinary Jolt into:

```text
pi_Jolt = pi_PIOP || pi_Dory
```

where `pi_PIOP` is the commitment plus stages 1-7 sumcheck transcript, and
`pi_Dory` is the stage-8 PCS opening proof. Dory assist keeps stages 1-7 as
native field-arithmetic verifier work and replaces expensive native Dory
verification with an auxiliary proof that the Dory verifier would accept.

That has a direct prover-design consequence: the ordinary `jolt-prover` flow
cannot treat stage 8 as an opaque final append. It must preserve a typed
`DoryBoundaryArtifacts` object that later code can consume without replaying
or reverse-engineering the final proof.

Representative boundary data:

```rust
pub struct DoryBoundaryArtifacts<F, PCS> {
    pub opening_plan: OpeningPlan<F>,
    pub evaluation_claims: Vec<EvaluationClaim<F>>,
    pub commitments: Vec<PCS::Commitment>,
    pub dory_proof: StructuredDoryProof<F, PCS>,
    pub verifier_setup_inputs: DoryVerifierSetupInputs<PCS>,
    pub transcript_checkpoints: DoryTranscriptCheckpoints<F>,
}
```

The exact type names will depend on `jolt-openings` and `jolt-dory`, but the
artifact should carry semantic fields. Recursion should not need a parser over
`JoltProof` to find the Dory verifier's public inputs.

## Dory Assist Prover Boundary

Dory assist may live as `jolt-prover::dory_assist` first, then move to a
dedicated crate if it becomes large. Either way, its role is prover
orchestration, not protocol ownership.

Target flow:

```text
base Jolt stages 1-7
  -> stage-8 Dory/opening boundary snapshot
  -> construct DoryAssistPublicInputs
  -> derive Dory verifier trace plan from the Dory proof and transcript
  -> generate operation-family witnesses
  -> prefix-pack witness polynomials into one dense trace
  -> commit dense trace with Hyrax over Grumpkin
  -> prove Dory-assist stage 1: packed GT exponentiation
  -> prove Dory-assist stage 2: batched operation, wiring, and public-input constraints
  -> prove Dory-assist stage 3: prefix-packing reduction
  -> prove Hyrax dense-trace opening
  -> attach verifier-owned DoryAssistProof payload
```

Protocol facts should live outside the prover:

```text
jolt-claims:
  operation family formulas, wiring formulas, public-input consistency,
  prefix-packing claim formulas, stage shapes, opening IDs

jolt-verifier:
  selected Dory-assist stage schedule and public proof payload shape

jolt-hyrax:
  Hyrax commitment/opening proof and verifier/R1CS helpers

jolt-sumcheck:
  generic sumcheck prover and verifier machinery

jolt-prover:
  orchestration, witness generation calls, transcript threading, proof assembly,
  rich artifacts
```

Heavy Dory-assist compute belongs behind kernels:

- Dory verifier trace extraction from the proof/transcript;
- G1/G2/GT operation witness generation;
- multi-Miller-loop witness generation;
- deterministic wiring edge generation from the Dory verifier computation DAG;
- stage-1/stage-2 sumcheck message computation;
- prefix-code assignment and dense trace materialization;
- Hyrax dense-trace commitment and opening helper computation.

The pairing boundary must be explicit configuration, because the current
recursion branch proves the Miller-loop trace and leaves final exponentiation
as a verifier-side boundary check:

```rust
pub enum PairingBoundaryMode {
    MillerLoopInsideFinalExpOutside,
    FullPairingInside,
}
```

If the protocol later proves final exponentiation inside the assist proof, this
should be a selected verifier/prover configuration change, not an implicit
change hidden in witness code.

## Wrapper Handoff

`jolt-wrapper` owns selected-verifier R1CS assembly and the wrapper SNARK
backend. `jolt-prover` should only produce the private witness inputs and
structured proof/artifact data required by that assembly.

Wrapper handoff shape:

```text
JoltProof
  + JoltProverArtifacts
  + ProtocolSelection
  -> WrapperWitnessInputs
  -> jolt-wrapper assembles selected-verifier R1CS
  -> jolt-wrapper proves with Spartan + HyperKZG or another backend
```

The wrapper witness input should include:

- selected verifier public inputs and preprocessing handles;
- proof fields as verifier-visible values;
- transcript checkpoints or absorbed-message witnesses where needed;
- sumcheck challenge witnesses if the wrapper derives challenges in-circuit;
- Dory-assist public inputs, stage outputs, and Hyrax opening witness data when
  Dory assist is selected;
- BlindFold verifier witness data when ZK Jolt is selected.

The self-contained wrapper target should derive transcript challenges inside
the R1CS. A split mode can expose challenges as public inputs only if the outer
verifier recomputes and binds them.

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
- trusted advice absent/present;
- untrusted advice absent/present;
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

Once full proving exists:

```text
manual jolt-prover proof
  -> manual jolt-verifier
  -> accepted in transparent and BlindFold mode
```

Stage-frontier tests should run both visibility modes as soon as the stage has
the data required by BlindFold. A transparent-only checkpoint is acceptable only
when the unsupported ZK path fails explicitly and the missing artifact is
tracked.

### ZK/BlindFold Tests

BlindFold coverage should be part of the first test plan, not a later audit
pass:

- committed round proof shape matches selected visibility;
- clear round coefficients are absent from public ZK proofs;
- round coefficients, blindings, commitments, and degrees are retained in
  private artifacts;
- committed output claims and blindings are retained in the same block order
  used by the BlindFold R1CS;
- input/output claim constraints match the concrete claim values for every
  relation, including optional advice relations;
- stage-8 extra constraint binds the hidden joint opening claim to the ZK
  evaluation commitment;
- tampering with committed round messages, output-claim commitments, BlindFold
  proof payload, or stage-8 ZK opening data is rejected.

### Advice Tests

Advice coverage should include real consumed advice, not only unused advice
bytes:

- no-advice, trusted-only, untrusted-only, and both-advice cases;
- advice-consuming guests, not just guests that ignore supplied advice;
- max advice sizes with small traces, including trace padding needed to embed
  advice openings into the main Dory opening;
- address-phase and no-address-phase advice dimensions;
- trusted advice commitment mismatch;
- untrusted advice commitment tampering;
- advice opening-point derivation from the unified stage-8 opening point;
- Lagrange selector factor for batching smaller advice polynomials into the
  main opening;
- transparent and BlindFold mode parity for the same advice fixture.

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

### Recursion And Dory-Assist Tests

Once Dory-assist plumbing exists, add tests at the boundary where recursion
depends on prover artifacts:

- stage-8 Dory boundary construction from ordinary opening artifacts;
- Dory-assist public inputs match the selected verifier's expected inputs;
- synthetic Dory verifier traces produce valid operation-family witnesses;
- prefix packing maps heterogeneous virtual polynomials into one dense opening
  claim with the expected selector weights;
- Hyrax commitment/opening verifies for the packed dense trace;
- tampering with Dory proof fields, evaluation claims, transcript checkpoints,
  prefix metadata, or Hyrax opening data is rejected;
- pairing-boundary mode is explicit and mismatch-tested.

The early version can use synthetic Dory verifier DAGs before the full Dory
proof path is wired. The important property is that tests exercise the same
artifact handoff that recursion and wrapping consume.

### Wrapper Handoff Tests

`jolt-prover` should not test wrapper proving, but it should test that its
exported witness inputs are complete and self-consistent:

- selected proof fields are present once and in verifier order;
- transcript witness data matches native transcript replay;
- Dory-assist witness inputs are present only when selected;
- BlindFold witness inputs are present only when selected;
- malformed or incomplete artifact bundles fail before R1CS assembly.

### No Bolt Gates

This spec does not require Bolt tests, generated artifact regeneration, or Bolt
equivalence gates. Bolt can later use these contracts as a target, but that is
not part of this work.

## Milestones

1. Scaffold `jolt-prover` with errors, inputs, preprocessing, artifacts, proof
   selection, advice artifacts, ZK artifacts, proof assembly, explicit RNG, and
   no Bolt dependency.
2. Add prover-side sumcheck driver APIs to `jolt-sumcheck`, or define the
   minimum local trait needed for Stage 1. The driver must support transparent
   and committed/BlindFold modes from the first version.
3. Add kernel trait boundaries and a first slow/reference kernel path.
4. Implement commitment phase with typed committed oracle artifacts and
   trusted/untrusted advice commitment artifacts.
5. Implement Stage 1 in transparent and BlindFold mode and verify with
   `jolt-verifier` Stage 1.
6. Implement Stage 2 in transparent and BlindFold mode, including product
   uni-skip and the five-instance batch.
7. Port Stages 3 through 7 incrementally, with transparent and BlindFold
   artifacts preserved at each checkpoint.
8. Implement advice claim-reduction flows for trusted/untrusted advice,
   including optional address phase and stage-8 opening dependencies.
9. Implement Stage 8 typed opening plan and PCS opening proof in transparent
   and BlindFold mode.
10. Construct and prove the Jolt-specific BlindFold instance from retained
    committed-sumcheck and stage-8 artifacts.
11. Add `DoryBoundaryArtifacts` from ordinary stage-8 opening data.
12. Activate full transparent and ZK prover/verifier E2E tests, including real
    advice fixtures.
13. Add field-inline prover support once field-inline protocol facts exist in
    `jolt-claims` and guest R1CS rows exist in `jolt-r1cs`.
14. Add Dory-assist public-input construction and synthetic trace-plan tests.
15. Add Dory-assist prover orchestration: witness generation, Hyrax commitment,
    assist sumchecks, prefix packing, and dense opening.
16. Add wrapper witness-input export for selected verifier schedules.

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

### Generic PCS Scope

The manual prover should be generic over PCS in principle, but Dory is the
first concrete compatibility target. Decide how generic the first milestone
must be before it becomes abstraction overhead.

### Field Inline Product Relation

When field inline is added, decide whether FMUL/FINV reuse existing product
virtualization or require a distinct field-product relation. This should be
settled in the field-inline protocol specs before prover kernels encode a fast
path.

### Dory Assist Crate Ownership

Start with `jolt-prover::dory_assist` if that keeps the first integration
small. Move to a dedicated crate when operation-family kernels, Hyrax packing,
and recursion artifact types become too large for the base prover crate.

### Pairing Boundary Policy

Decide when the protocol moves from "Miller loop inside, final exponentiation
outside" to "full pairing inside", and make that choice explicit in both
`ProtocolSelection` and prover configuration.

## Review Checklist

- Is `jolt-prover` clearly an orchestration crate?
- Is `jolt-verifier` clearly the canonical proof/protocol target?
- Is the `jolt-prover -> jolt-verifier` dependency one-way and intentional?
- Are heavy `compute_message` implementations outside protocol orchestration?
- Can protocol reviewers understand stage order, claims, transcript messages,
  and openings without reading optimized kernels?
- Can optimized kernels be swapped or tested against slow references?
- Does the design avoid Bolt dependencies completely?
- Does `jolt-equivalence` stay thin, with no hidden prover semantic
  reconstruction?
- Are artifacts rich enough for later stages, BlindFold, Dory assist, and
  wrapper work?
- Is the Dory/opening boundary structured enough for recursion without parsing
  opaque `JoltProof` bytes?
- Does wrapper handoff stop at witness inputs, leaving R1CS assembly to
  `jolt-wrapper`?
- Is the pairing-boundary mode explicit?
- Does backend granularity remain flexible without affecting verifier
  semantics?

## References

- [`jolt-verifier` model crate spec](./jolt-verifier-model-crate.md)
- [Field inline, Dory assist, and wrapper pipeline spec](./extended-jolt-field-inline-wrapper.md)
- Existing `jolt-core` prover implementation:
  `jolt-core/src/zkvm/prover.rs`
- Existing `jolt-core` sumcheck prover trait:
  `jolt-core/src/subprotocols/sumcheck_prover.rs`
- [Recursion references](../recursion_references.md)
- Local recursion paper:
  `/Users/markos/recursion-paper/protocol.tex`
