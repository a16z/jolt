# Spec: `jolt-verifier` Model Crate

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-16 |
| Status | approved |
| PR | TBD |

## Context

Jolt is moving toward a split where prover and verifier code live in separate
crates. Bolt also needs a concrete verifier-shaped artifact to generate toward.
`jolt-verifier` is that handwritten model.

The crate verifies proofs produced by the existing `jolt-core` prover, but it
does not wrap or call the `jolt-core` verifier. It should preserve verifier
semantics and transcript compatibility while using modular crates for the
generic protocol machinery.

The important architectural difference from `jolt-core` is dataflow. The model
verifier should make dependencies explicit with typed inputs and outputs. It
should not rely on a catch-all verifier object, a mutable opening accumulator,
or global Dory layout state.

## Goal

Build a standalone `jolt-verifier` crate that:

- verifies existing Jolt proof artifacts;
- is generic over modular PCS, vector-commitment, field, and transcript traits;
- uses concrete Dory/Pedersen/Blake2b instantiations only in compatibility tests
  backed by real core fixtures;
- keeps all legacy proof-format code under `compat`;
- models stage-by-stage verifier dataflow for future `jolt-prover` /
  `jolt-verifier` splitting;
- gives Bolt a clean verifier output shape to target.

## Non-Goals

- Do not add prover functionality.
- Do not require changing the existing proof format for the first compatible
  verifier.
- Do not replace the existing `jolt-core` verifier in this spec.
- Do not design a stable third-party API before the workspace split is correct.
- Do not introduce generic abstractions before a verifier stage proves they are
  useful.

## Design Principles

### Generic By Default

Verifier code should be parameterized over modular traits rather than hard-code
Dory. Dory is the compatibility instantiation, not the verifier architecture.

The public API should have this shape:

```rust
pub fn verify<F, PCS, VC, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: JoltProof<PCS, VC>,
    io: common::jolt_device::JoltDevice,
    trusted_advice_commitment: Option<PCS::Output>,
    zk: bool,
) -> Result<(), VerifierError>
where
    F: jolt_field::Field,
    PCS: jolt_openings::CommitmentScheme<Field = F>
        + jolt_openings::AdditivelyHomomorphic
        + jolt_openings::ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: jolt_transcript::AppendToTranscript
        + jolt_crypto::HomomorphicCommitment<F>,
    VC: jolt_crypto::VectorCommitment<Field = F>,
    VC::Output: jolt_crypto::HomomorphicCommitment<F>,
    T: jolt_transcript::Transcript<Challenge = F>;
```

`JoltVerifierPreprocessing<PCS, VC>` owns the verifier PCS setup and the optional
vector-commitment setup. When a backend can reuse generators from the PCS setup
source, construct preprocessing through the
`DeriveSetup<PCS::ProverSetup>` path; for Dory this means deriving the Pedersen
setup from the same public URS `g1_vec`/`h1` used by the PCS. The verifier API
still carries the derived `VC::Setup` explicitly because Dory's compact verifier
setup intentionally does not contain the full generator vectors.

The invariant is that verifier stage code is generic, and tests instantiate
concrete stacks such as Dory PCS, Pedersen vector commitments, BN254 scalar
field, and Blake2b transcript.

### Free Function Entry Point

Use a top-level `verify(...)` function. Do not expose a public
`JoltVerifier::new()` / `JoltVerifier::verify()` object API. Also avoid a
private `VerifierState` that simply bundles the whole verifier.

Use small structs for coherent concepts:

- `CheckedInputs`;
- `StageNInputs`;
- `StageNOutput`;
- stage-specific opening views;
- stage-8 final opening claims;
- ZK/BlindFold inputs.

### Compatibility Quarantine

Legacy proof-format behavior belongs in `compat`. The verifier proper should
not inspect legacy IDs, deserialize canonical fields, or reason about legacy
polynomial ordering.

Allowed in `compat`:

- direct aliases/imports of existing `jolt-core` types behind dev/test or an
  explicit compatibility feature;
- legacy IDs;
- canonical serialization shims;
- legacy config structs;
- canonical serialization and conversion for legacy polynomial-order metadata;
- conversion into verifier-owned model types.

Forbidden outside `compat`:

- direct dependency on `jolt-core`;
- `OpeningId` map lookups;
- Dory layout terminology;
- canonical serialization details;
- accumulator-shaped verifier state.

### Typed Opening Dataflow

Legacy standard-mode proofs may arrive as `OpeningId -> claim` maps. That is a
wire compatibility detail. Convert once near the boundary into typed data, then
pass named fields to stages.

`JoltProof` must not retain an ID-keyed opening-claim payload. The native proof
model carries either typed clear claims or a ZK proof payload:

```rust
pub enum JoltProofClaims<F, ZkProof> {
    Clear(ClearProofClaims<F>),
    Zk { blindfold_proof: ZkProof },
}

pub struct ClearProofClaims<F> {
    pub stage1: stage1::inputs::Stage1Claims<F>,
    pub stage2: stage2::inputs::Stage2Claims<F>,
    pub stage3: stage3::inputs::Stage3Claims<F>,
    pub stage4: stage4::inputs::Stage4Claims<F>,
    pub stage5: stage5::inputs::Stage5Claims<F>,
    pub stage6: stage6::inputs::Stage6Claims<F>,
    pub stage7: stage7::inputs::Stage7Claims<F>,
}
```

`compat/` is the only layer allowed to translate a legacy/core
`OpeningId -> claim` map into this explicit structure.

The same rule applies to committed polynomial commitments. Legacy/core proofs
carry commitments as a vector in proof payload order. `compat/` decodes that
vector once into the native typed proof shape:

```rust
pub struct JoltCommitments<C> {
    pub rd_inc: C,
    pub ram_inc: C,
    pub ra: JoltRaCommitments<C>,
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineCommitments<C>,
}

pub struct JoltRaCommitments<C> {
    pub instruction: Vec<C>,
    pub ram: Vec<C>,
    pub bytecode: Vec<C>,
}

#[cfg(feature = "field-inline")]
pub struct FieldInlineCommitments<C> {
    pub field_registers: FieldRegistersCommitments<C>,
}

#[cfg(feature = "field-inline")]
pub struct FieldRegistersCommitments<C> {
    pub rd_inc: C,
}
```

Field-inline register access selectors are not a commitment payload. The
`FieldRs1Ra`, `FieldRs2Ra`, and `FieldRdWa` openings are virtual field-inline
claims. When field inline is enabled, Stage 6 extends `BytecodeReadRaf` so those
virtual openings are checked against the field operands in the selected
bytecode row and are anchored through the existing committed
`BytecodeRa(i)@BytecodeReadRaf -> HammingWeightClaimReduction -> Stage 8` path.
There is no `FieldRegistersRa(i)` commitment.

Verifier preprocessing carries the field-inline bytecode side table when field
inline is enabled. The side table is parallel to ordinary bytecode rows and
contains the field op flags plus optional FR `rd`/`rs1`/`rs2` operand slots.
Stage 6 treats missing field-inline bytecode metadata as a verifier error.

`proof.rs` should not own commitment ordering helpers. The proof payload model
only stores the native shape. `compat/` owns legacy vector decoding, the
transcript preamble owns proof-payload commitment absorption, and Stage 8 owns
the final PCS batch order because it is part of the final opening check.

The final PCS batch order is not proof payload order: core transcript-binds
`RdInc`, `RamInc`, `InstructionRa*`, `RamRa*`, `BytecodeRa*`, but folds final
openings as `RamInc`, `RdInc`, `InstructionRa*`, `BytecodeRa*`, `RamRa*`,
advice. Stage code should not rebuild a `JoltCommittedPolynomial -> C` map or
zip independent ID and commitment vectors.

Clear proof claims are not a generic claim store. Each stage owns the
claims that are actually present in that stage's proof payload, while prior
stage data is consumed through typed stage outputs. For example, Stage 1 owns
the Spartan outer virtual-polynomial claims, including every flag claim, and
Stage 2 derives its RAM/read-write, instruction-reduction, and RAM-RAF input
claims from the verified Stage 1 output rather than duplicating them in
`Stage2Claims`. Stage 3 similarly consumes Stage 1/2 typed outputs for its
input claims and owns only its Stage 3 output openings. Stage 4 consumes the
prior typed outputs needed for register read/write and RAM val-check input
claims, and owns the Stage 4 output openings plus the optional typed advice
opening claims. Stage 5 consumes the verified Stage 2 and Stage 4 outputs it
actually needs, and owns the instruction read-RAF, RAM RA reduction, and
register value-evaluation output openings.

Generic facts:

```rust
// jolt-poly
pub struct Point<F> {
    coords: Vec<F>,
}

// jolt-openings
pub struct EvaluationClaim<F> {
    pub point: Point<F>,
    pub value: F,
}
```

`Point<F>` is intentionally lightweight. It is not parameterized by domain,
polynomial, stage, or endianness. Coordinate order is chosen explicitly at the
boundary with constructors such as `high_to_low`, `low_to_high`, and `concat`.

Jolt-specific opening views should live where they are used. Do not add a
top-level `openings.rs` module unless repeated concrete code justifies it. Good
initial homes are `compat::convert` for legacy decoding and
`stages/stage8/{inputs.rs, outputs.rs, verify.rs}` for the final opening plan.

### Jolt Formulas In `jolt-claims`

Jolt-specific claim math belongs in `jolt-claims`, not in `jolt-openings`,
`jolt-sumcheck`, or ad hoc verifier helper code.

The formula layer should describe the protocol equations. It should provide:

- symbolic expressions over openings, public values, and challenges;
- required-opening and required-challenge metadata;
- consistency claims such as same-evaluation constraints;
- named stage claim expressions for standard verification and for later
  BlindFold lowering;
- protocol-significant ordering metadata, such as the order of claim outputs in
  a batched stage;
- a shape Bolt can emit later.

`jolt-claims` should not become a computation crate. Formula evaluation in the
verifier means evaluating a symbolic expression against values computed by the
owning abstractions:

- `jolt-poly` owns generic MLE, equality, sparse/range, and Lagrange kernels;
- `jolt-sumcheck` owns sumcheck proof verification, reduction return shapes,
  batching coefficients, and batched-instance point slicing;
- `common` / `jolt-riscv` own VM memory layout, public I/O packing, and address
  remapping;
- `jolt-r1cs` owns R1CS matrix and linear-form computation;
- `jolt-blindfold` owns ZK verifier-equation construction from formula
  metadata;
- `jolt-transcript` owns generic transcript append/label mechanics;
- `jolt-verifier` owns transcript sequencing and final equality checks.

The generic `Expr::try_evaluate` API is acceptable because it only interprets a
formula against caller-supplied opening/challenge/public values. Any nontrivial
work needed to produce those values belongs to the abstraction that owns that
work, not to `jolt-claims`.

The intended organization is:

```text
jolt-claims/src/protocols/jolt/
  ids.rs
  stage.rs
  formulas/
    booleanity.rs
    bytecode.rs
    dimensions.rs
    instruction.rs
    ra.rs
    ram.rs
    registers.rs
    spartan.rs
    claim_reductions/
```

The same formula metadata should drive standard verifier checks and ZK
BlindFold construction. Do not duplicate the same formula in separate standard
and ZK implementations.

Stage-specific formulas may name public values such as RAM `UnmapAddress`,
RAM output-check masks, Spartan product `TauKernel`, or Lagrange weights. The
computation of those values still belongs to the relevant lower abstraction:
polynomial kernels in `jolt-poly`, VM layout/IO materialization in
`common`/`jolt-riscv`, sumcheck point slicing in `jolt-sumcheck`, and R1CS
linear algebra in `jolt-r1cs`.

### Sumcheck Model

`jolt-sumcheck` owns generic sumcheck proof data and verification:

- `ClearSumcheckProof<F>`;
- `CompressedSumcheckProof<F>`;
- `CommittedSumcheckProof<C>`;
- `SumcheckProof<F, C>`;
- `SumcheckStatement`, the round-count and degree statement used when scalar
  claims are hidden;
- `SumcheckProof::verify` for clear full-round proofs over an explicit domain;
- `SumcheckProof::verify_compressed_boolean` for single clear compressed
  boolean-hypercube proofs;
- `BatchedSumcheckVerifier::verify_compressed_boolean` for compressed batched
  clear boolean-hypercube proofs;
- `verify_committed_consistency` APIs for committed proof consistency without a
  clear scalar claim;
- `SumcheckDomain`, including `round_sum_coefficients(degree)`, the canonical
  coefficient vector for summing a round polynomial over that domain;
- `BooleanHypercube`;
- `CenteredIntegerDomain`.

Claim-taking verifier APIs are clear-only: they perform scalar round checks and
return evaluation claims. Committed verifier APIs take `SumcheckStatement`
instead of `SumcheckClaim`, check public committed-proof consistency, derive
transcript challenges, and return committed consistency data without absorbing
hidden scalar claims. BlindFold is responsible for the hidden claim relations.
ZK stage code must not manufacture placeholder `SumcheckClaim` values.

R1CS lowering for sumcheck round-sum checks must use the same
`SumcheckDomain::round_sum_coefficients` surface as native verification. This is
the shared hook for Boolean-hypercube and centered-domain uni-skip constraints;
do not reintroduce Boolean-only `s(0) + s(1)` logic in BlindFold lowering.

The verifier owns the Jolt policy:

- standard mode requires all stage sumchecks to be clear and requires standard
  opening claims;
- ZK mode requires all stage sumchecks to be committed and requires BlindFold;
- mixed clear/committed stage proofs are invalid.

Uni-skip is protocol structure, not a special sumcheck primitive. Model it as:

- one sumcheck over a centered integer domain;
- one remainder sumcheck over the boolean hypercube;
- an explicit formula-level handoff check.

### ZK And BlindFold

`jolt-blindfold` owns the generic BlindFold protocol:

- the `BlindFoldProtocol::builder()` construction API that consumes
  already-verified committed sumcheck consistency, explicit committed
  output-claim row ids plus `jolt-sumcheck::CommittedOutputClaims`, explicit
  `jolt-sumcheck::SumcheckDomainSpec` values, public/challenge scalars, and
  final-opening bindings;
- verifier-equation R1CS layout;
- relaxed instance/witness folding;
- vector-commitment openings;
- `jolt_blindfold::verify`.

The native modular BlindFold proof shape is compact and structured. It carries
only non-padding row commitments for the random relaxed witness:
round-coefficient rows, committed output-claim rows, and auxiliary rows. The
verifier reconstructs the full padded witness commitment vector internally by
inserting identity commitments for implicit padding rows. Do not reintroduce a
full padded `random_witness_row_commitments` proof field.

Committed output-claim rows are not anonymous capacity. The builder must reject
extra committed output rows unless they are accounted for by typed opening ids:
`row_count == ceil(opening_ids.len() / row_len)` for non-empty rows, and
`row_count == 0` for empty rows. This keeps the R1CS witness layout and the
committed proof artifact in one-to-one typed correspondence.

Final PCS evaluation bindings are explicit. A final-opening binding allocates
both the hidden evaluation variable and the hidden evaluation-commitment
blinding variable in dedicated padded BlindFold witness rows. Empty final
bindings are rejected. The proof carries fixed-coordinate folded-W openings for
those variables, and the verifier checks both that the opened coordinates match
`folded_eval_outputs` / `folded_eval_blindings` and that the rest of each
dedicated row is zero before the Spartan checks. This directly links the scalar
opened by the PCS hiding evaluation commitment to the value constrained inside
the folded BlindFold witness without revealing unrelated auxiliary witness
material.

`jolt-blindfold` should not re-export `jolt-r1cs` primitives or duplicate
sumcheck statement/input wrappers. Callers import `R1csBuilder`,
`ClaimSourceTable`, and `ClaimSources` from `jolt-r1cs` only when they are
working directly in test or low-level R1CS code. Production construction should
not pass a source table around. Sumcheck commitments, domains, and statements
come from `jolt-sumcheck`; BlindFold owns the source-variable linkage while
building the protocol.

`jolt-verifier` owns the Jolt-specific construction of the BlindFold instance.
Per-stage committed consistency should stay visible in the same linear
`stage*/verify.rs` flow as the clear checks. Shared committed-stage boundary
helpers live under `stages/zk/`, with explicit `inputs.rs` and `outputs.rs`
contracts just like the numbered stages. Do not add a top-level `zk` module or
untyped helper bags for BlindFold wiring.

The ZK verifier should use the same explicit sequencing discipline as the
clear verifier. The production top-level verifier must remain one linear
stage sequence; mode-specific behavior belongs inside typed stage outputs,
typed stage dependencies, and the stage verifier body, not in separate
clear/ZK orchestration paths. Do not add placeholder output structs for hidden
claims just to reserve later BlindFold work; carry committed consistency now
and add typed BlindFold inputs when the lowering consumes them. The ZK work
proceeds in three tracks:

- committed sumcheck track: consume committed sumcheck proofs, check
  Fiat-Shamir transcript consistency, check committed proof statement/order/round
  counts, derive challenges, and return typed outputs. These are internal
  consistency milestones, not production verifier phases. Validity at this
  layer means the public committed transcript is well formed; hidden
  polynomial-evaluation checks are intentionally out of scope until BlindFold is
  wired.
- BlindFold R1CS lowering track: lower the same typed stage claims and
  `jolt-claims` formula metadata into the BlindFold verifier R1CS. This should
  be incremental: add the R1CS inputs and constraints stage by stage, with tests
  that confirm the constructed instance has the expected shape and public
  sources before attempting the full proof.
- final ZK verification track: once the full BlindFold R1CS instance exists,
  convert the BlindFold proof artifact, wire the vector-commitment verifier,
  wire the PCS ZK opening path, append the `BlindFold` domain label to the
  existing verifier transcript, and call `jolt_blindfold::verify` on that same
  transcript. This call is production acceptance only once the modular
  BlindFold R1CS shape is equivalent to the proof artifact being verified.

A full core ZK proof contains one BlindFold proof for the whole Jolt protocol.
It cannot prove only a Stage 1 or Stage 2 prefix unless the fixture generator
also emits prefix BlindFold proofs. Prefix fixtures are useful, but not required
to start wiring ZK consistency milestones. Without prefix fixtures, a consistency
milestone may validate committed consistency and BlindFold R1CS construction,
then stop before calling `jolt_blindfold::verify`.

### Dory And Legacy Polynomial Ordering

There is no modular `DoryLayout`.

Dory verifies a commitment opening at a point. It should not know whether a Jolt
polynomial was linearized cycle-major or address-major before commitment.

The existing proof format contains a legacy ordering byte because current proofs
need it for transcript compatibility and for reconstructing some opening points.
Represent the verifier-owned value on the proof model with the non-Dory
`TracePolynomialOrder` from `jolt-claims`. Keep only legacy conversion shims in
`compat`.

Rules:

- no Dory layout type in stage code;
- no Dory layout type in `jolt-dory`;
- compatibility conversion may use the legacy order to construct typed
  `Point<F>` values;
- `compat` may serialize/deserialize the ordering byte for core artifact
  compatibility, but verifier logic imports the `jolt-claims` protocol type;
- stage 8 passes commitments, points, evaluations, and proof data to `jolt-dory`
  without exposing legacy ordering.

## Modular Crate Responsibilities

| Crate | Responsibility |
|-------|----------------|
| `common` | Public IO types, memory layout, shared constants |
| `jolt-field` | Field traits and concrete scalar fields |
| `jolt-transcript` | Fiat-Shamir transcript traits and implementations |
| `jolt-crypto` | Group, vector commitment, Pedersen, homomorphic commitment traits |
| `jolt-poly` | Polynomial types, evaluation helpers, integer-domain helpers, `Point<F>` |
| `jolt-openings` | `EvaluationClaim<F>`, PCS opening claims, RLC reduction |
| `jolt-sumcheck` | Clear/committed sumcheck proof models, domains, verifiers |
| `jolt-claims` | Symbolic Jolt formulas and formula metadata |
| `jolt-r1cs` | Generic R1CS builders and claim-expression lowering |
| `jolt-blindfold` | Generic BlindFold protocol and verifier |
| `jolt-dory` | Dory PCS commitments, proofs, setup, verification |
| `jolt-riscv` | Instruction identities and final-row semantics |
| `jolt-program` | Reusable program/preprocessing model pieces |
| `jolt-lookup-tables` | Lookup table definitions and final-row routing |

Verifier preprocessing should compose reusable `jolt-program` data with PCS and
vector-commitment setup. Keep it local to `jolt-verifier` only for verifier
setup ownership; program preprocessing itself belongs in `jolt-program`.

## Target Layout

```text
crates/jolt-verifier/src/
  lib.rs
  verifier.rs
  error.rs
  proof.rs
  preprocessing.rs

  compat/
    mod.rs
    codec.rs
    config.rs
    convert.rs
    ids.rs
    preprocessing.rs

  stages/
    mod.rs
    stage1/
      mod.rs
      inputs.rs
      outputs.rs
      verify.rs
    stage2/
      mod.rs
      inputs.rs
      outputs.rs
      verify.rs
    stage3/
      mod.rs
      inputs.rs
      outputs.rs
      verify.rs
    stage4/
      mod.rs
      inputs.rs
      outputs.rs
      verify.rs
    stage5/
      mod.rs
      inputs.rs
      outputs.rs
      verify.rs
    stage6/
      mod.rs
      inputs.rs
      outputs.rs
      verify.rs
    stage7/
      mod.rs
      inputs.rs
      outputs.rs
      verify.rs
    stage8/
      mod.rs
      inputs.rs
      outputs.rs
      verify.rs
    zk/
      mod.rs
      inputs.rs
      outputs.rs
      committed.rs
      blindfold.rs
```

Keep verifier preprocessing in top-level `preprocessing.rs`. Do not add a
top-level `openings.rs` unless there is substantial verifier-owned logic to put
there. Do not add `stages/blindfold.rs` or a top-level `zk` module;
Jolt-specific committed and BlindFold boundary helpers should stay under
`stages/zk/` and preserve explicit typed inputs/outputs.

## Verification Flow

The top-level verifier should make transcript-sensitive order visible:

```rust
pub fn verify<PCS, VC, T>(...) -> Result<(), VerifierError> {
    let checked = validate_inputs(preprocessing, &proof, &io)?;
    validate_proof_consistency(&proof, checked.zk)?;

    let mut transcript = new_transcript();
    bind_preamble(&checked, preprocessing, &proof, &mut transcript)?;
    bind_commitments(&checked, &proof, trusted_advice_commitment, &mut transcript)?;

    let proof_openings = convert_openings(&proof, &checked)?;

    let stage1 = stages::stage1::verify(&checked, preprocessing, &proof, &mut transcript)?;
    let stage2 = stages::stage2::verify(
        &checked,
        preprocessing,
        &proof,
        &mut transcript,
        stages::stage2::deps(&stage1),
    )?;
    let stage3 = stages::stage3::verify(
        &checked,
        preprocessing,
        &proof,
        &mut transcript,
        stages::stage3::deps(&stage1, &stage2),
    )?;
    let stage4 = stages::stage4::verify(..., stages::stage4::deps(&stage1, &stage2, &stage3))?;
    let stage5 = stages::stage5::verify(..., stages::stage5::deps(&stage2, &stage4))?;
    let stage6 = stages::stage6::verify(
        ...,
        stages::stage6::deps(&stage1, &stage2, &stage3, &stage4, &stage5),
    )?;
    let stage7 = stages::stage7::verify(
        ...,
        stages::stage7::deps(&stage1, &stage2, &stage3, &stage4, &stage5, &stage6),
    )?;

    let stage8 = stages::stage8::verify(
        &checked,
        preprocessing,
        &proof,
        trusted_advice_commitment,
        &mut transcript,
        stages::stage8::deps(&stage6, &stage7),
    )?;

    if checked.zk {
        let blindfold = stages::zk::blindfold::build(stages::zk::inputs::BlindFoldInputs {
            checked: &checked,
            preprocessing,
            proof: &proof,
            stage1: ...,
            stage2: ...,
            stage3: ...,
            stage4: ...,
            stage5: ...,
            stage6: ...,
            stage7: ...,
            stage8: ...,
        })?;
        transcript.append(&Label(b"BlindFold"));
        jolt_blindfold::verify::<F, VC, T>(
            &blindfold.protocol,
            proof.blindfold_proof()?,
            vc_setup,
            &mut transcript,
        )?;
        return Ok(());
    }

    Ok(())
}
```

Small helpers are fine for validation and repeated mechanics. Avoid helpers that
hide protocol checks or Fiat-Shamir ordering.

The Stage 1 port established the preferred style for stage code: direct,
linear calls into modular crates with local equality checks and transcript
absorbs visible in protocol order. Verbosity is acceptable when it exposes the
verification argument. Avoid verifier-local helper modules that merely dispatch
between sumcheck domains, wrap one equality check, or hide a transcript append;
move reusable behavior only when a concrete modular crate owns the concept.
Stage folders make the contract explicit: `inputs.rs` contains typed
clear proof claims and dependency structs, `outputs.rs` contains verified
stage outputs, and `verify.rs` contains the protocol flow.

Be conservative when adding new verifier-local structs or helper layers. A new
type should represent data that a later verifier step actually consumes, a
proof artifact that must be preserved, or a named protocol concept owned by the
stage. Do not add structs whose only purpose is to mirror `jolt-claims` opening
IDs, reserve future BlindFold lowering, or make clear/ZK branching look uniform.
When the existing modular APIs already encode the statement, domain, or opening
metadata, use those APIs directly.

The most important invariant is that `stage*/verify.rs` stays explicit and
linear. A stage verifier should read in protocol order:

1. load typed proof payload and typed dependencies;
2. derive dimensions;
3. sample transcript challenges;
4. fetch the relevant `jolt-claims` formula specs;
5. compute any required opening, challenge, and public values through the
   abstractions that own those computations;
6. evaluate the `jolt-claims` expressions against those named values;
7. call `jolt-sumcheck`;
8. compare the sumcheck reduction against the expected claim;
9. append opening claims in transcript order;
10. return typed outputs for later stages.

Do not hide transcript sampling, sumcheck calls, equality checks, or claim
appends behind stage-local helper abstractions. Helpers may prepare typed
values, but the verification argument must remain visible in `verify.rs`.
Prefer one linear stage verifier with narrow `match` blocks at the actual proof
representation boundary over separate clear and ZK helper codepaths.

## Stage Contract

Each stage should be a free function with explicit verifier inputs,
stage-local typed dependencies, and stage-specific output types:

```rust
pub struct Deps<'a, F> {
    pub previous_output: &'a PreviousStageOutput<F>,
}

pub fn deps<F>(previous_output: &PreviousStageOutput<F>) -> Deps<'_, F> {
    Deps { previous_output }
}

pub struct StageNOutput<F> {
    pub challenges: ...,
    pub claims: ...,
    pub opening_dependencies: ...,
    pub zk_claims: ...,
}

pub fn verify<PCS, VC, T>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC>,
    transcript: &mut T,
    deps: Deps<'_, PCS::Field>,
) -> Result<StageNOutput<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>;
```

Do not force a common stage trait or catch-all verifier context before repeated
structure justifies it. Dependencies should name the prior outputs a stage
actually consumes; do not pass all prior stages by default once concrete stage
contracts are known.

Stage implementation rules:

- use `jolt-sumcheck` directly for sumcheck verification; each stage should make
  the expected proof form and domain visible at the call site;
- use `jolt-claims` for Jolt-specific formula declarations, required symbol
  metadata, and consistency-check expressions;
- compute values referenced by those formulas in the crate that owns the
  underlying abstraction; do not place MLE, layout, public-IO materialization,
  sumcheck point slicing, transcript, or R1CS computation in `jolt-claims`;
- use `jolt-openings` for opening facts and final opening reduction;
- use `jolt-r1cs` and `jolt-blindfold` for ZK verifier-equation constraints;
- return named values needed by later stages instead of writing into a map.
- keep local structs only when they name real stage data. Do not add stage-wide
  traits, accumulators, or generic claim-state objects before concrete stages
  prove they remove meaningful complexity.

## Static Boundary Gate

Run the verifier boundary semgrep rules after each stage is wired:

```bash
semgrep --config .semgrep/jolt-verifier-boundaries.yml --error
```

These rules are intentionally structural. They enforce that:

- verifier docs/tests use precise fixture language (`baseline`, `consumer`,
  `lightweight`, etc.) instead of vague test taxonomy;
- `compat` imports and legacy/core-compat names stay under
  `crates/jolt-verifier/src/compat`;
- stage `inputs.rs` files expose typed fields/deps rather than
  `JoltOpeningId` router logic;
- sumcheck calls, formula evaluation, and transcript appends stay visible in
  stage `verify.rs` files;
- `jolt-claims` does not regain VM public-IO/address materialization helpers;
- `jolt-claims` names Spartan product public values but does not compute them.

If a future stage legitimately needs an exception, prefer moving the repeated
operation to the crate that owns the abstraction over weakening the rule.

Before implementing a new stage, do a short boundary investigation for that
stage and update the semgrep gate first when a predictable leak exists. The
investigation should identify:

- which formulas belong in `jolt-claims`;
- which value computations belong to `jolt-poly`, `jolt-sumcheck`,
  `jolt-program`, `jolt-riscv`, `common`, `jolt-r1cs`, or another owning crate;
- which data must be explicit typed proof payload versus prior-stage deps;
- which transcript operations must remain visible in `stage*/verify.rs`;
- any imports, helper names, or router patterns that should be forbidden before
  the implementation starts.

If the investigation reveals a concrete hazard, add or tighten a semgrep rule
in `.semgrep/jolt-verifier-boundaries.yml` before wiring the stage. This makes
the desired boundary fail fast during implementation rather than relying on a
review cleanup afterwards.

## Incremental Testing Strategy

There is no modular `jolt-prover` yet, so native prover/verifier parity is not
available for this crate. Until that exists, completeness is established by
using real `jolt-core` prover output and converting it through `compat` into the
`jolt-verifier` proof/preprocessing model. Soundness is established in two
tracks: transitivity against proofs rejected by the existing core verifier, and
direct tampering of every verifier-owned input field. ZK hiding has a separate
statistical-independence track that checks repeated ZK proofs for the same
program/input are randomized in the proof components that should be blinded.

The integration-test layout should be:

```text
crates/jolt-verifier/tests/
  completeness.rs      # Cargo integration-test target for the folder below
  soundness.rs         # Cargo integration-test target for the folder below
  statistical_independence.rs
  support/
    mod.rs             # shared test-only phase and fixture metadata

  completeness/
    mod.rs
    fixtures.rs
    cases.rs
    standard.rs
    zk.rs
    advice.rs

  soundness/
    mod.rs
    fixtures.rs
    core_transitivity/
      mod.rs
      preamble.rs
      commitments.rs
      proof_shape.rs
      openings.rs
      configs.rs
      zk.rs
    tampering/
      mod.rs
      preamble.rs
      commitments.rs
      proof_shape.rs
      sumcheck.rs
      openings.rs
      configs.rs
      output_claims.rs
      zk.rs

  statistical_independence/
    mod.rs
    fixtures.rs
    zk.rs
```

The root `completeness.rs`, `soundness.rs`, and
`statistical_independence.rs` files exist only because Cargo discovers
integration-test targets at the top level of `tests/`; the test logic belongs
in the corresponding folders. Use `core_transitivity/` for the Rust path of the
core-transitivity soundness track.

### Verifier Phases

The verifier now has full standard and ZK paths, so the integration harness does
not treat "reaching unimplemented code" as success. Active completeness tests
must accept with `Ok(())`; active soundness tests must reject with a concrete
verifier error. `VerifierError::Unimplemented` is a failure in both tracks.

Use a shared phase enum only as test-only metadata for fixture classification,
tamper-manifest coverage, and audit reporting:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VerifierPhase {
    Preamble,
    Commitments,
    Stage1,
    Stage2,
    Stage3,
    Stage4,
    Stage5,
    Stage6,
    Stage7,
    Stage8Openings,
    Zk,
}
```

Each test case declares the verifier phase that first observes the case:

```rust
struct TestCase {
    name: &'static str,
    zk: bool,
    fixture: FixtureId,
    checked_at: VerifierPhase,
}
```

In ZK mode, `Zk` names full BlindFold verification: all committed sumchecks have
checked, Stage 8 has used the PCS ZK opening path, the full BlindFold
protocol/R1CS instance has been built from typed stage outputs, the legacy core
BlindFold artifact has been translated under `compat`, and
`jolt_blindfold::verify` has accepted.

`Stage2` means the full core stage 2 boundary, not only the Spartan product
component: both the product uni-skip prelude and the following five-instance
batched sumcheck must be verified. Do not encode partial product-only progress
as a distinct production phase.

The phase metadata is test-harness metadata only. It must not leak into the
public verifier API, production verifier structs, or production `VerifierError`
variants. The harness interprets results directly:

- valid proof + `Ok(())` passes;
- valid proof + any verifier error fails;
- tampered proof + a concrete verifier error passes;
- tampered proof + `Ok(())` or `VerifierError::Unimplemented` fails.

ZK is wired with clear-style stage organization and full BlindFold proof
verification. The implementation sequence was:

- first, implement the committed sumcheck track stage by stage. Each ZK consistency
  path consumes committed sumcheck proofs, derives transcript challenges, checks
  committed proof statement/order/round counts, and returns typed outputs;
- second, wire the BlindFold R1CS lowering track from the typed stage outputs.
  Each stage contributes committed consistency, output-claim row ids, aliases
  for reused claims, public/challenge values, and `jolt-claims` formula
  metadata to `BlindFoldProtocol::builder()`. Tests should inspect the
  constructed R1CS shape and source coverage before the full proof is verified;
- third, once the full BlindFold R1CS instance is equivalent to the proof
  artifact, wire final ZK verification: convert the proof, verify
  vector-commitment openings, verify the PCS ZK opening path, and call
  `jolt_blindfold::verify`.

Full BlindFold proof verification is active. A full core ZK proof has a single
BlindFold proof for all stages, so integration tests verify the complete ZK
protocol instance against the full BlindFold proof. If we later want isolated
per-stage BlindFold proof verification, the core fixture generator must emit
dedicated prefix BlindFold proofs.

For future ZK changes, every clear stage output must keep the data BlindFold
needs: input claims, output claims, sumcheck points, batching coefficients,
public/advice decompositions, and opening claims used in final expressions.

If finer-grained regression coverage is needed, add `cfg(test)` or
integration-test helper code that calls explicit verifier substeps directly. Do
not add phase fields to production errors just to support the harness.

Core fixture generation is intentionally hardened in the fixture layer. The
test support uses a process-local mutex plus a Unix file lock so separate
nextest processes cannot generate core fixtures concurrently. ZK fixtures also
write a temporary serialized artifact after the first core-accepted generation;
later ZK tests deserialize that artifact instead of repeatedly invoking the
core ZK prover/verifier path. This keeps scheduling and stack pressure out of
the verifier API.

### Completeness

Completeness tests produce or load real proofs from `jolt-core`, convert them
through `compat`, and assert:

```text
if core verifier accepts the proof, jolt-verifier accepts it through the
full verifier path.
```

The fixture matrix should cover:

- standard mode and ZK mode;
- programs with and without trusted advice;
- different public input/output payload sizes, including empty inputs and
  trailing-zero outputs;
- panic and non-panic public I/O;
- small traces that can be generated live in CI;
- larger traces stored as fixtures once regeneration is too expensive;
- memory/RAM-heavy programs, register-heavy programs, instruction-lookup-heavy
  programs, and bytecode edge cases;
- multiple `ReadWriteConfig` and `OneHotConfig` shapes, including boundary
  values.

Recommended standard fixture set:

- `muldiv-guest`, inputs `[9, 5, 3]`: cheap arithmetic/division baseline fixture
  and the default exhaustive tamper base.
- `fibonacci-guest`, inputs `5` and `100`: same program with materially
  different trace lengths; good loop/register pressure with cheap generation.
- `memory-ops-guest`: explicit byte/halfword stores and signed/unsigned loads;
  useful for RAM read/write and output-shape coverage.
- `collatz-guest` with `collatz_convergence(19)` or a similarly small input:
  branch-heavy variable-length loop with 128-bit arithmetic.
- `sha2-guest` or `sha2-chain-guest` with a small iteration count: exercises
  inline-heavy byte-oriented computation without making every test expensive.
- `btreemap-guest` with a small `n` such as `10` or `50`: allocation/collection
  and long-trace coverage; keep it serialized and run as completeness/parity,
  not as the default exhaustive tamper base.
- `advice-demo-guest` or another real advice-using guest once advice fixture
  generation is stable: prefer actual advice consumption over an unused advice
  commitment placeholder.

Fixtures should include enough metadata to be self-describing:

```text
FixtureMetadata
- fixture id
- program/case name
- mode: standard or ZK
- feature set used to generate it
- artifact schema version
- core proof/preprocessing serialization version or source revision
- public I/O digest
- preprocessing digest
- trace length and RAM domain size
- `ReadWriteConfig`, `OneHotConfig`, and trace polynomial order
- whether trusted advice is present
- expected core verifier result
```

Core fixtures should be loaded from serialized artifacts by default. Live
generation is an explicit regeneration path, not the normal test path. The
artifact should contain canonical `jolt-core` bytes for verifier preprocessing,
proof, public I/O, and any trusted-advice commitment material needed by the
core verifier. Tests deserialize those core bytes first, optionally re-run the
core verifier as an oracle, then convert through `compat`.

Regenerate serialized fixtures only when the metadata contract changes: core
proof/preprocessing serialization changes, relevant verifier config changes,
guest bytecode changes, expected public I/O changes, or the target fixture
matrix changes. Store the regeneration command and expected metadata next to
the artifact so a stale fixture fails loudly instead of being silently reused.

Use one cheap fixture, currently `muldiv-guest`, for exhaustive field-by-field
tampering. Use the diverse fixtures for completeness/parity and selected
lightweight tampering. Exhaustively running every tamper mutation against every
guest should be opt-in, not the default harness shape.

The compatibility boundary under test is always:

```text
jolt-core proof/preprocessing/public I/O
  -> compat conversion
  -> JoltVerifierPreprocessing + JoltProof
  -> jolt-verifier::verify(...)
```

### Soundness: Core Transitivity

Core-transitivity tests use `jolt-core` as the oracle. For each invalid proof or
edge case:

```text
if core verifier rejects the artifact, jolt-verifier must also reject once the
artifact is converted through `compat`.
```

This track should include proofs made invalid through mutations to core proof
artifacts before compat conversion, especially:

- public I/O mismatches;
- wrong trusted advice commitment;
- changed preprocessing/verifier setup;
- malformed or inconsistent proof mode;
- missing or extra opening claims in standard mode;
- missing or extra BlindFold proof in ZK mode;
- changed configs, trace length, or RAM domain size;
- tampered commitments, sumcheck proof payloads, opening proof payloads, and
  final output claims.

Intentional hardening differences are allowed only when documented in the test
case metadata. The default rule is strict transitivity: reject if core rejects.

### Soundness: Direct Tampering

Direct tampering tests operate after compat conversion against
`JoltVerifierPreprocessing`, `JoltProof`, public I/O, and verifier inputs. They
should attempt to mutate every field that the verifier consumes, including:

- preamble data: public inputs, public outputs, panic flag, memory layout,
  trace length, RAM domain size, entry address, preprocessing digest;
- commitments: committed-polynomial order, values, duplicates, missing entries,
  trusted advice commitment, untrusted advice commitment;
- proof shape: clear vs committed stage proofs, mixed modes, opening claims in
  ZK mode, BlindFold proofs in standard mode;
- configs: every `JoltReadWriteConfig` and `JoltOneHotConfig` field, zero and
  out-of-range values, truncating numeric conversions;
- sumchecks: claimed sums, round polynomials or round commitments, compressed
  coefficients, first-round uni-skip proofs, final claims, batching
  coefficients once exposed;
- stage claim inputs: stage input/output claim formulas, MLE final claims,
  consistency claims, same-evaluation checks, output-claim checks;
- openings: missing/extra claims, wrong points, wrong values, wrong PCS proof,
  opening order mismatches, duplicate claims;
- ZK: wrong vector-commitment setup, wrong BlindFold proof, tampered committed
  round proofs, mismatched Dory ZK evaluation commitment, and malformed
  BlindFold public inputs.

Every tamper case should declare the verifier phase where the mutation first
becomes observable. For example, a proof-shape mutation is checked at `Preamble`
or `Commitments`; an output-claim mutation is checked at the stage that verifies
the final claim expression or at `Stage8Openings`; a BlindFold mutation is
checked at `Zk`.

### Statistical Independence

The statistical-independence track is a ZK-only regression suite. It is not a
formal proof of zero knowledge; it catches accidental deterministic proof
generation or accidental exposure of clear claims in the modular proof shape.

The default flow is release-only and uses fresh small-cycle ZK fixtures rather
than cached artifacts:

```text
same guest, same public input, same preprocessing
  -> generate independently randomized ZK core proofs
  -> compat conversion
  -> verify every proof through the full modular ZK verifier
  -> keep public/protocol shape stable
  -> bucket blinded proof components and run Dory-style distribution checks
```

The current fixture is the small `muldiv` guest so the trace is cheap enough for
manual release-mode sampling. The test should keep deterministic public material
stable: public I/O,
preprocessing digest, trace length, configs, entry address, and verifier setup.
It should require variation in at least one field from every ZK blinding family
that is present in the fixture:

- hiding PCS commitments produced by `commit_zk`;
- committed sumcheck round commitments;
- committed output-claim commitments;
- BlindFold proof commitments and auxiliary row commitments;
- PCS ZK evaluation commitment or equivalent hiding commitment exposed by the
  final opening proof.

The track should also assert that ZK proofs do not carry clear opening claim
payloads. Every sampled proof must verify successfully before its blinded
components are counted. This is a regression test for randomized proof
generation and accidental leakage in the proof shape; it is not a formal
zero-knowledge proof. When a cheap same-public-IO private-advice fixture exists,
extend this track with a Dory-style two-family comparison across distinct
private witnesses.

### Tamper Manifest

Direct tampering should be driven by an explicit manifest, not by ad hoc test
functions alone. For each stage, enumerate every verifier-owned field and every
typed claim value introduced by that stage. Each entry must classify the target
as exactly one of:

- checked at this stage;
- checked by a later named stage;
- checked by final opening verification;
- checked only by ZK/BlindFold verification;
- not verifier-owned, with the reason documented.

The manifest is test-only infrastructure. It must live under
`crates/jolt-verifier/tests/` or behind `cfg(test)`, and it must operate on the
native verifier proof/preprocessing objects after compat conversion. Do not add
manifest metadata, phase fields, or tamper hooks to the production
verifier API.

Each manifest entry should include:

```text
TamperTarget
- stable name
- mode: standard, ZK, or both
- verifier phase where rejection is first expected
- field path or typed claim path
- mutation strategy
- expected verifier behavior
- reason if the target is deferred or intentionally not verifier-owned
```

Mutation strategies should be simple and deterministic: offset a field element,
flip a boolean, change an enum variant, remove an item, add an extra item,
duplicate an item, swap ordering, truncate a vector, extend a vector, or replace
a proof payload with another well-typed invalid payload. The default fixture
flow is:

```text
real jolt-core proof
  -> compat conversion
  -> native JoltProof/JoltVerifierPreprocessing/public IO
  -> apply one manifest mutation
  -> verify rejection by the full verifier
```

For active targets that also exist in the legacy core proof shape, maintain the
pre-compat transitivity flow:

```text
real jolt-core proof
  -> apply one legacy/core-shape mutation
  -> assert jolt-core rejects or otherwise does not accept the artifact
  -> compat conversion of that same tampered artifact
  -> verify jolt-verifier rejects
```

This second flow is not a replacement for direct post-compat tampering. It
specifically protects the conversion boundary: compat must not normalize away,
drop, alias incorrectly, or repair malformed core proof data before the modular
verifier sees it. When core intentionally omits a claim that compat aliases to a
typed verifier field, the pre-compat test should mutate the legacy source claim
that compat uses for that alias and keep the alias documented in the test
helper.

Do not use synthetic proofs for direct tampering while the modular prover does
not exist. Executable verifier compatibility tests should use real `jolt-core`
proofs converted through `compat`; default builds may keep these tests ignored
until `core-fixtures` is enabled.

When a new typed stage input or output struct is added, the manifest must be
updated in the same change. A field may be deferred, but it may not be omitted.
This keeps the test suite honest about pass-through values such as claims that
are carried by one stage and checked by a later stage.

### Running

The default package test run should keep long-running or future-stage tests
ignored. Focused commands should make it easy to run one track:

```bash
cargo nextest run -p jolt-verifier --cargo-quiet
cargo nextest run -p jolt-verifier --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-verifier --features jolt-core-compat,zk --cargo-quiet
cargo nextest run -p jolt-verifier --features core-fixtures,zk --cargo-quiet
cargo nextest run -p jolt-verifier --release --features core-fixtures,zk zk_muldiv_jolt_proof_components_are_statistically_independent --run-ignored ignored-only --cargo-quiet
```

Use `jolt-core-compat` when only conversion code needs to compile. Use
`core-fixtures` when tests should live-generate `jolt-core` proofs; it enables
`jolt-core-compat` plus the `jolt-core/host` fixture-generation path.
`JOLT_VERIFIER_ZK_STAT_SAMPLES` controls the statistical sample count for the
ignored ZK independence test. It intentionally rejects debug builds unless
`JOLT_VERIFIER_ALLOW_DEBUG_STAT_TESTS=1` is set.

Tests should move from ignored to active when they are cheap and stable enough
for the default package run. Expensive fixture-generation tests can remain
ignored permanently, but their loaded-fixture equivalents should be active for
the full verifier path.

## Implementation Steps

Each step should be reviewed before the next step begins.

### Per-Stage Implementation Workflow

Use this workflow for every new verifier stage:

1. Identify the protocol checks the stage must perform. Write down the input
   claims, output claims, consistency checks, transcript challenges, public
   values, and opening claims the verifier must consume or produce.
2. Define the explicit typed proof payload and typed prior-stage dependencies
   first. Add or update `inputs.rs` and `outputs.rs` before writing the main
   verifier logic.
3. Investigate modular crate boundaries before implementation. Formulas and
   Jolt-specific claim declarations belong in `jolt-claims`; generic MLE,
   equality, Lagrange, range-mask, and point utilities belong in `jolt-poly`;
   sumcheck verification belongs in `jolt-sumcheck`; VM/public-IO material
   belongs in `jolt-program`, `jolt-riscv`, or `common`; R1CS/BlindFold
   constraints belong in `jolt-r1cs` or `jolt-blindfold`.
4. Resolve modular API pressure before adding verifier-local workarounds when
   practical. If a short local workaround is temporarily necessary, record it
   in the API pressure log with the proposed owning-crate fix.
5. Add or tighten semgrep checks before wiring the stage when the boundary
   investigation reveals a predictable leak.
6. Set up completeness and soundness tests before implementing `verify.rs`.
   Completeness should include the real core fixture shape expected to exercise
   the new verifier logic. Soundness should update the tamper manifest for every
   new field and claim, including deferred targets.
7. Leave tests ignored while required fixtures or native support do not exist,
   but make the intended coverage visible in the same change that adds the stage
   surface.
8. Implement `stage*/verify.rs` as explicit linear protocol logic: sample
   transcript challenges, call `jolt-sumcheck`, evaluate `jolt-claims`
   expressions with typed local values, compare claims, append transcript
   values, and return typed outputs.
9. Before adding a helper, wrapper struct, enum, or separate branch, check
   whether it carries real verifier state or only mirrors existing modular
   metadata. Prefer direct calls to the owning modular API and a small local
   `match` at the proof-representation boundary.
10. Unignore or activate tests whose verifier path is now implemented. Debug by
    fixing verifier logic first, then compat translation if the native typed
    payload is wrong.
11. Run the boundary semgrep gate, focused verifier tests, and affected modular
    crate tests before treating the stage as complete.

### 1. Generic Proof Model And Compat Boundary

Objective: make model proof data generic and isolate legacy fields.

Tasks:

- Express `JoltProof` in terms of modular proof data and generic commitment
  types.
- Keep concrete Dory/Pedersen/Blake2b instantiation in compatibility tests that
  use real core fixtures; do not introduce synthetic proof fixtures as a
  substitute for prover output.
- Represent trace polynomial ordering on the verifier-owned proof model with a
  non-Dory name; keep legacy conversion/serialization in `compat`.
- Keep canonical serialization for legacy fields in `compat::codec`.
- Validate proof mode structurally: all clear or all committed.

Review criteria:

- No Dory layout type appears in verifier stage code or modular Dory APIs.
- Proof validation rejects mixed clear/committed stages.
- Compatibility bytes still match existing proof artifacts.

### 2. Verifier Skeleton And Input Resolution

Objective: add the generic top-level verifier shape.

Tasks:

- Define the high-level `verify` API, including verifier preprocessing,
  public I/O, trusted advice commitment, proof, transcript, PCS, and VC
  generic bounds.
- Add `CheckedInputs`.
- Derive proof mode from explicit verifier inputs or proof metadata.
- Bind public preamble fields in prover-compatible order.
- Bind commitments in prover-compatible order.
- Add stage stubs with typed inputs and outputs.

Review criteria:

- No catch-all verifier state.
- Transcript-sensitive operations are visible.
- Generic verifier compiles with the concrete compatibility stack in tests.

### 3. Claim Evaluation Pattern And Compatibility Test Harness

Objective: establish the modular pattern for final sumcheck/MLE claim checks
and set up the completeness/soundness harness against the Step 2 API.

Tasks:

- Define how a stage evaluates `jolt-claims` input and output expressions
  using explicit stage-local claim inputs, not an accumulator or catch-all
  context.
- Define the `jolt-sumcheck` verifier return shape needed by `jolt-verifier`,
  including final evaluation claims, per-instance batching coefficients, and
  per-instance evaluation points.
- Include compressed batched verification support for core-compatible clear
  proofs, since stages such as core stage 2 provide one compressed proof for
  several instances rather than separate uncompressed proofs.
- Document the standard-mode equality:
  `sumcheck_final_claim == evaluated_output_expression`.
- Document the ZK-mode lowering of the same output expression into BlindFold
  final-output constraints.
- Add `tests/completeness/` integration-test skeletons that use real
  `jolt-core` prover output, cast through `compat`, and check that accepted core
  proofs are accepted by the modular verifier.
- Add `tests/soundness/core_transitivity/` skeletons that use `jolt-core`
  rejection as the oracle and require `jolt-verifier` rejection after compat
  conversion.
- Add `tests/soundness/tampering/` skeletons that mutate verifier inputs after
  compat conversion, with each mutation labeled by the first verifier phase that
  must observe it.
- Add shared fixture metadata and phase labels so individual stage tests can be
  audited as each verifier stage lands.

Review criteria:

- The MLE/final-claim pattern is explicit, typed, and local to each stage.
- `jolt-claims` and `jolt-sumcheck` compose without a verifier accumulator.
- Completeness and soundness tests are present as clear targets.
- Every future-stage soundness case records the verifier phase where it becomes
  enforceable.

### 4. Typed Opening Conversion

Objective: convert compatibility opening claims into verifier-native typed
opening inputs without letting legacy IDs or point construction leak into stage
logic.

Tasks:

- Convert legacy standard-mode opening maps into named opening-claim scalars
  keyed by `jolt_claims::protocols::jolt::JoltOpeningId`.
- Define stage-local opening views where first needed. These views should pair
  named opening scalars with stage-derived `jolt_poly::Point<F>` values only
  when the stage has the information needed to choose the point order.
- Add missing/extra checks for selected proof mode and shape.
- In Stage 8, assemble full `jolt_openings::VerifierOpeningClaim<F, C>` values
  directly from typed commitments, stage-derived points, and named opening
  scalars.

Review criteria:

- Stage code receives named opening scalars and stage-local opening views, not
  legacy IDs.
- Opening points use `jolt_poly::Point<F>`.
- Full `EvaluationClaim<F>` values are assembled only once the verifier has
  both the scalar claim and the stage-derived point.
- No top-level opening wrapper is introduced without concrete need.

### 5. Stage 1: Spartan Outer Split

Objective: port the first Spartan stage.

Tasks:

- Verify the centered-domain first-round sumcheck.
- Verify the boolean-hypercube remainder sumcheck.
- Check the handoff between first-round output and remainder input.
- Use `jolt-claims::protocols::jolt::formulas::spartan`.
- Return challenges, opening dependencies, sumcheck input/output claims, and
  the values ZK will later lower into BlindFold constraints.

Review criteria:

- Uni-skip is represented as ordinary sumchecks plus explicit consistency.
- No accumulator dependency.
- No verifier-local sumcheck-dispatch or claim-state helper module.
- Spartan outer claims are named fields, including all circuit flag claims,
  instead of an ID-keyed map or generic claim-state object.
- Proof shape, domain, equality checks, and transcript absorbs are visible in
  `stage1/verify.rs`.
- Any missing modular API is recorded as pressure.

### 6. Stage 2: Product Uni-Skip And Five-Instance Batch

Objective: port core stage 2 faithfully: the Spartan product uni-skip prelude
followed by the compressed batched sumcheck that core verifies in the same
stage.

Tasks:

- Verify the centered-domain first-round product uni-skip proof.
- Derive the product uni-skip input from the stage 1 product output claims and
  the transcript-sampled product high challenge.
- Verify `stage2_sumcheck_proof` as the core-compatible compressed batched
  sumcheck over these five instances:
  - RAM read/write checking;
  - Spartan product virtual remainder;
  - instruction lookup claim reduction;
  - RAM RAF evaluation;
  - public-output check.
- Add or extend `jolt-sumcheck` APIs so batched compressed verification returns
  the common challenge point, per-instance batching coefficients, and any round
  offsets needed to evaluate each instance's final expression explicitly.
- Use `jolt-claims` formulas and owning modular helpers for each instance's
  input and output expressions. Add helpers to the owning crate when verifier
  code would otherwise need to copy protocol math from `jolt-core`.
- Check the product uni-skip handoff into the product remainder instance
  explicitly.
- Return data needed by later stages, stage 8 openings, and eventual BlindFold
  construction: product uni-skip challenge, stage 2 challenge vector, batching
  coefficients, per-instance input claims, per-instance output claims, and
  opening dependencies.

Review criteria:

- Stage 1 and stage 2 outputs compose through typed fields.
- Stage 2 proof claims contain only Stage 2 payload claims; Stage 1-derived
  input claims flow through `Deps`.
- Product-only verification is not considered complete Stage 2 coverage.
- The compressed batch shape and the five instance order are visible in
  `stage2/verify.rs`.
- Batching coefficients and per-instance final-claim checks are explicit and
  typed.
- Formula checks use `jolt-claims` expressions, not copied ad hoc scalar math.
- `stage2/verify.rs` shows transcript challenge sampling, sumcheck calls,
  expression evaluation, equality checks, and opening-claim appends in protocol
  order.
- Formula-value computation is delegated to the owning abstraction rather than
  hidden inside `Stage2Claims`/`Deps` ID routers.

Current implementation status:

- standard Stage 2 is wired through the full core boundary: product uni-skip,
  the five-instance compressed batch, per-instance output expression checks,
  and the combined final-claim check;
- `jolt-claims` exposes the Stage 2 formula helpers and opening-order metadata
  needed by the verifier;
- `jolt-sumcheck` exposes the core-compatible compressed batched verifier
  return shape and checked per-instance point slicing used by Stage 2;
- `common` owns VM memory-layout remapping and public I/O byte-to-word packing;
- `jolt-poly` owns generic MLE helpers used by the formula public-value code;
- standard Stage 1/2/3/4/5/6/7/8 soundness tests now use the tamper manifest to
  mutate real core proofs after compat conversion, covering every current
  Stage 1/2/3/4/5/6/7 verifier-owned proof payload and typed claim target that is
  checked by the full clear verifier, plus Stage 8 commitments and final
  opening proof data;
- unchecked pass-through, advice, and
  ZK/BlindFold targets are explicitly recorded in the tamper manifest rather
  than left implicit;
- the standard verifier path is complete through Stage 8; ZK/BlindFold
  verification is also wired through the full path.

Boundary cleanup pressure:

- move any remaining non-formula computation currently exposed from
  `jolt-claims` to its owning abstraction;
- keep tightening the tamper manifest as new typed stage inputs/outputs are
  introduced so no verifier-owned field is added without a checked, deferred,
  final-opening, ZK-only, or non-verifier-owned classification;
- keep the stage verifier linear after this cleanup: it should still visibly
  sample challenges, verify sumchecks, evaluate formula expressions, compare,
  and append claims.

### 7. Stage 3: Shift, Instruction Input, Register Claim Reduction

Objective: port the stage containing Spartan shift, instruction input, and
register claim-reduction checks.

Tasks:

- Port `ShiftSumcheckVerifier` semantics.
- Port `InstructionInputSumcheckVerifier` semantics.
- Port register claim-reduction semantics.
- Batch claims with modular sumcheck APIs.

Review criteria:

- Batch input/output claim metadata is explicit.
- Stage output provides named dependencies for later stages and ZK.

Current implementation status:

- Stage 3 is organized as `stage3/{inputs.rs, outputs.rs, verify.rs}` and uses
  typed clear claims for:
  - Spartan shift output claims;
  - instruction-input virtualization output claims;
  - register claim-reduction output claims.
- `stage3/verify.rs` is linear and explicit: it samples the shift gamma,
  derives its powers via `jolt-field`, samples the instruction-input gamma and
  register-reduction gamma; verifies the compressed three-instance batch with
  `jolt-sumcheck`; evaluates `jolt-claims` input and output expressions;
  compares the batched final claim; then appends the non-aliased opening claims
  in core transcript order.
- `jolt-claims` now exposes Stage 3 opening-order metadata:
  `shift_input_openings`, `shift_output_openings`,
  `input_virtualization_input_openings`,
  `input_virtualization_output_openings`,
  `input_virtualization_consistency_openings`,
  `registers::claim_reduction_input_openings`, and
  `registers::claim_reduction_output_openings`.
- Stage 3 exposed and fixed an important transcript rule: when a later
  sumcheck opens the same underlying polynomial at the same point, core aliases
  the opening and does not append a duplicate opening claim. The modular
  verifier must still carry explicit typed values for formula evaluation, but
  transcript appends must follow the unique non-aliased core order.
- Standard completeness reaches Stage 3 for real core fixtures, including the
  advice fixture. The synthetic Dory/Pedersen fixture path has been removed so
  proof-shape and tampering checks do not accidentally rely on non-core proof
  data.
- Stage 3 soundness coverage is active for real core fixtures:
  compressed batch round-polynomial tampering, missing/extra round counts, and
  every typed Stage 3 output claim are tampered after compat conversion and
  rejected by the full verifier.
- Stage 3 ZK committed consistency checks
  the three committed sumcheck statements, derives transcript challenges, checks
  the committed output-claim commitment shape under the prevalidated VC setup,
  and returns typed public/consistency output for later stages and the full
  BlindFold path.
- Stage 3 has a dedicated append-order regression for clear opening
  claims. It records the transcript payload sequence and asserts that core
  aliases (`instruction_input.unexpanded_pc`, `registers.rs1_value`,
  `registers.rs2_value`) are carried as typed values for formula evaluation but
  are not appended as duplicate opening claims.

### 8. Stage 4: Register Read/Write And RAM Val Check

Objective: port register read/write checking and RAM value checking.

Tasks:

- Port register read/write checking.
- Build RAM initial state from reusable preprocessing/program data.
- Port RAM val-check formula, including advice/public decomposition.
- Preserve `ram_val_check_gamma` transcript absorption.

Review criteria:

- Advice accumulation is explicit and typed.
- Public values needed by ZK are represented as public inputs, not hidden in
  compatibility state.

Current implementation status:

- Stage 4 is organized as `stage4/{inputs.rs, outputs.rs, verify.rs}` and uses
  typed clear claims for:
  - optional untrusted and trusted advice openings;
  - register read/write output openings;
  - RAM val-check output openings.
- `stage4/verify.rs` is linear and explicit: it samples the register
  read/write gamma, derives the shared read-write address/cycle point from
  Stage 2, materializes the public initial RAM from `jolt-program`, computes
  typed advice contributions with `jolt-poly` block selectors, samples
  `ram_val_check_gamma`, verifies the compressed two-instance batch with
  `jolt-sumcheck`, evaluates the `jolt-claims` register and RAM output
  expressions, compares the combined final claim, then appends Stage 4 opening
  claims in core transcript order.
- `jolt-program` now owns `PublicInitialRam`, so the verifier can evaluate the
  bytecode/public-input contribution without reconstructing VM layout details
  locally. `jolt-poly` owns the generic MSB block-selector MLE helper used for
  advice contributions. `jolt-claims` exposes Stage 4 opening-order metadata
  for register read/write, RAM val-check, and advice openings.
- Stage 4 exposed another transcript gotcha: core appends
  `ram_val_check_gamma` with an empty byte payload, and that empty append still
  advances the transcript. The modular verifier keeps this behavior explicit,
  and a regression test asserts the exact packed label plus empty-byte append
  sequence.
- Stage 4 ZK committed consistency derives
  the same public read/write and RAM output-check points from typed Stage 2 ZK
  output, evaluates the public initial-RAM contribution, verifies the two
  committed sumcheck statements without constructing fake scalar claims, checks
  the committed output-claim commitment shape under the prevalidated VC setup,
  and returns typed public/consistency output for later stages and the full
  BlindFold path.
- Standard completeness reaches Stage 4 for real core fixtures, including the
  advice fixture. Stage 4 soundness coverage is active for real core fixtures:
  compressed batch round-polynomial tampering, missing/extra round counts, every
  typed Stage 4 output claim, and both optional advice claim fields are tampered
  after compat conversion and rejected by the full verifier.

### 9. Stage 5: Instruction Read-RAF, RAM RA Reduction, Register Values

Objective: port the read-RAF and value-evaluation checks in stage 5.

Tasks:

- Port instruction lookup read-RAF checking.
- Port RAM RA claim reduction.
- Port register value-evaluation checking.
- Use lookup and instruction semantics from modular crates where possible.

Review criteria:

- Lookup table semantics come from `jolt-lookup-tables` / `jolt-riscv` where
  available.
- Stage dependencies are explicit fields.

Current implementation status:

- Stage 5 is organized as `stage5/{inputs.rs, outputs.rs, verify.rs}` and uses
  typed clear claims for:
  - instruction read-RAF lookup-table flag openings;
  - instruction read-RAF virtual RA openings;
  - the instruction RAF flag opening;
  - RAM RA claim-reduction output opening;
  - register value-evaluation output openings.
- `stage5/verify.rs` is linear and explicit: it samples the instruction
  read-RAF gamma and RAM RA-reduction gamma, verifies the compressed
  three-instance batch with `jolt-sumcheck`, evaluates the `jolt-claims`
  input/output expressions, checks RAM address consistency across Stage 2 and
  Stage 4, compares the combined final claim, then appends Stage 5 opening
  claims in core transcript order.
- The Stage 5 dependency contract is deliberately narrow:
  `stage5::deps(&stage2, &stage4)`. Stage 5 consumes Stage 2 instruction/RAM
  outputs and Stage 4 RAM/register outputs, and does not receive unrelated
  prior stage outputs.
- `jolt-poly` now owns `OperandPolynomial`/`OperandSide`, used by instruction
  read-RAF to evaluate left/right interleaved operand polynomials. `jolt-claims`
  exposes Stage 5 opening metadata and point normalization helpers for
  instruction read-RAF, RAM RA reduction, and register value evaluation.
  `jolt-lookup-tables` owns lookup-table MLE evaluation.
- Stage 5 preserves the fact that instruction read-RAF publishes different
  opening points for different outputs: lookup-table flags and the RAF flag use
  the cycle point, while each virtual RA claim uses its address chunk plus the
  cycle point. The typed Stage 5 output records those points explicitly for
  final opening verification.
- Stage 5 ZK committed consistency checks
  the three-instance committed batch statement, validates the committed output
  claim chunk count against the hidden Stage 5 output-claim count, and records
  only public transcript/point data plus committed consistency. It does not
  manufacture hidden scalar claims; those relations remain deferred to
  BlindFold.
- Stage 2 and Stage 4 ZK outputs now expose the typed opening points Stage 5
  needs: Stage 2 provides the RAM RAF and RAM read-write opening points, and
  Stage 4 provides the RAM value-check and register read-write opening points.
  This keeps Stage 5 from reaching through generic batch consistency internals
  to recover protocol dependencies.
- Standard completeness reaches Stage 5 for real core fixtures, including the
  advice fixture. Stage 5 soundness coverage is active for real core fixtures:
  compressed batch round-polynomial tampering, missing/extra round counts, all
  read-RAF output claims, RAM RA claim-reduction output, and register
  value-evaluation outputs are tampered after compat conversion and rejected
  by the full verifier.
- The Stage 2 pass-through product remainder claims
  `write_lookup_output_to_rd` and `virtual_instruction` are not consumed by
  Stage 5; their tamper-manifest phase remains deferred to a later
  instruction/bytecode stage.

### 10. Stage 6: Bytecode, Booleanity, RA Virtualization, Increments, Advice Phase 1

Objective: port the broad Stage 6 batch without making the verifier look like a
generic opening-claim router. Stage 6 should be implemented as one batched
sumcheck, but it should be conceptualized and audited as smaller typed
components.

Core analogue:

```rust
let bytecode_read_raf = BytecodeReadRafSumcheckVerifier::gen(...);
let ram_hamming_booleanity =
    HammingBooleanitySumcheckVerifier::new(&opening_accumulator);
let booleanity = BooleanitySumcheckVerifier::new(...);
let ram_ra_virtual = RamRaVirtualSumcheckVerifier::new(...);
let lookups_ra_virtual = LookupsRaSumcheckVerifier::new(...);
let inc_reduction = IncClaimReductionSumcheckVerifier::new(...);
if trusted_advice_commitment.is_some() { push trusted advice cycle phase; }
if proof.untrusted_advice_commitment.is_some() { push untrusted advice cycle phase; }

BatchedSumcheck::verify(&proof.stage6_sumcheck_proof, instances, ...);
```

The modular verifier should preserve that instance order exactly, because the
order determines batching coefficients, transcript challenges, and BlindFold
constraint ordering.

Target Stage 6 proof payload:

```rust
pub struct Stage6Claims<F> {
    pub bytecode_read_raf: BytecodeReadRafOutputOpeningClaims<F>,
    pub ram_hamming_booleanity: RamHammingBooleanityOutputOpeningClaims<F>,
    pub booleanity: BooleanityOutputOpeningClaims<F>,
    pub ram_ra_virtualization: RamRaVirtualizationOutputOpeningClaims<F>,
    pub instruction_ra_virtualization: InstructionRaVirtualizationOutputOpeningClaims<F>,
    pub inc_claim_reduction: IncClaimReductionOutputOpeningClaims<F>,
    pub advice_cycle_phase: Stage6AdviceCyclePhaseClaims<F>,
}
```

`Stage6AdviceCyclePhaseClaims` should have typed `trusted` and `untrusted`
optional fields. Each present advice side should encode whether Stage 6 outputs
a cycle-phase intermediate claim for Stage 7 or directly outputs the final
advice claim when no address phase remains. That decision comes from typed
advice reduction dimensions, not from ad hoc `Option<F>` probing.

Target Stage 6 output:

```rust
pub struct Stage6ClearOutput<F> {
    pub public: Stage6PublicOutput<F>,
    pub output_claims: Stage6Claims<F>,
    pub batch: VerifiedStage6Batch<F>,
}

pub struct Stage6ZkOutput<F, C> {
    pub public: Stage6PublicOutput<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub bytecode_read_raf: BytecodeReadRafPublicOutput<F>,
    pub booleanity: BooleanityPublicOutput<F>,
    pub ram_hamming_booleanity: Stage6SumcheckPublicOutput<F>,
    pub ram_ra_virtualization: RamRaVirtualizationPublicOutput<F>,
    pub instruction_ra_virtualization: InstructionRaVirtualizationPublicOutput<F>,
    pub inc_claim_reduction: Stage6SumcheckPublicOutput<F>,
    pub trusted_advice_cycle_phase: Option<AdviceCyclePhasePublicOutput<F>>,
    pub untrusted_advice_cycle_phase: Option<AdviceCyclePhasePublicOutput<F>>,
}
```

`VerifiedStage6Batch` should expose each component result explicitly: input
claim, sumcheck point, opening point(s), public values used in the output
formula, expected output claim, and the combined batched final claim. It should
not expose a map keyed by `JoltOpeningId`.

The ZK output mirrors only public transcript/opening-point data plus committed
consistency. It must not contain hidden scalar output claims; BlindFold owns
those claim relations.

#### 10.1 Bytecode Read-RAF

Core analogue:
`jolt-core/src/zkvm/bytecode/read_raf_checking.rs`.

Formula owner:
`jolt_claims::protocols::jolt::formulas::bytecode::read_raf`.

Checks:

- sample the Stage 6 bytecode gamma powers in core order:
  outer gamma powers, then stage-specific gammas for prior Stage 1, Stage 2,
  Stage 3, Stage 4, and Stage 5 folded claims;
- compute the input claim from typed prior-stage outputs:
  `SpartanOuter`, `SpartanProductVirtualization`,
  `InstructionInputVirtualization`, `SpartanShift`,
  `RegistersReadWriteChecking`, `RegistersValEvaluation`, and
  `InstructionReadRaf`;
- enforce the formula consistency claim that
  `UnexpandedPC@SpartanShift` and
  `UnexpandedPC@InstructionInputVirtualization` are the same scalar before
  folding the Stage 3 bytecode input;
- include `PC@SpartanOuter`, `PC@SpartanShift`, and the public entry-point
  term exactly as core does;
- after sumcheck, evaluate the bytecode-row public expression at the bytecode
  address point and compare it against the product of
  `BytecodeRa(i)@BytecodeReadRaf` output claims.

Outputs:

- committed `BytecodeRa(i)` claims at the normalized bytecode read-RAF point;
- the bytecode address and cycle point used by Stage 7 hamming-weight claim
  reduction.

Boundary/API pressure before wiring:

- `jolt-claims` exposes typed opening-order helpers for bytecode read-RAF
  input/output claims. The verifier should not recreate private
  `JoltOpeningId` constructors from `bytecode.rs`.
- bytecode row evaluation over an MLE point is exposed as a typed
  `read_raf_public_values` helper. It returns the five stage values plus the
  Spartan RAF and entry terms; verifier code should not interpret
  `JoltInstructionRow` directly.
- `jolt-lookup-tables` provides aggregate `JoltInstruction` lookup-table
  dispatch, so bytecode evaluation does not duplicate instruction-kind
  matching.
- point normalization comes from `BytecodeReadRafDimensions`, matching core's
  address-then-cycle reversal.

#### 10.2 RAM Hamming Booleanity

Core analogue:
`jolt-core/src/zkvm/ram/hamming_booleanity.rs`.

Formula owner:
`jolt_claims::protocols::jolt::formulas::ram::hamming_booleanity`.

Checks:

- input claim is exactly zero;
- derive the reference cycle point from `LookupOutput@SpartanOuter`;
- after sumcheck, compute the public `EqCycle` value between that reference
  cycle point and the Stage 6 hamming-booleanity point;
- verify `EqCycle * (H^2 - H)` for
  `RamHammingWeight@RamHammingBooleanity`.

Outputs:

- virtual `RamHammingWeight@RamHammingBooleanity` at the normalized cycle
  point, consumed by Stage 7 hamming-weight claim reduction.

Boundary/API pressure before wiring:

- `jolt-claims` exposes `hamming_booleanity_output_openings()`.
- use `TraceDimensions::cycle_opening_point`; do not reimplement endianness in
  `jolt-verifier`.

#### 10.3 Canonical RA Booleanity

Core analogue:
`jolt-core/src/subprotocols/booleanity.rs`.

Formula owner:
`jolt_claims::protocols::jolt::formulas::booleanity::booleanity`.

Checks:

- input claim is exactly zero;
- derive the shared address/cycle reference from Stage 5
  `InstructionRa(0)@InstructionReadRaf` in the same way core does;
- sample the booleanity gamma once, then derive per-polynomial weights as
  `gamma^(2i)` over the canonical RA layout:
  `InstructionRa`, then `BytecodeRa`, then `RamRa`;
- after sumcheck, compute the public `EqAddressCycle` and verify the weighted
  sum of `ra_i^2 - ra_i` over every committed RA output claim at
  `Booleanity`.

Outputs:

- committed RA claims for every RA polynomial in the canonical layout at
  `Booleanity`;
- the shared booleanity address/cycle point consumed by Stage 7
  hamming-weight claim reduction.

Boundary/API pressure before wiring:

- `jolt-claims` exposes `booleanity_output_openings(layout, unsigned_inc_chunk_count)`.
  The verifier should consume the canonical `JoltRaPolynomialLayout` and the
  protocol-derived lattice increment chunk count, not assemble unrelated
  vectors by hand.
- if the Stage 5-derived address point is ever shorter than `log_k_chunk`,
  core samples extra transcript challenges. The modular API should make that
  policy explicit before the verifier wires it.

#### 10.4 RAM RA Virtualization

Core analogue:
`jolt-core/src/zkvm/ram/ra_virtual.rs`.

Formula owner:
`jolt_claims::protocols::jolt::formulas::ram::ra_virtualization`.

Checks:

- input claim is `RamRa@RamRaClaimReduction` from Stage 5;
- split the Stage 5 RAM RA claim-reduction opening point into address chunks
  using typed one-hot dimensions;
- after sumcheck, compute the public `EqCycle` between the reduced Stage 5
  cycle point and the Stage 6 RAM RA virtualization point;
- compare against `EqCycle * product(RamRa(i)@RamRaVirtualization)`.

Outputs:

- committed `RamRa(i)@RamRaVirtualization` claims at
  `(address_chunk_i, stage6_cycle)`.

Boundary/API pressure before wiring:

- `jolt-claims` exposes RAM RA virtualization input/output opening helpers.
- address chunking should come from the one-hot/dimension abstraction, not
  verifier-local arithmetic.

#### 10.5 Instruction RA Virtualization

Core analogue:
`jolt-core/src/zkvm/instruction_lookups/ra_virtual.rs`.

Formula owner:
`jolt_claims::protocols::jolt::formulas::instruction::ra_virtualization`.

Checks:

- reconstruct the full instruction address point from the Stage 5 virtual
  `InstructionRa(i)@InstructionReadRaf` openings;
- sample the virtualization gamma and compute the weighted input claim from
  Stage 5 virtual instruction RA claims;
- after sumcheck, compute the cycle equality coefficient and compare against
  the weighted products of committed `InstructionRa` chunks.

Outputs:

- committed `InstructionRa(i)@InstructionRaVirtualization` claims grouped by
  virtual RA chunk.

Boundary/API pressure before wiring:

- `jolt-claims` exposes instruction RA virtualization input/output helpers.
- reuse the Stage 5 multi-point read-RAF output shape rather than reconstructing
  instruction RA chunk points from raw IDs.

#### 10.6 Increment Claim Reduction

Core analogue:
`jolt-core/src/zkvm/claim_reductions/increments.rs`.

Formula owner:
`jolt_claims::protocols::jolt::formulas::claim_reductions::increments`.

Checks:

- sample the increment gamma;
- input claim folds:
  `RamInc@RamReadWriteChecking`,
  `RamInc@RamValCheck`,
  `RdInc@RegistersReadWriteChecking`, and
  `RdInc@RegistersValEvaluation`;
- after sumcheck, compute equality values from each prior cycle point to the
  Stage 6 increment point;
- verify the output formula for reduced `RamInc` and `RdInc`.

Outputs:

- committed `RamInc@IncClaimReduction` and
  `RdInc@IncClaimReduction` at the same normalized cycle point, both consumed
  by Stage 8 opening verification and possibly later reductions.

Boundary/API pressure before wiring:

- `jolt-claims` exposes increment input/output opening helpers.
- all cycle-point construction should use `TraceDimensions`.

#### 10.7 Advice Claim Reduction, Cycle Phase

Core analogue:
`jolt-core/src/zkvm/claim_reductions/advice.rs`.

Formula owner:
`jolt_claims::protocols::jolt::formulas::claim_reductions::advice`.

Checks:

- instantiate a trusted advice cycle-phase instance only when a trusted advice
  commitment is supplied;
- instantiate an untrusted advice cycle-phase instance only when
  `proof.untrusted_advice_commitment` is present;
- input claim is the corresponding advice opening emitted by Stage 4
  `RamValCheck`;
- output is either a cycle-phase intermediate claim for Stage 7 or, if no
  address phase remains, the final advice claim scaled by the typed final-scale
  public value.

Outputs:

- for each present advice side, either
  `AdviceClaimReductionCyclePhase` intermediate output or final
  `AdviceClaimReduction` output.

Boundary/API pressure before wiring:

- deriving `AdviceClaimReductionDimensions` must use
  `AdviceClaimReductionLayout`, `CommitmentMatrixShape`, and
  `TracePolynomialOrder` from `jolt-claims`; it must not import core
  `DoryGlobals` into `jolt-verifier`.
- `AdviceClaimReductionLayout` exposes cycle/address phase round counts,
  active cycle variable extraction, final opening-point normalization, and the
  dummy-gap final-scale factor used by core.
- typed advice payloads should make trusted and untrusted advice independent;
  verifier logic should not share a single untyped advice field.

#### 10.8 Stage 6 Testing Status

Stage 6 is implemented and feeds Stage 7 in the standard path.
The implementation uses the stage-folder shape established for Stages 1-5 and
verifies the whole real core Stage 6 batch as one compressed sumcheck while
keeping every component input and output typed.

Current coverage:

- real-core standard completeness fixtures exercise Stage 6 and the later full
  verifier path;
- direct soundness tampering mutates real core proofs after compat conversion;
- pre-compat transitivity tampering mutates legacy core proofs first, checks
  that core does not accept them, then casts the same tampered artifacts and
  checks modular rejection;
- tampering covers the Stage 6 compressed batch proof payload, missing/extra
  round counts, every scalar/vector field in `Stage6Claims`, and trusted and
  untrusted advice-cycle output branches;
- Stage 6 fixture and tamper cases are active with `Stage6` phase metadata;
- ZK committed consistency is active through Stage 6 on real core ZK fixtures.
  The ZK branch checks the Stage 6 committed batch statement, validates the
  committed output-claim chunk count from typed hidden-output counts, and
  returns typed public opening points consumed by later ZK stages and BlindFold.

Review criteria:

- Stage 6 remains linear at the verifier level: build typed dimensions, sample
  transcript challenges in core order, verify the batched sumcheck, evaluate
  each component formula, compare the combined final claim, append opening
  claims, and return typed outputs.
- Runtime dimensions come from typed config/dimensions, not hardcoded
  constants.
- Bytecode VM semantics, advice layout schedules, and point normalization live
  in their owning modular crates rather than in verifier-local helper logic.
- Advice support is first class and covers trusted/untrusted independently.
- ZK claim metadata covers every included sumcheck before the stage is treated
  as complete.

### 11. Stage 7: Hamming Weight And Advice Phase 2

Objective: port final claim reductions before opening verification.

Core analogue:

```rust
let hamming_weight = HammingWeightClaimReductionVerifier::new(...);
if trusted_advice_has_address_phase { push trusted AdviceClaimReductionVerifier; }
if untrusted_advice_has_address_phase { push untrusted AdviceClaimReductionVerifier; }

BatchedSumcheck::verify(&proof.stage7_sumcheck_proof, instances, ...);
```

Tasks:

- Port hamming-weight claim reduction.
- Port advice address-phase continuation when present.
- Return opening dependencies used by stage 8.

Target Stage 7 proof payload:

```rust
pub struct Stage7Claims<F> {
    pub hamming_weight_claim_reduction: HammingWeightClaimReductionOutputOpeningClaims<F>,
    pub advice_address_phase: Stage7AdviceAddressPhaseClaims<F>,
}
```

The hamming-weight payload should follow canonical RA layout:
`InstructionRa`, then `BytecodeRa`, then `RamRa`. Optional advice fields are
only present for commitments whose `AdviceClaimReductionLayout` still has an
address phase after Stage 6.

Verifier flow:

- derive `JoltFormulaDimensions` and `HammingWeightClaimReductionDimensions`
  from typed verifier config;
- sample the hamming-weight batching gamma before the batched sumcheck, exactly
  as core does;
- compute hamming-weight input from Stage 6 typed outputs:
  RAM hamming-weight, Booleanity RA claims, and virtualization/read-RAF RA
  claims;
- add trusted and untrusted advice address-phase instances only when the
  corresponding advice commitment exists and `address_phase_rounds > 0`;
- verify the Stage 7 compressed batched sumcheck;
- evaluate hamming output using `EqBooleanity` against the Stage 6 Booleanity
  address point and per-polynomial `EqVirtualization` against the Stage 6
  virtualization/read-RAF address points;
- evaluate advice final output with
  `AdviceClaimReductionLayout::address_phase_final_output_scale`, using the
  Stage 4 RAM-val advice opening point and Stage 6 cycle-phase variables;
- append Stage 7 opening claims in core order: all hamming-weight RA outputs,
  then trusted final advice if present, then untrusted final advice if present.

Important offset rule:

The hamming-weight instance uses the standard suffix point from batched
sumcheck verification. Advice address-phase instances use offset `0` in core.
The modular verifier must therefore extract advice address-phase points with
`try_instance_point_at(0, rounds)` rather than treating all instances as suffix
instances. Keep this visible in `stage7/verify.rs`.

Review criteria:

- Optional advice phases are explicit.
- Stage 8 can be built from typed outputs without legacy lookups.
- No generic opening-claim map or compat/core ID router appears in production
  Stage 7 code.

#### 11.1 Stage 7 Testing Status

Stage 7 is implemented and now feeds Stage 8 opening verification. The verifier
uses the same explicit stage-folder pattern as prior stages and verifies the
real core hamming-weight claim reduction plus optional trusted/untrusted advice
address-phase instances as one compressed batch.

Current coverage:

- real-core standard completeness fixtures reach Stage 7 and continue through
  Stage 8, including the advice consumer fixture;
- direct soundness tampering mutates converted native `JoltProof` values after
  compat conversion;
- pre-compat transitivity tampering mutates legacy core proofs first, checks
  that core does not accept them, then casts the same tampered artifacts and
  checks modular rejection;
- tampering covers the Stage 7 compressed batch proof payload, missing/extra
  round counts, every hamming-weight RA output vector, and trusted/untrusted
  advice address-phase final claims when those phases are present;
- Stage 7 fixture and tamper cases are active with `Stage7` phase metadata;
- ZK committed consistency is active through Stage 7 on real core ZK fixtures.
  The ZK branch checks the Stage 7 committed batch statement, validates the
  committed output-claim chunk count from typed hidden-output counts, and
  returns typed public opening points needed by Stage 8/BlindFold.

### 12. Stage 8: Joint Opening Verification

Objective: verify the final opening proof through modular opening/PCS APIs.

Tasks:

- Build final `VerifierOpeningClaim` values from stage outputs.
- Reduce opening claims with `jolt-openings`.
- Verify the PCS opening proof with the generic `PCS`.
- Instantiate Dory only in compatibility tests or aliases.

Review criteria:

- Dory receives only commitments, points, values, and proof data.
- Legacy polynomial ordering does not leak into stage 8.
- Standard-mode opening values are checked explicitly.

Current implementation status:

- Stage 8 is organized as `stage8/{inputs.rs, outputs.rs, verify.rs}`. Its
  dependency contract is narrow: `stage8::deps(&stage6, &stage7)`.
- The verifier derives the final opening claims from typed Stage 6 and Stage 7
  outputs, plus optional trusted/untrusted advice final openings. There is no
  Stage 8 clear claim payload and no ID-keyed opening map in production
  verifier code.
- `compat/convert.rs` owns decoding the legacy proof commitment vector into
  the typed `JoltCommitments` payload. `proof.rs` does not own proof-payload or
  final-opening ordering helpers.
- `stage8/verify.rs` validates the typed commitment layout, explicitly walks
  the core PCS-batch order, appends `rlc_claims` in core transcript order,
  samples gamma powers, combines the commitments with the generic additive PCS
  API, calls `PCS::verify`, and then binds opening inputs through the PCS API.
- The Dory compatibility adapter uses scalar transcript challenges for opening
  verification, matching the core Dory transcript.
- `jolt-openings::ZkOpeningScheme::verify_zk` returns the hidden evaluation
  commitment from the PCS proof, and `bind_zk_opening_inputs` is owned by the
  PCS trait. This keeps Stage 8 generic: it should not reach into Dory proof
  internals to extract `y_com` or know Dory transcript labels.
- Stage 8 ZK opening verification is wired through the generic PCS ZK API. The
  ZK branch builds the same final opening order as clear Stage 8, skips
  `rlc_claims`, combines commitments with the sampled gamma powers, calls
  `PCS::verify_zk`, binds the hidden evaluation commitment, and then returns a
  typed `Stage8Output::Zk` for the future BlindFold lowering.
- Pedersen/vector-commitment transcript payloads must use core's compressed
  curve-point encoding. This is owned by `jolt-crypto`'s BN254 G1/G2
  `AppendToTranscript` implementation; stages should not hand-roll commitment
  serialization.
- The standard path is complete: real core standard completeness fixtures,
  including the advice consumer fixture, are accepted through final opening
  verification.
- Stage 8 soundness coverage is active for real core fixtures: commitment
  order, commitment values, missing/extra commitments, joint opening proof, and
  final opening claim values are tampered after compat conversion and rejected.
  Final opening values are also tampered before compat conversion to guard the
  core-to-modular conversion boundary.
- ZK/BlindFold now reaches full verification on real core ZK fixtures. The
  verifier rejects mixed proof shapes, requires committed stage proofs plus a ZK
  claim payload in ZK mode, validates the vector-commitment setup once against
  the BlindFold generator capacity used by core, checks committed consistency
  through Stage 8's PCS ZK opening path, builds the typed
  `BlindFoldProtocol`/R1CS instance from Stage 1-8 ZK outputs, translates the
  legacy core BlindFold artifact under `compat`, and calls
  `jolt_blindfold::verify`.

### 13. ZK: Committed Consistency, BlindFold R1CS, And Final Verification

Objective: advance ZK verification in three tracks: committed sumcheck consistency,
incremental BlindFold R1CS lowering, and final BlindFold plus PCS ZK opening
verification once the whole protocol instance is available.

Current status:

- The concrete ZK shared code lives either inline in the relevant stage
  verifier or under `stages/zk/` when it is shared committed-boundary logic.
  That module mirrors the stage folder pattern with `inputs.rs`, `outputs.rs`,
  and concrete helpers; there is no top-level `zk` module.
- The top-level verifier enforces committed proof shape in ZK mode and rejects
  clear claims in ZK proofs.
- Clear stages 1-7 already call the clear-only claim-taking sumcheck APIs
  from `jolt-sumcheck`. ZK branches use statement-only committed consistency
  APIs so hidden scalar claims are never represented as placeholder values.
- Stage 1 through Stage 8 committed/ZK consistency is wired directly in the
  corresponding `stage*/verify.rs` files. These branches consume statement-only
  committed sumchecks,
  derive transcript challenges, require the committed output claims to have the
  chunk count implied by the prevalidated VC capacity and typed hidden-claim
  count, and return typed public/consistency outputs for BlindFold construction.
  Stage 8 additionally verifies the final PCS opening in ZK mode and returns
  the hidden evaluation commitment for BlindFold.
- `stages/zk/blindfold.rs` builds the full Jolt BlindFold protocol from typed
  Stage 1-8 ZK outputs. It passes committed consistency, committed output-claim
  rows, explicit opening ids, alias links for reused hidden claims, domain
  specs, public/challenge scalars, and the Stage 8 final-opening binding into
  `BlindFoldProtocol::builder()`. It does not pass maps or source tables through
  `jolt-verifier`; source-variable linkage remains internal to
  `jolt-blindfold`.
- ZK tampering tests now cover committed consistency through Stage 8: missing
  VC setup, Stage 1 committed round count, Stage 2 uni-skip and batch output
  commitment count, Stage 3 batch round count/degree/output commitments, and
  Stage 4, Stage 5, Stage 6, and Stage 7 batch round count/degree/output
  commitments, plus malformed/missing Dory ZK evaluation commitment in the
  joint opening proof.
- `JoltVerifierPreprocessing` carries the vector-commitment setup needed by
  BlindFold. `validate_inputs` checks that setup once against the shared
  BlindFold generator capacity (`MAX_BLINDFOLD_GENERATORS`), and the verifier's
  committed-stage boundary checks use the resulting capacity to validate
  output-claim commitment chunk counts. `jolt-sumcheck` does not know about this
  Jolt-specific VC sizing.
- `compat` converts legacy core BlindFold artifacts into the modular
  `jolt_blindfold::BlindFoldProof` shape during proof conversion. The
  production verifier owns only the modular proof type and calls
  `jolt_blindfold::verify` directly. Real core ZK fixtures now verify through
  this path after aligning core's BlindFold R1CS construction with the modular
  instance.
- The modular crates already provide most generic pieces: committed sumcheck
  proof consistency checks, statement-based BlindFold protocol construction,
  vector-commitment traits, and PCS ZK opening verification hooks.
- `jolt-blindfold` now uses the modular compact proof layout: random relaxed
  witness rows are encoded as separate round/output-claim/auxiliary row
  commitments, with identity padding reconstructed internally. It also binds
  final PCS evaluation outputs and blindings to concrete folded-W witness
  coordinates via additional vector-commitment openings before Spartan
  verification. Those final eval/blinding variables occupy dedicated padded
  witness rows; the verifier rejects non-zero material outside the opened slot
  in those rows.
- Core's BlindFold R1CS construction is aligned to the modular verifier
  instance: initial-claim constraints allocate witness variables when present,
  coefficient rows are not rounded up before output-claim rows, non-uniform
  round degrees drive row width, constants/challenges are baked into linear
  coefficients before R1CS lowering, and auxiliary variables are allocated only
  for products of hidden opening-derived values.
- Stage 2 product virtualization uses distinct public ids for product uni-skip
  Lagrange weights and product-remainder Lagrange weights. This prevents the
  global BlindFold public-value table from aliasing two different challenge
  points that clear verification computes locally.

Stage 8 ZK implementation status:

- `stage8::Deps` and `Stage8Output` are split into clear/ZK variants, mirroring
  Stage 6 and Stage 7. The clear variant keeps scalar opening claims and calls
  `PCS::verify`; the ZK variant carries only public/committed data:
  `opening_ids`, `constraint_coefficients`, final opening points,
  `joint_commitment`, and the PCS hiding evaluation commitment.
- The final opening order is derived from typed Stage 6/7 ZK outputs using the same
  order as clear Stage 8: `RamInc`, `RdInc`, `InstructionRa*`, `BytecodeRa*`,
  `RamRa*`, then optional trusted/untrusted advice.
- In ZK mode, Stage 8 does not append `rlc_claims` and does not construct scalar
  `VerifierOpeningClaim` values. Sample RLC gamma powers directly after the
  final opening list is known, matching core.
- It computes `constraint_coefficients = gamma_i * embedding_scale_i`. Dense
  increment commitments use `Eq(r_address, 0)`. Advice commitments use
  `advice_commitment_embedding_scale` against the advice final opening point.
- It combines commitments with `PCS::combine`, calls `PCS::verify_zk`, then calls
  `PCS::bind_zk_opening_inputs` with the returned hiding evaluation commitment.
- It returns the ZK Stage 8 output to the top-level verifier, which consumes it
  as the final-opening binding when constructing the BlindFold protocol. The
  top-level verifier appends the `BlindFold` domain label to the existing
  Fiat-Shamir transcript and continues that transcript through
  `jolt_blindfold::verify`; imported core ZK fixtures now pass this full path.

BlindFold public-input preparation status:

- The verifier builds the `BlindFoldProtocol` directly with `BlindFoldProtocol::builder()`
  from the explicit `jolt-verifier` stage outputs. The builder consumes
  already-verified committed consistency, committed output-claim row ids,
  committed output-claim row commitments, sumcheck domain specs, public/challenge
  scalars, and final-opening bindings; it does not re-verify committed proofs or
  own transcript replay.
- The builder input includes all seven committed stage proofs and the two
  uni-skip first-round proofs in the same chain order as core. Stage 8 is
  represented as a
  `FinalOpeningBinding`: the protocol relation
  `sum_i constraint_coefficients[i] * opening(opening_ids[i]) == joint_claim`,
  with `joint_claim` hidden behind the PCS evaluation commitment and
  BlindFold folded eval commitments. The final binding also carries the
  evaluation-commitment blinding as a witness variable, and the verifier checks
  fixed-coordinate folded-W openings for both the evaluation and blinding
  variables. These variables are isolated in dedicated witness rows so the
  coordinate openings do not reveal unrelated auxiliary witness values.
- It uses the Stage 8 ZK output's hiding evaluation commitment as the
  final-opening evaluation commitment when constructing `BlindFoldProtocol`.
- It does not introduce generic opening maps in production verifier code. Any
  source table needed for R1CS lowering is internal to `jolt-blindfold` and is
  populated from the builder's explicit output-claim row ids. The verifier only
  supplies typed ids, row lengths, commitments, and formulas.

Tasks:

- Keep core and modular BlindFold R1CS construction equivalent. Any future
  change to stage config ordering, output-claim row layout, witness packing,
  or final-eval coordinate bindings must update both sides or provide an
  explicit compat translation with tests.
- Keep the legacy core BlindFold proof adapter isolated under `compat`, with no
  core BlindFold type leaking into production verifier modules.
- Harden final proof verification by adding native modular fixtures once a
  modular prover exists.
- Add missing modular support before papering over it in `jolt-verifier`:
  transcript labels, challenge-power helpers, claim-expression lowering, and
  final opening commitments should live in their owning crates. Uni-skip and
  centered-domain R1CS round sums now use `jolt-sumcheck::SumcheckDomainSpec`
  plus `round_sum_coefficients`.
- Decide whether to add prefix BlindFold fixtures. If yes, add per-stage
  BlindFold proof verification. If no, keep per-stage coverage focused on
  committed consistency/R1CS construction and verify the full BlindFold proof in
  the complete ZK path.
- Keep ZK completeness and soundness tests active for the full ZK path. The
  current suite accepts real core ZK muldiv fixtures after compat conversion and
  rejects committed-stage, setup, and PCS ZK-eval tampering against the same
  fixture path.
- Add the `statistical_independence/` integration-test track for repeated ZK
  proofs over the same fixture, checking that blinded proof families vary while
  deterministic public metadata stays stable.

Review criteria:

- The same formula metadata drives clear checks and BlindFold lowering.
- Every committed sumcheck through Stage 8 is checked and represented in the
  BlindFold instance.
- Stage inputs and outputs remain typed and explicit; no compatibility-only map
  or phase/test-harness infrastructure leaks into the verifier API.
- No core BlindFold type leaks outside `compat`.
- ZK tests cover both verifier validity and hiding regressions: committed consistency
  soundness, BlindFold proof tampering, PCS ZK opening tampering, and
  statistical independence of randomized proof components.
- Full ZK mode accepts real core ZK fixtures only through the modular BlindFold
  proof verifier, vector commitment setup checks, and PCS ZK opening path.

### 14. End-To-End Compatibility

Objective: activate the full compatibility and soundness harness against
existing prover output.

Tasks:

- Unignore or activate full-path completeness tests for standard and ZK fixtures.
- Run the concrete Dory/Pedersen/Blake2b instantiation through converted
  `jolt-core` fixtures.
- Add `muldiv` coverage in standard and ZK modes.
- Activate core-transitivity tests once the relevant proof fields are consumed.
- Activate direct tampering tests once the relevant proof fields are consumed.
- Document any intentional hardening difference from the existing core verifier.

Review criteria:

- Existing prover proofs verify in `jolt-verifier` through the completeness
  harness.
- Standard and ZK modes are both covered.
- Soundness tests reject tampered artifacts at the expected verifier phases.
- Any intentional hardening difference is documented.

## API Pressure Log

Use this format when a stage reveals an awkward modular boundary:

```text
API Pressure Item
- Location:
- Problem:
- Temporary local workaround:
- Proposed modular crate change:
- Blocking for Bolt? yes/no
```

Expected pressure areas:

- `jolt-dory`: generic verifier/opening ergonomics.
- `jolt-openings`: claim-plan helpers needed by stage 8.
- `jolt-r1cs`: matrix helper or lowering gaps during Spartan/BlindFold wiring.
- `jolt-program`: preprocessing ownership if verifier stages need core-only
  bytecode or RAM structures.
- `jolt-lookup-tables` and `jolt-riscv`: final-row lookup routing and
  instruction semantics.

Do not add speculative abstractions before a concrete stage needs them.

## Validation

For verifier or modular-boundary changes:

```bash
cargo fmt -q --all
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

For focused crate work, run affected package tests, for example:

```bash
cargo nextest run -p jolt-verifier --cargo-quiet
cargo nextest run -p jolt-claims --cargo-quiet
cargo nextest run -p jolt-sumcheck --cargo-quiet
cargo nextest run -p jolt-openings --features test-utils --cargo-quiet
cargo nextest run -p jolt-blindfold --cargo-quiet
```

## Packaging Decisions

- `JoltProof` lives in `jolt-verifier`. Do not introduce a shared `jolt-proof`
  crate for this model.
- Verifier preprocessing should compose reusable `jolt-program` data with
  verifier-owned PCS/VC setup. Program preprocessing semantics stay in
  `jolt-program`; verifier setup ownership stays in `jolt-verifier`.
- Compatibility tests should use real proofs produced by `jolt-core`. Do not
  use synthetic proof fixtures while there is no `jolt-prover` crate producing
  modular-native proofs. Serialized fixtures are allowed when they are real
  `jolt-core` artifacts with regeneration commands and metadata.
- Do not expose convenience aliases for the Dory/Pedersen/Blake2b stack as part
  of the normal public API. Concrete aliases or helper types may exist under
  `cfg(test)` for integration tests.
