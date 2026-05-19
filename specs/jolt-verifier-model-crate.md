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
- stage-8 opening plan;
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
model carries either typed transparent claims or a ZK proof payload:

```rust
pub enum JoltProofClaims<F, ZkProof> {
    Transparent(TransparentProofClaims<F>),
    Zk { blindfold_proof: ZkProof },
}

pub struct TransparentProofClaims<F> {
    pub stage1: stage1::inputs::Stage1Claims<F>,
    pub stage2: stage2::inputs::Stage2Claims<F>,
    pub stage3: stage3::inputs::Stage3Claims<F>,
    pub stage4: stage4::inputs::Stage4Claims<F>,
    // Add later stages as they are wired.
}
```

`compat/` is the only layer allowed to translate a legacy/core
`OpeningId -> claim` map into this explicit structure.

Transparent proof claims are not a generic claim store. Each stage owns the
claims that are actually present in that stage's proof payload, while prior
stage data is consumed through typed stage outputs. For example, Stage 1 owns
the Spartan outer virtual-polynomial claims, including every flag claim, and
Stage 2 derives its RAM/read-write, instruction-reduction, and RAM-RAF input
claims from the verified Stage 1 output rather than duplicating them in
`Stage2Claims`. Stage 3 similarly consumes Stage 1/2 typed outputs for its
input claims and owns only its Stage 3 output openings. Stage 4 consumes the
prior typed outputs needed for register read/write and RAM val-check input
claims, and owns the Stage 4 output openings plus the optional typed advice
opening claims.

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
initial homes are `compat::convert` for legacy decoding and `stages/stage8.rs`
for the final opening plan.

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
- `SumcheckDomain`;
- `BooleanHypercube`;
- `CenteredIntegerDomain`.

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

- committed sumcheck transcript checks;
- verifier-equation R1CS layout;
- relaxed instance/witness folding;
- vector-commitment openings;
- `jolt_blindfold::verify`.

`jolt-verifier` owns the Jolt-specific construction of the BlindFold instance.
That construction should live in `stages/zk.rs`.

### Dory And Legacy Polynomial Ordering

There is no modular `DoryLayout`.

Dory verifies a commitment opening at a point. It should not know whether a Jolt
polynomial was linearized cycle-major or address-major before commitment.

The existing proof format contains a legacy ordering byte because current proofs
need it for transcript compatibility and for reconstructing some opening points.
Represent the verifier-owned value on the proof model with a non-Dory name such
as `TracePolynomialOrder`. Keep only legacy conversion and canonical
serialization shims in `compat`.

Rules:

- no Dory layout type in stage code;
- no Dory layout type in `jolt-dory`;
- compatibility conversion may use the legacy order to construct typed
  `Point<F>` values;
- `compat` may serialize/deserialize the ordering byte for core artifact
  compatibility, but verifier logic imports the verifier-owned proof type;
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
    stage5.rs
    stage6.rs
    stage7.rs
    stage8.rs
    zk.rs
```

Keep verifier preprocessing in top-level `preprocessing.rs`. Do not add a
top-level `openings.rs` unless there is substantial verifier-owned logic to put
there. Do not add `stages/blindfold.rs`; Jolt-specific BlindFold instance
construction is `stages/zk.rs`.

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
    let stage5 =
        stages::stage5::verify(..., stages::stage5::deps(&stage1, &stage2, &stage3, &stage4))?;
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
        &mut transcript,
        stages::stage8::deps(&stage1, &stage2, &stage3, &stage4, &stage5, &stage6, &stage7),
    )?;

    if checked.zk {
        stages::zk::verify(
            &checked,
            preprocessing,
            &proof,
            &mut transcript,
            stages::zk::deps(
                &stage1, &stage2, &stage3, &stage4, &stage5, &stage6, &stage7, &stage8,
            ),
        )?;
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
transparent proof claims and dependency structs, `outputs.rs` contains verified
stage outputs, and `verify.rs` contains the protocol flow.

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
direct tampering of every verifier-owned input field.

The integration-test layout should be:

```text
crates/jolt-verifier/tests/
  completeness.rs      # Cargo integration-test target for the folder below
  soundness.rs         # Cargo integration-test target for the folder below
  support/
    mod.rs             # shared test-only checkpoint and fixture metadata

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
```

The root `completeness.rs` and `soundness.rs` files exist only because Cargo
discovers integration-test targets at the top level of `tests/`; the test logic
belongs in the two folders. Use `core_transitivity/` for the Rust path of the
core-transitivity soundness track.

### Checkpoints

Tests must be stage-gated so they can be unlocked as the verifier is ported.
Use a shared checkpoint enum in the test harness:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VerifierCheckpoint {
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
    Full,
}
```

Each test case declares the first checkpoint where the case is meaningful:

```rust
struct TestCase {
    name: &'static str,
    zk: bool,
    fixture: FixtureId,
    first_checked_at: VerifierCheckpoint,
}
```

The harness has separate configured horizons for standard and ZK mode. The
standard frontier currently reaches `Stage4`; the ZK frontier remains at
`Commitments` while standard stages 5-8 are ported. A completeness case passes
when a valid proof reaches the configured horizon for its mode. Before the full
verifier exists, reaching the next unimplemented stage is success; after the
full verifier exists, success means `Ok(())`.

`Stage2` means the full core stage 2 boundary, not only the Spartan product
component. Do not advance the production frontier to `Stage2` until both the
product uni-skip prelude and the following five-instance batched sumcheck are
verified. The production verifier may validate an early Stage 2 substep before
returning `VerifierError::Unimplemented`, but that is only pre-frontier
hardening. Do not encode partial product-only progress as the production
`Stage2` checkpoint.

A soundness case is skipped or left ignored until
`horizon_for_mode(case.zk) >= first_checked_at`. Once unlocked, the modular
verifier must reject the tampered proof at or before the configured horizon for
that mode. This lets each stage add both positive and negative coverage without
waiting for the whole verifier.

Checkpointing is test-harness metadata only. It must not leak into the public
verifier API, production verifier structs, or production `VerifierError`
variants. The harness owns the current frontier:

```rust
struct VerifierFrontiers {
    standard: VerifierCheckpoint,
    zk: VerifierCheckpoint,
}

const CURRENT_VERIFIER_FRONTIERS: VerifierFrontiers = VerifierFrontiers {
    standard: VerifierCheckpoint::Stage4,
    zk: VerifierCheckpoint::Commitments,
};
```

The harness interprets results relative to the mode-specific frontier:

- valid proof + `Ok(())` passes when the full verifier is implemented;
- valid proof + `VerifierError::Unimplemented` passes only when the configured
  frontier says the proof was expected to reach currently unimplemented code;
- tampered proof whose `first_checked_at <= horizon_for_mode(case.zk)` must
  return a real verifier rejection, not `Ok(())` or the generic unimplemented
  sentinel;
- tampered proof whose first observable checkpoint is after the current frontier
  remains ignored or skipped.

ZK/BlindFold can be wired after the standard stages and standard opening
verification are complete. Prefix BlindFold fixtures are optional, not required
for standard-stage progress. A full core ZK proof has a single BlindFold proof
for all stages, so it is not by itself a partial frontier artifact. While ZK is
deferred, every standard stage output must still preserve the data BlindFold
will need later: input claims, output claims, sumcheck points, batching
coefficients, public/advice decompositions, and opening claims used in final
expressions. Once stages 1-8 are complete, wire `stages/zk.rs` against the full
stage-output data and activate full-verifier ZK tests.

If finer-grained progress is needed before production stages exist, add
`cfg(test)` or integration-test helper code that calls explicit verifier
substeps directly. Do not add checkpoint fields to production errors just to
support the harness.

### Completeness

Completeness tests produce or load real proofs from `jolt-core`, convert them
through `compat`, and assert:

```text
if core verifier accepts the proof, jolt-verifier accepts it through the
configured checkpoint.
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
tampering. Use the diverse fixtures for completeness/parity and selected lightweight
tampering. Exhaustively running every tamper mutation against every guest should
be opt-in, not the default harness shape.

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
relevant checkpoint is implemented.
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

Every tamper case should declare the checkpoint where the mutation must first be
observable. For example, a proof-shape mutation is checked at `Preamble` or
`Commitments`; an output-claim mutation is checked at the stage that verifies
the final claim expression or at `Stage8Openings`; a BlindFold mutation is
checked at `Zk`.

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
manifest metadata, checkpoint fields, or tamper hooks to the production
verifier API.

Each manifest entry should include:

```text
TamperTarget
- stable name
- mode: standard, ZK, or both
- first checkpoint where rejection is required
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
  -> verify rejection at or before the target checkpoint
```

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
```

Use `jolt-core-compat` when only conversion code needs to compile. Use
`core-fixtures` when tests should live-generate `jolt-core` proofs; it enables
`jolt-core-compat` plus the `jolt-core/host` fixture-generation path.

As stages land, tests should move from ignored to active when they are cheap and
stable enough for the default package run. Expensive fixture-generation tests
can remain ignored permanently, but their loaded-fixture equivalents should be
active once the verifier can reach their checkpoint.

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
   Completeness should include the real core fixture shape expected to reach
   the new frontier. Soundness should update the tamper manifest for every new
   field and claim, including deferred targets.
7. Leave future-frontier tests ignored or checkpoint-gated while they cannot
   pass, but make the intended coverage visible in the same change that adds
   the stage surface.
8. Implement `stage*/verify.rs` as explicit linear protocol logic: sample
   transcript challenges, call `jolt-sumcheck`, evaluate `jolt-claims`
   expressions with typed local values, compare claims, append transcript
   values, and return typed outputs.
9. Unignore or activate tests whose checkpoint is now implemented. Debug by
   fixing verifier logic first, then compat translation if the native typed
   payload is wrong.
10. Run the boundary semgrep gate, focused verifier tests, and affected modular
    crate tests before advancing the production frontier.

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

### 3. Claim Evaluation Pattern And Incremental Test Harness

Objective: establish the modular pattern for final sumcheck/MLE claim checks
and set up the incremental completeness/soundness harness against the Step 2
API.

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
  proofs reach the configured verifier checkpoint.
- Add `tests/soundness/core_transitivity/` skeletons that use `jolt-core`
  rejection as the oracle and require `jolt-verifier` rejection once the
  relevant checkpoint is implemented.
- Add `tests/soundness/tampering/` skeletons that mutate verifier inputs after
  compat conversion, with each mutation labeled by the first checkpoint that
  must observe it.
- Add shared fixture metadata and checkpoint gating so individual stage tests
  can be unignored or activated as each verifier stage lands.

Review criteria:

- The MLE/final-claim pattern is explicit, typed, and local to each stage.
- `jolt-claims` and `jolt-sumcheck` compose without a verifier accumulator.
- Completeness and soundness tests are present as clear targets but do not block
  the partially ported verifier.
- Every future-stage soundness case records the checkpoint where it becomes
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
- Add an `OpeningPlan` skeleton in or near `stage8.rs` that assembles full
  `jolt_openings::EvaluationClaim<F>` / verifier opening claims from typed
  commitments, stage-derived points, and named opening scalars.

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
- Product-only verification is not considered a complete `Stage2` checkpoint.
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
- standard Stage 1/2/3/4 soundness tests now use the tamper manifest to mutate real
  core proofs after compat conversion, covering every current Stage 1/2/3/4
  verifier-owned proof payload and typed claim target that is checked by the
  `Stage4` frontier;
- unchecked pass-through, final-opening, commitment-payload/order, advice, and
  ZK/BlindFold targets are explicitly recorded in the tamper manifest rather
  than left implicit;
- the standard verifier frontier is now `Stage4`; Stage 5 remains the next
  unimplemented standard-stage boundary.

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
  typed transparent claims for:
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
  rejected before the current frontier.
- Stage 3 has a dedicated append-order regression for transparent opening
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
  typed transparent claims for:
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
- Standard completeness reaches Stage 4 for real core fixtures, including the
  advice fixture. Stage 4 soundness coverage is active for real core fixtures:
  compressed batch round-polynomial tampering, missing/extra round counts, every
  typed Stage 4 output claim, and both optional advice claim slots are tampered
  after compat conversion and rejected before the current frontier.

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

### 10. Stage 6: Bytecode, Booleanity, RA Virtualization, Increments, Advice Phase 1

Objective: port the broad stage 6 batch.

Tasks:

- Port bytecode read-RAF checking.
- Port RAM hamming booleanity and generic booleanity checks.
- Port RAM RA virtualization and instruction RA virtualization.
- Port increment claim reduction.
- Port advice claim reduction phase 1 for trusted and untrusted advice.

Review criteria:

- Runtime dimensions come from typed config/dimensions, not hardcoded constants.
- Advice support is first class.
- ZK claim metadata covers every included sumcheck.

### 11. Stage 7: Hamming Weight And Advice Phase 2

Objective: port final claim reductions before opening verification.

Tasks:

- Port hamming-weight claim reduction.
- Port advice address-phase continuation when present.
- Return opening dependencies used by stage 8.

Review criteria:

- Optional advice phases are explicit.
- Stage 8 can be built from typed outputs without legacy lookups.

### 12. Stage 8: Joint Opening Verification

Objective: verify the final opening proof through modular opening/PCS APIs.

Tasks:

- Build `OpeningPlan` from stage outputs.
- Reduce opening claims with `jolt-openings`.
- Verify the PCS opening proof with the generic `PCS`.
- Instantiate Dory only in compatibility tests or aliases.

Review criteria:

- Dory receives only commitments, points, values, and proof data.
- Legacy polynomial ordering does not leak into stage 8.
- Standard-mode opening values are checked explicitly.

### 13. ZK: BlindFold Instance Construction

Objective: verify committed proofs with BlindFold.

Tasks:

- Add `stages/zk.rs`.
- Construct `jolt_blindfold::BlindFoldProtocol` from stage outputs and formulas.
- Convert existing BlindFold proof artifacts under `compat`.
- Call `jolt_blindfold::verify`.

Review criteria:

- The same formula metadata drives native and BlindFold checks.
- Every committed sumcheck is represented.
- No core BlindFold type leaks outside `compat`.

### 14. End-To-End Compatibility

Objective: activate the full compatibility and soundness harness against
existing prover output.

Tasks:

- Unignore or activate the full-checkpoint completeness tests for standard and
  ZK fixtures.
- Run the concrete Dory/Pedersen/Blake2b instantiation through converted
  `jolt-core` fixtures.
- Add `muldiv` coverage in standard and ZK modes.
- Activate core-transitivity tests whose checkpoints are fully implemented.
- Activate direct tampering tests whose checkpoints are fully implemented.
- Document any intentional hardening difference from the existing core verifier.

Review criteria:

- Existing prover proofs verify in `jolt-verifier` through the completeness
  harness.
- Standard and ZK modes are both covered.
- Soundness tests reject tampered artifacts at the correct checkpoints.
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
