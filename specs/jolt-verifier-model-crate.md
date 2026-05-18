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
- uses concrete Dory/Pedersen/Blake2b instantiations in compatibility tests and
  convenience aliases;
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
- legacy polynomial-order enum;
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

The formula layer should provide:

- symbolic expressions over openings, public values, and challenges;
- required-opening and required-challenge metadata;
- consistency claims such as same-evaluation constraints;
- native scalar evaluation for verifier checks;
- R1CS lowering for BlindFold;
- a shape Bolt can emit later.

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
Represent that under `compat` with a non-Dory name such as
`LegacyCoreLayout` or `TracePolynomialOrder`.

Rules:

- no Dory layout type in stage code;
- no Dory layout type in `jolt-dory`;
- compatibility conversion may use the legacy order to construct typed
  `Point<F>` values;
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
    layout.rs
    preprocessing.rs

  stages/
    mod.rs
    stage1.rs
    stage2.rs
    stage3.rs
    stage4.rs
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

- use `jolt-sumcheck` for sumcheck verification;
- use `jolt-claims` for Jolt-specific formulas and consistency checks;
- use `jolt-openings` for opening facts and final opening reduction;
- use `jolt-r1cs` and `jolt-blindfold` for ZK verifier-equation constraints;
- return named values needed by later stages instead of writing into a map.

## Implementation Steps

Each step should be reviewed before the next step begins.

### 1. Generic Proof Model And Compat Boundary

Objective: make model proof data generic and isolate legacy fields.

Tasks:

- Express `JoltProof` in terms of modular proof data and generic commitment
  types.
- Keep concrete Dory/Pedersen/Blake2b instantiation in tests and aliases.
- Move the legacy layout/order enum under `compat::layout`.
- Keep canonical serialization for legacy fields in `compat::codec`.
- Validate proof mode structurally: all clear or all committed.

Review criteria:

- No Dory layout type appears outside `compat`.
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

### 3. Claim Evaluation Pattern And Differential Test Harness

Objective: establish the modular pattern for final sumcheck/MLE claim checks
and set up the target end-to-end tests against the Step 2 API.

Tasks:

- Define how a stage evaluates `jolt-claims` input and output expressions
  using explicit stage-local resolved claims, not an accumulator or catch-all
  context.
- Define the `jolt-sumcheck` verifier return shape needed by `jolt-verifier`,
  including final evaluation claims, per-instance batching coefficients, and
  per-instance evaluation points.
- Document the standard-mode equality:
  `sumcheck_final_claim == evaluated_output_expression`.
- Document the ZK-mode lowering of the same output expression into BlindFold
  final-output constraints.
- Add ignored integration-test skeletons for standard and ZK compatibility
  against real `jolt-core` prover output.
- Add ignored differential-test skeletons that compare old core verifier
  acceptance against `jolt-verifier` acceptance once implemented.

Review criteria:

- The MLE/final-claim pattern is explicit, typed, and local to each stage.
- `jolt-claims` and `jolt-sumcheck` compose without a verifier accumulator.
- E2E tests are present as clear targets but do not block the partially ported
  verifier.

### 4. Typed Opening Conversion

Objective: convert compatibility opening claims into typed stage inputs.

Tasks:

- Convert legacy standard-mode opening maps into `EvaluationClaim<F>` values.
- Define stage-local opening views where first needed.
- Add missing/extra checks for selected proof mode and shape.
- Add an `OpeningPlan` skeleton in or near `stage8.rs`.

Review criteria:

- Stage code receives named openings, not legacy IDs.
- Opening points use `jolt_poly::Point<F>`.
- No top-level opening wrapper is introduced without concrete need.

### 5. Stage 1: Spartan Outer Split

Objective: port the first Spartan stage.

Tasks:

- Verify the centered-domain first-round sumcheck.
- Verify the boolean-hypercube remainder sumcheck.
- Check the handoff between first-round output and remainder input.
- Use `jolt-claims::protocols::jolt::formulas::spartan`.
- Return challenges, opening dependencies, and ZK claim metadata.

Review criteria:

- Uni-skip is represented as ordinary sumchecks plus explicit consistency.
- No accumulator dependency.
- Any missing modular API is recorded as pressure.

### 6. Stage 2: Spartan Product Split

Objective: port the product/remaining Spartan stage.

Tasks:

- Verify the centered-domain first-round product sumcheck.
- Verify the boolean-hypercube product remainder.
- Use `jolt-r1cs` helpers where they fit.
- Return data needed by later stages and ZK.

Review criteria:

- Stage 1 and stage 2 outputs compose through typed fields.
- Formula checks use `jolt-claims`, not copied ad hoc scalar math.

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

Objective: verify existing prover output.

Tasks:

- Add a minimal compatibility fixture path.
- Add at least one concrete Dory/Pedersen/Blake2b test instantiation.
- Add `muldiv` coverage in standard and ZK modes.
- Compare failures against the existing verifier where practical.

Review criteria:

- At least one existing prover proof verifies in `jolt-verifier`.
- Standard and ZK modes are both covered.
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
- Verifier preprocessing should be derived from `jolt-program` data or live in
  `jolt-program` when the semantics are shared by prover and verifier.
  `jolt-verifier` should not grow a parallel preprocessing model unless the data
  is verifier-specific.
- Differential compatibility tests should use real proofs produced by
  `jolt-core`. Do not use synthetic proof fixtures while there is no
  `jolt-prover` crate producing modular-native proofs.
- Do not expose convenience aliases for the Dory/Pedersen/Blake2b stack as part
  of the normal public API. Concrete aliases or helper types may exist under
  `cfg(test)` for integration tests.
