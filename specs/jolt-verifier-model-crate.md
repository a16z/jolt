# Spec: `jolt-verifier` Model Crate

| Field       | Value |
|-------------|-------|
| Author(s)   | Markos Georghiades, Codex |
| Created     | 2026-05-16 |
| Status      | draft |
| PR          | TBD |

## Summary

Jolt is moving toward splitting today's monolithic `jolt-core` crate into
separate prover and verifier crates. In parallel, the Bolt compiler work is
intended to emit verifier-oriented code against the newer modular crates. The
compiler target is hard to reason about in the abstract: producing code that is
semantically equivalent to the bespoke `jolt-core` verifier, but shaped around
the modular crates, requires a concrete model.

This spec proposes a handwritten `jolt-verifier` crate that verifies proofs
produced by the current `jolt-core` prover while depending primarily on the
modular crates. The crate is not just a wrapper around
`jolt_core::zkvm::verifier`; it is an audit-friendly target artifact that Bolt
can optimize toward. It should look recognizably like the existing
`jolt-core` verifier at the control-flow level, but use the cleaner modular
crate APIs where those APIs are ready. When the modular APIs are awkward or
insufficient, the implementation should use the handwritten verifier as pressure
to improve those APIs before Bolt relies on them.

## Intent

### Goal

Introduce `crates/jolt-verifier` as a standalone verifier crate with three
purposes:

- **Compatibility artifact**: verify proofs and verifier preprocessing produced
  by today's `jolt-core` prover/preprocessing path.
- **Architecture model**: demonstrate the target shape for a future
  `jolt-core` split into `jolt-prover` and `jolt-verifier`.
- **Bolt target artifact**: provide a concrete, handwritten verifier that the
  Bolt compiler can compare against and optimize toward.

The verifier's top-level flow should remain close to the current
`jolt-core/src/zkvm/verifier.rs`, but it should avoid a public stateful API,
a giant private verifier context, or small wrapper methods that hide
transcript-order-sensitive work. The preferred shape is one public `verify(...)`
function that performs validation, preamble binding, commitment binding, and
stage calls directly. Grouped arguments are encouraged when they represent a
coherent concept, but the dataflow should stay explicit:

```rust
pub fn verify<PCS, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS>,
    proof: JoltProof<PCS, T>,
    io: common::jolt_device::JoltDevice,
    trusted_advice_commitment: Option<PCS::Output>,
) -> Result<(), VerifierError> {
    let checked = validate_inputs(preprocessing, &proof, io)?;
    let mut transcript = T::new(b"Jolt");

    let preprocessing_digest = preprocessing.shared.digest();
    fiat_shamir_preamble(
        &checked.io,
        proof.ram_K,
        proof.trace_length,
        preprocessing.shared.bytecode.entry_address,
        &proof.rw_config,
        &proof.one_hot_config,
        proof.dory_layout,
        &preprocessing_digest,
        &mut transcript,
    );

    for commitment in &proof.commitments {
        transcript.append_serializable(b"commitment", commitment);
    }
    if let Some(commitment) = &proof.untrusted_advice_commitment {
        transcript.append_serializable(b"untrusted_advice", commitment);
    }
    if let Some(commitment) = &trusted_advice_commitment {
        transcript.append_serializable(b"trusted_advice", commitment);
    }

    let proof_openings = compat::convert::proof_openings(&proof, checked.shape())?;

    let stage1 = stages::stage1::verify(Stage1Inputs {
        proof: &proof,
        preprocessing,
        checked: &checked,
        transcript: &mut transcript,
        openings: proof_openings.stage1(),
    })?;
    let stage2 = stages::stage2::verify(Stage2Inputs {
        proof: &proof,
        preprocessing,
        checked: &checked,
        transcript: &mut transcript,
        openings: proof_openings.stage2(),
        stage1: &stage1,
    })?;
    let stage3 = stages::stage3::verify(Stage3Inputs {
        proof: &proof,
        preprocessing,
        checked: &checked,
        transcript: &mut transcript,
        openings: proof_openings.stage3(),
        stage1: &stage1,
        stage2: &stage2,
    })?;
    let stage4 = stages::stage4::verify(Stage4Inputs {
        proof: &proof,
        preprocessing,
        checked: &checked,
        transcript: &mut transcript,
        openings: proof_openings.stage4(),
        stage2: &stage2,
        stage3: &stage3,
    })?;
    let stage5 = stages::stage5::verify(Stage5Inputs {
        proof: &proof,
        preprocessing,
        checked: &checked,
        transcript: &mut transcript,
        openings: proof_openings.stage5(),
        stage4: &stage4,
    })?;
    let stage6 = stages::stage6::verify(Stage6Inputs {
        proof: &proof,
        preprocessing,
        checked: &checked,
        transcript: &mut transcript,
        openings: proof_openings.stage6(),
        stage4: &stage4,
        stage5: &stage5,
    })?;
    let stage7 = stages::stage7::verify(Stage7Inputs {
        proof: &proof,
        preprocessing,
        checked: &checked,
        transcript: &mut transcript,
        openings: proof_openings.stage7(),
        stage6: &stage6,
    })?;

    let opening_plan = OpeningPlan::from_stage_outputs(
        &stage1, &stage2, &stage3, &stage4, &stage5, &stage6, &stage7,
    )?;
    let stage8 = stages::stage8::verify(Stage8Inputs {
        proof: &proof,
        preprocessing,
        checked: &checked,
        transcript: &mut transcript,
        openings: &proof_openings,
        opening_plan,
    })?;

    stages::blindfold::verify_if_needed(BlindfoldInputs {
        proof: &proof,
        preprocessing,
        checked: &checked,
        transcript: &mut transcript,
        stages: (&stage1, &stage2, &stage3, &stage4, &stage5, &stage6, &stage7, &stage8),
    })?;
    Ok(())
}
```

The implementation should preserve this readable stage structure even if some
submodules are initially copied from `jolt-core` and then incrementally retargeted
to modular crates. The model verifier should prefer explicit typed stage dataflow
over a catch-all verifier object.

### Invariants

- `jolt-verifier` library code must not depend on `jolt-core`.
- `jolt-verifier` library code should not depend on `tracer` or `jolt-sdk`.
- `jolt-core` may be used as a dev-dependency only for differential tests and
  fixture generation.
- A proof accepted by the current `jolt-core` verifier must be accepted by
  `jolt-verifier`, modulo explicitly documented verifier hardening changes.
- A proof rejected by the current `jolt-core` verifier for a specified semantic
  reason should be rejected by `jolt-verifier` for the same reason.
- Transcript labels, transcript ordering, preprocessing digest, proof wire
  layout, and opening-claim namespaces must remain compatible with
  `jolt-core` prover output.
- Transcript-order-sensitive operations should be visible in the top-level
  `verify(...)` flow unless extracting them is necessary for reuse or targeted
  differential testing. Avoid one-line wrapper methods that make the verifier
  harder to compare against `jolt-core`.
- Do not model verifier progress with a mutable opening accumulator. The
  `jolt-core` accumulator is a compatibility source to understand, not the
  target architecture. The model verifier should convert wire-format opening
  claims into typed structures and pass named stage outputs forward explicitly.
- Stage dependencies should be represented with typed input/output structs and
  ownership where useful. If stage 7 needs a value produced by stage 6, stage 6
  should return a typed value that stage 7 consumes or borrows, rather than
  relying on a runtime map lookup.
- ZK and advice support are first-class requirements, not stretch goals. Stage
  outputs should carry the data needed for BlindFold bindings and advice checks
  as the stages are ported, even if final BlindFold verification is wired after
  the stage outputs exist.
- Standard and ZK proof modes must remain feature-gated in a way that is
  compatible with current `jolt-core` proof production.
- The first working version may keep Jolt-specific verifier stage logic in
  `jolt-verifier`, but generic algebra, transcript, PCS, polynomial, RISC-V,
  program preprocessing, and lookup table functionality should come from the
  modular crates whenever practical.
- Backwards-compatibility code must be isolated under `compat/`. The rest of
  the crate should consume typed model structs, not `jolt-core`-shaped wire maps
  or compatibility enums directly unless the type is explicitly part of the
  stable model.
- Any awkward adapter needed only because a modular API is insufficient should be
  called out as an API pressure item, not hidden as permanent glue.

### Non-Goals

- This spec does not require immediately deleting or changing
  `jolt-core/src/zkvm/verifier.rs`.
- This spec does not require the first `jolt-verifier` version to be generated
  by Bolt.
- This spec does not require proving support in `jolt-verifier`.
- This spec does not require changing the current `jolt-core` proof format in
  the first implementation slice.
- This spec does not require every Jolt-specific verifier subprotocol to be
  moved into a reusable modular crate before `jolt-verifier` exists.
- This spec does not require an external stable public API for third-party
  consumers. The crate is first a workspace model and compatibility target.

## Existing Modular Pieces

The following crates should be preferred over `jolt-core` internals:

| Crate | Intended verifier use |
|-------|------------------------|
| `common` | `JoltDevice`, `MemoryLayout`, constants shared by verifier inputs |
| `jolt-field` | field abstraction and BN254 scalar field |
| `jolt-transcript` | Fiat-Shamir transcript implementations |
| `jolt-crypto` | group, pairing, Pedersen, and commitment primitives |
| `jolt-poly` | multilinear/univariate polynomial abstractions and helpers |
| `jolt-openings` | PCS traits, opening reduction APIs, and generic typed opening facts |
| `jolt-dory` | Dory PCS implementation |
| `jolt-hyperkzg` | optional/future HyperKZG PCS implementation |
| `jolt-r1cs` | R1CS data structures where the API matches verifier needs |
| `jolt-sumcheck` | generic sumcheck verification APIs |
| `jolt-riscv` | source/final instruction identities, final rows, profiles |
| `jolt-program` | bytecode and RAM preprocessing, program image pipeline |
| `jolt-lookup-tables` | lookup table definitions and final-row lookup routing |

The following pieces are still effectively `jolt-core`-owned and must be ported,
compatibility-modeled, or extracted during this project:

- `JoltProof` wire format.
- `JoltVerifierPreprocessing` and `JoltSharedPreprocessing` wire formats.
- `OpeningId`, `PolynomialId`, `SumcheckId`, committed polynomial IDs, and
  virtual polynomial IDs.
- Typed opening-claim model compatible with the current `jolt-core`
  opening-claim namespace.
- Jolt-specific stage verifier params for Spartan, RAM, registers, bytecode,
  instruction lookups, and claim reductions.
- BlindFold verifier integration for ZK mode.
- Uni-skip verifier integration where the modular sumcheck crate does not yet
  expose the needed surface.
- Exact transcript preamble and Dory opening input binding.

## Target Crate Layout

Initial crate structure:

```text
crates/jolt-verifier/
  Cargo.toml
  src/lib.rs

  src/verifier.rs
  src/error.rs
  src/openings.rs
  src/commitments.rs
  src/claims/
    mod.rs
    expr.rs
    check.rs

  src/compat/
    mod.rs
    convert.rs
    proof.rs
    preprocessing.rs
    ids.rs
    config.rs
    transcript.rs

  src/stages/
    mod.rs
    stage1.rs
    stage2.rs
    stage3.rs
    stage4.rs
    stage5.rs
    stage6.rs
    stage7.rs
    stage8.rs
    blindfold.rs
```

This layout is intentionally conservative. `compat/` is the quarantine boundary
for backwards compatibility with today's `jolt-core` proof artifacts. It may
contain wire-compatible proof/preprocessing structs, legacy ID encodings, and
conversion helpers, but stage code should not traffic in those wire shapes
directly. `verifier.rs` owns the public `verify(...)` entry point and small
validation/grouping structs such as `CheckedInputs`. `openings.rs` owns typed
opening-claim structures, not a mutable accumulator. `stages/` preserves the
top-level verifier flow as free functions over stage-specific input structs.
`claims/` contains the local claim-expression/check abstraction that models the
future Bolt-generated formula layer. Stage-local helper modules may be extracted
later when repeated formulas or file size justify them, but the initial layout
should avoid a generic `protocols/` dumping ground.

## Public API Sketch

The first public API should be narrow:

```rust
pub type DefaultField = jolt_field::Fr;
pub type DefaultTranscript = jolt_transcript::Blake2bTranscript;
pub type DefaultPcs = jolt_dory::DoryScheme;

pub fn verify<PCS, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS>,
    proof: JoltProof<PCS, T>,
    io: common::jolt_device::JoltDevice,
    trusted_advice_commitment: Option<PCS::Output>,
) -> Result<(), VerifierError>
where
    PCS: /* modular PCS verifier bounds */,
    T: /* modular transcript bounds */;
```

The implementation should not expose a public `JoltVerifier::new()` /
`JoltVerifier::verify()` object API in the first version. It should also avoid a
private catch-all `VerifierState` or `VerificationContext` that simply bundles
the whole verifier. Prefer small concept-oriented structs:

- `CheckedInputs` for validated public IO, trace shape, `ram_K`, and derived
  params.
- `ProofOpenings` for typed standard-mode proof opening claims.
- `StageNInputs` for each stage's explicit inputs.
- `StageNOutput` for named challenges, claims, constraints, and values needed by
  later stages.
- `OpeningPlan` for the explicit stage-8 opening proof plan assembled from
  typed stage outputs.

The API should use `common::jolt_device::JoltDevice` directly. It must not use
`tracer::JoltDevice`, even though `tracer` currently re-exports the same common
type.

## Typed Dataflow

The model verifier should improve on `jolt-core` by making verifier dataflow
explicit and typed. The current `jolt-core` verifier accumulates opening claims
and opening points in a runtime map keyed by `OpeningId`. That is flexible, but
it makes stage dependencies harder to audit: missing, overwritten, or
misclassified claims can remain implicit until a later lookup.

`jolt-verifier` should instead use this pattern:

```text
jolt-core-compatible wire proof
  -> compatibility decode
  -> typed ProofOpenings / ZkOpeningCommitments
  -> typed StageNInputs
  -> typed StageNOutput
  -> explicit OpeningPlan for stage 8
```

Guidelines:

- A stage function should receive a `StageNInputs` struct whose fields describe
  exactly what the stage uses.
- A stage function should return a `StageNOutput` struct whose fields describe
  exactly what later stages or BlindFold need.
- Cross-stage dependencies should be Rust fields, references, or moved values,
  not runtime lookups into a generic accumulator.
- The standard-mode proof's wire-format opening map should be imported directly
  from `jolt-core` in compatibility tests or through a feature-gated converter,
  then converted once into a typed `ProofOpenings` structure with explicit
  missing/extra checks for the selected proof shape.
- Stage 8 should receive an explicit `OpeningPlan` assembled from typed stage
  outputs and typed proof openings. It should not reconstruct verifier intent by
  walking an accumulator map.
- If this typed model exposes missing abstractions in `jolt-openings` or
  `jolt-sumcheck`, prefer improving those modular APIs over reintroducing a
  broad accumulator.

## Claim Algebra And Bolt Target

The main Jolt-specific logic that is not already covered by modular crates is
the claim algebra for each verifier check:

- the summand/final-claim formula for each sumcheck instance;
- the input claim derived from prior openings, challenges, and constants;
- the output claim produced after verifier challenges are sampled;
- the semantic splitting of challenge vectors into cycle, address, bytecode,
  one-hot, or table coordinates;
- the opening dependencies required by stage 8;
- the ZK/BlindFold binding for the same claim formula.

This is the layer Bolt is expected to generate eventually. The handwritten
`jolt-verifier` should therefore model the shape of that generated output. It
should keep these formulas explicit, local, and uniform rather than scattering
ad hoc scalar math through stage orchestration.

The target pattern is one structured source for a claim formula that can support:

```text
claim expression
  -> native scalar evaluation for verifier checks
  -> typed opening dependencies
  -> typed stage output fields
  -> BlindFold input/output binding in ZK mode
  -> future Bolt IR/codegen target
```

This should improve on the current `jolt-core` pattern where the same math can
appear separately in methods such as `input_claim`,
`input_claim_constraint`, and `input_constraint_challenge_values`. The model
verifier may still port formulas incrementally from `jolt-core`, but the end
shape should make formula drift harder.

The modular crates should continue to own generic verifier machinery and public
data structures. `jolt-verifier` should own the Jolt-specific generated/manual
formula layer until a later shared protocol crate or Bolt IR makes a cleaner
home obvious.

## Opening Abstraction Ownership

Generic opening facts should live in `jolt-openings`, not in `jolt-verifier`.
`jolt-openings` already owns stateless PCS claim types and RLC reduction, so it
is the right home for a small typed layer over opening points and evaluations.
This typed layer should be added when the verifier implementation first needs
it; do not add it speculatively before the typed opening model is being built.

The intended generic surface is lightweight:

```rust
pub struct Point<F, Domain = ()> {
    pub coords: Vec<F>,
    _domain: PhantomData<Domain>,
}

pub struct Opening<F, Poly = (), Domain = ()> {
    pub point: Point<F, Domain>,
    pub eval: F,
    _poly: PhantomData<Poly>,
}

pub struct CommittedOpening<F, C, Poly = (), Domain = ()> {
    pub commitment: C,
    pub point: Point<F, Domain>,
    pub eval: F,
    _poly: PhantomData<Poly>,
}
```

`jolt-openings` may also provide erasure/conversion helpers from
`CommittedOpening<F, C, Poly, Domain>` into the existing untyped
`VerifierClaim<F, C>` used by PCS reduction.

Jolt-specific ownership remains in `jolt-verifier`:

- marker types such as `RamInc`, `InstructionRa`, `IncClaimReduction`, and
  `HammingWeightClaimReduction`;
- typed containers such as `ProofOpenings`, `StageNInputs`, `StageNOutput`, and
  `OpeningPlan`;
- compatibility conversion from legacy `OpeningId -> claim` maps;
- protocol formulas for Spartan, RAM, registers, bytecode, instruction lookups,
  claim reductions, and advice.

Do not put Jolt protocol formulas in `jolt-openings`. That crate should remain
PCS/opening-oriented and protocol-agnostic.

## Compatibility Boundary

Backwards compatibility is required, but it should not define the architecture
of the model verifier. All compatibility with the current `jolt-core` artifact
format should be isolated under `src/compat/`.

Allowed inside `compat/`:

- `jolt-core`-compatible proof and preprocessing wire structs.
- Legacy `OpeningId`, `PolynomialId`, `SumcheckId`, committed polynomial, and
  virtual polynomial encodings.
- `codec` helpers for canonical serialization that must match `jolt-core`.
- Wire-format maps such as standard-mode `OpeningId -> claim`.
- `convert` functions from wire structs into typed model structs.
- Transcript compatibility helpers that preserve current labels and ordering.

Not allowed outside `compat/` except at explicit conversion boundaries:

- Direct indexing into a legacy `OpeningId -> claim` map.
- Passing legacy wire proof substructures deep into stage code when a typed
  stage input is possible.
- Reconstructing verifier intent from compatibility IDs after typed conversion.
- Accumulator-shaped compatibility maps as verifier state.

Verifier-owned model types should use Rust `serde` for idiomatic serialization
where practical. The compatibility layer remains responsible for bridging from
current `jolt-core` canonical artifacts into those model types.

The intended boundary is:

```text
compat::convert::JoltCoreProof  (direct alias to jolt-core behind jolt-core-compat)
compat::preprocessing::WireVerifierPreprocessing
compat::ids::{OpeningId, ...}
compat::codec::{... canonical bridge helpers ...}
compat::convert::{From<JoltCoreProof>, preprocessing, proof_openings}
        |
        v
typed verifier model:
  JoltProof
  JoltVerifierPreprocessing
  ProofOpenings
  CheckedInputs
  StageNInputs / StageNOutput
  OpeningPlan
```

The exact type names can change during implementation, but this direction should
not: `compat/` decodes and validates old artifacts; the verifier proper uses the
new typed representation.

## Feature Policy

Initial features:

```toml
[features]
default = ["transcript-blake2b"]
zk = []
transcript-blake2b = []
transcript-keccak = []
transcript-poseidon = ["jolt-transcript/poseidon"]
```

Only one transcript feature should be enabled at a time, matching the existing
`jolt-core` policy. The default should match current `jolt-core` verifier
behavior: Blake2b when no explicit transcript feature is selected.

## Dependency Policy

Allowed normal dependencies:

```text
ark-serialize
blake2
common
jolt-crypto
jolt-dory
jolt-field
jolt-lookup-tables
jolt-openings
jolt-poly
jolt-program
jolt-r1cs
jolt-riscv
jolt-sumcheck
jolt-transcript
rayon
serde
thiserror
tracing
```

Forbidden normal dependencies:

```text
jolt-core
jolt-sdk
tracer
```

Allowed dev-dependencies:

```text
jolt-core
jolt-sdk
tracer
```

Only tests and fixture-generation helpers may use those dev-dependencies.

## API Pressure Log

During implementation, each uncomfortable adapter should be recorded in a
section like this:

```text
API Pressure Item
- Location:
- Problem:
- Temporary local workaround:
- Proposed modular crate change:
- Blocking for Bolt? yes/no
```

The implementation should prefer improving a modular crate API over teaching
Bolt to generate compatibility noise.

Likely pressure areas:

- `jolt-openings`: generic typed `Point` / `Opening` / `CommittedOpening`,
  erasure into `VerifierClaim`, and RLC reduction shape.
- `jolt-dory`: wire-compatible proof/setup/commitment serialization and Dory
  layout handling.
- `jolt-sumcheck`: Jolt-specific batched/uni-skip verification surfaces.
- `jolt-poly`: missing helpers that still exist only in `jolt-core::poly`.
- `jolt-r1cs`: whether Spartan verifier key and matrix helpers are reusable
  enough for stage 1 and stage 2.
- `jolt-lookup-tables`: final-row lookup routing APIs expected by verifier
  stage code.

## Implementation Plan

Each step below is intended to be small enough for explicit review. Do not
proceed from one step to the next until the previous step has been reviewed and
accepted.

### Step 0: Agree On This Spec

Goal:

- Review this document and agree on scope, crate shape, and stopping rules.

Expected changes:

- `specs/jolt-verifier-model-crate.md` only.

Checks:

- No code checks required.

Review stop:

- Confirm the crate is a handwritten Bolt target artifact, not a thin
  `jolt-core` wrapper.
- Confirm the first implementation slice should be scaffolding and wire-model
  only.

### Step 1: Add Empty Crate Scaffold

Goal:

- Add `crates/jolt-verifier` as a workspace crate that compiles but performs no
  verification.

Expected changes:

- Root `Cargo.toml`: add workspace member and workspace dependency entry.
- `crates/jolt-verifier/Cargo.toml`.
- `crates/jolt-verifier/src/lib.rs`.
- `crates/jolt-verifier/src/verifier.rs`.
- `crates/jolt-verifier/src/error.rs`.

Acceptance criteria:

- `cargo check -p jolt-verifier --quiet` passes.
- Library code does not depend on `jolt-core`, `tracer`, or `jolt-sdk`.

Review stop:

- Confirm crate metadata, feature names, and dependency policy before adding
  compatibility types.

### Step 2: Add Compatibility IDs

Goal:

- Port the verifier namespace IDs needed to deserialize current `jolt-core`
  proofs. These are compatibility-layer types, not the verifier's preferred
  internal data model.

Expected changes:

- `src/compat/ids.rs`.
- `src/compat/codec.rs`.
- `src/compat/mod.rs`.

Types:

- `SumcheckId`.
- `PolynomialId`.
- `OpeningId`.
- committed polynomial ID enum.
- virtual polynomial ID enum.

Acceptance criteria:

- IDs implement the same canonical serialization layout as `jolt-core`.
- IDs derive serde traits for model-side serialization.
- Unit tests cover representative variants.
- Differential tests compare serialized bytes against `jolt-core` for every ID
  variant.

Review stop:

- Decide whether any ID type should immediately move to `jolt-openings` or stay
  local until more verifier code is ported.
- Confirm that stages will not consume these IDs directly except at explicit
  conversion boundaries.

### Step 3: Add Config Compatibility

Goal:

- Port verifier config types required by the proof wire format and verifier
  setup.

Expected changes:

- `src/compat/config.rs`.

Types:

- `ReadWriteConfig`.
- `OneHotConfig`.
- `OneHotParams`.

Acceptance criteria:

- Validation behavior matches `jolt-core`.
- Canonical serialization layout matches `jolt-core`.
- Fallible construction uses idiomatic trait conversions where practical.
- Derived-parameter construction rejects zero-sized domains, truncating numeric
  conversions, and invalid one-hot configs before doing log/shift arithmetic.
- Unit tests cover invalid configs and known-good configs.

Review stop:

- Decide whether these config types belong in `jolt-verifier`,
  `jolt-program`, or a future shared VM config crate.

### Step 4: Add Proof Model and Core-Proof Conversion Boundary

Goal:

- Define the verifier-owned proof model shape without copying the current
  `jolt-core` proof struct.
- Import `jolt-core` proof types directly in tests or behind an explicit
  compatibility feature when core-proof conversion is needed.

Expected changes:

- `src/proof.rs`.
- `src/compat/convert.rs` behind a `jolt-core` compatibility feature if a
  library-level conversion shim is useful.

Acceptance criteria:

- `jolt-verifier` does not define a second copy of the full `jolt-core` proof
  struct.
- Direct `jolt-core` proof imports are used for compatibility tests and any
  feature-gated conversion shim.
- `verify_zk_consistency()` behavior matches `jolt-core` for imported core
  stage proof types where the conversion feature is enabled.
- The verifier-owned proof type is named `JoltProof`.
- Field-level canonical compatibility tests stay focused on the smaller
  compatibility types we own locally, such as IDs, config, and Dory layout.

Implementation note:

- The verifier-owned `JoltProof` groups stage proofs into `JoltStageProofs` and
  stores the runtime clear/ZK-specific data in `ProofPayload`.
- `compat::convert::JoltCoreProof` is an alias to
  `jolt_core::zkvm::proof_serialization::JoltProof` when the
  `jolt-core-compat` feature is enabled.
- `compat::convert` implements `From<JoltCoreProof>` for the verifier-owned
  model proof shape without duplicating the core proof definition.

Review stop:

- Confirm whether core-proof conversion should stay behind `jolt-core-compat`
  or move entirely into integration-test helpers.

### Step 4.1: Inventory Proof Compatibility Gaps

Goal:

- Record the concrete reasons `proof.rs` cannot yet use only modular crate
  proof types.
- Avoid choosing solutions in this step. Each issue below needs an explicit
  design decision before implementation proceeds.

Current desired direction:

- `proof.rs` should use concrete modular proof types wherever possible.
- `compat/` should import current `jolt-core` proof types directly, then convert
  them into the `jolt-verifier` model proof for compatibility and integration
  testing.
- `jolt-verifier` should not duplicate full `jolt-core` proof structs.

Issue inventory:

1. **Core sumcheck proof wrapper is not modular**

   Current `jolt-core` proof field type:

   ```rust
   SumcheckInstanceProof<F, C, FS>
   ```

   Current modular availability:

   - `jolt-sumcheck` exposes `SumcheckProof<F>` for clear round-polynomial
     sumcheck proofs.
   - `jolt-sumcheck` does not currently expose the `Clear` / `Zk` wrapper used
     by `jolt-core`.

   Compatibility issue:

   - Current core proofs encode whether each stage sumcheck is clear or ZK in
     `SumcheckInstanceProof`.
   - The verifier model cannot represent current stage sumcheck proofs with
     only `jolt-sumcheck::SumcheckProof<F>` without losing the ZK variant.

   Design decision needed:

   - Where should the clear/ZK sumcheck proof wrapper live?
   - What should its public name and field shape be?
   - Should it preserve the current `jolt-core` enum shape or define a cleaner
     modular shape with a compatibility conversion?

2. **ZK sumcheck round proof data is not modular**

   Current `jolt-core` proof field shape includes ZK sumcheck proofs whose data
   is Pedersen commitments to round polynomials, polynomial degrees, and output
   claim commitments.

   Current modular availability:

   - `jolt-sumcheck` documents committed-mode as future-facing in
     `RoundProof`, but does not expose the concrete committed/ZK proof type
     used by current `jolt-core`.
   - `jolt-crypto` exposes Pedersen commitment primitives, but not the
     Jolt-specific sumcheck ZK proof wrapper.

   Compatibility issue:

   - ZK proofs produced by current `jolt-core` cannot be represented by the
     current modular `SumcheckProof<F>`.
   - The proof mode consistency check needs a typed way to ask whether a stage
     proof is clear or ZK.

   Design decision needed:

   - Should committed/ZK sumcheck proof data live in `jolt-sumcheck`, a
     BlindFold-oriented crate, or `jolt-verifier`?
   - Should the `ProofMode` trait remain in `jolt-verifier::zk`, or should the
     modular proof type provide this directly?

3. **Univariate-skip proof types are core-only**

   Current `jolt-core` proof field type:

   ```rust
   UniSkipFirstRoundProofVariant<F, C, FS>
   ```

   Current modular availability:

   - `jolt-sumcheck` has generic round proof abstractions.
   - No modular crate currently exposes the Jolt univariate-skip first-round
     proof variant used by stages 1 and 2.

   Compatibility issue:

   - Stage 1 and stage 2 proofs include a high-degree first-round uni-skip
     proof.
   - The verifier model cannot make `JoltStageProofs` concrete using only
     existing modular types.

   Design decision needed:

   - Should univariate-skip live in `jolt-sumcheck`, a new dedicated module or
     crate, or `jolt-verifier`?
   - Should its clear/ZK wrapper mirror the core `UniSkipFirstRoundProofVariant`
     or use a new modular representation?

4. **BlindFold proof is core-only**

   Current `jolt-core` ZK proof field:

   ```rust
   BlindFoldProof<F, C>
   ```

   Current modular availability:

   - BlindFold verifier/proof types currently live under
     `jolt-core::subprotocols::blindfold`.
   - No modular crate currently exports the BlindFold proof type used by
     current ZK proofs.

   Compatibility issue:

   - A concrete ZK-capable `ProofPayload` cannot be expressed purely with
     modular crate types today.
   - ZK verification also needs BlindFold verifier inputs, stage configs, and
     R1CS construction data, not just the proof object.

   Design decision needed:

   - Should BlindFold move to its own modular crate, into `jolt-sumcheck`, or
     into another existing verifier-oriented crate?
   - Which BlindFold types are verifier-facing proof API, and which are
     implementation internals?

5. **Opening claims still use core accumulator-shaped IDs**

   Current `jolt-core` standard proof field:

   ```rust
   Claims<F>
   ```

   Current modular availability:

   - `jolt-openings` exposes generic `ProverClaim` and `VerifierClaim`.
   - `jolt-openings` does not know Jolt stage IDs, polynomial IDs, or the
     required set of openings for a specific Jolt proof shape.

   Compatibility issue:

   - Current core `Claims<F>` is keyed by `OpeningId`, which encodes
     `SumcheckId` plus committed/virtual polynomial IDs.
   - The model verifier should eventually consume typed per-stage openings, not
     an accumulator-shaped compatibility map.

   Design decision needed:

   - What is the verifier-owned typed opening structure?
   - Which part belongs in `jolt-openings` as generic opening claim machinery,
     and which part remains Jolt-protocol-specific?
   - Should compatibility `OpeningId` remain local to `jolt-verifier::compat`
     or move to a shared proof-format crate?

6. **Dory layout is not exposed by `jolt-dory`**

   Current `jolt-core` proof field:

   ```rust
   DoryLayout
   ```

   Current modular availability:

   - `jolt-dory` exposes `DoryCommitment`, `DoryProof`, setup types,
     streaming commitment, and ZK opening APIs.
   - `DoryLayout` currently lives in `jolt-core` Dory globals.

   Compatibility issue:

   - `proof.rs` currently carries a local `DoryLayout` model enum so it can
     represent the proof metadata without depending on `jolt-core`.
   - This is a smaller duplicate than the full proof struct, but it is still not
     imported from a modular crate.

   Design decision needed:

   - Where should `DoryLayout` live?
   - Is it part of Dory PCS configuration, Jolt proof metadata, preprocessing,
     or a shared proof-format/config crate?

7. **Concrete Dory proof types are modular but not yet wired into `proof.rs`**

   Current modular availability:

   - `jolt-dory` exposes `DoryCommitment` and `DoryProof`.
   - Current `proof.rs` is still generic over `Commitment` and `OpeningProof`.

   Compatibility issue:

   - We have enough modular Dory types to remove some generics from the default
     proof model, but doing so should be a deliberate design step because core
     compatibility conversion must translate from core PCS associated types.

   Design decision needed:

   - Should `JoltProof` itself be concrete over Dory, or should it stay generic
     with a `DefaultJoltProof` / `DoryJoltProof` alias?
   - Should non-Dory PCS support be a goal for this model crate?

8. **Transcript type appears in core proof subproofs**

   Current `jolt-core` stage proof types are generic over `FS: Transcript`.

   Current modular availability:

   - `jolt-transcript` exposes transcript traits and implementations.
   - Some modular proof types use serde and transcript-agnostic data, while
     current core proof subtypes include the transcript type parameter in their
     definitions.

   Compatibility issue:

   - `proof.rs` currently carries stage proof types generically, so the
     transcript type parameter leaks through compatibility conversion.
   - A concrete modular proof type may be able to avoid carrying transcript type
     parameters in the proof data.

   Design decision needed:

   - Should modular proof data types be transcript-agnostic?
   - Where should transcript choice enter: proof type, verifier function, or
     type alias?

9. **Serialization boundary differs between modular crates and core**

   Current core artifacts use arkworks canonical serialization for proof wire
   compatibility.

   Current modular availability:

   - Several modular crates use Rust `serde` as their primary model-facing
     serialization.
   - `jolt-dory` wraps canonical Dory proof bytes behind serde.

   Compatibility issue:

   - `jolt-verifier` wants serde-native model types where practical, while
     compatibility with current core proofs uses canonical serialization.
   - The conversion boundary must not blur model serialization with legacy core
     wire serialization.

   Design decision needed:

   - Which proof types should implement serde only?
   - Which compatibility paths should retain canonical decoding?
   - Should there be a shared codec crate or should compatibility codec remain
     local to `jolt-verifier`?

Review stop:

- Resolve these issues one at a time before making `proof.rs` concrete over
  modular crate types.

### Step 5: Add Preprocessing Wire Model

Goal:

- Define verifier preprocessing wire types compatible with `jolt-core` prover
  setup, plus conversion into verifier-facing typed preprocessing where useful.

Expected changes:

- `src/compat/preprocessing.rs`.
- `src/compat/convert.rs` if the preprocessing wire type is converted into a
  distinct verifier-facing type in this step.

Types:

- `JoltSharedPreprocessing`.
- `JoltVerifierPreprocessing`.
- optional ZK `BlindfoldSetup` placeholder if needed for the wire model.

Acceptance criteria:

- Reuse `jolt_program::preprocess::{BytecodePreprocessing, RAMPreprocessing}`.
- Reuse `common::jolt_device::MemoryLayout`.
- `digest()` matches `jolt-core` byte-for-byte for the same preprocessing.
- Serialization layout matches `jolt-core`.
- Wire compatibility remains under `compat/`; non-compat verifier modules
  consume typed preprocessing.

Review stop:

- Confirm whether `BlindfoldSetup` should be local compatibility code or moved
  toward `jolt-crypto`.

### Step 6: Add Transcript Compatibility Helpers

Goal:

- Encode the exact Fiat-Shamir preamble and commitment-binding order used by
  current `jolt-core`.

Expected changes:

- `src/compat/transcript.rs`.

Acceptance criteria:

- Preamble absorbs preprocessing digest, public IO, configs, trace length,
  `ram_K`, entry address, and Dory layout in the same order and with the same
  labels as `jolt-core`.
- Unit tests compare transcript challenges against `jolt-core` for a fixed
  synthetic input.

Review stop:

- Decide whether transcript labels should become shared constants in
  `jolt-transcript` or remain verifier-local.

### Step 7: Add Direct Verifier Entry Point Skeleton

Goal:

- Add the public `verify(...)` entry point and upfront validation, but no stage
  verification yet.

Expected changes:

- `src/verifier.rs`.
- `src/lib.rs` updated to re-export the public `verify(...)` API.

Acceptance criteria:

- No public stateful `JoltVerifier::new()` / `JoltVerifier::verify()` API is
  introduced.
- No private catch-all `VerifierState` / `VerificationContext` is introduced.
- Any grouping structs are concept-oriented, e.g. `CheckedInputs`, not a wrapper
  around the whole verifier.
- Memory layout checks match `jolt-core`.
- Input/output size checks match `jolt-core`.
- Output trailing-zero truncation matches `jolt-core`.
- Trace length validation matches `jolt-core`.
- `ram_K` min/max validation matches `jolt-core`.
- Config validation is called in the same place as `jolt-core`.

Review stop:

- Confirm the direct `common::jolt_device::JoltDevice` API and ensure no
  `tracer::JoltDevice` dependency was introduced.

### Step 8: Add Typed Opening Claim Model

Goal:

- Add a typed opening-claim model that replaces the current `jolt-core`
  verifier opening accumulator in the model crate.
- Add or use generic typed opening facts in `jolt-openings`; keep Jolt-specific
  marker types and opening containers in `jolt-verifier`. Add the generic
  `jolt-openings` layer only when this step needs it.

Expected changes:

- `crates/jolt-openings/src/claims.rs` or a new `typed.rs` module for generic
  `Point`, `Opening`, and `CommittedOpening`, only if the typed opening model
  needs those generic wrappers in this step.
- `src/openings.rs`.
- `src/compat/convert.rs` for the wire-claim-map to typed-openings boundary.

Acceptance criteria:

- Generic opening point/evaluation wrappers live in `jolt-openings`.
- Jolt-specific marker types and typed opening containers live in
  `jolt-verifier`.
- The standard-mode wire-format opening map is converted into typed
  `ProofOpenings` structures.
- Required claims are represented by named fields instead of map lookups.
- Missing or extra wire claims are rejected at the typed conversion boundary
  whenever the proof shape determines the exact required set. The required set
  should be derived from the proof/config/advice shape at runtime, not from a
  hardcoded global list.
- Stage output structs expose named opening claims/points needed by later
  stages and stage 8.
- No mutable opening accumulator is introduced.

Review stop:

- Confirm the split between generic `jolt-openings` typed facts and
  Jolt-specific `jolt-verifier` opening containers before porting stages.

### Step 9: Add Stage Skeletons

Goal:

- Add all stage methods with signatures and result structs, but leave protocol
  bodies stubbed.

Expected changes:

- `src/stages/mod.rs`.
- `src/stages/stage1.rs` through `stage8.rs`.
- `src/stages/blindfold.rs` behind `zk`.

Acceptance criteria:

- The top-level `verify(...)` function reads like the current `jolt-core`
  verifier, with transcript preamble and commitment binding visible inline.
- Each stage has a clear input/output result type.
- Stage functions take explicit `StageNInputs` structs, not a whole verifier
  context.
- Stage outputs include the typed data needed for ZK/BlindFold bindings and
  advice paths where applicable; do not defer those shapes until after the
  standard-mode verifier works.
- No stage is implemented yet beyond returning a structured
  `VerifierError::UnimplementedStage`.

Review stop:

- Confirm stage boundaries before moving protocol code.

### Step 10: Port Stage 1

Goal:

- Implement stage 1 verification using modular crates where practical.

Expected protocol areas:

- Spartan outer sumcheck.
- Stage 1 uni-skip first-round verification.
- Typed stage-1 opening claims and challenge outputs.

Acceptance criteria:

- Stage 1 transcript challenge sequence matches `jolt-core` on a fixture.
- Stage 1 typed output contains the same claims/challenges that `jolt-core`
  would have accumulated, checked by a differential fixture.
- Any copied Jolt-specific params remain stage-local or live under a reviewed
  check/formula module, not under a generic `src/protocols/` dumping ground.

Review stop:

- Review API pressure items before stage 2.

### Step 11: Port Stage 2

Goal:

- Implement stage 2 verification.

Expected protocol areas:

- Spartan product/shift-related verification.
- Stage 2 uni-skip verification.
- RAM/address-round alignment inputs needed by later stages.

Acceptance criteria:

- Stage 2 transcript and typed stage output match `jolt-core` on a fixture.
- Address-round challenge alignment is explicitly tested or asserted.

Review stop:

- Review any `jolt-r1cs`, `jolt-poly`, or uni-skip API pressure.

### Step 12: Port Stage 3

Goal:

- Implement stage 3 verification.

Expected protocol areas:

- Instruction lookup read-RAF checking.
- Bytecode read-RAF checking.
- Related typed opening claims and challenge outputs.

Acceptance criteria:

- Stage 3 transcript and typed stage output match `jolt-core`.
- Lookup-table routing uses `jolt-lookup-tables` where possible.

Review stop:

- Decide whether final-row lookup metadata APIs need changes.

### Step 13: Port Stage 4

Goal:

- Implement stage 4 verification.

Expected protocol areas:

- RAM value checking.
- Register value checking.
- Advice-related value accumulation if present.

Acceptance criteria:

- Stage 4 transcript and typed stage output match `jolt-core`.
- Advice fixtures are included or planned before full verification is claimed.

Review stop:

- Review advice handling and standard-mode claim reconstruction risks.

### Step 14: Port Stage 5

Goal:

- Implement stage 5 verification.

Expected protocol areas:

- RAM/register read-write checking.
- Hamming booleanity inputs that feed later reductions.

Acceptance criteria:

- Stage 5 transcript and typed stage output match `jolt-core`.
- `ReadWriteConfig` behavior matches standard and edge-case configs.

Review stop:

- Confirm no hidden tracer dependency was introduced.

### Step 15: Port Stage 6

Goal:

- Implement stage 6 verification.

Expected protocol areas:

- Claim reductions over instruction lookups, RAM RA, registers, increments,
  and advice phase 1.

Acceptance criteria:

- Stage 6 transcript and typed stage output match `jolt-core`.
- Trusted and untrusted advice paths are both represented.

Review stop:

- Review claim reduction APIs and decide what should move to modular crates.

### Step 16: Port Stage 7

Goal:

- Implement stage 7 verification.

Expected protocol areas:

- Hamming weight claim reduction.
- Advice phase 2.
- Final claim reduction outputs feeding stage 8.

Acceptance criteria:

- Stage 7 transcript and typed stage output match `jolt-core`.
- Advice end-to-end tests are ready before stage 8 is considered complete.

Review stop:

- Confirm stage 8 opening ID set and constraint coefficient derivation.

### Step 17: Port Stage 8 PCS Verification

Goal:

- Verify the final joint opening proof using modular PCS/opening APIs.

Expected protocol areas:

- Commitment map construction.
- RLC commitment combination.
- Dory opening input binding.
- Standard-mode direct opening evaluation binding.
- ZK-mode evaluation commitment binding if `zk` is enabled.

Acceptance criteria:

- Full standard-mode `muldiv` proof generated by `jolt-core` verifies with
  `jolt-verifier` as an interim check.
- Stage 8 preserves the ZK-mode evaluation commitment path needed by
  BlindFold; standard-mode success alone is not the final target.
- Dory proof/setup/commitment serialization is compatible with `jolt-core`.
- Any modular `jolt-dory` gaps are documented as API pressure items.

Review stop:

- Confirm the stage-8 API carries both standard and ZK opening data needed by
  the next BlindFold step.

### Step 18: Port BlindFold ZK Verification

Goal:

- Add ZK-mode BlindFold verification compatible with current `jolt-core`
  proofs.

Expected protocol areas:

- Stage config construction.
- Baked public inputs.
- Input/output claim constraints.
- Pedersen generator setup.
- BlindFold proof verification.

Acceptance criteria:

- Full ZK-mode `muldiv` proof generated by `jolt-core` verifies with
  `jolt-verifier`.
- BlindFold constraints remain synchronized with sumcheck claim formulas.
- ZK consistency checks reject mixed clear/ZK stage proofs.

Review stop:

- Decide whether BlindFold should become a modular crate or remain
  verifier-local until the prover split.

### Step 19: Add Full Differential Test Harness

Goal:

- Build repeatable end-to-end tests that compare `jolt-core` and
  `jolt-verifier`.

This is the full harness. Earlier steps should still add targeted differential
checks for the specific artifact they introduce, such as ID serialization,
preprocessing digest parity, transcript preamble parity, or one-stage
typed-output parity.

Expected tests:

- Generate `muldiv` proof with `jolt-core`, verify with both verifiers.
- Generate advice proof with `jolt-core`, verify with both verifiers.
- Standard mode.
- ZK mode.
- Serialized proof and preprocessing byte round trips.

Acceptance criteria:

- `cargo nextest run -p jolt-verifier --cargo-quiet --features ...` passes for
  supported modes.
- Existing required checks still pass:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

Review stop:

- Confirm the crate is useful as a Bolt target artifact.

### Step 20: Modular API Cleanup Pass

Goal:

- Use the completed handwritten verifier to improve modular crates and reduce
  compatibility glue.

Potential moves:

- Move reusable typed opening-plan pieces into `jolt-openings`.
- Move reusable verifier config into a shared VM config crate if warranted.
- Move generic BlindFold pieces out of `jolt-verifier`.
- Add verifier-friendly sumcheck/uni-skip APIs to `jolt-sumcheck`.
- Add Dory compatibility helpers to `jolt-dory`.
- Add missing polynomial helpers to `jolt-poly`.

Acceptance criteria:

- `jolt-verifier` gets smaller or simpler.
- Bolt target code shape gets simpler.
- No compatibility regression with `jolt-core` prover output.

Review stop:

- Decide the next split target: `jolt-prover`, shared proof format crate, or
  compiler integration.

## Testing Strategy

### Compile Checks

Run after each code-bearing step:

```bash
cargo check -p jolt-verifier --quiet
cargo fmt -q
```

When the crate reaches real verifier logic, use clippy:

```bash
cargo clippy -p jolt-verifier --all-targets -- -D warnings
```

### Differential Checks

Use `jolt-core` only in tests:

- Construct or generate proof/preprocessing with `jolt-core`.
- Serialize proof/preprocessing.
- Deserialize with `jolt-verifier`.
- Verify with `jolt-verifier`.
- Verify the same artifact with `jolt-core`.

### Required End-to-End Checks

Standard-mode `muldiv` is an interim checkpoint:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-verifier muldiv --cargo-quiet
```

The crate is not complete until ZK-mode and advice proof verification are also
covered. Before claiming verifier completeness:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
cargo nextest run -p jolt-verifier muldiv --cargo-quiet --features zk
```

Advice tests are required in both standard and ZK modes before treating claim
reconstruction and BlindFold bindings as stable.

## Review Discipline

This project should proceed one reviewed slice at a time.

For each implementation step:

1. State the exact step being implemented.
2. Make only the files needed for that step.
3. Run the checks listed for that step.
4. Record API pressure items immediately.
5. Stop for review before proceeding.

Do not combine stage ports. Do not silently move reusable code into modular
crates without review. Do not add Bolt-specific codegen conveniences until the
handwritten model is correct and readable.

## Open Questions

- Should `JoltProof` ultimately live in `jolt-verifier`, a future
  `jolt-proof`, or another shared crate?
- Should verifier config types move to `jolt-program` or remain local to the
  verifier/prover split?
- Should BlindFold become its own modular crate before or after the first
  complete `jolt-verifier`?
- Should `jolt-verifier` initially support only Dory, or should the public API
  stay generic over the modular PCS trait from the beginning?
- Should the first differential fixture be synthetic, `muldiv`, or an even
  smaller dedicated verifier fixture?
