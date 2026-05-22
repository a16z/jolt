# Spec: Wrapper Protocol And SNARK Backend

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-21 |
| Status | draft |
| PR | TBD |

## Purpose

The wrapper proves an R1CS encoding of the configured Jolt verifier computation.
The first backend is zero-knowledge Spartan + HyperKZG. The backend consumes
arbitrary R1CS; Jolt-specific protocol assembly happens before backend proving.

This spec owns the wrapper protocol axis:

```text
configured verifier computation
  -> R1CS assembly
  -> ZK Spartan + HyperKZG proof
```

Field inline and Dory assist add checks to the configured linear verifier flow.
Their protocol details are specified in
[field-inline-protocol.md](field-inline-protocol.md) and
[dory-assist-protocol.md](dory-assist-protocol.md). Proof-shape validation,
compile-time verifier config, and wrapper entry-point rules are specified in
[selected-verifier-integration.md](selected-verifier-integration.md).

## Scope

V1 scope:

```text
generic configured-verifier R1CS builder
transparent Jolt proof as private wrapper witness
in-circuit transcript replay
variable-challenge sumcheck R1CS
claim lowering over constants and variables
Hyrax R1CS integration when the configured verifier includes Dory assist
zero-knowledge Spartan + HyperKZG backend over arbitrary R1CS
```

Out of scope:

```text
making wrapper depend on BlindFold internals
wrapping the BlindFold verifier as the primary v1 ZK composition
folding / relaxed R1CS / row hiding
one universal circuit shape for every configured verifier mode
full gnark adapter
deleting the existing Groth16 transpiler path
```

## Design Intuition

The wrapper is a compiler from verifier computation to R1CS plus a SNARK
backend:

```text
configured verifier computation
  -> protocol-R1CS builder
  -> transcript constraints
  -> sumcheck verifier constraints
  -> claim equation constraints
  -> opening/PCS verifier constraints
  -> R1CS instance and witness
  -> ZK Spartan + HyperKZG proof
```

The backend must remain boring and zero-knowledge:

```text
input:
  ConstraintMatrices<F>
  witness Vec<F>
  public inputs Vec<F>

output:
  wrapper proof
```

It should not know about Jolt stages, field inline, Dory assist, BlindFold, or
transcript ordering. It must hide the R1CS witness, which includes the inner
transparent Jolt proof, clear opening claims, verifier intermediate values, and
any auxiliary variables allocated during wrapper assembly.

## Relationship To BlindFold

BlindFold already establishes the useful R1CS assembly pattern:

```text
build a protocol statement from verifier-stage semantics
allocate verifier-equation witness variables
lower claim expressions
append generic sumcheck constraints
connect each stage's output claim to the next protocol component
```

The wrapper reuses this pattern, not the BlindFold protocol. The primary v1
wrapped-ZK composition is:

```text
transparent Jolt proof
  -> private wrapper witness
  -> wrapper R1CS proves the transparent configured verifier accepts
  -> ZK Spartan + HyperKZG hides the verifier witness
```

In this composition, the wrapper R1CS does not prove the BlindFold verifier.
Base-layer BlindFold remains the standalone Jolt ZK path. If a future product
needs to wrap an already-BlindFolded Jolt proof, then the configured verifier
R1CS can include BlindFold verification as an optional composition, but that is
not the primary v1 wrapper target.

```text
BlindFold builder:
  configured stage semantics
  + committed sumcheck consistency
  + challenges baked as constants
  -> verifier-equation R1CS for the BlindFold proof

Wrapper builder:
  configured verifier computation
  + transparent Jolt proof as private witness data
  + transcript absorbs inside R1CS
  + challenge variables derived inside R1CS
  -> full verifier R1CS for ZK Spartan + HyperKZG
```

The wrapper does not need:

```text
Nova folding
relaxed R1CS
random satisfying instance
BlindFold witness row hiding
BlindFold-specific committed output claim rows
in-circuit BlindFold verification for the primary wrapped-ZK path
```

The wrapper does need:

```text
in-circuit Fiat-Shamir
challenge variables
public-input ordering
proof-data allocation
component R1CS helpers
```

## `jolt-wrapper` Layout

Target layout:

```text
crates/jolt-wrapper/
  Cargo.toml
  src/
    lib.rs
    config.rs
    proof.rs
    public_inputs.rs
    r1cs/
      mod.rs
      assembly.rs
      builder.rs
      sources.rs
      stages.rs
      instance.rs
      public_inputs.rs
      witness.rs
    snarks/
      mod.rs
      spartan_hyperkzg/
        mod.rs
        setup.rs
        prover.rs
        verifier.rs
        proof.rs
      gnark/
        mod.rs
```

`jolt-wrapper::r1cs` owns configured-verifier assembly. Reusable component
encodings live in the crates that own those protocols.

## Protocol Builder

`jolt-wrapper::r1cs` owns a generic protocol builder for the configured verifier
computation:

```rust
pub struct WrapperProtocolBuilder<F, Tr> {
    pub assembly: jolt_r1cs::R1csAssembly<F>,
    pub transcript: Tr,
    pub sources: WrapperClaimSources<F>,
}
```

`R1csAssembly` is a generic `jolt-r1cs` type: a builder plus stable public input
ordering and final matrix/witness export. `WrapperProtocolBuilder` adds
wrapper-specific transcript state, verifier source maps, and proof
artifact allocation.

Helper shape:

```rust
impl<F, Tr> WrapperProtocolBuilder<F, Tr> {
    pub fn alloc_public(
        &mut self,
        id: WrapperPublicInputId,
        value: F,
    ) -> Variable;

    pub fn alloc_witness(
        &mut self,
        id: WrapperWitnessId,
        value: F,
    ) -> Variable;

    pub fn absorb_public(&mut self, variable: Variable);
    pub fn absorb_witness(&mut self, variable: Variable);

    pub fn challenge(&mut self, id: WrapperChallengeId) -> Variable;
}
```

Each configured verifier stage lowers through an R1CS hook:

```rust
pub trait WrapperR1csStage<F> {
    type Error;

    fn append_r1cs<Tr>(
        &self,
        builder: &mut WrapperProtocolBuilder<F, Tr>,
    ) -> Result<(), Self::Error>;
}
```

The stage hook mirrors native verifier execution:

```text
allocate exactly the proof data the native verifier reads
absorb the same proof/public data in the same order
derive challenge variables
call component R1CS helpers
register openings and claims needed by later stages
```

## `jolt-r1cs` Substrate

`jolt-r1cs` provides generic construction machinery. It should not know Jolt
stage semantics.

Target additions:

```text
crates/jolt-r1cs/src/
  assembly.rs
  builder.rs
  constraint.rs
  lowering.rs
  matrices.rs
  protocol.rs
```

Responsibilities:

```text
assembly.rs:
  R1csAssembly, public-input registration, witness/matrix export

lowering.rs:
  SourceValue, ClaimSources, lower_claim_expr, assert_claim_expr_eq

protocol.rs:
  small generic helpers for claim-input -> component constraints ->
  claim-output stage lowering

constraints/:
  execution-relation constraints such as field-inline field_constraints
```

`R1csAssembly` should track public inputs explicitly:

```rust
pub struct R1csAssembly<F> {
    pub builder: R1csBuilder<F>,
    pub public_inputs: Vec<PublicInputBinding>,
}

pub struct PublicInputBinding {
    pub id: PublicInputId,
    pub variable: Variable,
}
```

Stable public-input ordering is part of the wrapper statement. It must be
deterministic across prover and verifier.

## Claim Lowering

`jolt-r1cs::lowering` should remain the single claim-lowering API. It needs to
generalize source values instead of adding a parallel symbolic-lowering module.

Current lowering treats openings as variables and challenges/publics as
constants. Wrapper lowering needs challenges and some public values to be R1CS
variables or linear combinations.

Adjusted shape:

```rust
pub enum SourceValue<F> {
    Constant(F),
    LinearCombination(LinearCombination<F>),
}

pub trait ClaimSources<F> {
    type Opening;
    type Challenge;
    type Public;

    fn opening(&mut self, id: &Self::Opening)
        -> Result<SourceValue<F>, ClaimLoweringError>;
    fn challenge(&mut self, id: &Self::Challenge)
        -> Result<SourceValue<F>, ClaimLoweringError>;
    fn public(&mut self, id: &Self::Public)
        -> Result<SourceValue<F>, ClaimLoweringError>;
}
```

`lower_claim_expr` keeps a constant fast path:

```text
only constants:
  produce a constant linear combination

one non-constant factor:
  stay linear

multiple non-constant factors:
  allocate multiplication constraints through R1csBuilder::multiply
```

This gives both modes through one API:

```text
BlindFold:
  ChallengeId -> SourceValue::Constant(F)
  PublicId    -> SourceValue::Constant(F)

Wrapper:
  ChallengeId -> SourceValue::LinearCombination(challenge variable)
  PublicId    -> SourceValue::LinearCombination(public input variable)
```

The existing `ClaimSourceTable` remains useful as the constant-source
implementation. The wrapper provides `WrapperClaimSources`, backed by
transcript-derived challenges, public inputs, opening variables, and stage-local
aliases.

## Sumcheck R1CS

`jolt-sumcheck::r1cs` already owns the generic sumcheck verifier equations.
The wrapper needs variable challenges.

The constant-challenge API stays for BlindFold and tests:

```text
round.challenge() -> F
```

The wrapper path adds:

```rust
pub struct SumcheckR1csRoundInput {
    pub degree: usize,
    pub challenge: Variable,
}
```

For each round:

```text
round-sum check:
  sum over verifier domain equals claim_in
  this remains a linear equality

evaluation-at-challenge check:
  s_i(r_i) = claim_out
  this uses variable challenge r_i
```

Use Horner constraints for the evaluation:

```text
eval = c_d
eval = eval * r + c_{d-1}
...
assert eval == claim_out
```

The wrapper obtains `r` from `jolt-transcript::r1cs`, so the same challenge
variable is used by transcript binding, claim expressions, and sumcheck
verifier equations.

## Transcript R1CS

`jolt-transcript` needs an `r1cs` module for in-circuit Fiat-Shamir:

```text
crates/jolt-transcript/src/r1cs/
  mod.rs
  sponge.rs
  poseidon.rs
  absorb.rs
  challenge.rs
```

API shape:

```rust
pub trait R1csTranscript<F> {
    fn absorb_scalar(&mut self, builder: &mut R1csBuilder<F>, value: Variable);
    fn absorb_public_scalar(&mut self, builder: &mut R1csBuilder<F>, value: Variable);
    fn challenge_scalar(&mut self, builder: &mut R1csBuilder<F>) -> Variable;
}
```

The initial wrapper target should use the SNARK-friendly transcript selected for
recursion. Native verifier replay and wrapper assembly must share the same
transcript order and scalar encoding.

## R1CS Ownership

Reusable R1CS encodings live with the crates that own the corresponding
protocol machinery:

```text
R1CS builder/matrices:
  jolt-r1cs::builder, jolt-r1cs::assembly, jolt-r1cs matrices

transcript hashing:
  jolt-transcript::r1cs

sumcheck verifier equations:
  jolt-sumcheck::r1cs, including variable-challenge wrapper checks

claim formulas:
  jolt-claims formulas lowered through jolt-r1cs::lowering

opening consistency:
  jolt-openings + jolt-r1cs helpers

field-inline field_constraints:
  jolt-r1cs::constraints::field_constraints

Future BlindFold verifier checks:
  jolt-blindfold, using the same constant-source claim/sumcheck R1CS helpers

Hyrax verifier checks:
  jolt-hyrax::r1cs

Dory-assist prefix packing:
  jolt-claims::protocols::dory_assist::packing R1CS helper
```

`jolt-wrapper::r1cs::assembly` sequences these helpers according to the
configured verifier flow exported by `jolt-verifier`.

## Assembly API

```rust
pub struct WrapperAssemblyInputs<F, PCS, VC, ZkProof, PcsAssistProof>
where
    PcsAssistProof: PcsProofAssist<PCS>,
{
    pub preprocessing: JoltVerifierPreprocessing<PCS, VC>,
    pub public_io: JoltDevice,
    pub proof: JoltProof<PCS, VC, ZkProof, PcsAssistProof>,
    pub public_inputs: WrapperPublicInputs<F>,
    pub witness_inputs: WrapperWitnessInputs<F>,
}

pub struct WrapperR1csInstance<F> {
    pub matrices: ConstraintMatrices<F>,
    pub witness: Vec<F>,
    pub public_inputs: Vec<F>,
}

pub fn assemble_configured_verifier_r1cs<
    F,
    PCS,
    VC,
    ZkProof,
    PcsAssistProof,
>(
    inputs: WrapperAssemblyInputs<F, PCS, VC, ZkProof, PcsAssistProof>,
) -> Result<WrapperR1csInstance<F>, WrapperError>
where
    PcsAssistProof: PcsProofAssist<PCS>,
{
    /* assemble configured verifier R1CS */
}
```

Wrapper assembly uses the same compile-time-derived `JOLT_VERIFIER_CONFIG` as
the configured inner verifier. `WrapperAssemblyInputs` does not contain a
runtime protocol selector; it validates `proof.protocol` against the configured
inner verifier before assembling R1CS.

Backend API:

```rust
pub fn prove_spartan_hyperkzg<F, P>(
    setup: &SpartanHyperKzgProverSetup<F, P>,
    instance: WrapperR1csInstance<F>,
) -> Result<WrapperProof<P>, WrapperError>;

pub fn verify_spartan_hyperkzg<F, P>(
    setup: &SpartanHyperKzgVerifierSetup<F, P>,
    proof: &WrapperProof<P>,
    public_inputs: &WrapperPublicInputs<F>,
) -> Result<(), WrapperError>;
```

## Challenge Binding

For a self-contained wrapper, transcript challenges must be derived inside the
R1CS from the same messages absorbed by native verification:

```text
proof data / public inputs
  -> transcript absorb constraints
  -> challenge variables
  -> sumcheck and claim constraints use those same variables
```

A split verifier can expose challenges as public inputs, but then the outer
verifier must recompute and bind them. The standalone wrapper target uses the
self-contained path.

## Wrapper Circuit Shape

V1 should use one wrapper circuit shape per configured verifier computation:

```text
ordinary Jolt wrapper:
  setup for ordinary configured verifier R1CS

field-inline Jolt wrapper:
  setup for FR-active configured verifier R1CS

Dory-assist Jolt wrapper:
  setup for Dory-assist configured verifier R1CS
```

Do not force one parameterized circuit with runtime inactive rows in v1. The
SNARK backend sees a fixed R1CS for each proof shape. If a future deployment
needs one universal circuit, selector-gated inactive protocol components can be
designed separately.

## Relationship To Existing Groth16 Transpiler

The existing `transpiler/` path symbolically runs the current verifier for
stages 1-7 and emits gnark code. It is useful reference infrastructure and a
near-term Groth16 path, but it is not the target modular architecture.

`jolt-wrapper` is the modular replacement path:

```text
current transpiler:
  jolt-core verifier execution -> MleAst -> gnark

modular wrapper:
  configured verifier computation -> component R1CS helpers -> generic SNARK backend
```

The future `snarks/gnark` adapter may reuse lessons from the transpiler, but
the wrapper should not use `MleAst` as the main verifier IR.

## ZK Composition

ZK configuration is owned by
[selected-verifier-integration.md](selected-verifier-integration.md).

The primary v1 wrapped-ZK composition is:

```text
inner proof:
  transparent Jolt proof, never published as a standalone artifact

wrapper R1CS:
  proves the transparent configured verifier accepts that proof

wrapper backend:
  zero-knowledge Spartan + HyperKZG hides the inner proof and all verifier
  witness data

outer public statement:
  public IO
  preprocessing / verifier-key digest
  configured protocol identifier
  wrapper public inputs required by the selected deployment
```

This avoids encoding the BlindFold verifier in the first wrapper circuit. It is
also the cleanest separation of concerns: transparent Jolt supplies a complete
information-theoretic verifier relation, while the wrapper SNARK supplies the
zero-knowledge hiding for the final proof.

The optional future composition remains:

```text
BlindFold before wrapper:
  Jolt prover produces a standalone BlindFold proof
  configured verifier checks that BlindFold proof
  wrapper proves that configured BlindFold-mode verifier computation
```

That path is useful only when the base Jolt proof must itself be a standalone
ZK artifact before wrapping. It should not drive the v1 wrapper architecture.

## Implementation Steps

Each step should be reviewed before continuing to the next.

1. Add `jolt-r1cs::R1csAssembly`.
   - Track stable public-input ordering.
   - Export matrices, witness, and publics.
   - Review gate: deterministic public-input order tests.

2. Generalize `jolt-r1cs::lowering`.
   - Add `SourceValue::{Constant, LinearCombination}`.
   - Keep existing constant-source tests passing.
   - Review gate: mixed constant/variable claim tests cover products and
     aliases.

3. Add variable-challenge `jolt-sumcheck::r1cs`.
   - Keep constant-challenge API for BlindFold.
   - Add Horner evaluation constraints for variable challenges.
   - Review gate: variable-challenge tests reject bad challenge assignments.

4. Add `jolt-transcript::r1cs`.
   - Implement the selected SNARK-friendly transcript gadget.
   - Match native transcript scalar encoding.
   - Review gate: R1CS challenge outputs match native transcript outputs for
     fixed absorption sequences.

5. Add wrapper protocol builder skeleton.
   - Implement `WrapperProtocolBuilder`, `WrapperClaimSources`, and configured
     stage hook traits.
   - Review gate: synthetic stage lowers to satisfied R1CS.

6. Assemble base configured verifier R1CS.
   - Start with a narrow synthetic verifier computation before full Jolt.
   - Review gate: native verifier and wrapper R1CS agree on accept/reject for
     the same fixture.

7. Add Dory-assist and Hyrax hooks.
   - Call `jolt-hyrax::r1cs` and Dory-assist claim/packing helpers when the
     configured verifier includes Dory assist.
   - Review gate: synthetic Dory-assist configured computation produces satisfied
     R1CS.

8. Add field-inline wrapper hooks.
   - Include FR stages when the configured verifier includes field inline.
   - Review gate: FR-off and FR-on configs produce distinct deterministic
     R1CS shapes.

9. Implement ZK `snarks/spartan_hyperkzg`.
   - Consume arbitrary `WrapperR1csInstance`.
   - Keep the backend independent from Jolt protocol types.
   - Hide the R1CS witness, including transparent inner proof data.
   - Review gate: synthetic arbitrary R1CS proof verifies and witness
     randomization changes proof bytes for the same public statement.

10. Add configured-verifier wrapper fixture.
   - Prove a small transparent configured verifier computation end to end with
     the transparent proof held as private wrapper witness.
   - Review gate: mutating transcript challenges, public inputs, sumcheck
     claims, or opening values causes wrapper verification failure.
