# Spec: Dory Assist Protocol

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-21 |
| Status | draft |
| PR | TBD |

## Purpose

Dory assist is the recursion-paper path for replacing expensive ordinary Dory
stage-8 verifier work with an auxiliary proof. It is the Dory-specific
implementation of the generic PCS proof-assist boundary exposed by
`jolt-verifier`.

From the `jolt-verifier` point of view, the Jolt proof carries a generic
optional PCS-assist payload, e.g. `Option<T>` where `T: PcsProofAssist<PCS>`.
The verifier config decides whether such a payload is required and which assist
implementation is selected. Dory-specific staging does not live in
`jolt-verifier`; it lives in a Dory-assist verifier crate that implements the
generic PCS-assist interface for the Dory PCS.

This spec owns the Dory-assist protocol semantics in `jolt-claims`, the
Dory-assist verifier crate, and the Hyrax commitment layer used by that proof.
Composition with field inline, wrapper proving, ZK, and proof configuration is
specified in
[selected-verifier-integration.md](selected-verifier-integration.md). The
wrapper R1CS compiler is specified in [wrapper-protocol.md](wrapper-protocol.md).

References:

- Recursion paper repo: <https://github.com/markosg04/recursion-paper>
- Quang's `quang/recursion-temp` branch is the concrete protocol reference for
  Dory-assist staging, operation families, wiring, prefix packing, and Hyrax
  opening flow.

The modular implementation is new code. The reference branch informs staging
and formulas; it is not copied as an architectural dependency.

## Scope

V1 scope:

```text
Dory-assist protocol facts in jolt-claims
Dory-assist verifier crate implementing PcsProofAssist for the Dory PCS
generic PCS-assist payload consumed by jolt-verifier
Hyrax dense-trace opening
multi-Miller-loop work proven inside the assist proof
final exponentiation checked natively by the Dory-assist verifier
wrapper-compatible R1CS hooks for sumcheck, claims, packing, and Hyrax
```

Out of scope:

```text
ordinary Dory verifier inside wrapper R1CS
pairing final exponentiation inside the Dory-assist SNARK
one-shot port of all recursion branch implementation internals
making Dory assist depend on field-inline-specific machinery
Dory-specific stage modules inside jolt-verifier
```

## Boundary Model

The ownership split is:

```text
jolt-verifier:
  generic configured verifier flow
  PcsProofAssist trait or equivalent verifier-facing abstraction
  Option<T: PcsProofAssist<PCS>> proof payload slot
  proof/config shape validation
  opening snapshot construction
  dispatch to the selected assist implementation

jolt-claims:
  Dory-assist protocol semantics
  stage IDs, public IDs, opening IDs, dimensions, and claim formulas
  operation-family and packing constraints

dory-assist-verifier:
  Dory-specific staged verifier organization
  proof payload type implementing PcsProofAssist for the Dory PCS
  transcript ordering for the Dory-assist stages
  Hyrax dense-trace opening verification
  native final-exponentiation/public pairing equality check

jolt-hyrax:
  reusable Hyrax commitment, opening, verifier, and R1CS helpers
```

`jolt-verifier` should not match on Dory-assist stages directly. A Dory-enabled
build selects the Dory PCS and a Dory-assist proof type; the generic opening
phase then validates `proof.pcs_assist` against the configured shape and calls
the selected `PcsProofAssist` implementation.

## Protocol Target

The Dory assist proof treats the Dory verifier computation as a typed execution
trace:

```text
public inputs:
  Dory proof artifact / joint PCS opening proof
  Dory verifier setup inputs
  Jolt evaluation claims and commitments from stage 8
  transcript-derived scalars

private witness:
  typed Dory verifier trace:
    GT exponentiation and multiplication
    G1/G2 scalar multiplication and addition
    multi-Miller-loop trace rows
    operation outputs and wiring values

commitment:
  Hyrax commitment to the packed dense trace
```

The proof establishes:

```text
local correctness:
  each operation family satisfies its algebraic relation

wiring correctness:
  outputs consumed by later operations match producer outputs

public-input consistency:
  public Dory proof inputs and Jolt evaluation claims are the values used in
  the operation trace

packing correctness:
  many native-size operation traces are packed into one dense polynomial

opening correctness:
  Hyrax opens that dense polynomial at the verifier's required point
```

## Stage Shape

The reference branch has moved toward a three-stage prefix-packing pipeline.
The modular protocol should preserve this high-level shape:

```text
stage 1:
  packed GT exponentiation sumcheck

stage 2:
  batched constraints:
    GT shift and claim reduction
    GT multiplication
    G1/G2 scalar multiplication
    G1/G2 addition
    multi-Miller-loop constraints
    AST-derived wiring and public-input constraints

stage 3:
  prefix packing reduction to one dense polynomial opening

PCS opening:
  Hyrax opening of the dense trace
```

Dory assist proves the multi-Miller-loop work. The Dory-assist verifier receives
the resulting public GT value, computes final exponentiation directly, and
checks the public pairing equality. Final exponentiation is cheap deterministic
verifier work and stays native in v1, but it is still Dory-specific verifier
logic owned by the Dory-assist verifier implementation rather than by
`jolt-verifier`.

## Component Model

The component split tracks the Dory verifier's algebraic domains:

```text
constraints:
  shared constraint families, poly types, arity, and sumcheck shapes

gt:
  GT exponentiation, GT multiplication, base powers, GT shift, GT wiring

g1:
  G1 addition, G1 scalar multiplication, scalar-mul shift, G1 wiring

g2:
  G2 addition, G2 scalar multiplication, scalar-mul shift, G2 wiring

pairing:
  multi-Miller-loop claims and public final-exponentiation check inputs

packing:
  prefix-packing layout and dense-opening claim
```

The Dory assist proof is a multi-stage virtualization protocol in the same
sense that base Jolt has multiple protocol components. The implementation
should mirror the base Jolt organization: small top-level modules define IDs,
stage shapes, dimensions, and public inputs; component modules own
relation-specific claims, openings, and wiring.

## `jolt-claims` Layout

Target layout:

```text
crates/jolt-claims/src/protocols/dory_assist/
  mod.rs
  config.rs
  ids.rs
  stage.rs
  proof_shape.rs
  public_inputs.rs
  dimensions.rs
  transcript.rs
  constraints/
    mod.rs
    shape.rs
    poly_types.rs
    sumcheck.rs
  gt/
    mod.rs
    claims.rs
    openings.rs
    wiring.rs
  g1/
    mod.rs
    claims.rs
    openings.rs
    wiring.rs
  g2/
    mod.rs
    claims.rs
    openings.rs
    wiring.rs
  pairing/
    mod.rs
    claims.rs
    openings.rs
    final_check.rs
  packing/
    mod.rs
    prefix.rs
    dense_opening.rs
```

Initial API shape:

```rust
pub struct DoryAssistConfig {
    pub pack_bits: usize,
}

pub enum DoryAssistStageId {
    PackedGtExp,
    BatchedConstraints,
    PrefixPacking,
}

pub enum DoryAssistChallengeId {
    Gt(GtChallenge),
    G1(G1Challenge),
    G2(G2Challenge),
    Pairing(PairingChallenge),
    Packing(PackingChallenge),
}

pub enum DoryAssistOpeningId {
    Gt(GtOpening),
    G1(G1Opening),
    G2(G2Opening),
    Pairing(PairingOpening),
    DenseTrace,
}

pub enum DoryAssistPublicId {
    DoryProofArtifact,
    VerifierSetupDigest,
    JoltEvaluationClaim,
    PairingFinalCheckInput,
}

pub struct DoryAssistDimensions {
    pub constraints: DoryAssistConstraintDimensions,
    pub gt: GtDimensions,
    pub g1: G1Dimensions,
    pub g2: G2Dimensions,
    pub pairing: PairingDimensions,
    pub packing: PrefixPackingDimensions,
}

pub struct DoryAssistStageClaims<F> {
    pub id: DoryAssistStageId,
    pub sumcheck: DoryAssistSumcheckSpec,
    pub input: DoryAssistInputClaimExpression<F>,
    pub output: DoryAssistOutputClaimExpression<F>,
    pub consistency: Vec<DoryAssistConsistencyClaim<F>>,
}

pub struct DoryAssistProtocolClaims<F> {
    pub stages: Vec<DoryAssistStageClaims<F>>,
}

pub struct DoryAssistPublicInputs<F> {
    pub jolt_evaluation_claims: JoltEvaluationClaims<F>,
    pub dory_proof: DoryProofPublicInput<F>,
    pub verifier_setup_digest: F,
    pub pairing_final_check_inputs: PairingFinalCheckInputs<F>,
}
```

`proof_shape.rs` should define the stage ordering, payload kinds, and validation
facts needed by downstream verifiers. The concrete serializable
`DoryAssistProof` payload lives in the Dory-assist verifier crate, but it should
remain typed by stage and operation family rather than routing claims through an
untyped opening map.

## Hyrax

`jolt-hyrax` factors Hyrax out as a reusable modular crate. Dory assist uses
Hyrax to commit to and open the packed dense trace. The first implementation is
a native transparent PCS adapter over `jolt_crypto::VectorCommitment`. Wrapper
assembly later adds `jolt-hyrax::r1cs` to prove the same opening verification
inside R1CS when the configured verifier is wrapped.

V1 native layout:

```text
crates/jolt-hyrax/
  Cargo.toml
  src/
    lib.rs
    dimensions.rs
    error.rs
    commitment.rs
    proof.rs
    setup.rs
    scheme.rs
```

Future R1CS layout:

```text
crates/jolt-hyrax/src/r1cs/
  mod.rs
  inputs.rs
  witness.rs
  constraints.rs
```

Hyrax should use existing abstractions rather than defining parallel ones:

```text
VectorCommitment
VectorCommitmentOpening
HomomorphicCommitment
Pedersen / PedersenSetup
JoltGroup
AppendToTranscript
jolt_openings::CommitmentScheme
jolt_openings::AdditivelyHomomorphic
```

Native API sketch:

```rust
pub struct HyraxDimensions {
    pub num_vars: usize,
    pub row_vars: usize,
    pub col_vars: usize,
}

pub struct HyraxCommitment<C> {
    pub rows: Vec<C>,
}

pub struct HyraxOpeningProof<F> {
    pub row_opening: VectorCommitmentOpening<F>,
}

pub struct HyraxScheme<VC: VectorCommitment> { ... }

impl<VC> CommitmentScheme for HyraxScheme<VC>
where
    VC: VectorCommitment<Field = F>,
    VC::Output: HomomorphicCommitment<F>;
```

The point split is fixed as:

```text
num_vars = row_vars + col_vars
row_point = point[..row_vars]
col_point = point[row_vars..]
```

Hyrax combines committed rows using `row_point`, then verifies the combined row
opening at `col_point`. The implementation should delegate row commitments,
row-combination openings, and commitment homomorphism to `VectorCommitment`,
`VectorCommitmentOpening`, and `HomomorphicCommitment`.

The default instantiation can be Pedersen-backed:

```text
PedersenHyrax<G> = Hyrax<Pedersen<G>>
```

but verifier APIs should be generic over the vector commitment abstraction.

Hyrax setup should compose with existing `jolt_crypto::DeriveSetup` impls. For
Dory assist, the intended path is:

```text
DoryProverSetup
  -> PedersenSetup<Bn254G1> via DeriveSetup<DoryProverSetup>
  -> HyraxProverSetup<Pedersen<Bn254G1>> using HyraxDimensions.row_len()
```

`HyraxDimensions` are not derivable from the PCS SRS and must be supplied by the
Dory-assist dense-trace layout.

## Generic PCS-Assist Integration

`jolt-verifier` owns the configured linear verifier flow and the generic
opening-phase assist boundary. Dory assist is one implementation of that
boundary, selected only when the configured PCS is Dory and the verifier config
requires the Dory-assist payload.

Verifier-facing API sketch:

```rust
pub trait PcsProofAssist<PCS: CommitmentScheme>: Sized {
    type Config;
    type Error;

    fn verify<T>(
        &self,
        config: &Self::Config,
        joint_opening_proof: &PCS::Proof,
        opening_snapshot: &PcsOpeningSnapshot<PCS>,
        transcript: &mut T,
    ) -> Result<(), Self::Error>
    where
        T: Transcript<Challenge = PCS::Field>;
}
```

The exact trait spelling can move during implementation, but the boundary
should preserve these semantics:

```text
ordinary configured Jolt:
  stages 1-7
  -> build opening snapshot
  -> proof.pcs_assist must be None
  -> verify joint_opening_proof through the ordinary PCS verifier

Dory-assisted configured Jolt:
  stages 1-7
  -> build the same opening snapshot
  -> proof.pcs_assist must be Some(DoryAssistProof)
  -> call DoryAssistProof::verify through the PcsProofAssist boundary
  -> do not also run the expensive ordinary Dory verifier path
```

The configured verifier flow and proof-shape rules live in
[selected-verifier-integration.md](selected-verifier-integration.md).

## `dory-assist-verifier` Layout

The Dory-assist verifier crate owns the concrete organization of the semantics
defined in `jolt-claims::protocols::dory_assist`.

Target layout:

```text
crates/jolt-dory-assist-verifier/
  Cargo.toml
  src/
    lib.rs
    config.rs
    proof.rs
    public_inputs.rs
    transcript.rs
    opening_snapshot.rs
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
    hyrax_opening/
      mod.rs
      inputs.rs
      outputs.rs
      verify.rs
    final_check.rs
```

The crate should expose a typed proof payload and implement the generic
PCS-assist boundary for the Dory PCS:

```rust
pub struct DoryAssistProof<F, T, HyraxOpeningProof> {
    pub stage1_proof: SumcheckInstanceProof<F, T>,
    pub stage2_proof: SumcheckInstanceProof<F, T>,
    pub stage3_packed_eval: F,
    pub opening_proof: HyraxOpeningProof,
    pub opening_claims: DoryAssistOpeningClaims<F>,
    pub dense_commitment: HyraxCommitment,
}

impl PcsProofAssist<DoryCommitmentScheme> for DoryAssistProof<...> {
    type Config = DoryAssistConfig;
    type Error = DoryAssistVerifierError;

    fn verify<T>(...) -> Result<(), Self::Error>
    where
        T: Transcript<Challenge = <DoryCommitmentScheme as CommitmentScheme>::Field>,
    {
        // stage 1, stage 2, stage 3, Hyrax opening, native final check
    }
}
```

The proof fields should remain typed by stage and operation family. Production
code should not route claims through an untyped opening map.

## R1CS And Wrapper Hooks

Dory assist contributes verifier computation to the wrapper:

```text
stage 1/2 sumcheck checks:
  jolt-sumcheck::r1cs

operation-family claim equations:
  jolt-claims formulas + jolt-r1cs lowering

wiring and public-input equations:
  dory_assist component claims + jolt-r1cs lowering

prefix packing:
  dory_assist::packing R1CS helper

dense Hyrax opening:
  jolt-hyrax::r1cs
```

The wrapper consumes these helpers through the configured verifier computation
and the selected `PcsProofAssist` implementation. Dory-specific formulas stay
in `jolt-claims`; Dory-specific stage organization stays in the Dory-assist
verifier crate; Hyrax-specific constraints stay in `jolt-hyrax`.

## Interaction With Field Inline

Dory assist is independent of field inline. If field inline is enabled, its
effects are already reflected in the composed verifier stage outputs and
opening data consumed by the Dory-opening path. Dory assist does not need a
separate field-inline mode.

## Implementation Steps

Each step should be reviewed before continuing to the next.

1. Add `jolt-hyrax` native API.
   - Define setup, commitment, proof, opening claim, and verifier APIs.
   - Use `jolt-crypto` vector commitment and group abstractions.
   - Review gate: synthetic Hyrax opening tests pass.

2. Add `jolt-hyrax::r1cs`.
   - Encode the verifier equations needed by Dory assist.
   - Keep R1CS helpers generic over `jolt-r1cs` builders.
   - Review gate: R1CS verifier constraints match native Hyrax verification on
     small fixtures.

3. Add `jolt-claims::protocols::dory_assist`.
   - Define config, IDs, dimensions, stage IDs, public IDs, and opening IDs.
   - Mirror the base Jolt structure rather than inventing a separate verifier
     spec abstraction.
   - Review gate: API shape matches the three-stage reference pipeline.

4. Add operation-family formulas.
   - Add GT, G1, G2, pairing, constraints, and packing modules.
   - Encode local constraints, output claims, and wiring formulas.
   - Review gate: formula tests cover each operation family independently.

5. Add prefix-packing formulas.
   - Define dense trace layout, prefix codewords, and dense-opening claim.
   - Review gate: packing tests catch incorrect prefix layout and opening-point
     normalization.

6. Add the Dory-assist verifier crate and proof shape.
   - Add typed proof payloads for stage 1, stage 2, stage 3, and Hyrax opening.
   - Organize verification stages around the semantic claims from
     `jolt-claims::protocols::dory_assist`.
   - Review gate: proof-shape validation rejects missing, extra, or misordered
     stage payloads.

7. Implement the generic PCS-assist boundary for Dory.
   - Implement `PcsProofAssist<DoryCommitmentScheme>` for the Dory-assist proof
     payload.
   - Build Dory-assist public inputs from the generic opening snapshot and the
     ordinary joint opening proof.
   - Review gate: `jolt-verifier` can dispatch to synthetic Dory-assist
     fixtures without Dory-specific stage modules.

8. Add multi-Miller-loop verification path.
   - Prove multi-Miller-loop trace constraints in Dory assist.
   - Keep final exponentiation as native Dory-assist verifier work.
   - Review gate: fixtures match the Quang reference branch for equivalent
     inputs.

9. Add wrapper R1CS hooks.
   - Lower Dory-assist stages through claim, sumcheck, packing, and Hyrax R1CS
     helpers.
   - Review gate: wrapper assembly can include synthetic Dory-assist stages in
     a satisfied R1CS.
