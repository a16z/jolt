# Spec: Dory Assist Protocol

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-21 |
| Status | draft |
| PR | TBD |

## Purpose

Dory assist is the recursion-paper path for replacing expensive ordinary Dory
stage-8 verifier work with additional verifier stages and an auxiliary proof.
It extends the selected Jolt protocol after the information-theoretic stages by
proving the expensive Dory verifier computation through a Dory-assist SNARK.

This spec owns the Dory-assist protocol axis and the Hyrax commitment layer used
by that proof. Composition with field inline, wrapper proving, ZK, and proof
selection is specified in
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
selected verifier stages after base Jolt stages 1-7
Hyrax dense-trace opening
multi-Miller-loop work proven inside the assist proof
final exponentiation checked natively by the selected verifier
wrapper-compatible R1CS hooks for sumcheck, claims, packing, and Hyrax
```

Out of scope:

```text
ordinary Dory verifier inside wrapper R1CS
pairing final exponentiation inside Dory assist
one-shot port of all recursion branch implementation internals
making Dory assist depend on field-inline-specific machinery
```

## Protocol Target

The Dory assist proof treats the Dory verifier computation as a typed execution
trace:

```text
public inputs:
  Dory proof artifact
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

Dory assist proves the multi-Miller-loop work. The selected verifier receives
the resulting public GT value, computes final exponentiation directly, and
checks the public pairing equality. Final exponentiation is cheap deterministic
verifier work and stays native in v1.

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
    pub enabled: bool,
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

The proof shape follows the reference branch's `RecursionProof` structure:

```rust
pub struct DoryAssistProof<F, T, HyraxProof> {
    pub stage1_proof: SumcheckInstanceProof<F, T>,
    pub stage2_proof: SumcheckInstanceProof<F, T>,
    pub stage3_packed_eval: F,
    pub opening_proof: HyraxProof,
    pub opening_claims: DoryAssistOpeningClaims<F>,
    pub dense_commitment: HyraxCommitment,
}
```

Exact proof fields can change during porting, but they should remain typed by
stage and operation family. Production code should not route claims through an
untyped opening map.

## Hyrax

`jolt-hyrax` factors Hyrax out as a reusable modular crate. Dory assist uses
Hyrax to commit to and open the packed dense trace. Wrapper assembly uses
`jolt-hyrax::r1cs` to prove the same opening verification inside R1CS when the
selected verifier is wrapped.

Target layout:

```text
crates/jolt-hyrax/
  Cargo.toml
  src/
    lib.rs
    commitment.rs
    proof.rs
    setup.rs
    transcript.rs
    verifier.rs
    opening.rs
    r1cs/
      mod.rs
      inputs.rs
      witness.rs
      constraints.rs
```

Hyrax should use `jolt-crypto` abstractions where practical:

```text
VectorCommitment
VectorCommitmentOpening
HomomorphicCommitment
Pedersen / PedersenSetup
JoltGroup
EvaluationClaim
Point
AppendToTranscript
```

Native API sketch:

```rust
pub struct HyraxCommitment<G> {
    pub rows: Vec<G>,
}

pub struct HyraxOpeningProof<F, G> {
    pub proof_data: Vec<F>,
    pub commitment_data: Vec<G>,
}

pub struct HyraxOpeningClaim<F, G> {
    pub commitment: HyraxCommitment<G>,
    pub point: Vec<F>,
    pub value: F,
}

pub fn verify_hyrax_opening<F, VC, T>(
    setup: &HyraxVerifierSetup<VC>,
    claim: &HyraxOpeningClaim<F, VC::Output>,
    proof: &HyraxOpeningProof<F, VC::Output>,
    transcript: &mut T,
) -> Result<(), HyraxError>
where
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>;
```

The default instantiation can be Pedersen-backed:

```text
PedersenHyrax<G> = Hyrax<Pedersen<G>>
```

but verifier APIs should be generic over the vector commitment abstraction.

## `jolt-verifier` Integration

`jolt-verifier` owns the selected stage schedule. Dory assist adds concrete
stages after the ordinary information-theoretic Jolt stages:

```text
ordinary selected Jolt:
  stages 1-7
  -> ordinary stage-8 PCS/Dory opening verification

selected Jolt with Dory assist:
  stages 1-7
  -> build Dory assist public inputs from stage-8 opening data
  -> dory_assist::stage1::verify
  -> dory_assist::stage2::verify
  -> dory_assist::stage3::verify
  -> dory_assist::hyrax_opening::verify through jolt-hyrax
  -> native final exponentiation/public pairing equality check
```

Target verifier layout:

```text
crates/jolt-verifier/src/
  proof.rs
  stages/
    mod.rs
    dory_assist/
      mod.rs
      public_inputs.rs
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
```

The detailed selected-schedule and proof-shape rules live in
[selected-verifier-integration.md](selected-verifier-integration.md).

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

The wrapper consumes these helpers through the selected verifier computation.
Dory-specific formulas stay in `jolt-claims`; Hyrax-specific constraints stay
in `jolt-hyrax`.

## Interaction With Field Inline

Dory assist is independent of field inline. If field inline is enabled, its
effects are already reflected in the base Jolt stage outputs and opening data
consumed by the Dory-opening path. Dory assist does not need a separate
field-inline mode.

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

6. Add Dory-assist proof shape.
   - Add typed proof payloads for stage 1, stage 2, stage 3, and Hyrax opening.
   - Review gate: proof-shape validation rejects missing, extra, or misordered
     stage payloads.

7. Add selected verifier stages.
   - Add concrete `jolt-verifier::stages::dory_assist` modules.
   - Build Dory-assist public inputs from ordinary stage-8 opening data.
   - Review gate: selected verifier can run synthetic Dory-assist fixtures.

8. Add multi-Miller-loop verification path.
   - Prove multi-Miller-loop trace constraints in Dory assist.
   - Keep final exponentiation as native public verifier work.
   - Review gate: fixtures match the Quang reference branch for equivalent
     inputs.

9. Add wrapper R1CS hooks.
   - Lower Dory-assist stages through claim, sumcheck, packing, and Hyrax R1CS
     helpers.
   - Review gate: wrapper assembly can include synthetic Dory-assist stages in
     a satisfied R1CS.
