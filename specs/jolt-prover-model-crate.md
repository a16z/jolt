# Spec: `jolt-prover` Model Crate

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-19 |
| Status | draft |
| PR | TBD |

## Purpose

`jolt-prover` is the handwritten prover model for the modular stack. It
produces `jolt-verifier` proofs without depending on Bolt.

The split is intentional:

```text
jolt-verifier  defines verifier-visible protocol semantics
jolt-prover    coordinates proving and assembles verifier-owned proofs
jolt-witness   exposes private data through reusable witness APIs
jolt-kernels   performs heavy optimized compute
```

Bolt remains a separate track. This crate should unblock modular proving,
ZK/BlindFold, advice, recursion, Dory assist, and wrapper work before Bolt is
ready.

## Core Decisions

- `jolt-prover -> jolt-verifier` is an intentional one-way dependency.
- `JoltProof` and verifier-visible `StageNOutput` types stay in
  `jolt-verifier`.
- `jolt-prover` builds verifier-owned proof and stage-output structs directly.
- Prover-local artifacts carry private execution state: witness handles,
  caches, opening inputs, ZK material, recursion data, and wrapper handoff data.
- ZK/BlindFold and advice are first-class from the first design.
- Committed sumchecks reuse `jolt-sumcheck` and `jolt-crypto` types.
- Witness generation is modular, namespace-generic, and consumed by
  `jolt-prover`.
- PCS implementation crates, such as `jolt-dory` over `dory-pcs`, are valid
  compute dependencies behind `jolt-openings`.
- Heavy compute belongs behind witness/kernel APIs, not in stage orchestration.
- `jolt-verifier` is the primary acceptance oracle as the prover frontier grows.
- `jolt-core` is a compatibility/parity oracle only.

## Non-Goals

- No Bolt dependency or generated prover/verifier code.
- No production calls into `jolt-core` prover internals.
- No transparent-only architecture.
- No dense-witness-only model.
- No wrapper R1CS assembly or wrapper SNARK proving inside `jolt-prover`.
- No stable external prover API yet.

## Ownership

| Crate | Owns |
|-------|------|
| `jolt-verifier` | Proof model, verifier stage outputs, selected verifier schedule |
| `jolt-prover` | Prover orchestration, transcript order, proof assembly, artifacts |
| `jolt-witness` | Generic witness traits plus protocol-specific witness providers |
| `jolt-kernels` | Optimized prover compute and map-reduce kernels |
| `jolt-claims` | Protocol IDs, formulas, opening/public/challenge metadata |
| `jolt-sumcheck` | Clear/committed sumcheck proofs, transcript replay, R1CS lowering |
| `jolt-crypto` | Vector commitments, Pedersen, homomorphic ops, setup derivation |
| `jolt-openings` | Opening plans, evaluation claims, RLC, PCS traits |
| `jolt-dory` / PCS crates | Concrete PCS implementations and optimized commit/open/eval kernels |
| `jolt-blindfold` | Generic BlindFold protocol from committed proofs |
| `jolt-wrapper` | Selected-verifier R1CS and wrapper SNARKs |

## Prover Flow

```text
inputs
  -> witness provider
  -> oracle/advice commitments
  -> stages 1..7
  -> stage 8 opening proof
  -> Dory-assist inputs when selected
  -> BlindFold proof when selected
  -> Dory-assist proof when selected
  -> wrapper witness handoff when requested
```

`jolt-prover` owns this order. Witness providers and kernels answer queries;
they do not decide stage order or transcript semantics.

## Public API

Representative shape:

```rust
pub fn prove<F, PCS, VC, T, K, W, R>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    inputs: JoltProverInputs<'_, F>,
    selection: ProofSelection,
    witness_builder: &mut W,
    kernels: &mut K,
    transcript: &mut T,
    rng: &mut R,
) -> Result<jolt_verifier::JoltProof<PCS, VC>, ProverError>
where
    F: jolt_field::Field,
    PCS: jolt_openings::CommitmentScheme<Field = F>,
    VC: jolt_crypto::VectorCommitment<Field = F>,
    T: jolt_transcript::Transcript<Challenge = F>,
    K: KernelSet<F>,
    W: WitnessBuilder<F>,
    R: rand_core::CryptoRngCore;
```

The primary API returns the verifier-owned proof. Private prover state is
captured by explicit sidecar sinks when a caller needs it:

```rust
pub fn prove_with_sidecars<F, PCS, VC, T, K, W, R, S>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    inputs: JoltProverInputs<'_, F>,
    selection: ProofSelection,
    witness_builder: &mut W,
    kernels: &mut K,
    transcript: &mut T,
    rng: &mut R,
    sidecars: &mut S,
) -> Result<jolt_verifier::JoltProof<PCS, VC>, ProverError>
where
    F: jolt_field::Field,
    PCS: jolt_openings::CommitmentScheme<Field = F>,
    VC: jolt_crypto::VectorCommitment<Field = F>,
    T: jolt_transcript::Transcript<Challenge = F>,
    K: KernelSet<F>,
    W: WitnessBuilder<F>,
    R: rand_core::CryptoRngCore,
    S: ProverSidecarSink<F, PCS, VC>;

pub trait ProverSidecarSink<F, PCS, VC>
where
    PCS: jolt_openings::CommitmentScheme<Field = F>,
    VC: jolt_crypto::VectorCommitment<Field = F>,
{
    fn record_stage<N>(&mut self, stage: StageArtifacts<F, VC, N>) -> Result<(), ProverError>;
    fn record_wrapper_inputs(&mut self, inputs: WrapperWitnessInputs<F>) -> Result<(), ProverError>;
    fn record_dory_assist_inputs(
        &mut self,
        inputs: DoryAssistInputs<F, PCS>,
    ) -> Result<(), ProverError>;
}
```

`prove` is the convenience entry point that uses a no-op sink. Wrapper,
recursion, Dory assist, and diagnostic flows call `prove_with_sidecars` with a
concrete sink that captures only the private data they consume.

Selection-specific bounds:

```text
Transparent:
  PCS: CommitmentScheme

BlindFold:
  PCS: CommitmentScheme + ZkOpeningScheme<HidingCommitment = VC::Output>
  VC: VectorCommitment<Field = F>
  VC::Output: HomomorphicCommitment<F> + AppendToTranscript
```

`rng` is explicit because committed rounds, output-claim commitments, ZK
openings, and BlindFold folding need auditable entropy.

## Proof Selection

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

The final type should align with `jolt-verifier::ProtocolSelection`.

Transparent mode:

```text
SumcheckProof::Clear(...)
public opening claims
stage 8 binds clear joint claim
```

BlindFold mode:

```text
SumcheckProof::Committed(CommittedSumcheckProof<VC::Output>)
private round coefficients/blindings
committed output-claim rows
ZK PCS evaluation commitment
BlindFold verifier-equation proof
```

Visibility is a protocol selection, not a cargo feature architecture.

## Committed Sumchecks

Use existing modular types:

```text
jolt_sumcheck::SumcheckProof<F, VC::Output>
jolt_sumcheck::CommittedSumcheckProof<VC::Output>
jolt_sumcheck::CommittedRoundWitness<F>
jolt_sumcheck::CommittedOutputClaims<VC::Output>
jolt_crypto::VectorCommitment
jolt_crypto::HomomorphicCommitment
jolt_crypto::DeriveSetup
jolt_blindfold::BlindFoldProtocol::from_committed_proofs
```

The prover privately builds `CommittedRoundWitness<F>`, commits with `VC`, puts
`CommittedSumcheckProof<VC::Output>` in the public stage proof, and retains
coefficients/blindings for BlindFold witness construction.

`JoltProverPreprocessing<PCS, VC>` should carry `VC::Setup`. Prefer deriving it:

```text
VC::Setup: DeriveSetup<PCS::ProverSetup>
```

when the PCS prover setup can be the source of vector-commitment generators.

## PCS Implementations

`jolt-prover` depends on PCS behavior through `jolt-openings` traits. Concrete
PCS crates are implementation dependencies, not protocol owners.

For Dory, `jolt-dory` may use `dory-pcs` directly for commitment construction,
evaluation proofs, ZK openings, and streaming hints. In this model the Dory PCS
implementation is a fine-grained kernel behind:

```text
CommitmentScheme
StreamingCommitment
AdditivelyHomomorphic
ZkOpeningScheme
```

The PCS implementation owns optimized compute and proof encoding for that PCS.
It does not own Jolt stage order, opening-claim formulas, transcript schedule,
or proof selection.

A Dory prover configuration can instantiate `PCS = jolt_dory::DoryScheme`; the
stage code should remain generic over the `jolt-openings` traits.

## Stage Model

Each stage is a free function with explicit dependencies:

```rust
pub fn prove<F, PCS, VC, T, K, W, R>(
    inputs: StageNInputs<'_, F>,
    deps: StageNDeps<'_, F>,
    selection: ProofVisibility,
    witness: &W,
    kernels: &mut K,
    transcript: &mut T,
    rng: &mut R,
) -> Result<StageNArtifacts<F, PCS, VC>, ProverError>;
```

Stage code should show:

- input claims;
- transcript absorbs/squeezes;
- clear or committed sumcheck calls;
- batching order;
- output claims;
- opening dependencies;
- private ZK artifacts.

## Verifier-Owned Protocol Objects

`jolt-verifier` defines the final protocol objects. `jolt-prover` imports those
types, computes their fields, and assembles them as the public proof surface.
Stage functions return verifier-owned protocol data plus a private sidecar for
prover-only state.

```rust
pub struct StageArtifacts<F, VC, Private> {
    pub verifier_output: jolt_verifier::stages::stageN::StageNOutput<F>,
    pub proof: jolt_verifier::stages::stageN::StageNProof<F, VC>,
    pub openings: StageOpeningDeps<F>,
    pub zk: Option<StageZkArtifacts<F, VC>>,
    pub private: Private,
}
```

Use two views with distinct responsibilities:

```text
verifier_output:
  verifier-owned protocol facts

private:
  witness handles, kernel state, coefficients, blindings, caches, handoff data
```

Verifier-owned types are the stable protocol target. Prover modules schedule
generic computation around those types and use private sidecars for data that
only the prover, BlindFold, recursion, or wrapper construction consumes.

## Witness Model

Witness generation is a private-data API. It must support base Jolt VM traces
and non-VM witnesses for Dory assist, wrapper, BlindFold, and recursion.

Use namespaces rather than hardcoding VM polynomial IDs:

```rust
pub trait WitnessNamespace {
    type CommittedId;
    type VirtualId;
    type OpeningId;
    type PublicId;
    type ChallengeId;
}
```

Base Jolt VM uses `jolt-claims` IDs:

```rust
pub enum JoltVmWitnessNamespace {}

impl WitnessNamespace for JoltVmWitnessNamespace {
    type CommittedId = jolt_claims::protocols::jolt::JoltCommittedPolynomial;
    type VirtualId = jolt_claims::protocols::jolt::JoltVirtualPolynomial;
    type OpeningId = jolt_claims::protocols::jolt::JoltOpeningId;
    type PublicId = jolt_claims::protocols::jolt::JoltPublicId;
    type ChallengeId = jolt_claims::protocols::jolt::JoltChallengeId;
}
```

Dory assist and wrapper define their own namespaces. Dory-assist witnesses are
not VM-shaped: they expose operation traces, packing witnesses, Miller-loop
witnesses, and public-input handles through a Dory-assist namespace.
Representative Dory-assist IDs are operation trace IDs, packing witness IDs,
Miller-loop witness IDs, and assist public-input IDs; they should not reuse the
Jolt VM witness namespace.

Composable witness traits:

```rust
pub trait WitnessProvider<F, N: WitnessNamespace> {
    fn dimensions(&self) -> &WitnessDimensions;
}

pub trait CommittedPolynomialWitness<F, N: WitnessNamespace>: WitnessProvider<F, N> {
    fn committed_ids(&self) -> &[N::CommittedId];
    fn committed_poly(&self, id: N::CommittedId) -> Result<PolynomialHandle<'_, F>, WitnessError>;
    fn eval_committed(&self, id: N::CommittedId, point: &[F]) -> Result<F, WitnessError>;
}

pub trait VirtualPolynomialWitness<F, N: WitnessNamespace>: WitnessProvider<F, N> {
    fn eval_virtual(
        &self,
        id: N::VirtualId,
        point: &[F],
        ctx: &VirtualEvalContext<F, N>,
    ) -> Result<F, WitnessError>;
}

pub trait OpeningWitnessProvider<F, N: WitnessNamespace>: WitnessProvider<F, N> {
    fn opening_witness(
        &self,
        id: N::OpeningId,
        point: &[F],
    ) -> Result<OpeningWitness<F>, WitnessError>;
}
```

Keep polynomial access handle-based:

```rust
pub enum PolynomialHandle<'a, F> {
    Dense(&'a [F]),
    Compact(CompactHandle<'a>),
    OneHot(OneHotHandle<'a>),
    Streaming(StreamingHandle<'a>),
    Lazy(LazyPolynomialHandle<'a, F>),
}
```

Do not force all witnesses into `Vec<F>`.

`jolt-prover` consumes a builder:

```rust
pub trait WitnessBuilder<F> {
    type Namespace: WitnessNamespace;
    type Witness: WitnessProvider<F, Self::Namespace>;

    fn build(
        &mut self,
        preprocessing: &JoltProgramPreprocessing,
        inputs: JoltProverInputs<'_, F>,
    ) -> Result<Self::Witness, WitnessError>;
}
```

Example implementations:

```text
Jolt VM:
  trace rows, advice tapes, final memory, committed oracles, virtual evals

Dory assist:
  operation traces, packing witnesses, Miller-loop witness, assist public inputs

Wrapper:
  selected verifier values, transcript state, R1CS assignment

BlindFold:
  committed round witnesses, output claim rows, auxiliary rows
```

Witness code does not own protocol order, transcript labels, stage selection,
or claim formulas.

## Kernels

Kernels own optimized compute. They may be coarse or fine-grained:

```text
coarse: prove a whole stage
medium: prove one batched sumcheck
fine: compute one relation message, bind, or evaluation
```

Kernels may own parallel map-reduce loops, sparse/dense evaluation,
prefix/suffix decomposition, RA processing, streaming commitment inputs, and
joint polynomial construction.

Kernels must not own claims, stage order, transcript labels, verifier
equations, or public proof shape.

## Advice

Trusted and untrusted advice share the same witness abstraction.

```text
trusted:
  commitment supplied from preprocessing/trusted setup
  verifier receives it outside JoltProof

untrusted:
  prover commits during proving
  commitment included in JoltProof
```

Artifacts retain padded words, commitment, opening hint, dimensions,
address-phase choice, advice openings, stage-8 selector factors, and BlindFold
claim constraints.

## Stage 8

Stage 8 is typed opening assembly:

```text
stage artifacts
  -> OpeningPlan
  -> RLC reduction
  -> joint polynomial / hint
  -> PCS opening proof
  -> visibility-specific transcript binding
  -> DoryAssistInputs when selected
```

Transparent mode binds the clear joint claim. BlindFold mode uses
`ZkOpeningScheme::open_zk` and retains the hidden evaluation output/blinding for
BlindFold.

When Dory assist is selected, stage 8 also returns typed `DoryAssistInputs`:
opening plan, evaluation claims, commitments, structured Dory proof data, setup
inputs, transcript checkpoints, and claim-to-polynomial mapping. Recursion and
Dory assist should not recover this by parsing opaque proof bytes.

## BlindFold

`jolt-prover` builds the Jolt-specific BlindFold protocol from committed stage
proofs:

```rust
jolt_blindfold::BlindFoldProtocol::from_committed_proofs(...)
```

Inputs come from `jolt-claims`, committed stage proofs, private
coefficients/blindings, stage-8 ZK opening data, and `VC::Setup`.

Changing a claim formula without updating its BlindFold metadata is a
correctness bug.

## Dory Assist And Recursion

Dory assist is a prover flow over a non-VM witness namespace:

```text
stage-8 DoryAssistInputs
  -> Dory-assist public inputs
  -> Dory verifier operation trace witness
  -> operation-family witnesses
  -> Miller-loop witness
  -> prefix-packed dense trace
  -> Hyrax commitment/opening
  -> DoryAssistProof
  -> verifier computes final exponentiation publicly
```

The Dory-assist witness provider should expose:

- operation trace witnesses for EC, field, and pairing operations;
- packing witnesses for dense/prefix-packed Hyrax rows;
- Miller-loop witnesses for line functions and intermediate values.

The policy is fixed: Dory assist proves Miller-loop work; the selected verifier
computes final exponentiation publicly. Selection only chooses whether assist is
emitted.

## Wrapper Handoff

`jolt-prover` exports witness inputs. `jolt-wrapper` owns R1CS assembly and
wrapper proving.

```text
prove(..., WrapperSidecarSink)
  -> JoltProof
  -> WrapperWitnessInputs captured by sink
  -> jolt-wrapper
```

Wrapper inputs may include proof fields in verifier order, verifier stage
outputs, transcript witnesses, BlindFold verifier data, Dory-assist public
inputs, and Hyrax witness data.

## Target Layout

```text
crates/jolt-prover/src/
  lib.rs
  inputs.rs
  preprocessing.rs
  artifacts.rs
  prover.rs
  selection.rs
  commitment/
  stages/
  openings/
  advice/
  zk/
  dory_assist/
  wrapper/

crates/jolt-witness/src/
  namespace.rs
  provider.rs
  polynomial.rs
  opening.rs
  r1cs.rs
  error.rs
  protocols/
    jolt_vm/
      mod.rs
      trace.rs
      advice.rs
      committed.rs
      virtuals.rs
    dory_assist/
      mod.rs
      namespace.rs
      operation_trace.rs
      packing.rs
      miller_loop.rs
    blindfold/
      mod.rs
      committed_rounds.rs
      output_claims.rs
    wrapper/
      mod.rs
      verifier_assignment.rs
      transcript.rs
```

This is a target shape, not a first-PR requirement.

## Testing

Required coverage:

- each implemented prover frontier is accepted by the matching `jolt-verifier`
  frontier;
- transparent and BlindFold modes;
- no advice, trusted-only, untrusted-only, both-advice;
- real advice-consuming guests;
- committed sumcheck and BlindFold artifact shape;
- witness provider reference checks for committed and virtual polynomial evals;
- stage-8 opening plan, ZK opening data, and Dory-assist inputs;
- tampering of transcript, dependencies, openings, committed claims, advice
  commitments, BlindFold inputs, and Dory-assist input data.

Bring-up should be top-down against `jolt-verifier`: implement the next prover
component, assemble the partial verifier-visible proof/checkpoint object, and
require the corresponding verifier stage/frontier to accept it. `jolt-core`
remains useful for fixture generation and parity checks, but it is not the
primary oracle for the modular prover.

## Milestones

1. Scaffold `jolt-prover`: selection, sidecar sinks, explicit RNG, `JoltProof`
   assembly.
2. Add `jolt-witness` traits and a trace-backed VM witness provider under
   `jolt_witness::protocols::jolt_vm`.
3. Add kernel boundaries and a slow/reference kernel path.
4. Implement commitments, including trusted/untrusted advice artifacts.
5. Implement stages 1-2 in transparent and BlindFold mode.
6. Port stages 3-7 with verifier-output sharing.
7. Implement advice reductions and optional address phase.
8. Implement stage 8, ZK opening path, and Dory-assist input data.
9. Prove the BlindFold instance from committed proofs.
10. Add full transparent/ZK prover-verifier E2E with advice.
11. Add Dory-assist witness module under `jolt_witness::protocols::dory_assist`
    and synthetic assist tests.
12. Add wrapper witness-input export.

## Open Questions

- Which protocol-specific witness modules are required in the first PR?
- What is the minimal `PolynomialHandle` surface needed by PCS openings and
  sumcheck kernels?
- How generic over PCS should milestone 1 be?
- Which Dory-assist pieces belong in `jolt-prover::dory_assist` versus
  `jolt_witness::protocols::dory_assist`?

## Review Checklist

- `jolt-verifier` remains the proof/protocol target.
- Verifier-visible outputs are shared where practical.
- Witness generation is modular and namespace-generic.
- ZK and advice are first class.
- Committed sumchecks reuse `jolt-sumcheck` / `jolt-crypto`.
- Heavy compute stays outside stage orchestration.
- Dory assist and wrapper can provide non-VM witnesses.
- No Bolt dependency.

## References

- [`jolt-verifier` model crate spec](./jolt-verifier-model-crate.md)
- [Field inline, Dory assist, and wrapper pipeline spec](./extended-jolt-field-inline-wrapper.md)
- [Recursion references](../recursion_references.md)
- `jolt-core/src/zkvm/prover.rs`
- `jolt-core/src/zkvm/witness.rs`
- `crates/jolt-sumcheck/src/committed.rs`
- `crates/jolt-crypto/src/commitment.rs`
- `crates/jolt-openings/src/schemes.rs`
- `crates/jolt-dory/src/scheme.rs`
