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
- Specialized proving entry points return typed auxiliary outputs for wrapper,
  recursion, Dory assist, and diagnostics.
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
| `jolt-prover` | Prover orchestration, transcript order, proof assembly, scoped prover state |
| `jolt-witness` | Generic witness traits plus protocol-specific witness providers |
| `jolt-kernels` | Optimized prover compute and map-reduce kernels |
| `jolt-program` | Program image, bytecode expansion, profile/extension legality, Jolt trace contract |
| `tracer` / execution backend | Concrete guest execution, advice I/O, normalized trace production |
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
  -> jolt-program image / expansion / preprocessing
  -> execution backend trace
  -> witness provider
  -> oracle/advice commitments
  -> stages 1..7
  -> stage 8 opening proof
  -> Dory-assist inputs when selected
  -> BlindFold proof when selected
  -> Dory-assist proof when selected
  -> wrapper witness handoff when requested
```

`jolt-prover` owns the proving order after program execution has produced a
Jolt trace. `jolt-program` owns program shape and the normalized Jolt trace
contract; the execution backend owns concrete execution; witness providers and
kernels answer proof-facing queries. None of those lower layers decide stage
order or transcript semantics.

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

The primary API returns the verifier-owned proof. Flows that need auxiliary
private data use explicit entry points with typed outputs:

```rust
pub struct WrapperProveOutput<F, PCS, VC>
where
    PCS: jolt_openings::CommitmentScheme<Field = F>,
    VC: jolt_crypto::VectorCommitment<Field = F>,
{
    pub proof: jolt_verifier::JoltProof<PCS, VC>,
    pub wrapper_inputs: WrapperWitnessInputs<F>,
}

pub fn prove_for_wrapper<F, PCS, VC, T, K, W, R>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    inputs: JoltProverInputs<'_, F>,
    selection: ProofSelection,
    witness_builder: &mut W,
    kernels: &mut K,
    transcript: &mut T,
    rng: &mut R,
) -> Result<WrapperProveOutput<F, PCS, VC>, ProverError>
where
    F: jolt_field::Field,
    PCS: jolt_openings::CommitmentScheme<Field = F>,
    VC: jolt_crypto::VectorCommitment<Field = F>,
    T: jolt_transcript::Transcript<Challenge = F>,
    K: KernelSet<F>,
    W: WitnessBuilder<F>,
    R: rand_core::CryptoRngCore;
```

Use the same pattern for `prove_for_dory_assist`, `prove_for_recursion`, and
diagnostic entry points. Each entry point owns a named return type with only the
data that flow consumes.

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
    pub protocol: jolt_verifier::JoltProtocolConfig,
    pub dory_assist: DoryAssistSelection,
    pub wrapper_handoff: WrapperHandoffSelection,
}
```

The Jolt proof protocol is selected as a concrete
`jolt_verifier::JoltProtocolConfig`. Visibility is derived from `protocol.zk`;
field-inline behavior is derived from `protocol.field_inline`. The prover writes
the same config into `JoltProof::protocol`, and the verifier rejects proofs whose
embedded config does not match its compile-time-selected config before any
transcript-dependent work runs.

`dory_assist` and `wrapper_handoff` are not changes to the base Jolt proof
shape. They select auxiliary proving outputs around the configured Jolt proof.

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

The production verifier can still choose its accepted protocol at compile time.
The prover API should make the target protocol explicit so tests, diagnostics,
and future multi-target proving can construct the exact proof shape the target
verifier accepts.

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
) -> Result<StageNProverOutput<F, VC, StageNPrivate<F, PCS, VC>>, ProverError>;
```

Stage code should show:

- input claims;
- transcript absorbs/squeezes;
- clear or committed sumcheck calls;
- batching order;
- output claims;
- opening dependencies;
- private ZK material.

## Verifier-Owned Protocol Objects

`jolt-verifier` defines the final protocol objects. `jolt-prover` imports those
types, computes their fields, and assembles them as the public proof surface.
Stage functions return verifier-owned protocol data plus scoped prover state
needed by later stages.

```rust
pub struct StageNProverOutput<F, VC, Private> {
    pub verifier_output: jolt_verifier::stages::stageN::StageNOutput<F>,
    pub proof: jolt_verifier::stages::stageN::StageNProof<F, VC>,
    pub openings: StageOpeningDeps<F>,
    pub zk: Option<StageZkMaterial<F, VC>>,
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
generic computation around those types. Prover-only state stays scoped to stage
outputs and specialized proving entry points.

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

Jolt VM + field inline:
  base trace rows, field_rows, FR register accesses, FR products, bridge rows,
  field-register committed oracles, field-inline virtual evals

Dory assist:
  operation traces, packing witnesses, Miller-loop witness, assist public inputs

Wrapper:
  selected verifier values, transcript state, R1CS assignment

BlindFold:
  committed round witnesses, output claim rows, auxiliary rows
```

Witness code does not own protocol order, transcript labels, stage selection,
or claim formulas.

## Program And Trace Inputs

`jolt-prover` should consume program and execution artifacts through the modular
program/Jolt-trace boundary, not through tracer internals:

```text
guest Rust / SDK
  -> jolt-program image, expansion, profile checks, preprocessing
  -> execution backend
  -> Jolt trace: TraceOutput<TraceSource>
  -> jolt-witness
  -> jolt-prover stages
```

`jolt-program` owns source decoding, expansion into final Jolt rows, bytecode
preprocessing, RAM/program preprocessing, and the backend-neutral execution
contract. It should validate that the selected profile supports field-inline
source rows before tracing begins.

The execution backend, typically `tracer`, owns concrete CPU/memory/device
execution and advice I/O. It adapts its local cycle representation into the
normalized Jolt trace contract. `jolt-prover` and `jolt-witness` should not
consume `tracer::Cycle`, `Cpu`, or lazy trace internals directly.

The Jolt trace row is the proof-facing trace row. It must carry enough data for
both ordinary Jolt and field inline:

```text
ordinary row data:
  final Jolt instruction row
  x-register reads/writes
  RAM access
  bytecode/source metadata needed by witness construction

field-inline row data, when enabled:
  field op kind / selector flags
  FR register operands and destination
  FR register read/write values
  bridge payloads for x-register <-> FR movement
```

Pure field operations should not create incidental ordinary x-register effects.
If an implementation keeps inert ordinary accesses for engineering reasons, it
must expose them explicitly as inert so the ordinary register witness and the
field-register witness stay consistent.

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

Advice state retains padded words, commitment, opening hint, dimensions,
address-phase choice, advice openings, stage-8 selector factors, and BlindFold
claim constraints.

## Field Inline Proving

Field inline should land in `jolt-prover`, not by threading a new prover path
through `jolt-core`. The prover architecture should treat field inline as a
configured Jolt VM extension whose protocol semantics live in
`jolt-claims::protocols::field_inline` and whose verifier composition is owned
by `jolt-verifier`.

Guest exposure should be SDK-level. On guest builds, field-element helper
methods or intrinsics emit field-inline source rows such as:

```text
FIELD_LOAD_FROM_X
FIELD_LOAD_IMM
FIELD_ADD
FIELD_SUB
FIELD_MUL
FIELD_INV
FIELD_ASSERT_EQ
FIELD_STORE_TO_X
```

`jolt-program` validates and expands these rows under the selected ISA/profile.
The execution backend executes their native field semantics and emits normalized
field-row data. `jolt-witness` is the first layer that turns that data into
committed and virtual polynomial witnesses.

The Jolt VM witness provider should expose ordinary Jolt witness data and, when
enabled, field-inline witness data through separate protocol IDs:

```text
ordinary VM namespace:
  JoltCommittedPolynomial
  JoltVirtualPolynomial
  JoltOpeningId

field-inline namespace:
  FieldInlineCommittedPolynomial
  FieldInlineVirtualPolynomial
  FieldInlineOpeningId
```

The two namespaces meet only at selected stage composition points. Bridge rows
reuse ordinary RV64 witness columns for `Rs1Value`, `RdWriteValue`, and `Imm`;
true field-inline columns are exposed under field-inline IDs.

The field-inline witness provider owns:

- `field_rows`: trace rows where native field instructions or bridge
  instructions are active;
- FR register read/write events for `FieldRs1`, `FieldRs2`, and `FieldRd`;
- `FieldRdInc` and `FieldRegistersRa(i)` committed polynomial material;
- virtual FR values: `FieldRs1Value`, `FieldRs2Value`, `FieldRdValue`,
  `FieldRegistersVal`, and write-address helpers;
- product witnesses for `FieldProduct = FieldRs1Value * FieldRs2Value` and
  `FieldInvProduct = FieldRs1Value * FieldRdValue`;
- bridge encodings between ordinary x-register values and native field
  elements.

Representative trace-to-witness mapping:

```text
FIELD_MUL fr3, fr1, fr2:
  field trace:
    read fr1, read fr2, write fr3
  witness:
    FieldRs1Value = fr1
    FieldRs2Value = fr2
    FieldRdValue = fr3
    FieldProduct = fr1 * fr2
    IsFieldMul = 1

FIELD_LOAD_FROM_X fr4, x10:
  ordinary trace:
    read x10
  field trace:
    write fr4
  witness:
    Rs1Value comes from the ordinary register witness
    FieldRdValue comes from the FR register witness
    bridge row enforces FieldRdValue = decode_x_register(Rs1Value, F)

FIELD_STORE_TO_X x11, fr4:
  field trace:
    read fr4
  ordinary trace:
    write x11
  witness:
    FieldRs1Value comes from the FR register witness
    RdWriteValue comes from the ordinary register witness
    bridge row enforces RdWriteValue = encode_field_register(FieldRs1Value, F)
```

Field inline v1 is native-field only: the field used by the Jolt proof and the
field used by field-inline arithmetic are the same field. Prover code should not
introduce quotient witnesses or non-native modular reduction for v1. The only
width-dependent work is trace/bridge encoding for the selected field
instantiation, such as two-limb or four-limb encodings.

Per-stage obligations:

```text
preamble:
  assemble proof.protocol from the selected verifier config
  commit/absorb FieldInlineCommitments only when field inline is enabled

stage 1:
  build the selected R1CS row layout: RV64 rows plus field_constraints rows
  compute selected Spartan outer openings
  reuse ordinary openings for bridge columns and append only FR-local openings
  retain committed output-claim rows/blindings in BlindFold mode

stage 2:
  add explicit field product lanes at the existing product point r_prod
  add FieldRegistersClaimReduction at the same r_prod
  keep product output ordering synchronized with jolt-verifier

stage 4:
  prove FieldRegistersReadWriteChecking in the read/write batch
  output FieldRegistersVal, FieldRdWa, and FieldRdInc read/write claims

stage 5:
  prove FieldRegistersValEvaluation in the val-evaluation batch
  output FieldRdWa and FieldRdInc val-evaluation claims

stage 6:
  reduce the stage-4/stage-5 FieldRdInc claims to one
  FieldRdInc@FieldRegistersIncClaimReduction claim

stage 8:
  include the reduced FieldRdInc opening in the final joint PCS opening plan
  after ordinary RamInc/RdInc and before RA/advice openings
```

The same obligations apply in Transparent and BlindFold mode. Transparent mode
puts clear field-inline opening claims in `JoltProofClaims::Clear`. BlindFold
mode commits field-inline sumcheck rounds and output-claim rows, keeps the
private coefficients/blindings for the BlindFold witness, and ensures the final
hidden PCS evaluation uses the same mixed Stage 8 opening ID order that
`jolt-verifier` lowers.

The prover should not recover field-inline order from ad hoc accumulators.
Stage code should use the canonical helpers in `jolt-claims` and the selected
stage order in `jolt-verifier`, then build a typed opening plan with explicit
Jolt and field-inline IDs.

## Stage 8

Stage 8 is typed opening assembly:

```text
stage outputs
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

When field inline is selected, the opening plan includes
`FieldRdInc@FieldRegistersIncClaimReduction` in the configured Stage 8 order.
This is ordinary final-opening work: the PCS backend sees another committed
polynomial/evaluation pair, while the typed opening plan preserves the protocol
identity needed by BlindFold, Dory assist, wrapper handoff, and diagnostics.

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

Field-inline relations follow the same rule: if a field-inline relation changes
its input/output claim formula or opening order, the prover's committed rows,
BlindFold witness construction, and final-opening plan must change with it.

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
prove_for_wrapper(...)
  -> JoltProof
  -> WrapperWitnessInputs
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
  state.rs
  prover.rs
  selection.rs
  commitment/
  stages/
    stage1/
    stage2/
    stage3/
    stage4/
    stage5/
    stage6/
    stage7/
    stage8/
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
      field_inline.rs
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
- committed sumcheck and BlindFold output shape;
- witness provider reference checks for committed and virtual polynomial evals;
- stage-8 opening plan, ZK opening data, and Dory-assist inputs;
- field-inline witness provider checks for field_rows, bridge rows,
  FieldRdInc, FieldRegistersRa(i), FR products, and virtual FR evals;
- field-inline transparent and BlindFold prover frontiers accepted by the
  matching `jolt-verifier` frontier;
- tampering of transcript, dependencies, openings, committed claims, advice
  commitments, field-inline commitments/claims, BlindFold inputs, and
  Dory-assist input data.

Bring-up should be top-down against `jolt-verifier`: implement the next prover
component, assemble the partial verifier-visible proof/checkpoint object, and
require the corresponding verifier stage/frontier to accept it. `jolt-core`
remains useful for fixture generation and parity checks, but it is not the
primary oracle for the modular prover.

## Milestones

1. Scaffold `jolt-prover`: selection, explicit RNG, `JoltProof` assembly, and
   specialized proving outputs.
2. Add `jolt-witness` traits and a trace-backed VM witness provider under
   `jolt_witness::protocols::jolt_vm`.
3. Add kernel boundaries and a slow/reference kernel path.
4. Implement commitments, including trusted/untrusted advice state.
5. Implement stages 1-2 in transparent and BlindFold mode.
6. Port stages 3-7 with verifier-output sharing.
7. Implement advice reductions and optional address phase.
8. Implement stage 8, ZK opening path, and Dory-assist input data.
9. Prove the BlindFold instance from committed proofs.
10. Add full transparent/ZK prover-verifier E2E with advice.
11. Add field-inline witness support and prover stage slices against the
    already-selected verifier flow.
12. Add full transparent/ZK prover-verifier E2E with field inline enabled.
13. Add Dory-assist witness module under `jolt_witness::protocols::dory_assist`
    and synthetic assist tests.
14. Add wrapper witness-input export.

## Open Questions

- Which protocol-specific witness modules are required in the first PR?
- What is the minimal `PolynomialHandle` surface needed by PCS openings and
  sumcheck kernels?
- How generic over PCS should milestone 1 be?
- Should field-inline witness support be implemented as one
  `jolt_vm::field_inline` provider extension or split into smaller
  `registers`, `products`, `bridge`, and `constraints` modules from the first
  PR?
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
- Field inline is implemented in `jolt-prover`/`jolt-witness`, not by adding a
  new production proving path through `jolt-core`.
- No Bolt dependency.

## References

- [`jolt-verifier` model crate spec](./jolt-verifier-model-crate.md)
- [Field inline protocol spec](./field-inline-protocol.md)
- [Field inline, Dory assist, and wrapper pipeline spec](./extended-jolt-field-inline-wrapper.md)
- [Recursion references](../recursion_references.md)
- `jolt-core/src/zkvm/prover.rs`
- `jolt-core/src/zkvm/witness.rs`
- `crates/jolt-sumcheck/src/committed.rs`
- `crates/jolt-crypto/src/commitment.rs`
- `crates/jolt-openings/src/schemes.rs`
- `crates/jolt-dory/src/scheme.rs`
