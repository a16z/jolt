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
jolt-backends  implements backend-specific prover compute over prover plans
```

Bolt remains a separate track. This crate should unblock modular proving,
ZK/BlindFold, advice, recursion, Dory assist, and wrapper work before Bolt is
ready.

## High-Level Vision

The modular prover stack should make the current Jolt prover easier to extend
without weakening its performance envelope. The architecture has one controlling
idea:

```text
protocol owns meaning and order
witness owns data access
backends own compute and memory
```

The desired end state is a prover where:

- `jolt-prover` is a handwritten protocol orchestrator, not a compute engine;
- `jolt-prover` can drive transparent, BlindFold, advice, field-inline,
  Dory-assist, and wrapper handoff flows through one typed proof model;
- `jolt-witness` is generic SNARK witness infrastructure that can serve Jolt
  VM, wrapper, Dory assist, and future polynomial-oracle protocols;
- `jolt-backends` can evolve from a canonical CPU backend to CUDA, Metal, or
  hybrid backends without changing transcript order or verifier-visible proof
  shape;
- `jolt-program` and execution backends are the only path from guest execution
  into witness data;
- `jolt-claims` and `jolt-verifier` remain the source of protocol semantics and
  verifier acceptance;
- field inline is a configured Jolt VM extension flowing through
  `jolt-riscv -> jolt-program -> tracer -> jolt-witness -> jolt-prover`, not a
  side channel into the prover.

This is a split, not a rewrite for cleanliness. The canonical CPU backend is
allowed to stay aggressively Jolt-specific internally. The important boundary
is that protocol decisions are outside backend compute, while the backend keeps
the current `jolt-core` prover's time, memory, streaming, and PCS
optimizations.

## Core Decisions

- `jolt-prover -> jolt-verifier` is an intentional one-way dependency.
- `JoltProof` and verifier-visible `StageNOutput` types stay in
  `jolt-verifier`.
- `jolt-prover` builds verifier-owned proof and stage-output structs directly.
- Specialized proving entry points return typed auxiliary outputs for wrapper,
  recursion, Dory assist, and diagnostics.
- ZK/BlindFold and advice are first-class from the first design.
- Committed sumchecks reuse `jolt-sumcheck` and `jolt-crypto` types.
- `jolt-witness` is reusable witness-oracle infrastructure for
  multilinear-polynomial proving systems; the Jolt VM witness provider is the
  first concrete provider, not the whole purpose of the crate.
- `jolt-prover` should be as generic as practical over `jolt-witness`
  namespaces, polynomial encodings, and materialized/streaming witness views.
- `jolt-prover` defines backend traits plus protocol-resolved plan/result
  types. Concrete compute implementations live in `jolt-backends`.
- `jolt-prover` must not production-depend on concrete backends. SDKs, tests,
  CLIs, or host integration code wire a selected backend into `jolt-prover`.
- PCS implementation crates, such as `jolt-dory` over `dory-pcs`, are valid
  compute dependencies behind `jolt-openings`.
- Heavy compute belongs behind witness/backend APIs, not in stage
  orchestration.
- The canonical CPU backend lives in `jolt_backends::cpu`. It may remain
  Jolt-specific and highly optimized internally, but it must execute
  protocol-resolved plans rather than owning protocol decisions; see
  [`jolt-prover` CPU Backend Port](./jolt-prover-cpu-backend-port.md).
- The canonical CPU backend must preserve the relevant prover-time, peak-memory,
  proof-shape, and streaming characteristics of the current `jolt-core` prover.
  Every ported frontier should account for the optimization IDs in
  [`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md).
- `jolt-verifier` is the primary acceptance oracle as the prover frontier grows.
- `jolt-core` is a compatibility/parity oracle only.

## Non-Negotiable Invariants

These invariants should be treated as review blockers:

- `jolt-prover` owns stage order, batching order, transcript labels, challenge
  derivation, proof visibility, opening order, and verifier-owned proof
  assembly.
- `jolt-backends` computes requested algebra and may cache or fuse work, but it
  does not choose claims, transcript events, verifier equations, or public proof
  shape.
- `jolt-witness` exposes data through oracle/view APIs; it does not own backend
  allocation policy, transcript order, PCS proofs, or protocol formulas.
- `jolt-program` owns program shape, bytecode expansion, preprocessing, and the
  normalized execution contract; `tracer` owns concrete execution and adapts to
  that contract.
- `tracer::Cycle`, CPU internals, lazy trace internals, and advice-tape
  internals must not become prover or witness dependencies.
- `jolt-claims` owns protocol IDs, formulas, opening/public/challenge metadata,
  and namespace semantics; witness and backend code must not rediscover these
  facts from ad hoc IDs.
- `jolt-verifier` remains the verifier-visible proof model and primary
  acceptance oracle.
- No production path may call into `jolt-core` prover internals. `jolt-core`
  may be used for fixture generation, differential checks, and parity
  measurement during migration.
- The CPU backend parity bar is performance-sensitive: current streaming
  commitments, Dory chunking/hints, compact polynomials, one-hot RA paths,
  split-eq/evaluation shortcuts, BlindFold private-material handling, advice
  layout, and memory-release behavior must be preserved or replaced by measured
  reviewed equivalents.
- The first backend boundary should be coarse enough to preserve the CPU fast
  path. Fine-grained `jolt-kernels` extraction is deferred until parity is
  demonstrated.
- Without the Cargo `field-inline` feature, prover/verifier/witness/claims/R1CS
  and tracer code should not expose field-inline logic. With the feature, FR-off
  profiles still have no field-inline metadata, commitments, claims,
  challenges, transcript absorbs, BlindFold rows, or dummy zero relations.
- Field-inline protocol surfaces follow `jolt-claims`: `FieldRdInc` is the
  current committed FR surface, while `FieldRs1Ra`, `FieldRs2Ra`, and
  `FieldRdWa` are virtual openings anchored through bytecode metadata and the
  ordinary committed `BytecodeRa(i)` path.
- Wrapper, recursion, and Dory-assist handoffs must consume typed auxiliary
  outputs, not parse opaque proof bytes or backend-private caches.

## Non-Goals

- No Bolt dependency or generated prover/verifier code.
- No production calls into `jolt-core` prover internals.
- No transparent-only architecture.
- No dense-witness-only model.
- No Jolt-VM-only witness abstraction.
- No first-design fine-grained compute crate. A future `jolt-kernels` crate can
  be extracted after the backend boundary is stable and perf-parity is
  demonstrated.
- No universal abstraction for every SNARK arithmetization; `jolt-witness`
  targets polynomial-oracle and multilinear-polynomial witness data.
- No wrapper R1CS assembly or wrapper SNARK proving inside `jolt-prover`.
- No stable external prover API yet.

## Ownership

| Crate | Owns |
|-------|------|
| `jolt-verifier` | Proof model, verifier stage outputs, selected verifier schedule |
| `jolt-prover` | Prover orchestration, transcript order, proof assembly, backend traits/plans, scoped prover state |
| `jolt-witness` | Generic witness-oracle infrastructure, polynomial encodings, public/opening witness APIs, and protocol-specific providers |
| `jolt-backends` | Canonical CPU backend plus future CUDA/Metal/hybrid backends; backend-specific compute and caches over `jolt-prover` plans |
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
backends answer proof-facing compute requests. None of those lower layers
decide stage order or transcript semantics.

## Public API

Representative shape:

```rust
pub fn prove<F, PCS, VC, T, B, W, R>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    inputs: JoltProverInputs<'_, F>,
    selection: ProofSelection,
    witness_builder: &mut W,
    backend: &mut B,
    transcript: &mut T,
    rng: &mut R,
) -> Result<jolt_verifier::JoltProof<PCS, VC>, ProverError>
where
    F: jolt_field::Field,
    PCS: jolt_openings::CommitmentScheme<Field = F>,
    VC: jolt_crypto::VectorCommitment<Field = F>,
    T: jolt_transcript::Transcript<Challenge = F>,
    W: WitnessBuilder<F>,
    B: ProverBackend<F, PCS, VC, Namespace = W::Namespace>,
    R: rand_core::CryptoRngCore;
```

`ProverBackend` is a `jolt-prover` trait. `jolt-backends` implements it for
`CpuBackend` first, and later for `CudaBackend`, `MetalBackend`, or hybrid
backends. This keeps the protocol dependency direction one-way:

```text
host/SDK/test harness
  -> jolt-prover
  -> jolt-witness

jolt-backends
  -> jolt-prover
  -> jolt-witness
```

The caller selects and passes the backend. `jolt-prover` does not import
`jolt-backends` in production code. A separate `jolt-prover-api` crate should
only be introduced if the real Cargo graph forces a cycle; it is not the first
design target.

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

pub fn prove_for_wrapper<F, PCS, VC, T, B, W, R>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    inputs: JoltProverInputs<'_, F>,
    selection: ProofSelection,
    witness_builder: &mut W,
    backend: &mut B,
    transcript: &mut T,
    rng: &mut R,
) -> Result<WrapperProveOutput<F, PCS, VC>, ProverError>
where
    F: jolt_field::Field,
    PCS: jolt_openings::CommitmentScheme<Field = F>,
    VC: jolt_crypto::VectorCommitment<Field = F>,
    T: jolt_transcript::Transcript<Challenge = F>,
    W: WitnessBuilder<F>,
    B: ProverBackend<F, PCS, VC, Namespace = W::Namespace>,
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
pub fn prove<F, PCS, VC, T, B, W, R>(
    inputs: StageNInputs<'_, F>,
    deps: StageNDeps<'_, F>,
    selection: ProofVisibility,
    witness: &W,
    backend: &mut B,
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
  witness oracles/views, backend state, coefficients, blindings, caches,
  handoff data
```

Verifier-owned types are the stable protocol target. Prover modules schedule
generic computation around those types. Prover-only state stays scoped to stage
outputs and specialized proving entry points.

## Witness Model

The dedicated witness design lives in
[`jolt-witness` Crate](./jolt-witness-crate.md). `jolt-prover` treats it as a
generic witness-oracle dependency:

```text
jolt-program::execution artifacts
  -> jolt_witness::protocols::jolt_vm::JoltVmWitness
  -> committed polynomial oracles/views
  -> virtual polynomial evaluators
  -> public/opening witness APIs
  -> jolt-prover plans and jolt-backends compute
```

The important prover-side constraints are:

- the Jolt VM provider is built from `jolt-program::execution`, not tracer
  internals;
- core witness APIs are namespace-generic and reusable by Dory assist, wrapper,
  BlindFold, recursion, and future protocols;
- `jolt-claims` supplies the logical/protocol semantics for namespaces,
  formulas, protocol-visible dimensions, openings, publics, and challenges;
- `jolt-witness` supplies data-access semantics for concrete executions:
  oracle descriptors, available views, source lifecycle, streaming contracts,
  and direct witness evaluation;
- witness oracles/views preserve materialized, compact, sparse/event-log, one-hot/RA,
  lazy, and streaming encodings;
- `jolt-prover` plans should reference witness data through standard
  `OracleRef`, `ViewRequirement`, and `RetentionHint`-style contracts from
  `jolt-witness`, so backends can choose allocation and access strategies
  without guessing protocol intent;
- witnesses do not own transcript order, challenge sampling, claim formulas,
  opening order, commitment construction, PCS proofs, or verifier-visible proof
  objects.

## Program And Trace Inputs

`jolt-prover` should consume program and execution artifacts through the modular
`jolt-program::execution` boundary, not through tracer internals or a separate
trace crate:

```text
guest Rust / SDK
  -> jolt-program image, expansion, profile checks, preprocessing
  -> execution backend
  -> jolt_program::execution::TraceOutput<T: TraceSource>
  -> jolt-witness
  -> jolt-prover stages
```

`jolt-program` owns source decoding, expansion into final Jolt rows, bytecode
preprocessing, RAM/program preprocessing, and the backend-neutral execution
contract. The standardized execution API is:

```text
jolt_program::execution::JoltProgram
jolt_program::execution::TraceInputs
jolt_program::execution::ExecutionBackend
jolt_program::execution::TraceSource
jolt_program::execution::TraceRow
jolt_program::execution::TraceOutput
```

There is no `jolt-trace` crate in this modular flow. Historical `jolt-trace`
decode/trace ownership has been folded into `jolt-program` and `jolt-riscv`;
host-facing default tracer conveniences belong in SDK/host layers. `jolt-program`
should validate that the selected profile supports field-inline source rows
before tracing begins.

The execution backend, typically `tracer`, owns concrete CPU/memory/device
execution and advice I/O. It adapts its local cycle representation into the
normalized `jolt_program::execution::TraceRow` contract. The dependency edge is
`tracer -> jolt-program`, not `jolt-program -> tracer`. `jolt-prover` and
`jolt-witness` should not consume `tracer::Cycle`, `Cpu`, lazy trace internals,
or advice-tape internals directly.

Field-inline program/tracer changes are specified separately in
[field-inline-program-tracer.md](./field-inline-program-tracer.md). That spec
owns the canonical FR memory, bytecode metadata, bridge-row trace payload, and
profile plumbing needed before `jolt-witness` or `jolt-prover` consume
field-inline data.

The intended modular dependency shape for witness/prover bring-up is:

```text
jolt-witness
  -> jolt-program      // execution trace and preprocessing types
  -> jolt-claims       // protocol IDs and formulas
  -> jolt-riscv        // normalized instruction rows and flags
  -> jolt-field/poly   // field and polynomial views

jolt-prover
  -> jolt-witness
  -> jolt-program
  -> jolt-verifier
  -> jolt-claims / jolt-sumcheck / jolt-openings / jolt-crypto

jolt-backends
  -> jolt-prover        // backend traits and protocol-resolved plans
  -> jolt-witness       // witness oracles, views, stream producers, metadata
  -> jolt-openings / jolt-dory / jolt-crypto / jolt-sumcheck
```

`tracer` may be used by `jolt-witness` and `jolt-prover` dev tests as a fixture
backend, but not as a production dependency. `jolt-prover` may use
`jolt-backends` in dev tests or examples; it should not rely on it as a
production dependency.

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

## Backends

Backends own optimized compute. They may execute work at varying granularity:

```text
coarse: prove a whole stage
medium: prove one batched sumcheck
fine: compute one relation message, bind, or evaluation
```

The first design should prefer coarse backend hooks where that helps preserve
the current `jolt-core` CPU fast path. `jolt_backends::cpu` may own
Jolt-specific relation code, fused passes, cached state, unsafe allocation
patterns, Dory contexts, sparse/dense evaluation, prefix/suffix decomposition,
RA processing, streaming commitment inputs, and joint polynomial construction.

The canonical CPU backend has a strict parity requirement with `jolt-core`.
Moving code behind the backend boundary must not accidentally:

- materialize dense `Vec<F>` where `jolt-core` keeps compact, one-hot, sparse,
  derived, or streaming data;
- break CycleMajor streaming commitments or Dory two-tier aggregation;
- drop PCS opening hints needed by Stage 8;
- duplicate equality-table, prefix/suffix, or RA state that is currently shared;
- move transcript, claim, or opening-order decisions into compute kernels;
- regress BlindFold committed-round/private-material construction;
- erase advice-specific Dory contexts, dimensions, or retention policy;
- make field-inline additions force work in FR-off profiles.

The porting rule is: if a current `jolt-core` optimization affects prover time,
peak memory, proof size, or witness materialization, the modular backend must
preserve it or record a measured, reviewed replacement in
[`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md).

Future CUDA, Metal, and hybrid backends should implement the same
`jolt-prover` plan/result traits. They can choose different internal
granularity: a GPU backend might accelerate only commitments and a few relation
messages at first, while a hybrid backend can delegate unsupported plans back
to CPU. The protocol layer should not need to know which internal kernels were
used.

`jolt-kernels` is a possible later extraction of reusable fine-grained compute
building blocks. It is not the first architectural boundary. The first
boundary is:

```text
jolt-prover protocol plans -> jolt-backends compute -> slot-keyed results
```

Backends must not own claims, stage order, transcript labels, verifier
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

The Jolt VM witness provider should expose baseline RV64 witness data and, when
enabled, field-inline witness data through separate protocol IDs:

```text
RV64 Jolt namespace:
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

Inside `jolt_witness::protocols::jolt_vm`, this should be structured as
`rv64/` plus optional `field_inline/`, not as a separate top-level
field-inline provider. Field inline is an extension of the same Jolt VM
execution and proof.

The field-inline witness provider owns:

- `field_rows`: trace rows where native field instructions or bridge
  instructions are active;
- field-inline bytecode metadata required by `jolt-claims` and bound in the
  preamble when field inline is selected;
- FR register read/write events for `FieldRs1`, `FieldRs2`, and `FieldRd`;
- `FieldRdInc` committed polynomial material;
- virtual FR values: `FieldRs1Value`, `FieldRs2Value`, `FieldRdValue`,
  `FieldRegistersVal`, `FieldRs1Ra`, `FieldRs2Ra`, and `FieldRdWa`;
- product witnesses for `FieldProduct = FieldRs1Value * FieldRs2Value` and
  `FieldInvProduct = FieldRs1Value * FieldRdValue`;
- bridge encodings between ordinary x-register values and native field
  elements.

`FieldRs1Ra`, `FieldRs2Ra`, and `FieldRdWa` are virtual openings. They are
anchored through the field-inline extension of `BytecodeReadRaf`, then through
the ordinary committed `BytecodeRa(i)` path. `FieldRdInc` is the current
field-register committed polynomial surface. The prover must not introduce a
separate committed `FieldRegistersRa(i)` surface unless `jolt-claims` changes
the selected protocol semantics.

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
  validate and bind field-inline bytecode metadata only when field inline is
  enabled
  commit/absorb field-inline commitments, currently FieldRdInc, only when field
  inline is enabled

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
  output FieldRegistersVal, FieldRs1Ra, FieldRs2Ra, FieldRdWa, and FieldRdInc
  read/write claims

stage 5:
  prove FieldRegistersValEvaluation in the val-evaluation batch
  output FieldRdWa and FieldRdInc val-evaluation claims

stage 6:
  prove field-inline bytecode read-RAF anchoring for FieldRs1Ra, FieldRs2Ra,
  and FieldRdWa
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

In a build without the Cargo `field-inline` feature, `jolt-prover` should not
compile field-inline stage code or name field-inline claims at all. In a build
with the feature, an FR-off profile still skips field-inline commitments,
claims, challenges, bytecode metadata, sumchecks, transcript absorbs, and
BlindFold rows entirely. There are no dummy zero FR relations in the FR-off
proof shape.

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

crates/jolt-backends/src/
  lib.rs
  cpu/
    mod.rs
    state.rs
    commitments.rs
    advice.rs
    stage1.rs
    stage2.rs
    stage3.rs
    stage4.rs
    stage5.rs
    stage6.rs
    stage7.rs
    stage8.rs
    blindfold.rs
    core_fast_path/
  cuda/        // future/optional
    mod.rs
  metal/       // future/optional
    mod.rs
  hybrid/      // future/optional
    mod.rs

crates/jolt-witness/src/
  lib.rs
  dimensions.rs
  encoding.rs
  namespace.rs
  provider.rs
  polynomial.rs
  opening.rs
  public.rs
  streaming.rs
  r1cs.rs
  error.rs
  protocols/
    jolt_vm/
      mod.rs
      advice.rs
      rv64/
        mod.rs
        trace.rs
        committed.rs
        virtuals.rs
      field_inline/
        mod.rs
        metadata.rs
        registers.rs
        products.rs
        bridge.rs
        bytecode_anchor.rs
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
- field-inline witness provider checks for field_rows, bytecode metadata,
  bridge rows, FieldRdInc, FR products, virtual FR evals, and virtual
  FieldRs1Ra/FieldRs2Ra/FieldRdWa anchoring through BytecodeRa(i);
- FR-off checks that field-inline commitments, claims, challenges, metadata,
  sumchecks, transcript absorbs, and BlindFold rows are absent;
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

## Agent-Directed Implementation Plan

The implementation can be managed by one lead Codex agent responsible for this
top-level spec and integration direction, with focused Codex agents working on
bounded slices. The lead agent owns:

- the architecture invariants in this spec;
- PR/slice ordering;
- dependency graph and feature-gate consistency;
- final review of every worker patch;
- integration tests and parity checks;
- deciding when a lower-level spec needs revision.

Worker agents should receive one self-contained slice with explicit write
ownership. They should not change unrelated specs or implementation areas
without handing the issue back to the lead. When high-risk design or porting
work is delegated, prefer GPT-5.5 with xhigh reasoning and priority/fast
service tier when available.

The lead agent may choose the operational workflow that best reduces merge
risk: Codex sub-agent forked workspaces, git worktrees, `tmux`-managed Codex CLI
sessions, or a shared checkout for strictly sequential work. For parallel code
edits, the default should be one git worktree per worker slice unless the
environment already provides isolated sub-agent workspaces. The lead is
responsible for creating, naming, reviewing, merging, and cleaning up those
workspaces.

Recommended agent roles:

```text
lead:
  owns jolt-prover-model-crate.md, integration review, and final merge shape

explorer:
  answers scoped codebase questions and inventories existing behavior

worker:
  edits one crate/spec slice with a disjoint write set

reviewer:
  audits a completed slice against invariants, tests, and performance risks
```

Useful parallel slices:

| Slice | Primary Spec | Typical Write Ownership |
|-------|--------------|-------------------------|
| `jolt-riscv` field-inline vocabulary and gates | [field-inline-program-tracer.md](./field-inline-program-tracer.md) | `crates/jolt-riscv/`, profile tests |
| `jolt-program` field metadata and trace contract | [field-inline-program-tracer.md](./field-inline-program-tracer.md) | `crates/jolt-program/`, program/preprocess tests |
| `tracer` FR execution adapter | [field-inline-program-tracer.md](./field-inline-program-tracer.md) | `tracer/`, execution fixtures |
| witness core APIs | [jolt-witness-crate.md](./jolt-witness-crate.md) | `crates/jolt-witness/` core modules |
| Jolt VM witness provider | [jolt-witness-crate.md](./jolt-witness-crate.md) | `crates/jolt-witness/src/protocols/jolt_vm/` |
| prover scaffold and plans | this spec | `crates/jolt-prover/` protocol scaffolding |
| CPU backend scaffold | [jolt-prover CPU backend port](./jolt-prover-cpu-backend-port.md) | `crates/jolt-backends/src/cpu/` |
| optimization audit / parity tests | [optimization inventory](./jolt-core-prover-optimization-inventory.md) | tests, benches, inventory updates |
| field-inline prover stages | [field-inline protocol spec](./field-inline-protocol.md) | `jolt-prover` field-inline stage slices |
| Dory assist / wrapper handoff | Dory-assist and wrapper specs | typed auxiliary outputs and fixtures |

Each worker prompt should include:

```text
objective:
  concrete crate/spec slice and expected output

write ownership:
  exact files or directories the worker may edit

required specs:
  one primary spec plus any companion specs

invariants:
  protocol/compute boundary, CPU parity, feature gates, no jolt-core production
  dependency, no tracer internals in witness/prover

validation:
  exact cargo fmt/clippy/nextest commands or docs-only checks

handoff:
  list changed files, tests run, unresolved questions, and any spec mismatch
```

### Workspace And Merge Discipline

Recommended worktree naming:

```text
../jolt-agent-riscv-field-inline
../jolt-agent-program-trace
../jolt-agent-tracer-field-inline
../jolt-agent-witness-core
../jolt-agent-witness-jolt-vm
../jolt-agent-prover-plans
../jolt-agent-cpu-backend
../jolt-agent-parity
```

Each worktree should use a slice branch, for example:

```text
agent/riscv-field-inline
agent/program-trace
agent/witness-core
agent/prover-plans
agent/cpu-backend
```

Workers must treat their worktree as disposable and their assigned write set as
authoritative. They should not rebase, merge, reset, or clean the lead
worktree. They should report:

```text
git status --short
git diff --stat
changed files
tests run
known conflicts or spec mismatches
```

The lead should integrate worker output by inspection, not blind merge. A
typical integration flow is:

1. Review the worker diff in its worktree.
2. Run the slice validation there.
3. Merge or cherry-pick into the lead worktree only after review.
4. Resolve conflicts in favor of the architecture invariants, not in favor of
   the newest patch.
5. Run the relevant combined validation in the lead worktree.
6. Remove or park the worker worktree only after the integrated state is clean.

If two slices need the same files, the lead should serialize them or split the
shared file behind a small lead-owned interface patch before delegation. Shared
files such as root `Cargo.toml`, workspace feature maps, central protocol ID
definitions, and top-level specs should normally be lead-owned.

The lead should review each worker result in this order:

1. Check write-set boundaries and reject unrelated refactors.
2. Check feature gates: no-feature builds should not expose field-inline logic;
   feature-enabled FR-off profiles should remain structurally ordinary Jolt.
3. Check ownership: protocol facts in `jolt-claims`/`jolt-verifier`, data
   access in `jolt-witness`, compute in `jolt-backends`.
4. Check CPU parity: map touched hot paths to optimization-inventory IDs or
   record why the slice is not performance-sensitive.
5. Run the slice validation commands and then the relevant integration frontier.
6. Update the top-level spec or companion spec if implementation exposes a
   sharper boundary.

When using Codex sub-agents, prefer forked workspaces for code-edit workers and
assign disjoint write sets. If using external Codex CLI agents under `tmux`, use
one window per slice and preferably one git worktree per worker. If workers
share a checkout, they must avoid overlapping files, report `git status`
before handoff, and never revert other workers' changes.

Suggested `tmux` layout:

```text
tmux session: jolt-prover-split

window 0 lead:
  top-level spec, integration review, git status, final validation

window 1 riscv-program-tracer:
  field-inline vocabulary, profile gates, trace contract

window 2 witness:
  witness core APIs and Jolt VM provider

window 3 prover-backend:
  prover plans, backend traits, CPU scaffold

window 4 parity:
  optimization inventory, differential fixtures, benchmarks
```

The lead should keep the mainline moving by doing immediate blocking work
locally and delegating only bounded sidecar tasks. Worker output is not accepted
because it compiles once; it is accepted when it preserves the architectural
invariants and passes the relevant verifier/parity frontier.

## Milestones

1. Design and scaffold `jolt-witness` core: namespaces, dimensions, public
   values, polynomial encodings, materialized/streaming views, virtual evals,
   and opening witnesses.
2. Add a trace-backed Jolt VM witness provider under
   `jolt_witness::protocols::jolt_vm`, including committed/virtual/advice
   witness checks against fixtures.
3. Scaffold `jolt-prover`: selection, explicit RNG, `JoltProof` assembly, and
   specialized proving outputs over the `jolt-witness` core traits, plus
   backend traits and protocol-resolved plan/result types.
4. Scaffold `jolt-backends` with `jolt_backends::cpu` implementing the initial
   `jolt-prover` backend traits; add slow/reference paths only where they help
   prove equivalence without disrupting the CPU fast path.
5. Implement commitments in the CPU backend, including trusted/untrusted
   advice state.
6. Implement stages 1-2 in transparent and BlindFold mode.
7. Port stages 3-7 with verifier-output sharing.
8. Implement advice reductions and optional address phase.
9. Implement stage 8, ZK opening path, and Dory-assist input data.
10. Prove the BlindFold instance from committed proofs.
11. Add full transparent/ZK prover-verifier E2E with advice.
12. Add field-inline witness support and prover stage slices against the
    already-selected verifier flow.
13. Add full transparent/ZK prover-verifier E2E with field inline enabled.
14. Add Dory-assist witness module under `jolt_witness::protocols::dory_assist`
    and synthetic assist tests.
15. Add wrapper witness-input export.

## Open Questions

- Which protocol-specific witness modules are required in the first PR?
- What is the minimal `PolynomialView` surface needed by PCS openings and
  backend sumcheck execution?
- Which streaming capabilities must be in the initial `PolynomialView`
  surface, and which can wait for optimized backend implementations?
- Which backend trait methods should be coarse first for CPU perf parity, and
  which are stable enough to expose at medium granularity?
- Should `jolt-backends` remain one crate with optional backend features, or
  split into per-backend crates after the CPU path stabilizes?
- How generic over PCS should the first `jolt-prover` scaffold be?
- Should field-inline witness support be implemented as one
  `jolt_vm::field_inline` provider extension or split into smaller
  `registers`, `products`, `bridge`, and `constraints` modules from the first
  PR?
- Which Dory-assist pieces belong in `jolt-prover::dory_assist` versus
  `jolt_witness::protocols::dory_assist`?

## Review Checklist

- `jolt-verifier` remains the proof/protocol target.
- Verifier-visible outputs are shared where practical.
- `jolt-witness` is reusable polynomial-oracle witness infrastructure, not a
  Jolt-VM-only abstraction.
- Witness generation is modular, namespace-generic, and explicit about
  materialized, compact, derived, and streaming polynomial encodings.
- ZK and advice are first class.
- Committed sumchecks reuse `jolt-sumcheck` / `jolt-crypto`.
- `jolt-prover` defines backend traits/plans and does not production-depend on
  concrete backends.
- Concrete compute lives in `jolt-backends`, with CPU first and GPU/hybrid
  backends possible without changing protocol orchestration.
- The canonical CPU backend preserves current `jolt-core` optimization and
  memory characteristics or records measured replacements against the
  optimization inventory.
- Heavy compute stays outside stage orchestration.
- Fine-grained `jolt-kernels` extraction is deferred until the backend boundary
  is perf-stable.
- Feature gates compile out field-inline prover/verifier/witness/tracer logic
  when disabled, while feature-enabled FR-off profiles remain structurally
  ordinary Jolt.
- Implementation slices have clear write ownership and are reviewed against
  this spec before integration.
- Dory assist and wrapper can provide non-VM witnesses.
- Field inline is implemented in `jolt-prover`/`jolt-witness`, not by adding a
  new production proving path through `jolt-core`.
- No Bolt dependency.

## References

- [`jolt-verifier` model crate spec](./jolt-verifier-model-crate.md)
- [`jolt-witness` crate spec](./jolt-witness-crate.md)
- [`jolt-prover` CPU backend port spec](./jolt-prover-cpu-backend-port.md)
- [`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md)
- [Field inline protocol spec](./field-inline-protocol.md)
- [Field inline `jolt-program` / `tracer` integration spec](./field-inline-program-tracer.md)
- [Field inline, Dory assist, and wrapper pipeline spec](./extended-jolt-field-inline-wrapper.md)
- [Recursion references](../recursion_references.md)
- `jolt-core/src/zkvm/prover.rs`
- `jolt-core/src/zkvm/witness.rs`
- `crates/jolt-sumcheck/src/committed.rs`
- `crates/jolt-crypto/src/commitment.rs`
- `crates/jolt-openings/src/schemes.rs`
- `crates/jolt-dory/src/scheme.rs`
