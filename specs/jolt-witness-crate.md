# Spec: `jolt-witness` Crate

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-25 |
| Status | draft |
| PR | TBD |

## Purpose

`jolt-witness` is reusable witness-oracle infrastructure for SNARK and
polynomial-IOP provers. Its first production job is to generate and expose the
Jolt VM witness from the standardized `jolt-program::execution` interface, but
the crate must not be shaped as a Jolt-only witness generator.

The core abstraction is:

```text
private execution/protocol data
  -> typed witness provider
  -> committed polynomial oracles
  -> storage/access views
  -> virtual polynomial evaluators
  -> public value providers
  -> opening/evaluation witnesses
  -> materialized, compact, lazy, and streaming encodings
```

`jolt-prover` consumes this witness layer to prove the base Jolt protocol.
Other protocol provers should be able to reuse the same machinery for Dory
assist, wrapper proving, BlindFold witness construction, recursion, and future
SNARK protocols that consume polynomial-oracle witness data.

## Core Requirements

- Implement the Jolt VM witness provider from `jolt-program::execution`
  artifacts, not from `tracer` internals.
- Keep the core APIs namespace-generic so non-VM protocols can define their own
  committed, virtual, opening, public, and challenge IDs.
- Treat `jolt-claims` as the source of protocol semantics for protocol-specific
  namespaces: logical IDs, formulas, protocol-visible dimensions, opening
  metadata, and selected-verifier claim metadata.
- Preserve efficient encodings instead of forcing every witness into `Vec<F>`.
- Support materialized, compact, sparse/event-log, one-hot/RA, lazy, and
  streaming views.
- Treat large materialization as backend-owned. `jolt-witness` provides logical
  oracles, source data, metadata, recipes, and stream producers; the selected
  backend owns dense columns, device buffers, caches, scratch space, and
  materialization lifetimes.
- Separate witness data from protocol ownership: witness providers do not own
  transcript order, challenge sampling, claim formulas, opening order,
  commitment construction, verifier-visible proof objects, or SNARK backend
  proving.
- Expose enough metadata for `jolt-prover`, `jolt-backends`, PCS code,
  BlindFold, Dory assist, and wrapper code to request data without depending on
  protocol-provider concrete structs.
- Make public values explicit, including values derived from execution,
  preprocessing, wrapper assignment, Dory-assist inputs, or BlindFold metadata.
- Make transcript-derived challenges explicit inputs to virtual evaluation.

## Non-Goals

- No dependency on `jolt-prover`.
- No dependency on `jolt-backends`.
- No production dependency on `tracer`.
- No commitment construction or PCS opening proof generation.
- No sumcheck scheduling, batching, or transcript logic.
- No claim formula ownership.
- No assumption that all witness data is dense, random-access, or field-native.
- No universal abstraction for every arithmetization. The target is
  polynomial-oracle and multilinear-polynomial witness data.
- No stable external API before the Jolt VM provider and at least one non-VM
  provider prototype exercise the core traits.

## Ownership

| Layer | Owns |
|-------|------|
| `jolt-witness` core | Generic witness traits, namespaces, dimensions, encodings, polynomial oracles/views, public values, virtual eval contexts, opening/evaluation witnesses |
| `jolt_witness::protocols::jolt_vm` | Jolt VM trace-backed witness provider built from `jolt-program::execution` artifacts |
| `jolt_witness::protocols::dory_assist` | Dory verifier operation-trace, packing, Miller-loop, and assist-public-input witness providers |
| `jolt_witness::protocols::wrapper` | Configured-verifier assignment witness providers for wrapper proving |
| `jolt_witness::protocols::blindfold` | Committed round, output-claim row, and auxiliary BlindFold witness providers |
| `jolt-claims` | Protocol semantics: namespace IDs, formulas, protocol-visible dimensions, opening/public/challenge metadata |
| `jolt-prover` | Protocol order, transcript, challenge sampling, selected plans, opening plans, proof assembly |
| `jolt-backends` | Backend-specific allocation, materialization, caching, and compute over witness views and `jolt-prover` plans |
| `jolt-program` | Program image, preprocessing, profile legality, execution contract, normalized trace rows |

## Dependency Shape

The core witness traits should stay independent of the Jolt VM:

```text
jolt-witness core
  -> jolt-field / field traits
  -> polynomial view traits or lightweight local view types
  -> lightweight ID traits where needed, not concrete protocol IDs

jolt_witness::protocols::jolt_vm
  -> jolt-witness core
  -> jolt-program       // execution API and preprocessing artifacts
  -> jolt-riscv         // normalized instruction/flag types where needed
  -> jolt-claims        // Jolt VM committed/virtual/opening/public IDs

jolt-prover
  -> jolt-witness
  -> jolt-program
  -> jolt-verifier / jolt-claims / jolt-sumcheck / jolt-openings

jolt-backends
  -> jolt-prover        // backend traits and protocol-resolved plans
  -> jolt-witness       // oracles, views, stream producers, metadata
```

If the optional provider modules create an awkward Cargo graph, split them
later into provider crates, for example `jolt-witness-jolt`,
`jolt-witness-wrapper`, or `jolt-witness-dory-assist`. That split should not be
the first design target; start with one crate and keep the core module clean.

## Relationship To `jolt-program`

The Jolt VM provider consumes the standardized execution boundary:

```text
guest Rust / SDK
  -> jolt-program image, expansion, profile checks, preprocessing
  -> execution backend
  -> jolt_program::execution::TraceOutput<T: TraceSource>
  -> jolt_witness::protocols::jolt_vm::JoltVmWitness
  -> jolt-prover / jolt-backends
```

The relevant interface is:

```text
jolt_program::execution::JoltProgram
jolt_program::execution::TraceInputs
jolt_program::execution::ExecutionBackend
jolt_program::execution::TraceSource
jolt_program::execution::TraceRow
jolt_program::execution::TraceOutput
```

`tracer` remains a concrete execution backend and a useful test fixture, but
`jolt-witness` should not consume `tracer::Cycle`, `Cpu`, lazy trace internals,
or advice-tape internals directly. The dependency edge is
`tracer -> jolt-program`, not `jolt-witness -> tracer`.

Field-inline support requires `jolt-program` and `tracer` to expose the
canonical field-register trace payloads described in
[field-inline-program-tracer.md](./field-inline-program-tracer.md). `jolt-witness`
then consumes those normalized artifacts; it does not infer field-register
semantics from tracer-private state.

## Relationship To `jolt-claims`

`jolt-claims` is the source of protocol semantics. `jolt-witness` is the source
of data-access semantics.

For any protocol namespace:

```text
jolt-claims owns:
  logical committed/virtual/opening/public/challenge IDs
  claim formulas and relation metadata
  protocol-visible dimension formulas
  opening formulas and opening order where protocol-visible
  public/challenge metadata
  BlindFold constraint metadata derived from those formulas

jolt-witness owns:
  provider construction from private inputs
  oracle descriptors for a concrete execution/protocol instance
  available views and access capabilities
  source lifecycle and streaming contracts
  direct witness evaluation APIs
  public value access APIs

jolt-backends owns:
  physical storage format
  dense/materialized allocation
  device/host buffers
  cache policy and scratch memory
  layout-specific compute choices
```

For Jolt VM, that means:

```text
jolt-claims:
  "RamRa(2) is this logical committed polynomial, used by these formulas and
  opening IDs."

jolt-program:
  "This execution produced these normalized rows, bytecode facts, memory
  accesses, advice observations, and profile-checked field-inline rows."

jolt-witness:
  "For this execution, RamRa(2) is available as these oracle views with these
  dimensions, layouts, streaming properties, and direct evaluation behavior."

jolt-backends:
  "For this backend, consume RamRa(2) as one-hot indices, event logs, streaming
  chunks, dense host memory, or device-resident buffers."
```

The provider may use `jolt-claims` IDs to name Jolt VM witness oracles, but it
must not rederive protocol formulas locally. If a formula or protocol-visible
opening rule changes, the semantic change belongs in `jolt-claims`; the witness
provider adapts by exposing the data those semantics reference.

The Jolt VM provider is responsible for adapting the normalized trace plus
program preprocessing into witness data:

```text
TraceOutput<T: TraceSource>
  + Jolt program preprocessing
  + selected protocol/profile config
  + trusted/untrusted advice inputs
  + public IO
  -> JoltVmWitness
```

The provider may derive observed quantities such as trace length, RAM address
support, advice lengths, row-family counts, and field-inline row counts. Final
protocol choices that affect verifier-visible facts, such as padding policy,
opening order, stage selection, and claim formulas, remain owned by
`jolt-prover`, `jolt-claims`, and `jolt-verifier`.

## Core API Surface

### Namespaces

Every protocol defines its own namespace:

```rust
pub trait WitnessNamespace {
    type CommittedId;
    type VirtualId;
    type OpeningId;
    type PublicId;
    type ChallengeId;
}
```

Base Jolt VM uses Jolt protocol IDs:

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

Dory assist, wrapper, and BlindFold define distinct namespaces. They must not
reuse Jolt VM IDs merely because they are also represented as polynomial data.

### Provider Construction

The generic builder shape should allow protocol-specific inputs without baking
Jolt VM assumptions into the core trait:

```rust
pub trait WitnessBuilder<F> {
    type Namespace: WitnessNamespace;
    type Config;
    type Inputs<'a>;
    type Witness<'a>: WitnessProvider<F, Self::Namespace>
    where
        Self: 'a;

    fn build<'a>(
        &mut self,
        config: &Self::Config,
        inputs: Self::Inputs<'a>,
    ) -> Result<Self::Witness<'a>, WitnessError>;
}
```

For Jolt VM:

```rust
pub struct JoltVmWitnessInputs<'a, T: jolt_program::execution::TraceSource> {
    pub program: &'a jolt_program::execution::JoltProgram,
    pub preprocessing: &'a JoltProgramPreprocessing,
    pub trace: jolt_program::execution::TraceOutput<T>,
    pub trusted_advice: TrustedAdviceInput<'a>,
    pub untrusted_advice: UntrustedAdviceInput<'a>,
    pub public_io: PublicIo<'a>,
}
```

The exact type names can change. The important rule is that the Jolt VM witness
input is a `jolt-program` execution artifact plus preprocessing and advice
inputs, not a tracer-local emulator artifact.

### Provider Traits

The core provider traits are composable:

```rust
pub trait WitnessProvider<F, N: WitnessNamespace> {
    fn dimensions(&self) -> &WitnessDimensions;
    fn capabilities(&self) -> &WitnessCapabilities;

    fn describe_oracle(
        &self,
        oracle: OracleRef<N>,
    ) -> Result<OracleDescriptor<N>, WitnessError>;

    fn oracle_view(
        &self,
        request: OracleViewRequest<N>,
    ) -> Result<OracleView<'_, F>, WitnessError>;
}

pub trait PublicWitness<F, N: WitnessNamespace>: WitnessProvider<F, N> {
    fn public_value(&self, id: N::PublicId) -> Result<PublicValue<F>, WitnessError>;
}

pub trait CommittedPolynomialWitness<F, N: WitnessNamespace>:
    WitnessProvider<F, N>
{
    fn committed_ids(&self) -> &[N::CommittedId];

    fn committed_oracle(
        &self,
        id: N::CommittedId,
    ) -> Result<PolynomialOracle<'_, F>, WitnessError>;

    fn eval_committed(
        &self,
        id: N::CommittedId,
        point: EvaluationPoint<'_, F>,
    ) -> Result<F, WitnessError>;
}

pub trait VirtualPolynomialWitness<F, N: WitnessNamespace>:
    WitnessProvider<F, N>
{
    fn eval_virtual(
        &self,
        id: N::VirtualId,
        point: EvaluationPoint<'_, F>,
        ctx: &VirtualEvalContext<'_, F, N>,
    ) -> Result<F, WitnessError>;
}

pub trait OpeningWitnessProvider<F, N: WitnessNamespace>:
    WitnessProvider<F, N>
{
    fn opening_witness(
        &self,
        request: OpeningWitnessRequest<'_, F, N>,
    ) -> Result<OpeningWitness<F>, WitnessError>;
}
```

`PublicValue<F>` should support field values and structured public data that
can be lowered by protocol code:

```text
field scalar
byte string / public IO slice
commitment digest
program/preprocessing digest
dimension/config value
structured assist input
```

Witness providers may compute values, cache values, or expose already-built
views. They do not absorb those values into transcripts.

### Challenge Boundary

Transcript-derived values enter witness evaluation through explicit contexts:

```rust
pub struct VirtualEvalContext<'a, F, N: WitnessNamespace> {
    pub challenges: ChallengeMap<'a, N::ChallengeId, F>,
    pub publics: PublicMap<'a, N::PublicId, F>,
    pub openings: OpeningEvalMap<'a, N::OpeningId, F>,
}
```

This keeps the ownership split clear:

```text
witness provider:
  knows private rows, private columns, derived witness data, and public values
  derived from its inputs

jolt-prover / protocol prover:
  samples challenges, decides formula order, supplies evaluation contexts,
  requests openings, and constructs proof objects
```

## Witness-Prover-Backend Interop Contract

The witness/prover/backend boundary should be standardized enough that:

```text
jolt-prover:
  names logical witness oracles and required access patterns in protocol plans

jolt-witness:
  describes available oracles, dimensions, encodings, views, stream semantics,
  and cheap direct evaluations

jolt-backends:
  chooses views, allocates materialized storage, manages device/host memory,
  caches derived data, and releases workspace allocations
```

The shared contract should live in `jolt-witness` because it must not depend on
any particular prover or backend. `jolt-prover` may embed these generic request
types inside protocol-resolved plans, and `jolt-backends` may use them to ask
providers for source views or stream producers.

Representative shared types:

```rust
pub enum OracleKind<C, V> {
    Committed(C),
    Virtual(V),
}

pub struct OracleRef<N: WitnessNamespace> {
    pub kind: OracleKind<N::CommittedId, N::VirtualId>,
}

pub struct OracleDescriptor<N: WitnessNamespace> {
    pub oracle: OracleRef<N>,
    pub dimensions: PolynomialDimensions,
    pub available_views: ViewCapabilities,
    pub layout: PolynomialLayout,
    pub padding: PaddingPolicy,
    pub lifecycle: SourceLifecycle,
    pub cost: ViewCostHints,
}

pub struct ViewRequirement {
    pub access: AccessPattern,
    pub preferred_layouts: LayoutPreference,
    pub accepted_encodings: EncodingPreference,
    pub streaming: StreamingRequirement,
    pub materialization: MaterializationPolicy,
    pub retention: RetentionHint,
}

pub struct OracleViewRequest<N: WitnessNamespace> {
    pub oracle: OracleRef<N>,
    pub requirement: ViewRequirement,
}

pub struct OracleView<'a, F> {
    pub descriptor: OracleDescriptorRef<'a>,
    pub view: PolynomialView<'a, F>,
    pub lease: ViewLease,
}
```

The exact Rust shape can evolve, but these concepts are required:

```text
OracleRef:
  logical protocol identity, not storage identity

OracleDescriptor:
  dimensions, layout, padding, available views, source lifecycle, and cost hints

ViewRequirement:
  consumer's required access pattern, accepted encodings, streaming needs,
  materialization policy, and retention horizon

OracleView:
  borrowed/source view, stream producer, lazy recipe, or small already-owned
  data returned by the provider

BackendWorkspace:
  backend-owned dense columns, device buffers, scratch buffers, cached eq
  tables, materialized RA state, Dory hints, and other large allocations
```

`BackendWorkspace` is not a `jolt-witness` type. The witness contract says what
can be read and how; the backend decides where large data lives.

### Access Patterns

The standard `AccessPattern` vocabulary should cover at least:

```text
direct_eval:
  evaluate at one multilinear point without materializing the oracle

random_coefficients:
  read arbitrary coefficient positions

sequential_coefficients:
  scan coefficients in canonical order

row_chunks:
  stream row-major chunks for commitment or upload

address_chunks:
  stream address-major chunks or event groups

bind_prefix / bind_suffix:
  bind some variables while preserving an efficient residual representation

vector_matrix_product:
  PCS-oriented VMP access for Dory/RLC paths

event_log:
  access sparse RAM/register/RA event rows directly
```

The provider may reject unsupported access with `WitnessError::Unsupported`.
The backend can then choose a different view, request explicit materialization,
or report that the selected backend cannot execute the plan.

### Materialization Policy

Materialization must be explicit:

```text
BorrowOnly:
  provider may return only borrowed/source views or streams

AllowBackendMaterialization:
  backend may allocate and fill materialized storage in its workspace from a
  provider view or recipe

RequireMaterialized:
  backend requires a concrete dense/compact/sparse representation and owns the
  resulting allocation

ForbidDense:
  dense fallback is not allowed for this request because it would violate memory
  policy or performance expectations
```

`jolt-witness` may own compact source data, event logs, normalized trace views,
small derived tables, or replayable stream producers. It should not eagerly
allocate every committed polynomial in dense field form. If dense data is
needed, the backend requests it and accounts for it in its workspace.

### Lifecycle And Retention

Lifecycle must be visible because memory pressure is a protocol-level and
backend-level constraint:

```text
SourceLifecycle:
  BorrowedFromWitness
  ReplayableStream
  SingleUseStream
  DerivedRecipe
  SmallOwnedValue

RetentionHint:
  Ephemeral
  StageLocal
  UntilOpening
  UntilBlindFold
  WholeProof
```

`jolt-prover` decides the retention horizon from protocol dependencies:
commitment hints needed by Stage 8, ZK coefficients needed by BlindFold, advice
state spanning stages, wrapper handoff data, and Dory-assist inputs.

`jolt-backends` decides how to satisfy that horizon: keep a borrowed view,
materialize compactly, upload to device memory, recompute from a recipe, replay
a stream, or drop and regenerate later. Backend choices must preserve the
protocol-visible values and satisfy the retention hint.

The intended flow is:

```text
1. jolt-prover builds a protocol plan with OracleRef + ViewRequirement slots.
2. backend inspects witness OracleDescriptor values.
3. backend requests OracleView values or stream producers from jolt-witness.
4. backend allocates/materializes only what it needs in BackendWorkspace.
5. backend returns slot-keyed outputs and retained backend artifacts.
6. jolt-prover advances the protocol and emits release/retention information.
7. backend drops or reuses workspace allocations according to that information.
```

This gives the pieces a stable contract without forcing a single storage format
or access strategy across CPU, CUDA, Metal, and hybrid backends.

## Polynomial Oracles And Views

Polynomial access must be oracle-based and view-oriented:

```rust
pub struct PolynomialOracle<'a, F> {
    pub metadata: PolynomialMetadata,
    pub views: PolynomialViews<'a, F>,
}

pub enum PolynomialView<'a, F> {
    Dense(DenseView<'a, F>),
    Compact(CompactView<'a>),
    OneHot(OneHotView<'a>),
    BitPacked(BitPackedView<'a>),
    EventLog(EventLogView<'a>),
    Streaming(StreamingPolynomialView<'a, F>),
    Lazy(LazyPolynomialView<'a, F>),
    Rlc(RlcPolynomialView<'a, F>),
}
```

The oracle is the logical proof-system object. A view is a concrete access
contract over that oracle. An oracle may expose multiple views, and a backend
chooses the one matching its compute strategy and memory budget.

Each oracle exposes metadata:

```rust
pub struct PolynomialMetadata {
    pub dimensions: PolynomialDimensions,
    pub encoding: PolynomialEncoding,
    pub access: PolynomialAccess,
    pub padding: PaddingPolicy,
    pub layout: PolynomialLayout,
}
```

Representative encodings:

```text
Dense:
  field elements in canonical domain order

Compact:
  bool/u8/u16/u32/u64/u128/i64/i128/S128-like small scalars promoted lazily

OneHot:
  support indices plus one-hot address/cycle dimensions

SparseEvents:
  event rows such as RAM/register read-write accesses or RA index streams

BitPacked:
  packed bits or small limbs with explicit endianness and signedness

Streaming:
  deterministic chunk producer with domain order, chunk shape, and replay
  guarantees

Lazy:
  derived column or virtualized polynomial with direct eval/bind capability

RLC:
  joint linear-combination view over committed/opening data, possibly streaming
```

Representative access capabilities:

```text
random_access_coefficients
sequential_chunks
direct_multilinear_eval
direct_bind_prefix
direct_bind_suffix
stream_commit_rows
stream_vector_matrix_product
materialize_dense
materialize_compact
```

The API should make expensive conversions explicit. A backend may choose to
materialize a dense vector, but the type system should not require dense
materialization to ask basic questions about shape, evaluation, streaming, or
commitment input.

## Dimensions And Layout

`WitnessDimensions` should describe the witness shape without requiring the
consumer to know the protocol provider internals:

```rust
pub struct WitnessDimensions {
    pub domains: DomainDimensions,
    pub advice: AdviceDimensions,
    pub one_hot: OneHotDimensions,
    pub memory: MemoryDimensions,
    pub field_inline: Option<FieldInlineDimensions>,
    pub provider_specific: ProviderDimensionMap,
}
```

For Jolt VM, dimensions include:

```text
raw trace length
padded trace length
log_T
RAM address dimension / ram_K
bytecode dimension
instruction lookup one-hot dimensions
bytecode one-hot dimensions
RAM one-hot dimensions
trusted/untrusted advice dimensions
main/advice PCS embedding dimensions
field-inline row/register/product dimensions when enabled
```

Padding and embedding policies that affect verifier-visible claims must be
selected by protocol/config code. The witness provider can report observed
support and expose views under the selected dimensions.

## Streaming Model

Streaming is a first-class witness capability because the Jolt CPU fast path
depends on avoiding full materialization of committed witness polynomials.

The initial streaming shape should be deterministic and backend-friendly:

```rust
pub struct StreamingPolynomialView<'a, F> {
    pub metadata: PolynomialMetadata,
    pub policy: StreamingPolicy,
    pub producer: StreamingProducer<'a, F>,
}
```

Required properties:

- stable chunk order;
- explicit row-major/address-major/domain-major layout;
- explicit padding behavior;
- deterministic replay if the view advertises replayability;
- clear error if the stream is single-use and a second pass is requested;
- chunk metadata sufficient for PCS streaming commitment and RLC construction;
- no transcript side effects.

Streaming views should support at least:

```text
committed witness row chunks for CycleMajor Dory commitment
advice chunk producers
Stage 8 RLC streaming over trace and advice data
future GPU-friendly chunk upload paths
```

AddressMajor or unsupported layouts may start with materialized fallback views,
but the fallback must be explicit in the view capabilities and benchmarked in
the CPU backend port.

## Jolt VM Provider

The first concrete provider is:

```text
jolt_witness::protocols::jolt_vm::JoltVmWitness
```

It consumes `jolt-program::execution` outputs and exposes the base Jolt witness
namespace.

Internally, the provider should separate the baseline RV64 witness from the
optional field-inline extension:

```text
jolt_witness::protocols::jolt_vm
  rv64/
    trace rows
    registers
    RAM
    bytecode
    instruction lookups
    advice

  field_inline/
    field rows
    field-inline bytecode metadata
    FR register events
    FieldRdInc
    virtual FR values
    product lanes
    bridge views
    bytecode anchoring views
```

Field inline is not a separate top-level witness provider. It is an optional
extension of one Jolt VM execution because it shares the trace domain, program
image, bytecode path, public IO, bridge rows, selected verifier flow, and final
proof with the RV64 witness.

### Inputs

```text
JoltProgram / program image metadata
program preprocessing and bytecode expansion artifacts
TraceOutput<T: TraceSource>
trusted advice input
untrusted advice input
public input/output bytes
selected profile/protocol config
```

### RV64 Witness Data

Committed witness data:

```text
RdInc
RamInc
InstructionRa(d)
BytecodeRa(d)
RamRa(d)
TrustedAdvice
UntrustedAdvice
```

Virtual witness data:

```text
program counter / bytecode row values
instruction flags
operand and output values
register read/write values
RAM read/write address and value views
lookup operands and lookup outputs
read/write helper polynomials
Hamming/booleanity helper views
advice-derived virtuals
```

Provider-owned derived data:

```text
normalized trace row views
RAM/register event logs
RA indices and support views
advice word polynomials
initial/final memory observations
cached derived columns when profitable
```

Protocol-owned facts:

```text
claim formulas
sumcheck batching order
stage order
opening ID order
transcript labels
clear vs BlindFold selection
proof object fields
```

### Field Inline

When field inline is enabled, the Jolt VM provider also exposes field-inline
witness data under field-inline IDs from `jolt-claims`:

```text
field_rows
field-inline bytecode metadata
FR register read/write event logs
FieldRs1Value
FieldRs2Value
FieldRdValue
FieldRegistersVal
FieldRs1Ra
FieldRs2Ra
FieldRdWa
FieldRdInc
FieldProduct
FieldInvProduct
FieldOpFlag(...)
bridge encodings between x-register values and field registers
```

Current native-field v1 placement:

```text
committed field-register polynomial surface:
  FieldRdInc

virtual field-register openings:
  FieldRegistersVal
  FieldRs1Ra
  FieldRs2Ra
  FieldRdWa
  FieldRs1Value
  FieldRs2Value
  FieldRdValue
  FieldProduct
  FieldInvProduct
```

`FieldRs1Ra`, `FieldRs2Ra`, and `FieldRdWa` are virtual openings, not new
committed polynomials. They are anchored through the field-inline extension of
`BytecodeReadRaf`, then through the ordinary committed `BytecodeRa(i)` path.
The witness provider must expose the bytecode-side metadata and virtual views
needed for that anchoring; it should not introduce a separate committed RA
surface unless `jolt-claims` changes the protocol semantics.

Field-inline bytecode metadata is required only when field inline is selected.
In a build without the Cargo `field-inline` feature, the Jolt VM witness
provider should not expose the field-inline namespace at all. In a build with
the feature, an FR-off profile exposes no field-inline commitments, claims,
challenges, bytecode metadata, sumcheck rows, transcript data, or dummy zero
relations. Provider/test fixtures should cover metadata length, active flag
count, operand shape, inactive-row cleanliness, and field-register bounds
because those are verifier-bound protocol inputs.

The provider should use `jolt-program` profile checks and normalized field-row
data. Pure field operations should have pure FR register events; bridge rows
should explicitly connect ordinary x-register witnesses to FR witnesses. If an
execution backend keeps incidental ordinary x-register accesses on pure field
rows for compatibility, those accesses must be exposed as inert rather than
silently entering ordinary register witness semantics. The provider should not
infer field-inline legality from raw tracer behavior.

## Non-VM Providers

The generic machinery should be exercised by at least one non-VM provider early
enough to catch VM-specific leakage.

### Dory Assist

Dory assist witnesses are not VM-shaped. They expose:

```text
typed Dory verifier operation traces
G1/G2/GT operation-family rows
multi-Miller-loop rows
packing witnesses for dense/prefix-packed Hyrax rows
public Dory opening snapshot inputs
Hyrax opening witness data
Miller-loop intermediate values
```

This provider should use a Dory-assist namespace with operation trace IDs,
packing IDs, opening IDs, public IDs, and challenge IDs. It should not reuse the
Jolt VM namespace.

### Wrapper

Wrapper witnesses expose configured-verifier assignment data:

```text
transparent Jolt proof fields in verifier order
clear opening claims
verifier stage intermediate values
transcript replay state
sumcheck verifier equation variables
opening snapshot variables
optional Dory-assist verifier variables
R1CS assignment slices
```

The wrapper provider should feed `jolt-wrapper` R1CS assembly. It should not
own the wrapper SNARK backend, nor the configured verifier protocol semantics.

### BlindFold

BlindFold witnesses expose data generated during committed proving:

```text
committed sumcheck round coefficients
round blindings
committed output-claim rows
output-row blindings
final hidden PCS evaluation and blinding
auxiliary rows for the BlindFold verifier-equation R1CS
```

The BlindFold provider may be built from `jolt-prover` private stage outputs,
but `jolt-witness` itself must not depend on `jolt-prover`. Use a plain input
struct or a conversion layer in `jolt-prover` if needed.

## Error Model

Errors should distinguish:

```text
unknown ID
unsupported capability
dimension mismatch
point dimension mismatch
encoding conversion unavailable
stream already consumed
provider input invalid
protocol/config mismatch
internal invariant failure
```

Witness providers should fail early on shape/config mismatches. They should not
silently materialize dense fallbacks unless the oracle metadata reports that
fallback and the caller requested it.

## Testing

Core tests:

- dense-reference evaluation for every `PolynomialView` representation;
- compact-vs-dense bind/eval parity;
- one-hot/RA-vs-dense parity on small domains;
- sparse event view parity against dense matrices on small traces;
- bit-packed decoding/encoding tests;
- streaming-vs-materialized chunk parity;
- opening witness request shape and point-dimension checks;
- public value and challenge-context plumbing tests.

Jolt VM provider tests:

- build from `jolt_program::execution::TraceOutput<T>` fixtures;
- no production dependency on `tracer` internals;
- committed polynomial ID coverage;
- virtual polynomial eval parity against dense/reference computations;
- public IO and advice value checks;
- trace length, padded length, `ram_K`, one-hot dimensions, and advice
  dimensions against `jolt-core` fixtures during the transition;
- CycleMajor streaming commitment input parity;
- AddressMajor materialized fallback parity where supported;
- field-inline row/register/product/bridge witness checks when enabled;
- field-inline bytecode metadata validation fixtures;
- field-inline virtual `FieldRs1Ra`/`FieldRs2Ra`/`FieldRdWa` anchoring checks
  against the ordinary committed `BytecodeRa(i)` path;
- FR-off checks that no field-inline metadata, commitments, claims, challenges,
  rows, or dummy zero relations are exposed.

Non-VM provider tests:

- Dory-assist synthetic operation-trace and packing witness checks;
- wrapper assignment shape checks against selected verifier fixtures;
- BlindFold committed-round/output-row witness shape checks;
- namespace isolation tests proving VM IDs are not required by non-VM providers.

Integration tests:

- `jolt-prover` accepts a Jolt VM witness provider and produces verifier-visible
  stage frontiers;
- `jolt_backends::cpu` can consume compact, one-hot, sparse, streaming, and
  lazy views without forcing dense materialization;
- Dory assist and wrapper prototypes can consume the same core witness traits.

## Milestones

1. Scaffold `jolt-witness` core: namespace traits, dimensions, encodings,
   polynomial oracles/views, public values, virtual eval contexts, opening requests,
   and errors.
2. Add dense, compact, one-hot, sparse-event, lazy, and streaming view
   reference tests.
3. Implement `jolt_witness::protocols::jolt_vm::rv64` from
   `jolt-program::execution` RV64 trace outputs.
4. Expose RV64 Jolt committed and virtual witness data with dense-reference
   parity tests.
5. Add advice, public IO, opening witness, and streaming commitment views.
6. Add field-inline witness views behind the selected Jolt profile/config:
   `FieldRdInc`, FR register virtuals, product lanes, bridge rows, and
   bytecode-anchored virtual RA/WA openings.
7. Add a minimal Dory-assist or wrapper provider prototype to validate that the
   core traits are not VM-specific.
8. Wire the provider into the first `jolt-prover` frontier and
   `jolt_backends::cpu` commitment path.

## Open Questions

- Should provider modules live in one crate initially, or should heavy optional
  providers be split once Cargo dependencies become clear?
- What is the smallest useful `PolynomialAccess` capability set for the first
  CPU commitment and Stage 1 frontiers?
- Should streaming views be synchronous iterators first, with async/GPU upload
  adapters added later?
- How much of the current `jolt-core` `MultilinearPolynomial` enum should move
  into `jolt-witness` versus stay as a CPU backend internal representation?
- Which field-inline dimensions belong in generic `WitnessDimensions` versus a
  provider-specific dimension map?
- Should `OpeningWitness` include PCS-specific hints directly, or should hints
  be backend-private side data keyed by opening slots?

## Review Checklist

- Core traits do not mention Jolt VM concrete types.
- Jolt VM provider consumes `jolt-program::execution`, not tracer internals.
- Witness providers do not sample challenges or own transcript labels.
- Witness providers do not own claim formulas, stage order, opening order, or
  proof object construction.
- Polynomial oracles/views preserve compact, sparse, one-hot, lazy, and streaming
  capabilities.
- Dense materialization is explicit and test-covered.
- Public values and challenge inputs are distinct.
- Non-VM provider prototypes can reuse the same core APIs.
- CPU backend perf-critical views needed by the optimization inventory are
  representable without abstraction-driven regressions.

## References

- [`jolt-prover` model crate spec](./jolt-prover-model-crate.md)
- [`jolt-prover` CPU backend port spec](./jolt-prover-cpu-backend-port.md)
- [`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md)
- [Dory assist protocol spec](./dory-assist-protocol.md)
- [Wrapper protocol spec](./wrapper-protocol.md)
- [Field inline protocol spec](./field-inline-protocol.md)
- [Field inline `jolt-program` / `tracer` integration spec](./field-inline-program-tracer.md)
- `jolt-core/src/zkvm/witness.rs`
- `jolt-core/src/zkvm/prover.rs`
- `jolt-core/src/poly/multilinear_polynomial.rs`
- `jolt-core/src/poly/compact_polynomial.rs`
- `jolt-core/src/poly/one_hot_polynomial.rs`
- `jolt-core/src/poly/ra_poly.rs`
- `jolt-core/src/poly/shared_ra_polys.rs`
- `jolt-core/src/poly/rlc_polynomial.rs`
