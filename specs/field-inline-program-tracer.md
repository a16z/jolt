# Spec: Field Inline `jolt-program` / `tracer` Integration

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-25 |
| Status | draft |
| PR | TBD |

## Purpose

This spec defines the non-witness work needed to support field-inline memory in
the canonical program and execution pipeline. The goal is to make
`jolt-program` expose stable, tracer-free field-inline program and trace
artifacts, while `tracer` remains the concrete backend that executes field
instructions and produces those artifacts.

The boundary is:

```text
jolt-program:
  field-inline source/final row legality
  field-inline bytecode metadata
  backend-neutral trace-row payloads
  preprocessing/profile fingerprints

tracer:
  concrete field-register memory
  native field instruction execution
  bridge execution
  adaptation into jolt-program trace rows

jolt-witness:
  consumes jolt-program artifacts and exposes witness oracles/views
```

`jolt-program` must not depend on `tracer`, `jolt-witness`, `jolt-prover`,
`jolt-backends`, PCS crates, transcripts, or prover-only witness generation.

## Scope

V1 field inline is native-field only:

```text
field-inline arithmetic modulus = modulus(F: JoltField)
```

The program/trace layer should still avoid being generic over `F`. It should
carry canonical field-value encodings selected by the field-inline profile. The
witness/prover side decodes those encodings into the concrete proof field.

V1 scope:

```text
field-inline source instruction/profile support
field-inline final bytecode rows or metadata hooks
field-inline bytecode metadata in program preprocessing
optional field-inline payload on execution trace rows
tracer field-register file with 16 slots
pure field op and x-register <-> FR bridge execution
field value canonical encoding
fixture coverage for trace/witness handoff
```

Out of scope:

```text
non-native modular arithmetic
limb range-check witnesses
quotient/reduction witnesses
PCS/prover/witness materialization
field-inline claim formulas
field-inline opening order
field-register committed polynomial construction
GPU/backend storage layout
```

## Gating Model

Field inline should use two separate gates, with the Cargo feature as the outer
availability gate:

```text
Cargo/build gate:
  Does this crate/binary include field-inline support code at all?

Program/profile gate:
  Does this particular program/proof instance enable field-inline semantics?
```

In a build without the Cargo `field-inline` feature, prover, verifier, witness,
claims, R1CS, and tracer code should not know how to handle field inline. The
build should not expose field-inline prover stages, verifier checks, witness
providers, field-register trace payloads, or tracer FR execution. Any FR-on
program/proof/artifact loaded into such a build must fail early with an
unsupported-feature error.

In a build with the Cargo `field-inline` feature, the program/profile gate is
the semantic gate. The same binary must still prove and verify FR-off programs
with no field-inline metadata, no field-inline trace payloads, no field-inline
commitments, no field-inline claims, and no dummy zero relations.

### Per-Crate Gates

| Crate | Cargo/build gate | Program/profile gate |
|-------|------------------|----------------------|
| `jolt-riscv` | Add `field-inline` for field-inline source/final row identities, profile extensions, static operand-shape metadata, and serialization of those rows. Ordinary row tags/names remain available and unchanged without it. | `JoltInstructionProfile` decides whether field-inline source/final rows are legal. FR-off profiles reject them and exclude them from supported dense indexes. |
| `jolt-program` | `field-inline = ["jolt-riscv/field-inline"]` gates accepting field-inline rows, emitting field-inline bytecode metadata, and exposing field-inline metadata/trace payload types. Without it, FR-on programs are rejected. | Selected profile/config decides whether field-inline rows are accepted, metadata is required, and metadata digest/fingerprint is present. |
| `tracer` | `field-inline = ["jolt-riscv/field-inline", "jolt-program/field-inline"]` gates concrete FR execution code, field-register state, field-inline instruction dispatch, and `FieldInlineTraceData` emission. | FR-on profile allocates/updates FR memory and emits `FieldInlineTraceData`; FR-off execution emits none. FR-on program on a tracer without support is an early unsupported-feature error. |
| `jolt-witness` | `field-inline` gates the Jolt VM field-inline provider extension and depends on `jolt-program/field-inline` and `jolt-claims/field-inline`. Generic witness infrastructure stays available. | FR-on artifacts expose field-inline oracle views; FR-off exposes no field-inline namespace data. FR-on artifacts without compiled support are rejected early. |
| `jolt-claims` | Existing `field-inline` Cargo feature gates field-inline protocol IDs, formulas, relation metadata, and opening/public/challenge definitions. Without it, prover/verifier code cannot name field-inline claims. | Selected Jolt protocol config decides whether the field-inline namespace is active in this proof. |
| `jolt-r1cs` / `jolt-verifier` | Existing `field-inline` Cargo feature gates verifier/R1CS field-inline checks. Without it, verifier code has no field-inline stage composition. | Proof/preprocessing profile and metadata decide whether field-inline checks are run. Unsupported FR-on proofs are rejected. |
| `jolt-prover` | `field-inline` Cargo feature should depend on `jolt-claims/field-inline`, `jolt-witness/field-inline`, and any field-inline R1CS/prover stage support. | FR-on proof configs request field-inline witness views and stages; FR-off skips the whole field-inline path. |
| `jolt-backends` | Generic polynomial/oracle storage should not need a field-inline gate. Optional optimized field-inline kernels may be gated. | Backends allocate only the oracle descriptors requested by the active witness/prover plan. |
| SDK / guest-facing crates | Guest intrinsics and host profile selection may be gated by a `field-inline` feature. | Generated program metadata selects an FR-on profile when field-inline guest ops are used. |

Recommended feature propagation:

```text
jolt-riscv/field-inline
  -> jolt-program/field-inline

jolt-program/field-inline
  -> tracer/field-inline

jolt-program/field-inline
  -> jolt-witness/field-inline
  -> jolt-prover/field-inline

jolt-claims/field-inline
  -> jolt-r1cs/field-inline
  -> jolt-verifier/field-inline
  -> jolt-prover/field-inline
```

### Gating Invariants

- Field-inline row identities must not be tracer-private and should not be
  hidden behind the generic inline source key once they carry VM memory/proof
  semantics.
- Disabling the Cargo feature may remove field-inline symbols/types/modules,
  but it must not change the meaning of ordinary row tags, canonical names, or
  ordinary-profile fingerprints.
- Enabling the Cargo feature may add field-inline tags and names, but ordinary
  tags should stay explicitly allocated and stable.
- FR-off behavior must be structurally identical to ordinary Jolt: no metadata,
  no commitments, no challenges, no claims, no transcript absorbs, no BlindFold
  rows, and no dummy zeros.
- FR-on behavior must fail early if any required layer was built without
  field-inline support.
- The selected profile, field-inline-enabled bit, metadata digest, field value
  encoding, and field-register parameters must be serialized in preprocessing
  and bound by the verifier/prover transcript path.
- Runtime checks should reject inconsistent artifacts, for example FR-on
  profile with missing metadata, FR-off profile with field-inline trace payload,
  or field-inline trace rows without matching final bytecode metadata.

## Ownership

| Component | Owns |
|-----------|------|
| `jolt-riscv` | Source/final instruction identities, profile/capability metadata, field-inline opcode/flag vocabulary |
| `jolt-program::expand` | Expansion from accepted source rows to final Jolt rows and field-inline bytecode metadata |
| `jolt-program::preprocess` | Field-inline metadata storage, validation, serialization, digest/profile fingerprint inclusion |
| `jolt-program::execution` | Backend-neutral field-inline trace payload types and trace contract |
| `tracer` | Concrete CPU/FR state, field arithmetic execution, bridge behavior, conversion to neutral trace rows |
| `jolt-claims` | Field-inline protocol semantics, formulas, relation IDs, opening/public/challenge metadata |
| `jolt-witness` | Field-inline witness data access over `jolt-program` trace/preprocessing artifacts |

## Required `jolt-riscv` Changes

`jolt-riscv` is the shared static instruction layer. Under the
`field-inline` feature, field inline must extend this crate before
`jolt-program`, `tracer`, or `jolt-witness` can agree on what a field-inline
row is. Without the feature, field-inline row identities and helpers should be
absent or unavailable, and decoding/loading FR-on artifacts should fail early.

### Instruction Vocabulary

Add field-inline instruction identities to the shared source/final instruction
vocabulary. The exact names can change, but v1 needs stable identities for:

```text
FieldLoadFromX
FieldLoadImm
FieldAdd
FieldSub
FieldMul
FieldInv
FieldAssertEq
FieldStoreToX
```

If a source field-inline instruction expands into one or more final Jolt rows,
`jolt-riscv` should still own both identities:

```text
SourceInstructionKind:
  accepted guest/source field-inline operations

JoltInstructionKind:
  final proof bytecode rows emitted by jolt-program expansion
```

Field-inline rows that have protocol meaning should not remain hidden behind a
generic `Inline` source key. The generic inline mechanism is useful for external
accelerated inlines, but field-inline VM memory has stable bytecode metadata and
proof semantics, so it needs first-class row identities.

### Profile And Fingerprint Metadata

Add field-inline profile/capability metadata:

```text
SourceExtension::JoltFieldInline or equivalent
JoltTargetExtension::FieldInline or equivalent
field-inline-enabled profile constants
profile fingerprint coverage
source/final dense-index coverage
```

Profile checks should answer:

```text
is this field-inline source row legal?
is this final field-inline row legal?
does this profile require field-inline bytecode metadata?
does this profile require field-inline trace payloads?
```

FR-off profiles must reject field-inline source rows and must not include
field-inline final rows in supported final-row indexing. In builds without the
Cargo `field-inline` feature, only FR-off profiles are constructible/exported.

### Static Field-Inline Classification

`jolt-riscv` should expose structural classification helpers that are shared by
`jolt-program` and `tracer`:

```rust
pub fn is_field_inline_source(kind: SourceInstructionKind) -> bool;
pub fn is_field_inline_jolt(kind: JoltInstructionKind) -> bool;
pub fn field_inline_operand_shape(kind: JoltInstructionKind) -> Option<FieldInlineOperandShape>;
```

The shape should be static metadata only:

```rust
pub struct FieldInlineOperandShape {
    pub op: FieldInlineOp,
    pub reads_fr_rs1: bool,
    pub reads_fr_rs2: bool,
    pub writes_fr_rd: bool,
    pub bridge_x_register_role: Option<FieldInlineXRegisterRole>,
    pub has_immediate: bool,
}
```

Useful helpers include:

```text
is pure field op
is x-register -> FR bridge
is FR -> x-register bridge
does row read ordinary x-register
does row write ordinary x-register
does row require FieldProduct / FieldInvProduct payload
```

These helpers describe row shape, not execution values. They should not perform
field arithmetic, build witness columns, or encode protocol claims.

### Stable Row Encoding

Final field-inline rows need stable tags/canonical names so bytecode metadata,
profile fingerprints, lookup routing, and witness/prover IDs do not drift
accidentally. Field-inline additions should update the same generated surfaces
ordinary rows use today:

```text
instruction kind enum generation
canonical names
JoltInstructionTag allocation
SourceInstructionRow / JoltInstructionRow conversion
profile dense indexes
serialization, if enabled
```

Tag allocation should be explicit and reviewable. Changing a field-inline tag or
canonical name after fixtures exist is a proof-shape compatibility change.
Ordinary tags and canonical names must stay identical in builds with and
without `field-inline`.

### Interface To `tracer`

`tracer` should continue to consume `jolt-riscv` as a lower-level vocabulary:

```text
jolt-riscv field-inline instruction identities
  -> tracer concrete instruction structs / enum variants
  -> tracer execution dispatch
  -> conversion back to JoltInstructionRow
  -> jolt_program::execution::TraceRow
```

The instruction list should remain single-sourced. If field-inline rows are
added to `jolt-riscv` macros, `tracer` should reuse those macro entries or a
field-inline-specific generated list rather than duplicating opcode names and
tags manually.

### Non-Goals

`jolt-riscv` must not own:

```text
field-register runtime state
native field arithmetic execution
FieldEncodedValue conversion into F
FieldRdInc construction
FieldRs1Ra / FieldRs2Ra / FieldRdWa anchoring semantics
sumcheck formulas
PCS/prover/opening behavior
```

## Required `jolt-program` Changes

### Profile And Program Artifact

`JoltProgram` and `JoltProgramPreprocessing` should carry the selected program
profile and a stable profile/config fingerprint. Field inline must be selected
before expansion and preprocessing, not inferred from trace rows later.

Required facts:

```text
selected source profile
selected final-row legality profile
field-inline enabled/disabled bit
field register log_k, currently 4
field value encoding profile
field-inline metadata digest/fingerprint
```

FR-off programs should have no field-inline metadata payload and no field-inline
trace-row requirements. FR-on programs require metadata for every final trace
row or enough indexed metadata to validate active/inactive rows unambiguously.

### Field-Inline Bytecode Metadata

`jolt-program::expand` / `preprocess` should produce a structural metadata
artifact for field inline:

```rust
pub struct FieldInlineBytecodeMetadata {
    pub rows: Vec<FieldInlineBytecodeRow>,
    pub field_register_log_k: u8,
    pub value_encoding: FieldValueEncoding,
    pub profile_fingerprint: ProfileFingerprint,
}

pub struct FieldInlineBytecodeRow {
    pub active: bool,
    pub op: Option<FieldInlineOp>,
    pub rs1: Option<FieldRegister>,
    pub rs2: Option<FieldRegister>,
    pub rd: Option<FieldRegister>,
    pub bridge_x_register: Option<u8>,
    pub immediate: Option<FieldEncodedValue>,
}
```

Exact names can change. The artifact must support verifier/prover validation
of:

```text
length matches final bytecode/trace domain expectations
active flag count
operand shape for each active operation
inactive rows are clean
field-register bounds: register < 2^field_register_log_k
bridge rows identify the ordinary x-register operand they connect to
```

This metadata is structural program data. `jolt-claims` defines how the
metadata is used in protocol formulas. `jolt-program` should not duplicate
field-inline claim formulas.

### Field Value Encoding

`jolt-program` should define a field-value encoding that is independent of
arkworks field types:

```rust
pub struct FieldEncodedValue {
    pub bytes_le: Vec<u8>,
}

pub struct FieldValueEncoding {
    pub byte_len: u16,
    pub limb_bits: u16,
    pub limb_count: u16,
    pub canonical: bool,
}
```

For v1, the encoding must be canonical for the selected proof field
instantiation. If both 128-bit and 254-bit fields are supported, they are
different encoding profiles, not different arithmetic protocols.

`jolt-program` should validate structural encoding length/canonicality where it
can do so without importing the proving field. Full conversion to `F` belongs
in `jolt-witness`.

### Execution Trace Contract

The current trace row is RV64-only:

```rust
pub struct TraceRow {
    pub instruction: JoltInstructionRow,
    pub registers: RegisterState,
    pub ram_access: RamAccess,
}
```

Field inline should extend this into an RV64 payload plus optional field-inline
payload. This can be an additive migration first, but the canonical shape is:

```rust
pub struct TraceRow {
    pub instruction: JoltInstructionRow,
    pub rv64: Rv64TraceData,
    pub field_inline: Option<FieldInlineTraceData>,
}

pub struct Rv64TraceData {
    pub registers: RegisterState,
    pub ram_access: RamAccess,
}
```

`FieldInlineTraceData` should carry the execution facts needed by witness
generation:

```rust
pub struct FieldInlineTraceData {
    pub op: FieldInlineOp,
    pub rs1: Option<FieldRegisterRead>,
    pub rs2: Option<FieldRegisterRead>,
    pub rd: Option<FieldRegisterWrite>,
    pub product: Option<FieldEncodedValue>,
    pub inv_product: Option<FieldEncodedValue>,
    pub bridge: Option<FieldInlineBridge>,
}

pub struct FieldRegisterRead {
    pub register: u8,
    pub value: FieldEncodedValue,
}

pub struct FieldRegisterWrite {
    pub register: u8,
    pub pre_value: FieldEncodedValue,
    pub post_value: FieldEncodedValue,
}

pub enum FieldInlineBridge {
    LoadFromX {
        x_register: u8,
        x_value: u64,
        field_value: FieldEncodedValue,
    },
    StoreToX {
        field_register: u8,
        field_value: FieldEncodedValue,
        x_register: u8,
        x_value: u64,
    },
}
```

Exact representation can evolve. The required semantics are:

```text
pure field rows:
  have field_inline payload
  have FR register events
  suppress ordinary x-register accesses in rv64 payload
  have no RAM access unless the selected operation explicitly requires one

bridge rows:
  have field_inline payload
  have the required ordinary x-register read/write in rv64 payload
  have the required FR read/write in field_inline payload
  expose the bridge encoding payload explicitly

FR-off rows:
  have field_inline = None
```

If `tracer` cannot immediately suppress incidental ordinary x-register accesses
on pure field rows, the trace contract must mark them as inert explicitly. The
canonical behavior is suppression.

### Trace Output And Streaming

`TraceOutput<T: TraceSource>` can keep the same outer shape, but the `TraceRow`
returned by `TraceSource` must be sufficient for `jolt-witness` without
downcasting to tracer internals.

The trace contract should also remain streaming-friendly:

```text
TraceSource::next_row() -> Option<TraceRow>
```

must expose all field-inline payload data row-by-row. Witness generation should
not need tracer lazy checkpoints, CPU state pointers, or advice-tape internals
to recover FR events.

## Required `tracer` Changes

### Field Register Memory

`tracer` should own a concrete FR register file:

```text
field_register_log_k = 4
16 slots
each slot stores FieldEncodedValue or backend-local native field value
```

The concrete execution backend may store native field values internally for
speed, but the boundary value emitted to `jolt-program::execution` must be the
canonical `FieldEncodedValue`.

### Field Operation Execution

Tracer execution must implement:

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

Execution behavior:

```text
FIELD_MUL:
  read fr1/fr2
  compute product
  write frd
  emit FieldProduct = fr1 * fr2

FIELD_INV:
  read fr1
  write frd
  emit FieldInvProduct = fr1 * frd
  enforce/return execution error on invalid inverse behavior according to the
  selected field-inline instruction semantics

FIELD_LOAD_FROM_X:
  read ordinary x-register
  decode to field value
  write FR register
  emit bridge payload

FIELD_STORE_TO_X:
  read FR register
  encode to x-register value
  write ordinary x-register
  emit bridge payload
```

Concrete arithmetic is tracer-owned. `jolt-program` names the encoded payload;
it does not execute field arithmetic.

### Adapter To `jolt-program`

`tracer` implements `jolt_program::execution::ExecutionBackend` by adapting its
internal execution state into:

```text
TraceRow {
  instruction,
  rv64,
  field_inline,
}
```

The adapter must not leak:

```text
tracer::Cycle
Cpu
LazyTraceIterator internals
advice tape internals
field register file internals
```

into `jolt-program` or `jolt-witness`.

## Field-Inline Memory Semantics

Field-register memory is a separate execution memory from RV64 registers:

```text
RV64 registers:
  x0..x31, ordinary register witness

Field registers:
  fr0..fr15, field-register witness
```

Pure field operations touch only field registers. Bridge operations are the
only rows that intentionally connect RV64 registers and field registers.

This split is required so `jolt-witness` can build:

```text
rv64/
  ordinary register/RAM/bytecode witnesses

field_inline/
  FR register event logs
  FieldRdInc
  FieldRs1Value / FieldRs2Value / FieldRdValue
  FieldProduct / FieldInvProduct
  bridge views
  bytecode anchoring views
```

without treating pure field ops as accidental ordinary x-register activity.

## Interactions With `jolt-witness`

`jolt-witness` should consume only:

```text
JoltProgram
JoltProgramPreprocessing
TraceOutput<T: TraceSource>
TraceRow
FieldInlineBytecodeMetadata
TraceInputs/public IO/advice inputs
```

It should not consume:

```text
tracer::Cycle
Cpu
field-register runtime structs
lazy trace internals
advice tape internals
```

`jolt-program` changes are sufficient only if `jolt-witness` can build all
field-inline oracle views from those public artifacts:

```text
field_rows
field-inline bytecode metadata
FR register read/write event logs
FieldRdInc source events
FieldRs1Value / FieldRs2Value / FieldRdValue views
FieldProduct / FieldInvProduct views
bridge views
bytecode anchoring views for FieldRs1Ra / FieldRs2Ra / FieldRdWa
```

## Testing

`jolt-riscv` tests:

- no-feature builds do not expose field-inline row helpers or FR-on profiles;
- ordinary row tags, canonical names, dense indexes, and FR-off profile
  fingerprints are identical with and without `field-inline`;
- field-inline source rows are accepted only by FR-on profiles;
- field-inline final rows are accepted only by FR-on final-row profiles;
- FR-off profile fingerprints and dense indexes exclude field-inline rows;
- FR-on profile fingerprints and dense indexes include field-inline rows
  deterministically;
- field-inline canonical names and tags are stable;
- field-inline operand-shape helpers classify pure field rows, bridge rows,
  product rows, and inverse rows correctly;
- generated tracer-facing instruction lists include every field-inline row
  exactly once.

Program/preprocess tests:

- FR-off programs have no field-inline metadata.
- FR-on metadata length and active rows match expanded bytecode.
- Metadata validates active flag count, operand shape, inactive-row
  cleanliness, and field-register bounds.
- Profile fingerprints differ between FR-off and FR-on where expected.
- Field value encodings round-trip through structural serialization.

Tracer execution tests:

- tracer builds without `field-inline` reject FR-on programs before execution;
- pure field op rows have field payload and no ordinary x-register activity;
- bridge rows have both RV64 and FR payloads;
- field register reads/writes produce correct pre/post values;
- `FieldProduct` and `FieldInvProduct` payloads match native execution;
- invalid field-register operands are rejected;
- FR-off execution emits no field-inline payloads.

Witness handoff fixtures:

- ordinary RV64 row;
- pure field op row;
- x-register to FR bridge row;
- FR to x-register bridge row;
- small mixed trace with field product, inverse, and bytecode anchoring.

Regression tests:

- `jolt-program` does not depend on `tracer`;
- `jolt-program` does not depend on `jolt-witness`, `jolt-prover`, or
  `jolt-backends`;
- `jolt-witness` field-inline provider can build from
  `TraceOutput<T: TraceSource>` without accessing tracer internals.

## Milestones

1. Add field-inline source/final instruction/profile vocabulary in
   `jolt-riscv` without changing tracer execution or prover code.
2. Add `FieldInlineBytecodeMetadata` to expansion/preprocessing and validate
   FR-on/FR-off behavior in `jolt-program`.
3. Extend `TraceRow` with `Rv64TraceData` plus optional
   `FieldInlineTraceData`.
4. Implement tracer FR register file and field operation execution.
5. Adapt tracer execution output into the new `jolt-program::execution`
   payloads.
6. Add fixture tests for pure field, bridge, and mixed traces.
7. Wire `jolt-witness::protocols::jolt_vm::field_inline` to the new artifacts.

## Review Checklist

- Field-inline instruction identities are first-class `jolt-riscv` vocabulary,
  not tracer-private opcodes.
- `jolt-riscv` field-inline metadata is structural only and contains no witness
  or prover semantics.
- Field-inline profile fingerprints include the new row identities.
- `jolt-program` remains tracer-free.
- `jolt-program` exposes structural field-inline artifacts, not protocol
  formulas.
- Field-inline metadata is absent in FR-off mode.
- Field-inline trace payload is absent in FR-off mode.
- Pure field ops do not create ordinary x-register witness activity.
- Bridge ops explicitly expose both RV64 and FR sides.
- Field values use canonical encoding and do not require `jolt-program` to be
  generic over `F`.
- `tracer` owns concrete field execution and field-register state.
- `jolt-witness` can build field-inline witness data without tracer internals.

## References

- [Field inline protocol spec](./field-inline-protocol.md)
- [`jolt-witness` crate spec](./jolt-witness-crate.md)
- [`jolt-prover` model crate spec](./jolt-prover-model-crate.md)
- [`jolt-program` crate spec](./bytecode-expansion-crate.md)
- [Source/Jolt instruction split](./source-jolt-instruction-split.md)
- [Compiler-native bytecode expansion](./compiler-native-bytecode-expansion.md)
- `crates/jolt-program/src/execution/trace.rs`
- `crates/jolt-program/src/expand/`
- `crates/jolt-program/src/preprocess/`
- `tracer/src/execution_backend.rs`
- `tracer/src/emulator/cpu.rs`
