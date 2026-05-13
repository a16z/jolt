# Spec: Source/Jolt Instruction Split

| Field       | Value |
|-------------|-------|
| Author(s)   | Quang Dao |
| Created     | 2026-05-12 |
| Status      | proposed |
| PR          | [#1522](https://github.com/a16z/jolt/pull/1522) |

## Summary

PR #1518 moved bytecode expansion into `jolt-program`, but it deliberately kept
one broad row type: decoded source rows and expanded Jolt bytecode rows both
flowed through `NormalizedInstruction { instruction_kind: JoltInstructionKind }`.
That kept the code working, but the type names hid an important phase boundary.
This PR should make that boundary real: decoded program instructions are source
instructions, expansion recipes consume source instructions, and
bytecode/preprocessing/tracing/proof rows are Jolt instructions. The initial
implementation slices rename the final row payload to `JoltInstructionRow`, rename the
typed final-row view to `JoltInstruction`, add a decoded `SourceInstructionRow` payload,
and cut the decode/expand boundary over so source rows flow in and final rows
flow out. The remaining cutover should replace the broad mirrored kind tags
with typed source/final row enums and remove source-only rows from the final
universe.

The same cutover should clean up registered inlines. Inline opcodes are source
program opcodes identified by `(opcode, funct3, funct7)`; they are not final
Jolt bytecode rows. The inline provider contract should therefore accept source
inline rows and return final Jolt rows, without routing through the old fake
final `JoltInstructionKind::Inline` tag.

This split should also prepare Jolt for a more modular instruction world without
turning this PR into a general profile system or a single all-knowing instruction
macro. The near-term requirement is to create universal source/final instruction
enums whose variants carry the existing marker structs, then let each crate
decorate those marker structs with the facts it owns. Profiles should work like
an MLIR conversion target: the operation universe is stable, while each selected
profile defines which source rows are accepted and which final rows are legal
after expansion. `jolt-program` should own decode and expansion facts. `tracer`
should own execution semantics. Lookup/proving crates should own lookup-table
and circuit metadata. The final form of this PR should not be broad mirrored
bare `SourceInstructionKind` / `JoltInstructionKind` tag enums, and it should not
move downstream proving-system details into `jolt-riscv`.

## Intent

### Goal

Introduce explicit source/final instruction row enums plus a profile legality
layer across decode, expansion, tracer conversion, bytecode preprocessing, and
inline expansion:

- universal instruction enums: the source of truth for shipped source/final row
  identities, canonical operation names, compact binary tags, and
  enum-to-marker-struct dispatch.
- profile legality: the source of truth for which source instructions decode
  under a selected profile and which final rows are legal after expansion.
- profile-local dense indexes: generated indexes used by profile-specific
  tables only, never the persistent identity of an instruction.
- crate-local decorations: the source of truth for crate-specific behavior and
  metadata such as decode encodings, expansion dispatch, tracer execution,
  lookup-table routing, circuit flags, and instruction flags.
- `SourceInstruction<T = SourceInstructionRow>`: decoded source-program instruction enum,
  with variants such as `ADD(Add<T>)`, `ADDW(AddW<T>)`, and
  `Inline(Inline<T>)`.
- `SourceInstructionRow`: decoded source row payload, including address, operands,
  compression metadata, and inline dispatch metadata when applicable.
- `JoltInstruction<T = JoltInstructionRow>`: final expanded bytecode/proof/tracer
  instruction enum, with variants such as `ADD(Add<T>)`, `LD(Ld<T>)`, and
  `VirtualSignExtendWord(VirtualSignExtendWord<T>)`.
- `JoltInstructionRow`: final row payload, including operands, address,
  virtual-sequence metadata, and compression-tail metadata.

The old `NormalizedInstruction` row should not remain as a compatibility shim.
If a temporary name is needed while editing, it must be removed before the PR is
ready for review.

The concrete `SourceInstruction<T>` and `JoltInstruction<T>` types should remain
closed Rust enums. That is useful for serialization, match exhaustiveness, static
dispatch, and proof performance. Unlike profile-specific generated enums, these
enums should represent Jolt's shipped operation universe. Profiles then provide
explicit positive legality checks, in the same spirit as MLIR's separation
between operation definitions and conversion-target legality.

Profile support is part of the main goal, not a stretch goal. The PR should
leave source/final row types as phase-specific universal operation enums, with
profile legality deciding what is accepted for a given compiled configuration.

### Invariants

- Decoding preserves all currently supported RV64 and Jolt custom source
  opcodes, including registered inline dispatch metadata.
- Source and final instruction universes are explicit and separate. Adding a
  source opcode should not automatically add a final Jolt bytecode row, and
  adding a target-only virtual row should not automatically make it decodable
  from guest program bytes.
- Instruction identity is not Rust declaration order. Each source/final row
  identity has a canonical operation name such as `rv64.add` or
  `jolt.virtual.sign_extend_word`. Compact numeric tags are a binary encoding
  of those names for serialization, fixtures, and other persistent cross-crate
  references. Profile-local dense indexes may be regenerated for selected
  legality sets, but they must be derived from canonical names/tags and tied to
  the selected profile/catalog fingerprint.
- Source/final enum variants carry the marker structs directly, e.g.
  `SourceInstruction::ADD(Add<SourceInstructionRow>)` and
  `JoltInstruction::VirtualSignExtendWord(VirtualSignExtendWord<JoltInstructionRow>)`.
  This makes enum dispatch delegate naturally to marker-struct impls without a
  separate variant-to-struct lookup table.
- Crate-specific facts are not centralized in `jolt-riscv`: they are declared
  by the crate that owns the behavior and are keyed by the same marker structs
  or source/final row enum variants.
- The current Jolt source profile is explicit in code. The target legality set
  is computed from the selected source extensions plus selected inline
  extensions: it is the positive set of final rows that can be emitted by those
  expansions.
- Expansion behavior is unchanged relative to `main` after PR #1518 for the
  representative checked fixture corpus. The durable guard is a compact
  canonical-row baseline in `jolt-eval` that hashes the serialized
  `SourceInstruction<SourceInstructionRow>` corpus and final `JoltInstruction<JoltInstructionRow>`
  expansion stream after normalizing intentional type-name changes.
- Final Jolt bytecode cannot contain source-only opcodes:
  `Inline`, `ADDW`, `LW`, `SW`, AMOs, traps, CSR rows, shifts, DIV/REM, advice
  source loads, and other source-only expansion inputs must be rejected before
  preprocessing.
- `JoltInstruction<T>` contains only final-row variants, so it must not contain
  `Inline`, `ADDW`, `LW`, `SW`, or other source-only expansion inputs. Profile
  target legality is a separate positive check over those final variants.
- Source and final rows carry different semantic metadata:
  source rows know decode/inline identity; final rows know virtual-sequence
  position.
- `rd = x0` rewriting remains centralized in expansion for source rows and is
  not reimplemented in tracer as an independent policy.
- Registered inline expansion remains behind a provider boundary. The provider
  may live in `tracer`, but its input must be a source inline row and its output
  must be validated `JoltInstruction` rows.
- Concrete execution semantics remain tracer-owned. The source/final enums must
  not own CPU state mutation, RAM/advice side effects, concrete cycle
  construction, or inline advice generation.
- Lookup/proving metadata remains owned by the lookup/proving side of the codebase.
  `JoltInstruction<T>` may provide stable final-row identities, but it must not
  encode lookup-table flags, circuit flags, instruction flags, or proof-system
  routing policy.
- Expansion definitions should stay readable for humans authoring and reviewing
  instruction lowerings. The refactor may change the underlying recipe and row
  types, but the call-site syntax for ordinary expansions should remain at
  least as easy to parse as the current builder style.
- The resulting instruction and expansion code should remain extraction
  friendly. The PR does not need to make Hax or Aeneas fully compile the
  extracted output today, but it should avoid Rust patterns that are inherently
  difficult to translate into proof-oriented languages.
- `jolt-program::expand` remains independent of tracer CPU state, advice tapes,
  concrete tracer cycles, PCS/prover code, and ELF parsing.
- Prover/verifier behavior is unchanged. Bytecode preprocessing, PC mapping,
  instruction flags, lookup-table routing, and trace witness generation must
  see the same final Jolt rows as before.
- Adding a future instruction must not renumber existing serialized source or
  final row identities. If a new row changes a profile-local dense index used by
  preprocessing or proving tables, the corresponding profile/catalog
  fingerprint must change so stale artifacts are rejected rather than silently
  reused.

### Non-Goals

- Do not redesign lookup-table metadata or reintroduce `LookupInstructionKind`.
  The typed lookup-backed view should be the final-row `JoltInstruction` enum
  itself, not a second parallel instruction enum.
- Do not move lookup-table flags, circuit flags, instruction flags, or other
  proving-system metadata into `jolt-riscv`.
- Do not change RISC-V or Jolt instruction semantics.
- Do not change the registered inline algorithms themselves.
- Do not require Hax/Aeneas extraction to compile in this PR. Current extraction
  tools have temporary limitations, and this PR should not contort otherwise
  good Rust APIs around those limitations.
- Do not introduce deprecated aliases, conversion shims, or dual public APIs.
  This is a full cutover.
- Do not make `jolt-program` depend on `tracer`.
- Do not implement downstream user-authored extension profiles in this PR. The
  shipped source profiles must be explicit and compile-time selected, but
  third-party profile composition can remain follow-up work.

## Evaluation

### Acceptance Criteria

- [x] `NormalizedInstruction` is removed or fully renamed into one of the two
      phase-specific types, with no compatibility alias.
- [x] `jolt-program::image::decode_instruction` returns
      `SourceInstruction<SourceInstructionRow>`.
- [x] `jolt-program::expand` accepts `SourceInstruction` at public boundaries
      and returns typed final instructions. The current implementation spells
      the final payload type as concrete `Vec<JoltInstruction>`; if the final
      enum is later made generic, this becomes `Vec<JoltInstruction<JoltInstructionRow>>`.
- [x] Recursive expansion internally distinguishes source helper dispatch from
      direct target-row emission: helper recursion is keyed by
      `SourceInstructionKind`, while direct final emission is keyed by
      `JoltInstructionKind`. Source-only recipe builders receive source-row
      context, not synthetic final `JoltInstructionRow` inputs.
- [x] `JoltInstruction<T>` no longer has `Inline` and contains only universal
      shipped final-row variants. Profile-specific target legality is checked by
      a positive computed target legality closure rather than by changing the
      enum shape.
- [x] `SourceInstruction<T>` contains decoded RV64 source opcodes, Jolt custom
      source opcodes, and one source-only `Inline(Inline<T>)` variant.
      Individual registered inline opcodes are represented by `SourceInlineKey`,
      not by one enum variant per inline package entry.
- [x] Source and final typed row universes are separate closed enums whose
      variants carry marker structs, e.g. `ADD(Add<T>)`, and are not generated
      differently for each profile. The legacy bare tag enums remain as compact
      row identities in this PR, but target legality is no longer derived from
      treating that broad tag enum as the final typed universe.
- [x] `SourceInstructionRow` does not carry a duplicate source kind field. The
      `SourceInstruction` enum variant is the source identity, and the row
      payload carries only row data.
- [x] Canonical operation names and stable `u16` Jolt tags are explicit for the
      universal source/final enums supported by this catalog. Serialization does
      not depend on Rust enum declaration order, generated display order, or
      which profile is selected.
- [x] Any dense instruction indexes used by profile-specific preprocessing,
      lookup, or proving tables are generated from compact tags for that
      selected profile and are not used as persistent instruction identity.
- [x] Decode metadata and operand parsing are declared in `jolt-program`, keyed
      by `SourceInstructionKind`, not duplicated as an unrelated target-row
      opcode list.
- [x] Source-only expansion dispatch is declared/generated in `jolt-program`,
      keyed by `SourceInstructionKind`, and does not live in `jolt-riscv`.
- [x] Lookup-table routing, circuit flags, and instruction flags remain owned by
      the lookup/proving crates. They are not fields in the `jolt-riscv`
      row enum definitions or in a mega `jolt_instruction!` declaration.
- [x] Tracer still owns concrete execution semantics. No row enum macro in
      `jolt-riscv` mutates CPU/RAM/advice state or constructs concrete tracer
      cycles.
- [x] Broad mirrored source/target tag enums are no longer the authoritative
      final implementation shape. Closed typed enums remain, with
      marker-struct payloads and profile legality layered on top; the remaining
      broad bare `JoltInstructionKind` tag enum is a compact identity bridge for
      existing row serialization and call sites.
- [x] The profile/extension layer uses the concrete names `SourceExtension`,
      `JoltTargetExtension`, `InlineExtension`, and `JoltInstructionProfile`.
- [x] The default profile corresponds to the current supported
      RV64IMAC Jolt behavior, with room for shipped source presets such as
      `RV64IM_JOLT`, `RV64IMAC_JOLT`, and `RV64IMAC_JOLT_ALL_INLINES`.
      Future profiles can be added by selecting source/inline extensions and
      recomputing positive legality sets without changing the row enum shapes.
- [x] Inline dispatch metadata is represented directly on `SourceInstruction`
      as a `SourceInlineKey` payload, not packed into `NormalizedOperands::imm`.
- [x] `InlineExpansionProvider` accepts source inline data and returns final
      `JoltInstruction` rows.
- [x] Concrete expansion files remain ergonomic for human authors: common
      lowering code should read like a small instruction sequence, not like
      serialized grammar data or generated tables.
- [x] The refactor does not introduce extraction-hostile Rust patterns in the
      instruction row enums, row types, or expansion pipeline.
- [x] Tracer concrete `Instruction`/`Cycle` APIs execute final Jolt rows, while
      decoded source instructions convert through the expansion path before
      trace execution.
- [x] No verifier-facing crate imports tracer just to name source or final row
      types.
- [x] Existing expansion fixture/hash tests still pass against `main`'s
      post-#1518 behavior, and any fixture hash regeneration is backed by a
      structural source-to-final stream equivalence check rather than reviewer
      trust in changed serialization bytes.
- [x] `cargo tree -p jolt-program` has no `tracer` dependency.

### Testing Strategy

Run the usual validation stack:

```bash
cargo fmt -q
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo nextest run -p jolt-program --cargo-quiet
cargo nextest run -p tracer --cargo-quiet --features test-utils
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

Add or update tests for:

- decoded inline rows preserving `(opcode, funct3, funct7)` without `imm`
  packing;
- provider-free expansion rejecting source inline rows;
- registered inline provider returning only target-legal `JoltInstruction`
  rows;
- final bytecode rejecting every source-only kind at preprocessing;
- canonical-name and compact-tag tests proving existing identities do not change
  when generated enum order changes or new instructions are appended;
- profile dense-index tests proving unsupported rows have no index and supported
  rows map to a contiguous profile-local range;
- profile/catalog fingerprint tests proving a changed legal row set changes the
  artifact identity used by preprocessing/proving;
- tracer execution of expanded rows after the source/final cutover;
- fixture parity for the existing representative source corpus;
- a `jolt-eval` invariant named `source_to_jolt_expansion_equivalence` that
  compares the new source-to-final expansion stream against the pre-refactor
  normalized expansion semantics modulo intentional row type and compact-tag
  renames.

### Performance

This should be a type-boundary refactor with no intended runtime regression.
Avoid extra allocation in the hot trace path: source rows should be expanded
once into final rows, and final rows should be reused by preprocessing/tracing
where the current code already does so. If the implementation introduces a
conversion allocation around tracer execution, remove it or document why it is
outside hot loops before review.

### Readability

The implementation should preserve the readability win from PR #1518: changing
the symbolic expansion plumbing must not make ordinary expansion files harder
to read. It is acceptable to rename types and builder methods when that makes
the source/final boundary clearer, but adding a new opcode or auditing an
existing lowering should still look like writing a short instruction sequence.

Prefer code shaped like this:

```rust
let mut asm = ExpansionBuilder::new(instruction);

asm.emit_r(
    Sub,
    rd(instruction)?,
    rs1(instruction)?,
    rs2(instruction)?,
);
asm.emit_i(
    VirtualSignExtendWord,
    rd(instruction)?,
    rd(instruction)?,
    0,
);

asm.finalize()
```

Avoid exposing representation-oriented recipe construction at ordinary call
sites unless it is genuinely clearer for that instruction:

```rust
ExpandedInstructionSequence::new(
    instruction,
    [
        ExpansionOp::Emit(RowTemplate::r(...)),
        ExpansionOp::Emit(RowTemplate::i(...)),
    ],
)
```

The row-enum/profile work has the same constraint. Metadata can become more
structured, but instruction authors should not need to mentally execute macro
grammar to understand whether an opcode is source-only, target-only,
lookup-backed, side-effecting, or part of a default profile.

### Expansion Equivalence

The expansion fixture hash may change when `NormalizedInstruction` is replaced
by phase-specific row types, even if the emitted program is semantically
unchanged. Regenerating that fixture is therefore not itself sufficient evidence.
The durable guard should live in `jolt-eval` as
`source_to_jolt_expansion_equivalence`: a compact canonical-row baseline over
the representative source corpus and its emitted final-row expansion stream.

The invariant should serialize enough row structure to fail on differences in
source opcode identity, final opcode identity, operands, addresses, virtual
sequence metadata, and compression-tail metadata. It does not need to keep the
entire pre-refactor byte stream alive as a compatibility layer; intentional
type-name changes, compact-tag changes, and the removal of the old
`JoltInstructionKind::Inline` final-row tag should be normalized into the new
canonical row representation before hashing.

### Extraction Friendliness

This PR should keep the code on a path toward clean extraction into theorem
prover targets such as Lean. That does not mean optimizing the Rust code around
today's Hax/Aeneas limitations, especially when those limitations are likely to
move. It does mean avoiding choices that are structurally unfriendly to
extraction.

Prefer:

- plain data types with explicit fields over packed, phase-dependent encodings;
- small enums and structs with local invariants over trait-object or callback
  heavy control flow;
- total conversion functions that return typed errors over implicit panics;
- simple iterator/loop structure where it keeps the code just as readable;
- profile/legalization metadata that can be inspected as data, not only through macro
  expansion side effects.

Avoid introducing:

- `dyn` dispatch or closure-heavy APIs in the core instruction/expansion path;
- hidden global state for profile/legalization decisions;
- unsafe code in row enum, decode, or expansion plumbing;
- encodings where a field has unrelated meanings depending on an instruction
  phase;
- procedural macro magic that makes the generated instruction set difficult to
  audit or mirror in extracted code.

Extraction-friendliness is subordinate to correctness, performance, and idiomatic
Rust, but it should influence tie-breakers when two designs are otherwise
comparable.

## Design

### Architecture

The target data flow is:

```text
ELF / decoded word
  -> SourceInstruction
  -> jolt-program::expand
  -> Vec<JoltInstruction>
  -> bytecode preprocessing / tracer execution / proof lookup metadata
```

`jolt-riscv` owns shared instruction row identity because it is the lowest common
crate shared by decode, expansion, tracer, bytecode preprocessing, and lookup
metadata. That ownership should stay deliberately small. It should provide:

- the marker structs such as `Add<T>`, `AddW<T>`, and
  `VirtualSignExtendWord<T>`;
- the universal source/final row enums whose variants carry those marker
  structs;
- stable enum variant names, canonical operation names, and compact binary tags
  used for serialization;
- row structs and profile-independent row-shape types shared across crates.

It should not provide a mega declaration that also names decode opcodes,
expansion bodies, side-effect policy, tracer execution, lookup-table routing,
circuit flags, or instruction flags. Those are real facts, but they belong to
the crates that use and test them.

The row enums should use the same pattern as the former `LookupInstruction`:
each variant wraps the associated marker struct instantiated with a row payload.
That means the enum itself owns the variant-to-struct relationship; there is no
separate mapping table to keep in sync.

```rust
pub enum SourceInstruction<T = SourceInstructionRow> {
    NoOp,
    Unimpl,
    ADD(Add<T>),
    ADDW(AddW<T>),
    LW(Lw<T>),
    SW(Sw<T>),
    Inline(Inline<T>),
    // ...all shipped source-program rows...
}

pub enum JoltInstruction<T = JoltInstructionRow> {
    NoOp,
    Unimpl,
    ADD(Add<T>),
    LD(Ld<T>),
    SD(Sd<T>),
    VirtualSignExtendWord(VirtualSignExtendWord<T>),
    VirtualHostIO(VirtualHostIO<T>),
    // ...all shipped final Jolt rows...
}
```

```rust
pub struct SourceInstructionRow {
    pub address: usize,
    pub operands: SourceOperands,
    pub inline: Option<SourceInlineKey>,
    pub is_compressed: bool,
}

pub struct SourceInlineKey {
    pub opcode: u8,
    pub funct3: u8,
    pub funct7: u8,
    pub extension: InlineExtension,
}

pub struct JoltInstructionRow {
    pub address: usize,
    pub operands: JoltOperands,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}
```

The exact operand/spec type names can change, but the ownership relationship
should not: the universal enums define what shipped row identities exist, the
selected source profile defines which source rows decode, and the target closure
defines which final rows may be emitted by this profile. Source operands may
continue to use normalized register fields for ordinary rows; inline dispatch
metadata should not be stored in an immediate field.

### Canonical Names, Tags, And Profile Indexes

Do not use Rust enum declaration order as instruction identity. Source and
final row enums should have canonical operation names that are explicit,
reviewed, and namespace-qualified. This follows the MLIR model: the durable
semantic identity is a dialect-like namespace plus an operation mnemonic, while
compact encodings are an implementation detail.

Examples:

```text
rv64.add
rv64.mul
rv64.fadd_s
jolt.virtual.sign_extend_word
jolt.field.bn254.mul
jolt.inline.sha2.compress
wasm.i32.add
```

RISC-V instruction encodings are useful decode metadata, not sufficient Jolt
row identities. A RISC-V instruction is usually identified by an encoding
pattern such as `(opcode, funct3, funct7)` or by compressed-instruction fields,
not by the 7-bit opcode alone. Several instructions share the same opcode,
compressed rows live in a different encoding space, and Jolt virtual/final-only
rows have no source-program RISC-V encoding at all. Therefore `jolt-program`
should own RISC-V encoding facts for decode, while `jolt-riscv` owns canonical
operation names for source/final row identity.

For Jolt's binary artifacts, every operation supported by the shipped Jolt
catalog has a compact stable tag. The name is the reviewable semantic identity;
the tag is the compact serialization of that identity for bytecode rows,
fixtures, preprocessing keys, and proof-adjacent tables. This tag only needs to
cover operations Jolt actually supports in the current catalog, not every
operation in every possible future source ISA. A stable `u16` tag is therefore
the preferred starting point: it keeps bytecode row serialization as compact as
today while decoupling identity from Rust enum declaration order.

Use metadata like this in the generated catalog data:

```rust
pub trait SourceOp {
    const CANONICAL_NAME: &'static str;
    const JOLT_TAG: JoltOpTag;
}

pub trait JoltOp {
    const CANONICAL_NAME: &'static str;
    const JOLT_TAG: JoltOpTag;
}

impl<T> SourceOp for Add<T> {
    const CANONICAL_NAME: &'static str = "rv64.add";
    const JOLT_TAG: JoltOpTag = JoltOpTag(0x0101);
}

impl<T> JoltOp for Add<T> {
    const CANONICAL_NAME: &'static str = "rv64.add";
    const JOLT_TAG: JoltOpTag = JoltOpTag(0x0101);
}

impl<T> SourceOp for AddW<T> {
    const CANONICAL_NAME: &'static str = "rv64.addw";
    const JOLT_TAG: JoltOpTag = JoltOpTag(0x0102);
}

impl<T> JoltOp for VirtualSignExtendWord<T> {
    const CANONICAL_NAME: &'static str = "jolt.virtual.sign_extend_word";
    const JOLT_TAG: JoltOpTag = JoltOpTag(0x8001);
}
```

The tag is allocated only for supported Jolt source/final rows. Future source
ISA operations may be known by canonical name without receiving a Jolt tag until
the selected Jolt catalog actually supports them. Numeric tag ranges may be used
as a readability aid, but they must not be the source of architectural meaning.
The robust partition is the canonical name namespace: `rv64.*`,
`jolt.virtual.*`, `jolt.field.*`, `jolt.inline.*`, `wasm.*`, and future
namespaces. If a row moves from source-only to final-only, or from an inline
helper to a first-class operation, its name and tag should change only if its
semantic identity changes.

Profile-local dense indexes are a separate concept. They may be compact and may
change when a profile's legal final-row set changes:

```rust
pub struct ProfileInstructionIndex(u16);

impl Rv64imacJolt {
    pub const fn jolt_dense_index(tag: JoltOpTag) -> Option<ProfileInstructionIndex> {
        match tag {
            tag!("rv64.add") => Some(ProfileInstructionIndex(0)),
            tag!("rv64.addi") => Some(ProfileInstructionIndex(1)),
            tag!("rv64.mul") => Some(ProfileInstructionIndex(2)),
            tag!("rv64.ld") => Some(ProfileInstructionIndex(3)),
            tag!("jolt.virtual.sign_extend_word") => Some(ProfileInstructionIndex(4)),
            _ => None,
        }
    }
}
```

Dense indexes are appropriate for profile-specific preprocessing, lookup, or
proving tables. Concrete examples include arrays of per-profile final-row
metadata, profile-specific legality bitsets, lookup-routing tables keyed by the
selected final-row set, and any proving key/preprocessing structure that wants a
contiguous `[0, profile_instruction_count)` coordinate. Dense indexes are not
appropriate for canonical identity, serialization source of truth, fixture
identity, or cross-profile references. Any artifact that relies on dense
indexes must be keyed by the selected profile/catalog fingerprint so that stale
tables cannot be reused after adding or removing legal rows.

Profiles should not change the Rust enum shape. This is intentionally closer to
MLIR than to profile-specific generated Rust APIs: operations exist in the
universe, and a conversion target/profile declares which operations are legal at
a given point in the pipeline. Selecting `RV64IM_JOLT` should therefore reject
atomic source rows during decode and exclude atomic-produced final rows from the
computed target legality set, but it should not generate a different
`SourceInstruction<T>` type from `RV64IMAC_JOLT`.

### Ownership Boundaries

The universal row enums own only cross-crate identity. Crate-local decorations
own the behavior-specific facts, using the same marker structs as join keys.

`jolt-program` owns source decoding, operand parsing, positive source/target
legality, and source-to-final expansion dispatch. Its local metadata can be a
trait, free-function table, or macro that expands to ordinary matches:

```rust
trait DecodeSpec {
    const SOURCE_EXTENSION: SourceExtension;
    const FORMAT: SourceFormat;
    const ENCODING: SourceEncoding;
}

impl DecodeSpec for Add<()> {
    const SOURCE_EXTENSION: SourceExtension = SourceExtension::Rv64I;
    const FORMAT: SourceFormat = SourceFormat::R;
    const ENCODING: SourceEncoding = SourceEncoding::R {
        opcode: 0b0110011,
        funct3: 0b000,
        funct7: 0b0000000,
    };
}

impl DecodeSpec for AddW<()> {
    const SOURCE_EXTENSION: SourceExtension = SourceExtension::Rv64I;
    const FORMAT: SourceFormat = SourceFormat::R;
    const ENCODING: SourceEncoding = SourceEncoding::R {
        opcode: 0b0111011,
        funct3: 0b000,
        funct7: 0b0000000,
    };
}
```

Expansion dispatch should be local to `jolt-program` as well:

```rust
trait SourceExpansion {
    fn expand_source(
        &self,
        allocator: &mut ExpansionAllocator,
    ) -> Result<Vec<JoltInstruction<JoltInstructionRow>>, ExpansionError>;
}

impl SourceExpansion for AddW<SourceInstructionRow> {
    fn expand_source(
        &self,
        allocator: &mut ExpansionAllocator,
    ) -> Result<Vec<JoltInstruction<JoltInstructionRow>>, ExpansionError> {
        expand_addw(self, allocator)
    }
}
```

`tracer` owns concrete execution semantics:

- CPU register and PC mutation;
- RAM and device side effects;
- advice tape interaction and inline advice generation;
- concrete `Instruction` / `Cycle` construction and execution tests;
- tracer-internal adapters from final `JoltInstruction` rows to concrete
  execution instructions.

That can be expressed with tracer-local implementations keyed by the shared
marker structs or by the final row enum. For example:

```rust
trait Execute {
    fn execute(cpu: &mut Cpu, operands: &JoltOperands) -> Cycle;
}

impl Execute for Add<()> {
    fn execute(cpu: &mut Cpu, operands: &JoltOperands) -> Cycle {
        existing_add_execute(cpu, operands)
    }
}
```

Lookup/proving code owns lookup metadata, lookup-table routing, circuit flags,
and instruction flags. Those facts may also be generated from a local macro, but
the macro should live with the lookup/proving owner, not in the universal row
enum definition:

```rust
trait LookupMetadata {
    const LOOKUP: LookupSupport;
    const CIRCUIT_FLAGS: &'static [CircuitFlag];
    const INSTRUCTION_FLAGS: &'static [InstructionFlag];
}

impl LookupMetadata for Add<()> {
    const LOOKUP: LookupSupport = LookupSupport::Instruction;
    const CIRCUIT_FLAGS: &'static [CircuitFlag] =
        &[CircuitFlag::AddOperands, CircuitFlag::WriteLookupOutputToRD];
    const INSTRUCTION_FLAGS: &'static [InstructionFlag] =
        &[InstructionFlag::LeftOperandIsRs1Value, InstructionFlag::RightOperandIsRs2Value];
}
```

Expansion bodies are also not moved into `jolt-riscv`, because `jolt-riscv`
must remain below `jolt-program` in the dependency graph. `jolt-riscv` can expose
`SourceInstruction::ADDW(AddW<SourceInstructionRow>)`. `jolt-program` says whether `ADDW`
is legal in the active profile, how it lowers, and can generate the dispatcher
that routes the `ADDW` variant through the `AddW` marker to the human-written
`expand_addw` body.

### Rows And Profiles

The current code generates both instruction-kind enums from one broad macro
list. That makes the type split mostly nominal: every source opcode also exists
as a final Jolt opcode unless a later legality check rejects it. This PR should
replace that with two universal row enums plus crate-local metadata whose
first-order facts are named deliberately and consistently.

Use these names unless implementation uncovers a concrete conflict:

```rust
pub enum SourceExtension {
    /// RV64I instruction semantics and 64-bit base encodings.
    Rv64I,
    /// RV64M multiply/divide source instructions.
    Rv64M,
    /// RV64A atomic source instructions.
    Rv64A,
    /// RVC compressed encodings. These decode into ordinary source rows.
    Rv64C,
    /// CSR source instructions currently accepted by decode.
    Zicsr,
    /// Privileged source instructions currently accepted by decode, such as MRET.
    RvPrivileged,
    /// Jolt custom guest opcodes decoded from the custom instruction space.
    JoltCustom,
    /// Registered inline source opcodes keyed by opcode/funct3/funct7.
    JoltInline,
}

pub enum JoltTargetExtension {
    /// Base integer arithmetic, comparisons, and immediate operations.
    IntegerCore,
    /// Integer multiplication operations retained as final Jolt instructions.
    IntegerMultiply,
    /// Branches, jumps, and other control-flow operations.
    ControlFlow,
    /// 64-bit load/store operations. Narrow memory operations lower into these.
    LoadStore64,
    /// Advice-producing and advice-consuming virtual operations.
    Advice,
    /// Host I/O virtual operations.
    HostIO,
    /// Virtual assertions used by expansion and inlines.
    VirtualAssertions,
    /// Virtual arithmetic helpers used by division, word ops, and carries.
    VirtualArithmetic,
    /// Virtual shift and rotate helpers used by source lowering.
    VirtualShifts,
    /// Bit-manipulation helpers used mainly by custom ops and crypto inlines.
    BitManipulation,
}

pub enum InlineExtension {
    Sha2,
    Keccak256,
    Blake2,
    Blake3,
    BigInt256,
    Secp256k1,
    Grumpkin,
    P256,
}
```

The durable requirement is that source decode support, target bytecode legality,
inline registration support, lookup-table support, side-effect metadata, circuit
flags, and instruction flags are separate compile-time facts with separate
owners. The PR should not add a separate `InstructionPhase` enum unless the
implementation needs it internally; phase is already determined by whether a row
identity appears in `SourceInstruction<T>`, `JoltInstruction<T>`, or both.
Closed universal enums are acceptable and desired, but the source of truth for
profile legality must be explicit profile metadata plus crate-local decorations,
not a monolithic enum and not `!is_source_only`.

The current default source profile should include:

- `SourceExtension::Rv64I`: base integer, 64-bit loads/stores, branches, jumps,
  `FENCE`, `ECALL`, and `EBREAK`;
- `SourceExtension::Rv64M`: multiply/divide/remainder source opcodes,
  including W-suffix forms;
- `SourceExtension::Rv64A`: LR/SC and AMO source opcodes;
- `SourceExtension::Rv64C`: compressed encodings, which uncompress into
  ordinary source instructions rather than adding separate source kinds;
- `SourceExtension::Zicsr`: `CSRRW` and `CSRRS`;
- `SourceExtension::RvPrivileged`: `MRET`;
- `SourceExtension::JoltCustom`: custom decoded source rows such as
  `VirtualRev8W`, `VirtualAssertEQ`, `VirtualHostIO`, `AdviceLB/LH/LW/LD`, and
  `VirtualAdviceLen`;
- `SourceExtension::JoltInline`: source rows whose dispatch payload is keyed by
  `(opcode, funct3, funct7)`.

The computed target legality closure for the current default profile should
include these `JoltTargetExtension` families:

- `JoltTargetExtension::IntegerCore`: final instructions such as `ADD`, `ADDI`,
  `SUB`, `LUI`, `AUIPC`, `AND`, `ANDI`, `OR`, `ORI`, `XOR`, `XORI`, `SLT`,
  `SLTI`, `SLTU`, and `SLTIU`;
- `JoltTargetExtension::IntegerMultiply`: final multiply instructions such as
  `MUL` and `MULHU`; source-only multiply/divide rows lower before final
  bytecode;
- `JoltTargetExtension::ControlFlow`: final branch/jump instructions such as
  `BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU`, `JAL`, `JALR`, and `FENCE`;
- `JoltTargetExtension::LoadStore64`: final `LD` and `SD` instructions;
- `JoltTargetExtension::Advice`: `VirtualAdvice`, `VirtualAdviceLen`, and
  `VirtualAdviceLoad`;
- `JoltTargetExtension::HostIO`: `VirtualHostIO`;
- `JoltTargetExtension::VirtualAssertions`: `VirtualAssertEQ`,
  `VirtualAssertLTE`, `VirtualAssertValidDiv0`,
  `VirtualAssertValidUnsignedRemainder`, `VirtualAssertMulUNoOverflow`,
  `VirtualAssertWordAlignment`, and `VirtualAssertHalfwordAlignment`;
- `JoltTargetExtension::VirtualArithmetic`: `VirtualMULI`,
  `VirtualMovsign`, `VirtualPow2`, `VirtualPow2I`, `VirtualPow2W`,
  `VirtualPow2IW`, `VirtualChangeDivisor`, `VirtualChangeDivisorW`,
  `VirtualSignExtendWord`, and `VirtualZeroExtendWord`;
- `JoltTargetExtension::VirtualShifts`: `VirtualSRL`, `VirtualSRLI`,
  `VirtualSRA`, `VirtualSRAI`, `VirtualShiftRightBitmask`,
  `VirtualShiftRightBitmaskI`, `VirtualROTRI`, and `VirtualROTRIW`;
- `JoltTargetExtension::BitManipulation`: `ANDN`, `VirtualRev8W`, and the
  `VirtualXORROT*` / `VirtualXORROTW*` rows used by crypto inlines.

Source sentinel rows `NoOp` and `Unimpl` are always available and should not be
modeled as extension-gated capabilities. Final bytecode treats `NoOp` as the
only target-legal sentinel row; `Unimpl` is decode/source-side only and must be
rejected before preprocessing.

The inline profile metadata should use the registered inline package names as
first-class entries, not treat every inline as one anonymous extension. This
does not mean `SourceInstruction<T>` gets one variant per inline operation.
Source decoding uses one `SourceInstruction::Inline(Inline<SourceInstructionRow>)` row plus
a `SourceInlineKey` payload; the `InlineExtension` profile gates which registered
`(opcode, funct3, funct7)` keys are accepted and which provider is allowed to
expand them. Current entries are:

- `InlineExtension::Sha2`: `SHA256_INLINE`, `SHA256_INIT_INLINE`;
- `InlineExtension::Keccak256`: `KECCAK256_INLINE`;
- `InlineExtension::Blake2`: `BLAKE2_INLINE`;
- `InlineExtension::Blake3`: `BLAKE3_INLINE`, `BLAKE3_KEYED64_INLINE`;
- `InlineExtension::BigInt256`: `BIGINT256_MUL_INLINE`;
- `InlineExtension::Secp256k1`: the `SECP256K1_*` inline family;
- `InlineExtension::Grumpkin`: `GRUMPKIN_DIVQ_ADV`, `GRUMPKIN_DIVR_ADV`;
- `InlineExtension::P256`: the `P256_*` inline family.

`Inline` itself is source-only and must remain illegal in finalized bytecode.
Registered inline providers may emit ordinary target rows plus virtual helper
rows, but provider output must be validated against the computed target legality
closure before preprocessing.

The near-term profile shape is source-driven:

```rust
pub struct JoltInstructionProfile {
    pub source_extensions: &'static [SourceExtension],
    pub inline_extensions: &'static [InlineExtension],
}

pub const RV64IMAC_JOLT: JoltInstructionProfile = JoltInstructionProfile {
    source_extensions: &[
        SourceExtension::Rv64I,
        SourceExtension::Rv64M,
        SourceExtension::Rv64A,
        SourceExtension::Rv64C,
        SourceExtension::Zicsr,
        SourceExtension::RvPrivileged,
        SourceExtension::JoltCustom,
        SourceExtension::JoltInline,
    ],
    inline_extensions: &[],
};

pub const RV64IMAC_JOLT_ALL_INLINES: JoltInstructionProfile = JoltInstructionProfile {
    source_extensions: RV64IMAC_JOLT.source_extensions,
    inline_extensions: &[
        InlineExtension::Sha2,
        InlineExtension::Keccak256,
        InlineExtension::Blake2,
        InlineExtension::Blake3,
        InlineExtension::BigInt256,
        InlineExtension::Secp256k1,
        InlineExtension::Grumpkin,
        InlineExtension::P256,
    ],
};
```

`JoltTargetExtension` remains useful profile metadata: it groups final rows into
semantic families such as integer core, host I/O, virtual arithmetic, and crypto
helper rows. It should not be a second hand-selected profile axis in this PR,
and it should not encode lookup-table or proving-system policy. Given a selected
`JoltInstructionProfile`, `jolt-program` derives the final target legality set
from:

- direct final rows emitted by source instructions enabled by
  `source_extensions`;
- recursive helper rows reachable from those expansions;
- final rows emitted by enabled `inline_extensions`;
- the target-legal `NoOp` sentinel row.

This makes `RV64IM_JOLT` naturally produce a smaller legal final closure than
`RV64IMAC_JOLT` when atomics/compressed-only source paths are disabled, without
requiring callers to maintain a parallel target list. The Rust enum remains the
same universal shipped final-row enum; the selected profile changes which rows
decode and which final rows pass preprocessing legality. Cross-profile proof
artifact compatibility is not a goal of this PR: circuit/preprocessing keys are
tied to the selected compile-time profile, compact tags, dense-index maps, and
legality sets.

Reserve these shipped preset names:

- `RV64IM_JOLT`: base RV64I+M profile without atomics or compressed encodings;
- `RV64IMAC_JOLT`: current base RV64IMAC source profile with the inline source
  mechanism available;
- `RV64IMAC_JOLT_ALL_INLINES`: current workspace-wide profile with all listed
  `InlineExtension` packages enabled.

Profile selection should be explicit and tied to shipped profile constants, not
runtime plugin loading. This PR threads a `JoltInstructionProfile` value through
decode, expansion, sequence stamping, and bytecode preprocessing boundaries so
those phases do not silently read `RV64IMAC_JOLT` as hidden global policy.
Top-level callers that want the current default behavior pass `RV64IMAC_JOLT`;
callers that expand registered inline packages pass a profile whose
`inline_extensions` contain the relevant package entries, such as
`RV64IMAC_JOLT_ALL_INLINES`. The selected profile affects legality tables and
closure checks, not the Rust enum shape.

Inline inventory registration remains link-time, but availability is
profile-checked. If a workspace links `jolt-inlines-sha2` while the selected
profile does not include `InlineExtension::Sha2`, decode/expansion must reject
the registered key before it can enter finalized bytecode. The PR does not need
to make such a crate fail to compile.

For this PR, the implementation must include the universal source/final row
enums, crate-local metadata, and shipped current source profiles above. It does
not need arbitrary third-party profile composition, but decode legality,
final-bytecode legality, and inline availability should all flow from
profile/legalization facts and crate-owned decorations rather than from a broad
shared instruction-kind list.

Lookup support should not introduce `LookupInstructionKind` or preserve a
parallel `LookupInstruction` enum. Keep the existing boundary in spirit:
`JoltInstruction<T>` is the typed lookup/proof-facing view of supported final
rows, and `LookupTableKind` identifies concrete lookup tables in
`jolt-lookup-tables`.
A future `LookupTableProfile` may name the compile-time table set for a
selected proof profile, but it should be table-oriented, not another
instruction-kind enum.

Expansion should expose:

```rust
pub fn expand_instruction(
    instruction: &SourceInstruction<SourceInstructionRow>,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<JoltInstruction>, ExpansionError>;

pub trait InlineExpansionProvider {
    fn expand_inline(
        &mut self,
        instruction: &SourceInstruction<SourceInstructionRow>,
        allocator: &mut ExpansionAllocator,
    ) -> Result<Vec<JoltInstruction>, ExpansionError>;
}
```

Inside `jolt-program::expand`, the builder should keep two operations:

- `emit_*`: append target-legal `JoltInstruction<JoltInstructionRow>` rows directly.
- `expand_*`: recursively lower helper `SourceInstruction<SourceInstructionRow>` rows.

This models the current semantics more accurately than pretending every
recursive helper is pure RISC-V or every emitted row is already final.

The source-only expansion dispatcher should be generated from `jolt-program`
metadata or from a small `jolt-program` macro keyed by the instruction marker
structs, not maintained as an unrelated final-kind match list. The
generated shape should read conceptually like:

```rust
match instruction {
    SourceInstruction::ADD(add) => emit_direct(add),
    SourceInstruction::ADDW(addw) => addw.expand_source(allocator),
    SourceInstruction::LW(lw) => lw.expand_source(allocator),
    SourceInstruction::Inline(inline) => inline_provider.expand_inline(inline, allocator),
    _ => Err(ExpansionError::UnsupportedInstruction),
}
```

That does not require moving `expand_addw` into `jolt-riscv`; it requires the
dispatch edge to be generated from `jolt-program`'s local `AddW` expansion
metadata instead of copied into a separate list that can drift from the source
row enum.

Tracer should become phase-aware:

- decode-facing APIs consume or produce `SourceInstruction`;
- expanded trace execution consumes `JoltInstruction` or concrete tracer
  `Instruction` built from `JoltInstruction`;
- execution semantics remain implemented on tracer concrete instruction types,
  not in `jolt-riscv` row enum macros;
- `RISCVInstruction` conversions should not require
  `From<SourceInstruction> + Into<SourceInstruction> + From<JoltInstruction> +
  Into<JoltInstruction>` on the same trait;
- inline source rows should never be reconstructed from a final row.

Registered inlines can still be implemented in `tracer` for this PR, because
they need inventory registration and advice-generation hooks. The important
cleanup is the boundary: `TracerInlineExpansionProvider` should no longer parse
a fake final `Inline` row, and `jolt-inlines-sdk::InlineOp` should either return
final `JoltInstruction` rows directly or use an assembler whose output is final
`JoltInstruction` rows. If moving every inline builder off tracer's
`Instruction` enum proves too large, keep that as an internal tracer adapter
only and ensure the public provider contract is already final-row typed.

### Alternatives Considered

Keeping `SourceInstructionKind` and `JoltInstructionKind` as mirror bare tag
enums is the smallest change, but it keeps the reviewer concern intact: names
improve, but the compiler cannot enforce phase boundaries and enum dispatch
still needs a separate mapping back to marker structs.

Generating profile-specific source/final enum shapes would maximize type-level
profile safety: if `RV64IM_JOLT` excludes atomics, the atomic variants could
literally be absent. That is not the preferred shape for this PR. It makes
downstream exhaustive matches profile-dependent and is less aligned with MLIR,
where operations exist in a stable universe and a conversion target decides
which operations are legal at a particular stage. Jolt should instead keep
stable universal shipped source/final enums and make profile legality explicit.

Putting every fact into an expanded `jolt_instruction!` declaration would remove
some match duplication, but it would also make `jolt-riscv` aware of details it
should not own: decode quirks, expansion bodies, tracer state mutation,
lookup-table routing, circuit flags, and instruction flags. That is the wrong
dependency direction. This PR should instead use universal marker-carrying
source/final row enums plus crate-local decorations.

Adding a separate `InstructionCatalogEntry` table beside the existing marker
structs and then filling it with every fact has the same problem in a different
shape. A separate table is not needed for the enum variant -> marker struct
mapping if the variants carry marker structs directly. A small macro/list may
still be used to avoid hand-writing both enums and stable serialization matches,
but it should generate `ADD(Add<T>)`-style variants rather than a separate
bare-kind enum plus a parallel mapping table.

Making the enums fully dynamic, MLIR-style operation IDs would maximize
extensibility but is the wrong first move for Jolt. Instruction identity feeds
serialization, bytecode preprocessing, static flag dispatch, lookup routing,
and proof circuits. Closed universal enums are a better fit: the operation
universe can be MLIR-like and stable while the compiled prover remains
specialized by explicit profile legality sets.

Using Cargo features alone as the profile system would also be too coarse.
Features can select profiles or extension groups, but they should not be the
only place where instruction identity or legality lives. Decode membership,
target legality, lookup table routing, side effects, and lowering behavior
should be queryable through their owning crates, keyed by shared instruction
identities.

Making only `jolt-program` phase-aware is also insufficient. Tracer currently
converts concrete `Instruction` through the same normalized row and also handles
some expansion-ish behavior during tracing. If tracer remains untyped by phase,
the next cleanup will immediately need another cross-crate churn.

Removing tracer's concrete `Instruction` enum entirely is too broad. The enum
still owns execution behavior, cycle construction, and tests. This PR should
separate source/final row APIs while preserving tracer's concrete execution
model where it is useful.

Moving concrete execution semantics into `jolt-riscv` would also cross the wrong
boundary. `jolt-riscv` can describe stable identity, source/final membership,
and shared row shape. It should not know how to mutate a tracer `Cpu`, update
RAM/device state, generate advice, or build concrete tracer cycles.

Moving lookup/proving metadata into `jolt-riscv` has the same ownership smell.
`jolt-riscv` should be blissfully unaware of the proving system's lookup-table
flags, circuit flags, and instruction flags. Those facts should be declared in
the lookup/proving owner and keyed by `JoltInstruction<T>` variants or marker
structs.

Moving all registered inline implementations into `jolt-program` would make the
provider-free crate depend on inventory/advice/tracer concerns. That reverses
the dependency boundary from #1518. A provider contract with final-row output is
the smaller clean architecture.

## Documentation

Update `specs/compiler-native-bytecode-expansion.md` after implementation to
mark the deferred source/final split complete and link to this spec. No Jolt
book update is required unless public SDK APIs expose the new names directly.

## Execution

Current implementation status:

- `NormalizedInstruction` has been removed in favor of phase-specific row
  payloads.
- `jolt-program` decode now produces `SourceInstruction<SourceInstructionRow>` values.
- `jolt-program::expand` public APIs now consume `SourceInstruction` values and
  return typed final `Vec<JoltInstruction>` values. `JoltProgram`,
  `jolt-core::{guest,host}` decode handoffs, and tracer conversion materialize
  `JoltInstructionRow` explicitly with `JoltInstructionRow::from`.
- `InlineExpansionProvider` now receives a source instruction plus the selected
  profile and returns typed final `JoltInstruction` rows.
- `SourceInstructionRow` now carries `SourceInlineKey { opcode, funct3, funct7 }` metadata
  directly, so tracer decode and registered inline expansion no longer recover
  source inline identity by routing through a fake final `JoltInstructionRow::Inline`.
- `SourceInstructionRow` no longer carries a duplicate `SourceInstructionKind`; the
  `SourceInstruction` enum variant is the source instruction identity, and row
  payloads carry only address, operand, inline, and compression data.
- `SourceInstructionKind` tag serialization is now direct rather than routed
  through `SourceInstructionKind::jolt_kind()`, so future source-only identities
  do not need a final-row identity just to serialize.
- `SourceInstruction<T>` variants now carry marker structs such as
  `ADD(Add<T>)` and `Inline(Inline<T>)` rather than raw `T` payloads.
- `JoltInstruction` now preserves row payload for `Noop`, exposes row
  materialization through `From<JoltInstruction> for JoltInstructionRow`, and normalizes
  the materialized row kind from the typed enum variant.
- `JoltInstruction<T = JoltInstructionRow>` is now generic, and its final typed enum
  variants omit source-only rows such as `ADDW`, `DIV`, `LW`, `SW`, atomics,
  CSRs, trap rows, and `Inline`.
- `jolt-riscv` now exposes `SourceExtension`, `JoltTargetExtension`,
  `InlineExtension`, shipped `JoltInstructionProfile` presets, positive
  source/final legality checks, profile-local dense indexes, and a profile
  fingerprint.
- `jolt-program` bytecode and program preprocessing record the selected profile
  fingerprint so serialized preprocessing artifacts carry the profile identity
  used for legality and dense-index derivation.
- `jolt-program` decode, expansion, sequence stamping, and bytecode
  preprocessing now take an explicit selected `JoltInstructionProfile` instead
  of reading the default profile from inside those phase boundaries.
- Inline inventory registrations now declare their `InlineExtension`, and the
  tracer inline provider rejects a registered `(opcode, funct3, funct7)` key
  when the selected profile does not enable that extension.
- Recursive expansion recipes now distinguish source helper expansion
  (`SourceInstructionKind`) from final-row emission (`JoltInstructionKind`).
  Source-only recipe builders carry their source context as `SourceInstructionRow`, so
  they no longer need a broad `JoltInstructionKind` tag just to access operands,
  address, or compressed-row metadata.
- Tracer source conversion now builds `SourceInstruction` directly, while
  `try_from_jolt_instruction_row` rejects final `Inline` rows.
- `SourceInstruction` no longer has an implicit `From<SourceInstruction> for
  JoltInstructionRow` conversion; native source rows that pass through expansion use a
  checked `TryFrom<&SourceInstruction>` conversion to `JoltInstructionRow` that
  rejects source-only rows.
- Remaining caveat: the legacy bare `JoltInstructionKind` row-tag enum is still
  broad for this PR slice, even though the typed `JoltInstruction<T>` enum and
  profile legality layer are final-only. It remains as a compact identity and
  serialization bridge for existing row APIs and tracer internals, not as the
  source-of-truth final instruction universe.

1. [x] Add `SourceInstructionRow`, `SourceInlineKey`, `JoltInstructionRow`, and operand aliases/types in
   `jolt-riscv`.
2. [x] Add universal `SourceInstruction<T = SourceInstructionRow>` and
   `JoltInstruction<T = JoltInstructionRow>` enums whose variants carry marker structs,
   following the former `LookupInstruction` pattern.
3. [x] Add explicit canonical operation names, stable `u16` Jolt tags, and
   serialization for those universal row enums. Tags must encode canonical names
   for rows supported by this catalog and must not depend on Rust enum
   declaration order or the selected profile.
4. [x] Add the shipped `JoltInstructionProfile` presets and positive source/target
   legality APIs. Profiles select legal rows; they do not change enum shape.
5. [x] Add generated profile-local dense-index maps where preprocessing/proving
   needs compact indexes. Dense indexes must be derived from compact tags, and
   preprocessing artifacts must record the selected profile/catalog fingerprint.
6. [x] Add `jolt-program`-owned decode metadata keyed by marker structs or source
   enum variants, then change ELF/word decode to return
   `SourceInstruction<SourceInstructionRow>` after profile legality validation.
7. [x] Add `jolt-program`-owned source expansion metadata/dispatch keyed by marker
   structs or source enum variants.
8. [x] Change `jolt-program::expand` public APIs and internal recipes to consume
   source rows and emit target rows.
9. [x] Remove `Inline` from the final `JoltInstruction<T>` universe and move inline
   metadata to source-only types.
10. [x] Cut bytecode preprocessing, `JoltProgram`, execution rows, and proof imports
   over to `JoltInstruction`.
11. [x] Update tracer conversions so decode/source paths and expanded execution paths
   are separate while preserving tracer ownership of concrete execution
   semantics.
12. [x] Update `TracerInlineExpansionProvider` and `jolt-inlines-sdk` boundaries so
   registered inline expansion accepts source inline rows and returns validated
   final rows.
13. [x] Move or preserve lookup/proving metadata in the lookup/proving owner, keyed
    by final instruction identities; do not add lookup-table flags, circuit
    flags, or instruction flags to the `jolt-riscv` row enum definitions.
14. [x] Delete obsolete normalized-row aliases, stale legality helpers, and any
    source-only variants left in `JoltInstruction<T>`.
15. [x] Add small default-profile legality tests that check current supported source
    extensions, inline extensions, and computed target legality closure
    accept/reject the expected source/final rows.
16. [x] Add canonical-name, compact-tag, and dense-index tests that prove adding a
    row does not rename existing operations or renumber existing serialized
    tags, unsupported rows have no profile-local dense index, and
    profile/catalog fingerprinting changes when dense maps or legality sets
    change.
17. [x] Add the `jolt-eval` `source_to_jolt_expansion_equivalence` invariant and use
    it to gate any expansion fixture/hash regeneration.
18. [x] Run the full validation stack and update the expansion fixture/hash only if
    row type serialization changes while structural expansion output remains
    unchanged.

## References

- [PR #1518](https://github.com/a16z/jolt/pull/1518)
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [MLIR Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [MLIR Bytecode Format](https://mlir.llvm.org/docs/BytecodeFormat/)
- [`specs/compiler-native-bytecode-expansion.md`](compiler-native-bytecode-expansion.md)
- [`specs/bytecode-expansion-crate.md`](bytecode-expansion-crate.md)
- Archived extraction audit:
  `/Users/quang.dao/Documents/Notes/jolt-pr1518-extraction-audit-hax-aeneas.md`
