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
flow through `NormalizedInstruction { instruction_kind: JoltInstructionKind }`.
That keeps the code working, but the type names still hide an important phase
boundary. This PR should make that boundary real: decoded program instructions
are source instructions, expansion recipes consume source instructions, and
bytecode/preprocessing/tracing/proof rows are Jolt instructions.

The same cutover should clean up registered inlines. Inline opcodes are source
program opcodes identified by `(opcode, funct3, funct7)`; they are not final
Jolt bytecode rows. The inline provider contract should therefore accept source
inline rows and return final Jolt rows, without routing through a fake
`JoltInstructionKind::Inline` row.

This split should also prepare Jolt for a more modular instruction world without
turning this PR into a general profile system or a single all-knowing instruction
macro. The near-term requirement is to create one shared instruction identity
registry, then let each crate decorate those identities with the facts it owns.
`jolt-riscv` should know stable marker names, enum variants, discriminants, row
shape, and source/final membership. `jolt-program` should own decode and
expansion facts. `tracer` should own execution semantics. Lookup/proving crates
should own lookup-table and circuit metadata. The final form of this PR should
not be broad mirrored `SourceInstructionKind` / `JoltInstructionKind` enums, and
it should not move downstream proving-system details into `jolt-riscv`.

## Intent

### Goal

Introduce an explicit instruction registry/profile layer, then use the selected
source profile and computed target closure to generate phase-specific
instruction identifiers and row types across decode, expansion, tracer
conversion, bytecode preprocessing, and inline expansion:

- shared registry entries: the source of truth for stable instruction identity,
  marker struct names, enum variant names, discriminants, and source/final
  membership.
- crate-local decorations: the source of truth for crate-specific behavior and
  metadata such as decode encodings, expansion dispatch, tracer execution,
  lookup-table routing, circuit flags, and instruction flags.
- `SourceInstructionKind`: generated decoded source-program opcode identity.
- `SourceInstruction`: decoded source row, including address, operands,
  compression metadata, and inline dispatch metadata when applicable.
- `JoltInstructionKind`: generated final expanded bytecode row identity.
- `JoltInstruction`: final bytecode/proof/tracer row, including operands,
  address, virtual-sequence metadata, and compression-tail metadata.

The old `NormalizedInstruction` row should not remain as a compatibility shim.
If a temporary name is needed while editing, it must be removed before the PR is
ready for review.

The concrete `*InstructionKind` types may remain closed Rust enums. That is
useful for serialization, match exhaustiveness, static dispatch, and proof
performance. They must be generated views over shared registry facts and the
selected source profile, not one hand-maintained global list that every phase
shares.

Registry/profile support is part of the main goal, not a stretch goal. The PR
should leave `SourceInstructionKind` and `JoltInstructionKind` as phase-specific
generated artifacts of registry/profile facts, not as the primary source of
instruction-set truth.

### Invariants

- Decoding preserves all currently supported RV64 and Jolt custom source
  opcodes, including registered inline dispatch metadata.
- Source and target instruction membership is explicit in the registry. Adding a
  source opcode should not automatically add a final Jolt bytecode row, and
  adding a target-only virtual row should not automatically make it decodable
  from guest program bytes.
- Shared instruction identity facts are centralized once, close to the
  instruction marker structs. Crate-specific facts are not centralized there:
  they are declared by the crate that owns the behavior and are keyed by the same
  marker structs or generated enum variants.
- The current Jolt source profile is explicit in code. The target instruction
  set is computed from the selected source extensions plus selected inline
  extensions: it is the set of final rows that can be emitted by those
  expansions.
- Expansion behavior is unchanged relative to `main` after PR #1518:
  `SourceInstruction -> Vec<JoltInstruction>` must match the existing expanded
  bytecode for the checked fixture corpus, modulo intentional type names.
- Final Jolt bytecode cannot contain source-only opcodes:
  `Inline`, `ADDW`, `LW`, `SW`, AMOs, traps, CSR rows, shifts, DIV/REM, advice
  source loads, and other source-only expansion inputs must be rejected before
  preprocessing.
- Generated `JoltInstructionKind` contains only final rows in the computed target
  closure consumed by bytecode preprocessing, tracer execution, and proof lookup
  metadata. In particular, it must not contain `Inline`.
- Source and final rows carry different semantic metadata:
  source rows know decode/inline identity; final rows know virtual-sequence
  position.
- `rd = x0` rewriting remains centralized in expansion for source rows and is
  not reimplemented in tracer as an independent policy.
- Registered inline expansion remains behind a provider boundary. The provider
  may live in `tracer`, but its input must be a source inline row and its output
  must be validated `JoltInstruction` rows.
- Concrete execution semantics remain tracer-owned. The shared registry must not
  own CPU state mutation, RAM/advice side effects, concrete cycle construction,
  or inline advice generation.
- Lookup/proving metadata remains owned by the lookup/proving side of the codebase.
  The shared registry may provide stable final-row identities, but it must not
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

### Non-Goals

- Do not redesign lookup-table metadata or reintroduce `LookupInstructionKind`.
  Keep `LookupInstruction` as the typed lookup-backed view over final
  `JoltInstruction` rows.
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

- [ ] `NormalizedInstruction` is removed or fully renamed into one of the two
      phase-specific types, with no compatibility alias.
- [ ] `jolt-program::image::decode_instruction` returns `SourceInstruction`.
- [ ] `jolt-program::expand` accepts `SourceInstruction` at public boundaries
      and returns `Vec<JoltInstruction>`.
- [ ] Recursive expansion internally distinguishes source helper dispatch from
      direct target-row emission without relying on a broad shared enum.
- [ ] Generated `JoltInstructionKind` no longer has `Inline` and contains only
      final rows reachable from the selected source extensions and enabled
      inline extensions.
- [ ] Generated `SourceInstructionKind` contains decoded RV64 source opcodes,
      Jolt custom source opcodes, and one source-only `Inline` kind when the
      source profile enables inline decoding. Individual registered inline
      opcodes are represented by `SourceInline`, not by one enum variant per
      inline package entry.
- [ ] The shared instruction registry records source membership and final-row
      membership separately. `SourceInstructionKind` and `JoltInstructionKind`
      are generated from those facts rather than from one shared variant list.
- [ ] Shared registry metadata is limited to cross-crate identity and row-shape
      facts: marker struct name, enum variant name, stable discriminant,
      source/final membership, and profile-independent row classification.
- [ ] Decode metadata and operand parsing are declared in `jolt-program`, keyed
      by instruction marker structs or generated `SourceInstructionKind`
      variants, not duplicated as an unrelated handwritten opcode list.
- [ ] Source-only expansion dispatch is declared/generated in `jolt-program`,
      keyed by instruction marker structs or generated `SourceInstructionKind`
      variants, and does not live in `jolt-riscv`.
- [ ] Lookup-table routing, circuit flags, and instruction flags remain owned by
      the lookup/proving crates. They are not fields in the shared `jolt-riscv`
      registry or in a mega `jolt_instruction!` declaration.
- [ ] Tracer still owns concrete execution semantics. No registry macro in
      `jolt-riscv` mutates CPU/RAM/advice state or constructs concrete tracer
      cycles.
- [ ] Broad mirrored source/target enums are not the final implementation
      shape. If closed enums remain, they are generated from registry/profile
      facts and do not themselves define instruction-set membership.
- [ ] The profile/extension layer uses the concrete names `SourceExtension`,
      `JoltTargetExtension`, `InlineExtension`, and `JoltInstructionProfile`.
- [ ] The default registry/profile corresponds to the current supported
      RV64IMAC Jolt behavior, with room for shipped source presets such as
      `RV64IM_JOLT`, `RV64IMAC_JOLT`, and `RV64IMAC_JOLT_ALL_INLINES`.
      Future profiles can be added by selecting source/inline extensions and
      recomputing the target closure without changing the row types' phase
      semantics.
- [ ] Inline dispatch metadata is represented directly on `SourceInstruction`
      or a `SourceInlineInstruction` payload, not packed into
      `NormalizedOperands::imm`.
- [ ] `InlineExpansionProvider` accepts source inline data and returns final
      `JoltInstruction` rows.
- [ ] Concrete expansion files remain ergonomic for human authors: common
      lowering code should read like a small instruction sequence, not like
      serialized grammar data or generated tables.
- [ ] The refactor does not introduce extraction-hostile Rust patterns in the
      instruction registry, row types, or expansion pipeline.
- [ ] Tracer concrete `Instruction`/`Cycle` APIs execute final Jolt rows, while
      decoded source instructions convert through the expansion path before
      trace execution.
- [ ] No verifier-facing crate imports tracer just to name source or final row
      types.
- [ ] Existing expansion fixture/hash tests still pass against `main`'s
      post-#1518 behavior, and any fixture hash regeneration is backed by a
      structural source-to-final stream equivalence check rather than reviewer
      trust in changed serialization bytes.
- [ ] `cargo tree -p jolt-program` has no `tracer` dependency.

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
- tracer execution of expanded rows after the source/final cutover;
- fixture parity for the existing representative source corpus;
- a `jolt-eval` invariant named `source_to_jolt_expansion_equivalence` that
  compares the new source-to-final expansion stream against the pre-refactor
  normalized expansion semantics modulo intentional row type and discriminant
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
    JoltInstructionKind::SUB,
    rd(instruction)?,
    rs1(instruction)?,
    rs2(instruction)?,
);
asm.emit_i(
    JoltInstructionKind::VirtualSignExtendWord,
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

The registry/profile work has the same constraint. Metadata can become more
structured, but instruction authors should not need to mentally execute macro
grammar to understand whether an opcode is source-only, target-only,
lookup-backed, side-effecting, or part of a default profile.

### Expansion Equivalence

The expansion fixture hash may change when `NormalizedInstruction` is replaced
by phase-specific row types, even if the emitted program is semantically
unchanged. Regenerating that fixture is therefore not itself sufficient evidence.
Before updating hashes, add a structural equivalence check that compares the new
`SourceInstruction -> Vec<JoltInstruction>` stream with the post-#1518 `main`
semantics modulo intentional type/discriminant renames and the removal of
`JoltInstructionKind::Inline` from final rows.

The durable invariant should live in `jolt-eval` as
`source_to_jolt_expansion_equivalence`. It should run over the representative
program corpus used by the expansion parity fixture and fail on differences in
final opcode identity, operands, addresses, virtual sequence metadata, and
compression-tail metadata. Once that invariant passes, the fixture hashes can be
regenerated if serialization bytes changed mechanically.

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
- registry/profile metadata that can be inspected as data, not only through macro
  expansion side effects.

Avoid introducing:

- `dyn` dispatch or closure-heavy APIs in the core instruction/expansion path;
- hidden global state for registry/profile decisions;
- unsafe code in registry, decode, or expansion plumbing;
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

`jolt-riscv` owns shared instruction identity because it is the lowest common
crate shared by decode, expansion, tracer, bytecode preprocessing, and lookup
metadata. That ownership should stay deliberately small. It should provide:

- the marker structs such as `Add<T>`, `AddW<T>`, and
  `VirtualSignExtendWord<T>`;
- the stable enum variant names and serialization discriminants;
- whether an identity can appear as a decoded source row, a final Jolt row, or
  both;
- row structs and profile-independent row-shape types shared across crates.

It should not provide a mega declaration that also names decode opcodes,
expansion bodies, side-effect policy, tracer execution, lookup-table routing,
circuit flags, or instruction flags. Those are real facts, but they belong to
the crates that use and test them.

Because Rust macros cannot discover scattered declarations across files, the
closed enums still need one explicit registry boundary. That registry can live
near the existing marker definitions and should be intentionally boring:

```rust
instruction_registry! {
    ADD => Add,
        discriminant = 0x0001,
        source = true,
        target = true;

    ADDW => AddW,
        discriminant = 0x0002,
        source = true,
        target = false;

    VirtualSignExtendWord => VirtualSignExtendWord,
        discriminant = 0x1001,
        source = false,
        target = true;

    Inline => Inline,
        discriminant = 0x7fff,
        source = true,
        target = false;
}
```

The exact syntax can change. The durable shape is that the registry is a small
identity table, not a universal semantics table. It should be enough to generate
`SourceInstructionKind`, `JoltInstructionKind`, stable serialization, and simple
membership predicates. It should not answer questions like "which lookup table
does this row use?" or "which tracer method mutates memory for this row?"

Generated row types should then be ordinary data:

```rust
pub struct SourceInstruction {
    // Generated from registry entries whose `source` field is present.
    pub kind: SourceInstructionKind,
    pub address: usize,
    pub operands: SourceOperands,
    pub inline: Option<SourceInline>,
    pub is_compressed: bool,
}

pub struct SourceInline {
    pub opcode: u32,
    pub funct3: u32,
    pub funct7: u32,
    pub extension: InlineExtension,
}

pub struct JoltInstruction {
    // Generated from reachable registry entries whose `target` field is present.
    pub kind: JoltInstructionKind,
    pub address: usize,
    pub operands: JoltOperands,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}
```

The exact operand/spec type names can change, but the ownership relationship
should not: the shared registry defines what identities exist, the selected
source profile defines what decodes, and the target closure defines which final
rows may be emitted. `SourceInstructionKind` / `JoltInstructionKind` are
generated identifiers over those registry/profile facts. Source operands may
continue to use normalized register fields for ordinary rows; inline dispatch
metadata should not be stored in an immediate field.

### Ownership Boundaries

The shared registry owns only cross-crate identity. Crate-local decorations own
the behavior-specific facts, using the same marker structs as join keys.

`jolt-program` owns source decoding, operand parsing, positive final-bytecode
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
        instruction: &SourceInstruction,
        allocator: &mut ExpansionAllocator,
    ) -> Result<Vec<JoltInstruction>, ExpansionError>;
}

impl SourceExpansion for AddW<()> {
    fn expand_source(
        instruction: &SourceInstruction,
        allocator: &mut ExpansionAllocator,
    ) -> Result<Vec<JoltInstruction>, ExpansionError> {
        expand_addw(instruction, allocator)
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
marker structs or by the generated final enum. For example:

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
the macro should live with the lookup/proving owner, not in `jolt-riscv`:

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
must remain below `jolt-program` in the dependency graph. `jolt-riscv` can say
that `AddW` is source-only. `jolt-program` says how `AddW` lowers and can
generate the dispatcher that routes `SourceInstructionKind::ADDW` through the
`AddW` marker to the human-written `expand_addw` body.

### Registry And Profiles

The current code generates both instruction-kind enums from one broad macro
list. That makes the type split mostly nominal: every source opcode also exists
as a final Jolt opcode unless a later legality check rejects it. This PR should
replace that with a small identity registry plus crate-local metadata whose
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

The durable requirement is that source decode support, target bytecode support,
inline registration support, lookup-table support, side-effect metadata, circuit
flags, and instruction flags are separate compile-time facts with separate
owners. The PR should not add a separate `InstructionPhase` enum unless the
implementation needs it internally; phase is already determined by whether a
registry entry has `source`, `target`, or both. Closed enums are acceptable as
generated compile-time artifacts, but the source of truth must be the
registry/profile plus crate-local decorations, not a monolithic enum.

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

The computed target closure for the current default profile should include these
`JoltTargetExtension` families:

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

Sentinel rows `NoOp` and `Unimpl` are always available and should not be modeled
as extension-gated capabilities.

The inline registry should use the registered inline package names as first-class
entries, not treat every inline as one anonymous extension. This does not mean
`SourceInstructionKind` gets one variant per inline operation. Source decoding
uses one `SourceInstructionKind::Inline` row plus a `SourceInline` payload; the
`InlineExtension` profile gates which registered `(opcode, funct3, funct7)` keys
are accepted and which provider is allowed to expand them. Current entries are:

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
rows, but provider output must be validated against the computed target closure
before preprocessing.

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
`JoltInstructionProfile`, the build derives the final target set from:

- direct final rows emitted by source instructions enabled by
  `source_extensions`;
- recursive helper rows reachable from those expansions;
- final rows emitted by enabled `inline_extensions`;
- sentinel rows such as `NoOp` and `Unimpl`.

This makes `RV64IM_JOLT` naturally produce a smaller final closure than
`RV64IMAC_JOLT` when atomics/compressed-only source paths are disabled, without
requiring callers to maintain a parallel target list. Cross-profile proof
artifact compatibility is not a goal of this PR: circuit/preprocessing keys are
tied to the selected compile-time profile.

Reserve these shipped preset names:

- `RV64IM_JOLT`: base RV64I+M profile without atomics or compressed encodings;
- `RV64IMAC_JOLT`: current base RV64IMAC source profile with the inline source
  mechanism available;
- `RV64IMAC_JOLT_ALL_INLINES`: current workspace-wide profile with all listed
  `InlineExtension` packages enabled.

Profile selection should be compile-time, not runtime plugin loading. For this
PR, use Cargo features or marker types to select one shipped profile for the
compiled crate graph; do not thread an arbitrary runtime
`&'static JoltInstructionProfile` through `JoltProgram::new`. If marker types
are more ergonomic than value-level profiles, use the same names as type names,
for example `profile::Rv64imacJolt`. Exported presets should still use the
constant names above.

Inline inventory registration remains link-time, but availability is
profile-checked. If a workspace links `jolt-inlines-sha2` while the selected
profile does not include `InlineExtension::Sha2`, decode/expansion must reject
the registered key before it can enter finalized bytecode. The PR does not need
to make such a crate fail to compile.

For this PR, the implementation must include the explicit shared registry,
crate-local metadata, and shipped current source profiles above. It does not
need arbitrary third-party profile composition, but source/target enum
generation, decode legality, final-bytecode legality, and inline availability
should all flow from registry/profile facts and crate-owned decorations rather
than from a broad shared instruction-kind list.

Lookup support should not introduce `LookupInstructionKind`. Keep the existing
boundary: `JoltInstructionKind` identifies final rows, `LookupInstruction` is
the typed lookup/proof-facing view of supported final rows, and
`LookupTableKind` identifies concrete lookup tables in `jolt-lookup-tables`.
A future `LookupTableProfile` may name the compile-time table set for a
selected proof profile, but it should be table-oriented, not another
instruction-kind enum.

Expansion should expose:

```rust
pub fn expand_instruction(
    instruction: &SourceInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<JoltInstruction>, ExpansionError>;

pub trait InlineExpansionProvider {
    fn expand_inline(
        &mut self,
        instruction: &SourceInstruction,
        allocator: &mut ExpansionAllocator,
    ) -> Result<Vec<JoltInstruction>, ExpansionError>;
}
```

Inside `jolt-program::expand`, the builder should keep two operations:

- `emit_*`: append target-legal `JoltInstructionKind` rows directly.
- `expand_*`: recursively lower helper `SourceInstructionKind` rows.

This models the current semantics more accurately than pretending every
recursive helper is pure RISC-V or every emitted row is already final.

The source-only expansion dispatcher should be generated from `jolt-program`
metadata or from a small `jolt-program` macro keyed by the instruction marker
structs, not maintained as an unrelated `match JoltInstructionKind` list. The
generated shape should read conceptually like:

```rust
match instruction.kind {
    SourceInstructionKind::ADDW => AddW::expand_source(instruction, allocator),
    SourceInstructionKind::LW => Lw::expand_source(instruction, allocator),
    SourceInstructionKind::Inline => inline_provider.expand_inline(instruction, allocator),
    _ if instruction.kind.is_direct_target() => emit_direct(instruction),
    _ => Err(ExpansionError::UnsupportedInstruction),
}
```

That does not require moving `expand_addw` into `jolt-riscv`; it requires the
dispatch edge to be generated from `jolt-program`'s local `AddW` expansion
metadata instead of copied into a separate list that can drift from the source
kind registry.

Tracer should become phase-aware:

- decode-facing APIs consume or produce `SourceInstruction`;
- expanded trace execution consumes `JoltInstruction` or concrete tracer
  `Instruction` built from `JoltInstruction`;
- execution semantics remain implemented on tracer concrete instruction types,
  not in `jolt-riscv` registry macros;
- `RISCVInstruction` conversions should not require
  `From<SourceInstruction> + Into<SourceInstruction> + From<JoltInstruction> +
  Into<JoltInstruction>` on the same trait;
- inline source rows should never be reconstructed from a final row.

Registered inlines can still be implemented in `tracer` for this PR, because
they need inventory registration and advice-generation hooks. The important
cleanup is the boundary: `TracerInlineExpansionProvider` should no longer parse
a fake final `JoltInstructionKind::Inline` row, and `jolt-inlines-sdk::InlineOp`
should either return final `JoltInstruction` rows directly or use an assembler
whose output is final `JoltInstruction` rows. If moving every inline builder off
tracer's `Instruction` enum proves too large, keep that as an internal tracer
adapter only and ensure the public provider contract is already final-row typed.

### Alternatives Considered

Keeping `SourceInstructionKind` and `JoltInstructionKind` as mirror enums is
the smallest change, but it keeps the reviewer concern intact: names improve,
but the compiler cannot enforce phase boundaries.

Putting every fact into an expanded `jolt_instruction!` declaration would remove
some match duplication, but it would also make `jolt-riscv` aware of details it
should not own: decode quirks, expansion bodies, tracer state mutation,
lookup-table routing, circuit flags, and instruction flags. That is the wrong
dependency direction. This PR should instead use a minimal shared registry plus
crate-local decorations.

Adding a separate `InstructionCatalogEntry` table beside the existing marker
structs and then filling it with every fact has the same problem in a different
shape. A separate table is acceptable only if it stays limited to shared identity
facts needed to generate the closed enums and stable discriminants.

Making the enums fully dynamic, MLIR-style operation IDs would maximize
extensibility but is the wrong first move for Jolt. Instruction identity feeds
serialization, bytecode preprocessing, static flag dispatch, lookup routing,
and proof circuits. Closed generated enums are a better fit: the registry can be
modular while the compiled prover remains specialized.

Using Cargo features alone as the registry would also be too coarse. Features
can select profiles or extension groups, but they should not be the only place
where instruction identity lives. Decode membership, target legality, lookup
table routing, side effects, and lowering behavior should be queryable through
their owning crates, keyed by shared instruction identities.

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
the lookup/proving owner and keyed by `JoltInstructionKind` or marker structs.

Moving all registered inline implementations into `jolt-program` would make the
provider-free crate depend on inventory/advice/tracer concerns. That reverses
the dependency boundary from #1518. A provider contract with final-row output is
the smaller clean architecture.

## Documentation

Update `specs/compiler-native-bytecode-expansion.md` after implementation to
mark the deferred source/final split complete and link to this spec. No Jolt
book update is required unless public SDK APIs expose the new names directly.

## Execution

1. Add `SourceInstruction`, `SourceInline`, `JoltInstruction`, and operand
   aliases/types in `jolt-riscv`.
2. Add a minimal shared instruction registry in `jolt-riscv` that records marker
   struct names, enum variant names, stable discriminants, and source/final
   membership.
3. Generate `SourceInstructionKind`, `JoltInstructionKind`, stable
   serialization, and simple membership predicates from that registry rather
   than from one broad shared list.
4. Add `jolt-program`-owned decode metadata keyed by marker structs or generated
   source enum variants, then change ELF/word decode to return
   `SourceInstruction`.
5. Add `jolt-program`-owned source expansion metadata/dispatch keyed by marker
   structs or generated source enum variants.
6. Change `jolt-program::expand` public APIs and internal recipes to consume
   source rows and emit target rows.
7. Remove `JoltInstructionKind::Inline` and move inline metadata to
   source-only types.
8. Cut bytecode preprocessing, `JoltProgram`, execution rows, and proof imports
   over to `JoltInstruction`.
9. Update tracer conversions so decode/source paths and expanded execution paths
   are separate while preserving tracer ownership of concrete execution
   semantics.
10. Update `TracerInlineExpansionProvider` and `jolt-inlines-sdk` boundaries so
   registered inline expansion accepts source inline rows and returns validated
   final rows.
11. Move or preserve lookup/proving metadata in the lookup/proving owner, keyed
    by final instruction identities; do not add lookup-table flags, circuit
    flags, or instruction flags to the `jolt-riscv` registry.
12. Delete obsolete normalized-row aliases, stale legality helpers, and any
    source-only variants left in `JoltInstructionKind`.
13. Add a small default-profile/registry test that checks current supported
    source extensions, inline extensions, and computed target closure generate
    the expected phase-specific kinds.
14. Add the `jolt-eval` `source_to_jolt_expansion_equivalence` invariant and use
    it to gate any expansion fixture/hash regeneration.
15. Run the full validation stack and update the expansion fixture/hash only if
    row type serialization changes while structural expansion output remains
    unchanged.

## References

- [PR #1518](https://github.com/a16z/jolt/pull/1518)
- [`specs/compiler-native-bytecode-expansion.md`](compiler-native-bytecode-expansion.md)
- [`specs/bytecode-expansion-crate.md`](bytecode-expansion-crate.md)
- Archived extraction audit:
  `/Users/quang.dao/Documents/Notes/jolt-pr1518-extraction-audit-hax-aeneas.md`
