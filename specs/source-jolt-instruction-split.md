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

This split should also prepare Jolt for a more modular instruction world. Jolt
will likely need compile-time profiles for different source ISA combinations
and different target bytecode/lookup-table sets: for example RV64I+M without C,
the current RV64IMAC profile, future floating-point support, or inline-specific
virtual rows that are only enabled when a cryptographic inline package is
compiled in. This PR should not build that full profile system, but it should
avoid baking in the current monolithic "everything exists everywhere" shape.

## Intent

### Goal

Introduce phase-specific instruction data types and use them across decode,
expansion, tracer conversion, bytecode preprocessing, and inline expansion:

- `SourceInstructionKind`: decoded source-program opcode identity.
- `SourceInstruction`: decoded source row, including address, operands,
  compression metadata, and inline dispatch metadata when applicable.
- `JoltInstructionKind`: final expanded bytecode row identity.
- `JoltInstruction`: final bytecode/proof/tracer row, including operands,
  address, virtual-sequence metadata, and compression-tail metadata.

The old `NormalizedInstruction` row should not remain as a compatibility shim.
If a temporary name is needed while editing, it must be removed before the PR is
ready for review.

The concrete enums may remain closed Rust enums for now. That is useful for
serialization, match exhaustiveness, static dispatch, and proof performance.
The important architectural change is that they should be treated as generated
views over catalog metadata, not as one hand-maintained global list that every
phase shares.

### Invariants

- Decoding preserves all currently supported RV64 and Jolt custom source
  opcodes, including registered inline dispatch metadata.
- Source and target instruction membership is explicit in the catalog. Adding a
  source opcode should not automatically add a final Jolt bytecode row, and
  adding a target-only virtual row should not automatically make it decodable
  from guest program bytes.
- Expansion behavior is unchanged relative to `main` after PR #1518:
  `SourceInstruction -> Vec<JoltInstruction>` must match the existing expanded
  bytecode for the checked fixture corpus, modulo intentional type names.
- Final Jolt bytecode cannot contain source-only opcodes:
  `Inline`, `ADDW`, `LW`, `SW`, AMOs, traps, CSR rows, shifts, DIV/REM, advice
  source loads, and other source-only expansion inputs must be rejected before
  preprocessing.
- `JoltInstructionKind` contains only target-legal rows consumed by bytecode
  preprocessing, tracer execution, and proof lookup metadata. In particular, it
  must not contain `Inline`.
- Source and final rows carry different semantic metadata:
  source rows know decode/inline identity; final rows know virtual-sequence
  position.
- `rd = x0` rewriting remains centralized in expansion for source rows and is
  not reimplemented in tracer as an independent policy.
- Registered inline expansion remains behind a provider boundary. The provider
  may live in `tracer`, but its input must be a source inline row and its output
  must be validated `JoltInstruction` rows.
- Expansion definitions should stay readable for humans authoring and reviewing
  instruction lowerings. The refactor may change the underlying recipe and row
  types, but the call-site syntax for ordinary expansions should remain at
  least as easy to parse as the current builder style.
- `jolt-program::expand` remains independent of tracer CPU state, advice tapes,
  concrete tracer cycles, PCS/prover code, and ELF parsing.
- Prover/verifier behavior is unchanged. Bytecode preprocessing, PC mapping,
  instruction flags, lookup-table routing, and trace witness generation must
  see the same final Jolt rows as before.

### Non-Goals

- Do not redesign lookup-table metadata or reintroduce `LookupInstructionKind`.
  Keep `LookupInstruction` as the typed lookup-backed view over final
  `JoltInstruction` rows.
- Do not change RISC-V or Jolt instruction semantics.
- Do not change the registered inline algorithms themselves.
- Do not require Hax/Aeneas extraction to compile in this PR.
- Do not introduce deprecated aliases, conversion shims, or dual public APIs.
  This is a full cutover.
- Do not make `jolt-program` depend on `tracer`.
- Do not implement user-defined extension profiles in this PR. The catalog
  should be shaped so profiles can be generated later, but the only required
  profile here is the current supported Jolt profile.

## Evaluation

### Acceptance Criteria

- [ ] `NormalizedInstruction` is removed or fully renamed into one of the two
      phase-specific types, with no compatibility alias.
- [ ] `jolt-program::image::decode_instruction` returns `SourceInstruction`.
- [ ] `jolt-program::expand` accepts `SourceInstruction` at public boundaries
      and returns `Vec<JoltInstruction>`.
- [ ] Recursive expansion internally distinguishes source helper dispatch from
      direct target-row emission without relying on a broad shared enum.
- [ ] `JoltInstructionKind` no longer has `Inline` and no longer contains rows
      rejected by final-bytecode legality.
- [ ] `SourceInstructionKind` remains broad enough to describe decoded RV64,
      Jolt custom source opcodes, and registered inline opcodes.
- [ ] The instruction catalog records source membership and target membership
      separately. `SourceInstructionKind` and `JoltInstructionKind` are generated
      from those memberships rather than from one shared variant list.
- [ ] The default catalog/profile corresponds to the current supported Jolt
      behavior. Future profiles can be added by selecting subsets/extensions of
      the catalog, without changing the row types' phase semantics.
- [ ] Inline dispatch metadata is represented directly on `SourceInstruction`
      or a `SourceInlineInstruction` payload, not packed into
      `NormalizedOperands::imm`.
- [ ] `InlineExpansionProvider` accepts source inline data and returns final
      `JoltInstruction` rows.
- [ ] Concrete expansion files remain ergonomic for human authors: common
      lowering code should read like a small instruction sequence, not like
      serialized grammar data or generated tables.
- [ ] Tracer concrete `Instruction`/`Cycle` APIs execute final Jolt rows, while
      decoded source instructions convert through the expansion path before
      trace execution.
- [ ] No verifier-facing crate imports tracer just to name source or final row
      types.
- [ ] Existing expansion fixture/hash tests still pass against `main`'s
      post-#1518 behavior.
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
- fixture parity for the existing representative source corpus.

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

The catalog/profile work has the same constraint. Metadata can become more
structured, but instruction authors should not need to mentally execute macro
grammar to understand whether an opcode is source-only, target-only,
lookup-backed, side-effecting, or part of a default profile.

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

`jolt-riscv` owns both row families:

```rust
pub enum SourceInstructionKind {
    ADD,
    ADDIW,
    LW,
    SW,
    CSRRW,
    Inline,
    VirtualRev8W,
    AdviceLD,
    // decoded source-program opcode identities
}

pub struct SourceInstruction {
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
}

pub enum JoltInstructionKind {
    ADD,
    ADDI,
    LD,
    SD,
    MUL,
    MULHU,
    VirtualAdvice,
    VirtualAssertEQ,
    VirtualSignExtendWord,
    // only rows legal after expansion
}

pub struct JoltInstruction {
    pub kind: JoltInstructionKind,
    pub address: usize,
    pub operands: JoltOperands,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}
```

The exact operand type names can change, but the phase split should be visible
at the type level. Source operands may continue to use normalized register
fields for ordinary rows; inline dispatch metadata should not be stored in an
immediate field.

### Catalog And Profiles

The current code generates both instruction-kind enums from one broad macro
list. That makes the type split mostly nominal: every source opcode also exists
as a final Jolt opcode unless a later legality check rejects it. This PR should
replace that with catalog metadata whose first-order facts are:

```rust
pub enum InstructionPhase {
    SourceOnly,
    TargetOnly,
    SourceAndTarget,
}

pub enum SourceExtension {
    Rv64I,
    Rv64M,
    Rv64A,
    Rv64C,
    JoltCustom,
    JoltInline,
}

pub enum TargetExtension {
    JoltCore,
    JoltMemory,
    JoltBranch,
    JoltVirtualArithmetic,
    JoltVirtualMemory,
    JoltVirtualAssert,
    JoltAdvice,
    JoltInlineSupport,
}
```

The exact type names are not important. The durable requirement is that source
decode support, target bytecode support, lookup-table support, and side-effect
metadata are separate catalog facts. Closed enums are acceptable as generated
compile-time artifacts, but the source of truth should be the catalog entries,
not a monolithic enum.

The long-term shape is profile-driven:

```rust
pub struct JoltInstructionProfile {
    pub source_extensions: &'static [SourceExtension],
    pub target_extensions: &'static [TargetExtension],
}

pub const RV64IMAC_JOLT: JoltInstructionProfile = JoltInstructionProfile {
    source_extensions: &[
        SourceExtension::Rv64I,
        SourceExtension::Rv64M,
        SourceExtension::Rv64A,
        SourceExtension::Rv64C,
        SourceExtension::JoltCustom,
        SourceExtension::JoltInline,
    ],
    target_extensions: &[
        TargetExtension::JoltCore,
        TargetExtension::JoltMemory,
        TargetExtension::JoltBranch,
        TargetExtension::JoltVirtualArithmetic,
        TargetExtension::JoltVirtualMemory,
        TargetExtension::JoltVirtualAssert,
        TargetExtension::JoltAdvice,
        TargetExtension::JoltInlineSupport,
    ],
};
```

Profiles should be compile-time selections, not runtime plugin loading. The
selected profile affects decoding, expansion, bytecode preprocessing, lookup
metadata, proving keys, and circuit shape, so it must remain visible to the
Rust type system and Cargo feature/build configuration. Jolt should ship common
presets, and downstream users should eventually be able to define their own
profile by composing supported catalog extensions.

For this PR, the implementation only needs the current default profile. The
catalog should nevertheless make extension membership explicit enough that a
future PR can generate smaller source/target enums or tables from selected
extensions without re-separating the phase model.

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

Tracer should become phase-aware:

- decode-facing APIs consume or produce `SourceInstruction`;
- expanded trace execution consumes `JoltInstruction` or concrete tracer
  `Instruction` built from `JoltInstruction`;
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

Making the enums fully dynamic, MLIR-style operation IDs would maximize
extensibility but is the wrong first move for Jolt. Instruction identity feeds
serialization, bytecode preprocessing, static flag dispatch, lookup routing,
and proof circuits. Closed generated enums are a better fit: the catalog can be
modular while the compiled prover remains specialized.

Using Cargo features alone as the catalog would also be too coarse. Features
can select profiles or extension groups, but they should not be the only place
where instruction semantics live. Decode membership, target legality, lookup
table routing, side effects, and lowering behavior should be queryable from one
instruction catalog.

Making only `jolt-program` phase-aware is also insufficient. Tracer currently
converts concrete `Instruction` through the same normalized row and also handles
some expansion-ish behavior during tracing. If tracer remains untyped by phase,
the next cleanup will immediately need another cross-crate churn.

Removing tracer's concrete `Instruction` enum entirely is too broad. The enum
still owns execution behavior, cycle construction, and tests. This PR should
separate source/final row APIs while preserving tracer's concrete execution
model where it is useful.

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
2. Split the instruction macro/catalog so it generates source and target enums
   from explicit source/target membership rather than one broad shared list.
3. Change ELF/word decode to return `SourceInstruction`.
4. Change `jolt-program::expand` public APIs and internal recipes to consume
   source rows and emit target rows.
5. Remove `JoltInstructionKind::Inline` and move inline metadata to
   source-only types.
6. Cut bytecode preprocessing, `JoltProgram`, execution rows, and proof imports
   over to `JoltInstruction`.
7. Update tracer conversions so decode/source paths and expanded execution paths
   are separate.
8. Update `TracerInlineExpansionProvider` and `jolt-inlines-sdk` boundaries so
   registered inline expansion accepts source inline rows and returns validated
   final rows.
9. Delete obsolete normalized-row aliases, stale legality helpers, and any
   source-only variants left in `JoltInstructionKind`.
10. Add a small default-profile/catalog test that checks current supported
    source extensions and target extensions generate the expected phase-specific
    kinds.
11. Run the full validation stack and update the expansion fixture/hash if the
    row type serialization changes but semantic output is unchanged.

## References

- [PR #1518](https://github.com/a16z/jolt/pull/1518)
- [`specs/compiler-native-bytecode-expansion.md`](compiler-native-bytecode-expansion.md)
- [`specs/bytecode-expansion-crate.md`](bytecode-expansion-crate.md)
- Archived extraction audit:
  `/Users/quang.dao/Documents/Notes/jolt-pr1518-extraction-audit-hax-aeneas.md`
