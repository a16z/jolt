# Spec: Compiler-Native RV64 Bytecode Expansion

| Field      | Value |
|------------|-------|
| Author(s)  | Quang Dao |
| Created    | 2026-05-05 |
| Revised    | 2026-05-09 |
| Status     | implemented design |
| Related PR | [#1490](https://github.com/a16z/jolt/pull/1490), [#1518](https://github.com/a16z/jolt/pull/1518) |

## Summary

`jolt-program::expand` lowers decoded RV64 source rows into final Jolt bytecode
rows through a compiler-style pipeline:

```text
decoded RV64 source row
  -> readable per-instruction lowering
  -> ExpandedInstructionSequence recipe
  -> ExpansionState materializer
  -> stamped Jolt bytecode rows
```

Concrete lowerers are written as assembly-like recipes using
`ExpansionBuilder`. The builder records what should be emitted, expanded,
allocated, and released. It does not own allocator state and does not recursively
call the public expander. One central materializer interprets the recipe,
resolves symbolic temps to real virtual registers, recursively expands helper
rows, validates target legality, and stamps sequence metadata.

The design favors production Rust readability and exact register-budget
invariants. Hax/Aeneas extraction is useful as an audit path, but current
extractor limitations do not dictate Rust style.

## Scope

This PR combines four related cleanups that reinforce the same boundary:

- RV64-only program construction and tracing. Historical RV32/SV32 execution
  paths are removed from tracer code, and ELF32/RV32 inputs are rejected at the
  `jolt-program::image` boundary.
- lookup-backed instruction identity is split out of the flat instruction enum.
  The deeper decoded-source versus expanded-Jolt split is intentionally left as
  follow-up work because it requires tracer's concrete `Instruction` and
  `Cycle` model to become phase-aware.
- provider-free expansion is moved from a recursive assembler with borrowed
  allocator state to a recipe/materializer pass.
- serialization derives in `jolt-riscv` and `jolt-program` are feature-gated so
  the no-default expansion surface does not pull serde or ark serialization into
  extraction experiments.

Word instructions such as `ADDW`, `LW`, and `AMO*W` remain RV64 word
operations. They are not RV32 execution support.

## Goals

- Keep program construction and bytecode expansion RV64-only.
- Preserve expansion behavior relative to the post-#1490 baseline.
- Keep `JoltInstructionKind`/`NormalizedInstruction` as the existing flat row
  identity for this PR while removing the separate lookup-kind enum.
- Preserve a decoder-facing `SourceInstructionKind` only as a preparatory mirror
  of the current flat enum, not as a completed source/final split.
- Keep concrete lowerers readable enough for humans to add or revise opcodes.
- Centralize allocator ownership, recursive helper expansion, target-legality
  validation, metadata stamping, recursion-depth checks, and `rd = x0` rewriting.
- Keep registered inline expansion behind an explicit provider boundary.
- Keep provider-free expansion independent of tracer CPU state, advice tapes,
  prover internals, transcript code, PCS code, and ELF parsing.
- Keep serialization dependencies out of the no-default extraction surface.

## Non-Goals

- Do not change RISC-V or Jolt instruction semantics.
- Do not reintroduce RV32 or ELF32 support.
- Do not add compatibility aliases for renamed instruction kinds.
- Do not model execution semantics in the expander. Expansion defines bytecode
  lowering, not operational execution.
- Do not move registered `jolt-inlines` implementations into the provider-free
  grammar.
- Do not require Hax/Aeneas extraction in CI.
- Do not complete the source-program versus final-Jolt instruction-kind split
  in this PR. That requires tracer-side API changes so decoded guest
  instructions and expanded trace/proof rows no longer share the same concrete
  conversion path.

## Instruction Phases

Instruction identity is only partially separated in this PR:

- `JoltInstructionKind` remains the canonical flat row identity used by
  `NormalizedInstruction`, bytecode preprocessing, tracer conversion traits, and
  proof code. It still contains both final bytecode rows and source-only rows
  that are expansion inputs.
- `SourceInstructionKind` is a decoder-facing mirror of the same current
  instruction set. It names the decode result, but it does not yet enforce a
  smaller source/final type boundary.
- `LookupInstructionKind` has been removed. `LookupInstruction` remains as the
  typed view of rows with lookup/circuit metadata.

This deliberately does **not** address the deeper reviewer concern that source
and Jolt/target instruction kinds are still effectively the same broad set. A
follow-up PR should split tracer's phase model first: decoded guest instructions
should convert through a source row, while expanded trace/proof rows should
convert through a final Jolt row. Only after that should `JoltInstructionKind`
shrink to final bytecode rows.

`NormalizedInstruction` remains the production row type for decoded and expanded
rows. It carries row identity, operands, address, sequence metadata, and
compressed-tail metadata. It does not carry execution state, advice payloads, or
proof-system fields.

Lookup-backed classification is explicit through `LookupInstruction::try_from`.
Source-only legality and target-bytecode legality are still derived from the
combined `JoltInstructionKind` set in this PR and should be moved to true
source/final types in the tracer-aware follow-up.

## Expansion Pipeline

Each source-only lowerer returns an `ExpandedInstructionSequence`:

```rust
let mut asm = ExpansionBuilder::new(*instruction);

let tmp = asm.allocate()?;
asm.expand_i(
    JoltInstructionKind::VirtualPow2,
    tmp.operand(),
    reg(rs2(instruction)?),
    0,
);
asm.emit_r(
    JoltInstructionKind::MUL,
    reg(rd(instruction)?),
    reg(rs1(instruction)?),
    tmp.operand(),
);
asm.release(tmp);

asm.finalize()
```

Builder methods have precise meaning:

- `emit_*` records a row that is already target-legal.
- `expand_*` records a helper row that must pass through provider-free
  expansion before it is appended to the source row's final bytecode sequence.
- `allocate()` creates a symbolic `TempId`.
- `release(temp)` records the end of a symbolic temp lifetime.
- `finalize()` returns the recipe for materialization.

Pure recording operations are infallible. Fallible operations are limited to
places where failure can actually occur: operand decoding, symbolic temp
allocation, recursive materialization, target validation, capacity checks, and
real virtual-register allocation/release.

### Recipe Data Model

`ExpandedInstructionSequence` is a recipe, not finalized bytecode:

```rust
pub(super) struct ExpandedInstructionSequence {
    source: NormalizedInstruction,
    ops: Vec<ExpansionOp>,
}

pub(super) enum ExpansionOp {
    Emit(RowTemplate),
    Expand(RowTemplate),
    Allocate(TempId),
    Release(TempId),
}
```

The operation meanings are fixed:

- `Emit` appends a row that is already legal final Jolt bytecode.
- `Expand` records a helper row that must be recursively lowered by the central
  provider-free driver.
- `Allocate` creates a symbolic temp lifetime.
- `Release` ends a symbolic temp lifetime.

Rows inside recipes use explicit operand provenance:

```rust
pub(super) struct TempId(u8);

pub(super) enum RegisterOperand {
    Register(u8),
    Temp(TempId),
}

pub(super) const fn reg(register: u8) -> RegisterOperand {
    RegisterOperand::Register(register)
}

impl TempId {
    pub(super) const fn operand(self) -> RegisterOperand {
        RegisterOperand::Temp(self)
    }
}
```

This keeps lowerers close to assembly while still making the register phase
visible at every row.

## Register Budget

Jolt has 32 architectural RISC-V registers and 96 virtual registers:

```text
x0..x31      architectural RISC-V registers
vr32..vr39   reserved persistent virtual registers
vr40..vr47   instruction expansion temps
vr48..vr127  registered-inline temps
```

The expander enforces this split exactly:

- `NUM_RESERVED_VIRTUAL_REGISTERS = 8`
- `NUM_VIRTUAL_INSTRUCTION_REGISTERS = 8`
- `ExpansionAllocator::allocate()` uses only `vr40..vr47`.
- `ExpansionAllocator::allocate_for_inline()` uses only `vr48..vr127`.

`TempId` is symbolic recipe syntax, but each live temp is materialized through
`ExpansionAllocator::allocate()`. The symbolic temp namespace is therefore
bounded by `NUM_VIRTUAL_INSTRUCTION_REGISTERS`, not by the full `u8` range. A
ninth symbolic temp is rejected while building the recipe.

The explicit `reg(x)` and `temp.operand()` spelling is intentional. It makes
operand provenance visible in lowerers: decoded architectural registers and
symbolic allocator temps have different lifetimes and different side-effect
implications.

## Materialization

`ExpansionState` owns the real `ExpansionAllocator` while interpreting
`ExpansionOp` recipes. It is responsible for:

- allocating real virtual registers for symbolic temps;
- rejecting duplicate, unallocated, and leaked symbolic temps;
- recursively expanding helper rows recorded with `expand_*`;
- enforcing recursion depth;
- preserving inline-provider isolation;
- stamping `is_first_in_sequence`, `virtual_sequence_remaining`, and
  compressed-tail metadata;
- rejecting source-only rows in final bytecode;
- enforcing final sequence capacity.

The lowerers stay declarative. The materializer owns stateful behavior.

### Metadata Policy

Sequence metadata is owned by the materializer:

- pass-through rows remain unchanged;
- synthetic sequences are stamped as one source-row expansion;
- recursive helper expansions participate in the outer source sequence;
- `is_first_in_sequence` and `virtual_sequence_remaining` are derived centrally;
- `is_compressed` is attached to the final row of the expanded sequence according
  to the existing compressed-tail policy.

Concrete lowerers should not mutate sequence metadata directly.

## Inline Provider Boundary

Registered inlines are expanded through `InlineExpansionProvider`.

Provider output is intentionally outside the provider-free grammar because
registered inlines may need tracer-side registration, advice generation, and
large inline-specific virtual-register use. The public expansion entry point
still validates provider rows, appends reset rows for inline registers that must
be cleared, and stamps the resulting sequence.

Inline register clearing uses the same allocator partition as tracer:
instruction expansion temps do not borrow from the inline register pool, and
inline registers do not consume `vr40..vr47`.

### Follow-Up: Systematic Inline Expansion

Registered inlines are still the least systematic part of the expansion stack.
Today they are built through tracer-side `InstrAssembler` and `InlineOp`
registrations:

- inline metadata is registered by `(opcode, funct3, funct7)`;
- sequence builders emit tracer `Instruction` values through generic
  `emit_r::<Op>`, `emit_i::<Op>`, `emit_s::<Op>`, and similar helpers;
- inline builders allocate from `allocate_for_inline()`, not from the eight
  instruction-temp registers used by provider-free source-row expansion;
- `finalize_inline()` appends zeroing rows for the inline virtual registers that
  were allocated;
- advice-producing inlines separately provide advice values during tracing.

The current PR should keep that provider boundary. A follow-up PR can make
inlines systematic without mixing them into provider-free expansion all at once.
The clean direction is:

1. Define an inline recipe layer parallel to `ExpandedInstructionSequence`.
   It should use explicit inline virtual-register operands and inline-local
   allocation, but keep the larger `vr48..vr127` register pool.
2. Split inline declarations into metadata, bytecode recipe construction, and
   advice production. Metadata should remain registration-friendly; bytecode
   construction should be testable without a live CPU; advice production can
   keep the CPU/MMU dependency.
3. Reuse the same final-row validation and metadata stamping policy as
   provider-free expansion. Inline recipes should not hand-stamp sequence fields.
4. Preserve `finalize_inline()` semantics: every inline register allocated from
   the inline pool must be zeroed exactly once at the end of the inline sequence.
5. Add parity fixtures per inline family using compact hashes, not huge expanded
   row JSON.
6. Move one small inline first, preferably an advice-store style inline or a
   compact cryptographic helper, before attempting SHA2, Keccak, Blake, or field
   arithmetic inlines.

This would give inlines the same compiler-style shape as provider-free
expansion while preserving the important distinction between source-row
expansion temps and large inline working sets.

## Serialization Boundary

`jolt-riscv` and `jolt-program` expose a default `serialization` feature for
normal workspace builds. `serde` and `ark-serialize` derives are gated behind
that feature.

No-default builds keep the provider-free expansion core independent of
serialization crates:

```bash
cargo clippy -p jolt-program --no-default-features -q --all-targets -- -D warnings
```

Workspace crates that need serialized preprocessing, tracing, or proving data
opt back into the feature explicitly.

## Validation

The expander is validated by focused unit tests and by compact parity checks:

- allocator partition tests for instruction temps and inline temps;
- symbolic temp tests for duplicate allocation, unallocated use, unallocated
  release, leak detection, and the exact eight-temp recipe bound;
- metadata stamping and source-only target-legality tests;
- `rd = x0` behavior for side-effecting and side-effect-free instructions;
- inline-provider validation and reset-row tests;
- LR/SC RAM-range and CSR-zero rejection tests;
- a compact hash fixture covering provider-free expansion parity.

When the compact parity fixture changes, treat it as a semantic review event:

- record the baseline commit or intended semantic change;
- inspect row-level diffs for each affected instruction family;
- regenerate from deterministic serialized `Vec<NormalizedInstruction>` bytes;
- rerun the dedicated parity test and the `muldiv` e2e checks.

The primary e2e correctness checks remain:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

## Extraction Status

Extraction is informational. It is useful for finding hidden coupling and
dependency creep, but production Rust should remain idiomatic.

Current Hax command:

```bash
cargo hax -C -p jolt-program --no-default-features \; into \
  --output-dir /tmp/jolt-program-hax-lean-ergonomic \
  -i '-** +jolt_program::expand::**' \
  lean
```

The command emits Lean for the selected expansion namespace. The generated file
is currently about 14k lines. It no longer includes serde or ark-serialization
dependencies in the no-default configuration, and it reflects the exact
instruction-temp register bound.

The emitted Lean does not typecheck end-to-end in a scratch Lake project. The
current blockers are in extraction packaging and prelude coverage:

- missing or stubbed cross-crate models for `common` and `jolt-riscv`;
- missing Hax Lean models for some `Vec` operations;
- missing iterator/fold helpers such as `Iterator.position` and `fold_return`;
- Lean typeclass/universe synthesis failures around generated derive support;
- downstream unknown identifiers after earlier environment failures.

These blockers are not production Rust requirements. The next extraction work
should improve the Lean environment or upstream Hax/Aeneas models rather than
rewriting clear Rust into extractor-specific control flow.

## Remaining Work Outside This PR

- Make registered inlines systematic using the inline recipe plan above.
- Decide whether very large lowerers such as division and AMO helpers should be
  split further for human readability. This should be motivated by code quality,
  not by extractor output size alone.
- Continue improving Hax/Aeneas support in a maintained Lean environment:
  multi-crate models for `common` and `jolt-riscv`, standard-library models for
  `Vec`/iterator/fold helpers, and generated derive/typeclass support.
- Add row-diff tooling around the compact parity fixture so fixture updates are
  easier to review.
- Split decoded source instructions from final Jolt bytecode rows across
  `jolt-riscv`, `jolt-program`, and `tracer`. This follow-up should change
  tracer's conversion traits and concrete instruction APIs before shrinking
  `JoltInstructionKind`.

## Implementation Checklist

- [x] RV64-only program expansion boundary.
- [x] `LookupInstructionKind` removed; `LookupInstruction` remains as the typed
      lookup-backed view.
- [ ] True source-vs-final instruction split. Deferred to a tracer-aware
      follow-up PR.
- [x] `ExpansionBuilder` recipe API.
- [x] `ExpansionState` materializer with owned allocator state.
- [x] Exact virtual-register partition for reserved, instruction-temp, and
      inline-temp registers.
- [x] Explicit symbolic temp lifetime validation.
- [x] Infallible pure recipe-recording methods.
- [x] Central target legality and metadata stamping.
- [x] Inline-provider validation and reset-row handling.
- [x] Serialization feature boundary.
- [x] Compact provider-free expansion parity fixture.
- [x] Informational Hax extraction pass.

## Acceptance Criteria

- [x] No live RV32/SV32 tracer execution path remains.
- [x] ELF32/RV32 inputs are rejected before expansion.
- [x] Lookup-backed instruction identity uses `LookupInstruction` rather than a
      separate kind enum.
- [ ] Decoded-source and expanded-bytecode instruction identities are fully
      separated. Deferred because tracer still uses `NormalizedInstruction` and
      the concrete `Instruction` enum for both decoded source instructions and
      expanded Jolt rows.
- [x] The old production `InstrAssembler<'a>` expansion path is removed from
      `jolt-program::expand`.
- [x] Concrete provider-free lowerers return `ExpandedInstructionSequence`
      recipes and do not accept `&mut ExpansionAllocator`.
- [x] The central driver is the only provider-free component that recursively
      expands helper rows.
- [x] Symbolic temp lifetimes are represented in recipe data and materialized
      centrally.
- [x] The exact eight-register instruction-temp pool is enforced before
      materialization.
- [x] Provider-free expansion rejects `Inline`; registered inline support stays
      behind `InlineExpansionProvider`.
- [x] `jolt-program::expand` remains independent of tracer CPU execution,
      prover, transcript, PCS, and ELF parser dependencies.
- [x] Formatting, clippy in host and host+zk modes, focused expansion tests, and
      `muldiv` e2e tests pass.
