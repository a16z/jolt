# Spec: Inline Expansion Grammar

| Field       | Value |
|-------------|-------|
| Author(s)   | Quang Dao |
| Created     | 2026-05-13 |
| Status      | proposed |
| PR          | follow-up to [#1522](https://github.com/a16z/jolt/pull/1522) |

## Summary

PR #1522 makes inline opcodes source-only and removes `Inline` from the final
Jolt instruction universe. That fixes the phase identity problem, but registered
inline expansion still uses the older tracer-oriented `InstrAssembler` API:
inline crates emit concrete `tracer::Instruction` values through an
inventory-discovered callback, and `tracer` then normalizes those instructions
back into final `JoltInstructionRow` values. This follow-up should move static
registered inline expansion onto the same explicit recipe/materializer grammar
used by `jolt-program::expand` for built-in source-only rows, while keeping
runtime advice generation in tracer-owned host code.

The result should be one static bytecode expansion substrate:

```text
built-in source-only expansion ─┐
                                v
                         ExpansionOp grammar
                                v
registered inline expansion ────┘
                                |
                                v
                   one materializer / one metadata path
                                |
                                v
                         JoltInstructionRow
```

## Intent

### Goal

Replace the tracer `InstrAssembler` static inline expansion path with a
`jolt-program` recipe builder that emits final/source row templates and is
materialized by the existing `ExpansionState`.

The implementation should introduce or generalize these boundaries:

- an inline-capable expansion builder in `jolt-program::expand` that can express
  the helper operations currently emitted by `tracer::utils::inline_helpers::InstrAssembler`;
- an inline provider contract whose static expansion output is an explicit
  recipe or final row sequence, not `Vec<tracer::Instruction>`;
- an SDK-facing `InlineOp` API that lets shipped inline crates declare static
  sequences without importing tracer instruction types;
- a separate runtime advice path for inlines that need CPU/memory-dependent
  witness values.

### Invariants

- Static inline expansion must emit the same final `JoltInstructionRow` sequence
  as the current `InstrAssembler` path, modulo intentionally documented bug
  fixes locked by equivalence tests.
- Registered inline source identity remains `(opcode, funct3, funct7)` plus
  `InlineExtension`; there is still no final `JoltInstructionKind::Inline`.
- Profile gating remains two-stage: `SourceExtension::JoltInline` permits inline
  source rows, and `InlineExtension` permits a particular registered inline
  package.
- Sequence metadata remains centralized. Final rows emitted by inline expansion
  must have the same `is_first_in_sequence`, `virtual_sequence_remaining`, source
  address, and compressed-tail metadata as today.
- Inline virtual-register allocation must preserve the current register layout:

  ```text
  vr32..vr39: reserved persistent virtual registers
  vr40..vr47: per-source-instruction temporary registers
  vr48..    : provider-allocated inline registers reset at inline finalization
  ```

- Static inline expansion must not depend on `Cpu`, memory devices, advice
  tapes, tracer `Cycle`, prover code, transcript code, file I/O, global mutable
  state, `Arc<Mutex<_>>`, or RAII drop semantics.
- Normal expansion errors should be represented as `ExpansionError`, not panics.
- Runtime inline advice generation may remain host/tracer-owned and may depend
  on `Cpu`; it must not be part of the formalization-critical static expansion
  grammar.
- The implementation PR must do a full static inline expansion cutover for all
  shipped inlines. Do not leave some inline packages on the old static
  `InstrAssembler -> tracer::Instruction` path as a compatibility layer.

### Non-Goals

- This spec does not require making runtime advice generation Lean-extractable.
  Advice depends on CPU and memory state and remains an execution concern.
- This spec does not require changing inline guest SDK semantics or inline
  opcode assignments.
- This spec does not require changing final Jolt bytecode semantics or lookup
  tables.
- This spec does not require removing tracer's execution-time virtual register
  allocator if it is still needed for concrete tracing after static expansion is
  cut over.
- This spec does not require replacing inventory registration in the same PR if
  the registration surface can call an extraction-friendly static builder.
  Inventory is a host discovery mechanism; it must not hide static expansion
  semantics inside callback-only code.

## Evaluation

### Acceptance Criteria

- [ ] `InlineOp::build_sequence` no longer returns `Vec<tracer::Instruction>`.
- [ ] Static inline expansion no longer imports or uses
  `tracer::utils::inline_helpers::InstrAssembler`.
- [ ] `TracerInlineExpansionProvider` no longer converts provider output through
  `try_jolt_instruction_row()` on tracer instructions.
- [ ] All shipped inline crates emit static expansion through the new
  `jolt-program` inline recipe/final-row builder:
  - [ ] `jolt-inlines/sha2`
  - [ ] `jolt-inlines/blake2`
  - [ ] `jolt-inlines/blake3`
  - [ ] `jolt-inlines/keccak256`
  - [ ] `jolt-inlines/bigint`
  - [ ] `jolt-inlines/secp256k1`
  - [ ] `jolt-inlines/grumpkin`
  - [ ] `jolt-inlines/p256`
- [ ] Inline virtual-register resets are produced through
  `ExpansionAllocator::take_registers_for_reset()` or an equivalent plain-owned
  allocator path, not through `VirtualRegisterAllocator::get_registers_for_reset()`.
- [ ] Static inline expansion has equivalence tests against the old emitted
  sequences or checked fixtures before deleting the old path.
- [ ] The old static `InstrAssembler` path is deleted or narrowed to
  execution-only code that is no longer used for program bytecode expansion.
- [ ] Existing inline tests continue to pass.

### Testing Strategy

Run the normal checks:

```bash
cargo fmt -q
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo nextest run -p jolt-program --cargo-quiet
cargo nextest run -p tracer --cargo-quiet --features test-utils
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

Add focused tests:

- one equivalence test per inline package comparing old static output to the new
  recipe output before deleting the old path;
- profile tests proving disabled `InlineExtension`s still reject registered
  inline rows;
- allocator tests proving inline registers are reset and ordinary instruction
  temps are released;
- metadata tests proving inline sequences are stamped by the same central
  metadata path as built-in expansion;
- a negative test proving provider-free expansion still rejects
  `SourceInstructionKind::Inline`.

### Performance

Static expansion should not regress guest program construction measurably. The
new builder may be more structured than `InstrAssembler`, but it should avoid
extra recursive conversions through tracer `Instruction` values. If performance
is measured, benchmark at least one inline-heavy guest or direct expansion of
all shipped inline fixtures before and after the migration.

## Design

### Architecture

Current registered inline static expansion path:

```text
SourceInstruction::Inline
  |
  v
TracerInlineExpansionProvider
  |
  +-- inventory lookup by (opcode, funct3, funct7)
  +-- profile.supports_inline(extension)
  +-- build_sequence(InstrAssembler, FormatInline)
  v
Vec<tracer::Instruction>
  |
  +-- try_jolt_instruction_row()
  +-- JoltInstruction::try_from(row)
  v
Vec<JoltInstruction>
  |
  v
stamp_inline_sequence(...)
  |
  v
Vec<JoltInstructionRow>
```

Target static expansion path:

```text
SourceInstruction::Inline
  |
  v
InlineExpansionProvider
  |
  +-- registration lookup by (opcode, funct3, funct7)
  +-- profile.supports_inline(extension)
  +-- build_recipe(InlineExpansionBuilder, InlineOperands)
  v
InlineExpansionRecipe / ExpandedInstructionSequence
  |
  v
ExpansionState materializer
  |
  +-- allocate/release explicit temps
  +-- append inline-register reset rows
  +-- validate final target rows
  +-- stamp sequence metadata centrally
  v
Vec<JoltInstructionRow>
```

Runtime advice remains separate:

```text
tracer executes inline source instruction
  |
  +-- registration lookup
  +-- build_advice(InlineAdviceContext, operands, Cpu)
  +-- trace expanded rows with advice populated
```

The static recipe builder should be close to the current `ExpansionBuilder`
style:

```rust
let mut asm = InlineExpansionBuilder::new(source)?;
let a = asm.allocate_inline()?;
let b = asm.allocate_inline()?;

asm.emit_i(JoltInstructionKind::LD, a, reg(rs1), offset);
asm.emit_r(JoltInstructionKind::XOR, b, a, reg(rs2));
asm.emit_s(JoltInstructionKind::SD, reg(rd), b, 0);

asm.release_inline(a)?;
asm.release_inline(b)?;
asm.finalize()
```

The exact API can differ, but it should preserve the important properties:
plain owned state, explicit allocation/release, `Result` errors, row-template
output, and no tracer instruction dependency.

### Modules To Touch

Expected core changes:

- `crates/jolt-program/src/expand/grammar.rs`
  - expose or generalize `ExpansionOp`, `RowTemplate`, `RegisterOperand`, and
    `ExpansionBuilder` enough for inline recipes;
  - add inline-specific helpers only where they express real inline invariants.
- `crates/jolt-program/src/expand/materialize.rs`
  - materialize inline recipes with the same row validation and metadata
    discipline as built-in recipes;
  - avoid a second metadata implementation.
- `crates/jolt-program/src/expand/allocator.rs`
  - make inline register allocation and reset reporting the single static
    expansion allocator path.
- `crates/jolt-program/src/expand/mod.rs`
  - revise `InlineExpansionProvider` so static provider output is not
    `Vec<JoltInstruction>` produced from tracer instructions;
  - keep provider-free expansion rejecting inline source rows.
- `jolt-inlines/sdk/src/host.rs`
  - revise `InlineOp` and registration macros;
  - split static sequence construction from runtime advice construction;
  - move reusable assembler helpers that are static-expansion-safe into the new
    builder API or thin extension traits over it.
- `tracer/src/instruction/inline.rs`
  - simplify `TracerInlineExpansionProvider`;
  - keep runtime tracing/advice behavior, but stop using tracer instruction
    normalization as the static expansion output path.
- `tracer/src/utils/inline_helpers.rs`
  - delete, move, or narrow the old `InstrAssembler` once shipped inlines no
    longer depend on it for static expansion.
- `tracer/src/utils/virtual_registers.rs`
  - keep only execution-time allocator behavior that remains necessary; static
    expansion should use `ExpansionAllocator`.

Expected inline package migrations:

- `jolt-inlines/sha2/src/sequence_builder.rs`
- `jolt-inlines/blake2/src/sequence_builder.rs`
- `jolt-inlines/blake3/src/sequence_builder.rs`
- `jolt-inlines/keccak256/src/sequence_builder.rs`
- `jolt-inlines/bigint/src/multiplication/sequence_builder.rs`
- `jolt-inlines/secp256k1/src/sequence_builder.rs`
- `jolt-inlines/grumpkin/src/sequence_builder.rs`
- `jolt-inlines/p256/src/sequence_builder.rs`

### Expected Change Size

Approximate size:

```text
pilot only:        400-900 LoC changed
full cutover:    2,000-4,000 LoC changed
worst case:      5,000+ LoC if old tracer helpers are deleted completely
```

The implementation PR should target the full static expansion cutover. A pilot
can be an internal sequence of commits, but the PR should not merge with only
some inline packages using the new grammar and others still using the old static
tracer builder path.

### Alternatives Considered

1. Keep the current provider adapter.

   This preserves working behavior and keeps this PR smaller, but leaves static
   inline expansion in callback-heavy tracer code and blocks a clean extraction
   story.

2. Move all inline registration and advice into `jolt-program`.

   This would make `jolt-program` depend on tracer CPU/memory/advice concerns.
   Static sequence expansion belongs near program construction; advice
   generation does not.

3. Add a second inline-only materializer.

   This would duplicate sequence metadata, legality checks, and allocator
   behavior. Inline expansion should share the existing `ExpansionState` path
   wherever possible.

## Documentation

No Jolt book update is required unless inline registration is a public SDK
surface. The implementation should update:

- `specs/source-jolt-instruction-split.md` to mark the follow-up complete;
- `specs/bytecode-expansion-crate.md` if it still describes the old
  `InstrAssembler`-based registered inline bridge as the current compromise;
- inline SDK docs/comments showing how to implement an `InlineOp`.

## Execution

Suggested implementation order:

1. Add the new inline recipe/builder API in `jolt-program::expand`.
2. Add a temporary test harness that compares old `InstrAssembler` output with
   new recipe output for one inline package.
3. Port the smallest representative inline first, then port the remaining
   shipped inline packages.
4. Split runtime advice construction away from static sequence construction.
5. Remove the static `InstrAssembler -> tracer::Instruction` provider path.
6. Run the full validation stack and inline package tests.

## References

- [PR #1522](https://github.com/a16z/jolt/pull/1522)
- [`specs/source-jolt-instruction-split.md`](source-jolt-instruction-split.md)
- [`specs/bytecode-expansion-crate.md`](bytecode-expansion-crate.md)
