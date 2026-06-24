# Spec: Inline Expansion Grammar

| Field       | Value |
|-------------|-------|
| Author(s)   | Quang Dao |
| Created     | 2026-05-13 |
| Revised     | 2026-05-18 |
| Status      | proposed |
| PR          | follow-up to [#1522](https://github.com/a16z/jolt/pull/1522) |

## Summary

PR #1522 makes inline opcodes source-only and removes `Inline` from the final
Jolt instruction universe. That fixes the phase identity problem, but registered
inline expansion still uses a tracer-oriented static path:

```text
SourceInstruction::Inline
  -> tracer-owned registration lookup
  -> build_sequence(InstrAssembler, FormatInline)
  -> Vec<tracer::Instruction>
  -> try_jolt_instruction_row()
  -> JoltInstruction::try_from(row)
  -> stamp_inline_sequence(...)
  -> Vec<JoltInstructionRow>
```

This follow-up moves registered static inline expansion onto the same explicit
recipe/materializer grammar used by `jolt-program::expand` for built-in
source-only rows. Runtime advice generation remains tracer-owned host behavior,
but the formal static bytecode expansion path must be tracer-free.

The target shape is:

```text
built-in source-only expansion ----+
                                   |
registered inline expansion -------+--> ExpansionOp grammar
                                             |
                                             v
                   one materializer / one metadata path
                                |
                                v
                         JoltInstructionRow
```

## Intent

### Goals

- Replace the static registered-inline `InstrAssembler -> tracer::Instruction`
  path with a `jolt-program` recipe builder and materializer path.
- Make static inline sequence construction independent of tracer instruction
  types, concrete tracer cycles, CPU state, memory devices, advice tapes, and
  runtime trace formatting.
- Preserve the final `JoltInstructionRow` stream emitted by the old static path,
  except for intentional documented bug fixes locked by fixtures.
- Keep runtime inline advice generation as an execution-time host/tracer path:
  the tracer executes the source inline row, selects the same registration key,
  calls the registration's advice provider with operands plus CPU/memory context,
  and checks the returned advice against the statically materialized
  `VirtualAdvice` rows.
- Port every shipped inline package in one full static cutover. Do not merge a
  partial state where some shipped inlines still use the old static tracer
  builder.
- Keep sequence metadata, recursive helper expansion, profile legality, virtual
  register allocation, inline reset rows, and target validation centralized.
- Make registered static inline recipes visible to the same extraction-critical
  `jolt-program::expand` surface as built-in source-only expansion.

### Non-Goals

- Do not change inline guest SDK semantics or opcode assignments.
- Do not change final Jolt bytecode semantics, lookup tables, or proof
  constraints.
- Do not make runtime advice generation Lean-extractable in this PR.
- Do not require Hax or Aeneas to produce typechecked Lean for all shipped
  inline recipes in this PR. The implementation should remove the tracer
  boundary that currently hides static inline recipes from extraction, but
  extractor backend and standard-library limitations remain separate follow-up
  work.
- Do not move CPU, RAM, trace-cycle, or advice execution semantics into
  `jolt-program`.
- Do not replace inventory registration if the registration surface calls a
  tracer-free static recipe builder. Inventory is a host discovery mechanism,
  not the place where static expansion semantics may be hidden.
- Do not keep a separate tracer-owned static virtual-register allocator,
  `VirtualRegisterGuard`, or RAII reset path for program bytecode expansion.
- Do not introduce compatibility wrappers, deprecated aliases, or alternate
  static expansion paths.

## Static API Contract

Static inline expansion must be tracer-free at the type boundary.

`StaticInlineOp::build_sequence` or the equivalent static sequence contract must
use only `jolt-program`, `jolt-riscv`, and SDK-owned static recipe types. No
public item, helper trait, macro expansion, generic bound, or re-export used by
static sequence construction may name:

- `tracer::`
- `Cpu`
- `FormatInline`
- `Instruction`
- `RISCVInstruction`
- `RISCVTrace`
- `RISCVCycle`
- `InstrAssembler`
- tracer instruction marker structs re-exported through the inline SDK

The inline SDK should expose separate surfaces:

```rust
trait StaticInlineOp {
    const OPCODE: u8;
    const FUNCT3: u8;
    const FUNCT7: u8;
    const EXTENSION: InlineExtension;
    const NAME: &'static str;

    fn build_sequence(
        builder: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError>;
}

trait InlineAdviceProvider {
    fn build_advice(
        context: InlineAdviceContext<'_>,
    ) -> Result<Option<InlineAdvice>, InlineAdviceError>;
}
```

The exact names may differ, but the split is required:

- Static metadata and recipe construction live in a tracer-free contract.
- Runtime advice lives in a host-only contract that may depend on tracer `Cpu`.
- The static contract must compile without pulling in tracer.
- Host registration may pair a `StaticInlineOp` with an optional
  `InlineAdviceProvider`.

The shipped inline crates may use SDK-owned aliases for instruction kinds, but
those aliases must resolve to `jolt_riscv::{SourceInstructionKind,
JoltInstructionKind}` or recipe helper types, not tracer instruction structs.

## Inline Identity And Operands

Registered inline source identity is the decoded custom-instruction key:

```text
(opcode, funct3, funct7)
```

For a decoded inline row, the authoritative static key is
`SourceInstructionRow.inline: Option<SourceInlineKey>`.
`InlineExtension` is registration/profile metadata resolved after lookup; it is
not part of the ELF/source key unless the instruction encoding is changed to
carry it explicitly. Duplicate registered triples are therefore invalid even if
the registrations name different `InlineExtension`s.

Required rules:

- `SourceInstructionKind::Inline` must have `row.inline = Some(key)`.
- `row.operands.rs1`, `row.operands.rs2`, and `row.operands.rd` must be present.
- `row.operands.rd` is the inline `rs3` operand.
- `row.operands.imm` is not an inline identity source after decode.
- Static inline lookup must not convert the row into `tracer::Instruction`,
  `INLINE`, `FormatInline`, or any tracer format type.
- Malformed inline rows return `ExpansionError`, not panics.

`InlineOperands` should be a small tracer-free value derived from the decoded
source row:

```rust
pub struct InlineOperands {
    pub rs1: u8,
    pub rs2: u8,
    pub rs3: u8,
}
```

## Registration And Profile Gating

Every registration must declare:

- opcode
- funct3
- funct7
- `InlineExtension`
- human-readable name
- static recipe builder
- optional runtime advice provider

Registration keys must be unique. Duplicate `(opcode, funct3, funct7)` entries
must be rejected deterministically or covered by a test that proves the linked
registration table contains no duplicates.

This PR does not change the guest bytecode trust model. It moves static inline
expansion onto the `jolt-program` recipe/materializer path under the existing
SDK/prover assumptions. Hardening arbitrary attacker-supplied custom inline
opcodes is a separate design, not part of this cutover.

Registered inline expansion must require both:

```rust
profile.supports_source(SourceInstructionKind::Inline)
profile.supports_inline(registration.extension)
```

Neither check implies the other. The final rows emitted by materialization must
also pass target legality under the active profile.

Provider-free expansion continues to reject `SourceInstructionKind::Inline` with
`ExpansionError::InlineProviderRequired`.

## Recipe And Builder Contract

Provider output must be a recipe, not finalized bytecode. The provider may emit
final target rows inside the recipe through `Emit` operations, but it must not
return `Vec<JoltInstruction>`, `Vec<JoltInstructionRow>`, or
`Vec<tracer::Instruction>` directly.

Required recipe model:

```rust
pub struct ExpandedInstructionSequence {
    source: SourceInstructionRow,
    ops: Vec<ExpansionOp>,
}

pub enum ExpansionOp {
    Emit(RowTemplate),
    Expand(SourceInstructionRowTemplate),
    Allocate(TempId),
    Release(TempId),
    AllocateInline(InlineTempId),
    ReleaseInline(InlineTempId),
}
```

The exact representation may differ, but the semantics must be equivalent:

- `Emit` records a row that is already target-legal final Jolt bytecode.
- `Expand` records a source helper row that must be recursively lowered by the
  central provider-free driver.
- `Allocate` and `Release` describe symbolic temps in the `vr40..vr47` pool.
- Inline allocation describes provider-owned inline registers in the `vr48..`
  pool.
- Source-expansion temps and registered-inline temps are separate logical
  domains. The spec names them `TempId` and `InlineTempId`; an implementation may
  use one internal enum or id type only if the domain is explicit and the public
  builder APIs do not allow inline temps and materializer source temps to be used
  interchangeably by accident.
- The builder records intent. The materializer owns stateful allocation,
  recursive expansion, reset-row generation, validation, and metadata stamping.

The public static builder should expose concrete tracer-free methods such as:

```rust
emit_r(kind: JoltInstructionKind, rd, rs1, rs2)
emit_i(kind: JoltInstructionKind, rd, rs1, imm)
emit_s(kind: JoltInstructionKind, rs1, rs2, imm)
emit_load(kind: JoltInstructionKind, rd, rs1, imm)
emit_advice(rd)
expand_r(kind: SourceInstructionKind, rd, rs1, rs2)
expand_i(kind: SourceInstructionKind, rd, rs1, imm)
```

These methods may be grouped or named differently, but static call sites must
not recover tracer-style generic emission over tracer instruction structs.

Normal expansion failures must be represented as `ExpansionError`, not panics or
unchecked `unwrap`/`expect`.

## Register And Reset Contract

The static expander must preserve the existing register layout:

```text
x0..x31       architectural RISC-V registers
vr32..vr39    reserved persistent virtual registers
vr40..vr47    materializer-owned source-expansion temps
vr48..vr127   registered-inline temps
```

Provider static recipes may allocate inline registers only through the inline
builder API. They must not write directly to reserved virtual registers or to
ordinary architectural registers other than `x0`.

Static inline rows with `rd` must satisfy one of these cases:

- `rd == x0`
- `rd` is a live provider-allocated inline register from `vr48..`
- `rd` is a materializer-allocated helper temp from `vr40..vr47` inside a
  recursively expanded helper sequence

Any other inline write target must return
`ExpansionError::InvalidInlineWriteTarget`.

Inline register lifetimes must not rely on Rust `Drop`, RAII guards,
`Arc<Mutex<_>>`, raw pointers, `unsafe`, or hidden allocator aliases. The
implementation must use a builder-owned lifetime model:

- `finish_inline` is the explicit lifetime boundary for one registered inline
  recipe.
- `release_inline` is available for providers that want to reuse an inline temp
  before `finish_inline`.
- Finalization centrally ends any remaining builder-owned inline lifetimes and
  records which real inline registers need reset rows.
- Finalization must error if a provider tries to release an unknown temp, reuse a
  released temp, or smuggle an inline temp into a source-helper temp slot.

Reset rows are produced by the materializer from the static inline allocator
state, not by tracer RAII helpers. Reset rows are appended after functional rows
and before metadata stamping. Every provider-allocated inline register that may
contain a value must be reset exactly once with:

```text
ADDI rd, x0, 0
```

## Materialization And Metadata

Inline materialization must share the same `ExpansionState` machinery used by
built-in source-only expansion. It must:

- allocate real registers for symbolic temps;
- materialize inline temp allocation through `ExpansionAllocator`;
- recursively expand helper source rows;
- append inline reset rows from the central allocator/materializer path;
- reject illegal target rows under the active profile;
- reject empty sequences;
- reject sequences too long to encode `virtual_sequence_remaining`;
- flatten functional rows, recursively expanded helper rows, and reset rows;
- stamp sequence metadata exactly once over the flattened sequence.

Metadata rules:

- Every row uses the top-level inline source address.
- Exactly one row has `is_first_in_sequence = true`.
- `virtual_sequence_remaining` counts down over the full flattened sequence.
- `is_compressed` is set only on the final flattened row.
- If reset rows are appended, a compressed inline marks the final reset row as
  compressed.
- Provider-written or pre-stamped metadata must not be trusted.

## Runtime Advice Contract

Inline runtime advice is a separate inline witness advice channel. It is not the
same thing as the SDK `TrustedAdvice`/`UntrustedAdvice` tape.

Each registration owns one static recipe and one optional advice provider under
the same `(opcode, funct3, funct7)` source identity. `InlineExtension` remains
profile metadata checked after lookup.

The runtime flow is:

```text
tracer executes SourceInstructionKind::Inline
  -> read SourceInstructionRow.inline and operands
  -> lookup the same registration used by static expansion
  -> profile.supports_source(SourceInstructionKind::Inline)
  -> profile.supports_inline(registration.extension)
  -> build_advice(InlineAdviceContext { operands, cpu, memory helpers })
  -> attach returned values to the materialized VirtualAdvice rows in order
```

Static expansion is authoritative for where advice is consumed. The advice
provider computes values only; it must not emit instructions, allocate registers,
or decide the static row sequence.

Advice rules:

- `InlineAdviceContext` may expose inline operands and CPU/memory read helpers.
- `InlineAdviceContext` must not expose `InlineExpansionBuilder` or allocation.
- The number and order of runtime advice values must match the materialized
  `VirtualAdvice` rows in the static sequence.
- `None` is valid only when the static sequence contains zero `VirtualAdvice`
  rows.
- Too few or too many values must produce a structured inline advice error where
  the surrounding API can return one. Existing trace APIs that cannot return a
  result may wrap that structured error, but the mismatch must not be silent.
- Advice generation remains profile-gated by the same registration selected for
  static expansion.

## Helper Semantics

The old `InstrAssembler` contains reusable helper logic that shipped inline
crates depend on. Any helper moved to the new static SDK must have a documented
tracer-free contract. At minimum:

| Helper family | Required contract |
|---------------|-------------------|
| `Value::{Imm, Reg}` | State whether immediates are `u64`, how they map into row immediates, and when constant folding is allowed. |
| `bin`, `add`, `xor`, `and` | Specify R-vs-I row selection, wrapping arithmetic, immediate ordering, and constant-fold results. |
| `srli`, `rotri32`, `rotri`, `rotr64`, `rotl64` | Specify shift/rotate width, mask encoding, zero-shift behavior, and emitted virtual row kinds. |
| `load_paired_u32` | Specify address alignment assumption, clobbered temp register, zero-extension behavior, and emitted rows. |
| `load_paired_u32_dirty` | Specify the dirty upper-bit precondition and which downstream operations may consume it. |
| `store_paired_u32` | Specify clobbered operands, zero-extension, shift/or sequence, and store row. |
| `emit_advice_stores` | Specify exact `VirtualAdvice`/store row order and advice arity contribution. |
| `MulAccExt` helpers | Specify clobbered registers, carry propagation contract, wrapping semantics, and emitted row sequence. |

Equivalence fixtures protect behavior, but helper contracts are still required
so future inline authors know which invariants are intentional.

## Trace File Policy

`.joltinline` trace files are developer fixtures/documentation for inline
sequences. They are not the source of truth for static expansion.

If trace generation remains supported:

- it must consume recipes or materialized `JoltInstructionRow`s directly;
- it must not round-trip through `tracer::Instruction` or
  `try_jolt_instruction_row()` for static sequence generation;
- placeholder replacement must cover address, `rs1`, `rs2`, `rd`/`rs3`,
  compressed-tail metadata, and reset rows;
- fixture generation must use the same static provider path used by program
  bytecode expansion.

## Dependency Contract

`jolt-program` must not depend on `tracer`.

The static inline SDK feature should depend only on:

- `jolt-program`
- `jolt-riscv`
- static inline-local crates needed for constants or pure helper logic

Host/runtime advice features may depend on:

- `tracer`
- `inventory`
- CPU/memory/advice utilities

Static sequence modules in shipped inline crates must not import tracer types
directly or indirectly through SDK re-exports.

## Extraction Impact And Limits

The extraction target for this follow-up is static bytecode expansion semantics,
not runtime execution. The implementation should remove the architectural
blocker that currently hides registered inline recipes behind tracer instruction
construction, `InstrAssembler`, CPU/advice state, RAII register guards, and
inventory callback plumbing.

Static recipe builders for shipped inlines must therefore be ordinary callable
Rust functions over explicit operands and recipe-builder types. They must be
usable from tests or extraction harnesses without constructing a tracer CPU,
executing inventory lookup, or converting through tracer instructions. Inventory
may remain a host registration mechanism, but it must not be the only way to
name the static recipe body.

Within extraction-targeted static recipe bodies, prefer first-order helper
shapes over higher-order generic callback helpers. A helper that takes
`F: FnMut(...)` or a similar generic closure can force extractor backends to
model Rust's `Fn*` traits and associated output constraints even when the helper
only names a fixed instruction schedule. For fixed schedules, use concrete
methods, small loops, or macro-expanded direct calls so the extracted body stays
close to the emitted instruction sequence.

This does not mean that `cargo hax -p jolt-program ...` alone will necessarily
contain every shipped inline recipe. If recipe bodies remain in
`jolt-inlines/*`, an extraction experiment must either extract those crates too
or use an explicit aggregation target that calls their static recipe entry
points. This spec does not require moving all inline recipe bodies into
`jolt-program`; it does require that the bodies be visible through direct,
tracer-free static entry points.

After this cutover, Hax and Charon should be able to see the registered static
recipe bodies in principle. A fully typechecked Lean model is still outside this
PR. Known remaining extractor issues in the current codebase include heap-backed
`Vec` APIs, iterator/`collect` paths, allocator and metadata loops, materializer
borrow loops, generated prelude coverage, and Aeneas support for ordinary Rust
control-flow shapes. Those are follow-up extraction-engineering problems rather
than reasons to keep static inline expansion behind tracer.

The next extraction follow-up should keep that scope narrow. In particular, it
should not require extracting the full `jolt-riscv` typed instruction universe
as the proof surface for inline expansion. Today, `SourceInstructionKind` and
`JoltInstructionKind` are zero-payload instantiations of the same generic typed
instruction enums used by runtime and lookup code, and each marker generated by
`jolt_instruction!` carries generic `Flags`, row conversion, and
`JoltInstructionRowData` impls. That shape is ergonomic Rust, but current Hax
experiments show that it produces a large trait-heavy Lean artifact before
inline expansion semantics are reached. A better follow-up is a small
extraction-facing instruction surface: compact source/final instruction tags,
plain row records, and first-order tag/row helper functions that the
`jolt-program` expansion code can target. The existing marker-type ergonomics
may remain for runtime, lookup, and inline-builder call sites, but extraction
should be able to bypass marker-wrapper trait impls, serde/canonical
serialization, and test-only iterator helpers.

Runtime advice remains deliberately outside the extraction target. Static
recipes may expose the number and order of `VirtualAdvice` rows; CPU-derived
advice values and trace-time memory/register reads remain host/tracer semantics.

## Acceptance Criteria

- [ ] Static recipe construction compiles through a tracer-free SDK surface.
- [ ] `InlineOp::build_sequence` is replaced or split so static sequence
      construction no longer returns `Vec<tracer::Instruction>` and no longer
      names tracer types.
- [ ] Runtime advice generation is a separate host/tracer-owned contract.
- [ ] Static inline lookup consumes `SourceInstructionRow.inline` directly and
      does not convert source rows to tracer `INLINE`/`FormatInline`.
- [ ] Registered inline expansion checks both source inline support and package
      `InlineExtension` support.
- [ ] Registration keys are unique and tested.
- [ ] The cutover preserves the existing guest bytecode trust model; it does not
      add partial raw custom-opcode filtering machinery.
- [ ] Provider output is an `ExpandedInstructionSequence` or equivalent recipe,
      not a direct final row/instruction vector.
- [ ] Every shipped static inline recipe has a direct tracer-free entry point
      usable by tests or extraction harnesses without inventory lookup.
- [ ] Inline reset rows are produced by the central expansion
      allocator/materializer path, not by `VirtualRegisterAllocator` or RAII
      helpers.
- [ ] Any remaining tracer virtual-register helper is runtime/test-only and
      cannot be reached by program bytecode expansion or static inline builders.
- [ ] Metadata stamping is performed once over the flattened inline sequence,
      including reset rows.
- [ ] Static inline rows reject illegal write targets with
      `ExpansionError::InvalidInlineWriteTarget`.
- [ ] All shipped inline crates emit static expansion through the new recipe
      builder:
      - [ ] `jolt-inlines/sha2`
      - [ ] `jolt-inlines/blake2`
      - [ ] `jolt-inlines/blake3`
      - [ ] `jolt-inlines/keccak256`
      - [ ] `jolt-inlines/bigint`
      - [ ] `jolt-inlines/secp256k1`
      - [ ] `jolt-inlines/grumpkin`
      - [ ] `jolt-inlines/p256`
- [ ] Every registered inline key has a checked static expansion fixture.
- [ ] The old static `InstrAssembler` path is deleted or made private to
      execution/test utilities and is unreachable from static inline builders.
- [ ] `jolt-inlines-sdk` does not re-export `InstrAssembler`,
      `tracer::instruction`, `FormatInline`, or other tracer static-builder
      types.
- [ ] `store_trace`/trace fixture generation no longer uses a static
      tracer-instruction roundtrip.
- [ ] Related specs are updated so no document still authorizes the old
      tracer-normalized static inline compromise.

## Testing Strategy

Run the normal checks:

```bash
cargo fmt -q
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo nextest run -p jolt-program --cargo-quiet
cargo nextest run -p tracer --cargo-quiet --features test-utils
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host,zk
git diff --check
```

Run inline package tests:

```bash
cargo nextest run -p jolt-inlines-sdk --features host --cargo-quiet
cargo nextest run -p jolt-inlines-sha2 --features host --cargo-quiet
cargo nextest run -p jolt-inlines-blake2 --features host --cargo-quiet
cargo nextest run -p jolt-inlines-blake3 --features host --cargo-quiet
cargo nextest run -p jolt-inlines-keccak256 --features host --cargo-quiet
cargo nextest run -p jolt-inlines-bigint --features host --cargo-quiet
cargo nextest run -p jolt-inlines-secp256k1 --features host --cargo-quiet
cargo nextest run -p jolt-inlines-grumpkin --features host --cargo-quiet
cargo nextest run -p jolt-inlines-p256 --features host --cargo-quiet
```

Add focused tests:

- Golden fixtures generated from the old path before the API break, one per
  registered `(opcode, funct3, funct7)` key.
- Fixture cases for normal operands, `rd = x0`, aliased operands,
  compressed/uncompressed source rows, and representative source addresses.
- Static parity tests comparing final `JoltInstructionRow`s after metadata
  stamping, not raw helper instructions.
- Profile tests for:
  - `SourceExtension::JoltInline` disabled;
  - package `InlineExtension` disabled;
  - both enabled;
  - unregistered inline key;
  - provider-free inline expansion.
- Malformed row tests for missing `inline`, `rs1`, `rs2`, or `rd`/`rs3`.
- Metadata tests for flattened helper expansions, reset rows, countdown values,
  source address, and compressed-tail behavior.
- Allocator tests for inline register exhaustion, double release, optional reuse
  policy, reset row uniqueness, and illegal write targets.
- Advice tests for each advice-bearing shipped inline, asserting exact
  `VirtualAdvice` count/order and too-few/too-many mismatch behavior.
- Registration table tests proving all linked keys are unique and expected.
- Source identity tests proving duplicate `(opcode, funct3, funct7)` triples are
  rejected even across different `InlineExtension`s, because ELF decode cannot
  disambiguate extension at the source-key level.
- Trace fixture tests proving placeholder replacement covers address, `rs1`,
  `rs2`, `rd`/`rs3`, compressed tail, and reset rows.

Add mechanical checks:

```bash
! rg 'InstrAssembler|inline_helpers|try_jolt_instruction_row|tracer::instruction|FormatInline|RISCVInstruction|RISCVTrace|RISCVCycle' \
  jolt-inlines jolt-inlines/sdk/src
cargo tree -p jolt-program | (! rg '^tracer ')
```

Adjust the denylist if runtime advice modules are split into an explicitly
host-only location. The denylist must still apply to static sequence builders
and static SDK helpers.

## Performance

Static expansion should not measurably regress guest program construction.

Required perf smoke:

- expand all shipped inline fixtures many times through the old path and the new
  recipe/materializer path;
- compare row counts, reset row counts, wall time, and allocation behavior;
- investigate any wall-time regression above 5 percent or any material increase
  in peak allocation count;
- preallocate recipe and row buffers where sequence length is known or easily
  bounded.

This is a smoke test, not a benchmark suite. It exists to catch accidental
quadratic materialization, excessive recursive conversion, or avoidable
allocation churn.

## Implementation Order

1. Generate and check in old-path golden fixtures for every registered inline
   key before changing the static `InlineOp` signature. These fixtures should be
   serialized final `JoltInstructionRow`s or stable hashes of those rows after
   metadata stamping. `.joltinline` trace files remain separate
   documentation/debug output, not the parity oracle.
2. Add tracer-free static recipe traits and builder APIs.
3. Add host-only runtime advice registration/adapters.
4. Move reusable static helper logic from `InstrAssembler` into SDK-owned,
   tracer-free helper traits with documented contracts.
5. Port one small inline and validate fixtures.
6. Port every shipped inline package in one full cutover.
7. Remove static SDK access to `InstrAssembler`, tracer instruction markers,
   `FormatInline`, and tracer-normalized static output.
8. Update trace fixture generation to consume recipes or `JoltInstructionRow`s.
9. Update related specs that still describe the old compromise.
10. Run the full validation stack.

## Modules To Touch

Expected core changes:

- `crates/jolt-program/src/expand/grammar.rs`
  - expose or generalize recipe types for inline use;
  - add inline-specific recipe operations only where they express real inline
    invariants.
- `crates/jolt-program/src/expand/materialize.rs`
  - materialize inline recipes through the same central state machine;
  - append inline reset rows;
  - stamp flattened metadata exactly once.
- `crates/jolt-program/src/expand/allocator.rs`
  - preserve the reserved/instruction/inline register split;
  - report inline reset registers without tracer allocator involvement.
- `crates/jolt-program/src/expand/mod.rs`
  - revise `InlineExpansionProvider` to accept source inline rows and return
    recipes;
  - enforce source/profile/provider-free legality.
- `jolt-inlines/sdk`
  - split static recipe declaration from host advice;
  - provide tracer-free helper traits and op aliases;
  - keep tracer re-exports out of static sequence construction.
- `tracer/src/instruction/inline.rs`
  - keep runtime execution and advice integration;
  - stop owning static bytecode semantics or tracer-normalized provider output.
- `tracer/src/utils/inline_helpers.rs`
  - delete or restrict to private runtime/test utility code after static helper
    logic is moved.
- `tracer/src/utils/inline_sequence_writer.rs`
  - consume materialized `JoltInstructionRow`s or recipes directly if trace
    files remain supported.

Expected inline package migrations:

- `jolt-inlines/sha2/src/sequence_builder.rs`
- `jolt-inlines/blake2/src/sequence_builder.rs`
- `jolt-inlines/blake3/src/sequence_builder.rs`
- `jolt-inlines/keccak256/src/sequence_builder.rs`
- `jolt-inlines/bigint/src/multiplication/sequence_builder.rs`
- `jolt-inlines/secp256k1/src/sequence_builder.rs`
- `jolt-inlines/grumpkin/src/sequence_builder.rs`
- `jolt-inlines/p256/src/sequence_builder.rs`

## Related Documentation

This spec supersedes older text in `specs/bytecode-expansion-crate.md` that
allowed registered inline static expansion to remain a tracer-owned
`InstrAssembler` bridge. That text must be removed or marked obsolete in the
implementation PR.

Also update:

- `specs/source-jolt-instruction-split.md` to mark the follow-up complete;
- inline SDK docs/comments showing how to implement static recipes and runtime
  advice providers;
- any `.joltinline` fixture docs if trace files remain supported.

Extraction follow-up:

- add or expose a slim extraction-facing instruction surface for
  `jolt-program::expand` that uses compact tags and row records, not the full
  generic marker-wrapper instruction enums;
- keep marker-type metadata ergonomics available for ordinary Rust builders
  (`emit_r::<ADD>(...)`, lookup tables, runtime trace code), but do not make
  those trait impls mandatory for extracting static expansion semantics;
- keep no-default extraction paths free of serialization dependencies and
  test-only helper iterators.

## Alternatives Considered

1. Keep the current provider adapter.

   This preserves working behavior and keeps the PR smaller, but leaves static
   inline expansion in callback-heavy tracer code and blocks a clean extraction
   story.

2. Return final rows directly from inline providers.

   This avoids some recipe API work, but creates a second materialization path
   and weakens central ownership of metadata, helper recursion, reset rows, and
   target validation.

3. Move all inline registration and advice into `jolt-program`.

   This would make `jolt-program` depend on tracer CPU/memory/advice concerns.
   Static sequence expansion belongs near program construction; advice
   generation does not.

4. Add a second inline-only materializer.

   This would duplicate sequence metadata, legality checks, allocator behavior,
   and capacity checks. Inline expansion should share `ExpansionState` wherever
   possible.

## References

- [PR #1522](https://github.com/a16z/jolt/pull/1522)
- [`specs/source-jolt-instruction-split.md`](source-jolt-instruction-split.md)
- [`specs/compiler-native-bytecode-expansion.md`](compiler-native-bytecode-expansion.md)
- [`specs/bytecode-expansion-crate.md`](bytecode-expansion-crate.md)
