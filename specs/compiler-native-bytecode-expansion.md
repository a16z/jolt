# Spec: Compiler-Native RV64 Bytecode Expansion

| Field      | Value |
|------------|-------|
| Author(s)  | Quang Dao |
| Created    | 2026-05-05 |
| Revised    | 2026-05-09 |
| Status     | target design |
| Related PR | [#1490](https://github.com/a16z/jolt/pull/1490), [#1518](https://github.com/a16z/jolt/pull/1518) |

## Summary

`jolt-program::expand` should be a small compiler pass over explicit instruction
phases, not a recursive Rust assembler that happens to produce the right rows.
The long-term target is:

```text
decoded RV64 source row
  -> human-readable per-instruction lowering
  -> ExpandedInstructionSequence / ExpansionOp recipe
  -> owned ExpansionState materializer
  -> final stamped Jolt bytecode rows
```

Concrete instruction lowerings should stay easy for humans to read and edit.
They should look like ordered assembly snippets:

```rust
let mut asm = ExpansionBuilder::new(*instruction);

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

But the builder must be declarative: it records expansion operations. It must
not own or borrow allocator-backed expansion state, and it must not recursively
call the public expander while a concrete instruction lowering is running.

The branch first improved naming, removed old RV32-facing paths, and replaced
the old `InstrAssembler<'a>` surface. The final target goes further: no
provider-free lowerer should own or borrow allocator-backed expansion state.
This spec replaces that interim builder shape with owned state, explicit
recipes, and a central materializer.

## Branch Changes So Far

The branch already contains three substantial changes relative to `main`. The
remaining owned-state recipe rewrite should build on these changes rather than
re-describe them as future work.

### RV64-Only Cleanup

The tracer/emulator path has been narrowed to RV64:

- `Xlen::Bit32` and SV32 handling were removed from `tracer`.
- MMU tracing now uses 8-byte RV64 words consistently.
- RV32-specific arithmetic/sign-extension branches were removed from instruction
  implementations and virtual helpers.
- stale RV32-facing tests, comments, and helper assumptions were removed or
  rewritten.
- `jolt-program::image` continues to reject ELF32/RV32 inputs at the image
  boundary.

This means the new expander does not need an `Xlen` parameter or RV32 fallback
paths. Word instructions such as `ADDW`, `LW`, and `AMO*W` remain RV64 word
operations, not evidence that RV32 execution is still supported.

### Instruction Phase Naming

The branch has already split the old ambiguous instruction naming:

- decoded source rows use `RiscvInstructionKind`;
- expanded Jolt bytecode/helper rows use `JoltInstructionKind`;
- lookup-backed/provable rows use `LookupInstructionKind`;
- the old typed `JoltInstructions` view has been renamed to
  `LookupInstruction`;
- lookup routing in `jolt-core`, instruction metadata in `jolt-riscv`, the
  SHA2 inline sequence builder, and the Z3 virtual-sequence verifier were moved
  onto the new phase names.

The remaining work is not to invent these names from scratch. It is to audit and
finish the split while implementing the recipe expander, so source-only
legality, lookup-backed legality, and side-effect classification sit on the
right phase APIs.

### Interim Expansion Refactor

The branch replaced the old production `assembler.rs` path with a more readable
builder-based expander:

- `assembler.rs` was deleted.
- `buffer.rs`, `core.rs`, `grammar.rs`, and `metadata.rs` were added under
  `crates/jolt-program/src/expand`.
- concrete expansion files now read as `asm.emit_*`, `asm.expand_*`,
  `asm.allocate`, and `asm.release` calls.
- `emit_*` means "append a target-legal row"; `expand_*` means "route a helper
  row through provider-free expansion".
- bounded row buffers, sequence metadata helpers, recursion-depth checks,
  target-legality checks, and explicit release calls were added.
- a compact hash fixture
  `crates/jolt-program/src/expand/fixtures/main_expand_parity_hashes.json`
  checks provider-free expansion parity without committing huge row JSON.

The final compiler-native cutover preserves that readable lowerer surface but
changes what it means: `ExpansionBuilder` records pure `ExpansionOp` recipes,
and owned `ExpansionState` materializes temps, helper expansion, and sequence
metadata centrally.

## Goals

- Keep Jolt program expansion RV64-only.
- Preserve expansion behavior relative to the post-#1490 baseline unless a
  change is explicitly called out.
- Keep concrete instruction expansions readable and close to the old hand-written
  assembly style.
- Make expansion lowering declarative: lowerers produce recipes; one central
  driver materializes recipes.
- Eliminate lifetime-bearing expansion state from provider-free expansion.
- Make temporary-register allocation, release, reuse, recursion, metadata
  stamping, and `rd = x0` rewriting explicit and centrally validated.
- Split instruction identity by phase:
  - `RiscvInstructionKind`: decoded source ISA identity;
  - `JoltInstructionKind`: Jolt bytecode/helper row identity used during
    expansion;
  - `LookupInstructionKind`: lookup-backed/provable row identity.
- Keep provider-free expansion isolated from tracer, CPU, memory-device,
  advice-tape, prover, transcript, ELF parser, and PCS dependencies.
- Make Hax/Aeneas extraction target production expansion code, not a hand-written
  model that can drift from production.

## Non-Goals

- Do not change RISC-V or Jolt instruction semantics.
- Do not reintroduce RV32 compatibility shims.
- Do not add deprecated aliases or migration layers for renamed instruction
  kinds.
- Do not model full execution semantics in this expansion pass. Expansion says
  how source rows lower into bytecode rows; it does not define the operational
  meaning of those rows.
- Do not port registered `jolt-inlines` recipes or advice producers into the
  provider-free grammar in this pass.
- Do not make Hax/Aeneas extraction a CI requirement unless maintainers ask for
  it.

## Current Problem

The old #1490 shape used `InstrAssembler<'a>`:

```rust
pub struct InstrAssembler<'a> {
    sequence: Vec<NormalizedInstruction>,
    allocator: &'a mut ExpansionAllocator,
}
```

The interim builder refactor removed that type from production expansion, but it
recreated the same fundamental borrow shape:

```rust
pub(super) struct ExpansionState<'a> {
    allocator: &'a mut ExpansionAllocator,
}

pub(super) struct ExpansionBuilder<'a, 'b> {
    source: &'b NormalizedInstruction,
    state: ExpansionState<'a>,
    sequence: ExpansionSequence,
}
```

That shape was nicer to read, but it was not the compiler-native design.
Extraction tools still saw a long-lived mutable borrow stored inside the
expansion engine, and recursive helper expansion was still encoded by Rust calls
from builder methods.

The correct boundary is:

- concrete lowerers produce data;
- the driver owns mutable state;
- the driver is the only component that interprets recursive expansion work.

## Target Data Model

### Row Type

`NormalizedInstruction` remains the production row type. It should not grow
execution semantics, advice payloads, or proof-system fields.

### Instruction Phase Types

Use explicit phase names:

```rust
pub enum RiscvInstructionKind { ... }
pub enum JoltInstructionKind { ... }
pub enum LookupInstructionKind { ... }
```

Expected ownership:

- `RiscvInstructionKind` belongs to decode and source-program representation.
- `JoltInstructionKind` belongs to expanded Jolt bytecode and virtual/helper
  rows.
- `LookupInstructionKind` belongs to lookup-table routing and proving.

The exact enum layout can be flat or nested, but conversions must be explicit.
For example:

```rust
impl JoltInstructionKind {
    pub fn lookup_kind(self) -> Option<LookupInstructionKind> { ... }
    pub fn is_source_only(self) -> bool { ... }
    pub fn has_side_effects(self) -> bool { ... }
}
```

### Expansion Recipes

Concrete lowerers return a recipe:

```rust
pub(super) struct ExpandedInstructionSequence {
    source: NormalizedInstruction,
    ops: Vec<ExpansionOp>,
}

pub(super) enum ExpansionOp {
    Emit(RowTemplate),
    Expand(RowTemplate),
    Allocate(u8),
    Release(u8),
}
```

The names can change, but the responsibilities should not:

- `Emit` appends a row that is already final Jolt bytecode.
- `Expand` appends a source/helper row that must go through the central expander.
- `Allocate` and `Release` describe symbolic temp lifetimes in the recipe.
  The recorded `u8` values are temporary placeholders, not concrete virtual
  registers.

### Row Templates And Symbolic Temps

Recipes can mention architectural registers and symbolic temps. Symbolic temps
are encoded as reserved placeholder register values inside `RowTemplate`, so the
current lowerer code can stay close to the old assembly-like style while still
deferring concrete virtual-register allocation to materialization.

```rust
pub(super) struct RowTemplate {
    instruction_kind: JoltInstructionKind,
    operands: TemplateOperands,
}
```

Materialization resolves `TempId` values to concrete virtual registers through
the owned `ExpansionAllocator`.

## Target Control Flow

### ExpansionBuilder

`ExpansionBuilder` is a pure lowering builder:

```rust
pub(super) struct ExpansionBuilder {
    source: NormalizedInstruction,
    ops: Vec<ExpansionOp>,
    next_temp: u8,
}
```

It must not contain:

- `&mut ExpansionAllocator`;
- `ExpansionState<'_>`;
- any lifetime parameter needed for allocator ownership;
- calls to `expand_instruction` or `expand_one_core`.

It should expose ergonomic methods:

```rust
impl ExpansionBuilder {
    pub fn new(source: NormalizedInstruction) -> Self;

    pub fn allocate(&mut self) -> Result<u8, ExpansionError>;
    pub fn release(&mut self, temp: u8) -> Result<(), ExpansionError>;
    pub fn release_many<const N: usize>(&mut self, temps: [u8; N])
        -> Result<(), ExpansionError>;

    pub fn emit_r(...);
    pub fn emit_i(...);
    pub fn emit_s(...);
    pub fn emit_b(...);
    pub fn emit_u(...);
    pub fn emit_j(...);

    pub fn expand_r(...);
    pub fn expand_i(...);
    pub fn expand_s(...);
    pub fn expand_b(...);
    pub fn expand_u(...);
    pub fn expand_j(...);
    pub fn expand_address(...);

    pub fn finalize(self) -> Result<ExpandedInstructionSequence, ExpansionError>;
}
```

`emit_*` methods should be infallible when they only record a syntactically valid
template. Operand validation that depends on source row shape can still return
`ExpansionError` at the lowerer boundary.

### ExpansionState

`ExpansionState` owns all mutable materialization state:

```rust
pub(super) struct ExpansionState {
    allocator: ExpansionAllocator,
    work: Vec<ExpansionOp>,
    fuel: u32,
}
```

It should provide the provider-free entry point:

```rust
pub(super) fn expand_one_core(
    source: NormalizedInstruction,
    state: &mut ExpansionState,
) -> Result<Vec<NormalizedInstruction>, ExpansionError>;
```

The public API may preserve the existing call shape as an adapter:

```rust
pub fn expand_instruction(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError>;
```

but the adapter must move allocator state into `ExpansionState`, run the owned
driver, then move allocator state back out. Borrowed allocator state must not be
stored in production core structures.

### Central Driver

The central driver handles:

- `rd = x0` canonicalization;
- side-effect-preserving `rd = x0` rewrites;
- source-only legality checks;
- recursive helper expansion;
- temp allocation and release;
- output capacity checks;
- recursion/fuel limits;
- sequence metadata stamping;
- provider-free `Inline` rejection.

In pseudocode:

```rust
fn expand_one_core(
    source: NormalizedInstruction,
    state: &mut ExpansionState,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    state.reset_for_source(source)?;
    let recipe = lower_source_row(source)?;
    state.push_recipe(recipe)?;

    while let Some(op) = state.pop_work()? {
        state.consume_fuel()?;
        match op {
            ExpansionOp::Emit(row) => materializer.emit(row)?,
            ExpansionOp::Expand(row) => materializer.extend(state.expand_helper(row)?)?,
            ExpansionOp::Allocate(temp) => materializer.bind_temp(temp, state.allocate()?)?,
            ExpansionOp::Release(temp) => state.release(materializer.resolve_release(temp)?)?,
        }
    }

    state.finish_source_sequence()
}
```

The important invariant is that recursive helper expansion is a driver action,
not a builder action.

## Concrete Lowerer Shape

Concrete lowerers should no longer accept `&mut ExpansionAllocator`.

Target signature:

```rust
pub(in crate::expand) fn expand_addiw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError>;
```

Simple example:

```rust
pub(in crate::expand) fn expand_subw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

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
}
```

Temp-heavy example:

```rust
pub(in crate::expand) fn expand_lw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate();
    let v1 = asm.allocate();

    asm.expand_address(
        JoltInstructionKind::VirtualAssertWordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    );
    asm.expand_i(
        JoltInstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        instruction.operands.imm,
    );
    asm.expand_i(JoltInstructionKind::ANDI, v1, v0, format_i_imm(-8));
    asm.expand_i(JoltInstructionKind::LD, v1, v1, 0);
    asm.expand_i(JoltInstructionKind::SLLI, v0, v0, 3);
    asm.expand_r(JoltInstructionKind::SRL, v1, v1, v0);
    asm.expand_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        v1,
        0,
    );

    asm.release_many([v0, v1]);
    asm.finalize()
}
```

This shape is intentionally close to the current readable builder code. The
difference is that the builder records a recipe; it does not materialize rows or
recurse.

## Inline And Advice Boundary

Provider-free expansion rejects `JoltInstructionKind::Inline`.

Registered inline expansion remains an adapter outside the provider-free core:

- the provider may return finalized `NormalizedInstruction` rows;
- provider output is not modeled as provider-free `ExpansionOp` values in this
  pass;
- advice payload assignment remains outside `NormalizedInstruction`;
- trusted/untrusted advice memory and polynomial commitments remain
  preprocessing/proof responsibilities.

Advice-load source rows such as `AdviceLB`, `AdviceLH`, `AdviceLW`, and
`AdviceLD` are provider-free lowerings because they lower to ordinary bytecode
and virtual advice-load rows. Registered inlines that synthesize advice remain
outside this grammar.

## Metadata Policy

Preserve the current metadata policy:

- pass-through rows remain unchanged;
- synthetic sequences are stamped as one source-row expansion;
- `is_compressed` is attached according to the current final-row policy;
- `virtual_sequence_remaining` and `is_first_in_sequence` are derived centrally;
- helper rows expanded recursively participate in the outer source sequence.

No concrete lowerer should mutate sequence metadata directly.

## Validation Strategy

Behavioral parity should be tested at three levels:

1. Targeted unit tests for allocator transitions, temp release ordering, invalid
   releases, buffer overflow, and metadata stamping.
2. Instruction-family tests for representative shallow, recursive, temp-heavy,
   memory, AMO, CSR/control-flow, division/remainder, and advice-load lowerings.
3. A compact baseline parity fixture that hashes serialized expanded rows for
   provider-free inputs.

The fixture should remain compact. Avoid checking in giant expanded-row JSON.
When fixture hashes change, treat it as a semantic review event:

- state the baseline commit or intended semantic change;
- inspect at least one expanded-row diff for each affected family;
- regenerate with exact serialized bytes of `Vec<NormalizedInstruction>`;
- run the dedicated parity test.

## Extraction Strategy

The extraction start set should be provider-free expansion internals:

- allocator transitions;
- row template and recipe construction;
- metadata stamping;
- one shallow lowerer such as `ADDIW` or `SUBW`;
- one recursive/temp-heavy lowerer such as `LW`;
- the owned `ExpansionState` driver.

Hax/Aeneas issues caused by missing standard-library models are useful to record,
but production code should first remove avoidable obstacles:

- no lifetime-bearing expansion state;
- no long-lived mutable borrow stored in the builder;
- no public trait callback in the provider-free core;
- no `impl IntoIterator` in extraction-critical signatures when a concrete slice
  or vector is enough;
- minimal serialization/preprocess/prover dependencies in the extracted call
  graph.

## Acceptance Criteria

Already satisfied by the current branch:

- [x] Historical RV32 tracer execution support is removed; no live `Xlen::Bit32`
      or SV32 path remains.
- [x] ELF32/RV32 inputs are rejected at the `jolt-program::image` boundary.
- [x] Decoded-source, expanded-bytecode, and lookup-backed instruction
      identities are named separately as `RiscvInstructionKind`,
      `JoltInstructionKind`, and `LookupInstructionKind`.
- [x] The old typed `JoltInstructions` view has been renamed to
      `LookupInstruction`.
- [x] Lookup-table routing uses `LookupInstructionKind` or equivalent explicit
      lookup-subset conversion APIs.
- [x] The old production `InstrAssembler<'a>` file/path is removed.
- [x] The current concrete lowerers use readable builder calls rather than raw
      grammar-node construction.
- [x] A compact provider-free expansion parity fixture exists.
- [x] `ExpansionState` owns `ExpansionAllocator`; no production
      `ExpansionState<'a>` remains.
- [x] `ExpansionBuilder` has no lifetime parameters and stores no
      `ExpansionState`.
- [x] Concrete lowerers return `ExpandedInstructionSequence` recipes, not
      finalized `Vec<NormalizedInstruction>` values.
- [x] Concrete provider-free lowerers do not accept `&mut ExpansionAllocator`.
- [x] Builder `expand_*` methods record recursive-helper work; they do not call
      `expand_instruction`, `expand_one_core`, or any central driver function.
- [x] The central driver is the only provider-free component that expands helper
      rows recursively.
- [x] Temp lifetimes are represented explicitly in recipe data and materialized
      centrally.

Satisfied by the owned recipe/materializer rewrite:

- [x] Materialization preserves current virtual-register allocation and reuse.
- [x] `rd = x0` handling is centralized and covered by tests for side-effecting
      and non-side-effecting rows.
- [x] Synthetic sequence metadata is stamped centrally and matches the baseline.
- [x] Pass-through rows preserve the current metadata policy.
- [x] Provider-free expansion rejects `Inline`; inline provider support remains
      an adapter outside the owned core.
- [x] The existing instruction phase split is audited after the recipe rewrite,
      with no stale `InstructionKind` ambiguity or misplaced legality predicate.
- [x] `jolt-program::expand` remains independent of tracer, CPU execution,
      prover, transcript, PCS, and ELF parser dependencies.
- [x] Existing provider-free parity tests pass.
- [x] `cargo fmt -q` passes.
- [x] `cargo clippy --all --features host -q --all-targets -- -D warnings`
      passes.
- [x] `cargo clippy --all --features host,zk -q --all-targets -- -D warnings`
      passes.
- [x] `cargo nextest run -p jolt-program --cargo-quiet` passes.
- [x] `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host`
      passes.
- [x] `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk`
      passes.
- [x] Hax/Aeneas extraction is rerun on the provider-free core and the remaining
      blockers are documented.

Remaining outside this PR:

- [ ] Prove or hand-port the emitted Hax/Aeneas Lean into a maintained Lean
      project once the extractor prelude/library-model blockers are resolved.
- [ ] Decide whether to further simplify concrete lowerers specifically for
      Aeneas by avoiding conditional expressions inside row operands and by
      replacing iterator-heavy metadata helpers with loop-shaped code.

## Implemented Work Items

These are retained as an implementation checklist for the PR history. They are
implemented unless explicitly listed above as outside this PR.

### 1. Preserve And Verify Already-Landed Scope

- Treat RV64-only tracer cleanup as already landed; avoid reintroducing `Xlen`
  branches or RV32 compatibility shims while rewriting expansion.
- Treat `RiscvInstructionKind` / `JoltInstructionKind` /
  `LookupInstructionKind` as the current branch direction; audit and finish it
  rather than reverting to `InstructionKind`.
- Treat the current readable builder surface as the ergonomic baseline for
  concrete lowerers.
- Keep the compact hash fixture strategy; do not replace it with giant JSON.

### 2. Clean Up The Current Interim State

- Remove any remaining production references to `ExpansionState<'a>` and
  `ExpansionBuilder<'a, 'b>`.
- Remove stale spec/test language that treats the borrowed-builder shape as the
  final design.
- Keep `EXTRACTION-AUDIT-NEVER-COMMIT.md` local and untracked.

### 3. Introduce Recipe Types

- Add `ExpandedInstructionSequence`.
- Add `ExpansionOp`.
- Add `RowTemplate` / `TemplateOperands` capable of carrying architectural
  registers and symbolic temps.
- Add `TempId` or equivalent symbolic temp identifiers.
- Add tests for recipe construction and invalid temp lifetime patterns.

### 4. Make ExpansionBuilder Pure

- Change `ExpansionBuilder::new` to accept the source row by value.
- Remove allocator/state fields from `ExpansionBuilder`.
- Make `emit_*` append `ExpansionOp::Emit`.
- Make `expand_*` append `ExpansionOp::Expand`.
- Make `allocate` / `release` record temp lifetime operations.
- Keep lowerer code visually close to the current readable builder style.

### 5. Make ExpansionState Owned

- Change `ExpansionState` to own `ExpansionAllocator`.
- Add work-stack, output-buffer, and fuel/depth state.
- Add owned-state entry points and adapter functions for the existing public API.
- Avoid `impl IntoIterator` in extraction-critical APIs where concrete types are
  straightforward.

### 6. Implement The Central Materializer

- Interpret `ExpansionOp` values.
- Resolve symbolic temps to concrete virtual registers.
- Centralize `rd = x0` rewriting.
- Centralize source-only legality checks.
- Centralize helper expansion.
- Centralize metadata stamping.
- Preserve pass-through behavior.
- Preserve inline reset behavior at the adapter boundary.

### 7. Convert Concrete Lowerers

- Update every lowerer signature to remove `&mut ExpansionAllocator`.
- Convert arithmetic and word-op lowerers.
- Convert shift lowerers.
- Convert load/store lowerers.
- Convert AMO lowerers.
- Convert LR/SC lowerers.
- Convert division/remainder lowerers.
- Convert CSR/control-flow lowerers.
- Convert advice-load lowerers.
- Remove old finalized-row helper paths once all call sites are converted.

### 8. Finish Instruction Phase Split

- Audit uses of `InstructionKind`, `JoltInstructionKind`, and
  `LookupInstructionKind`.
- Ensure decoded-source, expanded-bytecode, and lookup-backed concepts have
  explicit names and conversions.
- Ensure source-only and lookup-backed legality predicates live on the right
  phase type.
- Remove stale aliases or compatibility names.

### 9. Strengthen Parity And Regression Tests

- Keep compact hash fixture rather than giant JSON.
- Add row-diff tooling or test helper for affected fixture changes.
- Add targeted tests for:
  - recursive helper expansion order;
  - temp allocation/release order;
  - metadata stamping;
  - side-effecting `rd = x0`;
  - non-side-effecting `rd = x0`;
  - provider-free `Inline` rejection;
  - advice-load lowering.

### 10. Rerun Extraction Experiments

- Rerun Hax on allocator, metadata, one shallow lowerer, one temp-heavy lowerer,
  and the owned driver.
- Rerun Aeneas/Charon on the same slices.
- Record remaining blockers in local never-commit notes.
- Separate genuine design blockers from tool/prelude/library-model blockers.

### 11. Final Verification

- Run formatting.
- Run clippy in host and host+zk modes.
- Run `jolt-program` nextest.
- Run `jolt-core` `muldiv` in host and host+zk modes.
- Check dependency boundaries.
- Update the PR description with the final architecture and verification.
