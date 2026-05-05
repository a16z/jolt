# Spec: Extraction-Native Bytecode Expansion

| Field       | Value                                                                 |
|-------------|-----------------------------------------------------------------------|
| Author(s)   | Quang Dao                                                            |
| Created     | 2026-05-05                                                           |
| Status      | draft                                                                 |
| Related PR  | [#1490](https://github.com/a16z/jolt/pull/1490)                      |
| Baseline    | `quang/bytecode-expand-spec` at `a3448e6da44f`                        |
| Depends on  | `specs/bytecode-expansion-crate.md`                                   |

## Summary

PR #1490 moves bytecode expansion out of `tracer` and into `jolt-program::expand`. That crate boundary is the right direction for formal verification, but the current implementation at `a3448e6da44f` still uses an idiomatic recursive Rust assembler shape that is hard for Hax and Aeneas to extract:

- family expanders build `Vec<NormalizedInstruction>` values;
- `InstrAssembler<'a>` owns a sequence while borrowing `&'a mut ExpansionAllocator`;
- `InstrAssembler::emit` recursively calls `expand_instruction`;
- temporary-register release is encoded in Rust call-stack/control-flow order;
- metadata is stamped by mutating a finished slice;
- inline expansion is a trait callback inside the core dispatch path.

This spec proposes a second-phase rewrite of `jolt-program::expand` into an extraction-native production implementation. The goal is not to add a proof-only model next to production code. The production expander itself should become a first-order state machine over explicit data transitions, while preserving byte-for-byte output and keeping runtime performance the same or better.

The target design:

```text
Decoded NormalizedInstruction
  -> SourceRow
  -> rd=x0 normalization
  -> depth-first work stack of ExpansionOp
  -> shallow family lowering
  -> explicit temp release/reset operations
  -> bounded per-source output buffer
  -> single metadata stamping pass
  -> top-level Vec extension
```

## Goals

- Preserve expansion behavior exactly relative to PR #1490 at `a3448e6da44f`.
- Preserve recursive expansion order exactly.
- Preserve `rd = x0` behavior for all source and helper rows.
- Preserve virtual-register numbering, allocation reuse, reserved registers, and inline reset behavior.
- Preserve sequence metadata: source address, `virtual_sequence_remaining`, `is_first_in_sequence`, and compressed-instruction metadata.
- Keep `jolt-program::expand` free of tracer, CPU, memory-device, advice-tape, prover, transcript, ELF parser, and PCS dependencies.
- Make Hax/Aeneas extraction of provider-free RV64 expansion straightforward enough that the first real extraction target is the production core, not a hand-written mirror.
- Avoid performance regressions by removing recursive per-instruction heap allocation and using bounded stack-resident buffers for per-source expansion.

## Non-Goals

- Do not change instruction semantics.
- Do not change bytecode preprocessing, RAM preprocessing, or proof-system APIs except for call-site adjustments needed by the new expansion API.
- Do not formalize tracer custom inline registries in this phase.
- Do not make Aeneas/Hax extraction a hard CI requirement in the implementation PR unless maintainers explicitly ask for it.
- Do not keep the current recursive `InstrAssembler<'a>` implementation as a compatibility layer once the rewrite lands. This branch owns the new `jolt-program` implementation, so the rewrite should be a full cutover.

## Baseline: Current PR Shape At `a3448e6da44f`

The current PR implementation lives under:

- `crates/jolt-program/src/expand/mod.rs`
- `crates/jolt-program/src/expand/allocator.rs`
- `crates/jolt-program/src/expand/assembler.rs`
- per-family modules such as `arithmetic.rs`, `memory.rs`, `division.rs`, `shifts.rs`, and `control_flow.rs`

Current public entry points:

```rust
pub fn expand_instruction(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError>;

pub fn expand_instruction_with_provider<P: InlineExpansionProvider + ?Sized>(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    inline_provider: &mut P,
) -> Result<Vec<NormalizedInstruction>, ExpansionError>;

pub fn expand_program(
    instructions: impl IntoIterator<Item = NormalizedInstruction>,
) -> Result<Vec<NormalizedInstruction>, ExpansionError>;
```

Current family expanders are shaped like:

```rust
pub(super) fn expand_addiw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::ADDI,
        rd(instruction)?,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    asm.finalize()
}
```

This looks small, but `asm.emit_i` calls `expand_instruction` recursively. Aeneas/Hax do not see "ADDIW lowers to two rows"; they see ADDIW lower into a borrowed assembler that recursively invokes the full dispatch path.

Current `InstrAssembler` shape:

```rust
pub struct InstrAssembler<'a> {
    address: usize,
    is_compressed: bool,
    has_inline_instr_format: bool,
    sequence: Vec<NormalizedInstruction>,
    allocator: &'a mut ExpansionAllocator,
}
```

The lifetime-bearing field is the biggest extraction smell. It lets normal Rust code write ergonomic expansion snippets, but it gives extraction tools a long-lived mutable borrow stored inside another mutable structure.

### Baseline Extraction Experiment

The smallest metadata-only slice works:

```bash
cargo hax -C -p jolt-program \; into \
  -i '+!jolt_program::expand::metadata::set_sequence_metadata' \
  --output-dir /tmp/jolt-hax-bytecode-expand-metadata lean

/Users/quang.dao/Documents/Lean/aeneas/charon/charon/target/release/charon cargo \
  --preset=aeneas \
  --hide-allocator \
  --start-from crate::expand::metadata::set_sequence_metadata \
  --dest-file /tmp/jolt-program-metadata.llbc \
  -- -p jolt-program

/Users/quang.dao/Documents/Lean/aeneas/src/_build/install/default/bin/aeneas \
  -backend lean \
  -dest /tmp/jolt-aeneas-metadata-lean \
  -split-files \
  -namespace JoltProgram \
  /tmp/jolt-program-metadata.llbc
```

The next slice, `expand::arithmetic::expand_addiw`, already exposes the structural issues:

```bash
/Users/quang.dao/Documents/Lean/aeneas/charon/charon/target/release/charon cargo \
  --preset=aeneas \
  --hide-allocator \
  --start-from crate::expand::arithmetic::expand_addiw \
  --dest-file /tmp/jolt-program-addiw.llbc \
  -- -p jolt-program

/Users/quang.dao/Documents/Lean/aeneas/src/_build/install/default/bin/aeneas \
  -backend lean \
  -dest /tmp/jolt-aeneas-addiw-lean \
  -split-files \
  -namespace JoltProgram \
  /tmp/jolt-program-addiw.llbc
```

Observed Charon warning:

```text
warning: Could not reconstruct `Box` initialization; branching during `Box` initialization is not supported.
```

Observed Aeneas errors included:

```text
Unsupported operation: shallow-init-box(move v@...)
The input arguments don't have the proper type
The pushed variables and their values do not have the same type
Internal error, please file an issue
Expected an arrow type
Unreachable
```

These errors come from the shape of the extracted call graph, not from bytecode expansion semantics.

## Current Vs Target Shape

| Concern | Current PR at `a3448e6da44f` | Target shape |
|---------|-------------------------------|--------------|
| Expansion state | Split between `InstrAssembler<'a>` and borrowed `&mut ExpansionAllocator` | One owned `ExpansionState` containing allocator, work stack, and output buffer |
| Family lowering | Expander calls `asm.emit_*`; each emit recursively expands immediately | Expander appends shallow `ExpansionOp` values only |
| Recursion | Hidden inside `InstrAssembler::emit` | One central depth-first driver |
| Temp lifetime | Encoded by Rust control flow and post-`finalize()` releases | Explicit `ExpansionOp::Release(register)` markers |
| Output allocation | Per-source `Vec` plus recursive helper `Vec`s | Bounded per-source buffer plus one top-level program `Vec` |
| Metadata | Mutate `&mut [NormalizedInstruction]` after sequence is built | Construct final rows once from raw rows and sequence length |
| Allocator | `[bool; N]` plus `Vec<u8>` reset list | Bitsets for live registers and inline reset set |
| Inline expansion | Trait callback inside core dispatcher | Adapter outside provider-free core |
| API ergonomics | `impl IntoIterator`, trait generic provider path | Concrete slice/state core; ergonomic wrappers outside |
| Extracted call graph | Pulls in much of `jolt-program` and serde/ark derives | Small `expand` core module graph |

## Target Module Layout

Suggested module layout:

```text
crates/jolt-program/src/expand/
  mod.rs             public adapters and compatibility API
  core.rs            SourceRow, RawRow, ExpansionOp, ExpansionState, driver
  lower.rs           dispatch from RawRow to shallow lowerers
  lower/
    arithmetic.rs
    control_flow.rs
    division.rs
    memory.rs
    shifts.rs
  allocator.rs       bitset allocator transitions
  buffer.rs          fixed-capacity WorkStack and ExpansionBuffer
  metadata.rs        metadata stamping and sequence invariants
  operands.rs        total operand projection helpers
  inline.rs          provider adapter; outside extraction-critical core
  error.rs           small core error enum; display impls can be feature-gated
```

The important separation is `core + lower + allocator + buffer + metadata + operands` versus `inline + public ergonomic wrappers`. The extraction target should be the first group.

## Core Data Model

`SourceRow` is the decoded source instruction plus source metadata shared by every row in the final expanded sequence:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SourceRow {
    pub kind: InstructionKind,
    pub operands: NormalizedOperands,
    pub address: usize,
    pub is_compressed: bool,
}

impl From<NormalizedInstruction> for SourceRow {
    fn from(row: NormalizedInstruction) -> Self {
        Self {
            kind: row.instruction_kind,
            operands: row.operands,
            address: row.address,
            is_compressed: row.is_compressed,
        }
    }
}
```

`RawRow` is a bytecode row before source metadata is stamped:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawRow {
    pub kind: InstructionKind,
    pub operands: NormalizedOperands,
}
```

Family lowerers produce operations, not normalized rows:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpansionOp {
    Row(RawRow),
    Release(u8),
    ResetInlineRegisters,
}
```

The driver owns all mutable state:

```rust
pub(crate) struct ExpansionState {
    allocator: ExpansionAllocator,
    work: WorkStack<ExpansionOp>,
    output: ExpansionBuffer<RawRow>,
    fuel: u32,
}
```

No production expansion struct should contain a borrowed allocator or a lifetime parameter.

## Driver Design

The driver is the only recursive component. It should be implemented as an iterative depth-first work stack:

```rust
pub(crate) fn expand_one_core(
    source: SourceRow,
    state: &mut ExpansionState,
) -> Result<ExpandedRows, ExpansionError> {
    state.reset_for_source();
    state.push_work(ExpansionOp::Row(source.raw_row()))?;

    while let Some(op) = state.pop_work() {
        state.consume_fuel()?;
        match op {
            ExpansionOp::Row(row) => process_row(source, row, state)?,
            ExpansionOp::Release(register) => state.allocator.release(register)?,
            ExpansionOp::ResetInlineRegisters => emit_inline_resets(source, state)?,
        }
    }

    metadata::stamp(source, state.output.as_slice())
}
```

`process_row` handles the rules currently embedded in `expand_instruction`:

```rust
fn process_row(
    source: SourceRow,
    row: RawRow,
    state: &mut ExpansionState,
) -> Result<(), ExpansionError> {
    if row.operands.rd == Some(0) && !handles_rd_zero_internally(row.kind) {
        if has_side_effects(row.kind) {
            let tmp = state.allocator.allocate_instruction()?;
            let rewritten = row.with_rd(tmp);
            state.push_ops_reversed(&[
                ExpansionOp::Release(tmp),
                ExpansionOp::Row(rewritten),
            ])?;
            return Ok(());
        }
        state.output.push(noop_raw())?;
        return Ok(());
    }

    if is_final_kind(row.kind) {
        state.output.push(row)?;
    } else {
        let ops = lower::lower(row, &mut state.allocator)?;
        state.push_ops_reversed(ops.as_slice())?;
    }
    Ok(())
}
```

The stack must preserve current recursive order. If a lowerer emits `[A, B, C]`, the driver should process the full recursive expansion of `A`, then `B`, then `C`. With a LIFO stack, `push_ops_reversed` pushes `C`, then `B`, then `A`.

## Shallow Lowerers

Every family expander should become a shallow lowerer. It must not call the public `expand_instruction` or the central driver.

Current ADDIW:

```rust
asm.emit_i(InstructionKind::ADDI, rd, rs1, imm)?;
asm.emit_i(InstructionKind::VirtualSignExtendWord, rd, rd, 0)?;
```

Target ADDIW:

```rust
pub(crate) fn lower_addiw(
    row: RawRow,
    out: &mut OpBuffer,
) -> Result<(), ExpansionError> {
    let rd = operands::rd(row)?;
    let rs1 = operands::rs1(row)?;
    out.row(i(InstructionKind::ADDI, rd, rs1, row.operands.imm))?;
    out.row(i(InstructionKind::VirtualSignExtendWord, rd, rd, 0))?;
    Ok(())
}
```

For functions with temporary registers, releases become explicit operations at the same semantic point:

```rust
pub(crate) fn lower_mulh(
    row: RawRow,
    allocator: &mut ExpansionAllocator,
    out: &mut OpBuffer,
) -> Result<(), ExpansionError> {
    let rd = operands::rd(row)?;
    let rs1 = operands::rs1(row)?;
    let rs2 = operands::rs2(row)?;

    let v_sx = allocator.allocate_instruction()?;
    let v_sy = allocator.allocate_instruction()?;
    let v_tmp = allocator.allocate_instruction()?;

    out.row(i(InstructionKind::VirtualMovsign, v_sx, rs1, 0))?;
    out.row(i(InstructionKind::VirtualMovsign, v_sy, rs2, 0))?;
    out.row(r(InstructionKind::MUL, v_sx, v_sx, rs2))?;
    out.row(r(InstructionKind::MUL, v_sy, v_sy, rs1))?;
    out.row(r(InstructionKind::MULHU, v_tmp, rs1, rs2))?;
    out.row(r(InstructionKind::ADD, v_tmp, v_tmp, v_sx))?;
    out.row(r(InstructionKind::ADD, rd, v_tmp, v_sy))?;

    out.release(v_sx)?;
    out.release(v_sy)?;
    out.release(v_tmp)?;
    Ok(())
}
```

This preserves the current behavior: releases happen after the recursively expanded helper rows that use the temps.

Some current functions release temps mid-sequence, for example CSR and SC flows. Those become mid-sequence `Release` operations:

```rust
out.row(i(InstructionKind::ADDI, temp, rs1, 0))?;
out.row(i(InstructionKind::ADDI, rd, virtual_reg, 0))?;
out.row(i(InstructionKind::ADDI, virtual_reg, temp, 0))?;
out.release(temp)?;
out.row(...)?;
```

The driver processes that release after the preceding rows have recursively finalized and before subsequent rows.

## Allocator Design

The current allocator:

```rust
allocated: [bool; NUM_VIRTUAL_REGISTERS],
pending_clearing_inline: Vec<u8>,
```

should become bitset-based:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExpansionAllocator {
    live: u128,
    inline_touched: u128,
}
```

There are at most 96 virtual registers, so `u128` is enough. Register index `i` corresponds to virtual register `RISCV_REGISTER_BASE + i`.

Allocation:

```rust
pub(crate) fn allocate_in_range(
    &mut self,
    start: u8,
    end: u8,
    pool: RegisterPool,
) -> Result<u8, ExpansionError> {
    let mut index = start;
    while index < end {
        let bit = 1u128 << index;
        if self.live & bit == 0 {
            self.live |= bit;
            if matches!(pool, RegisterPool::Inline) {
                self.inline_touched |= bit;
            }
            return Ok(RISCV_REGISTER_BASE + index);
        }
        index += 1;
    }
    Err(ExpansionError::VirtualRegisterExhausted { pool })
}
```

Release:

```rust
pub(crate) fn release(&mut self, register: u8) -> Result<(), ExpansionError> {
    let index = virtual_index(register)?;
    let bit = 1u128 << index;
    if self.live & bit == 0 {
        return Err(ExpansionError::UnallocatedVirtualRegister { register });
    }
    self.live &= !bit;
    Ok(())
}
```

Inline reset:

```rust
pub(crate) fn inline_resets(&mut self, out: &mut OpBuffer) -> Result<(), ExpansionError> {
    let inline_mask = inline_register_mask();
    if self.live & inline_mask != 0 {
        return Err(ExpansionError::InlineRegistersStillAllocated);
    }

    let mut pending = self.inline_touched & inline_mask;
    while pending != 0 {
        let index = pending.trailing_zeros() as u8;
        pending &= !(1u128 << index);
        out.row(i(InstructionKind::ADDI, RISCV_REGISTER_BASE + index, 0, 0))?;
    }
    self.inline_touched &= !inline_mask;
    Ok(())
}
```

This design is both faster and easier to prove than `[bool; N] + Vec<u8>`.

## Buffer Design

Use local fixed-capacity buffers for per-source expansion:

```rust
pub(crate) struct FixedVec<T: Copy, const N: usize> {
    len: usize,
    data: [T; N],
}
```

`RawRow`, `ExpansionOp`, and `NormalizedInstruction` are `Copy`, so this can avoid `MaybeUninit` in the extraction-critical core. Overflow returns `ExpansionError::ExpansionBufferExceeded { capacity: N }`.

Suggested buffers:

```rust
pub(crate) type WorkStack = FixedVec<ExpansionOp, MAX_WORK_OPS_PER_SOURCE>;
pub(crate) type OpBuffer = FixedVec<ExpansionOp, MAX_SHALLOW_OPS_PER_LOWERING>;
pub(crate) type ExpansionBuffer = FixedVec<RawRow, MAX_FINAL_ROWS_PER_SOURCE>;
```

The exact capacities should be set from observed maximum expansion lengths plus margin, then guarded by tests over the curated parity corpus. This is not merely for extraction: bounded buffers remove recursive heap allocation from the runtime hot path.

The top-level program API can still return a heap-backed `Vec<NormalizedInstruction>`:

```rust
pub fn expand_program_slice(
    instructions: &[NormalizedInstruction],
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut state = ExpansionState::new();
    let mut expanded = Vec::with_capacity(estimate_expanded_len(instructions));
    for instruction in instructions {
        let rows = core::expand_one_core(SourceRow::from(*instruction), &mut state)?;
        expanded.extend_from_slice(rows.as_slice());
    }
    Ok(expanded)
}
```

Ergonomic `impl IntoIterator` wrappers can remain outside the extraction target.

## Metadata Stamping

Rows should not be partially initialized with placeholder metadata. The finalizer should construct `NormalizedInstruction` rows from `RawRow` values:

```rust
pub(crate) fn stamp_row(
    source: SourceRow,
    row: RawRow,
    index: usize,
    len: usize,
) -> Result<NormalizedInstruction, ExpansionError> {
    let remaining = len
        .checked_sub(index + 1)
        .ok_or(ExpansionError::MalformedExpansion)?;
    let remaining = u16::try_from(remaining)
        .map_err(|_| ExpansionError::ExpansionTooLong { len })?;

    Ok(NormalizedInstruction {
        instruction_kind: row.kind,
        address: source.address,
        operands: row.operands,
        virtual_sequence_remaining: Some(remaining),
        is_first_in_sequence: index == 0,
        is_compressed: index + 1 == len && source.is_compressed,
    })
}
```

For a side-effect-free `rd = x0` no-op, this still stamps the source address and compressed metadata according to current behavior.

## Inline Handling

The provider-free core should treat `InstructionKind::Inline` as unsupported:

```rust
fn process_row(...) -> Result<(), ExpansionError> {
    if row.kind == InstructionKind::Inline {
        return Err(ExpansionError::InlineProviderRequired);
    }
    ...
}
```

The provider-taking API should sit outside the extracted core:

```rust
pub trait InlineExpansionProvider {
    fn expand_inline(
        &mut self,
        source: SourceRow,
    ) -> Result<InlineExpansion, ExpansionError>;
}

pub struct InlineExpansion {
    rows: InlineRowBuffer,
}
```

The adapter can intercept inline source rows, ask tracer's registry for the normalized inline sequence, and push those rows into the same core driver. If provider-owned inline allocation needs reset rows, the provider should return explicit reset rows or an explicit reset operation. The core should not call a trait object while processing provider-free RV64 rows.

This preserves the dependency boundary from PR #1490: `jolt-program` still does not depend on `tracer`, inventory, CPU state, or advice tapes.

## Termination And Expansion Classes

The current recursive implementation relies on the absence of cycles in helper expansion. The new driver should make this explicit.

Add an expansion classification:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ExpansionRank {
    Final = 0,
    VirtualHelper = 1,
    ShallowSynthetic = 2,
    SourceOnly = 3,
}

pub(crate) const fn expansion_rank(kind: InstructionKind) -> ExpansionRank {
    match kind {
        InstructionKind::ADDIW
        | InstructionKind::ADDW
        | InstructionKind::DIV
        | InstructionKind::LW
        | InstructionKind::SW => ExpansionRank::SourceOnly,
        ...
        _ => ExpansionRank::Final,
    }
}
```

Every shallow lowerer should emit rows with strictly lower rank than the row being lowered, or a test should document the alternative termination measure for that family. Add tests that collect emitted kinds from each lowerer over representative operands and assert the expansion graph is acyclic.

The driver should also enforce a fuel bound:

```rust
const MAX_EXPANSION_OPS_PER_SOURCE: u32 = 4096;
```

Fuel exhaustion should be treated as an internal malformed-expansion error and should never occur in parity fixtures.

## Performance Expectations

This rewrite should not regress performance. It should improve or preserve it for three reasons:

- per-source recursive `Vec` allocation disappears;
- allocator operations become bit operations and bounded scans;
- metadata stamping becomes a single construction pass instead of mutate-after-build.

Potential cost:

- fixed buffers use stack space;
- the central work-stack driver adds an explicit dispatch loop.

The explicit loop replaces recursive Rust calls and should be comparable or cheaper. Stack size should be bounded by chosen capacities and checked in review.

Benchmark expectations:

- no measurable regression in decode-plus-expansion for representative guests;
- no measurable regression in trace length accounting;
- allocation count during expansion should drop relative to `a3448e6da44f`.

Suggested measurements:

```bash
cargo nextest run -p jolt-program --cargo-quiet
cargo nextest run -p tracer --cargo-quiet
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk

# Optional follow-up benchmark if maintainers want measured evidence:
cargo run --release -p jolt-core profile --name sha3 --format chrome
```

## API Shape

The extraction-critical API should be concrete:

```rust
pub(crate) fn expand_one_core(
    source: SourceRow,
    state: &mut ExpansionState,
) -> Result<ExpandedRows, ExpansionError>;

pub fn expand_program_slice(
    instructions: &[NormalizedInstruction],
) -> Result<Vec<NormalizedInstruction>, ExpansionError>;
```

Compatibility wrappers can keep call sites pleasant:

```rust
pub fn expand_program(
    instructions: impl IntoIterator<Item = NormalizedInstruction>,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let collected: Vec<_> = instructions.into_iter().collect();
    expand_program_slice(&collected)
}
```

If that collection is too costly for a hot call site, add a concrete streaming adapter outside the extracted module:

```rust
pub fn expand_program_iter<I>(
    instructions: I,
) -> Result<Vec<NormalizedInstruction>, ExpansionError>
where
    I: Iterator<Item = NormalizedInstruction>,
{
    let mut state = ExpansionState::new();
    let mut expanded = Vec::new();
    for instruction in instructions {
        let rows = core::expand_one_core(SourceRow::from(instruction), &mut state)?;
        expanded.extend_from_slice(rows.as_slice());
    }
    Ok(expanded)
}
```

The extracted core should not be generic over `IntoIterator`.

## Cargo Feature Shape

The extraction-critical module graph should compile without serialization and host-only dependencies.

Suggested feature split:

```toml
[features]
default = ["std", "serde", "ark-serialize"]
std = ["common/std"]
serde = ["dep:serde", "jolt-riscv/serde"]
ark-serialize = ["dep:ark-serialize", "jolt-riscv/ark-serialize"]
image = ["dep:object"]
```

If changing workspace feature defaults is too much for this phase, at minimum ensure a Hax/Aeneas start set rooted in `expand::core` does not reference serialization impls.

## Migration Plan

1. Add new `core`, `buffer`, and bitset `allocator` internals under `jolt-program::expand`.
2. Port one small family, such as ADDIW/ADDW/SUBW, to shallow lowering and prove parity against the current output.
3. Port arithmetic, shifts, memory, division, and control-flow families.
4. Replace `InstrAssembler<'a>` in production expansion code.
5. Update tracer inline adapter to produce explicit inline expansion rows or explicit reset rows.
6. Delete the old recursive assembler once all parity tests pass.
7. Run Hax/Aeneas again on:
   - metadata stamping,
   - allocator transitions,
   - ADDIW shallow lowering,
   - provider-free `expand_one_core`.
8. Run formatting, clippy, host tests, ZK tests, and dependency checks.

Do not leave both expanders in production. A temporary test-only reference path is acceptable during the rewrite, but the final branch should have one canonical production expander.

## Acceptance Criteria

- [ ] `jolt-program::expand` no longer has a production `InstrAssembler<'a>` that stores a borrowed allocator.
- [ ] Family lowerers are shallow and do not call `expand_instruction`.
- [ ] Recursive expansion happens in one central depth-first driver.
- [ ] Temporary-register release and inline reset are explicit expansion operations.
- [ ] Allocator state is represented by bitsets, not a heap-backed reset list.
- [ ] Per-source expansion uses bounded buffers, with explicit overflow errors.
- [ ] Metadata is stamped during final row construction, not by mutating already-built rows.
- [ ] Provider-free core expansion has a concrete, non-generic entry point over `SourceRow` and `ExpansionState`.
- [ ] Inline provider support is an adapter outside the provider-free core.
- [ ] Expansion output matches PR #1490 baseline fixtures exactly.
- [ ] Hax and Aeneas can extract metadata stamping and at least one shallow family lowerer without pulling in execution/preprocess/serialization modules.
- [ ] Dependency checks still show no `tracer` dependency from `jolt-program` or `jolt-riscv`.

## Open Questions

- What fixed capacities should be used for `WorkStack`, `OpBuffer`, and `ExpansionBuffer`? The answer should come from measuring the current curated expansion corpus plus a conservative margin.
- Should `ExpansionRank` be manually maintained or generated from a declarative lowering table?
- Should inline provider output be normalized rows, raw rows, or explicit `ExpansionOp` values?
- Should serialization derives on `NormalizedInstruction` be feature-gated in this phase, or is it enough to keep them outside the extraction start set?
- Does any current helper expansion rely on allocator reuse earlier than a simple ordered `Release` marker would allow? Parity tests should answer this before implementation lands.

