# Spec: RV64-Only Instruction Phases And Compiler-Native Bytecode Expansion

| Field       | Value                                                                 |
|-------------|-----------------------------------------------------------------------|
| Author(s)   | Quang Dao                                                            |
| Created     | 2026-05-05                                                           |
| Status      | draft                                                                 |
| Related PR  | [#1490](https://github.com/a16z/jolt/pull/1490), [#1518](https://github.com/a16z/jolt/pull/1518) |
| Baseline    | `main` after [#1490](https://github.com/a16z/jolt/pull/1490), merge commit `51d81a36e` |
| Depends on  | `specs/bytecode-expansion-crate.md`                                   |

## Summary

PR #1490 moves bytecode expansion out of `tracer` and into
`jolt-program::expand`. That crate boundary is the right direction for formal
verification, but the merged implementation still leaves two cleanup/design
issues ahead of the compiler-native rewrite:

1. `jolt-program` is RV64-only, but `tracer` still carries historical RV32
   execution, decode, uncompression, `Xlen::Bit32`, and documentation paths.
2. `InstructionKind` is a flat row identity while `JoltInstructions<T>` is a
   typed enum with a different and currently confusing meaning. The next pass
   should name source RISC-V kinds, expanded Jolt bytecode kinds, and
   lookup-backed kinds explicitly before baking the ambiguity into the
   extraction-oriented expander.

This spec therefore prioritizes the next PR sequence as:

```text
Phase 1: remove historical RV32 support from tracer and stale docs
Phase 2: split instruction phase identities
Phase 3: rewrite provider-free jolt-program::expand as a compiler-native core
```

Phase 1 should come first because it deletes width-dependent branches from the
emulator, decoder, virtual helpers, and expansion call sites before Phase 2
names the RV64 source/target row universe. Phase 2 should come before the
compiler-native rewrite because legality predicates and recipe boundaries are
clearer once source rows, expanded bytecode rows, and lookup-backed rows have
separate names.

After those two setup phases, the current `jolt-program::expand` implementation
still uses an idiomatic recursive Rust assembler shape that is hard for Hax and
Aeneas to extract:

- family expanders build `Vec<NormalizedInstruction>` values;
- `InstrAssembler<'a>` owns a sequence while borrowing `&'a mut ExpansionAllocator`;
- `InstrAssembler::emit` recursively calls `expand_instruction`;
- temporary-register release is encoded in Rust call-stack/control-flow order;
- metadata is stamped by mutating a finished slice;
- inline expansion is a trait callback inside the core dispatch path.

The compiler-native phase proposes a rewrite of `jolt-program::expand` into an
extraction-friendly production implementation. The goal is not to add a
proof-only model next to production code. The production expander itself should
become a first-order lowering pipeline over explicit data transitions, while
preserving byte-for-byte output and keeping runtime performance the same or
better.

This rewrite should also align with the MLIR-shaped Jolt work in the
`refactor/crates` branch, which currently treats Bolt as a compiler pipeline
over explicit dialects, passes, schema validation, typed plans, and generated
artifacts. Bytecode expansion is smaller than Bolt's prover/verifier pipeline,
but it has the same compiler shape: a source IR, a target IR, legality
constraints, lowering rules, resource materialization, validation, and
emission into production Rust data structures.

The target design:

```text
Decoded NormalizedInstruction
  -> Source(NormalizedInstruction)
  -> syntactic expansion recipe / shallow lowerer
  -> rd=x0 normalization
  -> depth-first work stack of ExpansionOp
  -> explicit temp release/reset operations
  -> bounded per-source output buffer
  -> Expanded(NormalizedInstruction)
  -> Stamped(NormalizedInstruction)
  -> top-level Vec extension
```

## Goals

- Remove historical RV32 support from `tracer` and stale dual-width docs so the
  host/tracer/program stack is consistently RV64IMAC-only.
- Split instruction identity by phase:
  - source RISC-V instruction kinds decoded from an RV64 program;
  - expanded Jolt bytecode row kinds consumed by preprocessing/proving;
  - lookup-backed instruction kinds used by lookup-table routing.
- Preserve expansion behavior exactly relative to `main` after PR #1490, except
  for changes that are explicit consequences of RV32 deletion or the
  instruction-kind phase split.
- Preserve recursive expansion order exactly.
- Preserve `rd = x0` behavior for all source and helper rows.
- Preserve virtual-register numbering, allocation reuse, reserved registers, and inline reset behavior.
- Preserve sequence metadata: source address, `virtual_sequence_remaining`, `is_first_in_sequence`, and compressed-instruction metadata.
- Make expansion recipes syntactic and inspectable enough to drive production Rust, Lean extraction/model generation, docs, and parity fixtures from one source of truth.
- Keep the recipe representation MLIR-ready: represent instruction emission, temps, metadata, and reset behavior as typed ops/attrs/regions that can later be moved into an MLIR dialect without changing expansion behavior.
- Keep `jolt-program::expand` free of tracer, CPU, memory-device, advice-tape, prover, transcript, ELF parser, and PCS dependencies.
- Make Hax/Aeneas extraction of provider-free RV64 expansion straightforward enough that the first real extraction target is the production core, not a hand-written mirror.
- Avoid performance regressions by removing recursive per-instruction heap allocation and using bounded stack-resident buffers for per-source expansion.

## Non-Goals

- Do not change instruction semantics.
- Do not add a new RV32 compatibility layer, aliases, or migration shims.
- Do not change committed bytecode/proof semantics merely to rename instruction
  kinds. The split should make phase boundaries explicit while preserving the
  current RV64 expanded rows and lookup behavior.
- Do not define a structured grammar for instruction execution semantics in the
  compiler-native expansion phase. The expansion grammar is only a syntactic
  lowering language: it says that a source row such as `DIV` maps to a sequence
  of bytecode rows, but it does not define the operational meaning of `DIV` or
  of those target rows. A separate semantics track should handle execution
  meaning and expansion-correctness theorems.
- Do not change bytecode preprocessing, RAM preprocessing, or proof-system APIs except for call-site adjustments needed by the new expansion API.
- Do not formalize tracer custom inline registries in this phase.
- Do not port registered `jolt-inlines` recipes or advice builders into the
  grammar in this implementation sequence. Advice handling is explicitly
  unresolved for the grammar; provider-owned inline/advice behavior stays
  behind the adapter boundary in this phase.
- Do not make Aeneas/Hax extraction a hard CI requirement in the implementation PR unless maintainers explicitly ask for it.
- Do not keep the current recursive `InstrAssembler<'a>` implementation as a compatibility layer once the rewrite lands. This branch owns the new `jolt-program` implementation, so the rewrite should be a full cutover.

## Baseline: Post-#1490 Expand Shape

The merged #1490 implementation lives under:

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

| Concern | Current post-#1490 shape | Target shape |
|---------|-------------------------------|--------------|
| Source of truth | Arbitrary Rust functions using an imperative assembler API | Inspectable syntactic expansion recipes or shallow lowerers |
| Expansion state | Split between `InstrAssembler<'a>` and borrowed `&mut ExpansionAllocator` | One owned `ExpansionState` containing allocator, work stack, and output buffer |
| Family lowering | Expander calls `asm.emit_*`; each emit recursively expands immediately | Expander appends shallow `ExpansionOp` values only |
| Recursion | Hidden inside `InstrAssembler::emit` | One central depth-first driver |
| Temp lifetime | Encoded by Rust control flow and post-`finalize()` releases | Explicit `ExpansionOp::Release(register)` markers |
| Output allocation | Per-source `Vec` plus recursive helper `Vec`s | Bounded per-source buffer plus one top-level program `Vec` |
| Metadata | Mutate `&mut [NormalizedInstruction]` after synthetic sequences are built; non-expanded source rows pass through unchanged | Explicit stage policy: source pass-through, expanded helper rows, stamped synthetic sequence |
| Allocator | `[bool; N]` plus `Vec<u8>` reset list | Bitsets for live registers and inline reset set |
| Inline expansion | Trait callback inside core dispatcher | Adapter outside provider-free core |
| API ergonomics | `impl IntoIterator`, trait generic provider path | Concrete slice/state core; ergonomic wrappers outside |
| Extracted call graph | Pulls in much of `jolt-program` and serde/ark derives | Small `expand` core module graph |

## Target Module Layout

Suggested module layout:

```text
crates/jolt-program/src/expand/
  mod.rs             public adapters and compatibility API
  grammar.rs         syntactic recipe data and checked helpers
  core.rs            RowStage, ExpansionOp, ExpansionState, driver
  lower.rs           dispatch from helper rows to shallow lowerers/recipe interpreter
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

The important separation is `grammar + core + lower + allocator + buffer + metadata + operands` versus `inline + public ergonomic wrappers`. The extraction target should be the first group.

## Compiler And MLIR Alignment

The `refactor/crates` branch at `4e6c4a635` contains Markos's Bolt-shaped view
of Jolt as an explicit compiler pipeline. Bolt's current docs describe the
generic lowering path as:

```text
protocol -> concrete -> party -> compute -> cpu -> Rust
```

The key rule is that compiler facts live in dialect ops, validators, lowering
passes, or typed plan data before Rust emission. Generated Rust is the final
artifact, not where protocol meaning should be rediscovered by string matching
or ad hoc helper logic. The Bolt paper draft in
`/Users/quang.dao/Documents/SNARKs/bolt` makes the same architectural point:
dialects are domain vocabularies of typed operations and attributes, passes are
partial functions between modules, and each lowering pass should state the
invariants it preserves.

Bytecode expansion should use the same mental model:

```text
rv64.source
  -> expand.canonical
  -> expand.lowered
  -> expand.allocated
  -> jolt.bytecode
  -> Rust Vec<NormalizedInstruction>
```

These names are conceptual phase markers, not a request to introduce MLIR now.
They should guide the Rust design so that a future MLIR dialect can be a
mechanical lift rather than another rewrite.

### Source And Target

Today source and target are both represented by `NormalizedInstruction`, and
that should stay true. The compiler boundary should be defined by legality
predicates and stage policy rather than by a new family of row wrapper types:

```rust
pub(crate) fn is_source_row(row: NormalizedInstruction) -> bool;
pub(crate) fn is_target_bytecode_row(row: NormalizedInstruction) -> bool;
```

The source IR is a decoded RV64 program row with address, compressed flag, and
source operands. It may contain instructions that are not legal final Jolt
bytecode rows, such as `ADDIW`, `MULH`, `LB`, `SCW`, `DIV`, `Inline`, and CSR
operations.

The target IR is the bytecode stream consumed by preprocessing and the proof
system: every row is legal for the final bytecode relation, internal virtual
register use has been materialized, sequence metadata is correct, and
pass-through or literal rows preserve their baseline metadata policy.

The expansion contract is syntactic. This module says which bytecode rows are
emitted for a decoded source row and how virtual registers, recursive helper
expansion, and sequence metadata are materialized. It does not define the
execution semantics of the source instruction or of the target instructions.
Those semantics can be hand-modeled separately if needed. For this PR branch,
row-for-row parity against the current Rust output remains the first acceptance
test except where this PR intentionally fixes a documented baseline bug.

### Execution Semantics Track

Execution semantics should be a separate artifact from bytecode expansion.
Today production execution lives in `tracer`, but `tracer` is an emulator with
CPU state, memory devices, advice plumbing, inline registries, and host
conveniences. It is the implementation to test against, not the cleanest
semantic source of truth.

The coherent split is:

```text
Expansion recipe:
  source NormalizedInstruction -> target bytecode rows

Execution semantics:
  instruction + abstract machine state -> next abstract machine state

Expansion correctness:
  executing the expanded target sequence refines executing the source row
```

The recommended first semantics pass is hand-modeled Lean, not MLIR and not
extraction from `tracer`. Define a small abstract machine state and a transition
relation for a provider-free instruction slice:

```text
step : NormalizedInstruction -> MachineState -> Result MachineState Trap
```

Then prove or test the expansion theorem for that slice:

```text
exec_seq (expand instr) (extend state)
  projects_to
step instr state
```

Start with a narrow family such as `ADDI`, `ADD`, shifts, loads/stores, and one
virtual helper instruction. That slice should reveal whether hand-modeled Lean
stays manageable or whether a separate semantic DSL is worth building.

If a semantic DSL becomes useful, it should be a different language from the
expansion grammar. It would describe effects such as register reads/writes,
memory reads/writes, branches, traps, and advice reads, and could generate a
Rust reference interpreter, Lean definitions, and MLIR op interfaces or
documentation. MLIR should act as a carrier for dialect structure, legality,
lowering, and op metadata; it should not be the first source of proof
semantics.

Existing RISC-V formal models such as Sail may be useful for the base ISA, but
they will not cover Jolt virtual helper instructions or Jolt-specific execution
conventions. Even if Sail is used for RV64 semantics, Jolt still needs its own
semantic layer for target bytecode rows.

### Advice Boundary

Advice should also be split by phase. The current code uses "advice" for three
different channels, and the compiler-native rewrite should preserve them while
naming the boundaries more precisely.

1. **Advice-load source instructions.**
   `AdviceLB`, `AdviceLH`, `AdviceLW`, and `AdviceLD` are decoded guest/source
   instructions. Provider-free expansion owns their syntactic lowering:

   ```text
   AdviceLB rd -> VirtualAdviceLoad rd, 1; SLLI rd, rd, 56; SRAI rd, rd, 56
   AdviceLH rd -> VirtualAdviceLoad rd, 2; SLLI rd, rd, 48; SRAI rd, rd, 48
   AdviceLW rd -> VirtualAdviceLoad rd, 4; SLLI rd, rd, 32; SRAI rd, rd, 32
   AdviceLD rd -> VirtualAdviceLoad rd, 8
   ```

   `VirtualAdviceLoad` execution reads bytes from the runtime advice tape and
   writes the result to `rd`. Expansion emits the row and byte length; it does
   not produce the tape contents.

2. **Trace-time advice payloads.**
   `VirtualAdvice` is a target bytecode row whose concrete tracer instruction
   carries an extra `advice: u64` payload. That payload is not part of
   `NormalizedInstruction`. Today it is patched by tracer-side logic, including
   registered inline `build_advice` functions and LR/SC success-bit handling for
   `SCW`/`SCD`.

   Expansion may emit `VirtualAdvice`, but provider-free expansion must not
   assign its payload. Inline/provider code may continue returning finalized
   rows for this phase, and tracer may continue patching concrete
   `VirtualAdvice` instructions during trace construction.

3. **Committed advice memory.**
   `trusted_advice` and `untrusted_advice` are byte arrays placed in Jolt device
   memory and committed as advice polynomials by the prover. This is
   preprocessing/proof behavior, not bytecode expansion behavior. The expansion
   rewrite should not move or reinterpret those commitments.

The better long-term shape is:

```text
ExpansionRecipe:
  emits bytecode rows, including VirtualAdvice and VirtualAdviceLoad

AdvicePlan:
  describes how trace-time VirtualAdvice payloads or advice-tape bytes are produced

ExecutionSemantics:
  defines how advice-consuming rows read those payloads/tape bytes
```

For the semantics follow-up, advice should first be modeled abstractly as an
oracle or tape in the Lean machine state. A concrete `AdvicePlan` or advice DSL
should be introduced only after one advice-bearing slice shows what state the
plan must observe and how slots/tape positions should be named. MLIR can carry
effect metadata such as `reads_advice_tape(byte_len)` or
`requires_trace_advice(slot)`, but it should not hide the source of advice
values inside expansion recipes.

### Lowering Pipeline

The proposed recipe/driver shape should map onto ordinary compiler passes:

```text
decode/uncompress
  produces rv64.source rows

canonicalize-source
  normalizes root-level rd=x0 behavior, pass-through rows, literal metadata policies, and documented baseline fixes

validate-source
  rejects malformed or unsupported source rows such as CSRRW/CSRRS with CSR address 0

legalize-expansion
  repeatedly rewrites illegal source/helper ops with instruction-family recipes

inline-fragments
  expands recipe fragments such as amo_pre64/amo_post64 with alpha-renamed temps

allocate-temps
  materializes symbolic temps into concrete virtual registers using deterministic first-fit allocation

reset-inline-temps
  emits reset rows for touched inline registers after checking no inline temps are live

materialize-metadata
  stamps synthetic sequence metadata and preserves pass-through/literal metadata

verify-target
  checks target legality, no live temps, bounded sequence size, and dependency hygiene

emit-rust
  appends final rows to Vec<NormalizedInstruction>
```

The current implementation conflates most of these passes inside
`InstrAssembler::emit`. The compiler-native rewrite should separate them in
the data model even if the initial Rust implementation fuses several passes for
performance.

### Compiler Concepts In This Rewrite

The rewrite is using standard compiler ideas under different names:

| Bytecode expansion concept | Compiler concept | Why it matters here |
|----------------------------|------------------|---------------------|
| `InstructionKind` families | Dialect operation set | Each instruction is an op with operand shape, side effects, and legality. |
| Source-only vs final kinds | Legalization target | Expansion is a conversion from illegal source ops to legal target ops. |
| `LowerStmt::Emit` | Rewrite pattern / conversion pattern | A recipe replaces one op with a sequence of lower-level ops. |
| Recursive helper expansion | Conversion driver / worklist legalization | Emitted helper ops must themselves be legalized until all target ops are legal. |
| Work fuel / recursion depth | Termination guard | Legalization needs a simple bound so accidental cycles become errors. |
| `Seq`, `If`, `WithTemp` | Structured regions/control flow | Recipes need regions instead of arbitrary Rust control flow. |
| `Fragment` | Symbolic helper / inlining | Shared lowering snippets should be named regions with explicit arguments and alpha-renamed locals. |
| Operand refs and conditions | Pure row syntax | Operand selection should be separate from stateful expansion effects. |
| `TempId` | Virtual register / temporary SSA value | Recipes should name symbolic temps before concrete register allocation. |
| `ExpansionAllocator` | Register allocation / bufferization | Concrete virtual-register numbers are resource materialization, not semantic lowering. |
| `Release` | Lifetime end / deallocation | Reuse requires explicit lifetime boundaries and validation. |
| `RowStage::{Source, Expanded, Stamped}` | Attribute/materialization policy | Metadata is an emitted attribute policy, not an incidental mutation. |
| Recipe checks | IR verifier/schema validation | Bad recipes should be rejected structurally where practical. |
| `expand_program_slice` | Compiler driver | The public API orchestrates passes and emits the final artifact. |

The most important adjustment to the previous grammar proposal is to treat
concrete virtual-register assignment as a materialization pass. Recipes should
prefer symbolic temps:

```rust
WithTemp {
    temp: T0,
    pool: RegisterPool::Instruction,
    body: Seq(&[
        Emit(i(ADDI, Temp(T0), SourceRs1, imm(0))),
        Emit(i(ADDI, SourceRd, Reserved(CsrTarget), imm(0))),
        Emit(i(ADDI, Reserved(CsrTarget), Temp(T0), imm(0))),
    ]),
}
```

The driver may still allocate the concrete register at `WithTemp` execution
time for performance and exact parity. Conceptually, however, the grammar has a
symbolic temporary whose live range is the `WithTemp` body. This is much closer
to MLIR/SSA form, and it makes the future MLIR lowering natural:

```mlir
%t0 = expand.alloc_temp {pool = "instruction"} : !expand.vreg
expand.emit "ADDI"(%t0, %rs1) {imm = 0}
expand.emit "ADDI"(%rd, %csr_target) {imm = 0}
expand.emit "ADDI"(%csr_target, %t0) {imm = 0}
expand.release_temp %t0
```

That IR can later be lowered to concrete Jolt bytecode by a register-allocation
pass that uses the same first-fit policy as today's Rust allocator.

### MLIR-Ready Shape

The Rust recipe surface should avoid choices that would be awkward in MLIR:

- Prefer named ops with typed operands/attrs over free-form closures.
- Prefer explicit regions (`Seq`, `If`, `WithTemp`) over Rust call-stack
  effects.
- Prefer symbolic temps plus a deterministic allocation pass over hard-coded
  register numbers in recipes.
- Store operand shapes, side-effect classes, and target legality as typed data
  or traits that a verifier can inspect.
- Keep baseline quirks and intentional fixes as explicit ops, attrs, or verifier
  rules. A true literal row can still be represented as
  `expand.literal_default_row`, but the CSR-zero fix should be represented as a
  validation failure such as `expand.fail unsupported_csr`, not hidden inside a
  Rust branch.
- Treat metadata stamping as attribute materialization at a phase boundary.
- Keep schema/validator tests close to the grammar, the same way Bolt validates
  transcript threading, party projection, kernel-free verifier IR, and import
  boundaries.

Possible future dialect split:

```text
riscv.norm      decoded RV64 source rows and operand formats
jolt.bytecode   final legal Jolt bytecode ops and sequence attrs
expand          temporary lowering ops: emit, alloc_temp, release_temp, reset_inline, literal
```

The near-term Rust implementation does not need to expose these names publicly.
It should, however, make each concept visible enough that moving to MLIR's
dialect conversion infrastructure later would mostly replace the driver and
grammar interpreter, not the specification.

### Literature Pointers

Useful compiler concepts to read alongside this design:

- **SSA form.** The classic reference is Cytron et al.,
  "Efficiently Computing Static Single Assignment Form and the Control
  Dependence Graph" (TOPLAS 1991). Bolt's paper draft uses SSA as the default
  representation style; our symbolic `TempId` discipline is the bytecode
  expansion analogue.
- **MLIR dialects and multi-level lowering.** Lattner et al.,
  "[MLIR: Scaling Compiler Infrastructure for Domain Specific Computation](https://research.google/pubs/mlir-scaling-compiler-infrastructure-for-domain-specific-computation/)"
  (CGO 2021) is the main reference. The relevant lesson is not "use MLIR now";
  it is "keep each abstraction level explicit and lower through typed dialects."
- **Dialect conversion / legalization.** MLIR's
  [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
  documentation describes conversion targets, rewrite patterns, type
  conversion, and legality. Our `is_target_bytecode_row`, lowering recipes, and
  worklist driver are the same idea in smaller Rust form.
- **Pattern rewriting.** MLIR's docs index points to generic DAG rewriting and
  table-driven declarative rewrite rules. Our recipes should be declarative
  rewrite patterns where possible, with Rust builders used only to construct
  inspectable recipe data.
- **Register allocation, liveness, and bufferization.** Temp materialization and
  explicit `Release` markers are a small register-allocation problem. The Bolt
  paper's `bufferize`/`cpu` discussion is the closest local analogy: pure SSA
  values become explicit mutable resources only near the target boundary.

## Declarative Expansion Grammar

Ari's [`Lost in Translation`](https://randomwalks.xyz/blog/translations/) reaches the right general conclusion for Jolt verification: arbitrary Rust is the wrong long-term source of truth. His article applies that idea to instruction execution semantics. This spec is narrower. It does not propose a CPU-state semantics AST or an execution-semantics DSL. The bytecode expansion grammar is an expansion-time syntax transformer: it turns one `NormalizedInstruction` source row into zero or more `NormalizedInstruction` bytecode rows while preserving allocation, recursive helper expansion, and metadata behavior.

The recipe surface must be expressive enough to model the current implementation, including the inconvenient parts:

- Emitted helper rows are not final output immediately. They go back through the central expansion driver, exactly like `InstrAssembler::emit` currently calls `expand_instruction`.
- Some source rows are pass-through rows, not synthetic sequences. A source `ADD` currently returns the input `NormalizedInstruction` unchanged; it does not get `virtual_sequence_remaining = Some(0)` or `is_first_in_sequence = true`.
- Synthetic sequences do get finalized metadata. Current family expanders use `InstrAssembler::finalize`, which rewrites every row's sequence metadata and puts `is_compressed` only on the last row.
- `rd = x0` has two separate behaviors: side-effect-free rows become a direct ADDI no-op, while side-effecting rows are recursively expanded after rewriting `rd` to a temporary register and releasing that temporary after expansion.
- Temporary-register allocation order is observable through emitted register numbers. Recipes or lowerer outputs must expose temp lifetimes at precise points, and the materialization pass must preserve today's deterministic allocation order.
- Temporary-register release timing matters for reuse inside longer expansions such as `SCD`, `SCW`, CSR updates, and division.
- Some branches depend on decoded operands or expansion parameters, for example `rd == rs1`, `rs1 == x0`, `csr == 0`, `word`, `signed`, `min`, and `remainder_output`.
- Shared snippets such as `amo_pre64` and `amo_post64` are real grammar fragments with parameters, not arbitrary Rust helper functions.
- Errors are expansion outcomes. Missing operands, unsupported CSRs, virtual-register exhaustion, and inline-provider requirements should be explicit recipe/driver results.
- Baseline quirks and deliberate baseline fixes must both be made visible. The historical `CSRRW`/`CSRRS` behavior for `csr == 0` returned `NormalizedInstruction::default()` directly, bypassing assembler finalization and producing an address-zero row that could be skipped by bytecode PC mapping. This PR intentionally rejects that source row as `UnsupportedCsr(0)`, so the compiler-native rewrite should model the case as a source validation failure or recipe `Fail`, not as a literal default row to preserve.

The key design rule is to keep the recipe surface first-order and inspectable,
not to pre-commit to a large expression language. Separate pure operand
selection from expansion-time statements where that buys clarity, but introduce
only the expression forms that real expansion families need.

The minimal durable grammar is likely closer to:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RegRef {
    Zero,
    SourceRd,
    SourceRs1,
    SourceRs2,
    Temp(TempId),
    Reserved(ReservedReg),
    Const(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImmRef {
    SourceImm,
    Const(i128),
    Derived(DerivedImm),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cond {
    RdEqZero,
    Rs1EqZero,
    RdEqRs1,
    CsrEq(u16),
    CsrSupported,
    Param(ExpansionParam),
}
```

`RegRef`, `ImmRef`, and `Cond` are pure. They may inspect the current source
row, recipe parameters, named temps, and reserved-register map, but they cannot
allocate, release, emit rows, or mutate output.

If a family truly needs compound immediate or condition expressions, add the
smallest first-order representation that covers that family. Avoid heap-backed
`Box` trees and Rust closures in the extraction-critical core. Constructor
helpers can still make recipes readable, but `finish()` should return ordinary
data that tests and extractors can inspect.

Rows are also pure specifications:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowTemplate {
    R {
        kind: InstructionKind,
        rd: RegRef,
        rs1: RegRef,
        rs2: RegRef,
    },
    I {
        kind: InstructionKind,
        rd: RegRef,
        rs1: RegRef,
        imm: ImmRef,
    },
    // Add S/B/J/U/Align variants only if direct row construction helpers are
    // not enough for the families being ported.
}
```

Expansion-time statements are the only layer that changes allocator or output state:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LowerStmt {
    Seq(&'static [LowerStmt]),
    Emit(RowTemplate),
    WithTemp {
        temp: TempId,
        pool: RegisterPool,
        body: &'static LowerStmt,
    },
    If {
        cond: Cond,
        then_body: &'static LowerStmt,
        else_body: &'static LowerStmt,
    },
    Fragment {
        fragment: FragmentId,
        args: FragmentArgs,
    },
    Release(TempId),
    ReturnLiteral(LiteralRow),
    Fail(ExpansionErrorKind),
    ResetInlineRegisters,
}
```

This enum is a sketch, not a required minimum surface. During implementation,
pare it down aggressively. `Seq`, `Emit`, `WithTemp`, `If`, `Fragment`,
`ReturnLiteral`, `Fail`, and reset/release markers are the concepts that seem
stable; the exact helper enums should follow the code.

`Emit(RowTemplate)` means "push a helper row back through the expansion
driver." It does not mean "append this final instruction to output." This
distinction is what preserves current recursive behavior without recursive Rust
calls.

`WithTemp` introduces a symbolic temp whose live range is exactly the statement body. The current Rust interpreter may materialize the concrete virtual register when entering the statement and lower to an explicit `Release` operation after the body. Nested `WithTemp` blocks are the grammar form for current mid-sequence release/reuse patterns:

```rust
// Shape of the current CSRRW rd == rs1 branch.
WithTemp {
    temp: T0,
    pool: RegisterPool::Instruction,
    body: Seq(&[
        Emit(i(ADDI, Temp(T0), SourceRs1, Const(0))),
        Emit(i(ADDI, SourceRd, Reserved(CsrTarget), Const(0))),
        Emit(i(ADDI, Reserved(CsrTarget), Temp(T0), Const(0))),
    ]),
}
```

The grammar compiler/interpreter should materialize `T0` at this point and emit the `Release(T0)` after the third row has recursively expanded and before any following statement. That gives the same observable allocation reuse as the current hand-written Rust while keeping the recipe itself in symbolic-temp form.

A simple lowerer becomes data:

```rust
pub(crate) const ADDIW_RECIPE: LowerStmt = Seq(&[
    Emit(i(ADDI, SourceRd, SourceRs1, SourceImm)),
    Emit(i(VirtualSignExtendWord, SourceRd, SourceRd, Const(0))),
]);
```

A parameterized family becomes either a recipe with parameters or a small recipe builder whose output is still grammar, not arbitrary emit calls:

```rust
pub(crate) fn load_recipe(
    kind: InstructionKind,
    sign_extension_shift: Option<i128>,
) -> Recipe {
    let mut recipe = RecipeBuilder::new();
    recipe.emit(i(kind, SourceRd, SourceRs1, SourceImm));

    if let Some(shift) = sign_extension_shift {
        let shift = recipe.imm_const(shift);
        recipe.emit(i(SLLI, SourceRd, SourceRd, shift));
        recipe.emit(i(SRAI, SourceRd, SourceRd, shift));
    }

    recipe.finish()
}
```

The exact Rust surface may differ, but the extraction requirement is that the
output is still inspected as recipe data or shallow operations. A builder that
can perform arbitrary Rust-side emission recreates the current problem.

Complex branches should be visible either as recipe data or as shallow lowerer
control flow that produces inspectable operations. For example, CSR update
behavior needs to preserve the current branch structure:

```rust
pub(crate) const CSRRW_RECIPE: LowerStmt = If {
    cond: CsrEq(0),
    then_body: &Fail(ExpansionErrorKind::UnsupportedCsr),
    else_body: &If {
        cond: CsrSupported,
        then_body: &If {
            cond: RdEqZero,
            then_body: &Emit(i(ADDI, Reserved(CsrTarget), SourceRs1, Const(0))),
            else_body: &If {
                cond: RdEqRs1,
                then_body: &CSRRW_SWAP_THROUGH_TEMP,
                else_body: &Seq(&[
                    Emit(i(ADDI, SourceRd, Reserved(CsrTarget), Const(0))),
                    Emit(i(ADDI, Reserved(CsrTarget), SourceRs1, Const(0))),
                ]),
            },
        },
        else_body: &Fail(ExpansionErrorKind::UnsupportedCsr),
    },
};
```

The `csr == 0` branch is intentionally a failure now. The earlier branch that returned `LiteralRow::DefaultNormalizedInstruction` was a useful warning sign because it bypassed normal source metadata; the current PR resolves that warning by rejecting the decoded row instead of preserving the literal. A grammar interpreter should attach the source CSR value to the `UnsupportedCsr` error so this case produces `UnsupportedCsr(0)` in the public API.

Shared fragments should be grammar fragments with typed arguments:

```rust
pub(crate) const AMO_PRE64: Fragment = Fragment::new(
    FragmentId::AmoPre64,
    &[Arg::Rs1, Arg::VRd, Arg::VDword, Arg::VShift],
    Seq(&[
        Emit(align(VirtualAssertWordAlignment, ArgReg(Rs1), Const(0))),
        Emit(i(ANDI, ArgReg(VShift), ArgReg(Rs1), FormatIImm(Const(-8)))),
        Emit(i(LD, ArgReg(VDword), ArgReg(VShift), Const(0))),
        Emit(i(SLLI, ArgReg(VShift), ArgReg(Rs1), Const(3))),
        Emit(r(SRL, ArgReg(VRd), ArgReg(VDword), ArgReg(VShift))),
    ]),
);
```

This keeps `amo_pre64` reusable without making extraction depend on arbitrary Rust helper control flow.

Recipe checks should run in normal Rust tests where practical:

- all temp uses are inside a live temp scope;
- no temp is released twice;
- every `Emit` row has the operands required by its row shape;
- fragments are acyclic when that can be checked from recipe data;
- every `ReturnLiteral` is named in a baseline-quirks test;
- maximum shallow ops, work-stack depth, and final rows stay within fixed capacities for the curated corpus.

There are two plausible implementation strategies:

1. Ordinary Rust recipe definitions interpreted by `expand::lower`, or shallow
   lowerers that return bounded `ExpansionOp` buffers. This is the best first
   implementation because Hax/Aeneas see a small driver plus data-shaped
   recipes/ops, not proc-macro syntax.
2. A proc-macro DSL for ergonomics, later. This can use `syn` the way Ari
   describes, but the macro must emit the same checked recipe definitions rather
   than opaque hand-coded lowerers. Proc macros parse syntax, not Rust types, so
   any type-dependent behavior must be explicit in the recipe data or delegated
   to small typed interfaces outside the extraction-critical core.

The production Rust path can be an interpreter, generated
interpreter-specialized code, or shallow lowerers that return the same operation
data. The Lean path should consume the same syntactic expansion definitions.
Hax/Aeneas then become one extraction path for the driver and core state
machine, not the only way to recover expansion behavior from arbitrary Rust.

## Core Data Model

Keep the runtime data model smaller than the first draft. `NormalizedInstruction`
is already the canonical row type, and most proposed types were wrappers around
it. The core should distinguish row stages and metadata policy without creating
a parallel instruction representation.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowStage {
    Source(NormalizedInstruction),
    Expanded(NormalizedInstruction),
    Stamped(NormalizedInstruction),
}
```

The exact enum name can change, but the phase distinction should stay explicit:

- `Source` rows are decoded inputs or helper rows that still need expansion
  canonicalization.
- `Expanded` rows are final bytecode rows before sequence metadata has been
  stamped.
- `Stamped` rows are ready to append to the top-level bytecode vector.

The current implementation does not stamp every input row. It stamps only
synthetic sequences built through `InstrAssembler::finalize`; rows that do not
expand are returned unchanged. Preserve that policy explicitly. It can be
represented by a small result enum, but avoid introducing separate `SourceRow`,
`RawRow`, `SequenceRow`, `ExpandedRows`, and `SourcePlan` types unless the
implementation demonstrates that the extra names buy real clarity.

A minimal result enum is enough:

```rust
pub(crate) enum ExpansionResult {
    PassThrough(NormalizedInstruction),
    Literal(NormalizedInstruction),
    Synthetic(ExpansionBuffer<NormalizedInstruction>),
}

pub(crate) enum InitialExpansion {
    Direct(ExpansionResult),
    Work(OpBuffer),
}
```

Family lowerers produce operations, not recursively finalized `Vec`s:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpansionOp {
    Row(NormalizedInstruction),
    Release(u8),
    ResetInlineRegisters,
}
```

`ExpansionOp::Row` carries a row that must go back through the central expansion
driver. It is not final output until the driver decides the kind is legal target
bytecode and the metadata policy has been applied. Historical CSR-zero
default-row behavior is excluded from the stage model because the PR now rejects
that row before materialization.

The driver owns all mutable state:

```rust
pub(crate) struct ExpansionState {
    allocator: ExpansionAllocator,
    work: WorkStack<ExpansionOp>,
    output: ExpansionBuffer<NormalizedInstruction>,
    fuel: u32,
}
```

No production expansion struct should contain a borrowed allocator or a lifetime parameter.

## Driver Design

The driver is the only recursive component. It should be implemented as an iterative depth-first work stack:

```rust
pub(crate) fn expand_one_core(
    source: NormalizedInstruction,
    state: &mut ExpansionState,
) -> Result<ExpansionResult, ExpansionError> {
    state.reset_for_source();

    let initial_ops = match prepare_source(source, state)? {
        InitialExpansion::Direct(rows) => return Ok(rows),
        InitialExpansion::Work(initial_ops) => initial_ops,
    };
    state.start_synthetic_sequence();
    state.push_ops_reversed(initial_ops.as_slice())?;

    while let Some(op) = state.pop_work() {
        state.consume_fuel()?;
        match op {
            ExpansionOp::Row(row) => process_row(source, row, state)?,
            ExpansionOp::Release(register) => state.allocator.release(register)?,
            ExpansionOp::ResetInlineRegisters => emit_inline_resets(source, state)?,
        }
    }

    Ok(ExpansionResult::Synthetic(state.output.clone()))
}
```

`process_row` handles the rules currently embedded in `expand_instruction`:

```rust
fn process_row(
    source: NormalizedInstruction,
    row: NormalizedInstruction,
    state: &mut ExpansionState,
) -> Result<(), ExpansionError> {
    if row.operands.rd == Some(0) && !handles_rd_zero_internally(row.instruction_kind) {
        if has_side_effects(row.instruction_kind) {
            let tmp = state.allocator.allocate_instruction()?;
            let rewritten = row.with_rd(tmp);
            state.push_ops_reversed(&[
                ExpansionOp::Release(tmp),
                ExpansionOp::Row(rewritten),
            ])?;
            return Ok(());
        }
        state.output.push(noop_for_source(source))?;
        return Ok(());
    }

    if is_final_kind(row.instruction_kind) {
        state.output.push(strip_sequence_metadata(row))?;
    } else {
        let ops = lower::lower(row, &mut state.allocator)?;
        state.push_ops_reversed(ops.as_slice())?;
    }
    Ok(())
}
```

`prepare_source` handles source-only pass-through before the synthetic driver starts:

```rust
fn prepare_source(
    source: NormalizedInstruction,
    state: &mut ExpansionState,
) -> Result<InitialExpansion, ExpansionError> {
    if source.operands.rd == Some(0) && !handles_rd_zero_internally(source.instruction_kind) {
        if has_side_effects(source.instruction_kind) {
            let tmp = state.allocator.allocate_instruction()?;
            let rewritten = source.with_rd(tmp);
            if is_final_kind(rewritten.instruction_kind) {
                let row = rewritten;
                state.allocator.release(tmp)?;
                return Ok(InitialExpansion::Direct(ExpansionResult::Literal(row)));
            }
            return Ok(InitialExpansion::Work(ops(&[
                ExpansionOp::Row(rewritten),
                ExpansionOp::Release(tmp),
            ])));
        }
        return Ok(InitialExpansion::Direct(ExpansionResult::Literal(noop_for_source(source))));
    }

    if is_final_kind(source.instruction_kind) {
        return Ok(InitialExpansion::Direct(ExpansionResult::PassThrough(source)));
    }

    Ok(InitialExpansion::Work(ops(&[ExpansionOp::Row(source)])))
}
```

This distinction is necessary for exact parity with the current PR. It also makes any future decision to stamp final source rows a conscious behavior change instead of an accidental consequence of the new architecture.

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
    row: NormalizedInstruction,
    out: &mut OpBuffer,
) -> Result<(), ExpansionError> {
    let rd = operands::rd(row)?;
    let rs1 = operands::rs1(row)?;
    out.row(i(InstructionKind::ADDI, rd, rs1, row.operands.imm))?;
    out.row(i(InstructionKind::VirtualSignExtendWord, rd, rd, 0))?;
    Ok(())
}
```

For functions with temporary registers, releases become explicit operations at
the same sequence point:

```rust
pub(crate) fn lower_mulh(
    row: NormalizedInstruction,
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

`ExpansionOp` and `NormalizedInstruction` are `Copy`, so this can avoid
`MaybeUninit` in the extraction-critical core. Overflow returns
`ExpansionError::ExpansionBufferExceeded { capacity: N }`.

Suggested buffers:

```rust
pub(crate) type WorkStack = FixedVec<ExpansionOp, MAX_WORK_OPS_PER_SOURCE>;
pub(crate) type OpBuffer = FixedVec<ExpansionOp, MAX_SHALLOW_OPS_PER_LOWERING>;
pub(crate) type ExpansionBuffer = FixedVec<NormalizedInstruction, MAX_FINAL_ROWS_PER_SOURCE>;
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
        let rows = core::expand_one_core(*instruction, &mut state)?;
        metadata::append_expanded_rows(rows, &mut expanded)?;
    }
    Ok(expanded)
}
```

Ergonomic `impl IntoIterator` wrappers can remain outside the extraction target.

## Metadata Stamping

Rows inside synthetic sequences should not be partially initialized with
placeholder metadata. The finalizer should construct or update
`NormalizedInstruction` rows from expanded rows:

```rust
pub(crate) fn stamp_row(
    source: NormalizedInstruction,
    mut row: NormalizedInstruction,
    index: usize,
    len: usize,
) -> Result<NormalizedInstruction, ExpansionError> {
    let remaining = len
        .checked_sub(index + 1)
        .ok_or(ExpansionError::MalformedExpansion)?;
    let remaining = u16::try_from(remaining)
        .map_err(|_| ExpansionError::ExpansionTooLong { len })?;

    row.address = source.address;
    row.virtual_sequence_remaining = Some(remaining);
    row.is_first_in_sequence = index == 0;
    row.is_compressed = index + 1 == len && source.is_compressed;
    Ok(row)
}
```

This stamping function is for `ExpansionResult::Synthetic` only.
`ExpansionResult::PassThrough` returns the original source row unchanged,
matching the current behavior for already-final rows. Root-level
`ExpansionResult::Literal` returns an explicitly constructed literal row without
stamping and must be covered by a baseline-quirk or normalization test.

For a side-effect-free `rd = x0` source no-op, current behavior returns an ADDI
no-op with the source address and compressed flag, but with
`virtual_sequence_remaining = None` and `is_first_in_sequence = false`. That
should be represented as `ExpansionResult::Literal(noop_for_source(source))`,
not as a one-row synthetic sequence, unless maintainers intentionally approve
the metadata behavior change.

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
        source: NormalizedInstruction,
    ) -> Result<InlineExpansion, ExpansionError>;
}

pub struct InlineExpansion {
    rows: InlineRowBuffer,
}
```

The adapter can intercept inline source rows and ask tracer's registry for the
normalized inline sequence. For this phase, that sequence may remain finalized
provider output rather than `ExpansionOp` data. If provider-owned inline
allocation needs reset rows, the provider should include those rows before
returning. The provider-free core should not call a trait object while
processing ordinary RV64 rows.

This preserves the dependency boundary from PR #1490: `jolt-program` still does not depend on `tracer`, inventory, CPU state, or advice tapes.

## Termination And Recursion Bound

The current recursive implementation relies on the absence of cycles in helper
expansion. The new driver should make accidental cycles explicit without
requiring a full rank system in the compiler-native expansion phase.

The driver should enforce a fuel or recursion-depth bound:

```rust
const MAX_EXPANSION_OPS_PER_SOURCE: u32 = 4096;
```

Fuel exhaustion should be treated as an internal malformed-expansion error and
should never occur in parity fixtures.

If the recipe surface later becomes declarative enough to build a useful
dependency graph, a rank or acyclicity validator can be added as a follow-up.
It is not required for the compiler-native expansion phase.

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
- allocation count during expansion should drop relative to the post-#1490
  expansion engine.

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
    source: NormalizedInstruction,
    state: &mut ExpansionState,
) -> Result<ExpansionResult, ExpansionError>;

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
        let rows = core::expand_one_core(instruction, &mut state)?;
        metadata::append_expanded_rows(rows, &mut expanded)?;
    }
    Ok(expanded)
}
```

The extracted core should not be generic over `IntoIterator`.

## Cargo Feature Shape

The extraction-critical module graph should avoid serialization and host-only dependencies in its call graph, but the first rewrite should not also change workspace feature defaults. Current `jolt-riscv` serialization derives are unconditional; feature-gating them is a useful follow-up only if Hax/Aeneas still pull those impls after the grammar/core rewrite.

For this phase, the practical rule is:

- `expand::grammar`, `expand::core`, `expand::allocator`, `expand::buffer`, and `expand::metadata` should not call serialization APIs;
- `image = ["dep:object"]` should remain outside the extraction target;
- Hax/Aeneas experiments should start from provider-free `expand` core functions, not public image/execution adapters.

## Next Implementation Pass Scope

Decision: the next implementation sequence should prioritize RV64-only cleanup
and instruction phase names before rewriting the provider-free expander. The
compiler-native rewrite remains the main extraction goal, but doing it after
these two setup phases makes the legality predicates, source/target split, and
review surface clearer.

The next PR may implement all three phases if it stays reviewable, but the
ordered scope is:

1. **Phase 1: fully remove historical RV32 support from `tracer`.**
   `jolt-program` already rejects ELF32/RV32. Finish the cutover by deleting
   live `tracer` RV32 execution/decode/uncompression paths and stale docs.
   Implemented in [#1518](https://github.com/a16z/jolt/pull/1518).
2. **Phase 2: introduce phase-specific instruction identities.** Keep a flat
   committed row identity where proof code needs it, but stop asking one name to
   mean source RISC-V op, expanded Jolt bytecode row, and lookup-backed row.
3. **Phase 3: provider-free compiler-native bytecode expansion.** Convert the
   current `jolt-program::expand` families into syntactic recipes or shallow
   lowerers, a central worklist driver, explicit temp materialization, bounded
   buffers, and explicit metadata policy.
4. **Later pass: inline extension/advice design.** Decide how generic inline
   infrastructure, selected concrete inline recipes, and advice generation
   should relate to the same expansion machinery. This may become an extension
   dialect, but that is not settled in this spec.

Do not combine the inline/advice pass with the three phases above. Inlines have
a different dependency shape from ordinary bytecode expansion: they involve
guest SDK encoding, link-time registration, host advice computation, tracer
CPU/memory access, inline-only virtual-register reset policy, and much larger
sequence capacities. Combining them with the provider-free rewrite would make
review and parity debugging too wide. The bytecode pass should instead build
the target recipe surface and resource materialization rules that a later inline
pass can reuse if that proves to be the right design.

### In Scope For Phase 1: RV32 Removal

Phase 1 should include the following production changes:

- Delete `Xlen::Bit32` as a live execution mode from `tracer`, or replace
  `Xlen` with a single RV64 marker/API where keeping a name is useful for
  documentation.
- Reject ELF32/RV32 at the tracer image/load boundary with a typed or explicit
  error instead of configuring the CPU/MMU for 32-bit execution.
- Remove RV32 branches from tracer decode/uncompression, compressed-instruction
  handling, ALU/div/rem helpers, MMU address normalization, trap-cause width,
  virtual helper instructions, and inline length accounting.
- Remove or rewrite tests that exist solely to validate RV32 behavior. Keep
  tests that validate RV64 word operations such as `ADDW`, `DIVW`, `REMUW`,
  word AMOs, and compressed RV64 encodings.
- Update stale comments and docs that describe RV32, ELF32, or dual-width
  expansion as supported behavior.
- Keep `jolt-program::image`'s existing ELF32/RV32 rejection and add tracer-side
  coverage so both program construction paths agree.

Phase 1 should include these tests and checks:

- focused tracer tests for ELF32/RV32 rejection;
- existing RV64 compressed-instruction tests;
- existing RV64 word-op, AMO/LR/SC, and inline-sequence tests;
- `cargo fmt -q`;
- `cargo clippy --all --features host -q --all-targets -- -D warnings`;
- `cargo nextest run -p tracer --cargo-quiet`;
- `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host`.

### In Scope For Phase 2: Instruction Phase Split

Phase 2 should introduce names that reflect the actual pipeline:

- `RiscvInstructionKind`: source-level RV64 program instruction identity. This
  includes source instructions that may expand away before proving, such as
  `DIV`, `REM`, atomics, CSR ops, advice-load source ops, and Jolt inline source
  opcodes.
- `JoltInstructionKind`: expanded executable Jolt bytecode row identity. These
  are the rows legal after bytecode expansion and before preprocessing/proving.
  Some rows are ordinary RISC-V-looking rows; some are Jolt virtual helper rows.
- `LookupInstructionKind`: the subset of expanded Jolt bytecode rows routed
  through instruction lookup tables.

The exact Rust spelling can change during implementation, but the split should
make these boundaries explicit. A Rust enum shaped as
`InstructionKind::{Riscv(...), Jolt(...), Lookup(...)}` is probably not the
right committed representation because bytecode rows, serialization, and lookup
indexing need stable flat discriminants. Prefer phase-specific public types plus
conversion/metadata APIs:

```rust
impl RiscvInstructionKind {
    pub const fn expands_to_jolt(self) -> bool;
}

impl JoltInstructionKind {
    pub fn lookup_kind(self) -> Option<LookupInstructionKind>;
}
```

Phase 2 should decide what happens to the existing names:

- either rename the current flat `InstructionKind` to `JoltInstructionKind` and
  introduce `RiscvInstructionKind` for decoded source rows;
- or keep `InstructionKind` only as an internal/stable flat row discriminant and
  expose phase-specific newtypes/enums around it.

The PR should not leave `JoltInstructions<T>` and `InstructionKind` with
overlapping unclear meanings. Either `JoltInstructions<T>` becomes a typed view
over the chosen Jolt bytecode kind, or it is renamed/narrowed so it no longer
sounds like "all instructions provable by Jolt" when it actually contains a
mixed subset.

Implementation direction for this PR: use the first option. The flat serialized
row discriminant is `JoltInstructionKind`; RV64 decode tables produce
`RiscvInstructionKind` and lower that into normalized `JoltInstructionKind`
rows; the typed metadata view formerly called `JoltInstructions` is now
`LookupInstruction`, with `LookupInstructionKind` available through
`JoltInstructionKind::lookup_kind()`.

Phase 2 should include these tests and checks:

- compile-time or unit tests showing source-only rows such as `DIV`/`REM` and
  atomics are distinguished from final expanded rows where appropriate;
- tests for `JoltInstructionKind::lookup_kind()` covering lookup-backed rows,
  memory/system-ish rows that are legal bytecode but not lookup-backed, and
  virtual helper rows;
- dependency checks showing `jolt-riscv` remains tracer-free;
- existing `jolt-program`, `jolt-lookup-tables`, `tracer`, and `jolt-core`
  instruction tests.

### In Scope For Phase 3: Compiler-Native Provider-Free Expansion

Phase 3 should include the following production changes:

- Use compiler-native vocabulary in docs and code where appropriate.
  Extraction remains a key acceptance signal, but the implementation shape is a
  compiler lowering pipeline.
- Add a small `expand::grammar` or equivalent recipe module with inspectable
  syntactic expansion data:
  - use `NormalizedInstruction` as the row representation, with a small
    `RowStage`/`ExpansionResult` policy for source, expanded, and stamped rows;
  - include only the operand-reference, condition, and statement forms that real
    expansion families need;
  - keep `WithTemp`/nested temp scopes or an equivalent explicit lifetime
    construct for recipes that need more than one temporary register;
  - keep `RegisterPool::{Instruction, Inline}` only if it is useful for the
    provider-free core and the inline adapter boundary.
- Add `expand::core` with concrete, non-generic entry points:
  - `expand_one_core(source: NormalizedInstruction, state: &mut ExpansionState)`;
  - a depth-first work stack of `ExpansionOp`;
  - source and target legality predicates;
  - a fuel or recursion-depth bound that returns an explicit error on runaway
    expansion.
- Replace the production recursive `InstrAssembler<'a>` with shallow family
  lowerers that only return recipe data or bounded `ExpansionOp` buffers.
- Materialize temps through `ExpansionAllocator` at explicit lifetime
  boundaries, preserving current first-fit virtual-register numbering and
  release order. Nested `WithTemp` scopes are the default representation for
  multi-temp sequences, but the implementation may use an equivalent explicit
  operation form if it is simpler.
- Replace heap-backed per-source sequence construction in the core with bounded
  buffers and explicit overflow errors.
- Preserve current metadata behavior exactly, including pass-through rows,
  synthetic sequence stamping, `is_compressed` placement, and any remaining
  literal rows. Also preserve the PR's documented baseline fixes, including
  rejection of `CSRRW`/`CSRRS` with CSR address `0`.
- Keep public expansion APIs available to current call sites, but route them
  through the new core. Ergonomic iterator wrappers may stay outside the
  extraction-critical module graph.
- Preserve inline support as an adapter outside provider-free core:
  `InstructionKind::Inline` remains illegal for `expand_one_core`, while
  `expand_instruction_with_provider` can still delegate to an
  `InlineExpansionProvider` that returns finalized `NormalizedInstruction` rows.
- Preserve current advice-channel behavior:
  provider-free expansion lowers `AdviceLB/LH/LW/LD` into `VirtualAdviceLoad`
  sequences, does not assign `VirtualAdvice` payloads, and does not move
  trusted/untrusted advice commitments out of preprocessing/proof code.

Phase 3 should also include the following tests and checks:

- A test-only parity harness that compares the new provider-free expander
  against the current PR output while porting families. The final production
  code should have one expander; the old assembler may remain only as a
  temporary test reference during the rewrite. Once tracer delegates to
  `jolt-program::expand`, tracer bridge tests are circular and should not be
  treated as expansion oracles.
- Fixture coverage for every provider-free source-only instruction kind with
  representative operand aliases, `rd = x0`, compressed metadata, CSR edge
  cases, branch/control-flow immediates, load/store offsets, AMO/LR/SC cases,
  advice-load cases, and division/remainder variants.
- Capacity tests that assert observed final row count, shallow op count, and
  work-stack depth stay below the chosen constants.
- Dependency checks showing no `tracer` dependency from `jolt-program` or
  `jolt-riscv`.
- Hax/Aeneas reruns on metadata stamping, allocator transitions, ADDIW shallow
  lowering, and provider-free `expand_one_core`.
- Standard repo verification for code changes:
  - `cargo fmt -q`;
  - `cargo clippy --all --features host -q --all-targets -- -D warnings`;
  - `cargo clippy --all --features host,zk -q --all-targets -- -D warnings`;
  - focused `cargo nextest run -p jolt-program --cargo-quiet`;
  - `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host`;
  - `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk`.

### Concrete Phase 3 Milestones

The compiler-native expansion phase should be reviewable as one focused
production rewrite of `jolt-program::expand`, with the following milestones:

1. **Freeze the behavioral baseline.**
   - Add golden/parity coverage for the current PR behavior before deleting the
     recursive assembler path.
   - Treat the old assembler as a temporary test oracle only while the rewrite is
     in flight. It should not survive as a compatibility layer or second
     production expander.
   - Capture row-for-row parity for metadata, virtual register allocation,
     helper-row recursion order, `rd = x0`, compressed source rows, remaining
     quirks, and documented fixes such as CSR address `0` rejection.

2. **Introduce the compiler data model.**
   - Keep `NormalizedInstruction` as the row type and add only the minimal
     stage/result wrappers needed to distinguish source rows, expanded helper
     rows, stamped rows, pass-through rows, and deliberate literals.
   - Add syntactic recipe data or shallow operation buffers with explicit temp
     ids, register-pool ids where useful, and recipe fragments where they
     reduce duplication.
   - Keep the model first-order: no borrowed assembler state, no recursive
     callback into public expansion APIs, no trait-object dispatch in the
     provider-free core.
   - Add validators/checks for source legality, target legality, fuel/depth,
     temp liveness where represented as recipe data, and bounded row capacity.

3. **Build the new core driver behind the existing public API.**
   - Implement `ExpansionState`, the bitset allocator, bounded row buffers, the
     explicit work stack, and the single metadata materialization pass.
   - Route public provider-free expansion through the new core as soon as the
     first family is ported, while keeping test-only parity checks available for
     families not yet ported.
   - Keep iterator conveniences and inline-provider adapters outside the
     extraction-critical modules.

4. **Port one small family end to end.**
   - Start with ADDIW/ADDW/SUBW or a similarly small arithmetic slice.
   - Validate byte-for-byte output parity against the old path.
   - Run Hax/Aeneas on the new allocator transitions, metadata stamping,
     shallow lowering, and `expand_one_core` slice before porting larger
     families.

5. **Port all provider-free expansion families.**
   - Move arithmetic, shifts, memory, division, control-flow, fragment helpers,
     AMO/LR/SC, CSR, and literal/pass-through behavior onto recipes or shallow
     lowerers that return inspectable operations.
   - Delete the recursive production assembler once the last provider-free family
     has parity coverage.
   - Preserve the inline boundary as an adapter returning finalized rows, with
     `InstructionKind::Inline` still illegal in provider-free `expand_one_core`.

6. **Close with extraction and repo verification.**
   - Re-run the narrow Hax/Aeneas experiments and record which modules now
     extract, which still fail, and whether failures are due to expansion-core
     structure or extractor/tool limitations.
   - Run the repo checks listed above.
   - Land the pass only if production behavior is single-path, parity-tested,
     dependency-light, and no slower on the expansion hot path.

The expected review scope for Phase 3 is therefore: "replace the provider-free
expansion engine with a compiler-native lowering core." It is not: "also
redesign all inline packages."

### Out Of Scope For The Three-Phase PR

The RV32 removal, instruction phase split, and compiler-native provider-free
expansion work should not:

- introduce a Melior/MLIR dependency;
- create `jolt-inline-ir` or a new inline registry crate;
- port SHA2, Keccak, Blake, BigInt, Secp, Grumpkin, or P-256 inline recipes;
- replace `inventory` registration;
- redesign guest inline SDK assembly macros;
- model inline advice as a grammar;
- assign `VirtualAdvice` payloads in provider-free expansion;
- change runtime advice-tape behavior for `VirtualAdviceLoad` or
  `VirtualAdviceLen`;
- move trusted/untrusted advice memory commitments into `jolt-program::expand`;
- change RV64 tracer execution semantics;
- change bytecode/RAM preprocessing semantics;
- feature-gate `serde` or `ark-serialize` unless extraction still pulls those
  impls after the compiler-native rewrite.

### Phase 3 File-Level Shape

The intended end state inside `crates/jolt-program/src/expand` is:

```text
allocator.rs       deterministic bitset allocator and reset tracking
buffer.rs          bounded buffers used by core lowering
core.rs            worklist driver, legality, staged row policy
grammar.rs         small syntactic recipe data and validators/checks
metadata.rs        sequence metadata materialization
operands.rs        operand decoding helpers
lower/
  arithmetic.rs    shallow recipe/lowering definitions
  control_flow.rs
  division.rs
  memory.rs
  shifts.rs
  fragments.rs     shared helper recipes such as amo_pre64/amo_post64
inline.rs          adapter boundary only, outside provider-free core
mod.rs             public API glue
```

The exact file names can shift during implementation, but ownership should not:
`core + grammar/recipe + allocator + buffer + metadata + lower` form the
compiler-native target; `inline + public ergonomic wrappers` stay outside the
extraction target.

## Follow-Up Inline Extension Pass

This section is separate from the execution-semantics track above. Inlines add
both expansion complexity and advice/execution questions, so their design should
not be used as the first semantics modeling slice.

The inline pass should start after Phase 3 has stabilized the target bytecode
recipe surface and materialization rules. Its goal is not "make current Rust
inline builders extract." The exact advice model is still unresolved. A future
inline spec should decide whether inlines become extension dialects, remain
adapter-provided finalized rows, or split bytecode expansion from advice
generation more explicitly.

One possible crate split is:

```text
jolt-inline-ir
  Generic InlineId, InlineOpDef, InlineRecipe, placeholder AdviceModel, validators.

jolt-program
  Consumes inline definitions and lowers source rows to final bytecode.
  Still has no tracer dependency.

jolt-inlines-{sha2,keccak256,bigint,...}
  Concrete extension dialects: guest SDK encoding plus typed inline defs.

jolt-inline-registry
  Optional feature-selected registry crate collecting chosen inline packages.

tracer
  Runtime adapter: executes final bytecode and evaluates advice plans against Cpu.
```

The first inline implementation slice should port one representative inline,
not all of them. A good first candidate is an inline with large but regular
memory/register structure, such as `bigint.mul256`, because it stresses static
loops, symbolic inline temp arrays, memory loads/stores, and reset rows without
also requiring the full hash/advice surface. A later slice can port an
advice-bearing modular arithmetic inline, then a hash compression inline with
static round schedules.

A future inline grammar may need:

- `InlineOpDef { id, name, operands, effects, recipe, advice }`;
- `InlineOperandSpec` that makes `rs3` a memory output pointer, not a normal
  architectural `rd`;
- static loops and recipe fragments with alpha-renamed temps;
- symbolic inline temp arrays over `RegisterPool::Inline`;
- explicit memory effect metadata;
- explicit `VirtualAdvice(slot)` rows tied to whatever advice model the follow-up spec chooses;
- reset policy that checks all inline temps are dead before materializing
  `ADDI temp, x0, 0` rows.

In MLIR terms, inlines should become extension ops such as
`jolt.inline.sha2.compress` or `jolt.inline.bigint.mul256`, not opaque Rust
callbacks. The compiler should legalize those ops into `expand` ops and then
into final `jolt.bytecode` rows using the same target legality, metadata, and
resource materialization machinery as provider-free expansion.

## Migration Plan

1. Remove historical RV32 support from `tracer`, including live `Xlen::Bit32`
   paths, RV32 decode/uncompression behavior, RV32-only tests, and stale docs.
2. Add tracer-side RV32/ELF32 rejection coverage and keep `jolt-program::image`
   RV32/ELF32 rejection coverage green.
3. Introduce phase-specific instruction identities:
   `RiscvInstructionKind`, `JoltInstructionKind`, and `LookupInstructionKind`
   or equivalent names with the same semantics.
4. Decide the final relationship between `InstructionKind` and
   `JoltInstructions<T>` so the code no longer has two overlapping names for
   different slices of the instruction universe.
5. Update decode, expansion, preprocessing, lookup-table routing, tracer
   adapters, and proof call sites to use the phase-specific types or explicit
   conversion APIs.
6. Freeze the provider-free expansion behavioral baseline after the RV32 and
   instruction-kind work is complete.
7. Add the small syntactic expansion recipe surface, compiler-phase vocabulary,
   and baseline-quirk fixtures.
8. Add new `core`, `buffer`, and bitset `allocator` internals under
   `jolt-program::expand`.
9. Represent temp lifetimes explicitly and materialize them through the
   allocator at `WithTemp` or equivalent boundaries.
10. Port one small family, such as ADDIW/ADDW/SUBW, to recipe-backed shallow
    lowering and prove parity against the current output.
11. Port arithmetic, shifts, memory, division, and control-flow families.
12. Replace `InstrAssembler<'a>` in production expansion code.
13. Preserve tracer inline adapter support as finalized rows outside
    provider-free core.
14. Delete the old recursive assembler once all parity tests pass.
15. Run Hax/Aeneas again on:
   - metadata stamping,
   - allocator transitions,
   - ADDIW shallow lowering,
   - provider-free `expand_one_core`.
16. Record the separate semantics follow-up: a hand-modeled Lean transition
    relation for a small provider-free slice, plus an expansion-correctness
    statement comparing source-row execution with target-sequence execution.
17. Run formatting, clippy, host tests, ZK tests, and dependency checks.

Do not leave both expanders in production. A temporary test-only reference path is acceptable during the rewrite, but the final branch should have one canonical production expander.

## Acceptance Criteria

- [ ] `tracer` no longer has a live RV32 execution mode, `Xlen::Bit32`, or
      RV32 decode/uncompression path.
- [ ] ELF32/RV32 inputs are rejected explicitly by both `jolt-program::image`
      and the remaining tracer image/load boundary.
- [ ] RV32-only tests and stale docs/comments are removed or rewritten, while
      RV64 word-op and RV64 compressed-instruction coverage remains.
- [x] Source-level instruction identity, expanded Jolt bytecode row identity,
      and lookup-backed instruction identity have distinct names/types or
      explicit conversion APIs.
- [x] The `InstructionKind` / `JoltInstructions<T>` relationship is resolved so
      the code no longer has two overlapping concepts with unclear phase
      ownership.
- [x] Lookup-table routing uses `LookupInstructionKind` or an equivalent
      explicit subset API such as `JoltInstructionKind::lookup_kind()`.
- [ ] `jolt-program::expand` no longer has a production `InstrAssembler<'a>` that stores a borrowed allocator.
- [ ] Expansion recipes are represented as syntactic lowering data or shallow operations; the grammar does not model instruction execution semantics.
- [ ] The spec keeps execution semantics on a separate track, starting with a hand-modeled Lean abstract machine slice and expansion-correctness theorem rather than extraction from `tracer` or MLIR-as-semantics.
- [ ] The spec names source, intermediate, and target phases with MLIR-ready legality predicates, even while Rust remains the implementation substrate.
- [ ] `NormalizedInstruction` remains the core row type, with only minimal stage/result wrappers for source, expanded, stamped, pass-through, and literal policies.
- [ ] Recipe temps have explicit lifetimes and are materialized into concrete virtual registers by a deterministic allocation/resource-materialization step.
- [ ] Recipe checks reject invalid operand shapes, unchecked literal rows, bounded-capacity overflow, and temp-lifetime mistakes where temp scopes are represented as data.
- [ ] Family lowerers are shallow and do not call `expand_instruction`.
- [ ] Recursive expansion happens in one central depth-first driver.
- [ ] Temporary-register release and inline reset are explicit expansion operations.
- [ ] Allocator state is represented by bitsets, not a heap-backed reset list.
- [ ] Per-source expansion uses bounded buffers, with explicit overflow errors.
- [ ] Synthetic sequence metadata is stamped during final row construction, not by mutating already-built rows.
- [ ] Source pass-through rows and deliberate literal rows preserve the current metadata policy exactly, with tests for the weird cases.
- [ ] Provider-free core expansion has a concrete, non-generic entry point over `NormalizedInstruction` and `ExpansionState`.
- [ ] Inline provider support is an adapter outside the provider-free core.
- [ ] Provider-free expansion preserves advice-load lowering exactly:
      `AdviceLB/LH/LW/LD` emit `VirtualAdviceLoad` plus sign extension where
      needed.
- [ ] `VirtualAdvice` payload assignment remains outside provider-free
      expansion; `NormalizedInstruction` does not grow an advice payload field.
- [ ] Trusted/untrusted advice memory and polynomial commitments remain
      preprocessing/proof responsibilities, not expansion responsibilities.
- [ ] Expansion output matches PR #1490 baseline fixtures exactly.
- [ ] Hax and Aeneas can extract metadata stamping and at least one shallow family lowerer without pulling in execution/preprocess/serialization modules.
- [ ] Dependency checks still show no `tracer` dependency from `jolt-program` or `jolt-riscv`.

## Resolved And Sharpened Questions

### Fixed Buffer Capacities

Initial answer: use conservative fixed capacities for provider-free expansion:

```rust
const MAX_FINAL_ROWS_PER_SOURCE: usize = 64;
const MAX_SHALLOW_OPS_PER_LOWERING: usize = 64;
const MAX_WORK_OPS_PER_SOURCE: usize = 128;
```

A throwaway measurement against the current `expand_instruction` implementation on representative provider-free rows found the largest final expansion was `SCW` at 37 rows. The next largest cases were word AMO min/max at 24 rows, `DIV`/`REM` at 24 rows, `DIVW`/`REMW` at 21 rows, word AMO arithmetic at 19 rows, and `AMOSWAPW` at 18 rows.

These constants are intentionally not tight. They leave room for the worklist
driver to carry explicit `Release` operations and for small future edits without
immediately changing stack layout. The implementation PR should still add a
fixture test that expands every provider-free source-only kind with
representative operands and asserts:

- final row count is below `MAX_FINAL_ROWS_PER_SOURCE`;
- maximum shallow recipe output is below `MAX_SHALLOW_OPS_PER_LOWERING`;
- maximum observed work-stack depth is below `MAX_WORK_OPS_PER_SOURCE`.

Inline expansion should not use these provider-free constants until inline recipes are also moved into the grammar. Registered inlines are outside this phase's extraction target and can continue using a heap-backed adapter or a separately measured inline capacity.

### Grammar Surface

Decision: start with ordinary Rust data or shallow lowerers plus a small
`RecipeBuilder` only where it pays for itself; do not start with a proc macro.

Plain data is the extraction-friendly source of truth, but the first
implementation should not overbuild the grammar. A builder is acceptable only
if `finish()` returns inspectable syntactic expansion data or bounded
`ExpansionOp` buffers. The builder must not be a new imperative assembler whose
methods hide arbitrary recursive emission logic.

Start with the smallest surface that covers the ported families. `Seq`,
`Emit`, `If`, `WithTemp`/explicit lifetime markers, `Fragment`, `Release`,
`ReturnLiteral`, and `Fail` are likely useful concepts. `RegRef`, `ImmRef`, and
`Cond` should be pared down to the forms actually needed by provider-free
expansion. Do not preserve the larger `RegExpr`/`ImmExpr`/`CondExpr` sketch just
because it appears in this spec.

A proc-macro DSL can be a later ergonomics layer, but only if it emits the same checked recipe definitions. That keeps Ari's article's lesson without inheriting the main proc-macro drawback: `syn` sees syntax, not Rust types. Type-dependent behavior should stay in the recipe data or in small typed adapters outside the extraction-critical core.

### MLIR Boundary

Decision: keep the immediate rewrite Rust-first, but make the Rust data model
MLIR-ready without implying an execution-semantics IR.

The implementation PR should not add a Melior/MLIR dependency to
`jolt-program::expand`. That would make an already large rewrite harder to
review and would move the extraction experiment into compiler-infrastructure
integration before the expansion behavior is clean. The right near-term target
is a small Rust expansion IR with explicit source/intermediate/target legality
checks, syntactic rewrite recipes or shallow lowerers, explicit temp lifetimes,
resource materialization, and focused validators/checks.

Those concepts should be named and shaped as if they may later become MLIR
dialects or ops. In particular, avoid recipe APIs that depend on Rust closures,
trait-object callbacks, hidden mutation, or call-stack effects. A future MLIR
version should be able to represent the same phases as `riscv.norm`,
`expand`, and `jolt.bytecode` dialects without changing the expansion contract
or the parity fixtures.

For execution semantics, MLIR should be treated as a carrier for op structure,
legality interfaces, lowering passes, and documentation, not as the first proof
semantics source. The first proof semantics source should be a Lean transition
relation over an abstract machine state. If a later semantic DSL is introduced,
it should be able to emit Lean definitions, a Rust reference interpreter, and
MLIR op metadata from the same effect descriptions.

### Execution Semantics Follow-Up

Decision: make instruction semantics a separate follow-up spec rather than
folding it into bytecode expansion.

The first follow-up should hand-model a provider-free slice in Lean:

- define an abstract `MachineState` with architectural registers, memory,
  program counter, trap state, and any Jolt target-state components needed by
  virtual helper rows;
- define `step` for a small source/target instruction slice;
- define `exec_seq` for target bytecode rows;
- define the projection from Jolt target state back to architectural state;
- prove or test one expansion-correctness theorem of the form "executing the
  expanded sequence refines executing the source row";
- differentially test the Lean/reference semantics against `tracer` for the
  same slice.

Only after that slice should the project decide whether a semantic DSL is worth
the cost. The DSL option is attractive if it can generate Lean semantics and a
Rust reference interpreter, but it should be driven by the hand-modeled slice
rather than designed in the abstract.

### Expansion Rank

Decision: do not require `ExpansionRank` in the compiler-native expansion
phase.

The central driver should enforce a fuel or recursion-depth bound and return an
explicit error on runaway expansion. That is sufficient for the next rewrite and
is much simpler to review.

If the recipe surface becomes declarative enough later, a dependency-graph or
rank validator can be added as a follow-up. The important Phase 3 invariant is
that recursive expansion goes through one bounded driver instead of being hidden
inside hand-written Rust call structure.

### Inline Provider Output

Decision for this phase: the provider-free core should reject `InstructionKind::Inline`; the tracer adapter should return finalized `NormalizedInstruction` rows, not core `ExpansionOp` values.

That matches the current shape: `TracerInlineExpansionProvider` builds inline sequences with tracer's inline assembler and its own `VirtualRegisterAllocator`, then returns normalized rows. The `ExpansionAllocator` passed into `expand_inline` is currently unused by the tracer provider. The provider also handles inline register reset rows before returning.

Trying to force current registered inlines into `ExpansionOp` now would broaden the PR into a second DSL migration. That should be a later spec if maintainers want inline recipes to share the same grammar. For this phase, inline support should be an adapter outside the extraction target:

```rust
pub enum InlineExpansion {
    FinalizedRows(Vec<NormalizedInstruction>),
}
```

The adapter must preserve the existing `rd = x0` remapping behavior before dispatching to the provider.

### Advice Channels

Decision: preserve today's three advice channels but name them separately.

Provider-free expansion owns only the `AdviceLB/LH/LW/LD` syntactic lowering to
`VirtualAdviceLoad` plus sign extension. `VirtualAdviceLoad` reads from the
runtime advice tape during execution; expansion records only the byte length.

`VirtualAdvice` is different: it is a target row whose concrete tracer
instruction carries a trace-time `advice: u64` payload. That payload is patched
by tracer/inline/LR-SC logic today and should remain outside provider-free
expansion. Do not add an advice payload to `NormalizedInstruction` in this
rewrite.

Trusted and untrusted advice bytes are a third channel: they are Jolt device
memory regions committed as advice polynomials by the prover. They should stay
in preprocessing/proof code and should not be folded into bytecode expansion.

The later semantics/advice follow-up can introduce an `AdvicePlan` if needed.
For now, model advice abstractly as an oracle/tape in the execution semantics
track and keep expansion responsible only for emitting the rows that consume
advice.

### Serialization Derives

Decision for the rewrite PR: do not feature-gate `serde` or `ark-serialize` as part of the first compiler-native rewrite. Keep the extraction start set rooted in `expand::grammar`, `expand::core`, `expand::allocator`, `expand::buffer`, and `expand::metadata`, and retest Hax/Aeneas after the call graph no longer goes through `InstrAssembler<'a>`.

Current `jolt-riscv` has unconditional `serde` and `ark_serialize` derives on `InstructionKind`, `NormalizedInstruction`, and `NormalizedOperands`; `jolt-program` also depends on both unconditionally. Feature-gating those derives may still be useful, but it is a workspace-facing dependency cleanup rather than the core bytecode-expansion redesign. If Hax/Aeneas still pull serialization impls after the grammar rewrite, make this the next narrow change:

```toml
[features]
default = ["std", "serde", "ark-serialize"]
serde = ["dep:serde"]
ark-serialize = ["dep:ark-serialize"]
```

and gate the derives/impls in `jolt-riscv`.

### Release Timing

Decision: ordered `ExpansionOp::Release(register)` is expressive enough for current provider-free expansion.

The current code has two release patterns:

- end-of-sequence releases, such as `MULH`, `MULHSU`, loads, stores, AMOs, shifts, and division;
- mid-sequence releases, such as CSR swap temps, `ECALL` temps, `EBREAK`/`MRET` jump-discard registers, and the staged temps in `SCD`/`SCW`.

Both are modeled by placing `Release` exactly after the last emitted row that may recursively use the temp. The central driver must process a row's full recursive expansion before moving to the following operation; with that rule, an ordered release marker preserves the current allocator reuse behavior. The parity suite should still include allocation-number assertions for `CSRRW rd == rs1`, `CSRRS rd == rs1`, `ECALL`, `SCD`, and `SCW`, because those are the cases most likely to reveal a release-order bug.

### `csr == 0`

Decision for this PR: reject `CSRRW`/`CSRRS` with CSR address `0` as `ExpansionError::UnsupportedCsr(0)`.

Before the final #1490 fixes, `expand_csrrw` and `expand_csrrs` returned
`NormalizedInstruction::default()` when `csr == 0`, bypassing the normal
assembler finalizer at the source level. The bug hunt found that this is not a
harmless no-op once expansion is owned by `jolt-program`: the default row's
address is `0`, and `BytecodePCMapper` skips address-zero rows, so a decoded
CSR-zero source row can fail to receive an entry-bytecode index.

The compiler-native rewrite should therefore make this a first-class source validation rule, not a legacy literal. The recipe may express it as an explicit `If { cond: CsrEq(0), then_body: Fail(UnsupportedCsr), ... }`, or the validator may reject it before recipe interpretation. Either way, tests should assert the public error is `UnsupportedCsr(0)` and should not include CSR-zero in parity fixtures that preserve historical output.
