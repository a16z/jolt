# Spec: `jolt-program` Crate

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao                      |
| Created     | 2026-05-01                     |
| Status      | in review                      |
| PR          | [#1490](https://github.com/a16z/jolt/pull/1490) |

## Summary

Jolt's bytecode expansion logic is still owned by the `tracer` crate even though `jolt-trace` now exposes host-facing program decoding, tracing, and bytecode preprocessing APIs. This spec proposes strengthening the existing `jolt-riscv` crate as the shared instruction-data crate, plus one new `jolt-program` crate that owns program-image decoding, bytecode expansion, and materialized Jolt program preprocessing behind separate internal modules.

The target is a modular pipeline from ELF bytes to expanded bytecode and program preprocessing, while keeping CPU execution, memory-device emulation, advice I/O, trace production, PCS setup, and commitment derivation out of the verifier-facing program dependency graph.

## Intent

### Goal

Introduce a `jolt-program` crate with three internal boundaries:

- `jolt_program::image`: deterministic program-image decoding from ELF bytes.
- `jolt_program::expand`: expansion from decoded RISC-V instructions into final Jolt bytecode rows.
- `jolt_program::preprocess`: materialized bytecode, RAM, and program preprocessing artifacts consumed by prover and verifier setup.

The new crate should own the path:

```text
ELF bytes
  -> DecodedProgramImage
  -> expanded bytecode
  -> JoltProgramPreprocessing
```

`tracer`, `jolt-trace`, and `jolt-core` should become consumers of this program pipeline instead of owners of bytecode expansion or materialized program preprocessing semantics.

### Invariants

- Expansion of a decoded RV64 instruction must produce exactly the same bytecode sequence as the current RV64 `Instruction::inline_sequence(&VirtualRegisterAllocator, Xlen::Bit64)` behavior.
- Recursive expansion must be preserved: helper-emitted instructions must themselves be expanded until the output contains the final bytecode rows consumed by tracing and bytecode preprocessing.
- `rd = x0` behavior must be preserved:
  - side-effect-free instructions become a no-op instruction,
  - side-effecting instructions write to a temporary virtual register,
  - CSR/trap-related instructions that handle `rd = x0` internally keep their current behavior.
- Sequence metadata must be preserved for every expanded row: `is_first_in_sequence`, `virtual_sequence_remaining`, source address, and compressed-instruction metadata.
- Virtual register allocation must preserve the existing layout and lifetime rules for reserved registers, per-instruction temporary registers, and inline-only registers that are reset during inline finalization.
- Bytecode preprocessing must continue to map `(address, virtual_sequence_remaining)` to the same dense bytecode table indices as before.
- Prover and verifier behavior must not change: any committed bytecode table, trace row, PC lookup, or instruction lookup derived from a program must remain identical modulo intentionally documented serialization changes.
- Program preprocessing built from an ELF must be deterministic and must match preprocessing derived through the existing prover path.

Add concrete `jolt-eval` invariants for bytecode expansion fixture consistency and program preprocessing determinism. These invariants should live under `jolt-eval/src/invariant/` and be wired into `JoltInvariants::all()`.

### Non-Goals

- This spec does not propose changing Jolt's instruction semantics.
- This spec does not propose changing guest compilation, memory initialization semantics, or trace execution beyond redirecting program call sites to the new program crate.
- This spec does not require completing Lean, Hax, or Aeneas extraction in the same implementation PR.
- This spec does not require redesigning bytecode commitments, lookup tables, or prover constraints.
- This spec does not require moving PCS setup, commitment derivation, bytecode/program-image opening hints, BlindFold setup, `JoltProverPreprocessing`, or `JoltVerifierPreprocessing` out of `jolt-core`.
- This spec does not require integrating bytecode-commitment PR [#1344](https://github.com/a16z/jolt/pull/1344). That work is a future integration constraint, not part of this implementation scope.
- This spec does not require supporting RV32 in `jolt-program`; RV32/ELF32 should be rejected by the new program pipeline.
- This spec does not require deleting historical RV32 branches from `tracer`; that cleanup can happen separately.
- This spec does not require exposing a stable public API for third-party consumers outside the Jolt workspace.
- This spec does not require moving CPU execution, lazy trace iteration, advice tapes, or memory-device emulation into verifier-facing crates.

## Evaluation

### Acceptance Criteria

- [ ] The existing `crates/jolt-riscv` crate owns shared instruction data types currently required by decoding, expansion, tracing, bytecode preprocessing, and verifier checks.
- [ ] `crates/jolt-riscv` no longer depends on `tracer`; `tracer` depends on `jolt-riscv` for static instruction data instead.
- [ ] `crates/jolt-program` exists and is a workspace member.
- [ ] `jolt-program::image` exposes deterministic ELF decoding into a `DecodedProgramImage` without depending on `tracer`.
- [ ] `jolt-program::expand` provides a pure RV64 expansion API over decoded `ProgramInstruction` values.
- [ ] `jolt-program::image` rejects ELF32/RV32 inputs with an explicit unsupported-architecture error.
- [ ] `jolt-program::preprocess` owns materialized bytecode/RAM/program preprocessing artifacts consumed by both prover and verifier setup.
- [ ] `jolt-riscv` implements `JoltInstruction` for `ProgramInstruction` and does not retain a blanket `impl<T: tracer::instruction::RISCVInstruction> JoltInstruction for T`.
- [ ] Any `JoltInstruction` impls for tracer's concrete instruction structs live in `tracer` as adapter impls, generated from the shared instruction-kind list where practical.
- [ ] `jolt-program` does not depend on PCS implementations, Dory setup, commitment derivation, bytecode/program-image opening hints, BlindFold setup, or prover-only witness generation.
- [ ] The expansion-critical `jolt-program::expand` module does not depend on CPU execution, lazy tracing, memory-device emulation, advice I/O, prover-only witness generation, transcripts, or ELF/object parsing.
- [ ] Any ELF/object parsing dependency is feature-gated and isolated to `jolt-program::image` so `jolt-program::expand` and `jolt-program::preprocess` remain usable without an object-file parser dependency.
- [ ] `tracer` no longer owns the canonical recursive bytecode expansion algorithm; it calls `jolt-program::expand` for expansion during trace execution and trace length accounting.
- [ ] `jolt-trace::decode` no longer performs expansion by directly calling `Instruction::inline_sequence` from `tracer`; it calls `jolt-program::image` and `jolt-program::expand`.
- [ ] `InstrAssembler`, `VirtualRegisterAllocator`, or their minimal expansion-facing equivalents live behind the `jolt-program::expand` module boundary.
- [ ] `InstrAssembler` borrows `&mut ExpansionAllocator` during emission and does not own, clone, or hide allocator state behind shared ownership.
- [ ] Expansion APIs return explicit errors for allocation or malformed-expansion failures instead of introducing new panics in the core expansion path.
- [ ] A program-preprocessing path is implemented as `ELF bytes -> DecodedProgramImage -> expanded bytecode -> JoltProgramPreprocessing`, and prover/verifier setup wraps that program preprocessing without re-decoding or re-expanding the program.
- [ ] The verifier-facing path does not depend on CPU execution, lazy tracing, memory-device emulation, advice I/O, or prover-only witness generation.
- [ ] `cargo tree -p tracer` shows `tracer` depending on `jolt-riscv` and `jolt-program`, but neither lower-level crate depending back on `tracer`.
- [ ] `cargo tree -p jolt-core --features host` does not require `tracer` solely to name program preprocessing or bytecode rows.
- [ ] Fixture consistency tests prove byte-for-byte or structure-for-structure parity with the current RV64 expansion output for supported RV64 instruction families.
- [ ] Tests cover recursive expansion, `rd = x0`, virtual-register clearing, compressed source instructions, and bytecode PC mapping.
- [ ] The crate dependency surface is suitable for future formal verification and extraction work: no CPU, memory-device, prover, transcript, or ELF parser dependency in `jolt-program::expand`.
- [ ] Program-preprocessing and verifier-facing modules document any deliberate choices that make Hax/Aeneas/Lean extraction harder, so those choices can be revisited in a follow-up rather than discovered after the split.

### Testing Strategy

Existing host-mode tests that decode, trace, preprocess bytecode, and prove guest programs must continue passing with `--features host`. Existing ZK-mode tests that consume bytecode preprocessing must continue passing with `--features host,zk`.

Add parity tests that compare the current implementation and the new crate during the transition. These should include:

- one test per instruction family with non-trivial inline expansion,
- representative loads, stores, arithmetic, division, remainder, shifts, AMOs, CSR, and trap-related instructions,
- explicit `rd = x0` cases for side-effect-free and side-effecting instructions,
- recursive helpers where an emitted helper instruction expands further,
- tests that assert exact `virtual_sequence_remaining` and `is_first_in_sequence` values,
- tests that build `BytecodePCMapper` from expanded output and verify stable PC indices,
- tests that `jolt-program::image` rejects ELF32/RV32 inputs with the expected unsupported-architecture error.

The parity process should be:

1. Before deleting the old production expansion entry points, add tests that call both the current `tracer` expansion path and the new `jolt-program::expand` path on the same decoded instruction corpus.
2. Use those dual-run tests to fix the new implementation until it matches old output exactly for instruction variant, normalized operands, address, compressed metadata, `is_first_in_sequence`, and `virtual_sequence_remaining`.
3. Generate checked-in fixture expectations from the old behavior for a curated corpus of decoded instructions and small ELF programs. The fixtures should serialize normalized expanded bytecode rows, not raw debug strings.
4. Cut all production call sites over to the new crate.
5. Delete the old production expansion implementation from `tracer`.
6. Keep the fixture consistency tests and property/invariant tests after deletion so CI continues to guard the new implementation without requiring the old implementation to remain in production.

Do not leave the old expansion implementation as a compatibility shim. A small test-only reference module may be used during implementation if it makes the transition safer, but the final merged production code should have one canonical expansion implementation.

Add `jolt-eval` invariants:

- `bytecode_expansion_fixture_consistency`: for generated or fixture decoded instructions, compare expanded output to fixture expectations, including instruction variants, normalized operands, flags, addresses, compressed metadata, `is_first_in_sequence`, and `virtual_sequence_remaining`. During the transition only, the fixture generator may compare the old expansion path to the new crate output before the old path is deleted.
- `program_preprocessing_determinism`: for small fixture ELFs or generated guest programs, build program preprocessing through the modular path repeatedly and compare the serialized preprocessing digest. During the transition only, also compare the old macro/host path to the new modular path before the old path is deleted.
- `bytecode_pc_mapping_consistency`: for expanded bytecode with inline sequences, assert that `BytecodePCMapper::get_pc(address, virtual_sequence_remaining)` matches fixture expectations and the dense instruction order produced by preprocessing.

After the full cutover, keep fixture consistency tests in the new crate so future changes to expansion semantics are intentional and reviewable.

### Performance

The refactor should have no measurable runtime regression in guest decoding, tracing, or bytecode preprocessing. The implementation should benchmark at least one representative guest decode/trace path before and after the change. Any additional allocation introduced by the crate boundary should be removed or justified before merge.

Add a Criterion benchmark under `jolt-eval` for decode-plus-expansion throughput over at least one small guest and one instruction-heavy fixture. This can become a performance objective after the benchmark is stable; it does not need to block the architectural split.

## Design

### Architecture

Current expansion architecture:

```text
jolt-trace::decode
  -> tracer::decode(elf)
  -> Instruction::inline_sequence(&VirtualRegisterAllocator, Xlen::Bit64)
      -> per-instruction inline_sequence methods
      -> InstrAssembler::add_to_sequence
          -> emitted_instruction.inline_sequence(...)
```

This means `jolt-trace` exposes bytecode-related APIs, but the expansion algorithm, recursive expansion policy, and virtual register allocator are still entangled with `tracer`.

Current program/prover/verifier preprocessing architecture:

```text
program.decode()
  -> expanded bytecode
  -> memory_init bytes
  -> program_size
  -> ELF entry address

JoltProgramPreprocessing::new(...)
  -> BytecodePreprocessing::preprocess(expanded bytecode, entry address)
  -> RAMPreprocessing::preprocess(memory_init)
  -> MemoryLayout
  -> max_padded_trace_length

JoltVerifierPreprocessing::new(...)
  -> JoltProgramPreprocessing
  -> PCS verifier setup
  -> optional BlindFold setup
```

At verification time, `JoltVerifier` consumes `JoltVerifierPreprocessing`; it does not need the original ELF, CPU execution, memory-device emulation, advice tapes, lazy trace iteration, or prover witness generation. However, if a user wants to derive preprocessing from an ELF, the program construction path needs deterministic ELF decoding, bytecode expansion, bytecode preprocessing, RAM preprocessing, memory layout, and preprocessing serialization/digesting.

Proposed architecture:

```text
jolt-program
  image
    -> decode_elf(elf)
    -> decoded RV64 program instructions, memory init, program end, entry address

  expand
    -> expand_instruction(decoded_program_instruction, allocator)
    -> expand_program(decoded_program_instructions)
    -> expansion metadata and errors

  preprocess
    -> BytecodePreprocessing
    -> BytecodePCMapper
    -> RAMPreprocessing
    -> JoltProgramPreprocessing

jolt-trace
  -> builds guest ELFs
  -> calls jolt-program image, expand, and preprocess APIs
  -> owns tracing-oriented host convenience APIs

tracer
  -> executes instructions and produces traces
  -> calls jolt-program::expand when trace-time expansion is required

jolt-core verifier
  -> consumes JoltVerifierPreprocessing
  -> depends on program preprocessing artifacts, not on tracer execution
```

`jolt-riscv` and `jolt-program` should own the instruction, expansion, and materialized program-preprocessing abstractions, not the full tracer. The implementation PR should produce this dependency direction:

```text
common / jolt-platform
        ^
        |
jolt-riscv
        ^
        |
jolt-program
  image -> expand -> preprocess
        ^
        |
jolt-core, jolt-trace, tracer
```

The `image -> expand -> preprocess` line is a module-level pipeline, not a license for every module to depend on every dependency. `jolt-program::expand` should remain pure and should not inherit the ELF parser dependency from `jolt-program::image`.

The implementation PR should land the full crate cutover described here rather than a partial first pass. If the implementation discovers a smaller dependency cleanup that should be split out first, do that in a prerequisite PR before the crate-cutover PR, not by merging this design with temporary ownership or compatibility layers.

The broader program boundary should be evaluated in the same design pass:

| Concern | Verifier needs it? | Proposed home |
|---------|--------------------|---------------|
| ELF parsing into decoded RV64 instructions, memory image, entry address, and program end | Only when deriving preprocessing from ELF; not during proof verification from serialized preprocessing | `jolt-program::image` |
| Recursive bytecode expansion | Yes, for program preprocessing from ELF; not during verification from already-serialized preprocessing | `jolt-program::expand` |
| Bytecode table padding and PC mapping | Yes; verifier needs metadata derived from bytecode preprocessing; committed-bytecode mode should replace full bytecode rows with trusted commitments outside this crate | `jolt-program::preprocess` |
| RAM preprocessing from ELF memory bytes | Yes; verifier checks RAM initialization claims against program preprocessing or future program-image commitments | `jolt-program::preprocess` |
| Memory layout from program size and I/O limits | Yes; verifier validates public I/O sizes and RAM bounds | `common` plus `jolt-program::preprocess` |
| Program preprocessing digest and serialization | Yes; verifier binds preprocessing or its committed digest to the proof context | `jolt-program::preprocess` |
| PCS verifier setup and optional BlindFold setup | Yes, but these are proof-system setup artifacts rather than program artifacts | stay in `jolt-core` verifier or a proof-system setup crate |
| CPU execution and trace production | No | `tracer` |
| Memory-device emulation and I/O side effects | No, except the verifier consumes public `JoltDevice`/memory layout data supplied with the proof API | `tracer`/host APIs plus `common` data types |
| Advice tape I/O | No for verification, except commitments/public inputs already represented in proof and verifier inputs | prover/host tracing crates |
| Prover witness generation and prover-facing polynomial preprocessing | No | `jolt-core` prover |

#### Crate Placement And Dependency Direction

New library code should live under `crates/jolt-program` and be added to the root workspace `members` and `[workspace.dependencies]`, matching the newer crate layout used by `jolt-trace`, `jolt-riscv`, `jolt-openings`, and related libraries. Existing top-level crates such as `tracer`, `common`, `jolt-core`, and `jolt-sdk` can depend on `jolt-program` through workspace dependencies. `jolt-riscv` already exists and should be refactored in place rather than replaced by a sibling instruction crate.

It is acceptable for `tracer` to import `jolt-program`, but only if `jolt-program` does not depend back on `tracer`. The current code does not yet satisfy that shape: the concrete `Instruction`, `Cycle`, `RISCVInstruction`, `NormalizedInstruction`, and per-instruction structs live in `tracer`, and `crates/jolt-riscv/Cargo.toml` has a direct `tracer = { workspace = true, features = ["std"] }` dependency for those types. Therefore `jolt-program::expand` cannot both depend on `tracer::instruction::*` and be imported by `tracer` without creating a dependency cycle.

The implementation should resolve this by moving the shared instruction vocabulary, normalized operand view, instruction flags, and decode-facing program representation into the existing `jolt-riscv` crate. This crate already owns Jolt's RISC-V instruction kinds and circuit/instruction flag metadata; the cutover should make it the lower crate that `tracer` consumes, not a wrapper around `tracer` types.

The concrete execution data structures should remain in `tracer` unless implementation work proves that moving a small piece is clearly better. In particular, `tracer::instruction::Instruction`, `Cycle`, `RISCVCycle<T>`, per-instruction structs such as `ADD`/`LW`, register-state types, RAM-access types, and execution traits stay in `tracer`. The lower shared representation should instead be an abstract program instruction, for example:

```rust
pub enum InstructionKind { Add, Lw, /* ... */ }

pub struct ProgramInstruction {
    pub kind: InstructionKind,
    pub operands: NormalizedOperands,
    pub address: u64,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}
```

`jolt-program::image`, `jolt-program::expand`, and `jolt-program::preprocess` should use `ProgramInstruction`, not tracer's concrete instruction structs. `tracer` should provide conversion at its boundary, such as `From<&tracer::instruction::Instruction> for ProgramInstruction` and `TryFrom<ProgramInstruction> for tracer::instruction::Instruction`, so execution can keep concrete structs while verifier-facing program preprocessing stays independent of `tracer`.

`JoltInstruction` follows the same ownership rule. `jolt-riscv` defines the trait and should implement it for local program representations such as `ProgramInstruction`. It must remove the current blanket implementation over `tracer::instruction::RISCVInstruction`, because that implementation forces `jolt-riscv` to name `tracer`. If tracer's concrete instruction structs still need to satisfy `JoltInstruction`, those impls belong in `tracer` because the concrete types are local there. They should be adapter impls, preferably generated by reusing `jolt_riscv::for_each_instruction_kind!`, and can delegate through conversion to `ProgramInstruction`.

After that, the intended dependency direction is:

```text
common / jolt-platform
        ^
        |
jolt-riscv
        ^
        |
jolt-program
        ^
        |
jolt-core, jolt-trace, tracer
```

This direction keeps `tracer` as an execution engine while allowing it to reuse the same bytecode expansion module as program preprocessing. It also gives formal verification tools a target that is not downstream of the emulator.

#### Proposed Workspace Layout

The implementation should refactor existing `crates/jolt-riscv` and introduce one new crate under `crates/`:

```text
crates/
  jolt-riscv/
    src/
      lib.rs
      decode.rs
      uncompress.rs
      normalized.rs
      operands.rs
      kind.rs
      program_instruction.rs
      traits.rs
      instructions/
        mod.rs
        add.rs
        addi.rs
        ...
        virtual_advice.rs
        virtual_assert_*.rs
        virtual_*.rs

  jolt-program/
    Cargo.toml
    src/
      lib.rs
      error.rs
      image/
        mod.rs
        elf.rs
      expand/
        mod.rs
        allocator.rs
        assembler.rs
        error.rs
        metadata.rs
        sequences/
          mod.rs
          arithmetic.rs
          bitwise.rs
          branch.rs
          csr.rs
          div_rem.rs
          load_store.rs
          shift.rs
          trap.rs
          virtuals.rs
      preprocess/
        mod.rs
        bytecode.rs
        ram.rs
        program.rs
        error.rs
```

The file list is intentionally concrete. Implementers may split individual instruction families differently if the resulting module boundary and dependency graph are the same, but the PR should not leave expansion logic scattered across `tracer`.

#### Crate Responsibilities

`crates/jolt-riscv` owns the data model that must be shared by decoding, expansion, tracing, bytecode preprocessing, and verifier checks. It already contains Jolt instruction kind wrappers, `JoltInstruction`, `JoltInstructions`, and circuit/instruction flag metadata; this PR should remove its current `tracer` dependency by making it own the abstract program-instruction vocabulary rather than tracer's concrete execution structs:

- `src/kind.rs`: own `InstructionKind`, the canonical names of real and virtual Jolt instructions, plus static metadata such as side-effect classification.
- `src/operands.rs`: own `NormalizedOperands` and operand accessors.
- `src/normalized.rs`: own any normalized instruction view needed by existing flag and lookup code, or fold that view into `ProgramInstruction` if the implementation can do so cleanly.
- `src/program_instruction.rs`: own `ProgramInstruction`, including kind, normalized operands, address, virtual sequence metadata, and compressed-instruction metadata.
- `src/jolt_instruction.rs`: define `JoltInstruction` and implement it for `ProgramInstruction`; do not implement it blanket-style for `tracer::instruction::RISCVInstruction`.
- `src/instructions/*.rs`: keep the existing Jolt instruction kind wrappers and flag declarations; do not move tracer's concrete execution structs into this crate as the default design.
- `src/decode.rs`: own RV64 opcode decoding into `ProgramInstruction`.
- `src/uncompress.rs`: own RV64 compressed RISC-V decompression used by ELF decode.
- `src/traits.rs`: define pure traits needed by `ProgramInstruction` and static metadata. Execution-specific traits must remain outside this crate.

The shared instruction list and macro input should move into `jolt-riscv`. Today `tracer/src/instruction/mod.rs` generates `Instruction` and `Cycle` from the same instruction list. After the split, `jolt-riscv` should own the canonical list and use it to generate `InstructionKind` plus pure metadata dispatch. `tracer` should reuse that same list by invoking an exported `macro_rules!` token tree from `jolt-riscv`, for example `jolt_riscv::for_each_instruction_kind!`, to define its concrete `Instruction`, `Cycle`, `RISCVCycle<T>`, and execution/trace dispatch. The implementation must not duplicate the long instruction list across crates.

`crates/jolt-program` owns program image decoding, bytecode expansion, and materialized program preprocessing. These are one package because they are one program-construction pipeline, but the internal modules should remain separate enough that dependency and formalization boundaries are still visible.

`jolt-program::image` owns deterministic ELF parsing:

- `image/elf.rs`: move the non-execution part of `tracer::decode`: parse ELF, reject ELF32/RV32, compute entry address, collect RAM image bytes, decode `.text` into RV64 `ProgramInstruction` values, and compute `program_end`.
- `image/mod.rs`: expose `DecodedProgramImage` and `decode_elf`.
- `error.rs`: replace warnings/panics in decode with explicit `ProgramError` or `DecodeError` values where possible. If current behavior must preserve `UNIMPL` insertion for invalid words, encode that policy explicitly.

The `object` dependency should be feature-gated and used only by `jolt-program::image`. The pure expansion and preprocessing modules should still be usable in builds that operate on decoded instructions or serialized preprocessing rather than ELF bytes.

`jolt-program::expand` owns source-instruction to final-bytecode-row expansion:

- `expand/allocator.rs`: move and simplify `VirtualRegisterAllocator`; remove `Arc<Mutex<_>>` unless cross-thread sharing is truly required by the expansion API.
- `expand/assembler.rs`: move `InstrAssembler`, `Value`, and inline emission helpers from `tracer/src/utils/inline_helpers.rs`.
- `expand/mod.rs`: provide `expand_instruction`, `expand_program`, and recursive expansion dispatch.
- `expand/metadata.rs`: assign `is_first_in_sequence`, `virtual_sequence_remaining`, source address, and compressed metadata consistently.
- `expand/sequences/*.rs`: move per-instruction inline expansion logic out of `tracer/src/instruction/*.rs`, grouped by instruction family.
- `expand/error.rs`: define `ExpansionError` for virtual register exhaustion, invalid inline write targets, malformed sequence metadata, and unsupported instructions.

`ExpansionAllocator` should be single-owner mutable state passed as `&mut ExpansionAllocator` through expansion. The current `VirtualRegisterAllocator` uses `Arc<Mutex<_>>` because the old API exposes only `&VirtualRegisterAllocator`, clones that allocator into `InstrAssembler` and `VirtualRegisterGuard`, and relies on guards deallocating through shared state on `Drop`. The new crate should make that state flow explicit instead: recursive expansion and inline finalization should borrow one allocator mutably, and any current per-CPU or per-thread allocator sharing should become per-expansion ownership unless an implementation can name a real cross-thread requirement. `InstrAssembler` should therefore borrow `&mut ExpansionAllocator` for the duration of emission rather than owning or cloning allocator state. Its field shape should make the borrow visible, for example an `InstrAssembler<'a>` containing an `&'a mut ExpansionAllocator` plus only the emission buffers and metadata needed for the active inline sequence.

`jolt-program::preprocess` owns materialized program preprocessing artifacts used by both prover and verifier:

- `preprocess/bytecode.rs`: move `BytecodePreprocessing`, `BytecodePCMapper`, and bytecode preprocessing errors from `jolt-core/src/zkvm/bytecode/mod.rs`.
- `preprocess/ram.rs`: move `RAMPreprocessing` and pure RAM initialization helpers from `jolt-core/src/zkvm/ram/mod.rs`. Dependency purity is binding for `compute_min_ram_K`: move it only if it stays dependency-light. If relocating it would drag `jolt-core`, PCS setup, prover-only modules, or proof-system configuration into `jolt-program::preprocess`, leave `compute_min_ram_K` in `jolt-core` and have it consume the pure RAM preprocessing surface from `jolt-program::preprocess`.
- `preprocess/program.rs`: move the current shared layer into the final `JoltProgramPreprocessing` type, including canonical serialization and `digest()`.
- `preprocess/error.rs`: consolidate bytecode/RAM/program preprocessing errors.

This module is intentionally program-level rather than verifier-only. The current code already has a `JoltSharedPreprocessing` layer containing `Arc<BytecodePreprocessing>`, `RAMPreprocessing`, `MemoryLayout`, and `max_padded_trace_length`; both `JoltProverPreprocessing` and `JoltVerifierPreprocessing` wrap that shared layer. This PR should cut over that layer to `JoltProgramPreprocessing` in `jolt-program::preprocess`. `JoltProverPreprocessing` and `JoltVerifierPreprocessing` should remain in `jolt-core` unless a later proof-system setup refactor moves them, because they add PCS prover/verifier setup and optional BlindFold setup rather than program preprocessing semantics.

`jolt-program::preprocess` should not own committed-bytecode or committed-program-image derivation. Those artifacts require PCS prover setup, Dory geometry, trusted commitments, and prover opening hints. They should remain in `jolt-core` or a future PCS-aware committed-program crate that consumes `JoltProgramPreprocessing`.

`tracer` remains the execution engine:

- keeps `Cpu`, `Memory`, `LazyTraceIterator`, `Cycle`, execution semantics, advice tape plumbing, trace-to-file support, and `jolt-emu`;
- imports `jolt-riscv` for instruction data;
- imports `jolt-program::expand` for trace-time expansion and trace length accounting;
- imports `jolt-program::image` for ELF decode where tracing from ELF is needed.

`jolt-core` remains the proof system:

- imports program preprocessing artifacts from `jolt-program::preprocess`;
- keeps prover/verifier protocols, sumchecks, commitments, witness generation, and proof serialization;
- should not import `tracer` just to name bytecode preprocessing artifacts.

`jolt-trace` remains a host-facing convenience crate:

- owns `Program`, guest build helpers, analyze helpers, and user-facing trace APIs;
- calls `jolt_program::image::decode_elf`, `jolt_program::expand::expand_program`, and `jolt_program::preprocess` instead of directly coupling expansion to `tracer`.

#### Dependency Table

Target dependency edges:

| Crate | May depend on | Must not depend on |
|-------|---------------|--------------------|
| `jolt-riscv` | `common`, `jolt-platform`, `serde`, `ark-serialize`, `strum`, `paste`, `derive_more` | `tracer`, `jolt-core`, `jolt-trace`, `jolt-program`, `object` |
| `jolt-program` | `jolt-riscv`, `common`, `ark-serialize`, `blake2`, `thiserror`, `serde`, optional `object` and `tracing` behind image/host features | `tracer`, `jolt-core`, `jolt-trace`, PCS implementations, Dory setup, prover-only modules |
| `tracer` | `jolt-riscv`, `jolt-program`, `common`, `jolt-platform` | `jolt-core` |
| `jolt-core` | `jolt-riscv`, `jolt-program`, proof-system crates, `common` | `tracer` for program preprocessing types |
| `jolt-trace` | `jolt-program`, `tracer`, `common` | owning canonical expansion semantics |

The implementation PR must not introduce a new cycle. Its final state should remove the `tracer -> jolt-program -> tracer` cycle risk entirely by moving shared instruction data below both crates.

#### Workspace And Cargo Changes

Root `Cargo.toml` changes:

- Add workspace member `"crates/jolt-program"`.
- Add workspace dependency `jolt-program = { path = "./crates/jolt-program", default-features = false }`.
- Keep the existing `jolt-riscv = { path = "./crates/jolt-riscv", ... }` workspace dependency, but update its feature policy if needed so verifier-critical consumers can use it without pulling in `tracer`.

`jolt-program` feature policy:

- default features should be minimal;
- `std` should gate file-system and allocation-heavy conveniences where possible;
- `image` or `elf` should gate the `object` parser dependency;
- verifier-critical modules should compile without pulling in `tracer`;
- test fixtures can use `test-utils`, but production expansion should not depend on randomized generation.

Existing crate `Cargo.toml` updates:

- `tracer/Cargo.toml`: add `jolt-riscv` and `jolt-program`.
- `crates/jolt-riscv/Cargo.toml`: remove the direct dependency on `tracer`; keep only lower-level dependencies needed by static instruction data, flags, decoding, and serialization.
- `crates/jolt-trace/Cargo.toml`: add `jolt-program`; keep `tracer` for execution only.
- `jolt-core/Cargo.toml`: add `jolt-riscv` and `jolt-program`; remove program-preprocessing dependence on `tracer` where feasible.
- `jolt-sdk/Cargo.toml` and `jolt-sdk/macros/Cargo.toml`: update imports only as needed for generated preprocessing functions.

The crate should provide explicit APIs along these lines:

```rust
pub fn expand_instruction(
    instruction: &ProgramInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<ProgramInstruction>, ExpansionError>;

pub fn expand_program(
    instructions: impl IntoIterator<Item = ProgramInstruction>,
) -> Result<Vec<ProgramInstruction>, ExpansionError>;
```

The exact type names can change during implementation, but the API should make state flow explicit. In particular, virtual-register allocation state should be visible as expansion state rather than hidden inside the tracer CPU.

The program-image and preprocessing APIs should be explicit about their inputs:

```rust
pub struct DecodedProgramImage {
    pub instructions: Vec<ProgramInstruction>,
    pub memory_init: Vec<(u64, u8)>,
    pub program_end: u64,
    pub entry_address: u64,
}

pub fn preprocess_program(
    expanded_bytecode: Vec<ProgramInstruction>,
    memory_init: Vec<(u64, u8)>,
    memory_layout: MemoryLayout,
    max_padded_trace_length: usize,
    entry_address: u64,
) -> Result<JoltProgramPreprocessing, PreprocessingError>;
```

The important design point is not these exact signatures; it is that program preprocessing should not have to import `tracer::emulator::Cpu`, `LazyTraceIterator`, advice-tape machinery, host `Program` build logic, PCS setup, or committed-bytecode derivation.

The implementation should also isolate formal-verification-friendly logic:

- prefer total functions returning `Result` over panics in core expansion paths,
- keep CPU, memory, ELF, prover, transcript, and device dependencies out of `jolt-program::expand`,
- keep macro-generated dispatch behind a narrow boundary owned by `jolt-riscv`, with `tracer` reusing `jolt_riscv::for_each_instruction_kind!` or the final exported macro name for concrete `Instruction` and `Cycle` generation,
- avoid concurrency primitives in allocator state unless they are needed by the API,
- document every invariant needed by Lean/Hax/Aeneas models, especially recursion and metadata assignment.

#### Formal Verification Readiness

Hax/Aeneas/Lean extraction is a reach goal for follow-up work, not a required deliverable for the crate-cutover PR. The implementation should nevertheless index on extraction strongly enough that the new boundaries do not need to be redesigned later.

For this PR, "extractable or close to extractable" means:

- `jolt-program::expand` core logic uses explicit input/output state and total functions where practical;
- allocator state is plain owned Rust data, not hidden behind `Arc<Mutex<_>>`, global state, interior mutability, or thread-local state;
- core expansion does not depend on file I/O, ELF parsing, logging, tracing spans, CPU execution, memory emulation, advice tapes, prover code, or transcript code;
- errors are represented with small enums instead of panics in normal control flow;
- recursive expansion has an explicit termination argument in comments and tests, such as "each recursive step expands into instructions from a strictly lower expansion class" or another reviewable measure;
- metadata assignment is centralized in one module so the Lean model can state and prove one theorem about `virtual_sequence_remaining` and `is_first_in_sequence`;
- APIs avoid trait objects, dynamic dispatch, async, macros that hide core semantics, and callback-heavy designs in the extraction-critical modules;
- any necessary macro-generated instruction dispatch expands to simple first-order functions that can be audited or extracted after expansion.

The right indexing is therefore:

- High for `jolt-program::expand`, because bytecode expansion is the main formalization target.
- Medium for `jolt-program::image`, because ELF parsing is verifier-facing when preprocessing from ELF, but it may be acceptable to trust or separately validate object-file parsing before formalizing expansion.
- Medium for `jolt-program::preprocess`, because bytecode/RAM preprocessing and digest binding are verifier-critical and prover-consumed, but PCS setup and commitment derivation remain trusted proof-system boundaries outside this crate.
- Low for `jolt-trace` and `tracer` execution internals in this PR, because they should become consumers of the verifier-facing crates rather than extraction targets.

This PR should not contort the code to satisfy a specific extractor immediately. It should keep the extraction-critical core small, pure, dependency-light, and well-specified so a follow-up PR can run Hax/Aeneas and then address concrete tool feedback without reopening the architecture.

### Alternatives Considered

Keeping expansion in `tracer` and documenting it there is the smallest code change, but it leaves the formalization target coupled to CPU execution, memory, advice, and trace infrastructure.

Treating `jolt-trace` as the expansion crate is also tempting because it already contains bytecode preprocessing, but `jolt-trace` is host-facing and still builds guests, reads ELF files, invokes the tracer, and owns program APIs. That crate is useful to users, but it is not the minimal formalization target.

Moving only `BytecodePreprocessing` is insufficient because preprocessing consumes already-expanded bytecode. The correctness-sensitive part for formalization is the path that turns one decoded source instruction into zero, one, or many final bytecode rows.

Creating separate `jolt-program-image`, `jolt-bytecode-expand`, and `jolt-program-preprocess` crates was considered, and it is a reasonable architecture if these modules later grow independent release or ownership needs. For this PR, separate crates look premature: they would add workspace churn and dependency edges without changing the true pipeline boundary. A single `jolt-program` crate with explicit `image`, `expand`, and `preprocess` modules gives the same conceptual separation with less mechanical overhead.

Creating one large `jolt-verifier-input` crate for ELF decode, expansion, bytecode preprocessing, RAM preprocessing, verifier setup, PCS setup, and committed-program data would blur the boundary between materialized program preprocessing and proof-system setup. `jolt-program` should stop at materialized program artifacts; proof-system setup and committed-program derivation stay outside.

### Related Branches And Future Integration

This spec was checked against two ongoing branches, but neither branch expands the scope of this PR.

The `refactor/crates` branch is useful as a point of comparison for proof-system setup boundaries. Its `jolt-zkvm::preprocessing` work computes module/protocol shape and PCS setup for compiled modules; it does not extract ELF/program bytecode or RAM preprocessing out of `jolt-core`. This supports keeping `jolt-program::preprocess` focused on materialized program artifacts while leaving PCS setup and prover/verifier keys in `jolt-core` or a later proof-system setup crate.

The bytecode-commitment PR [#1344](https://github.com/a16z/jolt/pull/1344), branch `amir/bytecode-commitment-merged`, is useful as a point of comparison for future committed-program integration. That branch distinguishes full program preprocessing from committed program preprocessing: the prover still needs full bytecode, RAM/program image data, and opening hints, while the verifier should only need metadata plus trusted bytecode and program-image commitments. This PR should not absorb that committed path. Instead, `jolt-program::preprocess` should provide the materialized input that committed-program preprocessing can consume later. Commitment derivation, trusted commitments, Dory geometry, and opening hints remain outside this crate.

The future-compatible shape is therefore:

```text
jolt-program::preprocess
  -> materialized JoltProgramPreprocessing
      -> bytecode preprocessing
      -> RAM/program-image preprocessing
      -> memory layout, trace bound, entry metadata, digest

jolt-core or future committed-program crate
  -> consumes JoltProgramPreprocessing
  -> derives committed bytecode/program-image artifacts using PCS setup
  -> gives prover full data plus hints
  -> gives verifier metadata plus trusted commitments
```

## Documentation

Update the Jolt book only if the crate is exposed to users or changes contributor-facing architecture. At minimum, add developer documentation describing:

- where bytecode expansion lives,
- which modules are verifier-critical and which crates are prover/host-only,
- how program preprocessing can be derived from serialized preprocessing or from ELF bytes,
- how decoded instructions become expanded bytecode rows,
- how `virtual_sequence_remaining` is assigned and consumed,
- how to add or modify an instruction expansion safely.

## Execution

### Files To Create

- `crates/jolt-riscv/src/decode.rs`
- `crates/jolt-riscv/src/uncompress.rs`
- `crates/jolt-riscv/src/normalized.rs`
- `crates/jolt-riscv/src/operands.rs`
- `crates/jolt-riscv/src/kind.rs`
- `crates/jolt-riscv/src/program_instruction.rs`
- `crates/jolt-riscv/src/traits.rs`
- `crates/jolt-program/Cargo.toml`
- `crates/jolt-program/src/lib.rs`
- `crates/jolt-program/src/error.rs`
- `crates/jolt-program/src/image/mod.rs`
- `crates/jolt-program/src/image/elf.rs`
- `crates/jolt-program/src/expand/mod.rs`
- `crates/jolt-program/src/expand/allocator.rs`
- `crates/jolt-program/src/expand/assembler.rs`
- `crates/jolt-program/src/expand/error.rs`
- `crates/jolt-program/src/expand/metadata.rs`
- `crates/jolt-program/src/expand/sequences/*.rs`
- `crates/jolt-program/src/preprocess/mod.rs`
- `crates/jolt-program/src/preprocess/bytecode.rs`
- `crates/jolt-program/src/preprocess/ram.rs`
- `crates/jolt-program/src/preprocess/program.rs`
- `crates/jolt-program/src/preprocess/error.rs`
- `jolt-eval/src/invariant/bytecode_expansion.rs`
- `jolt-eval/benches/decode_expand.rs`

### Files To Modify

- `Cargo.toml`: add the `crates/jolt-program` workspace member and workspace dependency; keep the existing `jolt-riscv` workspace entry.
- `Cargo.lock`: update after adding the workspace crate.
- `tracer/Cargo.toml`: depend on `jolt-riscv` and `jolt-program`.
- `tracer/src/lib.rs`: remove ELF decode ownership; tracer-specific ELF entry points should call `jolt_program::image::decode_elf` internally only when they also perform tracing/execution work.
- `tracer/src/emulator/cpu.rs`: keep tracer-local `Xlen` and concrete execution structs; call expansion APIs through conversion to/from `ProgramInstruction` where trace length or trace-time `rd = x0` expansion is needed.
- `tracer/src/instruction/**`: keep concrete instruction structs and execution-only methods in `tracer`; convert to/from `jolt-riscv`'s `ProgramInstruction` at the crate boundary; implement `JoltInstruction` for concrete tracer instruction types here if callers still need that trait on those types.
- `tracer/src/utils/inline_helpers.rs`: move expansion helpers to `jolt-program::expand`; delete or replace with imports during the same full cutover.
- `tracer/src/utils/virtual_registers.rs`: move allocator to `jolt-program::expand`; delete or replace imports.
- `crates/jolt-riscv/Cargo.toml`: remove `tracer` and add any lower-level dependencies needed by moved instruction data.
- `crates/jolt-riscv/src/lib.rs`: expose the canonical instruction-kind list, `InstructionKind`, `ProgramInstruction`, `NormalizedOperands`, pure traits, and flags without referencing `tracer`.
- `crates/jolt-riscv/src/instructions/**`: keep existing Jolt instruction kind wrappers and pure instruction metadata; do not import tracer concrete instruction structs.
- `crates/jolt-riscv/src/jolt_instruction.rs`: remove the blanket impl over `tracer::instruction::RISCVInstruction`; implement `JoltInstruction` over `ProgramInstruction` or another local static representation.
- `crates/jolt-trace/Cargo.toml`: add `jolt-program`; keep `tracer` only for execution.
- `crates/jolt-trace/src/program.rs`: call `jolt_program::image` and `jolt_program::expand` in `decode`.
- `crates/jolt-trace/src/bytecode.rs`: remove or delegate to `jolt-program::preprocess` if duplicate with `jolt-core` bytecode preprocessing.
- `jolt-core/Cargo.toml`: add `jolt-riscv` and `jolt-program`.
- `jolt-core/src/zkvm/bytecode/mod.rs`: move or reexport materialized bytecode preprocessing from `jolt-program::preprocess`.
- `jolt-core/src/zkvm/ram/mod.rs`: move or reexport pure RAM preprocessing from `jolt-program::preprocess`; leave prover/verifier sumcheck modules in `jolt-core`.
- `jolt-core/src/zkvm/verifier.rs`: use program preprocessing types from `jolt-program::preprocess`, while keeping `JoltVerifierPreprocessing` if PCS setup remains in `jolt-core`.
- `jolt-core/src/zkvm/prover.rs`: update imports for program preprocessing, bytecode preprocessing, RAM preprocessing, and program instruction types.
- `jolt-core/src/poly/**`, `jolt-core/src/subprotocols/**`, `jolt-core/src/zkvm/**`: update imports from `tracer::instruction` to `jolt-riscv` where the code only needs `ProgramInstruction`, `InstructionKind`, or static instruction data.
- `jolt-sdk/macros/src/lib.rs`: generated preprocessing functions should call the modular decode/expand/program-preprocess path.
- `jolt-eval/src/invariant/mod.rs`: register bytecode expansion invariants.
- `jolt-eval/src/objective/mod.rs`: add `decode_expand` objective only if the benchmark is promoted to a measured objective in this PR.
- `book/**` or developer docs: add an architecture page or section if maintainers want crate-boundary docs in the book.

### Files To Remove Or Empty

Remove these only after all call sites are cut over:

- `tracer/src/utils/inline_helpers.rs`
- `tracer/src/utils/virtual_registers.rs`
- decode/uncompress helpers under `tracer/src/instruction/` once `jolt-riscv` owns RV64 decode into `ProgramInstruction`

Do not delete tracer's concrete instruction structs, instruction-format structs, execution-specific trace implementations, or RV32 cleanup code in this PR.

### Implementation Checklist

1. Add empty `jolt-program` crate under `crates/`, workspace member, workspace dependency, and minimal module skeletons.
2. Refactor existing `jolt-riscv` so it no longer depends on `tracer`.
3. Add `InstructionKind`, `ProgramInstruction`, normalized operands, and local static instruction metadata to `jolt-riscv` while keeping tracer's concrete instruction structs in `tracer`.
4. Move RV64 opcode decode and RV64 compressed-instruction decompression into `jolt-riscv`; reject RV32/ELF32 in the new program pipeline.
5. Export the canonical instruction-kind list from `jolt-riscv` as a macro such as `jolt_riscv::for_each_instruction_kind!`, and use it from `tracer` to generate its concrete `Instruction`, `Cycle`, and `RISCVCycle<T>`.
6. Replace the `JoltInstruction` blanket impl over `tracer::instruction::RISCVInstruction` with `JoltInstruction for ProgramInstruction`; add concrete tracer adapter impls only in `tracer` if needed.
7. Update `tracer`, `jolt-core`, and `jolt-trace` imports to compile against the strengthened `jolt-riscv`.
8. Add `jolt_program::image::decode_elf` by moving the non-execution logic from `tracer::decode`.
9. Update all call sites of `tracer::decode` that only need ELF decoding to call `jolt_program::image::decode_elf`; keep `tracer::decode` only if it remains a tracer-specific execution API rather than a compatibility shim.
10. Move `VirtualRegisterAllocator`, `InstrAssembler`, recursive `add_to_sequence`, and per-instruction inline expansion logic into `jolt-program::expand`, with `InstrAssembler` borrowing `&mut ExpansionAllocator`.
11. Implement `expand_instruction` and `expand_program`.
12. Update `tracer` trace-time expansion and trace length accounting to use `jolt-program::expand`.
13. Update `jolt-trace::decode` to call `jolt_program::image::decode_elf` followed by `jolt_program::expand::expand_program`.
14. Move `BytecodePreprocessing`, `BytecodePCMapper`, pure `RAMPreprocessing`, and `JoltProgramPreprocessing` into `jolt-program::preprocess`.
15. Keep `compute_min_ram_K` in `jolt-core` if moving it would pull prover-only or proof-system dependencies into `jolt-program::preprocess`.
16. Update `jolt-core` prover/verifier code to consume the moved preprocessing types.
17. Update SDK macro-generated preprocessing to use the modular path.
18. Add expansion parity tests before removing old expansion entry points.
19. Add `jolt-eval` invariants for expansion fixture consistency, PC mapping consistency, and program preprocessing determinism.
20. Add the decode-plus-expansion Criterion benchmark under `jolt-eval`.
21. Remove old canonical expansion ownership from `tracer`.
22. Run formatting, clippy, host tests, ZK tests, and targeted crate dependency checks.

### Verification Commands

The implementation PR should run:

```bash
cargo fmt -q
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo nextest run --cargo-quiet
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
cargo nextest run -p jolt-riscv --cargo-quiet
cargo nextest run -p jolt-program --cargo-quiet
cargo run -p jolt-eval --bin measure-objectives -- --no-bench
cargo bench -p jolt-eval --bench decode_expand -- --quick
cargo tree -p tracer
cargo tree -p jolt-riscv
cargo tree -p jolt-program
```

The dependency checks should confirm that `jolt-riscv` and `jolt-program` do not depend on `tracer`, and that the default or no-ELF feature path for `jolt-program` does not pull object parsing into `jolt-program::expand` or `jolt-program::preprocess`.

## References

- `crates/jolt-trace/src/program.rs`: current `decode` path calls `tracer::decode` and then expands with `Instruction::inline_sequence`.
- `crates/jolt-riscv/Cargo.toml`: currently depends on `tracer`; this must invert so `tracer` depends on `jolt-riscv`.
- `jolt-core/src/zkvm/verifier.rs`: current home of `JoltSharedPreprocessing` and `JoltVerifierPreprocessing`; the program/shared layer should move to `jolt-program::preprocess`, while proof-system setup can remain in `jolt-core`.
- `jolt-core/src/zkvm/bytecode/mod.rs`: bytecode preprocessing and PC mapping consume expanded `Instruction` rows.
- `jolt-core/src/zkvm/ram/mod.rs`: RAM preprocessing consumes ELF memory initialization bytes.
- `jolt-sdk/macros/src/lib.rs`: generated preprocessing calls `program.decode()` before constructing the current `JoltSharedPreprocessing`; after this PR they should construct `JoltProgramPreprocessing` through the modular path.
- `tracer/src/instruction/mod.rs`: `Instruction::inline_sequence` handles `rd = x0`, dispatch, and recursive expansion entry points.
- `tracer/src/utils/inline_helpers.rs`: `InstrAssembler` recursively expands helper-emitted instructions through `add_to_sequence`.
- `tracer/src/utils/virtual_registers.rs`: virtual register layout, allocator state, and inline clearing behavior.
- `tracer/src/lib.rs`: current ELF decode extracts instructions, memory bytes, program end, entry address, and `Xlen`.
- `jolt-eval/README.md`: describes invariant and objective hooks to add for this refactor.
- PR [#1369](https://github.com/a16z/jolt/pull/1369): added `jolt-trace`, which is useful host-facing trace structure but does not by itself isolate bytecode expansion.
- PR [#1260](https://github.com/a16z/jolt/pull/1260): broader crate refactor discussion relevant to dependency direction and crate boundaries.
