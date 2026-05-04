# Spec: Bytecode Expansion Crate

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao                      |
| Created     | 2026-05-01                     |
| Status      | in review                      |
| PR          | [#1490](https://github.com/a16z/jolt/pull/1490) |

## Summary

Jolt's bytecode expansion logic is still owned by the `tracer` crate even though `jolt-trace` now exposes host-facing program decoding, tracing, and bytecode preprocessing APIs. This spec proposes strengthening the existing `jolt-riscv` crate as the shared instruction-data crate, plus dedicated crates for program-image decoding, bytecode expansion, and materialized Jolt program preprocessing. The target is a modular pipeline from ELF bytes to expanded bytecode and program preprocessing, while keeping CPU execution, memory-device emulation, advice I/O, trace production, PCS setup, and commitment derivation out of the verifier-facing dependency graph.

## Intent

### Goal

Introduce a `jolt-bytecode-expand` crate that owns expansion from decoded RISC-V instructions into Jolt bytecode rows, including recursive inline expansion, virtual register allocation, and bytecode-row metadata, while making `tracer`, `jolt-trace`, and program preprocessing consumers of that crate instead of owners of expansion semantics.

### Invariants

- Expansion of a decoded instruction must produce exactly the same bytecode sequence as the current `Instruction::inline_sequence(&VirtualRegisterAllocator, Xlen)` behavior.
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
- This spec does not propose changing ELF decoding, guest compilation, memory initialization, or trace execution beyond redirecting call sites to the new expansion crate.
- This spec does not require completing Lean, Hax, or Aeneas extraction in the same implementation PR.
- This spec does not require redesigning the bytecode commitment, lookup tables, or prover constraints.
- This spec does not require moving PCS setup, commitment derivation, bytecode/program-image opening hints, BlindFold setup, `JoltProverPreprocessing`, or `JoltVerifierPreprocessing` out of `jolt-core`.
- This spec does not require integrating bytecode-commitment PR [#1344](https://github.com/a16z/jolt/pull/1344). That work is a future integration constraint, not part of this implementation scope.
- This spec does not require exposing a stable public API for third-party consumers outside the Jolt workspace.
- This spec does not require moving CPU execution, lazy trace iteration, advice tapes, or memory-device emulation into verifier-facing crates.

## Evaluation

### Acceptance Criteria

- [ ] The existing `crates/jolt-riscv` crate owns shared instruction data types currently required by decoding, expansion, tracing, bytecode preprocessing, and verifier checks.
- [ ] `crates/jolt-riscv` no longer depends on `tracer`; `tracer` depends on `jolt-riscv` for static instruction data instead.
- [ ] `crates/jolt-program-image` exists, is a workspace member, and exposes deterministic ELF decoding into a `DecodedProgramImage` without depending on `tracer`.
- [ ] `crates/jolt-bytecode-expand` exists, is a workspace member, and provides a pure expansion API over decoded instructions and `Xlen`.
- [ ] `crates/jolt-program-preprocess` exists, is a workspace member, and owns materialized bytecode/RAM/program preprocessing artifacts consumed by both prover and verifier setup.
- [ ] `jolt-program-preprocess` does not depend on PCS implementations, Dory setup, commitment derivation, bytecode/program-image opening hints, BlindFold setup, or prover-only witness generation.
- [ ] `tracer` no longer owns the canonical recursive bytecode expansion algorithm; it calls `jolt-bytecode-expand` for expansion during trace execution and trace length accounting.
- [ ] `jolt-trace::decode` no longer performs expansion by directly calling `Instruction::inline_sequence` from `tracer`; it calls `jolt-program-image` and `jolt-bytecode-expand`.
- [ ] `InstrAssembler`, `VirtualRegisterAllocator`, or their minimal expansion-facing equivalents live behind the `jolt-bytecode-expand` crate boundary.
- [ ] Expansion APIs return explicit errors for allocation or malformed-expansion failures instead of introducing new panics in the core expansion path.
- [ ] A program-preprocessing path is implemented as `ELF bytes -> DecodedProgramImage -> expanded bytecode -> JoltProgramPreprocessing`, and prover/verifier setup wraps that program preprocessing without re-decoding or re-expanding the program.
- [ ] The verifier-facing path does not depend on CPU execution, lazy tracing, memory-device emulation, advice I/O, or prover-only witness generation.
- [ ] `cargo tree -p tracer` shows `tracer` depending on the lower-level instruction/expansion crates, but none of those lower-level crates depending back on `tracer`.
- [ ] `cargo tree -p jolt-core --features host` does not require `tracer` solely to name program preprocessing or bytecode rows.
- [ ] Fixture consistency tests prove byte-for-byte or structure-for-structure parity with the current expansion output for supported RV32 and RV64 instruction families.
- [ ] Tests cover recursive expansion, `rd = x0`, virtual-register clearing, compressed source instructions, and bytecode PC mapping.
- [ ] The crate dependency surface is suitable for future formal verification and extraction work: no CPU, memory-device, prover, transcript, or ELF parser dependency in `jolt-bytecode-expand`.
- [ ] Program-preprocessing and verifier-facing crates document any deliberate choices that make Hax/Aeneas/Lean extraction harder, so those choices can be revisited in a follow-up rather than discovered after the split.

### Testing Strategy

Existing host-mode tests that decode, trace, preprocess bytecode, and prove guest programs must continue passing with `--features host`. Existing ZK-mode tests that consume bytecode preprocessing must continue passing with `--features host,zk`.

Add parity tests that compare the current implementation and the new crate during the transition. These should include:

- one test per instruction family with non-trivial inline expansion,
- representative loads, stores, arithmetic, division, remainder, shifts, AMOs, CSR, and trap-related instructions,
- explicit `rd = x0` cases for side-effect-free and side-effecting instructions,
- recursive helpers where an emitted helper instruction expands further,
- tests that assert exact `virtual_sequence_remaining` and `is_first_in_sequence` values,
- tests that build `BytecodePCMapper` from expanded output and verify stable PC indices.

The parity process should be:

1. Before deleting the old production expansion entry points, add tests that call both the current `tracer` expansion path and the new `jolt-bytecode-expand` path on the same decoded instruction corpus.
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
  -> Instruction::inline_sequence(&VirtualRegisterAllocator, Xlen)
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
jolt-program-image
  -> decode_elf(elf)
  -> decoded instructions, memory init, program end, entry address, xlen

jolt-bytecode-expand
  -> expand_instruction(decoded_instruction, allocator, xlen)
  -> expand_program(decoded_instructions, xlen)
  -> expansion metadata and errors

jolt-program-preprocess
  -> BytecodePreprocessing
  -> BytecodePCMapper
  -> RAMPreprocessing
  -> JoltProgramPreprocessing

jolt-trace
  -> builds guest ELFs
  -> calls jolt-program-image and jolt-bytecode-expand
  -> calls jolt-program-preprocess when program preprocessing is needed
  -> owns tracing-oriented host convenience APIs

tracer
  -> executes instructions and produces traces
  -> calls jolt-bytecode-expand when trace-time expansion is required

jolt-core verifier
  -> consumes JoltVerifierPreprocessing
  -> depends on program preprocessing artifacts, not on tracer execution
```

`jolt-riscv` and the new crates should own the expansion-facing and program-preprocessing abstractions, not the full tracer. The implementation PR should produce this dependency direction:

```text
instruction representation / common types
        ^
        |
jolt-program-image      jolt-bytecode-expand
        ^                       ^
        |                       |
jolt-program-preprocess <------+
        ^
        |
jolt-core verifier, prover, jolt-trace
```

The implementation PR should land the full crate cutover described here rather than a partial first pass. If the implementation discovers a smaller dependency cleanup that should be split out first, do that in a prerequisite PR before the crate-split PR, not by merging this design with temporary ownership or compatibility layers.

The broader crate split should be evaluated in the same design pass:

| Concern | Verifier needs it? | Proposed home |
|---------|--------------------|---------------|
| ELF parsing into decoded instructions, memory image, entry address, program end, and `Xlen` | Only when deriving preprocessing from ELF; not during proof verification from serialized preprocessing | `jolt-program-image` or similarly small decode crate |
| Recursive bytecode expansion | Yes, for program preprocessing from ELF; not during verification from already-serialized preprocessing | `jolt-bytecode-expand` |
| Bytecode table padding and PC mapping | Yes; verifier needs metadata derived from bytecode preprocessing; committed-bytecode mode should replace full bytecode rows with trusted commitments outside this crate | `jolt-program-preprocess` for materialized preprocessing |
| RAM preprocessing from ELF memory bytes | Yes; verifier checks RAM initialization claims against program preprocessing or future program-image commitments | `jolt-program-preprocess` for materialized preprocessing |
| Memory layout from program size and I/O limits | Yes; verifier validates public I/O sizes and RAM bounds | `common` plus `jolt-program-preprocess` |
| Program preprocessing digest and serialization | Yes; verifier binds preprocessing or its committed digest to the proof context | `jolt-program-preprocess` |
| PCS verifier setup and optional BlindFold setup | Yes, but these are proof-system setup artifacts rather than bytecode-expansion artifacts | stay in `jolt-core` verifier or a proof-system setup crate |
| CPU execution and trace production | No | `tracer` |
| Memory-device emulation and I/O side effects | No, except the verifier consumes public `JoltDevice`/memory layout data supplied with the proof API | `tracer`/host APIs plus `common` data types |
| Advice tape I/O | No for verification, except commitments/public inputs already represented in proof and verifier inputs | prover/host tracing crates |
| Prover witness generation and prover-facing polynomial preprocessing | No | `jolt-core` prover |

#### Crate Placement And Dependency Direction

New library crates should live under `crates/` and be added to the root workspace `members` and `[workspace.dependencies]`, matching the newer crate layout used by `jolt-trace`, `jolt-riscv`, `jolt-openings`, and related libraries. Existing top-level crates such as `tracer`, `common`, `jolt-core`, and `jolt-sdk` can depend on those `crates/*` libraries through workspace dependencies. `jolt-riscv` already exists and should be refactored in place rather than replaced by a sibling instruction crate.

It is acceptable for `tracer` to import new `crates/*` libraries, but only if those libraries do not depend back on `tracer`. The current code does not yet satisfy that shape: the concrete `Instruction`, `Cycle`, `RISCVInstruction`, `NormalizedInstruction`, and per-instruction structs live in `tracer`, and `crates/jolt-riscv/Cargo.toml` has a direct `tracer = { workspace = true, features = ["std"] }` dependency for those types. Therefore a new `jolt-bytecode-expand` crate cannot both depend on `tracer::instruction::*` and be imported by `tracer` without creating a dependency cycle.

The implementation should resolve this by first moving the shared instruction representation, normalized instruction view, instruction flags, and decode-facing types into the existing `jolt-riscv` crate. This crate already owns Jolt's RISC-V instruction kinds and circuit/instruction flag metadata; the cutover should make it the lower crate that `tracer` consumes, not a wrapper around `tracer` types.

After that, the intended dependency direction is:

```text
common / jolt-platform
        ^
        |
jolt-riscv
        ^
        |
jolt-program-image      jolt-bytecode-expand
        ^                       ^
        |                       |
jolt-program-preprocess <------+
        ^                       ^
        |                       |
jolt-core, jolt-trace, tracer --+
```

This direction keeps `tracer` as an execution engine while allowing it to reuse the same bytecode expansion crate as program preprocessing. It also gives formal verification tools a target that is not downstream of the emulator.

#### Proposed Workspace Layout

The implementation should refactor existing `crates/jolt-riscv` and introduce three new crates under `crates/`:

```text
crates/
  jolt-riscv/
    src/
      lib.rs
      xlen.rs
      decode.rs
      uncompress.rs
      normalized.rs
      operands.rs
      instruction.rs
      traits.rs
      format/
        mod.rs
        format_assert_align.rs
        format_b.rs
        format_i.rs
        format_inline.rs
        format_j.rs
        format_load.rs
        format_r.rs
        format_s.rs
        format_u.rs
        format_virtual_advice.rs
        format_virtual_right_shift_i.rs
        format_virtual_right_shift_r.rs
      instructions/
        mod.rs
        add.rs
        addi.rs
        ...
        virtual_advice.rs
        virtual_assert_*.rs
        virtual_*.rs

  jolt-program-image/
    Cargo.toml
    src/
      lib.rs
      elf.rs
      error.rs

  jolt-bytecode-expand/
    Cargo.toml
    src/
      lib.rs
      allocator.rs
      assembler.rs
      error.rs
      expand.rs
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

  jolt-program-preprocess/
    Cargo.toml
    src/
      lib.rs
      bytecode.rs
      ram.rs
      program.rs
      error.rs
```

The file list is intentionally concrete. Implementers may split individual instruction families differently if the resulting crate boundary and dependency graph are the same, but the PR should not leave expansion logic scattered across `tracer`.

#### Crate Responsibilities

`crates/jolt-riscv` owns the data model that must be shared by decoding, expansion, tracing, bytecode preprocessing, and verifier checks. It already contains Jolt instruction kind wrappers, `JoltInstruction`, `JoltInstructions`, and circuit/instruction flag metadata; this PR should move the concrete static instruction representation below `tracer` into this crate and remove its current `tracer` dependency:

- `src/xlen.rs`: move `Xlen` out of `tracer::emulator::cpu` so decoders and expanders do not import CPU execution.
- `src/operands.rs`: own `NormalizedOperands` and operand accessors.
- `src/normalized.rs`: own `NormalizedInstruction`.
- `src/format/*.rs`: move instruction format structs from `tracer/src/instruction/format/`.
- `src/instructions/*.rs`: merge the concrete instruction structs and pure instruction metadata from `tracer/src/instruction/` with the existing Jolt instruction kind wrappers and flag declarations.
- `src/instruction.rs`: own the `Instruction` enum, `NoOp`, `UNIMPL`, normalize/with-rd/set-metadata helpers, side-effect metadata, and variant dispatch that does not require CPU execution.
- `src/decode.rs`: move `Instruction::decode` and opcode decoding.
- `src/uncompress.rs`: move compressed RISC-V decompression used by ELF decode.
- `src/traits.rs`: define pure traits such as `RiscvInstruction`, `Normalize`, `HasSideEffects`, and metadata mutation traits. Execution-specific traits must remain outside this crate.

The shared instruction list and macro input should move into `jolt-riscv`. Today `tracer/src/instruction/mod.rs` generates `Instruction` and `Cycle` from the same instruction list. After the split, `jolt-riscv` should own the canonical list and use it to generate `Instruction` plus pure instruction dispatch. `tracer` should reuse that same list, via an exported macro or generated table, to define `Cycle`, `RISCVCycle<T>`, and execution/trace dispatch. The implementation must not duplicate the long instruction list across crates.

`crates/jolt-program-image` owns deterministic ELF parsing:

- `src/elf.rs`: move the non-execution part of `tracer::decode`: parse ELF, compute entry address, detect RV32/RV64, collect RAM image bytes, decode `.text` into `Instruction`, and compute `program_end`.
- `src/lib.rs`: expose `DecodedProgramImage` and `decode_elf`.
- `src/error.rs`: replace warnings/panics in decode with explicit `DecodeError` values where possible. If current behavior must preserve `UNIMPL` insertion for invalid words, encode that policy explicitly.

`crates/jolt-bytecode-expand` owns source-instruction to final-bytecode-row expansion:

- `src/allocator.rs`: move and simplify `VirtualRegisterAllocator`; remove `Arc<Mutex<_>>` unless cross-thread sharing is truly required by the expansion API.
- `src/assembler.rs`: move `InstrAssembler`, `Value`, and inline emission helpers from `tracer/src/utils/inline_helpers.rs`.
- `src/expand.rs`: provide `expand_instruction`, `expand_program`, and recursive expansion dispatch.
- `src/metadata.rs`: assign `is_first_in_sequence`, `virtual_sequence_remaining`, source address, and compressed metadata consistently.
- `src/sequences/*.rs`: move per-instruction inline expansion logic out of `tracer/src/instruction/*.rs`, grouped by instruction family.
- `src/error.rs`: define `ExpansionError` for virtual register exhaustion, invalid inline write targets, malformed sequence metadata, and unsupported instructions.

`ExpansionAllocator` should be single-owner mutable state passed as `&mut ExpansionAllocator` through expansion. The current `VirtualRegisterAllocator` uses `Arc<Mutex<_>>` because the old API exposes only `&VirtualRegisterAllocator`, clones that allocator into `InstrAssembler` and `VirtualRegisterGuard`, and relies on guards deallocating through shared state on `Drop`. The new crate should make that state flow explicit instead: recursive expansion and inline finalization should borrow one allocator mutably, and any current per-CPU or per-thread allocator sharing should become per-expansion ownership unless an implementation can name a real cross-thread requirement. `InstrAssembler` should therefore borrow `&mut ExpansionAllocator` for the duration of emission rather than owning or cloning allocator state.

`crates/jolt-program-preprocess` owns materialized program preprocessing artifacts used by both prover and verifier:

- `src/bytecode.rs`: move `BytecodePreprocessing`, `BytecodePCMapper`, and bytecode preprocessing errors from `jolt-core/src/zkvm/bytecode/mod.rs`.
- `src/ram.rs`: move `RAMPreprocessing` and pure RAM initialization helpers from `jolt-core/src/zkvm/ram/mod.rs`. Move `compute_min_ram_K` only if it stays dependency-light; if it would drag prover-only modules into this crate, leave `compute_min_ram_K` in `jolt-core` and have it consume the pure RAM preprocessing surface from `jolt-program-preprocess`.
- `src/program.rs`: move the current shared layer into the final `JoltProgramPreprocessing` type, including canonical serialization and `digest()`.
- `src/error.rs`: consolidate bytecode/RAM/program preprocessing errors.

This crate is intentionally program-level rather than verifier-only. The current code already has a `JoltSharedPreprocessing` layer containing `Arc<BytecodePreprocessing>`, `RAMPreprocessing`, `MemoryLayout`, and `max_padded_trace_length`; both `JoltProverPreprocessing` and `JoltVerifierPreprocessing` wrap that shared layer. This PR should cut over that layer to `JoltProgramPreprocessing` in `jolt-program-preprocess`. `JoltProverPreprocessing` and `JoltVerifierPreprocessing` should remain in `jolt-core` unless a later proof-system setup refactor moves them, because they add PCS prover/verifier setup and optional BlindFold setup rather than program preprocessing semantics.

`jolt-program-preprocess` should not own committed-bytecode or committed-program-image derivation. Those artifacts require PCS prover setup, Dory geometry, trusted commitments, and prover opening hints. They should remain in `jolt-core` or a future PCS-aware committed-program crate that consumes `JoltProgramPreprocessing`.

`tracer` remains the execution engine:

- keeps `Cpu`, `Memory`, `LazyTraceIterator`, `Cycle`, execution semantics, advice tape plumbing, trace-to-file support, and `jolt-emu`;
- imports `jolt-riscv` for instruction data;
- imports `jolt-bytecode-expand` for trace-time expansion and trace length accounting;
- imports `jolt-program-image` for ELF decode where tracing from ELF is needed.

`jolt-core` remains the proof system:

- imports program preprocessing artifacts from `jolt-program-preprocess`;
- keeps prover/verifier protocols, sumchecks, commitments, witness generation, and proof serialization;
- should not import `tracer` just to name bytecode preprocessing artifacts.

`jolt-trace` remains a host-facing convenience crate:

- owns `Program`, guest build helpers, analyze helpers, and user-facing trace APIs;
- calls `jolt-program-image::decode_elf`, `jolt-bytecode-expand::expand_program`, and `jolt-program-preprocess` instead of directly coupling expansion to `tracer`.

#### Dependency Table

Target dependency edges:

| Crate | May depend on | Must not depend on |
|-------|---------------|--------------------|
| `jolt-riscv` | `common`, `jolt-platform`, `serde`, `ark-serialize`, `strum`, `paste`, `derive_more` | `tracer`, `jolt-core`, `jolt-trace`, `jolt-bytecode-expand`, `object` |
| `jolt-program-image` | `jolt-riscv`, `common`, `object`, `thiserror`, `tracing` | `tracer`, `jolt-core`, `jolt-trace`, `jolt-bytecode-expand` |
| `jolt-bytecode-expand` | `jolt-riscv`, `common`, `thiserror`, optional `tracing` | `tracer`, `jolt-core`, `jolt-trace`, `object`, prover crates |
| `jolt-program-preprocess` | `jolt-riscv`, `jolt-bytecode-expand` only for helper constructors, `common`, `ark-serialize`, `blake2`, `thiserror` | `tracer`, `jolt-trace`, PCS implementations, Dory setup, prover-only modules |
| `tracer` | `jolt-riscv`, `jolt-program-image`, `jolt-bytecode-expand`, `common`, `jolt-platform` | `jolt-program-preprocess` unless trace APIs explicitly need program preprocessing |
| `jolt-core` | `jolt-riscv`, `jolt-program-preprocess`, proof-system crates, `common` | `tracer` for program preprocessing types |
| `jolt-trace` | `jolt-program-image`, `jolt-bytecode-expand`, `jolt-program-preprocess`, `tracer`, `common` | owning canonical expansion semantics |

The implementation PR must not introduce a new cycle. Its final state should remove the `tracer -> jolt-bytecode-expand -> tracer` cycle risk entirely by moving shared instruction data below both crates.

#### Workspace And Cargo Changes

Root `Cargo.toml` changes:

- Add workspace members:
  - `"crates/jolt-program-image"`
  - `"crates/jolt-bytecode-expand"`
  - `"crates/jolt-program-preprocess"`
- Add workspace dependencies:
  - `jolt-program-image = { path = "./crates/jolt-program-image", default-features = false }`
  - `jolt-bytecode-expand = { path = "./crates/jolt-bytecode-expand", default-features = false }`
  - `jolt-program-preprocess = { path = "./crates/jolt-program-preprocess", default-features = false }`
- Keep the existing `jolt-riscv = { path = "./crates/jolt-riscv", ... }` workspace dependency, but update its feature policy if needed so verifier-critical consumers can use it without pulling in `tracer`.

New and refactored crate `Cargo.toml` feature policy:

- default features should be minimal;
- `std` feature should gate file-system and allocation-heavy conveniences where possible;
- verifier-critical crates should compile without pulling in `tracer`;
- test fixtures can use `test-utils`, but production expansion should not depend on randomized generation.

Existing crate `Cargo.toml` updates:

- `tracer/Cargo.toml`: add `jolt-riscv`, `jolt-program-image`, and `jolt-bytecode-expand`.
- `crates/jolt-riscv/Cargo.toml`: remove the direct dependency on `tracer`; keep only lower-level dependencies needed by static instruction data, flags, decoding, and serialization.
- `crates/jolt-trace/Cargo.toml`: add `jolt-program-image`, `jolt-bytecode-expand`, and possibly `jolt-program-preprocess`; keep `tracer` for execution only.
- `jolt-core/Cargo.toml`: add `jolt-riscv` and `jolt-program-preprocess`; remove program-preprocessing dependence on `tracer` where feasible.
- `jolt-sdk/Cargo.toml` and `jolt-sdk/macros/Cargo.toml`: update imports only as needed for generated preprocessing functions.

The crate should provide explicit APIs along these lines:

```rust
pub fn expand_instruction(
    instruction: &Instruction,
    allocator: &mut ExpansionAllocator,
    xlen: Xlen,
) -> Result<Vec<Instruction>, ExpansionError>;

pub fn expand_program(
    instructions: impl IntoIterator<Item = Instruction>,
    xlen: Xlen,
) -> Result<Vec<Instruction>, ExpansionError>;
```

The exact type names can change during implementation, but the API should make state flow explicit. In particular, virtual-register allocation state should be visible as expansion state rather than hidden inside the tracer CPU.

The program preprocessing API should be explicit about its inputs:

```rust
pub struct DecodedProgramImage {
    pub instructions: Vec<Instruction>,
    pub memory_init: Vec<(u64, u8)>,
    pub program_end: u64,
    pub entry_address: u64,
    pub xlen: Xlen,
}

pub fn preprocess_program(
    expanded_bytecode: Vec<Instruction>,
    memory_init: Vec<(u64, u8)>,
    memory_layout: MemoryLayout,
    max_padded_trace_length: usize,
    entry_address: u64,
) -> Result<JoltProgramPreprocessing, PreprocessingError>;
```

The important design point is not these exact signatures; it is that program preprocessing should not have to import `tracer::emulator::Cpu`, `LazyTraceIterator`, advice-tape machinery, host `Program` build logic, PCS setup, or committed-bytecode derivation.

The implementation should also isolate formal-verification-friendly logic:

- prefer total functions returning `Result` over panics in core expansion paths,
- keep CPU, memory, ELF, prover, transcript, and device dependencies out of the expansion crate,
- keep macro-generated dispatch behind a narrow boundary owned by `jolt-riscv`, with `tracer` reusing the same source list for `Cycle` generation,
- avoid concurrency primitives in allocator state unless they are needed by the API,
- document every invariant needed by Lean/Hax/Aeneas models, especially recursion and metadata assignment.

#### Formal Verification Readiness

Hax/Aeneas/Lean extraction is a reach goal for follow-up work, not a required deliverable for the crate-split PR. The implementation should nevertheless index on extraction strongly enough that the new boundaries do not need to be redesigned later.

For this PR, "extractable or close to extractable" means:

- `jolt-bytecode-expand` core logic uses explicit input/output state and total functions where practical;
- allocator state is plain owned Rust data, not hidden behind `Arc<Mutex<_>>`, global state, interior mutability, or thread-local state;
- core expansion does not depend on file I/O, ELF parsing, logging, tracing spans, CPU execution, memory emulation, advice tapes, prover code, or transcript code;
- errors are represented with small enums instead of panics in normal control flow;
- recursive expansion has an explicit termination argument in comments and tests, such as "each recursive step expands into instructions from a strictly lower expansion class" or another reviewable measure;
- metadata assignment is centralized in one module so the Lean model can state and prove one theorem about `virtual_sequence_remaining` and `is_first_in_sequence`;
- APIs avoid trait objects, dynamic dispatch, async, macros that hide core semantics, and callback-heavy designs in the extraction-critical modules;
- any necessary macro-generated instruction dispatch expands to simple first-order functions that can be audited or extracted after expansion.

The right indexing is therefore medium-high:

- High for `jolt-bytecode-expand`, because bytecode expansion is the main formalization target.
- Medium for `jolt-program-image`, because ELF parsing is verifier-facing when preprocessing from ELF, but it may be acceptable to trust or separately validate object-file parsing before formalizing expansion.
- Medium for `jolt-program-preprocess`, because bytecode/RAM preprocessing and digest binding are verifier-critical and prover-consumed, but PCS setup and commitment derivation remain trusted proof-system boundaries outside this crate.
- Low for `jolt-trace` and `tracer` execution internals in this PR, because they should become consumers of the verifier-facing crates rather than extraction targets.

This PR should not contort the code to satisfy a specific extractor immediately. It should keep the extraction-critical core small, pure, dependency-light, and well-specified so a follow-up PR can run Hax/Aeneas and then address concrete tool feedback without reopening the architecture.

### Alternatives Considered

Keeping expansion in `tracer` and documenting it there is the smallest code change, but it leaves the formalization target coupled to CPU execution, memory, advice, and trace infrastructure.

Treating `jolt-trace` as the expansion crate is also tempting because it already contains bytecode preprocessing, but `jolt-trace` is host-facing and still builds guests, reads ELF files, invokes the tracer, and owns program APIs. That crate is useful to users, but it is not the minimal formalization target.

Moving only `BytecodePreprocessing` is insufficient because preprocessing consumes already-expanded bytecode. The correctness-sensitive part for formalization is the path that turns one decoded source instruction into zero, one, or many final bytecode rows.

Creating one large `jolt-verifier-input` crate for ELF decode, expansion, bytecode preprocessing, RAM preprocessing, and program preprocessing is workable but less ideal. It would be easier to integrate, but it would blur the boundary between pure instruction expansion and program preprocessing. A more modular split gives formal verification tools a smaller bytecode-expansion target while still giving the verifier a clean dependency path.

### Related Branches And Future Integration

This spec was checked against two ongoing branches, but neither branch expands the scope of this PR.

The `refactor/crates` branch is useful as a point of comparison for proof-system setup boundaries. Its `jolt-zkvm::preprocessing` work computes module/protocol shape and PCS setup for compiled modules; it does not extract ELF/program bytecode or RAM preprocessing out of `jolt-core`. This supports keeping `jolt-program-preprocess` focused on materialized program artifacts while leaving PCS setup and prover/verifier keys in `jolt-core` or a later proof-system setup crate.

The bytecode-commitment PR [#1344](https://github.com/a16z/jolt/pull/1344), branch `amir/bytecode-commitment-merged`, is useful as a point of comparison for future committed-program integration. That branch distinguishes full program preprocessing from committed program preprocessing: the prover still needs full bytecode, RAM/program image data, and opening hints, while the verifier should only need metadata plus trusted bytecode and program-image commitments. This PR should not absorb that committed path. Instead, `jolt-program-preprocess` should provide the materialized input that committed-program preprocessing can consume later. Commitment derivation, trusted commitments, Dory geometry, and opening hints remain outside this crate.

The future-compatible shape is therefore:

```text
jolt-program-preprocess
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
- which crates are verifier-critical and which are prover/host-only,
- how program preprocessing can be derived from serialized preprocessing or from ELF bytes,
- how decoded instructions become expanded bytecode rows,
- how `virtual_sequence_remaining` is assigned and consumed,
- how to add or modify an instruction expansion safely.

## Execution

### Files To Create

- `crates/jolt-riscv/src/xlen.rs`
- `crates/jolt-riscv/src/decode.rs`
- `crates/jolt-riscv/src/uncompress.rs`
- `crates/jolt-riscv/src/normalized.rs`
- `crates/jolt-riscv/src/operands.rs`
- `crates/jolt-riscv/src/instruction.rs`
- `crates/jolt-riscv/src/traits.rs`
- `crates/jolt-riscv/src/format/*.rs`
- `crates/jolt-program-image/Cargo.toml`
- `crates/jolt-program-image/src/lib.rs`
- `crates/jolt-program-image/src/elf.rs`
- `crates/jolt-program-image/src/error.rs`
- `crates/jolt-bytecode-expand/Cargo.toml`
- `crates/jolt-bytecode-expand/src/lib.rs`
- `crates/jolt-bytecode-expand/src/allocator.rs`
- `crates/jolt-bytecode-expand/src/assembler.rs`
- `crates/jolt-bytecode-expand/src/error.rs`
- `crates/jolt-bytecode-expand/src/expand.rs`
- `crates/jolt-bytecode-expand/src/metadata.rs`
- `crates/jolt-bytecode-expand/src/sequences/*.rs`
- `crates/jolt-program-preprocess/Cargo.toml`
- `crates/jolt-program-preprocess/src/lib.rs`
- `crates/jolt-program-preprocess/src/bytecode.rs`
- `crates/jolt-program-preprocess/src/ram.rs`
- `crates/jolt-program-preprocess/src/program.rs`
- `crates/jolt-program-preprocess/src/error.rs`
- `jolt-eval/src/invariant/bytecode_expansion.rs`
- `jolt-eval/benches/decode_expand.rs`

### Files To Modify

- `Cargo.toml`: add workspace members and workspace dependencies for the three new crates; keep the existing `jolt-riscv` workspace entry.
- `Cargo.lock`: update after adding workspace crates.
- `tracer/Cargo.toml`: depend on `jolt-riscv`, `jolt-program-image`, and `jolt-bytecode-expand`.
- `tracer/src/lib.rs`: remove ELF decode ownership; tracer-specific ELF entry points should call `jolt_program_image::decode_elf` internally only when they also perform tracing/execution work.
- `tracer/src/emulator/cpu.rs`: import `Xlen` and `Instruction` from `jolt-riscv`; call expansion APIs where trace length or trace-time `rd = x0` expansion is needed.
- `tracer/src/instruction/**`: move pure instruction definitions to `jolt-riscv`; leave execution-only methods in `tracer` as extension impls or execution modules.
- `tracer/src/utils/inline_helpers.rs`: move expansion helpers to `jolt-bytecode-expand`; delete or replace with reexports only if required during same full cutover.
- `tracer/src/utils/virtual_registers.rs`: move allocator to `jolt-bytecode-expand`; delete or replace imports.
- `crates/jolt-riscv/Cargo.toml`: remove `tracer` and add any lower-level dependencies needed by moved instruction data.
- `crates/jolt-riscv/src/lib.rs`: expose the canonical instruction list, `Instruction`, `NormalizedInstruction`, `Xlen`, pure traits, and flags without referencing `tracer`.
- `crates/jolt-riscv/src/instructions/**`: merge existing Jolt instruction kind wrappers with moved concrete instruction structs and pure instruction metadata.
- `crates/jolt-riscv/src/jolt_instruction.rs`: remove the blanket impl over `tracer::instruction::RISCVInstruction`; implement `JoltInstruction` over the new local static instruction representation.
- `crates/jolt-trace/Cargo.toml`: add new crates; keep `tracer` only for execution.
- `crates/jolt-trace/src/program.rs`: call `jolt-program-image` and `jolt-bytecode-expand` in `decode`.
- `crates/jolt-trace/src/bytecode.rs`: remove or delegate to `jolt-program-preprocess` if duplicate with `jolt-core` bytecode preprocessing.
- `jolt-core/Cargo.toml`: add `jolt-riscv` and `jolt-program-preprocess`.
- `jolt-core/src/zkvm/bytecode/mod.rs`: move or reexport materialized bytecode preprocessing from `jolt-program-preprocess`.
- `jolt-core/src/zkvm/ram/mod.rs`: move or reexport pure RAM preprocessing from `jolt-program-preprocess`; leave prover/verifier sumcheck modules in `jolt-core`.
- `jolt-core/src/zkvm/verifier.rs`: use program preprocessing types from `jolt-program-preprocess`, while keeping `JoltVerifierPreprocessing` if PCS setup remains in `jolt-core`.
- `jolt-core/src/zkvm/prover.rs`: update imports for program preprocessing, bytecode preprocessing, RAM preprocessing, and instruction types.
- `jolt-core/src/poly/**`, `jolt-core/src/subprotocols/**`, `jolt-core/src/zkvm/**`: update imports from `tracer::instruction` to `jolt-riscv` where the code only needs static instruction data.
- `jolt-sdk/macros/src/lib.rs`: generated preprocessing functions should call the modular decode/expand/program-preprocess path.
- `jolt-eval/src/invariant/mod.rs`: register bytecode expansion invariants.
- `jolt-eval/src/objective/mod.rs`: add `decode_expand` objective only if the benchmark is promoted to a measured objective in this PR.
- `book/**` or developer docs: add an architecture page or section if maintainers want crate-boundary docs in the book.

### Files To Remove Or Empty

Remove these only after all call sites are cut over:

- `tracer/src/utils/inline_helpers.rs`
- `tracer/src/utils/virtual_registers.rs`
- pure instruction-format files under `tracer/src/instruction/format/`
- pure per-instruction data/normalize/decode files under `tracer/src/instruction/`

Do not delete execution-specific instruction trace implementations unless they have been moved to a tracer execution module in the same PR.

### Implementation Checklist

1. Add empty `jolt-program-image`, `jolt-bytecode-expand`, and `jolt-program-preprocess` crates under `crates/`, workspace members, workspace dependencies, and minimal `lib.rs` files.
2. Refactor existing `jolt-riscv` so it no longer depends on `tracer`.
3. Move `Xlen`, normalized operands, instruction formats, `NormalizedInstruction`, and the `Instruction` enum into `jolt-riscv`.
4. Move opcode decode and compressed-instruction decompression into `jolt-riscv`.
5. Update `tracer`, `jolt-core`, and `jolt-trace` imports to compile against the strengthened `jolt-riscv`.
6. Add `jolt-program-image::decode_elf` by moving the non-execution logic from `tracer::decode`.
7. Update all call sites of `tracer::decode` that only need ELF decoding to call `jolt-program-image::decode_elf`; keep `tracer::decode` only if it remains a tracer-specific execution API rather than a compatibility shim.
8. Move `VirtualRegisterAllocator`, `InstrAssembler`, recursive `add_to_sequence`, and per-instruction inline expansion logic into `jolt-bytecode-expand`.
9. Implement `expand_instruction` and `expand_program`.
10. Update `tracer` trace-time expansion and trace length accounting to use `jolt-bytecode-expand`.
11. Update `jolt-trace::decode` to call `jolt-program-image::decode_elf` followed by `jolt-bytecode-expand::expand_program`.
12. Move `BytecodePreprocessing`, `BytecodePCMapper`, pure `RAMPreprocessing`, and `JoltProgramPreprocessing` into `jolt-program-preprocess`.
13. Update `jolt-core` prover/verifier code to consume the moved preprocessing types.
14. Update SDK macro-generated preprocessing to use the modular path.
15. Add expansion parity tests before removing old expansion entry points.
16. Add `jolt-eval` invariants for expansion fixture consistency, PC mapping consistency, and program preprocessing determinism.
17. Add the decode-plus-expansion Criterion benchmark under `jolt-eval`.
18. Remove old canonical expansion ownership from `tracer`.
19. Run formatting, clippy, host tests, ZK tests, and targeted crate dependency checks.

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
cargo nextest run -p jolt-bytecode-expand --cargo-quiet
cargo nextest run -p jolt-program-image --cargo-quiet
cargo nextest run -p jolt-program-preprocess --cargo-quiet
cargo run -p jolt-eval --bin measure-objectives -- --no-bench
cargo bench -p jolt-eval --bench decode_expand -- --quick
cargo tree -p tracer
cargo tree -p jolt-riscv
cargo tree -p jolt-bytecode-expand
```

The dependency checks should confirm that `jolt-riscv`, `jolt-bytecode-expand`, `jolt-program-image`, and `jolt-program-preprocess` do not depend on `tracer`.

## References

- `crates/jolt-trace/src/program.rs`: current `decode` path calls `tracer::decode` and then expands with `Instruction::inline_sequence`.
- `jolt-core/src/zkvm/verifier.rs`: current home of `JoltSharedPreprocessing` and `JoltVerifierPreprocessing`; the program/shared layer should move to `jolt-program-preprocess`, while proof-system setup can remain in `jolt-core`.
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
