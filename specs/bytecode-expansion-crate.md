# Spec: Bytecode Expansion Crate

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao                      |
| Created     | 2026-05-01                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Jolt's bytecode expansion logic is still owned by the `tracer` crate even though `jolt-trace` now exposes host-facing program decoding, tracing, and bytecode preprocessing APIs. This spec proposes a dedicated bytecode expansion crate and asks whether verifier-critical preprocessing should be split into additional small crates in the same refactor. The target is a modular pipeline from ELF bytes to expanded bytecode and verifier preprocessing, while keeping CPU execution, memory-device emulation, advice I/O, and trace production out of the verifier-facing dependency graph.

## Intent

### Goal

Introduce a `jolt-bytecode-expand` crate that owns expansion from decoded RISC-V instructions into Jolt bytecode rows, including recursive inline expansion, virtual register allocation, and bytecode-row metadata, while making `tracer`, `jolt-trace`, and shared verifier preprocessing consumers of that crate instead of owners of expansion semantics.

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
- Verifier preprocessing built from an ELF must be deterministic and must match preprocessing derived through the existing prover path.

Add concrete `jolt-eval` invariants for bytecode expansion parity and verifier preprocessing determinism. These invariants should live under `jolt-eval/src/invariant/` and be wired into `JoltInvariants::all()`.

### Non-Goals

- This spec does not propose changing Jolt's instruction semantics.
- This spec does not propose changing ELF decoding, guest compilation, memory initialization, or trace execution beyond redirecting call sites to the new expansion crate.
- This spec does not require completing Lean, Hax, or Aeneas extraction in the same implementation PR.
- This spec does not require redesigning the bytecode commitment, lookup tables, or prover constraints.
- This spec does not require exposing a stable public API for third-party consumers outside the Jolt workspace.
- This spec does not require moving CPU execution, lazy trace iteration, advice tapes, or memory-device emulation into verifier-facing crates.

## Evaluation

### Acceptance Criteria

- [ ] `crates/jolt-riscv-instructions` exists, is a workspace member, and owns shared instruction data types currently required by decoding, expansion, tracing, bytecode preprocessing, and verifier checks.
- [ ] `crates/jolt-program-image` exists, is a workspace member, and exposes deterministic ELF decoding into a `DecodedProgramImage` without depending on `tracer`.
- [ ] `crates/jolt-bytecode-expand` exists, is a workspace member, and provides a pure expansion API over decoded instructions and `Xlen`.
- [ ] `crates/jolt-verifier-preprocess` exists, is a workspace member, and owns verifier-critical bytecode/RAM/shared preprocessing artifacts or explicitly documents any artifacts left in `jolt-core` for this PR.
- [ ] `tracer` no longer owns the canonical recursive bytecode expansion algorithm; it calls `jolt-bytecode-expand` for expansion during trace execution and trace length accounting.
- [ ] `jolt-trace::decode` no longer performs expansion by directly calling `Instruction::inline_sequence` from `tracer`; it calls `jolt-program-image` and `jolt-bytecode-expand`.
- [ ] `InstrAssembler`, `VirtualRegisterAllocator`, or their minimal expansion-facing equivalents live behind the `jolt-bytecode-expand` crate boundary.
- [ ] Expansion APIs return explicit errors for allocation or malformed-expansion failures instead of introducing new panics in the core expansion path.
- [ ] A verifier-preprocessing path is implemented as `ELF bytes -> DecodedProgramImage -> expanded bytecode -> shared preprocessing -> verifier preprocessing`.
- [ ] The verifier-facing path does not depend on CPU execution, lazy tracing, memory-device emulation, advice I/O, or prover-only witness generation.
- [ ] `cargo tree -p tracer` shows `tracer` depending on the new lower-level crates, but none of those lower-level crates depending back on `tracer`.
- [ ] `cargo tree -p jolt-core --features host` does not require `tracer` solely to name verifier/shared preprocessing or bytecode rows, except for any temporary dependency explicitly justified in the PR.
- [ ] Golden tests prove byte-for-byte or structure-for-structure parity with the current expansion output for supported RV32 and RV64 instruction families.
- [ ] Tests cover recursive expansion, `rd = x0`, virtual-register clearing, compressed source instructions, and bytecode PC mapping.
- [ ] The crate dependency surface is suitable for future formal verification and extraction work: no CPU, memory-device, prover, transcript, or ELF parser dependency in `jolt-bytecode-expand`.
- [ ] Verifier-facing crates document any deliberate choices that make Hax/Aeneas/Lean extraction harder, so those choices can be revisited in a follow-up rather than discovered after the split.

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
3. Generate checked-in golden fixtures from the old behavior for a curated corpus of decoded instructions and small ELF programs. The fixtures should serialize normalized expanded bytecode rows, not raw debug strings.
4. Cut all production call sites over to the new crate.
5. Delete the old production expansion implementation from `tracer`.
6. Keep the golden fixture tests and property/invariant tests after deletion so CI continues to guard the new implementation without requiring the old implementation to remain in production.

Do not leave the old expansion implementation as a compatibility shim. A small test-only reference module may be used during implementation if it makes the transition safer, but the final merged production code should have one canonical expansion implementation.

Add `jolt-eval` invariants:

- `bytecode_expansion_parity`: for generated or fixture decoded instructions, compare the old expansion path to the new crate output, including instruction variants, normalized operands, flags, addresses, compressed metadata, `is_first_in_sequence`, and `virtual_sequence_remaining`.
- `verifier_preprocessing_determinism`: for small fixture ELFs or generated guest programs, build shared verifier preprocessing through the old macro/host path and through the new modular path, then compare the serialized `JoltSharedPreprocessing::digest()`.
- `bytecode_pc_mapping_consistency`: for expanded bytecode with inline sequences, assert that `BytecodePCMapper::get_pc(address, virtual_sequence_remaining)` returns the same dense indices before and after the refactor.

After the full cutover, keep golden fixture tests or snapshot tests in the new crate so future changes to expansion semantics are intentional and reviewable.

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

Current verifier preprocessing architecture:

```text
program.decode()
  -> expanded bytecode
  -> memory_init bytes
  -> program_size
  -> ELF entry address

JoltSharedPreprocessing::new(...)
  -> BytecodePreprocessing::preprocess(expanded bytecode, entry address)
  -> RAMPreprocessing::preprocess(memory_init)
  -> MemoryLayout
  -> max_padded_trace_length

JoltVerifierPreprocessing::new(...)
  -> JoltSharedPreprocessing
  -> PCS verifier setup
  -> optional BlindFold setup
```

At verification time, `JoltVerifier` consumes `JoltVerifierPreprocessing`; it does not need the original ELF, CPU execution, memory-device emulation, advice tapes, lazy trace iteration, or prover witness generation. However, if a user wants to derive verifier preprocessing from an ELF, the verifier-critical construction path needs deterministic ELF decoding, bytecode expansion, bytecode preprocessing, RAM preprocessing, memory layout, and preprocessing serialization/digesting.

Proposed architecture:

```text
jolt-program-image
  -> decode_elf(elf)
  -> decoded instructions, memory init, program end, entry address, xlen

jolt-bytecode-expand
  -> expand_instruction(decoded_instruction, allocator, xlen)
  -> expand_program(decoded_instructions, xlen)
  -> expansion metadata and errors

jolt-verifier-preprocess
  -> BytecodePreprocessing
  -> BytecodePCMapper
  -> RAMPreprocessing
  -> JoltSharedPreprocessing or a lower-level shared-preprocessing equivalent

jolt-trace
  -> builds guest ELFs
  -> calls jolt-program-image and jolt-bytecode-expand
  -> calls jolt-verifier-preprocess when shared preprocessing is needed
  -> owns tracing-oriented host convenience APIs

tracer
  -> executes instructions and produces traces
  -> calls jolt-bytecode-expand when trace-time expansion is required

jolt-core verifier
  -> consumes JoltVerifierPreprocessing
  -> depends on verifier preprocessing artifacts, not on tracer execution
```

The new crate should own the expansion-facing abstractions, not the full tracer. The most conservative first version can still reuse the existing instruction representation, but the dependency direction should move toward:

```text
instruction representation / common types
        ^
        |
jolt-program-image      jolt-bytecode-expand
        ^                       ^
        |                       |
jolt-verifier-preprocess <------+
        ^
        |
jolt-core verifier, prover, jolt-trace
```

If the full split is too large for one PR, the implementation should still make the intended boundaries explicit and avoid further coupling. A minimum acceptable first pass is:

```text
instruction representation / common types
        ^
        |
jolt-bytecode-expand
        ^
        |
   tracer, jolt-trace
```

The broader crate split should be evaluated in the same design pass:

| Concern | Verifier needs it? | Proposed home |
|---------|--------------------|---------------|
| ELF parsing into decoded instructions, memory image, entry address, program end, and `Xlen` | Only when deriving preprocessing from ELF; not during proof verification from serialized preprocessing | `jolt-program-image` or similarly small decode crate |
| Recursive bytecode expansion | Yes, for verifier preprocessing from ELF; not during verification from already-serialized preprocessing | `jolt-bytecode-expand` |
| Bytecode table padding and PC mapping | Yes; verifier uses bytecode code size, bytecode rows, entry bytecode index, and PC mapping-derived commitments/checks | `jolt-verifier-preprocess` or a `jolt-bytecode` preprocessing crate |
| RAM preprocessing from ELF memory bytes | Yes; verifier checks RAM initialization claims against `RAMPreprocessing` | `jolt-verifier-preprocess` or `jolt-memory-preprocess` |
| Memory layout from program size and I/O limits | Yes; verifier validates public I/O sizes and RAM bounds | `common` plus shared preprocessing crate |
| Preprocessing digest and serialization | Yes; verifier binds preprocessing to Fiat-Shamir transcript | shared verifier preprocessing crate |
| PCS verifier setup and optional BlindFold setup | Yes, but these are proof-system setup artifacts rather than bytecode-expansion artifacts | stay in `jolt-core` verifier or a proof-system setup crate |
| CPU execution and trace production | No | `tracer` |
| Memory-device emulation and I/O side effects | No, except the verifier consumes public `JoltDevice`/memory layout data supplied with the proof API | `tracer`/host APIs plus `common` data types |
| Advice tape I/O | No for verification, except commitments/public inputs already represented in proof and verifier inputs | prover/host tracing crates |
| Prover witness generation and prover-facing polynomial preprocessing | No | `jolt-core` prover |

#### Crate Placement And Dependency Direction

New library crates should live under `crates/` and be added to the root workspace `members` and `[workspace.dependencies]`, matching the newer crate layout used by `jolt-trace`, `jolt-riscv`, `jolt-openings`, and related libraries. Existing top-level crates such as `tracer`, `common`, `jolt-core`, and `jolt-sdk` can depend on those `crates/*` libraries through workspace dependencies.

It is acceptable for `tracer` to import new `crates/*` libraries, but only if those libraries do not depend back on `tracer`. The current code does not yet satisfy that shape: the concrete `Instruction`, `Cycle`, `RISCVInstruction`, `NormalizedInstruction`, and per-instruction structs live in `tracer`, and `jolt-riscv` currently depends on `tracer` for those types. Therefore a new `jolt-bytecode-expand` crate cannot both depend on `tracer::instruction::*` and be imported by `tracer` without creating a dependency cycle.

The implementation should resolve this by first moving the shared instruction representation, normalized instruction view, instruction flags, and decode-facing types into a lower-level crate. Two plausible homes are:

- a strengthened `jolt-riscv` crate, if it becomes the owner of instruction data types instead of wrapping `tracer` types;
- a new `jolt-riscv-instructions` or `jolt-instruction` crate, if keeping `jolt-riscv` focused on Jolt circuit-facing flags is cleaner.

After that, the intended dependency direction is:

```text
common / jolt-platform
        ^
        |
jolt-riscv-instructions
        ^
        |
jolt-program-image      jolt-bytecode-expand
        ^                       ^
        |                       |
jolt-verifier-preprocess <------+
        ^                       ^
        |                       |
jolt-core, jolt-trace, tracer --+
```

This direction keeps `tracer` as an execution engine while allowing it to reuse the same bytecode expansion crate as verifier preprocessing. It also gives formal verification tools a target that is not downstream of the emulator.

#### Proposed Workspace Layout

The implementation should introduce these crates under `crates/`:

```text
crates/
  jolt-riscv-instructions/
    Cargo.toml
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

  jolt-verifier-preprocess/
    Cargo.toml
    src/
      lib.rs
      bytecode.rs
      ram.rs
      shared.rs
      error.rs
```

The file list is intentionally concrete. Implementers may split individual instruction families differently if the resulting crate boundary and dependency graph are the same, but the PR should not leave expansion logic scattered across `tracer`.

#### Crate Responsibilities

`crates/jolt-riscv-instructions` owns the data model that must be shared by decoding, expansion, tracing, bytecode preprocessing, and verifier checks:

- `src/xlen.rs`: move `Xlen` out of `tracer::emulator::cpu` so decoders and expanders do not import CPU execution.
- `src/operands.rs`: own `NormalizedOperands` and operand accessors.
- `src/normalized.rs`: own `NormalizedInstruction`.
- `src/format/*.rs`: move instruction format structs from `tracer/src/instruction/format/`.
- `src/instructions/*.rs`: move concrete instruction structs and pure instruction metadata from `tracer/src/instruction/`.
- `src/instruction.rs`: own the `Instruction` enum, `NoOp`, `UNIMPL`, normalize/with-rd/set-metadata helpers, side-effect metadata, and variant dispatch that does not require CPU execution.
- `src/decode.rs`: move `Instruction::decode` and opcode decoding.
- `src/uncompress.rs`: move compressed RISC-V decompression used by ELF decode.
- `src/traits.rs`: define pure traits such as `RiscvInstruction`, `Normalize`, `HasSideEffects`, and metadata mutation traits. Execution-specific traits must remain outside this crate.

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

`crates/jolt-verifier-preprocess` owns preprocessing artifacts used by both prover and verifier:

- `src/bytecode.rs`: move `BytecodePreprocessing`, `BytecodePCMapper`, and bytecode preprocessing errors from `jolt-core/src/zkvm/bytecode/mod.rs`.
- `src/ram.rs`: move `RAMPreprocessing`, `compute_min_ram_K`, and pure RAM initialization helpers from `jolt-core/src/zkvm/ram/mod.rs` if doing so does not drag prover-only modules with it.
- `src/shared.rs`: move or wrap `JoltSharedPreprocessing`, including canonical serialization and `digest()`.
- `src/error.rs`: consolidate bytecode/RAM/shared preprocessing errors.

`tracer` remains the execution engine:

- keeps `Cpu`, `Memory`, `LazyTraceIterator`, `Cycle`, execution semantics, advice tape plumbing, trace-to-file support, and `jolt-emu`;
- imports `jolt-riscv-instructions` for instruction data;
- imports `jolt-bytecode-expand` for trace-time expansion and trace length accounting;
- imports `jolt-program-image` for ELF decode where tracing from ELF is needed.

`jolt-core` remains the proof system:

- imports verifier preprocessing artifacts from `jolt-verifier-preprocess`;
- keeps prover/verifier protocols, sumchecks, commitments, witness generation, and proof serialization;
- should not import `tracer` just to name bytecode preprocessing artifacts.

`jolt-trace` remains a host-facing convenience crate:

- owns `Program`, guest build helpers, analyze helpers, and user-facing trace APIs;
- calls `jolt-program-image::decode_elf`, `jolt-bytecode-expand::expand_program`, and `jolt-verifier-preprocess` instead of directly coupling expansion to `tracer`.

#### Dependency Table

Target dependency edges:

| Crate | May depend on | Must not depend on |
|-------|---------------|--------------------|
| `jolt-riscv-instructions` | `common`, `jolt-platform`, `serde`, `ark-serialize`, `strum`, `paste`, `derive_more` | `tracer`, `jolt-core`, `jolt-trace`, `jolt-bytecode-expand`, `object` |
| `jolt-program-image` | `jolt-riscv-instructions`, `common`, `object`, `thiserror`, `tracing` | `tracer`, `jolt-core`, `jolt-trace`, `jolt-bytecode-expand` |
| `jolt-bytecode-expand` | `jolt-riscv-instructions`, `common`, `thiserror`, optional `tracing` | `tracer`, `jolt-core`, `jolt-trace`, `object`, prover crates |
| `jolt-verifier-preprocess` | `jolt-riscv-instructions`, `jolt-bytecode-expand` only for helper constructors, `common`, `ark-serialize`, `blake2`, `thiserror` | `tracer`, `jolt-trace`, prover-only modules |
| `tracer` | `jolt-riscv-instructions`, `jolt-program-image`, `jolt-bytecode-expand`, `common`, `jolt-platform` | `jolt-verifier-preprocess` unless trace APIs explicitly need shared preprocessing |
| `jolt-riscv` | `jolt-riscv-instructions` | `tracer` after the full cutover |
| `jolt-core` | `jolt-riscv-instructions`, `jolt-verifier-preprocess`, proof-system crates, `common` | `tracer` for verifier/shared preprocessing types |
| `jolt-trace` | `jolt-program-image`, `jolt-bytecode-expand`, `jolt-verifier-preprocess`, `tracer`, `common` | owning canonical expansion semantics |

The first implementation PR may temporarily leave some dependency edges in their old shape while moving files, but it must not introduce a new cycle. The final state of the implementation PR should remove the `tracer -> jolt-bytecode-expand -> tracer` cycle risk entirely.

#### Workspace And Cargo Changes

Root `Cargo.toml` changes:

- Add workspace members:
  - `"crates/jolt-riscv-instructions"`
  - `"crates/jolt-program-image"`
  - `"crates/jolt-bytecode-expand"`
  - `"crates/jolt-verifier-preprocess"`
- Add workspace dependencies:
  - `jolt-riscv-instructions = { path = "./crates/jolt-riscv-instructions", default-features = false }`
  - `jolt-program-image = { path = "./crates/jolt-program-image", default-features = false }`
  - `jolt-bytecode-expand = { path = "./crates/jolt-bytecode-expand", default-features = false }`
  - `jolt-verifier-preprocess = { path = "./crates/jolt-verifier-preprocess", default-features = false }`

New crate `Cargo.toml` feature policy:

- default features should be minimal;
- `std` feature should gate file-system and allocation-heavy conveniences where possible;
- verifier-critical crates should compile without pulling in `tracer`;
- test fixtures can use `test-utils`, but production expansion should not depend on randomized generation.

Existing crate `Cargo.toml` updates:

- `tracer/Cargo.toml`: add `jolt-riscv-instructions`, `jolt-program-image`, and `jolt-bytecode-expand`.
- `crates/jolt-riscv/Cargo.toml`: replace direct dependency on `tracer` with `jolt-riscv-instructions` after cutover.
- `crates/jolt-trace/Cargo.toml`: add `jolt-program-image`, `jolt-bytecode-expand`, and possibly `jolt-verifier-preprocess`; keep `tracer` for execution only.
- `jolt-core/Cargo.toml`: add `jolt-riscv-instructions` and `jolt-verifier-preprocess`; remove verifier/shared-preprocessing dependence on `tracer` where feasible.
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

The shared preprocessing API should be explicit about its inputs:

```rust
pub struct DecodedProgramImage {
    pub instructions: Vec<Instruction>,
    pub memory_init: Vec<(u64, u8)>,
    pub program_end: u64,
    pub entry_address: u64,
    pub xlen: Xlen,
}

pub fn preprocess_for_verifier(
    expanded_bytecode: Vec<Instruction>,
    memory_init: Vec<(u64, u8)>,
    memory_layout: MemoryLayout,
    max_padded_trace_length: usize,
    entry_address: u64,
) -> Result<JoltSharedPreprocessing, PreprocessingError>;
```

The important design point is not these exact signatures; it is that verifier preprocessing should not have to import `tracer::emulator::Cpu`, `LazyTraceIterator`, advice-tape machinery, or host `Program` build logic.

The implementation should also isolate formal-verification-friendly logic:

- prefer total functions returning `Result` over panics in core expansion paths,
- keep CPU, memory, ELF, prover, transcript, and device dependencies out of the expansion crate,
- keep macro-generated dispatch behind a narrow boundary, or generate simple first-order functions that tools can inspect,
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
- Medium for `jolt-verifier-preprocess`, because bytecode/RAM preprocessing and digest binding are verifier-critical, but some serialization and cryptographic setup boundaries may remain trusted.
- Low for `jolt-trace` and `tracer` execution internals in this PR, because they should become consumers of the verifier-facing crates rather than extraction targets.

This PR should not contort the code to satisfy a specific extractor immediately. It should keep the extraction-critical core small, pure, dependency-light, and well-specified so a follow-up PR can run Hax/Aeneas and then address concrete tool feedback without reopening the architecture.

### Alternatives Considered

Keeping expansion in `tracer` and documenting it there is the smallest code change, but it leaves the formalization target coupled to CPU execution, memory, advice, and trace infrastructure.

Treating `jolt-trace` as the expansion crate is also tempting because it already contains bytecode preprocessing, but `jolt-trace` is host-facing and still builds guests, reads ELF files, invokes the tracer, and owns program APIs. That crate is useful to users, but it is not the minimal formalization target.

Moving only `BytecodePreprocessing` is insufficient because preprocessing consumes already-expanded bytecode. The correctness-sensitive part for formalization is the path that turns one decoded source instruction into zero, one, or many final bytecode rows.

Creating one large `jolt-verifier-input` crate for ELF decode, expansion, bytecode preprocessing, RAM preprocessing, and shared preprocessing is workable but less ideal. It would be easier to integrate, but it would blur the boundary between pure instruction expansion and verifier preprocessing. A more modular split gives formal verification tools a smaller bytecode-expansion target while still giving the verifier a clean dependency path.

## Documentation

Update the Jolt book only if the crate is exposed to users or changes contributor-facing architecture. At minimum, add developer documentation describing:

- where bytecode expansion lives,
- which crates are verifier-critical and which are prover/host-only,
- how verifier preprocessing can be derived from serialized preprocessing or from ELF bytes,
- how decoded instructions become expanded bytecode rows,
- how `virtual_sequence_remaining` is assigned and consumed,
- how to add or modify an instruction expansion safely.

## Execution

### Files To Create

- `crates/jolt-riscv-instructions/Cargo.toml`
- `crates/jolt-riscv-instructions/src/lib.rs`
- `crates/jolt-riscv-instructions/src/xlen.rs`
- `crates/jolt-riscv-instructions/src/decode.rs`
- `crates/jolt-riscv-instructions/src/uncompress.rs`
- `crates/jolt-riscv-instructions/src/normalized.rs`
- `crates/jolt-riscv-instructions/src/operands.rs`
- `crates/jolt-riscv-instructions/src/instruction.rs`
- `crates/jolt-riscv-instructions/src/traits.rs`
- `crates/jolt-riscv-instructions/src/format/*.rs`
- `crates/jolt-riscv-instructions/src/instructions/*.rs`
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
- `crates/jolt-verifier-preprocess/Cargo.toml`
- `crates/jolt-verifier-preprocess/src/lib.rs`
- `crates/jolt-verifier-preprocess/src/bytecode.rs`
- `crates/jolt-verifier-preprocess/src/ram.rs`
- `crates/jolt-verifier-preprocess/src/shared.rs`
- `crates/jolt-verifier-preprocess/src/error.rs`
- `jolt-eval/src/invariant/bytecode_expansion.rs`
- `jolt-eval/benches/decode_expand.rs`

### Files To Modify

- `Cargo.toml`: add workspace members and workspace dependencies for the new crates.
- `Cargo.lock`: update after adding workspace crates.
- `tracer/Cargo.toml`: depend on `jolt-riscv-instructions`, `jolt-program-image`, and `jolt-bytecode-expand`.
- `tracer/src/lib.rs`: remove ELF decode ownership; tracer-specific ELF entry points should call `jolt_program_image::decode_elf` internally only when they also perform tracing/execution work.
- `tracer/src/emulator/cpu.rs`: import `Xlen` and `Instruction` from `jolt-riscv-instructions`; call expansion APIs where trace length or trace-time `rd = x0` expansion is needed.
- `tracer/src/instruction/**`: move pure instruction definitions to `jolt-riscv-instructions`; leave execution-only methods in `tracer` as extension impls or execution modules.
- `tracer/src/utils/inline_helpers.rs`: move expansion helpers to `jolt-bytecode-expand`; delete or replace with reexports only if required during same full cutover.
- `tracer/src/utils/virtual_registers.rs`: move allocator to `jolt-bytecode-expand`; delete or replace imports.
- `crates/jolt-riscv/Cargo.toml`: depend on `jolt-riscv-instructions` instead of `tracer` after instruction-type cutover.
- `crates/jolt-riscv/src/**`: update trait impls and wrappers to use instruction types from `jolt-riscv-instructions`.
- `crates/jolt-trace/Cargo.toml`: add new crates; keep `tracer` only for execution.
- `crates/jolt-trace/src/program.rs`: call `jolt-program-image` and `jolt-bytecode-expand` in `decode`.
- `crates/jolt-trace/src/bytecode.rs`: remove or delegate to `jolt-verifier-preprocess` if duplicate with `jolt-core` bytecode preprocessing.
- `jolt-core/Cargo.toml`: add `jolt-riscv-instructions` and `jolt-verifier-preprocess`.
- `jolt-core/src/zkvm/bytecode/mod.rs`: move or reexport verifier bytecode preprocessing from `jolt-verifier-preprocess`.
- `jolt-core/src/zkvm/ram/mod.rs`: move or reexport pure RAM preprocessing from `jolt-verifier-preprocess`; leave prover/verifier sumcheck modules in `jolt-core`.
- `jolt-core/src/zkvm/verifier.rs`: use shared preprocessing types from `jolt-verifier-preprocess`, while keeping `JoltVerifierPreprocessing` if PCS setup remains in `jolt-core`.
- `jolt-core/src/zkvm/prover.rs`: update imports for shared preprocessing, bytecode preprocessing, RAM preprocessing, and instruction types.
- `jolt-core/src/poly/**`, `jolt-core/src/subprotocols/**`, `jolt-core/src/zkvm/**`: update imports from `tracer::instruction` to `jolt-riscv-instructions` where the code only needs static instruction data.
- `jolt-sdk/macros/src/lib.rs`: generated `preprocess_shared_*` should call the modular decode/expand/preprocess path.
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

1. Add empty new crates under `crates/`, workspace members, workspace dependencies, and minimal `lib.rs` files.
2. Move `Xlen`, normalized operands, instruction formats, `NormalizedInstruction`, and the `Instruction` enum into `jolt-riscv-instructions`.
3. Move opcode decode and compressed-instruction decompression into `jolt-riscv-instructions`.
4. Update `tracer`, `jolt-riscv`, `jolt-core`, and `jolt-trace` imports to compile against `jolt-riscv-instructions`.
5. Add `jolt-program-image::decode_elf` by moving the non-execution logic from `tracer::decode`.
6. Update all call sites of `tracer::decode` that only need ELF decoding to call `jolt-program-image::decode_elf`; keep `tracer::decode` only if it remains a tracer-specific execution API rather than a compatibility shim.
7. Move `VirtualRegisterAllocator`, `InstrAssembler`, recursive `add_to_sequence`, and per-instruction inline expansion logic into `jolt-bytecode-expand`.
8. Implement `expand_instruction` and `expand_program`.
9. Update `tracer` trace-time expansion and trace length accounting to use `jolt-bytecode-expand`.
10. Update `jolt-trace::decode` to call `jolt-program-image::decode_elf` followed by `jolt-bytecode-expand::expand_program`.
11. Move `BytecodePreprocessing`, `BytecodePCMapper`, pure `RAMPreprocessing`, and `JoltSharedPreprocessing` or a shared-preprocessing equivalent into `jolt-verifier-preprocess`.
12. Update `jolt-core` prover/verifier code to consume the moved preprocessing types.
13. Update SDK macro-generated preprocessing to use the modular path.
14. Add expansion parity tests before removing old expansion entry points.
15. Add `jolt-eval` invariants for expansion parity, PC mapping consistency, and shared preprocessing determinism.
16. Add the decode-plus-expansion Criterion benchmark under `jolt-eval`.
17. Remove old canonical expansion ownership from `tracer`.
18. Run formatting, clippy, host tests, ZK tests, and targeted crate dependency checks.

### Verification Commands

The implementation PR should run:

```bash
cargo fmt -q
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo nextest run --cargo-quiet
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
cargo nextest run -p jolt-bytecode-expand --cargo-quiet
cargo nextest run -p jolt-program-image --cargo-quiet
cargo nextest run -p jolt-verifier-preprocess --cargo-quiet
cargo run -p jolt-eval --bin measure-objectives -- --no-bench
cargo bench -p jolt-eval --bench decode_expand -- --quick
cargo tree -p tracer
cargo tree -p jolt-bytecode-expand
```

The dependency checks should confirm that `jolt-bytecode-expand`, `jolt-program-image`, and `jolt-verifier-preprocess` do not depend on `tracer`.

## References

- `crates/jolt-trace/src/program.rs`: current `decode` path calls `tracer::decode` and then expands with `Instruction::inline_sequence`.
- `jolt-core/src/zkvm/verifier.rs`: `JoltSharedPreprocessing` and `JoltVerifierPreprocessing` are the verifier-facing preprocessing artifacts.
- `jolt-core/src/zkvm/bytecode/mod.rs`: bytecode preprocessing and PC mapping consume expanded `Instruction` rows.
- `jolt-core/src/zkvm/ram/mod.rs`: RAM preprocessing consumes ELF memory initialization bytes.
- `jolt-sdk/macros/src/lib.rs`: generated shared preprocessing calls `program.decode()` before constructing `JoltSharedPreprocessing`.
- `tracer/src/instruction/mod.rs`: `Instruction::inline_sequence` handles `rd = x0`, dispatch, and recursive expansion entry points.
- `tracer/src/utils/inline_helpers.rs`: `InstrAssembler` recursively expands helper-emitted instructions through `add_to_sequence`.
- `tracer/src/utils/virtual_registers.rs`: virtual register layout, allocator state, and inline clearing behavior.
- `tracer/src/lib.rs`: current ELF decode extracts instructions, memory bytes, program end, entry address, and `Xlen`.
- `jolt-eval/README.md`: describes invariant and objective hooks to add for this refactor.
- PR #1369: added `jolt-trace`, which is useful host-facing trace structure but does not by itself isolate bytecode expansion.
- PR #1260: broader crate refactor discussion relevant to dependency direction and crate boundaries.
