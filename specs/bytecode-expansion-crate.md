# Spec: `jolt-program` Crate

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao                      |
| Created     | 2026-05-01                     |
| Status      | in review                      |
| PR          | [#1490](https://github.com/a16z/jolt/pull/1490) |

## Summary

Jolt's bytecode expansion logic is still owned by the `tracer` crate even though `jolt-trace` now exposes host-facing program decoding, tracing, and bytecode preprocessing APIs. This spec proposes strengthening the existing `jolt-riscv` crate as the shared instruction vocabulary and normalized-row crate, plus one new `jolt-program` crate that owns program-image decoding, bytecode expansion, materialized Jolt program preprocessing, and the backend-neutral execution contract behind separate internal modules.

The target is a modular program-construction pipeline from ELF bytes to expanded bytecode and program preprocessing, plus a stable trait boundary that lets host/SDK code invoke a tracer without making lower `jolt-*` crates depend on tracer internals. CPU execution, memory-device emulation, advice I/O, concrete trace production, PCS setup, and commitment derivation stay out of the verifier-facing program dependency graph.

## Intent

### Goal

Introduce a `jolt-program` crate with three program-construction boundaries plus one backend-neutral execution boundary:

- `jolt_program::image`: deterministic program-image decoding from ELF bytes.
- `jolt_program::expand`: expansion from decoded RISC-V instructions into final Jolt bytecode rows.
- `jolt_program::preprocess`: materialized bytecode, RAM, and program preprocessing artifacts consumed by prover and verifier setup.
- `jolt_program::execution`: stable tracing input/output traits and data rows implemented by concrete execution backends such as `tracer`.

The new crate should own the path:

```text
ELF bytes
  -> DecodedProgramImage
  -> expanded bytecode
  -> JoltProgramPreprocessing
```

`tracer`, `jolt-trace`, `jolt-sdk`, and `jolt-core` should become consumers of these program-construction APIs instead of owners of bytecode expansion or materialized program preprocessing semantics. `tracer` should also become an implementation of the execution backend trait rather than a crate whose concrete internals leak into `jolt-program`, `jolt-riscv`, or proof-system APIs.

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

Keep the implementation PR's new checks focused on targeted `jolt-program`, `tracer`, and end-to-end parity tests. `jolt-eval` invariants for expansion fixture consistency and program preprocessing determinism are useful follow-up work, but are intentionally deferred so this refactor does not add extra evaluation scaffolding beyond the crate split itself.

### Non-Goals

- This spec does not propose changing Jolt's instruction semantics.
- This spec does not propose changing guest compilation, memory initialization semantics, or trace execution beyond redirecting program call sites to the new program crate and putting execution behind a stable backend trait.
- This spec does not require completing Lean, Hax, or Aeneas extraction in the same implementation PR.
- This spec does not require redesigning bytecode commitments, lookup tables, or prover constraints.
- This spec does not require moving PCS setup, commitment derivation, bytecode/program-image opening hints, BlindFold setup, `JoltProverPreprocessing`, or `JoltVerifierPreprocessing` out of `jolt-core`.
- This spec does not require integrating bytecode-commitment PR [#1344](https://github.com/a16z/jolt/pull/1344). That work is a future integration constraint, not part of this implementation scope.
- This spec does not require supporting RV32 in `jolt-program`; RV32/ELF32 should be rejected by the new program pipeline.
- This spec does not require deleting historical RV32 branches from `tracer`; that cleanup can happen separately.
- TODO(tracer RV32 cleanup): when RV32 support is fully removed from `tracer`, update stale comments and docs that still describe `Xlen`, RV32, ELF32, or dual-width expansion behavior as live supported paths.
- TODO(ISA profiles): keep `InstructionKind` as the canonical flat row/discriminant for bytecode, serialization, and proof indexing, but move the source declaration behind it to a hierarchical instruction-family/capability table. That table should generate the flat enum plus metadata such as `instruction_family`, `required_capability`, and `is_supported_by(profile)`, so future profiles like RV64IM, RV64IMA, RV64IMAC, and later floating-point extensions can be selected cleanly without destabilizing committed bytecode rows.
- Pivot from the original Claude-approved spec: this PR should add an `InlineExpansionProvider` hook now, even though the earlier plan left registered custom inline handling in tracer as future work. The hook is worthwhile long-term because it lets `jolt-program::expand` remain the canonical expansion entry point for source programs that contain Jolt custom inline opcodes, while keeping the inline registry, advice computation, and tracer CPU state outside `jolt-program`.
- This spec does not require exposing a stable public API for third-party consumers outside the Jolt workspace.
- This spec does not require moving CPU execution, lazy trace iteration, advice tapes, or memory-device emulation into verifier-facing crates. Only the backend-neutral execution contract and output row types belong in `jolt-program`; concrete execution remains in `tracer`.

## Evaluation

### Acceptance Criteria

- [x] The existing `crates/jolt-riscv` crate owns the shared instruction vocabulary, canonical `NormalizedInstruction` row, `JoltInstruction` conversion marker trait, and static metadata required by decoding, expansion, tracing, bytecode preprocessing, and verifier checks.
- [x] `crates/jolt-riscv` no longer depends on `tracer`; `tracer` depends on `jolt-riscv` for static instruction data instead.
- [x] `crates/jolt-program` exists and is a workspace member.
- [x] `jolt-program::image` exposes deterministic ELF decoding into a `DecodedProgramImage` without depending on `tracer`.
- [x] `jolt-program::expand` provides a pure RV64 expansion API over decoded `NormalizedInstruction` values.
- [x] `jolt-program::image` rejects ELF32/RV32 inputs with an explicit unsupported-architecture error.
- [x] `jolt-program` exposes an `ExecutableProgram` or equivalent built program artifact that packages decoded image data, expanded bytecode, and program preprocessing inputs without depending on `tracer`.
- [x] `jolt-program::execution` defines backend-neutral `ExecutionBackend`, `TraceSource`, `TraceInputs`, `TraceRow`, `TraceOutput`, and trace error types without importing `tracer`, `Cpu`, `Cycle`, `RISCVCycle<T>`, `LazyTraceIterator`, concrete memory-device internals, or advice-tape implementation details.
- [x] `TraceRow` and related execution contract types use `jolt-riscv` normalized instruction rows plus backend-neutral register/RAM/device data, not tracer concrete cycle types.
- [x] `tracer` implements `jolt_program::execution::ExecutionBackend` inside the `tracer` crate and adapts its concrete CPU/cycle/lazy-trace machinery to the neutral trace contract at its boundary.
- [x] SDK host-facing analyze/trace entry points invoke execution through a generic `B: ExecutionBackend` path, with the default tracer backend selected in `jolt-sdk` rather than in `jolt-trace`, `jolt-program`, or proof crates. Proof generation remains intentionally on the prover's concrete trace path in this PR because witness generation still requires lazy tracer checkpoints, advice tape output, and final-memory data that are not part of the neutral trace-row contract.
- [x] Tracer-internal implementation changes do not require changes to `jolt-riscv`, `jolt-program`, `jolt-core`, or SDK macros unless the stable execution contract or normalized program row semantics intentionally change.
- [x] `NormalizedInstruction` includes an `instruction_kind: InstructionKind` field plus normalized operands, address, virtual sequence metadata, and compressed-instruction metadata.
- [x] `jolt-riscv` documents that `InstructionKind` is the canonical flat row identity, while ISA/profile hierarchy is expressed as generated metadata rather than nested row variants.
- [x] `jolt-program::preprocess` owns materialized bytecode/RAM/program preprocessing artifacts consumed by both prover and verifier setup.
- [x] `JoltInstruction` is a marker/conversion trait equivalent to `Into<NormalizedInstruction> + TryFrom<NormalizedInstruction>`, not a second accessor abstraction over the same fields.
- [x] Any `JoltInstruction` impls for tracer's concrete instruction structs live in `tracer` as adapter impls, generated from the shared instruction-kind list where practical.
- [x] `jolt-riscv` does not retain a blanket `impl<T: tracer::instruction::RISCVInstruction> JoltInstruction for T`.
- [x] `jolt-program` does not depend on PCS implementations, Dory setup, commitment derivation, bytecode/program-image opening hints, BlindFold setup, or prover-only witness generation.
- [x] The expansion-critical `jolt-program::expand` module does not depend on CPU execution, lazy tracing, memory-device emulation, advice I/O, prover-only witness generation, transcripts, or ELF/object parsing.
- [x] Any ELF/object parsing dependency is feature-gated and isolated to `jolt-program::image` so `jolt-program::expand` and `jolt-program::preprocess` remain usable without an object-file parser dependency.
- [x] `tracer` no longer owns the canonical recursive bytecode expansion algorithm; it calls `jolt-program::expand` for expansion during trace execution and trace length accounting.
- [x] `jolt-trace::decode` no longer performs expansion by directly calling `Instruction::inline_sequence` from `tracer`; `jolt-trace` is now a tracer-free trait crate and no longer owns a decode path.
- [x] `jolt-trace` does not depend on `tracer`; tracing APIs that need a default tracer backend move to `jolt-sdk` or another host SDK layer that imports `tracer`.
- [x] `InstrAssembler`, `VirtualRegisterAllocator`, or their minimal expansion-facing equivalents live behind the `jolt-program::expand` module boundary.
- [x] `InstrAssembler` borrows `&mut ExpansionAllocator` during emission and does not own, clone, or hide allocator state behind shared ownership.
- [x] Expansion APIs return explicit errors for allocation or malformed-expansion failures instead of introducing new panics in the core expansion path.
- [x] `jolt-program::expand` exposes an `InlineExpansionProvider` trait and provider-taking expansion entry points for registered Jolt inline source opcodes, without depending on `tracer`, `inventory`, `Cpu`, advice tape internals, or concrete inline crates.
- [x] The default/provider-free expansion path returns an explicit unsupported-inline error for `InstructionKind::Inline` rows instead of silently treating them as already-expanded bytecode.
- [x] `tracer` implements the provider by adapting its existing inline registry and sequence builders to normalized `jolt-program` rows; inline advice generation remains trace-time execution behavior for this PR.
- [x] A program-preprocessing path is implemented as `ELF bytes -> DecodedProgramImage -> expanded bytecode -> JoltProgramPreprocessing`, and prover/verifier setup wraps that program preprocessing without re-decoding or re-expanding the program.
- [x] The verifier-facing path does not depend on CPU execution, lazy tracing, memory-device emulation, advice I/O, or prover-only witness generation.
- [x] `cargo tree -p tracer` shows `tracer` depending on `jolt-riscv` and `jolt-program`, but neither lower-level crate depending back on `tracer`.
- [x] `cargo tree -p jolt-program` shows no dependency on `tracer` while still compiling the execution trait module.
- [x] `cargo tree -p jolt-core --features host` does not require `tracer` solely to name program preprocessing or bytecode rows.
- [x] Fixture consistency tests prove byte-for-byte or structure-for-structure parity with the current RV64 expansion output for supported RV64 instruction families.
- [x] Tests cover recursive expansion, `rd = x0`, virtual-register clearing, compressed source instructions, and bytecode PC mapping.
- [x] The crate dependency surface is suitable for future formal verification and extraction work: no CPU, memory-device, prover, transcript, or ELF parser dependency in `jolt-program::expand`.
- [x] Program-preprocessing and verifier-facing modules document any deliberate choices that make Hax/Aeneas/Lean extraction harder, so those choices can be revisited in a follow-up rather than discovered after the split.

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

After the full cutover, keep fixture consistency tests in the new crate so future changes to expansion semantics are intentional and reviewable.

### Performance

The refactor should have no measurable runtime regression in guest decoding, tracing, or bytecode preprocessing. The implementation should benchmark at least one representative guest decode/trace path before and after the change. Any additional allocation introduced by the crate boundary should be removed or justified before merge.

Do not add a new `jolt-eval` benchmark in this PR unless a reviewer asks for one. Decode-plus-expansion benchmarking can become a measured objective after the architectural split is stable.

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

  execution
    -> ExecutableProgram
    -> ExecutionBackend and TraceSource traits
    -> backend-neutral trace rows, trace inputs, trace outputs, and trace errors

jolt-trace
  -> owns tracer-free trace traits and instruction/cycle-facing helper APIs
  -> may call jolt-program APIs only for tracer-free program utilities

tracer
  -> executes instructions and produces traces
  -> implements jolt_program::execution::ExecutionBackend
  -> calls jolt-program::expand when trace-time expansion is required

jolt-core verifier
  -> consumes JoltVerifierPreprocessing
  -> depends on program preprocessing artifacts, not on tracer execution
```

`jolt-riscv` should own the instruction vocabulary and canonical normalized row, while `jolt-program` should own expansion, materialized program-preprocessing abstractions, and the stable execution backend contract. Neither boundary should depend on the full tracer. The implementation PR should produce this dependency direction:

```text
common / jolt-platform
        ^
        |
jolt-riscv
        ^
        |
jolt-program
  image -> expand -> preprocess
  execution traits
        ^
        |
jolt-core, jolt-sdk, jolt-trace, tracer
        ^            ^
        |            |
 proof setup   default tracer backend selected here
```

The `image -> expand -> preprocess` line is a module-level program-construction pipeline, not a license for every module to depend on every dependency. `jolt-program::expand` should remain pure and should not inherit the ELF parser dependency from `jolt-program::image`. The `execution` module is orthogonal to that pipeline: it names how a built program is run by a backend, but it does not own an emulator.

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
| Built executable program artifact used as a backend input | No for verification from serialized preprocessing; yes for host proving/tracing | `jolt-program` root or `jolt-program::execution` |
| Stable execution backend trait and neutral trace row contract | No for proof verification; yes for host/SDK dependency isolation | `jolt-program::execution` |
| PCS verifier setup and optional BlindFold setup | Yes, but these are proof-system setup artifacts rather than program artifacts | stay in `jolt-core` verifier or a proof-system setup crate |
| CPU execution and trace production | No | `tracer` |
| Memory-device emulation and I/O side effects | No, except the verifier consumes public `JoltDevice`/memory layout data supplied with the proof API | concrete behavior in `tracer`; neutral public data in `common` or `jolt-program::execution` |
| Advice tape I/O | No for verification, except commitments/public inputs already represented in proof and verifier inputs | prover/host tracing crates |
| Prover witness generation and prover-facing polynomial preprocessing | No | `jolt-core` prover |

#### Crate Placement And Dependency Direction

New library code should live under `crates/jolt-program` and be added to the root workspace `members` and `[workspace.dependencies]`, matching the newer crate layout used by `jolt-trace`, `jolt-riscv`, `jolt-openings`, and related libraries. Existing top-level crates such as `tracer`, `common`, `jolt-core`, `jolt-trace`, and `jolt-sdk` can depend on `jolt-program` through workspace dependencies. `jolt-riscv` already exists and should be refactored in place rather than replaced by a sibling instruction crate.

It is acceptable for `tracer` to import `jolt-program`, but only if `jolt-program` does not depend back on `tracer`. That import should have two narrow purposes: consume `jolt-program` program-construction APIs and implement `jolt_program::execution::ExecutionBackend` for the tracer backend. The current code does not yet satisfy that shape: the concrete `Instruction`, `Cycle`, `RISCVInstruction`, `NormalizedInstruction`, and per-instruction structs live in `tracer`, and `crates/jolt-riscv/Cargo.toml` has a direct `tracer = { workspace = true, features = ["std"] }` dependency for those types. Therefore `jolt-program::expand` cannot both depend on `tracer::instruction::*` and be imported by `tracer` without creating a dependency cycle.

The implementation should resolve this by moving the shared instruction vocabulary, normalized instruction row, normalized operand view, instruction flags, and decode helpers into the existing `jolt-riscv` crate. This crate already owns Jolt's RISC-V instruction kinds and circuit/instruction flag metadata; the cutover should make it the lower catalog/row crate that `tracer` and `jolt-program` consume, not a wrapper around tracer execution types.

The concrete execution data structures should remain in `tracer` unless implementation work proves that moving a small piece is clearly better. In particular, `tracer::instruction::Instruction`, `Cycle`, `RISCVCycle<T>`, per-instruction structs such as `ADD`/`LW`, register-state types, RAM-access types, and execution traits stay in `tracer`. `jolt-riscv` should own the canonical bytecode/program row type:

```rust
pub enum InstructionKind { Add, Lw, /* ... */ }

pub struct NormalizedOperands { /* rs1, rs2, rd, imm */ }

pub struct NormalizedInstruction {
    pub instruction_kind: InstructionKind,
    pub operands: NormalizedOperands,
    pub address: u64,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}
```

`jolt-program::image`, `jolt-program::expand`, and `jolt-program::preprocess` should use `NormalizedInstruction`, not tracer's concrete instruction structs. `tracer` should provide conversion at its boundary, such as `From<&tracer::instruction::Instruction> for NormalizedInstruction` and `TryFrom<NormalizedInstruction> for tracer::instruction::Instruction`, so execution can keep concrete structs while verifier-facing program preprocessing stays independent of `tracer`.

Expansion APIs should return concrete `NormalizedInstruction` rows rather than something parameterized as `Vec<I: JoltInstruction>`. Expansion is a heterogeneous row-producing operation: one source instruction may emit different real and virtual instruction kinds, plus sequence metadata. `NormalizedInstruction` is the canonical representation of those rows.

`JoltInstruction` should not duplicate the field-access API of `NormalizedInstruction`. If it remains, it should be only a narrow marker/conversion trait:

```rust
pub trait JoltInstruction:
    Into<NormalizedInstruction> + TryFrom<NormalizedInstruction>
{}
```

That trait says a concrete instruction type can round-trip through the canonical normalized row when the conversion is supported. It must not become the primary bytecode row abstraction, and `jolt-program::expand` should not be generic over it. `TryFrom<NormalizedInstruction>` is intentionally weaker than `From<NormalizedInstruction>` because tracer may have execution-only variants or unsupported rows that cannot be reconstructed infallibly. Any `JoltInstruction` impls for tracer's concrete instruction structs belong in `tracer` because the concrete types are local there. They should be adapter impls, preferably generated by reusing `jolt_riscv::for_each_instruction_kind!`.

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
jolt-core, jolt-sdk, jolt-trace, tracer
```

This direction keeps `tracer` as an execution engine while allowing it to reuse the same bytecode expansion module as program preprocessing. It also gives formal verification tools a target that is not downstream of the emulator.

The intended stability invariant is stronger than "there is no dependency cycle": tracer-internal implementation changes should be isolated to `tracer`. `jolt-program::execution` is the contract tracer implements, so lower `jolt-*` crates should change only when the normalized program row, executable-program input, or execution backend contract changes intentionally. In particular, renaming tracer's `Cycle`, changing lazy trace iteration internals, reorganizing CPU state, or altering memory-device implementation details should not require edits to `jolt-riscv`, `jolt-program`, `jolt-core`, or SDK macro generation.

`jolt-trace` should also be upstream of tracer in the target design. It should own tracer-free trace traits and instruction/cycle-facing helper APIs, but it should not select or import the concrete tracer backend. Default backend selection belongs in `jolt-sdk`, which is already the user-facing host API layer and can import both `jolt-program` and `tracer`.

#### Proposed Workspace Layout

The implementation should refactor existing `crates/jolt-riscv` and introduce one new crate under `crates/`:

```text
crates/
  jolt-riscv/
    src/
      lib.rs
      uncompress.rs
      normalized.rs
      operands.rs
      kind.rs
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
        arithmetic.rs
        assembler.rs
        control_flow.rs
        division.rs
        error.rs
        memory.rs
        metadata.rs
        operands.rs
        shifts.rs
      preprocess/
        mod.rs
        bytecode.rs
        ram.rs
        program.rs
        error.rs
      execution/
        mod.rs
        backend.rs
        trace.rs
        error.rs
```

The file list is intentionally concrete. Implementers may split individual instruction families differently if the resulting module boundary and dependency graph are the same, but the PR should not leave expansion logic scattered across `tracer`.

#### Crate Responsibilities

`crates/jolt-riscv` owns the data model that must be shared by decoding, expansion, tracing, bytecode preprocessing, and verifier checks. It already contains Jolt instruction kind wrappers, `JoltInstruction`, `JoltInstructions`, and circuit/instruction flag metadata; this PR should remove its current `tracer` dependency by making it own the abstract instruction vocabulary, normalized row, and conversion marker rather than any tracer execution types:

- `src/kind.rs`: own `InstructionKind`, the canonical names of real and virtual Jolt instructions, plus static metadata such as side-effect classification.
- `src/operands.rs`: own `NormalizedOperands` and operand accessors.
- `src/normalized.rs`: own `NormalizedInstruction`, including `instruction_kind`, normalized operands, address, virtual sequence metadata, and compressed-instruction metadata.
- `src/jolt_instruction.rs`: define the marker trait as `Into<NormalizedInstruction> + TryFrom<NormalizedInstruction>`; do not implement it blanket-style for `tracer::instruction::RISCVInstruction`.
- `src/instructions/*.rs`: keep the existing Jolt instruction kind wrappers and flag declarations; do not move tracer's concrete execution structs into this crate as the default design.
- `src/uncompress.rs`: own RV64 compressed RISC-V decompression helpers used by ELF decode.
- `src/traits.rs`: define pure traits needed by instruction metadata. Execution-specific traits must remain outside this crate.

The shared instruction list and macro input should move into `jolt-riscv`. Today `tracer/src/instruction/mod.rs` generates `Instruction` and `Cycle` from the same instruction list. After the split, `jolt-riscv` should own the canonical list and use it to generate `InstructionKind` plus pure metadata dispatch. `tracer` should reuse that same list by invoking an exported `macro_rules!` token tree from `jolt-riscv`, for example `jolt_riscv::for_each_instruction_kind!`, to define its concrete `Instruction`, `Cycle`, `RISCVCycle<T>`, and execution/trace dispatch. The implementation must not duplicate the long instruction list across crates.

The canonical instruction list should not stay permanently flat at the declaration level. The row/discriminant type should remain flat because it is used in serialization, bytecode commitments, lookup indexing, and proof code, but the source list should be grouped by capability/family, for example `rv64i`, `rv64m`, `rv64a`, `zicsr`, `system`, `jolt_virtual`, and `jolt_inline`. That grouped declaration should generate the flat `InstructionKind` enum, family/capability metadata, profile checks, and tracer dispatch lists from one source of truth. Compressed instructions should be represented as an input decoding capability rather than as separate bytecode instruction kinds: `C.ADDI` decodes to `ADDI` with `is_compressed = true`, so disabling RV64C should reject 16-bit encodings at `jolt-program::image` rather than remove an `InstructionKind` variant.

The profile API can be a follow-up, but the intended shape is:

```rust
pub enum InstructionFamily {
    Rv64I,
    Rv64M,
    Rv64A,
    Zicsr,
    System,
    JoltVirtual,
    JoltInline,
}

pub enum IsaProfile {
    Rv64IM,
    Rv64IMA,
    Rv64IMAC,
}

impl InstructionKind {
    pub const fn family(self) -> InstructionFamily { /* generated */ }
}

impl IsaProfile {
    pub const fn supports_compressed(self) -> bool { /* generated */ }
    pub const fn supports_kind(self, kind: InstructionKind) -> bool { /* generated */ }
}
```

Jolt additions should be modeled separately from traditional RISC-V extensions. `JoltVirtual` rows are verifier/proof bytecode helpers produced by expansion, while `JoltInline` rows are source-program extension opcodes whose concrete identity is `(opcode, funct3, funct7)` plus operands. Treating both as ordinary RV64 ISA families would make future profile selection confusing: an RV64IM guest may still require Jolt virtual helper rows after expansion, and a guest may choose to enable or disable Jolt inline source opcodes independently from the base RISC-V ISA.

Registered custom inline expansion should be pluggable rather than tracer-owned or `jolt-program`-owned. `jolt-program::expand` should define the trait:

```rust
pub trait InlineExpansionProvider {
    fn expand_inline(
        &mut self,
        instruction: &NormalizedInstruction,
        allocator: &mut ExpansionAllocator,
    ) -> Result<Vec<NormalizedInstruction>, ExpansionError>;
}
```

and expose provider-taking entry points such as `expand_instruction_with_provider` and `expand_program_with_provider`. The provider-free `expand_instruction`/`expand_program` functions should still exist for ordinary RV64/Jolt-virtual expansion, but must reject source `InstructionKind::Inline` rows explicitly. `tracer` can then implement the provider by calling the existing inventory-backed `(opcode, funct3, funct7)` inline registry and normalizing the returned sequence. This is a deliberate pivot from leaving inlines entirely in tracer: it keeps the program construction pipeline unified while avoiding a larger refactor of `jolt-inlines-sdk::InlineOp`, old typed `InstrAssembler`, and `build_advice` in this PR.

`crates/jolt-program` owns program image decoding, bytecode expansion, materialized program preprocessing, and the backend-neutral execution contract. The first three are one package because they are one program-construction pipeline; the execution module is adjacent because it defines how a built program is handed to an execution backend. The internal modules should remain separate enough that dependency and formalization boundaries are still visible.

`jolt-program::image` owns deterministic ELF parsing:

- `image/elf.rs`: move the non-execution part of `tracer::decode`: parse ELF, reject ELF32/RV32, compute entry address, collect RAM image bytes, decode `.text` into RV64 `NormalizedInstruction` values, and compute `program_end`.
- `image/decode.rs` or an equivalent module: own RV64 opcode decoding into `NormalizedInstruction` using `jolt-riscv`'s `InstructionKind`, `NormalizedOperands`, and decode helpers.
- `image/mod.rs`: expose `DecodedProgramImage` and `decode_elf`.
- `error.rs`: replace warnings/panics in decode with explicit `ProgramError` or `DecodeError` values where possible. If current behavior must preserve `UNIMPL` insertion for invalid words, encode that policy explicitly.

The `object` dependency should be feature-gated and used only by `jolt-program::image`. The pure expansion and preprocessing modules should still be usable in builds that operate on decoded instructions or serialized preprocessing rather than ELF bytes.

`jolt-program::expand` owns source-instruction to final-bytecode-row expansion:

- `expand/allocator.rs`: move and simplify `VirtualRegisterAllocator`; remove `Arc<Mutex<_>>` unless cross-thread sharing is truly required by the expansion API.
- `expand/assembler.rs`: move `InstrAssembler`, `Value`, and inline emission helpers from `tracer/src/utils/inline_helpers.rs`.
- `expand/mod.rs`: provide `expand_instruction`, `expand_program`, and recursive expansion dispatch.
- `expand/metadata.rs`: assign `is_first_in_sequence`, `virtual_sequence_remaining`, source address, and compressed metadata consistently.
- `expand/error.rs`: define `ExpansionError` for virtual register exhaustion, invalid inline write targets, malformed sequence metadata, and unsupported instructions.

`ExpansionAllocator` should be single-owner mutable state passed as `&mut ExpansionAllocator` through expansion. The current `VirtualRegisterAllocator` uses `Arc<Mutex<_>>` because the old API exposes only `&VirtualRegisterAllocator`, clones that allocator into `InstrAssembler` and `VirtualRegisterGuard`, and relies on guards deallocating through shared state on `Drop`. The new crate should make that state flow explicit instead: recursive expansion and inline finalization should borrow one allocator mutably, and any current per-CPU or per-thread allocator sharing should become per-expansion ownership unless an implementation can name a real cross-thread requirement. `InstrAssembler` should therefore borrow `&mut ExpansionAllocator` for the duration of emission rather than owning or cloning allocator state. Its field shape should make the borrow visible, for example an `InstrAssembler<'a>` containing an `&'a mut ExpansionAllocator` plus only the emission buffers and metadata needed for the active inline sequence.

`jolt-program::preprocess` owns materialized program preprocessing artifacts used by both prover and verifier:

- `preprocess/bytecode.rs`: move `BytecodePreprocessing`, `BytecodePCMapper`, and bytecode preprocessing errors from `jolt-core/src/zkvm/bytecode/mod.rs`.
- `preprocess/ram.rs`: move `RAMPreprocessing` and pure RAM initialization helpers from `jolt-core/src/zkvm/ram/mod.rs`. Dependency purity is binding for `compute_min_ram_K`: move it only if it stays dependency-light. If relocating it would drag `jolt-core`, PCS setup, prover-only modules, or proof-system configuration into `jolt-program::preprocess`, leave `compute_min_ram_K` in `jolt-core` and have it consume the pure RAM preprocessing surface from `jolt-program::preprocess`.
- `preprocess/program.rs`: move the current shared layer into the final `JoltProgramPreprocessing` type, including canonical serialization and `digest()`.
- `preprocess/error.rs`: consolidate bytecode/RAM/program preprocessing errors.

This module is intentionally program-level rather than verifier-only. The current code already has a `JoltSharedPreprocessing` layer containing `Arc<BytecodePreprocessing>`, `RAMPreprocessing`, `MemoryLayout`, and `max_padded_trace_length`; both `JoltProverPreprocessing` and `JoltVerifierPreprocessing` wrap that shared layer. This PR should cut over that layer to `JoltProgramPreprocessing` in `jolt-program::preprocess`. `JoltProverPreprocessing` and `JoltVerifierPreprocessing` should remain in `jolt-core` unless a later proof-system setup refactor moves them, because they add PCS prover/verifier setup and optional BlindFold setup rather than program preprocessing semantics.

`jolt-program::preprocess` should not own committed-bytecode or committed-program-image derivation. Those artifacts require PCS prover setup, Dory geometry, trusted commitments, and prover opening hints. They should remain in `jolt-core` or a future PCS-aware committed-program crate that consumes `JoltProgramPreprocessing`.

`jolt-program::execution` owns the backend-neutral tracing contract, not a tracer implementation:

- `execution/backend.rs`: define `ExecutionBackend`, `TraceSource`, and any adapter traits needed by SDK/host APIs.
- `execution/trace.rs`: define `ExecutableProgram`, `TraceInputs`, `TraceRow`, `TraceOutput<T>`, and backend-neutral register/RAM/memory-image data needed by prover setup.
- `execution/error.rs`: define `TraceError` or a small error enum wrapping backend failures without exposing tracer concrete types.

The execution contract should make the conceptual boundary concrete:

```rust
pub trait ExecutionBackend {
    type Trace: TraceSource;

    fn trace(
        &mut self,
        program: &ExecutableProgram,
        inputs: TraceInputs,
    ) -> Result<TraceOutput<Self::Trace>, TraceError>;
}

pub trait TraceSource {
    fn next_row(&mut self) -> Option<TraceRow>;
}
```

The exact field names can change, but `TraceRow` should be expressed in terms of `NormalizedInstruction` plus backend-neutral register and RAM access data. It should not expose `tracer::Cycle`, `tracer::RISCVCycle<T>`, `Cpu`, `LazyTraceIterator`, concrete memory-device state, or advice-tape internals. The `ExecutableProgram` input should be built from `jolt-program::image`, `jolt-program::expand`, and preprocessing inputs so host code can pass one stable program artifact to any backend.

This module is the answer to the "changing tracer should not touch lower crates" concern. `jolt-program` defines what execution must provide; `tracer` implements it. If tracer reorganizes its CPU, cycle enum, lazy iterator, or memory-device implementation without changing the contract, those edits should stay in `tracer`.

`tracer` remains the execution engine:

- keeps `Cpu`, `Memory`, `LazyTraceIterator`, `Cycle`, execution semantics, advice tape plumbing, trace-to-file support, and `jolt-emu`;
- imports `jolt-riscv` for instruction data;
- imports `jolt-trace` only to implement tracer adapters for tracer-local cycle types;
- imports `jolt-program::expand` for trace-time expansion and trace length accounting;
- imports `jolt-program::image` for ELF decode where tracing from ELF is needed.
- implements `jolt_program::execution::ExecutionBackend` and adapts tracer-local `Cycle`/lazy trace output to `TraceRow`/`TraceOutput` at the crate boundary.

`jolt-core` remains the proof system:

- imports program preprocessing artifacts from `jolt-program::preprocess`;
- keeps prover/verifier protocols, sumchecks, commitments, witness generation, and proof serialization;
- should not import `tracer` just to name bytecode preprocessing artifacts.

`jolt-trace` remains a tracer-free trace trait crate:

- owns `JoltCycle` and any tracer-free instruction/cycle-facing helper APIs;
- does not depend on `tracer` and does not choose a concrete execution backend;
- deletes or moves APIs that return tracer `Cycle`, `LazyTraceIterator`, tracer `Memory`, or tracer-backed `ProgramSummary`. Those APIs belong in `jolt-sdk` if they need the default tracer backend, or in `jolt-program`/`jolt-trace` only if rewritten over tracer-free data.

`jolt-sdk` remains the macro and user-facing SDK layer:

- generated prove/analyze/preprocess code should build or receive an `ExecutableProgram` through `jolt-program`;
- generated host APIs should be generic over `B: ExecutionBackend` where practical;
- `jolt-sdk` owns the default host backend selection and may import `tracer` to construct the default `TracerBackend`, while keeping tracer concrete types out of `jolt-riscv`, `jolt-program`, `jolt-trace`, and verifier-facing preprocessing types.

#### Dependency Table

Target dependency edges:

| Crate | May depend on | Must not depend on |
|-------|---------------|--------------------|
| `jolt-riscv` | `common`, `jolt-platform`, `serde`, `ark-serialize`, `strum`, `paste`, `derive_more` | `tracer`, `jolt-core`, `jolt-trace`, `jolt-program`, `object` |
| `jolt-program` | `jolt-riscv`, `common`, `ark-serialize`, `blake2`, `thiserror`, `serde`, optional `object` and `tracing` behind image/host features | `tracer`, `jolt-core`, `jolt-trace`, PCS implementations, Dory setup, prover-only modules |
| `tracer` | `jolt-riscv`, `jolt-program`, `jolt-trace` for cycle-trait adapter impls, `common`, `jolt-platform` | `jolt-core` |
| `jolt-core` | `jolt-riscv`, `jolt-program`, proof-system crates, `common` | `tracer` for program preprocessing types |
| `jolt-trace` | `jolt-program`, `common`, `jolt-riscv` as needed for program rows | `tracer`, owning canonical expansion semantics, or owning the execution backend contract |
| `jolt-sdk` | `jolt-program`, `jolt-core`, `tracer` for default host backend wiring | making `jolt-riscv`, `jolt-program`, `jolt-trace`, or verifier-facing generated types depend on tracer concrete internals |

The implementation PR must not introduce a new cycle. Its final state should remove the `tracer -> jolt-program -> tracer` cycle risk entirely by moving shared instruction vocabulary and metadata below both crates.

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

- `tracer/Cargo.toml`: add `jolt-riscv`, `jolt-program`, and `jolt-trace` for tracer-local adapter impls.
- `crates/jolt-riscv/Cargo.toml`: remove the direct dependency on `tracer`; keep only lower-level dependencies needed by static instruction data, flags, decoding, and serialization.
- `crates/jolt-trace/Cargo.toml`: add `jolt-program`; remove `tracer`.
- `jolt-sdk/Cargo.toml`: import `tracer` for default host backend wiring.
- `jolt-core/Cargo.toml`: add `jolt-riscv` and `jolt-program`; remove program-preprocessing dependence on `tracer` where feasible.
- `jolt-sdk/Cargo.toml` and `jolt-sdk/macros/Cargo.toml`: update imports only as needed for generated preprocessing functions.

The crate should provide explicit APIs along these lines:

```rust
pub fn expand_instruction(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError>;

pub fn expand_program(
    instructions: impl IntoIterator<Item = NormalizedInstruction>,
) -> Result<Vec<NormalizedInstruction>, ExpansionError>;
```

The exact type names can change during implementation, but the API should make state flow explicit. In particular, virtual-register allocation state should be visible as expansion state rather than hidden inside the tracer CPU.

The program-image and preprocessing APIs should be explicit about their inputs:

```rust
pub struct DecodedProgramImage {
    pub instructions: Vec<NormalizedInstruction>,
    pub memory_init: Vec<(u64, u8)>,
    pub program_end: u64,
    pub entry_address: u64,
}

pub fn preprocess_program(
    expanded_bytecode: Vec<NormalizedInstruction>,
    memory_init: Vec<(u64, u8)>,
    memory_layout: MemoryLayout,
    max_padded_trace_length: usize,
    entry_address: u64,
) -> Result<JoltProgramPreprocessing, PreprocessingError>;
```

The program crate should also expose a built-program artifact and execution contract that host/SDK code can use without naming tracer concrete types:

```rust
pub struct ExecutableProgram {
    pub image: DecodedProgramImage,
    pub expanded_bytecode: Vec<NormalizedInstruction>,
    pub program_preprocessing: JoltProgramPreprocessing,
}

pub fn build_executable(elf: &[u8]) -> Result<ExecutableProgram, ProgramError>;

pub struct TraceInputs {
    /* public inputs, private inputs, and device/advice handles in neutral form */
}

pub struct TraceRow {
    pub instruction: NormalizedInstruction,
    pub registers: RegisterState,
    pub ram_access: RamAccess,
}

pub struct TraceOutput<T> {
    pub trace: T,
    pub device: JoltDevice,
    pub final_memory: Option<MemoryImage>,
}

pub trait ExecutionBackend {
    type Trace: TraceSource;

    fn trace(
        &mut self,
        program: &ExecutableProgram,
        inputs: TraceInputs,
    ) -> Result<TraceOutput<Self::Trace>, TraceError>;
}
```

`RegisterState`, `RamAccess`, `MemoryImage`, `TraceInputs`, and `TraceError` may live in `jolt-program::execution` or reuse existing neutral `common` types if those already capture the right semantics. They should not be aliases of tracer concrete execution structures.

The important design point is not these exact signatures; it is that program preprocessing should not have to import `tracer::emulator::Cpu`, `LazyTraceIterator`, advice-tape machinery, host `Program` build logic, PCS setup, or committed-bytecode derivation. Similarly, SDK and host execution APIs should depend on the `ExecutionBackend` contract rather than on tracer's concrete `Cycle` or CPU internals.

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
- Low for `jolt-trace` and `tracer` execution internals in this PR, because `jolt-trace` is only a host/trace trait boundary and tracer execution internals are not extraction targets.

This PR should not contort the code to satisfy a specific extractor immediately. It should keep the extraction-critical core small, pure, dependency-light, and well-specified so a follow-up PR can run Hax/Aeneas and then address concrete tool feedback without reopening the architecture.

#### Extraction-Native Follow-Up

The crate boundary introduced by this spec makes `jolt-program::expand` the right formalization target, but an extraction-native production rewrite is large enough to require separate review. See `specs/extraction-native-bytecode-expansion.md` for a follow-up proposal that compares PR #1490 at `a3448e6da44f` against a production design based on owned expansion state, shallow family lowering, explicit work-stack recursion, bitset allocation, and bounded per-source buffers.

### Alternatives Considered

Keeping expansion in `tracer` and documenting it there is the smallest code change, but it leaves the formalization target coupled to CPU execution, memory, advice, and trace infrastructure.

Treating `jolt-trace` as the expansion crate is also tempting because it historically contained bytecode preprocessing and host-facing trace conveniences. That crate is useful to users, but it is not the minimal formalization target and should not depend on the concrete tracer backend.

Moving only `BytecodePreprocessing` is insufficient because preprocessing consumes already-expanded bytecode. The correctness-sensitive part for formalization is the path that turns one decoded source instruction into zero, one, or many final bytecode rows.

Creating separate `jolt-program-image`, `jolt-bytecode-expand`, and `jolt-program-preprocess` crates was considered, and it is a reasonable architecture if these modules later grow independent release or ownership needs. For this PR, separate crates look premature: they would add workspace churn and dependency edges without changing the true pipeline boundary. A single `jolt-program` crate with explicit `image`, `expand`, and `preprocess` modules gives the same conceptual separation with less mechanical overhead.

Creating one large `jolt-verifier-input` crate for ELF decode, expansion, bytecode preprocessing, RAM preprocessing, verifier setup, PCS setup, and committed-program data would blur the boundary between materialized program preprocessing and proof-system setup. For proof-system setup, `jolt-program` should stop at materialized program artifacts and the neutral execution contract; PCS setup, prover/verifier keys, and committed-program derivation stay outside.

Defining `ExecutionBackend` in `tracer` would preserve the current ownership problem: SDK/host code that wants a generic backend contract would still import tracer to name the trait, and lower program crates could drift toward tracer concrete types. Defining the trait only in `jolt-sdk` would keep the SDK generic but would not give non-SDK host code or future proof-system setup code a common built-program execution contract. `jolt-program::execution` is the narrow middle ground: it defines the stable shape of execution inputs and outputs while leaving all concrete execution behavior in tracer.

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

### Implementation Findings

During the initial cutover, moving bytecode preprocessing to `NormalizedInstruction` exposed one proof-semantics subtlety that was previously hidden by `jolt-core`'s per-instruction concrete `Flags` impls. `virtual_sequence_remaining == Some(0)` does not by itself mean the R1CS `IsLastInSequence` flag should be set. In current proof semantics, `IsLastInSequence` is only used to skip `NextPCEqPCPlusOneIfInline` for `JALR` at the end of trap-related inline sequences, where the next PC may jump to a trap handler instead of advancing to the next virtual bytecode row. Other final helper instructions, such as an `ADDI` with `virtual_sequence_remaining == Some(0)`, must remain `VirtualInstruction` rows but must not set `IsLastInSequence`; otherwise the `NextPCEqPCPlusOneIfInline` constraint is incorrectly suppressed.

The canonical flag behavior should therefore live in `jolt-riscv`: `VirtualInstruction` is derived from `virtual_sequence_remaining.is_some()`, `DoNotUpdateUnexpandedPC` is derived from `virtual_sequence_remaining.unwrap_or(0) != 0`, and `IsLastInSequence` is derived from `instruction_kind == InstructionKind::JALR && virtual_sequence_remaining == Some(0)`. The implementation should keep an expanded-bytecode parity test that compares normalized flags against the existing concrete instruction flags so future changes to this behavior are intentional.

The allocator cutover also exposed several non-obvious expansion invariants that should stay covered by fixture consistency tests. First, helper instructions emitted by an expansion must themselves be canonicalized through `jolt-program::expand`; for example an emitted `SLLI` row may become `VirtualMULI`, and skipping recursive canonicalization changes the bytecode. Second, some signed DIV/REM expansions must delay temporary-register allocation until after nested multiplication helpers return; otherwise the existing eight-register virtual pool is exhausted. Third, `REMUW` intentionally reuses its advice register as scratch while `DIVUW` needs a separate quotient/temp split. Fourth, RV64 `SCW` spills through the reservation register before expanding the nested store because `SW` consumes almost the whole temporary pool. These are behavioral compatibility constraints, not merely implementation details.

RV64 ELF decode has one additional boundary to keep explicit. Registered custom inline opcodes carry dispatch metadata in `opcode`, `funct3`, and `funct7`; a generic `InstructionKind::Inline` row alone is not enough to reconstruct the concrete tracer instruction. The normalized inline row should therefore preserve this dispatch metadata, while the registered-inline sequence/advice registry can remain a separate execution-boundary question until it is intentionally moved out of tracer.

The execution-backend cutover should proceed incrementally. The implementation now provides the stable lower hook: `jolt-core::host::Program` can build a `jolt_program::ExecutableProgram` and run it through any `jolt_program::execution::ExecutionBackend`, while existing tracer-returning host convenience APIs remain available for compatibility. A follow-up SDK pass should move generated default host tracing/proving conveniences onto that hook, constructing `tracer::TracerBackend` in SDK wiring rather than asking lower program/preprocessing layers to name tracer internals.

## Documentation

Update the Jolt book only if the crate is exposed to users or changes contributor-facing architecture. At minimum, add developer documentation describing:

- where bytecode expansion lives,
- which modules are verifier-critical and which crates are prover/host-only,
- how program preprocessing can be derived from serialized preprocessing or from ELF bytes,
- how decoded instructions become expanded bytecode rows,
- how `virtual_sequence_remaining` is assigned and consumed,
- how host/SDK code selects an execution backend without making `jolt-program` depend on tracer,
- how to add or modify an instruction expansion safely.

## Execution

### Files To Create

- `crates/jolt-riscv/src/uncompress.rs`
- `crates/jolt-riscv/src/normalized.rs`
- `crates/jolt-riscv/src/operands.rs`
- `crates/jolt-riscv/src/kind.rs`
- `crates/jolt-riscv/src/traits.rs`
- `crates/jolt-program/Cargo.toml`
- `crates/jolt-program/src/lib.rs`
- `crates/jolt-program/src/error.rs`
- `crates/jolt-program/src/image/mod.rs`
- `crates/jolt-program/src/image/decode.rs`
- `crates/jolt-program/src/image/elf.rs`
- `crates/jolt-program/src/expand/mod.rs`
- `crates/jolt-program/src/expand/allocator.rs`
- `crates/jolt-program/src/expand/assembler.rs`
- `crates/jolt-program/src/expand/error.rs`
- `crates/jolt-program/src/expand/metadata.rs`
- `crates/jolt-program/src/preprocess/mod.rs`
- `crates/jolt-program/src/preprocess/bytecode.rs`
- `crates/jolt-program/src/preprocess/ram.rs`
- `crates/jolt-program/src/preprocess/program.rs`
- `crates/jolt-program/src/preprocess/error.rs`
- `crates/jolt-program/src/execution/mod.rs`
- `crates/jolt-program/src/execution/backend.rs`
- `crates/jolt-program/src/execution/trace.rs`
- `crates/jolt-program/src/execution/error.rs`

### Files To Modify

- `Cargo.toml`: add the `crates/jolt-program` workspace member and workspace dependency; keep the existing `jolt-riscv` workspace entry.
- `Cargo.lock`: update after adding the workspace crate.
- `tracer/Cargo.toml`: depend on `jolt-riscv` and `jolt-program`.
- `tracer/src/lib.rs`: remove ELF decode ownership; tracer-specific ELF entry points should call `jolt_program::image::decode_elf` internally only when they also perform tracing/execution work; implement or reexport the tracer backend type that satisfies `jolt_program::execution::ExecutionBackend`.
- `tracer/src/emulator/cpu.rs`: keep tracer-local `Xlen` and concrete execution structs; call expansion APIs through conversion to/from `NormalizedInstruction` where trace length or trace-time `rd = x0` expansion is needed.
- `tracer/src/**`: adapt tracer-local `Cycle`, `LazyTraceIterator`, register state, RAM accesses, device output, and final-memory output to `jolt_program::execution::{TraceRow, TraceSource, TraceOutput}` at the boundary without exposing concrete tracer types through `jolt-program`.
- `tracer/src/instruction/**`: keep concrete instruction structs and execution-only methods in `tracer`; convert to/from `jolt-riscv`'s `NormalizedInstruction` at the crate boundary; implement `JoltInstruction` for concrete tracer instruction types here if callers still need that trait on those types.
- `tracer/src/utils/inline_helpers.rs`: move expansion helpers to `jolt-program::expand`; delete or replace with imports during the same full cutover.
- `tracer/src/utils/virtual_registers.rs`: move allocator to `jolt-program::expand`; delete or replace imports.
- `crates/jolt-riscv/Cargo.toml`: remove `tracer` and add any lower-level dependencies needed by moved instruction data.
- `crates/jolt-riscv/src/lib.rs`: expose the canonical instruction-kind list, `InstructionKind`, `NormalizedInstruction`, `NormalizedOperands`, `JoltInstruction`, pure traits, and flags without referencing `tracer` or `jolt-program`.
- `crates/jolt-riscv/src/instructions/**`: keep existing Jolt instruction kind wrappers and pure instruction metadata; do not import tracer concrete instruction structs.
- `crates/jolt-riscv/src/normalized.rs`: define `NormalizedInstruction` with an `instruction_kind` field.
- `crates/jolt-riscv/src/jolt_instruction.rs`: remove the blanket impl over `tracer::instruction::RISCVInstruction`; define only the marker/conversion trait without depending on tracer.
- `crates/jolt-trace/Cargo.toml`: add `jolt-program`; remove `tracer`.
- `crates/jolt-trace/src/program.rs`, `analyze.rs`, `ram.rs`, and tracer-backed `bytecode.rs`: remove or move tracer-backed APIs to `jolt-sdk`; reintroduce only tracer-free versions over `jolt-program` or `jolt_program::execution`.
- `jolt-core/Cargo.toml`: add `jolt-riscv` and `jolt-program`.
- `jolt-core/src/zkvm/bytecode/mod.rs`: move or reexport materialized bytecode preprocessing from `jolt-program::preprocess`.
- `jolt-core/src/zkvm/ram/mod.rs`: move or reexport pure RAM preprocessing from `jolt-program::preprocess`; leave prover/verifier sumcheck modules in `jolt-core`.
- `jolt-core/src/zkvm/verifier.rs`: use program preprocessing types from `jolt-program::preprocess`, while keeping `JoltVerifierPreprocessing` if PCS setup remains in `jolt-core`.
- `jolt-core/src/zkvm/prover.rs`: update imports for program preprocessing, bytecode preprocessing, RAM preprocessing, and program instruction types.
- `jolt-core/src/poly/**`, `jolt-core/src/subprotocols/**`, `jolt-core/src/zkvm/**`: update imports from `tracer::instruction` to `jolt-riscv` where the code needs `NormalizedInstruction`, `InstructionKind`, `JoltInstruction`, or static instruction data.
- `jolt-sdk/macros/src/lib.rs`: generated preprocessing functions should call the modular decode/expand/program-preprocess path; generated prove/analyze/trace entry points should accept an execution backend through `jolt_program::execution::ExecutionBackend` or construct the default `tracer::TracerBackend` in SDK host wiring, without naming tracer internals in lower-level generated types.
- `book/**` or developer docs: add an architecture page or section if maintainers want crate-boundary docs in the book.

### Files To Remove Or Empty

Remove these only after all call sites are cut over:

- `tracer/src/utils/inline_helpers.rs`
- `tracer/src/utils/virtual_registers.rs`
- decode/uncompress helpers under `tracer/src/instruction/` once `jolt-program::image` owns RV64 decode into `NormalizedInstruction`

Do not delete tracer's concrete instruction structs, instruction-format structs, execution-specific trace implementations, or RV32 cleanup code in this PR.

### Implementation Checklist

1. [x] Add empty `jolt-program` crate under `crates/`, workspace member, workspace dependency, and minimal module skeletons.
2. [x] Refactor existing `jolt-riscv` so it no longer depends on `tracer`.
3. [x] Add `InstructionKind`, `NormalizedInstruction`, normalized operands, marker `JoltInstruction`, and local static instruction metadata to `jolt-riscv` while keeping tracer execution types out of `jolt-riscv`.
4. [x] Add an `instruction_kind: InstructionKind` field to `NormalizedInstruction` and use it as the row type returned by decode/expansion and consumed by preprocessing.
5. [x] Move RV64 opcode decode into `jolt-program::image` and RV64 compressed-instruction decompression helpers into `jolt-riscv`; reject RV32/ELF32 in the new program pipeline.
6. [x] Export the canonical instruction-kind list from `jolt-riscv` as a macro such as `jolt_riscv::for_each_instruction_kind!`, and use it from `tracer` to generate its concrete `Instruction`, `Cycle`, and `RISCVCycle<T>`.
7. [x] Remove the `JoltInstruction` blanket impl over `tracer::instruction::RISCVInstruction`; add concrete tracer adapter impls only in `tracer` if needed.
8. [x] Update `tracer`, `jolt-core`, and `jolt-trace` imports to compile against the strengthened `jolt-riscv` normalized row and `jolt-program` pipeline.
9. [x] Add `jolt_program::image::decode_elf` by moving the non-execution logic from `tracer::decode`.
10. [x] Update all call sites of `tracer::decode` that only need ELF decoding to call `jolt_program::image::decode_elf`; keep `tracer::decode` only if it remains a tracer-specific execution API rather than a compatibility shim.
11. [x] Move `VirtualRegisterAllocator`, `InstrAssembler`, recursive `add_to_sequence`, and per-instruction inline expansion logic into `jolt-program::expand`, with `InstrAssembler` borrowing `&mut ExpansionAllocator`.
12. [x] Implement `expand_instruction` and `expand_program`.
13. [x] Update `tracer` trace-time expansion and trace length accounting to use `jolt-program::expand`.
14. [x] Update `jolt-trace::decode` to call `jolt_program::image::decode_elf` followed by `jolt_program::expand::expand_program`; completed by removing `jolt-trace` decode ownership entirely.
15. [x] Remove `jolt-trace`'s dependency on `tracer`; move tracer-backed trace convenience APIs to `jolt-sdk` or rewrite them over `ExecutionBackend`.
16. [x] Move `BytecodePreprocessing`, `BytecodePCMapper`, pure `RAMPreprocessing`, and `JoltProgramPreprocessing` into `jolt-program::preprocess`.
17. [x] Keep `compute_min_ram_K` in `jolt-core` if moving it would pull prover-only or proof-system dependencies into `jolt-program::preprocess`.
18. [x] Add `ExecutableProgram` or the final equivalent built-program artifact that packages decoded image data, expanded bytecode, and preprocessing inputs for execution backends.
19. [x] Add `jolt-program::execution` with `ExecutionBackend`, `TraceSource`, `TraceInputs`, `TraceRow`, `TraceOutput`, and trace error types expressed without tracer concrete types.
20. [x] Implement `ExecutionBackend` in `tracer`, adapting `Cpu`, `Cycle`, `LazyTraceIterator`, register state, RAM accesses, device output, and final memory into the neutral execution contract at the tracer boundary.
21. [x] Update `jolt-core` prover/verifier code to consume the moved preprocessing types.
22. [x] Update SDK macro-generated preprocessing to use the modular path.
23. [x] Update SDK host-facing trace/analyze entry points to use `B: ExecutionBackend` or construct a default `tracer::TracerBackend`, keeping tracer concrete types out of lower generated APIs. Prove remains on the concrete prover witness API in this PR, as documented above.
24. [x] Add expansion parity tests before removing old expansion entry points.
25. [ ] Follow-up: add `jolt-eval` invariants for expansion fixture consistency, PC mapping consistency, and program preprocessing determinism if maintainers want evaluation-level coverage beyond the targeted parity tests.
26. [ ] Follow-up: add a decode-plus-expansion benchmark under `jolt-eval` if this becomes a measured objective.
27. [x] Remove old canonical expansion ownership from `tracer`.
28. Run formatting, clippy, host tests, ZK tests, and targeted crate dependency checks.

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
cargo tree -p tracer
cargo tree -p jolt-riscv
cargo tree -p jolt-program
cargo tree -p jolt-trace
cargo tree -p jolt-sdk --features host
```

The dependency checks should confirm that `jolt-riscv`, `jolt-program`, and `jolt-trace` do not depend on `tracer`, that the default or no-ELF feature path for `jolt-program` does not pull object parsing into `jolt-program::expand` or `jolt-program::preprocess`, and that any `jolt-sdk` tracer edge is confined to host/default-backend wiring rather than verifier-facing generated types.

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
- PR [#1369](https://github.com/a16z/jolt/pull/1369): added `jolt-trace`, which is useful host-facing trace structure but does not by itself isolate bytecode expansion.
- PR [#1260](https://github.com/a16z/jolt/pull/1260): broader crate refactor discussion relevant to dependency direction and crate boundaries.
