# Spec: Architectural trace (one row per source instruction)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The tracer emits one row per Jolt instruction, so a source instruction that expands to a virtual sequence produces several rows and none describes the instruction itself. A consumer that works on the source ISA directly, such as a profiler or a differential checker that wants one row per executed RISC-V instruction, must regroup rows and discard virtual-register traffic. It also cannot make decode reject what it does not handle: the smallest exported profile is `RV64IM_JOLT`, `tracer::decode` hard-codes `RV64IMAC_JOLT`, and no code consults `SourceExtension::Rv64C`. This spec covers one pull request in four parts. **Profile**: an `RV64I` profile that decode enforces, compressed encodings included. **Seam**: the execution seam carries any row type. **Backend**: `SourceTraceRow` and `SourceTracerBackend`, one flat row per executed instruction of an RV64I program. **Decode mode**: ELFs produced by the RISC-V architectural test framework (ACT4, `specs/act4-tests.md`) and by hand-written assembly place data words inside executable sections, and decode, which reads every word of such a section as an instruction, rejects them under `RV64I`; a mode chosen by the caller makes a word that does not decode a hole in the instruction list, an error only if it is fetched.

## Intent

### Goal

**Profile.**

```rust
// crates/jolt-riscv/src/profile.rs, re-exported from the crate root
pub const RV64I: JoltInstructionProfile = JoltInstructionProfile {
    source_extensions: &[SourceExtension::Rv64I],
    inline_extensions: &[],
};
// crates/jolt-program/src/error.rs, new variant of ProgramError
#[error("compressed instruction at {address:#x} is not legal in the selected profile")]
IllegalCompressedInstruction { address: u64 },
// tracer/src/lib.rs
pub fn decode_with_profile(elf: &[u8], profile: JoltInstructionProfile)
    -> Result<(Vec<Instruction>, Vec<(u64, u8)>, u64, u64), ProgramError>;
```

`decode_instruction` keeps its signature. `tracer::decode(elf)` keeps its signature and its panics and calls `decode_with_profile(elf, RV64IMAC_JOLT)`.

**Seam.** `TraceSource`, `ExecutionBackend` and `OwnedTrace` take the row type as a parameter that defaults to `TraceRow`.

```rust
// crates/jolt-program/src/execution/backend.rs
pub trait ExecutionBackend<R = TraceRow> {
    type Trace: TraceSource<R>;
    fn trace(&mut self, program: &JoltProgram, inputs: TraceInputs)
        -> Result<TraceOutput<Self::Trace>, TraceError>;
}
pub trait TraceSource<R = TraceRow> {
    fn next_row(&mut self) -> Option<R>;
    fn rows(&self) -> Option<&[R]> { None }
    fn shared_rows(&self) -> Option<Arc<Vec<R>>> { None }
}
// crates/jolt-program/src/execution/trace.rs
pub struct OwnedTrace<R = TraceRow> { /* Arc<Vec<R>>, cursor */ }
impl<R: Clone> TraceSource<R> for OwnedTrace<R>;
// JoltProgram::trace_with<R, B: ExecutionBackend<R>>(&self, &mut B, TraceInputs)
```

`OwnedTrace<R>` has `new`, `rows`, `From<Vec<R>>`, `Default` and `Clone` with no bound on `R`, and `into_rows` for `R: Clone`. `ChunkedExecutionBackend` is unchanged.

**Backend.**

```rust
// crates/jolt-program/src/execution/trace/source_row.rs, re-exported from `execution`
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceTraceRow { /* private: nine u64 words, a u32 instruction index, four tag bytes */ }
impl SourceTraceRow {
    pub fn new(instruction_index: u32, pc: u64, next_pc: u64,
               registers: RegisterState, ram_access: RamAccess) -> Self;
    pub const fn instruction_index(&self) -> u32;
    // pub const fn ..(&self) -> u64: pc, next_pc, rs1_value, rs2_value, rd_pre_value,
    // rd_post_value, ram_address, ram_pre_value, ram_post_value
    pub fn registers(&self) -> RegisterState;
    pub fn ram_access(&self) -> RamAccess;
}

// crates/jolt-program/src/execution/error.rs; TraceError gains SourceTrace(#[from] SourceTraceError)
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SourceTraceError { // each variant has an #[error] message printing its fields
    UnsupportedInstruction { pc: u64, kind: SourceInstructionKind },
    PcOutsideProgram { pc: u64 },
    MisalignedAccess { pc: u64, address: u64, width: u8 },
    StoreToProgramText { pc: u64, address: u64 },
    DeviceRegisterAccess { pc: u64, address: u64 },
    ProgramTextTooLarge { span: u64 },
}

// tracer/src/source_trace.rs, re-exported from the crate root
#[derive(Default, Debug, Clone)]
pub struct SourceTracerBackend { /* private: row capacity hint, 0 by default */ }
impl SourceTracerBackend {
    pub fn with_row_capacity(rows: usize) -> Self;
}
impl ExecutionBackend<SourceTraceRow> for SourceTracerBackend {
    type Trace = OwnedTrace<SourceTraceRow>;
}
```

```rust
let program = JoltProgram::from_elf_bytes_with_profile(elf, RV64I);
let config = MemoryConfig { program_size: Some(program_size), ..Default::default() };
let inputs = TraceInputs::new(input, Vec::new(), Vec::new(), config);
let output = program.trace_with(&mut SourceTracerBackend::with_row_capacity(n), inputs)?;
let rows: &[SourceTraceRow] = output.trace.rows();
```

`program_size` must be `Some` and cover the loaded image, as for `TracerBackend`.

The backend decodes `program.elf_bytes()` with `decode_with_mode(elf, RV64I, mode)` in the mode it holds, `Strict` by default; `program.profile` is not consulted, and an empty ELF returns `TraceError::MissingElfBytes`. It executes each decoded instruction through a new `pub(crate) fn Instruction::execute_direct(&self, cpu: &mut Cpu)`, generated beside `Instruction::execute`, which calls the per-instruction `RISCVInstruction::execute` with a discarded stack-local `RAMAccess`: no rd = x0 rewrite, no virtual sequence, no rows. `Mmu` gains a private `record_access: bool`, `true` by default, and `pub(crate) fn set_access_recording(&mut self, on: bool)`, which the backend turns off (invariant B10).

**Decode mode.**

```rust
// crates/jolt-program/src/image/elf.rs, re-exported from `image`
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodeMode {
    #[default]
    Strict,          // the decode of P1 to P6
    DataHoles,  // four-byte slots; a slot that does not decode is omitted (D2)
}
pub fn decode_elf_with_mode(elf: &[u8], profile: JoltInstructionProfile, mode: DecodeMode)
    -> Result<Rv64ProgramImage, ProgramError>;
// crates/jolt-program/src/error.rs, new variant of ProgramError
#[error("the selected decode mode is not defined for a profile with compressed instructions")]
DecodeModeUnsupportedByProfile,
// tracer/src/lib.rs
pub fn decode_with_mode(elf: &[u8], profile: JoltInstructionProfile, mode: DecodeMode)
    -> Result<(Vec<Instruction>, Vec<(u64, u8)>, u64, u64), ProgramError>;
// tracer/src/source_trace.rs
impl SourceTracerBackend {
    pub fn with_decode_mode(self, mode: DecodeMode) -> Self;
}
```

No existing signature changes. Each decode entry point gains a sibling, as `decode` gained `decode_with_profile`. The backend gains a chaining setter in the manner of `TraceInputs::with_advice_tape`, because a second constructor could not be combined with `with_row_capacity`. `decode_elf(elf, profile)` is `decode_elf_with_mode(elf, profile, DecodeMode::Strict)`, and `decode_with_profile(elf, profile)` is `decode_with_mode(elf, profile, DecodeMode::Strict)`. `SourceTracerBackend` holds a private `DecodeMode`, which is `Strict` after `default` and `with_row_capacity`. The private `decode_text_section` and `SourceExecution::new` take the mode as a parameter. One value thus selects both the list the backend executes and the list a consumer resolves `instruction_index` against (D6):

```rust
let mode = DecodeMode::DataHoles;
let image = decode_elf_with_mode(elf, RV64I, mode)?; // rows index image.instructions
let mut backend = SourceTracerBackend::with_row_capacity(n).with_decode_mode(mode);
```

### Invariants

**Profile.**

- P1. `RV64I.supports_source(kind)` holds exactly for the 52 kinds whose `source_extension` is `Rv64I` and for the two placeholder kinds that have none (`NoOp`, `Unimpl`), which decode never produces. The set is defined once, by `source_extension`.
- P2. Under a profile without `Rv64C`, `decode_instruction` returns `IllegalCompressedInstruction { address }` when `is_compressed` is set, and otherwise `MalformedImage("instruction address is not 4-byte aligned")` when `address % 4 != 0`. Both checks precede the decoding of the kind.
- P3. Under a profile that contains `Rv64C`, decode output is unchanged for every input.
- P4. Every other rejection keeps its error: `IllegalSourceInstruction(kind)` for a kind outside the profile, `MalformedImage` for an unknown encoding. A section whose extent passes the end of the address space is `MalformedImage`, not an overflow.
- P5. A profile remains a legality set: `decode_elf` does not expand, and the profile gains no field.
- P6. The `Strict` sweep of `decode_text_section` skips a zero halfword as padding under every profile, before the profile is consulted. An image accepted under `RV64I` is therefore a set of 32-bit instructions at 4-aligned addresses, possibly with holes.

**Seam.**

- A1. Every existing bound, implementation and call site compiles unchanged, and every backend produces the same rows.
- A2. `TraceSource::rows` and `shared_rows` return `None` once `next_row` has been called. The inherent `OwnedTrace::rows` returns every row regardless of the cursor.

**Backend.** Write `x[r]` for the value of register `r` before the instruction, with `x[0] = 0`, and `{rs1, rs2, rd, imm}` for the instruction's `NormalizedOperands`.

- B1. **One row per executed instruction.** `instruction_index` indexes `decode_elf_with_mode(elf, RV64I, mode)?.instructions` for the backend's mode, by default `decode_elf(elf, RV64I)?.instructions`, and that instruction's address is `pc`. The first `pc` is the ELF entry; each later `pc` is the previous row's `next_pc`.
- B2. **Registers.** `rs1` and `rs2` are recorded exactly when the operands name them, with `x[r]`. `rd` is recorded exactly when named, with `x[rd]` before and the register's value after; for `rd` = x0 both are 0. All values before the instruction are captured before it executes, so with `rs1 = rd`, `rs2 = rd` or all three equal the read is the old value. A value accessor returns 0 for an unnamed register, and `registers()` distinguishes an unnamed register (`None`) from x0 (`Some`, register 0).
- B3. **RAM.** A load (LB, LBU, LH, LHU, LW, LWU, LD) or store (SB, SH, SW, SD) of width `w` has `ea = x[rs1].wrapping_add(imm as u64)` and `ram_address = ea & !7`, a byte address. The recorded doubleword is the eight bytes from `ram_address`, little-endian. A load is `RamAccess::Read` with `ram_pre_value == ram_post_value`, also when `rd` = x0. A store is `RamAccess::Write` whose post-value is the pre-value with bytes `ea & 7 .. (ea & 7) + w` replaced by the low `w` bytes of `x[rs2]`; it is computed, not read back. Every other instruction is `RamAccess::NoOp` with all three RAM words 0; a consumer takes no memory value from such a row, and the backend tracks no placeholder cell.
- B4. **Packing.** `SourceTraceRow::new` packs what it is given and validates nothing: `registers()` and `ram_access()` return its arguments unchanged, including a read or write at address 0, which differs from `NoOp`. B1 to B3 are guarantees of the backend, not of the type.
- B5. **Replay.** Start from x0 to x31 = 0 and a memory holding the loaded ELF bytes, the input and advice bytes at their layout addresses, and 0 elsewhere (RAM, output region, control cells). Applied in order, every row's `rs1_value`, `rs2_value`, `rd_pre_value` and `ram_pre_value` equal the replayed state, and its `rd_post_value` and `ram_post_value` give the next one. At the end the replayed RAM has exactly the non-zero bytes of `final_memory`, and the replayed output region equals `device.outputs` extended with zeros.
- B6. **Supported set.** A program is traced exactly when decode under `RV64I` in the backend's mode accepts it. Under `Strict`, a compressed encoding, a misaligned instruction or a kind outside the base extension is rejected at trace start as `TraceError::Program`; under `DataHoles` decode rejects no word of text, and such a program fails only if it fetches a hole (D2, D6). Of the 52 base kinds, ECALL and EBREAK return `UnsupportedInstruction { pc, kind }` when reached: the row cannot record a trap entry, and `EBREAK::exec` is a self-loop, not a breakpoint. The other 50 execute. FENCE is a row with no operands and `next_pc = pc + 4`, which is exact for one hart executing synchronously.
- B7. **Static program, exact fetch.** The instruction executed at `pc` is the decoded instruction whose address is `pc`. Any other `pc`, whether it is the entry point, a hole, a target that is not 4-aligned or an address outside the text, returns `PcOutsideProgram { pc }`. The PC is set to `pc.wrapping_add(4)` before the instruction executes, so JAL and JALR link to `pc + 4`. This rejects an execution the backend does not support; it does not emulate the instruction-address-misaligned trap. A store of width `w` whose bytes `[ea, ea + w)` overlap any decoded instruction returns `StoreToProgramText { pc, address: ea }`; a store that shares a doubleword with an instruction without overlapping it is ordinary.
- B8. **Alignment.** `ea % w != 0` returns `MisalignedAccess { pc, address: ea, width: w }`.
- B9. **Control cells.** The doublewords at `layout.panic` and `layout.termination` are write-once logical cells with initial value 0. The first store to a cell, of any width at any offset inside it, is a `Write` with pre-value 0 and the post-value of B3. A second store to that cell, and a load of any width at any offset in either cell, return `DeviceRegisterAccess { pc, address: ea }`. The emulator's device still receives the store (a store covering `layout.panic` sets `device.panic`, whatever the value); the cell is never read from the device. The input and advice regions are read-only and the output region is ordinary zero-initialised memory.
- B10. **Access recording.** With `record_access` on, `Mmu` behaves as today on every path. With it off, `trace_load`, `trace_store_byte`, `trace_store_halfword` and `trace_store` return a default record without reading memory, after the `assert_effective_store_address` call the three store helpers make. No other permission, bounds or alignment check changes. The backend reads the doubleword of B3 with `Mmu::load_doubleword_raw(ea & !7)`, which is in bounds at the last doubleword of RAM.
- B11. **Errors and termination.** Each check precedes the execution of the instruction it rejects, in the order `PcOutsideProgram`, `UnsupportedInstruction`, `MisalignedAccess`, `DeviceRegisterAccess`, `StoreToProgramText`; an error returns no partial trace. The trace ends with the first row whose `next_pc == pc`, emitted once. A store to the termination cell does not end it. The backend emits no padding.
- B12. **Text span.** The text span runs from the lowest decoded instruction address to the highest plus 4. A span above `1 << 28` bytes returns `ProgramTextTooLarge { span }` before the emulator is built, which bounds the slot table (as many bytes as the span) at 256 MiB. Slot and instruction indices are converted with `u32::try_from`.
- B13. **State equivalence.** Take an execution that both backends complete with `device.panic` false. Group the rows of `TracerBackend`: a row with `virtual_sequence_remaining == None` is a group; a row with `is_first_in_sequence` and `Some(n)` opens a group that runs through the row with `Some(0)`. Groups and source rows correspond in order with equal addresses. After each pair, x0 to x31 and the PC are equal; at the end, RAM and the device (`outputs`, `panic`) are equal. Virtual registers, `trace_len`, `executed_instrs` and the call stack are not compared.
- B14. `Cpu::tick`, `Cpu::tick_operate`, the decode cache, the instruction executors and `TracerBackend` are not edited.

**Decode mode.** A *text range* is one of the disjoint address ranges into which `decode_elf` merges the executable sections.

- D1. **Strict.** `Strict` is the decode of P1 to P6, unchanged for every input and profile. Under a profile without `Rv64C` it sweeps a text range by halfwords and knows no data. It skips a zero halfword (P6). Any other halfword that does not end in `0b11` is `IllegalCompressedInstruction` (P2). A halfword that ends in `0b11` opens a 32-bit word, which is `MalformedImage` at an address that is not a multiple of 4 (P2) or when it is no RV64 encoding, `IllegalSourceInstruction` when its kind is outside the profile (P4), and otherwise an instruction, whether or not the program executes it. A range that ends inside a halfword, or inside a word so opened, is `MalformedImage`. A doubleword at a 4-aligned address that holds a 4-aligned address from `RAM_START_ADDRESS` to below `2^32`, such as a pointer to an instruction, is therefore always rejected: at its low halfword if that is nonzero, otherwise at the next, which is nonzero and lies at an address that is 2 modulo 4.
- D2. **Slots.** Under `DataHoles` the *slots* of a text range `[start, end)` are the words `[a, a + 4)` with `a` a multiple of 4, `start <= a` and `a + 4 <= end`, read in address order as little-endian `u32`. No halfword is examined on its own, and nothing resynchronises. A slot is an instruction exactly when `decode_instruction(word, a, false, profile)` returns `Ok`. On `Err` it is a hole, like the padding of P6: it is omitted from `instructions` and the error is not returned. At a slot the error is `MalformedImage` (no RV64 encoding, which covers the zero word and every word that does not end in `0b11`) or `IllegalSourceInstruction` (a kind outside the profile). Bytes of the range in no slot, before the first multiple of 4 or in a tail shorter than four bytes, are omitted likewise, and the truncation errors of `Strict` do not arise. `instructions` stays dense and in address order.
- D3. **What the mode leaves alone.** `memory_init`, `program_end` and `entry_address` are the same in both modes; `memory_init`, filled before any text is decoded, keeps every byte of a hole and of a fragment. The errors that concern no word of text are returned in both: an invalid ELF object, an ELF32 image, a section extent that overflows the address space, unreadable section data.
- D4. **Agreement.** An image that `Strict` accepts under a profile without `Rv64C` decodes to the same `Rv64ProgramImage` under `DataHoles`. The mode identifies no data: a data word that is an encoding the profile admits is an instruction of the list, at its own index. A program that treats it as data never fetches it, and B7 protects it from stores like any instruction.
- D5. **Profile.** The slot rule presupposes instructions of one width at 4-aligned addresses, which P2 gives exactly for a profile without `Rv64C`. With a profile that contains `Rv64C`, `decode_elf_with_mode` in `DataHoles` returns `DecodeModeUnsupportedByProfile` before it parses the ELF. Under every other profile the rule applies as stated.
- D6. **One mode per trace.** The backend decodes in the mode it holds, and `instruction_index` indexes that list (B1): a consumer resolves it with `decode_elf_with_mode(elf, RV64I, mode)` in the same mode. By D4 the choice matters only for an image that `Strict` rejects. A hole is an absent PC, not an instruction that fails: its slot in the backend's table is empty, a fetch of it returns `PcOutsideProgram { pc }` (B7), a load from it reads the image bytes (B5), and a store that overlaps only holes is an ordinary row.

No `jolt-eval` invariant is added. B5 and B13 are checked by test code inside `tracer`, because B13 needs the private step function; lifting it behind a public lockstep API is a follow-up.

### Non-Goals

- Closing virtual-sequence expansion under `RV64I`: the expansion of SLL emits `MUL`, so `expand_program` returns `ExpansionError::IllegalTargetInstruction`.
- Applying a profile in the emulator's runtime decode (`Cpu::decode_and_cache`), or changing `JoltProgram`, `build_jolt_program` or any default profile.
- Kinds outside the base extension, compressed encodings, self-modifying code, ECALL (and with it every host call: printing, cycle markers, runtime advice) and EBREAK.
- Guests built by the SDK, which targets `riscv64imac-*`; the gate rejects them.
- Telling data from code, or executing a word outside the base set. `DataHoles` makes an image with data in its text decodable; a program that fetches a hole still stops there.
- A chunked form of this trace, and serialisation of the row.
- Typed errors for accesses the memory map forbids. An access outside the map or beyond `heap_end`, a store to the input or advice regions and a store into the stack canary panic in `Mmu::assert_effective_address`, as in `TracerBackend`.
- Fixing the existing over-read: `Mmu::trace_load` reads eight bytes from `ea & !3`, which passes the end of RAM when `ea` is in its last four bytes. In `TracerBackend` guest narrow loads and stores do not reach it (their `trace` walks an expansion made of doubleword accesses), but `Cpu::read_string` and `Cpu::handle_advice_write` do, through `Mmu::load`.

## Evaluation

### Acceptance Criteria

- [ ] Profile: for every `kind` in `SourceInstructionKind::ALL`, `RV64I.supports_source(kind)` equals membership in a literal list of the 52 base instructions, or `source_extension(kind).is_none()`; `RV64I.fingerprint()` differs from that of every other exported profile.
- [ ] Profile: one hand-encoded word per base instruction decodes under `RV64I` to its kind; `mul`, `amoadd.w`, `csrrw`, `mret` and an inline-opcode word each return `IllegalSourceInstruction` with the matching kind.
- [ ] Profile: `c.addi x1, 1` (`0x0085`) returns `IllegalCompressedInstruction` under `RV64I` and `RV64IM_JOLT` and decodes to `ADDI` under `RV64IMAC_JOLT`, through `decode_instruction` and `decode_elf`; a 32-bit `nop` at `0x8000_0002` returns `MalformedImage` under `RV64I` and decodes under `RV64IMAC_JOLT`.
- [ ] Profile: `decode_with_profile(elf, RV64I)` returns the three instructions and entry address of `addi; addi; jal x0, 0`, and `Err(IllegalSourceInstruction(MUL))` when `mul` is appended; `decode(elf)` returns the same tuple for the first image. A section extent that overflows the address space returns `MalformedImage` without a panic.
- [ ] Seam: the change is confined to `execution/backend.rs` and `execution/trace.rs`, and the workspace builds, lints and passes every existing test with no other edit.
- [ ] Seam: a test-local `ExecutionBackend<u64>` returning `[7, 11, 13]`, driven through `trace_with`, yields those rows from `rows()`, from `shared_rows()` (the same allocation) and from `next_row()`. After the first `next_row`, `TraceSource::rows` and `shared_rows` return `None`; after the last, `next_row` returns `None` on every call; `into_rows` returns the rows while another handle to the `Arc` is alive.
- [ ] Seam: `OwnedTrace<R>` constructs, defaults and clones for an `R` that is neither `Clone` nor `Default`.
- [ ] Backend: every line of the matrix below holds.
- [ ] Backend: B1 and B5 hold on every matrix program that completes. B13 holds on each of them and on 64 seeded programs (seeds 0 to 63: a prologue that sets a reserved base register, 64 instructions drawn from the ALU, LUI, AUIPC, forward-branch and aligned load and store kinds over one heap buffer, then `jal x0, 0`).
- [ ] Decode mode: every line of the image table below, after the matrix, holds for `decode_elf_with_mode` under `RV64I`. Wherever a mode decodes, `memory_init` is every byte of the section in address order, `program_end` is its end and the entry is `0x8000_0000`.
- [ ] Decode mode: `decode_elf_with_mode(elf, RV64IMAC_JOLT, DataHoles)` returns `DecodeModeUnsupportedByProfile` for the first image of that table and for bytes that are no ELF. Under `RV64IM_JOLT` in `DataHoles`, the `mul` image of the table lists three instructions.
- [ ] Decode mode: the `Strict` column is today's behaviour, and no existing test, fixture or expected value changes.

| Area | Cases | Expected |
|------|-------|----------|
| Smoke | `addi x1, x0, 1; addi x2, x1, 2; jal x0, 0` | Three literal rows: (index 0, x1: 0 to 1), (index 1, rs1 value 1, x2: 0 to 3), (index 2, `next_pc == pc`) |
| Row type | `size_of`, `align_of`; `new` then accessors on literal inputs | 80 and 8, asserted at compile time. Each input returns unchanged, and these are pairwise distinct rows: rs1 absent and rs1 = x0 with value 0; rd absent and rd = x0; `NoOp`, a read at address 0 and a write at address 0, all with zero values |
| ALU | Every register and immediate kind: negative immediates; rd = rs1, rd = rs2, rd = rs1 = rs2; rd = x0; shift amounts 31, 32, 63, 64; W-form results on both sides of bit 31 | Hand-computed rows |
| Upper | LUI and AUIPC with bit 31 of the immediate set and clear | Sign-extended value; AUIPC relative to `pc` |
| Branches | All six kinds on signed and unsigned boundary operands, forward and backward, taken and not | `next_pc` is the target or `pc + 4`. A branch not taken to a target that is not 4-aligned is a normal row; taken, the trace returns `PcOutsideProgram { pc: target }` |
| Jumps | JAL, JALR; rd = x0; JALR with rs1 = rd; JALR to an odd address; JAL and JALR to an address that is 2 mod 4 | Link is `pc + 4`; the target uses the old rs1; bit 0 is cleared; 2 mod 4 returns `PcOutsideProgram` |
| Loads | LB, LBU at 8 offsets, LH, LHU at 4, LW, LWU at 2, LD, over a doubleword of distinct bytes with sign bits set; rd = x0 for one kind of each width | `ram_address`, pre = post = the doubleword, rd sign- or zero-extended; with rd = x0, rd values are 0 and the RAM words are still recorded |
| Stores | SB at 8 offsets, SH at 4, SW at 2, SD, into a doubleword of non-zero bytes; two successive overlapping stores | Post keeps the untouched bytes; the second store's pre is the first one's post |
| End of RAM | Every load and store kind at `heap_end - w`, including `lb x0` | Completes with the hand-computed rows |
| Misalignment | Every load and store kind with `w > 1` at every misaligned offset | `MisalignedAccess { pc, address, width }`, no panic |
| Gate | `ecall` and `ebreak` reached; `ecall` present and not reached; `mul` in the image; `c.addi` in the image; `fence` | `UnsupportedInstruction` with the kind; traces; `Program(IllegalSourceInstruction(MUL))`; `Program(IllegalCompressedInstruction { .. })`; a row with no operands |
| Fetch | Image `[0, jal]`; `addi; 0x0000_0000; jal`; falling off the end; jumps below and above the text | `PcOutsideProgram` with the entry, `0x8000_0004`, the end, the target |
| Text stores | With the image's last instruction at an address `a` that is a multiple of 8: SD at `a`, SB at `a + 3`, SW at the entry; SW and SB at `a + 4` | `StoreToProgramText { pc, address }` for the first three; ordinary rows for the last two |
| Termination | `beq x0, x0, 0`; `jal x0, 0`; `jal x1, 0`; `jalr x0, 0(x5)` with x5 = `pc`; a termination store followed by more instructions | The loop instruction is the last row and appears once; `jal x1, 0` records x1 = `pc + 4`; the store does not end the trace |
| Control cells | First store at the panic base, of 1 and of 0; first SB at panic `+1`; first SD to termination; a second store of any width or offset; LB at `+3` and LD at the base of either cell, before or after a store | `Write` with pre 0 and the replaced post; `device.panic` true, true, false, false; `DeviceRegisterAccess` for the second store and for each load |
| I/O | Copy an input doubleword to the output region, store 1 to the termination byte, `jal x0, 0` | Three RAM records and `device.outputs` holding the bytes |
| Text span | Table built from two instruction addresses `1 << 28` apart | `ProgramTextTooLarge { span }` |
| Data holes | With `with_decode_mode(DataHoles)`: `auipc x5, 0; ld x6, 16(x5); addi x0, x0, 0; jal x0, 12; .dword 0x8000_0018; jal x0, 0`. The same with `jal x0, 12` replaced by `addi x0, x0, 0`. The first image in the default mode | Five literal rows with indices 0 to 4, at offsets 0, 4, 8, 12 and 24; the load is a `Read` at `0x8000_0010` of the doubleword `0x8000_0018`, which x6 receives. `PcOutsideProgram { pc: 0x8000_0010 }`. `Program(IllegalCompressedInstruction { address: 0x8000_0010 })` |

In the image table the text is one section at `0x8000_0000` unless stated, `addi` is `addi x1, x0, 1` at the section start, and `jal` is `jal x0, 0`.

| Text | `Strict` | `DataHoles` |
|------|----------|------------------|
| `addi`; `.dword 0x8000_0010`; `jal` | `IllegalCompressedInstruction { address: 0x8000_0004 }` | `addi`; `jal` at `0x8000_000c` |
| The same with `.dword 0x8001_0000`, and with `.dword 0x8003_0000` | `IllegalCompressedInstruction { address: 0x8000_0006 }`; `MalformedImage` (alignment) | As above |
| The same with `.dword 0x0000_0013_8000_0010` | As the first line | `addi`; `addi x0, x0, 0` at `0x8000_0008`; `jal` at `0x8000_000c` |
| `addi`; `mul x3, x1, x2`; `jal` | `IllegalSourceInstruction(MUL)` | `addi`; `jal` at `0x8000_0008` |
| `addi`; the word `0x0000_0085` (`c.addi x1, 1`, then a zero halfword); `jal` | `IllegalCompressedInstruction { address: 0x8000_0004 }` | `addi`; `jal` at `0x8000_0008` |
| Bytes `00 00`, `13 00 00 00`, `00 00`, then `jal` | `MalformedImage` (alignment: the `nop` is read at `0x8000_0002`) | `jal` at `0x8000_0008` |
| `addi`; `jal`; then a tail of `13 00`, and of `13` alone | `MalformedImage` (truncated word; truncated halfword) | `addi`; `jal` at `0x8000_0004`; the tail only in `memory_init` |
| `addi`; `0x0000_0000`; `jal`. A section at `0x8000_0002` holding `00 00`, then `jal` | `addi`, `jal` at `0x8000_0008`. `jal` at `0x8000_0004` | The same |

### Testing Strategy

Programs are `u32` words built into an ELF with `test_elf::build_elf64`, which becomes `pub` under the `test-utils` feature as well as `cfg(test)`; its caller outside the crate is the benchmark. Fixtures use a `program_size` that is a multiple of eight, so `heap_end - w` is the last aligned address, and form data addresses with LUI, ADDI and SLLI from the test's `MemoryLayout`. Expected rows are literals computed from the RISC-V unprivileged ISA manual; none is produced by the tracer.

The image cases of the decode mode are sections of literal bytes, built by the `build_elf64` and `text_section` of the test module in `image/elf.rs`: `test_elf::build_elf64` takes whole `u32` words, so it cannot express a tail fragment or a section that starts off a multiple of 4, and `jolt-program` does not depend on `tracer`. The traced programs use `test_elf::build_elf64` like the rest of the matrix, and the checks of B1, B5 and B13 decode in the backend's mode. Every expected list follows from the encodings; neither mode is the oracle for the other.

B13 is checked by a private harness that steps two emulators in lockstep, as `test_execute_trace_state_lockstep` does: one `Emulator::tick(Some(&mut rows))` against one source step, asserting that the tick's rows form exactly one group at the source `pc`, then comparing registers and PC. The seeded programs are a maintained boundary test of B13 and are secondary to the hand-computed vectors: they show that the two backends agree, not that either is right.

Every existing test passes unmodified, except that test-local `RV64I_ONLY` profile constants become `jolt_riscv::RV64I`. There is no ZK-mode surface; the workspace must still lint under `host` and `host,zk`, and `tracer` must build with `field-inline`, whose instruction variants `execute_direct` covers through the enum's `cfg` attributes.

### Performance

- [ ] `size_of::<SourceTraceRow>() == 80`, asserted at compile time: 320 MiB at 2^22 rows. Rows go straight into one vector, with no intermediate `Vec<Cycle>` and no conversion pass.
- [ ] After setup, the step loop performs no per-instruction temporary allocation. The row vector is reserved from the capacity hint and otherwise grows amortised; the device output buffer is reserved to `max_output_size` during setup. Full-backtrace capture (`JOLT_BACKTRACE=full`), which boxes the register file on each call, is excluded. Test: a `cfg(test)` counting global allocator with a thread-local counter brackets the private step loop, with capacity reserved, on a loop of 2^12 iterations containing ALU work, a `jal x1` call and `jalr` return, a heap store and load, and an SB to successive output bytes; the allocation count across the loop is zero.
- [ ] A new `jolt-eval` objective `source_trace_gen` (via `/new-objective`; `jolt-eval` enables `tracer/test-utils`) runs three hand-assembled programs of about 2^22 executed instructions each: an ALU and branch loop, a mix of loads and stores of every width over a heap buffer, and a call and return loop that pushes and pops a stack frame. Criterion ids per program: `source` (`SourceTracerBackend`, capacity reserved), `reference` (`raw_trace_cycles` on the same ELF, `TRACER_PARALLEL` unset) and `scan` (one pass over `rows()` that wrapping-adds the ten numeric accessors of every row into a `u64` passed to `black_box`). Every id sets `Throughput::Elements` to the source row count, so `reference` is normalised per executed source instruction and not per row it emits. Figures are Criterion medians of 10 flat samples after warm-up, release profile, one thread, on an Apple M4 Max, recorded in the PR description. Gates: `source` is not slower than `reference` on any program, and `scan` costs at most 10 ns per row (8 GB/s).
- [ ] The existing path is cost-neutral: seam dispatch stays static, the `Mmu` switch adds one predictable branch to four helpers, and the `reference` ids of `trace_gen_fibonacci` and `trace_gen_sha2_chain` stay within run-to-run noise on the same machine.

The decode gate is off every hot path: decode runs once per program and gains one slice search and one remainder per instruction. So is the decode mode: `DataHoles` replaces the halfword sweep by one `decode_instruction` call per four bytes of text, `Strict` is the default, and the step loop, the slot table and the row are the same in both modes, so no benchmark id moves. A consumer reads `&[SourceTraceRow]` from `TraceSource::rows()`, or the `Arc` from `shared_rows()`, and splits it into parallel chunks without cloning. Each row carries its instruction index, its `next_pc`, and the old and new values of `rd` and of the RAM doubleword, so a chunk needs neither a replay of memory nor, beyond the previous row's `next_pc`, its neighbour.

## Design

### Architecture

`decode_instruction` is the single place where a profile is applied, so the compressed and alignment checks join the kind check there, and the backend's legality check is that gate under `RV64I` rather than a second list. `RV64IM_JOLT` changes behaviour: it now rejects compressed encodings; it has no caller outside tests.

The decode mode changes `decode_text_section` alone, after `decode_elf_with_mode` has checked D5. A hole is whatever `decode_instruction` rejects at a slot, so the mode adds no second list of encodings, and the backend's table already represents an absent instruction as an empty slot, so a hole adds no state there.

At trace start the backend decodes, builds a dense table of `u32` slots over the text span (one slot per four bytes, holding the instruction index or an empty marker), builds the emulator with `create_emulator`, turns access recording off and reserves the output buffer and the row vector. Each step then:

1. looks `pc` up in the table and rejects ECALL and EBREAK;
2. captures `x[rs1]`, `x[rs2]` and `x[rd]`;
3. for a load or store, computes `ea`, applies B8 and B9, and for a store compares `[ea, ea + w)` with the text span, consulting the one or two covered slots only inside it; then reads the doubleword at `ea & !7`, except in a control cell;
4. sets `cpu.pc = pc.wrapping_add(4)`, calls `execute_direct`, resets x0;
5. pushes the row with `next_pc = cpu.pc` and the new `x[rd]`, adds one to `cpu.trace_len` so panic diagnostics count source instructions, and stops if `next_pc == pc`.

The step replaces `Cpu::tick`. What it omits (trap entry, the pending-interrupt check, `wfi`) is unreachable without ECALL, CSR and WFI instructions. Teardown is `finish_emulator`, and the device, final memory and advice tape are returned as `TracerBackend` returns them. Kind, immediate and operand layout stay in the decoded image, referenced by `instruction_index`.

`record_access` is set in `Mmu::new`, copied by `save_state_with_empty_memory` and ignored by `capture_chunk_state`. The four tag bytes hold three register numbers and one byte of presence bits and RAM kind; the layout is private and pinned only by the round-trip test.

### Alternatives Considered

- **Profile.** A flag that disables expansion (expansion belongs to the pipeline that consumes the image, and the flag would enter the fingerprint); a profile parameter on `tracer::decode` (breaks a public signature); enforcing `Rv64C` in `decode_text_section` (a second enforcement point beside the public `decode_instruction`).
- **Seam.** A sibling trait and owned-trace type (a parallel copy of the seam); an inherent method on `TracerBackend`, as `trace_compact` (binds consumers to one backend); `ExecutionBackend<SourceTraceRow>` on `TracerBackend` itself (existing `backend.trace(..)` calls become ambiguous); a flag in `Cpu::tick_operate` (a branch in the existing per-instruction loop).
- **Legality.** Decoding under `program.profile` and re-checking each instruction against `RV64I` (a second pass that must also re-derive the alignment rule, since a profile with `Rv64C` admits a 32-bit instruction at an address that is 2 mod 4).
- **Decode mode.** Marking data by the symbol table or by the mapping symbols `$d` and `$x` (both are optional metadata: hand-written assembly seldom gives its data a symbol type or size, a stripped image has no symbols at all, and decode would still need a rule for an image without them); a flag per section, or a list of section names to exempt (the data shares its section with the code that jumps over it, and a name is one linker script's convention); decoding lazily at fetch (the instruction list would depend on the execution, so `instruction_index` would not name a fixed image, a consumer could not resolve it without the trace, and B7 could not protect an instruction not yet fetched); holes under `Strict` itself (every existing caller would lose the rejection of B6 at trace start); a field of the profile (P5, and the mode would enter the fingerprint).
- **Access records.** Splitting `Mmu::load` and `store` into a checked access and a record, with executors calling the former (every narrow executor changes or is duplicated, so ISA semantics gain a second definition; the switch leaves them untouched).
- **Program text.** Re-decoding after a store, as the reference emulator's decode cache does (`instruction_index` would no longer name a fixed image); an associative map keyed by PC (a probe on every step where the dense table costs one bounds check and one load).
- **EBREAK as termination.** Its executor already stalls the PC, but that is not what the instruction means, and termination has one definition without it.
- **Row.** A 64-byte row that aliases the `rd` and RAM slots (saves 64 MiB at 2^22 rows, makes every read depend on the instruction class; left open by the private fields).

## Documentation

No book change. Rustdoc on `RV64I` states its contents; rustdoc on `SourceTraceRow`, `SourceTraceError` and `SourceTracerBackend` states invariants B1 to B12; rustdoc on `DecodeMode` states D2 to D5, and on `with_decode_mode` D6.

## References

- The RISC-V Instruction Set Manual, Volume I: Unprivileged Architecture (source of the expected rows).
- `specs/x86-tracer-backend.md` (benchmark and equivalence conventions for execution backends).
- `specs/act4-tests.md` (the architectural tests whose images motivate the decode mode).
