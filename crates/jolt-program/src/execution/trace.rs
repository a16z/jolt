use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_riscv::{
    CircuitFlagSet, CircuitFlags, Flags, InstructionFlagSet, JoltCycle, JoltInstruction,
    JoltInstructionKind, JoltInstructionProfile, JoltInstructionRow, JoltInstructionTag,
    NormalizedOperands, RV64IMAC_JOLT,
};
use std::sync::Arc;

#[cfg(feature = "field-inline")]
use crate::field_inline::FieldInlineTraceData;

use super::{ExecutionBackend, TraceError, TraceSource};

/// A Jolt-ready program built from an RV64 ELF image.
///
/// This is the stage after `Rv64ProgramImage`: decoded RV64 instruction rows
/// have been expanded into the bytecode used by Jolt preprocessing, while the
/// original ELF bytes are still kept for backends that run the source program
/// from its ELF image.
#[derive(Debug, Clone)]
pub struct JoltProgram {
    elf_bytes: Vec<u8>,
    /// Final Jolt bytecode rows after expanding decoded RV64 instructions.
    pub expanded_bytecode: Vec<JoltInstructionRow>,
    /// Initial byte values for memory-backed ELF sections.
    pub memory_init: Vec<(u64, u8)>,
    /// End address of the loaded program image.
    pub program_end: u64,
    /// ELF entry point.
    pub entry_address: u64,
    /// Selected instruction legality/profile for this program.
    pub profile: JoltInstructionProfile,
}

impl Default for JoltProgram {
    fn default() -> Self {
        Self::from_elf_bytes(Vec::new())
    }
}

impl JoltProgram {
    pub fn from_elf_bytes(elf_bytes: Vec<u8>) -> Self {
        Self {
            elf_bytes,
            expanded_bytecode: Vec::new(),
            memory_init: Vec::new(),
            program_end: 0,
            entry_address: 0,
            profile: RV64IMAC_JOLT,
        }
    }

    pub fn from_parts(
        elf_bytes: Vec<u8>,
        expanded_bytecode: Vec<JoltInstructionRow>,
        memory_init: Vec<(u64, u8)>,
        program_end: u64,
        entry_address: u64,
    ) -> Self {
        Self::from_parts_with_profile(
            elf_bytes,
            expanded_bytecode,
            memory_init,
            program_end,
            entry_address,
            RV64IMAC_JOLT,
        )
    }

    pub fn from_parts_with_profile(
        elf_bytes: Vec<u8>,
        expanded_bytecode: Vec<JoltInstructionRow>,
        memory_init: Vec<(u64, u8)>,
        program_end: u64,
        entry_address: u64,
        profile: JoltInstructionProfile,
    ) -> Self {
        Self {
            elf_bytes,
            expanded_bytecode,
            memory_init,
            program_end,
            entry_address,
            profile,
        }
    }

    /// Creates a Jolt program from an RV64 program image and its expanded bytecode.
    ///
    /// `Rv64ProgramImage` contains the rows and memory decoded directly from
    /// the ELF. The caller supplies `expanded_bytecode`, which is the result of
    /// expanding those decoded rows into the bytecode used by Jolt.
    #[cfg(feature = "image")]
    pub fn from_rv64_image(
        elf_bytes: Vec<u8>,
        expanded_bytecode: Vec<JoltInstructionRow>,
        image: crate::image::Rv64ProgramImage,
    ) -> Self {
        Self::from_rv64_image_with_profile(elf_bytes, expanded_bytecode, image, RV64IMAC_JOLT)
    }

    #[cfg(feature = "image")]
    pub fn from_rv64_image_with_profile(
        elf_bytes: Vec<u8>,
        expanded_bytecode: Vec<JoltInstructionRow>,
        image: crate::image::Rv64ProgramImage,
        profile: JoltInstructionProfile,
    ) -> Self {
        Self::from_parts_with_profile(
            elf_bytes,
            expanded_bytecode,
            image.memory_init,
            image.program_end,
            image.entry_address,
            profile,
        )
    }

    pub fn elf_bytes(&self) -> &[u8] {
        &self.elf_bytes
    }

    pub fn trace_with<B: ExecutionBackend>(
        &self,
        backend: &mut B,
        inputs: TraceInputs,
    ) -> Result<TraceOutput<B::Trace>, TraceError> {
        backend.trace(self, inputs)
    }
}

#[derive(Default, Debug, Clone)]
pub struct TraceInputs {
    pub inputs: Vec<u8>,
    pub untrusted_advice: Vec<u8>,
    pub trusted_advice: Vec<u8>,
    pub memory_config: MemoryConfig,
    /// Runtime advice tape to seed execution with (the SDK's two-pass advice
    /// flow: pass 1 populates the tape, pass 2 consumes it). Read cursor
    /// always starts at 0.
    pub advice_tape: Option<Vec<u8>>,
}

impl TraceInputs {
    pub fn new(
        inputs: Vec<u8>,
        untrusted_advice: Vec<u8>,
        trusted_advice: Vec<u8>,
        memory_config: MemoryConfig,
    ) -> Self {
        Self {
            inputs,
            untrusted_advice,
            trusted_advice,
            memory_config,
            advice_tape: None,
        }
    }

    pub fn with_advice_tape(mut self, advice_tape: Option<Vec<u8>>) -> Self {
        self.advice_tape = advice_tape;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RegisterRead {
    pub register: u8,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RegisterWrite {
    pub register: u8,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RegisterState {
    pub rs1: Option<RegisterRead>,
    pub rs2: Option<RegisterRead>,
    pub rd: Option<RegisterWrite>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RamRead {
    pub address: u64,
    pub value: u64,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RamWrite {
    pub address: u64,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum RamAccess {
    Read(RamRead),
    Write(RamWrite),
    #[default]
    NoOp,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct MemoryImage {
    pub bytes: Vec<(u64, u8)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum RamAccessKind {
    NoOp,
    Read,
    Write,
}

/// Presence bits, RAM access kind, and immediate sign packed into one byte.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize),
    serde(transparent)
)]
#[repr(transparent)]
struct TraceRowMeta(u8);

impl TraceRowMeta {
    const RS1_PRESENT: u8 = 1 << 0;
    const RS2_PRESENT: u8 = 1 << 1;
    const RD_PRESENT: u8 = 1 << 2;
    const RAM_ACCESS_SHIFT: u8 = 3;
    const RAM_ACCESS_MASK: u8 = 0b11 << Self::RAM_ACCESS_SHIFT;
    const IMM_NEGATIVE: u8 = 1 << 5;

    fn new(registers: &RegisterState, ram_access: RamAccessKind, imm_negative: bool) -> Self {
        let mut bits = (ram_access as u8) << Self::RAM_ACCESS_SHIFT;
        bits |= Self::RS1_PRESENT * u8::from(registers.rs1.is_some());
        bits |= Self::RS2_PRESENT * u8::from(registers.rs2.is_some());
        bits |= Self::RD_PRESENT * u8::from(registers.rd.is_some());
        bits |= Self::IMM_NEGATIVE * u8::from(imm_negative);
        Self(bits)
    }

    #[inline]
    fn rs1_present(self) -> bool {
        self.0 & Self::RS1_PRESENT != 0
    }

    #[inline]
    fn rs2_present(self) -> bool {
        self.0 & Self::RS2_PRESENT != 0
    }

    #[inline]
    fn rd_present(self) -> bool {
        self.0 & Self::RD_PRESENT != 0
    }

    #[inline]
    fn ram_access(self) -> RamAccessKind {
        match (self.0 & Self::RAM_ACCESS_MASK) >> Self::RAM_ACCESS_SHIFT {
            1 => RamAccessKind::Read,
            2 => RamAccessKind::Write,
            _ => RamAccessKind::NoOp,
        }
    }

    #[inline]
    fn apply_imm_sign(self, magnitude: u64) -> i128 {
        let magnitude = magnitude as i128;
        if self.0 & Self::IMM_NEGATIVE != 0 {
            -magnitude
        } else {
            magnitude
        }
    }
}

/// Cached row class selecting the slot layout.
const IS_LOAD: u16 = 1 << CircuitFlags::Load as u16;
const IS_STORE: u16 = 1 << CircuitFlags::Store as u16;

/// `virtual_sequence_remaining` sentinel for `None`.
const VSR_NONE: u16 = u16::MAX;
/// Operand-id byte sentinel for an absent (`None`) operand.
const OPERAND_NONE: u8 = u8::MAX;

/// One execution cycle packed to 64 bytes. Four value slots use the final
/// memory-row layout (`specs/proof-trace-row-layout.md`):
///
/// - non-memory: `rs1`, `rs2`, `rd_pre`, `rd_post` (RAM values must be zero);
/// - load: `rs1`, `ram_address`, `rd_pre`, `rd_post` (= the RAM value; no rs2);
/// - store: `rs1`, `rs2` (= RAM post), `ram_pre`, `ram_address` (no rd).
///
/// Recorded register ids stay separate from instruction operand ids.
/// [`TraceRow::new`] rejects rows that violate the slot contract.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(feature = "field-inline"), derive(Copy))]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
#[repr(C)]
pub struct TraceRow {
    slots: [u64; 4],
    address: u64,
    /// Immediate magnitude; `meta` stores its sign.
    imm_abs: u64,
    /// Cached from the stored instruction metadata for fast witness extraction.
    circuit_flags: u16,
    kind_tag: u16,
    virtual_sequence_remaining: u16,
    instruction_flags: u8,
    meta: TraceRowMeta,
    /// Valid when the matching `meta` bit is set.
    rs1_register: u8,
    rs2_register: u8,
    rd_register: u8,
    /// Instruction operand ids; [`OPERAND_NONE`] means absent.
    rs1_operand: u8,
    rs2_operand: u8,
    rd_operand: u8,
    #[cfg(feature = "field-inline")]
    pub field_inline: Option<Arc<FieldInlineTraceData>>,
}

#[cfg(not(feature = "field-inline"))]
const _: () = assert!(
    std::mem::size_of::<TraceRow>() == 64,
    "TraceRow must stay 64 bytes; any size change should be intentional and reviewed"
);

impl Default for TraceRow {
    fn default() -> Self {
        Self::from_instruction(JoltInstructionRow::default())
    }
}

#[cold]
#[inline(never)]
#[expect(
    clippy::panic,
    reason = "fail-closed: silently packing a non-conforming row would corrupt aliased columns"
)]
fn contract_violation(kind: JoltInstructionKind, detail: &str) -> ! {
    panic!("trace row for {kind:?} violates the final memory-row contract: {detail}");
}

#[inline]
fn checked_operand_id(kind: JoltInstructionKind, id: Option<u8>) -> u8 {
    match id {
        None => OPERAND_NONE,
        Some(id) if id < OPERAND_NONE => id,
        Some(id) => contract_violation(kind, &format!("operand register id {id} is reserved")),
    }
}

impl TraceRow {
    /// Pack one cycle, rejecting invalid slot aliases.
    pub fn new(
        instruction: JoltInstructionRow,
        registers: RegisterState,
        ram_access: RamAccess,
    ) -> Self {
        let kind = instruction.instruction_kind;
        let (circuit_flags, instruction_flags) = match JoltInstruction::try_from(instruction) {
            Ok(instruction) => (instruction.circuit_flags(), instruction.instruction_flags()),
            // Source-only kinds have no flag semantics to cache.
            Err(_) => contract_violation(kind, "instruction kind has no JoltInstruction lowering"),
        };
        let is_load = circuit_flags.get(CircuitFlags::Load);
        let is_store = circuit_flags.get(CircuitFlags::Store);

        let rs1 = registers.rs1.unwrap_or(RegisterRead {
            register: 0,
            value: 0,
        });
        let rs2 = registers.rs2.unwrap_or(RegisterRead {
            register: 0,
            value: 0,
        });
        let rd = registers.rd.unwrap_or(RegisterWrite {
            register: 0,
            pre_value: 0,
            post_value: 0,
        });
        let (ram_kind, ram_address, ram_pre, ram_post) = match ram_access {
            RamAccess::Read(read) => (RamAccessKind::Read, read.address, read.value, read.value),
            RamAccess::Write(write) => (
                RamAccessKind::Write,
                write.address,
                write.pre_value,
                write.post_value,
            ),
            RamAccess::NoOp => (RamAccessKind::NoOp, 0, 0, 0),
        };

        let slots = if is_load {
            if registers.rs2.is_some() {
                contract_violation(kind, "load row reads rs2");
            }
            if ram_kind != RamAccessKind::NoOp
                && (ram_pre != rd.post_value || ram_post != rd.post_value)
            {
                contract_violation(kind, "load RAM value must equal the rd write value");
            }
            [rs1.value, ram_address, rd.pre_value, rd.post_value]
        } else if is_store {
            if registers.rd.is_some() {
                contract_violation(kind, "store row writes rd");
            }
            if ram_kind == RamAccessKind::Write && ram_post != rs2.value {
                contract_violation(kind, "store RAM write value must equal the rs2 value");
            }
            [rs1.value, rs2.value, ram_pre, ram_address]
        } else {
            if ram_address != 0 || ram_pre != 0 || ram_post != 0 {
                contract_violation(kind, "non-memory row carries RAM values");
            }
            [rs1.value, rs2.value, rd.pre_value, rd.post_value]
        };

        let imm = instruction.operands.imm;
        let Ok(imm_abs) = u64::try_from(imm.unsigned_abs()) else {
            contract_violation(kind, "immediate does not fit the u64 magnitude encoding");
        };
        let meta = TraceRowMeta::new(&registers, ram_kind, imm < 0);
        let virtual_sequence_remaining = match instruction.virtual_sequence_remaining {
            None => VSR_NONE,
            Some(VSR_NONE) => contract_violation(
                kind,
                "virtual_sequence_remaining collides with the sentinel",
            ),
            Some(remaining) => remaining,
        };

        Self {
            slots,
            address: instruction.address as u64,
            imm_abs,
            circuit_flags: circuit_flags.bits(),
            kind_tag: kind.tag().0,
            virtual_sequence_remaining,
            instruction_flags: instruction_flags.bits(),
            meta,
            rs1_register: rs1.register,
            rs2_register: rs2.register,
            rd_register: rd.register,
            rs1_operand: checked_operand_id(kind, instruction.operands.rs1),
            rs2_operand: checked_operand_id(kind, instruction.operands.rs2),
            rd_operand: checked_operand_id(kind, instruction.operands.rd),
            #[cfg(feature = "field-inline")]
            field_inline: None,
        }
    }

    pub fn from_instruction(instruction: JoltInstructionRow) -> Self {
        Self::new(instruction, RegisterState::default(), RamAccess::NoOp)
    }

    /// Reconstruct the instruction row.
    #[inline]
    pub fn instruction(&self) -> JoltInstructionRow {
        #[inline]
        fn operand(id: u8) -> Option<u8> {
            (id != OPERAND_NONE).then_some(id)
        }
        JoltInstructionRow {
            instruction_kind: self.instruction_kind(),
            address: self.address as usize,
            operands: NormalizedOperands {
                rs1: operand(self.rs1_operand),
                rs2: operand(self.rs2_operand),
                rd: operand(self.rd_operand),
                imm: self.meta.apply_imm_sign(self.imm_abs),
            },
            virtual_sequence_remaining: (self.virtual_sequence_remaining != VSR_NONE)
                .then_some(self.virtual_sequence_remaining),
            is_first_in_sequence: self.circuit_flags
                & (1 << CircuitFlags::IsFirstInSequence as u16)
                != 0,
            is_compressed: self.circuit_flags & (1 << CircuitFlags::IsCompressed as u16) != 0,
        }
    }

    /// Cached circuit flags.
    #[inline]
    pub fn circuit_flags(&self) -> CircuitFlagSet {
        CircuitFlagSet::from_bits(self.circuit_flags)
    }

    /// The row's cached instruction flags.
    #[inline]
    pub fn instruction_flags(&self) -> InstructionFlagSet {
        InstructionFlagSet::from_bits(self.instruction_flags)
    }

    /// Final instruction kind without reconstructing the row.
    #[inline]
    #[expect(clippy::expect_used, reason = "the tag is stored from a valid kind")]
    pub fn instruction_kind(&self) -> JoltInstructionKind {
        JoltInstructionKind::from_tag(JoltInstructionTag(self.kind_tag))
            .expect("trace row kind tag was stored from a valid instruction kind")
    }

    /// Whether the row's instruction is the canonical `NoOp`.
    #[inline]
    pub fn is_noop(&self) -> bool {
        self.kind_tag == const { JoltInstructionKind::NoOp.tag().0 }
    }

    /// Source RV64 instruction address (`instruction().address`).
    #[inline]
    pub fn address(&self) -> u64 {
        self.address
    }

    /// The instruction's immediate operand (`instruction().operands.imm`).
    #[inline]
    pub fn imm(&self) -> i128 {
        self.meta.apply_imm_sign(self.imm_abs)
    }

    #[inline]
    fn ram_slots(&self) -> (u64, u64, u64) {
        if self.circuit_flags & IS_LOAD != 0 {
            (self.slots[1], self.slots[3], self.slots[3])
        } else if self.circuit_flags & IS_STORE != 0 {
            (self.slots[3], self.slots[2], self.slots[1])
        } else {
            (0, 0, 0)
        }
    }

    #[inline]
    pub fn rs1_read(&self) -> Option<RegisterRead> {
        self.meta.rs1_present().then_some(RegisterRead {
            register: self.rs1_register,
            value: self.slots[0],
        })
    }

    #[inline]
    pub fn rs2_read(&self) -> Option<RegisterRead> {
        // Loads cannot reach this path with rs2 present.
        self.meta.rs2_present().then_some(RegisterRead {
            register: self.rs2_register,
            value: self.slots[1],
        })
    }

    #[inline]
    pub fn rd_write(&self) -> Option<RegisterWrite> {
        self.meta.rd_present().then_some(RegisterWrite {
            register: self.rd_register,
            pre_value: self.slots[2],
            post_value: self.slots[3],
        })
    }

    #[inline]
    pub fn registers(&self) -> RegisterState {
        RegisterState {
            rs1: self.rs1_read(),
            rs2: self.rs2_read(),
            rd: self.rd_write(),
        }
    }

    #[inline]
    pub fn ram_access(&self) -> RamAccess {
        let (address, pre_value, post_value) = self.ram_slots();
        match self.meta.ram_access() {
            RamAccessKind::Read => RamAccess::Read(RamRead {
                address,
                value: pre_value,
            }),
            RamAccessKind::Write => RamAccess::Write(RamWrite {
                address,
                pre_value,
                post_value,
            }),
            RamAccessKind::NoOp => RamAccess::NoOp,
        }
    }
}

impl JoltCycle for TraceRow {
    type Instruction = JoltInstructionRow;

    #[inline]
    fn instruction(&self) -> Self::Instruction {
        TraceRow::instruction(self)
    }

    #[inline]
    fn rs1_val(&self) -> Option<u64> {
        self.meta.rs1_present().then_some(self.slots[0])
    }

    #[inline]
    fn rs2_val(&self) -> Option<u64> {
        self.meta.rs2_present().then_some(self.slots[1])
    }

    #[inline]
    fn rd_vals(&self) -> Option<(u64, u64)> {
        self.meta
            .rd_present()
            .then_some((self.slots[2], self.slots[3]))
    }

    #[inline]
    fn ram_access_address(&self) -> Option<u64> {
        (self.meta.ram_access() != RamAccessKind::NoOp).then_some(self.ram_slots().0)
    }

    #[inline]
    fn ram_read_value(&self) -> Option<u64> {
        (self.meta.ram_access() != RamAccessKind::NoOp).then_some(self.ram_slots().1)
    }

    #[inline]
    fn ram_write_value(&self) -> Option<u64> {
        (self.meta.ram_access() == RamAccessKind::Write).then_some(self.ram_slots().2)
    }
}

#[derive(Debug, Clone)]
pub struct TraceOutput<T> {
    pub trace: T,
    pub device: JoltDevice,
    pub final_memory: Option<MemoryImage>,
    /// The populated runtime advice tape captured at guest termination
    /// (`None` when the backend produced no tape).
    pub advice_tape: Option<Vec<u8>>,
}

impl<T> TraceOutput<T> {
    /// `advice_tape` is a required parameter so that a backend (or a
    /// rebuild of an existing output) cannot silently discard a populated
    /// tape — the seam this field exists to plug.
    pub fn new(
        trace: T,
        device: JoltDevice,
        final_memory: Option<MemoryImage>,
        advice_tape: Option<Vec<u8>>,
    ) -> Self {
        Self {
            trace,
            device,
            final_memory,
            advice_tape,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct OwnedTrace {
    rows: Arc<Vec<TraceRow>>,
    next: usize,
}

impl OwnedTrace {
    pub fn new(rows: Vec<TraceRow>) -> Self {
        Self {
            rows: Arc::new(rows),
            next: 0,
        }
    }

    pub fn rows(&self) -> &[TraceRow] {
        self.rows.as_slice()
    }

    pub fn into_rows(self) -> Vec<TraceRow> {
        match Arc::try_unwrap(self.rows) {
            Ok(rows) => rows,
            Err(rows) => rows.as_ref().clone(),
        }
    }
}

impl From<Vec<TraceRow>> for OwnedTrace {
    fn from(rows: Vec<TraceRow>) -> Self {
        Self::new(rows)
    }
}

impl TraceSource for OwnedTrace {
    fn next_row(&mut self) -> Option<TraceRow> {
        // `TraceRow` is `Copy` only without `field-inline` (which adds a non-`Copy` `Arc`
        // field), so the row is copied or cloned to match the active build.
        #[cfg(not(feature = "field-inline"))]
        let row = self.rows.get(self.next).copied();
        #[cfg(feature = "field-inline")]
        let row = self.rows.get(self.next).cloned();
        self.next += usize::from(row.is_some());
        row
    }

    fn rows(&self) -> Option<&[TraceRow]> {
        // Pristine sources only: after `next_row` consumption the full slice
        // would diverge from the remaining stream.
        (self.next == 0).then(|| self.rows.as_slice())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;

    fn instruction(kind: JoltInstructionKind, operands: NormalizedOperands) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind: kind,
            address: 0x8000_0000,
            operands,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    /// Preserve presence and RAM variants even when values coincide.
    #[test]
    fn non_memory_state_round_trips_exactly() {
        let register_states = [
            RegisterState::default(),
            RegisterState {
                rs1: Some(RegisterRead {
                    register: 0,
                    value: 0,
                }),
                rs2: Some(RegisterRead {
                    register: 63,
                    value: u64::MAX,
                }),
                rd: Some(RegisterWrite {
                    register: 5,
                    pre_value: 7,
                    post_value: 7,
                }),
            },
            RegisterState {
                rs1: None,
                rs2: None,
                rd: Some(RegisterWrite {
                    register: 0,
                    pre_value: u64::MAX,
                    post_value: 0,
                }),
            },
        ];
        // Zero-valued accesses must keep their variant.
        let ram_accesses = [
            RamAccess::NoOp,
            RamAccess::Read(RamRead {
                address: 0,
                value: 0,
            }),
            RamAccess::Write(RamWrite {
                address: 0,
                pre_value: 0,
                post_value: 0,
            }),
        ];
        for registers in register_states {
            for ram_access in ram_accesses {
                let row = TraceRow::new(JoltInstructionRow::default(), registers, ram_access);
                assert_eq!(row.registers(), registers);
                assert_eq!(row.ram_access(), ram_access);
                assert_eq!(row.rs1_read(), registers.rs1);
                assert_eq!(row.rs2_read(), registers.rs2);
                assert_eq!(row.rd_write(), registers.rd);
            }
        }
    }

    #[test]
    fn load_row_round_trips_and_aliases() {
        let loaded = 0xdead_beefu64;
        let registers = RegisterState {
            rs1: Some(RegisterRead {
                register: 10,
                value: 0x1000,
            }),
            rs2: None,
            rd: Some(RegisterWrite {
                register: 11,
                pre_value: 5,
                post_value: loaded,
            }),
        };
        let ram_access = RamAccess::Read(RamRead {
            address: 0x2000,
            value: loaded,
        });
        let row = TraceRow::new(
            instruction(
                JoltInstructionKind::LD,
                NormalizedOperands {
                    rs1: Some(10),
                    rs2: None,
                    rd: Some(11),
                    imm: 8,
                },
            ),
            registers,
            ram_access,
        );
        assert_eq!(row.registers(), registers);
        assert_eq!(row.ram_access(), ram_access);
        assert_eq!(row.imm(), 8);
        assert_eq!(JoltCycle::ram_read_value(&row), Some(loaded));
        assert_eq!(JoltCycle::ram_write_value(&row), None);
        assert_eq!(JoltCycle::ram_access_address(&row), Some(0x2000));
    }

    #[test]
    fn store_row_round_trips_and_aliases() {
        let stored = 0x1234u64;
        let registers = RegisterState {
            rs1: Some(RegisterRead {
                register: 10,
                value: 0x3000,
            }),
            rs2: Some(RegisterRead {
                register: 12,
                value: stored,
            }),
            rd: None,
        };
        let ram_access = RamAccess::Write(RamWrite {
            address: 0x4000,
            pre_value: 0x5678,
            post_value: stored,
        });
        let row = TraceRow::new(
            instruction(
                JoltInstructionKind::SD,
                NormalizedOperands {
                    rs1: Some(10),
                    rs2: Some(12),
                    rd: None,
                    imm: -4,
                },
            ),
            registers,
            ram_access,
        );
        assert_eq!(row.registers(), registers);
        assert_eq!(row.ram_access(), ram_access);
        assert_eq!(row.imm(), -4);
        assert_eq!(JoltCycle::ram_read_value(&row), Some(0x5678));
        assert_eq!(JoltCycle::ram_write_value(&row), Some(stored));
    }

    /// Cached flags match instruction-derived flags.
    #[test]
    fn cached_flags_match_the_derivation() {
        let mut source = instruction(
            JoltInstructionKind::SD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: None,
                imm: 0,
            },
        );
        source.virtual_sequence_remaining = Some(0);
        source.is_compressed = true;
        for instruction in [source, JoltInstructionRow::default()] {
            let row = TraceRow::from_instruction(instruction);
            let decoded = JoltInstruction::try_from(row.instruction()).unwrap();
            assert_eq!(row.circuit_flags(), decoded.circuit_flags());
            assert_eq!(row.instruction_flags(), decoded.instruction_flags());
        }
    }

    /// Instruction operands remain independent of recorded register state.
    #[test]
    fn instruction_reconstruction_is_exact() {
        let mut source = instruction(
            JoltInstructionKind::ADDI,
            NormalizedOperands {
                rs1: Some(2),
                rs2: None,
                rd: Some(1),
                imm: -12345,
            },
        );
        source.virtual_sequence_remaining = Some(3);
        source.is_first_in_sequence = true;
        source.is_compressed = true;
        let row = TraceRow::new(
            source,
            RegisterState {
                rs1: Some(RegisterRead {
                    register: 200,
                    value: 9,
                }),
                rs2: None,
                rd: None,
            },
            RamAccess::NoOp,
        );
        assert_eq!(row.instruction(), source);
        assert_eq!(row.instruction_kind(), JoltInstructionKind::ADDI);
        assert_eq!(row.address(), 0x8000_0000);
        assert_eq!(row.imm(), -12345);
        // Recorded register ids stay independent of operand ids.
        assert_eq!(row.rs1_read().unwrap().register, 200);
    }

    /// Default is the canonical padding row.
    #[test]
    fn default_row_is_the_canonical_padding_row() {
        let row = TraceRow::default();
        assert_eq!(row.registers(), RegisterState::default());
        assert_eq!(row.ram_access(), RamAccess::NoOp);
        assert!(row.is_noop());
        assert_eq!(row.instruction(), JoltInstructionRow::default());
        assert_eq!(
            row,
            TraceRow::new(
                JoltInstructionRow::default(),
                RegisterState::default(),
                RamAccess::NoOp,
            )
        );
    }

    #[test]
    #[should_panic(expected = "violates the final memory-row contract")]
    fn non_memory_row_with_ram_traffic_is_rejected() {
        let _ = TraceRow::new(
            JoltInstructionRow::default(),
            RegisterState::default(),
            RamAccess::Write(RamWrite {
                address: 8,
                pre_value: 7,
                post_value: 11,
            }),
        );
    }

    #[test]
    #[should_panic(expected = "violates the final memory-row contract")]
    fn load_row_with_mismatched_ram_value_is_rejected() {
        let _ = TraceRow::new(
            instruction(
                JoltInstructionKind::LD,
                NormalizedOperands {
                    rs1: Some(10),
                    rs2: None,
                    rd: Some(11),
                    imm: 0,
                },
            ),
            RegisterState {
                rs1: None,
                rs2: None,
                rd: Some(RegisterWrite {
                    register: 11,
                    pre_value: 0,
                    post_value: 1,
                }),
            },
            RamAccess::Read(RamRead {
                address: 0x2000,
                value: 2,
            }),
        );
    }

    #[test]
    #[should_panic(expected = "violates the final memory-row contract")]
    fn oversized_immediate_is_rejected() {
        let _ = TraceRow::from_instruction(instruction(
            JoltInstructionKind::ADDI,
            NormalizedOperands {
                rs1: None,
                rs2: None,
                rd: None,
                imm: i128::MAX,
            },
        ));
    }

    #[cfg(not(feature = "field-inline"))]
    #[test]
    fn trace_row_is_64_bytes() {
        assert_eq!(std::mem::size_of::<TraceRow>(), 64);
    }
}
