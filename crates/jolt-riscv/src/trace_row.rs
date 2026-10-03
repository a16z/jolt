//! Proof-facing materialized trace row (`JoltTraceRow`).
//!
//! `JoltTraceRow` is a compact, `Copy` row shared by execution and proving after
//! final bytecode expansion. Its **logical proof columns** are
//! exposed through accessors, while the **physical storage** is private and free
//! to alias mutually-exclusive or equal values for final memory rows.
//!
//! # Captured state
//!
//! The per-cycle witness values are described by [`CapturedState`], a typed enum
//! over the three final row classes (`NonMemory` / `Load` / `Store`). Each
//! variant only names the columns that are independent for that class. The
//! constructor checks equalities between logical register and RAM observations;
//! the packed accessor view then stores one value per alias: a load's
//! `RamReadValue`, `RamWriteValue`, and `RdWriteValue` are one field, and a
//! store's `RamWriteValue` and `Rs2Value` are one field. The cached
//! `Load`/`Store` circuit flags determine the class on read, so the enum is the
//! accessor view while storage stays flat (no separate discriminant).
//!
//! # Crate boundaries
//!
//! This type lives in `jolt-riscv` and depends only on `jolt-riscv`-native
//! types. Producers supply logical register and RAM observations to the checked
//! constructor, which owns the final memory-row contract. The
//! lookup-table accessor lives in `jolt-lookup-tables`.
//!
//! # Logical vs physical
//!
//! Proof code must depend on the logical accessors (`rs1_value`, `ram_address`,
//! `captured_state`, ...), never on the private storage slots, so the physical
//! layout stays swappable.

use crate::{
    CircuitFlagSet, CircuitFlags, Flags, InstructionFlagSet, InstructionFlags, JoltCycle,
    JoltInstruction, JoltInstructionKind, JoltInstructionRow, JoltInstructionTag,
    NormalizedOperands, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
#[cfg(feature = "serialization")]
use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};

/// Largest register id storable in a register-id byte. `0xFF` is reserved as the
/// `None` sentinel, so ids must be `<= 254`. Jolt's register file
/// (`REGISTER_COUNT = 128`) sits well within this bound; the limit is a
/// storage-format detail, not a protocol fact.
const MAX_REGISTER_ID: u8 = u8::MAX - 1;

/// Sentinel stored in a register-id byte for an absent (`None`) operand.
const REGISTER_NONE: u8 = u8::MAX;

/// Field-inline builds use 24 circuit-flag bits; base builds retain the original
/// 16-bit layout. Six instruction flags and the immediate sign follow them.
const META_INSTRUCTION_FLAGS_SHIFT: u32 = if cfg!(feature = "field-inline") {
    24
} else {
    16
};
const META_CIRCUIT_FLAGS_MASK: u32 = (1 << META_INSTRUCTION_FLAGS_SHIFT) - 1;
const META_INSTRUCTION_FLAGS_MASK: u32 = (1u32 << (NUM_INSTRUCTION_FLAGS as u32)) - 1;
const META_IMM_NEGATIVE_SHIFT: u32 = META_INSTRUCTION_FLAGS_SHIFT + NUM_INSTRUCTION_FLAGS as u32;

const CAPTURE_RS1: u8 = 1;
const CAPTURE_RS2: u8 = 1 << 1;
const CAPTURE_RD: u8 = 1 << 2;
const VIRTUAL_SEQUENCE_PRESENT: u8 = 1 << 3;
const INTEGER_RS1: u8 = 1 << 4;
const INTEGER_RS2: u8 = 1 << 5;
const INTEGER_RD: u8 = 1 << 6;

const _: () = {
    assert!(NUM_CIRCUIT_FLAGS <= META_INSTRUCTION_FLAGS_SHIFT as usize);
    assert!(META_IMM_NEGATIVE_SHIFT < u32::BITS);
    let instruction_mask = META_INSTRUCTION_FLAGS_MASK << META_INSTRUCTION_FLAGS_SHIFT;
    let sign_mask = 1 << META_IMM_NEGATIVE_SHIFT;
    assert!(META_CIRCUIT_FLAGS_MASK & instruction_mask == 0);
    assert!((META_CIRCUIT_FLAGS_MASK | instruction_mask) & sign_mask == 0);
    let capture_mask = CAPTURE_RS1 | CAPTURE_RS2 | CAPTURE_RD;
    let integer_mask = INTEGER_RS1 | INTEGER_RS2 | INTEGER_RD;
    assert!(capture_mask.count_ones() == 3);
    assert!(integer_mask.count_ones() == 3);
    assert!(capture_mask & integer_mask == 0);
    assert!((capture_mask | integer_mask) & VIRTUAL_SEQUENCE_PRESENT == 0);
    assert!(capture_mask | integer_mask | VIRTUAL_SEQUENCE_PRESENT == 0x7f);
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct RegisterRead {
    pub register: u8,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct RegisterWrite {
    pub register: u8,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct RegisterState {
    pub rs1: Option<RegisterRead>,
    pub rs2: Option<RegisterRead>,
    pub rd: Option<RegisterWrite>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct RamRead {
    pub address: u64,
    pub value: u64,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct RamWrite {
    pub address: u64,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub enum RamAccess {
    Read(RamRead),
    Write(RamWrite),
    #[default]
    NoOp,
}

/// Witness values for a non-memory row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NonMemoryState {
    pub rs1_value: u64,
    pub rs2_value: u64,
    pub rd_pre_value: u64,
    pub rd_write_value: u64,
}

/// Witness values for a final load row.
///
/// `rd_write_value` is also `RamReadValue` and `RamWriteValue` (the loaded
/// value); the type collapses the three equal logical columns into one field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadState {
    pub rs1_value: u64,
    pub ram_address: u64,
    pub rd_pre_value: u64,
    pub rd_write_value: u64,
}

/// Witness values for a final store row.
///
/// `rs2_value` is also `RamWriteValue`; `ram_read_value` is the old memory
/// value. Stores write no register, so there is no `rd_*` field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreState {
    pub rs1_value: u64,
    pub rs2_value: u64,
    pub ram_read_value: u64,
    pub ram_address: u64,
}

/// The per-cycle witness values, typed by final row class.
///
/// This is the [`JoltTraceRow::captured_state`] view. Register *indices* are not
/// part of it; they come from the instruction's operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapturedState {
    NonMemory(NonMemoryState),
    Load(LoadState),
    Store(StoreState),
}

impl Default for CapturedState {
    fn default() -> Self {
        CapturedState::NonMemory(NonMemoryState::default())
    }
}

impl CapturedState {
    fn into_value_slots(self) -> TraceValueSlots {
        match self {
            CapturedState::NonMemory(s) => TraceValueSlots {
                slot0: s.rs1_value,
                slot1: s.rs2_value,
                slot2: s.rd_pre_value,
                slot3: s.rd_write_value,
            },
            CapturedState::Load(s) => TraceValueSlots {
                slot0: s.rs1_value,
                slot1: s.ram_address,
                slot2: s.rd_pre_value,
                slot3: s.rd_write_value,
            },
            CapturedState::Store(s) => TraceValueSlots {
                slot0: s.rs1_value,
                slot1: s.rs2_value,
                slot2: s.ram_read_value,
                slot3: s.ram_address,
            },
        }
    }
}

/// A final row violates its representation or observation contract.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TraceRowError {
    /// A final-row immediate does not fit the chosen signed-magnitude `u64`
    /// encoding.
    #[error("immediate |{imm}| does not fit the u64 magnitude encoding")]
    ImmTooWide { imm: i128 },
    /// A register id does not fit the compact `u8` storage (with `0xFF`
    /// reserved as the `None` sentinel).
    #[error("register id {id} exceeds the compact storage bound (max {max})", max = MAX_REGISTER_ID)]
    RegisterIdTooWide { id: u8 },
    #[error("instruction {kind:?} has no final instruction lowering")]
    UnsupportedInstruction { kind: JoltInstructionKind },
    #[error("captured {operand} register {actual} does not match integer operand {expected:?}")]
    RegisterMismatch {
        operand: &'static str,
        expected: Option<u8>,
        actual: u8,
    },
    #[error("RAM access does not match final instruction {kind:?}")]
    RamAccessMismatch { kind: JoltInstructionKind },
    #[error("memory values or register captures do not match final instruction {kind:?}")]
    MemoryValueMismatch { kind: JoltInstructionKind },
    #[error("no-op rows cannot capture register or RAM effects")]
    NoOpEffects,
    #[error("bytecode PC {pc} is invalid for {kind:?}: only no-ops occupy slot zero")]
    InvalidBytecodePc { kind: JoltInstructionKind, pc: u32 },
    #[error("no-op rows cannot be compressed or first in a virtual sequence")]
    NoOpMetadata,
}

/// Four aliased 64-bit value slots. Their logical meaning depends on the row's
/// class (derived from the cached `Load`/`Store` circuit flags); see the
/// [`JoltTraceRow`] accessors and [`JoltTraceRow::captured_state`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
struct TraceValueSlots {
    slot0: u64,
    slot1: u64,
    slot2: u64,
    slot3: u64,
}

/// Compact, copyable proof-facing trace row (balanced packed, 64 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct JoltTraceRow {
    values: TraceValueSlots,
    /// Source RV64 instruction address (guest architectural address).
    unexpanded_pc: u64,
    /// Magnitude of the immediate; sign is bit `META_IMM_NEGATIVE_SHIFT` of `meta`.
    imm_abs: u64,
    /// Compact local bytecode index (expanded "PC"); see [`JoltTraceRow::pc`].
    bytecode_pc: u32,
    /// Packed circuit flags, instruction flags, and immediate sign.
    meta: u32,
    /// Final Jolt instruction tag (stable identity, not a dense index). The
    /// lookup-table routing is derived from this in `jolt-lookup-tables`.
    jolt_tag: u16,
    virtual_sequence_remaining: u16,
    /// Full encoded operand ids, including field registers, or `0xFF` (None).
    rs1_id: u8,
    rs2_id: u8,
    rd_id: u8,
    /// Capture presence, sequence presence, and integer-operand presence.
    control: u8,
}

#[cfg(feature = "serialization")]
#[derive(Serialize, Deserialize)]
struct TraceRowWire {
    instruction: JoltInstructionRow,
    registers: RegisterState,
    ram_access: RamAccess,
    bytecode_pc: u32,
}

#[cfg(feature = "serialization")]
impl Serialize for JoltTraceRow {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        TraceRowWire {
            instruction: self.instruction(),
            registers: self.registers(),
            ram_access: self.ram_access(),
            bytecode_pc: self.bytecode_pc,
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serialization")]
impl<'de> Deserialize<'de> for JoltTraceRow {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = TraceRowWire::deserialize(deserializer)?;
        Self::new(
            wire.instruction,
            wire.registers,
            wire.ram_access,
            wire.bytecode_pc,
        )
        .map_err(Error::custom)
    }
}

const _: () = assert!(
    core::mem::size_of::<JoltTraceRow>() == 64,
    "JoltTraceRow must stay 64 bytes; any size change should be intentional and reviewed"
);

impl Default for JoltTraceRow {
    /// The canonical no-op/padding row, whose logical accessors match a
    /// `NoOp` cycle (in particular `IsNoop` is set).
    fn default() -> Self {
        Self::no_op()
    }
}

impl JoltTraceRow {
    /// Canonical no-op row.
    #[expect(
        clippy::expect_used,
        reason = "the canonical no-op satisfies the checked row contract"
    )]
    pub fn no_op() -> Self {
        Self::new(
            JoltInstructionRow::default(),
            RegisterState::default(),
            RamAccess::NoOp,
            0,
        )
        .expect("the canonical no-op satisfies the checked row contract")
    }

    /// Build a final row from logical observations and an already resolved PC.
    ///
    /// This checks slot aliasing, captured integer-register identities, storage
    /// bounds, and the reserved no-op PC. Accepted instruction metadata is
    /// reconstructed exactly. No-ops cannot be compressed or first in a
    /// sequence: their proof flags do not retain those booleans. The producer's
    /// program mapper must establish that the PC identifies this instruction
    /// in its bytecode.
    pub fn new(
        instruction: JoltInstructionRow,
        registers: RegisterState,
        ram_access: RamAccess,
        bytecode_pc: u32,
    ) -> Result<Self, TraceRowError> {
        let kind = instruction.instruction_kind;
        let lowered = JoltInstruction::try_from(instruction)
            .map_err(|_| TraceRowError::UnsupportedInstruction { kind })?;
        let circuit_flags = lowered.circuit_flags();
        let instruction_flags = lowered.instruction_flags();
        let is_noop = instruction_flags.get(InstructionFlags::IsNoop);
        if is_noop != (bytecode_pc == 0) {
            return Err(TraceRowError::InvalidBytecodePc {
                kind,
                pc: bytecode_pc,
            });
        }
        if is_noop && (instruction.is_first_in_sequence || instruction.is_compressed) {
            return Err(TraceRowError::NoOpMetadata);
        }
        if is_noop && (registers != RegisterState::default() || ram_access != RamAccess::NoOp) {
            return Err(TraceRowError::NoOpEffects);
        }

        let integer_operands = instruction.integer_operands();
        for (operand, expected, actual) in [
            (
                "rs1",
                integer_operands.rs1,
                registers.rs1.map(|read| read.register),
            ),
            (
                "rs2",
                integer_operands.rs2,
                registers.rs2.map(|read| read.register),
            ),
            (
                "rd",
                integer_operands.rd,
                registers.rd.map(|write| write.register),
            ),
        ] {
            if let Some(actual) = actual {
                if expected != Some(actual) {
                    return Err(TraceRowError::RegisterMismatch {
                        operand,
                        expected,
                        actual,
                    });
                }
            }
        }

        let rs1_value = registers.rs1.map_or(0, |read| read.value);
        let rs2_value = registers.rs2.map_or(0, |read| read.value);
        let rd_pre_value = registers.rd.map_or(0, |write| write.pre_value);
        let rd_write_value = registers.rd.map_or(0, |write| write.post_value);
        let state = if circuit_flags.get(CircuitFlags::Load) {
            let RamAccess::Read(read) = ram_access else {
                return Err(TraceRowError::RamAccessMismatch { kind });
            };
            if registers.rs2.is_some() || read.value != rd_write_value {
                return Err(TraceRowError::MemoryValueMismatch { kind });
            }
            CapturedState::Load(LoadState {
                rs1_value,
                ram_address: read.address,
                rd_pre_value,
                rd_write_value,
            })
        } else if circuit_flags.get(CircuitFlags::Store) {
            let RamAccess::Write(write) = ram_access else {
                return Err(TraceRowError::RamAccessMismatch { kind });
            };
            if registers.rd.is_some() || write.post_value != rs2_value {
                return Err(TraceRowError::MemoryValueMismatch { kind });
            }
            CapturedState::Store(StoreState {
                rs1_value,
                rs2_value,
                ram_read_value: write.pre_value,
                ram_address: write.address,
            })
        } else {
            if ram_access != RamAccess::NoOp {
                return Err(TraceRowError::RamAccessMismatch { kind });
            }
            CapturedState::NonMemory(NonMemoryState {
                rs1_value,
                rs2_value,
                rd_pre_value,
                rd_write_value,
            })
        };

        let imm = instruction.operands.imm;
        let imm_magnitude = imm.unsigned_abs();
        if imm_magnitude > u64::MAX as u128 {
            return Err(TraceRowError::ImmTooWide { imm });
        }

        let control = (u8::from(registers.rs1.is_some()) * CAPTURE_RS1)
            | (u8::from(registers.rs2.is_some()) * CAPTURE_RS2)
            | (u8::from(registers.rd.is_some()) * CAPTURE_RD)
            | (u8::from(instruction.virtual_sequence_remaining.is_some())
                * VIRTUAL_SEQUENCE_PRESENT)
            | (u8::from(integer_operands.rs1.is_some()) * INTEGER_RS1)
            | (u8::from(integer_operands.rs2.is_some()) * INTEGER_RS2)
            | (u8::from(integer_operands.rd.is_some()) * INTEGER_RD);

        Ok(Self {
            values: state.into_value_slots(),
            unexpanded_pc: instruction.address as u64,
            imm_abs: imm_magnitude as u64,
            bytecode_pc,
            meta: pack_meta(circuit_flags, instruction_flags, imm < 0),
            jolt_tag: kind.tag().0,
            virtual_sequence_remaining: instruction.virtual_sequence_remaining.unwrap_or(0),
            rs1_id: checked_register_id(instruction.operands.rs1)?,
            rs2_id: checked_register_id(instruction.operands.rs2)?,
            rd_id: checked_register_id(instruction.operands.rd)?,
            control,
        })
    }

    /// The per-cycle witness values, typed by row class.
    #[inline]
    pub fn captured_state(&self) -> CapturedState {
        if self.is_load() {
            CapturedState::Load(LoadState {
                rs1_value: self.values.slot0,
                ram_address: self.values.slot1,
                rd_pre_value: self.values.slot2,
                rd_write_value: self.values.slot3,
            })
        } else if self.is_store() {
            CapturedState::Store(StoreState {
                rs1_value: self.values.slot0,
                rs2_value: self.values.slot1,
                ram_read_value: self.values.slot2,
                ram_address: self.values.slot3,
            })
        } else {
            CapturedState::NonMemory(NonMemoryState {
                rs1_value: self.values.slot0,
                rs2_value: self.values.slot1,
                rd_pre_value: self.values.slot2,
                rd_write_value: self.values.slot3,
            })
        }
    }

    #[inline(always)]
    pub fn rs1_value(&self) -> u64 {
        self.values.slot0
    }

    #[inline(always)]
    pub fn rs2_value(&self) -> u64 {
        if self.is_load() {
            0
        } else {
            self.values.slot1
        }
    }

    #[inline(always)]
    pub fn rd_pre_value(&self) -> u64 {
        if self.is_store() {
            0
        } else {
            self.values.slot2
        }
    }

    #[inline(always)]
    pub fn rd_write_value(&self) -> u64 {
        if self.is_store() {
            0
        } else {
            self.values.slot3
        }
    }

    #[inline(always)]
    pub fn ram_address(&self) -> u64 {
        if self.is_load() {
            self.values.slot1
        } else if self.is_store() {
            self.values.slot3
        } else {
            0
        }
    }

    #[inline(always)]
    pub fn ram_read_value(&self) -> u64 {
        if self.is_load() {
            self.values.slot3
        } else if self.is_store() {
            self.values.slot2
        } else {
            0
        }
    }

    #[inline(always)]
    pub fn ram_write_value(&self) -> u64 {
        if self.is_load() {
            self.values.slot3
        } else if self.is_store() {
            self.values.slot1
        } else {
            0
        }
    }

    /// Expanded PC (local bytecode index) as a raw integer.
    #[inline(always)]
    pub fn pc(&self) -> u64 {
        self.bytecode_pc as u64
    }

    /// Source RV64 instruction address.
    #[inline(always)]
    pub fn unexpanded_pc(&self) -> u64 {
        self.unexpanded_pc
    }

    #[inline]
    pub fn address(&self) -> u64 {
        self.unexpanded_pc
    }

    #[inline]
    pub fn virtual_sequence_remaining(&self) -> Option<u16> {
        (self.control & VIRTUAL_SEQUENCE_PRESENT != 0).then_some(self.virtual_sequence_remaining)
    }

    /// Reconstruct the full encoded instruction, including field operands.
    #[inline]
    #[expect(
        clippy::expect_used,
        reason = "the private instruction tag is stored only by the checked constructor"
    )]
    pub fn instruction(&self) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind: self.instruction_kind().expect("checked instruction tag"),
            address: self.unexpanded_pc as usize,
            operands: NormalizedOperands {
                rs1: register_index(self.rs1_id),
                rs2: register_index(self.rs2_id),
                rd: register_index(self.rd_id),
                imm: self.imm(),
            },
            virtual_sequence_remaining: self.virtual_sequence_remaining(),
            is_first_in_sequence: self.circuit_flags().get(CircuitFlags::IsFirstInSequence),
            is_compressed: self.circuit_flags().get(CircuitFlags::IsCompressed),
        }
    }

    #[inline(always)]
    pub fn imm(&self) -> i128 {
        let magnitude = self.imm_abs as i128;
        if self.meta & (1 << META_IMM_NEGATIVE_SHIFT) != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    #[inline(always)]
    pub fn rs1_index(&self) -> Option<u8> {
        (self.control & INTEGER_RS1 != 0).then_some(self.rs1_id)
    }

    #[inline(always)]
    pub fn rs2_index(&self) -> Option<u8> {
        (self.control & INTEGER_RS2 != 0).then_some(self.rs2_id)
    }

    #[inline(always)]
    pub fn rd_index(&self) -> Option<u8> {
        (self.control & INTEGER_RD != 0).then_some(self.rd_id)
    }

    #[inline]
    pub fn rs1_read(&self) -> Option<RegisterRead> {
        (self.control & CAPTURE_RS1 != 0).then_some(RegisterRead {
            register: self.rs1_id,
            value: self.values.slot0,
        })
    }

    #[inline]
    pub fn rs2_read(&self) -> Option<RegisterRead> {
        (self.control & CAPTURE_RS2 != 0).then_some(RegisterRead {
            register: self.rs2_id,
            value: self.values.slot1,
        })
    }

    #[inline]
    pub fn rd_write(&self) -> Option<RegisterWrite> {
        (self.control & CAPTURE_RD != 0).then_some(RegisterWrite {
            register: self.rd_id,
            pre_value: self.values.slot2,
            post_value: self.values.slot3,
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
        if self.is_load() {
            RamAccess::Read(RamRead {
                address: self.ram_address(),
                value: self.ram_read_value(),
            })
        } else if self.is_store() {
            RamAccess::Write(RamWrite {
                address: self.ram_address(),
                pre_value: self.ram_read_value(),
                post_value: self.ram_write_value(),
            })
        } else {
            RamAccess::NoOp
        }
    }

    #[inline(always)]
    pub fn circuit_flags(&self) -> CircuitFlagSet {
        CircuitFlagSet::from_bits(self.meta & META_CIRCUIT_FLAGS_MASK)
    }

    #[inline(always)]
    pub fn instruction_flags(&self) -> InstructionFlagSet {
        InstructionFlagSet::from_bits(
            ((self.meta >> META_INSTRUCTION_FLAGS_SHIFT) & META_INSTRUCTION_FLAGS_MASK) as u8,
        )
    }

    #[inline(always)]
    pub fn is_load(&self) -> bool {
        self.meta & (1 << CircuitFlags::Load as u32) != 0
    }

    #[inline(always)]
    pub fn is_store(&self) -> bool {
        self.meta & (1 << CircuitFlags::Store as u32) != 0
    }

    #[inline(always)]
    pub fn is_noop(&self) -> bool {
        self.meta & (1 << (META_INSTRUCTION_FLAGS_SHIFT + InstructionFlags::IsNoop as u32)) != 0
    }

    /// Stable final-row identity, reconstructed from the cached tag.
    #[inline]
    pub fn instruction_kind(&self) -> Option<JoltInstructionKind> {
        JoltInstructionKind::from_tag(JoltInstructionTag(self.jolt_tag))
    }
}

impl JoltCycle for JoltTraceRow {
    type Instruction = JoltInstructionRow;

    #[inline]
    fn instruction(&self) -> Self::Instruction {
        JoltTraceRow::instruction(self)
    }

    #[inline]
    fn rs1_val(&self) -> Option<u64> {
        self.rs1_read().map(|read| read.value)
    }

    #[inline]
    fn rs2_val(&self) -> Option<u64> {
        self.rs2_read().map(|read| read.value)
    }

    #[inline]
    fn rd_vals(&self) -> Option<(u64, u64)> {
        self.rd_write()
            .map(|write| (write.pre_value, write.post_value))
    }

    #[inline]
    fn ram_access_address(&self) -> Option<u64> {
        (self.is_load() || self.is_store()).then_some(self.ram_address())
    }

    #[inline]
    fn ram_read_value(&self) -> Option<u64> {
        (self.is_load() || self.is_store()).then_some(JoltTraceRow::ram_read_value(self))
    }

    #[inline]
    fn ram_write_value(&self) -> Option<u64> {
        self.is_store()
            .then_some(JoltTraceRow::ram_write_value(self))
    }
}

#[inline]
fn pack_meta(
    circuit_flags: CircuitFlagSet,
    instruction_flags: InstructionFlagSet,
    imm_negative: bool,
) -> u32 {
    circuit_flags.bits()
        | ((instruction_flags.bits() as u32) << META_INSTRUCTION_FLAGS_SHIFT)
        | ((imm_negative as u32) << META_IMM_NEGATIVE_SHIFT)
}

#[inline(always)]
fn checked_register_id(id: Option<u8>) -> Result<u8, TraceRowError> {
    match id {
        None => Ok(REGISTER_NONE),
        Some(id) if id <= MAX_REGISTER_ID => Ok(id),
        Some(id) => Err(TraceRowError::RegisterIdTooWide { id }),
    }
}

#[inline(always)]
fn register_index(id: u8) -> Option<u8> {
    (id != REGISTER_NONE).then_some(id)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;
    use crate::CIRCUIT_FLAGS;

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

    #[test]
    fn layout_is_64_bytes_and_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<JoltTraceRow>();
        assert_eq!(core::mem::size_of::<JoltTraceRow>(), 64);
        assert_eq!(core::mem::align_of::<JoltTraceRow>(), 8);
    }

    #[test]
    fn default_is_canonical_no_op() {
        let row = JoltTraceRow::default();
        assert_eq!(row, JoltTraceRow::no_op());
        assert_eq!(row.instruction(), JoltInstructionRow::default());
        assert_eq!(row.registers(), RegisterState::default());
        assert_eq!(row.ram_access(), RamAccess::NoOp);
        assert_eq!(row.pc(), 0);
        assert_eq!(row.unexpanded_pc(), 0);
        assert_eq!(row.imm(), 0);
        assert_eq!(row.rs1_index(), None);
        assert!(row.is_noop());
        assert!(!row.is_load() && !row.is_store());
        assert_eq!(
            row.captured_state(),
            CapturedState::NonMemory(NonMemoryState::default())
        );
        let instruction = JoltInstructionRow {
            address: 0x8000_0000,
            operands: NormalizedOperands {
                rs1: Some(2),
                rs2: Some(3),
                rd: Some(4),
                imm: -7,
            },
            virtual_sequence_remaining: Some(3),
            ..Default::default()
        };
        let source_noop =
            JoltTraceRow::new(instruction, RegisterState::default(), RamAccess::NoOp, 0).unwrap();
        assert_eq!(source_noop.instruction(), instruction);
        assert_ne!(source_noop, row);
    }

    #[test]
    fn instruction_and_capture_presence_round_trip_independently() {
        for remaining in [None, Some(0), Some(u16::MAX)] {
            let mut source = instruction(
                JoltInstructionKind::ADDI,
                NormalizedOperands {
                    rs1: Some(2),
                    rs2: None,
                    rd: Some(1),
                    imm: -i128::from(u64::MAX),
                },
            );
            source.virtual_sequence_remaining = remaining;
            source.is_first_in_sequence = true;
            source.is_compressed = true;
            for read in [
                None,
                Some(RegisterRead {
                    register: 2,
                    value: 0,
                }),
            ] {
                let registers = RegisterState {
                    rs1: read,
                    ..Default::default()
                };
                let row = JoltTraceRow::new(source, registers, RamAccess::NoOp, 7).unwrap();
                assert_eq!(row.instruction(), source);
                assert_eq!(row.registers(), registers);
                assert_eq!(row.rs1_value(), 0);
                assert_eq!(row.rs1_index(), Some(2));
                assert_eq!(row.rs1_val(), read.map(|read| read.value));
                assert_eq!(row.rd_index(), Some(1));
                assert_eq!(row.rd_write(), None);
            }
        }
    }

    #[test]
    fn non_memory_state_round_trips_columns() {
        let source = instruction(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(2),
                rs2: Some(3),
                rd: Some(1),
                imm: 0,
            },
        );
        let registers = RegisterState {
            rs1: Some(RegisterRead {
                register: 2,
                value: 11,
            }),
            rs2: Some(RegisterRead {
                register: 3,
                value: 22,
            }),
            rd: Some(RegisterWrite {
                register: 1,
                pre_value: 33,
                post_value: 44,
            }),
        };
        let row = JoltTraceRow::new(source, registers, RamAccess::NoOp, 7).unwrap();
        assert_eq!(row.registers(), registers);
        assert_eq!(
            row.captured_state(),
            CapturedState::NonMemory(NonMemoryState {
                rs1_value: 11,
                rs2_value: 22,
                rd_pre_value: 33,
                rd_write_value: 44,
            })
        );
        assert_eq!(row.ram_read_value(), 0);
        assert_eq!(JoltCycle::ram_read_value(&row), None);
    }

    #[test]
    fn load_row_preserves_distinct_execution_and_proof_ram_views() {
        let source = instruction(
            JoltInstructionKind::LD,
            NormalizedOperands {
                rs1: Some(10),
                rs2: None,
                rd: Some(11),
                imm: 8,
            },
        );
        let registers = RegisterState {
            rs1: Some(RegisterRead {
                register: 10,
                value: 0x1000,
            }),
            rd: Some(RegisterWrite {
                register: 11,
                pre_value: 5,
                post_value: 0xdead_beef,
            }),
            ..Default::default()
        };
        let ram = RamAccess::Read(RamRead {
            address: 0x1008,
            value: 0xdead_beef,
        });
        let row = JoltTraceRow::new(source, registers, ram, 3).unwrap();
        assert_eq!(row.registers(), registers);
        assert_eq!(row.ram_access(), ram);
        assert_eq!(
            row.captured_state(),
            CapturedState::Load(LoadState {
                rs1_value: 0x1000,
                ram_address: 0x1008,
                rd_pre_value: 5,
                rd_write_value: 0xdead_beef,
            })
        );
        assert_eq!(row.rs2_value(), 0);
        assert_eq!(row.ram_read_value(), 0xdead_beef);
        assert_eq!(row.ram_write_value(), 0xdead_beef);
        assert_eq!(JoltCycle::ram_read_value(&row), Some(0xdead_beef));
        assert_eq!(JoltCycle::ram_write_value(&row), None);
    }

    #[test]
    fn store_row_round_trips_aliased_values() {
        let source = instruction(
            JoltInstructionKind::SD,
            NormalizedOperands {
                rs1: Some(10),
                rs2: Some(12),
                rd: None,
                imm: -4,
            },
        );
        let registers = RegisterState {
            rs1: Some(RegisterRead {
                register: 10,
                value: 0x3000,
            }),
            rs2: Some(RegisterRead {
                register: 12,
                value: 0x1234,
            }),
            rd: None,
        };
        let ram = RamAccess::Write(RamWrite {
            address: 0x2ffc,
            pre_value: 0x5678,
            post_value: 0x1234,
        });
        let row = JoltTraceRow::new(source, registers, ram, 9).unwrap();
        assert_eq!(row.registers(), registers);
        assert_eq!(row.ram_access(), ram);
        assert_eq!(
            row.captured_state(),
            CapturedState::Store(StoreState {
                rs1_value: 0x3000,
                rs2_value: 0x1234,
                ram_read_value: 0x5678,
                ram_address: 0x2ffc,
            })
        );
        assert_eq!(row.rd_pre_value(), 0);
        assert_eq!(row.rd_write_value(), 0);
        assert_eq!(JoltCycle::ram_write_value(&row), Some(0x1234));
    }

    #[test]
    fn malformed_register_and_memory_observations_are_rejected() {
        let source = instruction(
            JoltInstructionKind::ADDI,
            NormalizedOperands {
                rs1: Some(2),
                rs2: None,
                rd: Some(1),
                imm: 0,
            },
        );
        let registers = RegisterState {
            rs1: Some(RegisterRead {
                register: 200,
                value: 0,
            }),
            ..Default::default()
        };
        assert!(matches!(
            JoltTraceRow::new(source, registers, RamAccess::NoOp, 1),
            Err(TraceRowError::RegisterMismatch { .. })
        ));
        for (kind, ram) in [
            (
                JoltInstructionKind::ADD,
                RamAccess::Read(RamRead::default()),
            ),
            (
                JoltInstructionKind::ADD,
                RamAccess::Write(RamWrite::default()),
            ),
            (JoltInstructionKind::LD, RamAccess::NoOp),
            (
                JoltInstructionKind::LD,
                RamAccess::Write(RamWrite::default()),
            ),
            (JoltInstructionKind::SD, RamAccess::NoOp),
            (JoltInstructionKind::SD, RamAccess::Read(RamRead::default())),
        ] {
            let source = instruction(kind, NormalizedOperands::default());
            assert!(matches!(
                JoltTraceRow::new(source, RegisterState::default(), ram, 1),
                Err(TraceRowError::RamAccessMismatch { .. })
            ));
        }
        for (kind, ram) in [
            (
                JoltInstructionKind::LD,
                RamAccess::Read(RamRead {
                    address: 8,
                    value: 1,
                }),
            ),
            (
                JoltInstructionKind::SD,
                RamAccess::Write(RamWrite {
                    address: 8,
                    pre_value: 0,
                    post_value: 1,
                }),
            ),
        ] {
            let source = instruction(kind, NormalizedOperands::default());
            assert!(matches!(
                JoltTraceRow::new(source, RegisterState::default(), ram, 1),
                Err(TraceRowError::MemoryValueMismatch { .. })
            ));
        }
    }

    #[test]
    fn reserved_pc_and_noop_effects_are_rejected() {
        for (kind, pc) in [
            (JoltInstructionKind::NoOp, 1),
            (JoltInstructionKind::ADD, 0),
        ] {
            let source = instruction(kind, NormalizedOperands::default());
            assert!(matches!(
                JoltTraceRow::new(source, RegisterState::default(), RamAccess::NoOp, pc),
                Err(TraceRowError::InvalidBytecodePc { .. })
            ));
        }
        let captured_zero = RegisterState {
            rs1: Some(RegisterRead {
                register: 0,
                value: 0,
            }),
            ..Default::default()
        };
        assert!(matches!(
            JoltTraceRow::new(
                JoltInstructionRow::default(),
                captured_zero,
                RamAccess::NoOp,
                0
            ),
            Err(TraceRowError::NoOpEffects)
        ));
        for (is_first_in_sequence, is_compressed) in [(true, false), (false, true)] {
            let source = JoltInstructionRow {
                is_first_in_sequence,
                is_compressed,
                ..Default::default()
            };
            assert!(matches!(
                JoltTraceRow::new(source, RegisterState::default(), RamAccess::NoOp, 0),
                Err(TraceRowError::NoOpMetadata)
            ));
        }
    }

    #[test]
    fn storage_overflows_are_rejected() {
        let too_wide = instruction(
            JoltInstructionKind::ADDI,
            NormalizedOperands {
                imm: i128::MAX,
                ..Default::default()
            },
        );
        assert!(matches!(
            JoltTraceRow::new(too_wide, RegisterState::default(), RamAccess::NoOp, 1),
            Err(TraceRowError::ImmTooWide { .. })
        ));
        let reserved = instruction(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(REGISTER_NONE),
                ..Default::default()
            },
        );
        assert!(matches!(
            JoltTraceRow::new(reserved, RegisterState::default(), RamAccess::NoOp, 1),
            Err(TraceRowError::RegisterIdTooWide { .. })
        ));
    }

    #[test]
    fn flags_round_trip_through_meta() {
        for circuit_flag in CIRCUIT_FLAGS {
            let circuit_flags = CircuitFlagSet::default().set(circuit_flag);
            let instruction_flags = InstructionFlagSet::default().set(InstructionFlags::IsNoop);
            for imm_negative in [false, true] {
                let row = JoltTraceRow {
                    meta: pack_meta(circuit_flags, instruction_flags, imm_negative),
                    imm_abs: 7,
                    ..JoltTraceRow::default()
                };
                assert_eq!(row.circuit_flags(), circuit_flags);
                assert_eq!(row.instruction_flags(), instruction_flags);
                assert_eq!(row.imm(), if imm_negative { -7 } else { 7 });
            }
        }
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn field_operands_round_trip_while_integer_indices_use_the_canonical_projection() {
        for &kind in JoltInstructionKind::ALL {
            if crate::field_inline_operand_shape(kind).is_none() {
                continue;
            }
            let source = instruction(
                kind,
                NormalizedOperands {
                    rs1: Some(2),
                    rs2: Some(3),
                    rd: Some(4),
                    imm: 0,
                },
            );
            let expected = source.integer_operands();
            for (encoded, integer) in [
                (source.operands.rs1, expected.rs1),
                (source.operands.rs2, expected.rs2),
                (source.operands.rd, expected.rd),
            ] {
                assert!(integer.is_none() || integer == encoded);
            }
            let flags = JoltInstruction::try_from(source).unwrap().circuit_flags();
            let ram = if flags.get(CircuitFlags::Load) {
                RamAccess::Read(RamRead::default())
            } else {
                RamAccess::NoOp
            };
            let row = JoltTraceRow::new(source, RegisterState::default(), ram, 1).unwrap();
            assert_eq!(row.instruction(), source);
            assert_eq!(row.rs1_index(), expected.rs1);
            assert_eq!(row.rs2_index(), expected.rs2);
            assert_eq!(row.rd_index(), expected.rd);
            assert_eq!(row.registers(), RegisterState::default());
        }
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn memory_accumulation_keeps_field_rs2_out_of_integer_captures() {
        let source = instruction(
            JoltInstructionKind::FIELD_LOAD_ACCUMULATE_FROM_MEMORY,
            NormalizedOperands {
                rs1: Some(2),
                rs2: Some(3),
                rd: Some(4),
                imm: 8,
            },
        );
        let registers = RegisterState {
            rs1: Some(RegisterRead {
                register: 2,
                value: 0x1000,
            }),
            rd: Some(RegisterWrite {
                register: 4,
                pre_value: 9,
                post_value: 23,
            }),
            ..Default::default()
        };
        let ram = RamAccess::Read(RamRead {
            address: 0x1008,
            value: 23,
        });
        let row = JoltTraceRow::new(source, registers, ram, 1).unwrap();
        assert_eq!(row.instruction(), source);
        assert_eq!(row.rs1_index(), Some(2));
        assert_eq!(row.rs2_index(), None);
        assert_eq!(row.rd_index(), Some(4));
        assert_eq!(row.ram_read_value(), 23);
        assert_eq!(row.rd_write_value(), 23);
        let illegal = RegisterState {
            rs2: Some(RegisterRead {
                register: 3,
                value: 0,
            }),
            ..registers
        };
        assert!(matches!(
            JoltTraceRow::new(source, illegal, ram, 1),
            Err(TraceRowError::RegisterMismatch {
                operand: "rs2",
                expected: None,
                ..
            })
        ));
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn semantic_wire_round_trips_and_rechecks_observations() {
        let source = instruction(
            JoltInstructionKind::ADDI,
            NormalizedOperands {
                rs1: Some(2),
                rd: Some(1),
                imm: -17,
                ..Default::default()
            },
        );
        let registers = RegisterState {
            rs1: Some(RegisterRead {
                register: 2,
                value: 9,
            }),
            ..Default::default()
        };
        let row = JoltTraceRow::new(source, registers, RamAccess::NoOp, 5).unwrap();
        let mut wire = serde_json::to_value(row).unwrap();
        assert_eq!(
            serde_json::from_value::<JoltTraceRow>(wire.clone()).unwrap(),
            row
        );
        *wire.pointer_mut("/registers/rs1/register").unwrap() = serde_json::json!(3);
        assert!(serde_json::from_value::<JoltTraceRow>(wire).is_err());
        let mut wire = serde_json::to_value(row).unwrap();
        *wire.pointer_mut("/bytecode_pc").unwrap() = serde_json::json!(0);
        assert!(serde_json::from_value::<JoltTraceRow>(wire).is_err());
    }
}
