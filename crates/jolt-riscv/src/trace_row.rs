//! Proof-facing materialized trace row (`JoltTraceRow`).
//!
//! `JoltTraceRow` is a compact, `Copy` row built once after tracing and final
//! bytecode expansion. It replaces repeated prover-side adaptation of a raw
//! tracer cycle with a flat representation whose **logical proof columns** are
//! exposed through accessors, while the **physical storage** is private and free
//! to alias mutually-exclusive or equal values for final memory rows.
//!
//! # Crate boundaries
//!
//! This type lives in `jolt-riscv` (the instruction-vocabulary crate) and depends
//! only on `jolt-riscv`-native types. The pieces that require higher-level crates
//! live there instead:
//! - the `Cycle` → `JoltTraceRow` conversion lives in `tracer` (which imports
//!   this type and computes the bytecode PC);
//! - the lookup-table accessor lives in `jolt-lookup-tables` (which owns
//!   `LookupTableKind`), derived from the row's cached instruction tag.
//!
//! # Logical vs physical
//!
//! Proof code must depend on the logical accessors (`rs1_value`, `ram_address`,
//! `circuit_flags`, ...), never on the private storage slots. This keeps the
//! physical layout swappable: any layout whose accessors return identical values
//! is a valid drop-in, guarded by the `trace_row_accessor_parity` invariant.
//!
//! # Final memory-row contract
//!
//! The packing exploits the equalities that hold for *final* Jolt memory rows
//! (after narrow/atomic/store-conditional source ops have been lowered to final
//! `LD`/compute/`SD` rows):
//!
//! ```text
//! LD:  RamAddress = effective address
//!      RamReadValue = RamWriteValue = RdWriteValue
//! SD:  RamAddress = effective address
//!      RamReadValue = old memory value
//!      RamWriteValue = Rs2Value
//! ```
//!
//! Construction verifies this contract and fails loudly ([`TraceRowError`]) if a
//! row violates it, rather than silently packing inconsistent values.

use crate::{
    CircuitFlagSet, CircuitFlags, Flags, InstructionFlagSet, InstructionFlags, JoltInstruction,
    JoltInstructionKind, JoltInstructionRow, JoltInstructionTag,
};

/// Largest register id storable in `register_pack`. `0xFF` is reserved as the
/// `None` sentinel, so ids must be `<= 254`. Jolt's register file
/// (`REGISTER_COUNT = 128`) sits well within this bound; the limit is a
/// storage-format detail, not a protocol fact.
const MAX_REGISTER_ID: u8 = u8::MAX - 1;

/// Sentinel stored in `register_pack` for an absent (`None`) register operand.
const REGISTER_NONE: u8 = u8::MAX;

/// Error raised when a row's components cannot be represented in the layout.
///
/// Cycle-/bytecode-specific failures (source-only cycles, oversized bytecode
/// indices) are surfaced by the `tracer` conversion, not here, since this crate
/// has no notion of either.
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
    /// A row's logical values violate the final memory-row contract for its
    /// class (see module docs). The offending row must be lowered into
    /// canonical rows or stored in a less-packed layout before it can be used.
    #[error("memory-row contract violated for {kind:?}: {detail}")]
    MemoryRowContractViolation {
        kind: JoltInstructionKind,
        detail: &'static str,
    },
}

/// Physical class of a row, deciding how the four value slots are interpreted.
///
/// Derived at construction from the final instruction's `Load`/`Store` circuit
/// flags. A private packing detail, not a source of instruction identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
enum RowClass {
    #[default]
    NonMemory = 0,
    Load = 1,
    Store = 2,
}

/// Four aliased 64-bit value slots. Their logical meaning depends on
/// [`RowClass`]; see [`JoltTraceRow`] accessors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
struct TraceValueSlots {
    slot0: u64,
    slot1: u64,
    slot2: u64,
    slot3: u64,
}

/// The dynamic (witness) logical values for one cycle, before physical packing.
///
/// This is the producer-facing bundle: the `tracer` conversion fills it from a
/// raw cycle, so this crate's builder does not depend on any concrete cycle type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogicalValues {
    pub rs1_value: u64,
    pub rs2_value: u64,
    pub rd_pre_value: u64,
    pub rd_write_value: u64,
    pub ram_address: u64,
    pub ram_read_value: u64,
    pub ram_write_value: u64,
    pub rs1_index: Option<u8>,
    pub rs2_index: Option<u8>,
    pub rd_index: Option<u8>,
}

impl LogicalValues {
    pub const ZERO: Self = Self {
        rs1_value: 0,
        rs2_value: 0,
        rd_pre_value: 0,
        rd_write_value: 0,
        ram_address: 0,
        ram_read_value: 0,
        ram_write_value: 0,
        rs1_index: None,
        rs2_index: None,
        rd_index: None,
    };
}

/// Compact, copyable proof-facing trace row (balanced packed, 64 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct JoltTraceRow {
    values: TraceValueSlots,
    /// Source RV64 instruction address (guest architectural address).
    unexpanded_pc: u64,
    /// Magnitude of the immediate; sign in `imm_is_negative`.
    imm_abs: u64,
    /// Compact local bytecode index (expanded "PC"). Logical type is
    /// [`u64`]; see [`JoltTraceRow::bytecode_index`].
    bytecode_pc: u32,
    /// `rs1 | rs2 << 8 | rd << 16`, each byte a register id or `0xFF` (None).
    register_pack: u32,
    /// Final Jolt instruction tag (stable identity, not a dense index). The
    /// lookup-table routing is derived from this in `jolt-lookup-tables`.
    jolt_tag: u16,
    /// Cached circuit flags (avoids re-deriving from the instruction row).
    circuit_flags: CircuitFlagSet,
    /// Cached instruction flags.
    instruction_flags: InstructionFlagSet,
    imm_is_negative: bool,
    row_class: RowClass,
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
    pub fn no_op() -> Self {
        let instruction = JoltInstructionRow::default();
        let (circuit_flags, instruction_flags) = row_flags(&instruction);
        Self {
            values: TraceValueSlots::default(),
            unexpanded_pc: 0,
            imm_abs: 0,
            bytecode_pc: 0,
            register_pack: pack_register_ids(None, None, None).unwrap_or(NO_REGISTERS),
            jolt_tag: instruction.instruction_kind.tag().0,
            circuit_flags,
            instruction_flags,
            imm_is_negative: false,
            row_class: RowClass::NonMemory,
        }
    }

    /// Build a row from already-extracted logical values plus the final
    /// instruction row and its compact bytecode index.
    pub fn from_components(
        values: &LogicalValues,
        instruction: &JoltInstructionRow,
        bytecode_pc: u32,
    ) -> Result<Self, TraceRowError> {
        let (circuit_flags, instruction_flags) = row_flags(instruction);
        let is_load = circuit_flags.get(CircuitFlags::Load);
        let is_store = circuit_flags.get(CircuitFlags::Store);
        let kind = instruction.instruction_kind;

        let (row_class, slots) = if is_load {
            // RamReadValue = RamWriteValue = RdWriteValue; no rs2.
            if values.rs2_value != 0 {
                return Err(violation(kind, "load row has non-zero Rs2Value"));
            }
            if values.ram_read_value != values.rd_write_value
                || values.ram_write_value != values.rd_write_value
            {
                return Err(violation(
                    kind,
                    "load RamReadValue/RamWriteValue must equal RdWriteValue",
                ));
            }
            (
                RowClass::Load,
                TraceValueSlots {
                    slot0: values.rs1_value,
                    slot1: values.ram_address,
                    slot2: values.rd_pre_value,
                    slot3: values.rd_write_value,
                },
            )
        } else if is_store {
            // RamWriteValue = Rs2Value; no rd.
            if values.rd_pre_value != 0 || values.rd_write_value != 0 {
                return Err(violation(kind, "store row writes rd"));
            }
            if values.ram_write_value != values.rs2_value {
                return Err(violation(kind, "store RamWriteValue must equal Rs2Value"));
            }
            (
                RowClass::Store,
                TraceValueSlots {
                    slot0: values.rs1_value,
                    slot1: values.ram_write_value,
                    slot2: values.ram_read_value,
                    slot3: values.ram_address,
                },
            )
        } else {
            if values.ram_address != 0 || values.ram_read_value != 0 || values.ram_write_value != 0
            {
                return Err(violation(kind, "non-memory row carries RAM values"));
            }
            (
                RowClass::NonMemory,
                TraceValueSlots {
                    slot0: values.rs1_value,
                    slot1: values.rs2_value,
                    slot2: values.rd_pre_value,
                    slot3: values.rd_write_value,
                },
            )
        };

        let imm = instruction.operands.imm;
        let imm_magnitude = imm.unsigned_abs();
        if imm_magnitude > u64::MAX as u128 {
            return Err(TraceRowError::ImmTooWide { imm });
        }

        Ok(Self {
            values: slots,
            unexpanded_pc: instruction.address as u64,
            imm_abs: imm_magnitude as u64,
            bytecode_pc,
            register_pack: pack_register_ids(values.rs1_index, values.rs2_index, values.rd_index)?,
            jolt_tag: kind.tag().0,
            circuit_flags,
            instruction_flags,
            imm_is_negative: imm < 0,
            row_class,
        })
    }

    #[inline(always)]
    pub fn rs1_value(&self) -> u64 {
        self.values.slot0
    }

    #[inline(always)]
    pub fn rs2_value(&self) -> u64 {
        match self.row_class {
            RowClass::NonMemory | RowClass::Store => self.values.slot1,
            RowClass::Load => 0,
        }
    }

    #[inline(always)]
    pub fn rd_pre_value(&self) -> u64 {
        match self.row_class {
            RowClass::NonMemory | RowClass::Load => self.values.slot2,
            RowClass::Store => 0,
        }
    }

    #[inline(always)]
    pub fn rd_write_value(&self) -> u64 {
        match self.row_class {
            RowClass::NonMemory | RowClass::Load => self.values.slot3,
            RowClass::Store => 0,
        }
    }

    #[inline(always)]
    pub fn ram_address(&self) -> u64 {
        match self.row_class {
            RowClass::NonMemory => 0,
            RowClass::Load => self.values.slot1,
            RowClass::Store => self.values.slot3,
        }
    }

    #[inline(always)]
    pub fn ram_read_value(&self) -> u64 {
        match self.row_class {
            RowClass::NonMemory => 0,
            RowClass::Load => self.values.slot3,
            RowClass::Store => self.values.slot2,
        }
    }

    #[inline(always)]
    pub fn ram_write_value(&self) -> u64 {
        match self.row_class {
            RowClass::NonMemory => 0,
            RowClass::Load => self.values.slot3,
            RowClass::Store => self.values.slot1,
        }
    }

    /// Expanded PC (local bytecode index) as a raw integer.
    #[inline(always)]
    pub fn pc(&self) -> u64 {
        self.bytecode_pc as u64
    }

    /// Expanded PC as the logical `u64` type.
    #[inline(always)]
    pub fn bytecode_index(&self) -> u64 {
        self.bytecode_pc as u64
    }

    /// Source RV64 instruction address.
    #[inline(always)]
    pub fn unexpanded_pc(&self) -> u64 {
        self.unexpanded_pc
    }

    #[inline(always)]
    pub fn imm(&self) -> i128 {
        let magnitude = self.imm_abs as i128;
        if self.imm_is_negative {
            -magnitude
        } else {
            magnitude
        }
    }

    #[inline(always)]
    pub fn rs1_index(&self) -> Option<u8> {
        unpack_register_id(self.register_pack.to_le_bytes()[0])
    }

    #[inline(always)]
    pub fn rs2_index(&self) -> Option<u8> {
        unpack_register_id(self.register_pack.to_le_bytes()[1])
    }

    #[inline(always)]
    pub fn rd_index(&self) -> Option<u8> {
        unpack_register_id(self.register_pack.to_le_bytes()[2])
    }

    #[inline(always)]
    pub fn circuit_flags(&self) -> CircuitFlagSet {
        self.circuit_flags
    }

    #[inline(always)]
    pub fn instruction_flags(&self) -> InstructionFlagSet {
        self.instruction_flags
    }

    #[inline(always)]
    pub fn is_load(&self) -> bool {
        matches!(self.row_class, RowClass::Load)
    }

    #[inline(always)]
    pub fn is_store(&self) -> bool {
        matches!(self.row_class, RowClass::Store)
    }

    #[inline(always)]
    pub fn is_noop(&self) -> bool {
        self.instruction_flags.get(InstructionFlags::IsNoop)
    }

    /// Stable final-row identity, reconstructed from the cached tag.
    #[inline]
    pub fn instruction_kind(&self) -> Option<JoltInstructionKind> {
        JoltInstructionKind::from_tag(JoltInstructionTag(self.jolt_tag))
    }
}

/// Cached register-pack value for a row with no register operands.
const NO_REGISTERS: u32 = u32::from_le_bytes([REGISTER_NONE, REGISTER_NONE, REGISTER_NONE, 0]);

/// Circuit + instruction flags for a final instruction row.
///
/// `TryFrom<JoltInstructionRow>` is exhaustive over `instruction_kind`, so this
/// never actually fails; the fallback keeps the function total without a panic.
#[inline]
fn row_flags(instruction: &JoltInstructionRow) -> (CircuitFlagSet, InstructionFlagSet) {
    match JoltInstruction::try_from(*instruction) {
        Ok(instruction) => (instruction.circuit_flags(), instruction.instruction_flags()),
        Err(_) => (CircuitFlagSet::default(), InstructionFlagSet::default()),
    }
}

#[inline]
fn violation(kind: JoltInstructionKind, detail: &'static str) -> TraceRowError {
    TraceRowError::MemoryRowContractViolation { kind, detail }
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
fn pack_register_ids(
    rs1: Option<u8>,
    rs2: Option<u8>,
    rd: Option<u8>,
) -> Result<u32, TraceRowError> {
    Ok(u32::from_le_bytes([
        checked_register_id(rs1)?,
        checked_register_id(rs2)?,
        checked_register_id(rd)?,
        0,
    ]))
}

#[inline(always)]
fn unpack_register_id(byte: u8) -> Option<u8> {
    (byte != REGISTER_NONE).then_some(byte)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests may unwrap freely")]
mod tests {
    use super::*;
    use crate::NormalizedOperands;

    fn row(kind: JoltInstructionKind, operands: NormalizedOperands) -> JoltInstructionRow {
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
    fn layout_is_64_bytes() {
        assert_eq!(core::mem::size_of::<JoltTraceRow>(), 64);
        assert_eq!(core::mem::align_of::<JoltTraceRow>(), 8);
    }

    #[test]
    fn default_is_canonical_no_op() {
        let default = JoltTraceRow::default();
        assert_eq!(default, JoltTraceRow::no_op());
        assert_eq!(default.rs1_value(), 0);
        assert_eq!(default.rs2_value(), 0);
        assert_eq!(default.rd_pre_value(), 0);
        assert_eq!(default.rd_write_value(), 0);
        assert_eq!(default.ram_address(), 0);
        assert_eq!(default.ram_read_value(), 0);
        assert_eq!(default.ram_write_value(), 0);
        assert_eq!(default.pc(), 0);
        assert_eq!(default.unexpanded_pc(), 0);
        assert_eq!(default.imm(), 0);
        assert!(default.is_noop());
        assert!(!default.is_load() && !default.is_store());
    }

    #[test]
    fn non_memory_row_round_trips_register_columns() {
        let values = LogicalValues {
            rs1_value: 11,
            rs2_value: 22,
            rd_pre_value: 33,
            rd_write_value: 44,
            rs1_index: Some(2),
            rs2_index: Some(3),
            rd_index: Some(1),
            ..LogicalValues::ZERO
        };
        let instruction = row(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(2),
                rs2: Some(3),
                rd: Some(1),
                imm: 0,
            },
        );
        let r = JoltTraceRow::from_components(&values, &instruction, 7).unwrap();

        assert_eq!(r.rs1_value(), 11);
        assert_eq!(r.rs2_value(), 22);
        assert_eq!(r.rd_pre_value(), 33);
        assert_eq!(r.rd_write_value(), 44);
        assert_eq!(r.ram_address(), 0);
        assert_eq!(r.pc(), 7);
        assert_eq!(r.bytecode_index(), 7);
        assert_eq!(r.unexpanded_pc(), 0x8000_0000);
        assert_eq!(r.rs1_index(), Some(2));
        assert_eq!(r.rs2_index(), Some(3));
        assert_eq!(r.rd_index(), Some(1));
        assert!(!r.is_load() && !r.is_store());
    }

    #[test]
    fn load_row_aliases_ram_into_rd_slot() {
        let loaded = 0xdead_beefu64;
        let values = LogicalValues {
            rs1_value: 0x1000,
            rd_pre_value: 5,
            rd_write_value: loaded,
            ram_address: 0x2000,
            ram_read_value: loaded,
            ram_write_value: loaded,
            rs1_index: Some(10),
            rd_index: Some(11),
            ..LogicalValues::ZERO
        };
        let instruction = row(
            JoltInstructionKind::LD,
            NormalizedOperands {
                rs1: Some(10),
                rs2: None,
                rd: Some(11),
                imm: 8,
            },
        );
        let r = JoltTraceRow::from_components(&values, &instruction, 3).unwrap();

        assert!(r.is_load());
        assert_eq!(r.rs1_value(), 0x1000);
        assert_eq!(r.rs2_value(), 0, "loads have no rs2");
        assert_eq!(r.rd_pre_value(), 5);
        assert_eq!(r.rd_write_value(), loaded);
        assert_eq!(r.ram_address(), 0x2000);
        assert_eq!(r.ram_read_value(), loaded);
        assert_eq!(r.ram_write_value(), loaded);
        assert_eq!(r.imm(), 8);
    }

    #[test]
    fn store_row_aliases_rs2_into_ram_write_slot() {
        let stored = 0x1234u64;
        let old = 0x5678u64;
        let values = LogicalValues {
            rs1_value: 0x3000,
            rs2_value: stored,
            ram_address: 0x4000,
            ram_read_value: old,
            ram_write_value: stored,
            rs1_index: Some(10),
            rs2_index: Some(12),
            ..LogicalValues::ZERO
        };
        let instruction = row(
            JoltInstructionKind::SD,
            NormalizedOperands {
                rs1: Some(10),
                rs2: Some(12),
                rd: None,
                imm: -4,
            },
        );
        let r = JoltTraceRow::from_components(&values, &instruction, 9).unwrap();

        assert!(r.is_store());
        assert_eq!(r.rs1_value(), 0x3000);
        assert_eq!(r.rs2_value(), stored);
        assert_eq!(r.rd_pre_value(), 0, "stores have no rd");
        assert_eq!(r.rd_write_value(), 0, "stores have no rd");
        assert_eq!(r.ram_address(), 0x4000);
        assert_eq!(r.ram_read_value(), old);
        assert_eq!(r.ram_write_value(), stored);
        assert_eq!(r.imm(), -4);
    }

    #[test]
    fn rejects_load_contract_violation() {
        let values = LogicalValues {
            rd_write_value: 1,
            ram_read_value: 2,
            ram_write_value: 1,
            ..LogicalValues::ZERO
        };
        let instruction = row(JoltInstructionKind::LD, NormalizedOperands::default());
        let err = JoltTraceRow::from_components(&values, &instruction, 0).unwrap_err();
        assert!(matches!(
            err,
            TraceRowError::MemoryRowContractViolation { .. }
        ));
    }

    #[test]
    fn rejects_imm_overflow() {
        let values = LogicalValues::ZERO;
        let instruction = row(
            JoltInstructionKind::ADDI,
            NormalizedOperands {
                rs1: None,
                rs2: None,
                rd: None,
                imm: i128::MAX,
            },
        );
        let err = JoltTraceRow::from_components(&values, &instruction, 0).unwrap_err();
        assert!(matches!(err, TraceRowError::ImmTooWide { .. }));
    }

    #[test]
    fn rejects_register_id_too_wide() {
        let values = LogicalValues {
            rs1_index: Some(REGISTER_NONE),
            ..LogicalValues::ZERO
        };
        let instruction = row(JoltInstructionKind::ADD, NormalizedOperands::default());
        let err = JoltTraceRow::from_components(&values, &instruction, 0).unwrap_err();
        assert!(matches!(err, TraceRowError::RegisterIdTooWide { .. }));
    }
}
