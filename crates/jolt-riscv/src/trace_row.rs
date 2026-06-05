//! Proof-facing materialized trace row (`JoltTraceRow`).
//!
//! `JoltTraceRow` is a compact, `Copy` row built once after tracing and final
//! bytecode expansion. It replaces repeated prover-side adaptation of a raw
//! tracer cycle with a flat representation whose **logical proof columns** are
//! exposed through accessors, while the **physical storage** is private and free
//! to alias mutually-exclusive or equal values for final memory rows.
//!
//! # Captured state
//!
//! The per-cycle witness values are described by [`CapturedState`], a typed enum
//! over the three final row classes (`NonMemory` / `Load` / `Store`). Each
//! variant only names the columns that are independent for that class, so the
//! memory-row aliasing is enforced by the type rather than by runtime checks:
//! a load's `RamReadValue`, `RamWriteValue`, and `RdWriteValue` are one field,
//! and a store's `RamWriteValue` and `Rs2Value` are one field. The cached
//! `Load`/`Store` circuit flags determine the class on read, so the enum is the
//! producer/accessor view while storage stays flat (no separate discriminant).
//!
//! # Crate boundaries
//!
//! This type lives in `jolt-riscv` and depends only on `jolt-riscv`-native
//! types. The `Cycle` → `JoltTraceRow` conversion (and the contract checks that
//! the cycle's raw values collapse correctly) lives in `tracer`; the
//! lookup-table accessor lives in `jolt-lookup-tables`.
//!
//! # Logical vs physical
//!
//! Proof code must depend on the logical accessors (`rs1_value`, `ram_address`,
//! `captured_state`, ...), never on the private storage slots, so the physical
//! layout stays swappable.

use crate::{
    CircuitFlagSet, CircuitFlags, Flags, InstructionFlagSet, InstructionFlags, JoltInstruction,
    JoltInstructionKind, JoltInstructionRow, JoltInstructionTag, NormalizedOperands,
};

/// Largest register id storable in `register_pack`. `0xFF` is reserved as the
/// `None` sentinel, so ids must be `<= 254`. Jolt's register file
/// (`REGISTER_COUNT = 128`) sits well within this bound; the limit is a
/// storage-format detail, not a protocol fact.
const MAX_REGISTER_ID: u8 = u8::MAX - 1;

/// Sentinel stored in `register_pack` for an absent (`None`) register operand.
const REGISTER_NONE: u8 = u8::MAX;

/// Cached register-pack value for a row with no register operands.
const NO_REGISTERS: u32 = u32::from_le_bytes([REGISTER_NONE, REGISTER_NONE, REGISTER_NONE, 0]);

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
/// This is both the [`JoltTraceRow::from_components`] input and the
/// [`JoltTraceRow::captured_state`] view. Register *indices* are not part of it;
/// they come from the instruction's operands, so they are not duplicated.
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
    /// Pack into flat value slots, validating that the variant agrees with the
    /// row's `Load`/`Store` circuit flags.
    fn into_value_slots(
        self,
        is_load: bool,
        is_store: bool,
        kind: JoltInstructionKind,
    ) -> Result<TraceValueSlots, TraceRowError> {
        match self {
            CapturedState::NonMemory(s) => {
                if is_load || is_store {
                    return Err(class_mismatch(kind, "expected a load/store captured state"));
                }
                Ok(TraceValueSlots {
                    slot0: s.rs1_value,
                    slot1: s.rs2_value,
                    slot2: s.rd_pre_value,
                    slot3: s.rd_write_value,
                })
            }
            CapturedState::Load(s) => {
                if !is_load {
                    return Err(class_mismatch(
                        kind,
                        "load captured state for a non-load row",
                    ));
                }
                Ok(TraceValueSlots {
                    slot0: s.rs1_value,
                    slot1: s.ram_address,
                    slot2: s.rd_pre_value,
                    slot3: s.rd_write_value,
                })
            }
            CapturedState::Store(s) => {
                if !is_store {
                    return Err(class_mismatch(
                        kind,
                        "store captured state for a non-store row",
                    ));
                }
                Ok(TraceValueSlots {
                    slot0: s.rs1_value,
                    slot1: s.rs2_value,
                    slot2: s.ram_read_value,
                    slot3: s.ram_address,
                })
            }
        }
    }
}

/// Error raised when a captured state cannot be represented for an instruction.
///
/// Cycle-/bytecode-specific failures (source-only cycles, oversized bytecode
/// indices, and the cycle-value contract checks) are surfaced by the `tracer`
/// conversion, not here, since this crate has no notion of either.
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
    /// The captured-state variant disagrees with the instruction's `Load`/
    /// `Store` circuit flags.
    #[error("captured-state class does not match {kind:?}: {detail}")]
    StateClassMismatch {
        kind: JoltInstructionKind,
        detail: &'static str,
    },
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
    /// Magnitude of the immediate; sign in `imm_is_negative`.
    imm_abs: u64,
    /// Compact local bytecode index (expanded "PC"); see [`JoltTraceRow::pc`].
    bytecode_pc: u32,
    /// `rs1 | rs2 << 8 | rd << 16`, each byte a register id or `0xFF` (None).
    register_pack: u32,
    /// Final Jolt instruction tag (stable identity, not a dense index). The
    /// lookup-table routing is derived from this in `jolt-lookup-tables`.
    jolt_tag: u16,
    /// Cached circuit flags. Also the row-class discriminant: the `Load`/`Store`
    /// bits decide how the value slots are interpreted.
    circuit_flags: CircuitFlagSet,
    /// Cached instruction flags.
    instruction_flags: InstructionFlagSet,
    imm_is_negative: bool,
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
            register_pack: NO_REGISTERS,
            jolt_tag: instruction.instruction_kind.tag().0,
            circuit_flags,
            instruction_flags,
            imm_is_negative: false,
        }
    }

    /// Build a row from a captured state, the final instruction row, and its
    /// compact bytecode index.
    ///
    /// The captured-state variant must agree with the instruction's `Load`/
    /// `Store` flags; register indices and the immediate are taken from the
    /// instruction's operands (not the captured state), so they are not
    /// duplicated.
    pub fn from_components(
        state: CapturedState,
        instruction: &JoltInstructionRow,
        bytecode_pc: u32,
    ) -> Result<Self, TraceRowError> {
        let (circuit_flags, instruction_flags) = row_flags(instruction);
        let kind = instruction.instruction_kind;
        let values = state.into_value_slots(
            circuit_flags.get(CircuitFlags::Load),
            circuit_flags.get(CircuitFlags::Store),
            kind,
        )?;

        let imm = instruction.operands.imm;
        let imm_magnitude = imm.unsigned_abs();
        if imm_magnitude > u64::MAX as u128 {
            return Err(TraceRowError::ImmTooWide { imm });
        }

        Ok(Self {
            values,
            unexpanded_pc: instruction.address as u64,
            imm_abs: imm_magnitude as u64,
            bytecode_pc,
            register_pack: pack_register_ids(&instruction.operands)?,
            jolt_tag: kind.tag().0,
            circuit_flags,
            instruction_flags,
            imm_is_negative: imm < 0,
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
        self.circuit_flags.get(CircuitFlags::Load)
    }

    #[inline(always)]
    pub fn is_store(&self) -> bool {
        self.circuit_flags.get(CircuitFlags::Store)
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

/// Circuit + instruction flags for a final instruction row.
///
/// `TryFrom<JoltInstructionRow>` is exhaustive over `instruction_kind`, so this
/// never actually fails; the fallback keeps the function total without a panic.
#[inline]
fn row_flags(instruction: &JoltInstructionRow) -> (CircuitFlagSet, InstructionFlagSet) {
    JoltInstruction::try_from(*instruction)
        .map(|instruction| (instruction.circuit_flags(), instruction.instruction_flags()))
        .unwrap_or_default()
}

#[inline]
fn class_mismatch(kind: JoltInstructionKind, detail: &'static str) -> TraceRowError {
    TraceRowError::StateClassMismatch { kind, detail }
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
fn pack_register_ids(operands: &NormalizedOperands) -> Result<u32, TraceRowError> {
    Ok(u32::from_le_bytes([
        checked_register_id(operands.rs1)?,
        checked_register_id(operands.rs2)?,
        checked_register_id(operands.rd)?,
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
        assert_eq!(
            default.captured_state(),
            CapturedState::NonMemory(NonMemoryState::default())
        );
    }

    #[test]
    fn non_memory_row_round_trips_columns() {
        let state = CapturedState::NonMemory(NonMemoryState {
            rs1_value: 11,
            rs2_value: 22,
            rd_pre_value: 33,
            rd_write_value: 44,
        });
        let instruction = row(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(2),
                rs2: Some(3),
                rd: Some(1),
                imm: 0,
            },
        );
        let r = JoltTraceRow::from_components(state, &instruction, 7).unwrap();

        assert_eq!(r.rs1_value(), 11);
        assert_eq!(r.rs2_value(), 22);
        assert_eq!(r.rd_pre_value(), 33);
        assert_eq!(r.rd_write_value(), 44);
        assert_eq!(r.ram_address(), 0);
        assert_eq!(r.pc(), 7);
        assert_eq!(r.unexpanded_pc(), 0x8000_0000);
        // Register indices come from the instruction operands.
        assert_eq!(r.rs1_index(), Some(2));
        assert_eq!(r.rs2_index(), Some(3));
        assert_eq!(r.rd_index(), Some(1));
        assert!(!r.is_load() && !r.is_store());
        assert_eq!(r.captured_state(), state);
    }

    #[test]
    fn load_row_aliases_ram_into_rd_slot() {
        let loaded = 0xdead_beefu64;
        let state = CapturedState::Load(LoadState {
            rs1_value: 0x1000,
            ram_address: 0x2000,
            rd_pre_value: 5,
            rd_write_value: loaded,
        });
        let instruction = row(
            JoltInstructionKind::LD,
            NormalizedOperands {
                rs1: Some(10),
                rs2: None,
                rd: Some(11),
                imm: 8,
            },
        );
        let r = JoltTraceRow::from_components(state, &instruction, 3).unwrap();

        assert!(r.is_load());
        assert_eq!(r.rs1_value(), 0x1000);
        assert_eq!(r.rs2_value(), 0, "loads have no rs2");
        assert_eq!(r.rd_pre_value(), 5);
        assert_eq!(r.rd_write_value(), loaded);
        assert_eq!(r.ram_address(), 0x2000);
        assert_eq!(r.ram_read_value(), loaded);
        assert_eq!(r.ram_write_value(), loaded);
        assert_eq!(r.imm(), 8);
        assert_eq!(r.rs1_index(), Some(10));
        assert_eq!(r.rs2_index(), None);
        assert_eq!(r.rd_index(), Some(11));
        assert_eq!(r.captured_state(), state);
    }

    #[test]
    fn store_row_aliases_rs2_into_ram_write_slot() {
        let stored = 0x1234u64;
        let old = 0x5678u64;
        let state = CapturedState::Store(StoreState {
            rs1_value: 0x3000,
            rs2_value: stored,
            ram_read_value: old,
            ram_address: 0x4000,
        });
        let instruction = row(
            JoltInstructionKind::SD,
            NormalizedOperands {
                rs1: Some(10),
                rs2: Some(12),
                rd: None,
                imm: -4,
            },
        );
        let r = JoltTraceRow::from_components(state, &instruction, 9).unwrap();

        assert!(r.is_store());
        assert_eq!(r.rs1_value(), 0x3000);
        assert_eq!(r.rs2_value(), stored);
        assert_eq!(r.rd_pre_value(), 0, "stores have no rd");
        assert_eq!(r.rd_write_value(), 0, "stores have no rd");
        assert_eq!(r.ram_address(), 0x4000);
        assert_eq!(r.ram_read_value(), old);
        assert_eq!(r.ram_write_value(), stored);
        assert_eq!(r.imm(), -4);
        assert_eq!(r.captured_state(), state);
    }

    #[test]
    fn rejects_class_mismatch() {
        // A load instruction with a non-memory captured state.
        let state = CapturedState::NonMemory(NonMemoryState::default());
        let instruction = row(JoltInstructionKind::LD, NormalizedOperands::default());
        let err = JoltTraceRow::from_components(state, &instruction, 0).unwrap_err();
        assert!(matches!(err, TraceRowError::StateClassMismatch { .. }));

        // A non-memory instruction with a store captured state.
        let state = CapturedState::Store(StoreState::default());
        let instruction = row(JoltInstructionKind::ADD, NormalizedOperands::default());
        let err = JoltTraceRow::from_components(state, &instruction, 0).unwrap_err();
        assert!(matches!(err, TraceRowError::StateClassMismatch { .. }));
    }

    #[test]
    fn rejects_imm_overflow() {
        let instruction = row(
            JoltInstructionKind::ADDI,
            NormalizedOperands {
                rs1: None,
                rs2: None,
                rd: None,
                imm: i128::MAX,
            },
        );
        let err =
            JoltTraceRow::from_components(CapturedState::default(), &instruction, 0).unwrap_err();
        assert!(matches!(err, TraceRowError::ImmTooWide { .. }));
    }

    #[test]
    fn rejects_register_id_too_wide() {
        let instruction = row(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(REGISTER_NONE),
                rs2: None,
                rd: None,
                imm: 0,
            },
        );
        let err =
            JoltTraceRow::from_components(CapturedState::default(), &instruction, 0).unwrap_err();
        assert!(matches!(err, TraceRowError::RegisterIdTooWide { .. }));
    }
}
