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
    JoltInstructionKind, JoltInstructionRow, JoltInstructionTag,
};

/// Largest register id storable in a register-id byte. `0xFF` is reserved as the
/// `None` sentinel, so ids must be `<= 254`. Jolt's register file
/// (`REGISTER_COUNT = 128`) sits well within this bound; the limit is a
/// storage-format detail, not a protocol fact.
const MAX_REGISTER_ID: u8 = u8::MAX - 1;

/// Sentinel stored in a register-id byte for an absent (`None`) operand.
const REGISTER_NONE: u8 = u8::MAX;

/// `meta` bit layout: `circuit_flags` occupy the low 16 bits; instruction flags
/// the next 6; the immediate sign one more; the top 9 bits are spare.
const META_INSTRUCTION_FLAGS_SHIFT: u32 = 16;
const META_INSTRUCTION_FLAGS_MASK: u32 = (1u32 << (crate::NUM_INSTRUCTION_FLAGS as u32)) - 1;
const META_IMM_NEGATIVE_SHIFT: u32 = 22;

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
    /// Magnitude of the immediate; sign is bit `META_IMM_NEGATIVE_SHIFT` of `meta`.
    imm_abs: u64,
    /// Compact local bytecode index (expanded "PC"); see [`JoltTraceRow::pc`].
    bytecode_pc: u32,
    /// Packed flags + immediate sign: circuit flags in bits `0..16`, instruction
    /// flags in bits `16..22`, immediate sign in bit `22`, top 9 bits spare.
    meta: u32,
    /// Final Jolt instruction tag (stable identity, not a dense index). The
    /// lookup-table routing is derived from this in `jolt-lookup-tables`.
    jolt_tag: u16,
    /// `rs1`/`rs2`/`rd` register ids, or `0xFF` (None).
    rs1_id: u8,
    rs2_id: u8,
    rd_id: u8,
    /// Reserved layout slots (kept zero).
    _reserved: [u8; 3],
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
            meta: pack_meta(circuit_flags, instruction_flags, false),
            jolt_tag: instruction.instruction_kind.tag().0,
            rs1_id: REGISTER_NONE,
            rs2_id: REGISTER_NONE,
            rd_id: REGISTER_NONE,
            _reserved: [0; 3],
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
            meta: pack_meta(circuit_flags, instruction_flags, imm < 0),
            jolt_tag: kind.tag().0,
            rs1_id: checked_register_id(instruction.operands.rs1)?,
            rs2_id: checked_register_id(instruction.operands.rs2)?,
            rd_id: checked_register_id(instruction.operands.rd)?,
            _reserved: [0; 3],
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
        if self.meta & (1 << META_IMM_NEGATIVE_SHIFT) != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    #[inline(always)]
    pub fn rs1_index(&self) -> Option<u8> {
        register_index(self.rs1_id)
    }

    #[inline(always)]
    pub fn rs2_index(&self) -> Option<u8> {
        register_index(self.rs2_id)
    }

    #[inline(always)]
    pub fn rd_index(&self) -> Option<u8> {
        register_index(self.rd_id)
    }

    #[inline(always)]
    pub fn circuit_flags(&self) -> CircuitFlagSet {
        CircuitFlagSet::from_bits(self.meta as u16)
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
fn pack_meta(
    circuit_flags: CircuitFlagSet,
    instruction_flags: InstructionFlagSet,
    imm_negative: bool,
) -> u32 {
    (circuit_flags.bits() as u32)
        | ((instruction_flags.bits() as u32) << META_INSTRUCTION_FLAGS_SHIFT)
        | ((imm_negative as u32) << META_IMM_NEGATIVE_SHIFT)
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
fn register_index(id: u8) -> Option<u8> {
    (id != REGISTER_NONE).then_some(id)
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
        assert_eq!(default.rs1_index(), None);
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

    #[test]
    fn flags_round_trip_through_meta() {
        // Distinct circuit + instruction flags must survive the meta packing.
        let instruction = row(
            JoltInstructionKind::SD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: None,
                imm: 0,
            },
        );
        let (circuit_flags, instruction_flags) = row_flags(&instruction);
        let state = CapturedState::Store(StoreState::default());
        let r = JoltTraceRow::from_components(state, &instruction, 0).unwrap();
        assert_eq!(r.circuit_flags(), circuit_flags);
        assert_eq!(r.instruction_flags(), instruction_flags);
    }
}
