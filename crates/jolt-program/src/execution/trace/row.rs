#[cfg(feature = "field-inline")]
use std::sync::Arc;

use jolt_riscv::{
    CircuitFlagSet, CircuitFlags, Flags, InstructionFlagSet, JoltCycle, JoltInstruction,
    JoltInstructionKind, JoltInstructionRow, JoltInstructionTag, NormalizedOperands,
};
#[cfg(feature = "serialization")]
use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "field-inline")]
use crate::field_inline::FieldInlineTraceData;

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

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("trace row for {kind:?} violates the final memory-row contract: {reason}")]
pub struct TraceRowError {
    kind: JoltInstructionKind,
    reason: String,
}

impl TraceRowError {
    fn new(kind: JoltInstructionKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            reason: reason.into(),
        }
    }
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

const IS_LOAD: u16 = 1 << CircuitFlags::Load as u16;
const IS_STORE: u16 = 1 << CircuitFlags::Store as u16;
const VSR_NONE: u16 = u16::MAX;
const OPERAND_NONE: u8 = u8::MAX;

/// One execution cycle packed to 64 bytes. Four value slots use the final
/// memory-row layout (`specs/proof-trace-row-layout.md`):
///
/// - non-memory: `rs1`, `rs2`, `rd_pre`, `rd_post` (RAM values must be zero);
/// - load: `rs1`, `ram_address`, `rd_pre`, `rd_post` (= the RAM value; no rs2);
/// - store: `rs1`, `rs2` (= RAM post), `ram_pre`, `ram_address` (no rd).
///
/// Recorded register ids stay separate from instruction operand ids.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(feature = "field-inline"), derive(Copy))]
#[repr(C)]
pub struct TraceRow {
    slots: [u64; 4],
    address: u64,
    imm_abs: u64,
    circuit_flags: u16,
    kind_tag: u16,
    virtual_sequence_remaining: u16,
    instruction_flags: u8,
    meta: TraceRowMeta,
    rs1_register: u8,
    rs2_register: u8,
    rd_register: u8,
    rs1_operand: u8,
    rs2_operand: u8,
    rd_operand: u8,
    #[cfg(feature = "field-inline")]
    pub field_inline: Option<Arc<FieldInlineTraceData>>,
}

#[cfg(feature = "serialization")]
#[derive(Serialize, Deserialize)]
struct TraceRowWire {
    instruction: JoltInstructionRow,
    registers: RegisterState,
    ram_access: RamAccess,
    #[cfg(feature = "field-inline")]
    field_inline: Option<Arc<FieldInlineTraceData>>,
}

#[cfg(feature = "serialization")]
impl Serialize for TraceRow {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        TraceRowWire {
            instruction: self.instruction(),
            registers: self.registers(),
            ram_access: self.ram_access(),
            #[cfg(feature = "field-inline")]
            field_inline: self.field_inline.clone(),
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serialization")]
impl<'de> Deserialize<'de> for TraceRow {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = TraceRowWire::deserialize(deserializer)?;
        #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
        let mut row =
            Self::new(wire.instruction, wire.registers, wire.ram_access).map_err(Error::custom)?;
        #[cfg(feature = "field-inline")]
        {
            row.field_inline = wire.field_inline;
        }
        Ok(row)
    }
}

#[cfg(not(feature = "field-inline"))]
const _: () = assert!(
    std::mem::size_of::<TraceRow>() == 64,
    "TraceRow must stay 64 bytes; any size change should be intentional and reviewed"
);

impl Default for TraceRow {
    fn default() -> Self {
        let Ok(row) = Self::from_instruction(JoltInstructionRow::default()) else {
            unreachable!("the canonical padding row must satisfy the packed-row contract")
        };
        row
    }
}

fn checked_operand_id(kind: JoltInstructionKind, id: Option<u8>) -> Result<u8, TraceRowError> {
    match id {
        None => Ok(OPERAND_NONE),
        Some(id) if id < OPERAND_NONE => Ok(id),
        Some(id) => Err(TraceRowError::new(
            kind,
            format!("operand register id {id} is reserved"),
        )),
    }
}

impl TraceRow {
    /// Packs one cycle after checking every aliased-slot invariant.
    pub fn new(
        instruction: JoltInstructionRow,
        registers: RegisterState,
        ram_access: RamAccess,
    ) -> Result<Self, TraceRowError> {
        let kind = instruction.instruction_kind;
        let (circuit_flags, instruction_flags) = JoltInstruction::try_from(instruction)
            .map(|instruction| (instruction.circuit_flags(), instruction.instruction_flags()))
            .map_err(|_| {
                TraceRowError::new(kind, "instruction kind has no JoltInstruction lowering")
            })?;
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
                return Err(TraceRowError::new(kind, "load row reads rs2"));
            }
            if ram_kind != RamAccessKind::NoOp
                && (ram_pre != rd.post_value || ram_post != rd.post_value)
            {
                return Err(TraceRowError::new(
                    kind,
                    "load RAM value must equal the rd write value",
                ));
            }
            [rs1.value, ram_address, rd.pre_value, rd.post_value]
        } else if is_store {
            if registers.rd.is_some() {
                return Err(TraceRowError::new(kind, "store row writes rd"));
            }
            if ram_kind == RamAccessKind::Write && ram_post != rs2.value {
                return Err(TraceRowError::new(
                    kind,
                    "store RAM write value must equal the rs2 value",
                ));
            }
            [rs1.value, rs2.value, ram_pre, ram_address]
        } else {
            if ram_address != 0 || ram_pre != 0 || ram_post != 0 {
                return Err(TraceRowError::new(
                    kind,
                    "non-memory row carries RAM values",
                ));
            }
            [rs1.value, rs2.value, rd.pre_value, rd.post_value]
        };

        let imm = instruction.operands.imm;
        let imm_abs = u64::try_from(imm.unsigned_abs()).map_err(|_| {
            TraceRowError::new(kind, "immediate does not fit the u64 magnitude encoding")
        })?;
        let meta = TraceRowMeta::new(&registers, ram_kind, imm < 0);
        let virtual_sequence_remaining = match instruction.virtual_sequence_remaining {
            None => VSR_NONE,
            Some(VSR_NONE) => {
                return Err(TraceRowError::new(
                    kind,
                    "virtual_sequence_remaining collides with the sentinel",
                ))
            }
            Some(remaining) => remaining,
        };

        Ok(Self {
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
            rs1_operand: checked_operand_id(kind, instruction.operands.rs1)?,
            rs2_operand: checked_operand_id(kind, instruction.operands.rs2)?,
            rd_operand: checked_operand_id(kind, instruction.operands.rd)?,
            #[cfg(feature = "field-inline")]
            field_inline: None,
        })
    }

    pub fn from_instruction(instruction: JoltInstructionRow) -> Result<Self, TraceRowError> {
        Self::new(instruction, RegisterState::default(), RamAccess::NoOp)
    }

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

    #[inline]
    pub fn circuit_flags(&self) -> CircuitFlagSet {
        CircuitFlagSet::from_bits(self.circuit_flags)
    }

    #[inline]
    pub fn instruction_flags(&self) -> InstructionFlagSet {
        InstructionFlagSet::from_bits(self.instruction_flags)
    }

    #[inline]
    #[expect(clippy::expect_used, reason = "the tag is checked before storage")]
    pub fn instruction_kind(&self) -> JoltInstructionKind {
        JoltInstructionKind::from_tag(JoltInstructionTag(self.kind_tag))
            .expect("trace row kind tag was checked before storage")
    }

    #[inline]
    pub fn address(&self) -> u64 {
        self.address
    }

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
                let row =
                    TraceRow::new(JoltInstructionRow::default(), registers, ram_access).unwrap();
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
        )
        .unwrap();
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
        )
        .unwrap();
        assert_eq!(row.registers(), registers);
        assert_eq!(row.ram_access(), ram_access);
        assert_eq!(row.imm(), -4);
        assert_eq!(JoltCycle::ram_read_value(&row), Some(0x5678));
        assert_eq!(JoltCycle::ram_write_value(&row), Some(stored));
    }

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
            let row = TraceRow::from_instruction(instruction).unwrap();
            let decoded = JoltInstruction::try_from(row.instruction()).unwrap();
            assert_eq!(row.circuit_flags(), decoded.circuit_flags());
            assert_eq!(row.instruction_flags(), decoded.instruction_flags());
        }
    }

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
        )
        .unwrap();
        assert_eq!(row.instruction(), source);
        assert_eq!(row.instruction_kind(), JoltInstructionKind::ADDI);
        assert_eq!(row.address(), 0x8000_0000);
        assert_eq!(row.imm(), -12345);
        assert_eq!(row.rs1_read().unwrap().register, 200);
    }

    #[test]
    fn default_row_is_the_canonical_padding_row() {
        let row = TraceRow::default();
        assert_eq!(row.registers(), RegisterState::default());
        assert_eq!(row.ram_access(), RamAccess::NoOp);
        assert_eq!(row.instruction_kind(), JoltInstructionKind::NoOp);
        assert_eq!(row.instruction(), JoltInstructionRow::default());
        assert_eq!(
            row,
            TraceRow::new(
                JoltInstructionRow::default(),
                RegisterState::default(),
                RamAccess::NoOp,
            )
            .unwrap()
        );
    }

    #[test]
    fn non_memory_row_with_ram_traffic_is_rejected() {
        let error = TraceRow::new(
            JoltInstructionRow::default(),
            RegisterState::default(),
            RamAccess::Write(RamWrite {
                address: 8,
                pre_value: 7,
                post_value: 11,
            }),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("non-memory row carries RAM values"));
    }

    #[test]
    fn load_row_with_mismatched_ram_value_is_rejected() {
        let error = TraceRow::new(
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
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("load RAM value must equal the rd write value"));
    }

    #[test]
    fn oversized_immediate_is_rejected() {
        let error = TraceRow::from_instruction(instruction(
            JoltInstructionKind::ADDI,
            NormalizedOperands {
                rs1: None,
                rs2: None,
                rd: None,
                imm: i128::MAX,
            },
        ))
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("immediate does not fit the u64 magnitude encoding"));
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn serialization_round_trips_through_checked_fields() {
        let row = TraceRow::new(
            instruction(
                JoltInstructionKind::ADDI,
                NormalizedOperands {
                    rs1: Some(2),
                    rs2: None,
                    rd: Some(1),
                    imm: -17,
                },
            ),
            RegisterState {
                rs1: Some(RegisterRead {
                    register: 2,
                    value: 9,
                }),
                rd: Some(RegisterWrite {
                    register: 1,
                    pre_value: 3,
                    post_value: 12,
                }),
                ..RegisterState::default()
            },
            RamAccess::NoOp,
        )
        .unwrap();
        let encoded = serde_json::to_vec(&row).unwrap();
        let decoded = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(row, decoded);
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn deserialization_rejects_unknown_instruction_tag() {
        let row = TraceRow::default();
        let mut encoded = serde_json::to_value(row).unwrap();
        let tag = encoded
            .pointer_mut("/instruction/instruction_kind")
            .unwrap();
        *tag = serde_json::json!(1);
        assert!(serde_json::from_value::<TraceRow>(encoded).is_err());
    }

    #[cfg(not(feature = "field-inline"))]
    #[test]
    fn trace_row_is_64_bytes() {
        assert_eq!(std::mem::size_of::<TraceRow>(), 64);
    }
}
