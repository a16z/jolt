use jolt_program::{
    execution::{RegisterRead, RegisterWrite},
    field_inline::{
        FieldEncodedValue, FieldInlineBridge, FieldInlineTraceData, FieldRegisterRead,
        FieldRegisterWrite,
    },
};
use jolt_riscv::{FieldInlineOp, NormalizedOperands};
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::{
    emulator::cpu::Cpu,
    instruction::{format::format_field_inline::FormatFieldInline, RISCVInstruction},
};

use super::{normalize_register_value, InstructionRegisterState, RegisterSnapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum FieldAccess<W> {
    Read(FieldRegisterRead),
    ReadPair {
        rs1: FieldRegisterRead,
        rs2: FieldRegisterRead,
    },
    Write(W),
    ReadWrite {
        rs1: FieldRegisterRead,
        rd: W,
    },
    BinaryWrite {
        rs1: FieldRegisterRead,
        rs2: FieldRegisterRead,
        rd: W,
    },
}

impl<W: Copy> FieldAccess<W> {
    fn reads(self) -> (Option<FieldRegisterRead>, Option<FieldRegisterRead>) {
        match self {
            Self::Read(rs1) | Self::ReadWrite { rs1, .. } => (Some(rs1), None),
            Self::ReadPair { rs1, rs2 } | Self::BinaryWrite { rs1, rs2, .. } => {
                (Some(rs1), Some(rs2))
            }
            Self::Write(_) => (None, None),
        }
    }

    fn write(self) -> Option<W> {
        match self {
            Self::Write(rd) | Self::ReadWrite { rd, .. } | Self::BinaryWrite { rd, .. } => Some(rd),
            Self::Read(_) | Self::ReadPair { .. } => None,
        }
    }

    fn map_write<V>(self, f: impl FnOnce(W) -> V) -> FieldAccess<V> {
        match self {
            Self::Read(rs1) => FieldAccess::Read(rs1),
            Self::ReadPair { rs1, rs2 } => FieldAccess::ReadPair { rs1, rs2 },
            Self::Write(rd) => FieldAccess::Write(f(rd)),
            Self::ReadWrite { rs1, rd } => FieldAccess::ReadWrite { rs1, rd: f(rd) },
            Self::BinaryWrite { rs1, rs2, rd } => FieldAccess::BinaryWrite {
                rs1,
                rs2,
                rd: f(rd),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum IntegerAccess<W> {
    None,
    Read(RegisterRead),
    Write(W),
    ReadWrite { rs1: RegisterRead, rd: W },
}

impl<W: Copy> IntegerAccess<W> {
    fn read(self) -> Option<RegisterRead> {
        match self {
            Self::Read(rs1) | Self::ReadWrite { rs1, .. } => Some(rs1),
            Self::None | Self::Write(_) => None,
        }
    }

    fn write(self) -> Option<W> {
        match self {
            Self::Write(rd) | Self::ReadWrite { rd, .. } => Some(rd),
            Self::None | Self::Read(_) => None,
        }
    }

    fn map_write<V>(self, f: impl FnOnce(W) -> V) -> IntegerAccess<V> {
        match self {
            Self::None => IntegerAccess::None,
            Self::Read(rs1) => IntegerAccess::Read(rs1),
            Self::Write(rd) => IntegerAccess::Write(f(rd)),
            Self::ReadWrite { rs1, rd } => IntegerAccess::ReadWrite { rs1, rd: f(rd) },
        }
    }
}

/// Register inputs captured before execution, including every destination's old value.
pub struct FieldInlineBefore {
    field: FieldAccess<FieldRegisterRead>,
    integer: IntegerAccess<RegisterRead>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterStateFieldInline {
    field: FieldAccess<FieldRegisterWrite>,
    integer: IntegerAccess<RegisterWrite>,
}

impl Default for RegisterStateFieldInline {
    // EnumIter needs a complete cycle template; live captures never use this value.
    fn default() -> Self {
        Self {
            field: FieldAccess::Read(FieldRegisterRead {
                register: 0,
                value: FieldEncodedValue::zero(),
            }),
            integer: IntegerAccess::None,
        }
    }
}

impl InstructionRegisterState for RegisterStateFieldInline {
    #[cfg(any(feature = "test-utils", test))]
    fn random(_rng: &mut StdRng, _operands: &NormalizedOperands) -> Self {
        Self::default()
    }

    fn rs1_value(&self) -> Option<u64> {
        self.integer.read().map(|read| read.value)
    }

    fn rd_values(&self) -> Option<(u64, u64)> {
        self.integer
            .write()
            .map(|write| (write.pre_value, write.post_value))
    }
}

impl<I: RISCVInstruction<Format = FormatFieldInline>> RegisterSnapshot<I>
    for RegisterStateFieldInline
{
    type Before = FieldInlineBefore;

    #[expect(
        clippy::expect_used,
        reason = "Only field instructions select the field register snapshot"
    )]
    fn capture_pre(instruction: &I, cpu: &Cpu) -> Self::Before {
        let op = jolt_riscv::field_inline_source_op(instruction.source_kind())
            .expect("field snapshot requires a field instruction");
        let shape = jolt_riscv::field_inline_operand_shape_for_op(op);
        let operands = *instruction.operands();
        let field_rd = if shape.field_rd_in_rs2_slot {
            operands.rs2
        } else {
            operands.rd
        };
        let field_rs1 = if shape.field_rs1_is_field_rd {
            field_rd
        } else {
            operands.rs1
        };
        let read_field = |register: Option<u8>| {
            let register = register.unwrap_or(0);
            FieldRegisterRead {
                register,
                value: cpu.field_registers.read(register),
            }
        };
        let field = match (
            shape.reads_field_rs1,
            shape.reads_field_rs2,
            shape.writes_field_rd,
        ) {
            (true, false, false) => FieldAccess::Read(read_field(field_rs1)),
            (true, true, false) => FieldAccess::ReadPair {
                rs1: read_field(field_rs1),
                rs2: read_field(operands.rs2),
            },
            (false, false, true) => FieldAccess::Write(read_field(field_rd)),
            (true, false, true) => FieldAccess::ReadWrite {
                rs1: read_field(field_rs1),
                rd: read_field(field_rd),
            },
            (true, true, true) => FieldAccess::BinaryWrite {
                rs1: read_field(field_rs1),
                rs2: read_field(operands.rs2),
                rd: read_field(field_rd),
            },
            _ => panic!("unsupported field register operand shape"),
        };
        let x_operands = shape.x_operands(NormalizedOperands::from(operands));
        let read_x = |register: u8| RegisterRead {
            register,
            value: normalize_register_value(cpu, usize::from(register)),
        };
        let integer = match (x_operands.rs1, x_operands.rd) {
            (None, None) => IntegerAccess::None,
            (Some(rs1), None) => IntegerAccess::Read(read_x(rs1)),
            (None, Some(rd)) => IntegerAccess::Write(read_x(rd)),
            (Some(rs1), Some(rd)) => IntegerAccess::ReadWrite {
                rs1: read_x(rs1),
                rd: read_x(rd),
            },
        };
        FieldInlineBefore { field, integer }
    }

    fn capture_post(_instruction: &I, before: Self::Before, cpu: &Cpu) -> Self {
        Self {
            field: before.field.map_write(|before| FieldRegisterWrite {
                register: before.register,
                pre_value: before.value,
                post_value: cpu.field_registers.read(before.register),
            }),
            integer: before.integer.map_write(|before| RegisterWrite {
                register: before.register,
                pre_value: before.value,
                post_value: normalize_register_value(cpu, usize::from(before.register)),
            }),
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "The field instruction's completed capture supplies every required bridge operand"
    )]
    fn field_inline_trace(instruction: &I, state: &Self) -> Option<FieldInlineTraceData> {
        let op = jolt_riscv::field_inline_source_op(instruction.source_kind())
            .expect("field snapshot requires a field instruction");
        let (rs1, rs2) = state.field.reads();
        let rd = state.field.write();
        let bridge = match op {
            FieldInlineOp::LoadAccumulateFromRegister => {
                let x_read = state.integer.read().expect("ingress reads an x-register");
                Some(FieldInlineBridge::LoadAccumulateFromRegister {
                    x_register: x_read.register,
                    x_value: x_read.value,
                    field_value: rd.expect("ingress writes a field register").post_value,
                })
            }
            FieldInlineOp::LoadAccumulateFromMemory => {
                let x_read = state.integer.read().expect("memory ingress reads a base");
                let x_write = state
                    .integer
                    .write()
                    .expect("memory ingress writes an x-register");
                Some(FieldInlineBridge::LoadAccumulateFromMemory {
                    x_base: x_read.register,
                    x_register: x_write.register,
                    word: x_write.post_value,
                    field_value: rd.expect("ingress writes a field register").post_value,
                })
            }
            FieldInlineOp::AdviceLimb => {
                let field_read = rs1.expect("limb advice reads a field register");
                let x_write = state
                    .integer
                    .write()
                    .expect("limb advice writes an x-register");
                Some(FieldInlineBridge::AdviceLimb {
                    field_register: field_read.register,
                    field_value: field_read.value,
                    x_register: x_write.register,
                    x_value: x_write.post_value,
                })
            }
            FieldInlineOp::Add
            | FieldInlineOp::Sub
            | FieldInlineOp::Mul
            | FieldInlineOp::Inv
            | FieldInlineOp::AssertEq
            | FieldInlineOp::AssertZero
            | FieldInlineOp::LoadImm => None,
        };
        Some(FieldInlineTraceData {
            op: Some(op),
            rs1,
            rs2,
            rd,
            bridge,
        })
    }
}
