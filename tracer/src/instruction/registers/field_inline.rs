use jolt_program::field_inline::{
    FieldEncodedValue, FieldInlineTraceData, FieldRegisterRead, FieldRegisterWrite,
};
use jolt_riscv::{NormalizedOperands, RegisterRead, RegisterWrite};
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::{
    emulator::cpu::Cpu,
    instruction::{format::format_field_inline::FormatFieldInline, RISCVInstruction},
};

use super::{normalize_register_value, InstructionRegisterState, RegisterSnapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum FieldAccess {
    Read(FieldRegisterRead),
    ReadPair {
        rs1: FieldRegisterRead,
        rs2: FieldRegisterRead,
    },
    Write(FieldRegisterWrite),
    ReadWrite {
        rs1: FieldRegisterRead,
        rd: FieldRegisterWrite,
    },
    BinaryWrite {
        rs1: FieldRegisterRead,
        rs2: FieldRegisterRead,
        rd: FieldRegisterWrite,
    },
}

impl FieldAccess {
    fn reads(self) -> (Option<FieldRegisterRead>, Option<FieldRegisterRead>) {
        match self {
            Self::Read(rs1) | Self::ReadWrite { rs1, .. } => (Some(rs1), None),
            Self::ReadPair { rs1, rs2 } | Self::BinaryWrite { rs1, rs2, .. } => {
                (Some(rs1), Some(rs2))
            }
            Self::Write(_) => (None, None),
        }
    }

    fn write(self) -> Option<FieldRegisterWrite> {
        match self {
            Self::Write(rd) | Self::ReadWrite { rd, .. } | Self::BinaryWrite { rd, .. } => Some(rd),
            Self::Read(_) | Self::ReadPair { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum IntegerAccess {
    None,
    Read(RegisterRead),
    Write(RegisterWrite),
    ReadWrite {
        rs1: RegisterRead,
        rd: RegisterWrite,
    },
}

impl IntegerAccess {
    fn read(self) -> Option<RegisterRead> {
        match self {
            Self::Read(rs1) | Self::ReadWrite { rs1, .. } => Some(rs1),
            Self::None | Self::Write(_) => None,
        }
    }

    fn write(self) -> Option<RegisterWrite> {
        match self {
            Self::Write(rd) | Self::ReadWrite { rd, .. } => Some(rd),
            Self::None | Self::Read(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterStateFieldInline {
    field: FieldAccess,
    integer: IntegerAccess,
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
    #[expect(
        clippy::expect_used,
        reason = "Only field instructions select the field register snapshot"
    )]
    fn capture_pre(instruction: &I, cpu: &Cpu) -> Self {
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
        let write_field = |register| {
            let before = read_field(register);
            FieldRegisterWrite {
                register: before.register,
                pre_value: before.value,
                post_value: before.value,
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
            (false, false, true) => FieldAccess::Write(write_field(field_rd)),
            (true, false, true) => FieldAccess::ReadWrite {
                rs1: read_field(field_rs1),
                rd: write_field(field_rd),
            },
            (true, true, true) => FieldAccess::BinaryWrite {
                rs1: read_field(field_rs1),
                rs2: read_field(operands.rs2),
                rd: write_field(field_rd),
            },
            _ => panic!("unsupported field register operand shape"),
        };
        let x_operands = shape.x_operands(NormalizedOperands::from(operands));
        let read_x = |register: u8| RegisterRead {
            register,
            value: normalize_register_value(cpu, usize::from(register)),
        };
        let write_x = |register| {
            let before = read_x(register);
            RegisterWrite {
                register: before.register,
                pre_value: before.value,
                post_value: before.value,
            }
        };
        let integer = match (x_operands.rs1, x_operands.rd) {
            (None, None) => IntegerAccess::None,
            (Some(rs1), None) => IntegerAccess::Read(read_x(rs1)),
            (None, Some(rd)) => IntegerAccess::Write(write_x(rd)),
            (Some(rs1), Some(rd)) => IntegerAccess::ReadWrite {
                rs1: read_x(rs1),
                rd: write_x(rd),
            },
        };
        Self { field, integer }
    }

    fn capture_post(&mut self, _instruction: &I, cpu: &Cpu) {
        match &mut self.field {
            FieldAccess::Write(rd)
            | FieldAccess::ReadWrite { rd, .. }
            | FieldAccess::BinaryWrite { rd, .. } => {
                rd.post_value = cpu.field_registers.read(rd.register);
            }
            FieldAccess::Read(_) | FieldAccess::ReadPair { .. } => {}
        }
        match &mut self.integer {
            IntegerAccess::Write(rd) | IntegerAccess::ReadWrite { rd, .. } => {
                rd.post_value = normalize_register_value(cpu, usize::from(rd.register));
            }
            IntegerAccess::None | IntegerAccess::Read(_) => {}
        }
    }
}

impl RegisterStateFieldInline {
    pub(crate) fn to_field_inline_trace(self) -> FieldInlineTraceData {
        let (rs1, rs2) = self.field.reads();
        FieldInlineTraceData {
            rs1,
            rs2,
            rd: self.field.write(),
        }
    }
}
