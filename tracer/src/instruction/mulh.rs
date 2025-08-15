use crate::utils::virtual_registers::allocate_virtual_register;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD,
    format::{format_i::FormatI, format_r::FormatR, InstructionFormat},
    mul::MUL,
    mulhu::MULHU,
    virtual_movsign::VirtualMovsign,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = MULH,
    mask   = 0xfe00707f,
    match  = 0x02001033,
    format = FormatR,
    ram    = ()
);

impl MULH {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULH as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = match cpu.xlen {
            Xlen::Bit32 => cpu.sign_extend(
                (cpu.x[self.operands.rs1 as usize] * cpu.x[self.operands.rs2 as usize]) >> 32,
            ),
            Xlen::Bit64 => {
                (((cpu.x[self.operands.rs1 as usize] as i128)
                    * (cpu.x[self.operands.rs2 as usize] as i128))
                    >> 64) as i64
            }
        };
    }
}

impl RISCVTrace for MULH {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_sx = allocate_virtual_register();
        let v_sy = allocate_virtual_register();
        let v_0 = allocate_virtual_register();
        let v_1 = allocate_virtual_register();
        let v_2 = allocate_virtual_register();
        let v_3 = allocate_virtual_register();

        let mut sequence = vec![];

        let movsign_x = VirtualMovsign {
            address: self.address,
            operands: FormatI {
                rd: *v_sx,
                rs1: self.operands.rs1,
                imm: 0,
            },
            inline_sequence_remaining: Some(6),
            is_compressed: self.is_compressed,
        };
        sequence.push(movsign_x.into());

        let movsign_y = VirtualMovsign {
            address: self.address,
            operands: FormatI {
                rd: *v_sy,
                rs1: self.operands.rs2,
                imm: 0,
            },
            inline_sequence_remaining: Some(5),
            is_compressed: self.is_compressed,
        };
        sequence.push(movsign_y.into());

        let mulhu = MULHU {
            address: self.address,
            operands: FormatR {
                rd: *v_0,
                rs1: self.operands.rs1,
                rs2: self.operands.rs2,
            },
            inline_sequence_remaining: Some(4),
            is_compressed: self.is_compressed,
        };
        sequence.push(mulhu.into());

        let mulu_sx_y = MUL {
            address: self.address,
            operands: FormatR {
                rd: *v_1,
                rs1: *v_sx,
                rs2: self.operands.rs2,
            },
            inline_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
        };
        sequence.push(mulu_sx_y.into());

        let mulu_sy_x = MUL {
            address: self.address,
            operands: FormatR {
                rd: *v_2,
                rs1: *v_sy,
                rs2: self.operands.rs1,
            },
            inline_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.push(mulu_sy_x.into());

        let add_1 = ADD {
            address: self.address,
            operands: FormatR {
                rd: *v_3,
                rs1: *v_0,
                rs2: *v_1,
            },
            inline_sequence_remaining: Some(1),
            is_compressed: self.is_compressed,
        };
        sequence.push(add_1.into());

        let add_2 = ADD {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: *v_3,
                rs2: *v_2,
            },
            inline_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(add_2.into());

        sequence
    }
}
