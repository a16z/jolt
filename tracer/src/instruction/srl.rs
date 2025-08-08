use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{
        format_i::FormatI, format_r::FormatR,
        format_virtual_right_shift_r::FormatVirtualRightShiftR, InstructionFormat,
    },
    virtual_shift_right_bitmask::VirtualShiftRightBitmask,
    virtual_srl::VirtualSRL,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = SRL,
    mask   = 0xfe00707f,
    match  = 0x00005033,
    format = FormatR,
    ram    = ()
);

impl SRL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRL as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                .wrapping_shr(cpu.x[self.operands.rs2 as usize] as u32 & mask) as i64,
        );
    }
}

impl RISCVTrace for SRL {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn virtual_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_bitmask = virtual_register_index(6);

        let mut sequence = vec![];
        let mut virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(1);

        let bitmask = VirtualShiftRightBitmask {
            address: self.address,
            operands: FormatI {
                rd: v_bitmask,
                rs1: self.operands.rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(bitmask.into());
        virtual_sequence_remaining -= 1;

        let srl = VirtualSRL {
            address: self.address,
            operands: FormatVirtualRightShiftR {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                rs2: v_bitmask,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(srl.into());

        sequence
    }
}
