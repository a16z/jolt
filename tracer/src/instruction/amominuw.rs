use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use super::mul::MUL;
use super::sltu::SLTU;
use super::virtual_extend::VirtualExtend;
use super::virtual_move::VirtualMove;
use super::xori::XORI;
use super::RV32IMInstruction;
use crate::instruction::format::format_i::FormatI;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};
use common::constants::virtual_register_index;

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle,
};

declare_riscv_instr!(
    name   = AMOMINUW,
    mask   = 0xf800707f,
    match  = 0xc000202f,
    format = FormatR,
    ram    = ()
);

impl AMOMINUW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOMINUW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let compare_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Find the minimum (unsigned comparison) and store back to memory
        let new_value = if (original_value as u32) <= compare_value {
            original_value as u32
        } else {
            compare_value
        };
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value (sign extended)
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOMINUW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        match xlen {
            Xlen::Bit32 => self.inline_sequence_32(xlen),
            Xlen::Bit64 => self.inline_sequence_64(xlen),
        }
    }
}

impl AMOMINUW {
    fn inline_sequence_32(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_rd = virtual_register_index(7);
        let v_rs2 = virtual_register_index(8);
        let v_sel_rs2 = virtual_register_index(9);
        let v_sel_rd = virtual_register_index(10);
        let v_tmp = virtual_register_index(11);

        let mut sequence = vec![];
        let mut remaining = 10;
        remaining = amo_pre32(
            &mut sequence,
            self.address,
            self.is_compressed,
            self.operands.rs1,
            v_rd,
            remaining,
        );

        let mov = VirtualMove {
            address: self.address,
            operands: FormatI {
                rd: v_rs2,
                rs1: self.operands.rs2,
                imm: 0,
            },
            inline_sequence_remaining: Some(remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(mov.into());
        remaining -= 1;

        let mov = VirtualMove {
            address: self.address,
            operands: FormatI {
                rd: v_tmp,
                rs1: v_rd,
                imm: 0,
            },
            inline_sequence_remaining: Some(remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(mov.into());
        remaining -= 1;

        let sltu = SLTU {
            address: self.address,
            operands: FormatR {
                rd: v_sel_rs2,
                rs1: v_rs2,
                rs2: v_tmp,
            },
            inline_sequence_remaining: Some(remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(sltu.into());
        remaining -= 1;

        let xori = XORI {
            address: self.address,
            operands: FormatI {
                rd: v_sel_rd,
                rs1: v_sel_rs2,
                imm: 1,
            },
            inline_sequence_remaining: Some(remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(xori.into());
        remaining -= 1;

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_sel_rs2,
                rs2: self.operands.rs2,
            },
            inline_sequence_remaining: Some(remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(mul.into());
        remaining -= 1;

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_tmp,
                rs1: v_sel_rd,
                rs2: v_rd,
            },
            inline_sequence_remaining: Some(remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(mul.into());
        remaining -= 1;

        let add = ADD {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_tmp,
                rs2: v_rs2,
            },
            inline_sequence_remaining: Some(remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(add.into());
        remaining -= 1;

        amo_post32(
            &mut sequence,
            self.address,
            self.is_compressed,
            v_rs2,
            self.operands.rs1,
            self.operands.rd,
            v_rd,
            remaining,
        );

        sequence
    }

    fn inline_sequence_64(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_mask = virtual_register_index(10);
        let v_dword_address = virtual_register_index(11);
        let v_dword = virtual_register_index(12);
        let v_word = virtual_register_index(13);
        let v_shift = virtual_register_index(14);
        let v_rd = virtual_register_index(15);
        let v_rs2 = virtual_register_index(16);
        let v_sel_rs2 = virtual_register_index(17);
        let v_sel_rd = virtual_register_index(18);
        let v_tmp = virtual_register_index(19);

        let mut sequence = vec![];
        let mut inline_sequence_remaining = self.inline_sequence_remaining.unwrap_or(23);

        inline_sequence_remaining = amo_pre64(
            &mut sequence,
            self.address,
            self.is_compressed,
            self.operands.rs1,
            v_rd,
            v_dword_address,
            v_dword,
            v_shift,
            inline_sequence_remaining,
        );

        let ext = VirtualExtend {
            address: self.address,
            operands: FormatI {
                rd: v_rs2,
                rs1: self.operands.rs2,
                imm: 0,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(ext.into());
        inline_sequence_remaining -= 1;

        let ext = VirtualExtend {
            address: self.address,
            operands: FormatI {
                rd: v_tmp,
                rs1: v_rd,
                imm: 0,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(ext.into());
        inline_sequence_remaining -= 1;

        let sltu = SLTU {
            address: self.address,
            operands: FormatR {
                rd: v_sel_rs2,
                rs1: v_rs2,
                rs2: v_tmp,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(sltu.into());
        inline_sequence_remaining -= 1;

        let xori = XORI {
            address: self.address,
            operands: FormatI {
                rd: v_sel_rd,
                rs1: v_sel_rs2,
                imm: 1,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(xori.into());
        inline_sequence_remaining -= 1;

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_sel_rs2,
                rs2: self.operands.rs2,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(mul.into());
        inline_sequence_remaining -= 1;

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_tmp,
                rs1: v_sel_rd,
                rs2: v_rd,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(mul.into());
        inline_sequence_remaining -= 1;

        let add = ADD {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_tmp,
                rs2: v_rs2,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(add.into());
        inline_sequence_remaining -= 1;

        amo_post64(
            &mut sequence,
            self.address,
            self.is_compressed,
            v_rs2,
            v_dword_address,
            v_dword,
            v_shift,
            v_mask,
            v_word,
            self.operands.rd,
            v_rd,
            inline_sequence_remaining,
        );

        sequence
    }
}
