use serde::{Deserialize, Serialize};

use super::addi::ADDI;
use super::and::AND;
use super::andi::ANDI;
use super::format::format_i::FormatI;
use super::format::format_load::FormatLoad;
use super::format::format_s::FormatS;
use super::format::format_u::FormatU;
use super::format::format_virtual_halfword_alignment::HalfwordAlignFormat;
use super::ld::LD;
use super::lui::LUI;
use super::sd::SD;
use super::sll::SLL;
use super::slli::SLLI;
use super::srl::SRL;
use super::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use super::virtual_move::VirtualMove;
use super::virtual_sign_extend::VirtualSignExtend;
use super::virtual_sw::VirtualSW;
use super::xor::XOR;
use super::RAMWrite;
use super::RV32IMInstruction;
use super::VirtualInstructionSequence;
use common::constants::virtual_register_index;

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RAMAtomic, RISCVInstruction, RISCVTrace, RV32IMCycle,
};

declare_riscv_instr!(
    name   = AMOSWAPW,
    mask   = 0xf800707f,
    match  = 0x0800202f,
    format = FormatR,
    ram    = RAMAtomic
);

impl AMOSWAPW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <AMOSWAPW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1] as u64;
        let new_value = cpu.x[self.operands.rs2] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, memory_read)) => {
                // Store the read access
                ram_access.read = memory_read;
                word as i32 as i64
            }
            Err(_) => panic!("MMU load error"),
        };

        // Store the new value to memory
        let store_result = cpu.mmu.store_word(address, new_value);
        match store_result {
            Ok(memory_write) => {
                // Store the write access
                ram_access.write = memory_write;
            }
            Err(_) => panic!("MMU store error"),
        }

        // Return the original value
        cpu.x[self.operands.rd] = original_value;
    }
}

impl RISCVTrace for AMOSWAPW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for AMOSWAPW {
    fn virtual_sequence(&self, cpu: &Cpu) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_mask = virtual_register_index(10) as usize;
        let v_dword_address = virtual_register_index(11) as usize;
        let v_dword = virtual_register_index(12) as usize;
        let v_word = virtual_register_index(13) as usize;
        let v_shift = virtual_register_index(14) as usize;
        let v_rd = virtual_register_index(15) as usize;

        let mut sequence = vec![];
        let mut remaining = 15;
        remaining = amo_pre(
            cpu,
            &mut sequence,
            self.address,
            self.operands.rs1,
            v_rd,
            v_dword_address,
            v_dword,
            v_shift,
            remaining,
        );
        amo_post(
            cpu,
            &mut sequence,
            self.address,
            self.operands.rs2,
            v_dword_address,
            v_dword,
            v_shift,
            v_mask,
            v_word,
            self.operands.rd,
            v_rd,
            remaining,
        );

        sequence
    }
}

pub fn amo_pre(
    cpu: &Cpu,
    sequence: &mut Vec<RV32IMInstruction>,
    address: u64,
    rs1: usize,
    v_rd: usize,
    v_dword_address: usize,
    v_dword: usize,
    v_shift: usize,
    mut remaining: usize,
) -> usize {
    let assert_alignment = VirtualAssertWordAlignment {
        address,
        operands: HalfwordAlignFormat { rs1, imm: 0 },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.push(assert_alignment.into());
    remaining -= 1;

    let andi = ANDI {
        address,
        operands: FormatI {
            rd: v_dword_address,
            rs1,
            imm: -8i64 as u64,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.push(andi.into());
    remaining -= 1;

    let ld = LD {
        address,
        operands: FormatI {
            rd: v_dword,
            rs1: v_dword_address,
            imm: 0,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.push(ld.into());
    remaining -= 1;

    let slli = SLLI {
        address,
        operands: FormatI {
            rd: v_shift,
            rs1,
            imm: 3,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.extend(slli.virtual_sequence(cpu));
    remaining -= 1;

    let srl = SRL {
        address,
        operands: FormatR {
            rd: v_rd,
            rs1: v_dword,
            rs2: v_shift,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.extend(srl.virtual_sequence(cpu));
    remaining -= 2;

    remaining
}

pub fn amo_post(
    cpu: &Cpu,
    sequence: &mut Vec<RV32IMInstruction>,
    address: u64,
    rs2: usize,
    v_dword_address: usize,
    v_dword: usize,
    v_shift: usize,
    v_mask: usize,
    v_word: usize,
    rd: usize,
    v_rd: usize,
    mut remaining: usize,
) {
    let lui = LUI {
        address,
        operands: FormatU {
            rd: v_mask,
            imm: 0xffff_ffff,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.push(lui.into());
    remaining -= 1;

    let sll_mask = SLL {
        address,
        operands: FormatR {
            rd: v_mask,
            rs1: v_mask,
            rs2: v_shift,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.extend(sll_mask.virtual_sequence(cpu));
    remaining -= 2;

    let sll_value = SLL {
        address,
        operands: FormatR {
            rd: v_word,
            rs1: rs2,
            rs2: v_shift,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.extend(sll_value.virtual_sequence(cpu));
    remaining -= 2;

    let xor = XOR {
        address,
        operands: FormatR {
            rd: v_word,
            rs1: v_dword,
            rs2: v_word,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.push(xor.into());
    remaining -= 1;

    let and = AND {
        address,
        operands: FormatR {
            rd: v_word,
            rs1: v_word,
            rs2: v_mask,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.push(and.into());
    remaining -= 1;

    let xor_final = XOR {
        address,
        operands: FormatR {
            rd: v_dword,
            rs1: v_dword,
            rs2: v_word,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.push(xor_final.into());
    remaining -= 1;

    let sd = SD {
        address,
        operands: FormatS {
            rs1: v_dword_address,
            rs2: v_dword,
            imm: 0,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.push(sd.into());
    remaining -= 1;

    let signext = VirtualSignExtend {
        address,
        operands: FormatI {
            rd,
            rs1: v_rd,
            imm: 0,
        },
        virtual_sequence_remaining: Some(0),
    };
    sequence.push(signext.into());
    assert!(remaining == 0);
}
