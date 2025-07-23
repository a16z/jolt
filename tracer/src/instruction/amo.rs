
use super::and::AND;
use super::andi::ANDI;
use super::format::format_i::FormatI;
use super::format::format_s::FormatS;
use super::format::format_virtual_halfword_alignment::HalfwordAlignFormat;
use super::ld::LD;
use super::sd::SD;
use super::sll::SLL;
use super::slli::SLLI;
use super::srl::SRL;
use super::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use super::virtual_lw::VirtualLW;
use super::virtual_sw::VirtualSW;
use super::virtual_move::VirtualMove;
use super::virtual_sign_extend::VirtualSignExtend;
use super::xor::XOR;
use super::RV32IMInstruction;
use super::VirtualInstructionSequence;

use crate::instruction::ori::ORI;
use crate::instruction::srli::SRLI;

use crate::emulator::cpu::Xlen;

use super::format::format_r::FormatR;

pub fn amo_pre64(
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
    sequence.extend(slli.virtual_sequence(Xlen::Bit64));
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
    sequence.extend(srl.virtual_sequence(Xlen::Bit64));
    remaining -= 2;

    remaining
}

pub fn amo_post64(
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
    let ori = ORI {
        address,
        operands: FormatI {
            rd: v_mask,
            rs1: 0,
            imm: -1i64 as u64,
        },
        virtual_sequence_remaining: Some(8),
    };
    sequence.push(ori.into()); // v_mask gets 0xFFFFFFFF_FFFFFFFF
    remaining -= 1;

    let srli = SRLI {
        address,
        operands: FormatI {
            rd: v_mask,
            rs1: v_mask,
            imm: 32, // Logical right shift by 32 bits
        },
        virtual_sequence_remaining: Some(7),
    };
    sequence.push(srli.into()); // v_mask gets 0x00000000_FFFFFFFF
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
    sequence.extend(sll_mask.virtual_sequence(Xlen::Bit64));
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
    sequence.extend(sll_value.virtual_sequence(Xlen::Bit64));
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

pub fn amo_pre32(
    sequence: &mut Vec<RV32IMInstruction>,
    address: u64,
    rs1: usize,
    v_rd: usize,
    mut remaining: usize,
) -> usize {
    let assert_alignment = VirtualAssertWordAlignment {
        address,
        operands: HalfwordAlignFormat { rs1, imm: 0 },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.push(assert_alignment.into());
    remaining -= 1;

    let lw = VirtualLW {
        address,
        operands: FormatI {
            rd: v_rd,
            rs1,
            imm: 0,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.push(lw.into());
    remaining -= 1;

    remaining
}

pub fn amo_post32(
    sequence: &mut Vec<RV32IMInstruction>,
    address: u64,
    rs2: usize,
    rs1: usize,
    rd: usize,
    v_rd: usize,
    mut remaining: usize,
) {
    let sw = VirtualSW {
        address,
        operands: FormatS {
            rs1,
            rs2,
            imm: 0,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.push(sw.into());
    remaining -= 1;

    let vmove = VirtualMove {
        address,
        operands: FormatI {
            rd,
            rs1: v_rd,
            imm: 0,
        },
        virtual_sequence_remaining: Some(remaining),
    };
    sequence.push(vmove.into());
    remaining -= 1;
    
    println!("{}, remaining: {}", sequence.len(), remaining);
    assert!(remaining == 0, "sequence: {:?}, remaining: {}", sequence.len(), remaining);
}
