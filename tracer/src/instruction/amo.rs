use super::and::AND;
use super::andi::ANDI;
use super::ld::LD;
use super::ori::ORI;
use super::sd::SD;
use super::sll::SLL;
use super::slli::SLLI;
use super::srl::SRL;
use super::srli::SRLI;
use super::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use super::virtual_lw::VirtualLW;
use super::virtual_move::VirtualMove;
use super::virtual_sign_extend_word::VirtualSignExtendWord;
use super::virtual_sw::VirtualSW;
use super::xor::XOR;

use crate::utils::inline_helpers::InstrAssembler;

pub fn amo_pre64(asm: &mut InstrAssembler, rs1: u8, v_rd: u8, v_dword: u8, v_shift: u8) {
    asm.emit_halign::<VirtualAssertWordAlignment>(rs1, 0);
    // Use v_shift temporarily to hold aligned address
    asm.emit_i::<ANDI>(v_shift, rs1, -8i64 as u64);
    asm.emit_ld::<LD>(v_dword, v_shift, 0);
    // Now compute the actual shift value
    asm.emit_i::<SLLI>(v_shift, rs1, 3);
    asm.emit_r::<SRL>(v_rd, v_dword, v_shift);
}

#[allow(clippy::too_many_arguments)]
pub fn amo_post64(
    asm: &mut InstrAssembler,
    rs1: u8,
    rs2: u8,
    v_dword: u8,
    v_shift: u8,
    v_mask: u8,
    rd: u8,
    v_rd: u8,
) {
    asm.emit_i::<ORI>(v_mask, 0, -1i64 as u64);
    asm.emit_i::<SRLI>(v_mask, v_mask, 32);
    asm.emit_r::<SLL>(v_mask, v_mask, v_shift);
    // Use v_shift as temporary after it's been used for shifting
    asm.emit_r::<SLL>(v_shift, rs2, v_shift);
    asm.emit_r::<XOR>(v_shift, v_dword, v_shift);
    asm.emit_r::<AND>(v_shift, v_shift, v_mask);
    asm.emit_r::<XOR>(v_dword, v_dword, v_shift);
    // Recompute aligned address for store
    asm.emit_i::<ANDI>(v_mask, rs1, -8i64 as u64);
    asm.emit_s::<SD>(v_mask, v_dword, 0);
    asm.emit_i::<VirtualSignExtendWord>(rd, v_rd, 0);
}

pub fn amo_pre32(asm: &mut InstrAssembler, rs1: u8, v_rd: u8) {
    asm.emit_halign::<VirtualAssertWordAlignment>(rs1, 0);
    asm.emit_i::<VirtualLW>(v_rd, rs1, 0);
}

pub fn amo_post32(asm: &mut InstrAssembler, rs2: u8, rs1: u8, rd: u8, v_rd: u8) {
    asm.emit_s::<VirtualSW>(rs1, rs2, 0);
    asm.emit_i::<VirtualMove>(rd, v_rd, 0);
}
