use common::constants::RAM_START_ADDRESS;

use crate::emulator::cpu::Cpu;
use crate::instruction::addi::ADDI;
use crate::instruction::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use crate::instruction::virtual_lw::VirtualLW;
use crate::instruction::virtual_sign_extend_word::VirtualSignExtendWord;
use crate::instruction::virtual_sw::VirtualSW;
use crate::utils::inline_helpers::InstrAssembler;

pub fn amo_pre64(asm: &mut InstrAssembler, rs1: u8, v_rd: u8, v_dword: u8, v_shift: u8) {
    let _ = (v_dword, v_shift);
    asm.emit_align::<VirtualAssertWordAlignment>(rs1, 0);
    asm.emit_i::<VirtualLW>(v_rd, rs1, 0);
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
    let _ = (v_dword, v_shift, v_mask);
    asm.emit_s::<VirtualSW>(rs1, rs2, 0);
    asm.emit_i::<VirtualSignExtendWord>(rd, v_rd, 0);
}

pub fn amo_pre32(asm: &mut InstrAssembler, rs1: u8, v_rd: u8) {
    asm.emit_align::<VirtualAssertWordAlignment>(rs1, 0);
    asm.emit_i::<VirtualLW>(v_rd, rs1, 0);
}

pub fn amo_post32(asm: &mut InstrAssembler, rs2: u8, rs1: u8, rd: u8, v_rd: u8) {
    asm.emit_s::<VirtualSW>(rs1, rs2, 0);
    asm.emit_i::<ADDI>(rd, v_rd, 0);
}

pub fn reject_jolt_device_atomic(cpu: &Cpu, address: u64) {
    if cpu.mmu.jolt_device.is_some() && address < RAM_START_ADDRESS {
        panic!("Atomic operations over JoltDevice memory are unsupported");
    }
}
