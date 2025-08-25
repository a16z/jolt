use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::allocate_virtual_register;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::ori::ORI;
use super::slli::SLLI;
use super::virtual_sign_extend::VirtualSignExtend;
use super::{
    format::format_r::FormatR, virtual_shift_right_bitmask::VirtualShiftRightBitmask,
    virtual_srl::VirtualSRL, RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = SRLW,
    mask   = 0xfe00707f,
    match  = 0x0000003b | (0b101 << 12),
    format = FormatR,
    ram    = ()
);

impl SRLW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRLW as RISCVInstruction>::RAMAccess) {
        // SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate
        // on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is
        // given by rs2[4:0].
        let shamt = (cpu.x[self.operands.rs2 as usize] & 0x1f) as u32;
        cpu.x[self.operands.rd as usize] =
            ((cpu.x[self.operands.rs1 as usize] as u32) >> shamt) as i32 as i64;
    }
}

impl RISCVTrace for SRLW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_bitmask = allocate_virtual_register();
        let v_rs1 = allocate_virtual_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);
        asm.emit_i::<SLLI>(*v_rs1, self.operands.rs1, 32);
        asm.emit_i::<ORI>(*v_bitmask, self.operands.rs2, 32);
        asm.emit_i::<VirtualShiftRightBitmask>(*v_bitmask, *v_bitmask, 0);
        asm.emit_vshift_r::<VirtualSRL>(self.operands.rd, *v_rs1, *v_bitmask);
        asm.emit_i::<VirtualSignExtend>(self.operands.rd, self.operands.rd, 0);
        asm.finalize()
    }
}
