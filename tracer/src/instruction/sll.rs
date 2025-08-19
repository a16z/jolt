use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::allocate_virtual_register;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};
use serde::{Deserialize, Serialize};

use super::{
    format::format_r::FormatR, mul::MUL, virtual_pow2::VirtualPow2, RISCVInstruction, RISCVTrace,
    RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = SLL,
    mask   = 0xfe00707f,
    match  = 0x00001033,
    format = FormatR,
    ram    = ()
);

impl SLL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLL as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.x[self.operands.rs1 as usize]
                .wrapping_shl(cpu.x[self.operands.rs2 as usize] as u32 & mask),
        );
    }
}

impl RISCVTrace for SLL {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_pow2 = allocate_virtual_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);
        asm.emit_i::<VirtualPow2>(*v_pow2, self.operands.rs2, 0);
        asm.emit_r::<MUL>(self.operands.rd, self.operands.rs1, *v_pow2);
        asm.finalize()
    }
}
