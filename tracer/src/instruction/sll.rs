use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};
use serde::{Deserialize, Serialize};

use super::{
    format::format_r::FormatR, mul::MUL, virtual_pow2::VirtualPow2, Cycle, Instruction,
    RISCVInstruction, RISCVTrace,
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
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// SLL shifts left by multiplying by 2^shift_amount.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_pow2 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_i::<VirtualPow2>(*v_pow2, self.operands.rs2, 0);
        asm.emit_r::<MUL>(self.operands.rd, self.operands.rs1, *v_pow2);
        asm.finalize()
    }
}
