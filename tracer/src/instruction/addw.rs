use serde::{Deserialize, Serialize};

use super::{
    add::ADD,
    format::format_r::FormatR,
    virtual_sign_extend_word::VirtualSignExtendWord,
    Cycle,
    Instruction,
    RISCVInstruction,
    RISCVTrace,
};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterAllocator},
};

declare_riscv_instr!(
    name   = ADDW,
    mask   = 0xfe00707f,
    match  = 0x0000003b,
    format = FormatR,
    ram    = ()
);

impl ADDW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ADDW as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = cpu.x[self.operands.rs1 as usize]
            .wrapping_add(cpu.x[self.operands.rs2 as usize])
            as i32 as i64;
    }
}

impl RISCVTrace for ADDW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_r::<ADD>(self.operands.rd, self.operands.rs1, self.operands.rs2);
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, self.operands.rd, 0);
        asm.finalize()
    }
}
