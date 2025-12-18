use crate::{
    emulator::{cpu::GeneralizedCpu, memory::MemoryData},
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterAllocator},
};
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Xlen};

use super::addi::ADDI;
use super::virtual_sign_extend_word::VirtualSignExtendWord;
use super::Instruction;

use super::{
    format::{format_i::FormatI, normalize_imm},
    Cycle, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = ADDIW,
    mask   = 0x0000707f,
    match  = 0x0000001b,
    format = FormatI,
    ram    = ()
);

impl ADDIW {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <ADDIW as RISCVInstruction>::RAMAccess,
    ) {
        cpu.x[self.operands.rd as usize] = cpu.x[self.operands.rs1 as usize]
            .wrapping_add(normalize_imm(self.operands.imm, &cpu.xlen))
            as i32 as i64;
    }
}

impl RISCVTrace for ADDIW {
    fn trace<D: MemoryData>(&self, cpu: &mut GeneralizedCpu<D>, trace: Option<&mut Vec<Cycle>>) {
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
        asm.emit_i::<ADDI>(self.operands.rd, self.operands.rs1, self.operands.imm);
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, self.operands.rd, 0);
        asm.finalize()
    }
}
