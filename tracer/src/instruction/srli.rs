use crate::utils::inline_helpers::InstrAssembler;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::virtual_srli::VirtualSRLI,
};

use super::{
    format::format_i::FormatI, RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = SRLI,
    mask   = 0xfc00707f,
    match  = 0x00005013,
    format = FormatI,
    ram    = ()
);

impl SRLI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRLI as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                .wrapping_shr(self.operands.imm as u32 & mask) as i64,
        );
    }
}

impl RISCVTrace for SRLI {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let (shift, len) = match xlen {
            Xlen::Bit32 => (self.operands.imm & 0x1f, 32),
            Xlen::Bit64 => (self.operands.imm & 0x3f, 64),
        };
        let ones = (1u128 << (len - shift)) - 1;
        let bitmask = (ones << shift) as u64;

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);
        asm.emit_vshift_i::<VirtualSRLI>(self.operands.rd, self.operands.rs1, bitmask);
        asm.finalize()
    }
}
