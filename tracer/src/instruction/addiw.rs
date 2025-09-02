use crate::utils::inline_helpers::InstrAssembler;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::addi::ADDI;
use super::virtual_sign_extend::VirtualSignExtend;
use super::RV32IMInstruction;

use super::{
    format::{format_i::FormatI, normalize_imm},
    RISCVInstruction, RISCVTrace, RV32IMCycle,
};

declare_riscv_instr!(
    name   = ADDIW,
    mask   = 0x0000707f,
    match  = 0x0000001b,
    format = FormatI,
    ram    = ()
);

impl ADDIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ADDIW as RISCVInstruction>::RAMAccess) {
        // ADDIW is an RV64I instruction that adds the sign-extended 12-bit immediate to register
        // rs1 and produces the proper sign extension of a 32-bit result in rd. Overflows are
        // ignored and the result is the low 32 bits of the result sign-extended to 64 bits. Note,
        // ADDIW rd, rs1, 0 writes the sign extension of the lower 32 bits of register rs1 into
        // register rd (assembler pseudoinstruction SEXT.W).
        cpu.x[self.operands.rd as usize] = cpu.x[self.operands.rs1 as usize]
            .wrapping_add(normalize_imm(self.operands.imm, &cpu.xlen))
            as i32 as i64;
    }
}

impl RISCVTrace for ADDIW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, false);
        asm.emit_i::<ADDI>(self.operands.rd, self.operands.rs1, self.operands.imm);
        asm.emit_i::<VirtualSignExtend>(self.operands.rd, self.operands.rd, 0);
        asm.finalize()
    }
}
