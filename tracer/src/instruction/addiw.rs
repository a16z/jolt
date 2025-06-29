use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, normalize_imm, InstructionFormat},
    RISCVInstruction, RISCVTrace,
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
        cpu.x[self.operands.rd] =
            cpu.x[self.operands.rs1].wrapping_add(normalize_imm(self.operands.imm)) as i32 as i64;
    }
}

impl RISCVTrace for ADDIW {}
