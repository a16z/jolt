//! CSRRWI (CSR Read-Write Immediate) — Write a 5-bit immediate to CSR, read old value to rd.
//!
//! Encoding: `csr[31:20] | uimm[19:15] | funct3=101[14:12] | rd[11:7] | opcode=1110011[6:0]`
//!
//! The `csrwi csr, uimm` pseudo-instruction is `csrrwi x0, csr, uimm` (rd = 0, discard old value).
//!
//! The `rs1` field holds the zero-extended immediate `uimm`, not a register;
//! [`FormatI`] keeps it in `operands.rs1`.
//!
//! Tracing runs the inline sequence from `jolt_program::expand::control_flow`,
//! which keeps the CSRs in virtual registers and legalizes writes; see
//! `jolt_program::expand::control_flow::csr`.

use crate::instruction::registers::i::RegisterStateI;

use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = CSRRWI,
    mask   = 0x0000707f,
    match  = 0x00005073,
    format = FormatI,
    registers = RegisterStateI,
    ram    = ()
);

impl CSRRWI {
    fn csr_address(&self) -> u16 {
        (self.operands.imm & 0xfff) as u16
    }

    fn exec(&self, cpu: &mut Cpu, _: &mut <CSRRWI as RISCVInstruction>::RamAccess) {
        let csr_addr = self.csr_address();
        let old_val = cpu.read_csr_raw(csr_addr);

        cpu.write_csr_raw(csr_addr, u64::from(self.operands.rs1));

        if self.operands.rd != 0 {
            cpu.write_register(self.operands.rd as usize, cpu.sign_extend(old_val as i64));
        }
    }
}

impl RISCVTrace for CSRRWI {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Don't call self.execute() - the inline sequence handles all register writes.
        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}
