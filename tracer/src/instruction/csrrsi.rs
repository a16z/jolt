//! CSRRSI (CSR Read-Set Immediate) — Read CSR to rd, set the bits of a 5-bit immediate.
//!
//! Encoding: `csr[31:20] | uimm[19:15] | funct3=110[14:12] | rd[11:7] | opcode=1110011[6:0]`
//!
//! The `csrsi csr, uimm` pseudo-instruction is `csrrsi x0, csr, uimm`.
//! With `uimm = 0` the CSR is not written.
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
    name   = CSRRSI,
    mask   = 0x0000707f,
    match  = 0x00006073,
    format = FormatI,
    registers = RegisterStateI,
    ram    = ()
);

impl CSRRSI {
    fn csr_address(&self) -> u16 {
        (self.operands.imm & 0xfff) as u16
    }

    fn exec(&self, cpu: &mut Cpu, _: &mut <CSRRSI as RISCVInstruction>::RamAccess) {
        let csr_addr = self.csr_address();
        let old_val = cpu.read_csr_raw(csr_addr);

        if self.operands.rs1 != 0 {
            cpu.write_csr_raw(csr_addr, old_val | u64::from(self.operands.rs1));
        }

        if self.operands.rd != 0 {
            cpu.write_register(self.operands.rd as usize, cpu.sign_extend(old_val as i64));
        }
    }
}

impl RISCVTrace for CSRRSI {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Don't call self.execute() - the inline sequence handles all register writes.
        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}
