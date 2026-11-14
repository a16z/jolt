//! CSRRC (CSR Read-Clear) — Read CSR to rd, clear the bits set in rs1.
//!
//! Encoding: `csr[31:20] | rs1[19:15] | funct3=011[14:12] | rd[11:7] | opcode=1110011[6:0]`
//!
//! The `csrc csr, rs` pseudo-instruction is `csrrc x0, csr, rs` (clear only, discard old value).
//! With `rs1 = x0` the CSR is not written.
//!
//! Tracing runs the inline sequence from `jolt_program::expand::control_flow`,
//! which keeps the CSRs in virtual registers and legalizes writes; see
//! `jolt_program::expand::control_flow::csr`.

use crate::instruction::registers::i::RegisterStateI;

use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = CSRRC,
    mask   = 0x0000707f,
    match  = 0x00003073,
    format = FormatI,
    registers = RegisterStateI,
    ram    = ()
);

impl CSRRC {
    /// CSRRC has no single-step form. The CSRs live in virtual registers that
    /// only the inline sequence updates, so a direct update of `cpu.csr` could
    /// disagree with it. Callers must use `trace`.
    fn exec(&self, _cpu: &mut Cpu, _: &mut <CSRRC as RISCVInstruction>::RamAccess) {
        panic!("CSRRC must use trace(), not exec()");
    }
}

impl RISCVTrace for CSRRC {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Don't call self.execute() - the inline sequence handles all register writes.
        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}
