//! EBREAK — Breakpoint / program termination.
//!
//! Encoding: 0x00100073 (SYSTEM opcode, funct3=000, imm=1)
//!
//! In a zkVM context without a debugger, EBREAK serves as a termination point.
//! The emulator detects termination when PC doesn't change (prev_pc == pc).

use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = EBREAK,
    mask   = 0xffffffff,  // Exact match
    match  = 0x00100073,  // EBREAK encoding
    format = FormatI,
    ram    = ()
);

impl EBREAK {
    fn exec(&self, cpu: &mut Cpu, _: &mut <EBREAK as RISCVInstruction>::RAMAccess) {
        // Don't advance PC - emulator will detect prev_pc == pc and terminate
        cpu.pc = self.address;
    }
}

impl RISCVTrace for EBREAK {}
