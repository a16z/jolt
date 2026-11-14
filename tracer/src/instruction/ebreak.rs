//! EBREAK — Jolt's failure halt.
//!
//! Encoding: 0x00100073 (SYSTEM opcode, funct3=000, imm=1)
//!
//! EBREAK halts the hart, as it does under an external debugger with
//! `dcsr.ebreakm = 1` (RISC-V Debug Specification): M-mode EBREAK enters
//! Debug Mode instead of trapping, so `mepc`, `mcause`, `mtval` and `mstatus`
//! are unchanged. Tracing runs its one-row `Panic` lowering. `exec` below is
//! reached only by a direct `RISCVInstruction::execute` call and records the
//! same halt without the temporary link.

use crate::instruction::registers::i::RegisterStateI;

use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, HaltKind},
};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = EBREAK,
    mask   = 0xffffffff,
    match  = 0x00100073,
    format = FormatI,
    registers = RegisterStateI,
    ram    = ()
);

impl EBREAK {
    fn exec(&self, cpu: &mut Cpu, _: &mut <EBREAK as RISCVInstruction>::RamAccess) {
        cpu.halt(HaltKind::Panic, self.address, 0);
    }
}

impl RISCVTrace for EBREAK {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}
