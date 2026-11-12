//! MRET — Machine Return from Trap.
//!
//! Encoding: 0x30200073 (SYSTEM opcode, funct3=000, imm=0x302)
//!
//! # Privilege model
//!
//! Jolt's hart is M-mode only. The inline sequence (`jolt_program::expand`,
//! `expand_mret`) sets `mstatus.MIE := mstatus.MPIE`, `mstatus.MPIE := 1` and
//! jumps to mepc. MPP stays M, which is both its only legal value and the
//! least-privileged supported mode the specification asks MRET to leave in
//! it.
//!
//! `exec` below only jumps to the `mepc` held in `cpu.csr` and leaves
//! `mstatus` alone. It is reached only by a direct
//! `RISCVInstruction::execute` call. `Instruction::execute` and tracing run
//! the inline sequence, which keeps the CSRs in virtual registers.

use crate::instruction::registers::i::RegisterStateI;

use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = MRET,
    mask   = 0xffffffff,  // Exact match
    match  = 0x30200073,
    format = FormatI,
    registers = RegisterStateI,
    ram    = ()
);

const CSR_MEPC_ADDRESS: u16 = 0x341;

impl MRET {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MRET as RISCVInstruction>::RamAccess) {
        let mepc = cpu.read_csr_raw(CSR_MEPC_ADDRESS);
        cpu.pc = mepc;
    }
}

impl RISCVTrace for MRET {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Don't call self.execute() - the inline sequence's JALR handles the PC update.
        // The JALR reads mepc from virtual register vr36 and jumps to it.

        // Generate and execute inline sequence
        // The inline sequence reads mepc from virtual register (source of truth for proofs)
        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}
