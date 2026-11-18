//! ECALL (SYSTEM 0x0000_0073) — Environment call, always traps to mtvec.
//!
//! The inline sequence (`jolt_program::expand`, `expand_ecall`) takes an
//! environment-call-from-M-mode exception on Jolt's M-mode-only hart: it
//! writes mepc, mcause = 11 and mtval = 0, moves `mstatus.MIE` into
//! `mstatus.MPIE` and clears MIE (MPP stays M, its only legal value), and
//! jumps to `mtvec.BASE`. The CSRs live in virtual registers; see
//! `jolt_program::expand::control_flow::csr` for their encoding.
//!
//! `exec` below updates `cpu.csr` and is reached only by a direct
//! `RISCVInstruction::execute` call. `Instruction::execute` and tracing run
//! the inline sequence, which keeps the CSRs in virtual registers.

use crate::instruction::registers::i::RegisterStateI;

use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, PrivilegeMode, Trap, TrapType},
};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = ECALL,
    mask   = 0xffff_ffff,
    match  = 0x0000_0073,
    format = FormatI,
    registers = RegisterStateI,
    ram    = ()
);

impl ECALL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ECALL as RISCVInstruction>::RamAccess) {
        let trap_type = match cpu.privilege_mode {
            PrivilegeMode::User => TrapType::EnvironmentCallFromUMode,
            PrivilegeMode::Supervisor => TrapType::EnvironmentCallFromSMode,
            PrivilegeMode::Machine | PrivilegeMode::Reserved => TrapType::EnvironmentCallFromMMode,
        };

        cpu.raise_trap(
            Trap {
                trap_type,
                value: 0,
            },
            self.address,
        );
    }
}

impl RISCVTrace for ECALL {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}
