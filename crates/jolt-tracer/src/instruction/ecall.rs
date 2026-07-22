//! ECALL (SYSTEM 0x0000_0073) — Environment call, always traps to mtvec.
//!
//! The inline sequence writes mepc, mcause, mtval, and mstatus to their
//! virtual registers, then jumps unconditionally to the trap handler.
//!
//! # Privilege model
//!
//! Jolt targets M-mode-only execution (no S/U privilege levels) with no
//! interrupt hardware. mstatus is written as a constant `0x1800` (MPP=M-mode,
//! MIE=0, MPIE=0) rather than via read-modify-write. This is correct because:
//! - The privilege mode is always Machine — MPP is always 3.
//! - The MIE CSR (0x304) is not in the supported CSR whitelist and cannot be
//!   accessed by guest code. No interrupt sources exist (no timer, no CLINT,
//!   no PLIC), so MIE/MPIE bits are unused.
//! - The ZeroOS trap trampoline restores mstatus via `csrw` before `mret`,
//!   so the virtual register always holds the correct value across traps.

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
    ram    = ()
);

impl ECALL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ECALL as RISCVInstruction>::RAMAccess) {
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
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}
