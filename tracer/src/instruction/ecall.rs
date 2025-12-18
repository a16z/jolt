//! ECALL (SYSTEM 0x0000_0073) — currently only serves for Jolt cycle-tracking.
//! Although, this will be used for "pseudo-precompiles"

use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::{
        cpu::{GeneralizedCpu, PrivilegeMode, Trap, TrapType},
        memory::MemoryData,
    },
};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = ECALL,
    mask   = 0xffff_ffff,
    match  = 0x0000_0073,
    format = FormatI,
    ram    = ()
);

impl ECALL {
    /// **No architectural effects**
    /// Signals to emulator to record cycles through early exit of trap
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <ECALL as RISCVInstruction>::RAMAccess,
    ) {
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

impl RISCVTrace for ECALL {}
