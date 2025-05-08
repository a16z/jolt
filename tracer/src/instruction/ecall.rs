//! ECALL (SYSTEM 0x0000_0073) — currently only serves for Jolt-cycle tracking.
//! Although, this will be used for "pseudo-precompiles"
//!
//! It retires like a normal instruction; there is **no trap** because the
//! emulator has an early-exit path in `Cpu::handle_trap` that consumes the
//! marker.  From the ISA-level point of view we therefore treat it as a
//! “no-op”.

use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, PrivilegeMode, Trap, TrapType};

use super::{
    format::{format_i::FormatI, InstructionFormat}, RISCVInstruction, RISCVTrace,
};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ECALL {
    pub address: u64,
    pub operands: FormatI,
    pub virtual_sequence_remaining: Option<usize>,
}

impl RISCVInstruction for ECALL {
    const MASK: u32 = 0xffff_ffff;
    const MATCH: u32 = 0x0000_0073; // ECALL opcode

    type Format = FormatI;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64, validate: bool) -> Self {
        if validate {
            debug_assert_eq!(word, Self::MATCH);
        }
        Self {
            address,
            operands: FormatI::parse(word),
            virtual_sequence_remaining: None,
        }
    }

    /// **No architectural effects**
    /// Signals to emulator to record cycles through early exit of trap
    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {

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
