//! ECALL (SYSTEM 0x0000_0073) — used only for Jolt-cycle tracking.
//!
//! It retires like a normal instruction; there is **no trap** because the
//! emulator has an early-exit path in `Cpu::handle_trap` that consumes the
//! marker.  From the ISA-level point of view we therefore treat it as a
//! “no-op” that touches no architectural state.

use serde::{Deserialize, Serialize};

use crate::emulator::cpu::Cpu;

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVCycle, RISCVInstruction, RISCVTrace, RV32IMCycle,
};

// ---------------------------------------------------------------------------
//  Data type
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ECALL {
    pub address: u64,
    pub operands: FormatI,
    /// Virtual-sequence metadata (unused – always `None` here).
    pub virtual_sequence_remaining: Option<usize>,
}

// ---------------------------------------------------------------------------
//  RISCVInstruction impl
// ---------------------------------------------------------------------------
impl RISCVInstruction for ECALL {
    /// Exact bit-pattern match.
    const MASK:  u32 = 0xffff_ffff;
    const MATCH: u32 = 0x0000_0073; // ECALL opcode

    type Format    = FormatI;
    type RAMAccess = ();            // ECALL touches no memory

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

    /// **No architectural effect** – the real work is handled earlier by
    /// `Cpu::handle_trap`.  Leave all registers untouched.
    fn execute(&self, _: &mut Cpu, _: &mut Self::RAMAccess) {}
}

impl RISCVTrace for ECALL {}

