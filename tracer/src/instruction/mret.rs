//! MRET — Machine Return from Trap.
//!
//! Encoding: 0x30200073 (SYSTEM opcode, funct3=000, imm=0x302)
//!
//! # Privilege model
//!
//! Jolt targets M-mode-only execution with no interrupt hardware. MRET is
//! implemented as a single JALR to mepc — it does not modify mstatus.
//!
//! The full RISC-V spec requires MRET to restore MIE from MPIE, set MPIE=1,
//! and reset MPP to the least-privileged mode. These operations are omitted
//! because:
//! - There is only one privilege level (Machine) — MPP is always 3.
//! - No interrupt sources exist and the MIE CSR (0x304) is inaccessible,
//!   so MIE/MPIE bits are unused.
//! - The ZeroOS trap trampoline restores mstatus via `csrw mstatus, <saved>`
//!   before executing `mret`, so the virtual register holds the correct value
//!   without MRET needing to manipulate it.

use crate::instruction::registers::i::RegisterStateI;

use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = MRET,
    mask   = 0xffffffff,
    match  = 0x30200073,
    format = FormatI,
    registers = RegisterStateI,
    ram    = ()
);

const CSR_MEPC_ADDRESS: u16 = 0x341;

impl MRET {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MRET as RISCVInstruction>::RAMAccess) {
        let mepc = cpu.read_csr_raw(CSR_MEPC_ADDRESS);
        cpu.pc = mepc;
    }
}

impl RISCVTrace for MRET {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}

#[cfg(test)]
mod tests {
    use crate::instruction::Instruction;

    #[test]
    fn test_mret_decode() {
        let instr: u32 = 0x30200073;
        let address: u64 = 0x1000;

        let decoded = Instruction::decode(instr, address, false).expect("Failed to decode MRET");

        match decoded {
            Instruction::MRET(_mret) => {
                // MRET has no operands to check - it's a fixed encoding
            }
            _ => panic!("Expected MRET instruction, got {decoded:?}"),
        }
    }
}
