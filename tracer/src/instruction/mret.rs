//! MRET — Machine Return from Trap.
//!
//! Encoding: 0x30200073 (SYSTEM opcode, funct3=000, imm=0x302)
//!
//! For ZeroOS M-mode-only operation:
//! - Reads mepc (from virtual register vr35) and sets PC to that value
//! - In full RISC-V, MRET also restores privilege mode from mstatus.MPP
//!   and sets mstatus.MIE from mstatus.MPIE, but for single-privilege
//!   M-mode-only execution, these bits don't need to change.

use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    utils::inline_helpers::InstrAssembler,
    utils::virtual_registers::VirtualRegisterAllocator,
};

use super::{
    format::format_i::FormatI, jalr::JALR, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = MRET,
    mask   = 0xffffffff,  // Exact match
    match  = 0x30200073,  // MRET encoding: priv=0x302, funct3=000, opcode=1110011
    format = FormatI,
    ram    = ()
);

/// CSR address for mepc (Machine Exception Program Counter)
const CSR_MEPC_ADDRESS: u16 = 0x341;

impl MRET {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MRET as RISCVInstruction>::RAMAccess) {
        // Read mepc from CSR state and jump to it
        let mepc = cpu.read_csr_raw(CSR_MEPC_ADDRESS);
        cpu.pc = mepc;

        // In a full implementation, we would also:
        // 1. Set privilege mode to mstatus.MPP
        // 2. Set mstatus.MIE to mstatus.MPIE
        // 3. Set mstatus.MPP to U-mode (or M-mode if only M-mode is supported)
        // 4. Set mstatus.MPIE to 1
        //
        // For ZeroOS M-mode-only, single-core, no-interrupt use case,
        // privilege mode stays M and interrupt enable bits don't matter.
    }
}

impl RISCVTrace for MRET {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Don't call self.execute() - the inline sequence's JALR handles the PC update.
        // The JALR reads mepc from virtual register vr35 and jumps to it.

        // Generate and execute inline sequence
        // The inline sequence reads mepc from virtual register (source of truth for proofs)
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generate inline sequence for MRET.
    ///
    /// MRET jumps to mepc. The mepc virtual register (vr35) must have been written
    /// by a prior CSRRW instruction (typically in the trap handler before MRET).
    ///
    /// Layout:
    ///   0: JALR(x0, vr35, 0) - Jump to mepc (read directly from virtual register)
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let mepc_vr = allocator.mepc_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Index 0: Jump to mepc (read directly from virtual register)
        asm.emit_i::<JALR>(0, mepc_vr, 0);

        asm.finalize()
    }
}

#[cfg(test)]
mod tests {
    use crate::instruction::Instruction;

    /// Test decoding of `mret`
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
