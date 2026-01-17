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
    addi::ADDI,
    format::format_i::FormatI,
    jalr::JALR,
    virtual_advice::VirtualAdvice,
    Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

/// CSR address for mepc (Machine Exception Program Counter)
const CSR_MEPC_ADDRESS: u16 = 0x341;

declare_riscv_instr!(
    name   = MRET,
    mask   = 0xffffffff,  // Exact match
    match  = 0x30200073,  // MRET encoding: priv=0x302, funct3=000, opcode=1110011
    format = FormatI,
    ram    = ()
);

impl MRET {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MRET as RISCVInstruction>::RAMAccess) {
        // Read mepc and jump to it
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
        // Get mepc value before executing
        let mepc_val = cpu.read_csr_raw(CSR_MEPC_ADDRESS);

        // Execute the MRET (updates CPU pc)
        let mut ram_access = ();
        self.execute(cpu, &mut ram_access);

        // Generate inline sequence for proof verification
        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        // Fill in the advice value (mepc)
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = mepc_val;
        } else {
            panic!("MRET: Expected VirtualAdvice at index 0, got {:?}", inline_sequence[0]);
        }

        // Execute inline sequence to record in trace
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generate inline sequence for MRET.
    ///
    /// MRET reads mepc and jumps to it:
    ///   0: VirtualAdvice(temp)     - Get mepc value as advice
    ///   1: ADDI(vr_mepc, temp, 0)  - Write advice to vr35 (makes mepc "defined")
    ///   2: ADDI(temp2, vr_mepc, 0) - Read mepc from vr35 to temp2
    ///   3: JALR(x0, temp2, 0)      - Jump to mepc (rd=0, no return address saved)
    ///
    /// The write-then-read pattern ensures the virtual register is "defined"
    /// in the inline sequence before being used for the jump target.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let vr_mepc = allocator.mepc_register();

        let temp = allocator.allocate();
        let jump_target = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Index 0: Get mepc value as advice
        asm.emit_j::<VirtualAdvice>(*temp, 0);

        // Index 1: Write advice to vr_mepc (makes it "defined")
        asm.emit_i::<ADDI>(vr_mepc, *temp, 0);

        // Index 2: Read mepc from vr to temp for jump
        asm.emit_i::<ADDI>(*jump_target, vr_mepc, 0);

        // Index 3: Jump to mepc (JALR rd=0 means no return address saved)
        asm.emit_i::<JALR>(0, *jump_target, 0);

        asm.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::MRET;
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
            _ => panic!("Expected MRET instruction, got {:?}", decoded),
        }
    }
}
