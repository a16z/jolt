use serde::{Deserialize, Serialize};

use super::lw::LW;
use super::sw::SW;
use super::virtual_move::VirtualMove;
use super::xor::XOR;
use super::Instruction;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_amo::FormatAMO, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOXORW,
    mask   = 0xf800707f,
    match  = 0x2000202f,
    format = FormatAMO,
    ram    = ()
);

impl AMOXORW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOXORW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let xor_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // XOR the values and store back to memory
        let new_value = (original_value as u32) ^ xor_value;
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOXORW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// AMOXOR.W atomically performs bitwise XOR on a 32-bit word in memory with rs2.
    ///
    /// This atomic memory operation (AMO) instruction atomically loads a 32-bit word from
    /// the memory address in rs1, performs a bitwise XOR with the lower 32 bits of rs2,
    /// stores the result back to memory, and returns the original value sign-extended in rd.
    ///
    /// Implementation note:
    /// Unlike other AMO.W instructions, this uses direct LW/SW instructions rather than
    /// amo_pre/post helpers. This simplified approach works because:
    /// - The zkVM handles word-alignment internally
    /// - Single-threaded execution guarantees atomicity
    /// - No need for complex bit manipulation within doublewords
    ///
    /// The bitwise XOR operation is commonly used for:
    /// - Toggling flags or status bits atomically
    /// - Implementing spinlocks with toggle semantics
    /// - Checksum and parity calculations
    /// - Lock-free state machines with reversible transitions
    ///
    /// Return value handling:
    /// - The original 32-bit value is sign-extended to XLEN bits
    /// - Consistent behavior across different XLEN implementations
    ///
    /// XOR properties in concurrent contexts:
    /// - Commutative: order of XOR operations doesn't matter
    /// - Self-canceling: applying same XOR twice restores original
    /// - Useful for temporary locks that auto-release on second application
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_rs2 = allocator.allocate();
        let v_rd = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_ld::<LW>(*v_rd, self.operands.rs1, 0);
        asm.emit_r::<XOR>(*v_rs2, *v_rd, self.operands.rs2);
        asm.emit_s::<SW>(self.operands.rs1, *v_rs2, 0);
        asm.emit_i::<VirtualMove>(self.operands.rd, *v_rd, 0);
        asm.finalize()
    }
}
