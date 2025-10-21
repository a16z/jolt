use crate::instruction::ld::LD;
use crate::instruction::sd::SD;
use crate::instruction::Instruction;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{instruction::addi::ADDI, utils::inline_helpers::InstrAssembler};
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_amo::FormatAMO, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOSWAPD,
    mask   = 0xf800707f,
    match  = 0x0800302f,
    format = FormatAMO,
    ram    = ()
);

impl AMOSWAPD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOSWAPD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let new_value = cpu.x[self.operands.rs2 as usize] as u64;

        // Load the original doubleword from memory
        let load_result = cpu.mmu.load_doubleword(address);
        let original_value = match load_result {
            Ok((doubleword, _)) => doubleword as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Store the new value to memory
        cpu.mmu
            .store_doubleword(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOSWAPD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// AMOSWAP.D atomically swaps a 64-bit memory location with rs2.
    ///
    /// This atomic memory operation (AMO) instruction atomically loads a doubleword from
    /// the memory address in rs1, stores the value from rs2 to that location, and returns
    /// the original memory value in rd. This is an unconditional atomic exchange.
    ///
    /// In the zkVM context:
    /// - Atomicity is guaranteed by the single-threaded execution model
    /// - No compare-and-swap semantics needed (unlike LR/SC sequences)
    /// - Direct 64-bit operations on naturally aligned addresses
    ///
    /// AMOSWAP is commonly used for:
    /// - Implementing mutex acquisition (swap in lock value, check old)
    /// - Atomic pointer swapping in lock-free data structures
    /// - Message passing between threads (though zkVM is single-threaded)
    /// - Implementing semaphores and other synchronization primitives
    /// - Atomic queue operations (head/tail pointer updates)
    ///
    /// Implementation sequence:
    /// 1. Load original value from memory[rs1]
    /// 2. Store rs2 value to memory[rs1]
    /// 3. Return original value in rd
    ///
    /// This is the simplest AMO operation as it requires no computation,
    /// just an atomic exchange. The returned value can be examined to
    /// determine the previous state (e.g., was a lock already held).
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_rd = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_ld::<LD>(*v_rd, self.operands.rs1, 0);
        asm.emit_s::<SD>(self.operands.rs1, self.operands.rs2, 0);
        asm.emit_i::<ADDI>(self.operands.rd, *v_rd, 0);
        asm.finalize()
    }
}
