use serde::{Deserialize, Serialize};

use super::ld::LD;
use super::or::OR;
use super::sd::SD;
use crate::instruction::addi::ADDI;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_amo::FormatAMO, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOORD,
    mask   = 0xf800707f,
    match  = 0x4000302f,
    format = FormatAMO,
    ram    = ()
);

impl AMOORD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOORD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let or_value = cpu.x[self.operands.rs2 as usize] as u64;

        // Load the original doubleword from memory
        let load_result = cpu.mmu.load_doubleword(address);
        let original_value = match load_result {
            Ok((doubleword, _)) => doubleword as i64,
            Err(_) => panic!("MMU load error"),
        };

        // OR the values and store back to memory
        let new_value = (original_value as u64) | or_value;
        cpu.mmu
            .store_doubleword(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.write_register(self.operands.rd as usize, original_value);
    }
}

impl RISCVTrace for AMOORD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// AMOOR.D atomically performs bitwise OR on a 64-bit memory location with rs2.
    ///
    /// This atomic memory operation (AMO) instruction atomically loads a doubleword from
    /// the memory address in rs1, performs a bitwise OR with the value in rs2, stores
    /// the result back to memory, and returns the original memory value in rd.
    ///
    /// In the zkVM context:
    /// - Atomicity is guaranteed by the single-threaded execution model
    /// - No explicit memory barriers or locks needed
    /// - Direct 64-bit operations on naturally aligned addresses
    ///
    /// The bitwise OR operation is commonly used for:
    /// - Setting specific bits in shared flags or status words
    /// - Non-destructive bit accumulation in concurrent algorithms
    /// - Lock-free set operations on bitmasks
    /// - Atomic enabling of features or permissions
    ///
    /// Implementation sequence:
    /// 1. Load original value from memory[rs1]
    /// 2. Compute new_value = original | rs2
    /// 3. Store new_value back to memory[rs1]
    /// 4. Return original value in rd
    ///
    /// Memory ordering: Provides acquire-release semantics in multi-threaded contexts,
    /// ensuring all prior memory operations are visible before the OR and all subsequent
    /// operations see the OR's result, though this is implicit in zkVM's execution model.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let effective_rd = if self.operands.rd == 0 {
            *allocator.allocate()
        } else {
            self.operands.rd
        };
        let v_rs2 = allocator.allocate();
        let v_rd = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_ld::<LD>(*v_rd, self.operands.rs1, 0);
        asm.emit_r::<OR>(*v_rs2, *v_rd, self.operands.rs2);
        asm.emit_s::<SD>(self.operands.rs1, *v_rs2, 0);
        asm.emit_i::<ADDI>(effective_rd, *v_rd, 0);
        asm.finalize()
    }
}
