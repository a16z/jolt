use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::ld::LD;
use super::mul::MUL;
use super::sd::SD;
use super::slt::SLT;
use super::virtual_move::VirtualMove;
use super::xori::XORI;
use super::Instruction;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_r::FormatR, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOMAXD,
    mask   = 0xf800707f,
    match  = 0xa000302f,
    format = FormatR,
    ram    = ()
);

impl AMOMAXD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOMAXD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let compare_value = cpu.x[self.operands.rs2 as usize];

        // Load the original doubleword from memory
        let load_result = cpu.mmu.load_doubleword(address);
        let original_value = match load_result {
            Ok((doubleword, _)) => doubleword as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Find the maximum and store back to memory
        let new_value = if original_value >= compare_value {
            original_value
        } else {
            compare_value
        };
        cpu.mmu
            .store_doubleword(address, new_value as u64)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOMAXD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates inline sequence for atomic maximum operation (signed 64-bit).
    ///
    /// AMOMAX.D atomically loads a 64-bit value from memory, computes the maximum
    /// of that value and rs2 (treating both as signed), stores the maximum back
    /// to memory, and returns the original value in rd.
    ///
    /// The implementation uses a branchless maximum computation:
    /// 1. Load the current value from memory
    /// 2. Compare current value < rs2 to get selector bit (1 if rs2 is larger)
    /// 3. Invert selector to get bit for current value (1 if current is larger/equal)
    /// 4. Multiply rs2 by its selector bit (rs2 if larger, 0 otherwise)
    /// 5. Multiply current by its selector bit (current if larger/equal, 0 otherwise)
    /// 6. Add the two products to get the maximum
    /// 7. Store the maximum back to memory
    /// 8. Return the original value in rd
    ///
    /// This branchless approach avoids conditional execution, making it compatible
    /// with zkVM's constraint system while correctly handling signed comparison.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_rs2 = allocator.allocate();
        let v_rd = allocator.allocate();
        let v_sel_rs2 = allocator.allocate();
        let v_sel_rd = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_ld::<LD>(*v_rd, self.operands.rs1, 0);
        asm.emit_r::<SLT>(*v_sel_rs2, *v_rd, self.operands.rs2);
        asm.emit_i::<XORI>(*v_sel_rd, *v_sel_rs2, 1);
        asm.emit_r::<MUL>(*v_rs2, *v_sel_rs2, self.operands.rs2);
        asm.emit_r::<MUL>(*v_sel_rd, *v_sel_rd, *v_rd);
        asm.emit_r::<ADD>(*v_rs2, *v_sel_rd, *v_rs2);
        asm.emit_s::<SD>(self.operands.rs1, *v_rs2, 0);
        asm.emit_i::<VirtualMove>(self.operands.rd, *v_rd, 0);
        asm.finalize()
    }
}
