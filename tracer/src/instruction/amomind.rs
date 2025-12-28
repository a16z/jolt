use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::ld::LD;
use super::mul::MUL;
use super::sd::SD;
use super::slt::SLT;
use super::xori::XORI;
use super::Instruction;
use crate::emulator::cpu::GeneralizedCpu;
use crate::emulator::memory::MemoryData;
use crate::instruction::addi::ADDI;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{declare_riscv_instr, emulator::cpu::Xlen};

use super::{format::format_amo::FormatAMO, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOMIND,
    mask   = 0xf800707f,
    match  = 0x8000302f,
    format = FormatAMO,
    ram    = ()
);

impl AMOMIND {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <AMOMIND as RISCVInstruction>::RAMAccess,
    ) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let compare_value = cpu.x[self.operands.rs2 as usize];

        // Load the original doubleword from memory
        let load_result = cpu.mmu.load_doubleword(address);
        let original_value = match load_result {
            Ok((doubleword, _)) => doubleword as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Find the minimum and store back to memory
        let new_value = if original_value <= compare_value {
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

impl RISCVTrace for AMOMIND {
    fn trace<D: MemoryData>(&self, cpu: &mut GeneralizedCpu<D>, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates inline sequence for atomic minimum operation (signed 64-bit).
    ///
    /// AMOMIN.D atomically loads a 64-bit value from memory, computes the minimum
    /// of that value and rs2 (treating both as signed), stores the minimum back
    /// to memory, and returns the original value in rd.
    ///
    /// Uses branchless minimum computation:
    /// 1. Load current value from memory
    /// 2. Compare rs2 < current to get selector bit (1 if rs2 is smaller)
    /// 3. Invert to get selector for current (1 if current is smaller/equal)
    /// 4. Multiply and add to select minimum without branches
    /// 5. Store result and return original value
    ///
    /// Note the comparison order: SLT(rs2, current) not SLT(current, rs2),
    /// ensuring correct minimum selection semantics.
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
        asm.emit_r::<SLT>(*v_sel_rs2, self.operands.rs2, *v_rd);
        asm.emit_i::<XORI>(*v_sel_rd, *v_sel_rs2, 1);
        asm.emit_r::<MUL>(*v_rs2, *v_sel_rs2, self.operands.rs2);
        asm.emit_r::<MUL>(*v_sel_rd, *v_sel_rd, *v_rd);
        asm.emit_r::<ADD>(*v_rs2, *v_sel_rd, *v_rs2);
        asm.emit_s::<SD>(self.operands.rs1, *v_rs2, 0);
        asm.emit_i::<ADDI>(self.operands.rd, *v_rd, 0);
        asm.finalize()
    }
}
