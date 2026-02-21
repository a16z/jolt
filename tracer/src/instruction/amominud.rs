use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{instruction::addi::ADDI, utils::inline_helpers::InstrAssembler};
use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::ld::LD;
use super::mul::MUL;
use super::sd::SD;
use super::sltu::SLTU;
use super::sub::SUB;
use super::Instruction;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_amo::FormatAMO, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOMINUD,
    mask   = 0xf800707f,
    match  = 0xc000302f,
    format = FormatAMO,
    ram    = ()
);

impl AMOMINUD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOMINUD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let compare_value = cpu.x[self.operands.rs2 as usize] as u64;

        // Load the original doubleword from memory
        let load_result = cpu.mmu.load_doubleword(address);
        let original_value = match load_result {
            Ok((doubleword, _)) => doubleword as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Find the minimum (unsigned comparison) and store back to memory
        let new_value = if (original_value as u64) <= compare_value {
            original_value as u64
        } else {
            compare_value
        };
        cpu.mmu
            .store_doubleword(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.write_register(self.operands.rd as usize, original_value);
    }
}

impl RISCVTrace for AMOMINUD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates inline sequence for atomic minimum operation (unsigned 64-bit).
    ///
    /// AMOMINU.D atomically loads a 64-bit value from memory, computes the minimum
    /// of that value and rs2 (treating both as unsigned), stores the minimum back
    /// to memory, and returns the original value in rd.
    ///
    /// Uses the same branchless approach as AMOMIN.D but with unsigned comparison:
    /// 1. Load the current value from memory into v0
    /// 2. v1 = (rs2 < v0 ? 1 : 0)
    /// 3. v0 + (rs2 - v0) * v1
    ///    - If rs2 < v0, then v1 = 1, and the result is v0 + (rs2 - v0) * 1 = rs2
    ///    - If rs2 >= v0, then v1 = 0, and the result is v0 + (rs2 - v0) * 0 = v0
    /// 4. Store the minimum back to memory
    /// 5. Return the original value in rd
    ///
    /// The branchless multiplication technique ensures zkVM compatibility.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v0 = allocator.allocate();
        let v1 = allocator.allocate();
        let v2 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        asm.emit_ld::<LD>(*v0, self.operands.rs1, 0);
        asm.emit_r::<SLTU>(*v1, self.operands.rs2, *v0);
        asm.emit_r::<SUB>(*v2, self.operands.rs2, *v0);
        asm.emit_r::<MUL>(*v2, *v2, *v1);
        asm.emit_r::<ADD>(*v1, *v0, *v2);
        asm.emit_s::<SD>(self.operands.rs1, *v1, 0);
        asm.emit_i::<ADDI>(self.operands.rd, *v0, 0);

        asm.finalize()
    }
}
