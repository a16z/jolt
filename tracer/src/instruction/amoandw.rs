use serde::{Deserialize, Serialize};

use super::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use super::and::AND;
use super::Instruction;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_r::FormatR, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOANDW,
    mask   = 0xf800707f,
    match  = 0x6000202f,
    format = FormatR,
    ram    = ()
);

impl AMOANDW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOANDW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let and_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // AND the values and store back to memory
        let new_value = (original_value as u32) & and_value;
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOANDW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_rd = allocator.allocate();
        let v_rs2 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        match xlen {
            Xlen::Bit32 => {
                amo_pre32(&mut asm, self.operands.rs1, *v_rd);
                asm.emit_r::<AND>(*v_rs2, *v_rd, self.operands.rs2);
                amo_post32(&mut asm, *v_rs2, self.operands.rs1, self.operands.rd, *v_rd);
            }
            Xlen::Bit64 => {
                let v_mask = allocator.allocate();
                let v_dword_address = allocator.allocate();
                let v_dword = allocator.allocate();
                let v_word = allocator.allocate();
                let v_shift = allocator.allocate();

                amo_pre64(
                    &mut asm,
                    self.operands.rs1,
                    *v_rd,
                    *v_dword_address,
                    *v_dword,
                    *v_shift,
                );
                asm.emit_r::<AND>(*v_rs2, *v_rd, self.operands.rs2);
                amo_post64(
                    &mut asm,
                    *v_rs2,
                    *v_dword_address,
                    *v_dword,
                    *v_shift,
                    *v_mask,
                    *v_word,
                    self.operands.rd,
                    *v_rd,
                );
            }
        }

        asm.finalize()
    }
}
