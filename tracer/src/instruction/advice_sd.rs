use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{advice_tape_read, Cpu, Xlen},
    utils::inline_helpers::InstrAssembler,
};

use super::sd::SD;
use super::virtual_advice_load::VirtualAdviceLoad;
use super::{
    format::format_advice_s::FormatAdviceS, Cycle, Instruction, RAMWrite, RISCVInstruction,
    RISCVTrace,
};
use crate::utils::virtual_registers::VirtualRegisterAllocator;

declare_riscv_instr!(
    name   = AdviceSD,
    mask   = 0,
    match  = 0,
    format = FormatAdviceS,
    ram    = RAMWrite
);

impl AdviceSD {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <AdviceSD as RISCVInstruction>::RAMAccess) {
        // Read 8 bytes (doubleword) from the advice tape
        let advice_value = advice_tape_read(8).expect("Failed to read from advice tape");

        // Store the advice value to memory at address rs1 + imm
        *ram_access = cpu
            .mmu
            .store_doubleword(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                advice_value,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for AdviceSD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Store doubleword (64-bit) from advice tape to memory.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        match xlen {
            Xlen::Bit32 => panic!("SD is not supported in 32-bit mode"),
            Xlen::Bit64 => self.inline_sequence_64(allocator),
        }
    }
}

impl AdviceSD {
    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_dword = allocator.allocate();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);
        // Read 8 bytes from advice tape into v_dword register
        asm.emit_i::<VirtualAdviceLoad>(*v_dword, 0, 8);
        // Store v_dword to memory at rs1 + imm
        asm.emit_s::<SD>(self.operands.rs1, *v_dword, self.operands.imm);
        asm.finalize()
    }
}
