use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{advice_tape_read, Cpu, Xlen},
    instruction::format::format_advice_load_i::FormatAdviceLoadI,
    utils::inline_helpers::InstrAssembler,
};

use super::virtual_advice_load::VirtualAdviceLoad;
use super::{Cycle, Instruction, RISCVInstruction, RISCVTrace};
use crate::utils::virtual_registers::VirtualRegisterAllocator;

declare_riscv_instr!(
    name   = AdviceLD,
    mask   = 0,
    match  = 0,
    format = FormatAdviceLoadI,
    ram    = ()
);

impl AdviceLD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AdviceLD as RISCVInstruction>::RAMAccess) {
        // Read 8 bytes (doubleword) from the advice tape
        let advice_value = advice_tape_read(cpu, 8).expect("Failed to read from advice tape");
        // Store the advice value to register rd
        cpu.x[self.operands.rd as usize] = advice_value as i64;
    }
}

impl RISCVTrace for AdviceLD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Load doubleword (64-bit) from advice tape to register.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        match xlen {
            Xlen::Bit32 => panic!("LD is not supported in 32-bit mode"),
            Xlen::Bit64 => self.inline_sequence_64(allocator),
        }
    }
}

impl AdviceLD {
    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        // Read 8 bytes from advice tape into the register rd
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);
        asm.emit_j::<VirtualAdviceLoad>(self.operands.rd, 8);
        asm.finalize()
    }
}
