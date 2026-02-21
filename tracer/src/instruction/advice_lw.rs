use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{advice_tape_read, Cpu, Xlen},
    instruction::{format::format_advice_load_i::FormatAdviceLoadI, SLLI, SRAI},
    utils::inline_helpers::InstrAssembler,
};

use super::virtual_advice_load::VirtualAdviceLoad;
use super::{Cycle, Instruction, RISCVInstruction, RISCVTrace};
use crate::utils::virtual_registers::VirtualRegisterAllocator;

declare_riscv_instr!(
    name   = AdviceLW,
    mask   = 0,
    match  = 0,
    format = FormatAdviceLoadI,
    ram    = ()
);

impl AdviceLW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AdviceLW as RISCVInstruction>::RAMAccess) {
        // Read 4 bytes (word) from the advice tape
        let advice_value = advice_tape_read(cpu, 4).expect("Failed to read from advice tape");
        // Store the sign extended advice value to register rd
        cpu.write_register(self.operands.rd as usize, advice_value as i32 as i64);
    }
}

impl RISCVTrace for AdviceLW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Load word (32-bit) from advice tape to register.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        match xlen {
            Xlen::Bit32 => self.inline_sequence_32(allocator),
            Xlen::Bit64 => self.inline_sequence_64(allocator),
        }
    }
}

impl AdviceLW {
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        // Read 4 bytes from advice tape into the register rd
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);
        asm.emit_j::<VirtualAdviceLoad>(self.operands.rd, 4);
        asm.finalize()
    }
    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        // Read 4 bytes from advice tape into the register rd
        // And then shift twice to wipe upper 32 bits and sign extend
        // rd = (rd << 32) >>> 32
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);
        asm.emit_j::<VirtualAdviceLoad>(self.operands.rd, 4);
        asm.emit_i::<SLLI>(self.operands.rd, self.operands.rd, 32);
        asm.emit_i::<SRAI>(self.operands.rd, self.operands.rd, 32);
        asm.finalize()
    }
}
