use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::Cpu;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::InstructionFormat;
use crate::instruction::inline_sha256::{
    execute_sha256_compression_initial, Sha256SequenceBuilder, NEEDED_REGISTERS,
};
use crate::instruction::{
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = SHA256INIT,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0000100b,  // funct7=0x00, funct3=0x1, opcode=0x0B (custom-0)
    format = FormatR,
    ram    = ()
);

impl SHA256INIT {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SHA256INIT as RISCVInstruction>::RAMAccess) {
        // Load 16 input words from memory at rs1
        let mut input = [0u32; 16];
        for (i, word) in input.iter_mut().enumerate() {
            *word = cpu
                .mmu
                .load_word(cpu.x[self.operands.rs1].wrapping_add((i * 4) as i64) as u64)
                .expect("SHA256INIT: Failed to load input word")
                .0;
        }

        // Execute compression with default initial state and store result
        let result = execute_sha256_compression_initial(input);
        for (i, &word) in result.iter().enumerate() {
            cpu.mmu
                .store_word(
                    cpu.x[self.operands.rs2].wrapping_add((i * 4) as i64) as u64,
                    word,
                )
                .expect("SHA256INIT: Failed to store result");
        }
    }
}

impl RISCVTrace for SHA256INIT {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence();

        let mut trace = trace;
        for instr in virtual_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for SHA256INIT {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used as a scratch space
        let mut vr = [0; NEEDED_REGISTERS];
        (0..NEEDED_REGISTERS).for_each(|i| {
            vr[i] = virtual_register_index(i as u64) as usize;
        });
        let builder = Sha256SequenceBuilder::new(
            self.address,
            vr,
            self.operands.rs1,
            self.operands.rs2,
            true, // initial - uses BLOCK constants
        );
        builder.build()
    }
}
