use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::{Cpu, Xlen};
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::InstructionFormat;
use crate::instruction::inline_sha256::{
    execute_sha256_compression, Sha256SequenceBuilder, NEEDED_REGISTERS,
};
use crate::instruction::{
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = SHA256,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0000000b,  // funct7=0x00, funct3=0x0, opcode=0x0B (custom-0)
    format = FormatR,
    ram    = ()
);

impl SHA256 {
    fn exec(&self, cpu: &mut Cpu, _ram_access: &mut <SHA256 as RISCVInstruction>::RAMAccess) {
        // Load 16 input words from memory at rs1
        let mut input = [0u32; 16];
        for (i, word) in input.iter_mut().enumerate() {
            *word = cpu
                .mmu
                .load_word(cpu.x[self.operands.rs1 as usize].wrapping_add((i * 4) as i64) as u64)
                .expect("SHA256: Failed to load input word")
                .0;
        }

        // Load 8 initial state words from memory at rs2
        let mut iv = [0u32; 8];
        for (i, word) in iv.iter_mut().enumerate() {
            *word = cpu
                .mmu
                .load_word(cpu.x[self.operands.rs2 as usize].wrapping_add((i * 4) as i64) as u64)
                .expect("SHA256: Failed to load initial state")
                .0;
        }

        // Execute compression and store result at rs2
        let result = execute_sha256_compression(iv, input);
        for (i, &word) in result.iter().enumerate() {
            cpu.mmu
                .store_word(
                    cpu.x[self.operands.rs2 as usize].wrapping_add((i * 4) as i64) as u64,
                    word,
                )
                .expect("SHA256: Failed to store result");
        }
    }
}

impl RISCVTrace for SHA256 {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen);

        let mut trace = trace;
        for instr in virtual_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for SHA256 {
    fn virtual_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used as a scratch space
        let mut vr = [0; NEEDED_REGISTERS as usize];
        (0..NEEDED_REGISTERS).for_each(|i| {
            vr[i as usize] = virtual_register_index(i);
        });
        let builder = Sha256SequenceBuilder::new(
            self.address,
            self.is_compressed,
            vr,
            self.operands.rs1,
            self.operands.rs2,
            false, // not initial - uses custom IV from rs2
        );
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use crate::instruction::inline_sha256::test_utils::{sverify, Sha256CpuHarness, TestVectors};
    use crate::instruction::RISCVInstruction;

    #[test]
    fn test_sha256_direct_execution() {
        // Test against multiple canonical NIST test vectors
        for (desc, block, initial_state, expected) in TestVectors::get_standard_test_vectors() {
            let mut harness = Sha256CpuHarness::new();
            harness.load_block(&block);
            harness.load_state(&initial_state);
            Sha256CpuHarness::instruction_sha256().execute(&mut harness.harness.cpu, &mut ());
            let result = harness.read_state();

            sverify::assert_states_equal(
                &expected,
                &result,
                &format!("SHA256 direct execution: {desc}"),
            );
        }
    }

    #[test]
    fn test_sha256_exec_trace_equal() {
        // Test exec vs trace equivalence with canonical test vectors
        for (desc, block, initial_state, _expected) in TestVectors::get_standard_test_vectors() {
            sverify::assert_exec_trace_equiv_custom(
                &block,
                &initial_state,
                &format!("SHA256 exec vs trace: {desc}"),
            );
        }
    }

    #[test]
    fn measure_sha256_length() {
        use crate::instruction::VirtualInstructionSequence;
        let instr = Sha256CpuHarness::instruction_sha256();
        let sequence = instr.virtual_sequence(crate::emulator::cpu::Xlen::Bit32);
        let bytecode_len = sequence.len();
        println!(
            "SHA256 compression: bytecode length {}, {:.2} instructions per byte",
            bytecode_len,
            bytecode_len as f64 / 64.0,
        );
    }
}
