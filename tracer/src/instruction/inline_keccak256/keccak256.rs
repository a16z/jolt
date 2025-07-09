use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::InstructionFormat;
use crate::instruction::inline_keccak256::{
    execute_keccak_f, Keccak256SequenceBuilder, NEEDED_REGISTERS,
};
use crate::instruction::{
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = KECCAK256,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0200000b,  // funct7=0x01, funct3=0x0, opcode=0x0B (custom-0)
    format = FormatR,
    ram    = ()
);

impl KECCAK256 {
    fn exec(&self, cpu: &mut Cpu, _ram_access: &mut <KECCAK256 as RISCVInstruction>::RAMAccess) {
        // This is the "fast path" for emulation without tracing.
        // It performs the Keccak permutation using a native Rust implementation.

        // 1. Read the 25-lane (200-byte) state from memory pointed to by rs1.
        let mut state = [0u64; 25];
        let base_addr = cpu.x[self.operands.rs1] as u64;
        for (i, lane) in state.iter_mut().enumerate() {
            *lane = cpu
                .mmu
                .load_doubleword(base_addr.wrapping_add((i * 8) as u64))
                .expect("KECCAK256: Failed to load state from memory")
                .0;
        }

        // 2. Execute the Keccak-f permutation on the state.
        execute_keccak_f(&mut state);

        // 3. Write the permuted state back to memory.
        for (i, &lane) in state.iter().enumerate() {
            cpu.mmu
                .store_doubleword(base_addr.wrapping_add((i * 8) as u64), lane)
                .expect("KECCAK256: Failed to store state to memory");
        }
    }
}

impl RISCVTrace for KECCAK256 {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence();

        let mut trace = trace;
        for instr in virtual_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for KECCAK256 {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used as a scratch space
        let mut vr = [0; NEEDED_REGISTERS];
        (0..NEEDED_REGISTERS).for_each(|i| {
            vr[i] = virtual_register_index(i as u64) as usize;
        });
        let builder =
            Keccak256SequenceBuilder::new(self.address, vr, self.operands.rs1, self.operands.rs2);
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
    use crate::instruction::format::format_r::FormatR;

    const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

    #[test]
    fn test_keccak_state_equivalence() {
        // 1. Set up initial state and instruction
        let mut initial_state = [0u64; 25];
        for i in 0..25 {
            initial_state[i] = (i * 3 + 5) as u64; // Simple, predictable pattern
        }

        let instruction = KECCAK256 {
            address: 0,
            operands: FormatR {
                rs1: 10,
                rs2: 0,
                rd: 0,
            },
            virtual_sequence_remaining: None,
        };

        // 2. Set up the "exec" path CPU
        let mut cpu_exec = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu_exec.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let base_addr = DRAM_BASE;
        cpu_exec.x[10] = base_addr as i64;
        for (i, &lane) in initial_state.iter().enumerate() {
            cpu_exec
                .mmu
                .store_doubleword(base_addr + (i * 8) as u64, lane)
                .unwrap();
        }

        // 3. Set up the "trace" path CPU (must be identical)
        let mut cpu_trace = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu_trace.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        cpu_trace.x[10] = base_addr as i64;
        for (i, &lane) in initial_state.iter().enumerate() {
            cpu_trace
                .mmu
                .store_doubleword(base_addr + (i * 8) as u64, lane)
                .unwrap();
        }

        // 4. Run both paths
        instruction.exec(&mut cpu_exec, &mut ());
        instruction.trace(&mut cpu_trace, None);

        // 5. Assert that the final memory states are identical
        for i in 0..25 {
            let addr = base_addr + (i * 8) as u64;
            let val_exec = cpu_exec.mmu.load_doubleword(addr).unwrap().0;
            let val_trace = cpu_trace.mmu.load_doubleword(addr).unwrap().0;
            if val_exec != val_trace {
                println!(
                    "Mismatch at lane {}: exec {:#x}, trace {:#x}",
                    i, val_exec, val_trace
                );
            }
            assert_eq!(val_exec, val_trace, "State mismatch at lane {}", i);
        }
    }
}
