use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::InstructionFormat;
use crate::instruction::inline_keccak256::{
    execute_keccak_f, Keccak256SequenceBuilder, NEEDED_REGISTERS,
};
use crate::instruction::RAMRead;
use crate::instruction::RAMWrite;
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
    use crate::emulator::test_harness::CpuTestHarness;
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
    use crate::instruction::format::format_r::FormatR;
    use crate::instruction::inline_keccak256::test_constants::*;
    use crate::instruction::inline_keccak256::test_utils::*;
    use crate::instruction::inline_keccak256::{
        execute_keccak_f, Keccak256SequenceBuilder, NEEDED_REGISTERS,
    };

    const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

    // Test that the virtual sequence and direct execution paths produce the same end result
    #[test]
    fn test_keccak_state_equivalence() {
        for (desc, state) in TestVectors::get_standard_test_vectors() {
            println!("\n=== Testing: {} ===", desc);
            kverify::assert_exec_trace_equiv(&state, desc);
        }
    }

    // Test that the direct execution (exec method) works correctly against known test vectors
    #[test]
    fn test_keccak256_direct_execution() {
        // Test that the direct execution (exec method) works correctly against known test vectors
        // This validates the instruction logic before testing virtual sequences
        for (i, test_case) in keccak_test_vectors().iter().enumerate() {
            println!(
                "Running Keccak256 direct execution test case {}: {}",
                i + 1,
                test_case.description
            );

            let mut setup_exec = KeccakCpuHarness::new();
            setup_exec.load_state(&test_case.input);

            // Create and execute instruction
            setup_exec.execute_keccak_instruction();

            let result = setup_exec.read_state();
            assert_eq!(
                result, test_case.expected,
                "Keccak256 direct execution test case {} failed: {}\nInput: {:016x?}\nExpected: {:016x?}\nActual: {:016x?}",
                i + 1, test_case.description, test_case.input, test_case.expected, result
            );
        }
    }

    // Identifies exact point at which direct execution and virtual sequence diverge
    // by that virtual registers match (in contrast to test_keccak_state_equivalence which only tests the final state).
    #[test]
    fn debug_virtual_sequence_divergence() {
        for (description, initial_state) in TestVectors::get_standard_test_vectors() {
            println!(
                "\nTesting debug_virtual_sequence_divergence with test case: {} ===",
                description
            );
            // Test each round and step
            for round in 0..24 {
                for step in &["theta", "rho_and_pi", "chi", "iota"] {
                    // For debugging, you can uncomment this to see the progress.
                    // println!("\n--- Testing Round {} after {} ---", round, step);

                    // 1. Set up a fresh CPU for this specific test case.
                    let mut setup = KeccakCpuHarness::new();
                    setup.load_state(&initial_state);

                    // 2. Generate and execute the virtual sequence up to the current step.
                    let builder = super::Keccak256SequenceBuilder::new(0x1000, setup.vr, 10, 11);
                    let sequence = builder.build_up_to_step(round, step);
                    setup.execute_virtual_sequence(&sequence);
                    let vr_state = setup.read_vr();

                    // 3. Compute the correct, expected state using the reference implementation helper.
                    let expected_vr_state =
                        execute_reference_up_to_step(&initial_state, round as usize, step);

                    // 4. Assert that the states are equal. This will panic with a detailed
                    //    error message on the first divergence, stopping the test.
                    kverify::assert_states_equal(
                        &expected_vr_state,
                        &vr_state,
                        &format!(
                            "debug_virtual_sequence_divergence(round={}, step={})",
                            round, step
                        ),
                    );
                }
            }
        }
    }

    #[test]
    #[ignore] // For debugging purposes only
    fn debug_print_states() {
        // From https://github.com/XKCP/XKCP/blob/master/tests/TestVectors/KeccakF-1600-IntermediateValues.txt
        let initial_state = [0u64; 25];
        let expected_final_state = xkcp_vectors::AFTER_ONE_PERMUTATION;

        // EXEC PATH
        let mut setup_exec = KeccakCpuHarness::new();
        setup_exec.load_state(&initial_state);
        setup_exec.execute_keccak_instruction();
        let exec_result = setup_exec.read_state();

        // TRACE PATH
        let mut setup_trace = KeccakCpuHarness::new();
        setup_trace.load_state(&initial_state);
        setup_trace.trace_keccak_instruction();
        let trace_result = setup_trace.read_state();

        println!("DEBUGGING KECCAK-F STATE MISMATCH");
        for i in 0..25 {
            println!(
                "Lane {i:02}:  Exec: 0x{:016x}   Trace: 0x{:016x}   Expected: 0x{:016x}",
                exec_result[i], trace_result[i], expected_final_state[i]
            );
        }
    }

    #[test]
    #[ignore] // For debugging purposes only
    fn debug_rotation_issue() {
        println!("=== Debugging Keccak256 Rotation Issue ===");

        let mut setup = KeccakCpuHarness::new();

        // Test value that should reveal rotation issues
        let test_value: u64 = 0x123456789ABCDEF0;
        setup.harness.cpu.x[KeccakCpuHarness::RS1] = test_value as i64;

        // Get virtual register mapping
        let mut vr = [0; super::NEEDED_REGISTERS];
        for i in 0..super::NEEDED_REGISTERS {
            vr[i] = virtual_register_index(i as u64) as usize;
        }

        // Test rotation by 1 (common in Keccak theta step)
        println!("\nTesting ROTL by 1:");
        println!("Input: 0x{:016x}", test_value);
        println!("Expected: 0x{:016x}", test_value.rotate_left(1));

        // We need to access the builder's methods, but they're private
        // So let's test via the full sequence and examine the first rotation

        // Set up a test state with a pattern that makes rotations obvious
        setup.load_state(&custom_vectors::ROTATION_TEST);

        // Generate just the theta step which uses rotation
        let instruction = KeccakCpuHarness::instruction();
        let sequence = instruction.virtual_sequence();

        // Find the first VirtualMULI or rotation-related instruction
        let mut found_rotation = false;
        for (i, instr) in sequence.iter().enumerate() {
            match instr {
                RV32IMInstruction::VirtualMULI(muli) => {
                    println!(
                        "\nFound VirtualMULI at instruction {}: rd={}, rs1={}, imm=0x{:x}",
                        i, muli.operands.rd, muli.operands.rs1, muli.operands.imm
                    );
                    found_rotation = true;
                    break;
                }
                RV32IMInstruction::SLLI(slli) => {
                    println!(
                        "\nFound SLLI at instruction {}: rd={}, rs1={}, imm={}",
                        i, slli.operands.rd, slli.operands.rs1, slli.operands.imm
                    );
                }
                RV32IMInstruction::SRLI(srli) => {
                    println!(
                        "\nFound SRLI at instruction {}: rd={}, rs1={}, imm={}",
                        i, srli.operands.rd, srli.operands.rs1, srli.operands.imm
                    );
                }
                _ => {}
            }
        }

        if !found_rotation {
            println!("WARNING: No rotation operations found in sequence!");
        }

        // Execute a portion of the sequence and check intermediate values
        let mut instruction_count = 0;
        for instr in sequence.iter().take(100) {
            // Just first 100 instructions
            setup.execute_virtual_sequence(&[instr.clone()]);
            instruction_count += 1;

            // Check if any virtual registers have interesting values
            if instruction_count % 10 == 0 {
                println!("\nAfter {} instructions:", instruction_count);
                let vr_state = setup.read_vr();
                for i in 0..10 {
                    if vr_state[i] != 0 {
                        println!("  vr[{}] = 0x{:016x}", i, vr_state[i]);
                    }
                }
            }
        }

        println!("\n✅ Debug test completed");
    }
}
