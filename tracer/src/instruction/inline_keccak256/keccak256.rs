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
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
    use crate::instruction::format::format_r::FormatR;
    use crate::instruction::inline_keccak256::{
        execute_chi, execute_iota, execute_keccak256, execute_keccak_f, execute_rho_and_pi,
        execute_theta, ROTATION_OFFSETS, ROUND_CONSTANTS,
    };
    use crate::instruction::{RAMRead, RAMWrite};
    use common::constants::virtual_register_index;
    use hex_literal::hex;

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

    #[test]
    #[ignore] // For debugging purposes only
    fn debug_print_states() {
        // From https://github.com/XKCP/XKCP/blob/master/tests/TestVectors/KeccakF-1600-IntermediateValues.txt
        let initial_state_vec = vec![0u64; 25];
        let mut expected_final_state = [0u64; 25];
        let final_state_vec: Vec<u64> = vec![
            0xF1258F7940E1DDE7,
            0x84D5CCF933C0478A,
            0xD598261EA65AA9EE,
            0xBD1547306F80494D,
            0x8B284E056253D057,
            0xFF97A42D7F8E6FD4,
            0x90FEE5A0A44647C4,
            0x8C5BDA0CD6192E76,
            0xAD30A6F71B19059C,
            0x30935AB7D08FFC64,
            0xEB5AA93F2317D635,
            0xA9A6E6260D712103,
            0x81A57C16DBCF555F,
            0x43B831CD0347C826,
            0x01F22F1A11A5569F,
            0x05E5635A21D9AE61,
            0x64BEFEF28CC970F2,
            0x613670957BC46611,
            0xB87C5A554FD00ECB,
            0x8C3EE88A1CCF32C8,
            0x940C7922AE3A2614,
            0x1841F924A2C509E4,
            0x16F53526E70465C2,
            0x75F644E97F30A13B,
            0xEAF1FF7B5CECA249,
        ];
        expected_final_state.copy_from_slice(&final_state_vec);

        let instruction = KECCAK256 {
            address: 0,
            operands: FormatR {
                rs1: 10,
                rs2: 0,
                rd: 0,
            },
            virtual_sequence_remaining: None,
        };

        // EXEC PATH
        let mut cpu_exec = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu_exec.get_mut_mmu().init_memory(1024 * 1024);
        let base_addr = DRAM_BASE;
        cpu_exec.x[10] = base_addr as i64;
        for (i, &lane) in initial_state_vec.iter().enumerate() {
            cpu_exec
                .mmu
                .store_doubleword(base_addr + (i * 8) as u64, lane)
                .unwrap();
        }
        instruction.exec(&mut cpu_exec, &mut ());

        // TRACE PATH
        let mut cpu_trace = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu_trace.get_mut_mmu().init_memory(1024 * 1024);
        cpu_trace.x[10] = base_addr as i64;
        for (i, &lane) in initial_state_vec.iter().enumerate() {
            cpu_trace
                .mmu
                .store_doubleword(base_addr + (i * 8) as u64, lane)
                .unwrap();
        }
        instruction.trace(&mut cpu_trace, None);

        println!("DEBUGGING KECCAK-F STATE MISMATCH");
        for i in 0..25 {
            let addr = base_addr + (i * 8) as u64;
            let val_exec = cpu_exec.mmu.load_doubleword(addr).unwrap().0;
            let val_trace = cpu_trace.mmu.load_doubleword(addr).unwrap().0;
            println!(
                "Lane {i:02}:  Exec: 0x{:016x}   Trace: 0x{:016x}   Expected: 0x{:016x}",
                val_exec, val_trace, expected_final_state[i]
            );
        }
    }

    #[test]
    fn test_keccak256_direct_execution() {
        // Test that the direct execution (exec method) works correctly against known test vectors
        // This validates the instruction logic before testing virtual sequences

        let test_cases = vec![
            // Test case 1: All zeros input (standard test vector)
            TestCase {
                input: [0u64; 25],
                expected: [
                    0xF1258F7940E1DDE7,
                    0x84D5CCF933C0478A,
                    0xD598261EA65AA9EE,
                    0xBD1547306F80494D,
                    0x8B284E056253D057,
                    0xFF97A42D7F8E6FD4,
                    0x90FEE5A0A44647C4,
                    0x8C5BDA0CD6192E76,
                    0xAD30A6F71B19059C,
                    0x30935AB7D08FFC64,
                    0xEB5AA93F2317D635,
                    0xA9A6E6260D712103,
                    0x81A57C16DBCF555F,
                    0x43B831CD0347C826,
                    0x01F22F1A11A5569F,
                    0x05E5635A21D9AE61,
                    0x64BEFEF28CC970F2,
                    0x613670957BC46611,
                    0xB87C5A554FD00ECB,
                    0x8C3EE88A1CCF32C8,
                    0x940C7922AE3A2614,
                    0x1841F924A2C509E4,
                    0x16F53526E70465C2,
                    0x75F644E97F30A13B,
                    0xEAF1FF7B5CECA249,
                ],
                description: "All zeros input (XKCP test vector)",
            },
            // Test case 2: Simple pattern
            TestCase {
                input: {
                    let mut state = [0u64; 25];
                    for i in 0..25 {
                        state[i] = (i * 3 + 5) as u64;
                    }
                    state
                },
                expected: {
                    // We'll calculate this by running the reference implementation
                    let mut state = [0u64; 25];
                    for i in 0..25 {
                        state[i] = (i * 3 + 5) as u64;
                    }
                    execute_keccak_f(&mut state);
                    state
                },
                description: "Simple arithmetic pattern",
            },
            // Test case 3: Single bit set
            TestCase {
                input: {
                    let mut state = [0u64; 25];
                    state[0] = 1;
                    state
                },
                expected: {
                    let mut state = [0u64; 25];
                    state[0] = 1;
                    execute_keccak_f(&mut state);
                    state
                },
                description: "Single bit in first lane",
            },
        ];

        for (i, test_case) in test_cases.iter().enumerate() {
            println!(
                "Running Keccak256 direct execution test case {}: {}",
                i + 1,
                test_case.description
            );

            // Set up CPU and memory
            let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
            cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
            let base_addr = DRAM_BASE;
            cpu.x[10] = base_addr as i64;

            // Store input state to memory
            for (j, &lane) in test_case.input.iter().enumerate() {
                cpu.mmu
                    .store_doubleword(base_addr + (j * 8) as u64, lane)
                    .expect("Failed to store input lane");
            }

            // Create and execute instruction
            let instruction = KECCAK256 {
                address: 0,
                operands: FormatR {
                    rs1: 10,
                    rs2: 0,
                    rd: 0,
                },
                virtual_sequence_remaining: None,
            };

            instruction.exec(&mut cpu, &mut ());

            // Read result from memory
            let mut result = [0u64; 25];
            for (j, lane) in result.iter_mut().enumerate() {
                *lane = cpu
                    .mmu
                    .load_doubleword(base_addr + (j * 8) as u64)
                    .expect("Failed to load result lane")
                    .0;
            }

            // Compare with expected result
            assert_eq!(
                result, test_case.expected,
                "Keccak256 direct execution test case {} failed: {}\nInput: {:016x?}\nExpected: {:016x?}\nActual: {:016x?}",
                i + 1, test_case.description, test_case.input, test_case.expected, result
            );
        }

        println!("All Keccak256 direct execution test cases passed!");
    }

    #[test]
    fn test_execute_keccak256() {
        // Test vectors for the end-to-end Keccak-256 hash function.
        let e2e_vectors: &[(&[u8], [u8; 32])] = &[
            (
                b"",
                hex!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
            ),
            (
                b"abc",
                hex!("4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"),
            ),
        ];

        for (input, expected_hash) in e2e_vectors {
            let hash = execute_keccak256(input);
            assert_eq!(
                &hash,
                expected_hash,
                "Failed on e2e test vector for input: {:?}",
                std::str::from_utf8(input).unwrap_or("invalid utf-8")
            );
        }
    }

    #[test]
    fn test_execute_keccak_f() {
        // Test vectors for the Keccak-f[1600] permutation from XKCP.
        // https://github.com/XKCP/XKCP/blob/master/tests/TestVectors/KeccakF-1600-IntermediateValues.txt
        let after_one_permutation = [
            0xF1258F7940E1DDE7,
            0x84D5CCF933C0478A,
            0xD598261EA65AA9EE,
            0xBD1547306F80494D,
            0x8B284E056253D057,
            0xFF97A42D7F8E6FD4,
            0x90FEE5A0A44647C4,
            0x8C5BDA0CD6192E76,
            0xAD30A6F71B19059C,
            0x30935AB7D08FFC64,
            0xEB5AA93F2317D635,
            0xA9A6E6260D712103,
            0x81A57C16DBCF555F,
            0x43B831CD0347C826,
            0x01F22F1A11A5569F,
            0x05E5635A21D9AE61,
            0x64BEFEF28CC970F2,
            0x613670957BC46611,
            0xB87C5A554FD00ECB,
            0x8C3EE88A1CCF32C8,
            0x940C7922AE3A2614,
            0x1841F924A2C509E4,
            0x16F53526E70465C2,
            0x75F644E97F30A13B,
            0xEAF1FF7B5CECA249,
        ];
        let after_two_permutations = [
            0x2D5C954DF96ECB3C,
            0x6A332CD07057B56D,
            0x093D8D1270D76B6C,
            0x8A20D9B25569D094,
            0x4F9C4F99E5E7F156,
            0xF957B9A2DA65FB38,
            0x85773DAE1275AF0D,
            0xFAF4F247C3D810F7,
            0x1F1B9EE6F79A8759,
            0xE4FECC0FEE98B425,
            0x68CE61B6B9CE68A1,
            0xDEEA66C4BA8F974F,
            0x33C43D836EAFB1F5,
            0xE00654042719DBD9,
            0x7CF8A9F009831265,
            0xFD5449A6BF174743,
            0x97DDAD33D8994B40,
            0x48EAD5FC5D0BE774,
            0xE3B8C8EE55B7B03C,
            0x91A0226E649E42E9,
            0x900E3129E7BADD7B,
            0x202A9EC5FAA3CCE8,
            0x5B3402464E1C3DB6,
            0x609F4E62A44C1059,
            0x20D06CD26A8FBF5C,
        ];

        let mut state = [0u64; 25]; // Initial state is all zeros.

        // First permutation
        execute_keccak_f(&mut state);
        assert_eq!(
            state, after_one_permutation,
            "Failed on first permutation of Keccak-f"
        );

        // Second permutation
        execute_keccak_f(&mut state);
        assert_eq!(
            state, after_two_permutations,
            "Failed on second permutation of Keccak-f"
        );
    }

    #[test]
    fn test_step_by_step_round_0() {
        // Test the first round step by step using XKCP intermediate values
        let mut state = [0u64; 25]; // All zeros initially

        // After theta (Round 0):
        // All zeros remain zeros in theta step for all-zero input
        execute_theta(&mut state);
        let expected_after_theta = [0u64; 25];
        assert_eq!(state, expected_after_theta, "Round 0: Failed after theta");

        // After rho (Round 0):
        // All zeros remain zeros in rho step
        execute_rho_and_pi(&mut state);
        let expected_after_rho_pi = [0u64; 25];
        assert_eq!(
            state, expected_after_rho_pi,
            "Round 0: Failed after rho and pi"
        );

        // After chi (Round 0):
        // All zeros remain zeros in chi step
        execute_chi(&mut state);
        let expected_after_chi = [0u64; 25];
        assert_eq!(state, expected_after_chi, "Round 0: Failed after chi");

        // After iota (Round 0):
        // Only the first lane gets the round constant
        execute_iota(&mut state, ROUND_CONSTANTS[0]);
        let mut expected_after_iota = [0u64; 25];
        expected_after_iota[0] = 0x0000000000000001;
        assert_eq!(state, expected_after_iota, "Round 0: Failed after iota");
    }

    #[test]
    fn test_step_by_step_round_1() {
        // Start with the state after round 0
        let mut state = [0u64; 25];
        state[0] = 0x0000000000000001; // Result from round 0

        // Round 1 theta step
        execute_theta(&mut state);
        let expected_after_theta = [
            0x0000000000000001,
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000002,
            0x0000000000000000,
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000002,
            0x0000000000000000,
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000002,
            0x0000000000000000,
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000002,
            0x0000000000000000,
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000002,
        ];
        assert_eq!(state, expected_after_theta, "Round 1: Failed after theta");

        // Round 1 rho and pi step
        execute_rho_and_pi(&mut state);
        let expected_after_rho_pi = [
            0x0000000000008083u64,
            0x0000100000000000u64,
            0x0000000000008000u64,
            0x0000000000000001u64,
            0x0000100000008000u64,
            0x0000000000000000u64,
            0x0000200000200000u64,
            0x0000000000000000u64,
            0x0000200000000000u64,
            0x0000000000200000u64,
            0x0000000000000002u64,
            0x0000000000000200u64,
            0x0000000000000000u64,
            0x0000000000000202u64,
            0x0000000000000000u64,
            0x0000000010000400u64,
            0x0000000000000000u64,
            0x0000000000000400u64,
            0x0000000010000000u64,
            0x0000000000000000u64,
            0x0000010000000000u64,
            0x0000000000000000u64,
            0x0000010000000004u64,
            0x0000000000000000u64,
            0x0000000000000004u64,
        ];
        assert_eq!(
            state, expected_after_rho_pi,
            "Round 1: Failed after rho and pi"
        );

        // Round 1 chi step
        execute_chi(&mut state);
        let expected_after_chi = [
            0x0000000000000001u64,
            0x0000100000000000u64,
            0x0000000000008000u64,
            0x0000000000000001u64,
            0x0000100000008000u64,
            0x0000000000000000u64,
            0x0000200000200000u64,
            0x0000000000000000u64,
            0x0000200000000000u64,
            0x0000000000200000u64,
            0x0000000000000002u64,
            0x0000000000000200u64,
            0x0000000000000000u64,
            0x0000000000000202u64,
            0x0000000000000000u64,
            0x0000000010000400u64,
            0x0000000000000000u64,
            0x0000000000000400u64,
            0x0000000010000000u64,
            0x0000000000000000u64,
            0x0000010000000000u64,
            0x0000000000000000u64,
            0x0000010000000004u64,
            0x0000000000000000u64,
            0x0000000000000004u64,
        ];
        assert_eq!(state, expected_after_chi, "Round 1: Failed after chi");

        // Round 1 iota step
        execute_iota(&mut state, ROUND_CONSTANTS[1]);
        let expected_after_iota = [
            0x0000000000008083u64,
            0x0000100000000000u64,
            0x0000000000008000u64,
            0x0000000000000001u64,
            0x0000100000008000u64,
            0x0000000000000000u64,
            0x0000200000200000u64,
            0x0000000000000000u64,
            0x0000200000000000u64,
            0x0000000000200000u64,
            0x0000000000000002u64,
            0x0000000000000200u64,
            0x0000000000000000u64,
            0x0000000000000202u64,
            0x0000000000000000u64,
            0x0000000010000400u64,
            0x0000000000000000u64,
            0x0000000000000400u64,
            0x0000000010000000u64,
            0x0000000000000000u64,
            0x0000010000000000u64,
            0x0000000000000000u64,
            0x0000010000000004u64,
            0x0000000000000000u64,
            0x0000000000000004u64,
        ];
        assert_eq!(state, expected_after_iota, "Round 1: Failed after iota");
    }

    #[test]
    fn test_individual_step_functions() {
        // Test each step function individually to isolate any issues

        // Test theta with a known pattern
        let mut state = [0u64; 25];
        state[0] = 1;
        state[5] = 2;
        state[10] = 4;

        let original_state = state;
        execute_theta(&mut state);

        // Verify theta creates the expected column parity effects
        // This is a basic sanity check
        assert_ne!(state, original_state, "Theta should modify the state");

        // Test rho_and_pi - just verify it changes the state
        let mut state = [0u64; 25];
        state[1] = 0xFF;
        let original_state = state;
        execute_rho_and_pi(&mut state);

        // The function should move and rotate lanes, so state should change
        assert_ne!(state, original_state, "Rho and pi should modify the state");
        // Original position should be cleared (or changed)
        assert_ne!(
            state[1], 0xFF,
            "Original position should be different after rho_and_pi"
        );

        // Test chi
        let mut state = [0u64; 25];
        state[0] = 0xFF;
        state[1] = 0xAA;
        state[2] = 0x55;

        execute_chi(&mut state);

        // Chi should apply the non-linear transformation
        // Expected: state[0] ^= (~state[1] & state[2])
        let expected_0 = 0xFF ^ ((!0xAA) & 0x55);
        assert_eq!(state[0], expected_0, "Chi transformation failed");

        // Test iota
        let mut state = [0u64; 25];
        state[0] = 0x1234;

        execute_iota(&mut state, 0x5678);
        assert_eq!(
            state[0],
            0x1234 ^ 0x5678,
            "Iota should XOR round constant into first lane"
        );

        // Verify other lanes unchanged
        for i in 1..25 {
            assert_eq!(state[i], 0, "Iota should only affect first lane");
        }
    }

    #[test]
    fn test_rotation_offsets() {
        // Verify our rotation offsets match the XKCP reference
        let expected_offsets = [
            [0, 36, 3, 41, 18],
            [1, 44, 10, 45, 2],
            [62, 6, 43, 15, 61],
            [28, 55, 25, 21, 56],
            [27, 20, 39, 8, 14],
        ];

        assert_eq!(
            ROTATION_OFFSETS, expected_offsets,
            "Rotation offsets don't match XKCP reference"
        );
    }

    #[test]
    fn test_round_constants() {
        // Verify our round constants match the XKCP reference
        let expected_constants = [
            0x0000000000000001,
            0x0000000000008082,
            0x800000000000808a,
            0x8000000080008000,
            0x000000000000808b,
            0x0000000080000001,
            0x8000000080008081,
            0x8000000000008009,
            0x000000000000008a,
            0x0000000000000088,
            0x0000000080008009,
            0x000000008000000a,
            0x000000008000808b,
            0x800000000000008b,
            0x8000000000008089,
            0x8000000000008003,
            0x8000000000008002,
            0x8000000000000080,
            0x000000000000800a,
            0x800000008000000a,
            0x8000000080008081,
            0x8000000000008080,
            0x0000000080000001,
            0x8000000080008008,
        ];

        assert_eq!(
            ROUND_CONSTANTS, expected_constants,
            "Round constants don't match XKCP reference"
        );
    }

    #[test]
    fn test_virtual_sequence_step_by_step() {
        println!("=== Testing Virtual Sequence Step-by-Step ===");

        // Set up CPU and memory for virtual sequence execution
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let base_addr = DRAM_BASE;
        cpu.x[10] = base_addr as i64;

        // Initialize with all-zero state (same as XKCP test vectors)
        let initial_state = [0u64; 25];

        // Store initial state to memory
        for (j, &lane) in initial_state.iter().enumerate() {
            cpu.mmu
                .store_doubleword(base_addr + (j * 8) as u64, lane)
                .expect("Failed to store initial lane");
        }

        // Generate the virtual sequence
        let instruction = KECCAK256 {
            address: 0,
            operands: FormatR {
                rs1: 10,
                rs2: 0,
                rd: 0,
            },
            virtual_sequence_remaining: None,
        };

        let virtual_sequence = instruction.virtual_sequence();
        println!(
            "Generated virtual sequence with {} instructions",
            virtual_sequence.len()
        );

        // Expected state after each round (from XKCP test vectors)
        let expected_after_round_0 = {
            let mut state = [0u64; 25];
            state[0] = 0x0000000000000001; // After iota with RC[0]
            state
        };

        let expected_after_round_1 = [
            0x0000000000008083u64,
            0x0000100000000000u64,
            0x0000000000008000u64,
            0x0000000000000001u64,
            0x0000100000008000u64,
            0x0000000000000000u64,
            0x0000200000200000u64,
            0x0000000000000000u64,
            0x0000200000000000u64,
            0x0000000000200000u64,
            0x0000000000000002u64,
            0x0000000000000200u64,
            0x0000000000000000u64,
            0x0000000000000202u64,
            0x0000000000000000u64,
            0x0000000010000400u64,
            0x0000000000000000u64,
            0x0000000000000400u64,
            0x0000000010000000u64,
            0x0000000000000000u64,
            0x0000010000000000u64,
            0x0000000000000000u64,
            0x0000010000000004u64,
            0x0000000000000000u64,
            0x0000000000000004u64,
        ];

        // Execute virtual sequence and check intermediate states
        let mut instruction_count = 0;
        let mut round_count = 0;

        for (i, virtual_instr) in virtual_sequence.iter().enumerate() {
            // Execute the instruction
            match virtual_instr {
                RV32IMInstruction::LD(ld) => {
                    let mut ram_access = RAMRead::default();
                    ld.execute(&mut cpu, &mut ram_access);
                }
                RV32IMInstruction::SD(sd) => {
                    let mut ram_access = RAMWrite::default();
                    sd.execute(&mut cpu, &mut ram_access);
                }
                RV32IMInstruction::XOR(xor) => {
                    let mut ram_access = ();
                    xor.execute(&mut cpu, &mut ram_access);
                }
                RV32IMInstruction::XORI(xori) => {
                    let mut ram_access = ();
                    xori.execute(&mut cpu, &mut ram_access);
                }
                RV32IMInstruction::AND(and) => {
                    let mut ram_access = ();
                    and.execute(&mut cpu, &mut ram_access);
                }
                RV32IMInstruction::ANDI(andi) => {
                    let mut ram_access = ();
                    andi.execute(&mut cpu, &mut ram_access);
                }
                RV32IMInstruction::VirtualROTRI(rotri) => {
                    let mut ram_access = ();
                    rotri.execute(&mut cpu, &mut ram_access);
                }
                RV32IMInstruction::VirtualMULI(muli) => {
                    let mut ram_access = ();
                    muli.execute(&mut cpu, &mut ram_access);
                }
                RV32IMInstruction::ADDI(addi) => {
                    let mut ram_access = ();
                    addi.execute(&mut cpu, &mut ram_access);
                }
                RV32IMInstruction::LUI(lui) => {
                    let mut ram_access = ();
                    lui.execute(&mut cpu, &mut ram_access);
                }
                RV32IMInstruction::SRLI(srli) => {
                    let mut ram_access = ();
                    srli.execute(&mut cpu, &mut ram_access);
                }
                RV32IMInstruction::SLLI(slli) => {
                    let mut ram_access = ();
                    slli.execute(&mut cpu, &mut ram_access);
                }
                RV32IMInstruction::OR(or) => {
                    let mut ram_access = ();
                    or.execute(&mut cpu, &mut ram_access);
                }
                _ => {
                    panic!(
                        "Unexpected instruction type in Keccak256 virtual sequence: {:?}",
                        virtual_instr
                    );
                }
            }

            instruction_count += 1;

            // Check state at key points (this is a simplified check - we'd need to know
            // exactly when each round completes in the virtual sequence)
            if instruction_count == virtual_sequence.len() / 24 {
                // Rough estimate for end of round 0
                let current_state = read_state_from_memory(&mut cpu, base_addr);
                if current_state != expected_after_round_0 {
                    println!(
                        "DIVERGENCE DETECTED after ~round 0 (instruction {}):",
                        instruction_count
                    );
                    println!("Expected round 0 final state:");
                    print_state_hex(&expected_after_round_0);
                    println!("Actual state:");
                    print_state_hex(&current_state);

                    // Find first differing lane
                    for (lane_idx, (&expected, &actual)) in expected_after_round_0
                        .iter()
                        .zip(current_state.iter())
                        .enumerate()
                    {
                        if expected != actual {
                            println!(
                                "First difference at lane {}: expected 0x{:016x}, got 0x{:016x}",
                                lane_idx, expected, actual
                            );
                            break;
                        }
                    }
                    break;
                }
                round_count += 1;
            }
        }

        // Read final state
        let final_state = read_state_from_memory(&mut cpu, base_addr);

        // Compare with expected final state (after all 24 rounds)
        let expected_final_state = [
            0xF1258F7940E1DDE7,
            0x84D5CCF933C0478A,
            0xD598261EA65AA9EE,
            0xBD1547306F80494D,
            0x8B284E056253D057,
            0xFF97A42D7F8E6FD4,
            0x90FEE5A0A44647C4,
            0x8C5BDA0CD6192E76,
            0xAD30A6F71B19059C,
            0x30935AB7D08FFC64,
            0xEB5AA93F2317D635,
            0xA9A6E6260D712103,
            0x81A57C16DBCF555F,
            0x43B831CD0347C826,
            0x01F22F1A11A5569F,
            0x05E5635A21D9AE61,
            0x64BEFEF28CC970F2,
            0x613670957BC46611,
            0xB87C5A554FD00ECB,
            0x8C3EE88A1CCF32C8,
            0x940C7922AE3A2614,
            0x1841F924A2C509E4,
            0x16F53526E70465C2,
            0x75F644E97F30A13B,
            0xEAF1FF7B5CECA249,
        ];

        println!("=== Final State Comparison ===");
        println!("Expected final state:");
        print_state_hex(&expected_final_state);
        println!("Actual final state:");
        print_state_hex(&final_state);

        // Check if final state matches
        for (lane_idx, (&expected, &actual)) in expected_final_state
            .iter()
            .zip(final_state.iter())
            .enumerate()
        {
            if expected != actual {
                println!(
                    "FINAL STATE MISMATCH at lane {}: expected 0x{:016x}, got 0x{:016x}",
                    lane_idx, expected, actual
                );

                // This will fail the test and show us exactly where the issue is
                assert_eq!(
                    actual, expected,
                    "Virtual sequence produced incorrect result at lane {}",
                    lane_idx
                );
            }
        }

        println!("Virtual sequence step-by-step test completed successfully!");
    }

    #[test]
    fn test_virtual_sequence_granular() {
        println!("=== Testing Virtual Sequence Step-by-Step (Granular) ===");

        // Set up CPU and memory
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let base_addr = DRAM_BASE;
        cpu.x[10] = base_addr as i64;

        // Initialize with all-zero state (same as XKCP test vectors)
        let initial_state = [0u64; 25];

        // Store initial state to memory
        for (j, &lane) in initial_state.iter().enumerate() {
            cpu.mmu
                .store_doubleword(base_addr + (j * 8) as u64, lane)
                .expect("Failed to store initial lane");
        }

        // Get virtual register mapping
        let mut vr = [0; super::NEEDED_REGISTERS];
        for i in 0..super::NEEDED_REGISTERS {
            vr[i] = super::virtual_register_index(i as u64) as usize;
        }

        // Test load_state first
        println!("\n=== Testing load_state ===");
        let builder = super::Keccak256SequenceBuilder::new(0, vr, 10, 0);
        let load_sequence = builder.build_load_state_only();
        println!(
            "Generated load_state sequence with {} instructions",
            load_sequence.len()
        );

        // Execute load_state sequence
        for instr in &load_sequence {
            execute_instruction(&mut cpu, instr);
        }

        // Check if state was loaded into virtual registers
        println!("Checking virtual registers after load_state:");
        for i in 0..25 {
            let reg_val = cpu.x[vr[i]] as u64;
            println!("  vr[{}] (reg {}): 0x{:016x}", i, vr[i], reg_val);
            assert_eq!(
                reg_val, 0,
                "Virtual register {} should contain 0 after loading all-zero state",
                i
            );
        }

        // Now test round 0, step by step
        println!("\n=== Testing Round 0 ===");

        // Test after theta
        println!("\n--- After theta ---");
        let mut cpu_theta = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu_theta.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        cpu_theta.x[10] = base_addr as i64;

        // Store initial state
        for (j, &lane) in initial_state.iter().enumerate() {
            cpu_theta
                .mmu
                .store_doubleword(base_addr + (j * 8) as u64, lane)
                .expect("Failed to store initial lane");
        }

        let builder = super::Keccak256SequenceBuilder::new(0, vr, 10, 0);
        let theta_sequence = builder.build_up_to_step(0, "theta");
        println!(
            "Generated sequence up to theta with {} instructions",
            theta_sequence.len()
        );

        for instr in &theta_sequence {
            execute_instruction(&mut cpu_theta, instr);
        }

        // Check virtual registers after theta
        println!("Virtual registers after theta:");
        for i in 0..25 {
            let reg_val = cpu_theta.x[vr[i]] as u64;
            if reg_val != 0 {
                println!("  vr[{}] (reg {}): 0x{:016x}", i, vr[i], reg_val);
            }
        }

        // Test after iota (where we expect the first non-zero value)
        println!("\n--- After iota (full round 0) ---");
        let mut cpu_iota = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu_iota.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        cpu_iota.x[10] = base_addr as i64;

        // Store initial state
        for (j, &lane) in initial_state.iter().enumerate() {
            cpu_iota
                .mmu
                .store_doubleword(base_addr + (j * 8) as u64, lane)
                .expect("Failed to store initial lane");
        }

        let builder = super::Keccak256SequenceBuilder::new(0, vr, 10, 0);
        let iota_sequence = builder.build_up_to_step(0, "iota");
        println!(
            "Generated sequence up to iota with {} instructions",
            iota_sequence.len()
        );

        for instr in &iota_sequence {
            execute_instruction(&mut cpu_iota, instr);
        }

        // Check virtual registers after iota - vr[0] should be 0x1
        println!("Virtual registers after iota:");
        let mut found_nonzero = false;
        for i in 0..25 {
            let reg_val = cpu_iota.x[vr[i]] as u64;
            if reg_val != 0 {
                println!("  vr[{}] (reg {}): 0x{:016x}", i, vr[i], reg_val);
                found_nonzero = true;
                if i == 0 {
                    assert_eq!(
                        reg_val, 0x0000000000000001,
                        "After round 0 iota, vr[0] should be 0x1"
                    );
                }
            }
        }

        assert!(
            found_nonzero,
            "After round 0 iota, at least vr[0] should be non-zero!"
        );

        // Now test the full sequence with store_state
        println!("\n=== Testing Full Sequence with store_state ===");
        let mut cpu_full = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu_full.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        cpu_full.x[10] = base_addr as i64;

        // Store initial state
        for (j, &lane) in initial_state.iter().enumerate() {
            cpu_full
                .mmu
                .store_doubleword(base_addr + (j * 8) as u64, lane)
                .expect("Failed to store initial lane");
        }

        // Generate and execute the full sequence
        let instruction = KECCAK256 {
            address: 0,
            operands: FormatR {
                rs1: 10,
                rs2: 0,
                rd: 0,
            },
            virtual_sequence_remaining: None,
        };

        let full_sequence = instruction.virtual_sequence();
        println!(
            "Generated full sequence with {} instructions",
            full_sequence.len()
        );

        for instr in &full_sequence {
            execute_instruction(&mut cpu_full, instr);
        }

        // Check memory after full execution
        let final_state = read_state_from_memory(&mut cpu_full, base_addr);
        println!("Final state in memory:");
        print_state_hex(&final_state);

        // Expected final state from XKCP
        let expected_final = [
            0xF1258F7940E1DDE7,
            0x84D5CCF933C0478A,
            0xD598261EA65AA9EE,
            0xBD1547306F80494D,
            0x8B284E056253D057,
            0xFF97A42D7F8E6FD4,
            0x90FEE5A0A44647C4,
            0x8C5BDA0CD6192E76,
            0xAD30A6F71B19059C,
            0x30935AB7D08FFC64,
            0xEB5AA93F2317D635,
            0xA9A6E6260D712103,
            0x81A57C16DBCF555F,
            0x43B831CD0347C826,
            0x01F22F1A11A5569F,
            0x05E5635A21D9AE61,
            0x64BEFEF28CC970F2,
            0x613670957BC46611,
            0xB87C5A554FD00ECB,
            0x8C3EE88A1CCF32C8,
            0x940C7922AE3A2614,
            0x1841F924A2C509E4,
            0x16F53526E70465C2,
            0x75F644E97F30A13B,
            0xEAF1FF7B5CECA249,
        ];

        // Check if we got the correct result
        let mut all_correct = true;
        for (i, (&expected, &actual)) in expected_final.iter().zip(final_state.iter()).enumerate() {
            if expected != actual {
                println!(
                    "Lane {} mismatch: expected 0x{:016x}, got 0x{:016x}",
                    i, expected, actual
                );
                all_correct = false;
            }
        }

        if all_correct {
            println!("✅ Full virtual sequence produces CORRECT final result!");
        } else {
            println!("❌ Full virtual sequence produces INCORRECT final result");
        }

        println!("\n✅ Granular test identified where the issue occurs");
    }

    #[test]
    fn test_debug_rotation_issue() {
        println!("=== Debugging Keccak256 Rotation Issue ===");

        // Test our rotation implementation directly
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        // Test value that should reveal rotation issues
        let test_value: u64 = 0x123456789ABCDEF0;
        cpu.x[10] = test_value as i64;

        // Get virtual register mapping
        let mut vr = [0; super::NEEDED_REGISTERS];
        for i in 0..super::NEEDED_REGISTERS {
            vr[i] = virtual_register_index(i as u64) as usize;
        }

        // Test rotation by 1 (common in Keccak theta step)
        println!("\nTesting ROTL by 1:");
        println!("Input: 0x{:016x}", test_value);
        println!("Expected: 0x{:016x}", test_value.rotate_left(1));

        // Build a minimal sequence that just does one rotation
        let builder = super::Keccak256SequenceBuilder::new(0, vr, 10, 0);

        // We need to access the builder's methods, but they're private
        // So let's test via the full sequence and examine the first rotation

        // Instead, let's test the full sequence but with a known input
        let base_addr = DRAM_BASE;
        cpu.x[10] = base_addr as i64;

        // Set up a test state with a pattern that makes rotations obvious
        let test_state = [
            0x0000000000000001u64, // This should rotate to 0x0000000000000002 after ROTL(1)
            0x8000000000000000u64, // This should rotate to 0x0000000000000001 after ROTL(1)
            0xFFFFFFFFFFFFFFFFu64, // This should stay 0xFFFFFFFFFFFFFFFF after ROTL(1)
            0x123456789ABCDEF0u64, // This should rotate to 0x2468ACF13579BDE0 after ROTL(1)
            0x0F0F0F0F0F0F0F0Fu64, // Pattern to check
            0x00FF00FF00FF00FFu64,
            0x0000FFFF0000FFFFu64,
            0x00000000FFFFFFFFu64,
            0xAAAAAAAAAAAAAAAAu64,
            0x5555555555555555u64,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];

        // Store test state to memory
        for (i, &lane) in test_state.iter().enumerate() {
            cpu.mmu
                .store_doubleword(base_addr + (i * 8) as u64, lane)
                .expect("Failed to store test lane");
        }

        // Generate just the theta step which uses rotation
        let instruction = KECCAK256 {
            address: 0,
            operands: FormatR {
                rs1: 10,
                rs2: 0,
                rd: 0,
            },
            virtual_sequence_remaining: None,
        };

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
            execute_instruction(&mut cpu, instr);
            instruction_count += 1;

            // Check if any virtual registers have interesting values
            if instruction_count % 10 == 0 {
                println!("\nAfter {} instructions:", instruction_count);
                for i in 0..10 {
                    let val = cpu.x[vr[i]] as u64;
                    if val != 0 {
                        println!("  vr[{}] = 0x{:016x}", i, val);
                    }
                }
            }
        }

        println!("\n✅ Debug test completed");
    }

    #[test]
    fn test_virtual_sequence_detailed_divergence() {
        println!("=== Finding Exact Divergence Point in Virtual Sequence ===");

        // Set up CPU and memory
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let base_addr = DRAM_BASE;
        cpu.x[10] = base_addr as i64;

        // Initialize with all-zero state (same as XKCP test vectors)
        let initial_state = [0u64; 25];

        // Store initial state to memory
        for (j, &lane) in initial_state.iter().enumerate() {
            cpu.mmu
                .store_doubleword(base_addr + (j * 8) as u64, lane)
                .expect("Failed to store initial lane");
        }

        // Get virtual register mapping
        let mut vr = [0; super::NEEDED_REGISTERS];
        for i in 0..super::NEEDED_REGISTERS {
            vr[i] = virtual_register_index(i as u64) as usize;
        }

        // Test each round and step
        for round in 0..24 {
            for step in &["theta", "rho_and_pi", "chi", "iota"] {
                println!("\n--- Testing Round {} after {} ---", round, step);

                // Create a new CPU for this test
                let mut test_cpu = Cpu::new(Box::new(DefaultTerminal::new()));
                test_cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
                test_cpu.x[10] = base_addr as i64;

                // Store initial state
                for (j, &lane) in initial_state.iter().enumerate() {
                    test_cpu
                        .mmu
                        .store_doubleword(base_addr + (j * 8) as u64, lane)
                        .expect("Failed to store initial lane");
                }

                // Generate sequence up to this step
                let builder = super::Keccak256SequenceBuilder::new(0x1000, vr, 10, 11);
                let sequence = builder.build_up_to_step(round, step);

                println!("Generated {} instructions", sequence.len());

                // Execute the sequence
                for instr in &sequence {
                    execute_instruction(&mut test_cpu, instr);
                }

                // Read virtual registers
                let mut virtual_state = [0u64; 25];
                for i in 0..25 {
                    virtual_state[i] = test_cpu.x[vr[i]] as u64;
                }

                // Compute expected state using reference implementation
                let mut expected_state = initial_state;
                for r in 0..=round {
                    execute_theta(&mut expected_state);
                    if r == round && *step == "theta" {
                        break;
                    }

                    execute_rho_and_pi(&mut expected_state);
                    if r == round && *step == "rho_and_pi" {
                        break;
                    }

                    execute_chi(&mut expected_state);
                    if r == round && *step == "chi" {
                        break;
                    }

                    execute_iota(&mut expected_state, ROUND_CONSTANTS[r as usize]);
                    if r == round && *step == "iota" {
                        break;
                    }
                }

                // Compare states
                let mut all_match = true;
                for i in 0..25 {
                    if virtual_state[i] != expected_state[i] {
                        println!(
                            "MISMATCH at lane {}: virtual={:#018x}, expected={:#018x}",
                            i, virtual_state[i], expected_state[i]
                        );
                        all_match = false;
                    }
                }

                if all_match {
                    println!("✓ All lanes match!");
                } else {
                    println!("\n❌ DIVERGENCE FOUND: Round {} after {}", round, step);
                    println!(
                        "This is the first point where virtual sequence diverges from expected!"
                    );

                    // Print first few mismatched lanes for debugging
                    println!("\nFirst 5 lanes:");
                    for i in 0..5 {
                        println!(
                            "  Lane {}: virtual={:#018x}, expected={:#018x}",
                            i, virtual_state[i], expected_state[i]
                        );
                    }

                    return; // Stop at first divergence
                }
            }
        }
    }

    #[test]
    fn test_debug_iota_instruction_generation() {
        println!("=== Debugging Iota Instruction Generation ===");

        // Get virtual register mapping
        let mut vr = [0; super::NEEDED_REGISTERS];
        for i in 0..super::NEEDED_REGISTERS {
            vr[i] = virtual_register_index(i as u64) as usize;
        }

        // Create a sequence builder and manually call iota for different rounds
        for test_round in 0..3 {
            println!("\n--- Testing round {} iota ---", test_round);

            let mut builder = super::Keccak256SequenceBuilder::new(0x1000, vr, 10, 11);
            builder.round = test_round;

            // Get the initial sequence length
            let initial_len = builder.sequence.len();

            // Call iota
            builder.iota();

            // Get the new instructions
            let iota_instructions = &builder.sequence[initial_len..];

            println!(
                "Generated {} instructions for round {} iota",
                iota_instructions.len(),
                test_round
            );

            // Print the instructions
            for (i, instr) in iota_instructions.iter().enumerate() {
                match instr {
                    RV32IMInstruction::XORI(xori) => {
                        println!(
                            "  {}: XORI rd={}, rs1={}, imm=0x{:x}",
                            i, xori.operands.rd, xori.operands.rs1, xori.operands.imm
                        );
                    }
                    RV32IMInstruction::XOR(xor) => {
                        println!(
                            "  {}: XOR rd={}, rs1={}, rs2={}",
                            i, xor.operands.rd, xor.operands.rs1, xor.operands.rs2
                        );
                    }
                    RV32IMInstruction::LUI(lui) => {
                        println!(
                            "  {}: LUI rd={}, imm=0x{:x}",
                            i, lui.operands.rd, lui.operands.imm
                        );
                    }
                    RV32IMInstruction::ADDI(addi) => {
                        println!(
                            "  {}: ADDI rd={}, rs1={}, imm=0x{:x}",
                            i, addi.operands.rd, addi.operands.rs1, addi.operands.imm
                        );
                    }
                    _ => {
                        println!("  {}: {:?}", i, instr);
                    }
                }
            }

            // Check what constant should be used
            let expected_constant = ROUND_CONSTANTS[test_round as usize];
            println!("Expected constant: 0x{:016x}", expected_constant);
        }
    }

    #[test]
    fn test_debug_iota_round_constants() {
        println!("=== Debugging Iota Round Constants ===");

        // Check which round constants fit in 12 bits
        for (round, &constant) in ROUND_CONSTANTS.iter().enumerate() {
            let fits_in_12_bits = constant as i64 >= -2048 && constant as i64 <= 2047;
            println!(
                "Round {}: 0x{:016x} - fits in 12 bits: {}",
                round, constant, fits_in_12_bits
            );

            // For round 1, let's debug more
            if round == 1 {
                println!("  Round 1 constant: 0x{:016x} = {}", constant, constant);
                println!("  As i64: {}", constant as i64);
                println!(
                    "  Fits check: {} >= -2048 && {} <= 2047 = {}",
                    constant as i64, constant as i64, fits_in_12_bits
                );

                // Test what happens with XORI
                let truncated = (constant as i32) as u64;
                println!("  Truncated to 32-bit: 0x{:016x}", truncated);
                let sign_extended_12bit = ((constant & 0xFFF) as i32 as i64) as u64;
                println!("  12-bit sign extended: 0x{:016x}", sign_extended_12bit);
            }
        }

        // Test our load_32bit_immediate logic for round 1 constant
        println!("\n=== Testing load_32bit_immediate for round 1 ===");
        let round1_const = 0x0000000000008082u64;
        let value = round1_const as u32;
        let upper_20 = value >> 12;
        let lower_12 = value & 0xFFF;

        println!("Value: 0x{:08x}", value);
        println!("Upper 20 bits: 0x{:05x}", upper_20);
        println!("Lower 12 bits: 0x{:03x}", lower_12);

        let needs_adjustment = (lower_12 & 0x800) != 0;
        println!(
            "Needs adjustment: {} (bit 11 = {})",
            needs_adjustment,
            (lower_12 & 0x800) != 0
        );

        let adjusted_upper = if needs_adjustment {
            upper_20 + 1
        } else {
            upper_20
        };
        println!("Adjusted upper: 0x{:05x}", adjusted_upper);

        let sign_extended_lower = if lower_12 & 0x800 != 0 {
            (lower_12 | 0xFFFFF000) as i32 as i64
        } else {
            lower_12 as i64
        };
        println!(
            "Sign extended lower: 0x{:016x} = {}",
            sign_extended_lower as u64, sign_extended_lower
        );

        // Simulate LUI + ADDI
        let lui_result = (adjusted_upper as u64) << 12;
        println!("After LUI: 0x{:016x}", lui_result);
        let final_result = (lui_result as i64 + sign_extended_lower) as u64;
        println!("After ADDI: 0x{:016x}", final_result);
        println!("Expected: 0x{:016x}", round1_const);
        println!("Match: {}", final_result == round1_const);
    }

    #[test]
    fn test_trace_iota_operations() {
        println!("=== Tracing Iota Operations in Virtual Sequence ===");

        // Set up CPU and memory
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let base_addr = DRAM_BASE;
        cpu.x[10] = base_addr as i64;

        // Initialize with all-zero state
        let initial_state = [0u64; 25];
        for (j, &lane) in initial_state.iter().enumerate() {
            cpu.mmu
                .store_doubleword(base_addr + (j * 8) as u64, lane)
                .expect("Failed to store initial lane");
        }

        // Get virtual register mapping
        let mut vr = [0; super::NEEDED_REGISTERS];
        for i in 0..super::NEEDED_REGISTERS {
            vr[i] = virtual_register_index(i as u64) as usize;
        }

        // Generate sequence for first 2 rounds
        let builder = super::Keccak256SequenceBuilder::new(0x1000, vr, 10, 11);
        let sequence = builder.build_up_to_step(1, "iota");

        println!(
            "Generated {} instructions for up to round 1 iota",
            sequence.len()
        );

        // Track state of lane 0 through execution
        let mut prev_lane0 = 0u64;
        let mut instruction_count = 0;

        for instr in &sequence {
            execute_instruction(&mut cpu, instr);
            instruction_count += 1;

            let curr_lane0 = cpu.x[vr[0]] as u64;
            if curr_lane0 != prev_lane0 {
                println!(
                    "Instruction {}: Lane 0 changed from 0x{:016x} to 0x{:016x}",
                    instruction_count, prev_lane0, curr_lane0
                );

                // Check if this looks like an iota operation
                let diff = curr_lane0 ^ prev_lane0;
                for (round, &constant) in ROUND_CONSTANTS.iter().enumerate() {
                    if diff == constant {
                        println!(
                            "  -> This is iota with round {} constant: 0x{:016x}",
                            round, constant
                        );
                    }
                }

                prev_lane0 = curr_lane0;
            }
        }

        println!("\nFinal lane 0 value: 0x{:016x}", cpu.x[vr[0]] as u64);
        println!("Expected after round 0: 0x{:016x}", 0x0000000000000001u64);
        println!("Expected after round 1: 0x{:016x}", 0x0000000000008083u64);
    }

    #[test]
    fn test_trace_round1_iota_execution() {
        println!("=== Tracing Round 1 Iota Execution ===");

        // Set up CPU
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        // Get virtual register mapping
        let mut vr = [0; super::NEEDED_REGISTERS];
        for i in 0..super::NEEDED_REGISTERS {
            vr[i] = virtual_register_index(i as u64) as usize;
        }

        // Set up initial state - lane 0 should have value 0x1 (from round 0)
        cpu.x[vr[0]] = 0x1;

        // Create builder and generate round 1 iota instructions
        let mut builder = super::Keccak256SequenceBuilder::new(0x1000, vr, 10, 11);
        builder.round = 1;
        let initial_len = builder.sequence.len();
        builder.iota();
        let iota_instructions = &builder.sequence[initial_len..];

        println!(
            "Generated {} instructions for round 1 iota",
            iota_instructions.len()
        );
        println!("Initial lane 0 value: 0x{:016x}", cpu.x[vr[0]]);

        // Execute each instruction and trace the state
        for (i, instr) in iota_instructions.iter().enumerate() {
            println!("\nInstruction {}: {:?}", i, instr);

            // Print relevant register values before execution
            match instr {
                RV32IMInstruction::LUI(lui) => {
                    println!(
                        "  Before: rd (r{}) = 0x{:016x}",
                        lui.operands.rd, cpu.x[lui.operands.rd] as u64
                    );
                }
                RV32IMInstruction::ADDI(addi) => {
                    println!(
                        "  Before: rd (r{}) = 0x{:016x}, rs1 (r{}) = 0x{:016x}",
                        addi.operands.rd,
                        cpu.x[addi.operands.rd] as u64,
                        addi.operands.rs1,
                        cpu.x[addi.operands.rs1] as u64
                    );
                }
                RV32IMInstruction::XOR(xor) => {
                    println!("  Before: rd (r{}) = 0x{:016x}, rs1 (r{}) = 0x{:016x}, rs2 (r{}) = 0x{:016x}", 
                             xor.operands.rd, cpu.x[xor.operands.rd] as u64,
                             xor.operands.rs1, cpu.x[xor.operands.rs1] as u64,
                             xor.operands.rs2, cpu.x[xor.operands.rs2] as u64);
                }
                _ => {}
            }

            // Execute the instruction
            execute_instruction(&mut cpu, instr);

            // Print relevant register values after execution
            match instr {
                RV32IMInstruction::LUI(lui) => {
                    println!(
                        "  After: rd (r{}) = 0x{:016x}",
                        lui.operands.rd, cpu.x[lui.operands.rd] as u64
                    );
                }
                RV32IMInstruction::ADDI(addi) => {
                    println!(
                        "  After: rd (r{}) = 0x{:016x}",
                        addi.operands.rd, cpu.x[addi.operands.rd] as u64
                    );
                }
                RV32IMInstruction::XOR(xor) => {
                    println!(
                        "  After: rd (r{}) = 0x{:016x}",
                        xor.operands.rd, cpu.x[xor.operands.rd] as u64
                    );
                }
                _ => {}
            }
        }

        println!("\nFinal lane 0 value: 0x{:016x}", cpu.x[vr[0]] as u64);
        println!("Expected: 0x{:016x}", 0x0000000000008083u64);
    }

    #[test]
    fn test_trace_round3_iota_execution() {
        println!("=== Tracing Round 3 Iota Execution ===");

        // Set up CPU
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        // Get virtual register mapping
        let mut vr = [0; super::NEEDED_REGISTERS];
        for i in 0..super::NEEDED_REGISTERS {
            vr[i] = virtual_register_index(i as u64) as usize;
        }

        // Set up initial state - use the expected value before round 3 iota
        cpu.x[vr[0]] = 0x8838573acdebe243u64 as i64; // Correct value from chi step

        // Create builder and generate round 3 iota instructions
        let mut builder = super::Keccak256SequenceBuilder::new(0x1000, vr, 10, 11);
        builder.round = 3;
        let initial_len = builder.sequence.len();
        builder.iota();
        let iota_instructions = &builder.sequence[initial_len..];

        println!(
            "Generated {} instructions for round 3 iota",
            iota_instructions.len()
        );
        println!("Initial lane 0 value: 0x{:016x}", cpu.x[vr[0]] as u64);
        println!("Round 3 constant: 0x{:016x}", ROUND_CONSTANTS[3]);

        // Execute each instruction and trace the state
        for (i, instr) in iota_instructions.iter().enumerate() {
            println!("\nInstruction {}: ", i);

            // Print the instruction
            match instr {
                RV32IMInstruction::LUI(lui) => {
                    println!(
                        "  LUI rd={}, imm=0x{:x} (shifted: 0x{:x})",
                        lui.operands.rd,
                        lui.operands.imm >> 12,
                        lui.operands.imm
                    );
                }
                RV32IMInstruction::ADDI(addi) => {
                    println!(
                        "  ADDI rd={}, rs1={}, imm=0x{:x} ({})",
                        addi.operands.rd,
                        addi.operands.rs1,
                        addi.operands.imm,
                        addi.operands.imm as i64
                    );
                }
                RV32IMInstruction::SLLI(slli) => {
                    println!(
                        "  SLLI rd={}, rs1={}, imm={}",
                        slli.operands.rd, slli.operands.rs1, slli.operands.imm
                    );
                }
                RV32IMInstruction::OR(or) => {
                    println!(
                        "  OR rd={}, rs1={}, rs2={}",
                        or.operands.rd, or.operands.rs1, or.operands.rs2
                    );
                }
                RV32IMInstruction::XOR(xor) => {
                    println!(
                        "  XOR rd={}, rs1={}, rs2={}",
                        xor.operands.rd, xor.operands.rs1, xor.operands.rs2
                    );
                }
                _ => {
                    println!("  {:?}", instr);
                }
            }

            // Execute and show result
            let before_val = match instr {
                RV32IMInstruction::LUI(lui) => cpu.x[lui.operands.rd] as u64,
                RV32IMInstruction::ADDI(addi) => cpu.x[addi.operands.rd] as u64,
                RV32IMInstruction::SLLI(slli) => cpu.x[slli.operands.rd] as u64,
                RV32IMInstruction::OR(or) => cpu.x[or.operands.rd] as u64,
                RV32IMInstruction::XOR(xor) => cpu.x[xor.operands.rd] as u64,
                _ => 0,
            };

            execute_instruction(&mut cpu, instr);

            let after_val = match instr {
                RV32IMInstruction::LUI(lui) => cpu.x[lui.operands.rd] as u64,
                RV32IMInstruction::ADDI(addi) => cpu.x[addi.operands.rd] as u64,
                RV32IMInstruction::SLLI(slli) => cpu.x[slli.operands.rd] as u64,
                RV32IMInstruction::OR(or) => cpu.x[or.operands.rd] as u64,
                RV32IMInstruction::XOR(xor) => cpu.x[xor.operands.rd] as u64,
                _ => 0,
            };

            println!("  Result: 0x{:016x} -> 0x{:016x}", before_val, after_val);
        }

        println!("\nFinal lane 0 value: 0x{:016x}", cpu.x[vr[0]] as u64);
        println!("Expected: 0x{:016x}", 0x0838573a4deb6243u64);
    }

    // Helper function to execute any instruction
    fn execute_instruction(cpu: &mut Cpu, instr: &RV32IMInstruction) {
        match instr {
            RV32IMInstruction::LD(ld) => {
                let mut ram_access = RAMRead::default();
                ld.execute(cpu, &mut ram_access);
            }
            RV32IMInstruction::SD(sd) => {
                let mut ram_access = RAMWrite::default();
                sd.execute(cpu, &mut ram_access);
            }
            RV32IMInstruction::XOR(xor) => {
                let mut ram_access = ();
                xor.execute(cpu, &mut ram_access);
            }
            RV32IMInstruction::XORI(xori) => {
                let mut ram_access = ();
                xori.execute(cpu, &mut ram_access);
            }
            RV32IMInstruction::AND(and) => {
                let mut ram_access = ();
                and.execute(cpu, &mut ram_access);
            }
            RV32IMInstruction::ANDI(andi) => {
                let mut ram_access = ();
                andi.execute(cpu, &mut ram_access);
            }
            RV32IMInstruction::OR(or) => {
                let mut ram_access = ();
                or.execute(cpu, &mut ram_access);
            }
            RV32IMInstruction::SLLI(slli) => {
                let mut ram_access = ();
                slli.execute(cpu, &mut ram_access);
            }
            RV32IMInstruction::SRLI(srli) => {
                let mut ram_access = ();
                srli.execute(cpu, &mut ram_access);
            }
            RV32IMInstruction::VirtualROTRI(rotri) => {
                let mut ram_access = ();
                rotri.execute(cpu, &mut ram_access);
            }
            RV32IMInstruction::VirtualMULI(muli) => {
                let mut ram_access = ();
                muli.execute(cpu, &mut ram_access);
            }
            RV32IMInstruction::ADDI(addi) => {
                let mut ram_access = ();
                addi.execute(cpu, &mut ram_access);
            }
            RV32IMInstruction::LUI(lui) => {
                let mut ram_access = ();
                lui.execute(cpu, &mut ram_access);
            }
            _ => panic!(
                "Unexpected instruction type in Keccak256 virtual sequence: {:?}",
                instr
            ),
        }
    }

    fn read_state_from_memory(cpu: &mut Cpu, base_addr: u64) -> [u64; 25] {
        let mut state = [0u64; 25];
        for (j, lane) in state.iter_mut().enumerate() {
            *lane = cpu
                .mmu
                .load_doubleword(base_addr + (j * 8) as u64)
                .expect("Failed to load lane")
                .0;
        }
        state
    }

    fn print_state_hex(state: &[u64; 25]) {
        for (i, &lane) in state.iter().enumerate() {
            if i % 5 == 0 {
                println!();
            }
            print!("0x{:016x} ", lane);
        }
        println!();
    }

    struct TestCase {
        input: [u64; 25],
        expected: [u64; 25],
        description: &'static str,
    }

    #[test]
    fn test_find_round3_chi_value() {
        println!("=== Finding Round 3 Chi Output Value ===");

        // Execute reference implementation up to round 3 chi
        let mut state = [0u64; 25];

        // Execute rounds 0, 1, 2, and up to round 3 chi
        for round in 0..3 {
            execute_theta(&mut state);
            execute_rho_and_pi(&mut state);
            execute_chi(&mut state);
            execute_iota(&mut state, ROUND_CONSTANTS[round]);
        }

        // Now execute round 3 up to chi
        execute_theta(&mut state);
        execute_rho_and_pi(&mut state);
        execute_chi(&mut state);

        println!("State after round 3 chi:");
        println!("Lane 0: 0x{:016x}", state[0]);

        // Apply round 3 iota
        execute_iota(&mut state, ROUND_CONSTANTS[3]);
        println!("State after round 3 iota:");
        println!("Lane 0: 0x{:016x}", state[0]);
        println!("Expected: 0x0838573a4deb6243");
    }

    #[test]
    fn test_debug_load_64bit_immediate() {
        println!("=== Debugging load_64bit_immediate for Round 3 Constant ===");

        // Set up CPU
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        // Get virtual register mapping
        let mut vr = [0; super::NEEDED_REGISTERS];
        for i in 0..super::NEEDED_REGISTERS {
            vr[i] = virtual_register_index(i as u64) as usize;
        }

        // Create builder
        let mut builder = super::Keccak256SequenceBuilder::new(0x1000, vr, 10, 11);

        // Test loading the round 3 constant
        let round3_constant = 0x8000000080008000u64;
        println!("Loading constant: 0x{:016x}", round3_constant);

        // Manually trace what load_64bit_immediate should do
        let bits_63_32 = (round3_constant >> 32) as u32;
        let bits_31_0 = round3_constant as u32;
        println!("Upper 32 bits: 0x{:08x}", bits_63_32);
        println!("Lower 32 bits: 0x{:08x}", bits_31_0);

        // Load the constant
        let initial_len = builder.sequence.len();
        builder.load_64bit_immediate(round3_constant, vr[65]);
        let load_instructions = &builder.sequence[initial_len..];

        println!("\nGenerated {} instructions:", load_instructions.len());

        // Execute and trace each instruction
        for (i, instr) in load_instructions.iter().enumerate() {
            println!("\nInstruction {}: ", i);
            match instr {
                RV32IMInstruction::XOR(xor) => {
                    println!(
                        "  XOR rd={}, rs1={}, rs2={}",
                        xor.operands.rd, xor.operands.rs1, xor.operands.rs2
                    );
                }
                RV32IMInstruction::ADDI(addi) => {
                    println!(
                        "  ADDI rd={}, rs1={}, imm=0x{:x}",
                        addi.operands.rd, addi.operands.rs1, addi.operands.imm
                    );
                }
                RV32IMInstruction::SLLI(slli) => {
                    println!(
                        "  SLLI rd={}, rs1={}, imm={}",
                        slli.operands.rd, slli.operands.rs1, slli.operands.imm
                    );
                }
                RV32IMInstruction::OR(or) => {
                    println!(
                        "  OR rd={}, rs1={}, rs2={}",
                        or.operands.rd, or.operands.rs1, or.operands.rs2
                    );
                }
                _ => {
                    println!("  {:?}", instr);
                }
            }

            // Execute the instruction
            execute_instruction(&mut cpu, instr);

            // Print relevant register values
            println!("  After execution:");
            if let RV32IMInstruction::OR(_) = instr {
                println!("    vr[65] (r{}): 0x{:016x}", vr[65], cpu.x[vr[65]] as u64);
                println!("    vr[66] (r{}): 0x{:016x}", vr[66], cpu.x[vr[66]] as u64);
            }
            match instr {
                RV32IMInstruction::XOR(xor) => {
                    println!(
                        "    r{}: 0x{:016x}",
                        xor.operands.rd, cpu.x[xor.operands.rd] as u64
                    );
                }
                RV32IMInstruction::ADDI(addi) => {
                    println!(
                        "    r{}: 0x{:016x}",
                        addi.operands.rd, cpu.x[addi.operands.rd] as u64
                    );
                }
                RV32IMInstruction::SLLI(slli) => {
                    println!(
                        "    r{}: 0x{:016x}",
                        slli.operands.rd, cpu.x[slli.operands.rd] as u64
                    );
                }
                RV32IMInstruction::OR(or) => {
                    println!(
                        "    r{}: 0x{:016x}",
                        or.operands.rd, cpu.x[or.operands.rd] as u64
                    );
                }
                _ => {}
            }
        }

        println!("\nFinal value in vr[65]: 0x{:016x}", cpu.x[vr[65]] as u64);
        println!("Expected: 0x{:016x}", round3_constant);
        assert_eq!(
            cpu.x[vr[65]] as u64, round3_constant,
            "load_64bit_immediate failed!"
        );
    }

    #[test]
    fn test_debug_round6_constant() {
        println!("=== Debugging Round 6 Constant Loading ===");

        // The round 6 constant
        let round6_constant = 0x8000000080008081u64;
        println!("Round 6 constant: 0x{:016x}", round6_constant);

        // Break it down
        let bits_63_32 = (round6_constant >> 32) as u32;
        let bits_31_0 = round6_constant as u32;
        println!("Upper 32 bits: 0x{:08x}", bits_63_32);
        println!("Lower 32 bits: 0x{:08x}", bits_31_0);

        // Check if lower 32 bits would trigger sign extension workaround
        let upper_20 = bits_31_0 >> 12;
        let lower_12 = bits_31_0 & 0xFFF;
        println!("\nLower 32-bit breakdown:");
        println!("  Upper 20 bits: 0x{:05x}", upper_20);
        println!("  Lower 12 bits: 0x{:03x}", lower_12);
        println!("  Bit 19 set: {}", (upper_20 & 0x80000) != 0);

        // Test loading it
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        let mut vr = [0; super::NEEDED_REGISTERS];
        for i in 0..super::NEEDED_REGISTERS {
            vr[i] = virtual_register_index(i as u64) as usize;
        }

        let mut builder = super::Keccak256SequenceBuilder::new(0x1000, vr, 10, 11);

        // Load just the lower 32 bits
        let initial_len = builder.sequence.len();
        builder.load_32bit_immediate(bits_31_0, vr[65]);
        let instructions = &builder.sequence[initial_len..];

        println!(
            "\nGenerated {} instructions for lower 32 bits:",
            instructions.len()
        );

        for (i, instr) in instructions.iter().enumerate() {
            println!("Instruction {}: {:?}", i, instr);
            execute_instruction(&mut cpu, instr);
        }

        println!("\nFinal value: 0x{:016x}", cpu.x[vr[65]] as u64);
        println!("Expected: 0x{:016x}", bits_31_0 as u64);

        assert_eq!(
            cpu.x[vr[65]] as u64, bits_31_0 as u64,
            "Failed to load lower 32 bits correctly!"
        );
    }

    #[test]
    fn test_debug_specific_initial_state() {
        println!("=== Debugging Specific Initial State ===");

        // Use the same initial state as test_keccak_state_equivalence
        let mut initial_state = [0u64; 25];
        for i in 0..25 {
            initial_state[i] = (i * 3 + 5) as u64;
        }

        println!("Initial state:");
        print_state_hex(&initial_state);

        // Test direct execution
        let mut state_exec = initial_state;
        execute_keccak_f(&mut state_exec);
        println!("\nExpected result (from execute_keccak_f):");
        print_state_hex(&state_exec);

        // Test using the KECCAK256 instruction exec method
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let base_addr = DRAM_BASE;
        cpu.x[10] = base_addr as i64;

        // Store initial state
        for (i, &lane) in initial_state.iter().enumerate() {
            cpu.mmu
                .store_doubleword(base_addr + (i * 8) as u64, lane)
                .expect("Failed to store initial lane");
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

        instruction.exec(&mut cpu, &mut ());

        // Read result
        let mut result_exec = [0u64; 25];
        for (i, lane) in result_exec.iter_mut().enumerate() {
            *lane = cpu
                .mmu
                .load_doubleword(base_addr + (i * 8) as u64)
                .expect("Failed to load result lane")
                .0;
        }

        println!("\nResult from KECCAK256.exec:");
        print_state_hex(&result_exec);

        // Compare
        let mut matches = true;
        for i in 0..25 {
            if state_exec[i] != result_exec[i] {
                println!(
                    "Mismatch at lane {}: reference 0x{:016x}, exec 0x{:016x}",
                    i, state_exec[i], result_exec[i]
                );
                matches = false;
            }
        }

        if matches {
            println!("\n✅ KECCAK256.exec matches reference implementation");
        } else {
            println!("\n❌ KECCAK256.exec does NOT match reference implementation");
        }

        // Now test the virtual sequence
        let mut cpu_trace = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu_trace.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        cpu_trace.x[10] = base_addr as i64;

        // Store initial state
        for (i, &lane) in initial_state.iter().enumerate() {
            cpu_trace
                .mmu
                .store_doubleword(base_addr + (i * 8) as u64, lane)
                .expect("Failed to store initial lane");
        }

        instruction.trace(&mut cpu_trace, None);

        // Read result
        let mut result_trace = [0u64; 25];
        for (i, lane) in result_trace.iter_mut().enumerate() {
            *lane = cpu_trace
                .mmu
                .load_doubleword(base_addr + (i * 8) as u64)
                .expect("Failed to load result lane")
                .0;
        }

        println!("\nResult from KECCAK256.trace (virtual sequence):");
        print_state_hex(&result_trace);

        // Compare with expected
        matches = true;
        for i in 0..25 {
            if state_exec[i] != result_trace[i] {
                println!(
                    "Mismatch at lane {}: expected 0x{:016x}, got 0x{:016x}",
                    i, state_exec[i], result_trace[i]
                );
                matches = false;
            }
        }

        if matches {
            println!("\n✅ Virtual sequence matches reference implementation");
        } else {
            println!("\n❌ Virtual sequence does NOT match reference implementation");
        }
    }

    #[test]
    fn test_rotation_offsets_indexing() {
        println!("=== Testing Rotation Offsets Indexing ===");

        // According to the Keccak specification, the rotation offset for lane (x,y)
        // should be ROTATION_OFFSETS[y][x], not ROTATION_OFFSETS[x][y]

        // Test with a simple state where we can track the rotations
        let mut state = [0u64; 25];

        // Set specific bits in each lane to track rotations
        // We'll set bit 0 in each lane
        for i in 0..25 {
            state[i] = 1u64;
        }

        // Run the reference rho_and_pi
        let mut ref_state = state;
        execute_rho_and_pi(&mut ref_state);

        println!("After reference rho_and_pi:");
        for y in 0..5 {
            for x in 0..5 {
                let lane = ref_state[x + 5 * y];
                // Find which bit is set (it should be rotated from bit 0)
                let rotation = lane.trailing_zeros();
                println!("  Lane ({},{}): rotated by {} positions", x, y, rotation);
            }
        }

        // Check against expected rotations
        // According to the spec, lane (x,y) should be rotated by ROTATION_OFFSETS[y][x]
        println!("\nExpected rotations (using ROTATION_OFFSETS[y][x]):");
        for y in 0..5 {
            for x in 0..5 {
                println!(
                    "  Lane ({},{}): should rotate by {}",
                    x, y, ROTATION_OFFSETS[y][x]
                );
            }
        }

        // Verify the reference implementation is using the correct indexing
        let mut mismatches = 0;
        for y in 0..5 {
            for x in 0..5 {
                let expected_rotation = ROTATION_OFFSETS[x][y]; // This is what the impl uses
                let lane_after = ref_state[x + 5 * y];
                let actual_rotation = lane_after.trailing_zeros();

                // Note: We need to account for the pi permutation as well
                // The reference impl does both rho and pi together
            }
        }
    }

    #[test]
    fn test_virtual_rotation_implementation() {
        println!("=== Testing Virtual Rotation Implementation ===");

        // Test if the virtual sequence correctly implements 64-bit rotations
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        // Get virtual register mapping
        let mut vr = [0; super::NEEDED_REGISTERS];
        for i in 0..super::NEEDED_REGISTERS {
            vr[i] = virtual_register_index(i as u64) as usize;
        }

        // Test various rotation amounts with different test values
        let test_cases = vec![
            (0x0000000000000001u64, 1, 0x0000000000000002u64), // Simple rotation by 1
            (0x8000000000000000u64, 1, 0x0000000000000001u64), // MSB wraps to LSB
            (0x0123456789ABCDEFu64, 4, 0x123456789ABCDEF0u64), // Rotation by 4
            (0x0123456789ABCDEFu64, 32, 0x89ABCDEF01234567u64), // Rotation by 32 (swap halves)
            (0x0123456789ABCDEFu64, 36, 0x9ABCDEF012345678u64), // Rotation by 36
        ];

        for (i, (input, rotation, expected)) in test_cases.iter().enumerate() {
            println!(
                "\nTest case {}: rotate 0x{:016x} left by {} bits",
                i, input, rotation
            );
            println!("Expected: 0x{:016x}", expected);

            // Set up input value in a virtual register
            cpu.x[vr[0]] = *input as i64;

            // Create a minimal sequence that just does the rotation
            let mut builder = super::Keccak256SequenceBuilder::new(0, vr, 10, 0);
            use crate::instruction::inline_keccak256::Value::Reg;
            builder.rotl64(Reg(vr[0]), *rotation, vr[1]);
            builder.enumerate_sequence();

            // Execute the sequence
            for instr in &builder.sequence {
                execute_instruction(&mut cpu, instr);
            }

            // Check result
            let result = cpu.x[vr[1]] as u64;
            println!("Actual:   0x{:016x}", result);

            if result != *expected {
                println!("❌ MISMATCH!");

                // Debug: show the instructions generated
                println!("Generated instructions:");
                for (j, instr) in builder.sequence.iter().enumerate() {
                    println!("  {}: {:?}", j, instr);
                }
            } else {
                println!("✅ Correct");
            }

            assert_eq!(
                result, *expected,
                "Rotation test case {} failed: rotate 0x{:016x} by {} should be 0x{:016x}, got 0x{:016x}",
                i, input, rotation, expected, result
            );
        }
    }
}
