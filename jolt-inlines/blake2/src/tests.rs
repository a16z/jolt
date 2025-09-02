#[cfg(test)]
mod tests {

    use super::TestVectors;
    use crate::test_utils::blake2_verify;

    #[test]
    fn test_blake2_permutation_default() {
        let (mut state, message, expected_state, counter, is_final) = TestVectors::get_default_test();
        state[0] ^= 0x01010000 ^ (0u64 << 8) ^ 64u64;
        blake2_verify::assert_exec_trace_equiv(&state, &message, counter, is_final, &expected_state);
    }
}

pub struct TestVectors;
impl TestVectors {
    /// Get the default test case
    pub fn get_default_test() -> (
        [u64; 8],  // initial state
        [u64; 16], // message block
        [u64; 8],  // expected state
        u64, // counter
        bool, // is_final
    ) {
        // Initial state - Blake2b initialization vector
        let state = [
            0x6a09e667f3bcc908,
            0xbb67ae8584caa73b,
            0x3c6ef372fe94f82b,
            0xa54ff53a5f1d36f1,
            0x510e527fade682d1,
            0x9b05688c2b3e6c1f,
            0x1f83d9abfb41bd6b,
            0x5be0cd19137e2179,
        ];

        // Message block with "abc" in little-endian
        let mut message = [0u64; 16];
        message[0] = 0x0000000000636261u64; // "abc"

        // Expected state after Blake2b compression
        let expected_state = [
            0x0D4D1C983FA580BAu64, // BA 80 A5 3F 98 1C 4D 0D (little-endian)
            0xE9F6129FB697276Au64, // 6A 27 97 B6 9F 12 F6 E9
            0xB7C45A68142F214Cu64, // 4C 21 2F 14 68 5A C4 B7
            0xD1A2FFDB6FBB124Bu64, // 4B 12 BB 6F DB FF A2 D1
            0x2D79AB2A39C5877Du64, // 7D 87 C5 39 2A AB 79 2D
            0x95CC3345DED552C2u64, // C2 52 D5 DE 45 33 CC 95
            0x5A92F1DBA88AD318u64, // 18 D3 8A A8 DB F1 92 5A
            0x239900D4ED8623B9u64, // B9 23 86 ED D4 00 99 23
        ];

        (state, message, expected_state, 3u64, true)
    }
}



//     use super::*;
//     use crate::test_utils::*;
//     use tracer::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
//     use tracer::instruction::format::format_inline::FormatInline;
//     use tracer::instruction::{inline::INLINE, RISCVInstruction, RISCVTrace};

//     const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

//     // Test constants from RFC 7693 Appendix A (Blake2b with "abc")
//     const INITIAL_STATE: [u64; HASH_STATE_SIZE] = [
//         0x6a09e667f3bcc908,
//         0xbb67ae8584caa73b,
//         0x3c6ef372fe94f82b,
//         0xa54ff53a5f1d36f1,
//         0x510e527fade682d1,
//         0x9b05688c2b3e6c1f,
//         0x1f83d9abfb41bd6b,
//         0x5be0cd19137e2179,
//     ];

//     const EXPECTED_STATE: [u64; HASH_STATE_SIZE] = [
//         0x0D4D1C983FA580BAu64, // BA 80 A5 3F 98 1C 4D 0D (little-endian)
//         0xE9F6129FB697276Au64, // 6A 27 97 B6 9F 12 F6 E9
//         0xB7C45A68142F214Cu64, // 4C 21 2F 14 68 5A C4 B7
//         0xD1A2FFDB6FBB124Bu64, // 4B 12 BB 6F DB FF A2 D1
//         0x2D79AB2A39C5877Du64, // 7D 87 C5 39 2A AB 79 2D
//         0x95CC3345DED552C2u64, // C2 52 D5 DE 45 33 CC 95
//         0x5A92F1DBA88AD318u64, // 18 D3 8A A8 DB F1 92 5A
//         0x239900D4ED8623B9u64, // B9 23 86 ED D4 00 99 23
//     ];

//     fn get_pre_post_states() -> ([u64; HASH_STATE_SIZE], [u64; HASH_STATE_SIZE]) {
//         (INITIAL_STATE, EXPECTED_STATE)
//     }

//     /// Test macro to reduce repetitive setup and verification
//     macro_rules! test_blake2 {
//         ($test_name:ident, $exec_block:expr) => {
//             #[test]
//             fn $test_name() {
//                 let (mut initial_state, expected_state) = get_pre_post_states();
//                 // Apply Blake2b parameter block: h[0] ^= 0x01010000 ^ (kk << 8) ^ nn
//                 initial_state[0] ^= 0x01010000 ^ (0u64 << 8) ^ 64u64;

//                 // Message block with "abc" in little-endian
//                 let mut message_block = [0u64; MESSAGE_BLOCK_SIZE];
//                 message_block[0] = 0x0000000000636261u64; // "abc"

//                 let (counter, is_final) = (3u64, true);

//                 let instruction = INLINE {
//                     address: 0,
//                     operands: FormatInline {
//                         rs1: 10, // Points to state
//                         rs2: 11, // Points to message block + counter + final flag
//                         rs3: 0,
//                     },
//                     // BLAKE2 inline opcode values
//                     opcode: 0x2B, // custom-1 opcode
//                     funct3: 0x00,
//                     funct7: 0x00, // Blake2 specific encoding
//                     inline_sequence_remaining: None,
//                     is_compressed: false,
//                 };

//                 // Set up CPU
//                 let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
//                 cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
//                 let state_addr = DRAM_BASE;
//                 let message_addr = DRAM_BASE + 1024; // Separate address for message block
//                 cpu.x[10] = state_addr as i64; // rs1 points to state
//                 cpu.x[11] = message_addr as i64; // rs2 points to message block

//                 // Store initial state (8 words) at rs1
//                 store_words_to_memory(&mut cpu, state_addr, &initial_state)
//                     .expect("Failed to store initial state");
//                 // Store message block (16 words) at rs2
//                 store_words_to_memory(&mut cpu, message_addr, &message_block)
//                     .expect("Failed to store message block");
//                 // Store counter after message block
//                 store_words_to_memory(&mut cpu, message_addr + 128, &[counter])
//                     .expect("Failed to store counter");
//                 // Store final flag after counter
//                 store_words_to_memory(
//                     &mut cpu,
//                     message_addr + 136,
//                     &[if is_final { 1 } else { 0 }],
//                 )
//                 .expect("Failed to store final flag");

//                 // Execute the instruction
//                 $exec_block(&instruction, &mut cpu);

//                 // Verify results (Blake2b compression outputs 8 words)
//                 let mut result = [0u64; HASH_STATE_SIZE];
//                 for i in 0..HASH_STATE_SIZE {
//                     let addr = state_addr + (i * 8) as u64;
//                     result[i] = cpu.mmu.load_doubleword(addr).unwrap().0;
//                     assert_eq!(
//                         result[i], expected_state[i],
//                         "Mismatch at word {}: got {:#x}, expected {:#x}",
//                         i, result[i], expected_state[i]
//                     );
//                 }
//             }
//         };
//     }

//     test_blake2!(
//         test_exec_correctness,
//         |instruction: &INLINE, cpu: &mut Cpu| {
//             instruction.execute(cpu, &mut ());
//         }
//     );

//     test_blake2!(
//         test_trace_correctness,
//         |instruction: &INLINE, cpu: &mut Cpu| {
//             instruction.trace(cpu, None);
//         }
//     );

//     /// Test using the harness helper functions
//     #[test]
//     fn test_blake2_with_harness() {
//         let (mut initial_state, expected_state) = get_pre_post_states();
//         // Apply Blake2b parameter block
//         initial_state[0] ^= 0x01010000 ^ (0u64 << 8) ^ 64u64;

//         // Message block with "abc"
//         let mut message_block = [0u64; MESSAGE_BLOCK_SIZE];
//         message_block[0] = 0x0000000000636261u64; // "abc"

//         blake2_verify::assert_exec_trace_equiv(
//             &initial_state,
//             &message_block,
//             3u64,
//             true,
//             &expected_state,
//         );
//     }

//     /// Test edge cases
//     #[test] 
//     fn test_blake2_edge_cases() {
//         println!("\n=== Testing BLAKE2 edge cases ===");

//         // Edge case 1: Empty message
//         {
//             let mut initial_state = INITIAL_STATE;
//             initial_state[0] ^= 0x01010000 ^ 64u64; // Blake2b-512 parameters
            
//             let empty_message = [0u64; MESSAGE_BLOCK_SIZE];
            
//             // Expected state for empty message (needs proper expected values)
//             // For now, just ensure it doesn't crash
//             let mut harness = Blake2CpuHarness::new();
//             harness.load_blake2_data(&initial_state, &empty_message, 0, true);
            
//             let instruction = Blake2CpuHarness::instruction();
//             instruction.execute(&mut harness.harness.cpu, &mut ());
            
//             let _result = harness.read_state();
//             println!("  ✓ Empty message test passed");
//         }

//         // Edge case 2: Maximum counter value
//         {
//             let mut initial_state = INITIAL_STATE;
//             initial_state[0] ^= 0x01010000 ^ 64u64;
            
//             let mut message_block = [0u64; MESSAGE_BLOCK_SIZE];
//             message_block[0] = 0xFFFFFFFFFFFFFFFF;
            
//             let mut harness = Blake2CpuHarness::new();
//             harness.load_blake2_data(&initial_state, &message_block, u64::MAX, false);
            
//             let instruction = Blake2CpuHarness::instruction();
//             instruction.execute(&mut harness.harness.cpu, &mut ());
            
//             let _result = harness.read_state();
//             println!("  ✓ Maximum counter test passed");
//         }

//         // Edge case 3: All ones message
//         {
//             let mut initial_state = INITIAL_STATE;
//             initial_state[0] ^= 0x01010000 ^ 64u64;
            
//             let all_ones = [0xFFFFFFFFFFFFFFFF; MESSAGE_BLOCK_SIZE];
            
//             let mut harness = Blake2CpuHarness::new();
//             harness.load_blake2_data(&initial_state, &all_ones, 128, true);
            
//             let instruction = Blake2CpuHarness::instruction();
//             instruction.execute(&mut harness.harness.cpu, &mut ());
            
//             let _result = harness.read_state();
//             println!("  ✓ All ones message test passed");
//         }

//         println!("\n✅ All edge cases passed!\n");
//     }

//     /// Test multiple rounds
//     #[test]
//     fn test_blake2_multiple_rounds() {
//         let mut state = INITIAL_STATE;
//         state[0] ^= 0x01010000 ^ 64u64; // Blake2b-512 parameters
        
//         // Simulate multiple compression rounds
//         for round in 0..3 {
//             let mut message = [0u64; MESSAGE_BLOCK_SIZE];
//             message[0] = round as u64;
            
//             let is_final = round == 2;
//             let counter = (round + 1) * 128;
            
//             let mut harness = Blake2CpuHarness::new();
//             harness.load_blake2_data(&state, &message, counter as u64, is_final);
            
//             let instruction = Blake2CpuHarness::instruction();
//             instruction.execute(&mut harness.harness.cpu, &mut ());
            
//             state = harness.read_state();
//         }
        
//         // State should be modified after multiple rounds
//         assert_ne!(state, INITIAL_STATE, "State should change after compression rounds");
//     }
// }