#![cfg_attr(not(feature = "host"), no_std)]

pub mod sdk;
pub use sdk::*;

// Only include inline registration when compiling for host
#[cfg(feature = "host")]
mod exec;
#[cfg(feature = "host")]
mod trace_generator;

#[cfg(feature = "host")]
use tracer::{register_inline, list_registered_inlines};

// Initialize and register inlines
#[cfg(feature = "host")]
pub fn init_inlines() -> Result<(), String> {
    // Register SHA256 with funct3=0x00 and funct7=0x00 (matching the SDK's assembly instruction)
    register_inline(
        0x00,
        0x00,
        "SHA256_INLINE",
        std::boxed::Box::new(exec::sha2_exec),
        std::boxed::Box::new(trace_generator::sha2_virtual_sequence_builder),
    )?;

    // Register SHA256 with funct3=0x01 and funct7=0x00 (matching the SDK's assembly instruction)
    register_inline(
        0x01,
        0x00,
        "SHA256_INIT_INLINE",
        std::boxed::Box::new(exec::sha2_init_exec),
        std::boxed::Box::new(trace_generator::sha2_init_virtual_sequence_builder),
    )?;

    Ok(())
}

// Automatic registration when the library is loaded
#[cfg(feature = "host")]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register SHA256 inlines: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common;
    use tracer::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, cpu::Xlen};
    use tracer::instruction::format::format_r::FormatR;
    use tracer::instruction::{RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction};

    const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024 * 10; // 10MB to accommodate heap area

    // SHA256 initial hash values (FIPS 180-4)
    const SHA256_INITIAL_STATE: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Expected state after processing "abc" (first block)
    // This is the result of applying SHA256 compression to the padded "abc" message
    const EXPECTED_STATE_AFTER_ABC: [u32; 8] = [
        0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
        0xf20015ad,
    ];

    // Helper function to set up CPU with JoltDevice for tests
    fn setup_test_cpu() -> Cpu {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        cpu.update_xlen(Xlen::Bit32);
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        
        // Set up JoltDevice - this is required for memory operations
        let memory_config = common::jolt_device::MemoryConfig {
            bytecode_size: Some(1024 * 1024), // 1MB for bytecode
            ..Default::default()
        };
        let jolt_device = common::jolt_device::JoltDevice::new(&memory_config);
        cpu.get_mut_mmu().jolt_device = Some(jolt_device);
        
        cpu
    }

    // Helper function to get safe memory addresses in the heap area
    fn get_test_addresses() -> (u64, u64) {
        // Use addresses in the heap area (after stack_end which is at 0x80100000)
        let message_addr = 0x80110000;  // In heap area, after stack_end
        let state_addr = 0x80111000;    // 4KB after message_addr
        (message_addr, state_addr)
    }

    #[test]
    fn test_sha256_inline_compression_exec() {
        // The inlines might already be registered by the ctor
        let registered_before = list_registered_inlines();

        if registered_before.is_empty() {
            // If not registered yet, register them
            assert!(init_inlines().is_ok());
        }

        // Verify all inlines are registered
        let registered = list_registered_inlines();
        assert!(registered.len() >= 1); // At least our 1 inline

        // Create padded message block for "abc"
        // SHA256 uses 512-bit blocks (16 32-bit words)
        let mut message_block = [0u32; 16];
        // "abc" = 0x61, 0x62, 0x63
        message_block[0] = 0x61626380; // "abc" + 0x80 padding bit (big-endian)
                                       // message_block[1..14] remain 0
        message_block[15] = 0x00000018; // bit length = 24 bits (3 bytes * 8)

        let instruction_word: u32 = (0x00 << 25) | // funct7 = 0x00 (SHA256 inline)
                                   (11 << 20)    | // rs2 = 11
                                   (10 << 15)    | // rs1 = 10
                                   (0x0 << 12)   | // funct3 = 0x0 (INLINE)
                                   (0 << 7)      | // rd = 0
                                   0x2B; // opcode = 0x2B

        // Decode the instruction word to create the INLINE instruction
        let instruction = match RV32IMInstruction::decode(instruction_word, 0x1000) {
            Ok(instr) => instr,
            Err(e) => panic!("Failed to decode INLINE instruction: {}", e),
        };

        // Set up CPU
        let mut cpu = setup_test_cpu();
        let (message_addr, state_addr) = get_test_addresses();
        cpu.x[10] = message_addr as i64; // rs1 points to message
        cpu.x[11] = state_addr as i64; // rs2 points to state

        // Store message block (16 words) at rs1
        for (i, &word) in message_block.iter().enumerate() {
            cpu.mmu
                .store_word(message_addr + (i * 4) as u64, word)
                .expect("Failed to store message word");
        }

        // Store initial state (8 words) at rs2
        for (i, &word) in SHA256_INITIAL_STATE.iter().enumerate() {
            cpu.mmu
                .store_word(state_addr + (i * 4) as u64, word)
                .expect("Failed to store initial state");
        }

        // Execute the instruction
        match instruction {
            RV32IMInstruction::INLINE(inline_instr) => {
                inline_instr.exec(&mut cpu, &mut ());
            }
            _ => panic!("Expected INLINE instruction, got {:?}", instruction),
        }

        // Verify results (SHA256 compression outputs 8 words at rs2)
        let mut result = [0u32; 8];
        for i in 0..8 {
            let addr = state_addr + (i * 4) as u64;
            result[i] = cpu.mmu.load_word(addr).unwrap().0;
            assert_eq!(
                result[i], EXPECTED_STATE_AFTER_ABC[i],
                "Mismatch at word {}: got {:#010x}, expected {:#010x}",
                i, result[i], EXPECTED_STATE_AFTER_ABC[i]
            );
        }
    }

    #[test]
    fn test_sha256_inline_compression_trace() {
        // The inlines might already be registered by the ctor
        let registered_before = list_registered_inlines();

        if registered_before.is_empty() {
            // If not registered yet, register them
            assert!(init_inlines().is_ok());
        }

        // Verify all inlines are registered
        let registered = list_registered_inlines();
        assert!(registered.len() >= 1); // At least our 1 inline

        // Create padded message block for "abc"
        // SHA256 uses 512-bit blocks (16 32-bit words)
        let mut message_block = [0u32; 16];
        // "abc" = 0x61, 0x62, 0x63
        message_block[0] = 0x61626380; // "abc" + 0x80 padding bit (big-endian)
                                       // message_block[1..14] remain 0
        message_block[15] = 0x00000018; // bit length = 24 bits (3 bytes * 8)

        let instruction_word: u32 = (0x00 << 25) | // funct7 = 0x00 (SHA256 inline)
                                   (11 << 20)    | // rs2 = 11
                                   (10 << 15)    | // rs1 = 10
                                   (0x0 << 12)   | // funct3 = 0x0 (INLINE)
                                   (0 << 7)      | // rd = 0
                                   0x2B; // opcode = 0x2B

        // Decode the instruction word to create the INLINE instruction
        let instruction = match RV32IMInstruction::decode(instruction_word, 0x1000) {
            Ok(instr) => instr,
            Err(e) => panic!("Failed to decode INLINE instruction: {}", e),
        };

        // Set up CPU
        let mut cpu = setup_test_cpu();
        let (message_addr, state_addr) = get_test_addresses();
        cpu.x[10] = message_addr as i64; // rs1 points to message
        cpu.x[11] = state_addr as i64; // rs2 points to state

        // Store message block (16 words) at rs1
        for (i, &word) in message_block.iter().enumerate() {
            cpu.mmu
                .store_word(message_addr + (i * 4) as u64, word)
                .expect("Failed to store message word");
        }

        // Store initial state (8 words) at rs2
        for (i, &word) in SHA256_INITIAL_STATE.iter().enumerate() {
            cpu.mmu
                .store_word(state_addr + (i * 4) as u64, word)
                .expect("Failed to store initial state");
        }

        // Execute the instruction
        match instruction {
            RV32IMInstruction::INLINE(inline_instr) => {
                inline_instr.trace(&mut cpu, None);
            }
            _ => panic!("Expected INLINE instruction, got {:?}", instruction),
        }

        // Verify results (SHA256 compression outputs 8 words at rs2)
        let mut result = [0u32; 8];
        for i in 0..8 {
            let addr = state_addr + (i * 4) as u64;
            result[i] = cpu.mmu.load_word(addr).unwrap().0;
            assert_eq!(
                result[i], EXPECTED_STATE_AFTER_ABC[i],
                "Mismatch at word {}: got {:#010x}, expected {:#010x}",
                i, result[i], EXPECTED_STATE_AFTER_ABC[i]
            );
        }
    }

    #[test]
    fn test_sha256_direct_exec() {
        use tracer::instruction::inline_sha256::sha256::SHA256;

        // Create padded message block for "abc"
        let mut message_block = [0u32; 16];
        message_block[0] = 0x61626380; // "abc" + 0x80 padding bit (big-endian)
        message_block[15] = 0x00000018; // bit length = 24 bits (3 bytes * 8)

        // Set up CPU
        let mut cpu = setup_test_cpu();
        let (message_addr, state_addr) = get_test_addresses();
        cpu.x[10] = message_addr as i64; // rs1 points to message
        cpu.x[11] = state_addr as i64; // rs2 points to state

        // Store message block (16 words) at rs1
        for (i, &word) in message_block.iter().enumerate() {
            cpu.mmu
                .store_word(message_addr + (i * 4) as u64, word)
                .expect("Failed to store message word");
        }

        // Store initial state (8 words) at rs2
        for (i, &word) in SHA256_INITIAL_STATE.iter().enumerate() {
            cpu.mmu
                .store_word(state_addr + (i * 4) as u64, word)
                .expect("Failed to store initial state");
        }

        // Create SHA256 instruction directly
        let sha256_instr = SHA256 {
            address: 0x1000,
            operands: FormatR {
                rs1: 10,
                rs2: 11,
                rd: 0,
            },
            virtual_sequence_remaining: None,
        };

        sha256_instr.execute(&mut cpu, &mut ());

        // Verify results
        let mut result = [0u32; 8];
        for i in 0..8 {
            let addr = state_addr + (i * 4) as u64;
            result[i] = cpu.mmu.load_word(addr).unwrap().0;
            println!(
                "Word {}: {:#010x} (expected: {:#010x})",
                i, result[i], EXPECTED_STATE_AFTER_ABC[i]
            );
            assert_eq!(
                result[i], EXPECTED_STATE_AFTER_ABC[i],
                "Mismatch at word {}: got {:#010x}, expected {:#010x}",
                i, result[i], EXPECTED_STATE_AFTER_ABC[i]
            );
        }
    }

    #[test]
    fn test_sha256_direct_trace() {
        use tracer::instruction::inline_sha256::sha256::SHA256;

        // Create padded message block for "abc"
        let mut message_block = [0u32; 16];
        message_block[0] = 0x61626380; // "abc" + 0x80 padding bit (big-endian)
        message_block[15] = 0x00000018; // bit length = 24 bits (3 bytes * 8)

        // Set up CPU
        let mut cpu = setup_test_cpu();
        let (message_addr, state_addr) = get_test_addresses();
        cpu.x[10] = message_addr as i64; // rs1 points to message
        cpu.x[11] = state_addr as i64; // rs2 points to state

        // Store message block (16 words) at rs1
        for (i, &word) in message_block.iter().enumerate() {
            cpu.mmu
                .store_word(message_addr + (i * 4) as u64, word)
                .expect("Failed to store message word");
        }

        // Store initial state (8 words) at rs2
        for (i, &word) in SHA256_INITIAL_STATE.iter().enumerate() {
            cpu.mmu
                .store_word(state_addr + (i * 4) as u64, word)
                .expect("Failed to store initial state");
        }

        // Create SHA256 instruction directly
        let sha256_instr = SHA256 {
            address: 0x1000,
            operands: FormatR {
                rs1: 10,
                rs2: 11,
                rd: 0,
            },
            virtual_sequence_remaining: None,
        };

        // Trace the instruction
        let mut trace: Vec<RV32IMCycle> = Vec::new();
        sha256_instr.trace(&mut cpu, Some(&mut trace));

        println!("SHA256 direct trace generated {} instructions", trace.len());

        // Verify results
        let mut result = [0u32; 8];
        for i in 0..8 {
            let addr = state_addr + (i * 4) as u64;
            result[i] = cpu.mmu.load_word(addr).unwrap().0;
            println!(
                "Word {}: {:#010x} (expected: {:#010x})",
                i, result[i], EXPECTED_STATE_AFTER_ABC[i]
            );
            assert_eq!(
                result[i], EXPECTED_STATE_AFTER_ABC[i],
                "Mismatch at word {}: got {:#010x}, expected {:#010x}",
                i, result[i], EXPECTED_STATE_AFTER_ABC[i]
            );
        }
    }
}