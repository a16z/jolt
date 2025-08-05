use tracer::emulator::cpu::Cpu;
use tracer::instruction::inline::INLINE;
use tracer::instruction::{RISCVInstruction, RV32IMInstruction};
use tracer::{list_registered_inlines, register_inline};

pub mod generator;

/// SHA-256 initial hash values
pub const BLOCK: [u64; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-256 round constants (K)
pub const K: [u64; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub fn execute_sha256_compression(initial_state: [u32; 8], input: [u32; 16]) -> [u32; 8] {
    let mut a = initial_state[0];
    let mut b = initial_state[1];
    let mut c = initial_state[2];
    let mut d = initial_state[3];
    let mut e = initial_state[4];
    let mut f = initial_state[5];
    let mut g = initial_state[6];
    let mut h = initial_state[7];

    let mut w = [0u32; 64];

    w[..16].copy_from_slice(&input);

    // Calculate word schedule
    for i in 16..64 {
        // σ₁(w[i-2]) + w[i-7] + σ₀(w[i-15]) + w[i-16]
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    // Perform 64 rounds
    for i in 0..64 {
        let ch = (e & f) ^ ((!e) & g);
        let maj = (a & b) ^ (a & c) ^ (b & c);

        let sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22); // Σ₀(a)
        let sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25); // Σ₁(e)

        let t1 = h
            .wrapping_add(sigma1)
            .wrapping_add(ch)
            .wrapping_add(K[i] as u32)
            .wrapping_add(w[i]);
        let t2 = sigma0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    // Final IV addition
    [
        initial_state[0].wrapping_add(a),
        initial_state[1].wrapping_add(b),
        initial_state[2].wrapping_add(c),
        initial_state[3].wrapping_add(d),
        initial_state[4].wrapping_add(e),
        initial_state[5].wrapping_add(f),
        initial_state[6].wrapping_add(g),
        initial_state[7].wrapping_add(h),
    ]
}

fn sha2_exec(
    instr: &INLINE,
    cpu: &mut Cpu,
    _ram_access: &mut <INLINE as RISCVInstruction>::RAMAccess,
) {
    // Load 16 input words from memory at rs1
    let mut input = [0u32; 16];
    for (i, word) in input.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_word(cpu.x[instr.operands.rs1].wrapping_add((i * 4) as i64) as u64)
            .expect("SHA256: Failed to load input word")
            .0;
    }

    // Load 8 initial state words from memory at rs2
    let mut iv = [0u32; 8];
    for (i, word) in iv.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_word(cpu.x[instr.operands.rs2].wrapping_add((i * 4) as i64) as u64)
            .expect("SHA256: Failed to load initial state")
            .0;
    }

    // Execute compression and store result at rs2
    let result = execute_sha256_compression(iv, input);
    for (i, &word) in result.iter().enumerate() {
        cpu.mmu
            .store_word(
                cpu.x[instr.operands.rs2].wrapping_add((i * 4) as i64) as u64,
                word,
            )
            .expect("SHA256: Failed to store result");
    }
}

const VIRTUAL_REGISTER_COUNT: u64 = 32; //  see Section 6.1 of Jolt paper
pub const fn virtual_register_index(index: u64) -> u64 {
    index + VIRTUAL_REGISTER_COUNT
}

// Builder for sha256 - returns empty sequence as XOR is atomic
fn sha2_virtaul_sequence_builder(_address: u64, _rs1: usize, _rs2: usize) -> Vec<RV32IMInstruction> {
    // Virtual registers used as a scratch space
    let mut vr = [0; 32];
    (0..32).for_each(|i| {
        vr[i] = virtual_register_index(i as u64) as usize;
    });
    let builder = generator::Sha256SequenceBuilder::new(
        _address, vr, _rs1, _rs2, false, // not initial - uses custom IV from rs2
    );
    builder.build()
}

// Initialize and register inlines
pub fn init_inlines() -> Result<(), String> {
    // Register SHA256 with funct7=0x00
    register_inline(
        0x00,
        0x0,
        "SHA256_INLINE",
        Box::new(sha2_exec),
        Box::new(sha2_virtaul_sequence_builder),
    )?;

    // Also register with funct7=0x01 for compatibility
    register_inline(
        0x01,
        0x0,
        "SHA2_INLINE",
        Box::new(sha2_exec),
        Box::new(sha2_virtaul_sequence_builder),
    )?;

    Ok(())
}

// Optional: Use ctor for automatic registration when the library loads
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register inlines: {}", e);
    }
}

fn main() {
    // Manual registration (if not using ctor)
    // init_inlines().expect("Failed to register inlines");

    // List all registered inlines
    println!("Registered inlines:");
    for (key, name) in list_registered_inlines() {
        println!(
            "  funct3={:#04x}: -- funct7={:#04x}: {}",
            key.0, key.1, name
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common;
    use tracer::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
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