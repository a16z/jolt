use tracer::emulator::cpu::Cpu;
use tracer::instruction::precompile::PRECOMPILE;
use tracer::instruction::{RISCVInstruction, RV32IMInstruction};
use tracer::{register_precompile, list_registered_precompiles};

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



fn sha2_precompile(instr: &PRECOMPILE, cpu: &mut Cpu, _ram_access: &mut <PRECOMPILE as RISCVInstruction>::RAMAccess) {
    // Load 16 input words from memory at rs1
    let mut input = [0u32; 16];
    for (i, word) in input.iter_mut().enumerate() {
        *word = cpu.mmu
            .load_word(cpu.x[instr.operands.rs1].wrapping_add((i * 4) as i64) as u64)
            .expect("SHA256: Failed to load input word")
            .0;
    }

    // Load 8 initial state words from memory at rs2
    let mut iv = [0u32; 8];
    for (i, word) in iv.iter_mut().enumerate() {
        *word = cpu.mmu
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
fn sha2_builder(_address: u64, _rs1: usize, _rs2: usize) -> Vec<RV32IMInstruction> {
    // Virtual registers used as a scratch space
    let mut vr = [0; 32];
    (0..32).for_each(|i| {
        vr[i] = virtual_register_index(i as u64) as usize;
    });
    let builder = generator::Sha256SequenceBuilder::new(
        _address,
        vr,
        _rs1,
        _rs2,
        false, // not initial - uses custom IV from rs2
    );
    builder.build()
}

// Initialize and register precompiles
fn init_precompiles() -> Result<(), String> {
    // Register XOR with funct7=0x01
    register_precompile(0x01, "SHA2_PRECOMPILE", 
        Box::new(sha2_precompile), 
        Box::new(sha2_builder))?;
    
    Ok(())
}

// Optional: Use ctor for automatic registration when the library loads
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_precompiles() {
        eprintln!("Failed to register precompiles: {}", e);
    }
}

fn main() {
    // Manual registration (if not using ctor)
    // init_precompiles().expect("Failed to register precompiles");
    
    // List all registered precompiles
    println!("Registered precompiles:");
    for (funct7, name) in list_registered_precompiles() {
        println!("  funct7={:#04x}: {}", funct7, name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracer::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
    use tracer::instruction::format::format_r::FormatR;
    use tracer::instruction::{RISCVInstruction, RV32IMInstruction, RV32IMCycle, RISCVTrace};

    const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

    // SHA256 initial hash values (FIPS 180-4)
    const SHA256_INITIAL_STATE: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Expected state after processing "abc" (first block)
    // This is the result of applying SHA256 compression to the padded "abc" message
    const EXPECTED_STATE_AFTER_ABC: [u32; 8] = [
        0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223,
        0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad,
    ];

    #[test]
    fn test_sha256_precompile_compression_exec() {

        // The precompiles might already be registered by the ctor
        let registered_before = list_registered_precompiles();
        
        if registered_before.is_empty() {
            // If not registered yet, register them
            assert!(init_precompiles().is_ok());
        }
        
        // Verify all precompiles are registered
        let registered = list_registered_precompiles();
        assert!(registered.len() >= 1); // At least our 1 precompile

        // Create padded message block for "abc"
        // SHA256 uses 512-bit blocks (16 32-bit words)
        let mut message_block = [0u32; 16];
        // "abc" = 0x61, 0x62, 0x63
        message_block[0] = 0x61626380; // "abc" + 0x80 padding bit (big-endian)
        // message_block[1..14] remain 0
        message_block[15] = 0x00000018; // bit length = 24 bits (3 bytes * 8)

        // Create a PRECOMPILE instruction word with funct7=0x01 (SHA256)
        // Format: [funct7][rs2][rs1][funct3][rd][opcode]
        let instruction_word: u32 = (0x01 << 25) | // funct7 = 0x01 (SHA256 precompile)
                                   (11 << 20)    | // rs2 = 11
                                   (10 << 15)    | // rs1 = 10
                                   (0x2 << 12)   | // funct3 = 0x2 (PRECOMPILE)
                                   (0 << 7)      | // rd = 0
                                   0x3E;           // opcode = 0x3E (0b0111110)

        // Decode the instruction word to create the PRECOMPILE instruction
        let instruction = match RV32IMInstruction::decode(instruction_word, 0x1000) {
            Ok(instr) => instr,
            Err(e) => panic!("Failed to decode PRECOMPILE instruction: {}", e),
        };


        // Set up CPU
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let message_addr = DRAM_BASE;
        let state_addr = DRAM_BASE + 1024; // Separate address for state
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
            RV32IMInstruction::PRECOMPILE(precompile_instr) => {
                precompile_instr.exec(&mut cpu, &mut ());
            }
            _ => panic!("Expected PRECOMPILE instruction, got {:?}", instruction),
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
    fn test_sha256_precompile_compression_trace() {

        // The precompiles might already be registered by the ctor
        let registered_before = list_registered_precompiles();
        
        if registered_before.is_empty() {
            // If not registered yet, register them
            assert!(init_precompiles().is_ok());
        }
        
        // Verify all precompiles are registered
        let registered = list_registered_precompiles();
        assert!(registered.len() >= 1); // At least our 1 precompile

        // Create padded message block for "abc"
        // SHA256 uses 512-bit blocks (16 32-bit words)
        let mut message_block = [0u32; 16];
        // "abc" = 0x61, 0x62, 0x63
        message_block[0] = 0x61626380; // "abc" + 0x80 padding bit (big-endian)
        // message_block[1..14] remain 0
        message_block[15] = 0x00000018; // bit length = 24 bits (3 bytes * 8)

        // Create a PRECOMPILE instruction word with funct7=0x01 (SHA256)
        // Format: [funct7][rs2][rs1][funct3][rd][opcode]
        let instruction_word: u32 = (0x01 << 25) | // funct7 = 0x01 (SHA256 precompile)
                                   (11 << 20)    | // rs2 = 11
                                   (10 << 15)    | // rs1 = 10
                                   (0x2 << 12)   | // funct3 = 0x2 (PRECOMPILE)
                                   (0 << 7)      | // rd = 0
                                   0x3E;           // opcode = 0x3E (0b0111110)

        // Decode the instruction word to create the PRECOMPILE instruction
        let instruction = match RV32IMInstruction::decode(instruction_word, 0x1000) {
            Ok(instr) => instr,
            Err(e) => panic!("Failed to decode PRECOMPILE instruction: {}", e),
        };


        // Set up CPU
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let message_addr = DRAM_BASE;
        let state_addr = DRAM_BASE + 1024; // Separate address for state
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
            RV32IMInstruction::PRECOMPILE(precompile_instr) => {
                precompile_instr.trace(&mut cpu, None);
            }
            _ => panic!("Expected PRECOMPILE instruction, got {:?}", instruction),
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
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let message_addr = DRAM_BASE;
        let state_addr = DRAM_BASE + 1024; // Separate address for state
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
            println!("Word {}: {:#010x} (expected: {:#010x})", i, result[i], EXPECTED_STATE_AFTER_ABC[i]);
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
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let message_addr = DRAM_BASE;
        let state_addr = DRAM_BASE + 1024; // Separate address for state
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
            println!("Word {}: {:#010x} (expected: {:#010x})", i, result[i], EXPECTED_STATE_AFTER_ABC[i]);
            assert_eq!(
                result[i], EXPECTED_STATE_AFTER_ABC[i],
                "Mismatch at word {}: got {:#010x}, expected {:#010x}",
                i, result[i], EXPECTED_STATE_AFTER_ABC[i]
            );
        }
    }
}




// // Example 1: Simple XOR precompile
// fn xor_precompile(instr: &PRECOMPILE, cpu: &mut Cpu, _ram: &mut ()) {
//     let result = cpu.x[instr.operands.rs1] ^ cpu.x[instr.operands.rs2];
//     cpu.x[instr.operands.rd] = result;
//     println!("XOR precompile: x{} = x{} ^ x{} = {:#x} (at {:#x})", 
//              instr.operands.rd, instr.operands.rs1, instr.operands.rs2, result, instr.address);
// }

// // Builder for XOR - returns empty sequence as XOR is atomic
// fn xor_builder(_address: u64, _rs1: usize, _rs2: usize) -> Vec<RV32IMInstruction> {
//     // XOR doesn't need a virtual sequence, it's a single atomic operation
//     Vec::new()
// }

// // Example 2: Bit rotation precompile
// fn rotate_left_precompile(instr: &PRECOMPILE, cpu: &mut Cpu, _ram: &mut ()) {
//     let value = cpu.x[instr.operands.rs1] as u64;
//     let shift = (cpu.x[instr.operands.rs2] & 0x3F) as u32; // Use lower 6 bits for shift amount
//     let result = value.rotate_left(shift);
//     cpu.x[instr.operands.rd] = result as i64;
//     println!("ROTL precompile: x{} = rotl(x{}, {}) = {:#x} (at {:#x})", 
//              instr.operands.rd, instr.operands.rs1, shift, result, instr.address);
// }

// // Builder for ROTL - could decompose into shifts and ORs if needed
// fn rotl_builder(_address: u64, _rs1: usize, _rs2: usize) -> Vec<RV32IMInstruction> {
//     // For simplicity, return empty sequence
//     // In a real implementation, you might decompose this into:
//     // - SLL for left shift
//     // - SRL for right shift of (64-shift) positions  
//     // - OR to combine results
//     Vec::new()
// }

// // Example 3: Population count (popcount) precompile
// fn popcount_precompile(instr: &PRECOMPILE, cpu: &mut Cpu, _ram: &mut ()) {
//     let value = cpu.x[instr.operands.rs1] as u64;
//     let count = value.count_ones() as i64;
//     cpu.x[instr.operands.rd] = count;
//     println!("POPCOUNT precompile: x{} = popcount(x{}) = {} (at {:#x})", 
//              instr.operands.rd, instr.operands.rs1, count, instr.address);
// }

// // Builder for POPCOUNT - could decompose into bit manipulation if needed
// fn popcount_builder(_address: u64, _rs1: usize, _rs2: usize) -> Vec<RV32IMInstruction> {
//     // For simplicity, return empty sequence
//     // In a real implementation, you might generate a sequence of:
//     // - Shift and AND operations to isolate bits
//     // - ADD operations to sum them up
//     Vec::new()
// }

// // Initialize and register precompiles
// fn init_precompiles() -> Result<(), String> {
//     // Register XOR with funct7=0x01
//     register_precompile(0x01, "XOR_PRECOMPILE", 
//         Box::new(xor_precompile), 
//         Box::new(xor_builder))?;
    
//     // Register ROTL with funct7=0x02
//     register_precompile(0x02, "ROTL_PRECOMPILE", 
//         Box::new(rotate_left_precompile),
//         Box::new(rotl_builder))?;
    
//     // Register POPCOUNT with funct7=0x03
//     register_precompile(0x03, "POPCOUNT_PRECOMPILE", 
//         Box::new(popcount_precompile),
//         Box::new(popcount_builder))?;
    
//     Ok(())
// }

// // Optional: Use ctor for automatic registration when the library loads
// #[ctor::ctor]
// fn auto_register() {
//     if let Err(e) = init_precompiles() {
//         eprintln!("Failed to register precompiles: {}", e);
//     }
// }

// fn main() {
//     // Manual registration (if not using ctor)
//     // init_precompiles().expect("Failed to register precompiles");
    
//     // List all registered precompiles
//     println!("Registered precompiles:");
//     for (funct7, name) in list_registered_precompiles() {
//         println!("  funct7={:#04x}: {}", funct7, name);
//     }
    
//     // In actual usage, these precompiles would be invoked when the emulator
//     // encounters PRECOMPILE instructions with matching funct7 values:
//     //
//     // Assembly example:
//     // .word 0x02A382BE  # PRECOMPILE x5, x7, x10 with funct7=0x01 (XOR)
//     //                   # Binary: [0000001][01010][00111][010][00101][0111110]
//     //                   # funct7=0x01, rs2=10, rs1=7, funct3=2, rd=5, opcode=0x3E
    
//     println!("\nTo use in assembly:");
//     println!("  XOR:      .word 0x02A382BE  # funct7=0x01");
//     println!("  ROTL:     .word 0x04A382BE  # funct7=0x02");
//     println!("  POPCOUNT: .word 0x06A382BE  # funct7=0x03");
    
//     println!("\nNote: The builder functions can return virtual instruction sequences");
//     println!("that decompose complex operations into simpler RISC-V instructions.");
//     println!("This is useful for verification and analysis purposes.");
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
    
//     #[test]
//     fn test_precompile_registration() {
//         // The precompiles might already be registered by the ctor
//         let registered_before = list_registered_precompiles();
        
//         if registered_before.is_empty() {
//             // If not registered yet, register them
//             assert!(init_precompiles().is_ok());
//         }
        
//         // Verify all precompiles are registered
//         let registered = list_registered_precompiles();
//         assert!(registered.len() >= 3); // At least our 3 precompiles
        
//         // Check specific registrations
//         let mut found_xor = false;
//         let mut found_rotl = false;
//         let mut found_popcount = false;
        
//         for (funct7, name) in registered {
//             match funct7 {
//                 0x01 => {
//                     assert_eq!(name, "XOR_PRECOMPILE");
//                     found_xor = true;
//                 }
//                 0x02 => {
//                     assert_eq!(name, "ROTL_PRECOMPILE");
//                     found_rotl = true;
//                 }
//                 0x03 => {
//                     assert_eq!(name, "POPCOUNT_PRECOMPILE");
//                     found_popcount = true;
//                 }
//                 _ => {} // Ignore other precompiles that might be registered
//             }
//         }
        
//         assert!(found_xor && found_rotl && found_popcount);
//     }
    
//     #[test]
//     fn test_duplicate_registration() {
//         // Try to register the same funct7 twice (this should always fail)
//         let result = register_precompile(0x01, "DUPLICATE", Box::new(|_, _, _| {}), Box::new(|_address, _rs1, _rs2| Vec::new()));
//         assert!(result.is_err());
//         assert!(result.unwrap_err().contains("already registered"));
//     }
// } 