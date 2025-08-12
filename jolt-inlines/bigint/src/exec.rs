use tracer::{
    emulator::cpu::Cpu,
    instruction::{inline::INLINE, RISCVInstruction},
};

/// Execute 256-bit × 256-bit = 512-bit multiplication
/// Input: 4 u64 limbs for each operand (little-endian)
/// Output: 8 u64 limbs for the product (little-endian)
pub fn execute_bigint256_mul(a: [u64; 4], b: [u64; 4]) -> [u64; 8] {
    let mut result = [0u64; 8];

    // Schoolbook multiplication: compute all partial products
    // For each a[i] * b[j], add the 128-bit product to result[i+j]
    for i in 0..4 {
        for j in 0..4 {
            // Compute 64×64 = 128-bit product
            let product = (a[i] as u128) * (b[j] as u128);
            let lo = product as u64;
            let hi = (product >> 64) as u64;

            // Add to result[i+j] with carry propagation
            let k = i + j;
            
            // Add low part
            let (sum, carry1) = result[k].overflowing_add(lo);
            result[k] = sum;
            
            // Propagate carry through high part and beyond
            let mut carry = carry1 as u64;
            if hi != 0 || carry != 0 {
                // Add high part plus carry from low part
                let (sum_with_hi, carry_hi) = result[k + 1].overflowing_add(hi);
                let (sum_with_carry, carry_carry) = sum_with_hi.overflowing_add(carry);
                result[k + 1] = sum_with_carry;
                carry = (carry_hi as u64) + (carry_carry as u64);
                
                // Continue propagating carry if needed
                let mut idx = k + 2;
                while carry != 0 && idx < 8 {
                    let (sum, c) = result[idx].overflowing_add(carry);
                    result[idx] = sum;
                    carry = c as u64;
                    idx += 1;
                }
            }
        }
    }

    result
}

pub fn bigint_mul_exec(
    instr: &INLINE,
    cpu: &mut Cpu,
    _ram_access: &mut <INLINE as RISCVInstruction>::RAMAccess,
) {
    // Load 4 u64 words from memory at rs1 (first operand)
    let mut a = [0u64; 4];
    for i in 0..4 {
        a[i] = cpu
            .mmu
            .load_doubleword(cpu.x[instr.operands.rs1 as usize].wrapping_add((i * 8) as i64) as u64)
            .expect("BIGINT256_MUL: Failed to load operand A")
            .0;
    }

    // Load 4 u64 words from memory at rs2 (second operand)
    let mut b = [0u64; 4];
    for i in 0..4 {
        b[i] = cpu
            .mmu
            .load_doubleword(cpu.x[instr.operands.rs2 as usize].wrapping_add((i * 8) as i64) as u64)
            .expect("BIGINT256_MUL: Failed to load operand B")
            .0;
    }

    // Execute multiplication
    let result = execute_bigint256_mul(a, b);

    // Store 8 u64 result words back to memory at rs1
    for i in 0..8 {
        cpu.mmu
            .store_doubleword(
                cpu.x[instr.operands.rd as usize].wrapping_add((i * 8) as i64) as u64,
                result[i],
            )
            .expect("BIGINT256_MUL: Failed to store result");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracer::{
        emulator::{
            cpu::Cpu,
            default_terminal::DefaultTerminal,
        },
        instruction::format::format_r::FormatR,
        register_inline,
    };

    pub const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

    #[test]
    fn test_bigint_mul_exec() {        
        // Try to register the bigint multiplication inline (it may already be registered)
        let _ = register_inline(
            0x0B,
            0x00,
            0x01,
            "BIGINT256_MUL",
            Box::new(bigint_mul_exec),
            Box::new(|_, _, _, _, _, _| vec![]), // Dummy trace generator for testing
        );

        // Create a CPU with memory
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        // Test with the example values from the Python script
        let a_lo: u128 = 0xf3a8_9b7c_4d2e_1f0a_8b6c_5d3e_2f1a_0b9c;
        let a_hi: u128 = 0x9c8b_7a6f_5e4d_3c2b_1a09_8776_5544_3322;
        let b_lo: u128 = 0x1234_5678_9abc_def0_fedc_ba98_7654_3210;
        let b_hi: u128 = 0xa5b6_c7d8_e9fa_0b1c_2d3e_4f50_6172_8394;

        // Convert to u64 limbs (little-endian)
        let a_limbs = [
            (a_lo & 0xFFFFFFFFFFFFFFFF) as u64,
            ((a_lo >> 64) & 0xFFFFFFFFFFFFFFFF) as u64,
            (a_hi & 0xFFFFFFFFFFFFFFFF) as u64,
            ((a_hi >> 64) & 0xFFFFFFFFFFFFFFFF) as u64,
        ];

        let b_limbs = [
            (b_lo & 0xFFFFFFFFFFFFFFFF) as u64,
            ((b_lo >> 64) & 0xFFFFFFFFFFFFFFFF) as u64,
            (b_hi & 0xFFFFFFFFFFFFFFFF) as u64,
            ((b_hi >> 64) & 0xFFFFFFFFFFFFFFFF) as u64,
        ];

        // Set up memory locations (in valid memory range)
        let rs1_addr = 0x80002000;  // Start of general memory
        let rs2_addr = 0x80002100;  // Offset by 256 bytes
        let rd_addr = 0x80002200;  // Offset by 256 bytes
        

        // Store operand A at rs1_addr
        for i in 0..4 {
            cpu.mmu
                .store_doubleword(rs1_addr + (i * 8) as u64, a_limbs[i])
                .expect("Failed to store operand A");
        }

        // Store operand B at rs2_addr
        for i in 0..4 {
            cpu.mmu
                .store_doubleword(rs2_addr + (i * 8) as u64, b_limbs[i])
                .expect("Failed to store operand B");
        }

        // Set register values to point to memory locations
        cpu.x[10] = rs1_addr as i64; // rs1 = x10
        cpu.x[11] = rs2_addr as i64; // rs2 = x11
        cpu.x[12] = rd_addr as i64;  // rd = x12 (destination for result)

        // Create INLINE instruction
        let instr = INLINE {
            opcode: 0x0B,
            funct3: 0x00,
            funct7: 0x01,
            address: 0x0,
            operands: FormatR {
                rs1: 10,
                rs2: 11,
                rd: 12,  // Not used
            },
            inline_sequence_remaining: None,
            is_compressed: false,
        };

        // Execute the multiplication using exec()
        instr.exec(&mut cpu, &mut ());

        // Expected result from Python computation
        let expected = [
            0xc90fab9bbf1531c0_u64,
            0xaad973dd55fab9b8_u64,
            0xbca2ca1b10cfc4cf_u64,
            0xbb1ee1c3c94e6a79_u64,
            0xa6ae39a57f3091a7_u64,
            0x2dae6201791d3cf5_u64,
            0x50221be9fec14f26_u64,
            0x6555ab47e3e48c66_u64,
        ];

        // Read result from memory at rs1_addr (where result was stored)
        let mut result = [0u64; 8];
        for i in 0..8 {
            result[i] = cpu.mmu
                .load_doubleword(rd_addr + (i * 8) as u64)
                .expect("Failed to load result")
                .0;
        }

        // Verify the result
        assert_eq!(result, expected, "BigInt multiplication result mismatch");
    }
}