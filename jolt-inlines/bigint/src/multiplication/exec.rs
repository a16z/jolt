use tracer::{
    emulator::cpu::Cpu,
    instruction::{inline::INLINE, RISCVInstruction},
};

use super::{INPUT_LIMBS, OUTPUT_LIMBS};

/// Execute n-bit × n-bit = 2n-bit multiplication
/// Input: u64 limbs for each operand (little-endian)
/// Output: u64 limbs for the product (little-endian)
pub fn bigint_mul(lhs: [u64; INPUT_LIMBS], rhs: [u64; INPUT_LIMBS]) -> [u64; OUTPUT_LIMBS] {
    let mut result = [0u64; OUTPUT_LIMBS];

    // Schoolbook multiplication: compute all partial products
    // For each a[i] * b[j], add the 128-bit product to result[i+j]
    for (i, &lhs_limb) in lhs.iter().enumerate() {
        for (j, &rhs_limb) in rhs.iter().enumerate() {
            // Compute 64×64 = 128-bit product
            let product = (lhs_limb as u128) * (rhs_limb as u128);
            let low = product as u64;
            let high = (product >> 64) as u64;

            // Add to result[i+j] with carry propagation
            let result_position = i + j;

            // Add low part
            let (sum, carry1) = result[result_position].overflowing_add(low);
            result[result_position] = sum;

            // Propagate carry through high part and beyond
            let mut carry = carry1 as u64;
            if high != 0 || carry != 0 {
                // Add high part plus carry from low part
                let (sum_with_hi, carry_hi) = result[result_position + 1].overflowing_add(high);
                let (sum_with_carry, carry_carry) = sum_with_hi.overflowing_add(carry);
                result[result_position + 1] = sum_with_carry;
                carry = (carry_hi as u64) + (carry_carry as u64);

                // Continue propagating carry if needed
                let mut carry_position = result_position + 2;
                while carry != 0 && carry_position < OUTPUT_LIMBS {
                    let (sum, c) = result[carry_position].overflowing_add(carry);
                    result[carry_position] = sum;
                    carry = c as u64;
                    carry_position += 1;
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
    let mut a = [0u64; INPUT_LIMBS];
    for (i, limb) in a.iter_mut().enumerate().take(INPUT_LIMBS) {
        *limb = cpu
            .mmu
            .load_doubleword(cpu.x[instr.operands.rs1 as usize].wrapping_add((i * 8) as i64) as u64)
            .expect("BIGINT256_MUL: Failed to load operand A")
            .0;
    }

    // Load 4 u64 words from memory at rs2 (second operand)
    let mut b = [0u64; INPUT_LIMBS];
    for (i, limb) in b.iter_mut().enumerate().take(INPUT_LIMBS) {
        *limb = cpu
            .mmu
            .load_doubleword(cpu.x[instr.operands.rs2 as usize].wrapping_add((i * 8) as i64) as u64)
            .expect("BIGINT256_MUL: Failed to load operand B")
            .0;
    }

    // Execute multiplication
    let result = bigint_mul(a, b);

    // Store 8 u64 result words back to memory at rs1
    for (i, limb) in result.iter().enumerate().take(OUTPUT_LIMBS) {
        cpu.mmu
            .store_doubleword(
                cpu.x[instr.operands.rd as usize].wrapping_add((i * 8) as i64) as u64,
                *limb,
            )
            .expect("BIGINT256_MUL: Failed to store result");
    }
}
