use super::{INPUT_LIMBS, OUTPUT_LIMBS};

/// Execute n-bit × n-bit = 2n-bit multiplication
/// Input: u64 limbs for each operand (little-endian)
/// Output: u64 limbs for the product (little-endian)
pub fn bigint_mul(lhs: [u64; INPUT_LIMBS], rhs: [u64; INPUT_LIMBS]) -> [u64; OUTPUT_LIMBS] {
    let mut result = [0u64; OUTPUT_LIMBS];

    for (i, &lhs_limb) in lhs.iter().enumerate() {
        for (j, &rhs_limb) in rhs.iter().enumerate() {
            let product = (lhs_limb as u128) * (rhs_limb as u128);
            let low = product as u64;
            let high = (product >> 64) as u64;

            let result_position = i + j;

            let (sum, carry1) = result[result_position].overflowing_add(low);
            result[result_position] = sum;

            let mut carry = carry1 as u64;
            if high != 0 || carry != 0 {
                let (sum_with_hi, carry_hi) = result[result_position + 1].overflowing_add(high);
                let (sum_with_carry, carry_carry) = sum_with_hi.overflowing_add(carry);
                result[result_position + 1] = sum_with_carry;
                carry = (carry_hi as u64) + (carry_carry as u64);

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
