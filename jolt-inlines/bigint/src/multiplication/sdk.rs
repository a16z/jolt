//! BigInt multiplication implementation optimized for Jolt zkVM.
//!
//! This module provides 256-bit × 256-bit = 512-bit multiplication.

use super::{INPUT_LIMBS, OUTPUT_LIMBS};

/// Performs 256-bit × 256-bit multiplication
///
/// # Arguments
/// * `lhs` - First 256-bit operand as 4 u64 limbs (little-endian)
/// * `rhs` - Second 256-bit operand as 4 u64 limbs (little-endian)
///
/// # Returns
/// * 512-bit result as 8 u64 limbs (little-endian)
#[inline(always)]
pub fn bigint256_mul(lhs: [u64; INPUT_LIMBS], rhs: [u64; INPUT_LIMBS]) -> [u64; OUTPUT_LIMBS] {
    let mut result = [0u64; OUTPUT_LIMBS];
    unsafe {
        bigint256_mul_inline(lhs.as_ptr(), rhs.as_ptr(), result.as_mut_ptr());
    }
    result
}

/// Low-level interface to the BigInt multiplication inline instruction
///
/// # Arguments
/// * `a` - Pointer to 4 u64 words (32 bytes) for first operand
/// * `b` - Pointer to 4 u64 words (32 bytes) for second operand  
/// * `result` - Pointer to 8 u64 words (64 bytes) where result will be written
///
/// # Safety
/// - All pointers must be valid and properly aligned for u64 access (8-byte alignment)
/// - `a` and `b` must point to at least 32 bytes of readable memory
/// - `result` must point to at least 64 bytes of writable memory
/// - The memory regions may overlap (result can be the same as a or b)
#[cfg(not(feature = "host"))]
pub unsafe fn bigint256_mul_inline(a: *const u64, b: *const u64, result: *mut u64) {
    use jolt_inlines_common::constants::{bigint, INLINE_OPCODE};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const bigint::mul256::FUNCT3,
        funct7 = const bigint::mul256::FUNCT7,
        rd = in(reg) result,  // rd - output address
        rs1 = in(reg) a,      // rs1 - first operand address
        rs2 = in(reg) b,      // rs2 - second operand address
        options(nostack)
    );
}

/// Low-level interface to the BigInt multiplication inline instruction (host version)
///
/// # Arguments
/// * `a` - Pointer to 4 u64 words (32 bytes) for first operand
/// * `b` - Pointer to 4 u64 words (32 bytes) for second operand
/// * `result` - Pointer to 8 u64 words (64 bytes) where result will be written
///
/// # Safety
/// - All pointers must be valid and properly aligned for u64 access (8-byte alignment)
/// - `a` and `b` must point to at least 32 bytes of readable memory
/// - `result` must point to at least 64 bytes of writable memory
#[cfg(feature = "host")]
pub unsafe fn bigint256_mul_inline(a: *const u64, b: *const u64, result: *mut u64) {
    use crate::multiplication::exec;

    let a_array = *(a as *const [u64; INPUT_LIMBS]);
    let b_array = *(b as *const [u64; INPUT_LIMBS]);
    let result_array = exec::bigint_mul(a_array, b_array);
    core::ptr::copy_nonoverlapping(result_array.as_ptr(), result, OUTPUT_LIMBS);
}
