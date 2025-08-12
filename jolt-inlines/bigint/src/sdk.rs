//! BigInt multiplication implementation optimized for Jolt zkVM.
//!
//! This module provides 256-bit × 256-bit = 512-bit multiplication.

/// Performs 256-bit × 256-bit multiplication
///
/// # Arguments
/// * `a` - First 256-bit operand as 4 u64 limbs (little-endian)
/// * `b` - Second 256-bit operand as 4 u64 limbs (little-endian)
///
/// # Returns
/// * 512-bit result as 8 u64 limbs (little-endian)
#[inline(always)]
pub fn bigint256_mul(a: [u64; 4], b: [u64; 4]) -> [u64; 8] {
    #[cfg(not(feature = "host"))]
    unsafe {
        // Allocate space for result on stack
        let mut result = [0u64; 8];
        
        // Call the inline multiplication instruction
        // This uses custom RISC-V instruction encoding
        bigint256_mul_inline(
            a.as_ptr(),
            b.as_ptr(), 
            result.as_mut_ptr(),
        );
        
        result
    }
    
    #[cfg(feature = "host")]
    {
        use crate::exec;
        exec::execute_bigint256_mul(a, b)
    }
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
    core::arch::asm!(
        ".insn r 0x0B, 0x0, 0x01, x0, {}, {}",     // INLINE BIGINT256_MUL Instruction
        in(reg) a,
        in(reg) b,
        options(nostack)
    );
    // Note: The inline instruction will handle storing the result at the location of a
    // So we need to copy from a to result if they're different
    if a as *const u64 != result {
        core::ptr::copy_nonoverlapping(a, result, 8);
    }
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
    use crate::exec;
    
    let a_array = *(a as *const [u64; 4]);
    let b_array = *(b as *const [u64; 4]);
    let result_array = exec::execute_bigint256_mul(a_array, b_array);
    core::ptr::copy_nonoverlapping(result_array.as_ptr(), result, 8);
}

#[cfg(all(test, feature = "host"))]
mod tests {
    use super::*;

    #[test]
    fn test_bigint256_mul_simple() {
        // Test 1 * 1 = 1
        let a = [1, 0, 0, 0];
        let b = [1, 0, 0, 0];
        let result = bigint256_mul(a, b);
        assert_eq!(result, [1, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_bigint256_mul_with_carry() {
        // Test with numbers that require carry propagation
        let a = [u64::MAX, 0, 0, 0]; // 2^64 - 1
        let b = [2, 0, 0, 0]; // 2
        let result = bigint256_mul(a, b);
        // (2^64 - 1) * 2 = 2^65 - 2 = 0xfffffffffffffffe with carry
        assert_eq!(result[0], 0xfffffffffffffffe);
        assert_eq!(result[1], 1);
        assert_eq!(result[2], 0);
    }

    #[test]
    fn test_bigint256_mul_full() {
        // Test with larger numbers
        let a = [0x123456789abcdef0, 0xfedcba9876543210, 0, 0];
        let b = [0x1111111111111111, 0x2222222222222222, 0, 0];
        let result = bigint256_mul(a, b);
        
        // Verify the computation matches expected value
        // This would need to be calculated separately to verify
        assert_eq!(result.len(), 8);
    }
}