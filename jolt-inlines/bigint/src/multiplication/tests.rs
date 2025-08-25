#![cfg(all(test, feature = "host"))]

use super::{INPUT_LIMBS, OUTPUT_LIMBS};
use rand::{thread_rng, Rng};

mod bigint256_multiplication {
    use super::TestVectors;
    use crate::multiplication::trace_generator::bigint_mul_sequence_builder;
    use crate::test_utils::bigint_verify;
    use tracer::emulator::cpu::Xlen;

    #[test]
    fn test_bigint256_mul_default() {
        // Test with the default test vector
        let (lhs, rhs, expected) = TestVectors::get_default_test();
        bigint_verify::assert_exec_trace_equiv(&lhs, &rhs, &expected);
    }

    #[test]
    fn test_bigint256_mul_random() {
        // Test with 100 random inputs
        for i in 0..100 {
            let (lhs, rhs, expected) = TestVectors::generate_random_test();
            bigint_verify::assert_exec_trace_equiv(&lhs, &rhs, &expected);
        }
    }

    #[test]
    fn test_bigint256_mul_edge_cases() {
        println!("\n=== Testing BigInt256 multiplication edge cases ===");

        let edge_cases = TestVectors::get_edge_cases();

        for (i, (lhs, rhs, expected, description)) in edge_cases.iter().enumerate() {
            println!("\nEdge case #{}: {}", i + 1, description);
            bigint_verify::assert_exec_trace_equiv(lhs, rhs, expected);
        }

        println!("\n✅ All {} edge cases passed!\n", edge_cases.len());
    }

    #[test]
    fn test_bigint256_mul_sequence_instruction_count_print() {
        // Build the virtual instruction sequence and print its length for reference
        let seq = bigint_mul_sequence_builder(
            0,     // address
            false, // is_compressed
            Xlen::Bit64,
            10, // rs1
            11, // rs2
            12, // rd
        );
        println!("BIGINT256_MUL sequence length = {}", seq.len());
        assert!(seq.len() > 0);
    }
}

/// Test vectors for BigInt multiplication
pub struct TestVectors;
impl TestVectors {
    /// Get the default test case
    pub fn get_default_test() -> ([u64; INPUT_LIMBS], [u64; INPUT_LIMBS], [u64; OUTPUT_LIMBS]) {
        // Test case: multiply two 256-bit numbers
        let lhs = [
            0x8b6c5d3e2f1a0b9c_u64, // limb 0 (least significant)
            0xf3a89b7c4d2e1f0a_u64, // limb 1
            0x1a09877655443322_u64, // limb 2
            0x9c8b7a6f5e4d3c2b_u64, // limb 3 (most significant)
        ];

        let rhs = [
            0xfedcba9876543210_u64, // limb 0 (least significant)
            0x123456789abcdef0_u64, // limb 1
            0x2d3e4f5061728394_u64, // limb 2
            0xa5b6c7d8e9fa0b1c_u64, // limb 3 (most significant)
        ];

        // Expected result (512-bit)
        let result = [
            0xc90fab9bbf1531c0_u64, // limb 0 (least significant)
            0xaad973dd55fab9b8_u64, // limb 1
            0xbca2ca1b10cfc4cf_u64, // limb 2
            0xbb1ee1c3c94e6a79_u64, // limb 3
            0xa6ae39a57f3091a7_u64, // limb 4
            0x2dae6201791d3cf5_u64, // limb 5
            0x50221be9fec14f26_u64, // limb 6
            0x6555ab47e3e48c66_u64, // limb 7 (most significant)
        ];

        (lhs, rhs, result)
    }

    /// Generate random test inputs and compute the expected result using ark_works
    pub fn generate_random_test() -> ([u64; INPUT_LIMBS], [u64; INPUT_LIMBS], [u64; OUTPUT_LIMBS]) {
        use ark_ff::{BigInteger, BigInteger256};
        use rand::{thread_rng, Rng};

        let mut rng = thread_rng();

        // Generate random 256-bit numbers (4 limbs of 64 bits each)
        let lhs: [u64; INPUT_LIMBS] = core::array::from_fn(|_| rng.gen());
        let rhs: [u64; INPUT_LIMBS] = core::array::from_fn(|_| rng.gen());

        // Compute the expected result using ark_works multiplication
        let lhs_ark = BigInteger256::new(lhs);
        let rhs_ark = BigInteger256::new(rhs);

        let (low_result, high_result) = lhs_ark.mul(&rhs_ark);

        let mut result = [0u64; OUTPUT_LIMBS];
        result[..INPUT_LIMBS].copy_from_slice(&low_result.0);
        result[INPUT_LIMBS..].copy_from_slice(&high_result.0);

        (lhs, rhs, result)
    }

    /// Get edge cases for testing BigInt multiplication
    pub fn get_edge_cases() -> Vec<(
        [u64; INPUT_LIMBS],
        [u64; INPUT_LIMBS],
        [u64; OUTPUT_LIMBS],
        &'static str,
    )> {
        use ark_ff::{BigInteger, BigInteger256};

        let mut cases = Vec::new();

        // Helper function to compute multiplication using ark_works
        let compute = |lhs: [u64; INPUT_LIMBS], rhs: [u64; INPUT_LIMBS]| -> [u64; OUTPUT_LIMBS] {
            let lhs_ark = BigInteger256::new(lhs);
            let rhs_ark = BigInteger256::new(rhs);
            let (low_result, high_result) = lhs_ark.mul(&rhs_ark);
            let mut result = [0u64; OUTPUT_LIMBS];
            result[..INPUT_LIMBS].copy_from_slice(&low_result.0);
            result[INPUT_LIMBS..].copy_from_slice(&high_result.0);
            result
        };

        // Edge case 1: 0 * 0
        let zero = [0u64; INPUT_LIMBS];
        cases.push((zero, zero, compute(zero, zero), "0 * 0"));

        // Edge case 2: 0 * 1
        let one = [1u64, 0, 0, 0];
        cases.push((zero, one, compute(zero, one), "0 * 1"));

        // Edge case 3: 1 * 0
        cases.push((one, zero, compute(one, zero), "1 * 0"));

        // Edge case 4: 1 * 1
        cases.push((one, one, compute(one, one), "1 * 1"));

        // Edge case 5: MAX * 0
        let max = [u64::MAX; INPUT_LIMBS];
        cases.push((max, zero, compute(max, zero), "MAX * 0"));

        // Edge case 6: 0 * MAX
        cases.push((zero, max, compute(zero, max), "0 * MAX"));

        // Edge case 7: MAX * 1
        cases.push((max, one, compute(max, one), "MAX * 1"));

        // Edge case 8: 1 * MAX
        cases.push((one, max, compute(one, max), "1 * MAX"));

        // Edge case 9: MAX * MAX
        cases.push((max, max, compute(max, max), "MAX * MAX"));

        // Edge case 10: Power of 2 (2^64) * Power of 2 (2^128)
        let pow2_64 = [0, 1, 0, 0]; // 2^64
        let pow2_128 = [0, 0, 1, 0]; // 2^128
        cases.push((
            pow2_64,
            pow2_128,
            compute(pow2_64, pow2_128),
            "2^64 * 2^128",
        ));

        // Edge case 11: (2^256 - 1) * 2
        let two = [2u64, 0, 0, 0];
        cases.push((max, two, compute(max, two), "(2^256 - 1) * 2"));

        // Edge case 12: Single limb max * Single limb max
        let single_limb_max = [u64::MAX, 0, 0, 0];
        cases.push((
            single_limb_max,
            single_limb_max,
            compute(single_limb_max, single_limb_max),
            "Single limb MAX * Single limb MAX",
        ));

        // Edge case 13: Alternating bits pattern
        let alternating = [
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
        ];
        cases.push((
            alternating,
            alternating,
            compute(alternating, alternating),
            "Alternating bits * Alternating bits",
        ));

        // Edge case 14: Only MSB set in each limb
        let msb_only = [0x8000000000000000; INPUT_LIMBS];
        cases.push((
            msb_only,
            msb_only,
            compute(msb_only, msb_only),
            "MSB only * MSB only",
        ));

        // Edge case 15: Only LSB set in each limb
        let lsb_only = [1; INPUT_LIMBS];
        cases.push((
            lsb_only,
            lsb_only,
            compute(lsb_only, lsb_only),
            "LSB only * LSB only",
        ));

        // Edge case 16: Sequential values
        let sequential = [1, 2, 3, 4];
        let sequential2 = [5, 6, 7, 8];
        cases.push((
            sequential,
            sequential2,
            compute(sequential, sequential2),
            "Sequential values",
        ));

        // Edge case 17: Only highest limb non-zero
        let high_limb_only = [0, 0, 0, u64::MAX];
        cases.push((
            high_limb_only,
            high_limb_only,
            compute(high_limb_only, high_limb_only),
            "High limb only * High limb only",
        ));

        // Edge case 18: Only lowest limb non-zero
        let low_limb_only = [u64::MAX, 0, 0, 0];
        cases.push((
            low_limb_only,
            low_limb_only,
            compute(low_limb_only, low_limb_only),
            "Low limb only * Low limb only",
        ));

        // Edge case 19: Middle limbs only
        let middle_limbs = [0, u64::MAX, u64::MAX, 0];
        cases.push((
            middle_limbs,
            middle_limbs,
            compute(middle_limbs, middle_limbs),
            "Middle limbs * Middle limbs",
        ));

        // Edge case 20: Prime-like pattern (Mersenne prime approximation)
        let mersenne_like = [u64::MAX, u64::MAX, u64::MAX, 0x7FFFFFFFFFFFFFFF];
        cases.push((
            mersenne_like,
            two,
            compute(mersenne_like, two),
            "Mersenne-like * 2",
        ));

        cases
    }
}
