mod bigint256_multiplication {
    use crate::multiplication::sequence_builder::BigintMul256;
    use jolt_inlines_sdk::{
        assert_edge_cases_match_reference, assert_random_cases_match_reference,
        assert_reference_matches_harness,
    };

    #[test]
    fn test_bigint256_mul_default() {
        let lhs = [
            0x8b6c5d3e2f1a0b9c_u64,
            0xf3a89b7c4d2e1f0a_u64,
            0x1a09877655443322_u64,
            0x9c8b7a6f5e4d3c2b_u64,
        ];
        let rhs = [
            0xfedcba9876543210_u64,
            0x123456789abcdef0_u64,
            0x2d3e4f5061728394_u64,
            0xa5b6c7d8e9fa0b1c_u64,
        ];

        assert_reference_matches_harness::<BigintMul256>(&(lhs, rhs));
    }

    #[test]
    fn test_bigint256_mul_random() {
        assert_random_cases_match_reference::<BigintMul256>(0xB16_1A57, 100);
    }

    #[test]
    fn test_bigint256_mul_edge_cases() {
        assert_edge_cases_match_reference::<BigintMul256>();
    }
}
