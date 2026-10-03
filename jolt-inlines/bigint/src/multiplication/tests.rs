mod bigint256_multiplication {
    use crate::multiplication::sequence_builder::BigintMul256;
    use jolt_inlines_sdk::{
        assert_edge_cases_match_reference, assert_random_cases_match_reference,
    };

    #[test]
    fn test_bigint256_mul_random() {
        assert_random_cases_match_reference::<BigintMul256>(0xB16_1A57, 100);
    }

    #[test]
    fn test_bigint256_mul_edge_cases() {
        assert_edge_cases_match_reference::<BigintMul256>();
    }
}
