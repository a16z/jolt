#[macro_export]
macro_rules! jolt_materialization_mle_parity_test {
    ($test_name:ident, $jolt_strategy:ty, $F:ty, $M:expr)  => {

    #[test]
    fn $test_name() {
        use ark_std::log2;
        use crate::jolt::jolt_strategy::JoltStrategy;
        use crate::utils::index_to_field_bitvector;

        const M: usize = $M;

        let operand_bits = log2($M) as usize;
        let materialized: Vec<Vec<$F>> =
            <$jolt_strategy>::materialize_subtables();
        assert_eq!(materialized.len(), <$jolt_strategy as JoltStrategy<$F>>::num_subtables());

        for (subtable_index, materialized_table) in materialized.iter().enumerate() {
            for input_index in 0..M {
                assert_eq!(
                    materialized_table[input_index],
                    <$jolt_strategy>::evaluate_memory_mle(subtable_index, &index_to_field_bitvector(input_index, operand_bits)),
                    "Subtable {subtable_index} index {input_index} did not match between MLE and materialized subtable."
                );
            }
        }
    }
    };
}
