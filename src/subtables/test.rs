use ark_ff::PrimeField;
use ark_std::test_rng;

pub fn gen_random_point<F: PrimeField, const C: usize>(memory_bits: usize) -> [Vec<F>; C] {
  let mut rng = test_rng();
  std::array::from_fn(|_| {
    let mut r_i: Vec<F> = Vec::with_capacity(memory_bits);
    for _ in 0..memory_bits {
      r_i.push(F::rand(&mut rng));
    }
    r_i
  })
}

#[macro_export]
macro_rules! materialization_mle_parity_test {
    ($test_name:ident, $table_type:ty, $F:ty, $M:expr, $NUM_SUBTABLES:expr) => {
    #[test]
    fn $test_name() {
        use ark_std::log2;

        const C: usize = 4;
        const M: usize = $M;

        let operand_bits = log2($M) as usize;
        let materialized: [Vec<$F>; { <$table_type as SubtableStrategy<$F, C, M>>::NUM_SUBTABLES }] =
            <$table_type as SubtableStrategy<$F, C, M>>::materialize_subtables();

        for (subtable_index, materialized_table) in materialized.iter().enumerate() {
            for input_index in 0..M {
                assert_eq!(
                    materialized_table[input_index],
                    <$table_type as SubtableStrategy<$F, C, M>>::evaluate_subtable_mle(subtable_index, &index_to_field_bitvector(input_index, operand_bits)),
                    "Subtable {subtable_index} index {input_index} did not match between MLE and materialized subtable."
                );
            }
        }
    }
    };
}
