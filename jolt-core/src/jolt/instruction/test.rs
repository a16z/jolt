#[macro_export]
/// Tests the consistency of an instruction's `subtables``, `to_indices`, and `combine_lookups`
/// methods. In detail:
/// 1. Materializes each subtable in `subtables`
/// 2. Converts operands to subtable lookup indices using `to_indices`
/// 3. Combines the looked-up subtable entries using `combine_lookups`
/// 4. Checks that the result equals the expected value, given by the `lookup_output`
macro_rules! jolt_instruction_test {
    ($instr:expr) => {
        use ark_ff::PrimeField;

        let subtable_lookup_indices = $instr.to_indices(C, ark_std::log2(M) as usize);

        let mut subtable_values: Vec<Fr> = vec![];
        for (subtable, dimension_indices) in $instr.subtables::<Fr>(C, M) {
            let materialized_subtable = subtable.materialize(M);
            for i in dimension_indices.iter() {
                subtable_values.push(materialized_subtable[subtable_lookup_indices[i]]);
            }
        }

        let actual = $instr.combine_lookups(&subtable_values, C, M);
        let expected = Fr::from_u64($instr.lookup_entry()).unwrap();
        assert_eq!(actual, expected, "{:?}", $instr);
    };
}
