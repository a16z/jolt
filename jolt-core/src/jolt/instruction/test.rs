#[macro_export]
/// Tests the consistency of an instruction's `subtables``, `to_indices`, and `combine_lookups` 
/// methods. In detail:
/// 1. Materializes each subtable in `subtables`
/// 2. Converts operands to subtable lookup indices using `to_indices`
/// 3. Combines the looked-up subtable entries using `combine_lookups`
/// 4. Checks that the result equals the expected value, given by the RHS expression
macro_rules! jolt_instruction_test {
  ($instr:expr, $expected_value:expr) => {
    let materialized_subtables: Vec<_> = $instr
      .subtables::<Fr>(C)
      .iter()
      .map(|subtable| subtable.materialize(M))
      .collect();

    let subtable_lookup_indices = $instr.to_indices(C, ark_std::log2(M) as usize);

    let mut subtable_values: Vec<Fr> = Vec::with_capacity(C * $instr.subtables::<Fr>(C).len());
    for subtable in materialized_subtables {
      for lookup_index in subtable_lookup_indices.iter() {
        subtable_values.push(subtable[*lookup_index]);
      }
    }

    let actual = $instr.combine_lookups(&subtable_values, C, M);
    let expected = $expected_value;

    assert_eq!(actual, expected, "{:?}", $instr);
  };
}
