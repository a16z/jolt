#[macro_export]
macro_rules! jolt_instruction_test {
  ($instr:expr, $expected_value:expr) => {
    let materialized_subtables: Vec<_> = $instr
      .subtables::<Fr>()
      .iter()
      .map(|subtable| subtable.materialize(M))
      .collect();

    let subtable_lookup_indices = $instr.to_indices(C, ark_std::log2(M) as usize);
    // for idx in subtable_lookup_indices.clone() {
    //   println!("{}", idx);
    // }

    let mut subtable_values: Vec<Fr> = Vec::with_capacity(C * $instr.subtables::<Fr>().len());
    for subtable in materialized_subtables {
      for lookup_index in subtable_lookup_indices.iter() {
        subtable_values.push(subtable[*lookup_index]);
      }
    }

    let actual = $instr.combine_lookups(&subtable_values, C, M);
    let expected = $expected_value;

    // if (actual != expected) {
    //   println!("{:?}", $instr);
    //   println!("expected: {:?}", expected);
    //   println!("actual: {:?}", actual);
    // }

    assert_eq!(
      actual,
      expected,
      "{:?}",
      $instr
    );
  };
}
