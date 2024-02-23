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

use ark_ff::PrimeField;
use crate::jolt::instruction::JoltInstruction;
use rand_core::SeedableRng;

const TEST_C: usize = 4;
const TEST_M: usize = 1 << 16;

/// Tests `num_iters` random instructions for their parity between the `lookup_entry()`
/// and `lookup_entry_u64` functions. Needs `concrete_instruction: I` to call `.random(&self)`.
pub fn lookup_entry_u64_parity_random<F, I>(num_iters: usize, concrete_instruction: I) 
where
    F: PrimeField,
    I: JoltInstruction
{
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);
    let mut instructions: Vec<I> = Vec::new();
    for _ in 0..num_iters {
        let instruction = concrete_instruction.random(&mut rng);
        instructions.push(instruction);
    }
    lookup_entry_u64_parity::<F, I>(instructions);
}

/// Tests `instructions` for their parity between the `lookup_entry()`
/// and `lookup_entry_u64` functions.
pub fn lookup_entry_u64_parity<F, I>(instructions: Vec<I>) 
where
    F: PrimeField,
    I: JoltInstruction 
{
    for instruction in instructions {
        let fr = instruction.lookup_entry::<F>(TEST_C, TEST_M);
        let u = instruction.lookup_entry_u64();
        assert_eq!(fr, F::from(u), "Entries did not match for {instruction:?}");
    }
}