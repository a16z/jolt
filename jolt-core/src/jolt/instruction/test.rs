use crate::{
    field::JoltField,
    subprotocols::sparse_dense_shout::{LookupBits, SparseDenseSumcheckAlt},
};
use rand::{rngs::StdRng, SeedableRng};

#[macro_export]
/// Tests the consistency of an instruction's `subtables``, `to_indices`, and `combine_lookups`
/// methods. In detail:
/// 1. Materializes each subtable in `subtables`
/// 2. Converts operands to subtable lookup indices using `to_indices`
/// 3. Combines the looked-up subtable entries using `combine_lookups`
/// 4. Checks that the result equals the expected value, given by the `lookup_output`
macro_rules! jolt_instruction_test {
    ($instr:expr) => {
        use $crate::field::JoltField;

        let subtable_lookup_indices = $instr.to_indices(C, ark_std::log2(M) as usize);

        let mut subtable_values: Vec<Fr> = vec![];
        for (subtable, dimension_indices) in $instr.subtables::<Fr>(C, M) {
            let materialized_subtable = subtable.materialize(M);
            for i in dimension_indices.iter() {
                subtable_values.push(Fr::from_u64(
                    materialized_subtable[subtable_lookup_indices[i]] as u64,
                ));
            }
        }

        let actual = $instr.combine_lookups(&subtable_values, C, M);
        let expected = Fr::from_u64($instr.lookup_entry());
        assert_eq!(actual, expected, "{:?}", $instr);
    };
}

#[macro_export]
/// Tests the consistency and correctness of a virtual instruction sequence.
/// In detail:
/// 1. Sets the registers to given values for `x` and `y`.
/// 2. Constructs an `RVTraceRow` with the provided opcode and register values.
/// 3. Generates the virtual instruction sequence using the specified instruction type.
/// 4. Iterates over each row in the virtual sequence and validates the state changes.
/// 5. Verifies that the registers `r_x` and `r_y` have not been modified (not clobbered).
/// 6. Ensures that the result of the instruction sequence is correctly written to the `rd` register.
/// 7. Checks that no unintended modifications have been made to other registers.
macro_rules! jolt_virtual_sequence_test {
    ($instr_type:ty, $opcode:expr) => {
        use crate::jolt::vm::rv32i_vm::RV32I;
        use ark_std::test_rng;
        use common::constants::REGISTER_COUNT;
        use rand_chacha::rand_core::RngCore;

        let mut rng = test_rng();
        let r_x = rng.next_u64() % 32;
        let r_y = rng.next_u64() % 32;
        let rd = rng.next_u64() % 32;
        let x = rng.next_u32() as u64;
        let y = if r_y == r_x { x } else { rng.next_u32() as u64 };
        let result = <$instr_type>::sequence_output(x, y);

        let mut registers = vec![0u64; REGISTER_COUNT as usize];
        registers[r_x as usize] = x;
        registers[r_y as usize] = y;

        let trace_row = RVTraceRow {
            instruction: ELFInstruction {
                address: rng.next_u64(),
                opcode: $opcode,
                rs1: Some(r_x),
                rs2: Some(r_y),
                rd: Some(rd),
                imm: None,
                virtual_sequence_remaining: None,
            },
            register_state: RegisterState {
                rs1_val: Some(x),
                rs2_val: Some(y),
                rd_post_val: Some(result),
            },
            memory_state: None,
            advice_value: None,
        };

        let virtual_sequence = <$instr_type>::virtual_trace(trace_row);
        assert_eq!(virtual_sequence.len(), <$instr_type>::SEQUENCE_LENGTH);

        for row in virtual_sequence {
            if let Some(rs1_val) = row.register_state.rs1_val {
                assert_eq!(registers[row.instruction.rs1.unwrap() as usize], rs1_val);
            }
            if let Some(rs2_val) = row.register_state.rs2_val {
                assert_eq!(registers[row.instruction.rs2.unwrap() as usize], rs2_val);
            }

            let lookup = RV32I::try_from(&row).unwrap();
            let output = lookup.lookup_entry();
            if let Some(rd) = row.instruction.rd {
                registers[rd as usize] = output;
                assert_eq!(
                    registers[rd as usize],
                    row.register_state.rd_post_val.unwrap()
                );
            } else {
                assert!(output == 1)
            }
        }

        for (index, val) in registers.iter().enumerate() {
            if index as u64 == r_x {
                // Check that r_x hasn't been clobbered
                assert_eq!(*val, x);
            } else if index as u64 == r_y {
                // Check that r_y hasn't been clobbered
                assert_eq!(*val, y);
            } else if index as u64 == rd {
                // Check that result was written to rd
                assert_eq!(*val, result as u64);
            } else if index < 32 {
                // None of the other "real" registers were touched
                assert_eq!(*val, 0, "Other 'real' registers should not be touched");
            }
        }
    };
}

#[macro_export]
macro_rules! instruction_mle_test_small {
    ($test_name:ident, $instruction_type:ty) => {
        #[test]
        fn $test_name() {
            use crate::{field::JoltField, utils::index_to_field_bitvector};

            let materialized = <$instruction_type>::default().materialize();
            for (i, entry) in materialized.iter().enumerate() {
                assert_eq!(
                    Fr::from_u64(*entry),
                    <$instruction_type>::default()
                        .evaluate_mle(&index_to_field_bitvector(i as u64, 16)),
                    "MLE did not match materialized table at index {i}",
                );
            }
        }
    };
}

#[macro_export]
macro_rules! instruction_mle_test_large {
    ($test_name:ident, $instruction_type:ty) => {
        #[test]
        fn $test_name() {
            use crate::{field::JoltField, utils::index_to_field_bitvector};

            let mut rng = test_rng();

            for _ in 0..1000 {
                let index = rng.next_u64();
                assert_eq!(
                    Fr::from_u64(<$instruction_type>::default().materialize_entry(index)),
                    <$instruction_type>::default()
                        .evaluate_mle(&index_to_field_bitvector(index, 64)),
                    "MLE did not match materialized table at index {index}",
                );
            }
        }
    };
}

#[macro_export]
macro_rules! instruction_update_function_test {
    ($test_name:ident, $instruction_type:ty) => {
        #[test]
        fn $test_name() {
            use crate::{field::JoltField, utils::index_to_field_bitvector};
            use ark_std::{test_rng, Zero};

            let mut rng = test_rng();
            let instr = <$instruction_type>::default();
            const WORD_SIZE: usize = 32;

            for _ in 0..1000 {
                let index = rng.next_u64();
                let mut t_parameters: Vec<Fr> = index_to_field_bitvector(index, 2 * WORD_SIZE);
                let mut r_prev = None;

                for j in 0..2 * WORD_SIZE {
                    let r_j = Fr::random(&mut rng);
                    let b_j = if t_parameters[j].is_zero() { 0 } else { 1 };

                    let b_next = if j == 2 * WORD_SIZE - 1 {
                        None
                    } else {
                        Some(t_parameters[j + 1].to_u64().unwrap() as u8)
                    };

                    let actual: Fr = (0..instr.eta())
                        .map(|l| {
                            instr.multiplicative_update(l, j, r_j, b_j, r_prev, b_next)
                                * instr.subtable_mle(l, &t_parameters)
                                + instr.additive_update(l, j, r_j, b_j, r_prev, b_next)
                        })
                        .sum();

                    t_parameters[j] = r_j;
                    r_prev = Some(r_j);
                    let expected = instr.evaluate_mle(&t_parameters);

                    assert_eq!(actual, expected);
                }
            }
        }
    };
}

pub fn prefix_suffix_test<F: JoltField, I: SparseDenseSumcheckAlt<F>>() {
    let num_prefixes = I::NUM_PREFIXES;
    let num_suffixes = I::NUM_SUFFIXES;

    let mut rng = StdRng::seed_from_u64(12345);

    for _ in 0..1000 {
        let mut prefix_checkpoints: Vec<Option<F>> = vec![None; num_prefixes];
        let instr = I::default().random(&mut rng);
        let lookup_index = instr.to_lookup_index();

        println!("{instr:?} -> {lookup_index}");

        let result = F::from_u64(instr.materialize_entry(lookup_index));

        let mut j = 0;
        let mut r: Vec<u8> = vec![];
        for phase in 0..3 {
            let suffix_len = (3 - phase) * 16;
            let (mut prefix, suffix) =
                LookupBits::new(lookup_index, 64 - phase * 16).split(suffix_len);

            let suffix_evals: Vec<_> = (0..num_suffixes)
                .map(|l| F::from_u32(I::suffix_mle(l, suffix)))
                .collect();

            for _ in 0..16 {
                let r_x = if j % 2 == 1 {
                    Some(F::from_u8(*r.last().unwrap()))
                } else {
                    None
                };

                let c = prefix.pop_msb();

                let prefix_evals: Vec<_> = (0..num_prefixes)
                    .map(|l| I::prefix_mle(l, &prefix_checkpoints, r_x, c as u32, prefix, j))
                    .collect();

                let combined = I::combine(&prefix_evals, &suffix_evals);
                println!("{j} {prefix} {suffix}");
                if combined != result {
                    for (i, x) in prefix_evals.iter().enumerate() {
                        println!("prefix_evals[{i}] = {x}");
                    }
                    for (i, x) in suffix_evals.iter().enumerate() {
                        println!("suffix_evals[{i}] = {x}");
                    }
                }

                assert_eq!(combined, result);
                r.push(c);

                if r.len() % 2 == 0 {
                    I::update_prefix_checkpoints(
                        &mut prefix_checkpoints,
                        F::from_u8(r[r.len() - 2]),
                        F::from_u8(r[r.len() - 1]),
                        j,
                    );
                }

                j += 1;
            }
        }
    }
}
