use crate::{
    field::JoltField,
    jolt::{
        instruction::{
            prefixes::{PrefixCheckpoint, Prefixes},
            suffixes::SuffixEval,
        },
        vm::rv32i_vm::RV32I,
    },
    subprotocols::sparse_dense_shout::{LookupBits, PrefixSuffixDecomposition},
    utils::index_to_field_bitvector,
};
use ark_std::test_rng;
use common::constants::REGISTER_COUNT;
use rand::{rngs::StdRng, SeedableRng};
use rand_core::RngCore;
use strum::{EnumCount, IntoEnumIterator};
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::{JoltInstruction, VirtualInstructionSequence};

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

/// Tests the consistency and correctness of a virtual instruction sequence.
/// In detail:
/// 1. Sets the registers to given values for `x` and `y`.
/// 2. Constructs an `RVTraceRow` with the provided opcode and register values.
/// 3. Generates the virtual instruction sequence using the specified instruction type.
/// 4. Iterates over each row in the virtual sequence and validates the state changes.
/// 5. Verifies that the registers `r_x` and `r_y` have not been modified (not clobbered).
/// 6. Ensures that the result of the instruction sequence is correctly written to the `rd` register.
/// 7. Checks that no unintended modifications have been made to other registers.
pub fn jolt_virtual_sequence_test<I: VirtualInstructionSequence>(opcode: RV32IM) {
    let mut rng = test_rng();

    for _ in 0..1000 {
        let r_x = rng.next_u64() % 32;
        let r_y = rng.next_u64() % 32;
        let mut rd = rng.next_u64() % 32;
        while rd == 0 {
            rd = rng.next_u64() % 32;
        }
        let x = if r_x == 0 { 0 } else { rng.next_u32() as u64 };
        let y = if r_y == r_x {
            x
        } else if r_y == 0 {
            0
        } else {
            rng.next_u32() as u64
        };
        let result = I::sequence_output(x, y);

        let mut registers = vec![0u64; REGISTER_COUNT as usize];
        registers[r_x as usize] = x;
        registers[r_y as usize] = y;

        let trace_row = RVTraceRow {
            instruction: ELFInstruction {
                address: rng.next_u64(),
                opcode,
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
            precompile_input: None,
            precompile_output_address: None,
        };

        let virtual_sequence = I::virtual_trace(trace_row);
        assert_eq!(virtual_sequence.len(), I::SEQUENCE_LENGTH);

        for row in virtual_sequence {
            if let Some(rs1_val) = row.register_state.rs1_val {
                assert_eq!(
                    registers[row.instruction.rs1.unwrap() as usize],
                    rs1_val,
                    "{row:?}"
                );
            }
            if let Some(rs2_val) = row.register_state.rs2_val {
                assert_eq!(
                    registers[row.instruction.rs2.unwrap() as usize],
                    rs2_val,
                    "{row:?}"
                );
            }

            let lookup = RV32I::try_from(&row).unwrap();
            let output = lookup.lookup_entry();
            if let Some(rd) = row.instruction.rd {
                registers[rd as usize] = output;
                assert_eq!(
                    registers[rd as usize],
                    row.register_state.rd_post_val.unwrap(),
                    "{row:?}"
                );
            } else {
                assert!(output == 1, "{row:?}")
            }
        }

        for (index, val) in registers.iter().enumerate() {
            if index as u64 == r_x {
                if r_x != rd {
                    // Check that r_x hasn't been clobbered
                    assert_eq!(*val, x);
                }
            } else if index as u64 == r_y {
                if r_y != rd {
                    // Check that r_y hasn't been clobbered
                    assert_eq!(*val, y);
                }
            } else if index as u64 == rd {
                // Check that result was written to rd
                assert_eq!(*val, result as u64);
            } else if index < 32 {
                // None of the other "real" registers were touched
                assert_eq!(*val, 0, "Other 'real' registers should not be touched");
            }
        }
    }
}

pub fn instruction_mle_random_test<F: JoltField, I: JoltInstruction + Default>() {
    let mut rng = test_rng();

    for _ in 0..1000 {
        let index = rng.next_u64();
        assert_eq!(
            F::from_u64(I::default().materialize_entry(index)),
            I::default().evaluate_mle(&index_to_field_bitvector(index, 64)),
            "MLE did not match materialized table at index {index}",
        );
    }
}

pub fn instruction_mle_full_hypercube_test<F: JoltField, I: JoltInstruction + Default>() {
    let materialized = I::default().materialize();
    for (i, entry) in materialized.iter().enumerate() {
        assert_eq!(
            F::from_u64(*entry),
            I::default().evaluate_mle(&index_to_field_bitvector(i as u64, 16)),
            "MLE did not match materialized table at index {i}",
        );
    }
}

pub fn materialize_entry_test<F: JoltField, I: JoltInstruction + Default>() {
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10000 {
        let instr = I::default().random(&mut rng);
        assert_eq!(
            instr.lookup_entry(),
            instr.materialize_entry(instr.to_lookup_index())
        );
    }
}

pub fn prefix_suffix_test<F: JoltField, I: PrefixSuffixDecomposition<32>>() {
    let mut rng = StdRng::seed_from_u64(12345);

    for _ in 0..1000 {
        let mut prefix_checkpoints: Vec<PrefixCheckpoint<F>> = vec![None.into(); Prefixes::COUNT];
        let instr = I::default().random(&mut rng);
        let lookup_index = instr.to_lookup_index();

        let result = F::from_u64(instr.materialize_entry(lookup_index));

        let mut j = 0;
        let mut r: Vec<u8> = vec![];
        for phase in 0..4 {
            let suffix_len = (3 - phase) * 16;
            let (mut prefix_bits, suffix_bits) =
                LookupBits::new(lookup_index, 64 - phase * 16).split(suffix_len);

            let suffix_evals: Vec<_> = instr
                .suffixes()
                .iter()
                .map(|suffix| SuffixEval::from(F::from_u32(suffix.suffix_mle::<32>(suffix_bits))))
                .collect();

            for _ in 0..16 {
                let r_x = if j % 2 == 1 {
                    Some(F::from_u8(*r.last().unwrap()))
                } else {
                    None
                };

                let c = prefix_bits.pop_msb();

                let prefix_evals: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<32, F>(
                            &prefix_checkpoints,
                            r_x,
                            c as u32,
                            prefix_bits,
                            j,
                        )
                    })
                    .collect();

                let combined = instr.combine(&prefix_evals, &suffix_evals);
                if combined != result {
                    println!("{instr:?} -> {lookup_index}");
                    println!("{j} {prefix_bits} {suffix_bits}");
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
                    Prefixes::update_checkpoints::<32, F>(
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
