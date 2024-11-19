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
                subtable_values.push(materialized_subtable[subtable_lookup_indices[i]]);
            }
        }

        let actual = $instr.combine_lookups(&subtable_values, C, M);
        let expected = Fr::from_u64($instr.lookup_entry()).unwrap();
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
