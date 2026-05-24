// SPDX-License-Identifier: Apache-2.0

//! Parity and emulator tests for the Poseidon2-Goldilocks inline.

use crate::exec::{execute_poseidon2_permutation, POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8};

const P: u64 = crate::GOLDILOCKS_MODULUS;

fn u128_mul_mod(a: u64, b: u64) -> u64 {
    ((a as u128) * (b as u128) % (P as u128)) as u64
}

#[test]
fn exec_mul_mod_matches_u128_for_known_cases() {
    let cases: &[(u64, u64)] = &[
        (0, 0),
        (1, 1),
        (P - 1, P - 1),
        (P - 1, 1),
        (2, 3),
        (0xC0000000_00000000, 0xC0000000_00000000),
        (0x80000000_00000001, 0x80000000_00000001),
        (0xFFFFFFFF_FFFFFFFF_u64 % P, 0xFFFFFFFF_FFFFFFFF_u64 % P),
        (12345, P - 1),
    ];
    for &(a, b) in cases {
        assert_eq!(crate::exec::mul_mod(a, b), u128_mul_mod(a, b));
    }
}

#[test]
fn exec_mul_mod_matches_u128_random_stress() {
    let mut seed: u64 = 0xDEADBEEFCAFEBABE;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };
    for _ in 0..100_000 {
        let a = next() % P;
        let b = next() % P;
        assert_eq!(crate::exec::mul_mod(a, b), u128_mul_mod(a, b));
    }
}

mod plonky3_parity {
    use p3_goldilocks::{default_goldilocks_poseidon2_8, Goldilocks};
    use p3_symmetric::Permutation;

    use super::{execute_poseidon2_permutation, P};

    fn plonky3_permute(state_u64: [u64; 8]) -> [u64; 8] {
        use p3_field::{PrimeCharacteristicRing, PrimeField64};

        let perm = default_goldilocks_poseidon2_8();
        let mut state: [Goldilocks; 8] = state_u64.map(Goldilocks::from_u64);
        perm.permute_mut(&mut state);
        state.map(|f| f.as_canonical_u64())
    }

    fn plonky3_permute_generic(state_u64: [u64; 8]) -> [u64; 8] {
        use p3_field::{PrimeCharacteristicRing, PrimeField64};
        use p3_goldilocks::{
            Poseidon2ExternalLayerGoldilocks, Poseidon2InternalLayerGoldilocks,
            GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL, GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL,
            GOLDILOCKS_POSEIDON2_RC_8_INTERNAL,
        };
        use p3_poseidon2::{ExternalLayerConstants, Poseidon2};

        let external = ExternalLayerConstants::<Goldilocks, 8>::new(
            GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL.to_vec(),
            GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL.to_vec(),
        );
        let internal = GOLDILOCKS_POSEIDON2_RC_8_INTERNAL.to_vec();
        let perm: Poseidon2<
            Goldilocks,
            Poseidon2ExternalLayerGoldilocks<8>,
            Poseidon2InternalLayerGoldilocks,
            8,
            7,
        > = Poseidon2::new(external, internal);

        let mut state: [Goldilocks; 8] = state_u64.map(Goldilocks::from_u64);
        perm.permute_mut(&mut state);
        state.map(|f| f.as_canonical_u64())
    }

    fn assert_matches_plonky3(initial: [u64; 8]) {
        let mut ours = initial;
        execute_poseidon2_permutation(&mut ours);
        assert_eq!(ours, plonky3_permute(initial));
        assert_eq!(ours, plonky3_permute_generic(initial));
    }

    #[test]
    fn permute_all_zero_matches_plonky3() {
        assert_matches_plonky3([0u64; 8]);
    }

    #[test]
    fn permute_known_input_matches_plonky3() {
        assert_matches_plonky3([1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn permute_large_values_match_plonky3() {
        assert_matches_plonky3([P - 1, P - 2, P - 3, P - 4, P - 5, P - 6, P - 7, P - 8]);
    }

    #[test]
    fn permute_stress_matches_plonky3() {
        let mut seed: u64 = 0x0BADC0DEF00DCAFE;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };

        for _ in 0..200 {
            assert_matches_plonky3([
                next() % P,
                next() % P,
                next() % P,
                next() % P,
                next() % P,
                next() % P,
                next() % P,
                next() % P,
            ]);
        }
    }

    #[test]
    fn round_constants_match_plonky3_layout() {
        use p3_field::PrimeField64;
        use p3_goldilocks::{
            GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL, GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL,
            GOLDILOCKS_POSEIDON2_RC_8_INTERNAL,
        };

        let mut idx = 0;
        for round in 0..4 {
            for elem in 0..8 {
                assert_eq!(
                    super::POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8[idx],
                    GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL[round][elem].as_canonical_u64()
                );
                idx += 1;
            }
        }
        for round in 0..22 {
            assert_eq!(
                super::POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8[idx],
                GOLDILOCKS_POSEIDON2_RC_8_INTERNAL[round].as_canonical_u64()
            );
            idx += 1;
        }
        for round in 0..4 {
            for elem in 0..8 {
                assert_eq!(
                    super::POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8[idx],
                    GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL[round][elem].as_canonical_u64()
                );
                idx += 1;
            }
        }
        assert_eq!(idx, 86);
    }

    #[test]
    fn internal_diagonal_matches_plonky3() {
        use crate::exec::POSEIDON2_INTERNAL_DIAG;
        use p3_field::PrimeField64;
        use p3_goldilocks::MATRIX_DIAG_8_GOLDILOCKS;

        for i in 0..8 {
            assert_eq!(
                POSEIDON2_INTERNAL_DIAG[i],
                MATRIX_DIAG_8_GOLDILOCKS[i].as_canonical_u64()
            );
        }
    }
}

#[test]
fn sequence_builder_emits_nonempty_instruction_list() {
    use jolt_inlines_sdk::host::InlineOp;
    use tracer::utils::inline_sequence_writer::SequenceInputs;

    let inputs = SequenceInputs::default();
    let instructions = crate::sequence_builder::Poseidon2GoldilocksPermutation::build_sequence(
        (&inputs).into(),
        (&inputs).into(),
    );
    assert!(instructions.len() >= 100);
}

#[test]
fn sequence_builder_emission_is_deterministic() {
    use jolt_inlines_sdk::host::InlineOp;
    use tracer::utils::inline_sequence_writer::SequenceInputs;

    let inputs1 = SequenceInputs::default();
    let inputs2 = SequenceInputs::default();
    let seq1 = crate::sequence_builder::Poseidon2GoldilocksPermutation::build_sequence(
        (&inputs1).into(),
        (&inputs1).into(),
    );
    let seq2 = crate::sequence_builder::Poseidon2GoldilocksPermutation::build_sequence(
        (&inputs2).into(),
        (&inputs2).into(),
    );
    assert_eq!(seq1.len(), seq2.len());
    let dbg1: Vec<String> = seq1.iter().map(|i| format!("{i:?}")).collect();
    let dbg2: Vec<String> = seq2.iter().map(|i| format!("{i:?}")).collect();
    assert_eq!(dbg1, dbg2);
}

#[cfg(test)]
mod emulator {
    use core::array;

    use super::*;
    use jolt_inlines_sdk::host::{
        instruction::{ld::LD, sd::SD},
        FormatInline, InlineOp, InlineOp as InlineOpTrait, InstrAssembler, Instruction,
        VirtualRegisterGuard,
    };
    use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

    fn create_harness(output_size: usize) -> InlineTestHarness {
        let layout = InlineMemoryLayout::single_input(
            POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8.len() * 8,
            output_size,
        );
        InlineTestHarness::new(layout)
    }

    fn execute_inline_permutation(initial_state: &[u64; 8]) -> [u64; 8] {
        let mut harness = create_harness(64);
        harness.setup_registers();
        harness.load_input64(&POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8);
        harness.load_state64(initial_state);
        let inline_instr = InlineTestHarness::create_default_instruction(
            crate::INLINE_OPCODE,
            crate::POSEIDON2_GOLDILOCKS_FUNCT3,
            crate::POSEIDON2_GOLDILOCKS_FUNCT7,
        );
        harness.execute_inline(inline_instr);
        let result_vec = harness.read_output64(8);
        let mut result = [0u64; 8];
        result.copy_from_slice(&result_vec);
        result
    }

    #[test]
    fn emulator_permute_all_zero_matches_reference() {
        let mut reference = [0u64; 8];
        execute_poseidon2_permutation(&mut reference);
        assert_eq!(execute_inline_permutation(&[0u64; 8]), reference);
    }

    #[test]
    fn emulator_permute_known_input_matches_reference() {
        let initial = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let mut reference = initial;
        execute_poseidon2_permutation(&mut reference);
        assert_eq!(execute_inline_permutation(&initial), reference);
    }

    #[test]
    fn emulator_permute_stress_matches_reference() {
        let mut seed: u64 = 0xFACEFEED0BADBEEF;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };
        for _ in 0..50 {
            let initial = [
                next() % P,
                next() % P,
                next() % P,
                next() % P,
                next() % P,
                next() % P,
                next() % P,
                next() % P,
            ];
            let mut reference = initial;
            execute_poseidon2_permutation(&mut reference);
            assert_eq!(execute_inline_permutation(&initial), reference);
        }
    }

    struct IdentityPermutation;

    impl InlineOpTrait for IdentityPermutation {
        const OPCODE: u32 = crate::INLINE_OPCODE;
        const FUNCT3: u32 = 0x05;
        const FUNCT7: u32 = 0x05;
        const NAME: &'static str = "IDENTITY_TEST_INLINE";

        fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
            let vr: [VirtualRegisterGuard; 8] =
                array::from_fn(|_| asm.allocator.allocate_for_inline());
            let mut asm = asm;
            for i in 0..8 {
                asm.emit_ld::<LD>(*vr[i], operands.rs1, (i * 8) as i64);
            }
            for i in 0..8 {
                asm.emit_s::<SD>(operands.rs1, *vr[i], (i * 8) as i64);
            }
            drop(vr);
            asm.finalize_inline()
        }
    }

    struct AddRcOnlyTest;

    impl InlineOpTrait for AddRcOnlyTest {
        const OPCODE: u32 = crate::INLINE_OPCODE;
        const FUNCT3: u32 = 0x06;
        const FUNCT7: u32 = 0x06;
        const NAME: &'static str = "ADD_RC_ONLY_TEST_INLINE";

        fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
            let mut builder =
                crate::sequence_builder::Poseidon2GoldilocksSequenceBuilder::new_for_test(
                    asm, operands,
                );
            builder.test_load_p_and_state_and_add_rc_full(0);
            builder.test_store_and_finalize()
        }
    }

    struct MdsOnlyTest;

    impl InlineOpTrait for MdsOnlyTest {
        const OPCODE: u32 = crate::INLINE_OPCODE;
        const FUNCT3: u32 = 0x07;
        const FUNCT7: u32 = 0x09;
        const NAME: &'static str = "MDS_ONLY_TEST";

        fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
            let mut builder =
                crate::sequence_builder::Poseidon2GoldilocksSequenceBuilder::new_for_test(
                    asm, operands,
                );
            builder.test_load_p_state_mds_only();
            builder.test_store_and_finalize()
        }
    }

    struct IntDiffOnlyTest;

    impl InlineOpTrait for IntDiffOnlyTest {
        const OPCODE: u32 = crate::INLINE_OPCODE;
        const FUNCT3: u32 = 0x07;
        const FUNCT7: u32 = 0x0A;
        const NAME: &'static str = "INT_DIFF_ONLY_TEST";

        fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
            let mut builder =
                crate::sequence_builder::Poseidon2GoldilocksSequenceBuilder::new_for_test(
                    asm, operands,
                );
            builder.test_load_p_state_intdiff_only();
            builder.test_store_and_finalize()
        }
    }

    struct SboxOnlyTest;

    impl InlineOpTrait for SboxOnlyTest {
        const OPCODE: u32 = crate::INLINE_OPCODE;
        const FUNCT3: u32 = 0x07;
        const FUNCT7: u32 = 0x0C;
        const NAME: &'static str = "SBOX_ONLY_TEST";

        fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
            let mut builder =
                crate::sequence_builder::Poseidon2GoldilocksSequenceBuilder::new_for_test(
                    asm, operands,
                );
            builder.test_load_p_state_sbox_only();
            builder.test_store_and_finalize()
        }
    }

    struct MulPairsTest;

    impl InlineOpTrait for MulPairsTest {
        const OPCODE: u32 = crate::INLINE_OPCODE;
        const FUNCT3: u32 = 0x07;
        const FUNCT7: u32 = 0x12;
        const NAME: &'static str = "MUL_PAIRS_TEST";

        fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
            let mut builder =
                crate::sequence_builder::Poseidon2GoldilocksSequenceBuilder::new_for_test(
                    asm, operands,
                );
            builder.test_load_p_state_mul_pairs();
            builder.test_store_and_finalize()
        }
    }

    jolt_inlines_sdk::register_inlines! {
        trace_file: "poseidon2_test_inlines_trace.joltinline",
        extension: jolt_inlines_sdk::host::InlineExtension::Poseidon2Goldilocks,
        ops: [IdentityPermutation, AddRcOnlyTest, MdsOnlyTest, IntDiffOnlyTest, SboxOnlyTest, MulPairsTest],
    }

    fn run_inline_with_state(funct3: u32, funct7: u32, initial: &[u64; 8]) -> [u64; 8] {
        let mut harness = create_harness(64);
        harness.setup_registers();
        harness.load_input64(&POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8);
        harness.load_state64(initial);
        let instr =
            InlineTestHarness::create_default_instruction(crate::INLINE_OPCODE, funct3, funct7);
        harness.execute_inline(instr);
        let v = harness.read_output64(8);
        let mut out = [0u64; 8];
        out.copy_from_slice(&v);
        out
    }

    #[test]
    fn inline_mul_mod_stress_vs_u128() {
        let mut seed: u64 = 0xBADC0FFEE0DDF00D;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };
        for _ in 0..100 {
            let mut state = [0u64; 16];
            for i in 0..8 {
                state[i] = next() % P;
                state[i + 8] = next() % P;
            }
            let layout = InlineMemoryLayout::single_input(8, 128);
            let mut harness = InlineTestHarness::new(layout);
            harness.setup_registers();
            harness.load_input64(&[0u64]);
            harness.load_state64(&state);
            let instr = InlineTestHarness::create_default_instruction(
                crate::INLINE_OPCODE,
                MulPairsTest::FUNCT3,
                MulPairsTest::FUNCT7,
            );
            harness.execute_inline(instr);
            let result = harness.read_output64(8);
            for i in 0..8 {
                assert_eq!(result[i], u128_mul_mod(state[i], state[i + 8]));
            }
        }
    }

    #[test]
    fn add_rc_only_inline_matches_reference() {
        let result =
            run_inline_with_state(AddRcOnlyTest::FUNCT3, AddRcOnlyTest::FUNCT7, &[0u64; 8]);
        let mut expected = [0u64; 8];
        expected.copy_from_slice(&POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8[..8]);
        assert_eq!(result, expected);
    }

    #[test]
    fn mds_only_matches_reference() {
        let initial = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let got = run_inline_with_state(MdsOnlyTest::FUNCT3, MdsOnlyTest::FUNCT7, &initial);
        let mut expected = initial;
        crate::exec::external_mds(&mut expected);
        assert_eq!(got, expected);
    }

    #[test]
    fn int_diff_only_matches_reference() {
        let initial = [11u64, 22, 33, 44, 55, 66, 77, 88];
        let got = run_inline_with_state(IntDiffOnlyTest::FUNCT3, IntDiffOnlyTest::FUNCT7, &initial);
        let mut expected = initial;
        crate::exec::internal_diffusion(&mut expected);
        assert_eq!(got, expected);
    }

    #[test]
    fn sbox_only_matches_reference() {
        let initial: [u64; 8] = POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8[..8]
            .try_into()
            .unwrap();
        let got = run_inline_with_state(SboxOnlyTest::FUNCT3, SboxOnlyTest::FUNCT7, &initial);
        let mut expected = [0u64; 8];
        for i in 0..8 {
            expected[i] = crate::exec::sbox(initial[i]);
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn identity_inline_preserves_state() {
        let layout = InlineMemoryLayout::single_input(8, 64);
        let mut harness = InlineTestHarness::new(layout);
        harness.setup_registers();
        harness.load_input64(&[0u64]);
        let initial: [u64; 8] = [11, 22, 33, 44, 55, 66, 77, 88];
        harness.load_state64(&initial);
        let instr = InlineTestHarness::create_default_instruction(
            crate::INLINE_OPCODE,
            IdentityPermutation::FUNCT3,
            IdentityPermutation::FUNCT7,
        );
        harness.execute_inline(instr);
        assert_eq!(harness.read_output64(8), initial);
    }
}
