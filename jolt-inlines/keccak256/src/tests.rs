#![cfg(all(test, feature = "host"))]

mod exec {
    use crate::sequence_builder::Keccak256Permutation;
    use crate::test_utils::*;
    use jolt_inlines_sdk::spec::InlineSpec;

    #[test]
    fn test_keccak256_direct_execution() {
        for (i, test_case) in keccak_test_vectors().iter().enumerate() {
            let mut harness = Keccak256Permutation::create_harness();
            Keccak256Permutation::load(&mut harness, &test_case.input);
            harness.execute_inline(Keccak256Permutation::instruction());
            let result = Keccak256Permutation::read(&mut harness);
            assert_eq!(
                result, test_case.expected,
                "Keccak256 direct execution test case {} failed: {}\nInput: {:016x?}\nExpected: {:016x?}\nActual: {:016x?}",
                i + 1, test_case.description, test_case.input, test_case.expected, result
            );
        }
    }

    #[test]
    fn test_execute_keccak256() {
        let e2e_vectors: &[(&[u8], [u8; 32])] = &[
            (
                b"",
                hex_literal::hex!(
                    "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
                ),
            ),
            (
                b"abc",
                hex_literal::hex!(
                    "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
                ),
            ),
        ];

        for (input, expected_hash) in e2e_vectors {
            let hash = crate::spec::execute_keccak256(input);
            assert_eq!(&hash, expected_hash);
        }
    }
}

mod exec_trace_equivalence {
    use crate::sequence_builder::Keccak256Permutation;
    use crate::test_constants::*;
    use jolt_inlines_sdk::spec::InlineSpec;

    #[test]
    fn test_keccak_against_reference() {
        let initial_state = [0u64; 25];
        let expected_final_state = xkcp_vectors::AFTER_ONE_PERMUTATION;
        let mut harness = Keccak256Permutation::create_harness();
        Keccak256Permutation::load(&mut harness, &initial_state);
        harness.execute_inline(Keccak256Permutation::instruction());
        let trace_result = Keccak256Permutation::read(&mut harness);
        for i in 0..25 {
            assert_eq!(trace_result[i], expected_final_state[i]);
        }
    }
}

mod exec_unit {
    use crate::sequence_builder::ROUND_CONSTANTS;
    use crate::spec::{execute_chi, execute_iota, execute_rho_and_pi, execute_theta};
    use crate::test_constants::xkcp_vectors;
    use crate::NUM_LANES;

    #[test]
    fn test_execute_theta() {
        let mut state = [0u64; NUM_LANES];
        state[0] = 1;
        state[5] = 2;
        state[10] = 4;
        let original_state = state;
        execute_theta(&mut state);
        assert_ne!(
            state, original_state,
            "theta: state unchanged; expected column parity diffusion"
        );
    }

    #[test]
    fn test_execute_rho_and_pi() {
        let mut state = [0u64; NUM_LANES];
        state[1] = 0xFF;
        let original_state = state;
        execute_rho_and_pi(&mut state);
        assert_ne!(
            state, original_state,
            "rho+pi: state unchanged; expected rotations and permutation"
        );
        assert_ne!(
            state[1], 0xFF,
            "rho+pi: lane [1] not moved/rotated as expected"
        );
    }

    #[test]
    fn test_execute_chi() {
        let mut state = [0u64; NUM_LANES];
        state[0] = 0xFF;
        state[1] = 0xAA;
        state[2] = 0x55;
        execute_chi(&mut state);
        let expected_0 = 0xFF ^ ((!0xAA) & 0x55);
        assert_eq!(
            state[0], expected_0,
            "chi: A[0] mismatch (expected {expected_0:#x}, got {:#x})",
            state[0]
        );
    }

    #[test]
    fn test_execute_iota() {
        let mut state = [0u64; NUM_LANES];
        state[0] = 0x1234;
        execute_iota(&mut state, 0x5678);
        assert_eq!(state[0], 0x1234 ^ 0x5678, "iota: A[0,0] mismatch");
        state
            .into_iter()
            .enumerate()
            .skip(1)
            .for_each(|(i, s)| assert_eq!(s, 0, "iota: lane {i} changed unexpectedly"));
    }

    #[test]
    fn test_step_by_step_round_1() {
        let mut state = [0u64; NUM_LANES];
        state[0] = 0x0000000000000001;
        let round = 1;
        let expected_states = &xkcp_vectors::EXPECTED_AFTER_ROUND1;
        type StepFn = fn(&mut [u64; NUM_LANES]);
        let steps: &[(&str, StepFn, [u64; NUM_LANES])] = &[
            ("theta", execute_theta, expected_states.theta),
            ("rho and pi", execute_rho_and_pi, expected_states.rho_pi),
            ("chi", execute_chi, expected_states.chi),
        ];
        for &(name, step_fn, expected) in steps {
            step_fn(&mut state);
            assert_eq!(state, expected, "round 1: mismatch after {name}");
        }
        execute_iota(&mut state, ROUND_CONSTANTS[round]);
        assert_eq!(state, expected_states.iota, "round 1: mismatch after iota");
    }
}
