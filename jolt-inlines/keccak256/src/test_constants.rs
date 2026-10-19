pub struct TestVectors;

use crate::Keccak256State;

impl TestVectors {
    pub fn get_standard_test_vectors() -> Vec<(&'static str, Keccak256State)> {
        vec![
            ("zero state", [0u64; 25]),
            ("simple pattern", Self::create_simple_pattern()),
            (
                "xkcp first permutation result",
                xkcp_vectors::AFTER_ONE_PERMUTATION,
            ),
        ]
    }

    pub fn create_simple_pattern() -> Keccak256State {
        core::array::from_fn(|i| (i * 3 + 5) as u64)
    }
}

pub mod xkcp_vectors {
    //! Test constants and vectors for Keccak256 instruction tests
    //!
    //! These constants are extracted from XKCP test vectors and other reference implementations
    //! to avoid duplication and accidental modification during test refactoring.

    use super::Keccak256State;

    #[derive(Debug, PartialEq)]
    pub struct ExpectedKeccakRoundState {
        pub theta: Keccak256State,
        pub rho_pi: Keccak256State,
        pub chi: Keccak256State,
        pub iota: Keccak256State,
    }

    /// XKCP test vector: Result after one Keccak-f[1600] permutation on all-zero input
    /// Source: https://github.com/XKCP/XKCP/blob/master/tests/TestVectors/KeccakF-1600-IntermediateValues.txt
    pub const AFTER_ONE_PERMUTATION: Keccak256State = [
        0xF1258F7940E1DDE7,
        0x84D5CCF933C0478A,
        0xD598261EA65AA9EE,
        0xBD1547306F80494D,
        0x8B284E056253D057,
        0xFF97A42D7F8E6FD4,
        0x90FEE5A0A44647C4,
        0x8C5BDA0CD6192E76,
        0xAD30A6F71B19059C,
        0x30935AB7D08FFC64,
        0xEB5AA93F2317D635,
        0xA9A6E6260D712103,
        0x81A57C16DBCF555F,
        0x43B831CD0347C826,
        0x01F22F1A11A5569F,
        0x05E5635A21D9AE61,
        0x64BEFEF28CC970F2,
        0x613670957BC46611,
        0xB87C5A554FD00ECB,
        0x8C3EE88A1CCF32C8,
        0x940C7922AE3A2614,
        0x1841F924A2C509E4,
        0x16F53526E70465C2,
        0x75F644E97F30A13B,
        0xEAF1FF7B5CECA249,
    ];

    pub const EXPECTED_AFTER_ROUND1_THETA: Keccak256State = [
        0x0000000000000001,
        0x0000000000000001,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000002,
        0x0000000000000000,
        0x0000000000000001,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000002,
        0x0000000000000000,
        0x0000000000000001,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000002,
        0x0000000000000000,
        0x0000000000000001,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000002,
        0x0000000000000000,
        0x0000000000000001,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000002,
    ];

    pub const EXPECTED_AFTER_ROUND1_CHI: Keccak256State = [
        0x0000000000000001u64, // After chi, before iota
        0x0000100000000000u64,
        0x0000000000008000u64,
        0x0000000000000001u64,
        0x0000100000008000u64,
        0x0000000000000000u64,
        0x0000200000200000u64,
        0x0000000000000000u64,
        0x0000200000000000u64,
        0x0000000000200000u64,
        0x0000000000000002u64,
        0x0000000000000200u64,
        0x0000000000000000u64,
        0x0000000000000202u64,
        0x0000000000000000u64,
        0x0000000010000400u64,
        0x0000000000000000u64,
        0x0000000000000400u64,
        0x0000000010000000u64,
        0x0000000000000000u64,
        0x0000010000000000u64,
        0x0000000000000000u64,
        0x0000010000000004u64,
        0x0000000000000000u64,
        0x0000000000000004u64,
    ];

    pub const EXPECTED_AFTER_ROUND1_RHO_PI: Keccak256State = [
        0x0000000000000001u64,
        0x0000100000000000u64,
        0x0000000000000000u64,
        0x0000000000000000u64,
        0x0000000000008000u64,
        0x0000000000000000u64,
        0x0000000000200000u64,
        0x0000000000000000u64,
        0x0000200000000000u64,
        0x0000000000000000u64,
        0x0000000000000002u64,
        0x0000000000000000u64,
        0x0000000000000000u64,
        0x0000000000000200u64,
        0x0000000000000000u64,
        0x0000000010000000u64,
        0x0000000000000000u64,
        0x0000000000000400u64,
        0x0000000000000000u64,
        0x0000000000000000u64,
        0x0000000000000000u64,
        0x0000000000000000u64,
        0x0000010000000000u64,
        0x0000000000000000u64,
        0x0000000000000004u64,
    ];

    pub const EXPECTED_AFTER_ROUND1_IOTA: Keccak256State = [
        0x0000000000008083u64,
        0x0000100000000000u64,
        0x0000000000008000u64,
        0x0000000000000001u64,
        0x0000100000008000u64,
        0x0000000000000000u64,
        0x0000200000200000u64,
        0x0000000000000000u64,
        0x0000200000000000u64,
        0x0000000000200000u64,
        0x0000000000000002u64,
        0x0000000000000200u64,
        0x0000000000000000u64,
        0x0000000000000202u64,
        0x0000000000000000u64,
        0x0000000010000400u64,
        0x0000000000000000u64,
        0x0000000000000400u64,
        0x0000000010000000u64,
        0x0000000000000000u64,
        0x0000010000000000u64,
        0x0000000000000000u64,
        0x0000010000000004u64,
        0x0000000000000000u64,
        0x0000000000000004u64,
    ];

    pub const EXPECTED_AFTER_ROUND1: ExpectedKeccakRoundState = ExpectedKeccakRoundState {
        theta: EXPECTED_AFTER_ROUND1_THETA,
        rho_pi: EXPECTED_AFTER_ROUND1_RHO_PI,
        chi: EXPECTED_AFTER_ROUND1_CHI,
        iota: EXPECTED_AFTER_ROUND1_IOTA,
    };
}
