/// Common test vectors used across multiple tests
pub struct TestVectors;

pub type Keccak256State = [u64; 25];

impl TestVectors {
    /// Get standard test vectors for state testing
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

    /// Create a simple arithmetic pattern for testing
    pub fn create_simple_pattern() -> Keccak256State {
        core::array::from_fn(|i| (i * 3 + 5) as u64)
    }

    /// Get rotation test vectors
    pub fn get_rotation_test_vectors() -> Vec<(u64, u32, u64)> {
        vec![
            (0x0000000000000001u64, 1, 0x0000000000000002u64), // Simple rotation by 1
            (0x8000000000000000u64, 1, 0x0000000000000001u64), // MSB wraps to LSB
            (0x0123456789ABCDEFu64, 4, 0x123456789ABCDEF0u64), // Rotation by 4
            (0x0123456789ABCDEFu64, 32, 0x89ABCDEF01234567u64), // Rotation by 32 (swap halves)
            (0x0123456789ABCDEFu64, 36, 0x9ABCDEF012345678u64), // Rotation by 36
        ]
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

    /// XKCP test vector: Result after two Keccak-f[1600] permutations on all-zero input
    pub const AFTER_TWO_PERMUTATIONS: Keccak256State = [
        0x2D5C954DF96ECB3C,
        0x6A332CD07057B56D,
        0x093D8D1270D76B6C,
        0x8A20D9B25569D094,
        0x4F9C4F99E5E7F156,
        0xF957B9A2DA65FB38,
        0x85773DAE1275AF0D,
        0xFAF4F247C3D810F7,
        0x1F1B9EE6F79A8759,
        0xE4FECC0FEE98B425,
        0x68CE61B6B9CE68A1,
        0xDEEA66C4BA8F974F,
        0x33C43D836EAFB1F5,
        0xE00654042719DBD9,
        0x7CF8A9F009831265,
        0xFD5449A6BF174743,
        0x97DDAD33D8994B40,
        0x48EAD5FC5D0BE774,
        0xE3B8C8EE55B7B03C,
        0x91A0226E649E42E9,
        0x900E3129E7BADD7B,
        0x202A9EC5FAA3CCE8,
        0x5B3402464E1C3DB6,
        0x609F4E62A44C1059,
        0x20D06CD26A8FBF5C,
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

    /// Expected state after Round 1 rho and pi steps
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
