#[cfg(test)]
/// Common test vectors used across multiple SHA-256 tests
pub struct TestVectors;

pub type Sha256Block = [u32; 16];
pub type Sha256State = [u32; 8];

impl TestVectors {
    /// Get standard test vectors for SHA-256 compression function testing
    /// These vectors test the compression function directly (block + initial state -> final state)
    pub fn get_standard_test_vectors() -> Vec<(&'static str, Sha256Block, Sha256State, Sha256State)>
    {
        vec![
            (
                "all zeros block with initial IV",
                Self::zero_block(),
                Self::sha256_initial_iv(),
                nist_vectors::ZERO_BLOCK_WITH_INITIAL_IV_RESULT,
            ),
            (
                "NIST pattern block with initial IV",
                nist_vectors::NIST_PATTERN_BLOCK,
                Self::sha256_initial_iv(),
                nist_vectors::NIST_PATTERN_BLOCK_RESULT,
            ),
            (
                "all ones block with initial IV",
                Self::all_ones_block(),
                Self::sha256_initial_iv(),
                nist_vectors::ALL_ONES_BLOCK_RESULT,
            ),
        ]
    }

    /// Create a zero block for testing
    pub fn zero_block() -> Sha256Block {
        [0u32; 16]
    }

    /// Create an all-ones block for testing  
    pub fn all_ones_block() -> Sha256Block {
        [0xFFFFFFFF; 16]
    }

    /// SHA-256 initial hash value (IV) as specified in FIPS 180-4
    pub fn sha256_initial_iv() -> Sha256State {
        [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ]
    }
}

pub mod nist_vectors {
    //! Test constants and vectors for SHA-256 compression function tests
    //!
    //! These constants are extracted from NIST test vectors and other reference implementations
    //! to provide canonical test cases for the SHA-256 compression function.

    use super::{Sha256Block, Sha256State};

    /// NIST test vector: All-zero 512-bit block compressed with initial IV
    /// Input block: 64 bytes of zeros (512 bits)
    /// Input state: SHA-256 initial IV
    /// Expected output from NIST SHA-256 test vectors
    pub const ZERO_BLOCK_WITH_INITIAL_IV_RESULT: Sha256State = [
        0xda5698be, 0x17b9b469, 0x62335799, 0x779fbeca, 0x8ce5d491, 0xc0d26243, 0xbafef9ea,
        0x1837a9d8,
    ];

    /// NIST test vector: Pattern block (000102030405...3e3f) compressed with initial IV
    /// This is the pattern 00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f repeated
    pub const NIST_PATTERN_BLOCK: Sha256Block = [
        0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f, 0x10111213, 0x14151617, 0x18191a1b,
        0x1c1d1e1f, 0x20212223, 0x24252627, 0x28292a2b, 0x2c2d2e2f, 0x30313233, 0x34353637,
        0x38393a3b, 0x3c3d3e3f,
    ];

    pub const NIST_PATTERN_BLOCK_RESULT: Sha256State = [
        0xfc99a2df, 0x88f42a7a, 0x7bb9d180, 0x33cdc6a2, 0x0256755f, 0x9d5b9a50, 0x44a9cc31,
        0x5abe84a7,
    ];

    /// NIST test vector: All-ones block (ffffffff repeated) compressed with initial IV
    pub const ALL_ONES_BLOCK_RESULT: Sha256State = [
        0xef0c748d, 0xf4da50a8, 0xd6c43c01, 0x3edc3ce7, 0x6c9d9fa9, 0xa1458ade, 0x56eb86c0,
        0xa64492d2,
    ];
}
