//! Test constants for BLAKE2 implementation

use hex_literal::hex;

/// Test vector input for BLAKE2b
pub const BLAKE2B_TEST_INPUT: &[u8] = b"The quick brown fox jumps over the lazy dog";

/// Expected output for BLAKE2b test vector
pub const BLAKE2B_TEST_OUTPUT: [u8; 64] = hex!("
    a8add4bdddfd93e4877d2746e62817b1
    16364a1fa7bc148d95090bc7333b3673
    f82401cf7aa2e4cb1ecd90296e3f14cb
    5413f8ed77be73045b13914cdcd6a918
");

/// Test vector input for BLAKE2s
pub const BLAKE2S_TEST_INPUT: &[u8] = b"The quick brown fox jumps over the lazy dog";

/// Expected output for BLAKE2s test vector
pub const BLAKE2S_TEST_OUTPUT: [u8; 32] = hex!("
    606beeec743ccbeff6cbcdf5d5302aa8
    55c559c9c3d7b33b054d9f4024cdedcc
");
