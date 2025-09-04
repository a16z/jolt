#[cfg(test)]
mod tests {

    use super::TestVectors;
    use crate::test_utils::blake2_verify;

    #[test]
    fn test_blake2_permutation_default() {
        let (mut state, message, expected_state, counter, is_final) = TestVectors::get_default_test();
        state[0] ^= 0x01010000 ^ (0u64 << 8) ^ 64u64;
        blake2_verify::assert_exec_trace_equiv(&state, &message, counter, is_final, &expected_state);
    }
}

pub struct TestVectors;
impl TestVectors {
    /// Get the default test case
    pub fn get_default_test() -> (
        [u64; 8],  // initial state
        [u64; 16], // message block
        [u64; 8],  // expected state
        u64, // counter
        bool, // is_final
    ) {
        // Initial state - Blake2b initialization vector
        let state = [
            0x6a09e667f3bcc908,
            0xbb67ae8584caa73b,
            0x3c6ef372fe94f82b,
            0xa54ff53a5f1d36f1,
            0x510e527fade682d1,
            0x9b05688c2b3e6c1f,
            0x1f83d9abfb41bd6b,
            0x5be0cd19137e2179,
        ];

        // Message block with "abc" in little-endian
        let mut message = [0u64; 16];
        message[0] = 0x0000000000636261u64; // "abc"

        // Expected state after Blake2b compression
        let expected_state = [
            0x0D4D1C983FA580BAu64, // BA 80 A5 3F 98 1C 4D 0D (little-endian)
            0xE9F6129FB697276Au64, // 6A 27 97 B6 9F 12 F6 E9
            0xB7C45A68142F214Cu64, // 4C 21 2F 14 68 5A C4 B7
            0xD1A2FFDB6FBB124Bu64, // 4B 12 BB 6F DB FF A2 D1
            0x2D79AB2A39C5877Du64, // 7D 87 C5 39 2A AB 79 2D
            0x95CC3345DED552C2u64, // C2 52 D5 DE 45 33 CC 95
            0x5A92F1DBA88AD318u64, // 18 D3 8A A8 DB F1 92 5A
            0x239900D4ED8623B9u64, // B9 23 86 ED D4 00 99 23
        ];

        (state, message, expected_state, 3u64, true)
    }
}