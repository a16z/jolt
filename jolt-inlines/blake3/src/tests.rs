use crate::IV;

#[cfg(test)]
mod tests {

    use super::TestVectors;
    use crate::test_utils::blake3_verify;

    #[test]
    fn test_blake3_compression_default() {
        let (chaining_value, message, counter, block_len, flags, expected_state) = 
            TestVectors::get_default_test();
        
        blake3_verify::assert_exec_trace_equiv(
            &chaining_value,
            &message,
            &counter,
            block_len,
            flags,
            &expected_state,
        );
    }

    #[test]
    fn test_blake3_chaining_value_update() {
        // Test that the first 8 words of the working state represent the updated chaining value
        let (chaining_value, message, counter, block_len, flags, expected_state) = 
            TestVectors::get_default_test();
        
        // Extract expected chaining value from the first 8 words of the expected state
        let mut expected_chaining = [0u32; 8];
        expected_chaining.copy_from_slice(&expected_state[0..8]);
        
        blake3_verify::assert_exec_trace_equiv_chaining(
            &chaining_value,
            &message,
            &counter,
            block_len,
            flags,
            &expected_chaining,
        );
    }
}

pub struct TestVectors;
impl TestVectors {
    /// Get the default test case for Blake3 compression
    pub fn get_default_test() -> (
        [u32; 8],   // initial chaining value
        [u32; 16],  // message block
        [u32; 2],   // counter
        u32,        // block length
        u32,        // flags
        [u32; 16],  // expected working state
    ) {
        // Use Blake3 IV as the initial chaining value
        let chaining_value = IV;

        // Test message block
        let message = [
            50462976, 117835012, 185207048, 252579084, 
            319951120, 387323156, 454695192, 522067228,
            589439264, 656811300, 724183336, 791555372, 
            858927408, 926299444, 993671480, 1061043516,
        ];

        // Counter (low, high)
        let counter = [0, 0];

        // Block length
        let block_len = 64;

        // Flags: CHUNK_START | CHUNK_END | ROOT
        let flags = 1u32 | 2u32 | 8u32;

        // Expected results after Blake3 compression
        let expected_state = [
            0x4171ed4e, 0xd45c4aea, 0x6b6088b7, 0xe2463fd2, 
            0xac9caf12, 0x7ddcaceb, 0xc76d4c1f, 0x981b51f2, 
            0x6cc59cfc, 0xe3ff31b8, 0xe1e7a83e, 0xb209dfd1, 
            0x6727fd6e, 0xaa660067, 0xb123d082, 0x1babe8df,
        ];

        (chaining_value, message, counter, block_len, flags, expected_state)
    }

    /// Get a test case with non-default counter values
    pub fn get_test_with_counter() -> (
        [u32; 8],   // initial chaining value  
        [u32; 16],  // message block
        [u32; 2],   // counter
        u32,        // block length
        u32,        // flags
        [u32; 16],  // expected working state
    ) {
        // Use Blake3 IV as the initial chaining value
        let chaining_value = IV;

        // Test message block (all zeros for simplicity)
        let message = [0u32; 16];

        // Non-zero counter
        let counter = [1, 0];

        // Block length
        let block_len = 64;

        // Flags: CHUNK_START only
        let flags = 1u32;

        // This would need actual expected values from a Blake3 reference implementation
        // For now, using placeholder values - these would need to be updated
        let expected_state = [
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ];

        (chaining_value, message, counter, block_len, flags, expected_state)
    }

    /// Get a test case with different flags
    pub fn get_test_with_flags() -> (
        [u32; 8],   // initial chaining value
        [u32; 16],  // message block
        [u32; 2],   // counter
        u32,        // block length
        u32,        // flags
        [u32; 16],  // expected working state
    ) {
        // Use Blake3 IV as the initial chaining value
        let chaining_value = IV;

        // Test message block (incrementing pattern)
        let message = [
            0x00000001, 0x00000002, 0x00000003, 0x00000004,
            0x00000005, 0x00000006, 0x00000007, 0x00000008,
            0x00000009, 0x0000000a, 0x0000000b, 0x0000000c,
            0x0000000d, 0x0000000e, 0x0000000f, 0x00000010,
        ];

        // Counter
        let counter = [0, 0];

        // Block length (partial block)
        let block_len = 32;

        // Flags: CHUNK_START | CHUNK_END
        let flags = 1u32 | 2u32;

        // This would need actual expected values from a Blake3 reference implementation
        // For now, using placeholder values - these would need to be updated
        let expected_state = [
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ];

        (chaining_value, message, counter, block_len, flags, expected_state)
    }
}