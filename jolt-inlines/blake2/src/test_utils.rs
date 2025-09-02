//! Test utilities for BLAKE2 implementation

use tracer::test_utils::{setup_test_context, TestContext};

/// Create a test context for BLAKE2 testing
pub fn create_blake2_test_context() -> TestContext {
    setup_test_context()
}

/// Helper function to compare byte arrays
pub fn compare_bytes(actual: &[u8], expected: &[u8]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    
    for (a, e) in actual.iter().zip(expected.iter()) {
        if a != e {
            return false;
        }
    }
    
    true
}

/// Helper to format bytes as hex string for debugging
pub fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join("")
}
