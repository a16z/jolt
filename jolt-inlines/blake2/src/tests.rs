//! Tests for BLAKE2 implementation

use crate::test_constants::*;
use crate::test_utils::*;

#[test]
fn test_blake2b_basic() {
    let mut output = [0u8; 64];
    crate::blake2b(BLAKE2B_TEST_INPUT, &mut output);
    
    assert!(
        compare_bytes(&output, &BLAKE2B_TEST_OUTPUT),
        "BLAKE2b output mismatch.\nExpected: {}\nGot: {}",
        bytes_to_hex(&BLAKE2B_TEST_OUTPUT),
        bytes_to_hex(&output)
    );
}

#[test]
fn test_blake2s_basic() {
    let mut output = [0u8; 32];
    crate::blake2s(BLAKE2S_TEST_INPUT, &mut output);
    
    assert!(
        compare_bytes(&output, &BLAKE2S_TEST_OUTPUT),
        "BLAKE2s output mismatch.\nExpected: {}\nGot: {}",
        bytes_to_hex(&BLAKE2S_TEST_OUTPUT),
        bytes_to_hex(&output)
    );
}

#[test]
fn test_blake2b_empty_input() {
    let input = b"";
    let mut output = [0u8; 64];
    crate::blake2b(input, &mut output);
    
    // TODO: Add expected output for empty input
    assert_eq!(output.len(), 64);
}

#[test]
fn test_blake2s_empty_input() {
    let input = b"";
    let mut output = [0u8; 32];
    crate::blake2s(input, &mut output);
    
    // TODO: Add expected output for empty input
    assert_eq!(output.len(), 32);
}

#[test]
fn test_blake2b_keyed() {
    let key = b"secret_key";
    let input = b"test message";
    let mut output = [0u8; 64];
    crate::blake2b_keyed(key, input, &mut output);
    
    // TODO: Add expected output for keyed BLAKE2b
    assert_eq!(output.len(), 64);
}

#[test]
fn test_blake2s_keyed() {
    let key = b"secret_key";
    let input = b"test message";
    let mut output = [0u8; 32];
    crate::blake2s_keyed(key, input, &mut output);
    
    // TODO: Add expected output for keyed BLAKE2s
    assert_eq!(output.len(), 32);
}
