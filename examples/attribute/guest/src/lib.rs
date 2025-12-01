#![cfg_attr(feature = "guest", no_std)]

// Test #[allow(arithmetic_overflow)]
#[jolt::provable]
#[allow(arithmetic_overflow)]
fn test_allow_overflow() -> u8 {
    u8::MAX + 1 // Would fail without #[allow]
}

// Test #[allow(unused_variables)]
#[jolt::provable]
#[allow(unused_variables)]
fn test_allow_unused() -> u32 {
    let unused = 42; // Would warn without #[allow]
    100
}

// Test #[inline]
#[jolt::provable]
#[inline]
fn test_inline() -> u32 {
    5 + 5
}

// Test multiple attributes
#[jolt::provable]
#[allow(arithmetic_overflow)]
#[inline]
fn test_multiple_attrs() -> u8 {
    u8::MAX + u8::MAX
}
