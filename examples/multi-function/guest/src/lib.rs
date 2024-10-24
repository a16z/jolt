#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable]
fn add(x: u32, y: u32) -> u32 {
    x + y
}

#[jolt::provable]
fn mul(x: u32, y: u32) -> u32 {
    x * y
}
