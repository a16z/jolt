#![cfg_attr(feature = "guest", no_std)]
#![no_main]

#[jolt::provable]
fn muldiv(a: u32, b: u32, c: u32) -> u32 {
    a * b / c
}
