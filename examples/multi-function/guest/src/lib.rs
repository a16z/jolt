#![cfg_attr(feature = "guest", no_std)]
#![no_main]

#[jolt::func]
fn add(x: u32, y: u32) -> u32 {
    x + y
}

#[jolt::func]
fn mul(x: u32, y: u32) -> u32 {
    x * y
}

