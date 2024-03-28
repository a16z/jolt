#![cfg_attr(feature = "guest", no_std)]
#![cfg_attr(feature = "guest", no_main)]

#[jolt_sdk::main]
fn add(x: u32, y: u32) -> u32 {
    x + y
}

#[jolt_sdk::main]
fn mul(x: u32, y: u32) -> u32 {
    x * y
}

