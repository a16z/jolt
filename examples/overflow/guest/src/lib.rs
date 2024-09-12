#![cfg_attr(feature = "guest", no_std)]
#![no_main]

use jolt_sdk::prelude::*;

#[cfg(feature = "guest")]
#[panic_handler]
fn panic_handler(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[jolt::provable]
pub fn stack_overflow_risk(n: u32) -> u32 {
    if n == 0 {
        0
    } else {
        1 + stack_overflow_risk(n - 1)
    }
}

#[jolt::provable]
pub fn memory_overflow_risk(n: u32) -> u32 {
    let mut vec = Vec::new();
    for i in 0..n {
        vec.push(i);
    }
    vec.len() as u32
}
