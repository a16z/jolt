#![cfg_attr(feature = "guest", no_std)]
#![no_main]

extern crate alloc;
use alloc::vec::Vec;

#[jolt::provable]
fn alloc(n: u32) -> u32 {
    let mut v = Vec::<u32>::new();
    for i in 0..100 {
        v.push(i);
    }

    v[n as usize]
}
