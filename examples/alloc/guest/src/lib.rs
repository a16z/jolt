#![cfg_attr(feature = "guest", no_std)]
#![no_main]

extern crate alloc;
use alloc::vec::Vec;

#[jolt::provable]
fn alloc(n: u32) -> u32 {
    // let mut v = Vec::<u32>::with_capacity(5);
    // v.push(5);

    //ALLOCATOR.free_memory() as u32
    5
}

