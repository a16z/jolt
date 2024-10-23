#![cfg_attr(feature = "guest", no_std)]
#![no_main]

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

#[jolt::provable(stack_size = 1024)]
fn allocate_stack() -> u32 {
    let arr = [1u32; 1024];
    arr.iter().sum()
}

#[jolt::provable(stack_size = 8192)]
fn allocate_stack_with_increased_size() -> u32 {
    allocate_stack()
}

#[jolt::provable(memory_size = 4096)]
fn allocate_heap() -> u32 {
    let mut vectors = Vec::new();

    loop {
        let v = vec![1u32; 1024];
        vectors.extend(v);
    }
}
