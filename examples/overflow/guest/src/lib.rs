#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;
use alloc::vec::Vec;

#[jolt::provable(stack_size = 2048)] // Increase stack size to prevent overflow
fn overflow_stack() -> u32 {
    // Use Vec instead of an array to avoid stack overflow
    let arr: Vec<u32> = vec![1u32; 1024];
    arr.iter().sum()
}

#[jolt::provable(stack_size = 8192)]
fn allocate_stack_with_increased_size() -> u32 {
    // Increase stack size to prevent overflow
    overflow_stack()
}

#[jolt::provable(memory_size = 8192)] // Increase memory size
fn overflow_heap() -> u32 {
    let mut vectors = Vec::new();
    let mut allocated_memory = 0; // Logic to track allocated memory

    // Limit the amount of allocated memory
    while allocated_memory < 1024 * 1024 * 100 { // For example, limit to 100 MB
        let v = vec![1u32; 1024];
        vectors.push(v);
        allocated_memory += v.len() * std::mem::size_of::<u32>();
    }

    allocated_memory as u32 // Return the amount of allocated memory
}
