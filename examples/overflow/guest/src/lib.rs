#![cfg_attr(feature = "guest", no_std)]
#![no_main]

const ARRAY_SIZE: usize = 2048;

#[jolt::provable(stack_size = 1024)]
fn overflow_stack() -> u32 {
    let arr = [1; ARRAY_SIZE];

    let sum = arr.iter().fold(0u32, |acc, &x| acc.wrapping_add(x));
    sum
}
