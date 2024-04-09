# Troubleshooting
## Insufficient Memory or Stack Size
Jolt provides reasonable defaults for the total allocated memory and stack size. It is however possible that the defaults are not sufficient, leading to unpredictable errors within our tracer. To fix this we can try to increase these sizes. We suggest starting with the stack size first as this is much more likely to run out.

Below is an example of manually specifying both the total memory and stack size.
```rust
#![cfg_attr(feature = "guest", no_std)]
#![no_main]

extern crate alloc;
use alloc::vec::Vec;

#[jolt::provable(stack_size = 10000, memory_size = 10000000)]
fn waste_memory(size: u32, n: u32) {
    let mut v = Vec::new();
    for i in 0..size {
        vec.push(i);
    }
}
```

## Maximum Input or Output Size Exceeded
Jolt restricts the size of the inputs and outputs to 4096 bytes. This value is currently not configurable, but we will be adding support for this soon.

## Guest Attempts to Compile Standard Library
Sometimes after installing the toolchain the guest does still tries to compile with the standard library which will fail with a large number of errors that certain items such as `Result` are referenced and not available. This generally happens when one tries to run jolt before installing the toolchain. To address, try rerunning `jolt install-toolchain`, restarting your terminal, and delete both your rust target directory and any files under `/tmp` that begin with jolt.
