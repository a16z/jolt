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
        v.push(i);
    }
}
```

## Maximum Input or Output Size Exceeded
Jolt restricts the size of the inputs and outputs to 4096 bytes by default. Using inputs and outputs that exceed this size will lead to errors. These values can be configured via the macro.

```rust
#![cfg_attr(feature = "guest", no_std)]
#![no_main]

#[jolt::provable(max_input_size = 10000, max_output_size = 10000)]
fn sum(input: &[u8]) -> u32 {
    let mut sum = 0;
    for value in input {
        sum += *value as u32;
    }

    sum
}
```

## Guest Attempts to Compile Standard Library
Sometimes after installing the toolchain the guest still tries to compile with the standard library which will fail with a large number of errors that certain items such as `Result` are referenced and not available. This generally happens when one tries to run jolt before installing the toolchain. To address, try rerunning `jolt install-toolchain`, restarting your terminal, and delete both your rust target directory and any files under `/tmp` that begin with jolt.

## Guest Fails to Compile on the Host
By default, Jolt will attempt to compile the guest for the host architecture. This is useful if you want to run and test the guest's tagged functions directly. If you know your guest code cannot compile on the host (for example, if your guest uses inline RISCV assembly), you can specify to only build for the guest architecture.
```rust
#[jolt::provable(guest_only)]
fn inline_asm() -> (i32, u32, i32, u32) {
    use core::arch::asm;

    let mut data: [u8; 8] = [0; 8];
    unsafe {
        let ptr = data.as_mut_ptr();

        // Store Byte (SB instruction)
        asm!(
        "sb {value}, 0({ptr})",
        ptr = in(reg) ptr,
        value = in(reg) 0x12,
        );
    }
}
```


## Getting Help
If none of the above solve the problem, please create a Github issue with a detailed bug report including the Jolt commit hash, the hardware or container configuration used, and a minimal guest program to reproduce the bug.
