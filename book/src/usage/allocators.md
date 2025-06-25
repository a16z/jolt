# Allocators
Jolt provides an allocator which supports most containers such as `Vec` and `Box`. This is useful for Jolt users who would like to write `no_std` code rather than using Jolt's [standard library support](./stdlib.md). To use these containers, they must be explicitly imported from `alloc`. The `alloc` crate is automatically provided by rust and does not need to be added to the `Cargo.toml` file.

## Example
```rust
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
```
